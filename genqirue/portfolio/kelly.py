"""
Robust Kelly Criterion with Uncertainty and CVaR Constraints

Implements portfolio optimization for cycling betting with:
1. Full Kelly: f* = (bp - q) / b
2. Robust Kelly: Accounts for probability uncertainty
3. CVaR constraints: Limits tail risk
4. Correlation handling: Between riders on same team
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from ..domain.entities import Position, Portfolio, MarketState
from ..models.base import StrategyMixin

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Kelly sizing variants."""
    FULL_KELLY = "full"
    HALF_KELLY = "half"
    QUARTER_KELLY = "quarter"
    ROBUST_KELLY = "robust"
    FRACTIONAL_KELLY = "fractional"


@dataclass
class KellyParameters:
    """Parameters for Kelly optimization."""
    # Kelly variant
    method: SizingMethod = SizingMethod.HALF_KELLY
    fractional_kelly: float = 0.25  # For FRACTIONAL_KELLY
    
    # Robust Kelly uncertainty penalty (γ in PLAN.md formula)
    robust_gamma: float = 1.0
    
    # Risk constraints
    max_position_pct: float = 0.25  # Never allocate >25% to single position
    max_portfolio_var: float = 0.05  # Max portfolio variance
    cvar_alpha: float = 0.95  # CVaR confidence level
    max_cvar: float = 0.10  # Maximum CVaR (10%)
    
    # Team correlation handling
    team_correlation_boost: float = 0.3  # Additional correlation for teammates
    
    # Minimum edge threshold (basis points)
    min_edge_bps: float = 50.0


class RobustKellyOptimizer(StrategyMixin):
    """
    Robust Kelly optimizer with CVaR constraints.
    
    Implements the formula from PLAN.md:
    f_i* = [b_i*p̂_i - q_i] / [b_i * (1 - p̂_i²(b_i+1)²σ_i² / (p̂_i²(b_i+1)²))]
    
    With constraints:
    - Σf_i ≤ 1.0 (full allocation)
    - f_i ≥ 0 (no short selling in betting)
    - f_i ≤ max_position_pct (single position limit)
    - quad_form(f, cov) ≤ max_portfolio_var (variance)
    - CVaR_α(f) ≤ max_cvar (tail risk)
    """
    
    def __init__(self, params: Optional[KellyParameters] = None):
        self.params = params or KellyParameters()
        self.positions_history: List[Portfolio] = []
        
    def calculate_kelly_fractions(
        self,
        probabilities: np.ndarray,
        odds: np.ndarray,
        prob_stds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Kelly fractions for all positions.
        
        Args:
            probabilities: Array of win probabilities
            odds: Array of decimal odds
            prob_stds: Array of probability standard deviations (for robust)
            
        Returns:
            Dictionary with 'full', 'robust', 'half', 'quarter' fractions
        """
        n = len(probabilities)
        
        # Full Kelly
        b = odds - 1  # Net odds
        q = 1 - probabilities
        
        full_kelly = np.zeros(n)
        for i in range(n):
            if b[i] > 0:
                f = (b[i] * probabilities[i] - q[i]) / b[i]
                full_kelly[i] = max(0.0, f)
        
        # Robust Kelly with uncertainty
        robust_kelly = np.zeros(n)
        if prob_stds is not None:
            for i in range(n):
                robust_kelly[i] = self.robust_kelly_fraction(
                    probabilities[i],
                    odds[i],
                    prob_stds[i],
                    self.params.robust_gamma
                )
        else:
            robust_kelly = full_kelly.copy()
        
        return {
            'full': full_kelly,
            'robust': robust_kelly,
            'half': full_kelly * 0.5,
            'quarter': full_kelly * 0.25,
            'fractional': full_kelly * self.params.fractional_kelly
        }
    
    def optimize_portfolio(
        self,
        positions: List[Position],
        team_assignments: Optional[Dict[int, int]] = None
    ) -> Portfolio:
        """
        Optimize portfolio using convex optimization with constraints.
        
        Args:
            positions: List of candidate positions
            team_assignments: Dict mapping rider_id to team_id for correlation
            
        Returns:
            Optimized portfolio
        """
        if not positions:
            return Portfolio(positions=[])
        
        n = len(positions)
        
        # Extract probabilities and odds
        probs = np.array([p.market_state.model_prob for p in positions])
        odds_arr = np.array([p.market_state.back_odds for p in positions])
        prob_stds = np.array([p.market_state.model_prob_uncertainty for p in positions])
        
        # Filter by minimum edge
        edges = np.array([p.market_state.edge_bps for p in positions])
        valid_mask = edges >= self.params.min_edge_bps
        
        if not valid_mask.any():
            logger.info("No positions meet minimum edge threshold")
            return Portfolio(positions=[])
        
        # Calculate base Kelly fractions
        kelly_fracs = self.calculate_kelly_fractions(
            probs[valid_mask],
            odds_arr[valid_mask],
            prob_stds[valid_mask]
        )
        
        if self.params.method == SizingMethod.FULL_KELLY:
            base_fractions = kelly_fracs['full']
        elif self.params.method == SizingMethod.ROBUST_KELLY:
            base_fractions = kelly_fracs['robust']
        elif self.params.method == SizingMethod.HALF_KELLY:
            base_fractions = kelly_fracs['half']
        elif self.params.method == SizingMethod.QUARTER_KELLY:
            base_fractions = kelly_fracs['quarter']
        else:
            base_fractions = kelly_fracs['fractional']
        
        # Build covariance matrix (accounting for team correlations)
        cov_matrix = self._build_covariance_matrix(
            positions, valid_mask, team_assignments
        )
        
        # Convex optimization for final sizing
        if CVXPY_AVAILABLE and n > 1:
            final_fractions = self._convex_optimize(
                base_fractions,
                cov_matrix,
                probs[valid_mask],
                odds_arr[valid_mask]
            )
        else:
            # Simple constraint application
            final_fractions = self._apply_constraints_simple(
                base_fractions,
                cov_matrix
            )
        
        # Update positions with final stakes
        valid_positions = [p for i, p in enumerate(positions) if valid_mask[i]]
        for i, pos in enumerate(valid_positions):
            pos.kelly_fraction = kelly_fracs['full'][i]
            pos.robust_kelly_fraction = kelly_fracs['robust'][i]
            pos.half_kelly_fraction = kelly_fracs['half'][i]
            pos.stake = final_fractions[i]
        
        portfolio = Portfolio(positions=valid_positions)
        portfolio.correlation_matrix = cov_matrix
        
        # Calculate portfolio metrics
        self._calculate_portfolio_metrics(portfolio, probs[valid_mask], odds_arr[valid_mask])
        
        self.positions_history.append(portfolio)
        
        return portfolio
    
    def _build_covariance_matrix(
        self,
        positions: List[Position],
        valid_mask: np.ndarray,
        team_assignments: Optional[Dict[int, int]]
    ) -> np.ndarray:
        """
        Build covariance matrix accounting for team correlations.
        
        Riders on the same team have positively correlated outcomes
        (team tactics, shared resources).
        """
        n_valid = valid_mask.sum()
        cov = np.eye(n_valid) * 0.01  # Base variance 1%
        
        if team_assignments is None or n_valid < 2:
            return cov
        
        valid_positions = [p for i, p in enumerate(positions) if valid_mask[i]]
        
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                rider_i = valid_positions[i].market_state.selection_id
                rider_j = valid_positions[j].market_state.selection_id
                
                team_i = team_assignments.get(rider_i)
                team_j = team_assignments.get(rider_j)
                
                if team_i is not None and team_i == team_j:
                    # Same team - positive correlation
                    cov[i, j] = self.params.team_correlation_boost
                    cov[j, i] = self.params.team_correlation_boost
                else:
                    # Different teams - slight positive correlation (same race)
                    cov[i, j] = 0.05
                    cov[j, i] = 0.05
        
        return cov
    
    def _convex_optimize(
        self,
        base_fractions: np.ndarray,
        cov_matrix: np.ndarray,
        probabilities: np.ndarray,
        odds: np.ndarray
    ) -> np.ndarray:
        """
        Convex optimization with CVaR and variance constraints.
        
        Uses cvxpy to solve:
        maximize: expected_return - risk_penalty
        subject to: sum(f) <= 1, f >= 0, f <= max_position,
                    quad_form(f, cov) <= max_var, CVaR <= max_cvar
        """
        if not CVXPY_AVAILABLE:
            return self._apply_constraints_simple(base_fractions, cov_matrix)
        
        n = len(base_fractions)
        
        # Decision variable: allocation fractions
        f = cp.Variable(n)
        
        # Expected returns
        b = odds - 1
        expected_returns = probabilities * b - (1 - probabilities)
        
        # Objective: maximize Kelly utility (log wealth approximation)
        # Use linear approximation for convexity: E[return] - 0.5 * variance
        objective = cp.Maximize(
            expected_returns @ f - 0.5 * cp.quad_form(f, cov_matrix * 100)
        )
        
        # Constraints
        constraints = [
            cp.sum(f) <= 1.0,  # Full allocation
            f >= 0,  # No short selling
            f <= self.params.max_position_pct,  # Single position limit
            cp.quad_form(f, cov_matrix) <= self.params.max_portfolio_var,  # Variance
        ]
        
        # CVaR constraint (simplified - assumes normal distribution)
        # CVaR_α ≈ μ - σ * φ(Φ^{-1}(α)) / (1-α)
        portfolio_std = cp.sqrt(cp.quad_form(f, cov_matrix))
        z_alpha = 1.645  # 95% quantile
        phi_z = 0.103  # φ(1.645)
        cvar_approx = -portfolio_std * phi_z / (1 - self.params.cvar_alpha)
        constraints.append(cvar_approx <= self.params.max_cvar)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            result = problem.solve(solver=cp.SCS, verbose=False)
            
            if f.value is not None:
                return np.maximum(0, f.value)
        except Exception as e:
            logger.warning(f"Convex optimization failed: {e}")
        
        # Fallback to simple constraints
        return self._apply_constraints_simple(base_fractions, cov_matrix)
    
    def _apply_constraints_simple(
        self,
        fractions: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply constraints without convex optimization.
        """
        f = fractions.copy()
        
        # Non-negative
        f = np.maximum(0, f)
        
        # Max position limit
        f = np.minimum(f, self.params.max_position_pct)
        
        # Normalize to sum <= 1
        total = f.sum()
        if total > 1.0:
            f = f / total
        
        # Check variance constraint
        portfolio_var = f @ cov_matrix @ f
        if portfolio_var > self.params.max_portfolio_var:
            # Scale down
            scale = np.sqrt(self.params.max_portfolio_var / portfolio_var)
            f = f * scale
        
        return f
    
    def _calculate_portfolio_metrics(
        self,
        portfolio: Portfolio,
        probabilities: np.ndarray,
        odds: np.ndarray
    ):
        """Calculate portfolio-level risk metrics."""
        n = len(portfolio.positions)
        if n == 0:
            return
        
        stakes = np.array([p.stake for p in portfolio.positions])
        
        # Expected return
        b = odds - 1
        portfolio.expected_return = np.sum(
            stakes * (probabilities * b - (1 - probabilities))
        )
        
        # Portfolio variance
        if portfolio.correlation_matrix is not None:
            portfolio.portfolio_variance = stakes @ portfolio.correlation_matrix @ stakes
        
        # CVaR (simplified - assumes independence)
        portfolio.cvar_95 = self._calculate_cvar(probabilities, odds, stakes)
    
    def _calculate_cvar(
        self,
        probabilities: np.ndarray,
        odds: np.ndarray,
        stakes: np.ndarray,
        n_scenarios: int = 1000
    ) -> float:
        """
        Calculate CVaR via Monte Carlo simulation.
        """
        n = len(probabilities)
        
        # Simulate outcomes
        outcomes = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            # Simulate which bets win (Bernoulli)
            wins = np.random.random(n) < probabilities
            
            # Calculate P&L
            pnl = np.where(
                wins,
                stakes * (odds - 1),
                -stakes
            )
            outcomes[i] = pnl.sum()
        
        # CVaR is mean of worst (1-α) outcomes
        var_threshold = np.percentile(outcomes, (1 - self.params.cvar_alpha) * 100)
        cvar = outcomes[outcomes <= var_threshold].mean()
        
        return cvar
    
    def get_position_sizing_recommendation(
        self,
        market_state: MarketState,
        bankroll: float = 100.0
    ) -> Dict[str, float]:
        """
        Get position sizing recommendation for a single market.
        
        Args:
            market_state: Market state with probability and odds
            bankroll: Total bankroll for absolute stake calculation
            
        Returns:
            Dictionary with stake recommendations
        """
        prob = market_state.model_prob
        odds_val = market_state.back_odds
        prob_std = market_state.model_prob_uncertainty
        
        # Calculate all Kelly variants
        kelly_fracs = self.calculate_kelly_fractions(
            np.array([prob]),
            np.array([odds_val]),
            np.array([prob_std])
        )
        
        return {
            'full_kelly_stake': kelly_fracs['full'][0] * bankroll,
            'half_kelly_stake': kelly_fracs['half'][0] * bankroll,
            'quarter_kelly_stake': kelly_fracs['quarter'][0] * bankroll,
            'robust_kelly_stake': kelly_fracs['robust'][0] * bankroll,
            'recommended_stake': kelly_fracs['half'][0] * bankroll,
            'kelly_fraction': kelly_fracs['half'][0],
            'expected_value': self.calculate_ev(prob, odds_val),
        }
    
    def generate_report(self, portfolio: Portfolio) -> str:
        """Generate human-readable portfolio report."""
        lines = [
            "=" * 60,
            "PORTFOLIO OPTIMIZATION REPORT",
            "=" * 60,
            f"Total Positions: {len(portfolio.positions)}",
            f"Total Stake: {portfolio.total_stake:.2%}",
            f"Expected Return: {portfolio.expected_return:.2%}",
            f"Portfolio Variance: {portfolio.portfolio_variance:.4f}",
            f"CVaR (95%): {portfolio.cvar_95:.2%}",
            "",
            "POSITIONS:",
            "-" * 60,
        ]
        
        for pos in sorted(portfolio.positions, key=lambda p: -p.stake):
            lines.append(
                f"  {pos.market_state.selection_id}: "
                f"stake={pos.stake:.2%}, "
                f"edge={pos.market_state.edge_bps:.0f}bps, "
                f"strategy={pos.originating_strategy}"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class CVaROptimizer:
    """
    Alternative optimizer focusing specifically on CVaR minimization.
    """
    
    def __init__(self, alpha: float = 0.95, max_cvar: float = 0.05):
        self.alpha = alpha
        self.max_cvar = max_cvar
    
    def optimize(
        self,
        returns: np.ndarray,  # Scenario returns matrix (n_scenarios x n_assets)
        probabilities: np.ndarray  # Win probabilities
    ) -> np.ndarray:
        """
        Minimize CVaR using Rockafellar-Uryasev formulation.
        """
        if not CVXPY_AVAILABLE:
            # Fallback to equal weight
            n = len(probabilities)
            return np.ones(n) / n
        
        n_scenarios, n_assets = returns.shape
        
        # Decision variables
        w = cp.Variable(n_assets)  # Portfolio weights
        z = cp.Variable(n_scenarios)  # Auxiliary variables
        q = cp.Variable()  # VaR threshold
        
        # Objective: minimize CVaR
        objective = cp.Minimize(
            q + (1 / (n_scenarios * (1 - self.alpha))) * cp.sum(z)
        )
        
        # Constraints
        constraints = [
            z >= 0,
            z >= -returns @ w - q,  # z_i >= loss_i - q
            cp.sum(w) == 1,
            w >= 0,
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if w.value is not None:
            return np.maximum(0, w.value)
        
        return np.ones(n_assets) / n_assets
