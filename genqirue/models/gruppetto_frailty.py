"""
Strategy 2: Gruppetto Outlier Detection

Competing risks survival model with Bayesian Cox Proportional Hazards
and time-varying frailty. Identifies riders hiding form in the gruppetto
who may perform well on transition stages.

Key insights:
- Riders in gruppetto on mountain stages may be preserving energy
- Martingale residuals from Cox model serve as proxy for frailty
- High frailty on mountain stage + low gruppetto time loss = hidden form
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

from .base import SurvivalModel, ModelDiagnostics, PosteriorSummary, StrategyMixin

logger = logging.getLogger(__name__)


@dataclass
class SurvivalRecord:
    """Single survival observation for a rider-stage."""
    rider_id: int
    stage_id: int
    stage_date: datetime
    stage_type: str  # 'mountain', 'flat', etc.
    
    # Survival outcome
    time_to_cutoff: float  # Minutes to time cut (or censored time)
    event_observed: bool   # 1 if OTL/DNF, 0 if finished inside cut
    
    # Covariates
    gc_position: Optional[int] = None
    gc_time_behind: float = 0.0
    gruppetto_indicator: int = 0  # 1 if in gruppetto
    gruppetto_time_loss: float = 0.0  # Time lost to gruppetto winner
    
    # Rider characteristics
    rider_specialty_scores: Optional[Dict[str, int]] = None


class GruppettoFrailtyModel(SurvivalModel, StrategyMixin):
    """
    Bayesian Cox PH model with time-varying frailty.
    
    Model specification:
    λ_i(t) = λ_0(t) * exp(β^T * X_i + b_i)
    
    where:
    - λ_0(t): Baseline hazard (modeled as piecewise constant)
    - β: Fixed effects coefficients
    - X_i: Covariates (GC position, gruppetto status, etc.)
    - b_i ~ N(0, σ²_b): Rider-specific frailty (random effect)
    
    The frailty term captures unobserved heterogeneity in rider "toughness".
    High frailty + gruppetto on mountains = potential hidden form.
    """
    
    def __init__(
        self,
        model_name: str = "gruppetto_frailty",
        random_seed: int = 42,
        mcmc_samples: int = 1000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        n_hazard_pieces: int = 5,  # Number of pieces for baseline hazard
        frailty_prior_sigma: float = 1.0,
    ):
        super().__init__(
            model_name=model_name,
            random_seed=random_seed,
            mcmc_samples=mcmc_samples,
            mcmc_tune=mcmc_tune,
            mcmc_chains=mcmc_chains
        )
        self.n_hazard_pieces = n_hazard_pieces
        self.frailty_prior_sigma = frailty_prior_sigma
        
        # Data storage
        self.survival_data: List[SurvivalRecord] = []
        self.rider_ids: List[int] = []
        self.n_riders: int = 0
        
        # Fitted parameters (point estimates for fast inference)
        self.frailty_estimates: Dict[int, float] = {}
        self.hidden_form_probs: Dict[int, float] = {}
        self.beta_estimates: Dict[str, float] = {}
        
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build the Bayesian Cox PH model with frailty.
        
        Args:
            data: Dictionary with keys:
                - 'survival_data': List[SurvivalRecord]
                - 'rider_ids': List of unique rider IDs
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required for Bayesian Cox model")
        
        self.survival_data = data.get('survival_data', [])
        self.rider_ids = data.get('rider_ids', [])
        self.n_riders = len(self.rider_ids)
        
        if not self.survival_data or self.n_riders == 0:
            raise ValueError("No survival data provided")
        
        # Prepare design matrix
        X, event, time, rider_idx = self._prepare_data()
        n_obs = len(event)
        
        # Create cut points for piecewise constant baseline hazard
        time_cut_points = np.linspace(0, time.max(), self.n_hazard_pieces + 1)
        
        with pm.Model() as model:
            # === Fixed Effects ===
            # Coefficients for observed covariates
            beta_gc_pos = pm.Normal('beta_gc_pos', mu=0, sigma=0.5)
            beta_gc_time = pm.Normal('beta_gc_time', mu=0, sigma=0.01)  # Per second
            beta_gruppetto = pm.Normal('beta_gruppetto', mu=0, sigma=1.0)
            beta_gruppetto_loss = pm.Normal('beta_gruppetto_loss', mu=0, sigma=0.01)
            
            # === Random Effects (Frailty) ===
            # Rider-specific frailty: b_i ~ N(0, sigma_b^2)
            sigma_b = pm.HalfNormal('sigma_b', sigma=self.frailty_prior_sigma)
            frailty_raw = pm.Normal('frailty_raw', mu=0, sigma=1, shape=self.n_riders)
            frailty = pm.Deterministic('frailty', frailty_raw * sigma_b)
            
            # === Baseline Hazard (Piecewise Constant) ===
            log_hazard_base = pm.Normal(
                'log_hazard_base', 
                mu=np.log(1e-3), 
                sigma=2, 
                shape=self.n_hazard_pieces
            )
            
            # === Linear Predictor ===
            # η_i = β^T * X_i + b_rider[i]
            eta = (
                beta_gc_pos * X[:, 0] +
                beta_gc_time * X[:, 1] +
                beta_gruppetto * X[:, 2] +
                beta_gruppetto_loss * X[:, 3] +
                frailty[rider_idx]
            )
            
            # Hazard rate for each observation
            # λ_i(t) = λ_0(t) * exp(η_i)
            hazard_piece = np.digitize(time, time_cut_points) - 1
            hazard_piece = np.clip(hazard_piece, 0, self.n_hazard_pieces - 1)
            
            log_hazard = log_hazard_base[hazard_piece] + eta
            hazard = pm.math.exp(log_hazard)
            
            # === Survival Likelihood ===
            # For event: contribution is log(hazard)
            # For censored: contribution is -cumulative_hazard
            # Simplified: using Poisson approximation for Cox model
            
            # More accurate: partial likelihood approximation
            # Here we use a parametric approximation (piecewise exponential)
            piece_widths = np.diff(time_cut_points)
            piece_widths = np.append(piece_widths, piece_widths[-1])
            
            # Cumulative hazard for each observation
            cum_hazard = pm.math.sum(
                hazard[:, None] * piece_widths[None, :] * 
                (time[:, None] > time_cut_points[:-1][None, :]),
                axis=1
            )
            
            # Log-likelihood
            logp = pm.math.sum(
                event * log_hazard - cum_hazard
            )
            
            pm.Potential('log_likelihood', logp)
            
            return model
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data matrices from survival records.
        
        Returns:
            X: Design matrix (n_obs, n_features)
            event: Event indicator (n_obs,)
            time: Survival/censoring time (n_obs,)
            rider_idx: Rider index for random effects (n_obs,)
        """
        n_obs = len(self.survival_data)
        
        X = np.zeros((n_obs, 4))  # gc_pos, gc_time, gruppetto, gruppetto_loss
        event = np.zeros(n_obs)
        time = np.zeros(n_obs)
        rider_idx = np.zeros(n_obs, dtype=int)
        
        rider_to_idx = {r: i for i, r in enumerate(self.rider_ids)}
        
        for i, record in enumerate(self.survival_data):
            X[i, 0] = record.gc_position if record.gc_position else 0
            X[i, 1] = record.gc_time_behind
            X[i, 2] = record.gruppetto_indicator
            X[i, 3] = record.gruppetto_time_loss
            
            event[i] = record.event_observed
            time[i] = record.time_to_cutoff
            rider_idx[i] = rider_to_idx.get(record.rider_id, 0)
        
        # Standardize continuous features
        X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-8)
        X[:, 1] = (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)
        X[:, 3] = (X[:, 3] - X[:, 3].mean()) / (X[:, 3].std() + 1e-8)
        
        return X, event, time, rider_idx
    
    def fit(self, data: Dict[str, Any], **kwargs) -> 'GruppettoFrailtyModel':
        """Fit the model and extract frailty estimates."""
        super().fit(data, **kwargs)
        
        # Extract frailty estimates for each rider
        if self.idata is not None:
            self._extract_frailty_estimates()
            self._compute_hidden_form_probs()
        
        return self
    
    def _extract_frailty_estimates(self):
        """Extract posterior mean frailty for each rider."""
        if not PYMC_AVAILABLE or self.idata is None:
            return
        
        frailty_samples = self.idata.posterior['frailty'].values
        # Shape: (chains, samples, n_riders)
        frailty_mean = frailty_samples.reshape(-1, self.n_riders).mean(axis=0)
        
        self.frailty_estimates = {
            rider_id: float(frailty_mean[i])
            for i, rider_id in enumerate(self.rider_ids)
        }
        
        logger.info(f"Extracted frailty estimates for {len(self.frailty_estimates)} riders")
    
    def _compute_hidden_form_probs(self):
        """
        Compute probability of hidden good form based on frailty pattern.
        
        High frailty on mountain stages while staying close to gruppetto
        suggests the rider is preserving energy rather than struggling.
        """
        if not self.frailty_estimates:
            return
        
        frailty_values = np.array(list(self.frailty_estimates.values()))
        mean_frailty = np.mean(frailty_values)
        std_frailty = np.std(frailty_values) + 1e-8
        
        for rider_id, frailty in self.frailty_estimates.items():
            # Z-score
            z_score = (frailty - mean_frailty) / std_frailty
            
            # High frailty (above 1.5 SD) indicates potential hidden form
            if z_score > 1.5:
                # Probability of being in good form
                self.hidden_form_probs[rider_id] = 1 - self.logistic(
                    -(frailty - mean_frailty) / std_frailty
                )
            else:
                self.hidden_form_probs[rider_id] = 0.0
        
        logger.info(
            f"Identified {sum(1 for p in self.hidden_form_probs.values() if p > 0)} "
            f"riders with potential hidden form"
        )
    
    def compute_frailty(self) -> Dict[int, float]:
        """Return frailty estimates for all riders."""
        return self.frailty_estimates.copy()
    
    def get_hidden_form_prob(self, rider_id: int) -> float:
        """Get probability of hidden good form for a rider."""
        return self.hidden_form_probs.get(rider_id, 0.0)
    
    def cumulative_hazard(self, times: np.ndarray) -> np.ndarray:
        """
        Compute cumulative hazard function.
        
        Args:
            times: Time points in minutes
            
        Returns:
            Cumulative hazard at each time point
        """
        if self.idata is None:
            return np.zeros_like(times)
        
        # Extract baseline hazard samples
        log_hazard_samples = self.idata.posterior['log_hazard_base'].values
        # Shape: (chains, samples, n_pieces)
        
        log_hazard_mean = log_hazard_samples.reshape(-1, self.n_hazard_pieces).mean(axis=0)
        hazard_mean = np.exp(log_hazard_mean)
        
        # Piecewise constant hazard
        time_cut_points = np.linspace(0, 200, self.n_hazard_pieces + 1)  # 200 min max
        piece_widths = np.diff(time_cut_points)
        
        cum_hazard = np.zeros_like(times, dtype=float)
        for i, t in enumerate(times):
            piece_idx = np.digitize(t, time_cut_points) - 1
            piece_idx = min(piece_idx, self.n_hazard_pieces - 1)
            
            # Sum hazard up to this piece
            if piece_idx > 0:
                cum_hazard[i] += np.sum(hazard_mean[:piece_idx] * piece_widths[:piece_idx])
            
            # Add partial contribution from current piece
            if piece_idx >= 0:
                time_in_piece = t - time_cut_points[piece_idx]
                cum_hazard[i] += hazard_mean[piece_idx] * time_in_piece
        
        return cum_hazard
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Predict OTL/DNF probability for new observations.
        
        Args:
            new_data: Dictionary with survival records
            
        Returns:
            Dictionary with 'otl_prob' and 'frailty' arrays
        """
        if not self.frailty_estimates:
            return {'otl_prob': np.array([]), 'frailty': np.array([])}
        
        survival_data = new_data.get('survival_data', [])
        rider_ids = new_data.get('rider_ids', [])
        
        otl_probs = []
        frailties = []
        
        for record in survival_data:
            rider_id = record.rider_id
            frailty = self.frailty_estimates.get(rider_id, 0.0)
            
            # Simplified OTL probability using cumulative hazard
            time = record.time_to_cutoff
            cum_haz = self.cumulative_hazard(np.array([time]))[0]
            
            # Adjust by frailty
            adjusted_cum_haz = cum_haz * np.exp(frailty)
            otl_prob = 1 - np.exp(-adjusted_cum_haz)
            
            otl_probs.append(otl_prob)
            frailties.append(frailty)
        
        return {
            'otl_prob': np.array(otl_probs),
            'frailty': np.array(frailties)
        }
    
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """
        Calculate betting edge based on frailty signal.
        
        Strategy: Bet on riders with high hidden form probability
        on transition stages (flat/hilly after mountains).
        
        Args:
            prediction: Output from predict()
            market_odds: Decimal odds
            
        Returns:
            Edge in basis points
        """
        rider_id = prediction.get('rider_id')
        stage_type = prediction.get('stage_type', '')
        
        # Only bet on transition stages after gruppetto
        if stage_type not in ['flat', 'hilly']:
            return 0.0
        
        hidden_form_prob = self.get_hidden_form_prob(rider_id)
        if hidden_form_prob < 0.3:  # Threshold
            return 0.0
        
        # Estimate win probability (simplified)
        win_prob = hidden_form_prob * 0.15  # Base rate adjustment
        
        # Calculate edge
        if market_odds <= 1:
            return 0.0
        
        implied_prob = 1.0 / market_odds
        edge = win_prob - implied_prob
        
        return edge * 10000  # Convert to basis points
    
    def get_sql_schema(self) -> str:
        """
        Return SQL schema for storing frailty results.
        
        Matches the exact schema specified in PLAN.md.
        """
        return """
        -- Rider frailty table for Strategy 2
        CREATE TABLE IF NOT EXISTS rider_frailty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rider_id INTEGER NOT NULL,
            frailty_estimate REAL NOT NULL,
            hidden_form_prob REAL NOT NULL DEFAULT 0.0,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            FOREIGN KEY (rider_id) REFERENCES riders(id),
            UNIQUE(rider_id, computed_at)
        );
        
        CREATE INDEX IF NOT EXISTS idx_frailty_rider ON rider_frailty(rider_id);
        CREATE INDEX IF NOT EXISTS idx_frailty_hidden_form ON rider_frailty(hidden_form_prob) 
            WHERE hidden_form_prob > 0;
        
        -- View for hidden form detection (as specified in PLAN.md)
        CREATE VIEW IF NOT EXISTS rider_hidden_form AS
        SELECT 
            rf.rider_id,
            r.name as rider_name,
            rf.frailty_estimate,
            rf.hidden_form_prob,
            rf.computed_at,
            CASE 
                WHEN rf.frailty_estimate > (
                    SELECT AVG(frailty_estimate) + 1.5 * STDDEV(frailty_estimate) 
                    FROM rider_frailty 
                    WHERE computed_at = (SELECT MAX(computed_at) FROM rider_frailty)
                )
                THEN 1 - (
                    SELECT CUME_DIST() OVER (ORDER BY frailty_estimate)
                    FROM rider_frailty rf2
                    WHERE rf2.rider_id = rf.rider_id
                    AND rf2.computed_at = rf.computed_at
                )
                ELSE 0 
            END as hidden_form_signal
        FROM rider_frailty rf
        JOIN riders r ON rf.rider_id = r.id
        WHERE rf.computed_at = (SELECT MAX(computed_at) FROM rider_frailty);
        """


class FastFrailtyEstimator:
    """
    Fast (non-Bayesian) frailty estimator for real-time use.
    Uses sksurv CoxPH model when available, falls back to simplified model.
    """
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.frailty_estimates: Dict[int, float] = {}
        
    def fit(self, survival_data: List[SurvivalRecord]) -> 'FastFrailtyEstimator':
        """Fit using sksurv if available."""
        if not SKSURV_AVAILABLE or len(survival_data) < 10:
            self._fit_simple(survival_data)
            return self
        
        try:
            # Prepare sksurv format
            y = np.array([
                (record.event_observed, record.time_to_cutoff)
                for record in survival_data
            ], dtype=[('event', bool), ('time', float)])
            
            X = np.array([
                [
                    record.gc_position or 0,
                    record.gc_time_behind,
                    record.gruppetto_indicator,
                    record.gruppetto_time_loss
                ]
                for record in survival_data
            ])
            
            self.model = CoxPHSurvivalAnalysis()
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Compute martingale residuals as proxy for frailty
            self._compute_martingale_residuals(survival_data, X, y)
            
        except Exception as e:
            logger.warning(f"CoxPH fitting failed: {e}, using simple model")
            self._fit_simple(survival_data)
        
        return self
    
    def _fit_simple(self, survival_data: List[SurvivalRecord]):
        """Simple frailty estimation based on gruppetto patterns."""
        rider_stats: Dict[int, List[float]] = {}
        
        for record in survival_data:
            rider_id = record.rider_id
            if record.stage_type == 'mountain':
                # Frailty proxy: gruppetto time loss / (1 + gruppetto_indicator)
                if rider_id not in rider_stats:
                    rider_stats[rider_id] = []
                
                if record.gruppetto_indicator:
                    # Time loss normalized by GC gap
                    proxy = record.gruppetto_time_loss / max(record.gc_time_behind, 60)
                    rider_stats[rider_id].append(proxy)
        
        # Compute average frailty per rider
        for rider_id, proxies in rider_stats.items():
            if proxies:
                self.frailty_estimates[rider_id] = np.mean(proxies)
        
        self.is_fitted = True
    
    def _compute_martingale_residuals(
        self, 
        survival_data: List[SurvivalRecord],
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        Compute martingale residuals as frailty proxy.
        
        Martingale residual = observed - expected events
        High residual = rider survived longer than predicted (good form)
        """
        if self.model is None:
            return
        
        # Get risk scores
        risk_scores = self.model.predict(X)
        
        # Compute simple martingale-like residuals
        for i, record in enumerate(survival_data):
            rider_id = record.rider_id
            
            # Lower risk score than expected = good form
            expected = 0.5  # Baseline expectation
            observed = 1.0 if not record.event_observed else 0.0
            
            residual = observed - expected * np.exp(risk_scores[i])
            
            if rider_id not in self.frailty_estimates:
                self.frailty_estimates[rider_id] = []
            self.frailty_estimates[rider_id].append(residual)
        
        # Average across observations
        for rider_id in self.frailty_estimates:
            residuals = self.frailty_estimates[rider_id]
            self.frailty_estimates[rider_id] = np.mean(residuals) if residuals else 0.0
    
    def get_frailty(self, rider_id: int) -> float:
        """Get frailty estimate for a rider."""
        return self.frailty_estimates.get(rider_id, 0.0)
