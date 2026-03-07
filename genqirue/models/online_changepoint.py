"""
Strategy 12: Attack Confirmation via Bayesian Online Changepoint Detection

Implements Adams & MacKay (2007) Bayesian Online Changepoint Detection
with Weibull hazard function for detecting power-based attacks in real-time.

Key insights:
- Power surges above rider's 95th percentile indicate potential attacks
- Changepoint probability > 0.8 + yesterday's Z-score > 2.0 = bet signal
- Must operate with <50ms latency for live betting
"""
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if numba not available
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

from .base import OnlineModel, StrategyMixin

logger = logging.getLogger(__name__)


class AttackSignal(Enum):
    """Attack detection signal levels."""
    NONE = 0
    SUSPECTED = 1
    CONFIRMED = 2
    ESTABLISHED = 3


@dataclass
class PowerObservation:
    """Single power/telemetry observation."""
    timestamp: datetime
    rider_id: int
    stage_id: int
    
    # Power metrics
    power_watts: float
    power_z_score: float = 0.0  # Normalized power
    normalized_power: float = 0.0  # Power relative to FTP
    
    # Context
    gradient_pct: float = 0.0
    speed_kmh: float = 0.0
    cadence: Optional[int] = None
    heart_rate: Optional[int] = None
    
    # Derived
    is_attack: bool = False  # Ground truth if available


@dataclass
class ChangepointState:
    """Current state of the changepoint detector."""
    run_length: int = 0  # r_t: time since last changepoint
    changepoint_prob: float = 0.0  # P(r_t = 0 | data)
    
    # Distribution parameters
    mean: float = 0.0
    variance: float = 1.0
    
    # For student-t robustness
    dof: float = 10.0
    
    # Cached run length distribution
    run_length_dist: np.ndarray = field(default_factory=lambda: np.ones(1))
    
    # History for diagnostics
    run_length_history: List[int] = field(default_factory=list)
    cp_prob_history: List[float] = field(default_factory=list)


# Pre-computed hazard function lookup for speed
HazardCache = {}


def weibull_hazard(t: float, shape: float = 2.0, scale: float = 100.0) -> float:
    """
    Weibull hazard function.
    
    h(t) = (shape/scale) * (t/scale)^(shape-1)
    
    Args:
        t: Time since changepoint
        shape: Weibull shape parameter (k)
        scale: Weibull scale parameter (lambda)
    
    Returns:
        Hazard rate
    """
    if t <= 0:
        return 1.0
    return (shape / scale) * (t / scale) ** (shape - 1)


def constant_hazard(t: float, lambda_val: float = 250.0) -> float:
    """
    Constant hazard function (simpler alternative).
    
    Args:
        t: Time (ignored for constant hazard)
        lambda_val: Expected run length between changepoints
    
    Returns:
        Constant hazard rate = 1/lambda
    """
    return 1.0 / lambda_val


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _update_run_length_numba(
        x_t: float,
        R_prev: np.ndarray,
        hazard: np.ndarray,
        pred_probs: np.ndarray,
        max_run: int
    ) -> Tuple[np.ndarray, float]:
        """
        Numba-accelerated run length update.
        
        This is the core BOCD algorithm from Adams & MacKay 2007.
        
        Args:
            x_t: New observation
            R_prev: Previous run length distribution
            hazard: Hazard values for each run length
            pred_probs: Predictive probabilities for each run length
            max_run: Maximum run length to track
        
        Returns:
            (new run length distribution, changepoint probability)
        """
        n = len(R_prev)
        R_new = np.zeros(min(n + 1, max_run))
        
        # Growth: run length increases by 1
        # P(r_t = i+1 | data) ∝ P(r_{t-1} = i | data) * (1 - H(i)) * P(x_t | params_i)
        growth = R_prev * (1 - hazard[:n]) * pred_probs[:n]
        
        # Changepoint: run length resets to 0
        # P(r_t = 0 | data) = sum_i P(r_{t-1} = i | data) * H(i) * P(x_t | params_0)
        cp_prob = np.sum(R_prev * hazard[:n] * pred_probs[:n])
        
        # Build new distribution
        R_new[0] = cp_prob
        if n < max_run:
            R_new[1:n+1] = growth
        else:
            R_new[1:] = growth[:-1]
        
        # Normalize
        total = np.sum(R_new)
        if total > 0:
            R_new = R_new / total
        
        return R_new, cp_prob
    
    @jit(nopython=True, cache=True)
    def _student_t_pred_prob(
        x: float,
        mean: float,
        variance: float,
        dof: float
    ) -> float:
        """
        Student-t predictive probability density.
        
        More robust to outliers than Gaussian.
        """
        # Simplified Student-t density
        # p(x | params) ∝ (1 + (x-mean)^2 / (dof*variance))^(-(dof+1)/2)
        scaled_diff = (x - mean) ** 2 / (dof * max(variance, 1e-10))
        log_prob = -(dof + 1) / 2 * np.log(1 + scaled_diff)
        return np.exp(log_prob)


class BayesianChangepointDetector(OnlineModel, StrategyMixin):
    """
    Bayesian Online Changepoint Detection for attack confirmation.
    
    Algorithm (Adams & MacKay 2007):
    1. Maintain distribution P(r_{t-1} | x_{1:t-1}) over run lengths
    2. For new observation x_t:
       - Compute predictive P(x_t | params for each run length)
       - Update: P(r_t | x_{1:t}) using hazard function
    3. If P(r_t = 0 | data) > threshold, declare changepoint
    
    Key decision rule from PLAN.md:
    - Bet when P(|r_t - r_{t-1}| > 0 | Data) > 0.8 AND yesterday's Z-score > 2.0
    """
    
    def __init__(
        self,
        max_run_length: int = 1000,
        hazard_function: str = "weibull",
        hazard_params: Optional[Dict] = None,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        prior_dof: float = 10.0,
        changepoint_threshold: float = 0.8,
        z_score_threshold: float = 2.0,
        use_numba: bool = True
    ):
        """
        Initialize the changepoint detector.
        
        Args:
            max_run_length: Maximum run length to track (truncate for efficiency)
            hazard_function: 'weibull' or 'constant'
            hazard_params: Parameters for hazard function
            prior_mean: Prior mean for observations
            prior_var: Prior variance for observations
            prior_dof: Prior degrees of freedom (Student-t robustness)
            changepoint_threshold: P(r_t = 0) threshold for detection
            z_score_threshold: Historical Z-score threshold for betting
            use_numba: Whether to use Numba JIT acceleration
        """
        self.max_run_length = max_run_length
        self.hazard_function = hazard_function
        self.hazard_params = hazard_params or {'shape': 2.0, 'scale': 100.0}
        
        # Prior parameters (updated online)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_dof = prior_dof
        
        # Decision thresholds
        self.changepoint_threshold = changepoint_threshold
        self.z_score_threshold = z_score_threshold
        
        # State
        self.state = ChangepointState(
            run_length=0,
            changepoint_prob=0.0,
            mean=prior_mean,
            variance=prior_var,
            dof=prior_dof,
            run_length_dist=np.ones(1)  # Start with P(r=0) = 1
        )
        
        # Rider-specific detectors (for multi-rider tracking)
        self.rider_detectors: Dict[int, 'BayesianChangepointDetector'] = {}
        
        # Performance tracking
        self.update_times_ms: List[float] = []
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        # Historical data for Z-score lookup
        self.yesterday_z_scores: Dict[int, float] = {}
        
        logger.info(
            f"Initialized ChangepointDetector: "
            f"threshold={changepoint_threshold}, z_threshold={z_score_threshold}, "
            f"numba={self.use_numba}"
        )
    
    def _get_hazard(self, run_lengths: np.ndarray) -> np.ndarray:
        """Compute hazard for array of run lengths."""
        if self.hazard_function == "weibull":
            shape = self.hazard_params.get('shape', 2.0)
            scale = self.hazard_params.get('scale', 100.0)
            return np.array([weibull_hazard(float(r), shape, scale) for r in run_lengths])
        else:
            lambda_val = self.hazard_params.get('lambda', 250.0)
            return np.full(len(run_lengths), 1.0 / lambda_val)
    
    def _compute_predictive_probs(
        self, 
        x_t: float,
        run_lengths: np.ndarray
    ) -> np.ndarray:
        """
        Compute P(x_t | parameters_for_run_length).
        
        Uses recursive update of mean and variance for each run length.
        """
        n = len(run_lengths)
        probs = np.zeros(n)
        
        # Simple Gaussian predictive (can be extended to Student-t)
        for i, r in enumerate(run_lengths):
            # Run length r means we've seen r observations since changepoint
            # Posterior parameters after r observations
            n_obs = r + 1  # Including prior as 1 pseudo-observation
            
            # Update formula for normal with known variance
            posterior_var = self.prior_var * (1 + 1.0 / max(n_obs, 1))
            
            # Predictive probability (simplified)
            probs[i] = np.exp(-0.5 * ((x_t - self.state.mean) ** 2) / max(posterior_var, 1e-10))
        
        return probs
    
    def update(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update detector with new observation.
        
        Args:
            observation: Dictionary with:
                - 'power_z_score': Normalized power reading
                - 'rider_id': Rider identifier
                - 'timestamp': Observation time
                
        Returns:
            Dictionary with detection results
        """
        import time
        start_time = time.perf_counter()
        
        x_t = observation.get('power_z_score', 0.0)
        rider_id = observation.get('rider_id', 0)
        
        # Run the update
        R_prev = self.state.run_length_dist
        run_lengths = np.arange(len(R_prev))
        
        # Compute hazard and predictive probabilities
        hazard = self._get_hazard(run_lengths)
        pred_probs = self._compute_predictive_probs(x_t, run_lengths)
        
        # Update run length distribution
        if self.use_numba and NUMBA_AVAILABLE:
            R_new, cp_prob = _update_run_length_numba(
                x_t, R_prev, hazard, pred_probs, self.max_run_length
            )
        else:
            R_new, cp_prob = self._update_run_length_python(
                x_t, R_prev, hazard, pred_probs
            )
        
        # Update state
        self.state.run_length_dist = R_new
        self.state.changepoint_prob = cp_prob
        self.state.run_length = int(np.argmax(R_new))  # MAP estimate
        
        # Update sufficient statistics (online mean/variance)
        if cp_prob > 0.5:
            # Changepoint detected - reset statistics
            self.state.mean = x_t
            self.state.variance = self.prior_var
            self.state.dof = self.prior_dof
        else:
            # No changepoint - recursive update
            n = len(R_prev)
            self.state.mean = (n * self.state.mean + x_t) / (n + 1)
            residual = x_t - self.state.mean
            self.state.variance = (
                (n * self.state.variance + residual ** 2) / (n + 1)
            )
        
        # Track history
        self.state.run_length_history.append(self.state.run_length)
        self.state.cp_prob_history.append(cp_prob)
        
        # Trim history if too long
        if len(self.state.run_length_history) > 10000:
            self.state.run_length_history = self.state.run_length_history[-5000:]
            self.state.cp_prob_history = self.state.cp_prob_history[-5000:]
        
        # Track latency
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.update_times_ms.append(elapsed_ms)
        if len(self.update_times_ms) > 1000:
            self.update_times_ms = self.update_times_ms[-500:]
        
        # Generate signal
        signal = self._generate_signal(rider_id, cp_prob)
        
        return {
            'changepoint_prob': cp_prob,
            'run_length': self.state.run_length,
            'signal': signal.name,
            'signal_level': signal.value,
            'latency_ms': elapsed_ms,
            'should_bet': signal.value >= AttackSignal.CONFIRMED.value
        }
    
    def _update_run_length_python(
        self,
        x_t: float,
        R_prev: np.ndarray,
        hazard: np.ndarray,
        pred_probs: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Pure Python implementation of run length update.
        """
        n = len(R_prev)
        R_new = np.zeros(min(n + 1, self.max_run_length))
        
        # Growth
        growth = R_prev * (1 - hazard[:n]) * pred_probs[:n]
        
        # Changepoint
        cp_prob = np.sum(R_prev * hazard[:n] * pred_probs[:n])
        
        # Build new distribution
        R_new[0] = cp_prob
        if n < self.max_run_length:
            R_new[1:n+1] = growth
        else:
            R_new[1:] = growth[:-1]
        
        # Normalize
        total = np.sum(R_new)
        if total > 0:
            R_new = R_new / total
        
        return R_new, cp_prob
    
    def _generate_signal(
        self, 
        rider_id: int, 
        changepoint_prob: float
    ) -> AttackSignal:
        """
        Generate attack signal based on changepoint probability and history.
        
        Decision rule from PLAN.md:
        - Bet when P(|r_t - r_{t-1}| > 0 | Data) > 0.8 AND yesterday's Z-score > 2.0
        """
        # Check if this is a new changepoint (run length reset)
        is_new_cp = changepoint_prob > self.changepoint_threshold
        
        # Get yesterday's Z-score for this rider
        yesterday_z = self.yesterday_z_scores.get(rider_id, 0.0)
        
        if is_new_cp and yesterday_z > self.z_score_threshold:
            return AttackSignal.CONFIRMED
        elif is_new_cp:
            return AttackSignal.SUSPECTED
        elif changepoint_prob > 0.5:
            return AttackSignal.ESTABLISHED
        else:
            return AttackSignal.NONE
    
    def update_yesterday_z_score(self, rider_id: int, z_score: float):
        """Update yesterday's Z-score for a rider (from historical data)."""
        self.yesterday_z_scores[rider_id] = z_score
    
    def get_latency_ms(self) -> float:
        """Return average update latency."""
        if not self.update_times_ms:
            return 0.0
        return np.mean(self.update_times_ms[-100:])
    
    def get_rider_detector(self, rider_id: int) -> 'BayesianChangepointDetector':
        """Get or create detector for specific rider."""
        if rider_id not in self.rider_detectors:
            self.rider_detectors[rider_id] = BayesianChangepointDetector(
                max_run_length=self.max_run_length,
                hazard_function=self.hazard_function,
                hazard_params=self.hazard_params.copy(),
                changepoint_threshold=self.changepoint_threshold,
                z_score_threshold=self.z_score_threshold,
                use_numba=self.use_numba
            )
        return self.rider_detectors[rider_id]
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Not used for online model - use update() instead."""
        return {'changepoint_prob': np.array([self.state.changepoint_prob])}
    
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """
        Calculate betting edge based on attack confirmation.
        
        Strategy: Bet on confirmed attacks in breakaway markets.
        """
        signal_level = prediction.get('signal_level', 0)
        cp_prob = prediction.get('changepoint_prob', 0.0)
        
        if signal_level < AttackSignal.CONFIRMED.value:
            return 0.0
        
        # Estimate win probability based on signal strength
        # Confirmed attack significantly increases breakaway success probability
        win_prob = 0.15 + 0.25 * cp_prob  # 15-40% depending on confidence
        
        if market_odds <= 1:
            return 0.0
        
        implied_prob = 1.0 / market_odds
        edge = win_prob - implied_prob
        
        return edge * 10000
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detector diagnostics."""
        return {
            'mean_latency_ms': self.get_latency_ms(),
            'max_latency_ms': max(self.update_times_ms) if self.update_times_ms else 0,
            'current_run_length': self.state.run_length,
            'current_changepoint_prob': self.state.changepoint_prob,
            'num_updates': len(self.state.run_length_history),
            'numba_enabled': self.use_numba
        }


class PowerZScoreCalculator:
    """
    Calculates power Z-scores for attack detection.
    
    Normalizes power relative to rider's historical distribution.
    """
    
    def __init__(self, window_size: int = 300):
        self.window_size = window_size
        self.rider_stats: Dict[int, Dict] = {}
    
    def update_stats(self, rider_id: int, power: float):
        """Update running statistics for a rider."""
        if rider_id not in self.rider_stats:
            self.rider_stats[rider_id] = {
                'powers': [],
                'mean': power,
                'var': 100.0,  # Initial variance assumption
                'n': 1
            }
        
        stats = self.rider_stats[rider_id]
        stats['powers'].append(power)
        
        # Keep only recent observations
        if len(stats['powers']) > self.window_size:
            stats['powers'] = stats['powers'][-self.window_size:]
        
        # Update online mean and variance (Welford's algorithm)
        n = len(stats['powers'])
        if n > 1:
            old_mean = stats['mean']
            new_mean = old_mean + (power - old_mean) / n
            stats['var'] = stats['var'] + (power - old_mean) * (power - new_mean)
            stats['mean'] = new_mean
            stats['n'] = n
    
    def calculate_z_score(self, rider_id: int, power: float) -> float:
        """Calculate Z-score for a power reading."""
        if rider_id not in self.rider_stats:
            return 0.0
        
        stats = self.rider_stats[rider_id]
        if stats['n'] < 10:
            return 0.0
        
        std = np.sqrt(stats['var'] / max(stats['n'] - 1, 1))
        if std < 1:
            return 0.0
        
        return (power - stats['mean']) / std


# Numba-accelerated function as specified in PLAN.md
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def update_run_length(
        x_t: float,
        R_prev: np.ndarray,
        hazard: float,
        pred_prob: float
    ) -> np.ndarray:
        """
        Numba-optimized run length update (from PLAN.md specification).
        
        This is a simplified version for single observation update.
        """
        growth = R_prev * pred_prob * (1 - hazard)
        cp_prob = np.sum(R_prev * pred_prob * hazard)
        
        R_new = np.zeros_like(R_prev)
        R_new[0] = cp_prob
        R_new[1:] = growth[:-1]
        
        # Normalize
        total = np.sum(R_new)
        if total > 0:
            R_new = R_new / total
        
        return R_new
