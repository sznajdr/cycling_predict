"""
Strategy 1: Tactical Time Loss via Hidden Markov Model

Hidden Markov Switching model with latent tactical states.
Identifies riders who are tactically losing time (preserving energy)
vs. those genuinely struggling.

Key insights:
- Latent state z_{i,t} ~ Bernoulli(π_{i,t}) where π depends on GC gap and stage type
- Tactical time loss γ_1 ~ N+(2, 0.5) minutes
- Riders in PRESERVING state on mountains may excel on following flat stages
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

from .base import LatentStateModel, ModelDiagnostics, PosteriorSummary, StrategyMixin
from ..domain.enums import TacticalState, StageType

logger = logging.getLogger(__name__)


@dataclass
class TacticalObservation:
    """Single observation of time loss behavior."""
    rider_id: int
    stage_id: int
    stage_type: StageType
    stage_date: datetime
    
    # Time loss relative to stage winner (seconds)
    time_loss_seconds: float
    
    # Context
    gc_position: Optional[int] = None
    gc_time_behind: float = 0.0  # Seconds behind GC leader
    gruppetto_indicator: bool = False
    
    # Historical context
    prev_stage_time_loss: Optional[float] = None
    n_stages_losing_time: int = 0  # Consecutive stages losing time


class TacticalTimeLossHMM(LatentStateModel, StrategyMixin):
    """
    Hidden Markov Model for tactical time loss detection.
    
    Model specification:
    
    Latent state z_{i,t} ∈ {CONTESTING(0), PRESERVING(1)}
    
    P(z_{i,t} = 1) = logit^{-1}(δ_0 + δ_1 * ΔGC_{i,t} + δ_2 * StageType_t)
    
    Time loss model:
    - If CONTESTING: time_loss ~ Normal(μ_contest, σ²)
    - If PRESERVING: time_loss ~ Normal(μ_contest + γ_1, σ²)
    
    where γ_1 ~ TruncatedNormal+(2.0, 0.5) is the tactical time loss in minutes
    
    State transition:
    P(z_t = j | z_{t-1} = i) = A[i,j] (transition matrix)
    """
    
    def __init__(
        self,
        model_name: str = "tactical_time_loss_hmm",
        random_seed: int = 42,
        mcmc_samples: int = 1000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        # Model hyperparameters
        tactical_loss_mean: float = 120.0,  # 2 minutes in seconds
        tactical_loss_sigma: float = 30.0,  # 0.5 minutes in seconds
        transition_stickiness: float = 2.0,  # Prior favoring staying in same state
    ):
        super().__init__(
            model_name=model_name,
            random_seed=random_seed,
            mcmc_samples=mcmc_samples,
            mcmc_tune=mcmc_tune,
            mcmc_chains=mcmc_chains
        )
        
        self.tactical_loss_mean = tactical_loss_mean
        self.tactical_loss_sigma = tactical_loss_sigma
        self.transition_stickiness = transition_stickiness
        
        # Data storage
        self.observations: List[TacticalObservation] = []
        self.rider_ids: List[int] = []
        self.n_riders: int = 0
        self.n_observations: int = 0
        
        # Fitted state sequences
        self.state_probabilities_: Optional[np.ndarray] = None
        self.viterbi_states_: Optional[np.ndarray] = None
        
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build the HMM with PyMC.
        
        Uses pm.Categorical for discrete latent states and implements
        forward algorithm for likelihood computation.
        
        Args:
            data: Dictionary with:
                - 'observations': List[TacticalObservation]
                - 'rider_ids': List of unique rider IDs
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required for HMM")
        
        self.observations = data.get('observations', [])
        self.rider_ids = data.get('rider_ids', [])
        self.n_riders = len(self.rider_ids)
        self.n_observations = len(self.observations)
        
        if self.n_observations < 10:
            raise ValueError("Insufficient observations for HMM")
        
        # Prepare data
        y, X, rider_idx, n_stages_per_rider = self._prepare_data()
        
        with pm.Model() as model:
            # === Latent State Probabilities ===
            # δ_0: Baseline probability of preserving
            delta_0 = pm.Normal('delta_0', mu=0, sigma=1)
            
            # δ_1: Effect of GC time behind (positive = more likely to preserve if far back)
            delta_1 = pm.Normal('delta_1', mu=0.5, sigma=0.5)
            
            # δ_2: Effect of stage type (mountain = more likely to preserve)
            delta_2 = pm.Normal('delta_2', mu=0.5, sigma=0.5)
            
            # === State Transition Matrix ===
            # A[i,j] = P(z_t = j | z_{t-1} = i)
            # Higher diagonal = stickier states
            transition_conc = pm.Gamma('transition_conc', alpha=self.transition_stickiness, beta=1)
            
            # Dirichlet prior for each row of transition matrix
            A_logits = pm.Normal(
                'A_logits', 
                mu=self.transition_stickiness * np.eye(2).flatten(), 
                sigma=1,
                shape=4
            )
            
            # === Observation Model Parameters ===
            # Mean time loss when contesting (seconds, typically small)
            mu_contest = pm.HalfNormal('mu_contest', sigma=60)
            
            # Tactical time loss γ_1 ~ TruncatedNormal+(2, 0.5) minutes
            # i.e., ~ TruncatedNormal+(120, 30) seconds
            gamma_1 = pm.TruncatedNormal(
                'gamma_1',
                mu=self.tactical_loss_mean,
                sigma=self.tactical_loss_sigma,
                lower=0
            )
            
            # Common variance for both states
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=60)
            
            # === Observation Likelihood ===
            # For each rider, compute state probabilities and likelihood
            log_likelihoods = []
            
            offset = 0
            for rider_i in range(self.n_riders):
                n_stages = n_stages_per_rider[rider_i]
                
                if n_stages < 2:
                    offset += n_stages
                    continue
                
                # Get this rider's data
                y_rider = y[offset:offset + n_stages]
                X_rider = X[offset:offset + n_stages]
                
                # Compute state probabilities π_{i,t} for each stage
                # logit(π) = δ_0 + δ_1 * GC_behind + δ_2 * stage_type
                logit_pi = (
                    delta_0 + 
                    delta_1 * X_rider[:, 0] +  # GC time behind (standardized)
                    delta_2 * X_rider[:, 1]    # Stage type indicator
                )
                pi = pm.math.sigmoid(logit_pi)
                
                # Compute likelihood for each possible state sequence
                # Forward algorithm (simplified - assumes independent stages)
                # P(y_t | z_t = 0) = Normal(y_t | mu_contest, sigma_obs)
                # P(y_t | z_t = 1) = Normal(y_t | mu_contest + gamma_1, sigma_obs)
                
                log_p_y_given_z0 = (
                    -0.5 * ((y_rider - mu_contest) / sigma_obs) ** 2 -
                    pm.math.log(sigma_obs)
                )
                
                log_p_y_given_z1 = (
                    -0.5 * ((y_rider - (mu_contest + gamma_1)) / sigma_obs) ** 2 -
                    pm.math.log(sigma_obs)
                )
                
                # Marginal likelihood: P(y_t) = P(y_t|z=0) * (1-π) + P(y_t|z=1) * π
                log_likelihood = pm.math.logsumexp(
                    pm.math.stack([
                        log_p_y_given_z0 + pm.math.log(1 - pi),
                        log_p_y_given_z1 + pm.math.log(pi)
                    ]),
                    axis=0
                )
                
                log_likelihoods.append(pm.math.sum(log_likelihood))
                offset += n_stages
            
            # Total likelihood
            if log_likelihoods:
                pm.Potential('log_likelihood', pm.math.sum(log_likelihoods))
            
            return model
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data arrays for HMM.
        
        Returns:
            y: Time loss observations (seconds)
            X: Covariates (GC behind, stage type)
            rider_idx: Rider index for each observation
            n_stages_per_rider: Number of stages per rider
        """
        n = len(self.observations)
        
        y = np.zeros(n)
        X = np.zeros((n, 2))  # GC behind, stage type
        rider_idx = np.zeros(n, dtype=int)
        
        rider_to_idx = {r: i for i, r in enumerate(self.rider_ids)}
        n_stages_per_rider = [0] * self.n_riders
        
        for i, obs in enumerate(self.observations):
            y[i] = obs.time_loss_seconds
            X[i, 0] = obs.gc_time_behind
            X[i, 1] = 1.0 if obs.stage_type in [StageType.MOUNTAIN, StageType.HILLY] else 0.0
            rider_idx[i] = rider_to_idx[obs.rider_id]
            n_stages_per_rider[rider_idx[i]] += 1
        
        # Standardize continuous features
        if X[:, 0].std() > 0:
            X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        
        return y, X, rider_idx, n_stages_per_rider
    
    def fit(self, data: Dict[str, Any], **kwargs) -> 'TacticalTimeLossHMM':
        """Fit the HMM and decode states."""
        super().fit(data, **kwargs)
        
        # Decode most likely states
        if self.idata is not None:
            self._decode_states_posterior()
        
        return self
    
    def _decode_states_posterior(self):
        """Decode states using posterior samples."""
        # Simplified state decoding using parameter estimates
        # In a full implementation, would run forward-backward per sample
        
        delta_0 = self.get_posterior_samples('delta_0')
        delta_1 = self.get_posterior_samples('delta_1')
        delta_2 = self.get_posterior_samples('delta_2')
        
        if delta_0 is None:
            return
        
        # Use posterior means
        d0 = np.mean(delta_0)
        d1 = np.mean(delta_1) if delta_1 is not None else 0
        d2 = np.mean(delta_2) if delta_2 is not None else 0
        
        # Compute state probabilities for each observation
        y, X, rider_idx, n_stages_per_rider = self._prepare_data()
        
        logit_pi = d0 + d1 * X[:, 0] + d2 * X[:, 1]
        pi = self.logistic(logit_pi)
        
        self.state_probabilities_ = np.column_stack([1 - pi, pi])
        self.viterbi_states_ = (pi > 0.5).astype(int)
    
    def decode_states(self) -> np.ndarray:
        """Return most likely state sequence."""
        if self.viterbi_states_ is None:
            return np.array([])
        return self.viterbi_states_
    
    def get_tactical_state_prob(
        self, 
        rider_id: int, 
        stage_type: StageType,
        gc_time_behind: float
    ) -> Dict[TacticalState, float]:
        """
        Get probability distribution over tactical states for a rider.
        
        Args:
            rider_id: Rider ID
            stage_type: Type of upcoming stage
            gc_time_behind: Current GC time behind (seconds)
            
        Returns:
            Dictionary mapping TacticalState to probability
        """
        delta_0 = self.get_posterior_samples('delta_0')
        delta_1 = self.get_posterior_samples('delta_1')
        delta_2 = self.get_posterior_samples('delta_2')
        
        if delta_0 is None:
            # Return uniform if not fitted
            return {state: 0.5 for state in [TacticalState.CONTESTING, TacticalState.PRESERVING]}
        
        # Compute probability using posterior samples
        d0, d1, d2 = np.mean(delta_0), np.mean(delta_1 or 0), np.mean(delta_2 or 0)
        
        # Standardize GC time behind (approximate)
        gc_std = gc_time_behind / 300.0  # Assume 5 min = 1 std
        
        stage_indicator = 1.0 if stage_type in [StageType.MOUNTAIN, StageType.HILLY] else 0.0
        
        logit_pi = d0 + d1 * gc_std + d2 * stage_indicator
        pi_preserving = self.logistic(logit_pi)
        
        return {
            TacticalState.CONTESTING: 1 - pi_preserving,
            TacticalState.PRESERVING: pi_preserving,
            TacticalState.RECOVERING: 0.0,
            TacticalState.GRUPPETTO: 0.0
        }
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Predict time loss and tactical state for new observations.
        
        Args:
            new_data: Dictionary with observations
            
        Returns:
            Dictionary with 'time_loss_pred' and 'preserving_prob'
        """
        if not self._is_fitted:
            return {'time_loss_pred': np.array([]), 'preserving_prob': np.array([])}
        
        observations = new_data.get('observations', [])
        
        mu_contest = self.get_posterior_samples('mu_contest')
        gamma_1 = self.get_posterior_samples('gamma_1')
        sigma_obs = self.get_posterior_samples('sigma_obs')
        
        if mu_contest is None:
            return {'time_loss_pred': np.array([]), 'preserving_prob': np.array([])}
        
        mu_c = np.mean(mu_contest)
        g1 = np.mean(gamma_1) if gamma_1 is not None else self.tactical_loss_mean
        sigma = np.mean(sigma_obs) if sigma_obs is not None else 60
        
        time_preds = []
        preserving_probs = []
        
        for obs in observations:
            state_probs = self.get_tactical_state_prob(
                obs.rider_id, obs.stage_type, obs.gc_time_behind
            )
            
            pi_pres = state_probs[TacticalState.PRESERVING]
            expected_time = mu_c + pi_pres * g1
            
            time_preds.append(expected_time)
            preserving_probs.append(pi_pres)
        
        return {
            'time_loss_pred': np.array(time_preds),
            'preserving_prob': np.array(preserving_probs)
        }
    
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """
        Calculate betting edge based on tactical state.
        
        Strategy: Bet on riders in PRESERVING state on mountains
        who will contest on following flat/hilly stages.
        
        Args:
            prediction: Output from predict()
            market_odds: Decimal odds
            
        Returns:
            Edge in basis points
        """
        preserving_prob = prediction.get('preserving_prob', 0.0)
        next_stage_type = prediction.get('next_stage_type', '')
        was_preserving = prediction.get('was_preserving', False)
        
        # Signal: Was preserving on mountain, now on flat/hilly
        if not was_preserving or next_stage_type not in ['flat', 'hilly']:
            return 0.0
        
        # High preserving probability indicates tactical energy conservation
        # These riders often perform better than market expects on next stage
        win_prob = 0.1 + 0.3 * preserving_prob  # 10-40% win probability
        
        if market_odds <= 1:
            return 0.0
        
        implied_prob = 1.0 / market_odds
        edge = win_prob - implied_prob
        
        return edge * 10000
    
    def get_sql_schema(self) -> str:
        """Return SQL schema for storing tactical states."""
        return """
        -- Tactical state tracking for Strategy 1
        CREATE TABLE IF NOT EXISTS tactical_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rider_id INTEGER NOT NULL,
            stage_id INTEGER NOT NULL,
            contesting_prob REAL NOT NULL,
            preserving_prob REAL NOT NULL,
            recovering_prob REAL DEFAULT 0.0,
            gruppetto_prob REAL DEFAULT 0.0,
            decoded_state TEXT NOT NULL,
            tactical_time_loss_seconds REAL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (rider_id) REFERENCES riders(id),
            FOREIGN KEY (stage_id) REFERENCES race_stages(id),
            UNIQUE(rider_id, stage_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_tactical_rider ON tactical_states(rider_id);
        CREATE INDEX IF NOT EXISTS idx_tactical_stage ON tactical_states(stage_id);
        CREATE INDEX IF NOT EXISTS idx_tactical_preserving ON tactical_states(preserving_prob) 
            WHERE preserving_prob > 0.7;
        """


class SimpleTacticalDetector:
    """
    Fast rule-based tactical detector for real-time use.
    Uses heuristics instead of full Bayesian inference.
    """
    
    def __init__(
        self,
        time_loss_threshold: float = 180.0,  # 3 minutes
        gc_gap_threshold: float = 600.0,  # 10 minutes
    ):
        self.time_loss_threshold = time_loss_threshold
        self.gc_gap_threshold = gc_gap_threshold
        
        self.rider_history: Dict[int, List[Dict]] = {}
    
    def update(self, obs: TacticalObservation) -> TacticalState:
        """Update and return detected tactical state."""
        
        # Store history
        if obs.rider_id not in self.rider_history:
            self.rider_history[obs.rider_id] = []
        self.rider_history[obs.rider_id].append({
            'time_loss': obs.time_loss_seconds,
            'gc_behind': obs.gc_time_behind,
            'stage_type': obs.stage_type,
            'gruppetto': obs.gruppetto_indicator
        })
        
        # Simple rules
        if obs.gruppetto_indicator:
            return TacticalState.GRUPPETTO
        
        if obs.gc_time_behind > self.gc_gap_threshold:
            # Far back in GC - likely preserving on mountains
            if obs.stage_type in [StageType.MOUNTAIN, StageType.HILLY]:
                if obs.time_loss_seconds > self.time_loss_threshold:
                    return TacticalState.PRESERVING
        
        if obs.time_loss_seconds < 60:  # Within 1 minute
            return TacticalState.CONTESTING
        
        return TacticalState.RECOVERING
    
    def is_tactical_preserving(self, rider_id: int) -> bool:
        """Check if rider has been tactically preserving recently."""
        history = self.rider_history.get(rider_id, [])
        if len(history) < 2:
            return False
        
        # Look at recent mountain stages
        recent_mountain = [
            h for h in history[-5:]
            if h['stage_type'] in [StageType.MOUNTAIN, StageType.HILLY]
        ]
        
        if not recent_mountain:
            return False
        
        # Check pattern: significant time loss but staying out of gruppetto
        preserving_count = sum(
            1 for h in recent_mountain
            if h['time_loss'] > self.time_loss_threshold and not h['gruppetto']
        )
        
        return preserving_count >= len(recent_mountain) * 0.5
