"""
Base Bayesian Model class for all Genqirue betting strategies.
Provides PyMC infrastructure, posterior storage, and convergence diagnostics.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable
from datetime import datetime
import logging
import warnings

import numpy as np
import pandas as pd

# PyMC imports
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportanceError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian models will not function.")

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """Convergence diagnostics from MCMC sampling."""
    r_hat: float = 1.0
    ess_bulk: float = 0.0
    ess_tail: float = 0.0
    divergences: int = 0
    treedepth_exceeded: int = 0
    effective_sample_size: float = 0.0
    converged: bool = False
    
    def __post_init__(self):
        # Convergence criteria: r_hat < 1.01 and ESS > 400 per chain
        self.converged = (
            self.r_hat < 1.01 and 
            self.ess_bulk >= 400 and 
            self.divergences == 0
        )


@dataclass
class PosteriorSummary:
    """Summary statistics for a posterior distribution."""
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    hdi_3: float = 0.0  # 3% HDI lower bound
    hdi_97: float = 0.0  # 97% HDI upper bound
    hdi_width: float = 0.0
    
    @classmethod
    def from_az_summary(cls, summary_df: pd.DataFrame, var_name: str) -> 'PosteriorSummary':
        """Create from Arviz summary dataframe."""
        if var_name not in summary_df.index:
            return cls()
        row = summary_df.loc[var_name]
        return cls(
            mean=row.get('mean', 0.0),
            std=row.get('sd', 0.0),
            median=row.get('50%', 0.0),
            hdi_3=row.get('hdi_3%', 0.0),
            hdi_97=row.get('hdi_97%', 0.0),
            hdi_width=row.get('hdi_97%', 0.0) - row.get('hdi_3%', 0.0)
        )


class BayesianModel(ABC):
    """
    Abstract base class for all Bayesian betting models.
    
    Each strategy implements:
    - build_model(): Define PyMC model structure
    - fit(): Run MCMC sampling
    - predict(): Generate predictions from posterior
    - get_edge(): Calculate betting edge
    
    Attributes:
        model_name: Unique identifier for this model instance
        model: PyMC model object (created in build_model)
        idata: Arviz InferenceData object with posterior samples
        diagnostics: Model convergence diagnostics
    """
    
    def __init__(
        self, 
        model_name: str,
        random_seed: int = 42,
        mcmc_samples: int = 1000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        target_accept: float = 0.95
    ):
        self.model_name = model_name
        self.random_seed = random_seed
        self.mcmc_samples = mcmc_samples
        self.mcmc_tune = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.target_accept = target_accept
        
        self.model: Optional[pm.Model] = None
        self.idata: Optional[az.InferenceData] = None
        self.diagnostics: ModelDiagnostics = ModelDiagnostics()
        self.posterior_summaries: Dict[str, PosteriorSummary] = {}
        
        self._is_built = False
        self._is_fitted = False
        
        logger.info(f"Initialized {self.__class__.__name__}: {model_name}")
    
    @abstractmethod
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build the PyMC model specification.
        
        Args:
            data: Dictionary containing all data needed for the model
            
        Returns:
            PyMC Model instance
        """
        pass
    
    def fit(self, data: Dict[str, Any], **kwargs) -> 'BayesianModel':
        """
        Fit the model using MCMC sampling.
        
        Args:
            data: Training data dictionary
            **kwargs: Additional arguments passed to pm.sample()
            
        Returns:
            self for method chaining
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC is not available")
        
        logger.info(f"Fitting {self.model_name}...")
        
        # Build model if not already built
        if not self._is_built:
            self.model = self.build_model(data)
            self._is_built = True
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Default sampling kwargs
        sample_kwargs = {
            'draws': self.mcmc_samples,
            'tune': self.mcmc_tune,
            'chains': self.mcmc_chains,
            'target_accept': self.target_accept,
            'random_seed': self.random_seed,
            'return_inferencedata': True,
            'cores': min(4, self.mcmc_chains),
            **kwargs
        }
        
        try:
            with self.model:
                self.idata = pm.sample(**sample_kwargs)
            
            self._is_fitted = True
            self._compute_diagnostics()
            self._summarize_posterior()
            
            logger.info(
                f"Fitting complete. R-hat: {self.diagnostics.r_hat:.4f}, "
                f"ESS: {self.diagnostics.ess_bulk:.0f}, "
                f"Divergences: {self.diagnostics.divergences}"
            )
            
            if not self.diagnostics.converged:
                logger.warning(
                    f"Model may not have converged! "
                    f"R-hat: {self.diagnostics.r_hat:.4f}, "
                    f"Divergences: {self.diagnostics.divergences}"
                )
                
        except Exception as e:
            logger.error(f"MCMC sampling failed: {e}")
            raise
        
        return self
    
    def _compute_diagnostics(self):
        """Compute convergence diagnostics from posterior samples."""
        if self.idata is None:
            return
        
        try:
            summary = az.summary(self.idata)
            
            # Aggregate across all parameters
            r_hats = summary.get('r_hat', pd.Series([1.0]))
            ess_bulk_vals = summary.get('ess_bulk', pd.Series([0.0]))
            ess_tail_vals = summary.get('ess_tail', pd.Series([0.0]))
            
            self.diagnostics = ModelDiagnostics(
                r_hat=float(r_hats.max()),
                ess_bulk=float(ess_bulk_vals.min()),
                ess_tail=float(ess_tail_vals.min()),
                divergences=getattr(self.idata.sample_stats, 'diverging', pd.Series([0])).sum().item(),
                effective_sample_size=float(ess_bulk_vals.sum())
            )
        except Exception as e:
            logger.warning(f"Could not compute diagnostics: {e}")
    
    def _summarize_posterior(self):
        """Create posterior summaries for all parameters."""
        if self.idata is None:
            return
        
        try:
            summary = az.summary(self.idata, hdi_prob=0.94)
            for var_name in summary.index:
                self.posterior_summaries[var_name] = PosteriorSummary.from_az_summary(
                    summary, var_name
                )
        except Exception as e:
            logger.warning(f"Could not summarize posterior: {e}")
    
    @abstractmethod
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate predictions from fitted model.
        
        Args:
            new_data: Dictionary with new observations
            
        Returns:
            Dictionary mapping prediction names to arrays of posterior predictive samples
        """
        pass
    
    @abstractmethod
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """
        Calculate betting edge in basis points.
        
        Args:
            prediction: Output from predict()
            market_odds: Decimal odds from betting market
            
        Returns:
            Edge in basis points (100 bps = 1% edge)
        """
        pass
    
    def posterior_predictive(self, data: Dict[str, Any]) -> az.InferenceData:
        """
        Run posterior predictive check.
        
        Args:
            data: Data for posterior predictive sampling
            
        Returns:
            InferenceData with posterior predictive samples
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before posterior predictive")
        
        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.idata,
                var_names=['obs'] if 'obs' in self.model.named_vars else None,
                random_seed=self.random_seed
            )
        return ppc
    
    def save_posterior(self, filepath: str):
        """Save posterior to NetCDF file."""
        if self.idata is None:
            raise RuntimeError("No posterior to save")
        self.idata.to_netcdf(filepath)
        logger.info(f"Saved posterior to {filepath}")
    
    def load_posterior(self, filepath: str):
        """Load posterior from NetCDF file."""
        self.idata = az.from_netcdf(filepath)
        self._is_fitted = True
        self._compute_diagnostics()
        self._summarize_posterior()
        logger.info(f"Loaded posterior from {filepath}")
    
    def get_param(self, param_name: str) -> Optional[PosteriorSummary]:
        """Get posterior summary for a parameter."""
        return self.posterior_summaries.get(param_name)
    
    def get_posterior_samples(self, var_name: str) -> Optional[np.ndarray]:
        """Get raw posterior samples for a variable."""
        if self.idata is None or 'posterior' not in self.idata:
            return None
        try:
            return self.idata.posterior[var_name].values.flatten()
        except KeyError:
            return None


class StrategyMixin:
    """
    Mixin providing common functionality for betting strategies.
    """
    
    @staticmethod
    def logistic(x: float) -> float:
        """Logistic sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def logit(p: float) -> float:
        """Logit (inverse logistic) function."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))
    
    @staticmethod
    def calculate_ev(prob: float, odds: float) -> float:
        """Calculate expected value of a bet."""
        if odds <= 1:
            return -1.0
        return prob * (odds - 1) - (1 - prob)
    
    @staticmethod
    def kelly_fraction(prob: float, odds: float) -> float:
        """
        Calculate full Kelly fraction.
        f* = (bp - q) / b where b = odds - 1, p = win prob, q = 1 - p
        """
        if odds <= 1:
            return 0.0
        b = odds - 1
        q = 1 - prob
        f = (b * prob - q) / b
        return max(0.0, min(1.0, f))
    
    @staticmethod
    def robust_kelly_fraction(
        prob: float, 
        odds: float, 
        prob_std: float,
        gamma: float = 1.0  # Risk aversion
    ) -> float:
        """
        Robust Kelly fraction accounting for probability uncertainty.
        
        f* = f_kelly * (1 - (prob_std^2 * (b+1)^2) / (p^2 * (b+1)^2))
        
        Args:
            prob: Point estimate of win probability
            odds: Decimal odds
            prob_std: Standard deviation of probability estimate
            gamma: Risk aversion parameter (higher = more conservative)
        """
        if odds <= 1 or prob <= 0:
            return 0.0
        
        b = odds - 1
        f_kelly = (b * prob - (1 - prob)) / b
        
        if f_kelly <= 0:
            return 0.0
        
        # Uncertainty penalty
        penalty = gamma * (prob_std ** 2) * ((b + 1) ** 2) / ((prob ** 2) * ((b + 1) ** 2))
        f_robust = f_kelly * (1 - penalty)
        
        return max(0.0, min(1.0, f_robust))


class LatentStateModel(BayesianModel):
    """
    Base class for models with discrete latent states (HMMs, etc.).
    """
    
    @abstractmethod
    def decode_states(self) -> np.ndarray:
        """
        Decode most likely latent state sequence (Viterbi or MAP).
        
        Returns:
            Array of state indices
        """
        pass
    
    def state_probabilities(self) -> np.ndarray:
        """
        Get posterior state probabilities at each time point.
        
        Returns:
            Array of shape (n_states, n_timesteps)
        """
        pass


class SurvivalModel(BayesianModel):
    """
    Base class for survival/frailty models (Strategies 2, 14).
    """
    
    @abstractmethod
    def compute_frailty(self) -> Dict[int, float]:
        """
        Compute frailty estimates for each rider.
        
        Returns:
            Dictionary mapping rider_id to frailty score
        """
        pass
    
    @abstractmethod
    def cumulative_hazard(self, times: np.ndarray) -> np.ndarray:
        """
        Compute cumulative hazard function.
        
        Args:
            times: Time points
            
        Returns:
            Cumulative hazard at each time point
        """
        pass


class OnlineModel(ABC):
    """
    Interface for real-time/updateable models (Strategies 10-13).
    These models support fast online updates (<100ms latency).
    """
    
    @abstractmethod
    def update(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update model with new observation.
        
        Args:
            observation: New data point
            
        Returns:
            Updated state/predictions
        """
        pass
    
    @abstractmethod
    def get_latency_ms(self) -> float:
        """Return average update latency in milliseconds."""
        pass
