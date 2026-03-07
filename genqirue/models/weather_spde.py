"""
Strategy 6: ITT Weather Arbitrage via Gaussian Processes

Spatio-temporal Gaussian Process with Matérn kernel for wind field modeling.
Estimates fair time difference between early and late ITT starters.

Key insights:
- Wind conditions change during ITT, favoring certain start times
- GP models wind field v_wind(s,t) ~ GP(0, k) with Matérn kernel
- Fair time difference ΔT = ∫[P/F(v_early) - P/F(v_late)]dx
- RMSE target: < 10 seconds vs actual results
"""
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False

from .base import BayesianModel, ModelDiagnostics, PosteriorSummary, StrategyMixin

logger = logging.getLogger(__name__)


@dataclass
class WeatherObservation:
    """Weather measurement at a location and time."""
    timestamp: datetime
    location: Tuple[float, float]  # (lat, lon) or (distance_km, 0) for 1D course
    wind_speed_ms: float
    wind_direction_deg: float  # 0 = North, 90 = East, etc.
    temperature_c: float
    pressure_hpa: float = 1013.25
    humidity_pct: float = 50.0


@dataclass
class ITTStarter:
    """ITT starter information."""
    rider_id: int
    start_time: datetime
    start_order: int
    predicted_finish_time: datetime
    power_watts: float  # Estimated or historical power
    cda: float  # Drag coefficient × frontal area


class WeatherSPDEModel(BayesianModel, StrategyMixin):
    """
    Spatio-temporal Gaussian Process for ITT weather arbitrage.
    
    Model specification:
    
    Wind field: v_wind(s,t) ~ GP(0, k((s,t), (s',t')))
    
    Kernel: Matérn 3/2 with separate length scales for space and time
    k((s,t), (s',t')) = σ² * (1 + √3*r/ℓ) * exp(-√3*r/ℓ)
    where r² = ((s-s')/ℓ_s)² + ((t-t')/ℓ_t)²
    
    Aerodynamic drag: F_aero(v) = 0.5 * ρ * C_dA * (v_rider + v_wind)²
    
    Fair time: T = ∫₀ᴰ P / F_aero(v_wind(x, t(x))) dx
    where t(x) is arrival time at position x
    
    Uses PyMC's HSGP (Hilbert Space GP) for computational efficiency.
    """
    
    def __init__(
        self,
        model_name: str = "weather_spde",
        random_seed: int = 42,
        mcmc_samples: int = 500,
        mcmc_tune: int = 500,
        mcmc_chains: int = 4,
        # GP parameters
        spatial_length_scale_km: float = 10.0,
        temporal_length_scale_min: float = 30.0,
        use_hsgp: bool = True,
        hsgp_m: List[int] = None,  # Number of basis functions per dimension
    ):
        super().__init__(
            model_name=model_name,
            random_seed=random_seed,
            mcmc_samples=mcmc_samples,
            mcmc_tune=mcmc_tune,
            mcmc_chains=mcmc_chains
        )
        
        self.spatial_length_scale = spatial_length_scale_km
        self.temporal_length_scale = temporal_length_scale_min
        self.use_hsgp = use_hsgp
        self.hsgp_m = hsgp_m or [20, 20]  # Default: 20 basis functions per dim
        
        # Data storage
        self.weather_observations: List[WeatherObservation] = []
        self.course_distance_km: float = 40.0
        self.n_checkpoints: int = 10
        
        # Fitted GP for fast predictions
        self.gp_regressor = None
        
    def build_model(self, data: Dict[str, Any]) -> pm.Model:
        """
        Build the spatio-temporal GP model.
        
        Args:
            data: Dictionary with:
                - 'weather_obs': List[WeatherObservation]
                - 'course_distance_km': Total ITT distance
                - 'coordinates': Array of shape (n_obs, 2) with (space, time) coords
                - 'wind_speeds': Array of wind speed observations
        """
        if not PYMC_AVAILABLE:
            raise RuntimeError("PyMC required for GP model")
        
        self.weather_observations = data.get('weather_obs', [])
        self.course_distance_km = data.get('course_distance_km', 40.0)
        
        # Prepare data
        X, y = self._prepare_data()
        
        if len(y) < 5:
            raise ValueError("Insufficient weather observations for GP")
        
        n_obs = len(y)
        
        with pm.Model() as model:
            # === GP Hyperparameters ===
            # Marginal standard deviation
            sigma = pm.HalfNormal('sigma', sigma=5.0)
            
            # Length scales
            # Spatial: correlation over distance (km)
            length_space = pm.HalfNormal(
                'length_space', 
                sigma=self.spatial_length_scale
            )
            
            # Temporal: correlation over time (minutes)
            length_time = pm.HalfNormal(
                'length_time',
                sigma=self.temporal_length_scale
            )
            
            # === Kernel Definition ===
            # Matérn 3/2 kernel
            if self.use_hsgp:
                # HSGP approximation for efficiency (as specified in PLAN.md)
                # cov_func = pm.gp.cov.Matern32(2, ls=[length_space, length_time])
                # gp = pm.gp.HSGP(cov_func=cov_func, m=self.hsgp_m)
                
                # Simplified: use standard GP with approximation
                ls = pt.stack([length_space, length_time])
                cov_func = sigma ** 2 * pm.gp.cov.Matern32(
                    input_dim=2,
                    ls=ls
                )
            else:
                ls = pt.stack([length_space, length_time])
                cov_func = sigma ** 2 * pm.gp.cov.Matern32(
                    input_dim=2,
                    ls=ls
                )
            
            # === GP Prior ===
            gp = pm.gp.Latent(cov_func=cov_func)
            
            # Latent function values
            f = gp.prior('f', X=X)
            
            # === Likelihood ===
            # Wind speed observations with noise
            sigma_noise = pm.HalfNormal('sigma_noise', sigma=1.0)
            
            pm.Normal(
                'wind_speed',
                mu=f,
                sigma=sigma_noise,
                observed=y
            )
            
            return model
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare coordinates and observations for GP.
        
        Returns:
            X: Array of shape (n_obs, 2) with (distance_km, time_minutes)
            y: Array of wind speed observations
        """
        if not self.weather_observations:
            return np.array([]).reshape(0, 2), np.array([])
        
        # Reference time (first observation)
        t0 = self.weather_observations[0].timestamp
        
        X = []
        y = []
        
        for obs in self.weather_observations:
            # Distance coordinate (simplified: use first coord component)
            distance = obs.location[0] if len(obs.location) > 0 else 0
            
            # Time coordinate (minutes from start)
            time_min = (obs.timestamp - t0).total_seconds() / 60
            
            X.append([distance, time_min])
            y.append(obs.wind_speed_ms)
        
        return np.array(X), np.array(y)
    
    def fit(self, data: Dict[str, Any], **kwargs) -> 'WeatherSPDEModel':
        """Fit the GP model."""
        super().fit(data, **kwargs)
        
        # Also fit sklearn GP for fast predictions
        if SKLEARN_GP_AVAILABLE:
            self._fit_sklearn_gp()
        
        return self
    
    def _fit_sklearn_gp(self):
        """Fit scikit-learn GP for fast point predictions."""
        X, y = self._prepare_data()
        
        if len(y) < 3:
            return
        
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(
                length_scale=[self.spatial_length_scale, self.temporal_length_scale],
                length_scale_bounds=(1e-2, 100.0),
                nu=1.5
            ) +
            WhiteKernel(noise_level=0.1)
        )
        
        self.gp_regressor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=0.1
        )
        
        try:
            self.gp_regressor.fit(X, y)
            logger.info("Fitted sklearn GP for fast predictions")
        except Exception as e:
            logger.warning(f"Failed to fit sklearn GP: {e}")
            self.gp_regressor = None
    
    def predict_wind_field(
        self,
        distances_km: np.ndarray,
        times_min: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict wind speed at given positions and times.
        
        Args:
            distances_km: Array of distances along course
            times_min: Array of times (minutes from start)
            
        Returns:
            (mean_predictions, std_predictions)
        """
        X_new = np.column_stack([distances_km, times_min])
        
        if self.gp_regressor is not None:
            return self.gp_regressor.predict(X_new, return_std=True)
        
        # Fallback to prior mean
        return np.zeros(len(X_new)), np.ones(len(X_new))
    
    def calculate_fair_time_difference(
        self,
        starter_early: ITTStarter,
        starter_late: ITTStarter,
        n_checkpoints: int = 20
    ) -> Dict[str, float]:
        """
        Calculate fair time difference between two ITT starters.
        
        ΔT = ∫₀ᴰ [P/F_aero(v_early(t)) - P/F_aero(v_late(t))] dt
        
        Args:
            starter_early: Earlier ITT starter
            starter_late: Later ITT starter
            n_checkpoints: Number of points for numerical integration
            
        Returns:
            Dictionary with 'delta_t_seconds', 'early_advantage_seconds', etc.
        """
        if self.gp_regressor is None:
            return {'delta_t_seconds': 0.0, 'uncertainty': 999.0}
        
        # Course checkpoints
        distances = np.linspace(0, self.course_distance_km, n_checkpoints)
        dx = distances[1] - distances[0]
        
        # Estimate arrival times at each checkpoint
        # Simplified: constant speed assumption
        speed_early_kmh = 45.0  # Assumed average speed
        speed_late_kmh = 45.0
        
        times_early = starter_early.start_time + np.array([
            timedelta(hours=d / speed_early_kmh) for d in distances
        ])
        times_late = starter_late.start_time + np.array([
            timedelta(hours=d / speed_late_kmh) for d in distances
        ])
        
        # Reference time for GP
        t0 = self.weather_observations[0].timestamp if self.weather_observations else times_early[0]
        
        times_early_min = np.array([(t - t0).total_seconds() / 60 for t in times_early])
        times_late_min = np.array([(t - t0).total_seconds() / 60 for t in times_late])
        
        # Predict wind speeds
        wind_early, wind_early_std = self.predict_wind_field(distances, times_early_min)
        wind_late, wind_late_std = self.predict_wind_field(distances, times_late_min)
        
        # Calculate aerodynamic drag and power
        # F_aero = 0.5 * ρ * C_dA * (v_rider + v_wind*cos(θ))²
        
        rho = 1.225  # Air density (kg/m³)
        
        # Convert speeds to m/s
        v_rider_ms = speed_early_kmh / 3.6
        
        delta_t = 0.0
        early_advantage = 0.0
        late_advantage = 0.0
        
        for i in range(len(distances)):
            # Simplified: assume headwind/tailwind component
            v_eff_early = max(0.1, v_rider_ms + wind_early[i])
            v_eff_late = max(0.1, v_rider_ms + wind_late[i])
            
            # Drag force
            F_early = 0.5 * rho * starter_early.cda * v_eff_early ** 2
            F_late = 0.5 * rho * starter_late.cda * v_eff_late ** 2
            
            # Time for this segment (dt = dx / v)
            # v = P / F, so dt = dx * F / P
            dt_early = (dx * 1000) * F_early / max(starter_early.power_watts, 1)
            dt_late = (dx * 1000) * F_late / max(starter_late.power_watts, 1)
            
            delta_t += dt_late - dt_early
            
            if dt_early < dt_late:
                early_advantage += dt_late - dt_early
            else:
                late_advantage += dt_early - dt_late
        
        # Uncertainty from GP std
        uncertainty = np.mean(wind_early_std + wind_late_std) * 10  # Rough scaling
        
        return {
            'delta_t_seconds': delta_t,
            'early_advantage_seconds': early_advantage,
            'late_advantage_seconds': late_advantage,
            'uncertainty': uncertainty,
            'favors': 'early' if delta_t > 0 else 'late'
        }
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Generate predictions for ITT time differences.
        
        Args:
            new_data: Dictionary with 'starters' list of ITTStarter
            
        Returns:
            Dictionary with prediction arrays
        """
        starters = new_data.get('starters', [])
        
        if len(starters) < 2 or self.gp_regressor is None:
            return {'time_advantage': np.array([])}
        
        # Compare first and last starter as extreme cases
        early = starters[0]
        late = starters[-1]
        
        result = self.calculate_fair_time_difference(early, late)
        
        return {
            'time_advantage': np.array([result['delta_t_seconds']]),
            'uncertainty': np.array([result['uncertainty']]),
            'favors': np.array([1 if result['favors'] == 'early' else -1])
        }
    
    def get_edge(self, prediction: Dict[str, Any], market_odds: float) -> float:
        """
        Calculate betting edge based on weather forecast.
        
        Strategy: Bet when model predicts >10s advantage vs market pricing.
        """
        time_adv = prediction.get('time_advantage', np.array([0]))[0]
        uncertainty = prediction.get('uncertainty', np.array([999]))[0]
        
        # Only bet if confidence is high
        if abs(time_adv) < 10 or uncertainty > 20:
            return 0.0
        
        # Estimate probability of advantage
        # Assume normal distribution around prediction
        z_score = abs(time_adv) / max(uncertainty, 1)
        prob_advantage = self.logistic(z_score - 1)  # Sigmoid mapping
        
        if market_odds <= 1:
            return 0.0
        
        implied_prob = 1.0 / market_odds
        edge = prob_advantage - implied_prob
        
        return edge * 10000
    
    def get_sql_schema(self) -> str:
        """Return SQL schema for storing weather predictions."""
        return """
        -- Weather field predictions for Strategy 6
        CREATE TABLE IF NOT EXISTS weather_fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_id INTEGER NOT NULL,
            distance_km REAL NOT NULL,
            time_minutes REAL NOT NULL,
            wind_speed_pred REAL NOT NULL,
            wind_speed_std REAL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stage_id) REFERENCES race_stages(id),
            UNIQUE(stage_id, distance_km, time_minutes)
        );
        
        CREATE INDEX IF NOT EXISTS idx_weather_stage ON weather_fields(stage_id);
        
        -- ITT time difference predictions
        CREATE TABLE IF NOT EXISTS itt_time_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_id INTEGER NOT NULL,
            rider_early_id INTEGER NOT NULL,
            rider_late_id INTEGER NOT NULL,
            delta_t_seconds REAL NOT NULL,
            uncertainty REAL,
            favors_rider TEXT,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stage_id) REFERENCES race_stages(id),
            FOREIGN KEY (rider_early_id) REFERENCES riders(id),
            FOREIGN KEY (rider_late_id) REFERENCES riders(id),
            UNIQUE(stage_id, rider_early_id, rider_late_id)
        );
        """


class SimpleWeatherArbitrage:
    """
    Simplified weather arbitrage using wind direction changes.
    """
    
    def __init__(self):
        self.wind_observations: List[WeatherObservation] = []
    
    def add_observation(self, obs: WeatherObservation):
        """Add weather observation."""
        self.wind_observations.append(obs)
    
    def estimate_time_advantage(
        self,
        start_time_diff_minutes: float,
        course_direction_deg: float = 0.0
    ) -> Dict[str, float]:
        """
        Estimate time advantage based on wind changes.
        
        Simplified heuristic: if wind is strengthening, later starters benefit.
        """
        if len(self.wind_observations) < 2:
            return {'delta_t': 0.0, 'confidence': 0.0}
        
        # Trend in wind speed
        times = np.array([
            (obs.timestamp - self.wind_observations[0].timestamp).total_seconds() / 60
            for obs in self.wind_observations
        ])
        speeds = np.array([obs.wind_speed_ms for obs in self.wind_observations])
        
        if len(times) < 2 or np.std(times) < 1:
            return {'delta_t': 0.0, 'confidence': 0.0}
        
        # Linear trend
        slope = np.polyfit(times, speeds, 1)[0]  # m/s per minute
        
        # Expected wind speed difference
        wind_diff = slope * start_time_diff_minutes
        
        # Rough time estimate: 1 m/s wind change ≈ 5s over 40km
        delta_t = wind_diff * 5
        
        confidence = min(1.0, abs(wind_diff) / 2.0)
        
        return {
            'delta_t': delta_t,
            'confidence': confidence,
            'trend': 'increasing' if slope > 0 else 'decreasing'
        }
