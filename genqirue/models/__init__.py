"""Models layer: 15 betting strategies."""
from .base import (
    BayesianModel,
    StrategyMixin,
    LatentStateModel,
    SurvivalModel,
    OnlineModel,
    ModelDiagnostics,
    PosteriorSummary,
)

# Implemented strategies
from .gruppetto_frailty import (
    GruppettoFrailtyModel,
    SurvivalRecord,
    FastFrailtyEstimator,
)

from .online_changepoint import (
    BayesianChangepointDetector,
    ChangepointState,
    PowerObservation,
    PowerZScoreCalculator,
)

from .tactical_hmm import (
    TacticalTimeLossHMM,
    TacticalObservation,
    SimpleTacticalDetector,
)

from .weather_spde import (
    WeatherSPDEModel,
    WeatherObservation,
    ITTStarter,
    SimpleWeatherArbitrage,
)

from .stage_ranker import (
    StageRankingModel,
    RiderSignals,
    StageRankingResult,
)

__all__ = [
    # Base classes
    "BayesianModel",
    "StrategyMixin",
    "LatentStateModel",
    "SurvivalModel",
    "OnlineModel",
    "ModelDiagnostics",
    "PosteriorSummary",
    
    # Strategy 2: Gruppetto Frailty
    "GruppettoFrailtyModel",
    "SurvivalRecord",
    "FastFrailtyEstimator",
    
    # Strategy 12: Online Changepoint
    "BayesianChangepointDetector",
    "ChangepointState",
    "PowerObservation",
    "PowerZScoreCalculator",
    
    # Strategy 1: Tactical HMM
    "TacticalTimeLossHMM",
    "TacticalObservation",
    "SimpleTacticalDetector",
    
    # Strategy 6: Weather SPDE
    "WeatherSPDEModel",
    "WeatherObservation",
    "ITTStarter",
    "SimpleWeatherArbitrage",

    # Stage Ranking Model
    "StageRankingModel",
    "RiderSignals",
    "StageRankingResult",
]
