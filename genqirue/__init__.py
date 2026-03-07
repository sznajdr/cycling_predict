"""
Genqirue: Bayesian Cycling Betting Engine

A production-grade betting intelligence system implementing 15 research-grade
statistical models for professional cycling (Grand Tours) betting.

Quick Start:
    from genqirue.models.gruppetto_frailty import GruppettoFrailtyModel
    from genqirue.portfolio.kelly import RobustKellyOptimizer
    
    # Strategy 2: Frailty detection (START HERE)
    model = GruppettoFrailtyModel()
    model.fit(data)
    frailty = model.compute_frailty()
    
    # Portfolio optimization
    optimizer = RobustKellyOptimizer()
    portfolio = optimizer.optimize_portfolio(positions)

Architecture:
    genqirue/
    ├── domain/      # Entities and enums
    ├── models/      # 15 betting strategies
    ├── portfolio/   # Kelly optimization, risk management
    ├── data/        # Schema and ETL
    ├── inference/   # Real-time and batch inference
    ├── execution/   # Bet placement
    └── validation/  # Backtesting and scoring
"""

__version__ = "0.1.0"
__author__ = "Genqirue Team"

# Core exports
from .domain.enums import (
    StageType,
    TacticalState,
    RiskType,
    MarketType,
    ConfidenceLevel,
    GamePhase,
    WeatherCondition,
)

from .domain.entities import (
    RiderState,
    StageContext,
    MarketState,
    Position,
    Portfolio,
    StrategyOutput,
)

from .models.base import (
    BayesianModel,
    StrategyMixin,
    LatentStateModel,
    SurvivalModel,
    OnlineModel,
)

# Strategy exports (in recommended implementation order)
from .models.gruppetto_frailty import GruppettoFrailtyModel, FastFrailtyEstimator
from .models.online_changepoint import BayesianChangepointDetector
from .models.tactical_hmm import TacticalTimeLossHMM
from .models.weather_spde import WeatherSPDEModel

from .portfolio.kelly import (
    RobustKellyOptimizer,
    KellyParameters,
    SizingMethod,
)

__all__ = [
    # Version
    "__version__",
    
    # Enums
    "StageType",
    "TacticalState", 
    "RiskType",
    "MarketType",
    "ConfidenceLevel",
    "GamePhase",
    "WeatherCondition",
    
    # Entities
    "RiderState",
    "StageContext",
    "MarketState",
    "Position",
    "Portfolio",
    "StrategyOutput",
    
    # Base classes
    "BayesianModel",
    "StrategyMixin",
    "LatentStateModel",
    "SurvivalModel",
    "OnlineModel",
    
    # Strategy models (priority order)
    "GruppettoFrailtyModel",
    "FastFrailtyEstimator",
    "BayesianChangepointDetector",
    "TacticalTimeLossHMM",
    "WeatherSPDEModel",
    
    # Portfolio
    "RobustKellyOptimizer",
    "KellyParameters",
    "SizingMethod",
]
