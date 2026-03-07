"""Domain layer: entities and enums for the betting engine."""
from .enums import (
    StageType,
    TacticalState,
    RiskType,
    MarketType,
    ConfidenceLevel,
    GamePhase,
    WeatherCondition,
    RecoveryStatus,
    AttackSignal,
    STAGE_TYPE_SPECIALTY_MAP,
    TACTICAL_STATE_PRIORITY,
)

from .entities import (
    RiderPhysicalAttributes,
    RiderSpecialtyScores,
    RiderState,
    StageContext,
    MarketState,
    GapState,
    BreakawayGameState,
    Position,
    Portfolio,
    StrategyOutput,
)

__all__ = [
    # Enums
    "StageType",
    "TacticalState",
    "RiskType",
    "MarketType",
    "ConfidenceLevel",
    "GamePhase",
    "WeatherCondition",
    "RecoveryStatus",
    "AttackSignal",
    "STAGE_TYPE_SPECIALTY_MAP",
    "TACTICAL_STATE_PRIORITY",
    
    # Entities
    "RiderPhysicalAttributes",
    "RiderSpecialtyScores",
    "RiderState",
    "StageContext",
    "MarketState",
    "GapState",
    "BreakawayGameState",
    "Position",
    "Portfolio",
    "StrategyOutput",
]
