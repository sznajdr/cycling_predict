"""
Domain enums for the Genqirue betting engine.
Defines all categorical types used across the 15 betting strategies.
"""
from enum import Enum, auto
from typing import Dict, List


class StageType(Enum):
    """Classification of stage parcours characteristics."""
    FLAT = "flat"
    HILLY = "hilly"
    MOUNTAIN = "mountain"
    ITT = "itt"  # Individual Time Trial
    TTT = "ttt"  # Team Time Trial
    PROLOGUE = "prologue"
    ROAD = "road"  # Generic road stage
    COBBLES = "cobbles"  # Cobbled classics


class TacticalState(Enum):
    """
    Latent tactical states for Strategy 1 (Tactical Time Loss HMM).
    Represents hidden rider intentions during a stage.
    """
    CONTESTING = auto()  # Riding full gas, trying to win
    PRESERVING = auto()  # Saving energy, GC protection mode
    RECOVERING = auto()  # Post-crash/medical recovery
    GRUPPETTO = auto()   # Outlier group, time cut risk


class RiskType(Enum):
    """
    Risk categories for joint frailty models (Strategy 14).
    """
    DESCENT = "descent"
    CORNER = "corner"
    WET = "wet"
    COBBLES = "cobbles"
    MECHANICAL = "mechanical"
    CRASH = "crash"


class MarketType(Enum):
    """Betting market types supported by the engine."""
    WINNER = "winner"  # Stage/race winner
    TOP_3 = "top_3"  # Podium finish
    TOP_10 = "top_10"  # Top 10 finish
    H2H = "h2h"  # Head-to-head matchup
    GC_POSITION = "gc_position"  # Final GC position
    KOM = "kom"  # Mountains classification
    POINTS = "points"  # Points/sprint classification
    COMBATIVITY = "combativity"  # Most aggressive rider
    BREAKAWAY = "breakaway"  # Makes the breakaway


class ConfidenceLevel(Enum):
    """
    Confidence tiers for bet sizing (mapped to Kelly fractions).
    """
    EXTREME = 0.95  # Very high confidence
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.55
    SPECULATIVE = 0.40


class GamePhase(Enum):
    """
    Game-theoretic phase for breakaway games (Strategy 8).
    """
    EARLY = "early"  # First third of stage
    MIDDLE = "middle"  # Middle third
    LATE = "late"  # Final third/decisive phase
    SPRINT = "sprint"  # Final 5km


class WeatherCondition(Enum):
    """
    Weather states affecting rider performance.
    """
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    WINDY = "windy"
    CROSSWIND = "crosswind"
    HOT = "hot"  # >30°C
    COLD = "cold"  # <10°C


class RecoveryStatus(Enum):
    """
    Post-incident recovery states (Strategy 3, 14).
    """
    OPTIMAL = 1.0
    GOOD = 0.9
    MODERATE = 0.75
    IMPAIRED = 0.55
    CRITICAL = 0.35


class AttackSignal(Enum):
    """
    Changepoint detection signals (Strategy 12).
    """
    NONE = 0
    SUSPECTED = 1  # Initial power surge
    CONFIRMED = 2  # Gap opening
    ESTABLISHED = 3  # Breakaway confirmed


# Utility mappings
STAGE_TYPE_SPECIALTY_MAP: Dict[StageType, str] = {
    StageType.FLAT: "sp_sprint",
    StageType.HILLY: "sp_hills",
    StageType.MOUNTAIN: "sp_climber",
    StageType.ITT: "sp_time_trial",
    StageType.TTT: "sp_time_trial",
    StageType.PROLOGUE: "sp_time_trial",
    StageType.ROAD: "sp_gc",
    StageType.COBBLES: "sp_one_day_races",
}

# Priority ordering for tactical states (lower = more aggressive)
TACTICAL_STATE_PRIORITY = {
    TacticalState.CONTESTING: 1,
    TacticalState.PRESERVING: 2,
    TacticalState.RECOVERING: 3,
    TacticalState.GRUPPETTO: 4,
}
