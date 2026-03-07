"""
Domain entities for the Genqirue betting engine.
Core data structures representing riders, stages, markets, and system state.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from .enums import (
    StageType, TacticalState, RiskType, MarketType, 
    ConfidenceLevel, GamePhase, WeatherCondition, RecoveryStatus, AttackSignal
)


@dataclass
class RiderPhysicalAttributes:
    """Physical characteristics affecting performance."""
    height_m: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = field(init=False)
    
    def __post_init__(self):
        if self.height_m and self.weight_kg and self.height_m > 0:
            self.bmi = self.weight_kg / (self.height_m ** 2)
        else:
            self.bmi = None


@dataclass
class RiderSpecialtyScores:
    """PCS specialty scores (0-100) for rider profiling."""
    one_day_races: int = 50
    gc: int = 50
    time_trial: int = 50
    sprint: int = 50
    climber: int = 50
    hills: int = 50
    
    def get_score_for_stage_type(self, stage_type: StageType) -> int:
        """Get relevant specialty score for stage type."""
        from .enums import STAGE_TYPE_SPECIALTY_MAP
        attr = STAGE_TYPE_SPECIALTY_MAP.get(stage_type, "gc")
        return getattr(self, attr, 50)


@dataclass
class RiderState:
    """
    Complete rider state at a point in time.
    Used across all strategies for decision making.
    """
    # Identity
    rider_id: int
    pcs_url: str
    name: str
    nationality: str
    birthdate: Optional[str] = None
    
    # Physical
    physical: RiderPhysicalAttributes = field(default_factory=RiderPhysicalAttributes)
    specialties: RiderSpecialtyScores = field(default_factory=RiderSpecialtyScores)
    
    # Dynamic state (updated per race/stage)
    age: float = 25.0  # Current age in years
    current_team_id: Optional[int] = None
    current_team_class: Optional[str] = None
    
    # Form and fatigue indicators
    frailty_estimate: float = 0.0  # From Strategy 2 (Gruppetto)
    hidden_form_prob: float = 0.0  # Probability of hidden good form
    recovery_factor: float = 1.0  # From Strategy 3 (Medical PK)
    fatigue_index: float = 0.0  # Cumulative fatigue (0-1)
    
    # Tactical state (Strategy 1 HMM)
    tactical_state: TacticalState = TacticalState.CONTESTING
    tactical_state_probs: Dict[TacticalState, float] = field(default_factory=dict)
    
    # Changepoint detection (Strategy 12)
    attack_signal: AttackSignal = field(init=False)
    run_length: int = 0  # Time since last changepoint
    changepoint_prob: float = 0.0
    
    # Weather-specific (Strategy 6, 7)
    weather_sensitivity: float = 0.5  # Individual weather sensitivity
    crosswind_skill: float = 0.5  # Bike handling in wind
    
    # Historical context
    recent_results: List[Dict] = field(default_factory=list)
    gc_position: Optional[int] = None
    gc_time_behind: float = 0.0  # seconds behind leader
    
    def __post_init__(self):
        if not self.tactical_state_probs:
            self.tactical_state_probs = {state: 0.25 for state in TacticalState}
        self.attack_signal = AttackSignal.NONE


@dataclass
class StageContext:
    """
    Stage characteristics and environmental conditions.
    """
    # Identity
    stage_id: int
    race_id: int
    stage_number: Optional[int]  # None for one-day races
    stage_date: date
    
    # Parcours
    stage_type: StageType
    distance_km: float
    vertical_m: Optional[int] = None
    profile_score: Optional[int] = None  # PCS difficulty 0-100
    
    # Climbs (for mountain stages)
    climbs: List[Dict] = field(default_factory=list)
    
    # Environmental (Strategy 6, 7)
    avg_temp_c: Optional[float] = None
    weather: WeatherCondition = WeatherCondition.CLEAR
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    precipitation_mm: float = 0.0
    
    # Race dynamics
    is_rest_day: bool = False  # Strategy 5
    days_since_rest: int = 0
    avg_speed_winner_kmh: Optional[float] = None
    
    # Field strength
    startlist_quality_score: Optional[int] = None
    
    # Temporal positioning
    race_phase: GamePhase = GamePhase.EARLY
    stages_remaining: int = 0


@dataclass 
class MarketState:
    """
    Betting market state and odds information.
    """
    market_type: MarketType
    selection_id: int  # rider_id or other identifier
    
    # Odds (decimal format)
    back_odds: float = 0.0
    lay_odds: float = 0.0
    implied_prob: float = field(init=False)
    
    # Market metadata
    volume_matched: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    # Our estimates
    model_prob: float = 0.5
    model_prob_uncertainty: float = 0.1  # Standard deviation
    edge_bps: float = field(init=False)  # Edge in basis points
    
    def __post_init__(self):
        if self.back_odds > 1:
            self.implied_prob = 1.0 / self.back_odds
        else:
            self.implied_prob = 0.0
        self.edge_bps = (self.model_prob - self.implied_prob) * 10000


@dataclass
class GapState:
    """
    Real-time breakaway gap dynamics (Strategy 13).
    """
    gap_seconds: float = 0.0
    gap_velocity: float = 0.0  # Rate of change (positive = growing)
    catch_prob: float = 0.5
    
    # Ornstein-Uhlenbeck parameters
    ou_theta: float = 0.1  # Mean reversion rate
    ou_mu: float = 120.0  # Equilibrium gap (seconds)
    ou_sigma: float = 30.0  # Volatility
    
    # Kalman filter state
    kf_mean: float = 0.0
    kf_variance: float = 100.0


@dataclass
class BreakawayGameState:
    """
    Game-theoretic state for breakaway decisions (Strategy 8).
    """
    # State space S = (GC_positions, Stage_wins, Remaining_stages)
    gc_positions: Dict[int, int] = field(default_factory=dict)  # rider_id -> position
    stage_wins: Dict[int, int] = field(default_factory=dict)  # rider_id -> wins
    remaining_stages: int = 0
    
    # Quantal Response Equilibrium parameters
    qre_lambda: float = 1.0  # Rationality parameter
    
    # Coalition structure (hedonic game)
    coalitions: List[set] = field(default_factory=list)
    coalition_values: Dict[frozenset, float] = field(default_factory=dict)


@dataclass
class Position:
    """
    Betting position with sizing and metadata.
    """
    market_state: MarketState
    
    # Kelly-optimal sizing
    kelly_fraction: float = 0.0  # Full Kelly
    robust_kelly_fraction: float = 0.0  # With uncertainty penalty
    half_kelly_fraction: float = 0.0  # Conservative
    
    # Risk management
    cvar_95: float = 0.0  # Conditional VaR at 95%
    max_drawdown: float = 0.0
    
    # Strategy attribution
    originating_strategy: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Execution
    stake: float = 0.0  # Final stake after all constraints
    entry_odds: float = 0.0
    status: str = "pending"  # pending, open, closed, void


@dataclass
class Portfolio:
    """
    Collection of positions with covariance structure.
    """
    positions: List[Position] = field(default_factory=list)
    
    # Portfolio metrics
    total_stake: float = field(init=False)
    expected_return: float = 0.0
    portfolio_variance: float = 0.0
    cvar_95: float = 0.0
    
    # Correlation matrix between positions
    correlation_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.total_stake = sum(p.stake for p in self.positions)


@dataclass
class StrategyOutput:
    """
    Standardized output from any betting strategy.
    """
    strategy_name: str
    rider_id: int
    stage_id: int
    
    # Probability estimates
    win_prob: float = 0.0
    win_prob_std: float = 0.0  # Posterior uncertainty
    
    # Value assessment
    edge_bps: float = 0.0
    expected_value: float = 0.0
    
    # Model diagnostics
    r_hat: float = 1.0  # Convergence diagnostic
    ess: float = 0.0  # Effective sample size
    
    # Latent variables (strategy-specific)
    latent_states: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    computed_at: datetime = field(default_factory=datetime.now)
