# Genqirue: Bayesian Cycling Betting Engine

Production-grade betting intelligence system implementing 15 research-grade statistical models for professional cycling betting.

## Architecture

```
genqirue/
├── domain/          # Core entities and enums
├── models/          # 15 betting strategies (Bayesian models)
├── portfolio/       # Kelly optimization, CVaR constraints
├── data/            # Schema extensions for betting data
├── inference/       # Real-time and batch inference (to implement)
├── execution/       # Bet placement logic (to implement)
└── validation/      # Backtesting and scoring (to implement)
```

## Implemented Strategies

| Priority | Strategy | Model | File | Status |
|----------|----------|-------|------|--------|
| 1 | Gruppetto Outlier | Bayesian Cox PH with frailty | `gruppetto_frailty.py` | ✅ |
| 2 | Attack Confirmation | Online Changepoint Detection | `online_changepoint.py` | ✅ |
| 3 | Tactical Time Loss | Hidden Markov Model | `tactical_hmm.py` | ✅ |
| 4 | ITT Weather | Spatio-temporal GP | `weather_spde.py` | ✅ |
| 5 | Portfolio | Robust Kelly + CVaR | `kelly.py` | ✅ |

### Strategy 2: Gruppetto Frailty (START HERE)

**Concept**: Riders in the gruppetto (autobus) on mountain stages may be preserving energy rather than genuinely struggling. These riders often perform well on subsequent transition stages.

**Model**: Bayesian Cox Proportional Hazards with time-varying frailty
```
λ_i(t) = λ_0(t) * exp(β^T * X_i + b_i)
where b_i ~ N(0, σ²_b) is rider-specific frailty
```

**Key Signal**: High frailty on mountain + gruppetto time loss < threshold = hidden form

**Usage**:
```python
from genqirue.models import GruppettoFrailtyModel

model = GruppettoFrailtyModel()
model.fit({
    'survival_data': survival_records,
    'rider_ids': unique_riders
})

frailty = model.compute_frailty()
hidden_form = model.get_hidden_form_prob(rider_id)
```

### Strategy 12: Attack Confirmation

**Concept**: Detect attacks in real-time from power telemetry using Bayesian Online Changepoint Detection (Adams & MacKay 2007).

**Model**: BOCD with Weibull hazard
```
P(r_t | x_{1:t}) = P(r_{t-1}, x_t | x_{1:t-1}) / Σ P(r_{t-1}, x_t | x_{1:t-1})
```

**Key Signal**: P(changepoint) > 0.8 AND yesterday's Z-score > 2.0

**Usage**:
```python
from genqirue.models import BayesianChangepointDetector

detector = BayesianChangepointDetector()
result = detector.update({
    'power_z_score': 2.5,
    'rider_id': rider_id,
    'timestamp': datetime.now()
})

if result['should_bet']:
    print(f"Attack confirmed! Latency: {result['latency_ms']:.1f}ms")
```

### Strategy 1: Tactical Time Loss

**Concept**: Hidden Markov Model distinguishing between genuine struggling (RECOVERING) and tactical energy preservation (PRESERVING).

**Model**: HMM with 4 latent states
```
z_{i,t} ~ Bernoulli(π_{i,t})
π_{i,t} = logit^{-1}(δ_0 + δ_1 * ΔGC_{i,t} + δ_2 * StageType_t)
```

**Key Signal**: Rider in PRESERVING state on mountain → likely to contest next flat stage

### Strategy 6: Weather SPDE

**Concept**: Gaussian Process model of wind field to estimate fair time difference between early and late ITT starters.

**Model**: Spatio-temporal GP with Matérn kernel
```
v_wind(s,t) ~ GP(0, k((s,t), (s',t')))
ΔT = ∫[P/F(v_early) - P/F(v_late)]dx
```

**Target**: RMSE < 10 seconds vs actual results

### Portfolio: Robust Kelly

**Concept**: Kelly-optimal bet sizing with uncertainty penalty and CVaR tail risk constraints.

**Formula** (see `docs/MODELS.md`):
```
f* = (b*p̂ - q)/b * (1 - σ²(p̂)*(b+1)²/(p̂²*(b+1)²))
```

**Constraints**:
- Σf ≤ 1.0 (full allocation)
- f ≤ 0.25 (max 25% per position)
- quad_form(f, cov) ≤ 0.05 (variance)
- CVaR_95 ≤ 0.10 (tail risk)

**Usage**:
```python
from genqirue.portfolio import RobustKellyOptimizer

optimizer = RobustKellyOptimizer()
portfolio = optimizer.optimize_portfolio(positions, team_assignments)

print(optimizer.generate_report(portfolio))
```

## Database Schema

The betting engine extends the base `cycling.db` schema with tables for:

- `rider_frailty` - Frailty estimates and hidden form probabilities
- `tactical_states` - HMM decoded states per rider-stage
- `weather_fields` - GP wind field predictions
- `telemetry_changepoints` - Real-time changepoint detection
- `strategy_outputs` - All model predictions
- `positions` - Betting positions and P&L tracking
- `backtest_results` - Performance validation

Apply extensions:
```bash
python fetch_odds.py --init-schema
```

## Critical Implementation Order

1. **Strategy 2** (Gruppetto Frailty) - provides frailty scores used by Strategies 1, 3, 4, 14
2. **Strategy 12** (Online Changepoint) - highest value real-time signal
3. **Strategy 1** (Tactical HMM) - core alpha generator
4. **Strategy 6** (Weather SPDE) - high edge potential
5. **Portfolio** (Kelly) - must handle posterior uncertainty

## Acceptance Criteria

| Criterion | Target | Test |
|-----------|--------|------|
| Strategy 2 correlation | > 0.3 | Frailty vs next-day performance |
| Strategy 12 latency | < 50ms | Attack detection time |
| Strategy 6 RMSE | < 10s | ITT time difference prediction |
| Kelly max position | ≤ 25% | Single bet constraint |
| All models | Posterior | Predictive distributions |

## Dependencies

```bash
pip install -r requirements.txt
```

Core requirements:
- `pymc >= 5.0` - Bayesian inference
- `scikit-survival` - Cox models
- `cvxpy` - Portfolio optimization
- `numba` - Speed for Strategy 12

## Quick Start

```python
import sqlite3
from genqirue.models import GruppettoFrailtyModel, TacticalTimeLossHMM
from genqirue.portfolio import RobustKellyOptimizer

# Load data from scraped database
conn = sqlite3.connect('data/cycling.db')

# Strategy 2: Fit frailty model
frailty_model = GruppettoFrailtyModel()
frailty_model.fit(load_survival_data(conn))

# Strategy 1: Fit tactical HMM
hmm_model = TacticalTimeLossHMM()
hmm_model.fit(load_tactical_data(conn))

# Generate predictions for upcoming stage
positions = []
for rider_id in startlist:
    # Get frailty
    frailty = frailty_model.get_hidden_form_prob(rider_id)
    
    # Get tactical state
    state_probs = hmm_model.get_tactical_state_prob(
        rider_id, stage_type, gc_time_behind
    )
    
    # Create position if edge exists
    if frailty > 0.3 and state_probs['PRESERVING'] > 0.6:
        positions.append(create_position(rider_id, market_odds))

# Optimize portfolio
optimizer = RobustKellyOptimizer()
portfolio = optimizer.optimize_portfolio(positions, team_assignments)

# Place bets
for pos in portfolio.positions:
    if pos.stake > 0:
        place_bet(pos)
```

## Future Strategies (Not Yet Implemented)

| Strategy | Model | Complexity |
|----------|-------|------------|
| 3 | Medical PK | Two-compartment pharmacokinetic |
| 4 | Youth Fade | Functional PCA on aging curves |
| 5 | Rest Day | Interrupted Time Series |
| 7 | Weather H2H | Vector field Langevin |
| 8 | Breakaway | POSG with QRE |
| 9 | Super-Domestique | Dirichlet Process Mixture |
| 10 | Mechanical | Marked Hawkes process |
| 11 | Domestique | HJB differential game |
| 13 | Gap Closing | Ornstein-Uhlenbeck |
| 14 | Post-Crash | Joint frailty models |
| 15 | Cobbles | Reliability theory with copulas |

## License

Proprietary - For research and educational use only.
