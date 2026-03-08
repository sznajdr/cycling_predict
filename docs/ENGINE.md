# Genqirue: Bayesian Cycling Betting Engine

Production-grade betting intelligence system implementing 15 research-grade statistical models for professional cycling betting. Full mathematical specification in [`docs/MODELS.md`](MODELS.md).

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [All Strategies — Status Overview](#2-all-strategies--status-overview)
3. [Implemented Strategies](#3-implemented-strategies)
4. [Strategies To Implement](#4-strategies-to-implement)
5. [Portfolio Optimizer](#5-portfolio-optimizer)
6. [Database Schema](#6-database-schema)
7. [Implementation Order and Dependencies](#7-implementation-order-and-dependencies)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Dependencies by Package](#9-dependencies-by-package)
10. [Quick Start](#10-quick-start)

---

## 1. Architecture

```
genqirue/
├── domain/          # Core entities and enums
│   ├── entities.py  # RiderState, MarketState, Position, Portfolio dataclasses
│   └── enums.py     # StageType, TacticalState, MarketType, RiskType
├── models/          # All 15 betting strategies
│   ├── base.py                 # BayesianModel ABC
│   ├── gruppetto_frailty.py    # Strategy 2 — IMPLEMENTED
│   ├── tactical_hmm.py         # Strategy 1 — IMPLEMENTED
│   ├── weather_spde.py         # Strategy 6 — IMPLEMENTED
│   ├── online_changepoint.py   # Strategy 12 — IMPLEMENTED
│   ├── medical_pk.py           # Strategy 3 — to implement
│   ├── functional_pca.py       # Strategy 4 — to implement
│   ├── interrupted_ts.py       # Strategy 5 — to implement
│   ├── weather_sde.py          # Strategy 7 — to implement
│   ├── breakaway_game.py       # Strategy 8 — to implement
│   ├── mixed_membership.py     # Strategy 9 — to implement
│   ├── hawkes_mechanical.py    # Strategy 10 — to implement
│   ├── chase_game.py           # Strategy 11 — to implement
│   ├── gap_ou_process.py       # Strategy 13 — to implement
│   ├── joint_frailty.py        # Strategy 14 — to implement
│   └── cobble_reliability.py   # Strategy 15 — to implement
├── portfolio/
│   └── kelly.py     # Robust Kelly + CVaR optimiser — IMPLEMENTED
├── data/
│   └── schema_extensions.sql  # All betting DB tables
├── inference/       # Real-time and batch inference (to implement)
└── execution/       # Bet placement logic (to implement)
```

All models inherit from `BayesianModel` in `base.py` and implement four methods:

```python
def build_model(self, data: dict) -> None: ...   # define PyMC model
def fit(self, data: dict) -> None: ...           # sample posterior
def predict(self, new_data: dict) -> dict: ...   # win prob + uncertainty
def get_edge(self, prediction: dict, market_odds: float) -> float: ... # bps
```

---

## 2. All Strategies — Status Overview

| # | Strategy | Category | Model class | File | Status |
|---|----------|----------|-------------|------|--------|
| 1 | Tactical Time Loss | Pre-race | `TacticalTimeLossHMM` | `tactical_hmm.py` | Implemented |
| 2 | Gruppetto Frailty | Pre-race | `GruppettoFrailtyModel` | `gruppetto_frailty.py` | Implemented |
| 3 | Medical Communiqué | Pre-race | `MedicalPKModel` | `medical_pk.py` | To implement |
| 4 | Youth Fade | Pre-race | `YouthFadeModel` | `functional_pca.py` | To implement |
| 5 | Rest Day Regression | Pre-race | `RestDayModel` | `interrupted_ts.py` | To implement |
| 6 | ITT Weather Arbitrage | Environmental | `WeatherSPDEModel` | `weather_spde.py` | Implemented |
| 7 | Weather Mismatch H2H | Environmental | `WeatherSDEModel` | `weather_sde.py` | To implement |
| 8 | Desperation Breakaway | Game theory | `BreakawayGameModel` | `breakaway_game.py` | To implement |
| 9 | Super-Domestique Choke | Game theory | `MixedMembershipModel` | `mixed_membership.py` | To implement |
| 10 | Mechanical Incident | Real-time | `HawkesMechanicalModel` | `hawkes_mechanical.py` | To implement |
| 11 | Domestique Chokehold | Game theory | `ChaseGameModel` | `chase_game.py` | To implement |
| 12 | Attack Confirmation | Real-time | `BayesianChangepointDetector` | `online_changepoint.py` | Implemented |
| 13 | Gap Closing Calculus | Real-time | `GapOUModel` | `gap_ou_process.py` | To implement |
| 14 | Post-Crash Confidence | Risk | `JointFrailtyModel` | `joint_frailty.py` | To implement |
| 15 | Rain on Cobbles | Risk | `CobbleReliabilityModel` | `cobble_reliability.py` | To implement |

---

## 3. Implemented Strategies

---

### Strategy 2: Gruppetto Frailty

**File:** `genqirue/models/gruppetto_frailty.py`

**The edge.** Riders in the gruppetto on mountain stages may be preserving energy rather than genuinely struggling. The market prices them as weak candidates for the following flat or hilly stage. The frailty term separates riders who are hiding form from those who are at their actual limit.

**Model.**

```
λ_i(t) = λ_0(t) · exp(β^T · X_i + b_i)

b_i ~ Normal(0, σ²_b)    [rider-specific frailty]
```

Covariates `X_i`: GC position, seconds behind leader, gruppetto flag, time lost to leader. The frailty `b_i` captures everything the covariates miss — unexplained resilience. Positive `b_i` means the rider survived longer than observable characteristics predict.

**Data inputs.** `rider_results` (result_category = 'stage' and 'gc'), `race_stages` (stage_type), `startlist_entries`.

**Output.** `frailty_estimate` and `hidden_form_prob` per rider, stored in `rider_frailty`. Hidden form probability is derived from the empirical CDF of frailty estimates.

**Key signal.** `hidden_form_prob > 0.5` on or after a mountain stage → bet rider for the next transition stage.

**Downstream use.** Frailty scores feed into Strategies 1, 3, 4, and 14 as a covariate.

**Usage.**

```python
from genqirue.models import GruppettoFrailtyModel, FastFrailtyEstimator
from genqirue.models.gruppetto_frailty import SurvivalRecord

# Full Bayesian model (MCMC, slow but calibrated)
model = GruppettoFrailtyModel()
model.fit({'survival_data': survival_records, 'rider_ids': unique_riders})
frailty = model.compute_frailty()
hidden_form = model.get_hidden_form_prob(rider_id)

# Fast approximation (martingale residuals, no MCMC)
estimator = FastFrailtyEstimator()
estimator.fit(survival_records)
score = estimator.get_frailty(rider_id)
```

---

### Strategy 12: Attack Confirmation (BOCPD)

**File:** `genqirue/models/online_changepoint.py`

**The edge.** Live markets on breakaway survival move on information arriving in real time. Confirming whether an attack is structural — not a test effort — before the TV feed processes it creates a timing edge on live markets.

**Model.** Bayesian Online Changepoint Detection (Adams & MacKay, 2007). Maintains a posterior distribution over run length `r_t` — time since the last structural break in the gap or power time series:

```
P(r_t | x_{1:t}) ∝ Σ_{r_{t-1}} P(x_t | r_t, x_{(t-k):t}) · P(r_t | r_{t-1})
```

The hazard function uses a Weibull distribution over run lengths. The posterior update runs via a Numba-JIT kernel under 100ms:

```python
@jit(nopython=True)
def update_run_length(x_t, R_prev, hazard, pred_prob):
    growth  = R_prev * pred_prob * (1 - hazard)
    cp_prob = np.sum(R_prev * pred_prob * hazard)
    R_new   = np.zeros_like(R_prev)
    R_new[0]  = cp_prob
    R_new[1:] = growth[:-1]
    return R_new / np.sum(R_new)
```

**Data inputs.** Live telemetry: power Z-score, gap time series from live timing. In batch mode: `rider_results` time gaps as a proxy.

**Output.** `changepoint_prob`, `run_length`, `attack_signal` (0–3 confidence scale) stored per rider per stage in `telemetry_changepoints`.

**Key signal.** `changepoint_prob > 0.8` AND prior-stage Z-score > 2.0 → bet the rider is in a real attack.

**Latency target.** < 50ms per update.

**Usage.**

```python
from genqirue.models import BayesianChangepointDetector

detector = BayesianChangepointDetector()
result = detector.update({
    'power_z_score': 2.5,
    'rider_id': rider_id,
    'timestamp': datetime.now()
})

if result['should_bet']:
    print(f"Attack confirmed: {result['changepoint_prob']:.2f}  latency: {result['latency_ms']:.1f}ms")
```

---

### Strategy 1: Tactical Time Loss HMM

**File:** `genqirue/models/tactical_hmm.py`

**The edge.** Every time gap is a mixture of fitness and tactics. A GC rider managing effort and one who cracked both show the same observed time loss. Identifying who was preserving lets you bet them on the following stage.

**Model.** Hidden Markov Model with latent states `z_{i,t}`:

```
P(z_{i,t} = PRESERVING) = sigmoid(δ_0 + δ_1 · ΔGC_{i,t} + δ_2 · StageType_t)

time_loss | CONTESTING  ~ Normal(μ, σ²)
time_loss | PRESERVING  ~ Normal(μ + γ, σ²),   γ ~ Normal⁺(2, 0.5)
```

`γ` is the tactical time loss — the extra time deliberately conceded — constrained positive and centred on 2 minutes.

**Data inputs.** `rider_results` (time_behind_winner_seconds, result_category = 'stage' and 'gc'), `race_stages` (stage_type).

**Output.** `contesting_prob`, `preserving_prob`, `decoded_state`, `tactical_time_loss_seconds` per rider per stage in `tactical_states`.

**Key signal.** `preserving_prob > 0.7` on a mountain stage → bet the rider for the following flat or hilly stage.

**Usage.**

```python
from genqirue.models import TacticalTimeLossHMM, SimpleTacticalDetector
from genqirue.models.tactical_hmm import TacticalObservation

# Full HMM (requires ≥ 10 observations)
model = TacticalTimeLossHMM()
model.fit({'observations': observations, 'rider_ids': unique_riders})
state_probs = model.get_tactical_state_prob(rider_id, stage_type, gc_time_behind)

# Simple heuristic detector (no MCMC, works on sparse data)
detector = SimpleTacticalDetector()
for obs in observations:
    detector.update(obs)
is_preserving = detector.is_tactical_preserving(rider_id)
```

---

### Strategy 6: ITT Weather Arbitrage

**File:** `genqirue/models/weather_spde.py`

**The edge.** ITT start windows span 3–4 hours. Wind conditions over that window can differ by 30–90 seconds for riders on key exposed sections — swamping typical GC gaps. When weather data arrives after markets open, there is a pricing window.

**Model.** Gaussian Process over the wind field along the course, implemented via PyMC's HSGP approximation for computational tractability:

```python
with pm.Model() as weather_model:
    gp = pm.gp.HSGP(cov_func=matern_kernel, m=[20, 20, 10])
    f = gp.prior("f", X=spatiotemporal_coords)
```

Fair time difference between an early and late starter:

```
ΔT = ∫_0^D [P / F_aero(v_wind(t_early)) - P / F_aero(v_wind(t_late))] dx
```

`F_aero` is the aerodynamic drag force; `P` is the rider's power output.

**Data inputs.** Weather API (wind speed, direction at points along the course), ITT start order and times from race organisers, `race_stages` (distance_km, stage_type = 'itt').

**Output.** `delta_t_seconds` and `uncertainty` per early/late rider pair, stored in `itt_time_predictions`.

**Key signal.** When `|ΔT|` exceeds the current H2H market margin and `uncertainty` is low, there is edge on the H2H or winner market.

**Acceptance criterion.** RMSE < 10 seconds vs actual ITT time differences on historical data.

---

## 4. Strategies To Implement

Each section below gives everything needed to begin implementation: the edge, the model, the data it reads, the output it writes, where it connects to the portfolio, its dependencies, and acceptance criteria.

---

### Strategy 3: Medical Communiqué

**File to create:** `genqirue/models/medical_pk.py`
**Class name:** `MedicalPKModel`

**The edge.** Crash and illness communiqués arrive with a lag and are priced crudely — either the market ignores them or overreacts with a blunt discount. A pharmacokinetic model provides a precise, time-varying performance penalty curve calibrated to the specific type and severity of incident. The market rarely has this.

**Model.** Two-compartment pharmacokinetic model. Physical trauma is treated as a drug concentration decaying over time:

```
dC_trauma/dt = -k_el · C_trauma

Perf(t) = Perf_baseline · (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

- `k_el` — elimination rate constant; estimated per rider from historical return-to-form data after documented crashes (scraped from ProCyclingStats race communiqués or news sources)
- `EC_50` — concentration at which performance is halved; varies by incident type (crash vs illness vs mechanical)
- `C_trauma(t_0) = dose` — initial trauma concentration at the time of incident, scaled to severity

For a crash on stage `t_0`, the predicted performance multiplier on stage `t_0 + d` is:

```
Perf_penalty(d) = 1 - (dose · exp(-k_el · d)) / (EC_50 + dose · exp(-k_el · d))
```

Fit `k_el` and `EC_50` via hierarchical Bayesian model over historical incidents. Use robust Kelly with the posterior standard deviation as uncertainty input.

**Data inputs.**
- Medical communiqués (external; ingested via a parser or manually entered)
- `rider_results` — pre-incident baseline performance by stage type
- `pk_parameters` table (read/write; stores fitted parameters per rider per incident)

**Output.** A `Perf_penalty(d)` curve per rider per incident, and adjusted win probability estimates for upcoming stages. Stored in `pk_parameters`; strategy output stored in `strategy_outputs`.

**Integration.** The PK penalty scales the frailty-adjusted win probability from Strategy 2 before passing to the Kelly optimizer.

**Dependencies.** `pymc >= 5.0`, `numpy`, `scipy`.

**Acceptance criterion.** Predicted performance penalty at day d=2 post-crash correlates with observed performance delta (vs pre-crash baseline) at r > 0.4 on historical crash data.

---

### Strategy 4: Youth Fade

**File to create:** `genqirue/models/functional_pca.py`
**Class name:** `YouthFadeModel`

**The edge.** Age-related performance decline is not linear and not uniform across rider types. A 32-year-old sprinter and a 32-year-old climber are at different points on their respective trajectories. The market prices age as a blunt heuristic (crude discounts to older riders). The model prices age as a personalised trajectory, giving a precise current-season expectation for each rider.

**Model.** Functional Principal Component Analysis applied to career performance trajectories. Each rider's career arc `X_i(t)` is decomposed as:

```
X_i(t) = μ(t) + Σ_k ξ_{ik} · φ_k(t) + ε_i(t)
```

- `μ(t)` — population-average aging curve across all professional riders
- `φ_k(t)` — the k-th principal mode of variation in career arc shape (e.g. early-peaking vs late-peaking riders)
- `ξ_{ik}` — rider-specific loading on each mode; learned from career history
- `ε_i(t)` — residual noise

Fit using `scikit-fda` or a custom implementation. Supplement with a Gompertz-Makeham survival component for the abandonment hazard at age t.

**Data inputs.**
- `rider_results` — full career result history (all years), filtered to result_category = 'stage', ranked stage finishes
- `riders` — birthdate, specialty scores (sp_climber, sp_sprint, sp_gc, etc.)
- `rider_teams` — team class by season (proxy for competitive level)

**Output.** For each rider, a predicted performance score for the current season relative to career trajectory, and a deviation from expected trajectory (positive = underpriced, negative = overpriced). Stored in `strategy_outputs`.

**Integration.** Trajectory deviation is used as an additional covariate in the Kelly optimizer's probability estimate. Riders materially below their predicted trajectory are faded; those above are backed when the market hasn't adjusted.

**Implementation notes.**
- Require at least 3 seasons of data per rider to compute meaningful loadings
- Specialty scores from `riders` table can be used to cluster riders before fitting separate population curves per archetype (sprinter, climber, GC)
- Use leave-one-season-out cross-validation to calibrate trajectory predictions

**Dependencies.** `scikit-fda` or `numpy` with manual FPCA, `scipy`.

**Acceptance criterion.** Trajectory deviation in season N correlates with performance change from season N-1 to N at r > 0.25 on held-out riders.

---

### Strategy 5: Rest Day Regression

**File to create:** `genqirue/models/interrupted_ts.py`
**Class name:** `RestDayModel`

**The edge.** Rest days reset physical state in ways the market treats as noise. Riders who were trending downward before a rest day often recover; riders peaking before a rest day sometimes fade. An interrupted time series model separates the systematic rest-day effect from underlying form, producing better baseline form estimates entering the post-rest-day stages.

**Model.** Interrupted Time Series with ARMA errors:

```
Y_t = β_0 + β_1·Time_t + β_2·Intervention_t + β_3·TimeAfter_t + Σ_j φ_j·Y_{t-j} + ε_t
```

- `Intervention_t` — binary indicator for rest day
- `β_2` — immediate level shift on return from rest
- `β_3` — slope change in performance trajectory post-rest
- `φ_j` — autoregressive coefficients capturing momentum

Bayesian Structural Time Series (BSTS) alternative handles multiple rest days and non-stationary trends more cleanly:

```
Y_t = μ_t + β^T·X_t + ε_t
μ_t = μ_{t-1} + δ_{t-1} + η_t     [local linear trend]
δ_t = δ_{t-1} + ζ_t                [slope]
```

Fit both; use BSTS for races with multiple rest days (grand tours), ITS for shorter stage races.

**Data inputs.**
- `rider_results` (stage-by-stage GC time gaps for each rider through a race)
- `race_stages` (stage_date, to identify rest days as gaps in the date sequence)

**Output.** A post-rest adjusted performance baseline and slope estimate per rider, used to update win probability estimates for the first 2–3 stages after a rest day. Stored in `strategy_outputs`.

**Integration.** Adjusts the prior probability before the Kelly optimizer runs. Riders flagged for above-average post-rest recovery (positive `β_2` historically) are backed; those with flat or negative response are not.

**Dependencies.** `pymc >= 5.0` (for BSTS), `statsmodels` (for classical ITS), `numpy`, `pandas`.

**Acceptance criterion.** Out-of-sample prediction of post-rest-day stage performance (top-10 rate) improves over a naive form baseline by at least 15% relative.

---

### Strategy 7: Weather Mismatch H2H

**File to create:** `genqirue/models/weather_sde.py`
**Class name:** `WeatherSDEModel`

**The edge.** H2H markets on cobble sectors, crosswind stages, and wet descents misprice handling ability and technical skill. Some riders are structurally less sensitive to wind and rough surfaces — their time is more predictable regardless of conditions. The market conflates this structural robustness with raw speed.

**Model.** Langevin stochastic differential equation for bike velocity in stochastic wind:

```
m · dv/dt = F_drive - F_drag(v, v_wind) - F_gravity + σ_wind · ξ(t)

F_drag(v, v_wind) = (1/2) · ρ · CdA · (v + v_wind)²
```

`ξ(t)` is white noise modelling wind gusts. Integrating this SDE over a stage profile produces a distribution over finishing times per rider under different wind realisations. The key metric is **output variance** — riders with lower finishing time variance under gust scenarios are preferred in crosswind or cobble conditions regardless of their mean finishing time.

**Parameters to estimate per rider:**
- `CdA` — drag coefficient × frontal area (proxy: rider weight and height from `riders` table)
- `P_max` — power output (proxy: specialty scores, past winning time on similar stages)
- `σ_gust_sensitivity` — calibrated from historical performance variance on windy stages

**Data inputs.**
- `riders` (height_m, weight_kg, sp_one_day_races, sp_hills — proxies for handling ability)
- `rider_results` (time gaps on cobble/crosswind stages — use won_how and stage profile data)
- `race_stages` (stage_type, profile_score — identify crosswind-exposure stages)
- External weather data at race time (wind speed, direction)

**Output.** A distribution over finishing time for each rider under the current weather scenario, and a probability estimate that rider A beats rider B on the H2H market. Stored in `strategy_outputs` and `positions`.

**Integration.** Produces H2H market probability estimates that feed directly into the Kelly optimizer as market_type = 'h2h'.

**Dependencies.** `sdeint` or `torchsde`, `numpy`, `scipy`.

**Acceptance criterion.** Model-implied H2H probabilities have positive ROI against actual H2H market prices on crosswind and cobble stages in a walk-forward backtest.

---

### Strategy 8: Desperation Breakaway

**File to create:** `genqirue/models/breakaway_game.py`
**Class name:** `BreakawayGameModel`

**The edge.** Late in a stage race, riders who have lost GC contention and have no stage win have strong strategic incentive to attack in breakaways — regardless of physical form. The market prices these riders on form signals. The model prices them on incentive, which is highest precisely when form signals are worst.

**Model.** Partially Observable Stochastic Game (POSG). State space:

```
S = (GC_positions, Stage_wins, Remaining_stages, Stage_type)
```

Each rider's action is stochastic, modelled via Quantal Response Equilibrium (QRE) — a noisy best-response:

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_{a'} exp(λ · Q_i(s, a'))
```

- `Q_i(s, a_i)` — expected value of action `a_i` from state `s` for rider i (computed via dynamic programming or learned via CFR)
- `λ` — rationality parameter controlling how sharply riders best-respond; estimated from historical breakaway behavior

Strategic incentive score for rider i before stage t:

```
incentive(i, t) = w_gc · I(GC_rank_i > threshold)
               + w_win · I(stage_wins_i = 0)
               + w_stage · (remaining_stages / total_stages)
               + w_type · I(stage_type = breakaway_favorable)
```

**Data inputs.**
- `rider_results` (GC rank before each stage, stage wins so far in race)
- `race_stages` (stage_number, stage_type, remaining stages)
- `startlist_entries` (rider field for each race — affects breakaway coalition size)

**Output.** A per-rider breakaway probability for the current stage, and a stage win probability adjusted for strategic incentive. Stored in `strategy_outputs`.

**Integration.** Feeds into the Kelly optimizer for winner and top-10 markets. Particularly useful in stages 6–10 of grand tours when GC hierarchy is established but sprint/GC riders dominate market pricing.

**Implementation notes.**
- A simplified version using only the incentive score (no full POSG) is a valid starting point
- Full POSG requires `open_spiel` or a custom implementation of Counterfactual Regret Minimization
- Validate the incentive score first (does high incentive correlate with breakaway participation?) before adding the full game-theoretic layer

**Dependencies.** `numpy`, `scipy`; optionally `open_spiel` or `ray[rllib]` for full POSG.

**Acceptance criterion.** Breakaway participation rate for riders with `incentive_score > 0.7` exceeds the field average by at least 2×, measured on historical stage race data.

---

### Strategy 9: Super-Domestique Choke

**File to create:** `genqirue/models/mixed_membership.py`
**Class name:** `MixedMembershipModel`

**The edge.** Some riders perform measurably below their individual ceiling when riding in protection duties — a persistent effect that isn't priced into individual stage markets. Other riders are unaffected or even elevate. The market treats domestique status as a categorical variable; the model treats it as a continuous, rider-specific loading.

**Model.** Mixed Membership Model (Latent Dirichlet Allocation-style) with a Dirichlet Process prior over rider archetypes:

```
θ_i ~ Dirichlet(α)            [rider-specific type weights]
w_{i,t} ~ Σ_k θ_{ik} · Normal(μ_k, Σ_k)
```

Latent types `k`:
- **Leader** — performance unaffected or improved when given freedom
- **Domestique** — performance suppressed under protection duties
- **Opportunist** — performs to individual ceiling regardless of role

The observed variable `w_{i,t}` is a composite performance score for rider i on stage t (function of time gap, rank, winning probability). The loading `θ_{i,domestique}` for each rider is estimated from career history.

**Data inputs.**
- `rider_results` — full career stage results for all riders
- `startlist_entries` and `rider_teams` — to infer protection duties (proxy: whether a rider's team leader was in the top 10 GC)
- `riders` — specialty scores as priors on archetype membership

**Output.** Per-rider archetype weights `θ_i`, and for each upcoming stage, a predicted performance score conditioned on the rider's expected role. High `θ_{i,domestique}` + protection duty → fade the rider. Stored in `strategy_outputs`.

**Integration.** The archetype-conditioned performance score adjusts the win probability before Kelly optimization. Particularly useful for identifying overpriced riders who are listed individually but riding in a clearly supportive role.

**Dependencies.** `pymc >= 5.0` (Dirichlet Process via `pm.Mixture`), `numpy`, `scikit-learn` (for initialisation).

**Acceptance criterion.** Riders with `θ_{domestique} > 0.6` in confirmed protection roles have statistically lower stage rank (worse performance) than their historical baseline at p < 0.05.

---

### Strategy 10: Mechanical Incident

**File to create:** `genqirue/models/hawkes_mechanical.py`
**Class name:** `HawkesMechanicalModel`

**The edge.** Mechanical incidents (punctures, chain drops, crashes) cluster in time and space — cobble sectors, technical descents, bunch sprints. Live markets reprice slowly after a mechanical. If the model can estimate recovery probability and time cost before the market adjusts, there is a window of 15–60 seconds per incident.

**Model.** Marked Hawkes Process — a self-exciting point process where each incident increases the short-term probability of further incidents:

```
λ_t = μ + Σ_{t_i < t} φ(t - t_i, m_i)

φ(Δt, m) = α · m · exp(-β · Δt)
```

- `μ` — baseline incident rate (depends on stage type and km)
- `φ` — excitation kernel; each incident of mark `m` decays with rate `β`
- `m_i` — mark: incident type (puncture, crash, mechanical) and severity

For each incident, the model estimates:

```
P(rejoin_peloton | incident_type, gap_at_incident, km_remaining, team_support)
time_to_rejoin ~ LogNormal(μ_type, σ²_type)
```

A Dirichlet Process prior on incident marks allows the model to learn new incident patterns without pre-specifying the mark space.

**Data inputs.**
- Live telemetry feed: incident type, timestamp, rider, km position
- `race_stages` (stage profile — identify high-risk sections)
- Historical incident data (to calibrate `α`, `β`, `μ` per stage type)

**Output.** Updated win probabilities and abandonment probabilities per rider following each incident. Real-time output (< 100ms per update). Stored in `telemetry_changepoints` and `strategy_outputs`.

**Latency target.** < 100ms per incident update.

**Dependencies.** `numba` (for latency), `tick` or custom Hawkes implementation, `numpy`.

**Acceptance criterion.** Rejoining probability predictions have Brier score < 0.20 on historical incident data.

---

### Strategy 11: Domestique Chokehold

**File to create:** `genqirue/models/chase_game.py`
**Class name:** `ChaseGameModel`

**The edge.** When a team's leader is protected by domestiques, rivals can time their attack to strike precisely when the domestiques are dropped — forcing the leader to respond before they intended. The market prices attacks on pace. The model prices attacks on the strategic timing of when protection expires, which is a different and more exploitable quantity.

**Model.** Two-player differential game between a breakaway rider and a chasing protected leader. The state space is `x = (gap, remaining_domestiques, km_remaining)`. The value function satisfies the Hamilton-Jacobi-Bellman equation:

```
∂V/∂t + min_{u_chase} max_{u_break} [∇V · f(x, u_break, u_chase) + g(x)] = 0
```

- `u_break` — breakaway power output
- `u_chase` — chase power output (or that of the domestique covering for the leader)
- `g(x)` — stage win payoff at `km_remaining = 0`

The Nash equilibrium gives:
- The optimal power allocation for the breakaway rider at each state
- The threshold gap at which the chase will be abandoned (leader switches to individual effort)

**Data inputs.**
- `rider_results` — historical power proxies (time gaps on climbs vs flat sections as a power surrogate)
- `race_stages` — remaining km at each intermediate time check
- Live timing: current gap, number of domestiques still with the leader (estimated from TV/timing data)

**Output.** For each stage breakaway scenario, the probability that the breakaway survives given the current state. Updated in real time as km ticks down. Stored in `strategy_outputs`.

**Integration.** Produces live breakaway survival probability. Feeds into winner and top-3 markets during live betting windows.

**Dependencies.** `scipy.integrate` (for HJB PDE), `numpy`; optionally `casadi` for optimal control formulation.

**Acceptance criterion.** Breakaway survival probability at 50km to go predicts actual outcome better than a naive time-gap-only model on historical race data.

---

### Strategy 13: Gap Closing Calculus

**File to create:** `genqirue/models/gap_ou_process.py`
**Class name:** `GapOUModel`

**The edge.** Live catch probability is priced on the current gap and broadcaster commentary. A mean-reverting gap (chase is working) and a diverging gap (breakaway has settled) look identical in the short run but imply very different catch probabilities over the remaining distance. The OU process distinguishes these regimes.

**Model.** Ornstein-Uhlenbeck process for the gap between breakaway and peloton:

```
dG_t = θ(μ - G_t) dt + σ dW_t
```

- `θ` — mean-reversion speed; large θ means the chase is strongly pulling the gap back
- `μ` — equilibrium gap; 0 implies the peloton will eventually catch
- `σ` — volatility; noise in the gap signal

The catch probability is the **first passage time** probability — probability that `G_t` hits zero by the finish:

```
P(catch | G_0, θ, μ, σ, D_remaining) = 2 · Φ(-G_0 / √(σ² / (2θ) · (1 - exp(-2θ·t_finish))))
```

where `t_finish` is estimated time remaining at the current pace.

An Extended Kalman Filter estimates `θ`, `μ`, and `σ` from live timing splits in real time:

```
State vector: [G_t, θ, μ, σ]
Observation:  G_t (gap measured at each timing point)
Transition:   OU dynamics (linearised for EKF)
```

**Data inputs.**
- Live timing: gap between breakaway and peloton at each intermediate timing point (typically every 10–20km)
- `race_stages` — distance_km and estimated average speed (to convert km remaining to time remaining)

**Output.** Updated catch probability with 90% credible interval, refreshed at each timing point. Latency target < 100ms. Stored in `strategy_outputs` (real-time column).

**Integration.** Produces live market probability for "peloton catches breakaway" binary market, and updates winner probabilities for riders in the break vs peloton. Feeds Kelly optimizer for live betting.

**Dependencies.** `filterpy` (EKF implementation) or `numpy` with manual EKF, `scipy`.

**Acceptance criterion.** Catch probability at 30km to go predicts actual outcome with AUC > 0.75 on historical race data.

---

### Strategy 14: Post-Crash Confidence

**File to create:** `genqirue/models/joint_frailty.py`
**Class name:** `JointFrailtyModel`

**The edge.** The market prices physical crash damage via observable data (injury reports, DNF risk). It does not price the confidence effect — the persistent tendency of riders who have crashed to brake earlier on descents, hold wider lines, and choose safer positioning in the bunch. This effect is real, persistent, and systematically underestimated because it is not directly observable in results data.

**Model.** Joint frailty model with shared random effects across multiple risk types:

```
λ_{ij}(t) = λ_{0j}(t) · exp(β^T · X_{ij} + b_i + ε_{ij})

b_i ~ Normal(0, σ²_b)        [shared frailty across all risk types]
ε_{ij} ~ Normal(0, σ²_{ε_j}) [risk-type-specific residual]
```

Risk types `j`: descent performance, corner performance, wet-condition performance, bunch sprint performance.

The shared frailty `b_i` captures a rider's overall confidence level — affecting all risk types. A rider with high crash history has elevated `b_i` across all four types, not just descents. `ε_{ij}` captures additional type-specific variation (a rider can be a poor descender but a confident cornerist).

**Bayesian Network** over risk types encodes conditional independence structure:

```
Descent_risk ← b_i → Corner_risk
                ↓
          Wet_risk ← Technical_skill
```

**Data inputs.**
- `rider_results` (stage finishes and time gaps on stages with profile characteristics matching each risk type)
- `race_stages` (won_how, stage_type, profile_score — identify descent/corner/wet-heavy stages)
- External: incident reports (crashes, mechanicals) with date and severity

**Output.** Per-rider, per-risk-type frailty decomposition. A high `b_i` + upcoming technical stage → fade the rider. High `b_i` + flat stage → no signal. Stored in `rider_frailty` (extended schema) and `strategy_outputs`.

**Integration.** The confidence-adjusted frailty feeds into the Kelly optimizer. Particularly useful for one-day classics with technical finishes (Liège–Bastogne–Liège, Paris-Roubaix) and mountain stages with dangerous descents.

**Dependencies.** `pymc >= 5.0`, `numpy`, `pgmpy` or `bnlearn` (for Bayesian Network structure).

**Acceptance criterion.** Riders with `b_i > 1 SD` above mean in a given risk type underperform their pre-crash baseline on risk-type-matching stages at p < 0.05.

---

### Strategy 15: Rain on Cobbles

**File to create:** `genqirue/models/cobble_reliability.py`
**Class name:** `CobbleReliabilityModel`

**The edge.** Wet cobble sectors produce correlated failures — when one rider punctures, others are more likely to (same surface conditions, same pacing dynamics). The market prices each rider's puncture risk independently. The model prices it jointly, which substantially changes the probability that any given rider survives the full cobble sequence with the front group.

**Model.** Clayton copula for sector-to-sector survival correlation:

```
C_θ(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
```

`θ` is the tail dependence parameter — how correlated failures are in the worst case. In wet conditions `θ` rises (stronger tail dependence). In dry conditions `θ` approaches 0 (independent sectors). Estimate `θ` from historical cobble stage data as a function of rain and surface conditions.

Individual sector survival probability for rider i on sector s:

```
P(survive_s | rider_i) = f(CdA_i, power_i, surface_condition_s, speed_into_sector)
```

Joint survival across all cobble sectors:

```
P(survive_all | rider_i) ≠ Π_s P(survive_s | rider_i)
                         = C_θ(P(survive_1), ..., P(survive_N))   [correlated]
```

**Dynamic programming** for optimal pacing strategy given survival risk. Value function over (sector, current_speed, energy_remaining):

```
V_s(v, E) = max_{v'} [-λ_s(v') · DNF_cost + stage_time(v') + V_{s+1}(v', E - power_cost(v'))]
```

Riders whose optimal pacing deviates from flat-out (the market's implicit assumption) have mispriced winner and H2H markets.

**Data inputs.**
- `race_stages` (stage_type, profile_score — identify cobble stages)
- `race_climbs` (sector characteristics as cobble sector proxy — km_before_finish, length)
- `riders` (weight_kg, sp_one_day_races — proxy for cobble ability and power-to-weight)
- External: rain forecast, sector condition reports

**Output.** Joint survival probability across all cobble sectors per rider, and the implied stage win probability adjusted for joint survival structure. Stored in `strategy_outputs`. The pacing deviation signal is stored separately for H2H positioning.

**Integration.** Produces winner and H2H probabilities for cobble classics and mixed stages. Feeds Kelly optimizer for outright, top-3, and H2H markets simultaneously.

**Dependencies.** `copulas` or `scipy` (for Clayton copula), `numpy`, `scipy.optimize` (for DP).

**Acceptance criterion.** Joint survival probability (model) is better calibrated than independent survival probability (naive) measured by Brier score on historical cobble stage outcomes.

---

## 5. Portfolio Optimizer

**File:** `genqirue/portfolio/kelly.py`

Kelly fraction for a single bet:

```
f* = (b·p - (1-p)) / (b - 1)
```

where `b` is decimal odds. The system defaults to **quarter-Kelly** (`f = f*/4`).

When a model provides a posterior standard deviation `σ_p`, Robust Kelly automatically reduces stake:

```
f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
```

Portfolio constraints solved via CVXPY:

```python
constraints = [
    cp.sum(f) <= 1.0,              # total allocation cap
    f >= 0,
    f <= 0.25,                     # max 25% per position
    cp.quad_form(f, cov) <= 0.05,  # variance bound
]
```

CVaR at 95% is the expected loss in the worst 5% of scenarios — the tail risk measure that governs ruin probability.

**Usage.**

```python
from genqirue.portfolio import RobustKellyOptimizer, KellyParameters

params = KellyParameters(
    method='half_kelly',
    max_position_pct=0.10,
    min_edge_bps=50
)
optimizer = RobustKellyOptimizer(params)
portfolio = optimizer.optimize_portfolio(positions, team_assignments)
```

---

## 6. Database Schema

The betting engine extends `cycling.db` with the following tables (applied via `python scripts/fetch_odds.py --init-schema`):

| Table | Purpose | Written by |
|-------|---------|------------|
| `rider_frailty` | Frailty estimates and hidden form probabilities | Strategy 2, 14 |
| `tactical_states` | HMM decoded states per rider-stage | Strategy 1 |
| `weather_fields` | GP wind field predictions (GP posterior) | Strategy 6 |
| `itt_time_predictions` | Fair ΔT between ITT starters | Strategy 6 |
| `telemetry_changepoints` | Real-time BOCPD run lengths and attack signals | Strategy 12 |
| `pk_parameters` | PK model elimination rates and EC50 per incident | Strategy 3 |
| `strategy_outputs` | All model win probability predictions | All strategies |
| `positions` | Betting positions: stake, entry odds, P&L | Portfolio |
| `backtest_results` | Walk-forward backtest metrics | Backtester |
| `model_versions` | Model version tracking and hyperparameters | All strategies |
| `bookmaker_odds` | Raw odds snapshots from Betclic | Odds scraper |
| `bookmaker_odds_latest` | View: most recent snapshot per selection | Odds scraper |

---

## 7. Implementation Order and Dependencies

```
Strategy 2 (Gruppetto Frailty)   ← implement first; produces frailty scores
    ↓ provides frailty covariate
Strategy 1 (Tactical HMM)        ← core batch alpha generator
Strategy 3 (Medical PK)          ← uses frailty baseline
Strategy 14 (Post-Crash)         ← extends Strategy 2's frailty framework
    ↓ all feed into
Portfolio (Robust Kelly)          ← must handle uncertainty from all models

Strategy 12 (BOCPD)              ← implement second; highest value real-time signal
Strategy 13 (Gap OU)             ← depends on live timing infrastructure
Strategy 10 (Hawkes Mechanical)  ← shares live infrastructure with 12, 13
Strategy 11 (Chase Game)         ← depends on live timing infrastructure

Strategy 6 (Weather SPDE)        ← independent; external weather dependency
Strategy 7 (Weather H2H)         ← builds on weather infrastructure from 6

Strategy 8 (Breakaway)           ← independent; uses race state from DB
Strategy 9 (Super-Domestique)    ← independent; long career history needed
Strategy 4 (Youth Fade)          ← independent; long career history needed
Strategy 5 (Rest Day)            ← independent; single-race data sufficient
Strategy 15 (Cobbles)            ← independent; race calendar dependent
```

---

## 8. Acceptance Criteria

| Strategy | Metric | Target |
|----------|--------|--------|
| 2 Frailty | Spearman ρ vs next-day transition stage performance | > 0.30 |
| 1 Tactical | Top-3 rate for PRESERVING-flagged riders on next stage | > 2× field average |
| 3 Medical PK | Correlation of PK penalty with observed performance delta | > 0.40 |
| 4 Youth Fade | Trajectory deviation predicts YoY performance change | r > 0.25 |
| 5 Rest Day | Post-rest top-10 rate vs naive form baseline | +15% relative |
| 6 Weather ITT | RMSE vs actual ΔT between early/late ITT starters | < 10 seconds |
| 7 Weather H2H | Walk-forward ROI on crosswind/cobble H2H bets | > 0% |
| 8 Breakaway | Incentive score vs actual breakaway participation rate | > 2× field average |
| 9 Domestique | Performance suppression for high-domestique-loading riders | p < 0.05 |
| 10 Hawkes | Brier score on rejoining probability post-mechanical | < 0.20 |
| 11 Chase | Breakaway survival at 50km vs naive gap model | AUC improvement |
| 12 BOCPD | Attack confirmation latency | < 50ms |
| 13 Gap OU | Catch probability at 30km AUC | > 0.75 |
| 14 Post-Crash | Performance below baseline on risk-type-matching stages | p < 0.05 |
| 15 Cobbles | Brier score: joint vs independent survival probability | Joint < Independent |
| Portfolio | Max single position | ≤ 25% |
| Portfolio | All models | Posterior predictive distributions, not point estimates |

---

## 9. Dependencies by Package

| Package | Strategies | Purpose |
|---------|------------|---------|
| `pymc >= 5.0` | 1, 2, 3, 4, 5, 6, 9, 14 | Bayesian inference and MCMC |
| `arviz` | 1, 2, 3, 6 | Posterior analysis, R-hat, ESS, convergence |
| `scikit-survival` or `lifelines` | 2, 14 | Cox PH baseline, martingale residuals |
| `numba` | 12 | JIT-compiled update kernel for < 50ms latency |
| `cvxpy` | Portfolio | Convex optimisation for Kelly constraints |
| `gstools` or `PyKrige` | 6 | Spatio-temporal GP / kriging for wind fields |
| `sdeint` or `torchsde` | 7, 13 | Numerical integration of SDEs and EKF |
| `scikit-fda` | 4 | Functional data analysis (FPCA on aging curves) |
| `statsmodels` | 5 | ITS with ARMA errors |
| `tick` | 10 | Hawkes process fitting |
| `filterpy` | 13 | Extended Kalman Filter |
| `scipy` | 8, 11, 15 | Optimisation, copulas, numerical integration |
| `copulas` | 15 | Clayton copula evaluation |
| `pgmpy` or `bnlearn` | 14 | Bayesian Network structure for risk types |
| `open_spiel` or `ray[rllib]` | 8, 11 | Multi-agent game theory (full POSG) |
| `numpy`, `pandas` | All | Data wrangling and numerical computation |

Install all: `pip install -r requirements.txt`

---

## 10. Quick Start

```python
import sqlite3
from genqirue.models import GruppettoFrailtyModel, TacticalTimeLossHMM
from genqirue.models import BayesianChangepointDetector
from genqirue.portfolio import RobustKellyOptimizer, KellyParameters

conn = sqlite3.connect('data/cycling.db')

# Strategy 2: fit frailty model on historical mountain stage data
frailty_model = GruppettoFrailtyModel()
frailty_model.fit(load_survival_data(conn))

# Strategy 1: fit tactical HMM
hmm_model = TacticalTimeLossHMM()
hmm_model.fit(load_tactical_data(conn))

# Generate positions for upcoming stage
positions = []
for rider_id in startlist:
    frailty = frailty_model.get_hidden_form_prob(rider_id)
    state   = hmm_model.get_tactical_state_prob(rider_id, stage_type, gc_gap)

    if frailty > 0.3 and state['PRESERVING'] > 0.6:
        positions.append(create_position(rider_id, market_odds))

# Optimise portfolio
params = KellyParameters(method='quarter_kelly', max_position_pct=0.25, min_edge_bps=50)
optimizer = RobustKellyOptimizer(params)
portfolio = optimizer.optimize_portfolio(positions, team_assignments)

for pos in portfolio.positions:
    if pos.stake > 0:
        print(f"Rider {pos.market_state.selection_id}: stake={pos.stake:.1%} odds={pos.market_state.back_odds:.1f}")
```

---

## License

Proprietary. For research and educational use only.
