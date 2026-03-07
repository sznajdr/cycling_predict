# Statistical Models Reference

Mathematical specification of all 15 betting strategies. Four are implemented; the remainder are fully specified and ready for implementation.

---

## Implementation Status

| # | Strategy | Category | Model | Status |
|---|----------|----------|-------|--------|
| 1 | Tactical Time Loss | Pre-race | HMM | Implemented |
| 2 | Gruppetto Frailty | Pre-race | Cox PH + frailty | Implemented |
| 3 | Medical Communiqué | Pre-race | Two-compartment PK | Specified |
| 4 | Youth Fade | Pre-race | Functional PCA | Specified |
| 5 | Rest Day Regression | Pre-race | ITS / BSTS | Specified |
| 6 | ITT Weather Arbitrage | Environmental | GP / SPDE | Implemented |
| 7 | Weather Mismatch H2H | Environmental | Langevin SDE | Specified |
| 8 | Desperation Breakaway | Game theory | POSG / QRE | Specified |
| 9 | Super-Domestique Choke | Game theory | Dirichlet Process | Specified |
| 10 | Mechanical Incident | Real-time | Marked Hawkes | Specified |
| 11 | Domestique Chokehold | Game theory | HJB differential game | Specified |
| 12 | Attack Confirmation | Real-time | BOCPD | Implemented |
| 13 | Gap Closing Calculus | Real-time | OU + EKF | Specified |
| 14 | Post-Crash Confidence | Risk | Joint frailty | Specified |
| 15 | Rain on Cobbles | Risk | Clayton copula + DP | Specified |

**Implementation order** (dependency chain): 2 → 12 → 1 → 6 → portfolio

---

## Category I: Pre-Race Form Signals

Overnight batch. Run on historical data, produce ranked rider lists before markets open.

---

### Strategy 1: Tactical Time Loss HMM

**File:** `genqirue/models/tactical_hmm.py`

**The edge.** Time gaps on mountain stages are a mix of fitness and tactics. A GC rider managing effort and a rider who cracked both show the same observed time loss. The model separates these states. Riders flagged as PRESERVING are bets for the following flat stage; those flagged as CONTESTING and still losing time are not.

**Model.** Hidden Markov Model with two latent states:

```
z_{i,t} ∈ {CONTESTING, PRESERVING}

P(z_{i,t} = PRESERVING) = sigmoid(δ_0 + δ_1 · ΔGC_{i,t} + δ_2 · StageType_t)

time_loss | CONTESTING  ~ Normal(μ, σ²)
time_loss | PRESERVING  ~ Normal(μ + γ, σ²),   γ ~ Normal⁺(2, 0.5)
```

`γ` is the tactical time loss — approximately 2 minutes — constrained positive. Fitted via MCMC using PyMC with `pm.Categorical` for latent states.

**Signal:** `P(PRESERVING) > 0.7` on a mountain stage → bet rider in the following transition stage.

---

### Strategy 2: Gruppetto Frailty

**File:** `genqirue/models/gruppetto_frailty.py`

**The edge.** Gruppetto riders on mountain stages are managing effort, not struggling. The market prices them poorly for the following flat stage. The question is which gruppetto riders are sandbagging versus at their genuine limit. The frailty term answers this.

**Model.** Bayesian Cox Proportional Hazards with rider-level random effects:

```
λ_i(t) = λ_0(t) · exp(β^T · X_i + b_i)

b_i ~ Normal(0, σ²_b)   [frailty term]
```

- `λ_0(t)` — baseline hazard (population dropout rate over time)
- `X_i` — covariates: GC position, seconds behind leader, gruppetto flag, time lost
- `b_i` — **frailty**: rider-specific random effect capturing unobservable resilience

A large positive `b_i` means the rider survived longer than their observable characteristics predict. Riders ranked by frailty after mountain stages are the transition-stage bets.

**Acceptance criterion:** frailty scores correlate with next-day transition stage performance at ρ > 0.3.

---

### Strategy 3: Medical Communiqué

**File:** `genqirue/models/medical_pk.py` *(to implement)*

**The edge.** Crash and illness news arrives with a lag and is priced crudely — either ignored or overreacted to. A pharmacokinetic model of trauma recovery gives a time-varying performance penalty the market doesn't have.

**Model.** Two-compartment pharmacokinetic model treating physical trauma as a decaying drug concentration:

```
dC_trauma/dt = -k_el · C_trauma

Perf(t) = Perf_baseline · (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

`k_el` is the elimination rate (recovery speed), estimated from historical return-to-form data after documented crashes. `EC_50` is the concentration at which performance is halved. The model outputs a predicted performance penalty curve by day post-incident.

---

### Strategy 4: Youth Fade

**File:** `genqirue/models/functional_pca.py` *(to implement)*

**The edge.** Age-related decline is not linear and not the same across rider types. Sprinters fade differently from climbers. GC riders peak later. The market prices age as a blunt heuristic; the model prices it as a personalised trajectory.

**Model.** Functional Principal Component Analysis applied to career performance trajectories:

```
X_i(t) = μ(t) + Σ_k ξ_{ik} · φ_k(t) + ε_i(t)
```

`μ(t)` is the population-average aging curve. `φ_k(t)` are the principal modes of variation — different shapes of career arc. `ξ_{ik}` are rider-specific loadings. Riders whose current performance exceeds their predicted trajectory are underpriced; those below are overpriced.

---

### Strategy 5: Rest Day Regression

**File:** `genqirue/models/interrupted_ts.py` *(to implement)*

**The edge.** Rest days reset physical state in ways the market treats as noise. An interrupted time series model separates the systematic rest-day effect from underlying form trends.

**Model.** Interrupted Time Series with ARMA errors:

```
Y_t = β_0 + β_1·Time_t + β_2·Intervention_t + β_3·TimeAfter_t
      + Σ_j φ_j · Y_{t-j} + ε_t
```

`Intervention_t` marks the rest day. `β_2` captures the immediate level shift; `β_3` captures the slope change after. The Bayesian Structural Time Series (BSTS) alternative adds a local trend component and handles multiple rest days cleanly.

---

## Category II: Environmental / Physical

---

### Strategy 6: ITT Weather Arbitrage

**File:** `genqirue/models/weather_spde.py`

**The edge.** ITT markets are efficient on form, often wrong on weather. A long ITT start window spans 3–4 hours. Riders who start into a headwind on key exposed sections versus a tailwind can differ by 30–90 seconds — swamping typical GC separations. When weather data arrives after markets open, there is a window.

**Model.** Gaussian Process over the wind field along the course:

```
w(s, t) ~ GP(μ(s,t), K((s,t), (s',t')))
```

The Matérn kernel captures spatial correlation (nearby sections have correlated wind) and temporal correlation (conditions 30 minutes apart are more similar than conditions 3 hours apart). For each rider, the model integrates expected headwind/tailwind exposure along their trajectory given their start time:

```
ΔT = ∫_0^D [P / F_aero(v_wind(t_early)) - P / F_aero(v_wind(t_late))] dx
```

The SPDE formulation (via sparse matrices using PyMC's `pm.gp.HSGP`) makes the computation tractable for large spatial domains.

**Acceptance criterion:** RMSE < 10 seconds vs actual ITT time differences between early and late starters.

---

### Strategy 7: Weather Mismatch H2H

**File:** `genqirue/models/weather_sde.py` *(to implement)*

**The edge.** H2H markets on cobble sectors and crosswind stages misprice handling ability. Some riders are structurally better in gusts. The market doesn't separate weather sensitivity from baseline speed.

**Model.** Langevin stochastic differential equation for bike velocity in stochastic wind:

```
m · dv/dt = F_drive - F_drag - F_gravity + σ_wind · ξ(t)
```

`ξ(t)` is white noise modelling wind gusts. Integrating this SDE produces a distribution over finishing times for each rider under different wind realisations. Riders with lower variance in the output — those whose times are less sensitive to wind — are preferred in crosswind conditions regardless of raw speed.

---

## Category III: Game Theory

---

### Strategy 8: Desperation Breakaway

**File:** `genqirue/models/breakaway_game.py` *(to implement)*

**The edge.** Late in a stage race, GC-irrelevant riders have strong incentive to go in breakaways. The market prices them on form. The game-theoretic model prices them on their strategic incentive, which is highest precisely when form signals are worst.

**Model.** Partially Observable Stochastic Game with states `S = (GC_positions, Stage_wins, Remaining_stages)`. Quantal Response Equilibrium:

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_a exp(λ · Q_i(s, a))
```

`Q_i` is the value of each action given the current state. `λ` controls how sharply riders best-respond. Riders with strong strategic incentives (GC lost, no stage win, few stages remaining) get higher breakaway probability regardless of form.

---

### Strategy 9: Super-Domestique Choke

**File:** `genqirue/models/mixed_membership.py` *(to implement)*

**The edge.** Some riders perform below their individual ability when leading domestique duties. Others elevate. The market treats domestiques as a category; the model treats them as a distribution.

**Model.** Mixed Membership Model (LDA-style) with Dirichlet Process prior:

```
θ_i ~ Dirichlet(α)
w_{i,t} ~ Σ_k θ_{ik} · Normal(μ_k, Σ_k)
```

Each rider is a mixture of latent types — leader, domestique, opportunist. Riders with high weight on the domestique component are expected to underperform their physical ceiling in protection duties.

---

### Strategy 11: Domestique Chokehold

**File:** `genqirue/models/chase_game.py` *(to implement)*

**The edge.** The optimal time to attack a protected leader is before the domestiques are dropped — forcing early work. The market prices attacks on pace; the model prices attacks on when protection expires.

**Model.** Differential game between attacker and chaser. The value function is governed by the Hamilton-Jacobi-Bellman equation:

```
∂V/∂t + min_{u_chase} max_{u_break} [∇V · f(x, u_break, u_chase) + g(x)] = 0
```

State `x` is the gap plus remaining domestique count. The Nash equilibrium power allocation determines when the chase will be abandoned. Riders attacking precisely when domestique protection expires have higher breakaway survival probability.

---

## Category IV: Real-Time / Live

Latency requirement: under 100ms per update.

---

### Strategy 10: Mechanical Incident

**File:** `genqirue/models/hawkes_mechanical.py` *(to implement)*

**The edge.** Mechanical incidents cluster in time and space. Markets reprice slowly after a mechanical. If the model estimates recovery probability before the market adjusts, there is a window.

**Model.** Marked Hawkes Process — self-exciting point process where each incident increases short-term probability of further incidents:

```
λ_t = μ + Σ_{t_i < t} φ(t - t_i, m_i)
```

`φ` is the excitation kernel; `m_i` is the mark (type and severity of incident). A Dirichlet Process updates allow the model to learn new incident patterns in real time. The model outputs time-to-rejoin, probability of successful chase, and residual abandonment probability.

---

### Strategy 12: Attack Confirmation (BOCPD)

**File:** `genqirue/models/online_changepoint.py`

**The edge.** Live markets on breakaway survival move on information arriving in real time. If the model can confirm an attack is structural — not a test, not a reaction — before the TV feed processes it, there is a timing edge.

**Model.** Bayesian Online Changepoint Detection (Adams & MacKay, 2007). Maintains a posterior over run length — time since the last structural break in the gap time series:

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) · P(r_t | r_{t-1})
```

At each new observation the posterior updates in under 100ms (Numba-accelerated).

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

**Decision rule:** bet when `P(changepoint) > 0.8` AND yesterday's Z-score `> 2.0`.

**Acceptance criterion:** attack confirmed within 30 seconds of power surge on historical validation data.

---

### Strategy 13: Gap Closing Calculus

**File:** `genqirue/models/gap_ou_process.py` *(to implement)*

**The edge.** Live catch probability is priced on the current gap and eyeball assessment. The model prices on gap dynamics — whether the gap is mean-reverting (chase will catch) or diverging (breakaway survives). These are structurally different situations that look the same in the short run.

**Model.** Ornstein-Uhlenbeck process for the gap:

```
dG_t = θ(μ - G_t) dt + σ dW_t
```

`θ` controls mean-reversion speed; `μ` is the equilibrium gap. The first passage time — probability the gap hits zero by the finish — gives catch probability directly. An Extended Kalman Filter estimates `θ`, `μ`, and `σ` in real time from live timing splits.

---

## Category V: Risk Modelling

---

### Strategy 14: Post-Crash Confidence

**File:** `genqirue/models/joint_frailty.py` *(to implement)*

**The edge.** After a significant crash, the market prices physical damage. It does not price the confidence loss — earlier braking on descents, wider lines, less aggressive positioning. This persistent penalty the market systematically underestimates.

**Model.** Joint frailty model with shared random effects across multiple risk types:

```
λ_{ij}(t) = λ_{0j}(t) · exp(β^T · X_{ij} + b_i + ε_{ij})
```

`b_i` is a rider-level shared frailty across all risk types (descent, corner, wet). `ε_{ij}` is risk-type-specific. A rider with high crash frailty on descents but normal frailty elsewhere is specifically exposed to technical finishes — not to flat stages. Bayesian Networks encode conditional dependencies between risk types.

---

### Strategy 15: Rain on Cobbles

**File:** `genqirue/models/cobble_reliability.py` *(to implement)*

**The edge.** Wet cobble sectors produce correlated failures — when one rider punctures, others are more likely to. Markets price rider puncture probability independently. The model prices it jointly.

**Model.** Clayton copulas for sector-to-sector performance correlation:

```
C_θ(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
```

`θ` controls tail dependence — how much sector failures cluster. In wet conditions `θ` rises. The joint survival probability across all cobble sectors is materially lower than the product of individual probabilities.

Dynamic programming solves the optimal pacing strategy given this survival risk:

```
V_s(v) = min_{v'} [λ_s(v') · DNF_cost + L_s/v' + V_{s+1}(v')]
```

---

## Portfolio: Robust Kelly

**File:** `genqirue/portfolio/kelly.py`

Kelly fraction maximises expected log-wealth (bankroll growth rate):

```
f* = (b·p - (1-p)) / (b - 1)
```

where `b` is decimal odds. Full Kelly is theoretically optimal but sensitive to model miscalibration. **The system defaults to quarter-Kelly** (`f = f*/4`).

When the model provides a posterior standard deviation `σ_p` on the probability estimate, Robust Kelly automatically derisks:

```
f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
```

Portfolio constraints are solved via CVXPY:

```python
constraints = [
    cp.sum(f) <= 1.0,              # total allocation
    f >= 0,
    f <= 0.25,                     # max 25% per position
    cp.quad_form(f, cov) <= 0.05,  # variance bound
    # CVaR_95 <= 0.10              # tail risk bound
]
```

CVaR (Conditional Value at Risk at 95%) is the expected loss in the worst 5% of scenarios — the appropriate tail risk measure for a betting book where ruin is permanent.

---

## Dependencies

| Package | Purpose | Strategies |
|---------|---------|------------|
| `pymc >= 5.0` | Bayesian inference, MCMC | 1, 2, 6 |
| `arviz` | Posterior analysis, convergence diagnostics | 1, 2, 6 |
| `scikit-survival` or `lifelines` | Cox PH models | 2, 14 |
| `numba` | <100ms inference via JIT | 12 |
| `cvxpy` | Portfolio convex optimisation | Portfolio |
| `gstools` or `PyKrige` | Geostatistics, GP spatial models | 6 |
| `sdeint` or `torchsde` | Stochastic differential equations | 7, 13 |
| `ray[rllib]` or `open_spiel` | Multi-agent RL, game theory | 8, 11 |

Install all: `pip install -r requirements.txt`
