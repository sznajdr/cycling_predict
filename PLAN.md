```markdown
# Genqirue: Bayesian Cycling Betting Engine - Implementation Prompt

Build a production-grade betting intelligence system implementing the 15 research-grade statistical models from the provided mathematical framework. This is a hierarchical Bayesian system for professional cycling (Grand Tours) betting.

## Core Mathematical Requirements

Implement these specific stochastic models exactly as specified:

### Category I: Hierarchical Bayesian Models
1. **Tactical Time Loss (Strategy 1)**: Hidden Markov Switching model with latent state $z_{i,t} \sim \text{Bernoulli}(\pi_{i,t})$ where $\pi_{i,t} = \text{logit}^{-1}(\delta_0 + \delta_1 \cdot \Delta GC_{i,t} + \delta_2 \cdot StageType_t)$. Use PyMC with `pm.Categorical` for latent states and truncated normal $\mathcal{N}^+(2, 0.5)$ for tactical time loss $\gamma_1$.

2. **Gruppetto Outlier (Strategy 2)**: Competing risks survival with Bayesian Cox PH and time-varying frailty $b_i \sim \mathcal{N}(0, \sigma^2_b)$. Calculate martingale residuals as proxy for frailty. SQL schema must support `rider_frailty` table with hidden_form_prob calculation.

3. **Medical Communiqué (Strategy 3)**: Two-compartment PK model $\frac{dC_{trauma}}{dt} = -k_{el} \cdot C_{trauma}$ with $Perf(t) = Perf_{baseline} \cdot \left(1 - \frac{C_{trauma}(t)}{EC_{50} + C_{trauma}(t)}\right)$. Implement Robust Kelly sizing with posterior uncertainty.

### Category II: Functional & Longitudinal
4. **Youth Fade (Strategy 4)**: Functional PCA on aging curves $X_i(t) = \mu(t) + \sum_{k=1}^K \xi_{ik} \phi_k(t) + \epsilon_i(t)$. Use Gompertz-Makeham with functional covariates for survival.

5. **Rest Day Regression (Strategy 5)**: Interrupted Time Series with ARMA errors: $Y_t = \beta_0 + \beta_1 \cdot Time_t + \beta_2 \cdot Intervention_t + \beta_3 \cdot TimeAfter_t + \sum_{j=1}^p \phi_j Y_{t-j} + \epsilon_t$. Include BSTS (Bayesian Structural Time Series) alternative.

### Category III: Physics & SPDEs
6. **ITT Weather Arbitrage (Strategy 6)**: Spatio-temporal Gaussian Process with Matérn kernel $k((s,t),(s',t'))$ for wind field $v_{wind}(s,t) \sim \mathcal{GP}(0, k)$. Implement Physics-Informed Neural Network (PINN) layer for Navier-Stokes drag calculations. Calculate fair time difference $\Delta T = \int_0^D \frac{P}{F_{aero}(v_{wind}(t_{early}))} - \frac{P}{F_{aero}(v_{wind}(t_{late}))} dx$.

7. **Weather Mismatch H2H (Strategy 7)**: Vector field analysis with Langevin equation $m \frac{dv}{dt} = F_{drive} - F_{drag} - F_{gravity} + \sigma_{wind} \cdot \xi(t)$ for bike handling in gusts.

### Category IV: Game Theory
8. **Desperation Breakaway (Strategy 8)**: Partially Observable Stochastic Game (POSG) with states $S = (GC\_positions, Stage\_wins, Remaining\_stages)$. Implement Quantal Response Equilibrium (QRE): $P(a_i|s) = \frac{\exp(\lambda \cdot Q_i(s,a_i))}{\sum_{a'} \exp(\lambda \cdot Q_i(s,a'))}$ using Counterfactual Regret Minimization (CFR) or MADDPG. Include hedonic game theory for coalition formation with characteristic function $v(S)$.

9. **Super-Domestique Choke (Strategy 9)**: Mixed Membership Model (LDA-style) with $\theta_i \sim \text{Dir}(\alpha)$ and $w_{i,t} \sim \sum_{k=1}^K \theta_{ik} \cdot \mathcal{N}(\mu_k, \Sigma_k)$. Use Dirichlet Process Mixture Models (Bayesian nonparametric).

### Category V: Online Learning & Stochastic Calculus
10. **Mechanical Incident (Strategy 10)**: Marked Hawkes process $\lambda_t = \mu + \sum_{t_i < t} \phi(t-t_i, m_i)$ with Dirichlet Process updates for winner probabilities. Implement <50ms latency SGD updates using GPU (TensorRT).

11. **Domestique Chokehold (Strategy 11)**: Differential game theory with Hamilton-Jacobi-Bellman equation $\frac{\partial V}{\partial t} + \min_{u_{chase}} \max_{u_{break}} \left[\nabla V \cdot f(x, u_{break}, u_{chase}) + g(x)\right] = 0$. Solve Nash Bargaining for Pareto optimal power allocation.

12. **Attack Confirmation (Strategy 12)**: Bayesian Online Changepoint Detection (Adams & MacKay 2007): $P(r_t | x_{1:t}) = \frac{P(r_{t-1}, x_t | x_{1:t-1})}{\sum_{r_{t-1}} P(r_{t-1}, x_t | x_{1:t-1})}$ with Weibull hazard. Use Particle Filter for non-linear dynamics. Decision rule: bet when $P(|r_t - r_{t-1}| > 0 | Data) > 0.8$ AND yesterday's Z-score > 2.0.

13. **Gap Closing Calculus (Strategy 13)**: Ornstein-Uhlenbeck process $dG_t = \theta(\mu - G_t)dt + \sigma dW_t$ with First Passage Time calculation for catch probability. Implement Extended Kalman Filter for real-time parameter estimation from live telemetry.

14. **Post-Crash Confidence (Strategy 14)**: Joint frailty models $\lambda_{ij}(t) = \lambda_{0j}(t) \exp(\beta^T X_{ij} + b_i + \epsilon_{ij})$ with shared frailty across risk types (descent, corner, wet). Use Bayesian Networks for conditional dependencies.

15. **Rain on Cobbles (Strategy 15)**: Reliability theory with Clayton copulas $C_\theta(u,v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}$ for sector-to-sector correlation. Dynamic programming pacing optimization via Bellman equation $V_s(v) = \min_{v'} \left[\lambda_s(v') \cdot DNF\_cost + \frac{L_s}{v'} + V_{s+1}(v')\right]$.

## System Architecture
```

genqirue/ ├── domain/ │ ├── entities.py # RiderState, StageContext, MarketState dataclasses │ └── enums.py # StageType, TacticalState, RiskType ├── models/ │ ├── base.py # BayesianModel ABC with PyMC infrastructure │ ├── tactical_hmm.py # Strategy 1 │ ├── gruppetto_frailty.py # Strategy 2 (CRITICAL: implement first) │ ├── medical_pk.py # Strategy 3 (Pharmacokinetic) │ ├── functional_pca.py # Strategy 4 (scikit-fda or custom) │ ├── interrupted_ts.py # Strategy 5 (Causal inference) │ ├── weather_spde.py # Strategy 6 (Gaussian Processes) │ ├── weather_sde.py # Strategy 7 (Langevin/SDE) │ ├── breakaway_game.py # Strategy 8 (OpenSpiel or custom POSG) │ ├── mixed_membership.py # Strategy 9 (Dirichlet Process) │ ├── hawkes_mechanical.py # Strategy 10 (Hawkes + online learning) │ ├── chase_game.py # Strategy 11 (Differential games) │ ├── online_changepoint.py# Strategy 12 (Numba-optimized, <50ms) │ ├── gap_ou_process.py # Strategy 13 (Kalman filtering) │ ├── joint_frailty.py # Strategy 14 (Shared frailty) │ └── cobble_reliability.py# Strategy 15 (Copulas + DP) ├── data/ │ ├── schema.sql # PostgreSQL/ClickHouse schema (refer to SQL in doc) │ ├── etl.py # ProCyclingStats/Strava ingestion │ └── features.py # Feature engineering pipeline ├── inference/ │ ├── real_time.py # Latency-critical inference (strategies 10-13) │ └── batch.py # Overnight model refitting ├── portfolio/ │ ├── kelly.py # Robust Kelly with uncertainty penalty │ └── cvar.py # CVaR constraints, Markowitz optimization ├── execution/ │ ├── odds_feed.py # Market data ingestion │ └── position_sizing.py # Bet placement logic └── validation/ ├── backtest.py # Walk-forward analysis ├── scoring.py # Brier score, CRPS, RPS └── calibration.py # Reliability diagrams, drift detection

plain

Copy

````plain

## Technical Implementation Details

### Infrastructure Base Class
Create `models/base.py` with:
- PyMC model specification interface
- Automatic Arviz posterior storage
- Sample post-processing for strategy outputs
- Convergence diagnostics (R-hat, ESS)

### Critical Implementation Order
1. **START HERE**: `gruppetto_frailty.py` (Strategy 2) - provides frailty scores used by Strategies 1, 3, 4, 14
2. **Second**: `online_changepoint.py` (Strategy 12) - highest value real-time signal
3. **Third**: `tactical_hmm.py` (Strategy 1) - core alpha generator
4. **Fourth**: `weather_spde.py` (Strategy 6) - high edge potential, complex implementation
5. **Fifth**: `kelly.py` (portfolio optimization) - must handle posterior uncertainty

### Specific Code Requirements

For Strategy 2 (Gruppetto), implement this SQL view exactly:
```sql
CREATE VIEW rider_hidden_form AS
SELECT rider_id,
       SUM(event - cumulative_hazard) OVER (
           PARTITION BY rider_id 
           ORDER BY stage_date 
           ROWS UNBOUNDED PRECEDING
       ) as frailty_estimate,
       CASE 
           WHEN frailty_estimate > (AVG(frailty) OVER () + 1.5*STDDEV(frailty) OVER ()) 
           THEN 1 - CDF_NORMAL(frailty_estimate, 0, STDDEV(frailty) OVER ())
           ELSE 0 
       END as hidden_form_prob
FROM survival_analysis
WHERE stage_type = 'mountain';
````

For Strategy 12 (Changepoint), use Numba JIT:

Python

Copy

```python
@jit(nopython=True)
def update_run_length(x_t, R_prev, hazard, pred_prob):
    growth = R_prev * pred_prob * (1 - hazard)
    cp_prob = np.sum(R_prev * pred_prob * hazard)
    R_new = np.zeros_like(R_prev)
    R_new[0] = cp_prob
    R_new[1:] = growth[:-1]
    return R_new / np.sum(R_new)
```

For Strategy 6 (Weather), use PyMC's HSGP approximation:

Python

Copy

```python
with pm.Model() as weather_model:
    # Use Hilbert Space GP for computational efficiency
    gp = pm.gp.HSGP(cov_func=matern_kernel, m=[20, 20, 10])
    f = gp.prior("f", X=spatiotemporal_coords)
```

### Data Schema Requirements

Implement tables for:

- `rider_frailty` (martingale residuals, hidden_form_prob)
    
- `telemetry_changepoints` (run_length rt​ , changepoint_prob)
    
- `pk_parameters` (elimination rates kel​ , EC50​ per rider)
    
- `weather_fields` (GP posterior samples for SPDE)
    
- `breakaway_games` (POSG state histories)
    
- `positions` (live telemetry for Kalman filtering)
    

### Portfolio Math

Implement Robust Kelly:

fi∗​=bi​p^​i​(bi​+1)−1​⋅(1−p^​i2​(bi​+1)2σi2​⋅(bi​+1)2​)

With CVaR constraint using `cvxpy`:

Python

Copy

```python
constraints = [
    cp.sum(f) <= 1.0,
    f >= 0,
    cp.quad_form(f, cov_matrix) <= 0.05,  # Variance
    # Add CVaR constraint here
]
```

## Acceptance Criteria

1. **Strategy 2** produces frailty scores correlating with next-day transition stage performance (correlation > 0.3)
    
2. **Strategy 12** detects attacks within 30 seconds of power surge (validation on historical attack data)
    
3. **Strategy 6** calculates ΔT between early/late ITT starters with RMSE < 10 seconds vs actual results
    
4. **Kelly optimizer** never allocates >25% to single position, handles covariance between riders on same team
    
5. **All models** provide posterior predictive distributions (not just point estimates) for uncertainty quantification
    
6. **Inference latency**: Strategies 10-13 < 100ms per update, Strategies 1-9 < 5 minutes batch processing
    

## Dependencies

- `pymc >= 5.0` (Bayesian inference)
    
- `arviz` (posterior analysis)
    
- `lifelines` or `scikit-survival` (Cox models, Strategy 2)
    
- `sdeint` or `torchsde` (SDEs, Strategy 13)
    
- `gstools` or `pykrige` (Geostatistics, Strategy 6)
    
- `cvxpy` (Portfolio optimization)
    
- `numba` (Strategy 12 speed)
    
- `clickhouse-driver` or `asyncpg` (Database)
    
- `ray[rllib]` or `open_spiel` (Strategy 8 multi-agent)
    

Begin with `models/base.py` and `models/gruppetto_frailty.py`. Ensure all models inherit from base class and implement `build_model()`, `fit()`, `predict()`, and `get_edge()` methods.