# Onboarding


---

---

### Pricing Form After a Rest Day → Interrupted Time Series (Strategy 5)

A rider is fading through a grand tour: 2nd, 4th, 7th, 11th across successive stages. The rest day hits. You know some riders reset; others carry the fatigue through. You mentally adjust post-rest prices based on historical response patterns.

The BSTS/ITS model decomposes this precisely:

```
Y_t = β_0 + β_1·Time_t + β_2·Intervention_t + β_3·TimeAfter_t + Σ_j φ_j·Y_{t-j} + ε_t
```

- `β_2`: immediate level shift at the rest day (bump or drop)
- `β_3`: trajectory change after (does the trend flatten, reverse, steepen?)
- `φ_j`: autoregressive momentum terms (what trajectory were they on entering it?)

The market applies a population-average rest-day adjustment, or none. The model estimates individual rest-day response per rider from their history.

---

### Spotting Desperation Breakaways → Quantal Response Equilibrium (Strategy 8)

It's Stage 18. A rider is 45 minutes down in GC — irrelevant for the overall, no stage wins, two stages remaining. You watch them attack a transition stage and think: of course. The incentive structure overrides form. You price them above their wattage suggests.

The POSG model formalizes this:

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_a exp(λ · Q_i(s, a))
```

State `s` encodes: GC deficit, stage wins, stages remaining, stage type. Q-value `Q_i(s, attack)` is high for any rider where GC is lost, wins are zero, and opportunities are running out — independent of watts. The market prices on observed form signals; the model prices on incentive.

---

### Gruppetto Riders Sandbagging → Cox PH Frailty (Strategy 2)

A gruppetto finisher on a mountain stage shows 12 minutes down. That gap is a mixture of physical limit and tactical conservation. The question is which portion is which.

The frailty model separates them:

```
λ_i(t) = λ_0(t) · exp(β^T · X_i + b_i),   b_i ~ Normal(0, σ²)
```

The frailty term `b_i` is the rider-specific random effect absorbing everything the observable covariates (weight, recent results, specialty scores) miss. Positive `b_i` — the rider survived substantially longer than form predicts. That excess is the sandbagging signal. The model flags high-frailty gruppetto riders for the next transition stage where the market hasn't adjusted.

---

### Kelly Staking → Robust Kelly + CVaR Optimizer

You find edge: model says 25%, market implies 20% at odds of 4.0. Kelly gives `f* = (bp - q)/b`. But full Kelly on miscalibrated models produces ruin. You size down.

The optimizer applies three constraints:

1. **Quarter-Kelly default:** `f = f*/4` — ~30% less long-run growth, ~4× lower drawdown variance
2. **Robust Kelly:** When the model returns a posterior standard deviation `σ_p`, stake is automatically reduced:
   ```
   f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
   ```
   Higher model uncertainty → tighter stake. High-odds bets under an uncertain model are penalised more than short-odds bets under the same uncertainty.
3. **CVaR constraint:** Portfolio-level bound on expected loss in the worst 5% of scenarios, enforced via CVXPY.

Code: `genqirue/portfolio/kelly.py`.

---

## Part 2: Why the Market Misprices

| Inefficiency | Book approach | Model approach |
|---|---|---|
| **ITT wind** | Expected conditions at market open | GP over full wind-field distribution; integrate over all realisations weighted by forecast probability |
| **Gruppetto riders** | Observed time loss → weak signal | Cox PH frailty: decompose time loss into physical limit vs. tactical conservation |
| **Post-crash recovery** | Binary: out or fine | PK model: continuous recovery curve with rider-specific elimination rates |
| **Rest day effects** | Population average or ignored | ITS/BSTS: individual rest-day response estimated from career history |
| **Cobble sectors** | Independent rider pricing | Clayton copula: joint tail dependence — when one punctures, conditional survival probability of others drops |
| **Desperation attacks** | Form-based probability | QRE: incentive-weighted probability that overrides form when GC is irrelevant |

---

## Part 3: Setup

**Requirements:** Python 3.11 or 3.13. The `procyclingstats` library must live adjacent to this repo:

```
parent_folder/
  cycling_predict/
  procyclingstats/
```

```bash
git clone https://github.com/ramonvermeulen/procyclingstats.git ../procyclingstats
pip install -e ../procyclingstats
pip install -r requirements.txt
python scripts/fetch_odds.py --init-schema
python scripts/quickstart.py
```

`quickstart.py` succeeding confirms the environment is functional. For full setup automation: `python scripts/setup_team.py`.

---

## Part 4: Data Pipeline

```
ProCyclingStats ──▶ SQLite DB ──▶ Models ──▶ Betclic Odds ──▶ Positions
  pipeline/           data/       genqirue/    fetch_odds.py   kelly.py
  runner.py         cycling.db
```

**Scraping.** `pipeline/runner.py` hits PCS at ~1 req/s via a persistent job queue (`fetch_queue`). Every job is idempotent — stop and restart anytime, the queue resumes. Each job transitions through: `pending → in_progress → completed` (or `failed → permanent_fail` after max retries).

**Database layers:**

| Layer | Tables | Content |
|---|---|---|
| Historical results | `riders`, `races`, `race_stages`, `rider_results`, `startlist_entries` | Stage results, time gaps, rider profiles, stage metadata |
| Real-time telemetry | `telemetry_changepoints` | Live power data for BOCPD (Strategy 12) |
| Weather | `weather_fields`, `itt_time_predictions` | Wind field snapshots, expected ITT time deltas |
| Market data | `bookmaker_odds`, `bookmaker_odds_latest` | Betclic odds snapshots; view returns most recent per selection |
| Model outputs | `rider_frailty`, `tactical_states`, `strategy_outputs` | Fitted model signals, ranked stage output with edge and Kelly |

**Key columns:**
- `rider_results.time_behind_winner_seconds` — raw time gap (input to Strategy 1)
- `rider_results.result_category` — `'stage'` or `'gc'`
- `race_stages.stage_type` — `'flat'`, `'hilly'`, `'mountain'`, `'itt'`, `'ttt'`
- `riders.sp_climber`, `sp_sprint`, `sp_gc` — specialty scores 0–100
- `bookmaker_odds.back_odds`, `fair_odds` — raw and hold-adjusted prices
- `bookmaker_odds.market_type` — `'winner'`, `'top_3'`, `'h2h'`, etc.

---

## Part 5: First Execution

**Configure races** in `config/races.yaml`:

```yaml
year: 2026

races:
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]
```

**Scrape:**

```bash
python -m pipeline.runner
python scripts/monitor.py          # watch progress in a second terminal
```

One race with 4 years of history: 20–60 minutes. A grand tour with 5 years: 4–6 hours. Full schema and job-type reference: [`docs/SCRAPE.md`](docs/SCRAPE.md).

**Rank a specific stage:**

```bash
python scripts/rank_stage.py paris-nice 2026 1
python scripts/rank_stage.py paris-nice 2026 1 --run-models   # fit frailty + tactical first
python scripts/rank_stage.py paris-nice 2026 3 --top 20       # top 20 only
python scripts/rank_stage.py paris-nice 2026 1 --save         # persist to strategy_outputs
```

`rank_stage.py` combines up to six pre-race signals (specialty with finish-type blending, cross-race recent form, historical, frailty, tactical, GC relevance) into a probability distribution over the startlist. Uphill finish detection (via `race_climbs` cumulative distance mapping) adjusts specialty column blending and applies a power-to-weight factor for mountain stages. It then joins live Betclic odds from `bookmaker_odds_latest`, computes edge in basis points, and sizes stakes via half-Kelly. Signals for which no data is available are omitted; the remaining weights are renormalized automatically.

Full signal documentation and output format: [`docs/RANKING.md`](docs/RANKING.md).

**Fit models and generate signals (multi-strategy workflow):**

```bash
python scripts/example_betting_workflow.py
```

Fits Strategies 1 and 2 on your scraped data, queries `bookmaker_odds_latest` for live odds (falls back to simulated fair-market odds if none), runs the Kelly optimizer, and prints recommended positions.

**Sample output:**

```
Top Opportunities (edge > 50bps):
Rider               Model Prob   Market Prob   Edge (bps)   Kelly Stake
Rider A             0.085        0.042         430          2.1%
Rider B             0.062        0.031         310          1.5%
```

**Run the backtest:**

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --strategy frailty --kelly 0.1 --save-bets bets.csv
```

Walk-forward: train on all races before R, predict R, record outcome, advance. No lookahead.

**Reading the backtest output:**

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

- **Top3%:** Naive baseline in a 150-rider field is 2%. Frailty at 5.6% is 2.8× — directionally real, but the confidence interval at n=72 is ±2.7pp. Target 200+ bets before treating any metric as stable.
- **ROI:** Against a simulated zero-margin market. Subtract 5–15% for real book overround.
- **MaxDD:** 15% on quarter-Kelly is expected. Above 40% suggests model miscalibration or over-sizing.
- **Spearman ρ:** Significance threshold at n=72 is ρ > 0.23. At 0.077, not yet resolved from noise.

---

## Part 6: The Sharp Strategies

### Strategy 6 (ITT Weather): Structural Arb

The setup: ITT start window spans 3–4 hours. Weather forecast updates after markets open. Early starters into a headwind vs. late starters into a tailwind typically differ by 30–90 seconds — swamping GC separations of 10–30 seconds.

```
ΔT = ∫_0^D [P/F_aero(v_wind(t_early)) - P/F_aero(v_wind(t_late))] dx
```

The model integrates the expected time delta along the course under the updated forecast. When markets are still priced on the original forecast, the edge is the difference in ΔT between the two wind realisations. Back late starters in H2H, lay early starters. The `σ_p` on the GP posterior is high when forecast confidence is low — Robust Kelly sizes down automatically.

### Strategy 12 (BOCPD): Latency Arbitrage

Live markets reprice on TV. TV lags GPS/power data by 15–30 seconds. BOCPD confirms structural pace changes in <50ms:

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) · P(r_t | r_{t-1})
```

Bet signal: `P(changepoint) > 0.8` AND prior-stage Z-score > 2.0. At that threshold, the model has confirmed the attack is structural before the TV feed has processed it. Infrastructure requirement: direct power/GPS data feed, sub-second API latency.

### Pre-Race vs. Real-Time

| | Pre-Race (1–9) | Real-Time (10, 12, 13) |
|---|---|---|
| Latency tolerance | Hours (overnight batch) | < 100ms |
| Data source | Historical DB | Live telemetry |
| Edge type | Information asymmetry | Speed asymmetry |
| Position sizing | Larger (more time, better liquidity) | Smaller, faster, higher Sharpe |
| Implementation | PyMC (MCMC) | Numba JIT |

---

## Part 7: Extending the System

All new strategies inherit from `BayesianModel` in `genqirue/models/base.py` and implement four methods:

```python
def build_model(self, data: dict) -> None:
    """Define the PyMC model — priors, likelihood, parameters."""

def fit(self, data: dict) -> None:
    """Run MCMC sampling — calibrate to historical data."""

def predict(self, new_data: dict) -> dict:
    """Return win probability and posterior uncertainty for new data."""

def get_edge(self, prediction: dict, market_odds: float) -> float:
    """Return edge in basis points. Positive = value bet."""
```

The portfolio optimizer calls `get_edge()` and `predict()['uncertainty']` uniformly across all strategies — the interface is what makes multi-strategy Kelly sizing possible.

**Implementation checklist** (see `CONTRIBUTING.md` for detail):
1. `genqirue/models/<strategy_name>.py` — inherit `BayesianModel`
2. `genqirue/data/schema_extensions.sql` — add tables if new data is needed
3. `tests/betting/test_strategies.py` — unit tests, no network calls
4. `genqirue/models/__init__.py` — export the class
5. `docs/ENGINE.md` — update strategy status table and acceptance criterion

**Kelly parameters** (adjust in `genqirue/portfolio/kelly.py`):

```python
KellyParameters(
    method='quarter_kelly',   # 'full', 'half', 'quarter', 'eighth'
    max_position_pct=0.25,    # single-position cap
    min_edge_bps=50,          # minimum edge threshold
    cvar_limit=0.10           # CVaR bound at 95%
)
```

Quarter-Kelly is the right default until model calibration is confirmed over 200+ bets. Half-Kelly is appropriate when the backtest Spearman ρ is significant and the out-of-sample period matches the in-sample metrics.

---

## Part 8: Trader Override Protocol

Three situations where domain knowledge should override model output:

**1. News not yet in PCS.** Rider had a fever last night; PCS communiqué hasn't appeared. Model is bullish based on form. Fade them manually.

**2. Team dynamics not captured.** Model sees two strong riders on the same team. You know the team has publicly committed to one leader. The second rider is working, not contesting.

**3. Course changes.** Organizers announced a detour; weather model is using the original route.

Override options:

```python
# Halve probability, double uncertainty
if rider_id == 123:
    prediction['win_prob'] *= 0.5
    prediction['uncertainty'] *= 2

# Exclude entirely
if rider_id in manual_exclusions:
    continue
```

Log every override: reason, timestamp, expected vs. actual outcome. Review monthly. If manual overrides are not adding positive expectation, stop overriding.

---

## Reference

| File | Purpose |
|------|---------|
| `docs/ENGINE.md` | Implementation logic, data inputs/outputs, acceptance criteria for all 15 strategies |
| `docs/MODELS.md` | Mathematical specifications — equations, priors, dependency chain |
| `docs/RANKING.md` | Stage ranking model — six signals, finish-type blending, weight matrix, softmax calibration, CLI usage |
| `docs/COMMANDS.md` | Complete CLI, SQL, scheduling, monitoring reference |
| `docs/ODDS.md` | Betclic scraper walkthrough — market types, H2H handling, name matching |
| `docs/SCRAPE.md` | Scraping pipeline — schema, job types, execution flow |
| `genqirue/models/base.py` | Abstract base class — required interface for all strategies |
| `genqirue/portfolio/kelly.py` | Robust Kelly + CVaR optimizer |
| `config/races.yaml` | Race configuration |

**MCMC convergence diagnostics** (logged after each `fit()` call):
- `R-hat < 1.01`: chains have converged
- `ESS > 400`: effective sample size is adequate
- Zero divergences: sampler is geometrically sound

When R-hat is elevated, the model is uncertain — Robust Kelly will downsize stakes automatically. Do not bet on high-R-hat models until you understand why the chains disagree.
