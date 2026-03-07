# Cycling Predict

A Bayesian betting engine for professional cycling. Scrapes ProCyclingStats, fits probabilistic models that identify market mispricings, and validates them through walk-forward backtesting.

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [The Models](#2-the-models)
3. [Portfolio Construction](#3-portfolio-construction)
4. [Backtesting](#4-backtesting)
5. [Project Structure](#5-project-structure)
6. [Setup](#6-setup)
7. [Running It](#7-running-it)
8. [Live Odds (Betclic)](#8-live-odds-betclic)
9. [Interpreting Results](#9-interpreting-results)
10. [Quick Command Reference](#10-quick-command-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What the System Does

Three stages:

**Stage 1 — Data.** A scraper pulls historical results from ProCyclingStats: stage results, time gaps, startlists, rider profiles, stage metadata (type, elevation, distance). This goes into a local SQLite database.

**Stage 2 — Modelling.** Statistical models fit on that historical data and produce probability estimates for riders in upcoming stages. Each model targets a specific market inefficiency — situations where the observed data implies a different probability than what the market is pricing.

**Stage 3 — Validation.** A walk-forward backtester replays history in strict chronological order — training only on what was available before each race, predicting each race, observing outcomes — and produces a P&L record you can interrogate. No lookahead.

The output is a ranked list of riders with model-implied probabilities, edge estimates against market odds, and Kelly-sized stakes. Whether you act on it is up to you.

---

## 2. The Models

Fifteen strategies across five categories. Four are implemented; the rest are mathematically specified and scheduled. Each targets a structural mispricing the market consistently fails to price.

---

### Pre-race form signals (Strategies 1–5)

Overnight batch. Run on historical data, produce ranked rider lists before markets open.

---

#### Strategy 1: Tactical HMM (Hidden Markov Model) — IMPLEMENTED

**The edge.** Every time gap on a stage is a mixture of fitness and tactics. The market cannot separate a GC rider who cracked from one who soft-pedalled — they both show 2 minutes down. If the model can identify who was managing effort, that rider is a bet for the next stage. The one who cracked is not.

**What the model does.** A **Hidden Markov Model** with two latent states:

- **CONTESTING** — racing at capacity; time loss reflects true fitness
- **PRESERVING** — deliberately holding back; time loss is tactical

The latent state `z_{i,t}` is unobservable. The observed variable is time loss. The model posits:

```
P(z_{i,t} = PRESERVING) = sigmoid(delta_0 + delta_1 * GC_gap + delta_2 * IsHardStage)
```

Riders far down on GC have less incentive to fight on hard stages. Time loss conditional on state:

```
time_loss | CONTESTING  ~ Normal(mu, sigma^2)
time_loss | PRESERVING  ~ Normal(mu + gamma, sigma^2)
```

`gamma` is the tactical time loss — approximately 2 minutes — constrained positive. Fitted via MCMC. Riders with high `P(PRESERVING)` on mountain stages are flagged for the following flat stage.

---

#### Strategy 2: Gruppetto Frailty (Cox Proportional Hazards) — IMPLEMENTED

**The edge.** Gruppetto riders on a mountain stage are managing effort, not struggling. The market prices them as poor candidates for the following flat stage. They are often fresher than GC riders who emptied themselves. The question is which gruppetto riders are sandbagging versus which are at their actual limit.

**What the model does.** A **Cox Proportional Hazards survival model** with rider-level random effects. The "event" is abandonment and the "time" is how long into a stage a rider held on.

```
lambda_i(t) = lambda_0(t) * exp(beta^T * X_i + b_i)
```

- `lambda_0(t)` — baseline hazard (how fast riders drop out in general)
- `X_i` — covariates: GC position, seconds behind leader, gruppetto flag, time lost
- `beta` — shared coefficients
- `b_i ~ Normal(0, sigma^2)` — **frailty term**: rider-specific random effect capturing everything the covariates miss

A large positive `b_i` means the rider survived longer than their observable characteristics predict. That unexplained resilience is the signal. Riders ranked by frailty after mountain stages are the transition-stage bets.

---

#### Strategy 3: Medical Communiqué (Two-Compartment PK Model)

**The edge.** Crash and illness news arrives in medical communiqués with a lag, and the market prices it crudely — either ignoring it or overreacting. A pharmacokinetic model of trauma recovery gives a precise time-varying performance penalty, which the market doesn't have.

**What the model does.** A **two-compartment pharmacokinetic model** treats physical trauma like a drug concentration decaying over time:

```
dC_trauma/dt = -k_el * C_trauma
Perf(t) = Perf_baseline * (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

`k_el` is the elimination rate — how fast a rider recovers — estimated from historical return-to-form data after documented crashes. `EC_50` is the concentration at which performance is halved. The model outputs a predicted performance penalty curve by day post-incident. When the market hasn't fully adjusted odds to match the implied penalty, there is edge.

---

#### Strategy 4: Youth Fade (Functional PCA on Aging Curves)

**The edge.** Age-related performance decline isn't linear and isn't the same for every rider type. Sprinters fade differently from climbers. GC riders peak later. The market prices age as a blunt heuristic. The model prices it as a personalised trajectory.

**What the model does.** **Functional Principal Component Analysis** applied to career performance trajectories:

```
X_i(t) = mu(t) + sum_k(xi_ik * phi_k(t)) + epsilon_i(t)
```

`mu(t)` is the population-average aging curve. `phi_k(t)` are the principal modes of variation — different shapes of career arc. `xi_ik` are rider-specific loadings that determine which mode best describes their trajectory. Riders whose current performance is above their predicted trajectory by this model are underpriced; riders below are overpriced.

---

#### Strategy 5: Rest Day Regression (Interrupted Time Series / BSTS)

**The edge.** Rest days reset the physical state in ways the market treats as noise. Riders who were declining before a rest day often recover; riders who were peaking sometimes fade. An interrupted time series model separates the systematic rest-day effect from underlying form.

**What the model does.** An **Interrupted Time Series** model with ARMA errors:

```
Y_t = beta_0 + beta_1*Time_t + beta_2*Intervention_t + beta_3*TimeAfter_t
      + sum_j(phi_j * Y_{t-j}) + epsilon_t
```

`Intervention_t` marks the rest day. `beta_2` captures the immediate level shift; `beta_3` captures the slope change. The **Bayesian Structural Time Series (BSTS)** alternative adds a local trend component and a seasonality structure, which fits better when races span multiple rest days.

---

### Environmental / physical (Strategies 6–7)

---

#### Strategy 6: ITT Weather Arbitrage (Gaussian Process / SPDE) — IMPLEMENTED

**The edge.** ITT markets are efficient on form. They are often wrong on weather. A long ITT start window spans 3–4 hours. Riders who start into a headwind on the key exposed sections versus a tailwind can differ by 30–90 seconds — a margin that swamps typical GC separations. When weather data arrives after markets open, there is a window.

**What the model does.** A **Gaussian Process** over the wind field along the course:

```
w(s, t) ~ GP(mu(s,t), K((s,t), (s',t')))
```

The kernel captures spatial correlation (nearby sections have correlated wind) and temporal correlation (conditions 30 minutes apart are more similar than conditions 3 hours apart). For each rider, the model integrates expected headwind/tailwind exposure along their trajectory given their start time. This produces an adjusted time estimate that feeds into win probabilities.

The SPDE formulation approximates the full GP using sparse matrices — computationally tractable for large spatial domains.

---

#### Strategy 7: Weather Mismatch H2H (Langevin SDE)

**The edge.** Head-to-head markets on cobble sectors and crosswind stages misprice handling ability. Some riders are structurally better in gusts — wider base, lower CdA, different bike setup. The market doesn't separate weather sensitivity from baseline speed.

**What the model does.** A **Langevin stochastic differential equation** for bike velocity in stochastic wind:

```
m * dv/dt = F_drive - F_drag - F_gravity + sigma_wind * xi(t)
```

`xi(t)` is white noise modelling wind gusts. The model integrates this SDE to get a distribution over finishing times for each rider under different wind scenarios. Riders with lower variance in the output — those whose times are less sensitive to wind realisation — are preferred in crosswind conditions regardless of raw speed.

---

### Game theory (Strategies 8–9, 11)

---

#### Strategy 8: Desperation Breakaway (POSG / Quantal Response Equilibrium)

**The edge.** Late in a stage race, GC-irrelevant riders have strong incentive to go in breakaways. The market prices them as it would any other stage — on form. The game-theoretic model prices them on their strategic incentive, which is highest precisely when form signals are worst.

**What the model does.** A **Partially Observable Stochastic Game** with states `S = (GC_positions, Stage_wins, Remaining_stages)`. Each rider's action is probabilistic, modelled via **Quantal Response Equilibrium** — a noisy best-response:

```
P(a_i | s) = exp(lambda * Q_i(s, a_i)) / sum_a(exp(lambda * Q_i(s, a)))
```

`Q_i` is the value of each action given the current state. `lambda` is the rationality parameter — how sharply riders best-respond. Riders with strong strategic incentives (GC lost, no stage win, few stages remaining) are assigned higher breakaway probability regardless of form. This is the structural mispricing.

---

#### Strategy 9: Super-Domestique Choke (Mixed Membership / Dirichlet Process)

**The edge.** Some riders perform below their individual ability when leading domestique duties — sacrifice is priced in, but the psychological loading isn't. Other riders elevate. The market treats domestiques as a category; the model treats them as a distribution.

**What the model does.** A **Mixed Membership Model** (LDA-style) with a **Dirichlet Process** prior:

```
theta_i ~ Dirichlet(alpha)
w_{i,t} ~ sum_k(theta_{ik} * Normal(mu_k, Sigma_k))
```

Each rider is a mixture of latent types — leader, domestique, opportunist. The mixing weights `theta_i` are learned from career performance patterns. Riders with high weight on the domestique component are expected to underperform their physical ceiling when in protection duties. When the market doesn't discount for this, there is edge fading them.

---

#### Strategy 11: Domestique Chokehold (Hamilton-Jacobi-Bellman Differential Game)

**The edge.** When a team's leader is protected by domestiques, the optimal strategy for rivals is to attack before those domestiques are dropped — forcing the leader to work earlier than planned. The market prices attacks on pace; the model prices attacks on the strategic timing of when protection expires.

**What the model does.** A **differential game** between attacker and chaser, with the value function governed by the Hamilton-Jacobi-Bellman equation:

```
dV/dt + min_{u_chase} max_{u_break} [nabla_V * f(x, u_break, u_chase) + g(x)] = 0
```

The state `x` is the gap plus remaining domestique count. The model solves for the Nash equilibrium power allocation — how much effort the breakaway should commit, and when the chase will be abandoned. Riders attacking precisely when domestique protection expires have higher breakaway survival probability than the market implies.

---

### Real-time / live (Strategies 10, 12–13)

These run within the race window. Latency requirements: under 100ms per update.

---

#### Strategy 10: Mechanical Incident (Marked Hawkes Process)

**The edge.** Mechanical incidents (punctures, chain drops, crashes) cluster in time and space — cobble sectors, descents, bunch sprints. Markets reprice slowly after a mechanical. If the model can estimate recovery probability and time cost before the market fully adjusts, there is a window.

**What the model does.** A **Marked Hawkes Process** — a self-exciting point process where each incident increases the short-term probability of further incidents:

```
lambda_t = mu + sum_{t_i < t} phi(t - t_i, m_i)
```

`phi` is the excitation kernel; `m_i` is the mark (type and severity of incident). The process captures clustering. When a rider has a mechanical, the model estimates time-to-rejoin, probability of successful chase, and residual probability of abandonment. This feeds into updated win and podium probabilities. **Dirichlet Process** updates allow the model to learn new incident patterns in real time.

---

#### Strategy 12: Attack Confirmation (BOCPD) — IMPLEMENTED

**The edge.** Live markets on breakaway survival and stage winner move on information arriving in real time. If the model can confirm an attack is structural — not a test, not a reaction — before the TV feed processes it, there is a timing edge.

**What the model does.** **Bayesian Online Changepoint Detection** (Adams & MacKay 2007). The model maintains a posterior over "run length" — time since the last structural break in the gap time series:

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) * P(r_t | r_{t-1})
```

At each new observation the posterior updates. When the probability of a changepoint exceeds a threshold, the attack is classified as confirmed. The update runs in under 100ms.

---

#### Strategy 13: Gap Closing Calculus (Ornstein-Uhlenbeck + Extended Kalman Filter)

**The edge.** Live markets on catch probability are priced on the current gap and eyeball assessment. The model prices on the gap dynamics — whether the gap is mean-reverting (chase will catch) or diverging (breakaway survives). These are structurally different situations that produce the same observed gap in the short run.

**What the model does.** An **Ornstein-Uhlenbeck process** for the gap:

```
dG_t = theta * (mu - G_t) * dt + sigma * dW_t
```

`theta` controls mean-reversion speed; `mu` is the equilibrium gap. When `theta` is large and `G_t > mu`, the gap is closing. The **first passage time** — probability the gap hits zero by the finish — gives the catch probability directly.

An **Extended Kalman Filter** estimates `theta`, `mu`, and `sigma` in real time from live timing splits. As new splits arrive, the parameter estimates update and catch probability reprices accordingly.

---

### Risk modelling (Strategies 14–15)

---

#### Strategy 14: Post-Crash Confidence (Joint Frailty Model)

**The edge.** After a significant crash, the market prices the physical damage. It does not price the confidence loss — the tendency to brake earlier on descents, hold wider lines, choose less aggressive lines through technical sections. This is a separate, persistent performance penalty the market systematically underestimates.

**What the model does.** A **joint frailty model** with shared random effects across multiple risk types:

```
lambda_{ij}(t) = lambda_{0j}(t) * exp(beta^T * X_{ij} + b_i + epsilon_{ij})
```

`b_i` is a rider-level shared frailty across all risk types (descent, corner, wet). `epsilon_{ij}` is risk-type-specific. A rider with high crash frailty on descents but normal frailty elsewhere is specifically exposed to technical finishes — not to flat stages. **Bayesian Networks** encode conditional dependencies between risk types, allowing targeted position-type bets rather than blanket fade.

---

#### Strategy 15: Rain on Cobbles (Clayton Copula + Dynamic Programming)

**The edge.** Wet cobble sectors produce correlated failures across a field — when one rider punctures, others are more likely to. Markets price rider puncture probability independently. The model prices it jointly, which changes the probability that any given rider survives with the front group intact.

**What the model does.** **Clayton copulas** for sector-to-sector performance correlation:

```
C_theta(u, v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
```

`theta` controls tail dependence — how much sector failures cluster. In wet conditions `theta` rises, meaning sector outcomes become more correlated. The joint survival probability for a rider completing all cobble sectors is materially lower than the product of individual sector probabilities.

**Dynamic programming** solves the optimal pacing strategy given this survival risk:

```
V_s(v) = min_{v'} [lambda_s(v') * DNF_cost + L_s/v' + V_{s+1}(v')]
```

Riders whose optimal pacing strategy deviates from what the market assumes (flat-out) have mispriced outright and H2H markets.

---

## 3. Portfolio Construction

Each model produces a probability estimate `p_model` per rider for a given market. The Kelly fraction:

```
f* = (b * p - (1 - p)) / (b - 1)
```

where `b` is the decimal odds. This maximises expected log-wealth (i.e. the growth rate of the bankroll). Full Kelly is theoretically optimal but brutally sensitive to model miscalibration — a 10% overestimate of probability translates directly into overbetting and can cause ruin.

**This system defaults to quarter-Kelly** (`f = f* / 4`). This sacrifices roughly half the theoretical growth rate in exchange for much lower variance. For a model you have never bet live before, this is the appropriate starting point.

**Robust Kelly** is available when the model provides a posterior standard deviation `sigma_p` on the probability estimate:

```
f_robust = f_kelly * (1 - gamma * sigma_p^2 * b^2 / p^2)
```

Higher uncertainty → smaller stake. Early in the season when posteriors are wide, this automatically derisks the book.

**Portfolio-level constraints** are solved via convex optimisation (CVXPY):
- No single position > 25% of bankroll
- Portfolio variance bounded above
- CVaR at 95% bounded above

CVaR (Conditional Value at Risk) is the expected loss in the worst 5% of scenarios — a tail risk measure that matters more than variance for betting books where ruin is permanent.

---

## 4. Backtesting

Walk-forward validation is the only honest way to test a betting model. For each race in chronological order:

```
training_data = all stage results from races that finished BEFORE this race
for each stage S in the race:
    training_data += stages 1 to S-1 of this race
    fit model on training_data
    predict stage S
    record actual outcome and P&L
```

No future data touches the training window at any point. This is what would have happened running the system live.

The system tests three strategies:

| Strategy | What it tests |
|----------|--------------|
| `frailty` | Do high-frailty gruppetto riders podium more on transition stages? |
| `tactical` | Do riders flagged as PRESERVING outperform on the following flat stage? |
| `baseline` | Random selection — your null hypothesis |

Beating the baseline is the minimum bar. If frailty or tactical cannot beat random selection, they are generating no information.

---

## 5. Project Structure

```
cycling_predict/
|
|-- pipeline/                   # Data collection
|   |-- runner.py               # Entry point: run this to scrape PCS
|   |-- fetcher.py              # HTTP requests, rate limiting
|   |-- pcs_parser.py           # HTML parsing for ProCyclingStats
|   |-- db.py                   # Schema definitions, all DB writes
|   |-- betclic_scraper.py      # Betclic odds scraper (hub + event pages)
|   `-- queue.py                # Persistent job queue (resume-safe)
|
|-- genqirue/                   # Betting engine
|   |-- models/
|   |   |-- base.py             # Abstract base classes
|   |   |-- gruppetto_frailty.py   # Strategy 2: Cox PH + frailty
|   |   |-- tactical_hmm.py        # Strategy 1: HMM
|   |   |-- weather_spde.py        # Strategy 6: GP/SPDE for ITTs
|   |   `-- online_changepoint.py  # Strategy 12: BOCPD
|   |-- portfolio/
|   |   `-- kelly.py            # Robust Kelly + CVaR optimiser
|   |-- domain/
|   |   |-- entities.py         # RiderState, MarketState, Position, Portfolio
|   |   `-- enums.py            # StageType, TacticalState, MarketType, etc.
|   `-- data/
|       `-- schema_extensions.sql  # Betting tables on top of the scraping schema
|
|-- backtesting/
|   |-- engine.py               # Walk-forward backtester
|   `-- __init__.py
|
|-- config/
|   `-- races.yaml              # Which races and years to scrape
|
|-- tests/                      # Automated tests
|-- data/cycling.db             # Created by scraper, not in git
|-- fetch_odds.py               # CLI for Betclic odds scraping
|-- run_backtest.py             # Backtest CLI
|-- example_betting_workflow.py # End-to-end worked example
`-- monitor.py                  # Watch scraping progress
```

---

## 6. Setup

**Requirements:** Python 3.11 or 3.13. Check with `python --version`.

The scraping library (`procyclingstats`) must live in the folder adjacent to this project:

```
parent_folder/
  cycling_predict/    ← this repo
  procyclingstats/    ← scraping library, cloned separately
```

Install:

```
pip install -e ../procyclingstats
pip install -r requirements.txt
```

Verify:

```
python quickstart.py
```

---

## 7. Running It

### Scrape data

Configure `config/races.yaml` — add any race available on ProCyclingStats:

```yaml
year: 2026

races:
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]

  - name: Tour de France
    pcs_slug: tour-de-france
    type: stage_race
    history_years: [2021, 2022, 2023, 2024, 2025]
```

`pcs_slug` is the URL identifier from `procyclingstats.com/race/{slug}`.

Run:

```
python -m pipeline.runner
```

Safe to stop (Ctrl+C) and resume at any time — the queue persists. Takes ~20–60 minutes per race (rate-limited to ~1 req/sec). Monitor in a second terminal:

```
python monitor.py
```

### Apply betting schema (first time only)

```
python fetch_odds.py --init-schema
```

### Run the example workflow

```
python example_betting_workflow.py
```

Fits the frailty and tactical models on your scraped data and prints a portfolio report — which riders have signal, how much edge against what odds, recommended stakes.

### Run the backtest

```
python run_backtest.py
```

Options:

```
--strategy [frailty|tactical|baseline|all]   Default: all
--kelly 0.1                                  Kelly fraction (default: 0.25)
--top-k 3                                    Riders to bet per stage (default: 3)
--no-top3                                    Win market instead of podium
--bankroll 5000                              Starting bankroll (default: 1000)
--save-bets bets.csv                         Export every bet to CSV
```

---

## 8. Live Odds (Betclic)

The models produce probabilities. To compute edge and Kelly fractions you need market odds. `fetch_odds.py` scrapes Betclic's cycling hub and stores every selection into `bookmaker_odds`. The example workflow and Kelly optimizer use those prices directly; if no match is found for a rider they fall back to simulated odds.

### First-time setup

Apply the odds schema (safe to run after `--init-schema` from setup — idempotent):

```
python fetch_odds.py --init-schema
```

### Test against a known event before writing anything

```
python fetch_odds.py --dry-run --event-url https://www.betclic.fr/cyclisme-scycling/paris-nice-c5649/paris-nice-2026-m1052180106760192
```

Prints rider names, raw odds, hold-adjusted fair odds, and the implied overround — no DB writes.

### Scrape all live events

```
python fetch_odds.py
```

Discovers every cycling event on the hub, extracts odds, and inserts rows into `bookmaker_odds`. Each run gets a UUID `scrape_run_id`; the `bookmaker_odds_latest` view always reflects the most recent snapshot per selection.

### Query what's stored

```
python -c "
import sqlite3; conn = sqlite3.connect('data/cycling.db')
print(conn.execute('SELECT market_type, COUNT(*) FROM bookmaker_odds GROUP BY market_type').fetchall())
"
```

### Run on a schedule

Odds move. Run every 30 minutes while markets are open. On Linux/macOS:

```
*/30 6-22 * * * cd /path/to/cycling_predict && python fetch_odds.py >> logs/odds.log 2>&1
```

See [ODDS_README.md](ODDS_README.md) for the full walkthrough: market type mappings, H2H row splitting, name matching logic, troubleshooting, and how to extend the classifier for unrecognised French labels.

---

## 9. Interpreting Results

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

**Top3%** — Podium rate across all bets. In a field of ~150 riders, the naive rate is 2% (3/150). Frailty at 5.6% is roughly 3× the null. Tactical at 3.7% is roughly 2×. The question is whether this holds over a larger sample.

**ROI** — Profit over total staked, against a simulated fair market (no bookmaker margin). Real bookmakers take 5–15% margin, so translate accordingly. 136% simulated ROI in a fair market might be ~80–90% against real best-available odds — still meaningful if it holds, but the sample here is tiny.

**MaxDD** — Maximum peak-to-trough bankroll decline. 15% for frailty is manageable. Anything above 40–50% becomes a serious psychological and bankroll management issue even with positive long-run expectation.

**Spearman** — Rank correlation between model scores and actual finishing positions. 0.077 for frailty is positive but weak. More important is whether it's statistically significant — with 72 bets, the threshold for p < 0.05 is approximately rho > 0.23. Export to CSV and run the test.

**The sample size problem.** With 72 bets, the standard error on the 5.6% top-3 rate is:

```
SE = sqrt(0.056 * 0.944 / 72) ≈ 2.7%
```

The gap over the 2% naive baseline is 3.6 percentage points, or 1.3 standard errors. Suggestive, not conclusive. You need roughly 200+ bets before the numbers stabilise. Scrape 3–4 years of Paris-Nice, Criterium du Dauphine, Tour de Suisse, and the three grand tours and you will have enough data to draw real conclusions.

---

## 10. Quick Command Reference

```
# Install
pip install -e ../procyclingstats
pip install -r requirements.txt

# Verify
python quickstart.py

# Scrape PCS data
python -m pipeline.runner

# Monitor scraping
python monitor.py

# Apply all schemas (betting + odds tables)
python fetch_odds.py --init-schema

# --- Odds scraping ---

# Dry-run a single known event (no DB writes)
python fetch_odds.py --dry-run --event-url <betclic-event-url>

# Dry-run full hub (shows all live selections, no DB writes)
python fetch_odds.py --dry-run

# Full hub scrape → writes to bookmaker_odds
python fetch_odds.py

# Check what's stored
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); print(conn.execute('SELECT market_type, COUNT(*) FROM bookmaker_odds GROUP BY market_type').fetchall())"

# --- Betting workflow ---

# Full example workflow (uses real odds when available, simulated otherwise)
python example_betting_workflow.py

# Backtest
python run_backtest.py
python run_backtest.py --save-bets bets.csv

# Reset stuck scraping jobs
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"

# Run tests
pytest tests/ -v
```

---

## 11. Troubleshooting

**"ModuleNotFoundError: No module named 'procyclingstats'"**
```
pip install -e ../procyclingstats
```

**"ValueError: HTML from given URL is invalid"**
PCS is rate-limiting. Wait 5–10 minutes and restart. The queue resumes safely.

**"sqlite3.OperationalError: no such table"**
If the missing table is `bookmaker_odds` or `bookmaker_odds_latest`, run `python fetch_odds.py --init-schema`. For other tables, no PCS data has been scraped yet — run `python -m pipeline.runner` first.

**Jobs stuck "in_progress"**
The scraper crashed mid-job. Reset:
```
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"
```

**Frailty backtest shows 0 bets**
The frailty model generates signals from mountain stage data. If the first race in the database has no prior history, or has no mountain stages, it produces no output. Scrape multiple races across multiple years so there is always mountain stage training data available before the test race.

**PyMC/pytensor warning about g++**
Cosmetic. Models still run. To fix: `conda install gxx`. To suppress: set env variable `PYTENSOR_FLAGS=cxx=`.
