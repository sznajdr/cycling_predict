# Cycling Predict

A Bayesian betting engine for professional cycling. Scrapes ProCyclingStats, fits probabilistic models that surface market mispricings, sizes stakes via robust Kelly under CVaR constraints, and validates everything through strict walk-forward backtesting.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
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

## 1. Pipeline Overview

Three stages run sequentially. Each stage's output is the next stage's input.

**Stage 1 — Scraping.** `pipeline/runner.py` hits ProCyclingStats at ~1 req/s via a persistent job queue (`fetch_queue` table). Every job is idempotent and resume-safe. Output: rider profiles, stage results, time gaps, startlists, and stage metadata (type, elevation, distance) in `data/cycling.db`.

**Stage 2 — Modelling.** `genqirue/` reads from `data/cycling.db` and fits probabilistic models on the historical record. Each model answers a specific question: given what happened in stages 1..N-1, what is the probability distribution over riders for stage N? The models do not ask which rider is fastest — they ask which rider is mispriced relative to the market, given information the market has not encoded.

**Stage 3 — Validation.** `backtesting/engine.py` replays history in strict chronological order — train on t<T, predict T, observe outcome, advance T. The backtest produces a bet-by-bet P&L ledger. There is no lookahead at any point.

**Stage 4 — Live pricing.** `fetch_odds.py` scrapes Betclic's HTML for current cycling markets. Scraped odds feed into Stage 2's edge calculation: `edge = model_prob - (1 / market_odds)`. Without live odds, the workflow uses simulated fair-market prices.

---

## 2. The Models

Fifteen strategies across five categories. Four are implemented; the rest are mathematically specified and ready for implementation. Full mathematical specification (equations, priors, acceptance criteria, dependency chain): [`docs/MODELS.md`](docs/MODELS.md). Implementation guide with data inputs, outputs, code examples, and per-strategy acceptance criteria for all 15: [`docs/ENGINE.md`](docs/ENGINE.md).

---

### Pre-race form signals (Strategies 1–5)

Overnight batch. Train on historical data, produce ranked rider lists before markets open.

---

#### Strategy 1: Tactical HMM — IMPLEMENTED

**Mispricing.** Every time gap is a mixture of fitness and tactics. The market cannot separate a GC contender who cracked from one who soft-pedalled on a climbing stage — both show 2 minutes down. A rider in the PRESERVING state on a mountain stage is underpriced on the following flat.

**Model.** Two-state Hidden Markov Model with emission probabilities conditioned on observed gap and stage type:

```
P(z_{i,t} = PRESERVING) = sigmoid(δ_0 + δ_1 · ΔGC_{i,t} + δ_2 · StageType_t)
```

`z_{i,t} ∈ {CONTESTING, PRESERVING}` is latent. Viterbi decoding recovers the most probable state sequence; riders with `P(PRESERVING) > 0.7` on mountain stages are flagged for the following stage.

---

#### Strategy 2: Gruppetto Frailty — IMPLEMENTED

**Mispricing.** Gruppetto finishers on mountain stages are managing effort. The market prices them as poor candidates for the following flat. The question is which gruppetto riders are sandbagging versus which are genuinely at their limit.

**Model.** Cox Proportional Hazards with rider-level frailty:

```
λ_i(t) = λ_0(t) · exp(β^T · X_i + b_i),   b_i ~ Normal(0, σ²)
```

The frailty term `b_i` is the rider-specific random effect absorbing everything the observable covariates miss. Large positive `b_i` — the rider survived substantially longer than form, weight, and recent performance predict. That excess is the signal.

---

#### Strategy 3: Medical Communiqué (Two-Compartment PK Model)

**Mispricing.** Crash and illness information arrives with a lag and is priced crudely as a binary flag. The pharmacokinetic model gives a continuous, time-decaying performance penalty:

```
dC_trauma/dt = -k_el · C_trauma
Perf(t) = Perf_baseline · (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

Market treats the rider as either "out" or "fine". The model prices the E-max curve between those poles.

---

#### Strategy 4: Youth Fade (Functional PCA on Aging Curves)

**Mispricing.** Age-related decline is nonlinear, heterogeneous across rider types, and priced by the market as a blunt heuristic. Functional PCA on career-length performance trajectories estimates a personalised decline function `X_i(t) = μ(t) + Σ_k ξ_{ik} · φ_k(t) + ε_i(t)` — the market uses the population mean `μ(t)`; the model uses the individual trajectory.

---

#### Strategy 5: Rest Day Regression (Interrupted Time Series / BSTS)

**Mispricing.** Rest days produce a systematic physical reset the market treats as noise. A Bayesian Structural Time Series model decomposes form into trend, seasonality, and a rest-day intervention component — isolating the causal effect of the rest day from underlying trajectory.

---

### Environmental / physical (Strategies 6–7)

---

#### Strategy 6: ITT Weather Arbitrage (Gaussian Process / SPDE) — IMPLEMENTED

**Mispricing.** An ITT start window spans 3–4 hours. Riders who start into a headwind on key exposed sections vs. a tailwind can differ by 30–90 seconds — swamping typical GC separations. The market prices on expected conditions; the model prices on the full wind-field distribution.

**Model.** Gaussian Process over the spatiotemporal wind field along the route:

```
w(s, t) ~ GP(μ(s,t), K((s,t), (s',t')))
ΔT = ∫_0^D [P/F_aero(v_wind(t_early)) - P/F_aero(v_wind(t_late))] dx
```

The SPDE formulation (via R-INLA or PyMC-BART sparse matrices) makes the full-field GP computationally tractable. Output is an expected time-gap delta between early and late starters at each aero-critical segment.

---

#### Strategy 7: Weather Mismatch H2H (Langevin SDE)

**Mispricing.** H2H markets on cobble sectors price riders on raw speed. The Langevin SDE assigns each rider a distribution over finishing times under different wind realisations. Riders with lower output variance (smoother power application, better handling) are underpriced in crosswind H2H regardless of FTP.

---

### Game theory (Strategies 8–9, 11)

---

#### Strategy 8: Desperation Breakaway (POSG / Quantal Response Equilibrium)

**Mispricing.** Riders who are GC-irrelevant have high strategic incentive to attack irrespective of form — the expected value of a solo win dominates the expected value of a peloton sprint. Market prices them on form; the model prices them on incentive structure via Quantal Response Equilibrium:

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_a exp(λ · Q_i(s, a))
```

---

#### Strategy 9: Super-Domestique Choke (Mixed Membership / Dirichlet Process)

**Mispricing.** Some riders systematically underperform their individual ability on days when they are leading domestique duties. The market treats domestique as a role that cancels form; the Dirichlet Process model identifies latent rider types — some respond to leadership duties with elevated performance, some suppress it.

---

#### Strategy 11: Domestique Chokehold (Hamilton-Jacobi-Bellman Differential Game)

**Mispricing.** The optimal moment to attack a protected leader is before the domestiques are dropped, not after — because after, the protected rider can accelerate freely. The HJB differential game gives the Nash equilibrium power allocation and the threshold crossing at which the chase is rationally abandoned. Markets price on the leader's advantage; the model prices on the structural gap between equilibrium power and sustainable power.

---

### Real-time / live (Strategies 10, 12–13)

Target latency: under 100ms per update.

---

#### Strategy 10: Mechanical Incident (Marked Hawkes Process)

**Mispricing.** Mechanical incidents cluster in time and space. The Hawkes process captures self-exciting clustering across riders: `λ_t = μ + Σ φ(t - t_i, m_i)`. When an incident occurs, updated win probabilities incorporate time-to-rejoin distribution and abandonment probability before the live market reprices.

---

#### Strategy 12: Attack Confirmation (BOCPD) — IMPLEMENTED

**Mispricing.** Live markets on breakaway survival move on TV coverage, which lags the GPS/power data by 15–30 seconds. Bayesian Online Changepoint Detection on live power metrics confirms whether a pace change is structural or noise before the feed processes it:

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) · P(r_t | r_{t-1})
```

Implemented with Numba JIT: updates in <50ms. Bet signal: `P(changepoint) > 0.8` AND prior-stage Z-score > 2.0.

---

#### Strategy 13: Gap Closing Calculus (Ornstein-Uhlenbeck + Extended Kalman Filter)

**Mispricing.** Catch probability is priced on current gap and eyeball peloton speed. The OU process on gap dynamics distinguishes mean-reverting (chase closes, breakaway caught) from diverging (gap expands, breakaway survives):

```
dG_t = θ(μ - G_t)dt + σ dW_t
```

The EKF estimates `(θ, μ, σ)` in real time from live timing splits. When `θ̂` is strongly positive and the gap is above `μ`, the breakaway is structural — the market is underpricing survival.

---

### Risk modelling (Strategies 14–15)

---

#### Strategy 14: Post-Crash Confidence (Joint Frailty Model)

**Mispricing.** The market prices physical crash damage (time out, injuries). It does not price the persistent confidence penalty — earlier braking into corners, wider lines, less aggressive positioning on descents. A joint frailty model with shared random effects across risk types (descent speed, corner commitment, wet-road exposure) isolates the confidence component from the physical recovery curve.

---

#### Strategy 15: Rain on Cobbles (Clayton Copula + Dynamic Programming)

**Mispricing.** Wet cobble sectors produce correlated failures — a rider who punctures or crashes on sector 1 is more likely to abandon than expected given only sector-1 difficulty, because the stress compounds. Clayton copulas model the joint tail dependence across sectors:

```
C_θ(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
```

Dynamic programming on the resulting joint survival surface gives optimal pacing per sector. Riders whose implied strategy (from their power output profile) deviates from the DP-optimal strategy are flagged as misprice candidates.

---

## 3. Portfolio Construction

Kelly fraction: `f* = (b·p - (1-p)) / b`. The system defaults to **quarter-Kelly** — full Kelly is theoretically optimal but catastrophically sensitive to model miscalibration. Quarter-Kelly trades ~30% of long-run growth for a roughly 4× reduction in drawdown variance.

**Robust Kelly** shrinks `f*` when the model supplies a posterior standard deviation `σ_p` on its probability estimate:

```
f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
```

Higher posterior uncertainty → tighter sizing. The penalty scales with leverage `b` — a high-odds bet under an uncertain model is penalised more than the same uncertainty at short odds.

**Portfolio-level constraints** (CVXPY, `genqirue/portfolio/kelly.py`):
- Single-position cap: 25% of bankroll
- Portfolio variance bounded above
- CVaR at 95% bounded above

---

## 4. Backtesting

Walk-forward validation — strict temporal ordering, no lookahead:

```
for each race R in chronological order:
    training_data = all results from races that completed before R
    for each stage S in R:
        training_data += stages 1..(S-1) of R
        fit model on training_data
        predict stage S
        record outcome → P&L ledger
```

Three strategies tested:

| Strategy | Signal |
|----------|--------|
| `frailty` | High-frailty gruppetto riders on transition stages |
| `tactical` | Riders in PRESERVING state on mountain → following flat |
| `baseline` | Random selection (null hypothesis) |

---

## 5. Project Structure

```
cycling_predict/
|
|-- pipeline/                   # Data collection
|   |-- runner.py               # Entry point: scrape PCS data
|   |-- fetcher.py              # HTTP requests, rate limiting
|   |-- pcs_parser.py           # HTML parsing for ProCyclingStats
|   |-- db.py                   # Schema definitions, all DB writes
|   |-- betclic_scraper.py      # Betclic odds scraper
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
|   |   `-- enums.py            # StageType, TacticalState, MarketType
|   `-- data/
|       `-- schema_extensions.sql  # Betting tables on top of the scraping schema
|
|-- backtesting/
|   |-- engine.py               # Walk-forward backtester
|   `-- __init__.py
|
|-- docs/
|   |-- MODELS.md               # Mathematical specification — all 15 strategies
|   |-- ENGINE.md               # Implementation guide — data flow, acceptance criteria
|   |-- SCRAPE.md               # Scraping pipeline — schema, job types, execution flow
|   |-- ODDS.md                 # Betclic scraper — walkthrough and troubleshooting
|   `-- DEPLOYMENT.md           # Production: Docker, cron, PostgreSQL, monitoring
|
|-- config/
|   `-- races.yaml              # Which races and years to scrape
|
|-- scripts/
|   |-- export_race_data.py     # Export / import race data for sharing
|   `-- setup_team.py           # Automated setup for new team members
|
|-- tests/
|   |-- betting/
|   |   `-- test_strategies.py  # Unit tests (no network)
|   |-- test_connection.py      # PCS connectivity check
|   |-- test_race.py            # Race meta + startlist roundtrip
|   `-- test_rider.py           # Rider scrape + DB roundtrip
|
|-- data/cycling.db             # Created by scraper, not in git
|-- fetch_odds.py               # Betclic odds CLI
|-- run_backtest.py             # Backtest CLI
|-- example_betting_workflow.py # End-to-end worked example
|-- quickstart.py               # Quick demo (no scraped data required)
|-- monitor.py                  # Watch scraping progress
|-- COMMANDS.md                 # Complete CLI and SQL reference
`-- CONTRIBUTING.md             # Development workflow and standards
```

---

## 6. Setup

**Requirements:** Python 3.11 or 3.13.

The scraping library (`procyclingstats`) must live adjacent to this project:

```
parent_folder/
  cycling_predict/    <- this repo
  procyclingstats/    <- cloned separately
```

Install:

```bash
pip install -e ../procyclingstats
pip install -r requirements.txt
python fetch_odds.py --init-schema   # apply all DB schemas
python quickstart.py                 # verify
```

Automated setup for a new team member:

```bash
python scripts/setup_team.py
```

---

## 7. Running It

### Scrape

Configure which races and years in `config/races.yaml`:

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

```bash
python -m pipeline.runner
python monitor.py          # watch progress in a second terminal
```

Rate-limited to ~1 req/s. Safe to stop (Ctrl+C) and resume — the queue persists. Each race takes 20–60 minutes depending on history depth. Full schema and job-type reference: [`docs/SCRAPE.md`](docs/SCRAPE.md).

### Fit models and generate signals

```bash
python example_betting_workflow.py
```

Fits the frailty and tactical models on your scraped data, queries live odds from `bookmaker_odds_latest`, and prints a portfolio report with edge estimates and Kelly stakes.

### Run the backtest

```bash
python run_backtest.py
python run_backtest.py --strategy frailty --kelly 0.1 --save-bets bets.csv
```

---

## 8. Live Odds (Betclic)

`fetch_odds.py` scrapes Betclic's cycling hub page, parses market blocks from raw HTML via regex (`"name":"([^"]+)",[^{]{0,200}?"odds":([\d.]+)`), computes hold-adjusted fair odds (`fair_prob = implied_prob / Σ implied_probs`), and inserts every selection into `bookmaker_odds`. Each run gets a UUID `scrape_run_id`; the `bookmaker_odds_latest` view returns the most recent snapshot per selection.

```bash
# Apply schema (first time)
python fetch_odds.py --init-schema

# Test a single event without writing to DB
python fetch_odds.py --dry-run --event-url <betclic-event-url>

# Full hub scrape → DB
python fetch_odds.py
```

Schedule for live coverage:

```bash
# Every 30 minutes, 06:00–22:00 (Linux/macOS cron)
*/30 6-22 * * * cd /path/to/cycling-predict && python fetch_odds.py >> logs/odds.log 2>&1
```

Name matching between Betclic selections and the `riders` table uses a two-pass join: exact Unicode match, then accent-stripped ASCII fallback. When no match is found, `example_betting_workflow.py` falls back to simulated fair-market odds.

Full walkthrough — market type mappings, H2H row handling, troubleshooting, classifier extension: [`docs/ODDS.md`](docs/ODDS.md).

---

## 9. Interpreting Results

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

**Top3% vs. null.** In a field of ~150, the naive top-3 rate is 2%. Frailty at 5.6% is 2.8× the null — consistent with a real signal, but the confidence interval is wide at n=72.

**ROI** is against a simulated fair market (zero margin). Real books take 5–15%; adjust accordingly. A strategy that looks marginally positive here may be negative against the actual overround.

**MaxDD** is peak-to-trough bankroll decline. 15% on frailty is consistent with quarter-Kelly sizing. MaxDD above 40–50% on a positive-expectation strategy usually means over-sizing or a model calibration problem.

**Spearman ρ = 0.077** on 72 bets is not significant (p < 0.05 threshold ≈ ρ > 0.23). The rank correlation signal needs more data to resolve from noise.

**Sample size.** With 72 bets, the standard error on the 5.6% top-3 rate is ~2.7%. The frailty edge over the 2% null is 1.3 standard errors — directionally interesting, not conclusive. Target 200+ bets before treating any metric as stable. To get there: scrape 3–4 years across Paris-Nice, Critérium du Dauphiné, Tour de Suisse, and the three grand tours.

```bash
python run_backtest.py --save-bets bets.csv
# then run Spearman test on the exported CSV
```

---

## 10. Quick Command Reference

```bash
# Setup
pip install -e ../procyclingstats && pip install -r requirements.txt
python fetch_odds.py --init-schema

# Scrape
python -m pipeline.runner
python monitor.py

# Odds
python fetch_odds.py --dry-run --event-url <url>
python fetch_odds.py --dry-run
python fetch_odds.py
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); print(conn.execute('SELECT market_type, COUNT(*) FROM bookmaker_odds GROUP BY market_type').fetchall())"

# Models and workflow
python quickstart.py
python example_betting_workflow.py

# Backtest
python run_backtest.py
python run_backtest.py --strategy frailty --kelly 0.1 --save-bets bets.csv

# Tests
pytest tests/betting/ -v     # unit tests, no network
pytest tests/ -v             # all tests

# Reset stuck scraping jobs
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit()"
```

Complete reference — SQL queries, scheduling, monitoring, git workflow: [`COMMANDS.md`](COMMANDS.md).

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'procyclingstats'`**
```bash
pip install -e ../procyclingstats
```

**`ValueError: HTML from given URL is invalid`**
PCS is rate-limiting. Wait 5–10 minutes; the queue resumes from where it stopped.

**`sqlite3.OperationalError: no such table`**
- If the missing table is `bookmaker_odds` or `bookmaker_odds_latest`: `python fetch_odds.py --init-schema`
- Otherwise: no PCS data scraped yet — run `python -m pipeline.runner` first

**Jobs stuck `in_progress`**
Scraper crashed mid-job:
```bash
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit()"
```

**Frailty backtest shows 0 bets**
The frailty model needs mountain-stage data to generate signals. If the database contains only flat races, or the first race has no prior history, output is empty. Scrape multiple races across multiple years.

**PyMC/pytensor warning about g++**
Cosmetic — models run. To suppress: set `PYTENSOR_FLAGS=cxx=`. To fix: `conda install gxx`.

**No odds matched in `_lookup_real_odds`**
```bash
python -c "
import sqlite3; conn = sqlite3.connect('data/cycling.db')
print(conn.execute(\"SELECT DISTINCT participant_name FROM bookmaker_odds WHERE participant_name LIKE '%name%'\").fetchall())
print(conn.execute(\"SELECT name FROM riders WHERE name LIKE '%name%'\").fetchall())
"
```
See [`docs/ODDS.md`](docs/ODDS.md) for name-matching details and how to extend the French market label classifier.
