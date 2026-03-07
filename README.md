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

The output is a ranked list of riders with model-implied probabilities, edge estimates against market odds, and Kelly-sized stakes.

---

## 2. The Models

Fifteen strategies across five categories. Four are implemented; the rest are mathematically specified and ready for implementation. Each targets a structural mispricing the market consistently fails to price.

Full mathematical specification (equations, acceptance criteria, dependency chain, dependencies): [`docs/MODELS.md`](docs/MODELS.md)

Implementation guide for all 15 strategies — implemented and not yet implemented — with usage examples, data inputs, outputs, integration points, and per-strategy acceptance criteria: [`docs/ENGINE.md`](docs/ENGINE.md)

---

### Pre-race form signals (Strategies 1–5)

Overnight batch. Run on historical data, produce ranked rider lists before markets open.

---

#### Strategy 1: Tactical HMM (Hidden Markov Model) — IMPLEMENTED

**The edge.** Every time gap on a stage is a mixture of fitness and tactics. The market cannot separate a GC rider who cracked from one who soft-pedalled — they both show 2 minutes down. If the model can identify who was managing effort, that rider is a bet for the next stage.

**What the model does.** A Hidden Markov Model with two latent states:

- **CONTESTING** — racing at capacity; time loss reflects true fitness
- **PRESERVING** — deliberately holding back; time loss is tactical

```
P(z_{i,t} = PRESERVING) = sigmoid(δ_0 + δ_1 · ΔGC_{i,t} + δ_2 · StageType_t)
```

Riders with high `P(PRESERVING)` on mountain stages are flagged for the following flat stage.

---

#### Strategy 2: Gruppetto Frailty (Cox Proportional Hazards) — IMPLEMENTED

**The edge.** Gruppetto riders on a mountain stage are managing effort, not struggling. The market prices them as poor candidates for the following flat stage. The question is which gruppetto riders are sandbagging versus which are at their actual limit.

**What the model does.** A Cox Proportional Hazards survival model with rider-level random effects:

```
λ_i(t) = λ_0(t) · exp(β^T · X_i + b_i)
```

`b_i ~ Normal(0, σ²)` is the **frailty term** — rider-specific random effect capturing everything the covariates miss. A large positive `b_i` means the rider survived longer than their observable characteristics predict. That unexplained resilience is the signal.

---

#### Strategy 3: Medical Communiqué (Two-Compartment PK Model)

**The edge.** Crash and illness news arrives with a lag and is priced crudely. A pharmacokinetic model gives a precise time-varying performance penalty:

```
dC_trauma/dt = -k_el · C_trauma
Perf(t) = Perf_baseline · (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

---

#### Strategy 4: Youth Fade (Functional PCA on Aging Curves)

**The edge.** Age-related decline is not linear and not the same across rider types. The market prices age as a blunt heuristic; the model prices it as a personalised trajectory via Functional PCA on career performance curves.

---

#### Strategy 5: Rest Day Regression (Interrupted Time Series / BSTS)

**The edge.** Rest days reset physical state in ways the market treats as noise. An interrupted time series model separates the systematic rest-day effect from underlying form.

---

### Environmental / physical (Strategies 6–7)

---

#### Strategy 6: ITT Weather Arbitrage (Gaussian Process / SPDE) — IMPLEMENTED

**The edge.** A long ITT start window spans 3–4 hours. Riders who start into a headwind on key exposed sections versus a tailwind can differ by 30–90 seconds — swamping typical GC separations.

**What the model does.** A Gaussian Process over the wind field along the course:

```
w(s, t) ~ GP(μ(s,t), K((s,t), (s',t')))
ΔT = ∫_0^D [P/F_aero(v_wind(t_early)) - P/F_aero(v_wind(t_late))] dx
```

The SPDE formulation approximates the GP using sparse matrices for computational tractability.

---

#### Strategy 7: Weather Mismatch H2H (Langevin SDE)

**The edge.** H2H markets on cobble sectors misprice handling ability. The Langevin SDE model gives each rider a distribution over finishing times under different wind realisations — riders with lower output variance are preferred in crosswind regardless of raw speed.

---

### Game theory (Strategies 8–9, 11)

---

#### Strategy 8: Desperation Breakaway (POSG / Quantal Response Equilibrium)

**The edge.** Late-race GC-irrelevant riders have high strategic incentive to attack regardless of form. The market prices them on form; the model prices them on incentive.

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_a exp(λ · Q_i(s, a))
```

---

#### Strategy 9: Super-Domestique Choke (Mixed Membership / Dirichlet Process)

**The edge.** Some riders perform below their individual ability when leading domestique duties. The market treats domestiques as a category; the Dirichlet Process model treats them as a distribution of latent types.

---

#### Strategy 11: Domestique Chokehold (Hamilton-Jacobi-Bellman Differential Game)

**The edge.** The optimal time to attack a protected leader is before the domestiques are dropped. The Hamilton-Jacobi-Bellman equation gives the Nash equilibrium power allocation and when the chase will be abandoned.

---

### Real-time / live (Strategies 10, 12–13)

Latency requirement: under 100ms per update.

---

#### Strategy 10: Mechanical Incident (Marked Hawkes Process)

**The edge.** Mechanical incidents cluster in time and space. The Hawkes process captures self-exciting clustering; updated win probabilities incorporate time-to-rejoin and abandonment probability before the market reprices.

---

#### Strategy 12: Attack Confirmation (BOCPD) — IMPLEMENTED

**The edge.** Live markets on breakaway survival move on information in real time. Bayesian Online Changepoint Detection confirms whether an attack is structural before the TV feed processes it.

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) · P(r_t | r_{t-1})
```

Updates in under 100ms (Numba-accelerated). Bet when `P(changepoint) > 0.8` AND yesterday's Z-score > 2.0.

---

#### Strategy 13: Gap Closing Calculus (Ornstein-Uhlenbeck + Extended Kalman Filter)

**The edge.** Catch probability is priced on the current gap and eyeball assessment. The OU process prices on gap dynamics — mean-reverting (chase will catch) versus diverging (breakaway survives). The EKF estimates parameters in real time from live timing splits.

---

### Risk modelling (Strategies 14–15)

---

#### Strategy 14: Post-Crash Confidence (Joint Frailty Model)

**The edge.** The market prices physical crash damage. It does not price confidence loss — earlier braking, wider lines, less aggressive positioning. A joint frailty model with shared random effects across risk types (descent, corner, wet) isolates the persistent confidence penalty.

---

#### Strategy 15: Rain on Cobbles (Clayton Copula + Dynamic Programming)

**The edge.** Wet cobble sectors produce correlated failures. Clayton copulas model joint survival probability across sectors:

```
C_θ(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}
```

Dynamic programming solves optimal pacing given this survival risk, identifying riders whose actual strategy deviates from the flat-out assumption the market uses.

---

## 3. Portfolio Construction

Kelly fraction maximises expected log-wealth:

```
f* = (b·p - (1-p)) / (b - 1)
```

**The system defaults to quarter-Kelly** (`f = f*/4`). Full Kelly is theoretically optimal but sensitive to model miscalibration.

**Robust Kelly** shrinks stakes when the model provides a posterior standard deviation `σ_p`:

```
f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
```

**Portfolio-level constraints** via CVXPY:
- No single position > 25% of bankroll
- Portfolio variance bounded above
- CVaR at 95% bounded above

CVaR (Conditional Value at Risk) is the expected loss in the worst 5% of scenarios — a tail risk measure that matters more than variance for betting books where ruin is permanent.

---

## 4. Backtesting

Walk-forward validation — no lookahead at any point:

```
training_data = all stage results from races that finished BEFORE this race
for each stage S in the race:
    training_data += stages 1 to S-1 of this race
    fit model on training_data
    predict stage S
    record actual outcome and P&L
```

Three strategies tested:

| Strategy | What it tests |
|----------|--------------|
| `frailty` | Do high-frailty gruppetto riders podium more on transition stages? |
| `tactical` | Do riders flagged as PRESERVING outperform on the following flat stage? |
| `baseline` | Random selection — the null hypothesis |

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
|   |   `-- enums.py            # StageType, TacticalState, MarketType, etc.
|   `-- data/
|       `-- schema_extensions.sql  # Betting tables on top of the scraping schema
|
|-- backtesting/
|   |-- engine.py               # Walk-forward backtester
|   `-- __init__.py
|
|-- docs/                       # All documentation
|   |-- MODELS.md               # Mathematical specification of all 15 strategies
|   |-- ENGINE.md               # Implementation guide — all strategies, data flow, acceptance criteria
|   |-- SCRAPE.md               # Scraping pipeline — schema, job types, execution flow
|   |-- ODDS.md                 # Betclic odds scraper — step-by-step walkthrough
|   `-- DEPLOYMENT.md           # Production deployment, Docker, cron, monitoring
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
|   |   `-- test_strategies.py  # Strategy unit tests (no network)
|   |-- test_connection.py      # PCS connectivity check
|   |-- test_race.py            # Race meta + startlist roundtrip
|   `-- test_rider.py           # Rider scrape + DB roundtrip
|
|-- data/cycling.db             # Created by scraper, not in git
|-- logs/                       # Runtime logs, not in git
|-- fetch_odds.py               # CLI for Betclic odds scraping
|-- run_backtest.py             # Backtest CLI
|-- example_betting_workflow.py # End-to-end worked example
|-- quickstart.py               # Quick demo (no config required)
|-- reset_stage_jobs.py         # One-time migration helper
|-- monitor.py                  # Watch scraping progress
|-- COMMANDS.md                 # Complete CLI and SQL command reference
`-- CONTRIBUTING.md             # Development workflow and standards
```

---

## 6. Setup

**Requirements:** Python 3.11 or 3.13.

The scraping library (`procyclingstats`) must live in the folder adjacent to this project:

```
parent_folder/
  cycling_predict/    <- this repo
  procyclingstats/    <- scraping library, cloned separately
```

Install:

```bash
pip install -e ../procyclingstats
pip install -r requirements.txt
```

Apply all database schemas:

```bash
python fetch_odds.py --init-schema
```

Verify:

```bash
python quickstart.py
```

Automated setup for a new team member:

```bash
python scripts/setup_team.py
```

---

## 7. Running It

### Scrape data

Configure `config/races.yaml`:

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

Run:

```bash
python -m pipeline.runner
```

Safe to stop (Ctrl+C) and resume at any time — the queue persists. Takes ~20–60 minutes per race (rate-limited to ~1 req/sec). Monitor in a second terminal:

```bash
python monitor.py
```

See [`docs/SCRAPE.md`](docs/SCRAPE.md) for the full schema reference, job type documentation, execution flow, and resume-safety details.

### Run the example workflow

```bash
python example_betting_workflow.py
```

Fits the frailty and tactical models on your scraped data and prints a portfolio report.

### Run the backtest

```bash
python run_backtest.py
python run_backtest.py --strategy frailty --kelly 0.1 --save-bets bets.csv
```

---

## 8. Live Odds (Betclic)

The models produce probabilities. To compute edge and Kelly fractions you need market odds. `fetch_odds.py` scrapes Betclic's cycling hub and stores every selection into `bookmaker_odds`. The workflow uses real prices when a rider matches; falls back to simulated odds otherwise.

### First-time schema setup

```bash
python fetch_odds.py --init-schema
```

### Test before writing anything

```bash
python fetch_odds.py --dry-run --event-url <betclic-event-url>
```

Prints rider names, raw odds, hold-adjusted fair odds, and the implied overround — no DB writes.

### Full hub scrape

```bash
python fetch_odds.py
```

Discovers every live cycling event, extracts odds, inserts rows into `bookmaker_odds`. Each run gets a UUID `scrape_run_id`; the `bookmaker_odds_latest` view always reflects the most recent snapshot per selection.

### Run on a schedule

```bash
# Every 30 minutes, 06:00–22:00 (Linux/macOS)
*/30 6-22 * * * cd /path/to/cycling-predict && python fetch_odds.py >> logs/odds.log 2>&1
```

See [`docs/ODDS.md`](docs/ODDS.md) for the full walkthrough: market type mappings, H2H row handling, name-matching logic, troubleshooting, and extending the classifier.

---

## 9. Interpreting Results

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

**Top3%** — Podium rate across all bets. In a field of ~150 riders, the naive rate is 2% (3/150). Frailty at 5.6% is roughly 3× the null.

**ROI** — Profit over total staked, against a simulated fair market (no bookmaker margin). Real bookmakers take 5–15% margin — translate accordingly.

**MaxDD** — Maximum peak-to-trough bankroll decline. 15% is manageable; above 40–50% becomes a serious issue even with positive long-run expectation.

**Spearman** — Rank correlation between model scores and actual finishing positions. With 72 bets, the threshold for p < 0.05 is approximately ρ > 0.23. Export to CSV and run the test:

```bash
python run_backtest.py --save-bets bets.csv
```

**The sample size problem.** With 72 bets, the standard error on the 5.6% top-3 rate is approximately 2.7%. The gap over the 2% naive baseline is 1.3 standard errors — suggestive, not conclusive. You need roughly 200+ bets before the numbers stabilise. Scrape 3–4 years of Paris-Nice, Criterium du Dauphine, Tour de Suisse, and the three grand tours to get there.

---

## 10. Quick Command Reference

```bash
# Setup
pip install -e ../procyclingstats && pip install -r requirements.txt
python fetch_odds.py --init-schema

# Scrape PCS data
python -m pipeline.runner
python monitor.py                    # watch progress

# Betclic odds
python fetch_odds.py --dry-run --event-url <url>   # test single event
python fetch_odds.py --dry-run                     # test full hub
python fetch_odds.py                               # full scrape → DB

# Verify odds stored
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); print(conn.execute('SELECT market_type, COUNT(*) FROM bookmaker_odds GROUP BY market_type').fetchall())"

# Betting workflow
python quickstart.py
python example_betting_workflow.py

# Backtesting
python run_backtest.py
python run_backtest.py --strategy frailty --kelly 0.1 --save-bets bets.csv

# Testing
pytest tests/betting/ -v             # unit tests (no network)
pytest tests/ -v                     # all tests

# Reset stuck scraping jobs
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit()"
```

See [`COMMANDS.md`](COMMANDS.md) for the complete reference including all SQL queries, scheduling, monitoring, and git workflow.

---

## 11. Troubleshooting

**"ModuleNotFoundError: No module named 'procyclingstats'"**
```bash
pip install -e ../procyclingstats
```

**"ValueError: HTML from given URL is invalid"**
PCS is rate-limiting. Wait 5–10 minutes and restart. The queue resumes safely.

**"sqlite3.OperationalError: no such table"**
If the missing table is `bookmaker_odds` or `bookmaker_odds_latest`: run `python fetch_odds.py --init-schema`. For other tables, no PCS data has been scraped yet — run `python -m pipeline.runner` first.

**Jobs stuck "in_progress"**
The scraper crashed mid-job. Reset:
```bash
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit()"
```

**Frailty backtest shows 0 bets**
The frailty model generates signals from mountain stage data. If the first race in the database has no prior history, or has no mountain stages, it produces no output. Scrape multiple races across multiple years.

**PyMC/pytensor warning about g++**
Cosmetic. Models still run. To fix: `conda install gxx`. To suppress: set env variable `PYTENSOR_FLAGS=cxx=`.

**No odds matching in `_lookup_real_odds`**
```bash
python -c "
import sqlite3; conn = sqlite3.connect('data/cycling.db')
print(conn.execute(\"SELECT DISTINCT participant_name FROM bookmaker_odds WHERE participant_name LIKE '%name%'\").fetchall())
print(conn.execute(\"SELECT name FROM riders WHERE name LIKE '%name%'\").fetchall())
"
```
See [`docs/ODDS.md`](docs/ODDS.md) for name-matching details and how to extend the classifier for unrecognised French market labels.
