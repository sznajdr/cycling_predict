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
8. [Interpreting Results](#8-interpreting-results)
9. [Quick Command Reference](#9-quick-command-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What the System Does

Three stages:

**Stage 1 — Data.** A scraper pulls historical results from ProCyclingStats: stage results, time gaps, startlists, rider profiles, stage metadata (type, elevation, distance). This goes into a local SQLite database.

**Stage 2 — Modelling.** Statistical models fit on that historical data and produce probability estimates for riders in upcoming stages. Each model targets a specific market inefficiency — situations where the observed data implies a different probability than what the market is pricing.

**Stage 3 — Validation.** A walk-forward backtester replays history in strict chronological order — training only on what was available before each race, predicting each race, observing outcomes — and produces a P&L record you can interrogate. No lookahead.

The output is a ranked list of riders with model-implied probabilities, edge estimates against market odds, and Kelly-sized stakes. Whether you act on it is up to you.

---

## 2. The Models

Four models are implemented. They target different structural edges in cycling markets.

---

### Strategy 2: Gruppetto Frailty (Cox Proportional Hazards)

**The edge.** On mountain stages, gruppetto riders are not struggling — they are managing effort. The market prices them as poor candidates for the following day's flat stage. They are often fresher than GC riders who emptied themselves on the climb. The question is which gruppetto riders are sandbagging versus which are genuinely at their limit.

**What the model does.** It fits a **Cox Proportional Hazards survival model** with rider-level random effects. Survival analysis was built to model time-to-event data (originally mortality in clinical trials). Here the "event" is abandonment (DNF or OTL) and the "time" is how long into a stage a rider held on.

The hazard function for rider *i* at time *t* is:

```
lambda_i(t) = lambda_0(t) * exp(beta^T * X_i + b_i)
```

- `lambda_0(t)` — baseline hazard rate (how fast riders drop out in general)
- `X_i` — observed covariates: GC position, seconds behind leader, whether they rode gruppetto, how much time they lost
- `beta` — shared coefficients across all riders
- `b_i ~ Normal(0, sigma^2)` — **frailty term**: a rider-specific random effect that captures everything the covariates don't

The frailty term is the signal. A rider with a large positive `b_i` survived longer than the model predicted from their observable characteristics. They absorbed more mountain stress than their GC position and time gaps suggest. That unexplained resilience — the residual — is the hidden form indicator.

Riders ranked by frailty after mountain stages are the candidates for transition stage bets. The model outputs a frailty score per rider; higher = more unexplained survival capacity.

---

### Strategy 1: Tactical HMM (Hidden Markov Model)

**The edge.** A rider's time loss on any stage is a mixture of fitness and tactics. The market cannot distinguish a GC rider who legitimately cracked from one who soft-pedalled to protect a leader or save legs. If the model can separate these cases, the soft-pedaller is a bet for the next stage; the rider who cracked is not.

**What the model does.** A **Hidden Markov Model** with two latent states:

- **CONTESTING** — racing at capacity, time loss reflects true fitness
- **PRESERVING** — deliberately not racing, time loss is tactical

The latent state `z_{i,t}` is unobservable. The observed variable is time loss. The model posits:

```
P(z_{i,t} = PRESERVING) = sigmoid(delta_0 + delta_1 * GC_gap + delta_2 * IsHardStage)
```

Riders far down on GC have less incentive to fight on hard stages — they are riding for other objectives. The time loss conditional on state is:

```
time_loss | CONTESTING  ~ Normal(mu, sigma^2)
time_loss | PRESERVING  ~ Normal(mu + gamma, sigma^2)
```

`gamma` is the tactical time loss — approximately 2 minutes — constrained to be positive. The model is fitted via MCMC (Markov Chain Monte Carlo sampling of the posterior distribution).

Riders with high `P(PRESERVING)` on mountain stages are flagged as candidates for the following flat stage. The model outputs the posterior probability of each state per rider per stage.

---

### Strategy 6: Weather SPDE (Gaussian Process)

**The edge.** ITT markets are efficient on form but often poor on weather. The start window of a long ITT can span 3–4 hours. Wind conditions 90 minutes into the window can be completely different from conditions at the start. Riders who hit a tailwind on a key exposed section versus a headwind can differ by 30–90 seconds on a 40km course — a gap that dwarfs typical margins. When weather data arrives after markets open, or when the market simply hasn't adjusted, there's edge.

**What the model does.** It models the wind field along the ITT course as a **Gaussian Process** (GP) — a probability distribution over functions. The GP kernel captures:

- Spatial correlation: nearby course segments have correlated wind
- Temporal correlation: wind at 10:00 is more like wind at 10:30 than 14:00

```
w(s, t) ~ GP(mu(s,t), K((s,t), (s',t')))
```

For each rider, given their start time and the time-varying wind field estimate, the model integrates expected headwind/tailwind exposure along their course trajectory. Riders starting into a forecasted tailwind on the key exposed sections get a time adjustment applied to their base time trial estimate. This feeds into adjusted win probabilities.

The SPDE (Stochastic Partial Differential Equation) formulation is a computationally efficient approximation to a full GP — it uses a sparse matrix representation that scales to large spatial domains.

---

### Strategy 12: Online Changepoint Detection (BOCPD)

**The edge.** Live in-race betting markets on breakaway survival, stage winner, and attack confirmation move on information that arrives in real time. If the model can confirm an attack statistically before the TV feed fully processes it, or detect that a gap is structurally growing (not just noise), there is a timing edge in live markets.

**What the model does.** Bayesian Online Changepoint Detection (BOCPD, Adams & MacKay 2007). The model maintains a posterior distribution over "run length" — the time elapsed since the most recent structural break in the time series of power readings or time gaps:

```
P(r_t = k | x_{1:t}) ∝ Σ P(x_t | r_t, x_{(t-k):t}) * P(r_t | r_{t-1})
```

At each new observation, the posterior updates in real time. When the posterior probability of a changepoint exceeds a threshold, the model classifies it as a confirmed attack. The update runs in under 100ms, which is within the reaction window for live markets.

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
|   |-- runner.py               # Entry point: run this to scrape
|   |-- fetcher.py              # HTTP requests, rate limiting
|   |-- pcs_parser.py           # HTML parsing for ProCyclingStats
|   |-- db.py                   # Schema definitions, all DB writes
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
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.executescript(open('genqirue/data/schema_extensions.sql').read()); conn.commit(); conn.close()"
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

## 8. Interpreting Results

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

## 9. Quick Command Reference

```
# Install
pip install -e ../procyclingstats
pip install -r requirements.txt

# Verify
python quickstart.py

# Scrape
python -m pipeline.runner

# Monitor scraping
python monitor.py

# Apply betting schema (once)
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.executescript(open('genqirue/data/schema_extensions.sql').read()); conn.commit(); conn.close()"

# Full example workflow
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

## 10. Troubleshooting

**"ModuleNotFoundError: No module named 'procyclingstats'"**
```
pip install -e ../procyclingstats
```

**"ValueError: HTML from given URL is invalid"**
PCS is rate-limiting. Wait 5–10 minutes and restart. The queue resumes safely.

**"sqlite3.OperationalError: no such table"**
No data scraped yet. Run `python -m pipeline.runner` first.

**Jobs stuck "in_progress"**
The scraper crashed mid-job. Reset:
```
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"
```

**Frailty backtest shows 0 bets**
The frailty model generates signals from mountain stage data. If the first race in the database has no prior history, or has no mountain stages, it produces no output. Scrape multiple races across multiple years so there is always mountain stage training data available before the test race.

**PyMC/pytensor warning about g++**
Cosmetic. Models still run. To fix: `conda install gxx`. To suppress: set env variable `PYTENSOR_FLAGS=cxx=`.
