# Cycling Predict

A Bayesian betting engine for professional cycling races. Scrapes historical race data, fits probabilistic models that identify market inefficiencies, and validates them through rigorous walk-forward backtesting before any real money is involved.

---

## Table of Contents

1. [What This Is — and What It Is Not](#1-what-this-is--and-what-it-is-not)
2. [The Core Idea: Finding Edge](#2-the-core-idea-finding-edge)
3. [How the Models Work](#3-how-the-models-work)
   - [Strategy 2: Gruppetto Frailty (Cox Proportional Hazards)](#strategy-2-gruppetto-frailty)
   - [Strategy 1: Tactical HMM (Hidden Markov Model)](#strategy-1-tactical-hmm)
   - [Strategy 6: Weather SPDE (Gaussian Process)](#strategy-6-weather-spde)
   - [Strategy 12: Online Changepoint Detection](#strategy-12-online-changepoint-detection)
4. [Portfolio Construction: The Kelly Criterion](#4-portfolio-construction-the-kelly-criterion)
5. [Backtesting: Walk-Forward Validation](#5-backtesting-walk-forward-validation)
6. [Project Structure](#6-project-structure)
7. [Setup](#7-setup)
8. [Step-by-Step Usage](#8-step-by-step-usage)
   - [Step 1: Scrape Data](#step-1-scrape-data)
   - [Step 2: Run the Models](#step-2-run-the-models)
   - [Step 3: Run the Backtest](#step-3-run-the-backtest)
   - [Step 4: Interpret Results](#step-4-interpret-results)
9. [Quick Command Reference](#9-quick-command-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Glossary](#11-glossary)

---

## 1. What This Is — and What It Is Not

**What it is:** A research and modelling framework. You point it at ProCyclingStats.com, it downloads years of race history, fits statistical models that look for systematic patterns the betting market misprices, and then tests those models on historical races to see whether the patterns hold up out-of-sample.

**What it is not:** A guaranteed money printer. Betting markets are populated by professionals. Finding and sustaining an edge is hard. The purpose of this system is to make the question "do I actually have edge?" answerable with data before risking capital.

**The honest version:** You may run this, backtest it across 50 races, and find that the models have no edge after all. That is a valid and useful result. The machinery is here so the answer comes from evidence, not intuition.

---

## 2. The Core Idea: Finding Edge

A betting market assigns implied probabilities to outcomes. If a bookmaker offers odds of 10.0 on a rider winning a stage, the implied probability is 1/10 = 10%. The actual probability of that rider winning might be 10%, 5%, or 18% — the market is not omniscient.

**Edge** is the difference between your estimated probability and the market's implied probability:

```
Edge = P(model) - P(market)
     = P(model) - 1 / decimal_odds
```

If Edge > 0, the expected value of that bet is positive. If Edge < 0, the market is smarter than you on this outcome and you should not bet. If you can consistently find situations where your model is right and the market is wrong, you make money in the long run.

The challenge: you cannot simply look at past races and pick riders who won — that is lookahead bias and it is worthless for prediction. Every model must be trained exclusively on data that would have been available before the race you are predicting. The backtesting engine here enforces this mechanically.

---

## 3. How the Models Work

This system implements four models, each targeting a different structural inefficiency in cycling markets.

---

### Strategy 2: Gruppetto Frailty

**The observation.** In mountain stages, a subset of riders — sprinters, classics specialists, domestiques protecting a leader — form the gruppetto: a collective group at the back that rides tempo to finish inside the time cut without racing. These riders are not struggling; they are conserving energy. On the following day's flat or rolling stage, they may be fresher than the GC riders who fought on the mountain.

**The model.** This is a Cox Proportional Hazards survival model with rider-level random effects (frailty).

Survival analysis originally modelled time-to-death in medical trials. Here it models time-to-abandonment (DNF or OTL). The hazard function for rider *i* at time *t* is:

```
lambda_i(t) = lambda_0(t) * exp(beta^T * X_i + b_i)
```

Where:
- `lambda_0(t)` is the baseline hazard — the underlying rate at which riders drop out
- `X_i` is a vector of observed covariates: GC position, time behind leader, whether they were in the gruppetto, how much time they lost
- `beta` are the fixed effect coefficients (shared across all riders)
- `b_i ~ Normal(0, sigma^2)` is the **frailty term** — a rider-specific random effect capturing unobserved heterogeneity

A rider with high frailty who ends up in the gruppetto on a mountain stage survived longer than the model predicted from their observed covariates. This residual survival — the gap between what the model expected and what happened — is the signal. A large positive residual on a mountain stage means the rider was not struggling as much as their gruppetto membership implies. They have hidden form.

Practically: we fit the model on all stages preceding the one we want to predict, extract the frailty estimates, and rank riders. The ones with the highest frailty are our candidates for outperforming on the next transition stage.

---

### Strategy 1: Tactical HMM

**The observation.** A rider's time loss on any given stage is not purely a function of fitness. It also reflects tactical decisions: protecting a leader, setting up a sprint, staying out of danger before a key mountain stage. Two riders with identical time losses may be in completely different physiological states.

**The model.** A Hidden Markov Model (HMM) with two latent states:

- **State 0 — CONTESTING:** The rider is racing at full capacity. Time loss reflects true fitness.
- **State 1 — PRESERVING:** The rider is deliberately not racing. Time loss is tactical, not physiological.

The latent state `z_{i,t}` is unobserved. What we observe is the time loss. The HMM posits:

```
P(z_{i,t} = PRESERVING) = sigmoid(delta_0 + delta_1 * GC_gap + delta_2 * IsHillOrMountain)
```

Riders far behind on GC have less incentive to fight; on hard stages this effect is amplified. The time loss distribution is a mixture:

```
time_loss | z = CONTESTING  ~ Normal(mu, sigma^2)
time_loss | z = PRESERVING  ~ Normal(mu + gamma, sigma^2)
```

where `gamma ~ TruncatedNormal+(mean=120s, sd=30s)` is the tactical time loss — the extra seconds a rider deliberately concedes when preserving.

The key insight: if the model assigns high `P(PRESERVING)` to a rider on a mountain stage, that rider may have significant untapped energy heading into the next flat stage. We bet on them for the following day.

---

### Strategy 6: Weather SPDE

**The observation.** Individual time trials (ITTs) are raced alone against the clock. Wind is a significant performance modifier — and critically, wind changes over the course of the start window. Riders who start at different times face different conditions. This is not priced efficiently by markets because it requires integrating real-time weather data with rider start times.

**The model.** A Stochastic Partial Differential Equation (SPDE) approach using a Gaussian Process to model the spatiotemporal wind field along the ITT course:

```
w(s, t) ~ GP(mu(s,t), K((s,t), (s',t')))
```

where the kernel `K` captures spatial correlation (nearby course segments have correlated wind) and temporal correlation (wind at 10:00 is more similar to 10:30 than to 14:00).

For each rider, we integrate the wind effect along their individual trajectory through this field. Riders starting into a headwind versus a tailwind can have their effective time shifted by 30–90 seconds on a 40km ITT, which is enormous relative to typical gaps. When the market has not adjusted for this (because the weather data came in after markets opened), we have edge.

---

### Strategy 12: Online Changepoint Detection

**The observation.** During a race, power data and time gaps update in real time. An attack — a decisive acceleration that opens a gap — follows a characteristic statistical signature in the gap dynamics: variance increases, the mean shifts upward, and the change is approximately abrupt.

**The model.** Bayesian online changepoint detection using the BOCPD algorithm (Adams & MacKay, 2007). The model maintains a posterior distribution over run lengths — the time elapsed since the last changepoint:

```
P(r_t = k | x_{1:t}) proportional to sum over r_{t-1} of P(x_t | r_t, x_{(t-r_t):t}) * P(r_t | r_{t-1})
```

At each new observation `x_t` (power reading or time gap), the posterior is updated in O(t) time. When the posterior probability of a changepoint exceeds a threshold, we classify it as a confirmed attack. This is used for in-race live betting markets where the signal must arrive within 100ms.

---

## 4. Portfolio Construction: The Kelly Criterion

Each model outputs a probability estimate `p_model` for an outcome. Given decimal odds `b`, the optimal fraction of bankroll to stake — the fraction that maximises long-run log wealth — is the Kelly fraction:

```
f* = (b * p - (1 - p)) / (b - 1)
   = (b * p - q) / (b - 1)
```

where `p = p_model` and `q = 1 - p`. Note:
- If `p * b < 1` (negative expected value), Kelly gives `f* < 0`, meaning do not bet.
- Full Kelly is optimal in theory but highly sensitive to probability estimation errors. A 10% overestimate of probability can lead to ruin.
- This system defaults to **Quarter-Kelly** (`f = f* / 4`), which sacrifices some expected growth rate in exchange for much lower variance and drawdown.

**Robust Kelly.** When the model provides a probability estimate with uncertainty (a posterior standard deviation `sigma_p`), we penalise the Kelly fraction for that uncertainty:

```
f_robust = f_kelly * (1 - gamma * sigma_p^2 * (b)^2 / p^2)
```

Higher uncertainty means smaller stake. This is particularly important for Bayesian models where the posterior distribution is wide early in the season.

**Portfolio-level constraints.** The system uses convex optimisation (via CVXPY) to solve the multi-asset Kelly problem subject to:
- No single position exceeds 25% of bankroll
- Portfolio variance stays below a threshold
- CVaR at 95% confidence stays bounded

CVaR (Conditional Value at Risk) measures the expected loss in the worst 5% of scenarios. It is a more conservative risk measure than variance because it focuses on tail behaviour, which is where ruin actually comes from.

---

## 5. Backtesting: Walk-Forward Validation

Before trusting any model's output, you need to know whether it would have made money historically — and crucially, under realistic conditions.

**The lookahead problem.** The naive approach — fit a model on all your data, see how well it predicts the data — is worthless for betting. You are measuring how well the model memorised data it was trained on. A model that fits noise perfectly will look brilliant in-sample and fail catastrophically out-of-sample.

**Walk-forward validation** solves this by enforcing strict temporal ordering:

```
For each race R in chronological order:
    training_data = all stage results from races that FINISHED before R started
    for each stage S in R:
        training_data += all prior stages of R (stages 1 to S-1)
        fit model on training_data
        predict stage S
        observe actual outcome
        record P&L
```

No data from stage S or later is ever used when predicting stage S. This simulates exactly what would have happened if you were running the model live.

**What the metrics mean:**

| Metric | What it measures | Good sign |
|--------|-----------------|-----------|
| Top3% | Fraction of bets where the rider podiums | Higher than 3/field_size |
| Win% | Fraction of bets where the rider wins the stage | Higher than 1/field_size |
| ROI | Total profit / total staked | > 0% |
| Max Drawdown | Worst peak-to-trough bankroll decline | < 30% |
| Spearman rho | Rank correlation between predicted scores and actual finish positions | > 0, p < 0.05 |
| Brier Score | Mean squared error of probability estimates | < 0.25 |

The **baseline** strategy (random rider selection) is your null hypothesis. If your model cannot consistently beat random selection, it has no predictive power worth acting on.

**Important:** With fewer than ~20 races in the database, ROI figures are dominated by variance. A single large win or loss can swing the headline number by 100%. Collect more data before drawing conclusions.

---

## 6. Project Structure

```
cycling_predict/
|
|-- pipeline/                   # Data collection
|   |-- runner.py               # Main scraper — run this first
|   |-- fetcher.py              # HTTP + rate limiting
|   |-- pcs_parser.py           # HTML parsing for ProCyclingStats
|   |-- db.py                   # Database schema + write operations
|   `-- queue.py                # Job queue (resume-safe scraping)
|
|-- genqirue/                   # Betting engine
|   |-- models/
|   |   |-- base.py             # Abstract base: BayesianModel, SurvivalModel, etc.
|   |   |-- gruppetto_frailty.py   # Strategy 2: Cox PH frailty model
|   |   |-- tactical_hmm.py        # Strategy 1: Hidden Markov Model
|   |   |-- weather_spde.py        # Strategy 6: Gaussian Process / SPDE
|   |   `-- online_changepoint.py  # Strategy 12: BOCPD
|   |-- portfolio/
|   |   `-- kelly.py            # Robust Kelly + CVaR portfolio optimiser
|   |-- domain/
|   |   |-- entities.py         # Data structures (RiderState, MarketState, etc.)
|   |   `-- enums.py            # StageType, TacticalState, MarketType, etc.
|   `-- data/
|       `-- schema_extensions.sql  # Betting tables (frailty scores, tactical states)
|
|-- backtesting/
|   |-- engine.py               # Walk-forward backtester
|   `-- __init__.py
|
|-- config/
|   `-- races.yaml              # Which races and years to scrape
|
|-- tests/
|   |-- test_connection.py      # Verify ProCyclingStats is reachable
|   |-- test_rider.py           # Verify rider scraping works
|   |-- test_race.py            # Verify race scraping works
|   `-- betting/
|       `-- test_strategies.py  # Unit tests for betting models
|
|-- data/                       # Created automatically
|   `-- cycling.db              # SQLite database (not in git — too large)
|
|-- logs/                       # Created automatically
|   `-- pipeline.log
|
|-- run_backtest.py             # CLI for backtesting
|-- example_betting_workflow.py # End-to-end worked example
|-- monitor.py                  # Check scraping progress
`-- quickstart.py               # Verify installation
```

---

## 7. Setup

### Requirements

- **Python 3.11 or 3.13**. Download from https://www.python.org/downloads/
- **The procyclingstats library** — a separate scraping library that must live in the parent folder alongside this project.

Check your Python version by opening a terminal and running:

```
python --version
```

### Install

Navigate to the project folder in your terminal, then run:

```
pip install -e ../procyclingstats
pip install -r requirements.txt
```

The first line installs the scraping library from the adjacent folder. The second installs everything else:

- **PyMC** — probabilistic programming for Bayesian models (MCMC sampling, HMMs, GPs)
- **scikit-survival** — survival analysis (Cox PH model for the frailty estimator)
- **cvxpy** — convex optimisation (Kelly portfolio construction)
- **pandas / numpy / scipy** — data manipulation and statistics
- **pytest** — running tests

### Verify installation

```
python quickstart.py
```

This runs a self-test. If it completes without errors, everything is working.

---

## 8. Step-by-Step Usage

### Step 1: Scrape Data

#### Configure what to scrape

Open `config/races.yaml`. It looks like this:

```yaml
year: 2026

races:
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]
```

`pcs_slug` is the URL identifier used by ProCyclingStats — for Paris-Nice the URL is `procyclingstats.com/race/paris-nice`, so the slug is `paris-nice`. To add Tour de France, you would add:

```yaml
  - name: Tour de France
    pcs_slug: tour-de-france
    type: stage_race
    history_years: [2021, 2022, 2023, 2024, 2025]
```

Start with one or two races. You can always add more later.

#### Run the scraper

```
python -m pipeline.runner
```

The scraper downloads, in order:
1. Race metadata (dates, categories, UCI points scale)
2. Stage details (distances, elevation profiles, stage type: flat / hilly / mountain / ITT)
3. Startlists (which riders entered each race)
4. Stage results (finishing positions, time gaps to winner in seconds)
5. Rider profiles (nationality, height, weight, PCS specialty scores)

**Speed:** Roughly one HTTP request per second. The scraper rate-limits itself deliberately — ProCyclingStats is a free resource and excessive scraping gets IP addresses blocked.

**Duration:** Expect 20–60 minutes for your first race with several historical years.

**It is safe to interrupt** (Ctrl+C) and resume at any time. The scraper uses a persistent job queue — it will pick up exactly where it left off.

#### Monitor progress

Open a second terminal window while the scraper runs:

```
python monitor.py
```

Output:

```
=== Queue status ===
  race_meta            completed       8
  stage_results        in_progress     1
  rider_profile        pending         287
```

The three statuses are: `pending` (not yet started), `in_progress` (currently running), `completed`.

#### Check what you have

```
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print('Races:',   conn.execute('SELECT COUNT(DISTINCT id) FROM races').fetchone()[0])
print('Riders:',  conn.execute('SELECT COUNT(*) FROM riders').fetchone()[0])
print('Results:', conn.execute('SELECT COUNT(*) FROM rider_results').fetchone()[0])
conn.close()
"
```

---

### Step 2: Run the Models

#### End-to-end example

```
python example_betting_workflow.py
```

This script runs the full pipeline on whatever data you have scraped:
1. Loads stage results from the database
2. Fits the Frailty model (Strategy 2)
3. Fits the Tactical detector (Strategy 1)
4. Generates frailty scores and tactical state probabilities for all riders
5. Feeds those signals into the Kelly optimiser
6. Prints a portfolio report: which riders to bet on, how much, and the expected edge

#### Apply the betting database schema

Before running models for the first time, extend the database with tables for storing model output:

```
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
conn.executescript(open('genqirue/data/schema_extensions.sql').read())
conn.commit()
conn.close()
print('Done.')
"
```

This adds tables for `rider_frailty`, `tactical_states`, and the views that join them with race results.

---

### Step 3: Run the Backtest

```
python run_backtest.py
```

This runs three strategies in walk-forward mode across all races in the database and prints a comparison table.

#### Options

```
python run_backtest.py --strategy frailty     # Only the frailty model
python run_backtest.py --strategy tactical    # Only the tactical HMM
python run_backtest.py --strategy baseline    # Only the random baseline

python run_backtest.py --kelly 0.1            # More conservative Kelly fraction (10% instead of 25%)
python run_backtest.py --top-k 5             # Bet on top 5 riders per stage (default: 3)
python run_backtest.py --no-top3             # Bet on stage win instead of podium (top-3)
python run_backtest.py --bankroll 5000        # Start with 5000 instead of 1000

python run_backtest.py --save-bets bets.csv   # Export every individual bet to a CSV file
```

---

### Step 4: Interpret Results

#### The output table

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

**Column by column:**

**Bets** — Total number of individual bets placed across all races. More bets = more statistical power.

**Races** — Number of races that contributed at least one bet. This is your effective sample size for drawing conclusions. Fewer than 20 is unreliable.

**Top3%** — Of all bets placed, what fraction of those riders actually finished in the top 3. The naive baseline (a random pick in a field of N riders) would give approximately 3/N. For a typical stage race field of 150 riders, that is 2%. If your model shows 5–6%, it is identifying riders who podium more than twice as often as chance.

**Win%** — Same as Top3% but for stage wins. The naive baseline is approximately 1/N.

**ROI** — Return on Investment: total profit divided by total amount staked, expressed as a percentage. This is calculated against a simulated fair market (no bookmaker margin). A real bookmaker will take a 5–15% margin, so an ROI of 8% in simulation might be 0% or negative in practice. An ROI of 30%+ in simulation is worth investigating further with real odds data.

**Bankroll** — The final bankroll after all bets, starting from 1000. This includes compounding — each bet is sized as a fraction of the current bankroll, not the starting bankroll.

**MaxDD** — Maximum Drawdown: the worst peak-to-trough decline as a percentage of the peak bankroll. A drawdown of 15% means at some point the bankroll fell 15% from its highest point before recovering. Drawdowns above 40–50% are psychologically difficult to sustain even if the long-run expectation is positive.

**Spearman** — Spearman rank correlation between the model's predicted scores and actual finishing positions (higher score should predict lower rank number). A value of 0.077 means a weak but positive correlation — the model's ranking slightly anticipates how riders actually finish. A value of 0 means the model's rankings have no relationship to outcomes. Statistical significance requires both the value and the p-value — export to CSV with `--save-bets` and test significance separately.

#### How to tell if a result is real

The question is always: is this ROI signal, or noise?

With 4 races and 72 bets, the standard error on your top-3 hit rate is approximately:

```
SE = sqrt(p * (1-p) / n) = sqrt(0.056 * 0.944 / 72) ≈ 0.027
```

So you are seeing 5.6% ± 2.7% (one standard error), and the naive baseline is 2%. The gap is 3.6 percentage points, which is roughly 1.3 standard errors. That is suggestive but not conclusive. At 1.96 standard errors (p = 0.05), you need the gap to be wider, which requires either more data or a larger effect size.

**Practical rule of thumb:** Do not trust ROI figures from fewer than 200 individual bets. Scrape more races and rerun.

---

## 9. Quick Command Reference

```
# Install
pip install -e ../procyclingstats
pip install -r requirements.txt

# Verify installation
python quickstart.py

# Scrape data
python -m pipeline.runner

# Monitor scraping (in a second terminal)
python monitor.py

# Apply betting schema (first time only)
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.executescript(open('genqirue/data/schema_extensions.sql').read()); conn.commit(); conn.close()"

# Run the full example workflow
python example_betting_workflow.py

# Run backtest (all strategies)
python run_backtest.py

# Run backtest and save every bet to CSV
python run_backtest.py --save-bets bets.csv

# Reset stuck scraping jobs
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"

# Run tests
pytest tests/ -v
```

---

## 10. Troubleshooting

**"ModuleNotFoundError: No module named 'procyclingstats'"**
The scraping library is not installed. Run:
```
pip install -e ../procyclingstats
```
Make sure the `procyclingstats` folder exists in the parent directory alongside `cycling_predict`.

**"ValueError: HTML from given URL is invalid"**
ProCyclingStats is temporarily blocking requests. Wait 5–10 minutes and restart. Check `logs/pipeline.log` for details.

**"sqlite3.OperationalError: no such table"**
The database does not exist yet. Run the scraper first:
```
python -m pipeline.runner
```

**Jobs stuck in "in_progress" after a crash**
The scraper crashed mid-job and left some entries locked. Reset them:
```
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"
```

**"Database is locked"**
Another process (the scraper, or a previous terminal session) is still holding the database open. Close any other terminals that might be running the scraper, or restart.

**PyMC/pytensor warning about g++**
This is cosmetic. The Bayesian models will still run, but without C compilation they will be slower. To fix on Windows:
```
conda install gxx
```
To suppress the warning permanently, set the environment variable `PYTENSOR_FLAGS=cxx=` (empty value).

**Backtest shows 0 bets for frailty**
The frailty model requires mountain stage data in the training window to produce non-zero scores. This means it cannot generate signals for the very first race in the database (no prior history) or for races that only have flat stages. Scrape more races, or add races that include mountain stages (Paris-Nice, Criterium du Dauphine, Tour de Suisse, any grand tour).

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Bayesian inference** | A statistical framework where you start with a prior belief about a parameter, observe data, and update to a posterior belief using Bayes' theorem: P(params \| data) ∝ P(data \| params) * P(params) |
| **BOCPD** | Bayesian Online Changepoint Detection. An algorithm that maintains a real-time posterior over when the last structural break in a time series occurred |
| **Brier Score** | Mean squared error between predicted probabilities and binary outcomes. Ranges from 0 (perfect) to 1 (perfectly wrong). A random model on a 1/N outcome has Brier score ≈ (1 - 1/N) |
| **Cox Proportional Hazards** | A survival model where the hazard (instantaneous dropout rate) for each rider is the baseline hazard multiplied by an exponential function of their covariates |
| **CVaR** | Conditional Value at Risk. The expected loss in the worst alpha% of scenarios. More sensitive to tail risk than variance |
| **Domestique** | A rider whose primary job is to support their team's leader rather than race for personal result |
| **Edge** | The difference between your model probability and the market's implied probability. Positive edge = positive expected value |
| **Frailty** | In survival analysis, a rider-specific random effect capturing unobserved heterogeneity in their "true" fitness or toughness |
| **Gaussian Process** | A probability distribution over functions, fully specified by a mean function and a covariance kernel. Used here to model the spatiotemporal wind field during ITTs |
| **GC** | General Classification — the overall standings in a stage race, based on cumulative time |
| **Gruppetto** | A group of riders at the back of the peloton on mountain stages who cooperate to ride tempo and finish within the time cut, rather than racing |
| **HMM** | Hidden Markov Model. A probabilistic model where an observed sequence (e.g. time losses) is generated by an unobserved sequence of latent states (e.g. CONTESTING vs PRESERVING) |
| **ITT** | Individual Time Trial. A stage raced alone against the clock, no drafting |
| **Kelly Criterion** | The bet sizing formula f* = (b*p - q) / b that maximises the expected log of wealth, derived by John Kelly in 1956 |
| **Lookahead bias** | Using information about the future when training or testing a model. Produces optimistic backtest results that will not replicate in live trading |
| **MCMC** | Markov Chain Monte Carlo. A class of algorithms for sampling from complex posterior distributions when analytical solutions are intractable |
| **PCS** | ProCyclingStats.com — the primary data source |
| **ROI** | Return on Investment: profit / amount staked |
| **Spearman correlation** | A rank-based correlation coefficient measuring whether higher model scores tend to correspond to better (lower-numbered) finishing positions |
| **SPDE** | Stochastic Partial Differential Equation. Used here as a computationally efficient approach to Gaussian Process regression on spatial domains |
| **Walk-forward validation** | A backtesting methodology where the model is retrained at each time step using only historically available data, then tested on the next unseen period |
