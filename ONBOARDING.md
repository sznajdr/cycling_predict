# Genqirue Onboarding Guide
## From Betting Intelligence to Computational Edge

**For:** Professional cycling traders with deep domain expertise, strong mathematical intuition, and zero programming experience.

**Goal:** Translate your existing edge — the patterns you see in races, the tells you read in communiqués, the form signals you price intuitively — into systematic, backtested, deployable models.

---

## Before You Start: The Checklist

- [ ] You understand why a gruppetto rider might be sandbagging
- [ ] You can price the impact of a rest day without looking at data
- [ ] You know what "closing line value" means and why it matters
- [ ] You've never written Python, never used Git, never queried a database
- [ ] You're comfortable with probability, Bayesian thinking, and differential equations at the conceptual level

If all boxes are checked, proceed. This guide assumes intelligence, not technical literacy.

---

# Part 1: The Conceptual Translation (No Code)

## Your Mental Models, Formalized

You already have a pricing framework in your head. This codebase formalizes it. Here's the mapping:

### "Reading the Race Communiqué" → Bayesian Updating (Strategy 3: Medical PK)

**What you do now:** A rider crashes on Stage 4. You read the communiqué: "superficial abrasions, will continue." You know from experience that "superficial" means anything from "fine tomorrow" to "limping through the mountains for a week. You mentally adjust your price — not to zero, not to full strength, but to some discounted curve that recovers over time.

**What the code does:** The two-compartment pharmacokinetic model treats physical trauma like a drug concentration decaying in the body:

```
dC_trauma/dt = -k_el · C_trauma
Perf(t) = Perf_baseline · (1 - C_trauma(t) / (EC_50 + C_trauma(t)))
```

Think of `C_trauma` as the "severity stock" in the rider's body. It decays at rate `k_el` (elimination rate — how fast they recover). The performance penalty follows an E-max curve: small trauma = small penalty, large trauma = large penalty, but with diminishing returns at the extreme.

**The edge:** The market treats the rider as either "out" or "fine." You know the truth is a continuous recovery curve. The PK model gives you that curve with parameters estimated from historical crash-recovery data.

### "Pricing a Rider's Form After a Rest Day" → Interrupted Time Series (Strategy 5)

**What you do now:** You track a rider's GC trajectory through a grand tour. They were fading — 2nd, 4th, 7th, 11th — then the rest day hits. You know some riders reset; others don't. You mentally adjust your price for the post-rest stages based on historical patterns.

**What the code does:** The interrupted time series model separates the systematic rest-day effect from underlying form:

```
Y_t = β_0 + β_1·Time_t + β_2·Intervention_t + β_3·TimeAfter_t + Σ_j φ_j·Y_{t-j} + ε_t
```

- `β_2` = immediate level shift (the rest day bump or drop)
- `β_3` = slope change (does their trajectory flatten, steepen, or reverse?)
- `φ_j` = autoregressive terms (momentum — are they trending up or down entering the rest day?)

**The edge:** The market treats rest day form as noise. You know it's signal. The ITS model quantifies that signal per rider based on their historical rest-day response pattern.

### "Spotting Desperation Breakaways" → Partially Observable Stochastic Games (Strategy 8)

**What you do now:** It's Stage 18 of a grand tour. A GC rider is 45 minutes down — irrelevant for the overall. They have zero stage wins. You see them attack on a transition stage and think: "Of course. They have nothing to lose." You price them higher than their form suggests because the incentive structure overrides form.

**What the code does:** The Quantal Response Equilibrium model prices actions based on strategic incentive, not just ability:

```
P(a_i | s) = exp(λ · Q_i(s, a_i)) / Σ_a exp(λ · Q_i(s, a))
```

The state `s` captures: GC position, stage wins so far, stages remaining, stage type. The Q-value `Q_i(s, a_i)` is the expected value of attacking from that state. A rider with GC lost + no wins + few stages remaining has high Q(attack) regardless of their watts.

**The edge:** The market prices on form signals (power data, recent results). You price on incentive structure. The POSG model formalizes that structure.

### "Kelly Staking" → Robust Kelly + CVaR Optimizer (`portfolio/kelly.py`)

**What you do now:** You find an edge — your model says 25%, the market says 20% (implied by odds of 4.0). You know Kelly says bet `f* = (bp - q)/b` where `b` is odds, `p` is your probability, `q = 1-p`. But you also know full Kelly is reckless — one bad run and you're ruined. You size down intuitively.

**What the code does:** Three layers of protection:

1. **Quarter-Kelly default:** `f = f*/4` — trades ~30% of long-run growth for ~4× drawdown variance reduction
2. **Robust Kelly:** When the model supplies a posterior standard deviation `σ_p` on its probability estimate, stake is automatically reduced:
   ```
   f_robust = f_kelly · (1 - γ · σ_p² · b² / p²)
   ```
   Higher uncertainty → tighter sizing. The penalty scales with leverage — uncertain longshots are penalized more than uncertain favorites.
3. **CVaR constraint:** The portfolio optimizer bounds the expected loss in the worst 5% of scenarios. This is the tail risk measure that governs ruin probability.

**The edge:** You already know Kelly. The code adds robustness to model uncertainty and portfolio-level risk constraints you can't enforce mentally at scale.

---

## Why Traditional Sportsbook Pricing Fails

| Market Inefficiency | Traditional Book Approach | Genqirue Approach |
|---------------------|---------------------------|-------------------|
| **Weather in ITTs** | Price on expected conditions at race time | Gaussian Process over full wind-field distribution; integrate over all possible conditions weighted by probability |
| **Gruppetto riders** | Price on observed time loss | Cox PH frailty model separates sandbagging (positive frailty) from genuine struggle (negative frailty) |
| **Post-crash form** | Binary: "out" or "fine" | PK model gives continuous recovery curve with rider-specific elimination rates |
| **Rest day effects** | Ignore or apply population average | ITS/BSTS isolates individual rest-day response pattern |
| **Cobble sector correlation** | Price each rider independently | Clayton copula models joint tail dependence — when one punctures, others are more likely to |

The books price on observable outcomes. You price on latent structure. This codebase makes that latent structure explicit and tradable.

---

# Part 2: The Environment (Hand-Holding Setup)

## Step 1: Open Your Terminal

**macOS:**
1. Press `Cmd + Space`
2. Type "Terminal"
3. Press Enter

**Windows:**
1. Press `Windows key`
2. Type "Command Prompt" or "PowerShell"
3. Press Enter

This is your command interface — think of it as the "betting terminal" where you place orders, but instead of calling a broker, you're running code.

## Step 2: Install Python

**Check if Python is installed:**

```bash
python --version
```

**What this means:** You're asking the computer "What version of Python do you have?" Python is the programming language this codebase uses.

**Expected output:** `Python 3.11.x` or `Python 3.13.x`

**If you see an error or version < 3.11:**

**macOS:**
```bash
# Install Homebrew (package manager)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

**Windows:**
1. Go to https://python.org/downloads
2. Download Python 3.11.x
3. Run installer — **check "Add Python to PATH"**
4. Restart Command Prompt
5. Verify: `python --version`

## Step 3: Navigate to Your Project Folder

**What this means:** You need to tell the terminal "I'm working on the cycling project now." This is like opening a specific workbook in Excel.

```bash
# macOS — if you put the project in Documents
cd ~/Documents/cycling_predict

# Windows — if you put the project in Documents
cd C:\Users\YourName\Documents\cycling_predict
```

**What `cd` means:** "Change Directory" — move to a different folder.

**Verify you're in the right place:**
```bash
ls        # macOS — list files
 dir       # Windows — list files
```

You should see files like `README.md`, `requirements.txt`, `quickstart.py`.

## Step 4: The "Sibling Repository" Concept

**What this means:** This project (`cycling_predict`) needs another project (`procyclingstats`) to function. They must live next to each other in the same parent folder — like two workbooks in the same file cabinet drawer.

**Required folder structure:**
```
parent_folder/
  cycling_predict/    ← you are here
  procyclingstats/    ← must be here, adjacent
```

**If you don't have `procyclingstats`:**

```bash
# Go up one level (to the parent folder)
cd ..

# Clone the procyclingstats repository
git clone https://github.com/ramonvermeulen/procyclingstats.git

# Go back into cycling_predict
cd cycling_predict
```

**What `git clone` means:** "Download a copy of this code repository." Think of it as making a local copy of a shared workbook.

**What `cd ..` means:** "Go up one folder level" — like clicking the "up" arrow in a file browser.

## Step 5: Install Dependencies

**What this means:** The project needs specific tools (Python libraries) to run. This is like installing Excel add-ins that the workbook requires.

```bash
# Install the procyclingstats library (the sibling repo)
pip install -e ../procyclingstats

# This means: "Install the Python package from the ../procyclingstats folder in 'editable' mode"
# -e = editable mode (changes to the code are reflected immediately)
# ../ = go up one folder, then into procyclingstats

# Install all other required libraries
pip install -r requirements.txt

# This means: "Read the requirements.txt file and install everything listed"
# -r = read from file
```

**Common Error 1: `pip: command not found`**
- **Fix:** Python didn't install correctly or isn't in your PATH. Reinstall Python and check "Add Python to PATH."

**Common Error 2: `ModuleNotFoundError: No module named 'procyclingstats'`**
- **Fix:** You didn't install the sibling repo. Run `pip install -e ../procyclingstats` again.

**Common Error 3: Permission denied**
- **macOS fix:** Add `sudo` before the command: `sudo pip install ...`
- **Windows fix:** Run Command Prompt as Administrator (right-click → Run as Administrator)

## Step 6: Set Up the Database

**What this means:** The project stores all data in a SQLite database — a single file (`data/cycling.db`) that acts like a structured Excel workbook with multiple sheets (tables) and strict relationships between them.

```bash
python fetch_odds.py --init-schema

# This means: "Run the Python script fetch_odds.py with the --init-schema flag"
# --init-schema = create all database tables for the first time
```

**What this creates:**
- `data/cycling.db` — the database file
- Tables for riders, races, stages, results, odds, and betting positions

**Common Error: `sqlite3.OperationalError: unable to open database file`**
- **Fix:** The `data/` folder doesn't exist. Create it: `mkdir data` then retry.

## Step 7: Install a SQLite Browser (Recommended)

**What this means:** You can "see" the database without writing SQL. This is like viewing an Excel workbook without knowing VBA.

**Download:** https://sqlitebrowser.org/dl/

**Why you want this:**
- Browse tables visually (like Excel sheets)
- Run SQL queries with a GUI (like Excel formulas)
- Export data to CSV
- Inspect the schema (table structures)

**To open the database:**
1. Open DB Browser for SQLite
2. File → Open Database
3. Navigate to `cycling_predict/data/cycling.db`
4. Click "Browse Data" to see tables

## Step 8: Verify Everything Works

```bash
python quickstart.py

# This means: "Run the quickstart demo script"
```

**What this does:**
- Loads sample data (or uses synthetic data if no races scraped yet)
- Fits the frailty and tactical models
- Prints betting opportunities

**Success indicators:**
- No red error messages
- Output shows model fitting progress (MCMC sampling)
- Final table shows riders with signal scores

**Common Error 1: `ModuleNotFoundError: No module named 'pymc'`**
- **Fix:** Dependencies didn't install. Run `pip install -r requirements.txt` again.

**Common Error 2: `FileNotFoundError: data/cycling.db`**
- **Fix:** Database not created. Run `python fetch_odds.py --init-schema`.

**Common Error 3: PyMC warnings about g++**
- **What this means:** PyMC (the Bayesian library) wants a C++ compiler for faster sampling. It's cosmetic — the models still run.
- **To suppress:** `export PYTENSOR_FLAGS=cxx=` (macOS/Linux) or `set PYTENSOR_FLAGS=cxx=` (Windows)
- **To fix properly:** `conda install gxx` (if using Anaconda)

---

# Part 3: The Data Pipeline (What Happens Before You Model)

## The Visual Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  ProCyclingStats │────▶│   SQLite DB  │────▶│   Models    │────▶│  Betclic Odds │────▶│ Positions │
│   (pcs.com)     │     │(data/cycling │     │(15 strategies│     │  (live market)│     │ (your bets)│
│                 │     │     .db)     │     │             │     │               │     │           │
└─────────────────┘     └──────────────┘     └─────────────┘     └──────────────┘     └───────────┘
       Stage 1                Stage 2              Stage 3              Stage 4            Stage 5
```

**Stage 1 — Scraping:** `pipeline/runner.py` hits ProCyclingStats at ~1 request per second. This is like manually copying race results into a spreadsheet, but automated and resume-safe.

**Stage 2 — Database:** Everything stores in `data/cycling.db`. Think of this as a very strict Excel workbook where every sheet has defined columns and relationships.

**Stage 3 — Modeling:** `genqirue/` reads from the database and fits probabilistic models. Each model answers: "Given history, what's the probability distribution over riders for the next stage?"

**Stage 4 — Live Odds:** `fetch_odds.py` scrapes Betclic's HTML for current markets. This gives you the "market price" to compare against your "model price."

**Stage 5 — Positions:** The Kelly optimizer sizes stakes based on edge: `edge = model_prob - (1 / market_odds)`.

## The Job Queue Mental Model

**What this means:** The scraper uses a persistent "to-do list" called `fetch_queue`. Every race, stage, rider, and result is a job on this list.

**Job states:**
- `pending` — waiting to be processed
- `in_progress` — currently being fetched
- `completed` — done, data stored
- `failed` — error occurred, will retry
- `permanent_fail` — failed after max retries

**Why this matters:** You can stop and restart the scraper anytime. It resumes from where it left off. This is like having an auto-save feature that also tracks what you've already done.

**To see the queue status:**
```bash
python monitor.py
```

**To reset stuck jobs** (if the scraper crashed mid-job):
```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\")
conn.commit()
print('Reset stuck jobs')
"
```

## Rate Limiting in Betting Terms

**What this means:** The scraper waits ~1 second between requests. This is like "steaming" a betting site slowly to avoid getting limited like a sharp bettor at a soft book.

**Why 1 req/s?**
- ProCyclingStats will block you if you go faster
- It's polite to the site operators
- It doesn't matter for your use case — you're not high-frequency trading, you're building overnight models

**The queue is your friend:** Start the scraper, let it run overnight. Check progress with `python monitor.py` in the morning.

## The Four Data Layers You Must Understand

### Layer 1: Historical Results

**Tables:** `rider_results`, `race_stages`, `riders`, `races`

**What this contains:** Every stage result, GC standing, time gap, startlist, and rider profile from all scraped races.

**Key columns to know:**
- `rider_results.time_behind_winner_seconds` — the raw time gap (input to Strategy 1)
- `rider_results.result_category` — 'stage' for stage result, 'gc' for GC standing
- `race_stages.stage_type` — 'flat', 'hilly', 'mountain', 'itt', 'ttt'
- `riders.sp_climber`, `sp_sprint`, `sp_gc` — specialty scores (0-100)

**Analogy:** This is like having a complete historical database of every horse race, with finishing times, positions, and jockey/trainer info.

### Layer 2: Real-Time Telemetry

**Tables:** `telemetry_changepoints` (Strategy 12), future tables for Strategies 10, 13

**What this contains:** Live power data, gap times, GPS positions — the data that arrives during a race.

**Latency requirement:** < 100ms per update for real-time strategies.

**Analogy:** This is like having live sectional times and stride data for horses during a race — information the market doesn't have yet.

### Layer 3: Weather Inputs

**Tables:** `weather_fields`, `itt_time_predictions`

**What this contains:** Wind speed/direction at points along the course, predicted time differences between ITT starters.

**Analogy:** This is like knowing the wind conditions at different parts of a track and which horses are starting into a headwind vs. tailwind.

### Layer 4: Market Data

**Tables:** `bookmaker_odds`, `bookmaker_odds_latest` (view)

**What this contains:** Raw odds from Betclic, hold-adjusted fair odds, market overround.

**Key columns:**
- `back_odds` — the odds offered (decimal)
- `fair_odds` — odds after removing bookmaker margin
- `market_type` — 'winner', 'top3', 'h2h', etc.

**Analogy:** This is the tote board — the current market prices you're trying to beat.

---

# Part 4: First End-to-End Execution (The "Hello World" of Cycling Models)

## Step 1: Configure What to Scrape

**File:** `config/races.yaml`

**What this means:** This file tells the scraper which races and years to fetch. It's like setting up your watchlist.

**Example configuration:**
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

**What each field means:**
- `pcs_slug` — the URL slug on ProCyclingStats (e.g., `pcs.com/race/paris-nice`)
- `history_years` — which historical editions to scrape for model training
- `type` — `stage_race` or `one_day_race`

## Step 2: Run the Scraper

```bash
python -m pipeline.runner

# This means: "Run the pipeline module's runner script"
# -m = run as a module (required for this project structure)
```

**What this does:**
1. Reads `config/races.yaml`
2. Creates jobs in `fetch_queue` for every race, stage, and rider
3. Processes jobs at ~1 req/s
4. Stores data in `data/cycling.db`

**To monitor progress** (in a second terminal):
```bash
python monitor.py
```

**Expected time:** 20-60 minutes per race depending on history depth. A full grand tour with 5 years of history might take 4-6 hours.

**What "done" looks like:**
- `monitor.py` shows all jobs as `completed`
- Row counts in core tables look reasonable (hundreds of riders, thousands of results)

## Step 3: Run Your First Models

**What we're doing:** Running Strategy 2 (Gruppetto Frailty) + Strategy 1 (Tactical HMM) only. These are the two implemented pre-race models.

```bash
python example_betting_workflow.py

# This means: "Run the example workflow that ties everything together"
```

**What this does:**
1. Loads data from `data/cycling.db`
2. Fits the frailty model on mountain stage gruppetto data
3. Fits the tactical HMM on time-loss data
4. Queries `bookmaker_odds_latest` for live odds (or uses simulated odds if none)
5. Calculates edge for each rider
6. Runs Robust Kelly optimizer
7. Prints recommended positions

**Understanding the output:**

```
Strategy 2 (Frailty) fitted: 847 riders, 3,421 survival records
Strategy 1 (Tactical HMM) fitted: 2,156 state observations

Top Opportunities (edge > 50bps):
┌─────────────────────┬──────────┬─────────────┬──────────┬──────────┐
│ Rider               │ Model Prob │ Market Prob │ Edge (bps) │ Kelly Stake │
├─────────────────────┼──────────┼─────────────┼──────────┼──────────┤
│ Rider A             │ 0.085    │ 0.042       │ 430      │ 2.1%     │
│ Rider B             │ 0.062    │ 0.031       │ 310      │ 1.5%     │
└─────────────────────┴──────────┴─────────────┴──────────┴──────────┘
```

**What "bps" means:** Basis points — 100 bps = 1 percentage point. An edge of 430 bps means your model says 8.5% vs. market 4.2% — a 4.3 percentage point difference.

## Step 4: Interpret Frailty Scores in Betting Terms

**The frailty score** (`b_i` in the equations) is like knowing a horse was held back in a prep race.

**Positive frailty** = rider survived longer than their observable characteristics predict
- They were in the gruppetto but didn't crack
- They lost time tactically, not physically
- **Bet signal:** Back them on the next transition stage

**Negative frailty** = rider survived less long than predicted
- They were genuinely at their limit
- The gruppetto was a survival effort, not tactical
- **Bet signal:** Fade them or avoid

**How to read the frailty table:**
```
Rider: Tadej Pogačar
Frailty: +0.73 (90th percentile)
Hidden Form Prob: 0.87
Signal: STRONG BUY (next transition stage)
```

**What this means:** Pogačar's frailty is in the top 10% of all riders — he was sandbagging, not struggling. The model is 87% confident he has hidden form. If the next stage is flat or hilly and the market hasn't adjusted, there's edge.

## Step 5: Run the Backtest

```bash
python run_backtest.py

# This means: "Run the walk-forward backtest on all implemented strategies"
```

**What this does:** Replays history in strict chronological order:
- Train on all races before Race R
- Predict Race R
- Record outcome → P&L
- Add Race R to training data
- Move to next race

**This is critical:** No lookahead. The model only knows what was knowable at the time.

**Sample output:**
```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
tactical        27      4    3.7%   3.7%    39.6%   1070.61   11.3%     0.000
baseline        93      4    1.1%   0.0%   -52.0%    596.34   45.3%     0.000
```

**How to read this as a trader:**

| Metric | What It Means | Interpretation |
|--------|---------------|----------------|
| **Bets** | Number of positions taken | Sample size — 72 bets is small; 200+ is meaningful |
| **Top3%** | Podium rate | Naive baseline is 2% (3/150). Frailty at 5.6% is 2.8× — real signal. |
| **ROI** | Return on investment | Against fair market (no margin). Subtract 5-15% for real book overround. |
| **Bankroll** | Final bankroll vs. start | 1686 = 68.6% gain on 1000 unit starting bankroll |
| **MaxDD** | Maximum drawdown | Peak-to-trough decline. 15% is healthy for quarter-Kelly. >40% suggests over-sizing. |
| **Spearman ρ** | Rank correlation | Measures if model rankings match actual outcomes. ρ > 0.23 is significant at n=72. |

**Statistical significance note:** At 72 bets, the standard error on the 5.6% top-3 rate is ~2.7%. The edge over the 2% null is 1.3 standard errors — directionally interesting, not conclusive. Target 200+ bets before treating metrics as stable.

## Step 6: Place Your First Paper Trade

**What this means:** The `example_betting_workflow.py` outputs recommended stakes but doesn't place real bets. This is your "paper trading" environment — practice without risk.

**To save bets to a file for tracking:**
```bash
python run_backtest.py --strategy frailty --kelly 0.1 --save-bets my_first_bets.csv

# This means: "Run frailty-only backtest with conservative 10% Kelly and save all bets to CSV"
```

**The CSV contains:**
- Race and stage
- Rider name
- Model probability
- Market odds at entry
- Edge (bps)
- Kelly stake
- Outcome (win/loss)
- P&L

**Track this like you would any betting ledger.** Over 200+ bets, the edge should materialize (or not — that's what backtesting tells you).

---

# Part 5: Extending the System (From Consumer to Creator)

## How to Add a New Strategy

**The `BayesianModel` ABC** (Abstract Base Class) is like a standardized bet submission form. Every strategy must implement the same interface so the portfolio optimizer can use it.

**Location:** `genqirue/models/base.py`

**Required methods:**
```python
def build_model(self, data: dict) -> None:
    """Define the PyMC model — specify priors, likelihood, parameters."""
    
def fit(self, data: dict) -> None:
    """Run MCMC sampling — calibrate the model to historical data."""
    
def predict(self, new_data: dict) -> dict:
    """Return win probability and uncertainty for new data."""
    
def get_edge(self, prediction: dict, market_odds: float) -> float:
    """Calculate edge in basis points vs. market odds."""
```

**Analogy:** This is like a standardized form every tipster must fill out: "What's your probability? What's your confidence? How does this compare to market price?" The optimizer can then compare across tipsters (strategies) and size accordingly.

## Example: Implementing Strategy 3 (Medical PK)

**File to create:** `genqirue/models/medical_pk.py`

**Class name:** `MedicalPKModel`

**The edge you bring:** You know that crash recovery follows a predictable curve, and the market prices it as binary (out/fine) rather than continuous.

**Step 1 — Create the file:**
```python
import pymc as pm
import numpy as np
from .base import BayesianModel

class MedicalPKModel(BayesianModel):
    """Two-compartment PK model for post-crash performance recovery."""
    
    def build_model(self, data: dict) -> None:
        """
        Data expected:
        - 'incidents': list of (rider_id, incident_date, incident_type, severity)
        - 'performances': list of (rider_id, stage_date, performance_score)
        """
        with pm.Model() as self.model:
            # Priors on PK parameters
            k_el = pm.HalfNormal('k_el', sigma=0.5)  # elimination rate
            EC50 = pm.HalfNormal('EC50', sigma=0.3)  # concentration at 50% effect
            
            # Rider-specific elimination rates (some recover faster)
            rider_k_el = pm.HalfNormal('rider_k_el', 
                                       sigma=0.2, 
                                       shape=data['n_riders'])
            
            # Likelihood: observed performance = baseline * penalty(d)
            # ... (implementation continues)
    
    def fit(self, data: dict) -> None:
        with self.model:
            self.trace = pm.sample(1000, tune=500, cores=4)
    
    def predict(self, new_data: dict) -> dict:
        # Calculate performance penalty for days since incident
        days = new_data['days_since_incident']
        k_el = self.trace.posterior['k_el'].mean()
        EC50 = self.trace.posterior['EC50'].mean()
        
        C_trauma = new_data['severity'] * np.exp(-k_el * days)
        penalty = C_trauma / (EC50 + C_trauma)
        
        return {
            'performance_multiplier': 1 - penalty,
            'uncertainty': self.trace.posterior['k_el'].std()
        }
    
    def get_edge(self, prediction: dict, market_odds: float) -> float:
        model_prob = prediction['performance_multiplier'] * self.baseline_win_prob
        market_prob = 1 / market_odds
        return (model_prob - market_prob) * 10000  # convert to bps
```

**Step 2 — Register the strategy:**

Add to `genqirue/models/__init__.py`:
```python
from .medical_pk import MedicalPKModel

__all__ = [
    # ... existing models
    'MedicalPKModel',
]
```

**Step 3 — Add acceptance criteria:**

In `docs/ENGINE.md`, add:
```
**Acceptance criterion:** Predicted performance penalty at day d=2 post-crash 
correlates with observed performance delta (vs pre-crash baseline) at r > 0.4 
on historical crash data.
```

## How to Scrape a Specific Race Not in Config

**Scenario:** Paris-Roubaix is tomorrow. It's not in your `races.yaml`. You need data fast.

**Step 1 — Add to config temporarily:**
```yaml
races:
  - name: Paris-Roubaix
    pcs_slug: paris-roubaix
    type: one_day_race
    history_years: [2022, 2023, 2024]  # get recent history for context
```

**Step 2 — Run scraper for just this race:**
```bash
python -m pipeline.runner
```

**Step 3 — Check what was scraped:**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print('Riders:', conn.execute(\"SELECT COUNT(*) FROM riders\").fetchone()[0])
print('Stages:', conn.execute(\"SELECT COUNT(*) FROM race_stages\").fetchone()[0])
print('Results:', conn.execute(\"SELECT COUNT(*) FROM rider_results\").fetchone()[0])
"
```

**Step 4 — Run models:**
```bash
python example_betting_workflow.py
```

## Modifying Kelly Parameters for Your Risk Appetite

**Location:** `genqirue/portfolio/kelly.py`

**Current defaults:**
```python
@dataclass
class KellyParameters:
    method: str = 'quarter_kelly'  # 'full', 'half', 'quarter', 'eighth'
    max_position_pct: float = 0.25  # 25% max per position
    min_edge_bps: float = 50       # 50 basis points minimum edge
    cvar_alpha: float = 0.95       # 95% CVaR constraint
    cvar_limit: float = 0.10       # 10% max CVaR
```

**Conservative settings** (lower variance, lower growth):
```python
params = KellyParameters(
    method='eighth_kelly',      # very conservative
    max_position_pct=0.10,      # 10% max per position
    min_edge_bps=100,           # only bet strong edges
    cvar_limit=0.05             # tight tail risk control
)
```

**Aggressive settings** (higher variance, higher growth):
```python
params = KellyParameters(
    method='half_kelly',        # more aggressive
    max_position_pct=0.40,      # larger positions
    min_edge_bps=25,            # bet smaller edges
    cvar_limit=0.20             # looser tail risk
)
```

**When to use what:**
- **Quarter-Kelly (default):** Good starting point. Balanced growth/drawdown.
- **Eighth-Kelly:** When learning the system. Survives model miscalibration.
- **Half-Kelly:** When you have high confidence in model calibration and deep bankroll.
- **Full Kelly:** Theoretical optimum. Dangerous in practice due to model error.

---

# Part 6: The "Sharp" Edge Cases

## Strategy 6 (ITT Weather): The Purest Arb

**Why this is the cleanest edge:** Physical time vs. market consensus time.

**The setup:**
- ITT start window: 3-4 hours
- Weather forecast updates after markets open
- Early starters into headwind vs. late starters into tailwind = 30-90 second difference
- Typical GC separation: 10-30 seconds

**The math:**
```
ΔT = ∫_0^D [P/F_aero(v_wind(t_early)) - P/F_aero(v_wind(t_late))] dx
```

**What this means:** The model integrates the expected time difference along the entire course, accounting for wind direction, rider power, and aerodynamic drag. When the weather forecast updates, the model knows the true expected time difference before the market reprices.

**Execution:**
1. Weather data updates at 10:00 AM
2. Markets opened at 8:00 AM based on earlier forecast
3. Model calculates: early starters +45 seconds vs. late starters
4. Market still prices on old forecast
5. **Bet:** Back late starters in H2H markets, lay early starters

**Risk:** Weather forecast error. The model's uncertainty (`σ_p`) should be high if forecast confidence is low — Robust Kelly will downsize accordingly.

## Strategy 12 (BOCPD): Latency Arbitrage on Live Markets

**The edge:** Live markets move on TV coverage. TV coverage lags GPS/power data by 15-30 seconds.

**The model:** Bayesian Online Changepoint Detection confirms structural pace changes in < 50ms:

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

**What this means:** The model maintains a probability distribution over "how long since the last structural change" (run length). When a power surge occurs, it updates in milliseconds whether this is a real attack or noise.

**Bet signal:** `P(changepoint) > 0.8` AND prior-stage Z-score > 2.0

**Execution:**
1. Power data shows surge at 14:23:15.200
2. Model confirms attack at 14:23:15.250 (< 50ms)
3. TV coverage shows attack at 14:23:45.000 (30s lag)
4. Market reprices at 14:23:50.000
5. **You had 30+ seconds to bet before the market moved.**

**Infrastructure requirement:** Direct data feed (not TV), sub-second latency to betting API.

## Pre-Race vs. Real-Time Strategies

| Aspect | Pre-Race (Strategies 1-9) | Real-Time (Strategies 10, 12, 13) |
|--------|---------------------------|-----------------------------------|
| **Latency tolerance** | Hours (overnight batch) | < 100ms |
| **Update frequency** | Once per stage | Continuous |
| **Data source** | Historical database | Live telemetry |
| **Edge type** | Information asymmetry | Speed asymmetry |
| **Capital capacity** | Higher (more time to size) | Lower (fast execution required) |
| **Implementation** | Python + PyMC | Python + Numba JIT |

**Portfolio implication:** Pre-race strategies can take larger positions (more time for line movement, better liquidity). Real-time strategies are smaller, faster, higher Sharpe.

## The Trader Override Protocol

**When to override the model:**

1. **Rider-specific news not in database:**
   - "Rider X had a fever last night" — not in PCS communiqué yet
   - Model likes Rider X based on form
   - You know to fade them

2. **Team dynamics not captured:**
   - Model sees two strong riders on same team
   - You know the team has publicly committed to one leader
   - Second rider will work, not contest

3. **Course changes:**
   - Weather model uses original route
   - Organizers announced a detour due to landslide
   - Model predictions invalid

**How to override:**

**Option 1 — Exclude rider from consideration:**
```sql
-- Add to rider_exclusions table (create if doesn't exist)
INSERT INTO rider_exclusions (rider_id, race_id, stage_id, reason, excluded_at)
VALUES (123, 456, 789, 'Fever reported, not in PCS yet', datetime('now'));
```

**Option 2 — Manual probability adjustment:**
```python
# In your strategy wrapper, before calling optimizer
if rider_id == 123:  # rider with fever
    prediction['win_prob'] *= 0.5  # halve their probability
    prediction['uncertainty'] *= 2  # double uncertainty
```

**Option 3 — Zero stake override:**
```python
# In position creation, add exclusion check
if rider_id in manual_exclusions:
    continue  # skip this rider entirely
```

**Documentation requirement:** Log every override with reason, timestamp, and expected vs. actual outcome. Review monthly. Overrides should improve performance; if they don't, stop overriding.

---

# Technical Specifications Reference

## Key Files and Their Purpose

| File | Purpose | When to Read It |
|------|---------|-----------------|
| `ENGINE.md` | Implementation logic, data flow, acceptance criteria | When implementing a new strategy |
| `MODELS.md` | Mathematical specifications, equations, priors | When you need the formal model definition |
| `COMMANDS.md` | CLI reference, SQL queries, maintenance tasks | When you need a specific command |
| `genqirue/models/base.py` | Abstract base class for all strategies | When creating a new strategy |
| `genqirue/portfolio/kelly.py` | Robust Kelly + CVaR optimizer | When modifying staking logic |
| `pipeline/runner.py` | Data scraping entry point | When the scraper breaks |
| `config/races.yaml` | Race configuration | When adding new races |

## "Fitting the Model" = "Calibrating the Pricing Algorithm"

**What MCMC sampling does:** PyMC runs thousands of parallel "universes" to explore the parameter space. Each sample is a possible world where the parameters take different values consistent with the observed data.

**Convergence diagnostics:**
- **R-hat < 1.01:** Chains have converged (agree on the answer)
- **ESS > 400:** Effective sample size is adequate (enough independent samples)
- **No divergences:** The sampler didn't get stuck in problematic regions

**What this means in betting terms:** The model has "settled" on a consistent probability estimate given the historical data. If R-hat is high, the model is uncertain — Robust Kelly will downsize stakes.

## SQLite Schema as Excel Workbook

**Analogy:** SQLite is Excel with strict relationships.

| Excel Concept | SQLite Equivalent |
|---------------|-------------------|
| Workbook file | `data/cycling.db` |
| Sheet | Table (e.g., `riders`, `races`) |
| Row | Record (e.g., one rider) |
| Column | Field (e.g., `name`, `birthdate`) |
| Formula | SQL query or computed column |
| VLOOKUP | SQL JOIN |
| Pivot table | SQL GROUP BY |
| Data validation | Schema constraints (NOT NULL, FOREIGN KEY) |

**Key tables:**
- `riders` — rider profiles (name, birthdate, specialty scores)
- `races` — race metadata (name, year, PCS slug)
- `race_stages` — stage details (number, type, distance, elevation)
- `rider_results` — stage results (rank, time gaps, result category)
- `startlist_entries` — who started each race
- `bookmaker_odds` — market odds snapshots
- `fetch_queue` — scraping job queue

---

# Troubleshooting Guide

## The 5 Most Common Errors

### 1. `ModuleNotFoundError: No module named 'procyclingstats'`

**Cause:** Sibling repo not installed or not in the right location.

**Fix:**
```bash
# Verify folder structure
ls ..  # should show both cycling_predict and procyclingstats

# Reinstall
pip install -e ../procyclingstats
```

### 2. `sqlite3.OperationalError: no such table: X`

**Cause:** Database schema not applied or table doesn't exist.

**Fix:**
```bash
# Apply schema
python fetch_odds.py --init-schema

# If still failing, check which table is missing
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print([t[0] for t in tables])
"
```

### 3. Jobs stuck `in_progress`

**Cause:** Scraper crashed mid-job.

**Fix:**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
n = conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\").rowcount
conn.commit()
print(f'Reset {n} stuck jobs')
"
```

### 4. Frailty backtest shows 0 bets

**Cause:** No mountain stage data in database, or no transition stages after mountains.

**Fix:**
```bash
# Check what stage types you have
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print(conn.execute(\"SELECT stage_type, COUNT(*) FROM race_stages GROUP BY stage_type\").fetchall())
"

# If no 'mountain' stages, scrape more races with mountain stages
# Edit config/races.yaml to include grand tours
```

### 5. `ValueError: HTML from given URL is invalid`

**Cause:** ProCyclingStats is rate-limiting you.

**Fix:**
- Wait 5-10 minutes
- The queue will resume from where it stopped
- Check `monitor.py` to see if jobs are progressing again

---

# Glossary: Betting Terms → Code Terms

| Betting Term | Code Term | What It Means |
|--------------|-----------|---------------|
| Pricing a market | `predict()` method | Generate probability distribution over outcomes |
| Staking | `optimize_portfolio()` | Kelly calculation + risk constraints |
| Edge | `get_edge()` | Model prob minus market-implied prob |
| Form | Latent state `z_t` | Hidden fitness/tactical state |
| Variance | Posterior std `σ_p` | Model uncertainty on probability |
| Bankroll | `bankroll` parameter | Starting capital for backtest |
| Drawdown | `max_dd` metric | Peak-to-trough decline |
| Closing line | `bookmaker_odds_latest` | Final market price before event |
| Steam move | Odds change between scrapes | Detected by comparing odds snapshots |
| Paper trade | Backtest with `--save-bets` | Simulate bets without real money |

---

# Next Steps

1. **Complete the setup** — Run through Part 2 until `python quickstart.py` succeeds
2. **Scrape your first race** — Start with a single race in `config/races.yaml`
3. **Run the backtest** — Understand the output metrics
4. **Paper trade** — Track model predictions vs. outcomes without risk
5. **Iterate** — Add strategies, refine parameters, build your edge

---

**Remember:** This system doesn't replace your domain expertise — it formalizes it. You already know that gruppetto riders can be sandbagging, that rest days reset form, that crashes have recovery curves. The code just makes those intuitions precise, backtestable, and tradable at scale.

Your edge is in the model assumptions, not the code. The code is just the delivery mechanism.

---

*For questions, refer to:*
- `ENGINE.md` — implementation details
- `MODELS.md` — mathematical specifications
- `COMMANDS.md` — complete CLI reference
