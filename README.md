# Cycling Predict - Complete Beginner's Guide

A step-by-step guide to scraping cycling data and building predictive models for cycling races. No prior experience required.

---

## Table of Contents

1. [What is This?](#what-is-this)
2. [Prerequisites](#prerequisites)
3. [Quick Start (5 Minutes)](#quick-start-5-minutes)
4. [Understanding the Directory Structure](#understanding-the-directory-structure)
5. [Step 1: Scraping Data](#step-1-scraping-data)
6. [Step 2: Understanding Your Database](#step-2-understanding-your-database)
7. [Step 3: Running the Betting Models](#step-3-running-the-betting-models)
8. [Step 4: Making Predictions](#step-4-making-predictions)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

---

## What is This?

This project has two main parts:

1. **Data Pipeline** (`pipeline/` folder) - Scrapes cycling race data from ProCyclingStats.com
2. **Betting Engine** (`genqirue/` folder) - Uses machine learning models to predict race outcomes

**What you can do:**
- Scrape historical data for any race (Tour de France, Giro, Paris-Nice, etc.)
- Build predictive models using 15 different strategies
- Identify riders with "hidden form" ( Strategy 2: Gruppetto )
- Detect tactical riders who are saving energy (Strategy 1: Tactical HMM)
- Optimize betting portfolios using Kelly criterion

---

## Prerequisites

### Required Software

1. **Python 3.11 or 3.13** (must be installed)
   - Download from: https://www.python.org/downloads/
   - Check: Open terminal and type `python --version`

2. **Git** (optional, for updates)
   - Download from: https://git-scm.com/downloads

### Install Dependencies

Open your terminal/command prompt in the project folder:

```powershell
# Windows PowerShell
cd C:\Users\danie\Downloads\nkls\cycling_predict

# Install the procyclingstats library (required for scraping)
pip install -e ..\procyclingstats

# Install all other dependencies
pip install -r requirements.txt
```

This installs:
- PyMC (Bayesian statistics)
- scikit-survival (survival analysis)
- cvxpy (portfolio optimization)
- pandas, numpy (data manipulation)
- pytest (testing)

---

## Quick Start (5 Minutes)

### Test Everything Works

```powershell
# 1. Test connection to ProCyclingStats
python tests/test_connection.py

# Expected output: PASS: TodayRaces.finished_races() returned list (N items)

# 2. Test rider scraping
python tests/test_rider.py

# Expected output: PASS lines for rider scraping

# 3. Test race scraping
python tests/test_race.py

# Expected output: PASS lines for race scraping
```

If all tests pass, you're ready to go!

---

## Understanding the Directory Structure

```
cycling_predict/
├── config/
│   └── races.yaml          # Configure which races to scrape
├── data/
│   └── cycling.db          # SQLite database (created automatically)
├── logs/
│   └── pipeline.log        # Scraping logs
├── pipeline/               # SCRAPING CODE (Part 1)
│   ├── db.py              # Database operations
│   ├── fetcher.py         # HTTP scraping functions
│   ├── pcs_parser.py      # URL/time parsing
│   ├── queue.py           # Job queue management
│   └── runner.py          # Main scraper
├── genqirue/              # BETTING MODELS (Part 2)
│   ├── domain/            # Data structures
│   ├── models/            # 15 betting strategies
│   │   ├── base.py       # Base model class
│   │   ├── gruppetto_frailty.py    # Strategy 2 (START HERE)
│   │   ├── tactical_hmm.py         # Strategy 1
│   │   ├── online_changepoint.py   # Strategy 12
│   │   └── weather_spde.py         # Strategy 6
│   ├── portfolio/         # Kelly optimization
│   │   └── kelly.py
│   └── data/
│       └── schema_extensions.sql   # Betting database schema
├── tests/
│   ├── test_connection.py
│   ├── test_rider.py
│   ├── test_race.py
│   └── betting/           # Betting model tests
├── example_betting_workflow.py  # EXAMPLE: Complete workflow
├── monitor.py             # Check scraping progress
└── reset_stage_jobs.py    # Fix stuck jobs
```

---

## Step 1: Scraping Data

### Configure What to Scrape

Edit `config/races.yaml`:

```yaml
year: 2026  # Primary prediction year

races:
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]

  # Uncomment more races as you get comfortable:
  # - name: Tour de France
  #   pcs_slug: tour-de-france
  #   type: stage_race
  #   history_years: [2021, 2022, 2023, 2024, 2025]
```

**For beginners:** Start with just Paris-Nice (already uncommented). Each race has:
- `pcs_slug`: The URL part (e.g., `procyclingstats.com/race/paris-nice`)
- `type`: `stage_race` (multi-day) or `one_day` (single day)
- `history_years`: Which years to scrape

### Run the Scraper

```powershell
# Start scraping (this will take 20-40 minutes for first run)
python -m pipeline.runner
```

**What happens:**
1. Creates `data/cycling.db` (SQLite database)
2. Downloads race metadata (dates, stages, categories)
3. Downloads startlists (which riders participated)
4. Downloads stage results (who won, time gaps)
5. Downloads rider profiles (height, weight, specialties)
6. Downloads historical results for each rider

**Monitor progress:**
```powershell
# In a new terminal while scraper is running:
python monitor.py

# Shows queue status like:
# === Queue status ===
#   race_meta            completed       4
#   stage_results        in_progress     1
#   rider_profile        pending         312
```

**Speed:** ~1 second per HTTP request (be nice to their servers!)

### Resume Scraping (If Interrupted)

```powershell
# Safe to stop anytime with Ctrl+C
# Then just restart:
python -m pipeline.runner

# It will resume from where it stopped!
```

### Check What You Have

```powershell
# Open the database with Python
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
cursor = conn.cursor()

# List races
cursor.execute('SELECT display_name, year FROM races')
print('RACES:')
for row in cursor.fetchall():
    print(f'  {row[0]} {row[1]}')

# Count riders
cursor.execute('SELECT COUNT(*) FROM riders')
print(f'RIDERS: {cursor.fetchone()[0]}')

# Count results
cursor.execute('SELECT COUNT(*) FROM rider_results')
print(f'RESULTS: {cursor.fetchone()[0]}')

conn.close()
"
```

---

## Step 2: Understanding Your Database

The database `data/cycling.db` contains these main tables:

### Core Tables

| Table | What It Stores |
|-------|---------------|
| `races` | Race info (name, year, dates, category) |
| `race_stages` | Individual stages (date, distance, profile) |
| `riders` | Rider profiles (name, nationality, height, weight) |
| `teams` | Teams (name, class, nationality) |
| `startlist_entries` | Who started which race |
| `rider_results` | Stage results (rank, time gaps, points) |

### Key Queries for Beginners

```sql
-- Get riders for a specific race
SELECT r.name, r.nationality, t.name as team
FROM startlist_entries sl
JOIN riders r ON sl.rider_id = r.id
JOIN teams t ON sl.team_id = t.id
JOIN races ra ON sl.race_id = ra.id
WHERE ra.pcs_slug = 'paris-nice' AND ra.year = 2024;

-- Get stage results
SELECT r.name, rs.stage_number, rr.rank, rr.time_behind_winner_seconds
FROM rider_results rr
JOIN riders r ON rr.rider_id = r.id
JOIN race_stages rs ON rr.stage_id = rs.id
JOIN races ra ON rs.race_id = ra.id
WHERE ra.pcs_slug = 'paris-nice' AND ra.year = 2024
  AND rs.stage_number = 1 AND rr.result_category = 'stage'
ORDER BY CAST(rr.rank AS INTEGER);
```

Run queries:
```powershell
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
query = '''
SELECT r.name, r.nationality 
FROM riders r 
LIMIT 10
'''
for row in conn.execute(query):
    print(row)
conn.close()
"
```

---

## Step 3: Running the Betting Models

### Apply Betting Schema

First, extend the database with betting tables:

```powershell
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
conn.executescript(open('genqirue/data/schema_extensions.sql').read())
conn.commit()
conn.close()
print('Betting schema applied!')
"
```

### Strategy Overview

The models are organized by priority (start with Strategy 2):

| Priority | Strategy | What It Does | File |
|----------|----------|--------------|------|
| **1** | Gruppetto Frailty | Finds riders hiding form in gruppetto | `gruppetto_frailty.py` |
| **2** | Online Changepoint | Detects attacks in real-time | `online_changepoint.py` |
| **3** | Tactical HMM | Identifies tactical energy preservation | `tactical_hmm.py` |
| **4** | Weather SPDE | ITT weather advantages | `weather_spde.py` |
| **5** | Kelly Portfolio | Optimal bet sizing | `portfolio/kelly.py` |

### Run the Complete Example

```powershell
python example_betting_workflow.py
```

This will:
1. Load scraped data from `cycling.db`
2. Fit Strategy 2 (Gruppetto Frailty) on 112 riders
3. Fit Strategy 1 (Tactical HMM) on 1,158 observations
4. Analyze Stage 5 of Paris-Nice 2024
5. Generate signals for 136 riders
6. Optimize portfolio with Kelly criterion

---

## Step 4: Making Predictions

### Basic Prediction Script

Create a new file `my_prediction.py`:

```python
"""
My first cycling prediction script.
Predicts winners for a specific stage.
"""
import sqlite3
from genqirue.models import FastFrailtyEstimator, SimpleTacticalDetector
from genqirue.models.gruppetto_frailty import SurvivalRecord
from genqirue.models.tactical_hmm import TacticalObservation
from genqirue.domain import StageType
from datetime import datetime

def load_data(conn, race_slug, year):
    """Load data from database."""
    query = '''
    SELECT 
        rr.rider_id,
        rs.stage_number,
        rs.stage_date,
        rs.stage_type,
        rr.time_behind_winner_seconds,
        rr.rank
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN races r ON rs.race_id = r.id
    WHERE r.pcs_slug = ? AND r.year = ?
      AND rr.result_category = 'stage'
    '''
    return conn.execute(query, (race_slug, year)).fetchall()

def main():
    # Connect to database
    conn = sqlite3.connect('data/cycling.db')
    
    print("Loading data for Paris-Nice 2024...")
    data = load_data(conn, 'paris-nice', 2024)
    
    # Prepare survival records for Strategy 2
    survival_records = []
    for row in data:
        rider_id, stage_num, stage_date, stage_type, time_loss, rank = row
        
        # Infer gruppetto status (time loss > 15 minutes)
        gruppetto = 1 if (time_loss or 0) > 900 else 0
        
        record = SurvivalRecord(
            rider_id=rider_id,
            stage_id=stage_num,
            stage_date=datetime.strptime(stage_date, '%Y-%m-%d') if stage_date else datetime.now(),
            stage_type=stage_type or 'road',
            time_to_cutoff=45.0,
            event_observed=False,
            gc_position=int(rank) if rank and rank not in ['DNF', 'DNS'] else 150,
            gc_time_behind=time_loss or 0,
            gruppetto_indicator=gruppetto,
            gruppetto_time_loss=(time_loss or 0) - 600 if gruppetto else 0
        )
        survival_records.append(record)
    
    # Fit frailty model
    print(f"Fitting frailty model on {len(survival_records)} records...")
    estimator = FastFrailtyEstimator()
    estimator.fit(survival_records)
    
    # Find riders with hidden form
    print("\nRiders with Hidden Form (Strategy 2):")
    print("-" * 50)
    
    # Get rider names
    rider_ids = list(estimator.frailty_estimates.keys())
    hidden_form_riders = []
    
    for rider_id in rider_ids:
        frailty = estimator.get_frailty(rider_id)
        # High frailty = potential hidden form
        if frailty > 0.5:
            # Get rider name
            name = conn.execute(
                'SELECT name FROM riders WHERE id = ?', (rider_id,)
            ).fetchone()
            name = name[0] if name else f"Rider {rider_id}"
            
            hidden_form_riders.append((name, frailty))
    
    # Sort by frailty
    hidden_form_riders.sort(key=lambda x: -x[1])
    
    for name, frailty in hidden_form_riders[:10]:
        print(f"  {name}: frailty={frailty:.3f}")
    
    conn.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
```

Run it:
```powershell
python my_prediction.py
```

### Understanding the Output

**Frailty Score Interpretation:**
- `frailty > 0.5`: Rider spent time in gruppetto but didn't lose excessive time
- This suggests they were **preserving energy**, not struggling
- These riders often perform better than expected on transition stages

### Make Your Own Predictions

Modify the script to:

1. **Change the race:**
```python
data = load_data(conn, 'tour-de-france', 2023)  # Analyze TdF 2023
```

2. **Change the signal threshold:**
```python
if frailty > 0.3:  # Lower threshold = more riders
```

3. **Add tactical state analysis:**
```python
from genqirue.models.tactical_hmm import TacticalTimeLossHMM

hmm = TacticalTimeLossHMM()
# ... fit and get tactical state probabilities
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'procyclingstats'"

```powershell
# Make sure you're in the right directory
cd C:\Users\danie\Downloads\nkls\cycling_predict

# Install the library
pip install -e ..\procyclingstats
```

### "ValueError: HTML from given URL is invalid"

- The website might be blocking you
- Wait 5 minutes and try again
- Check `logs/pipeline.log` for details

### "sqlite3.OperationalError: no such table"

You haven't scraped any data yet:
```powershell
python -m pipeline.runner
```

### Jobs stuck "in_progress"

```powershell
# Reset stuck jobs
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\")
conn.commit()
conn.close()
print('Stuck jobs reset!')
"
```

### Database is locked

The scraper is still writing to it. Either:
- Wait for it to finish
- Stop the scraper with Ctrl+C
- Check: `python monitor.py` to see if anything is running

### PyMC warnings about g++

This is fine - models will still work, just slower:
```powershell
# To fix (optional):
conda install gxx
```

---

## Next Steps

### 1. Learn the Models

Read the detailed guides:
- `genqirue/README.md` - Deep dive into betting strategies
- `SCRAPE_README.md` - Complete scraping documentation

### 2. Run Tests

```powershell
# Test all betting models
pytest tests/betting/test_strategies.py -v

# Test scraping
python tests/test_connection.py
python tests/test_rider.py
python tests/test_race.py
```

### 3. Experiment

Try modifying:
- `config/races.yaml` - Add more races
- `example_betting_workflow.py` - Change analysis parameters
- Signal thresholds in models

### 4. Advanced: Implement More Strategies

The 15 strategies from PLAN.md:
- ✅ Strategy 1: Tactical Time Loss (HMM)
- ✅ Strategy 2: Gruppetto Frailty (Cox PH)
- ✅ Strategy 6: ITT Weather (Gaussian Process)
- ✅ Strategy 12: Attack Confirmation (Changepoint)
- ⏳ Strategy 3-5, 7-11, 13-15 (to implement)

### 5. Visualization

Add visualization to see patterns:
```python
import matplotlib.pyplot as plt

# Plot frailty distribution
frailties = list(estimator.frailty_estimates.values())
plt.hist(frailties, bins=20)
plt.xlabel('Frailty Score')
plt.ylabel('Number of Riders')
plt.title('Rider Frailty Distribution')
plt.show()
```

---

## Quick Command Reference

```powershell
# Setup
pip install -e ..\procyclingstats
pip install -r requirements.txt

# Test
python tests/test_connection.py

# Scrape
python -m pipeline.runner

# Monitor
python monitor.py

# Apply betting schema
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.executescript(open('genqirue/data/schema_extensions.sql').read()); conn.commit(); conn.close()"

# Run example
python example_betting_workflow.py

# Run your script
python my_prediction.py

# Reset stuck jobs
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\"); conn.commit(); conn.close()"

# Check database
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); print('Riders:', conn.execute('SELECT COUNT(*) FROM riders').fetchone()[0]); print('Results:', conn.execute('SELECT COUNT(*) FROM rider_results').fetchone()[0]); conn.close()"
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **Gruppetto** | The group of riders at the back on mountain stages |
| **Frailty** | Statistical measure of "toughness" - high frailty = potentially hiding form |
| **HMM** | Hidden Markov Model - detects hidden states (like "preserving energy") |
| **Kelly Criterion** | Mathematical formula for optimal bet sizing |
| **PCS** | ProCyclingStats.com - the data source |
| **Stage Race** | Multi-day race (Tour de France, Giro, etc.) |
| **ITT** | Individual Time Trial - race against the clock |
| **CVaR** | Conditional Value at Risk - measures potential losses |

---

## Support

If something breaks:
1. Check `logs/pipeline.log` for errors
2. Run `python monitor.py` to see queue status
3. Check database integrity: `python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); conn.execute('PRAGMA integrity_check'); print('OK'); conn.close()"`

---

**Happy predicting! 🚴‍♂️📊**
