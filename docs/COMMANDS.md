# Command Reference

Complete reference for every CLI entry point, database operation, and maintenance task in the project.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [PCS Data Scraping](#2-pcs-data-scraping)
3. [Betclic Odds Scraping](#3-betclic-odds-scraping)
4. [Betting Workflow](#4-betting-workflow)
5. [Stage Ranking](#5-stage-ranking)
6. [Head-to-Head (H2H) Matchup Predictions](#6-head-to-head-h2h-matchup-predictions)
7. [Weather Analysis (ITT)](#7-weather-analysis-itt-wind-arbitrage)
8. [Backtesting](#8-backtesting)
9. [Database Administration](#9-database-administration)
10. [Testing](#10-testing)
11. [Monitoring](#11-monitoring)
12. [Scheduled Automation](#12-scheduled-automation)
13. [Git Workflow](#13-git-workflow)

---

## 1. Environment Setup

### First-time setup (new machine)

```bash
# Clone the repo
git clone https://github.com/your-org/cycling-predict.git
cd cycling-predict

# Install the procyclingstats scraping library (required sibling repo)
pip install -e ../procyclingstats

# Install all other dependencies
pip install -r requirements.txt

# Apply database schema (creates data/cycling.db if it doesn't exist)
python scripts/fetch_odds.py --init-schema
```

### Automated setup (new team member)

```bash
python scripts/setup_team.py
```

Runs all setup steps in order: Python version check, venv creation, dependency installation, schema application, import tests.

### Verify everything works

```bash
# Check PCS connectivity (requires network)
python tests/test_connection.py

# Check scraping + DB roundtrip (requires network, ~2 HTTP requests each)
python tests/test_rider.py
python tests/test_race.py

# Quick end-to-end demo (requires scraped data)
python scripts/quickstart.py
```

### Virtual environment (optional but recommended)

```bash
# Create
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate

# Deactivate
deactivate
```

---

## 2. PCS Data Scraping

### Run the scraper

```bash
python -m pipeline.runner
```

Reads `config/races.yaml`, seeds the job queue for all configured races, and processes jobs in priority order until the queue is empty. Safe to stop (Ctrl+C) and resume at any time.

### Monitor progress in a second terminal

```bash
python scripts/monitor.py
```

Prints a live queue status table (pending/in_progress/completed per job type) and row counts for all core tables.

### Reset stuck jobs

If the scraper was killed mid-job, that job stays `in_progress` indefinitely. Reset all:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
n = conn.execute(\"UPDATE fetch_queue SET status='pending' WHERE status='in_progress'\").rowcount
conn.commit()
print(f'Reset {n} stuck jobs')
"
```

### Force re-scrape of a specific race

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
slug, year = 'tour-de-france', 2024
conn.execute(\"DELETE FROM data_freshness WHERE entity_type='race_meta' AND entity_key=?\",
             (f'{slug}/{year}',))
conn.execute(\"UPDATE fetch_queue SET status='pending', retries=0 WHERE job_type='race_meta' AND pcs_slug=? AND year=?\",
             (slug, year))
conn.commit()
print('Done — re-run pipeline.runner to re-fetch')
"
```

### Reset permanently failed jobs

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
n = conn.execute(\"UPDATE fetch_queue SET status='pending', retries=0, last_error=NULL WHERE status='permanent_fail'\").rowcount
conn.commit()
print(f'Reset {n} permanent_fail jobs to pending')
"
```

### One-time migration: add stage metadata columns

Run this once if upgrading from an older schema that lacked stage enrichment fields:

```bash
python scripts/reset_stage_jobs.py
python -m pipeline.runner
```

---

## 3. Betclic Odds Scraping

### Apply odds schema (first time only)

```bash
python scripts/fetch_odds.py --init-schema
```

Creates the `bookmaker_odds` table, `bookmaker_odds_latest` view, and indexes. Idempotent — safe to run repeatedly.

### Dry-run a single event (no DB writes)

```bash
python scripts/fetch_odds.py --dry-run --event-url <betclic-event-url>
```

Prints a table of all selections with raw odds, hold-adjusted fair odds, and market overround. No data is written. Use this to verify a URL works before running a full scrape.

### Dry-run the full hub (no DB writes)

```bash
python scripts/fetch_odds.py --dry-run
```

Discovers all live cycling events on the Betclic hub and prints every selection. No data is written.

### Full hub scrape (writes to DB)

```bash
python scripts/fetch_odds.py
```

Discovers all live events, extracts odds, and inserts rows into `bookmaker_odds`. Each run gets a UUID `scrape_run_id`. The `bookmaker_odds_latest` view always reflects the most recent snapshot per selection.

### Scrape a single event (writes to DB)

```bash
python scripts/fetch_odds.py --event-url <betclic-event-url>
```

### Use a custom DB path

```bash
python scripts/fetch_odds.py --db path/to/custom.db
```

---

## 4. Betting Workflow

### Quick demo (Strategies 1 + 2, requires scraped data)

```bash
python quickstart.py
```

Loads Paris-Nice 2024 data (or whatever is in the DB), fits the frailty and tactical models, and prints the top betting opportunities with signal strength scores.

### Full stage analysis with portfolio optimization

```bash
python scripts/example_betting_workflow.py
```

Fits Strategies 1 and 2, generates positions for all riders with signals above threshold, runs the Robust Kelly optimizer, and prints recommended stakes with edge estimates. Uses real Betclic odds from `bookmaker_odds_latest` where available; falls back to simulated odds otherwise.

### Run with real odds (scrape first)

```bash
python fetch_odds.py && python example_betting_workflow.py
```

---

## 5. Stage Ranking

Pre-race ranking that combines up to six signals (specialty with finish-type blending + power-to-weight, cross-race recent form, historical, frailty, tactical, GC relevance) into softmax probabilities over the full startlist, joins live Betclic odds, and computes edge and Kelly stakes.

Full documentation: [`docs/RANKING.md`](docs/RANKING.md).

---

## 6. Head-to-Head (H2H) Matchup Predictions

Quick H2H probability calculator for betting matchups. Supports individual riders vs individual riders, or riders vs "The Field" (Das Feld).

### How It Works

The H2H script uses the stage ranking model probabilities to compute conditional win probabilities:

```
P(A beats B) = P(A wins) / (P(A wins) + P(B wins))
```

For "The Field" matchups:
```
P(A beats Field) = P(A wins) / (1 - P(A wins))
```

**Important:** The script reads **saved** probabilities from `strategy_outputs` table. It does NOT compute fresh probabilities. 

**Workflow:**
1. Run `rank_stage.py --save` to compute and save fresh probabilities
2. Run `h2h.py` to query those saved probabilities

**Data Freshness Warning:**
If data is >1 hour old, you'll see:
```
WARNING: Data is 7.7 hours old. Run with --save to refresh:
  python scripts/rank_stage.py paris-nice 2026 2 --save
```

This ensures H2H probabilities always match the ranking output exactly.

### Single Matchup

```bash
python scripts/h2h.py paris-nice 2026 2 -m "Zingle vs Godon"
```

Output:
```
Zingle vs Godon: 58.3% / 41.7% | Fair odds: @1.71 / @2.40
```

### Multiple Matchups

```bash
python scripts/h2h.py paris-nice 2026 2 \
    -m "Zingle vs Godon" \
    -m "Girmay vs Fretin" \
    -m "Lamperti vs Braet"
```

### The Field Matchups

Bet on whether a specific rider wins, or anyone else (the field):

```bash
python scripts/h2h.py paris-nice 2026 2 -m "Lamperti vs Das Feld"
```

Output:
```
Lamperti vs Das Feld: 8.5% / 91.5% | Fair odds: @11.76 / @1.09
```

Useful for breakaway vs peloton scenarios where the market prices a rider against the entire field.

### From File

Create a file `matchups.txt`:
```
# H2H Matchups Template
# Lines starting with # are comments
# Format: Rider A vs Rider B
# Supports "Das Feld" or "The Field" for field bets

Bryan Coquard vs Pascal Ackermann
Jonas Vingegaard vs The Field
Wilco Kelderman vs Aleksandr Vlasov
```

Run:
```bash
python scripts/h2h.py paris-nice 2026 2 -f matchups.txt
```

Output:
```
======================================================================
Paris Nice 2026 Stage 2 - H2H Predictions
======================================================================
Rider A                   vs Rider B                   | Prob A | Fair Odds
----------------------------------------------------------------------
Bryan Coquard             vs Pascal Ackermann          |  73.9% | @1.35 / @3.82
Jonas Vingegaard          vs The Field                 |   1.8% | @56.52 / @1.02
Wilco Kelderman           vs Aleksandr Vlasov          |  66.2% | @1.51 / @2.96
======================================================================
```

### Interactive Mode

```bash
python scripts/h2h.py paris-nice 2026 2
> Zingle vs Godon
Zingle vs Godon: 58.3% / 41.7% | Fair odds: @1.71 / @2.40

> Girmay vs Das Feld
Girmay vs Das Feld: 12.4% / 87.6% | Fair odds: @8.06 / @1.14

> done
```

Type `quit`, `exit`, or `done` to exit interactive mode.

### Different Races

```bash
# Tirreno Stage 1 ITT
python scripts/h2h.py tirreno-adriatico 2026 1 -m "Ganna vs Hayter"

# Paris-Nice Stage 5
python scripts/h2h.py paris-nice 2026 5 -m "Vingegaard vs Ayuso"
```

---

## 7. Weather Analysis (ITT Wind Arbitrage)

Strategy 6 implementation - analyzes wind conditions for Individual Time Trials to find riders with advantageous/disadvantaged start times. **Completely free - no API key required.**

### Quick ITT Analysis

```bash
# Auto-fetch from free providers (Open-Meteo/MET Norway/NOAA)
python scripts/weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# European race (uses MET Norway - very accurate)
python scripts/weather_race_analyzer.py --race paris-nice --year 2026 --stage 1

# Specify course bearing (0=North, 90=East, 180=South, 270=West)
python scripts/weather_race_analyzer.py --race tour-de-france --year 2025 --stage 21 --bearing 180
```

### Manual Forecast Entry (No Internet)

```bash
# Enter forecast manually - perfect for race day updates
python scripts/weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1 \
    --manual "14:00:5.2@180,15:00:6.8@200,16:00:4.1@220"

# Format: HH:MM:windspeed@direction
# Example: 14:00:5.2@180 = 2pm, 5.2 m/s wind from 180 degrees (South)
```

### Understanding Output

| Metric | What it means |
|--------|---------------|
| **Time Delta** | Seconds gained/lost vs neutral conditions (negative = faster) |
| **Advantage Score** | 0-100 scale, >60 = good conditions, <40 = poor conditions |
| **Spread** | Max time difference between any two start times (>15s = strong opportunity) |
| **BACK list** | Riders with tailwind advantage - potential value bets |
| **FADE list** | Riders with headwind disadvantage - potential lays |

### Free Weather Providers (No API Key)

The system automatically selects the best free provider:

| Provider | Coverage | Best For |
|----------|----------|----------|
| **MET Norway** | Europe | Tirreno, Paris-Nice, Giro, Vuelta |
| **NOAA** | USA | Tour of California, Tour of Utah |
| **Open-Meteo** | Global | Everywhere else |

All providers are completely free with no registration required.

---

## 8. Backtesting

### Run full walk-forward backtest (all strategies)

```bash
python scripts/run_backtest.py
```

### Options

```bash
python scripts/run_backtest.py --strategy frailty     # frailty only
python run_backtest.py --strategy tactical    # tactical HMM only
python run_backtest.py --strategy baseline    # random baseline
python run_backtest.py --strategy all         # all (default)

python scripts/run_backtest.py --kelly 0.25           # Kelly fraction (default: 0.25)
python run_backtest.py --top-k 3              # Riders to bet per stage (default: 3)
python run_backtest.py --no-top3              # Win market instead of podium market
python run_backtest.py --bankroll 5000        # Starting bankroll (default: 1000)
python scripts/run_backtest.py --save-bets bets.csv   # Export every bet to CSV
```

### Example — conservative frailty run

```bash
python scripts/run_backtest.py --strategy frailty --kelly 0.1 --top-k 5 --bankroll 10000 --save-bets results.csv
```

### Interpreting output

```
Strategy      Bets  Races   Top3%   Win%      ROI  Bankroll   MaxDD  Spearman
frailty         72      4    5.6%   1.4%   136.8%   1686.74   15.0%     0.077
```

- **Top3%** — podium rate; naive baseline is 2% (3/150 riders)
- **ROI** — profit over total staked, against a fair market (no margin)
- **MaxDD** — peak-to-trough drawdown; above 40% becomes difficult to sustain
- **Spearman** — rank correlation between model scores and actual positions; needs ρ > 0.23 for p < 0.05 at n = 72

---

## 9. Database Administration

### Connect to the database

```bash
# Python
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db')"

# SQLite CLI
sqlite3 data/cycling.db
```

### Check what's in the DB

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
tables = ['races','riders','race_stages','startlist_entries','rider_results','bookmaker_odds']
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t:<25} {n:>8}')
"
```

### Queue health check

```sql
SELECT job_type, status, COUNT(*) AS n
FROM fetch_queue
GROUP BY 1, 2
ORDER BY 1, 2;

SELECT job_type, pcs_slug, year, retries, last_error
FROM fetch_queue
WHERE status IN ('failed','permanent_fail')
ORDER BY job_type;
```

### Odds verification

```sql
-- Row counts by market type
SELECT market_type, COUNT(*) AS n, ROUND(AVG(back_odds),2) AS avg_odds
FROM bookmaker_odds
GROUP BY market_type
ORDER BY n DESC;

-- Latest snapshot — favourite prices for winner market
SELECT participant_name, back_odds, fair_odds, scraped_at
FROM bookmaker_odds_latest
WHERE market_type = 'winner'
ORDER BY back_odds ASC
LIMIT 10;

-- Unclassified market labels (extend classifier if recurring)
SELECT DISTINCT market_label_raw
FROM bookmaker_odds
WHERE market_type = 'unknown';

-- Price history for a selection
SELECT scraped_at, back_odds, fair_odds
FROM bookmaker_odds
WHERE participant_name_norm = 'pogacar'
  AND market_type = 'winner'
ORDER BY scraped_at;
```

### Feature engineering queries

```sql
-- GC standings entering a stage
SELECT ri.name, rr.rank, rr.time_behind_winner_seconds
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
JOIN races r ON r.id = s.race_id
WHERE r.pcs_slug = 'tour-de-france'
  AND r.year = 2023
  AND s.stage_number = 16
  AND rr.result_category = 'gc'
ORDER BY CAST(rr.rank AS INTEGER);

-- Mountain stage podium rate per rider
SELECT ri.name,
       COUNT(*) AS mountain_stages,
       SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS podiums,
       ROUND(100.0 * SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 1) AS podium_pct
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
WHERE s.stage_type = 'mountain' AND rr.result_category = 'stage' AND rr.rank GLOB '[0-9]*'
GROUP BY rr.rider_id
HAVING mountain_stages >= 5
ORDER BY podium_pct DESC;

-- Average time gap per rider per stage type (Strategy 1 input feature)
SELECT ri.name, s.stage_type, AVG(rr.time_behind_winner_seconds) AS avg_gap_s, COUNT(*) AS n
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
WHERE rr.result_category = 'stage' AND rr.time_behind_winner_seconds IS NOT NULL
GROUP BY rr.rider_id, s.stage_type
HAVING n >= 3
ORDER BY ri.name, s.stage_type;
```

### Apply schema extensions (all betting tables)

```bash
python scripts/fetch_odds.py --init-schema
```

Or directly:

```bash
sqlite3 data/cycling.db < genqirue/data/schema_extensions.sql
```

---

## 10. Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test files

```bash
pytest tests/test_connection.py -v      # PCS connectivity check (requires network)
pytest tests/test_rider.py -v           # Rider scrape + DB roundtrip
pytest tests/test_race.py -v            # Race meta + startlist scrape
pytest tests/betting/test_strategies.py -v  # Strategy unit tests (no network)
```

### Run with coverage

```bash
pytest tests/ --cov=genqirue --cov-report=html
open htmlcov/index.html
```

### Run only unit tests (no network)

```bash
pytest tests/betting/ -v
```

---

## 11. Monitoring

### Scraping progress

```bash
python monitor.py
```

Run in a second terminal while `pipeline.runner` is active. Refreshes queue state and table row counts on demand.

### Log file

```bash
# Linux/macOS
tail -f logs/pipeline.log

# Windows
Get-Content logs\pipeline.log -Wait
```

Log format per completed job:
```
2026-03-07 14:23:11 INFO pipeline.runner: race_meta done: paris-nice/2022 (7 stages queued)
2026-03-07 14:23:13 INFO pipeline.runner: stage_results done: race/paris-nice/2022/stage-1 (87 rows)
```

### Enable debug logging

Edit `pipeline/runner.py` and change the logging level from `INFO` to `DEBUG`. This shows every skipped rider and unrecognised result URL.

---

## 12. Scheduled Automation

### PCS scraper — cron (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Run daily at 06:00 and 18:00
0 6,18 * * * cd /path/to/cycling-predict && venv/bin/python -m pipeline.runner >> logs/cron.log 2>&1
```

### Odds scraper — cron (Linux/macOS)

```bash
# Every 30 minutes, 06:00–22:00
*/30 6-22 * * * cd /path/to/cycling-predict && venv/bin/python scripts/fetch_odds.py >> logs/odds.log 2>&1
```

### Windows Task Scheduler

Create a task for each command:
- **Program:** `python`
- **Arguments:** `-m pipeline.runner` (or `fetch_odds.py`)
- **Start in:** `path\to\cycling-predict`
- **Trigger:** Daily / Repeat every 30 minutes

### GitHub Actions (automated scraping in CI)

See `.github/workflows/scrape.yml` — triggered daily at 06:00 UTC and on manual dispatch. Uploads `data/cycling.db` as an artifact.

---

## 13. Git Workflow

### Daily development

```bash
# Pull latest
git pull origin main

# Create feature branch
git checkout -b feature/strategy-3-medical-pk

# Make changes, stage, commit
git add pipeline/betclic_scraper.py genqirue/models/medical_pk.py
git commit -m "Strategy 3: implement two-compartment PK model"

# Push branch and open PR
git push origin feature/strategy-3-medical-pk
```

### Branch naming conventions

| Prefix | Use for |
|--------|---------|
| `feature/` | New strategy or feature |
| `bugfix/` | Bug fix |
| `docs/` | Documentation only |
| `data/` | Config changes (new races, years) |

### Exporting race data (share without the full DB)

```bash
python scripts/export_race_data.py --race tour-de-france --year 2024
```

Creates a zip file suitable for sharing via cloud storage. To import:

```bash
python scripts/export_race_data.py --import-zip tour-de-france_2024.zip
```
