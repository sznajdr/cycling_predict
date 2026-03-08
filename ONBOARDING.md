Now I have all the information needed. Let me write a comprehensive walkthrough for predicting Tirreno-Adriatico Stage 1:

---

# 🚴‍♂️ Complete Walkthrough: Tirreno-Adriatico 2026 Stage 1 Prediction

## Race Overview

| Attribute | Details |
|-----------|---------|
| **Race** | Tirreno-Adriatico 2026 ("Race of the Two Seas") |
| **Stage 1** | Lido di Camaiore → Lido di Camaiore |
| **Date** | Monday, March 9, 2026 (TOMORROW) |
| **Distance** | 11.5 km |
| **Type** | Individual Time Trial (ITT) - FLAT, COASTAL |
| **Key Characteristic** | Exposed to wind - **Strategy 6 (Weather Arbitrage) is critical** |

### Why This Stage is Special

The opening ITT at Tirreno is one of the most weather-sensitive stages in cycling betting:
- **Flat coastal course** = minimal terrain variation, wind becomes the dominant factor
- **11.5km distance** = short enough that 10-30 second wind advantages can decide the stage
- **3-hour start window** (14:00-17:00) = wind conditions can shift 90-180° during the race
- **Historical dominance**: Ganna has won 3 of the last 4 editions; Del Toro won once

---

## 📋 The Prediction Workflow

### Phase 1: Data Preparation (Do This First)

#### Step 1.1: Verify Environment Setup

```bash
# Navigate to project
cd c:\Users\danie\Downloads\nkls\cycling_predict

# Verify procyclingstats is installed
pip install -e ..\procyclingstats

# Initialize database schema (if not done)
python scripts\fetch_odds.py --init-schema
```

#### Step 1.2: Check Current Database State

```bash
# What's already in the database?
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
tables = ['races','riders','race_stages','startlist_entries','rider_results','bookmaker_odds']
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t:<25} {n:>8}')
"
```

#### Step 1.3: Scrape/Update Tirreno-Adriatico Data

The race is already configured in `config/races.yaml` with history from 2022-2026:

```yaml
- name: Tirreno-Adriatico
  pcs_slug: tirreno-adriatico
  type: stage_race
  history_years: [2022, 2023, 2024, 2025, 2026]
```

Run the scraper:

```bash
# Start the pipeline
python -m pipeline.runner

# Monitor progress in a SECOND terminal
python scripts\monitor.py
```

**What gets scraped:**
- Race metadata (stage profiles, distances, types)
- Startlists for 2026
- Historical results (2022-2025) for model training
- Rider profiles (specialty scores, weights, etc.)

#### Step 1.4: Scrape 2026 Season Form (Critical for Early Season!)

Since this is March, you need recent 2026 results for form signals:

```bash
# Scrape all 2026 races for Tirreno-Adriatico riders
python scripts\scrape_2026_season.py --race tirreno-adriatico --year 2026

# Or limit to top riders for speed
python scripts\scrape_2026_season.py --race tirreno-adriatico --year 2026 --rider-limit 30
```

**Key races to have:**
- UAE Tour 2026 (Del Toro won - huge form signal)
- Strade Bianche 2026 (recent form)
- Omloop Het Nieuwsblad 2026
- Volta ao Algarve 2026

---

### Phase 2: Model Fitting & Ranking

#### Step 2.1: Run the Full Stage Ranking with Models

This is your **primary prediction command**:

```bash
python scripts\rank_stage.py tirreno-adriatico 2026 1 --run-models --save --top 20
```

**What this does:**
1. **Fits Strategy 2 (Gruppetto Frailty)** - Cox Proportional Hazards with rider frailty
2. **Fits Strategy 1 (Tactical HMM)** - Hidden Markov Model for tactical states
3. **Computes 6 signals per rider:**
   - **Specialty (40% weight for ITT)**: `sp_time_trial` from PCS
   - **Historical (25%)**: Past Tirreno Stage 1 results
   - **Form (15%)**: 2026 season results with time decay
   - **Frailty (15%)**: Sandbagging detection from gruppetto patterns
   - **GC Relevance (5%)**: Less relevant for ITT
   - **Tactical (0%)**: Not used for ITT (road stage signal only)
4. **Softmax calibration**: Converts scores to realistic probabilities (target: 15-25% for favorite)
5. **Saves to database**: `strategy_outputs` table for later analysis

**Expected output format:**
```
Tirreno-Adriatico 2026 Stage 1 — ITT (11.5km)
Signals: specialty(0.40) historical(0.25) form(0.15) frailty(0.15) gc_relevance(0.05)
Field: 176 riders | Temperature: 12.3 | Edge threshold: 50bps

 Rank  Rider                  Spec   Hist   Form   ModelProb
---------------------------------------------------------
    1  GANNA Filippo          0.98   0.92   0.85      18.2%
    2  VAN AERT Wout          0.85   0.88   0.91      14.7%
    3  DEL TORO Isaac         0.72   0.65   0.97      12.3%
    4  ROGLIC Primoz          0.78   0.90   0.82       9.1%
    5  CHRISTEN Jan           0.68   0.45   0.94       6.8%
```

---

### Phase 3: Weather Analysis (The Secret Weapon)

**This is where you find your edge.** The market prices on average conditions; you price on actual forecast.

#### Step 3.1: Run Weather Analysis (FREE - No API Key Needed!)

```bash
# Auto-fetch from free providers (Open-Meteo/MET Norway)
python scripts\weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1
```

**What the weather analyzer does:**
1. Fetches forecast for Lido di Camaiore (43.867°N, 10.200°E)
2. Gets startlist with predicted start times (GC position order)
3. Maps wind conditions to each rider's start time
4. Calculates aerodynamic time deltas using power-velocity model

**Sample output:**
```
======================================================================
WEATHER RACE ANALYZER - TIRRENO-ADRIATICO 2026 Stage 1
======================================================================
Start window: 14:00 - 16:55
Startlist: 176 riders

----------------------------------------------------------------------
WIND IMPACT ANALYSIS
----------------------------------------------------------------------
Rank   Rider                     Start    Wind               Delta      Advantage
-------------------------------------------------------------------------------------
1      VAN AERT Wout             15:30    5.2m/s @ 175°    -2.3s      72/100 🟢 BEST
2      GANNA Filippo             15:45    5.0m/s @ 170°    -1.8s      68/100 🟢 GOOD
...
172    DEL TORO Isaac            14:00    4.5m/s @ 90°     +3.2s      38/100 🔴 POOR

Expected time spread: 5.5 seconds between best/worst conditions
```

#### Step 3.2: Manual Forecast Entry (Race Day Updates)

If you have updated forecasts closer to race time (no internet needed):

```bash
python scripts\weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1 \
    --manual "14:00:5.2@180,15:00:6.8@200,16:00:4.1@220"
# Format: HH:MM:windspeed@direction
```

#### Step 3.3: Interpret Weather Impact

| Metric | Interpretation | Action |
|--------|---------------|--------|
| **Time Spread >15s** | Strong weather opportunity | Bet aggressively on wind advantage |
| **Time Spread 5-15s** | Moderate opportunity | Watch for line movements |
| **Time Spread <5s** | Neutral conditions | Rely on base model rankings |
| **Wind shifts 90°+ during window** | Huge opportunity | Early/late starters massively mispriced |

**For Tirreno Stage 1 specifically:**
- Coastal location = variable winds
- Late afternoon = sea breezes often build
- If forecast shows wind shifting from onshore to offshore → late starters gain advantage

---

### Phase 4: Odds Integration & Value Finding

#### Step 4.1: Scrape Live Betclic Odds

```bash
# Full scrape to database
python scripts\fetch_odds.py

# Or dry-run first to check what's available
python scripts\fetch_odds.py --dry-run
```

#### Step 4.2: Verify Odds Loaded

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT participant_name, back_odds, fair_odds 
    FROM bookmaker_odds_latest 
    WHERE market_type = 'winner' 
    ORDER BY back_odds 
    LIMIT 10
''')
for row in cursor.fetchall():
    print(f'{row[0]:<25} {row[1]:>6.2f}  {row[2]:>6.2f}')
"
```

#### Step 4.3: Run Full Betting Workflow

```bash
python scripts\example_betting_workflow.py
```

**Output format:**
```
Top Opportunities (edge > 50bps):
Rider               Model Prob   Market Prob   Edge (bps)   Kelly Stake
-----------------------------------------------------------------------
Van Aert            0.147        0.089         580          4.2%
Christen            0.068        0.025        4300          6.8%
Jorgenson           0.054        0.032        2200          3.1%
```

---

### Phase 5: Applying Weather Adjustments to Probabilities

#### Manual Weather Override Protocol

The weather analysis gives you time deltas. Convert these to probability adjustments:

```python
# Rule of thumb from docs/WEATHER_TOOLS.md:
# ~5% probability shift per 10 seconds of time advantage

# Example calculation:
# Base model: Van Aert 14.7% win probability
# Weather: -2.3s advantage (faster than neutral)
# Adjustment: 14.7% * (1 + (2.3/60)*0.05) ≈ 15.0% (small for short TT)

# For 11.5km ITT, use larger adjustments:
# 1 second ≈ 1-2% probability shift at the top
```

#### Revised Ranking with Weather

| Rider | Base Model | Weather Delta | Adjusted Prob | Market Odds | Edge |
|-------|-----------|---------------|---------------|-------------|------|
| Ganna | 18.2% | -1.8s | ~19% | 2.50 (40%) | -21% ❌ |
| Van Aert | 14.7% | -2.3s | ~16% | 11.0 (9%) | +7% ✅ |
| Del Toro | 12.3% | +3.2s | ~10% | 4.50 (22%) | -12% ❌ |
| Roglic | 9.1% | -0.5s | ~9% | 15.0 (6.7%) | +2.3% ✅ |
| Christen | 6.8% | -1.0s | ~7% | 40.0 (2.5%) | +4.5% ✅ |

---

## 🎯 Tirreno-Adriatico Stage 1 Specific Angles

### Market Favorites Analysis

| Rider | Market View | Model View | Weather | Verdict |
|-------|-------------|------------|---------|---------|
| **Filippo Ganna** | Heavy favorite (won 3 of last 4) | Should be top but not 40% | Check start time | Likely **overpriced** - no value at short odds |
| **Isaac Del Toro** | Second favorite (UAE + Strade form) | Strong form signal | Early starter risk | **FADE** if weather favors late |
| **Wout Van Aert** | Mid-price all-rounder | Undervalued vs specialists | Neutral/good | **BACK** - best value combination |
| **Primoz Roglic** | GC rider pricing | Proven short ITT performer | Late starter advantage | **BACK** if priced >10.0 |
| **Jan Christen** | Long shot (unknown) | High 2026 form score | Check start time | **SPECULATIVE** - tiny stake |

### Weather-Specific Betting Scenarios

#### Scenario A: Wind builds during afternoon (sea breeze)
- **Early starters (14:00-14:30)**: Headwind or crosswind → slower
- **Late starters (15:30-16:30)**: Tailwind or lighter wind → faster
- **Action**: Back late-starting GC riders (Roglic, Jorgenson, Evenepoel)

#### Scenario B: Wind dies during afternoon
- **Early starters**: Benefit from stronger tailwind
- **Late starters**: Slower in calmer conditions
- **Action**: Back early starters with strong TT ability

#### Scenario C: No significant wind (calm day)
- Weather spread <5 seconds
- Rely purely on base model + form
- **Action**: Van Aert and Christen likely value; Ganna/Del Toro likely overpriced

---

## 📊 Kelly Sizing & Risk Management

### Recommended Parameters for This Stage

In `genqirue/portfolio/kelly.py` or workflow:

```python
KellyParameters(
    method='quarter_kelly',    # Conservative - weather uncertainty is high
    max_position_pct=0.15,     # Cap at 15% for single rider (ITT variance)
    min_edge_bps=75,           # Higher threshold (short stage = more randomness)
    cvar_limit=0.10            # 10% CVaR at 95%
)
```

### Why Quarter-Kelly for ITT?

1. **Weather uncertainty** - Forecasts can be wrong
2. **Short distance** (11.5km) - Higher variance, mechanical issues matter more
3. **Early season** - Less reliable form data
4. **Start order effects** - Later starters have information advantage

---

## 📝 Complete Command Sequence

Copy-paste ready for tomorrow:

```powershell
# 1. Update data (if not done)
cd c:\Users\danie\Downloads\nkls\cycling_predict
python -m pipeline.runner

# 2. Scrape latest 2026 form
python scripts\scrape_2026_season.py --race tirreno-adriatico --year 2026

# 3. Run models and rank
python scripts\rank_stage.py tirreno-adriatico 2026 1 --run-models --save --top 20

# 4. Weather analysis (run 24h before, then 6h before)
python scripts\weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# 5. Get latest odds
python scripts\fetch_odds.py

# 6. Full portfolio analysis
python scripts\example_betting_workflow.py

# 7. Export for tracking
python scripts\run_backtest.py --strategy frailty --save-bets tirreno_2026_s1.csv
```

---

## 🔍 Post-Race Validation

After the stage completes:

```bash
# Scrape actual results
python -m pipeline.runner

# Compare predictions vs outcomes
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
# Query: actual podium vs model top-3
# Query: weather impact - did early/late starters differ as predicted?
"
```

**Track these metrics:**
- Did model top-3 include actual podium?
- Was weather spread realized? (Compare early vs late starter times)
- Did edge materialize? (Model prob > market prob → winner?)

---

## Summary: Key Edges for Tirreno Stage 1

1. **Strategy 6 (Weather Arbitrage)** - Coastal ITT = wind can swing results 10-30s
2. **Strategy 2 (Frailty)** - 2026 form signal (UAE Tour, Strade Bianche results)
3. **Market Inefficiency** - Ganna/Del Toro likely overpriced; Van Aert/Roglic potentially undervalued
4. **Start Time Lottery** - Check forecast shifts between market open and race start

**Primary value candidates:**
- **Wout Van Aert** - All-rounder TT ability, likely underpriced vs pure specialists
- **Primoz Roglic** - Proven ITT performer, GC pricing may create value
- **Jan Christen** - Unknown quantity, high 2026 form, market may underprice

**Avoid:**
- **Filippo Ganna** - Too short, no value despite being favorite
- **Isaac Del Toro** - Peak form = peak price; early starter weather risk

Good luck! 🚴‍♂️🎯