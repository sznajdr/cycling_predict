# Example: Finding Value Bets for Tirreno-Adriatico 2026 Stage 1

This guide walks through using the Genqirue betting engine to find value bets for the opening Individual Time Trial (ITT) of Tirreno-Adriatico 2026.

---

## Race Overview

| Attribute | Details |
|-----------|---------|
| **Race** | Tirreno-Adriatico 2026 ("Race of the Two Seas") |
| **Stage 1** | Lido Di Camaiore → Lido Di Camaiore |
| **Date** | Monday, March 9, 2026 |
| **Distance** | 11.5 km |
| **Type** | Individual Time Trial (ITT) |
| **Profile** | Flat, coastal - exposed to wind |

### Key Favorites (Market View)

| Stars | Riders |
|-------|--------|
| ⭐⭐⭐⭐⭐ | Isaac Del Toro (UAE) - Won UAE Tour 2026, 2nd at Strade Bianche |
| ⭐⭐⭐⭐ | Roglic (Red Bull), Tiberi (Bahrain), Jorgenson (Visma) |
| ⭐⭐⭐ | Pellizzari, Hindley, Jan Christen, Buitrago |
| ⭐⭐ | Van Aert, Van der Poel, Ganna, Carapaz, Bernal |

---

## Prerequisites

Ensure your environment is set up:

```bash
# Verify procyclingstats is installed
pip install -e ../procyclingstats

# Check database exists and schema is applied
python fetch_odds.py --init-schema
```

---

## Step 1: Configure the Race

Add Tirreno-Adriatico to your race configuration (`config/races_2026_early.yaml`):

```yaml
year: 2026

races:
  # ... other races ...

  # March - Tirreno-Adriatico
  - name: Tirreno-Adriatico
    pcs_slug: tirreno-adriatico
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]
```

---

## Step 2: Scrape Historical Data

Run the pipeline to collect historical results and rider profiles:

```bash
# Start the scraper
python -m pipeline.runner

# Monitor progress in a second terminal
python monitor.py
```

**Expected output:**
```
=== Queue status ===
  race_meta            pending         5
  race_startlist       pending         5
  stage_results        pending         35
  rider_profile        pending         200
```

**Note:** One race with 4 years of history takes ~20-60 minutes. The queue is resume-safe.

---

## Step 3: Discover 2026 Season Form

Since this is an early-season race, scrape recent 2026 results for Tirreno riders:

```bash
# Find all 2026 races for Tirreno-Adriatico riders
python scripts/scrape_2026_season.py --race tirreno-adriatico --year 2026

# Optional: Limit to top riders for faster scraping
python scripts/scrape_2026_season.py --race tirreno-adriatico --year 2026 --rider-limit 30
```

**This adds races like:**
- UAE Tour (Del Toro won, strong form signal)
- Strade Bianche (recent result)
- Omloop Het Nieuwsblad
- Volta ao Algarve

---

## Step 4: Fit Models and Generate Rankings

### Option A: Quick Rank (uses existing model data)

```bash
python rank_stage.py tirreno-adriatico 2026 1 --top 20
```

### Option B: Full Model Fit (recommended for Stage 1)

```bash
# Fit frailty + tactical models, then rank
python rank_stage.py tirreno-adriatico 2026 1 --run-models --save --top 20
```

**What this does:**
1. Loads historical data from 2022-2025
2. Fits **Strategy 1 (Tactical HMM)** on 4421+ observations
3. Fits **Strategy 2 (Gruppetto Frailty)** using Cox PH survival analysis
4. Computes 5 signals per rider:
   - Specialty (ITT score from PCS)
   - Historical (past Tirreno Stage 1 results)
   - Form (2026 season results)
   - Frailty (sandbagging detection)
   - GC Relevance
5. Runs softmax to get win probabilities
6. Saves to `strategy_outputs` table

**Sample Output:**
```
Tirreno-Adriatico 2026 Stage 1 - ITT (11.5km)
Signals: specialty(0.40) historical(0.25) form(0.15) frailty(0.15) gc_relevance(0.05)
Field: 176 riders | Temperature: 12.3 | Edge threshold: 50bps

 Rank  Rider                  Spec  Hist   Form  ModelProb
---------------------------------------------------------
    1  GANNA Filippo          0.98  0.92   0.85     18.2%
    2  VAN AERT Wout          0.85  0.88   0.91     14.7%
    3  DEL TORO Isaac         0.72  0.65   0.97     12.3%
    4  ROGLIC Primoz          0.78  0.90   0.82      9.1%
    5  CHRISTEN Jan           0.68  0.45   0.94      6.8%
```

---

## Step 5: Scrape Live Odds

Get current Betclic odds to compare against model probabilities:

```bash
# Full odds scrape
python fetch_odds.py

# Or dry-run first to check markets
python fetch_odds.py --dry-run
```

**Verify odds loaded:**
```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT participant_name, back_odds, fair_odds 
    FROM bookmaker_odds_latest 
    WHERE market_type = 'winner' 
    ORDER BY back_odds 
    LIMIT 10
\"\"\")
for row in cursor.fetchall():
    print(f'{row[0]:<25} {row[1]:>6.2f}  {row[2]:>6.2f}')
"
```

---

## Step 6: Find Value Bets (Edge Calculation)

Run the full betting workflow to identify +EV opportunities:

```bash
python example_betting_workflow.py
```

**Output format:**
```
Top Opportunities (edge > 50bps):
Rider               Model Prob   Market Prob   Edge (bps)   Kelly Stake
-----------------------------------------------------------------------
Van Aert            0.147        0.089         580          4.2%
Christen            0.068        0.025        4300          6.8%
Jorgenson           0.054        0.032        2200          3.1%
Roglic              0.091        0.065        2600          2.4%
```

### Understanding the Output

| Column | Meaning |
|--------|---------|
| **Model Prob** | Genqirue's estimated win probability (0-1) |
| **Market Prob** | Implied probability from bookmaker odds (1/odds) |
| **Edge (bps)** | Basis points of edge (100 bps = 1%) |
| **Kelly Stake** | Recommended bankroll percentage (quarter-Kelly) |

**Bet when:**
- Edge > 50 bps (default threshold)
- Model uncertainty (σ_p) is low
- CVaR constraint allows the position

---

## Strategy-Specific Edges for Stage 1

### Strategy 6: ITT Weather Arbitrage ⭐⭐⭐⭐⭐

**Critical for this stage!** Lido di Camaiore is coastal - wind conditions can swing results by 10-30 seconds.

```python
# Pseudocode for weather model
from genqirue.models.weather_spde import ITTWeatherModel

model = ITTWeatherModel()
# Input: wind forecast updates after market opens
# Output: time delta between early/late starters
```

**Betting angle:**
- If wind forecast shifts to favor late starters → back late-starting GC riders
- Early starters into headwind → underpriced if market hasn't adjusted

### Strategy 2: Frailty (Form Detection)

Riders with strong 2026 results but not market favorites:

| Rider | 2026 Form | Market Price | Edge |
|-------|-----------|--------------|------|
| Del Toro | Won UAE Tour, 2nd Strade | Short | Likely overpriced |
| Christen | Unknown/new | Long | Potential value |
| Van Aert | ??? | Medium | Value if form is good |

### Strategy 1: Tactical State

Less relevant for Stage 1 ITT (no previous stage), but useful for Stage 2+.

---

## Manual Override Considerations

Per the [ONBOARDING.md](ONBOARDING.md) protocol, consider manual adjustments:

### Override UP (increase probability)
- **Van Aert:** If he raced Strade Bianche strongly (form confirmation)
- **Roglic:** Proven short ITT performer, especially early season
- **Ganna:** But likely no value - market prices him too short

### Override DOWN (decrease probability)
- **Del Toro:** Peak form may mean peak market price - no value despite ability
- Riders who crashed at Strade Bianche (check medical communiqués)

### Exclude entirely
- Any rider with reported illness/mechanical issues pre-race

---

## Kelly Sizing Parameters

Edit `genqirue/portfolio/kelly.py` for this stage:

```python
KellyParameters(
    method='quarter_kelly',    # Conservative default
    max_position_pct=0.20,     # Cap at 20% for single rider
    min_edge_bps=50,           # Minimum 0.5% edge
    cvar_limit=0.10            # 10% CVaR at 95%
)
```

**Why quarter-Kelly for ITT?**
- Weather uncertainty is high
- Short distance = higher variance
- Early season = less form data

---

## Post-Race Validation

After the stage completes, update PCS data and validate:

```bash
# Scrape results
python -m pipeline.runner

# Run backtest to see how this stage performed
python run_backtest.py --strategy all --save-bets tirreno_2026_s1.csv
```

**Key metrics to track:**
- Did model top-3 include actual podium?
- Was edge realized (model prob > market prob)?
- Weather impact: Did Strategy 6 capture wind effects?

---

## Quick Reference Commands

```bash
# Full workflow (copy-paste ready)
cd /path/to/cycling_predict

# 1. Setup
python fetch_odds.py --init-schema

# 2. Scrape
python -m pipeline.runner
python monitor.py

# 3. 2026 season data
python scripts/scrape_2026_season.py --race tirreno-adriatico --year 2026

# 4. Rank Stage 1
python rank_stage.py tirreno-adriatico 2026 1 --run-models --save

# 5. Get odds
python fetch_odds.py

# 6. Portfolio analysis
python example_betting_workflow.py

# 7. Backtest
python run_backtest.py --strategy frailty --kelly 0.25 --save-bets results.csv
```

---

## Expected Value Bet Candidates

Based on market dynamics and model signals:

### Primary Targets (Likely Value)
| Rider | Rationale |
|-------|-----------|
| **Wout Van Aert** | All-rounder, strong short ITT, market may undervalue vs pure specialists |
| **Matteo Jorgenson** | American GC rider, less market attention, strong TT ability |
| **Jan Christen** | Young talent, unknown to casual market, UAE form indicator |

### Avoid (Likely Overpriced)
| Rider | Rationale |
|-------|-----------|
| **Filippo Ganna** | Pure ITT specialist, will be shortest odds, zero value |
| **Isaac Del Toro** | Hot form = favorite price, market efficient here |

### Watch List
| Rider | Rationale |
|-------|-----------|
| **Primoz Roglic** | GC rider pricing, check if stage win odds > overall odds |
| **Mathieu van der Poel** | Can surprise, check if priced for sprints not TT |

---

## Risk Factors

1. **Weather:** Coastal wind can reverse predictions (Strategy 6 handles this)
2. **Short Distance:** 11.5km = higher variance, luck plays bigger role
3. **Early Season:** Limited 2026 data for form signals
4. **Start Order:** Later starters know times to beat = information advantage

---

## Additional Resources

- [ONBOARDING.md](ONBOARDING.md) - Full system overview and theory
- [COMMANDS.md](COMMANDS.md) - Complete CLI reference
- [docs/RANKING.md](docs/RANKING.md) - Stage ranking model documentation
- [docs/ENGINE.md](docs/ENGINE.md) - Strategy implementation details

---

## Summary

**For Tirreno-Adriatico Stage 1 ITT:**

1. ✅ Scrape historical data (4 years)
2. ✅ Add 2026 season races for form
3. ✅ Run models with `--run-models` flag
4. ✅ Check weather forecasts for Strategy 6 edge
5. ✅ Compare model vs market odds
6. ✅ Size bets with quarter-Kelly
7. ✅ Log all positions for post-race review

**Good luck! 🚴‍♂️**
