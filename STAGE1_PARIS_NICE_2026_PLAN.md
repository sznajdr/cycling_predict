# Stage 1 Paris-Nice 2026 - Value Bet Finding Plan

## Race Overview

| Attribute | Details |
|-----------|---------|
| **Stage** | 1 of 8 |
| **Route** | Achères → Carrières-Sous-Poissy |
| **Profile** | Flat/slightly hilly with demanding finale |
| **Key Feature** | Côte de Chanteloup-les-Vignes (1.1km @ 8.3%) - 50km from finish |
| **Stage Type** | Hilly/Sprinter-unfriendly |
| **Date** | Today |

### Stage Profile Analysis

The stage looks flat on paper but has a **tricky, demanding last 50km** with the Côte de Chanteloup-les-Vignes (1.1km at 8.3%). This climb is hard enough to drop pure sprinters but not selective enough for pure climbers. 

**Winner Profile:** Punchy sprinter, puncheur, or strong classics rider who can survive the climb and win from a reduced bunch sprint or late attack.

---

## Step-by-Step Value Finding Process

### Step 1: Check Data Availability

```powershell
# Check if Paris-Nice 2026 data is in your database
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
cursor = conn.cursor()

# Check race
race = cursor.execute('SELECT id, display_name FROM races WHERE pcs_slug = ? AND year = ?', 
                      ('paris-nice', 2026)).fetchone()
if race:
    print(f'✓ Race found: {race[1]} (ID: {race[0]})')
    
    # Check startlist
    startlist = cursor.execute('SELECT COUNT(*) FROM startlist_entries WHERE race_id = ?', 
                               (race[0],)).fetchone()
    print(f'✓ Startlist: {startlist[0]} riders')
    
    # Check Stage 1
    stage = cursor.execute('SELECT id, stage_number, stage_type FROM race_stages 
                           WHERE race_id = ? AND stage_number = ?', 
                          (race[0], 1)).fetchone()
    if stage:
        print(f'✓ Stage 1 found: Type = {stage[2]}')
    else:
        print('✗ Stage 1 not found')
else:
    print('✗ Paris-Nice 2026 not found - run scraper first')
conn.close()
"
```

**If data is missing:**
```powershell
# Add 2026 to config and scrape
python -m pipeline.runner
```

---

### Step 2: Fetch Live Odds

```powershell
# Scrape Betclic odds for Paris-Nice Stage 1
python fetch_odds.py --dry-run

# If it finds the event, run full scrape
python fetch_odds.py
```

**Verify odds are stored:**
```sql
SELECT participant_name, back_odds, fair_odds
FROM bookmaker_odds_latest
WHERE market_type = 'winner'
ORDER BY back_odds
LIMIT 20;
```

---

### Step 3: Run Pre-Race Analysis

#### Option A: Quick Analysis (5 minutes)
```powershell
python quickstart.py
```

This will:
- Load historical Paris-Nice data
- Fit Strategy 2 (Gruppetto Frailty)
- Fit Strategy 1 (Tactical HMM)
- Show riders with hidden form

#### Option B: Full Stage Ranking (Recommended)
```powershell
# Fit models first
python rank_stage.py paris-nice 2026 1 --run-models

# View rankings
python rank_stage.py paris-nice 2026 1 --top 30

# Save to database
python rank_stage.py paris-nice 2026 1 --save
```

---

### Step 4: Identify Key Riders

Based on the preview article and stage profile, focus on these rider types:

#### Tier 1: Favorites (Short odds, look for each-way value)
| Rider | Why | Model Signal to Check |
|-------|-----|----------------------|
| **Juan Ayuso** | Top favorite, won Algarve, good form | Tactical state (PRESERVING?) |
| **Oscar Onley** | 4th at Algarve, punchy, stage winner potential | Frailty score, specialty match |
| **Ethan Hayter** | Strong classics rider, reduced sprint | Hidden form from gruppetto? |

#### Tier 2: Outsiders (Medium odds, value potential)
| Rider | Why | Model Signal to Check |
|-------|-----|----------------------|
| **Marc Soler** | Won PN 2018, motivated, punchy | Frailty, tactical state |
| **Lenny Martinez** | Won stage last year, aggressive | Attack signal (Strategy 12) |
| **Ewen Costiou** | Team leader, punchy climber | GC time loss pattern |

#### Tier 3: Longshots (Big odds, lottery tickets)
| Rider | Why | Model Signal to Check |
|-------|-----|----------------------|
| **Mads Pedersen** | Strong classics, can survive climb | Specialty scores |
| **Matteo Trentin** | Experienced, reduced bunch sprints | Historical PN results |
| **Benoît Cosnefroy** | French, aggressive, home pressure | Combativity awards |

---

### Step 5: Check Model Signals

#### Strategy 2: Gruppetto Frailty (Hidden Form)
```sql
-- Riders with high frailty = potential hidden form
SELECT r.name, rf.frailty_estimate, rf.hidden_form_prob
FROM rider_frailty rf
JOIN riders r ON rf.rider_id = r.id
WHERE rf.hidden_form_prob > 0.3
ORDER BY rf.hidden_form_prob DESC
LIMIT 15;
```

**What to look for:**
- Riders who were in gruppetto on recent mountain stages but didn't lose excessive time
- High frailty score suggests tactical conservation, not weakness
- These riders often overperform on transition/hilly stages

#### Strategy 1: Tactical Time Loss (HMM)
```sql
-- Riders in PRESERVING state (saving energy)
SELECT r.name, ts.preserving_prob, ts.decoded_state
FROM tactical_states ts
JOIN riders r ON ts.rider_id = r.id
JOIN race_stages rs ON ts.stage_id = rs.id
JOIN races ra ON rs.race_id = ra.id
WHERE ra.pcs_slug = 'paris-nice' 
  AND ra.year IN (2024, 2025)
  AND ts.preserving_prob > 0.6
ORDER BY ts.preserving_prob DESC
LIMIT 15;
```

**What to look for:**
- Riders who tactically lost time on previous stages
- PRESERVING state = conserving energy for later
- Stage 1 is perfect for these riders to "wake up"

#### Strategy 6: Weather (if ITT - not applicable here)
Skip for Stage 1 (not a time trial)

---

### Step 6: Historical Performance Query

```sql
-- Riders with good Paris-Nice history
SELECT r.name, 
       COUNT(*) as pn_stages,
       AVG(CAST(rr.rank AS FLOAT)) as avg_rank,
       SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 3 THEN 1 ELSE 0 END) as podiums
FROM rider_results rr
JOIN race_stages rs ON rr.stage_id = rs.id
JOIN riders r ON rr.rider_id = r.id
JOIN races ra ON rs.race_id = ra.id
WHERE ra.pcs_slug = 'paris-nice'
  AND rr.result_category = 'stage'
  AND rs.stage_type IN ('hilly', 'flat', 'road')
GROUP BY rr.rider_id
HAVING pn_stages >= 3
ORDER BY podiums DESC, avg_rank ASC
LIMIT 15;
```

---

### Step 7: Calculate Value Bets

After running `example_betting_workflow.py` or `rank_stage.py`, check for edge:

```sql
-- Top value opportunities (edge > 50 bps)
SELECT rider_name, 
       ROUND(model_prob * 100, 2) as model_pct,
       ROUND((1.0/back_odds) * 100, 2) as market_pct,
       ROUND(edge_bps, 0) as edge_bps,
       ROUND(kelly_pct * 100, 2) as kelly_stake
FROM strategy_outputs
WHERE strategy_name = 'stage_ranking'
  AND edge_bps > 50
ORDER BY edge_bps DESC
LIMIT 10;
```

---

## Expected Value Bet Scenarios

### Scenario A: Reduced Bunch Sprint (Most Likely - 60%)
The climb drops 30-40% of the peloton, leaving ~80-100 riders. A punchy sprinter wins.

**Potential value:**
- Mads Pedersen @ 12-15x (if market overprices pure sprinters)
- Oscar Onley @ 8-10x (if market hasn't adjusted to his Algarve form)

### Scenario B: Late Attack (25%)
A strong puncheur attacks on the climb or just after, holds off reduced peloton.

**Potential value:**
- Marc Soler @ 20-25x (experienced, knows the race)
- Lenny Martinez @ 15-18x (aggressive, won stage last year)

### Scenario C: Breakaway Survival (15%)
Strong break goes early, climb is hard enough to prevent complete regrouping.

**Potential value:**
- Look for riders with high frailty (Strategy 2) who are not GC threats
- Team workers with freedom

---

## Recommended Bets (Template)

| Rider | Book Odds | Model Prob | Market Prob | Edge | Kelly Stake | Notes |
|-------|-----------|------------|-------------|------|-------------|-------|
| ? | ? | ?% | ?% | ? bps | ?% | Fill after running models |
| ? | ? | ?% | ?% | ? bps | ?% | Fill after running models |
| ? | ? | ?% | ?% | ? bps | ?% | Fill after running models |

**Risk Management:**
- Max 3-4 riders (diversification)
- Quarter-Kelly stakes (conservative)
- Total exposure < 10% of bankroll for Stage 1

---

## Quick Execution Checklist

```bash
# 1. Verify data
python -c "import sqlite3; conn = sqlite3.connect('data/cycling.db'); print('Races:', conn.execute('SELECT COUNT(*) FROM races WHERE pcs_slug=\"paris-nice\"').fetchone()[0]); print('Riders:', conn.execute('SELECT COUNT(*) FROM riders').fetchone()[0]); conn.close()"

# 2. Fetch odds
python fetch_odds.py

# 3. Run analysis
python rank_stage.py paris-nice 2026 1 --run-models --save

# 4. Check output
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print('TOP 10 MODEL RANKINGS:')
for row in conn.execute('SELECT rank_position, rider_name, model_prob, back_odds, edge_bps FROM strategy_outputs WHERE strategy_name=\"stage_ranking\" ORDER BY CAST(rank_position AS INTEGER) LIMIT 10'):
    print(f'{row[0]:2}. {row[1]:25} Prob: {row[2]:.2%} Odds: {row[3]:6.2f} Edge: {row[4]:5.0f}bps')
conn.close()
"

# 5. Place bets based on output
```

---

## Key Insights from Preview

1. **Ayuso is favorite** but market may overprice him for a tricky Stage 1
2. **Onley** is in great form (4th Algarve) - model may rate him higher than market
3. **Soler** is experienced but market may undervalue his 2018 win
4. **Stage is selective** - pure sprinters will struggle (look for riders with high `sp_sprint` but also decent `sp_climber`)

---

## Model Weights for Stage 1

Edit in `genqirue/portfolio/kelly.py` or ranking config:

```python
# For hilly Stage 1, emphasize:
weights = {
    'specialty': 0.30,      # Important - need right profile
    'historical': 0.20,     # Some weight to PN history
    'frailty': 0.25,        # KEY - hidden form from gruppetto
    'tactical': 0.15,       # Less important Stage 1
    'gc_relevance': 0.10    # Less important Stage 1
}
```

---

## Post-Race Review

After Stage 1, log:
1. Which model signals worked?
2. Which riders were flagged but didn't perform?
3. Were there any news/events models missed?
4. Update model confidence based on results

This feedback loop improves predictions for Stages 2-8.
