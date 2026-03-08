# 2026 Season Data Plan

## The Problem

Your models ran but show poor data quality:
- **0/154 riders** have 2026 results
- Specialty scores show **"—"** (missing)
- Model using defaults: **0.50*** (no recent form)
- Historical weight dominates (2022-2025)

## Why This Happens

The 2026 season just started. Riders have form from:
- ✅ January: Tour Down Under, Australia races
- ✅ February: UAE Tour, Algarve, Omloop, Strade Bianche
- ⏳ March: Paris-Nice (happening now)

**We scraped Paris-Nice only** - we need earlier 2026 races!

---

## Solution Options

### Option A: Quick Fix - Add Key Races (Recommended)

Scrape the most important pre-Paris-Nice 2026 races:

```powershell
# Add to config/races.yaml temporarily:

- name: Volta ao Algarve
  pcs_slug: volta-ao-algarve
  type: stage_race
  history_years: [2026]

- name: Omloop Het Nieuwsblad
  pcs_slug: omloop-het-nieuwsblad
  type: one_day
  history_years: [2026]

- name: Strade Bianche
  pcs_slug: strade-bianche
  type: one_day
  history_years: [2026]
```

Then run:
```powershell
python -m pipeline.runner
```

**Pros:** Fast, targeted
**Cons:** Manual work, might miss some riders

---

### Option B: Automated - Scrape Per Rider

Use the new script to find and scrape all 2026 races for Paris-Nice riders:

```powershell
python scrape_2026_season.py --race paris-nice --year 2026 --max-races 30
```

This will:
1. Check each Paris-Nice rider's 2026 race history
2. Find ~20-30 unique races
3. Scrape any missing ones
4. Build up 2026 form database

**Pros:** Comprehensive, automated
**Cons:** Takes longer, might scrape races you don't care about

---

### Option C: Manual Rider Results

For each key rider, scrape their individual results:

```powershell
# In Python:
from procyclingstats import RiderResults

rider = RiderResults("rider/juan-ayuso-pesquera")
results = rider.results()  # All 2026 races
```

Already done by the pipeline! The `rider_results` job type does this.

**Check if it ran:**
```sql
SELECT r.name, COUNT(rr.id) as results_2026
FROM riders r
JOIN rider_results rr ON r.id = rr.rider_id
JOIN races ra ON rr.race_id = ra.id
WHERE ra.year = 2026
GROUP BY r.id
ORDER BY results_2026 DESC
LIMIT 10;
```

---

## Step-by-Step for Right Now

### Step 1: Check What 2026 Data Exists

```powershell
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')

# Races in 2026
races = conn.execute('SELECT display_name, year FROM races WHERE year = 2026').fetchall()
print('2026 RACES IN DATABASE:')
for r in races:
    print(f'  {r[0]}')

# Rider results count
results = conn.execute('''
    SELECT COUNT(*) 
    FROM rider_results rr
    JOIN races r ON rr.race_id = r.id
    WHERE r.year = 2026
''').fetchone()[0]
print(f'\nTotal 2026 rider results: {results}')

conn.close()
"
```

### Step 2: Add Key Early 2026 Races

Edit `config/races.yaml`:

```yaml
year: 2026

races:
  # Add these BEFORE Paris-Nice:
  
  - name: Volta ao Algarve
    pcs_slug: volta-ao-algarve
    type: stage_race
    history_years: [2026]

  - name: Omloop Het Nieuwsblad
    pcs_slug: omloop-het-nieuwsblad
    type: one_day
    history_years: [2026]

  - name: Strade Bianche
    pcs_slug: strade-bianche
    type: one_day
    history_years: [2026]

  - name: UAE Tour
    pcs_slug: uae-tour
    type: stage_race
    history_years: [2026]

  - name: Tour Down Under
    pcs_slug: tour-down-under
    type: stage_race
    history_years: [2026]

  # Your existing Paris-Nice
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025, 2026]
```

### Step 3: Scrape New Races

```powershell
# This will add the new races to the job queue
python -m pipeline.runner

# Monitor progress
python monitor.py
```

### Step 4: Verify 2026 Data

```powershell
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')

# Count 2026 results per rider in Paris-Nice
race = conn.execute('SELECT id FROM races WHERE pcs_slug=? AND year=?', 
                    ('paris-nice', 2026)).fetchone()

riders = conn.execute('''
    SELECT r.name, COUNT(rr.id) as results_2026
    FROM startlist_entries sl
    JOIN riders r ON sl.rider_id = r.id
    LEFT JOIN rider_results rr ON r.id = rr.rider_id
    LEFT JOIN races ra ON rr.race_id = ra.id AND ra.year = 2026
    WHERE sl.race_id = ?
    GROUP BY r.id
    ORDER BY results_2026 DESC
    LIMIT 20
''', (race[0],)).fetchall()

print('TOP 20 RIDERS BY 2026 RESULTS:')
for r in riders:
    print(f'  {r[0]:<30} {r[1]:>3} races')

conn.close()
"
```

### Step 5: Re-run Models

```powershell
# Now with 2026 data!
python rank_stage.py paris-nice 2026 1 --run-models --save
```

---

## Expected Outcome

**Before (what you saw):**
```
Rank  Rider              Spec   Hist  ModelProb
------------------------------------------------
1     ZINGLE Axel         —     0.97     17.0%    <- No 2026 data, using historical
2     Milan Menten       1.00   0.50*     5.7%    <- * = no race history this year
...
```

**After (with 2026 data):**
```
Rank  Rider              Spec   Hist  ModelProb  Form2026
---------------------------------------------------------
1     Juan Ayuso         0.85   0.75     22.0%   Algarve winner, UAE 2nd
2     Oscar Onley        0.82   0.70     18.0%   Algarve 4th, strong ITT
3     Kasper Asgreen     0.78   0.65     12.0%   Omloop winner
...
```

---

## Quick Alternative: Use Historical Data Only

If scraping takes too long, you can run the models with **historical data only** (2022-2025) but weight it differently:

Edit `rank_stage.py` weights:
```python
weights = {
    'specialty': 0.40,      # Increase (from 0.30)
    'historical': 0.50,     # High weight for multi-year history
    'frailty': 0.10,        # Decrease (no recent gruppetto data)
}
```

This is less accurate but usable today.

---

## Recommendation

**For Paris-Nice Stage 1 (today):**

1. **Quick:** Add Algarve, Omloop, Strade Bianche to config (5 min)
2. **Scrape:** Run pipeline while you do other prep (30 min)
3. **Re-run:** Models with fresh 2026 data
4. **Bet:** Based on actual 2026 form

**Timeline:**
- 12:30 - Add races, start scraper
- 13:00 - Check data, re-run models  
- 13:30 - Final predictions ready
- 15:00 - Race starts

---

## Key Races to Prioritize

| Race | PCS Slug | Importance | Riders Covered |
|------|----------|------------|----------------|
| Volta ao Algarve | volta-ao-algarve | ⭐⭐⭐ HIGH | Ayuso, Onley, many PN riders |
| Omloop Het Nieuwsblad | omloop-het-nieuwsblad | ⭐⭐⭐ HIGH | Classics specialists |
| Strade Bianche | strade-bianche | ⭐⭐⭐ HIGH | Puncheurs, climbers |
| UAE Tour | uae-tour | ⭐⭐ Medium | Early season climbers |
| Tour Down Under | tour-down-under | ⭐⭐ Medium | Aussies, some Europeans |

Start with **Algarve + Omloop + Strade Bianche** - these cover most Paris-Nice contenders!
