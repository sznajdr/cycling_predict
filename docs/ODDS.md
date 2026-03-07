# Betclic Odds Scraper — Step-by-Step Guide

Scrapes live cycling odds from Betclic, stores them in SQLite, and feeds them into the betting workflow so Kelly sizing uses real market prices instead of simulated ones.

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Apply the Schema](#3-step-1--apply-the-schema)
4. [Step 2 — Dry-Run a Single Event](#4-step-2--dry-run-a-single-event)
5. [Step 3 — Full Hub Scrape](#5-step-3--full-hub-scrape)
6. [Step 4 — Verify Rows in the DB](#6-step-4--verify-rows-in-the-db)
7. [Step 5 — Run the Betting Workflow](#7-step-5--run-the-betting-workflow)
8. [Scheduled / Repeated Scraping](#8-scheduled--repeated-scraping)
9. [Understanding the Data](#9-understanding-the-data)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. How It Works

Three steps happen every time you run the scraper:

**Step A — Hub discovery.** GET `https://www.betclic.fr/cyclisme-scycling`. Extract every event URL matching the pattern `/cyclisme-scycling/.../...-mXXXXXXXXX`. Each distinct `mXXX` suffix is one event (race or stage market).

**Step B — Event scraping.** GET each event page. The page embeds JSON-like market data in raw HTML. Two regex patterns extract it:

- Primary: `"name":"<selection>",...,"odds":<decimal>`
- Fallback: `"name":"<selection>",...,"price":<decimal>`

Selections are grouped by their enclosing market `"label"` field. A market label is something like `"Vainqueur"`, `"Top 3"`, or `"Duel — Pogacar vs Vingegaard"`.

**Step C — Fair odds computation.** Within each market group:

```
implied_prob  = 1.0 / back_odds
overround     = sum(implied_prob for all selections in market)
fair_prob     = implied_prob / overround
fair_odds     = 1.0 / fair_prob
```

`fair_odds` is the hold-adjusted true price — what the odds would be if the bookmaker took zero margin. `back_odds - fair_odds` represents the margin you need to overcome to have positive expectation.

---

## 2. Prerequisites

- Database created by `python -m pipeline.runner` (only the `data/cycling.db` file needs to exist; it can be empty)
- If the DB doesn't exist yet, `fetch_odds.py` will create it automatically and apply the schema

No additional packages required — the scraper uses only the Python standard library (`urllib`, `re`, `sqlite3`, `unicodedata`).

---

## 3. Step 1 — Apply the Schema

This creates the `bookmaker_odds` table and the `bookmaker_odds_latest` view. Safe to run repeatedly — every statement uses `IF NOT EXISTS`.

```bash
python fetch_odds.py --init-schema
```

Expected output:

```
Schema applied (bookmaker_odds table + view + indexes).
```

What was created in the DB:

| Object | Purpose |
|--------|---------|
| `bookmaker_odds` | Raw snapshot table — one row per selection per scrape run |
| `bookmaker_odds_latest` | View that returns only the most recent snapshot per (event, market, selection) |
| `idx_bookmaker_odds_event` | Index on `(event_id, market_type, scraped_at)` |
| `idx_bookmaker_odds_participant` | Index on `(participant_name, market_type, scraped_at)` |

---

## 4. Step 2 — Dry-Run a Single Event

Before writing anything to the DB, verify the scraper can extract odds from a known event page.

```bash
python fetch_odds.py --dry-run --event-url https://www.betclic.fr/cyclisme-scycling/paris-nice-c5649/paris-nice-2026-m1052180106760192
```

Expected output (columns truncated for readability):

```
DRY RUN — 87 selections from https://www.betclic.fr/...

  participant_name        market_type  back_odds  fair_odds  implied_prob  fair_prob  market_label_raw
  ----------------------  -----------  ---------  ---------  ------------  ---------  ----------------
  Tadej Pogacar           winner       1.85       1.72       0.541         0.581      Vainqueur
  Jonas Vingegaard        winner       4.50       4.19       0.222         0.239      Vainqueur
  Remco Evenepoel         winner       7.00       6.51       0.143         0.154      Vainqueur
  ...
  Pogacar vs Vingegaard   h2h          1.30       1.23       0.769         0.814      Duel
  Vingegaard vs Pogacar   h2h          3.50       3.31       0.286         0.302      Duel
  ...

  87 selections
```

If `(no rows)` is printed, the event URL has changed or Betclic has moved the market. Try a fresh URL from the hub:

```bash
# See what events are currently live
python fetch_odds.py --dry-run
```

That runs the full hub discovery dry-run and prints every selection from every live event.

---

## 5. Step 3 — Full Hub Scrape

```bash
python fetch_odds.py
```

This:
1. GETs the cycling hub and collects all event URLs
2. GETs each event page and extracts odds
3. Writes every selection to `bookmaker_odds` (duplicates skipped via `INSERT OR IGNORE`)
4. Prints a summary line

Expected output:

```
09:14:32 INFO pipeline.betclic_scraper: Found 12 event URLs on hub
09:14:45 INFO pipeline.betclic_scraper: Scrape complete: run_id=a3f1..., events=12, rows_attempted=634, inserted=634
Done. 634 rows inserted.
```

Each run gets a UUID `scrape_run_id` — every row from one scrape shares the same ID. This lets you reconstruct any historical snapshot: `SELECT * FROM bookmaker_odds WHERE scrape_run_id = 'a3f1...'`.

**Overround consistency.** Re-running the scraper within minutes will insert 0 new rows — the `UNIQUE` constraint on `(bookmaker, event_id, market_type, participant_raw, scrape_run_id)` prevents exact duplicates. Odds that have moved between runs will get a new row because the `scrape_run_id` is different. `bookmaker_odds_latest` always shows the most recent snapshot per selection.

---

## 6. Step 4 — Verify Rows in the DB

Check what's stored and how many rows per market type:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
rows = conn.execute('''
    SELECT market_type, COUNT(*) as n, ROUND(AVG(back_odds),2) as avg_odds
    FROM bookmaker_odds
    GROUP BY market_type
    ORDER BY n DESC
''').fetchall()
for r in rows:
    print(r)
"
```

Example output:

```
('winner', 187, 32.14)
('h2h', 156, 1.93)
('top_3', 94, 4.81)
('top_10', 74, 2.24)
('unknown', 42, 15.60)
('gc_position', 38, 8.30)
('kom', 23, 4.10)
```

`unknown` market type means the label didn't match any pattern in the classifier. Check what labels are being missed:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
rows = conn.execute('''
    SELECT DISTINCT market_label_raw
    FROM bookmaker_odds
    WHERE market_type = 'unknown'
    LIMIT 20
''').fetchall()
for r in rows:
    print(r[0])
"
```

Check the `bookmaker_odds_latest` view returns the expected latest snapshot:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
rows = conn.execute('''
    SELECT participant_name, market_type, back_odds, fair_odds, scraped_at
    FROM bookmaker_odds_latest
    WHERE market_type = 'winner'
    ORDER BY back_odds ASC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"
```

---

## 7. Step 5 — Run the Betting Workflow

With odds in the DB, the betting workflow automatically uses real market prices:

```bash
python example_betting_workflow.py
```

The `_lookup_real_odds()` function joins `bookmaker_odds_latest` to the `riders` table on name. It tries two join conditions:

1. **Exact lowercase match** — `LOWER(bo.participant_name) = LOWER(r.name)`. Works when Betclic uses the same name as PCS.
2. **Accent-stripped match** — compares accent-stripped versions. Handles `Pogačar` → `Pogacar`.

If neither matches, the function returns `None` and the workflow falls back to `20.0 + np.random.exponential(10)`. You'll see this in the output:

```
+ Pogačar: prob=8.3%, odds=2.1, edge=4820bps, signals=[HIDDEN_FORM]     ← real odds
+ Rider 441: prob=1.2%, odds=23.7, edge=120bps, signals=[TRANSITION_STAGE]  ← simulated odds
```

If a rider you expect to match isn't matching, query both sides:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
# Check name as stored in bookmaker_odds
print(conn.execute(\"SELECT DISTINCT participant_name FROM bookmaker_odds WHERE participant_name LIKE '%Pogac%'\").fetchall())
# Check name as stored in riders
print(conn.execute(\"SELECT name FROM riders WHERE name LIKE '%Pogac%'\").fetchall())
"
```

---

## 8. Scheduled / Repeated Scraping

Odds move. Run the scraper on a schedule to build a time series of snapshots.

**Linux/macOS cron** (every 30 minutes, 06:00–22:00):

```
*/30 6-22 * * * cd /path/to/cycling_predict && python fetch_odds.py >> logs/odds.log 2>&1
```

**Windows Task Scheduler** — create a task running:
```
python C:\path\to\cycling_predict\fetch_odds.py
```
Trigger: repeat every 30 minutes.

Each run appends new rows with a fresh `scrape_run_id`. The `bookmaker_odds_latest` view always reflects the most recent prices. To query price movement on a selection over time:

```sql
SELECT scraped_at, back_odds, fair_odds
FROM bookmaker_odds
WHERE participant_name_norm = 'pogacar'
  AND market_type = 'winner'
ORDER BY scraped_at;
```

---

## 9. Understanding the Data

### Market types

| `market_type` value | What it is | French Betclic phrases matched |
|---------------------|-----------|-------------------------------|
| `winner` | Outright stage or race winner | `vainqueur`, `victoire` |
| `top_3` | Podium finish | `podium`, `top 3`, `dans le top 3` |
| `top_10` | Top-10 finish | `top 10`, `dans le top 10` |
| `h2h` | Head-to-head between two riders | `duel`, `confrontation`, `h2h`, `vs`, `face à face` |
| `gc_position` | Overall classification markets | `classement général`, `maillot jaune` |
| `kom` | King of the mountains jersey | `montagne`, `grimpeur`, `maillot à pois` |
| `points` | Points classification jersey | `points`, `maillot vert` |
| `combativity` | Most combative rider award | `combatif` |
| `breakaway` | Breakaway market | `échappée`, `breakaway` |
| `unknown` | Label matched no pattern — review and extend classifier if recurring |

### Key columns

| Column | Meaning |
|--------|---------|
| `back_odds` | Raw bookmaker decimal odds |
| `implied_prob` | `1.0 / back_odds` |
| `market_total_impl_prob` | Sum of `implied_prob` across all selections in this market — the overround. A value of 1.10 means 10% margin |
| `fair_prob` | `implied_prob / overround` — your break-even probability |
| `fair_odds` | `1.0 / fair_prob` — the odds you'd need to be offered in a zero-margin market |
| `participant_name` | Cleaned rider name, prefix-stripped |
| `participant_name_norm` | Accent-stripped ASCII version for fuzzy DB joins |
| `participant_raw` | Original label from Betclic, unchanged |
| `scrape_run_id` | UUID shared by all rows from one scrape run — use to query a historical snapshot |
| `event_id` | The `mXXXXXXXXX` suffix from the Betclic URL — stable identifier for an event |

### H2H rows

When Betclic labels a selection as `"Pogacar vs Vingegaard"`, the scraper produces **two rows** — one for each rider — both with `participant_raw = "Pogacar vs Vingegaard"`. This makes it possible to look up either rider's H2H odds by name. The `market_type` will be `h2h` for both.

---

## 10. Troubleshooting

**`(no rows)` on dry-run for a known URL**

The page structure may have changed, or Betclic is serving a bot-detection page. Check:

```bash
python -c "
import urllib.request
req = urllib.request.Request(
    'https://www.betclic.fr/cyclisme-scycling',
    headers={'User-Agent': 'Mozilla/5.0'}
)
html = urllib.request.urlopen(req).read().decode('utf-8', errors='replace')
print(html[:2000])
"
```

If you see a CAPTCHA or empty body, Betclic is rate-limiting. Wait a few minutes and retry.

---

**`sqlite3.OperationalError: no such table: bookmaker_odds`**

The schema hasn't been applied yet. Run:

```bash
python fetch_odds.py --init-schema
```

---

**`sqlite3.OperationalError: no such view: bookmaker_odds_latest`**

Same fix — schema not applied. The view is defined in `schema_extensions.sql` and created by `--init-schema`.

---

**0 rows inserted despite successful scrape**

All rows were duplicates of a previous run in the same session. The `UNIQUE` constraint on `(bookmaker, event_id, market_type, participant_raw, scrape_run_id)` prevents re-insertion of identical rows **within the same run**. Across runs this doesn't happen because `scrape_run_id` is a fresh UUID each time. If you're seeing this, check you're not accidentally running the same script instance twice.

---

**Rider not matching in `_lookup_real_odds`**

Betclic may use a different name variant. Query both:

```python
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print(conn.execute("SELECT DISTINCT participant_name FROM bookmaker_odds WHERE participant_name LIKE '%name%'").fetchall())
print(conn.execute("SELECT name FROM riders WHERE name LIKE '%name%'").fetchall())
```

If the names are structurally different (abbreviated surname, middle name included, etc.), there is currently no fuzzy matching beyond accent stripping. You can manually insert a mapping or extend the join logic in `_lookup_real_odds`.

---

**`market_type = 'unknown'` for a market you want**

The classifier uses substring matching on French labels. Add the phrase to `_MARKET_LABEL_MAP` in `pipeline/betclic_scraper.py`:

```python
_MARKET_LABEL_MAP = [
    ...
    (["your new phrase"],  "existing_market_type"),
    ...
]
```

Check what's unclassified:

```sql
SELECT DISTINCT market_label_raw FROM bookmaker_odds WHERE market_type = 'unknown';
```

---

**`HTTP 403` or `HTTP 429` errors in the log**

Betclic is blocking requests. The scraper uses a single `User-Agent` header. If blocking persists, add a short sleep between events:

```python
# In pipeline/betclic_scraper.py, inside scrape_all():
import time
for url in event_urls:
    rows = process_event(url, scrape_run_id, scraped_at)
    all_rows.extend(rows)
    time.sleep(1.5)  # add this line
```
