# Cycling Predict — Scraping Pipeline Reference

End-to-end walkthrough of the data collection system: what it scrapes, how it
stores data, the exact execution order, and how to operate it.

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Prerequisites & Setup](#2-prerequisites--setup)
3. [Configuration](#3-configuration)
4. [Database Schema](#4-database-schema)
5. [How the Queue Works](#5-how-the-queue-works)
6. [Job Types — Complete Reference](#6-job-types--complete-reference)
7. [Execution Flow — Step by Step](#7-execution-flow--step-by-step)
8. [Rate Limiting & Error Handling](#8-rate-limiting--error-handling)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Monitoring Progress](#10-monitoring-progress)
11. [Resume Safety](#11-resume-safety)
12. [Useful SQL Queries](#12-useful-sql-queries)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Directory Structure

```
cycling_predict/
├── config/
│   └── races.yaml              # Which races + years to scrape
├── data/
│   └── cycling.db              # SQLite database (created at runtime)
├── logs/
│   └── pipeline.log            # Full run log (created at runtime)
├── pipeline/
│   ├── __init__.py
│   ├── db.py                   # Schema, connection, all upsert/query helpers
│   ├── fetcher.py              # HTTP scraping wrappers (one function per job type)
│   ├── pcs_parser.py           # URL parsing, time parsing, rank normalisation
│   ├── queue.py                # Job queue: claim, complete, seed, freshness checks
│   └── runner.py               # Orchestrator: routes jobs, main() entry point
├── tests/
│   ├── test_connection.py      # Smoke test — can we reach PCS?
│   ├── test_rider.py           # Rider scrape + DB roundtrip
│   └── test_race.py            # Race meta + startlist scrape + DB roundtrip
├── monitor.py                  # Quick progress snapshot (run anytime while pipeline runs)
├── .gitignore
└── requirements.txt
```

---

## 2. Prerequisites & Setup

### Install dependencies

```bash
cd path/to/cycling_predict

# Install the procyclingstats library in editable mode from the sibling folder
pip install -e ../procyclingstats

# Install PyYAML (only other dependency)
pip install -r requirements.txt
```

### Verify connectivity (run before anything else)

```bash
python tests/test_connection.py
```

Expected output:
```
PASS: TodayRaces.finished_races() returned list (N items)
```

If this fails, PCS is unreachable or Cloudflare is blocking you. Nothing else
will work until this passes.

### Validate scraping + DB roundtrip

```bash
python tests/test_rider.py   # ~3 seconds (2 HTTP requests)
python tests/test_race.py    # ~2 seconds (2 HTTP requests)
```

Both should print all `PASS` lines. These are the only tests that touch the
network — everything else is in-memory SQLite.

---

## 3. Configuration

**`config/races.yaml`** controls which races and years are scraped.

```yaml
year: 2026   # primary prediction year (not yet used by pipeline directly)

races:
  - name: Tour de France
    pcs_slug: tour-de-france      # matches procyclingstats.com URL fragment
    type: stage_race              # stage_race | one_day
    history_years: [2021, 2022, 2023, 2024, 2025]
```

### Key fields

| Field | Purpose |
|-------|---------|
| `pcs_slug` | Appended directly to PCS URLs, e.g. `race/tour-de-france/2022` |
| `type` | `stage_race` → has stages; `one_day` → single result |
| `history_years` | Each year generates its own set of jobs |

### To test with a single race first

Comment out all races except one and uncomment only e.g. Paris-Nice:

```yaml
races:
  - name: Paris-Nice
    pcs_slug: paris-nice
    type: stage_race
    history_years: [2022, 2023, 2024, 2025]
```

Validate it works fully before uncommenting the rest. Paris-Nice 2022–2025 =
4 races × ~7 stages = roughly 400–600 total jobs.

---

## 4. Database Schema

The database lives at `data/cycling.db`. It is created automatically on first
run. WAL journal mode is enabled for safe concurrent reads while the pipeline
writes.

### Core tables

#### `races`
One row per race × year.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| pcs_slug | TEXT | e.g. `tour-de-france` |
| display_name | TEXT | e.g. `Tour de France` |
| year | INTEGER | |
| startdate | TEXT | `YYYY-MM-DD` |
| enddate | TEXT | `YYYY-MM-DD` |
| category | TEXT | e.g. `Men Elite` |
| uci_tour | TEXT | e.g. `UCI Worldtour` |
| is_one_day_race | INTEGER | 0 or 1 |
| fetched_at | TIMESTAMP | |

**Unique constraint:** `(pcs_slug, year)`

---

#### `race_stages`
One row per stage (or one row for the entire one-day race).

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| race_id | INTEGER FK → races | |
| stage_number | INTEGER | NULL for one-day races |
| stage_type | TEXT | `flat`, `hilly`, `mountain`, `itt`, `ttt`, `prologue`, `road` |
| stage_date | TEXT | `YYYY-MM-DD` |
| distance_km | REAL | Filled by `stage_results` job |
| pcs_stage_url | TEXT UNIQUE | e.g. `race/tour-de-france/2022/stage-3` |
| is_one_day_race | INTEGER | 0 or 1 |
| vertical_m | INTEGER | Meters climbed — filled by `stage_results` |
| profile_score | INTEGER | PCS difficulty score 0–100 — filled by `stage_results` |
| avg_temp_c | REAL | Average temperature — filled by `stage_results` |
| avg_speed_winner_kmh | REAL | Wind/conditions proxy — filled by `stage_results` |
| won_how | TEXT | e.g. `Solo breakaway`, `Sprint` — filled by `stage_results` |
| startlist_quality_score | INTEGER | Field strength at race start — filled by `stage_results` |

**Unique constraints:** `pcs_stage_url` (primary dedup key); `(race_id, stage_number)`
> Note: `(race_id, stage_number)` cannot deduplicate one-day races since
> `stage_number` is NULL — `pcs_stage_url` handles that case.

---

#### `riders`
One row per rider. Initially a stub (name + nationality only); enriched to
full profile when `rider_profile` job completes.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| pcs_url | TEXT UNIQUE | e.g. `rider/tadej-pogacar` |
| name | TEXT | |
| nationality | TEXT | 2-char country code, e.g. `SI` |
| birthdate | TEXT | `YYYY-M-D` format from PCS |
| height_m | REAL | |
| weight_kg | REAL | |
| sp_one_day_races | INTEGER | PCS specialty score (0–100) |
| sp_gc | INTEGER | |
| sp_time_trial | INTEGER | |
| sp_sprint | INTEGER | |
| sp_climber | INTEGER | |
| sp_hills | INTEGER | |
| fetched_at | TIMESTAMP | Updated every time profile is re-fetched |

---

#### `teams`
One row per team × season (e.g. `team/uae-team-emirates-2022`).

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| pcs_url | TEXT UNIQUE | Includes year: `team/jumbo-visma-2022` |
| name | TEXT | |
| class | TEXT | `WT`, `Pro`, `1.1`, etc. |
| nationality | TEXT | |

---

#### `startlist_entries`
Which riders started which race, with their team and race number.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| race_id | INTEGER FK → races | |
| rider_id | INTEGER FK → riders | |
| team_id | INTEGER FK → teams | |
| rider_number | INTEGER | Bib number; NULL if not yet assigned |

**Unique constraint:** `(race_id, rider_id)`

---

#### `rider_results`
The central fact table. One row per rider × stage × result category.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| rider_id | INTEGER FK → riders | |
| race_id | INTEGER FK → races | |
| stage_id | INTEGER FK → race_stages | |
| result_category | TEXT | See values below |
| rank | TEXT | `"1"`, `"2"`, `"DNF"`, `"DNS"`, `"DSQ"`, `"OTL"` |
| time_seconds | INTEGER | Absolute stage or cumulative GC time in seconds |
| time_behind_winner_seconds | INTEGER | Gap to stage/GC winner; 0 for winner |
| pcs_points | REAL | |
| uci_points | REAL | |
| team_id | INTEGER FK → teams | Snapshot of team at race time |
| fetched_at | TIMESTAMP | |

**`result_category` values:**

| Value | Meaning | Source |
|-------|---------|--------|
| `stage` | Stage finish result | `Stage.results()` or `RiderResults.results()` |
| `gc` | General classification after the stage | `Stage.gc()` |
| `points` | Points classification after the stage | `Stage.points()` |
| `mountains` | KOM classification after the stage | `Stage.kom()` |
| `youth` | Best young rider classification | `Stage.youth()` |
| `combativity` | Most combative rider award | `RaceCombativeRiders` |

**Unique constraint:** `(rider_id, stage_id, result_category)`
> One row per rider per stage per classification. Re-runs update rather than duplicate.

**Indexes:**
```sql
idx_results_rider_race  ON rider_results(rider_id, race_id, result_category)
idx_results_stage_rank  ON rider_results(stage_id, rank)
idx_results_rider_date  ON rider_results(rider_id, fetched_at)
```

---

#### `rider_teams`
Career team history per rider, one row per rider × season × team.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| rider_id | INTEGER FK → riders | |
| season | INTEGER | Year |
| team_id | INTEGER FK → teams | |
| team_class | TEXT | `WT`, `Pro`, etc. |
| since | TEXT | `MM-DD` — first day in team this season |
| until | TEXT | `MM-DD` — last day in team this season |

**Unique constraint:** `(rider_id, season, team_id)`

---

#### `race_climbs`
All named climbs in a race, with physical characteristics.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| race_id | INTEGER FK → races | |
| climb_name | TEXT | |
| climb_url | TEXT | PCS location URL |
| length_km | REAL | Climb length in km |
| steepness_pct | REAL | Average gradient % |
| top_m | INTEGER | Altitude at summit in metres |
| km_before_finish | INTEGER | How far from the finish the summit is |

**Unique constraint:** `(race_id, climb_name)`

---

### Queue tables

#### `fetch_queue`
The job queue. Every unit of work is a row here.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| job_type | TEXT | See job types below |
| pcs_slug | TEXT | Race slug OR rider URL OR stage URL (see per-job notes) |
| year | INTEGER | Race year; `0` sentinel for rider jobs |
| status | TEXT | `pending`, `in_progress`, `completed`, `failed`, `permanent_fail` |
| priority | INTEGER | Lower = runs first |
| retries | INTEGER | How many times this job has failed |
| max_retries | INTEGER | Default 3; hit max → `permanent_fail` |
| next_attempt_at | TIMESTAMP | Exponential backoff delays failed jobs |
| last_error | TEXT | Error message from last failure |
| created_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |

**Unique constraint:** `(job_type, pcs_slug, year)`
> This is what makes `seed_queue` idempotent — re-running never creates duplicates.

**Indexes:**
```sql
idx_queue_status_prio  ON fetch_queue(status, priority, next_attempt_at)
idx_queue_lookup       ON fetch_queue(job_type, pcs_slug, year, status)
```

#### `data_freshness`
TTL cache. Prevents re-fetching data that was recently collected.

| Column | Type | Notes |
|--------|------|-------|
| entity_type | TEXT | e.g. `race_meta`, `rider_profile` |
| entity_key | TEXT | e.g. `tour-de-france/2022` or `rider/tadej-pogacar` |
| last_fetched_at | TIMESTAMP | |
| data_hash | TEXT | Reserved for future change detection |
| ttl_seconds | INTEGER | Default 86400 (24 hours) |

**Primary key:** `(entity_type, entity_key)`

---

## 5. How the Queue Works

### Claiming jobs

The runner loops: `claim_next_job → process_job → complete_job → repeat`.

`claim_next_job` does an atomic SELECT + UPDATE:
```sql
SELECT ... FROM fetch_queue
WHERE status = 'pending' AND next_attempt_at <= now
ORDER BY priority ASC, next_attempt_at ASC
LIMIT 1
-- then immediately:
UPDATE fetch_queue SET status = 'in_progress' WHERE id = ?
```

This means if you kill the runner mid-job, that job stays `in_progress`
forever. To reset stale in-progress jobs:
```sql
UPDATE fetch_queue SET status = 'pending' WHERE status = 'in_progress';
```

### Failure & backoff

On failure, `complete_job` increments `retries` and sets `next_attempt_at` to
`now + 2^retries seconds` (2s, 4s, 8s). At `max_retries` (default 3) the job
becomes `permanent_fail` and is skipped forever.

### Freshness checks

`is_fresh(entity_type, entity_key)` checks `data_freshness` — if the entity
was fetched within TTL (24h default), the job is skipped. This is checked
**before** adding jobs (in `seed_queue` and before queuing rider jobs), not
during claim. So a fresh entity simply never gets a new queue entry.

### Sentinel year for rider jobs

SQLite treats `NULL IS DISTINCT FROM NULL` in UNIQUE constraints — two rows
with `year = NULL` are considered different. To maintain idempotency for rider
jobs (which have no meaningful year), `year = 0` is used as a sentinel.

---

## 6. Job Types — Complete Reference

### Priority ordering (lower runs first)

```
1  →  race_meta
2  →  race_startlist
3  →  stage_results
4  →  combativity, race_climbs
5  →  rider_profile
6  →  rider_results
```

---

### `race_meta` — Priority 1

**What it fetches:** Race overview page
**PCS URL:** `race/{pcs_slug}/{year}`
**Scraper class:** `Race`

**Fields collected:**

| Field | PCS method |
|-------|-----------|
| display_name | `.name()` |
| startdate / enddate | `.startdate()` / `.enddate()` |
| category | `.category()` |
| uci_tour | `.uci_tour()` |
| is_one_day_race | `.is_one_day_race()` |
| stages list | `.stages("date", "profile_icon", "stage_name", "stage_url")` |

**What it writes:**
- Upserts 1 row in `races`
- Upserts N rows in `race_stages` (one per stage, or one for one-day races)
  - `stage_type` is inferred from `profile_icon` + stage name at this point
  - `distance_km` is NULL here — filled later by `stage_results`

**What it chains:**
- `stage_results` job per stage (priority 3)
- `combativity` job for the race (priority 4)
- `race_climbs` job for the race (priority 4)
- `race_startlist` job (priority 2)

**`pcs_slug` column stores:** race slug (e.g. `tour-de-france`)
**`year` column stores:** race year (e.g. `2022`)

---

### `race_startlist` — Priority 2

**What it fetches:** Race startlist page
**PCS URL:** `race/{pcs_slug}/{year}/startlist`
**Scraper class:** `RaceStartlist`

**Fields collected per entry:**

| Field | Notes |
|-------|-------|
| rider_url | e.g. `rider/tadej-pogacar` |
| rider_name | Full name as shown on PCS |
| team_url | Includes year, e.g. `team/uae-team-emirates-2022` |
| team_name | |
| nationality | 2-char country code |
| rider_number | Bib number; NULL if race hasn't started |

**What it writes:**
- Upserts a team stub (name only, no class/nationality yet) in `teams`
- Upserts a rider stub (name + nationality only) in `riders`
- Upserts 1 row in `startlist_entries` per rider

**What it chains (for each rider, if not already fresh):**
- `rider_profile` job (priority 5, `year = 0`)
- `rider_results` job (priority 6, `year = 0`)

**`pcs_slug` column stores:** race slug
**`year` column stores:** race year

---

### `stage_results` — Priority 3

**What it fetches:** Individual stage result page
**PCS URL:** `race/{pcs_slug}/{year}/stage-{N}` (or `race/{pcs_slug}/{year}` for one-day races)
**Scraper class:** `Stage`

**Stage metadata collected (written to `race_stages`):**

| Column | PCS method | Notes |
|--------|-----------|-------|
| distance_km | `.distance()` | Fills the NULL from race_meta |
| vertical_m | `.vertical_meters()` | May be NULL for flat stages |
| profile_score | `.profile_score()` | Numerical difficulty |
| avg_temp_c | `.avg_temperature()` | May be NULL |
| avg_speed_winner_kmh | `.avg_speed_winner()` | Wind/conditions proxy |
| won_how | `.won_how()` | e.g. `Solo breakaway`, `Sprint of large group` |
| startlist_quality_score | `.race_startlist_quality_score()[0]` | Field strength |

Written via `UPDATE race_stages SET ... WHERE pcs_stage_url = ?` using
`COALESCE` — existing non-NULL values are never overwritten.

**Classification results collected (written to `rider_results`):**

| `result_category` | Source method | Time field |
|-------------------|---------------|-----------|
| `stage` | `.results()` | Stage time (absolute or +gap) |
| `gc` | `.gc()` | Cumulative GC time |
| `points` | `.points()` | No time |
| `mountains` | `.kom()` | No time |
| `youth` | `.youth()` | Cumulative GC time |

**Time gap logic:**
- Winner = first row in the list (rank 1); their `time_s` = absolute time
- Non-winner with `time_s < winner_time_s` = PCS showed a "+gap" string
  (e.g. `+0:05:32`); `time_behind = time_s`, `time_seconds = winner + gap`
- Non-winner with `time_s >= winner_time_s` = absolute time;
  `time_behind = time_s - winner_time_s`
- Riders not in `riders` table (not in our startlist) are silently skipped

**`pcs_slug` column stores:** full stage URL (e.g. `race/paris-nice/2022/stage-3`)
**`year` column stores:** race year (for UNIQUE constraint)

---

### `combativity` — Priority 4

**What it fetches:** Most combative riders list for the whole race
**PCS URL:** `race/{pcs_slug}/{year}/results/comative-riders`
**Scraper class:** `RaceCombativeRiders`

**Fields collected:**

| Field | Notes |
|-------|-------|
| rider_url | Award winner |
| stage_url | Which stage the award was for |

**What it writes:**
- Inserts into `rider_results` with `result_category = 'combativity'`, `rank = '1'`
- One row per stage that had an award winner

**`pcs_slug` column stores:** race slug
**`year` column stores:** race year

---

### `race_climbs` — Priority 4

**What it fetches:** All climbs listed in a race's route
**PCS URL:** `race/{pcs_slug}/{year}/route/climbs`
**Scraper class:** `RaceClimbs`

**Fields collected per climb:**

| `race_climbs` column | PCS field | Notes |
|----------------------|-----------|-------|
| climb_name | `climb_name` | |
| climb_url | `climb_url` | PCS location URL |
| length_km | `length` | km |
| steepness_pct | `steepness` | average % gradient |
| top_m | `top` | altitude at summit in metres |
| km_before_finish | `km_before_finnish` | note PCS typo in source |

**What it writes:**
- Upserts into `race_climbs`, one row per named climb

**`pcs_slug` column stores:** race slug
**`year` column stores:** race year

---

### `rider_profile` — Priority 5

**What it fetches:** Rider profile page
**PCS URL:** `rider/{rider-slug}`
**Scraper class:** `Rider`

**Fields collected:**

| `riders` column | PCS method | Notes |
|-----------------|-----------|-------|
| name | `.name()` | |
| nationality | `.nationality()` | 2-char country code |
| birthdate | `.birthdate()` | `YYYY-M-D` (no zero-padding) |
| height_m | `.height()` | May be NULL |
| weight_kg | `.weight()` | May be NULL |
| sp_one_day_races | `.points_per_speciality()["one_day_races"]` | 0–100 |
| sp_gc | `.points_per_speciality()["gc"]` | |
| sp_time_trial | `.points_per_speciality()["time_trial"]` | |
| sp_sprint | `.points_per_speciality()["sprint"]` | |
| sp_climber | `.points_per_speciality()["climber"]` | |
| sp_hills | `.points_per_speciality()["hills"]` | |

**Team history collected:**
- `.teams_history()` → list of `{season, team_url, team_name, class, since, until}`
- Each entry upserts a team row (with class filled in) and a `rider_teams` row

**`pcs_slug` column stores:** rider URL (e.g. `rider/tadej-pogacar`)
**`year` column stores:** `0` (sentinel — rider jobs have no year)

---

### `rider_results` — Priority 6

**What it fetches:** All-time results for a rider
**PCS URL:** `rider/{rider-slug}/results`
**Scraper class:** `RiderResults`

**Fields collected per result row:**

| Field | Notes |
|-------|-------|
| stage_url | Parsed to determine race, year, stage number, category |
| rank | Numeric string or `DNF`/`DNS` etc. |
| pcs_points | |
| uci_points | |

**How results are stored:**
1. `parse_stage_url(stage_url)` extracts: `race_slug`, `year`, `stage_number`, `result_category`
2. Look up `race_id` — if race not in our DB, skip (rider may have raced things we don't track)
3. Look up `stage_id` — if stage not in DB, skip
4. Insert into `rider_results`

**URL patterns handled by `parse_stage_url`:**

| URL pattern | stage_number | result_category |
|-------------|-------------|-----------------|
| `race/{slug}/{year}/stage-N` | N | `stage` |
| `race/{slug}/{year}/prologue` | 0 | `stage` |
| `race/{slug}/{year}/gc` | NULL | `gc` |
| `race/{slug}/{year}/points` | NULL | `points` |
| `race/{slug}/{year}/mountains` or `/kom` | NULL | `mountains` |
| `race/{slug}/{year}/youth` | NULL | `youth` |
| `race/{slug}/{year}/combativity` | NULL | `combativity` |
| `race/{slug}/{year}` (no suffix) | NULL | `stage` (one-day race) |

**`pcs_slug` column stores:** rider URL
**`year` column stores:** `0` (sentinel)

---

## 7. Execution Flow — Step by Step

### Phase 0: Initialisation (runs once per execution)

```
python -m pipeline.runner
    │
    ├── setup_logging()
    │     Creates logs/pipeline.log + stream handler
    │
    ├── load_config()
    │     Reads config/races.yaml
    │
    ├── get_connection()
    │     Opens data/cycling.db with WAL mode + foreign keys
    │
    ├── init_db()
    │     CREATE TABLE IF NOT EXISTS for all 9 tables
    │     _migrate_db(): ALTER TABLE ADD COLUMN for new columns (safe to repeat)
    │
    ├── init_queue()
    │     CREATE TABLE IF NOT EXISTS fetch_queue + data_freshness
    │     _migrate_queue_if_needed(): detects old CHECK constraint, recreates if needed
    │
    └── seed_queue()
          For each race × year in config:
            if NOT is_fresh("race_meta", "{slug}/{year}"):
                add_job("race_meta", slug, year, priority=1)
            if NOT is_fresh("race_startlist", "{slug}/{year}"):
                add_job("race_startlist", slug, year, priority=2)
          (INSERT OR IGNORE — safe to call repeatedly)
```

### Phase 1: Race metadata (priority 1)

```
claim_next_job() → race_meta job
    │
    ├── fetch_race_meta(slug, year)
    │     GET race/{slug}/{year}
    │     Parses: name, dates, category, uci_tour, is_one_day_race, stages[]
    │     Sleep 1s
    │
    ├── upsert_race()
    │     INSERT OR UPDATE races
    │
    ├── upsert_stage() × N
    │     One per stage; distance_km = NULL at this point
    │     stage_type inferred from profile_icon + stage name
    │
    ├── add_job("stage_results", stage_url, year, priority=3) × N
    ├── add_job("combativity", slug, year, priority=4)
    ├── add_job("race_climbs", slug, year, priority=4)
    ├── mark_fresh("race_meta", "{slug}/{year}")
    └── add_job("race_startlist", slug, year, priority=2)
```

### Phase 2: Startlist (priority 2)

```
claim_next_job() → race_startlist job
    │
    ├── fetch_startlist(slug, year)
    │     GET race/{slug}/{year}/startlist
    │     Returns ~150–180 entries
    │     Sleep 1s
    │
    ├── For each entry:
    │     upsert_team(stub)         → teams
    │     upsert_rider(stub)        → riders (name + nationality only)
    │     upsert_startlist_entry()  → startlist_entries
    │     if not fresh: add_job("rider_profile", rider_url, year=0, priority=5)
    │     if not fresh: add_job("rider_results", rider_url, year=0, priority=6)
    │
    └── mark_fresh("race_startlist", "{slug}/{year}")
```

### Phase 3: Stage results (priority 3)

Runs once per stage. For a 7-stage race × 4 years = 28 jobs at this priority.

```
claim_next_job() → stage_results job
    │
    ├── fetch_stage_results(pcs_stage_url)
    │     GET race/{slug}/{year}/stage-N
    │     Collects meta: distance, vertical_m, profile_score, avg_temp,
    │                    avg_speed_winner, won_how, startlist_quality_score
    │     Collects results: stage, gc, points, mountains, youth
    │     Sleep 1s
    │
    ├── update_stage_meta()
    │     UPDATE race_stages SET ... WHERE pcs_stage_url = ?
    │     (COALESCE — never overwrites non-NULL with NULL)
    │
    └── insert_rider_result() × riders × categories
          Skips riders not in our startlist (not in riders table)
          Handles winner time vs +gap detection
```

### Phase 4: Combativity + Race climbs (priority 4)

Run once per race — small jobs.

```
claim_next_job() → combativity job
    ├── fetch_combativity(slug, year)
    │     GET race/{slug}/{year}/results/comative-riders
    │     Sleep 1s
    └── insert_rider_result() per award (result_category = 'combativity')

claim_next_job() → race_climbs job
    ├── fetch_race_climbs(slug, year)
    │     GET race/{slug}/{year}/route/climbs
    │     Sleep 1s
    └── upsert_race_climb() per climb
```

### Phase 5: Rider profiles (priority 5)

One job per unique rider across all startlists. In a full backfill across
multiple races and years, riders who appear in multiple startlists are
de-duplicated — `rider_profile` is queued only once (UNIQUE constraint +
freshness check).

```
claim_next_job() → rider_profile job
    ├── fetch_rider_profile(rider_url)
    │     GET rider/{slug}
    │     Collects: name, nationality, birthdate, height, weight,
    │               6 specialty scores, teams_history
    │     Sleep 1s
    ├── upsert_rider()     — full update
    └── insert_rider_teams()
          Upserts teams + rider_teams rows for each season in career
```

### Phase 6: Rider results (priority 6)

One job per unique rider. Fetches the rider's entire result history across
all races and years, filters to only races/stages in our DB.

```
claim_next_job() → rider_results job
    ├── fetch_rider_results(rider_url)
    │     GET rider/{slug}/results
    │     Returns flat list: all results across all years
    │     Sleep 1s
    └── For each result:
          parse_stage_url(stage_url) → race_slug, year, stage_number, category
          get_race_id() → skip if race not tracked
          get_stage_id() → skip if stage not tracked
          insert_rider_result()
```

---

## 8. Rate Limiting & Error Handling

### Rate limiting

Every `fetch_*` function calls `time.sleep(1.0)` before returning — regardless
of success or failure. This is a hard floor of 1 HTTP request per second.

The `procyclingstats` library (via `cloudscraper`) handles Cloudflare
challenges and has its own internal retry logic (3 attempts, exponential
backoff on 403s).

### Pipeline-level retries

Failed jobs get exponential backoff via the queue:

```
Attempt 1 fails → retry in 2s   (2^1)
Attempt 2 fails → retry in 4s   (2^2)
Attempt 3 fails → retry in 8s   (2^3)
Attempt 4 fails → permanent_fail (never retried again)
```

`permanent_fail` jobs are visible in `monitor.py` and can be manually reset:
```sql
UPDATE fetch_queue SET status='pending', retries=0, last_error=NULL
WHERE status='permanent_fail';
```

### What gets skipped silently

- Riders in stage results not in our `riders` table (they weren't in a
  startlist we scraped)
- Stage URLs in `rider_results` that don't match any race/stage in our DB
  (the rider raced things we don't track)
- Any result row missing `rider_url`

These are logged at DEBUG level. To see them: change `setup_logging()` level
to `logging.DEBUG`.

---

## 9. Running the Pipeline

### First time (fresh start)

```bash
cd path/to/cycling_predict

# 1. Verify tests pass
python tests/test_connection.py
python tests/test_rider.py
python tests/test_race.py

# 2. Edit config/races.yaml — uncomment only one race to validate end-to-end
#    (Paris-Nice is already the only active race)

# 3. Run the pipeline
python -m pipeline.runner
```

### Adding more races

1. Uncomment races in `config/races.yaml`
2. Run again — `seed_queue` adds new jobs for the new races only. Already-
   completed work is skipped (freshness check + UNIQUE constraint)

```bash
python -m pipeline.runner
```

### Full backfill (all 12 races, 4–5 years each)

Uncomment everything in `config/races.yaml`, then run. Estimated time depends
on network conditions:

| Phase | Jobs (approx) | Time at 1 req/s |
|-------|---------------|-----------------|
| race_meta | ~55 | ~1 min |
| race_startlist | ~55 | ~1 min |
| stage_results | ~400 | ~7 min |
| combativity + race_climbs | ~110 | ~2 min |
| rider_profile | ~800 unique riders | ~13 min |
| rider_results | ~800 unique riders | ~13 min |
| **Total** | **~2220** | **~40 min** |

> These are conservative estimates. The actual rider count will be higher
> once multi-race startlists are accumulated (same rider in multiple races =
> still only one `rider_profile` job due to deduplication).

### Killing and restarting

Safe at any point. On restart:
- `seed_queue` is idempotent — no duplicates created
- Completed jobs stay `completed` — not re-run
- The `in_progress` job from when you killed it stays stuck; reset it:
  ```sql
  UPDATE fetch_queue SET status='pending' WHERE status='in_progress';
  ```
- Then rerun `python -m pipeline.runner`

---

## 10. Monitoring Progress

Run in a second terminal while the pipeline is running:

```bash
python monitor.py
```

Example output mid-run:

```
=== Queue status ===
  combativity          completed       4
  race_climbs          completed       4
  race_meta            completed       4
  race_startlist       completed       4
  rider_profile        in_progress     1
  rider_profile        pending         312
  rider_results        pending         313
  stage_results        completed       28

=== Counts ===
  races                     4
  riders                    313
  teams                     28
  race_stages               30
  startlist_entries         556
  rider_results             4820
```

### Reading the log

```bash
# Tail the log file live
type logs\pipeline.log   # Windows
# or open it in any text editor — it's plain text
```

Each completed job logs one INFO line:
```
2026-03-07 14:23:11 INFO pipeline.runner: race_meta done: paris-nice/2022 (7 stages queued)
2026-03-07 14:23:13 INFO pipeline.runner: stage_results done: race/paris-nice/2022/stage-1 (87 rows stored)
2026-03-07 14:23:14 INFO pipeline.runner: rider_profile done: rider/tadej-pogacar
```

---

## 11. Resume Safety

The pipeline is safe to stop and restart at any time. Here is exactly what
happens in each scenario:

| Scenario | What happens on restart |
|----------|------------------------|
| Stopped cleanly after a job completed | Continues from next pending job |
| Killed mid-job (e.g. Ctrl+C) | That job stays `in_progress`; reset with the SQL below, then restart |
| DB deleted accidentally | Re-run from scratch — `seed_queue` regenerates everything |
| Race config changed (new races added) | `seed_queue` adds new jobs only; existing completed jobs untouched |
| Race config changed (races removed) | Existing DB data is kept; no new jobs added for removed races |
| Re-run after full completion | `seed_queue` finds everything fresh (within 24h TTL); zero new jobs; runner exits immediately |

**Reset stuck in-progress jobs:**
```sql
UPDATE fetch_queue SET status='pending' WHERE status='in_progress';
```

**Force re-fetch of a specific race (e.g. data changed):**
```sql
-- Remove freshness record to trigger re-scrape
DELETE FROM data_freshness
WHERE entity_type='race_meta' AND entity_key='tour-de-france/2024';
-- Then reset queue jobs for it
UPDATE fetch_queue SET status='pending', retries=0
WHERE job_type='race_meta' AND pcs_slug='tour-de-france' AND year=2024;
```

---

## 12. Useful SQL Queries

Connect to the DB with any SQLite tool (DB Browser for SQLite, DBeaver, or
the Python sqlite3 module). File path: `data/cycling.db`.

### Queue health check

```sql
-- Full status breakdown
SELECT job_type, status, COUNT(*) as n
FROM fetch_queue
GROUP BY 1, 2
ORDER BY 1, 2;

-- Failed jobs with error messages
SELECT job_type, pcs_slug, year, retries, last_error
FROM fetch_queue
WHERE status IN ('failed', 'permanent_fail')
ORDER BY job_type, pcs_slug;
```

### Data completeness

```sql
-- How complete is rider_results vs what we expect?
SELECT
    r.display_name,
    r.year,
    COUNT(DISTINCT sl.rider_id) as startlist_size,
    COUNT(DISTINCT rr.rider_id) as riders_with_results
FROM races r
LEFT JOIN startlist_entries sl ON sl.race_id = r.id
LEFT JOIN rider_results rr ON rr.race_id = r.id AND rr.result_category = 'stage'
GROUP BY r.id
ORDER BY r.year, r.display_name;

-- Stage meta completeness
SELECT
    pcs_stage_url,
    distance_km, vertical_m, profile_score,
    avg_temp_c, avg_speed_winner_kmh, won_how
FROM race_stages
ORDER BY race_id, stage_number;
```

### Feature engineering previews

```sql
-- GC standings entering Stage 17 of TdF 2023
SELECT ri.name, rr.rank, rr.time_behind_winner_seconds
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
JOIN races r ON r.id = s.race_id
WHERE r.pcs_slug = 'tour-de-france'
  AND r.year = 2023
  AND s.stage_number = 16        -- GC after stage 16 = entering 17
  AND rr.result_category = 'gc'
ORDER BY CAST(rr.rank AS INTEGER);

-- Mountain stage podium rate per rider
SELECT
    ri.name,
    COUNT(*) as mountain_stages,
    SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 3 THEN 1 ELSE 0 END) as podiums,
    ROUND(100.0 * SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 3 THEN 1 ELSE 0 END) / COUNT(*), 1) as podium_pct
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
WHERE s.stage_type = 'mountain'
  AND rr.result_category = 'stage'
  AND rr.rank GLOB '[0-9]*'
GROUP BY rr.rider_id
HAVING mountain_stages >= 5
ORDER BY podium_pct DESC;

-- Combativity award leaders (aggression proxy for Strategy 8)
SELECT ri.name, COUNT(*) as combativity_awards
FROM rider_results rr
JOIN riders ri ON rr.rider_id = ri.id
WHERE rr.result_category = 'combativity'
GROUP BY rr.rider_id
ORDER BY 2 DESC
LIMIT 20;

-- Average time residual per rider per stage type (key feature for Strategy 1)
SELECT
    ri.name,
    s.stage_type,
    AVG(rr.time_behind_winner_seconds) as avg_gap_s,
    COUNT(*) as n_stages
FROM rider_results rr
JOIN race_stages s ON rr.stage_id = s.id
JOIN riders ri ON rr.rider_id = ri.id
WHERE rr.result_category = 'stage'
  AND rr.time_behind_winner_seconds IS NOT NULL
GROUP BY rr.rider_id, s.stage_type
HAVING n_stages >= 3
ORDER BY ri.name, s.stage_type;

-- Climbs in Paris-Nice with gradient > 8%
SELECT rc.climb_name, rc.length_km, rc.steepness_pct, rc.top_m, rc.km_before_finish
FROM race_climbs rc
JOIN races r ON rc.race_id = r.id
WHERE r.pcs_slug = 'paris-nice'
  AND rc.steepness_pct > 8
ORDER BY r.year, rc.km_before_finish;
```

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'procyclingstats'`

```bash
pip install -e ../procyclingstats
```

### `ValueError: HTML from given URL is invalid`

PCS returned a 404 (race doesn't exist for that year) or Cloudflare blocked
the request. The job will be retried with backoff. Check `last_error` in the
queue:
```sql
SELECT pcs_slug, year, last_error FROM fetch_queue WHERE status = 'permanent_fail';
```

### `stage not in DB: race/...`

A `stage_results` job ran before `race_meta` completed (shouldn't happen
given the priority ordering). Reset and re-run:
```sql
UPDATE fetch_queue SET status='pending', retries=0 WHERE status='permanent_fail';
```

### All rider_results rows have `time_behind_winner_seconds = NULL`

The stage result page may not have shown times (e.g. TTT, or stage result
before the race finished). This is normal for some stages. GC times are
always populated.

### `data/cycling.db` is very large

Expected sizes after a full backfill (12 races × ~5 years):
- ~50–80 MB before `rider_results` (meta only)
- ~200–400 MB after all `rider_results` (full history for ~800 riders)

WAL files (`cycling.db-wal`, `cycling.db-shm`) appear while the pipeline
is running — these are normal and auto-cleaned on clean shutdown.

### Checking what a specific job fetched

```sql
SELECT * FROM data_freshness
WHERE entity_key LIKE '%tour-de-france%'
ORDER BY last_fetched_at DESC;
```
