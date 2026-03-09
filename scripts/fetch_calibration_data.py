"""
Calibration Data Seeder
=======================

Seeds the fetch queue with historical stage-result data from high-quality races
needed to calibrate the quality-weighted specialty model in stage_ranker.py.

Usage:
    python scripts/fetch_calibration_data.py [--db PATH] [--dry-run]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db import get_connection, init_db, DB_PATH
from pipeline.queue import init_queue, add_job

# ---------------------------------------------------------------------------
# Calibration targets
# ---------------------------------------------------------------------------

_CALIBRATION_YEARS = list(range(2023, 2026))  # 2020–2025 inclusive

CALIBRATION_TARGETS = [
    # --- Grand Tours ---
    ('tour-de-france',    _CALIBRATION_YEARS),
    ('giro-d-italia',     _CALIBRATION_YEARS),
    ('vuelta-a-espana',   _CALIBRATION_YEARS),

    # --- Monuments & major WT one-days ---
    ('paris-roubaix',           _CALIBRATION_YEARS),
    ('milan-san-remo',          _CALIBRATION_YEARS),
    ('ronde-van-vlaanderen',    _CALIBRATION_YEARS),
    ('liege-bastogne-liege',    _CALIBRATION_YEARS),
    ('il-lombardia',            _CALIBRATION_YEARS),
    ('strade-bianche',          _CALIBRATION_YEARS),
    ('gent-wevelgem',           _CALIBRATION_YEARS),
    ('amstel-gold-race',        _CALIBRATION_YEARS),
    ('la-fleche-wallonne',      _CALIBRATION_YEARS),

    # --- Major WT stage races ---
    ('criterium-du-dauphine',   _CALIBRATION_YEARS),
    ('tour-de-suisse',          _CALIBRATION_YEARS),
    ('volta-a-catalunya',       _CALIBRATION_YEARS),
    ('tour-de-romandie',        _CALIBRATION_YEARS),
    ('tour-de-pologne',         _CALIBRATION_YEARS),
    ('uae-tour',                _CALIBRATION_YEARS),
]

# One-day races (monuments + WT one-days) — used for stage-count estimate
_ONE_DAY_SLUGS = {
    'paris-roubaix',
    'milan-san-remo',
    'ronde-van-vlaanderen',
    'liege-bastogne-liege',
    'il-lombardia',
    'strade-bianche',
    'gent-wevelgem',
    'amstel-gold-race',
    'la-fleche-wallonne',
}

# Grand Tours — used for stage-count estimate
_GRAND_TOUR_SLUGS = {
    'tour-de-france',
    'giro-d-italia',
    'vuelta-a-espana',
}

# Request-count estimates per race×year
# race_meta: 1 req
# stage_results: 7 avg (1 for one-days, 21 for GT, 7 for stage races)
# rider_profile + rider_results: 150 riders × 2 = 300 reqs
# Total: ~308 per race×year; GT: 1+21+300=322; one-day: 1+1+300=302
def _estimate_reqs(slug: str) -> int:
    if slug in _GRAND_TOUR_SLUGS:
        return 322   # 1 + 21 + 300
    elif slug in _ONE_DAY_SLUGS:
        return 302   # 1 + 1 + 300
    else:
        return 308   # 1 + 7 + 300


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def check_completed(conn, slug: str, year: int) -> bool:
    """Return True if stage_results for this race×year is already completed."""
    row = conn.execute(
        """
        SELECT status FROM fetch_queue
        WHERE job_type = 'stage_results' AND pcs_slug = ? AND year = ?
        """,
        (slug, year),
    ).fetchone()
    if row is None:
        return False
    return row[0] == 'completed'


def seed_race_year(conn, slug: str, year: int, dry_run: bool) -> int:
    """Seed jobs for one race×year. Returns number of jobs added (0 if dry-run)."""
    jobs = [
        ('race_meta',     1),
        ('stage_results', 2),
        ('rider_profile', 3),
        ('rider_results', 4),
    ]
    if dry_run:
        return len(jobs)
    for job_type, priority in jobs:
        add_job(conn, job_type, slug, year=year, priority=priority)
    return len(jobs)


def main():
    parser = argparse.ArgumentParser(
        description='Seed fetch queue with calibration data for quality-weighted specialty model'
    )
    parser.add_argument('--db', default=DB_PATH, help='Path to cycling.db (default: data/cycling.db)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be seeded without writing')
    args = parser.parse_args()

    conn = get_connection(args.db)
    conn.row_factory = None  # use plain tuples for status check

    # Ensure schema exists
    init_db(conn)
    init_queue(conn)

    print("Calibration Data Seeder")
    print("=======================")

    total_race_years = sum(len(years) for _, years in CALIBRATION_TARGETS)
    n_races = len(CALIBRATION_TARGETS)
    print(f"Checking {n_races} target races × {len(_CALIBRATION_YEARS)} years = {total_race_years} race×year combos")
    if args.dry_run:
        print("(DRY RUN — no jobs will be written)\n")
    else:
        print()

    # ---------------------------------------------------------------------------
    # Coverage pass
    # ---------------------------------------------------------------------------
    already_done = 0
    newly_seeded_combos = 0
    total_jobs = 0
    total_reqs_estimate = 0

    coverage_lines = []

    for slug, years in CALIBRATION_TARGETS:
        scraped = []
        missing = []
        for year in years:
            if check_completed(conn, slug, year):
                scraped.append(year)
            else:
                missing.append(year)

        already_done += len(scraped)

        year_display = ' '.join(str(y) for y in years)
        status = f"[ {len(scraped)}/{len(years)} scraped ]"
        missing_note = ''
        if missing and scraped:
            missing_note = f"  ({','.join(str(y) for y in missing)} missing)"
        coverage_lines.append((slug, year_display, status, missing_note))

        # Seed missing combos
        for year in missing:
            n_jobs = seed_race_year(conn, slug, year, args.dry_run)
            total_jobs += n_jobs
            newly_seeded_combos += 1
            total_reqs_estimate += _estimate_reqs(slug)

    # ---------------------------------------------------------------------------
    # Print coverage table
    # ---------------------------------------------------------------------------
    print("Coverage:")
    max_slug_len = max(len(s) for s, _, _, _ in coverage_lines)
    for slug, year_display, status, missing_note in coverage_lines:
        print(f"  {slug:<{max_slug_len}}  {year_display}  {status}{missing_note}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    est_seconds = total_reqs_estimate  # 1 req/s
    est_hours = est_seconds / 3600.0

    print()
    print("Summary:")
    print(f"  Already complete:   {already_done} / {total_race_years} race×years")
    if args.dry_run:
        print(f"  Would seed:        {newly_seeded_combos} × 4 job types = {total_jobs} jobs  (dry run)")
    else:
        print(f"  Newly seeded:      {newly_seeded_combos} × 4 job types = {total_jobs} jobs")
    print(f"  Estimated time:    ~{newly_seeded_combos} × 5 min = ~{est_hours:.1f} hours"
          " (conservative; rider jobs deduplicated)")
    print()
    print("Start scraping: python -m pipeline.runner")

    conn.close()


if __name__ == '__main__':
    main()
