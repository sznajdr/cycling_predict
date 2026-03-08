"""
Scrape 2026 Season Data for Target Riders
==========================================

For each rider in a target race (e.g., Paris-Nice 2026):
1. Find all their 2026 races from PCS
2. Scrape those races first
3. Build up 2026 season form data

Usage:
    python scrape_2026_season.py --race paris-nice --year 2026
"""
import argparse
import sqlite3
import sys
import time
from datetime import datetime

# Use existing pipeline infrastructure
from procyclingstats import RiderResults
from pipeline.db import get_connection
from pipeline.fetcher import fetch_race_meta
from pipeline.runner import main as pipeline_main


def get_target_riders(race_slug, year):
    """Get all riders from target race startlist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    race = cursor.execute(
        'SELECT id FROM races WHERE pcs_slug = ? AND year = ?',
        (race_slug, year)
    ).fetchone()
    
    if not race:
        print(f"Race {race_slug} {year} not found!")
        return []
    
    riders = cursor.execute('''
        SELECT DISTINCT sl.rider_id, r.pcs_url, r.name
        FROM startlist_entries sl
        JOIN riders r ON sl.rider_id = r.id
        WHERE sl.race_id = ?
        ORDER BY r.name
    ''', (race[0],)).fetchall()
    
    conn.close()
    return riders


def get_rider_2026_races(pcs_url):
    """Get all 2026 races for a rider."""
    try:
        results_url = f"{pcs_url}/results"
        rr = RiderResults(results_url)
        results = rr.results("stage_url", "rank", "date", "pcs_points", "uci_points")
        
        races_2026 = []
        for result in results:
            # Parse stage URL to get race info
            stage_url = result.get('stage_url', '')
            if not stage_url:
                continue
            
            # Extract year from URL (format: race/slug/year/...)
            parts = stage_url.split('/')
            if len(parts) >= 3:
                try:
                    year_str = parts[-2] if len(parts) >= 2 else ''
                    year = int(year_str) if year_str.isdigit() else 0
                    if year == 2026:
                        race_slug = parts[-3] if len(parts) >= 3 else ''
                        if race_slug and race_slug not in [r['slug'] for r in races_2026]:
                            races_2026.append({
                                'slug': race_slug,
                                'year': year,
                                'url': f"race/{race_slug}/{year}"
                            })
                except (ValueError, IndexError):
                    continue
        
        return races_2026
        
    except Exception as e:
        print(f"    Error getting races: {e}")
        return []


def race_exists_in_db(race_slug, year):
    """Check if race is already in database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    existing = cursor.execute(
        'SELECT id FROM races WHERE pcs_slug = ? AND year = ?',
        (race_slug, year)
    ).fetchone()
    
    conn.close()
    return existing is not None


def add_race_to_queue(race_slug, year):
    """Add race to fetch_queue for scraping."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if already in queue
    existing = cursor.execute('''
        SELECT id FROM fetch_queue 
        WHERE job_type = 'race_meta' AND pcs_slug = ? AND year = ?
    ''', (race_slug, year)).fetchone()
    
    if existing:
        return True, "Already in queue"
    
    # Add to queue
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO fetch_queue 
            (job_type, pcs_slug, year, status, priority, max_retries)
            VALUES ('race_meta', ?, ?, 'pending', 1, 3)
        ''', (race_slug, year))
        
        # Also add startlist job
        cursor.execute('''
            INSERT OR IGNORE INTO fetch_queue 
            (job_type, pcs_slug, year, status, priority, max_retries)
            VALUES ('race_startlist', ?, ?, 'pending', 2, 3)
        ''', (race_slug, year))
        
        conn.commit()
        return True, "Added to queue"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Scrape 2026 season data for target race riders')
    parser.add_argument('--race', default='paris-nice', help='Target race slug (default: paris-nice)')
    parser.add_argument('--year', type=int, default=2026, help='Target year (default: 2026)')
    parser.add_argument('--max-races', type=int, default=50, help='Max races to add (default: 50)')
    parser.add_argument('--rider-limit', type=int, default=0, help='Limit to N riders (0 = all)')
    parser.add_argument('--run-pipeline', action='store_true', help='Auto-run pipeline after adding races')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"2026 SEASON DATA SCRAPER")
    print(f"Target: {args.race} {args.year}")
    print("="*70)
    print()
    
    # Get target riders
    print("Step 1: Getting target riders...")
    riders = get_target_riders(args.race, args.year)
    
    if not riders:
        print("No riders found!")
        return
    
    if args.rider_limit > 0:
        riders = riders[:args.rider_limit]
    
    print(f"Found {len(riders)} riders in startlist")
    print()
    
    # Collect all 2026 races
    print("Step 2: Finding 2026 races for each rider...")
    all_races = {}  # slug -> race_info
    
    for i, (rider_id, pcs_url, name) in enumerate(riders, 1):
        print(f"  [{i}/{len(riders)}] {name}...", end=" ", flush=True)
        
        try:
            races = get_rider_2026_races(pcs_url)
            print(f"found {len(races)} races")
            
            for race in races:
                key = (race['slug'], race['year'])
                if key not in all_races:
                    all_races[key] = race
        except Exception as e:
            print(f"error: {e}")
        
        time.sleep(0.5)  # Rate limit
    
    print()
    print(f"Total unique 2026 races found: {len(all_races)}")
    print()
    
    # Filter out target race itself
    all_races = {k: v for k, v in all_races.items() 
                 if not (k[0] == args.race and k[1] == args.year)}
    
    print(f"After excluding target race: {len(all_races)} races")
    print()
    
    # Check which races need scraping
    print("Step 3: Checking which races need scraping...")
    races_to_add = []
    
    for key, race in list(all_races.items())[:args.max_races]:
        if race_exists_in_db(race['slug'], race['year']):
            print(f"  {race['slug']} {race['year']}: Already in DB")
        else:
            print(f"  {race['slug']} {race['year']}: Will add to queue")
            races_to_add.append(race)
    
    print()
    print(f"Races to add to queue: {len(races_to_add)}")
    print()
    
    if not races_to_add:
        print("All races already in database!")
        return
    
    # Add to queue
    print("Step 4: Adding races to job queue...")
    added = 0
    failed = 0
    
    for race in races_to_add:
        success, msg = add_race_to_queue(race['slug'], race['year'])
        if success:
            if "Already" in msg:
                print(f"  {race['slug']}: {msg}")
            else:
                print(f"  {race['slug']}: Added")
                added += 1
        else:
            print(f"  {race['slug']}: Failed - {msg}")
            failed += 1
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Riders analyzed: {len(riders)}")
    print(f"Races found: {len(all_races)}")
    print(f"Races added to queue: {added}")
    print(f"Races already in DB: {len(all_races) - len(races_to_add)}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()
    
    if args.run_pipeline and added > 0:
        print("Step 5: Running pipeline to scrape new races...")
        print()
        pipeline_main()
    else:
        print("Next steps:")
        print(f"  1. Run: python -m pipeline.runner")
        print(f"  2. Or use: python monitor.py (to watch progress)")
        print(f"  3. When done: python rank_stage.py {args.race} {args.year} 1 --run-models")


if __name__ == '__main__':
    main()
