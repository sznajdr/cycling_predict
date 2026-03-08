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

# Add parent directory for procyclingstats
sys.path.insert(0, '..')

from procyclingstats import Rider, RiderResults, Race
from pipeline.db import get_connection, upsert_rider, upsert_race, upsert_stage
from pipeline.db import upsert_startlist_entry, upsert_rider_result
from pipeline.fetcher import fetch_rider_profile, fetch_rider_results


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
        rider_results = RiderResults(pcs_url)
        results = rider_results.results()
        
        races_2026 = []
        for result in results:
            # Parse stage URL to get race info
            stage_url = result.get('stage_url', '')
            if not stage_url:
                continue
            
            # Extract year from URL
            parts = stage_url.split('/')
            if len(parts) >= 3:
                try:
                    year = int(parts[-2]) if parts[-2].isdigit() else 0
                    if year == 2026:
                        race_slug = parts[-3] if len(parts) >= 3 else ''
                        if race_slug and race_slug not in [r['slug'] for r in races_2026]:
                            races_2026.append({
                                'slug': race_slug,
                                'year': year,
                                'url': f"race/{race_slug}/{year}"
                            })
                except:
                    continue
        
        return races_2026
        
    except Exception as e:
        print(f"    Error getting races: {e}")
        return []


def scrape_race_if_missing(race_slug, year):
    """Scrape a race if not already in database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if exists
    existing = cursor.execute(
        'SELECT id FROM races WHERE pcs_slug = ? AND year = ?',
        (race_slug, year)
    ).fetchone()
    
    if existing:
        return True, "Already in DB"
    
    conn.close()
    
    # Scrape it
    try:
        print(f"      Scraping {race_slug} {year}...")
        
        # Fetch race meta
        from pipeline.fetcher import fetch_race_meta
        race_data = fetch_race_meta(race_slug, year)
        
        if not race_data:
            return False, "Failed to fetch"
        
        # Store race
        conn = get_connection()
        race_id = upsert_race(conn, race_data)
        
        # Store stages
        for stage in race_data.get('stages', []):
            upsert_stage(conn, race_id, stage)
        
        conn.commit()
        conn.close()
        
        time.sleep(1)  # Rate limit
        return True, "Scraped successfully"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Scrape 2026 season data for target race riders')
    parser.add_argument('--race', default='paris-nice', help='Target race slug (default: paris-nice)')
    parser.add_argument('--year', type=int, default=2026, help='Target year (default: 2026)')
    parser.add_argument('--max-races', type=int, default=50, help='Max races to scrape (default: 50)')
    parser.add_argument('--rider-limit', type=int, default=0, help='Limit to N riders (0 = all)')
    
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
    all_races = {}  # slug -> year
    
    for i, (rider_id, pcs_url, name) in enumerate(riders, 1):
        print(f"  [{i}/{len(riders)}] {name}...", end=" ")
        
        races = get_rider_2026_races(pcs_url)
        print(f"found {len(races)} races")
        
        for race in races:
            key = (race['slug'], race['year'])
            if key not in all_races:
                all_races[key] = race
    
    print()
    print(f"Total unique 2026 races to check: {len(all_races)}")
    print()
    
    # Filter out target race itself
    all_races = {k: v for k, v in all_races.items() 
                 if not (k[0] == args.race and k[1] == args.year)}
    
    print(f"After excluding target race: {len(all_races)} races")
    print()
    
    # Sort by importance (you might want to prioritize certain races)
    # For now, just take first N
    races_to_scrape = list(all_races.values())[:args.max_races]
    
    # Scrape races
    print("Step 3: Scraping missing races...")
    print()
    
    success_count = 0
    fail_count = 0
    already_count = 0
    
    for i, race in enumerate(races_to_scrape, 1):
        print(f"  [{i}/{len(races_to_scrape)}] {race['slug']} {race['year']}...", end=" ")
        
        success, msg = scrape_race_if_missing(race['slug'], race['year'])
        
        if success:
            if msg == "Already in DB":
                already_count += 1
                print("already in DB")
            else:
                success_count += 1
                print("scraped ✓")
        else:
            fail_count += 1
            print(f"failed: {msg}")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Riders analyzed: {len(riders)}")
    print(f"Races found: {len(all_races)}")
    print(f"Races attempted: {len(races_to_scrape)}")
    print(f"  - Already in DB: {already_count}")
    print(f"  - Scraped: {success_count}")
    print(f"  - Failed: {fail_count}")
    print()
    print("Next steps:")
    print("  1. Run: python -m pipeline.runner (to process job queue)")
    print("  2. Run: python rank_stage.py paris-nice 2026 1 --run-models")
    print("  3. Check updated predictions with 2026 form data")


if __name__ == '__main__':
    main()
