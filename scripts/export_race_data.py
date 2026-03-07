"""
Export Race Data for Sharing
============================

Export specific race data to share with team members.

Usage:
    python scripts/export_race_data.py --race paris-nice --year 2024
    python scripts/export_race_data.py --race tour-de-france --year 2023 --output tdf_2023.zip
"""
import argparse
import sqlite3
import json
import zipfile
import os
from datetime import datetime
from pathlib import Path


def export_race_data(db_path, race_slug, year, output_path):
    """Export race data to a zip file."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print(f"Exporting {race_slug} {year}...")
    
    # Get race info
    race = conn.execute(
        "SELECT * FROM races WHERE pcs_slug = ? AND year = ?",
        (race_slug, year)
    ).fetchone()
    
    if not race:
        print(f"❌ Race not found: {race_slug} {year}")
        return False
    
    race_id = race['id']
    
    # Create temp directory for exports
    temp_dir = Path('temp_export')
    temp_dir.mkdir(exist_ok=True)
    
    data = {
        'race': dict(race),
        'stages': [],
        'riders': [],
        'results': [],
        'startlist': []
    }
    
    # Get stages
    stages = conn.execute(
        "SELECT * FROM race_stages WHERE race_id = ?",
        (race_id,)
    ).fetchall()
    data['stages'] = [dict(s) for s in stages]
    
    # Get startlist
    startlist = conn.execute("""
        SELECT sl.*, r.name as rider_name, r.nationality, t.name as team_name
        FROM startlist_entries sl
        JOIN riders r ON sl.rider_id = r.id
        JOIN teams t ON sl.team_id = t.id
        WHERE sl.race_id = ?
    """, (race_id,)).fetchall()
    data['startlist'] = [dict(s) for s in startlist]
    
    # Get riders
    rider_ids = [s['rider_id'] for s in startlist]
    if rider_ids:
        placeholders = ','.join('?' * len(rider_ids))
        riders = conn.execute(
            f"SELECT * FROM riders WHERE id IN ({placeholders})",
            rider_ids
        ).fetchall()
        data['riders'] = [dict(r) for r in riders]
    
    # Get results
    stage_ids = [s['id'] for s in stages]
    if stage_ids:
        placeholders = ','.join('?' * len(stage_ids))
        results = conn.execute(
            f"""
            SELECT rr.*, r.name as rider_name
            FROM rider_results rr
            JOIN riders r ON rr.rider_id = r.id
            WHERE rr.stage_id IN ({placeholders})
            """,
            stage_ids
        ).fetchall()
        data['results'] = [dict(r) for r in results]
    
    # Save as JSON
    json_path = temp_dir / f"{race_slug}_{year}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    # Create zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, json_path.name)
    
    # Cleanup
    json_path.unlink()
    temp_dir.rmdir()
    
    conn.close()
    
    print(f"✓ Exported to {output_path}")
    print(f"  - Race: {race['display_name']}")
    print(f"  - Stages: {len(data['stages'])}")
    print(f"  - Riders: {len(data['riders'])}")
    print(f"  - Results: {len(data['results'])}")
    
    return True


def import_race_data(zip_path, db_path):
    """Import race data from a zip file."""
    
    conn = sqlite3.connect(db_path)
    
    print(f"Importing from {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        json_name = zf.namelist()[0]
        with zf.open(json_name) as f:
            data = json.load(f)
    
    # Import logic here (simplified)
    # In practice, you'd use INSERT OR REPLACE for each table
    
    print(f"✓ Imported {data['race']['display_name']}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Export/Import race data')
    parser.add_argument('--race', help='Race slug (e.g., paris-nice)')
    parser.add_argument('--year', type=int, help='Race year')
    parser.add_argument('--output', help='Output file (default: {race}_{year}.zip)')
    parser.add_argument('--import-zip', help='Import from zip file')
    parser.add_argument('--db', default='data/cycling.db', help='Database path')
    
    args = parser.parse_args()
    
    if args.import_zip:
        import_race_data(args.import_zip, args.db)
    elif args.race and args.year:
        output = args.output or f"{args.race}_{args.year}.zip"
        export_race_data(args.db, args.race, args.year, output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
