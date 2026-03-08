"""
Simple Live Race View - Console Version
=======================================
No Streamlit needed - just prints to console.

Usage:
    python scripts/simple_live_view.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup


def get_db():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DB_PATH = PROJECT_ROOT / 'data' / 'cycling.db'
    return sqlite3.connect(DB_PATH)


def show_race_picker():
    """Show available races."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.id, r.display_name, r.year, r.pcs_slug,
               COUNT(DISTINCT rs.id) as stages,
               COUNT(DISTINCT sl.rider_id) as riders
        FROM races r
        LEFT JOIN race_stages rs ON rs.race_id = r.id
        LEFT JOIN startlist_entries sl ON sl.race_id = r.id
        GROUP BY r.id
        ORDER BY r.startdate DESC
        LIMIT 10
    """)
    
    races = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*60)
    print("AVAILABLE RACES")
    print("="*60)
    
    for i, race in enumerate(races, 1):
        print(f"{i}. {race[1].encode('ascii', errors='ignore').decode()} {race[2]} - {race[4]} stages, {race[5]} riders")
    
    print("\nEnter race number (or 0 to exit): ", end="")
    choice = input().strip()
    
    try:
        idx = int(choice) - 1
        if idx < 0:
            return None
        return races[idx]
    except:
        return races[0] if races else None


def show_stages(race_id, race_name):
    """Show stages for selected race."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, stage_number, stage_date, stage_type, distance_km, pcs_stage_url
        FROM race_stages
        WHERE race_id = ?
        ORDER BY stage_number
    """, (race_id,))
    
    stages = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*60)
    print(f"STAGES for {race_name}")
    print("="*60)
    
    for stage in stages:
        print(f"Stage {stage[1]}: {stage[3]} ({stage[4]}km) - {stage[2]}")
    
    print("\nEnter stage number to monitor: ", end="")
    choice = input().strip()
    
    try:
        stage_num = int(choice)
        for stage in stages:
            if stage[1] == stage_num:
                return stage
        return stages[0] if stages else None
    except:
        return stages[0] if stages else None


def scrape_live(pcs_slug, year, stage_num):
    """Scrape live results."""
    url = f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for live indicator
        page_text = soup.get_text().lower()
        is_live = 'live' in page_text or 'live timing' in page_text
        
        # Extract results
        results = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:20]:  # Top 20
                cols = row.find_all('td')
                if len(cols) >= 3:
                    rank = cols[0].get_text(strip=True)
                    rider = cols[1].get_text(strip=True)
                    time = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                    if rank and rider:
                        results.append((rank, rider, time))
        
        return {
            'is_live': is_live,
            'url': url,
            'results': results[:20],
            'timestamp': datetime.now()
        }, None
        
    except Exception as e:
        return None, str(e)


def show_live_race(pcs_slug, year, stage_num, race_name):
    """Show live race view."""
    import os
    import time
    
    print("\n" + "="*60)
    print(f"📊 {race_name} - Stage {stage_num} LIVE VIEW")
    print("="*60)
    print("\nPress Ctrl+C to stop, or wait for auto-refresh\n")
    
    try:
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "="*60)
            print(f"📊 {race_name} - Stage {stage_num}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            # Scrape live data
            live_data, error = scrape_live(pcs_slug, year, stage_num)
            
            if error:
                print(f"\n⚠️  Error: {error}")
                print(f"URL: https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}")
                if "403" in str(error):
                    print("\nNOTE: PCS is blocking automated requests (Cloudflare protection)")
                    print("📌 Open the URL manually in your browser for live timing")
            elif live_data:
                if live_data['is_live']:
                    print("\n*** RACE IS LIVE! ***")
                else:
                    print("\nRace not live yet or finished")
                
                print(f"\nLive Results (Top 20):")
                print("-"*60)
                print(f"{'Rank':<6} {'Rider':<40} {'Time':<10}")
                print("-"*60)
                
                if live_data['results']:
                    for rank, rider, time in live_data['results']:
                        print(f"{rank:<6} {rider:<40} {time:<10}")
                else:
                    print("No results data available yet...")
                
                print(f"\nRefreshing in 30 seconds...")
                print("Press Ctrl+C to stop")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Stopped monitoring")


def show_startlist(race_id):
    """Show race startlist."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT r.name, r.nationality, t.name as team,
               r.sp_climber, r.sp_sprint, r.sp_hills
        FROM startlist_entries sl
        JOIN riders r ON sl.rider_id = r.id
        JOIN teams t ON sl.team_id = t.id
        WHERE sl.race_id = ?
        ORDER BY r.sp_hills DESC, r.sp_sprint DESC
        LIMIT 30
    """, (race_id,))
    
    riders = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*80)
    print("👥 STARTLIST (Top 30 by profile)")
    print("="*80)
    print(f"{'Rider':<30} {'Team':<25} {'Hills':<8} {'Sprint':<8}")
    print("-"*80)
    
    for rider in riders:
        name = rider[0][:28].encode('ascii', errors='ignore').decode()
        team = (rider[2][:23] if rider[2] else "").encode('ascii', errors='ignore').decode()
        hills = rider[4] or 0
        sprint = rider[5] or 0
        print(f"{name:<30} {team:<25} {hills:<8} {sprint:<8}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*60)
        print("CYCLING PREDICT - LIVE CONSOLE VIEW")
        print("="*60)
        print("\n1. Pick Race and Watch Live")
        print("2. View Startlist")
        print("3. Exit")
        print("\nChoice: ", end="")
        
        choice = input().strip()
        
        if choice == "1":
            race = show_race_picker()
            if race:
                stage = show_stages(race[0], f"{race[1]} {race[2]}")
                if stage:
                    show_live_race(race[3], race[2], stage[1], f"{race[1]} {race[2]}")
        
        elif choice == "2":
            race = show_race_picker()
            if race:
                show_startlist(race[0])
                print("\nPress Enter to continue...")
                input()
        
        elif choice == "3" or choice == "0":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
