"""
Race Viewer - Works with Local Database
=======================================
Shows model predictions and race data without live scraping.

Usage:
    python race_viewer.py
"""
import sqlite3
import sys
import os


def get_db():
    return sqlite3.connect('data/cycling.db')


def show_races():
    """Show available races."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.id, r.display_name, r.year, r.pcs_slug,
               COUNT(DISTINCT rs.id) as stages,
               COUNT(DISTINCT sl.rider_id) as riders,
               r.startdate
        FROM races r
        LEFT JOIN race_stages rs ON rs.race_id = r.id
        LEFT JOIN startlist_entries sl ON sl.race_id = r.id
        GROUP BY r.id
        ORDER BY r.startdate DESC
        LIMIT 10
    """)
    
    races = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*70)
    print("AVAILABLE RACES")
    print("="*70)
    
    for i, race in enumerate(races, 1):
        name_clean = race[1].encode('ascii', 'ignore').decode() if race[1] else "Unknown"
        print(f"{i:2}. {name_clean} {race[2]} - {race[4]} stages, {race[5]} riders")
    
    return races


def show_stages(race_id, race_name):
    """Show stages."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, stage_number, stage_date, stage_type, distance_km
        FROM race_stages
        WHERE race_id = ?
        ORDER BY stage_number
    """, (race_id,))
    
    stages = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*70)
    print(f"STAGES FOR: {race_name}")
    print("="*70)
    
    for stage in stages:
        print(f"Stage {stage[1]}: {stage[3]} ({stage[4]}km) - {stage[2]}")
    
    return stages


def show_startlist(race_id):
    """Show startlist with specialties."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT r.name, r.nationality, t.name as team,
               r.sp_climber, r.sp_sprint, r.sp_hills, r.sp_gc
        FROM startlist_entries sl
        JOIN riders r ON sl.rider_id = r.id
        JOIN teams t ON sl.team_id = t.id
        WHERE sl.race_id = ?
        ORDER BY r.sp_hills DESC, r.sp_sprint DESC
        LIMIT 40
    """, (race_id,))
    
    riders = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*90)
    print("STARTLIST - TOP 40 BY HILLS SPECIALTY")
    print("="*90)
    print(f"{'#':<4} {'Rider':<28} {'Team':<28} {'Hill':<6} {'Spr':<6} {'GC':<6}")
    print("-"*90)
    
    for i, rider in enumerate(riders, 1):
        name = rider[0][:26].encode('ascii', 'ignore').decode() if rider[0] else ""
        team = (rider[2][:26] if rider[2] else "").encode('ascii', 'ignore').decode()
        hill = rider[4] or 0
        sprint = rider[5] or 0
        gc = rider[6] or 0
        print(f"{i:<4} {name:<28} {team:<28} {hill:<6} {sprint:<6} {gc:<6}")


def show_model_predictions(race_id, stage_id):
    """Show model predictions if available."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.name, so.win_prob, so.edge_bps
        FROM strategy_outputs so
        JOIN riders r ON so.rider_id = r.id
        WHERE so.strategy_name = 'stage_ranking'
          AND so.stage_id = ?
        ORDER BY so.win_prob DESC
        LIMIT 20
    """, (stage_id,))
    
    predictions = cursor.fetchall()
    conn.close()
    
    if not predictions:
        print("\nNo model predictions found.")
        print("Run: python rank_stage.py paris-nice 2026 1 --run-models")
        return False
    
    print("\n" + "="*90)
    print("MODEL PREDICTIONS & VALUE BETS")
    print("="*90)
    print(f"{'Rider':<30} {'Prob':<8} {'Edge':<10} {'Signal':<10}")
    print("-"*90)
    
    for pred in predictions:
        name = pred[0][:28].encode('ascii', 'ignore').decode() if pred[0] else ""
        prob = pred[1] or 0
        edge = pred[2] or 0
        
        signal = ""
        if edge > 100:
            signal = "STRONG BUY"
        elif edge > 50:
            signal = "BUY"
        elif edge > 0:
            signal = "WEAK BUY"
        
        print(f"{name:<30} {prob*100:>6.1f}% {edge:>+7.0f}bps {signal}")
    
    return True


def show_live_link(pcs_slug, year, stage_num):
    """Show PCS live link."""
    url = f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}"
    
    print("\n" + "="*70)
    print("LIVE RACE LINK")
    print("="*70)
    print(f"Open in browser: {url}")
    print("\nThe live page will show:")
    print("  - Live timing and positions")
    print("  - Results as they happen")
    print("  - Breakaway gaps")
    print("="*70)


def show_rider_analysis(race_id, pcs_slug, year, stage_num):
    """Show detailed rider analysis."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get frailty scores
    cursor.execute("""
        SELECT r.name, rf.frailty_estimate, rf.hidden_form_prob
        FROM rider_frailty rf
        JOIN riders r ON rf.rider_id = r.id
        JOIN startlist_entries sl ON sl.rider_id = r.id
        WHERE sl.race_id = ?
          AND rf.hidden_form_prob > 0.2
        ORDER BY rf.hidden_form_prob DESC
        LIMIT 10
    """, (race_id,))
    
    frailty_riders = cursor.fetchall()
    conn.close()
    
    if frailty_riders:
        print("\n" + "="*70)
        print("STRATEGY 2: RIDERS WITH HIDDEN FORM")
        print("="*70)
        print(f"{'Rider':<30} {'Frailty':<10} {'Hidden%':<10}")
        print("-"*70)
        
        for rider in frailty_riders:
            name = rider[0][:28].encode('ascii', 'ignore').decode() if rider[0] else ""
            print(f"{name:<30} {rider[1]:>8.3f} {rider[2]*100:>8.1f}%")
        
        print("\nThese riders may have been conserving energy recently.")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*70)
        print("CYCLING PREDICT - RACE VIEWER")
        print("="*70)
        print("\n1. View Race Analysis")
        print("2. Exit")
        print("\nChoice: ", end="")
        
        try:
            choice = input().strip()
        except EOFError:
            break
        
        if choice == "1":
            races = show_races()
            if not races:
                print("No races found!")
                continue
            
            print("\nSelect race number: ", end="")
            try:
                idx = int(input().strip()) - 1
                if idx < 0 or idx >= len(races):
                    continue
            except:
                continue
            
            race = races[idx]
            race_name = f"{race[1]} {race[2]}"
            
            # Show stages
            stages = show_stages(race[0], race_name)
            if not stages:
                continue
            
            print("\nSelect stage number: ", end="")
            try:
                stage_num = int(input().strip())
                stage = None
                for s in stages:
                    if s[1] == stage_num:
                        stage = s
                        break
                if not stage:
                    stage = stages[0]
                    stage_num = stage[1]
            except:
                stage = stages[0]
                stage_num = stage[1]
            
            # Show analysis
            show_startlist(race[0])
            show_model_predictions(race[0], stage[0])
            show_rider_analysis(race[0], race[3], race[2], stage_num)
            show_live_link(race[3], race[2], stage_num)
            
            print("\nPress Enter to continue...")
            try:
                input()
            except EOFError:
                break
        
        elif choice == "2" or choice == "":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
