"""
Stage 1 Paris-Nice 2026 - Quick Value Bet Finder
================================================

Run this script to get value bet recommendations for today's stage.

Usage:
    python analyze_stage1_pn2026.py
"""
import sqlite3
import sys
from datetime import datetime

# Try to import models
try:
    from genqirue.models import FastFrailtyEstimator, SimpleTacticalDetector
    from genqirue.models.gruppetto_frailty import SurvivalRecord
    from genqirue.models.tactical_hmm import TacticalObservation
    from genqirue.domain import StageType
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False


def check_database():
    """Check if Paris-Nice 2026 data exists."""
    conn = sqlite3.connect('data/cycling.db')
    cursor = conn.cursor()
    
    print("="*60)
    print("STAGE 1 PARIS-NICE 2026 - VALUE BET ANALYSIS")
    print("="*60)
    print()
    
    # Check race
    race = cursor.execute(
        'SELECT id, display_name FROM races WHERE pcs_slug = ? AND year = ?', 
        ('paris-nice', 2026)
    ).fetchone()
    
    if not race:
        print("[ERROR] Paris-Nice 2026 not found in database!")
        print("   Run: python -m pipeline.runner")
        conn.close()
        return None
    
    print(f"[OK] Race: {race[1]} 2026")
    
    # Check startlist
    startlist_count = cursor.execute(
        'SELECT COUNT(*) FROM startlist_entries WHERE race_id = ?', 
        (race[0],)
    ).fetchone()[0]
    print(f"[OK] Startlist: {startlist_count} riders")
    
    # Check Stage 1
    stage = cursor.execute(
        '''SELECT rs.id, rs.stage_number, rs.stage_type, rs.distance_km 
           FROM race_stages rs 
           WHERE rs.race_id = ? AND rs.stage_number = ?''', 
        (race[0], 1)
    ).fetchone()
    
    if not stage:
        print("[ERROR] Stage 1 not found!")
        conn.close()
        return None
    
    print(f"[OK] Stage 1: {stage[2]}, {stage[3]}km")
    print()
    
    return conn, race[0], stage[0]


def get_stage1_startlist(conn, race_id):
    """Get startlist with rider details."""
    query = '''
    SELECT DISTINCT 
        sl.rider_id,
        r.name,
        r.nationality,
        r.sp_climber,
        r.sp_sprint,
        r.sp_hills,
        r.sp_gc,
        t.name as team_name
    FROM startlist_entries sl
    JOIN riders r ON sl.rider_id = r.id
    JOIN teams t ON sl.team_id = t.id
    WHERE sl.race_id = ?
    ORDER BY r.sp_hills DESC, r.sp_sprint DESC
    LIMIT 50
    '''
    
    return conn.execute(query, (race_id,)).fetchall()


def analyze_specialty_scores(riders):
    """Analyze which riders have the best profile for Stage 1."""
    print("="*60)
    print("SPECIALTY ANALYSIS (Best Profile for Stage 1)")
    print("="*60)
    print()
    print(f"{'Rider':<30} {'Hills':<8} {'Sprint':<8} {'Team':<25}")
    print("-"*70)
    
    # Stage 1: Need hilly + sprint (puncheur/sprinter)
    scored_riders = []
    for rider in riders:
        name, hills, sprint = rider[1], rider[3] or 0, rider[4] or 0
        team = rider[7] or "Unknown"
        # Score = hills * 0.6 + sprint * 0.4 (hilly stage with sprint finish)
        score = hills * 0.6 + sprint * 0.4
        scored_riders.append((name, hills, sprint, team, score))
    
    # Sort by combined score
    scored_riders.sort(key=lambda x: -x[4])
    
    for name, hills, sprint, team, score in scored_riders[:15]:
        name_clean = name.encode('ascii', errors='ignore').decode('ascii')
        team_clean = team[:25].encode('ascii', errors='ignore').decode('ascii')
        print(f"{name_clean:<30} {hills:<8} {sprint:<8} {team_clean:<25}")
    
    print()
    return scored_riders[:15]


def check_historical_pn_performance(conn, rider_ids):
    """Check historical Paris-Nice performance for these riders."""
    print("="*60)
    print("HISTORICAL PARIS-NICE PERFORMANCE")
    print("="*60)
    print()
    
    if not rider_ids:
        print("No riders to check")
        return
    
    placeholders = ','.join('?' * len(rider_ids))
    query = f'''
    SELECT r.name,
           COUNT(*) as stages,
           AVG(CAST(rr.rank AS FLOAT)) as avg_rank,
           MIN(CAST(rr.rank AS INTEGER)) as best_rank,
           SUM(CASE WHEN CAST(rr.rank AS INTEGER) <= 10 THEN 1 ELSE 0 END) as top10s
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN riders r ON rr.rider_id = r.id
    JOIN races ra ON rs.race_id = ra.id
    WHERE ra.pcs_slug = 'paris-nice'
      AND rr.rider_id IN ({placeholders})
      AND rr.result_category = 'stage'
    GROUP BY rr.rider_id
    HAVING stages >= 2
    ORDER BY avg_rank ASC
    LIMIT 10
    '''
    
    results = conn.execute(query, rider_ids).fetchall()
    
    if not results:
        print("No significant Paris-Nice history found for these riders")
        print()
        return
    
    print(f"{'Rider':<25} {'Stages':<8} {'Avg':<8} {'Best':<8} {'Top10s':<8}")
    print("-"*70)
    
    for name, stages, avg_rank, best, top10s in results:
        avg_rank = avg_rank if avg_rank else 999
        name_clean = name.encode('ascii', errors='ignore').decode('ascii')
        print(f"{name_clean:<25} {stages:<8} {avg_rank:<8.1f} {best:<8} {top10s:<8}")
    
    print()


def check_odds_availability(conn):
    """Check if Betclic odds are available."""
    print("="*60)
    print("ODDS AVAILABILITY")
    print("="*60)
    print()
    
    # Check if odds table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bookmaker_odds'")
    if not cursor.fetchone():
        print("[WARN] Odds table not found. Run: python fetch_odds.py --init-schema")
        print()
        return False
    
    # Check latest odds
    latest = conn.execute('''
        SELECT COUNT(*) as count, MAX(scraped_at) as last_update
        FROM bookmaker_odds_latest
        WHERE market_type = 'winner'
    ''').fetchone()
    
    if latest[0] == 0:
        print("[WARN] No odds data found. Run: python fetch_odds.py")
        print()
        return False
    
    print(f"✓ Odds available: {latest[0]} selections")
    print(f"  Last update: {latest[1]}")
    print()
    
    # Show top 10 favorites
    print("Top 10 by odds:")
    print(f"{'Rider':<30} {'Back Odds':<12} {'Fair Odds':<12}")
    print("-"*55)
    
    for row in conn.execute('''
        SELECT participant_name, back_odds, fair_odds
        FROM bookmaker_odds_latest
        WHERE market_type = 'winner'
        ORDER BY back_odds ASC
        LIMIT 10
    '''):
        print(f"{row[0]:<30} {row[1]:<12.2f} {row[2]:<12.2f}")
    
    print()
    return True


def run_model_analysis(conn, race_id):
    """Run Strategy 1 and Strategy 2 models."""
    if not MODELS_AVAILABLE:
        print("[WARN] Models not available - skipping model analysis")
        print()
        return
    
    print("="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    print()
    
    # Get historical data for Paris-Nice
    query = '''
    SELECT rr.rider_id, rs.stage_number, rs.stage_type,
           rr.time_behind_winner_seconds, rr.rank, rs.stage_date
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN races ra ON rs.race_id = ra.id
    WHERE ra.pcs_slug = 'paris-nice'
      AND ra.year IN (2022, 2023, 2024, 2025)
      AND rr.result_category = 'stage'
    ORDER BY rs.stage_date, rr.rider_id
    '''
    
    results = conn.execute(query).fetchall()
    
    if len(results) < 50:
        print(f"⚠️  Only {len(results)} historical results found")
        print("   Models may be less accurate with limited data")
        print()
    
    # Prepare survival records for Strategy 2
    survival_records = []
    for row in results:
        rider_id, stage_num, stage_type, time_loss, rank, stage_date = row
        gruppetto = 1 if (time_loss or 0) > 900 else 0
        
        try:
            from datetime import datetime
            date_obj = datetime.strptime(stage_date, '%Y-%m-%d') if stage_date else datetime.now()
        except:
            from datetime import datetime
            date_obj = datetime.now()
        
        record = SurvivalRecord(
            rider_id=rider_id,
            stage_id=stage_num or 0,
            stage_date=date_obj,
            stage_type=stage_type or 'road',
            time_to_cutoff=45.0,
            event_observed=False,
            gc_position=int(rank) if rank and str(rank).isdigit() else 150,
            gc_time_behind=time_loss or 0,
            gruppetto_indicator=gruppetto,
            gruppetto_time_loss=(time_loss or 0) - 600 if gruppetto else 0
        )
        survival_records.append(record)
    
    # Fit frailty model
    print("Fitting Strategy 2 (Gruppetto Frailty)...")
    estimator = FastFrailtyEstimator()
    estimator.fit(survival_records)
    print(f"[OK] Analyzed {len(estimator.frailty_estimates)} riders")
    print()
    
    # Get top riders from startlist
    startlist = conn.execute('''
        SELECT DISTINCT sl.rider_id, r.name
        FROM startlist_entries sl
        JOIN riders r ON sl.rider_id = r.id
        WHERE sl.race_id = ?
    ''', (race_id,)).fetchall()
    
    # Check for hidden form
    print("RIDERS WITH HIDDEN FORM (High Frailty Score):")
    print(f"{'Rider':<30} {'Frailty':<10}")
    print("-"*45)
    
    hidden_form_riders = []
    for rider_id, name in startlist:
        frailty = estimator.get_frailty(rider_id)
        if frailty > 0.3:
            hidden_form_riders.append((name, frailty))
    
    hidden_form_riders.sort(key=lambda x: -x[1])
    for name, frailty in hidden_form_riders[:10]:
        name_clean = name.encode('ascii', errors='ignore').decode('ascii')
        print(f"{name_clean:<30} {frailty:<10.3f}")
    
    print()


def print_recommendations():
    """Print betting recommendations based on analysis."""
    print("="*60)
    print("BETTING RECOMMENDATIONS")
    print("="*60)
    print()
    
    print("STAGE 1 PROFILE: Hilly finish, Côte de Chanteloup-les-Vignes (1.1km @ 8.3%)")
    print("Expected outcome: Reduced bunch sprint or late attack")
    print()
    
    print("RIDER TYPES TO TARGET:")
    print("  + Punchy sprinters (high sp_sprint + decent sp_hills)")
    print("  + Puncheurs (high sp_hills)")
    print("  + Riders with hidden form (high frailty score)")
    print("  + Riders with good Paris-Nice history")
    print()
    
    print("RIDERS TO AVOID:")
    print("  - Pure sprinters (will be dropped on the climb)")
    print("  - Pure climbers (finish not hard enough)")
    print("  - GC favorites early in race (conservative)")
    print()
    
    print("NEXT STEPS:")
    print("  1. Run: python fetch_odds.py (to get live odds)")
    print("  2. Run: python rank_stage.py paris-nice 2026 1 --run-models")
    print("  3. Check for edge > 50bps in output")
    print("  4. Place quarter-Kelly stakes on top 3 value opportunities")
    print()


def main():
    """Main analysis function."""
    # Check database
    result = check_database()
    if not result:
        sys.exit(1)
    
    conn, race_id, stage_id = result
    
    # Get startlist
    riders = get_stage1_startlist(conn, race_id)
    print(f"Analyzing top {len(riders)} riders from startlist...")
    print()
    
    # Analyze specialty scores
    top_riders = analyze_specialty_scores(riders)
    
    # Check historical performance
    rider_ids = [r[0] for r in riders]
    check_historical_pn_performance(conn, rider_ids[:20])
    
    # Check odds
    odds_available = check_odds_availability(conn)
    
    # Run models
    run_model_analysis(conn, race_id)
    
    # Print recommendations
    print_recommendations()
    
    conn.close()
    
    print("="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
