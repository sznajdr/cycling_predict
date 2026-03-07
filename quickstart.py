"""
Quick Start Script for Cycling Predict
=====================================

This script demonstrates the complete workflow in one command.
Run this after scraping data to see the betting models in action.

Usage:
    python quickstart.py

What it does:
1. Checks your database
2. Fits Strategy 2 (Gruppetto Frailty)
3. Fits Strategy 1 (Tactical HMM)
4. Analyzes the latest stage
5. Shows top betting signals
"""
import sqlite3
import sys
from datetime import datetime

# Import our betting models
from genqirue.models import FastFrailtyEstimator, SimpleTacticalDetector
from genqirue.models.gruppetto_frailty import SurvivalRecord
from genqirue.models.tactical_hmm import TacticalObservation
from genqirue.domain import StageType


def check_database():
    """Check if database exists and has data."""
    try:
        conn = sqlite3.connect('data/cycling.db')
        cursor = conn.cursor()
        
        # Count key tables
        cursor.execute("SELECT COUNT(*) FROM races")
        num_races = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM riders")
        num_riders = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM rider_results")
        num_results = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"Database Status:")
        print(f"  - Races: {num_races}")
        print(f"  - Riders: {num_riders}")
        print(f"  - Results: {num_results}")
        
        if num_races == 0:
            print("\n⚠️  No data found! Run: python -m pipeline.runner")
            return False
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        print("Run: python -m pipeline.runner to create database")
        return False


def get_race_data(conn, race_slug='paris-nice', year=2024):
    """Get all race data for analysis."""
    
    # Get race stages
    query = """
    SELECT rs.id, rs.stage_number, rs.stage_date, rs.stage_type, rs.distance_km
    FROM race_stages rs
    JOIN races r ON rs.race_id = r.id
    WHERE r.pcs_slug = ? AND r.year = ?
    ORDER BY rs.stage_number
    """
    stages = conn.execute(query, (race_slug, year)).fetchall()
    
    # Get rider results with GC info
    query = """
    SELECT 
        rr.rider_id,
        rs.stage_number,
        rs.stage_date,
        rs.stage_type,
        rr.time_behind_winner_seconds,
        rr.rank,
        gc.time_behind_winner_seconds as gc_time_behind,
        gc.rank as gc_position
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN races r ON rs.race_id = r.id
    LEFT JOIN rider_results gc ON rr.rider_id = gc.rider_id 
        AND rr.stage_id = gc.stage_id 
        AND gc.result_category = 'gc'
    WHERE r.pcs_slug = ? AND r.year = ?
      AND rr.result_category = 'stage'
    ORDER BY rs.stage_number, rr.rider_id
    """
    results = conn.execute(query, (race_slug, year)).fetchall()
    
    return stages, results


def analyze_frailty(results):
    """Strategy 2: Gruppetto Frailty Analysis"""
    print("\n" + "="*60)
    print("STRATEGY 2: GRUPPETTO FRAILTY ANALYSIS")
    print("="*60)
    
    # Convert to survival records
    survival_records = []
    for row in results:
        rider_id, stage_num, stage_date, stage_type, time_loss, rank, gc_time, gc_pos = row
        
        gruppetto = 1 if (time_loss or 0) > 900 else 0
        
        try:
            date_obj = datetime.strptime(stage_date, '%Y-%m-%d') if stage_date else datetime.now()
        except:
            date_obj = datetime.now()
        
        record = SurvivalRecord(
            rider_id=rider_id,
            stage_id=stage_num or 0,
            stage_date=date_obj,
            stage_type=stage_type or 'road',
            time_to_cutoff=45.0,
            event_observed=False,
            gc_position=int(gc_pos) if gc_pos and str(gc_pos).isdigit() else 150,
            gc_time_behind=gc_time or time_loss or 0,
            gruppetto_indicator=gruppetto,
            gruppetto_time_loss=(time_loss or 0) - 600 if gruppetto else 0
        )
        survival_records.append(record)
    
    # Fit model
    print(f"Fitting frailty model on {len(survival_records)} observations...")
    estimator = FastFrailtyEstimator()
    estimator.fit(survival_records)
    
    print(f"✓ Analyzed {len(estimator.frailty_estimates)} riders")
    
    return estimator


def analyze_tactical_states(results):
    """Strategy 1: Tactical Time Loss Analysis"""
    print("\n" + "="*60)
    print("STRATEGY 1: TACTICAL TIME LOSS ANALYSIS")
    print("="*60)
    
    observations = []
    for row in results:
        rider_id, stage_num, stage_date, stage_type, time_loss, rank, gc_time, gc_pos = row
        
        # Map stage type
        st = stage_type or 'road'
        if st == 'mountain':
            stage_type_enum = StageType.MOUNTAIN
        elif st == 'flat':
            stage_type_enum = StageType.FLAT
        elif st == 'hilly':
            stage_type_enum = StageType.HILLY
        else:
            stage_type_enum = StageType.ROAD
        
        try:
            date_obj = datetime.strptime(stage_date, '%Y-%m-%d') if stage_date else datetime.now()
        except:
            date_obj = datetime.now()
        
        obs = TacticalObservation(
            rider_id=rider_id,
            stage_id=stage_num or 0,
            stage_type=stage_type_enum,
            stage_date=date_obj,
            time_loss_seconds=time_loss or 0,
            gc_position=int(gc_pos) if gc_pos and str(gc_pos).isdigit() else None,
            gc_time_behind=gc_time or 0
        )
        observations.append(obs)
    
    # Process with detector
    print(f"Processing {len(observations)} tactical observations...")
    detector = SimpleTacticalDetector()
    
    for obs in observations:
        detector.update(obs)
    
    print(f"✓ Processed tactical states")
    
    return detector


def find_opportunities(conn, frailty_estimator, tactical_detector, race_slug='paris-nice', year=2024):
    """Find betting opportunities."""
    print("\n" + "="*60)
    print("BETTING OPPORTUNITIES")
    print("="*60)
    
    # Get startlist for last stage
    query = """
    SELECT DISTINCT sl.rider_id, r.name
    FROM startlist_entries sl
    JOIN riders r ON sl.rider_id = r.id
    JOIN races ra ON sl.race_id = ra.id
    WHERE ra.pcs_slug = ? AND ra.year = ?
    LIMIT 20
    """
    
    riders = conn.execute(query, (race_slug, year)).fetchall()
    
    opportunities = []
    
    for rider_id, name in riders:
        signals = []
        
        # Check frailty
        frailty = frailty_estimator.get_frailty(rider_id)
        if frailty > 0.3:
            signals.append(('HIDDEN_FORM', frailty))
        
        # Check tactical preserving
        if tactical_detector.is_tactical_preserving(rider_id):
            signals.append(('TACTICAL_PRESERVE', 0.7))
        
        if signals:
            avg_signal = sum(s[1] for s in signals) / len(signals)
            opportunities.append({
                'name': name,
                'rider_id': rider_id,
                'signals': [s[0] for s in signals],
                'strength': avg_signal
            })
    
    # Sort by signal strength
    opportunities.sort(key=lambda x: -x['strength'])
    
    print(f"\nTop 10 Betting Opportunities:\n")
    print(f"{'Rider':<30} {'Signals':<40} {'Strength':<10}")
    print("-" * 80)
    
    for opp in opportunities[:10]:
        signals_str = ', '.join(opp['signals'])
        print(f"{opp['name']:<30} {signals_str:<40} {opp['strength']:.2f}")
    
    return opportunities


def main():
    """Main quickstart workflow."""
    print("="*60)
    print("CYCLING PREDICT - QUICKSTART")
    print("="*60)
    print("\nThis script will demonstrate the complete workflow.")
    print("Make sure you've already scraped data with: python -m pipeline.runner")
    print()
    
    # Step 1: Check database
    print("Step 1: Checking database...")
    if not check_database():
        sys.exit(1)
    
    # Connect to database
    conn = sqlite3.connect('data/cycling.db')
    
    # Step 2: Get data
    print("\nStep 2: Loading race data...")
    try:
        stages, results = get_race_data(conn, 'paris-nice', 2024)
        print(f"✓ Loaded {len(stages)} stages, {len(results)} results")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure you've scraped Paris-Nice 2024")
        conn.close()
        sys.exit(1)
    
    # Step 3: Run Strategy 2
    frailty_estimator = analyze_frailty(results)
    
    # Step 4: Run Strategy 1
    tactical_detector = analyze_tactical_states(results)
    
    # Step 5: Find opportunities
    opportunities = find_opportunities(conn, frailty_estimator, tactical_detector)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Analyzed {len(frailty_estimator.frailty_estimates)} riders")
    print(f"✓ Found {len(opportunities)} betting opportunities")
    print("\nNext steps:")
    print("  1. Run full analysis: python example_betting_workflow.py")
    print("  2. Create your own script: copy my_prediction.py template")
    print("  3. Add more races: edit config/races.yaml")
    print("\n" + "="*60)
    
    conn.close()


if __name__ == '__main__':
    main()
