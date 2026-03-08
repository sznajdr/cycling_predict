"""
Example workflow: Using Genqirue betting engine with scraped cycling data.

This demonstrates how to:
1. Load data from the existing cycling.db
2. Fit Strategy 2 (Gruppetto Frailty) - the critical first model
3. Get predictions for an upcoming stage
4. Optimize portfolio with Kelly criterion
5. Generate betting recommendations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Genqirue imports
from genqirue.models import (
    GruppettoFrailtyModel,
    FastFrailtyEstimator,
    TacticalTimeLossHMM,
    SimpleTacticalDetector,
    BayesianChangepointDetector,
)
from genqirue.models.gruppetto_frailty import SurvivalRecord
from genqirue.models.tactical_hmm import TacticalObservation
from genqirue.portfolio import RobustKellyOptimizer, KellyParameters
from genqirue.domain import (
    StageType,
    TacticalState,
    MarketState,
    Position,
    RiderState,
)


def load_survival_data_from_db(
    conn: sqlite3.Connection,
    race_slug: str = 'paris-nice',
    year: int = 2024
) -> Dict[str, Any]:
    """
    Load survival data from cycling.db for Gruppetto Frailty model.
    
    Creates survival records from rider_results and race_stages.
    """
    query = """
    SELECT 
        rr.rider_id,
        rr.stage_id,
        rs.stage_number,
        rs.stage_date,
        rs.stage_type,
        rr.rank,
        rr.time_behind_winner_seconds,
        rr.result_category,
        r.pcs_slug as rider_slug
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN races r ON rs.race_id = r.id
    WHERE r.pcs_slug = ?
      AND r.year = ?
      AND rr.result_category = 'stage'
    ORDER BY rs.stage_date, rr.rank
    """
    
    df = pd.read_sql_query(query, conn, params=(race_slug, year))
    
    if df.empty:
        print(f"No data found for {race_slug} {year}")
        return {'survival_data': [], 'rider_ids': []}
    
    survival_data = []
    
    for _, row in df.iterrows():
        # Infer gruppetto status from time loss
        # Gruppetto typically forms when time loss > 15-20 minutes on mountains
        time_loss = row['time_behind_winner_seconds'] or 0
        gruppetto = time_loss > 900  # 15 minutes
        
        # Infer GC position from rank
        try:
            gc_pos = int(row['rank']) if row['rank'] not in ['DNF', 'DNS', 'DSQ', 'OTL'] else 150
        except:
            gc_pos = 150
        
        record = SurvivalRecord(
            rider_id=row['rider_id'],
            stage_id=row['stage_id'],
            stage_date=datetime.strptime(row['stage_date'], '%Y-%m-%d'),
            stage_type=row['stage_type'] or 'road',
            time_to_cutoff=45.0,  # Assume 45 min time cut
            event_observed=False,  # Simplified: assume all finished
            gc_position=gc_pos,
            gc_time_behind=time_loss,
            gruppetto_indicator=1 if gruppetto else 0,
            gruppetto_time_loss=time_loss - 600 if gruppetto else 0
        )
        survival_data.append(record)
    
    rider_ids = df['rider_id'].unique().tolist()
    
    return {
        'survival_data': survival_data,
        'rider_ids': rider_ids
    }


def load_tactical_data_from_db(
    conn: sqlite3.Connection,
    race_slug: str = 'paris-nice',
    year: int = 2024
) -> Dict[str, Any]:
    """
    Load tactical time loss data for HMM model.
    """
    query = """
    SELECT 
        rr.rider_id,
        rr.stage_id,
        rs.stage_number,
        rs.stage_date,
        rs.stage_type,
        rr.time_behind_winner_seconds,
        gc.rank as gc_position,
        gc.time_behind_winner_seconds as gc_time_behind
    FROM rider_results rr
    JOIN race_stages rs ON rr.stage_id = rs.id
    JOIN races r ON rs.race_id = r.id
    LEFT JOIN rider_results gc ON rr.rider_id = gc.rider_id 
        AND rr.stage_id = gc.stage_id 
        AND gc.result_category = 'gc'
    WHERE r.pcs_slug = ?
      AND r.year = ?
      AND rr.result_category = 'stage'
    ORDER BY rs.stage_date, rr.rider_id
    """
    
    df = pd.read_sql_query(query, conn, params=(race_slug, year))
    
    if df.empty:
        return {'observations': [], 'rider_ids': []}
    
    observations = []
    
    for _, row in df.iterrows():
        time_loss = row['time_behind_winner_seconds'] or 0
        
        # Map stage type string to enum
        stage_type_str = row['stage_type'] or 'road'
        if stage_type_str == 'mountain':
            stage_type = StageType.MOUNTAIN
        elif stage_type_str == 'flat':
            stage_type = StageType.FLAT
        elif stage_type_str == 'hilly':
            stage_type = StageType.HILLY
        else:
            stage_type = StageType.ROAD
        
        obs = TacticalObservation(
            rider_id=row['rider_id'],
            stage_id=row['stage_id'],
            stage_type=stage_type,
            stage_date=datetime.strptime(row['stage_date'], '%Y-%m-%d'),
            time_loss_seconds=time_loss,
            gc_position=row['gc_position'],
            gc_time_behind=row['gc_time_behind'] or 0
        )
        observations.append(obs)
    
    rider_ids = df['rider_id'].unique().tolist()
    
    return {
        'observations': observations,
        'rider_ids': rider_ids
    }


def get_upcoming_stage_startlist(
    conn: sqlite3.Connection,
    race_slug: str = 'paris-nice',
    year: int = 2024,
    stage_number: int = 5
) -> List[Dict[str, Any]]:
    """
    Get startlist for an upcoming stage with relevant data.
    """
    query = """
    SELECT 
        sl.rider_id,
        r.name as rider_name,
        r.nationality,
        r.birthdate,
        t.pcs_url as team_url,
        rs.stage_type,
        rs.profile_score,
        rs.vertical_m,
        rs.distance_km,
        -- Get previous stage GC position
        (SELECT rr.rank 
         FROM rider_results rr 
         JOIN race_stages rs2 ON rr.stage_id = rs2.id
         WHERE rr.rider_id = sl.rider_id 
           AND rs2.stage_number = ? - 1
           AND rr.result_category = 'gc'
         LIMIT 1) as prev_gc_rank,
        (SELECT rr.time_behind_winner_seconds
         FROM rider_results rr 
         JOIN race_stages rs2 ON rr.stage_id = rs2.id
         WHERE rr.rider_id = sl.rider_id 
           AND rs2.stage_number = ? - 1
           AND rr.result_category = 'gc'
         LIMIT 1) as prev_gc_time_behind
    FROM startlist_entries sl
    JOIN riders r ON sl.rider_id = r.id
    JOIN teams t ON sl.team_id = t.id
    JOIN race_stages rs ON sl.race_id = rs.race_id
    JOIN races ra ON sl.race_id = ra.id
    WHERE ra.pcs_slug = ?
      AND ra.year = ?
      AND rs.stage_number = ?
    """
    
    df = pd.read_sql_query(
        query, 
        conn, 
        params=(stage_number, stage_number, race_slug, year, stage_number)
    )
    
    return df.to_dict('records')


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'cycling.db'


def _lookup_real_odds(rider_id: int, market_type: str, db_path=DB_PATH) -> float | None:
    """Return latest back_odds from bookmaker_odds_latest, or None if unavailable."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("""
                SELECT bo.back_odds FROM bookmaker_odds_latest bo
                JOIN riders r ON (
                    LOWER(bo.participant_name) = LOWER(r.name)
                    OR LOWER(bo.participant_name_norm) = LOWER(
                        REPLACE(REPLACE(REPLACE(r.name,'ä','a'),'ö','o'),'ü','u'))
                )
                WHERE r.id = ? AND bo.market_type = ?
                ORDER BY bo.scraped_at DESC LIMIT 1
            """, (rider_id, market_type)).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


def analyze_stage(
    conn: sqlite3.Connection,
    race_slug: str = 'paris-nice',
    year: int = 2024,
    stage_number: int = 5
) -> None:
    """
    Complete stage analysis using all available models.
    """
    print("=" * 70)
    print(f"GENQIRUE STAGE ANALYSIS: {race_slug.upper()} {year} Stage {stage_number}")
    print("=" * 70)
    
    # === STEP 1: Fit Gruppetto Frailty Model (Strategy 2) ===
    print("\n[1] Fitting Gruppetto Frailty Model (Strategy 2)...")
    
    survival_data = load_survival_data_from_db(conn, race_slug, year)
    
    if survival_data['survival_data']:
        # Use fast estimator for demo
        frailty_estimator = FastFrailtyEstimator()
        frailty_estimator.fit(survival_data['survival_data'])
        print(f"    [OK] Fitted frailty for {len(frailty_estimator.frailty_estimates)} riders")
    else:
        print("    ! No survival data available")
        frailty_estimator = None
    
    # === STEP 2: Fit Tactical HMM (Strategy 1) ===
    print("\n[2] Fitting Tactical Time Loss HMM (Strategy 1)...")
    
    tactical_data = load_tactical_data_from_db(conn, race_slug, year)
    
    if len(tactical_data['observations']) > 10:
        # Use simple detector for demo (full HMM requires more data)
        tactical_detector = SimpleTacticalDetector()
        for obs in tactical_data['observations']:
            tactical_detector.update(obs)
        print(f"    [OK] Processed {len(tactical_data['observations'])} tactical observations")
    else:
        print("    ! Insufficient tactical data")
        tactical_detector = None
    
    # === STEP 3: Analyze Upcoming Stage ===
    print(f"\n[3] Analyzing Stage {stage_number} Startlist...")
    
    startlist = get_upcoming_stage_startlist(conn, race_slug, year, stage_number)
    print(f"    Found {len(startlist)} riders in startlist")
    
    # === STEP 4: Generate Signals and Create Positions ===
    print("\n[4] Generating Trading Signals...")
    
    positions = []
    signals_found = []
    
    for rider in startlist:
        rider_id = rider['rider_id']
        signals = []
        
        # Signal 2A: Hidden form (Gruppetto Frailty)
        if frailty_estimator:
            frailty = frailty_estimator.get_frailty(rider_id)
            if frailty > 0.5:
                signals.append(('HIDDEN_FORM', frailty))
        
        # Signal 1A: Tactical preserving
        if tactical_detector and tactical_detector.is_tactical_preserving(rider_id):
            signals.append(('TACTICAL_PRESERVE', 0.7))
        
        # Check if rider was in gruppetto on previous mountain stage
        # and now on flat/hilly stage - classic transition setup
        stage_type = rider.get('stage_type', 'road')
        prev_gc_time = rider.get('prev_gc_time_behind') or 0
        
        if prev_gc_time > 900 and stage_type in ['flat', 'hilly']:
            signals.append(('TRANSITION_STAGE', 0.6))
        
        if signals:
            signals_found.append({
                'rider_id': rider_id,
                'name': rider['rider_name'],
                'team': rider['team_url'],
                'signals': signals,
                'stage_type': stage_type
            })
    
    print(f"    Found signals for {len(signals_found)} riders")
    
    # === STEP 5: Create Positions for Strong Signals ===
    print("\n[5] Creating Positions...")
    
    for signal_data in signals_found:
        # Calculate combined signal strength
        avg_signal = np.mean([s[1] for s in signal_data['signals']])
        
        # Only trade strong signals
        if avg_signal > 0.5:
            # Estimate win probability from signal
            # Base rate for stage winner ~ 1/150, adjust by signal
            base_prob = 0.0067  # ~1/150
            adjusted_prob = min(0.25, base_prob * (1 + avg_signal * 3))
            
            # Use real bookmaker odds when available, fall back to random simulation
            market_odds = _lookup_real_odds(signal_data['rider_id'], 'winner') \
                          or (20.0 + np.random.exponential(10))
            
            market = MarketState(
                market_type='winner',
                selection_id=signal_data['rider_id'],
                back_odds=market_odds,
                model_prob=adjusted_prob,
                model_prob_uncertainty=adjusted_prob * 0.3
            )
            
            pos = Position(
                market_state=market,
                originating_strategy='+'.join([s[0] for s in signal_data['signals']]),
                confidence='high' if avg_signal > 0.7 else 'medium'
            )
            
            positions.append(pos)
            
            rider_name = signal_data['name'].encode('ascii', errors='ignore').decode('ascii')
            print(f"    + {rider_name}: "
                  f"prob={adjusted_prob:.1%}, "
                  f"odds={market_odds:.1f}, "
                  f"edge={market.edge_bps:.0f}bps, "
                  f"signals={[s[0] for s in signal_data['signals']]}")
    
    # === STEP 6: Portfolio Optimization ===
    print(f"\n[6] Optimizing Portfolio ({len(positions)} positions)...")
    
    if positions:
        params = KellyParameters(
            method='half_kelly',
            max_position_pct=0.10,  # Conservative: max 10% per rider
            min_edge_bps=50  # Minimum 0.5% edge
        )
        optimizer = RobustKellyOptimizer(params)
        
        # Create team assignments for correlation handling
        team_assignments = {
            s['rider_id']: hash(s['team']) % 1000  # Simple team ID
            for s in signals_found
        }
        
        portfolio = optimizer.optimize_portfolio(positions, team_assignments)
        
        print(f"    Total Stake: {portfolio.total_stake:.1%}")
        print(f"    Expected Return: {portfolio.expected_return:.2%}")
        print(f"    Portfolio Variance: {portfolio.portfolio_variance:.4f}")
        print(f"    CVaR (95%): {portfolio.cvar_95:.2%}")
        
        print("\n    RECOMMENDED POSITIONS:")
        for pos in sorted(portfolio.positions, key=lambda p: -p.stake):
            if pos.stake > 0.001:  # Only show non-zero stakes
                print(f"      • Rider {pos.market_state.selection_id}: "
                      f"stake={pos.stake:.1%} ({pos.half_kelly_fraction:.1%} Kelly), "
                      f"strategy={pos.originating_strategy}")
    else:
        print("    No positions meeting criteria")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


def main():
    """Main entry point."""
    db_path = 'data/cycling.db'
    
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        
        # Run analysis for Paris-Nice 2024 Stage 5
        analyze_stage(conn, 'paris-nice', 2024, stage_number=5)
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        print("\nNote: This example requires data from the scraping pipeline.")
        print("Run 'python -m pipeline.runner' first to populate cycling.db")
        
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    main()
