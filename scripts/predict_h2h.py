#!/usr/bin/env python3
"""
Head-to-Head (H2H) Prediction Script for Cycling Stages

Usage:
    python scripts/predict_h2h.py tirreno-adriatico 2026 1
    
Then enter matchups in format: "Rider A vs. Rider B"

Example:
    Antonio Tiberi vs. Matteo Jorgenson
    Filippo Ganna vs. Ethan Hayter
    done
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_rider_probabilities(conn, race_slug, year, stage_number):
    """Get model win probabilities for all riders in a stage."""
    cursor = conn.execute('''
        SELECT r.name, so.win_prob, so.edge_bps, so.latent_states_json
        FROM strategy_outputs so
        JOIN riders r ON so.rider_id = r.id
        JOIN race_stages rs ON so.stage_id = rs.id
        JOIN races ra ON rs.race_id = ra.id
        WHERE so.strategy_name = 'stage_ranking'
          AND ra.pcs_slug = ?
          AND ra.year = ?
          AND rs.stage_number = ?
    ''', (race_slug, year, stage_number))
    
    probs = {}
    for row in cursor.fetchall():
        probs[row[0].lower().strip()] = {
            'name': row[0],
            'win_prob': row[1],
            'edge_bps': row[2],
            'signals': row[3]
        }
    return probs


def normalize_name(name):
    """Normalize rider name for matching."""
    return name.lower().strip().replace('  ', ' ')


def find_rider(probs, name_query):
    """Find rider in database with fuzzy matching."""
    name_norm = normalize_name(name_query)
    
    # Exact match
    if name_norm in probs:
        return probs[name_norm]
    
    # Partial match
    for key, data in probs.items():
        if name_norm in key or key in name_norm:
            return data
        # Check last name only
        if name_norm.split()[-1] in key:
            return data
    
    return None


def calculate_h2h_prob(prob_a, prob_b):
    """
    Calculate probability that A beats B in a H2H matchup.
    
    Formula: P(A beats B) = P(A wins) / (P(A wins) + P(B wins))
    
    This assumes the H2H is conditional on either A or B winning
    (ignoring other riders).
    """
    if prob_a + prob_b == 0:
        return 0.5
    return prob_a / (prob_a + prob_b)


def format_percent(p):
    """Format percentage with indicators."""
    if p >= 0.60:
        return f"{p*100:5.1f}% [STRONG]"
    elif p >= 0.55:
        return f"{p*100:5.1f}% [FAVORITE]"
    elif p <= 0.40:
        return f"{p*100:5.1f}% [UNDERDOG]"
    else:
        return f"{p*100:5.1f}% [TOSSUP]"


def analyze_h2h(probs, rider_a_name, rider_b_name):
    """Analyze a single H2H matchup."""
    rider_a = find_rider(probs, rider_a_name)
    rider_b = find_rider(probs, rider_b_name)
    
    if not rider_a:
        return f"[ERROR] Rider not found: {rider_a_name}"
    if not rider_b:
        return f"[ERROR] Rider not found: {rider_b_name}"
    
    prob_a = rider_a['win_prob']
    prob_b = rider_b['win_prob']
    
    h2h_prob = calculate_h2h_prob(prob_a, prob_b)
    
    # Calculate fair odds
    fair_odds_a = 1 / h2h_prob if h2h_prob > 0 else 999
    fair_odds_b = 1 / (1 - h2h_prob) if h2h_prob < 1 else 999
    
    result = f"""
+------------------------------------------------------------------+
|  H2H MATCHUP: {rider_a['name']:<25} vs {rider_b['name']:<26}|
+------------------------------------------------------------------+
|  Model Win Probabilities:                                        |
|    {rider_a['name']:<30} {prob_a*100:5.1f}%                       |
|    {rider_b['name']:<30} {prob_b*100:5.1f}%                       |
+------------------------------------------------------------------+
|  H2H Probability (A beats B):  {format_percent(h2h_prob)}         |
|  Fair H2H Odds:                                                |
|    {rider_a['name']:<30} @{fair_odds_a:5.2f}                      |
|    {rider_b['name']:<30} @{fair_odds_b:5.2f}                      |
+------------------------------------------------------------------+
"""
    return result


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("Usage: python scripts/predict_h2h.py <race-slug> <year> <stage-number>")
        print("\nExample:")
        print("  python scripts/predict_h2h.py tirreno-adriatico 2026 1")
        sys.exit(1)
    
    race_slug = sys.argv[1]
    year = int(sys.argv[2])
    stage_num = int(sys.argv[3])
    
    conn = sqlite3.connect('data/cycling.db')
    
    print(f"Loading model probabilities for {race_slug} {year} Stage {stage_num}...")
    probs = get_rider_probabilities(conn, race_slug, year, stage_num)
    
    if not probs:
        print(f"[ERROR] No model data found. Run: python scripts/rank_stage.py {race_slug} {year} {stage_num} --run-models --save")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(probs)} riders")
    print("\nEnter H2H matchups in format: 'Rider A vs. Rider B'")
    print("Type 'done' or press Ctrl+C when finished\n")
    
    # Pre-defined matchups for Tirreno Stage 1
    default_matchups = [
        "Antonio Tiberi vs. Matteo Jorgenson",
        "Filippo Ganna vs. Ethan Hayter",
        "Giulio Pellizzari vs. Santiago Buitrago",
        "Huub Artz vs. Jan Tratnik",
        "Ilan Van Wilder vs. Isaac Del Toro",
        "Jan Christen vs. Ben Healy",
        "Jonathan Milan vs. Soren Waerenskjold",
        "Max Walscheid vs. Felix Großschartner",
        "Pello Bilbao vs. Jai Hindley",
        "Primoz Roglic vs. Thymen Arensman",
        "Wout Van Aert vs. Mathieu Van Der Poel",
    ]
    
    print("Default matchups for Tirreno Stage 1:")
    for i, m in enumerate(default_matchups, 1):
        print(f"  {i}. {m}")
    print("\nPress Enter to analyze these, or type custom matchups:")
    
    matchups = []
    while True:
        try:
            line = input("> ").strip()
            if line.lower() in ('done', 'exit', 'quit'):
                break
            if not line and not matchups:
                # Use default matchups
                matchups = default_matchups
                break
            if not line:
                continue
            if 'vs.' in line.lower() or 'vs' in line.lower():
                matchups.append(line)
            else:
                print("  [!] Format should be: 'Rider A vs. Rider B'")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n")
            break
    
    print("\n" + "="*70)
    print(f"H2H PREDICTIONS: {race_slug.upper()} {year} STAGE {stage_num}")
    print("="*70)
    
    for matchup in matchups:
        # Parse matchup
        if 'vs.' in matchup:
            parts = matchup.split('vs.')
        elif 'vs' in matchup:
            parts = matchup.split('vs')
        else:
            continue
        
        if len(parts) == 2:
            rider_a = parts[0].strip()
            rider_b = parts[1].strip()
            print(analyze_h2h(probs, rider_a, rider_b))
    
    print("\nTIP: For H2H betting, look for matchups where model probability > 55%")
    print("       This indicates a statistically significant edge.")


if __name__ == '__main__':
    main()
