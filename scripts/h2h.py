#!/usr/bin/env python3
"""
H2H Prediction CLI - Quick head-to-head matchup predictions

Usage:
    # Interactive mode (enter matchups one by one)
    python scripts/h2h.py paris-nice 2026 2
    
    # Single matchup
    python scripts/h2h.py paris-nice 2026 2 --matchup "Zingle vs Godon"
    
    # Multiple matchups
    python scripts/h2h.py paris-nice 2026 2 --matchup "Zingle vs Godon" --matchup "Girmay vs Fretin"
    
    # From file (one matchup per line)
    python scripts/h2h.py paris-nice 2026 2 --file matchups.txt
    
    # Include "The Field" matchups
    python scripts/h2h.py paris-nice 2026 2 --matchup "Lamperti vs Das Feld"
"""

import argparse
import sqlite3
import unicodedata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def normalize(name):
    """Strip accents and normalize for matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii", "ignore").decode().strip().lower()


def safe_str(s):
    """Safe string for Windows console."""
    return s.encode('ascii', 'replace').decode('ascii')


def is_field(name):
    """Check if name is 'The Field' (Das Feld)."""
    n = normalize(name)
    return 'feld' in n or 'field' in n


def get_probs(conn, race, year, stage):
    """Get model win probabilities for all riders."""
    cursor = conn.execute('''
        SELECT r.name, r.pcs_url, so.win_prob, so.computed_at
        FROM strategy_outputs so
        JOIN riders r ON so.rider_id = r.id
        JOIN race_stages rs ON so.stage_id = rs.id
        JOIN races ra ON rs.race_id = ra.id
        WHERE so.strategy_name = 'stage_ranking'
          AND ra.pcs_slug = ? AND ra.year = ? AND rs.stage_number = ?
    ''', (race, year, stage))
    
    probs = {}
    computed_at = None
    for row in cursor.fetchall():
        orig_name = row[0]
        pcs_url = row[1]
        win_prob = row[2]
        if computed_at is None and row[3]:
            computed_at = row[3]
        norm_name = normalize(orig_name)
        probs[norm_name] = (orig_name, win_prob)
        # Also by last name
        last = norm_name.split()[-1] if ' ' in norm_name else norm_name
        probs[last] = (orig_name, win_prob)
        # Also by PCS URL
        if pcs_url:
            url_key = normalize(pcs_url.replace('rider/', ''))
            probs[url_key] = (orig_name, win_prob)
    return probs, computed_at


def find_rider(rider, probs):
    """Find rider in probability dict."""
    name = normalize(rider)
    if name in probs:
        return probs[name]
    last = name.split()[-1] if ' ' in name else name
    if last in probs:
        return probs[last]
    for key, val in probs.items():
        if last in key or key in last:
            return val
    return None


def calculate_h2h(a_name, b_name, probs):
    """Calculate H2H probability."""
    a_is_field = is_field(a_name)
    b_is_field = is_field(b_name)
    
    if a_is_field and b_is_field:
        return None, "ERROR: Both cannot be 'The Field'"
    
    if a_is_field:
        # Field vs Rider
        b = find_rider(b_name, probs)
        if not b:
            return None, f"{safe_str(b_name)}: NOT FOUND"
        prob_b = b[1]
        prob_field = 1 - prob_b
        if prob_field <= 0:
            prob_field = 0.001  # Avoid division by zero
        prob_a_wins = prob_field / (prob_field + prob_b)
        odds_a = 1 / prob_a_wins
        odds_b = 1 / (1 - prob_a_wins)
        return ("The Field", b[0], prob_a_wins, odds_a, odds_b), None
    
    if b_is_field:
        # Rider vs Field
        a = find_rider(a_name, probs)
        if not a:
            return None, f"{safe_str(a_name)}: NOT FOUND"
        prob_a = a[1]
        prob_field = 1 - prob_a
        if prob_field <= 0:
            prob_field = 0.001
        prob_a_wins = prob_a / (prob_a + prob_field)
        odds_a = 1 / prob_a_wins
        odds_b = 1 / (1 - prob_a_wins)
        return (a[0], "The Field", prob_a_wins, odds_a, odds_b), None
    
    # Normal H2H
    a = find_rider(a_name, probs)
    b = find_rider(b_name, probs)
    
    if not a:
        return None, f"{safe_str(a_name)}: NOT FOUND"
    if not b:
        return None, f"{safe_str(b_name)}: NOT FOUND"
    
    prob_a = a[1] / (a[1] + b[1]) if (a[1] + b[1]) > 0 else 0.5
    odds_a = 1 / prob_a if prob_a > 0 else 999
    odds_b = 1 / (1 - prob_a) if prob_a < 1 else 999
    
    return (a[0], b[0], prob_a, odds_a, odds_b), None


def parse_matchup(text):
    """Parse matchup text like 'Rider A vs Rider B'."""
    text = text.strip()
    # Try various separators
    for sep in [' vs. ', ' vs ', ' v ', ' - ', ' / ']:
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    return None, None


def print_header(race, year, stage):
    """Print prediction header."""
    race_display = race.replace('-', ' ').title()
    print(f"\n{'='*70}")
    print(f"{race_display} {year} Stage {stage} - H2H Predictions")
    print(f"{'='*70}")
    print(f"{'Rider A':<25} vs {'Rider B':<25} | Prob A | Fair Odds")
    print(f"{'-'*70}")


def print_result(result):
    """Print a single H2H result."""
    name_a, name_b, prob_a, odds_a, odds_b = result
    prob_b = 1 - prob_a
    print(f"{safe_str(name_a):<25} vs {safe_str(name_b):<25} | {prob_a*100:5.1f}% | @{odds_a:.2f} / @{odds_b:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='H2H matchup predictions for cycling stages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/h2h.py paris-nice 2026 2
  
  # Single matchup
  python scripts/h2h.py paris-nice 2026 2 -m "Zingle vs Godon"
  
  # Multiple matchups
  python scripts/h2h.py paris-nice 2026 2 -m "Zingle vs Godon" -m "Girmay vs Fretin"
  
  # From file
  python scripts/h2h.py paris-nice 2026 2 -f matchups.txt
  
  # The Field matchups
  python scripts/h2h.py paris-nice 2026 2 -m "Lamperti vs Das Feld"
        """
    )
    parser.add_argument('race', help='Race slug (e.g., paris-nice, tirreno-adriatico)')
    parser.add_argument('year', type=int, help='Year (e.g., 2026)')
    parser.add_argument('stage', type=int, help='Stage number (e.g., 1, 2)')
    parser.add_argument('-m', '--matchup', action='append', dest='matchups',
                       help='Matchup in format "Rider A vs Rider B" (can be used multiple times)')
    parser.add_argument('-f', '--file', dest='matchup_file',
                       help='File containing matchups (one per line)')
    
    args = parser.parse_args()
    
    # Connect to DB
    conn = sqlite3.connect('data/cycling.db')
    
    # Get probabilities
    try:
        probs, computed_at = get_probs(conn, args.race, args.year, args.stage)
    except Exception as e:
        print(f"Error: Could not load model data. Run: python scripts/rank_stage.py {args.race} {args.year} {args.stage} --save")
        sys.exit(1)
    
    if not probs:
        print(f"Error: No model data found for {args.race} {args.year} Stage {args.stage}")
        print(f"Run: python scripts/rank_stage.py {args.race} {args.year} {args.stage} --save")
        sys.exit(1)
    
    # Check data freshness
    if computed_at:
        from datetime import datetime, timezone
        try:
            computed_dt = datetime.fromisoformat(computed_at.replace('Z', '+00:00'))
            # computed_at is stored as UTC (datetime.utcnow()); compare to UTC now
            if computed_dt.tzinfo is None:
                computed_dt = computed_dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - computed_dt).total_seconds() / 3600
            if age_hours > 1:
                print(f"\nWARNING: Data is {age_hours:.1f} hours old. Run with --save to refresh:")
                print(f"  python scripts/rank_stage.py {args.race} {args.year} {args.stage} --save\n")
        except:
            pass
    
    # Collect matchups
    matchups = []
    
    if args.matchup_file:
        try:
            with open(args.matchup_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        a, b = parse_matchup(line)
                        if a and b:
                            matchups.append((a, b))
        except FileNotFoundError:
            print(f"Error: File not found: {args.matchup_file}")
            sys.exit(1)
    
    if args.matchups:
        for m in args.matchups:
            a, b = parse_matchup(m)
            if a and b:
                matchups.append((a, b))
    
    # Print header
    print_header(args.race, args.year, args.stage)
    
    # Process matchups or go interactive
    if matchups:
        for a, b in matchups:
            result, error = calculate_h2h(a, b, probs)
            if error:
                print(f"  {error}")
            else:
                print_result(result)
    else:
        # Interactive mode
        print("\nEnter matchups in format: 'Rider A vs Rider B'")
        print("Type 'done' or press Ctrl+D to exit\n")
        
        while True:
            try:
                line = input("> ").strip()
                if line.lower() in ('done', 'exit', 'quit'):
                    break
                if not line:
                    continue
                
                a, b = parse_matchup(line)
                if not a or not b:
                    print("  Format: 'Rider A vs Rider B'")
                    continue
                
                result, error = calculate_h2h(a, b, probs)
                if error:
                    print(f"  {error}")
                else:
                    print_result(result)
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n")
                break
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
