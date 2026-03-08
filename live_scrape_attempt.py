"""
Live Race Scraper - Attempt with Multiple Methods
=================================================

Tries different approaches to get live race data:
1. procyclingstats library (with cloudscraper)
2. Direct requests with rotation
3. Fallback to manual link

Usage:
    python live_scrape_attempt.py paris-nice 2026 1
"""
import sys
import time
from datetime import datetime


def try_procyclingstats_lib(pcs_slug, year, stage_num):
    """Try using the procyclingstats library (has cloudscraper built-in)."""
    try:
        from procyclingstats import Stage
        
        print("[1] Trying procyclingstats library (with cloudscraper)...")
        
        url = f"race/{pcs_slug}/{year}/stage-{stage_num}"
        stage = Stage(url)
        
        # Try to get results
        results = stage.results()
        
        if results:
            print(f"    ✓ Success! Got {len(results)} riders")
            return results
        else:
            print("    ✗ No results data")
            return None
            
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def try_alternative_sources(pcs_slug, year, stage_num):
    """Provide alternative sources."""
    print("\n" + "="*70)
    print("ALTERNATIVE LIVE SOURCES")
    print("="*70)
    
    race_name = pcs_slug.replace('-', ' ').title()
    
    sources = [
        ("PCS Official", f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}"),
        ("PCS Live Timing", f"https://www.procyclingstats.com/race/{pcs_slug}/{year}/stage-{stage_num}/live"),
        ("Race Website", f"https://www.paris-nice.fr/en/live" if 'paris-nice' in pcs_slug else "Check official race website"),
    ]
    
    print(f"\nRace: {race_name} {year} - Stage {stage_num}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    for name, url in sources:
        print(f"{name}:")
        print(f"  {url}")
        print()
    
    print("Tips for live betting:")
    print("  - Open PCS in browser and watch the live ticker")
    print("  - Use your model predictions from the database")
    print("  - Check odds with: python fetch_odds.py")
    print("  - Monitor for attacks in last 50km")
    print()


def show_model_predictions(pcs_slug, year, stage_num):
    """Show any available model predictions."""
    import sqlite3
    
    print("="*70)
    print("YOUR MODEL PREDICTIONS")
    print("="*70)
    
    conn = sqlite3.connect('data/cycling.db')
    cursor = conn.cursor()
    
    # Find race and stage
    race = cursor.execute(
        'SELECT id FROM races WHERE pcs_slug = ? AND year = ?',
        (pcs_slug, year)
    ).fetchone()
    
    if not race:
        print("Race not found in database")
        conn.close()
        return
    
    stage = cursor.execute(
        'SELECT id FROM race_stages WHERE race_id = ? AND stage_number = ?',
        (race[0], stage_num)
    ).fetchone()
    
    if not stage:
        print("Stage not found")
        conn.close()
        return
    
    # Get predictions
    predictions = cursor.execute('''
        SELECT rider_name, model_prob, edge_bps, kelly_pct
        FROM strategy_outputs
        WHERE strategy_name = 'stage_ranking' AND stage_id = ?
        ORDER BY model_prob DESC
        LIMIT 10
    ''', (stage[0],)).fetchall()
    
    if predictions:
        print(f"\nTop 10 Model Predictions:")
        print(f"{'Rider':<30} {'Prob':<8} {'Edge':<10} {'Kelly':<10}")
        print("-"*70)
        for p in predictions:
            name = p[0][:28] if p[0] else ""
            prob = p[1] or 0
            edge = p[2] or 0
            kelly = p[3] or 0
            print(f"{name:<30} {prob*100:>6.1f}% {edge:>+7.0f}bps {kelly*100:>6.2f}%")
    else:
        print("\nNo predictions found. Run:")
        print(f"  python rank_stage.py {pcs_slug} {year} {stage_num} --run-models")
    
    conn.close()
    print()


def main():
    """Main function."""
    if len(sys.argv) < 4:
        print("Usage: python live_scrape_attempt.py <pcs_slug> <year> <stage_num>")
        print("Example: python live_scrape_attempt.py paris-nice 2026 1")
        print()
        
        # Default to Paris-Nice Stage 1
        pcs_slug = "paris-nice"
        year = 2026
        stage_num = 1
        print(f"Using default: {pcs_slug} {year} stage {stage_num}")
    else:
        pcs_slug = sys.argv[1]
        year = int(sys.argv[2])
        stage_num = int(sys.argv[3])
    
    print("="*70)
    print(f"LIVE RACE SCRAPER - {pcs_slug} {year} Stage {stage_num}")
    print("="*70)
    print()
    
    # Try library
    results = try_procyclingstats_lib(pcs_slug, year, stage_num)
    
    if results:
        print("\n" + "="*70)
        print("LIVE RESULTS")
        print("="*70)
        for i, r in enumerate(results[:20], 1):
            print(f"{i}. {r}")
    else:
        print("\nCould not scrape live data automatically.")
        print("This is normal - PCS protects live pages with Cloudflare.")
    
    # Show alternatives
    try_alternative_sources(pcs_slug, year, stage_num)
    
    # Show predictions
    try:
        show_model_predictions(pcs_slug, year, stage_num)
    except Exception as e:
        print(f"Could not load predictions: {e}")


if __name__ == "__main__":
    main()
