"""
CLI entry point for the walk-forward backtester.

Usage
-----
# Run all strategies (recommended):
python scripts/run_backtest.py

# Run a single strategy:
python scripts/run_backtest.py --strategy frailty
python scripts/run_backtest.py --strategy tactical
python scripts/run_backtest.py --strategy baseline

# Customise parameters:
python scripts/run_backtest.py --bankroll 5000 --kelly 0.25 --top-k 3 --no-top3

# Save detailed CSV of every bet:
python scripts/run_backtest.py --save-bets bets.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import logging
import sys
from pathlib import Path

from backtesting import CyclingBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    p = argparse.ArgumentParser(description="Cycling betting backtest")
    p.add_argument(
        "--strategy",
        choices=["frailty", "tactical", "baseline", "all"],
        default="all",
        help="Which strategy to backtest (default: all)",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    p.add_argument(
        "--kelly",
        type=float,
        default=0.25,
        help="Kelly fraction — 0.25 = quarter-Kelly (default: 0.25)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Bet on top-K riders by model signal per stage (default: 3)",
    )
    p.add_argument(
        "--no-top3",
        action="store_true",
        help="Use win market instead of top-3 market",
    )
    p.add_argument(
        "--min-records",
        type=int,
        default=20,
        help="Minimum training records before fitting (default: 20)",
    )
    p.add_argument(
        "--save-bets",
        metavar="FILE",
        help="Save every bet record to a CSV file",
    )
    return p.parse_args()


def save_bets_csv(results: dict, filepath: str) -> None:
    rows = []
    for strategy, result in results.items():
        for b in result.bet_records:
            rows.append({
                "strategy": b.strategy,
                "race": b.race_slug,
                "year": b.year,
                "stage": b.stage_num,
                "stage_type": b.stage_type,
                "rider_id": b.rider_id,
                "rider_name": b.rider_name,
                "predicted_score": round(b.predicted_score, 4),
                "predicted_prob": round(b.predicted_prob, 4),
                "naive_odds": round(b.naive_odds, 2),
                "stake_fraction": round(b.stake_fraction, 4),
                "stake_amount": round(b.stake_amount, 2),
                "actual_rank": b.actual_rank if b.actual_rank < 999 else "DNF",
                "top3": b.is_top3,
                "top5": b.is_top5,
                "top10": b.is_top10,
                "winner": b.is_winner,
                "pnl": round(b.pnl, 2),
                "bankroll_after": round(b.bankroll_after, 2),
            })

    if not rows:
        print("No bets to save.")
        return

    path = Path(filepath)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} bet records to {path}")


def main():
    args = parse_args()

    backtester = CyclingBacktester(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        top_k_bets=args.top_k,
        min_train_records=args.min_records,
        bet_on_top3=not args.no_top3,
    )

    print(f"\nBacktesting cycling strategies...")
    print(f"  Bankroll:      {args.bankroll:,.0f}")
    print(f"  Kelly:         {args.kelly:.0%}")
    print(f"  Top-K bets:    {args.top_k}")
    print(f"  Market:        {'Top-3' if not args.no_top3 else 'Win'}")
    print(f"  Min records:   {args.min_records}")

    try:
        if args.strategy == "all":
            results = backtester.run_all()
        else:
            result = backtester.run(args.strategy)
            results = {args.strategy: result}

        if not results:
            print(
                "\nNo data found. Make sure you have scraped some data first:\n"
                "  python -m pipeline.runner\n"
            )
            sys.exit(1)

        backtester.print_report(results)

        if args.save_bets:
            save_bets_csv(results, args.save_bets)

    except Exception as exc:
        logging.exception(f"Backtest failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
