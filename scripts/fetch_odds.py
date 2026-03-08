"""
CLI entry point for scraping Betclic cycling odds.

Usage:
    python scripts/fetch_odds.py             # scrape all events from hub
    python scripts/fetch_odds.py --init-schema   # apply bookmaker_odds schema only
    python scripts/fetch_odds.py --event-url URL # scrape single event
    python scripts/fetch_odds.py --dry-run       # print results, don't write to DB
    python scripts/fetch_odds.py --dry-run --event-url URL
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

from pipeline.db import get_connection, init_db, init_betting_schema
from pipeline.betclic_scraper import (
    scrape_all,
    scrape_event_urls,
    process_event,
    insert_bookmaker_odds_batch,
    HUB_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no rows)")
        return
    cols = ["participant_name", "market_type", "back_odds", "fair_odds",
            "implied_prob", "fair_prob", "market_label_raw"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  " + "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        line = "  " + "  ".join(
            str(r.get(c, "")).ljust(widths[c]) for c in cols
        )
        print(line)
    print(f"\n  {len(rows)} selections")


def cmd_init_schema(conn) -> None:
    init_betting_schema(conn)
    print("Schema applied (bookmaker_odds table + view + indexes).")


def cmd_scrape_all(conn) -> None:
    inserted = scrape_all(conn)
    print(f"Done. {inserted} rows inserted.")


def cmd_event(conn, event_url: str, dry_run: bool) -> None:
    scrape_run_id = str(uuid.uuid4())
    scraped_at = datetime.now(timezone.utc).isoformat()
    rows = process_event(event_url, scrape_run_id, scraped_at)

    if not rows:
        print(f"No odds extracted from: {event_url}")
        return

    if dry_run:
        print(f"\nDRY RUN — {len(rows)} selections from {event_url}\n")
        _print_table(rows)
    else:
        inserted = insert_bookmaker_odds_batch(conn, rows)
        print(f"Inserted {inserted} rows (of {len(rows)} attempted) from {event_url}")


def cmd_dry_run_all(conn) -> None:
    scrape_run_id = str(uuid.uuid4())
    scraped_at = datetime.now(timezone.utc).isoformat()
    event_urls = scrape_event_urls(HUB_URL)
    if not event_urls:
        print("No event URLs found on hub.")
        return

    all_rows = []
    for url in event_urls:
        rows = process_event(url, scrape_run_id, scraped_at)
        all_rows.extend(rows)

    print(f"\nDRY RUN — {len(all_rows)} total selections across {len(event_urls)} events\n")
    _print_table(all_rows)


def parse_args():
    p = argparse.ArgumentParser(description="Fetch Betclic cycling odds")
    p.add_argument("--init-schema", action="store_true",
                   help="Apply bookmaker_odds schema and exit")
    p.add_argument("--event-url", metavar="URL",
                   help="Scrape a single event URL instead of the full hub")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing to DB")
    p.add_argument("--db", default=None, metavar="PATH",
                   help="Override DB path (default: data/cycling.db)")
    return p.parse_args()


def main():
    args = parse_args()
    conn = get_connection(args.db) if args.db else get_connection()

    init_db(conn)
    init_betting_schema(conn)

    if args.init_schema:
        cmd_init_schema(conn)
        return

    if args.event_url:
        cmd_event(conn, args.event_url, dry_run=args.dry_run)
        return

    if args.dry_run:
        cmd_dry_run_all(conn)
        return

    cmd_scrape_all(conn)


if __name__ == "__main__":
    main()
