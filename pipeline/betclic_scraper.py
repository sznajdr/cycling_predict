"""
Betclic cycling odds scraper.

Scrapes event URLs from the Betclic cycling hub, extracts odds from each
event page using regex on raw HTML, computes hold-adjusted fair odds, and
stores snapshots to the bookmaker_odds table.
"""
import logging
import re
import sqlite3
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from typing import Optional

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

HUB_URL = "https://www.betclic.fr/cyclisme-scycling"

# Matches paths like /cyclisme-scycling/paris-nice-c5649/paris-nice-2026-m1052180106760192
EVENT_URL_RE = re.compile(r'/cyclisme-scycling/[^"\'>\s]+/[a-z0-9\-]+-m\d+')

# Primary and fallback odds key names in Betclic JSON-within-HTML
ODDS_RE = re.compile(r'"name":"([^"]+)",[^{]{0,200}?"odds":([\d.]+)')
ODDS_ALT_RE = re.compile(r'"name":"([^"]+)",[^{]{0,200}?"price":([\d.]+)')

# Market label used to group selections (capture the surrounding market object)
MARKET_LABEL_RE = re.compile(
    r'"label"\s*:\s*"([^"]+)"[^}]{0,500}?"name"\s*:\s*"([^"]+)"[^}]{0,200}?"(?:odds|price)"\s*:\s*([\d.]+)',
    re.DOTALL
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# French market label → MarketType value (first substring match wins, case-insensitive)
_MARKET_LABEL_MAP = [
    (["vainqueur", "victoire"],                          "winner"),
    (["podium", "top 3", "dans le top 3"],               "top_3"),
    (["top 10", "dans le top 10"],                       "top_10"),
    (["duel", "confrontation", "h2h", " vs ",
      "face à face", "face a face"],                     "h2h"),
    (["classement général", "maillot jaune"],            "gc_position"),
    (["montagne", "grimpeur", "maillot à pois"],         "kom"),
    (["points", "maillot vert"],                         "points"),
    (["combatif"],                                       "combativity"),
    (["échappée", "breakaway"],                          "breakaway"),
]

# Prefixes to strip when extracting rider name from selection label
_NAME_PREFIX_RE = re.compile(
    r'^(?:victoire\s*[-–]\s*|top\s*\d+\s*[-–]\s*|podium\s*[-–]\s*|'
    r'vainqueur\s*[-–]\s*|classement\s*[-–]\s*)',
    re.IGNORECASE
)

# URL-path patterns used when HTML label parsing fails (JS-rendered pages)
_URL_MARKET_RULES = [
    (re.compile(r'/(etape-\d+|stage-\d+)[-/]', re.IGNORECASE), 'winner'),
    (re.compile(r'/(podium|top-?3)[-/]',         re.IGNORECASE), 'top_3'),
    (re.compile(r'/top-?10[-/]',                 re.IGNORECASE), 'top_10'),
    (re.compile(r'/(duel|h2h|confrontation)[-/]', re.IGNORECASE), 'h2h'),
    (re.compile(r'/[a-z\-]+-20\d{2}-m\d+$',     re.IGNORECASE), 'gc_position'),
]


def classify_market_from_url(event_url: str) -> str:
    """Classify market type from the Betclic event URL path."""
    path = event_url.split('betclic.fr')[-1]
    for pattern, market_type in _URL_MARKET_RULES:
        if pattern.search(path):
            return market_type
    return 'unknown'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %s fetching %s", exc.code, url)
    except Exception as exc:
        logger.warning("Error fetching %s: %s", url, exc)
    return None


def _normalize_name(name: str) -> str:
    """Strip accents and return ASCII lowercase for fuzzy join."""
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii", "ignore").decode().strip()


def classify_market(label: str) -> str:
    """Map a French market label string to a MarketType value."""
    low = label.lower()
    for phrases, market_type in _MARKET_LABEL_MAP:
        if any(p in low for p in phrases):
            return market_type
    return "unknown"


def extract_participant_name(raw_label: str, market_type: str) -> list[tuple[str, str]]:
    """
    Return list of (participant_name, participant_raw) tuples.

    H2H labels with ' vs ' or ' - ' between two names produce two rows.
    """
    label = raw_label.strip()

    if market_type == "h2h":
        for sep in (" vs ", " - ", " VS ", " Vs "):
            if sep in label:
                parts = label.split(sep, 1)
                return [(p.strip(), raw_label) for p in parts if p.strip()]

    # Strip known French prefixes
    name = _NAME_PREFIX_RE.sub("", label).strip()
    return [(name, raw_label)]


def _extract_market_blocks(html: str) -> list[dict]:
    """
    Parse HTML for market-grouped odds blocks.

    Strategy: find occurrences of "label":"..." then collect all
    name/odds pairs that follow within the same JSON object boundary.
    Returns list of {label, selections: [{name, odds}]}.
    """
    # Find positions of all market labels
    label_re = re.compile(r'"label"\s*:\s*"([^"]+)"')
    sel_re = re.compile(r'"name"\s*:\s*"([^"]+)"[^}]{0,300}?"(?:odds|price)"\s*:\s*([\d.]+)')

    blocks = []
    label_positions = [(m.start(), m.group(1)) for m in label_re.finditer(html)]

    for i, (pos, label) in enumerate(label_positions):
        # Determine end of this block: either the next label or +8000 chars
        end = label_positions[i + 1][0] if i + 1 < len(label_positions) else pos + 8000
        chunk = html[pos:end]
        selections = [
            {"name": m.group(1), "odds": float(m.group(2))}
            for m in sel_re.finditer(chunk)
            if float(m.group(2)) > 1.0  # skip malformed 0/1 values
        ]
        if selections:
            blocks.append({"label": label, "selections": selections})

    return blocks


def compute_fair_odds(selections: list[dict]) -> list[dict]:
    """
    Given list of {name, odds}, compute hold-adjusted fair odds.
    Adds implied_prob, market_total_impl_prob, fair_prob, fair_odds.
    """
    for sel in selections:
        sel["implied_prob"] = 1.0 / sel["odds"]

    total = sum(s["implied_prob"] for s in selections)
    for sel in selections:
        sel["market_total_impl_prob"] = total
        sel["fair_prob"] = sel["implied_prob"] / total if total > 0 else None
        sel["fair_odds"] = (1.0 / sel["fair_prob"]) if sel["fair_prob"] else None

    return selections


# ---------------------------------------------------------------------------
# Core scraping functions
# ---------------------------------------------------------------------------

def scrape_event_urls(hub_url: str = HUB_URL) -> list[str]:
    """Fetch cycling hub page and return unique absolute event URLs."""
    html = _get(hub_url)
    if not html:
        return []
    base = "https://www.betclic.fr"
    seen = set()
    urls = []
    for m in EVENT_URL_RE.finditer(html):
        path = m.group(0)
        full = base + path
        if full not in seen:
            seen.add(full)
            urls.append(full)
    logger.info("Found %d event URLs on hub", len(urls))
    return urls


def scrape_event_odds(event_url: str) -> list[dict]:
    """
    Fetch a single event page and return raw selection dicts.
    Each dict has: label, name, odds.
    Returns [] on any error.
    """
    html = _get(event_url)
    if not html:
        return []

    blocks = _extract_market_blocks(html)
    if not blocks:
        # Fallback: flat regex scan without market grouping
        matches = ODDS_RE.findall(html) or ODDS_ALT_RE.findall(html)
        if matches:
            blocks = [{"label": "unknown", "selections": [
                {"name": n, "odds": float(o)} for n, o in matches if float(o) > 1.0
            ]}]

    rows = []
    for block in blocks:
        for sel in block["selections"]:
            rows.append({"label": block["label"], "name": sel["name"], "odds": sel["odds"]})

    logger.debug("Event %s: %d raw selections across %d markets", event_url, len(rows), len(blocks))
    return rows


def _event_id_from_url(url: str) -> str:
    """Extract the mXXXXXXXXX suffix from a Betclic event URL."""
    m = re.search(r'-(m\d+)$', url.rstrip('/'))
    return m.group(1) if m else url.split('/')[-1]


def process_event(event_url: str, scrape_run_id: str, scraped_at: str) -> list[dict]:
    """
    Scrape one event, classify markets, compute fair odds.
    Returns list of row dicts ready for DB insertion.
    """
    try:
        raw = scrape_event_odds(event_url)
        if not raw:
            return []

        event_id = _event_id_from_url(event_url)

        # Group raw rows by market label
        by_label: dict[str, list] = defaultdict(list)
        for r in raw:
            by_label[r["label"]].append({"name": r["name"], "odds": r["odds"]})

        output_rows = []
        for label, selections in by_label.items():
            market_type = classify_market(label) if label != 'unknown' else classify_market_from_url(event_url)
            enriched = compute_fair_odds(selections)

            for sel in enriched:
                participants = extract_participant_name(sel["name"], market_type)
                for participant_name, participant_raw in participants:
                    output_rows.append({
                        "bookmaker": "betclic",
                        "event_url": event_url,
                        "event_id": event_id,
                        "market_type": market_type,
                        "market_label_raw": label,
                        "participant_name": participant_name,
                        "participant_name_norm": _normalize_name(participant_name),
                        "participant_raw": sel["name"],
                        "back_odds": sel["odds"],
                        "implied_prob": sel["implied_prob"],
                        "market_total_impl_prob": sel.get("market_total_impl_prob"),
                        "fair_prob": sel.get("fair_prob"),
                        "fair_odds": sel.get("fair_odds"),
                        "scraped_at": scraped_at,
                        "scrape_run_id": scrape_run_id,
                        "race_id": None,
                    })

        return output_rows

    except Exception as exc:
        logger.warning("Failed to process event %s: %s", event_url, exc)
        return []


# ---------------------------------------------------------------------------
# DB insertion
# ---------------------------------------------------------------------------

def insert_bookmaker_odds_batch(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert rows into bookmaker_odds, ignoring duplicates. Returns inserted count."""
    if not rows:
        return 0

    sql = """
        INSERT OR IGNORE INTO bookmaker_odds (
            bookmaker, event_url, event_id, market_type, market_label_raw,
            participant_name, participant_name_norm, participant_raw,
            back_odds, implied_prob, market_total_impl_prob, fair_prob, fair_odds,
            scraped_at, scrape_run_id, race_id
        ) VALUES (
            :bookmaker, :event_url, :event_id, :market_type, :market_label_raw,
            :participant_name, :participant_name_norm, :participant_raw,
            :back_odds, :implied_prob, :market_total_impl_prob, :fair_prob, :fair_odds,
            :scraped_at, :scrape_run_id, :race_id
        )
    """
    conn.executemany(sql, rows)
    conn.commit()
    inserted = conn.execute("SELECT changes()").fetchone()[0]
    return inserted


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def scrape_all(conn: sqlite3.Connection, hub_url: str = HUB_URL) -> int:
    """
    Scrape all cycling events from the Betclic hub and store to DB.
    Returns total rows inserted.
    """
    scrape_run_id = str(uuid.uuid4())
    scraped_at = datetime.now(timezone.utc).isoformat()

    event_urls = scrape_event_urls(hub_url)
    if not event_urls:
        logger.warning("No event URLs found — aborting scrape")
        return 0

    all_rows: list[dict] = []
    for url in event_urls:
        rows = process_event(url, scrape_run_id, scraped_at)
        all_rows.extend(rows)

    inserted = insert_bookmaker_odds_batch(conn, all_rows)
    logger.info(
        "Scrape complete: run_id=%s, events=%d, rows_attempted=%d, inserted=%d",
        scrape_run_id, len(event_urls), len(all_rows), inserted
    )
    return inserted
