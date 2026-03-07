"""Test: single race meta + startlist scrape + store."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pipeline.db import (
    get_connection, init_db,
    upsert_race, upsert_rider, upsert_team, upsert_startlist_entry,
    get_race_id, get_rider_id, get_team_id,
)
from pipeline.fetcher import fetch_race_meta, fetch_startlist

PCS_SLUG = "tour-de-france"
YEAR = 2022
failures = 0


def check(label, condition, detail=""):
    global failures
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))
        failures += 1


# ---------------------------------------------------------------------------
# Race meta
# ---------------------------------------------------------------------------
print("\n--- fetch_race_meta ---")
meta = fetch_race_meta(PCS_SLUG, YEAR)

check("meta is not None", meta is not None)
if meta:
    check("is_one_day_race=False", meta.get("is_one_day_race") is False,
          str(meta.get("is_one_day_race")))
    check("startdate present", bool(meta.get("startdate")), str(meta.get("startdate")))
    check("enddate present", bool(meta.get("enddate")), str(meta.get("enddate")))
    check("display_name present", bool(meta.get("display_name")))
    check("stages list non-empty", isinstance(meta.get("stages"), list) and len(meta["stages"]) > 0,
          f"stages count: {len(meta.get('stages', []))}")

# ---------------------------------------------------------------------------
# Startlist
# ---------------------------------------------------------------------------
print("\n--- fetch_startlist ---")
sl = fetch_startlist(PCS_SLUG, YEAR)

check("startlist is list", isinstance(sl, list))
check(
    "~176 startlist entries (140–200)",
    140 <= len(sl) <= 200,
    f"got {len(sl)} entries",
)
if sl:
    first = sl[0]
    check("entry has rider_url", bool(first.get("rider_url")))
    check("entry has team_name", bool(first.get("team_name")))

# ---------------------------------------------------------------------------
# In-memory DB roundtrip
# ---------------------------------------------------------------------------
print("\n--- in-memory DB roundtrip ---")
conn = get_connection(":memory:")
init_db(conn)

# Upsert race
upsert_race(conn, {
    "pcs_slug": PCS_SLUG,
    "display_name": meta["display_name"] if meta else "Tour de France",
    "year": YEAR,
    "startdate": meta.get("startdate") if meta else None,
    "enddate": meta.get("enddate") if meta else None,
    "category": meta.get("category") if meta else None,
    "uci_tour": meta.get("uci_tour") if meta else None,
    "is_one_day_race": False,
})
race_id = get_race_id(conn, PCS_SLUG, YEAR)
check("race stored", race_id is not None)

# Upsert 3 rider stubs + startlist entries
test_entries = (sl[:3] if sl else [
    {"rider_url": "rider/test-rider-1", "rider_name": "Rider One",
     "team_url": "team/team-a-2022", "team_name": "Team A", "rider_number": 1},
    {"rider_url": "rider/test-rider-2", "rider_name": "Rider Two",
     "team_url": "team/team-b-2022", "team_name": "Team B", "rider_number": 2},
    {"rider_url": "rider/test-rider-3", "rider_name": "Rider Three",
     "team_url": "team/team-a-2022", "team_name": "Team A", "rider_number": 3},
])

for entry in test_entries:
    rider_url = entry.get("rider_url", "")
    team_url = entry.get("team_url", "")
    if not rider_url:
        continue
    if team_url:
        upsert_team(conn, {"pcs_url": team_url, "name": entry.get("team_name"),
                            "class": None, "nationality": None})
    upsert_rider(conn, {"pcs_url": rider_url, "name": entry.get("rider_name"),
                         "nationality": entry.get("nationality")})
    rider_id = get_rider_id(conn, rider_url)
    team_id = get_team_id(conn, team_url) if team_url else None
    upsert_startlist_entry(conn, race_id, rider_id, team_id, entry.get("rider_number"))

count = conn.execute(
    "SELECT COUNT(*) as n FROM startlist_entries WHERE race_id=?", (race_id,)
).fetchone()["n"]
check(f"3 startlist entries stored", count == 3, f"got {count}")

conn.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"{'All tests passed!' if failures == 0 else f'{failures} test(s) FAILED'}")
sys.exit(failures)
