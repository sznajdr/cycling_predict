"""Test: single rider scrape + store."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pipeline.db import get_connection, init_db, upsert_rider, insert_rider_result, get_rider_id, upsert_stage, upsert_race, get_race_id, get_stage_id
from pipeline.fetcher import fetch_rider_profile, fetch_rider_results
from pipeline.pcs_parser import normalize_rank

RIDER_URL = "rider/tadej-pogacar"
failures = 0


def check(label, condition, detail=""):
    global failures
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))
        failures += 1


# ---------------------------------------------------------------------------
# Fetch profile
# ---------------------------------------------------------------------------
print("\n--- fetch_rider_profile ---")
profile = fetch_rider_profile(RIDER_URL)

check("profile is not None", profile is not None)
if profile:
    check("profile has name", bool(profile.get("name")))
    check("profile has nationality", bool(profile.get("nationality")))
    check(
        "height_m is float or None",
        profile.get("height_m") is None or isinstance(profile["height_m"], float),
        str(profile.get("height_m")),
    )
    check(
        "weight_kg is float or None",
        profile.get("weight_kg") is None or isinstance(profile["weight_kg"], float),
        str(profile.get("weight_kg")),
    )
    check("teams_history is list", isinstance(profile.get("teams_history"), list))

# ---------------------------------------------------------------------------
# Fetch results
# ---------------------------------------------------------------------------
print("\n--- fetch_rider_results ---")
results = fetch_rider_results(RIDER_URL)

check("results is non-empty list", isinstance(results, list) and len(results) > 0)
if results:
    top_ranks = [r.get("rank") for r in results if r.get("rank") is not None]
    normalized = [normalize_rank(r) for r in top_ranks]
    has_win = any(n == "1" for n in normalized)
    check("at least one rank=1 result", has_win, f"normalized ranks sample: {normalized[:10]}")

# ---------------------------------------------------------------------------
# In-memory DB roundtrip
# ---------------------------------------------------------------------------
print("\n--- in-memory DB roundtrip ---")
conn = get_connection(":memory:")
init_db(conn)

# Upsert rider
upsert_rider(conn, {
    "pcs_url": RIDER_URL,
    "name": profile["name"] if profile else "Test Rider",
    "nationality": profile.get("nationality") if profile else "SI",
    "height_m": profile.get("height_m") if profile else None,
    "weight_kg": profile.get("weight_kg") if profile else None,
})
rider_id = get_rider_id(conn, RIDER_URL)
check("rider stored and retrieved", rider_id is not None)

# Upsert a dummy race + stage for result insertion
upsert_race(conn, {
    "pcs_slug": "tour-de-france",
    "display_name": "Tour de France",
    "year": 2022,
    "startdate": "2022-07-01",
    "enddate": "2022-07-24",
    "category": "Men Elite",
    "uci_tour": "UCI Worldtour",
    "is_one_day_race": False,
})
race_id = get_race_id(conn, "tour-de-france", 2022)
upsert_stage(conn, race_id=race_id, stage_number=1,
             stage_type="flat", stage_date="2022-07-01",
             pcs_stage_url="race/tour-de-france/2022/stage-1")
stage_id = get_stage_id(conn, race_id, 1)
check("stage stored and retrieved", stage_id is not None)

# Insert result
insert_rider_result(
    conn,
    rider_id=rider_id,
    race_id=race_id,
    stage_id=stage_id,
    result_category="stage",
    rank="5",
    pcs_points=20.0,
    uci_points=10.0,
)

row = conn.execute(
    "SELECT rank, pcs_points FROM rider_results WHERE rider_id=? AND stage_id=?",
    (rider_id, stage_id),
).fetchone()
check("result stored correctly", row is not None and row["rank"] == "5" and row["pcs_points"] == 20.0)

conn.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"{'All tests passed!' if failures == 0 else f'{failures} test(s) FAILED'}")
sys.exit(failures)
