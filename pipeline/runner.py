import logging
import sys
from pathlib import Path

import yaml

from .db import (
    get_connection, init_db,
    upsert_race, upsert_rider, upsert_team, upsert_stage,
    upsert_startlist_entry, insert_rider_result, insert_rider_teams,
    get_race_id, get_rider_id, get_team_id, get_stage_id,
    update_stage_meta, upsert_race_climb,
)
from .fetcher import (
    fetch_race_meta, fetch_startlist, fetch_stage_results,
    fetch_combativity, fetch_race_climbs,
    fetch_rider_profile, fetch_rider_results,
)
from .pcs_parser import parse_stage_url, normalize_rank, parse_pcs_time
from .queue import (
    init_queue, seed_queue, claim_next_job, complete_job,
    add_job, mark_fresh, is_fresh,
)

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "races.yaml"
_LOG_PATH = _ROOT / "logs" / "pipeline.log"


def setup_logging():
    _LOG_PATH.parent.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(_LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path=_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------

def _handle_race_meta(conn, job):
    pcs_slug = job["pcs_slug"]
    year = job["year"]
    entity_key = f"{pcs_slug}/{year}"

    data = fetch_race_meta(pcs_slug, year)
    if data is None:
        return False, "fetch_race_meta returned None"

    upsert_race(conn, data)

    race_id = get_race_id(conn, pcs_slug, year)

    # Upsert stages for stage races
    for s in data.get("stages", []):
        upsert_stage(
            conn,
            race_id=race_id,
            stage_number=s["stage_number"],
            stage_type=s["stage_type"],
            stage_date=s["stage_date"],
            distance_km=s["distance_km"],
            pcs_stage_url=s["pcs_stage_url"],
            is_one_day_race=False,
        )

    # For one-day races, create a single stage row
    if data["is_one_day_race"]:
        pcs_stage_url = f"race/{pcs_slug}/{year}"
        upsert_stage(
            conn,
            race_id=race_id,
            stage_number=None,
            stage_type="road",
            stage_date=data.get("startdate"),
            distance_km=None,
            pcs_stage_url=pcs_stage_url,
            is_one_day_race=True,
        )

    # Queue stage_results jobs (priority 3) for every stage we just created
    stage_urls = [s["pcs_stage_url"] for s in data.get("stages", []) if s["pcs_stage_url"]]
    if data["is_one_day_race"]:
        stage_urls = [f"race/{pcs_slug}/{year}"]
    for stage_url in stage_urls:
        if not is_fresh(conn, "stage_results", stage_url):
            add_job(conn, "stage_results", stage_url, year=year, priority=3)

    # Queue combativity + race_climbs (priority 4, one per race)
    if not is_fresh(conn, "combativity", entity_key):
        add_job(conn, "combativity", pcs_slug, year=year, priority=4)
    if not is_fresh(conn, "race_climbs", entity_key):
        add_job(conn, "race_climbs", pcs_slug, year=year, priority=4)

    mark_fresh(conn, "race_meta", entity_key)
    add_job(conn, "race_startlist", pcs_slug, year=year, priority=2)
    log.info("race_meta done: %s/%s (%d stages queued)", pcs_slug, year, len(stage_urls))
    return True, None


def _handle_race_startlist(conn, job):
    pcs_slug = job["pcs_slug"]
    year = job["year"]
    entity_key = f"{pcs_slug}/{year}"

    race_id = get_race_id(conn, pcs_slug, year)
    if race_id is None:
        return False, f"race not in DB: {pcs_slug}/{year}"

    entries = fetch_startlist(pcs_slug, year)
    if not entries:
        return False, "fetch_startlist returned empty"

    for entry in entries:
        rider_url = entry.get("rider_url", "")
        if not rider_url:
            continue

        team_url = entry.get("team_url", "")
        team_name = entry.get("team_name", "")

        # Upsert team stub
        team_id = None
        if team_url:
            upsert_team(conn, {
                "pcs_url": team_url,
                "name": team_name,
                "class": None,
                "nationality": None,
            })
            team_id = get_team_id(conn, team_url)

        # Upsert rider stub (minimal)
        upsert_rider(conn, {
            "pcs_url": rider_url,
            "name": entry.get("rider_name"),
            "nationality": entry.get("nationality"),
        })
        rider_id = get_rider_id(conn, rider_url)

        upsert_startlist_entry(
            conn,
            race_id=race_id,
            rider_id=rider_id,
            team_id=team_id,
            rider_number=entry.get("rider_number"),
        )

        # Queue profile + results if not already fresh
        # year=0 is a sentinel: rider jobs don't have a year, but NULL would
        # break the UNIQUE(job_type, pcs_slug, year) constraint in SQLite.
        if not is_fresh(conn, "rider_profile", rider_url):
            add_job(conn, "rider_profile", rider_url, year=0, priority=5)
        if not is_fresh(conn, "rider_results", rider_url):
            add_job(conn, "rider_results", rider_url, year=0, priority=6)

    mark_fresh(conn, "race_startlist", entity_key)
    log.info("race_startlist done: %s/%s (%d riders)", pcs_slug, year, len(entries))
    return True, None


def _handle_stage_results(conn, job):
    pcs_stage_url = job["pcs_slug"]  # full stage URL stored in pcs_slug column

    row = conn.execute(
        "SELECT id, race_id FROM race_stages WHERE pcs_stage_url=?",
        (pcs_stage_url,),
    ).fetchone()
    if row is None:
        return False, f"stage not in DB: {pcs_stage_url}"

    stage_id = row["id"]
    race_id = row["race_id"]

    data = fetch_stage_results(pcs_stage_url)
    if data is None:
        return False, "fetch_stage_results returned None"

    # Tier 1: patch enrichment columns onto the stage row
    if data.get("meta"):
        update_stage_meta(conn, pcs_stage_url, data["meta"])

    stored = 0
    for category, rows in data.items():
        if category == "meta":
            continue
        if not rows:
            continue

        # Determine winner's time for gap calculation.
        # Winner (rank 1) is first in the list; their time is an absolute value.
        # Non-winners may show a relative "+gap" string which parse_pcs_time
        # returns as a small number of seconds. We detect this by checking
        # whether the parsed value is less than the winner's time.
        winner_time_s = None
        for r in rows:
            t = parse_pcs_time(str(r.get("time") or ""))
            if t is not None:
                winner_time_s = t
                break

        for r in rows:
            rider_url = r.get("rider_url", "")
            if not rider_url:
                continue

            rider_id = get_rider_id(conn, rider_url)
            if rider_id is None:
                continue  # rider not in our startlist

            # Rank: prefer numeric rank; fall back to status (DNF/DNS/etc.)
            rank_raw = r.get("rank")
            status = str(r.get("status") or "").strip().upper()
            if status and status not in ("DF", ""):
                rank = status
            else:
                rank = normalize_rank(rank_raw)

            time_s = parse_pcs_time(str(r.get("time") or ""))

            # Compute time_behind_winner_seconds
            gap_s = None
            if time_s is not None and winner_time_s is not None:
                if rank == "1":
                    gap_s = 0
                elif time_s < winner_time_s:
                    # time_s is a relative gap (e.g. parsed from "+0:05:32")
                    gap_s = time_s
                    time_s = winner_time_s + gap_s
                else:
                    gap_s = time_s - winner_time_s

            try:
                insert_rider_result(
                    conn,
                    rider_id=rider_id,
                    race_id=race_id,
                    stage_id=stage_id,
                    result_category=category,
                    rank=rank,
                    time_seconds=time_s,
                    time_behind_winner_seconds=gap_s,
                    pcs_points=r.get("pcs_points"),
                    uci_points=r.get("uci_points"),
                )
                stored += 1
            except Exception as e:
                log.debug("insert skipped (%s/%s): %s", pcs_stage_url, category, e)

    mark_fresh(conn, "stage_results", pcs_stage_url)
    log.info("stage_results done: %s (%d rows stored)", pcs_stage_url, stored)
    return True, None


def _handle_combativity(conn, job):
    pcs_slug = job["pcs_slug"]
    year = job["year"]
    entity_key = f"{pcs_slug}/{year}"

    race_id = get_race_id(conn, pcs_slug, year)
    if race_id is None:
        return False, f"race not in DB: {pcs_slug}/{year}"

    rows = fetch_combativity(pcs_slug, year)
    stored = 0
    for row in rows:
        rider_url = row.get("rider_url", "")
        stage_url = row.get("stage_url", "")
        if not rider_url or not stage_url:
            continue

        rider_id = get_rider_id(conn, rider_url)
        if rider_id is None:
            continue

        parsed = parse_stage_url(stage_url)
        if parsed is None:
            continue
        stage_id = get_stage_id(conn, race_id, parsed["stage_number"],
                                parsed["pcs_stage_url"])
        if stage_id is None:
            continue

        try:
            insert_rider_result(
                conn,
                rider_id=rider_id,
                race_id=race_id,
                stage_id=stage_id,
                result_category="combativity",
                rank="1",
            )
            stored += 1
        except Exception as e:
            log.debug("combativity insert skipped (%s): %s", stage_url, e)

    mark_fresh(conn, "combativity", entity_key)
    log.info("combativity done: %s/%s (%d awards)", pcs_slug, year, stored)
    return True, None


def _handle_race_climbs(conn, job):
    pcs_slug = job["pcs_slug"]
    year = job["year"]
    entity_key = f"{pcs_slug}/{year}"

    race_id = get_race_id(conn, pcs_slug, year)
    if race_id is None:
        return False, f"race not in DB: {pcs_slug}/{year}"

    rows = fetch_race_climbs(pcs_slug, year)
    for row in rows:
        if not row.get("climb_name"):
            continue
        upsert_race_climb(conn, race_id, row)

    mark_fresh(conn, "race_climbs", entity_key)
    log.info("race_climbs done: %s/%s (%d climbs)", pcs_slug, year, len(rows))
    return True, None


def _handle_rider_profile(conn, job):
    rider_url = job["pcs_slug"]  # pcs_slug stores rider_url for rider jobs

    data = fetch_rider_profile(rider_url)
    if data is None:
        return False, "fetch_rider_profile returned None"

    upsert_rider(conn, data)
    rider_id = get_rider_id(conn, rider_url)

    teams = data.get("teams_history", [])
    if teams:
        insert_rider_teams(conn, rider_id, teams)

    mark_fresh(conn, "rider_profile", rider_url)
    log.info("rider_profile done: %s", rider_url)
    return True, None


def _handle_rider_results(conn, job):
    rider_url = job["pcs_slug"]

    rider_id = get_rider_id(conn, rider_url)
    if rider_id is None:
        return False, f"rider not in DB: {rider_url}"

    rows = fetch_rider_results(rider_url)
    stored = 0

    for row in rows:
        stage_url = row.get("stage_url", "")
        if not stage_url:
            continue

        parsed = parse_stage_url(stage_url)
        if parsed is None:
            continue

        race_slug = parsed["race_slug"]
        year = parsed["year"]
        stage_number = parsed["stage_number"]
        result_category = parsed["result_category"]
        pcs_stage_url = parsed["pcs_stage_url"]

        race_id = get_race_id(conn, race_slug, year)
        if race_id is None:
            continue  # race not tracked

        stage_id = get_stage_id(conn, race_id, stage_number, pcs_stage_url)
        if stage_id is None:
            continue  # stage not tracked

        rank_raw = row.get("rank")
        rank = normalize_rank(rank_raw)

        # Parse time if available
        time_seconds = None
        tbw_seconds = None
        time_raw = row.get("time")
        if time_raw:
            time_seconds = parse_pcs_time(str(time_raw))

        pcs_points = row.get("pcs_points")
        uci_points = row.get("uci_points")

        try:
            insert_rider_result(
                conn,
                rider_id=rider_id,
                race_id=race_id,
                stage_id=stage_id,
                result_category=result_category,
                rank=rank,
                time_seconds=time_seconds,
                time_behind_winner_seconds=tbw_seconds,
                pcs_points=pcs_points,
                uci_points=uci_points,
            )
            stored += 1
        except Exception as e:
            log.debug("insert_rider_result skipped (%s/%s): %s", rider_url, stage_url, e)

    mark_fresh(conn, "rider_results", rider_url)
    log.info("rider_results done: %s (%d stored)", rider_url, stored)
    return True, None


_HANDLERS = {
    "race_meta":      _handle_race_meta,
    "race_startlist": _handle_race_startlist,
    "stage_results":  _handle_stage_results,
    "combativity":    _handle_combativity,
    "race_climbs":    _handle_race_climbs,
    "rider_profile":  _handle_rider_profile,
    "rider_results":  _handle_rider_results,
}


def process_job(conn, job):
    handler = _HANDLERS.get(job["job_type"])
    if handler is None:
        log.error("Unknown job_type: %s", job["job_type"])
        complete_job(conn, job["id"], success=False,
                     retries=job["retries"], error_msg="unknown job_type")
        return

    try:
        success, error = handler(conn, job)
    except Exception as e:
        success, error = False, str(e)
        log.exception("process_job crashed for job %s", job["id"])

    complete_job(conn, job["id"], success=success,
                 retries=job["retries"], error_msg=error)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    config = load_config()

    conn = get_connection()
    init_db(conn)
    init_queue(conn)

    seed_queue(conn, config)
    log.info("Queue seeded. Starting processing loop.")

    processed = 0
    while True:
        job = claim_next_job(conn)
        if job is None:
            break
        process_job(conn, job)
        processed += 1

    log.info("Done. Processed %d jobs.", processed)
    conn.close()


if __name__ == "__main__":
    main()
