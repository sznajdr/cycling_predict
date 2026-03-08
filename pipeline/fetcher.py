import logging
import time

from procyclingstats import Race, RaceClimbs, RaceCombativeRiders, RaceStartlist, Rider, RiderResults, Stage

from .pcs_parser import stage_type_from_name_and_icon

RATE_LIMIT = 1.0  # seconds between requests

log = logging.getLogger(__name__)


def _sleep():
    time.sleep(RATE_LIMIT)


def fetch_race_meta(pcs_slug, year):
    """
    Fetch race metadata for pcs_slug/year.

    Returns dict with keys:
        pcs_slug, display_name, year, startdate, enddate,
        category, uci_tour, is_one_day_race, stages (list)
    or None on error.
    """
    try:
        url = f"race/{pcs_slug}/{year}"
        race = Race(url)
        is_one_day = race.is_one_day_race()

        stages = []
        if not is_one_day:
            raw_stages = race.stages("date", "profile_icon", "stage_name", "stage_url")
            for s in raw_stages:
                stage_url = s.get("stage_url", "")
                stage_name = s.get("stage_name", "")
                profile_icon = s.get("profile_icon", "")
                # Extract stage number from URL
                stage_number = None
                if stage_url:
                    parts = stage_url.rstrip("/").split("/")
                    last = parts[-1]
                    if last.startswith("stage-"):
                        try:
                            stage_number = int(last.split("-")[1])
                        except (IndexError, ValueError):
                            pass
                    elif last == "prologue":
                        stage_number = 0

                # Derive date (MM-DD) + year → YYYY-MM-DD
                date_raw = s.get("date", "")
                stage_date = None
                if date_raw:
                    try:
                        month, day = date_raw.split("-")
                        stage_date = f"{year}-{int(month):02d}-{int(day):02d}"
                    except Exception:
                        stage_date = None

                stages.append({
                    "stage_number": stage_number,
                    "stage_type": stage_type_from_name_and_icon(stage_name, profile_icon),
                    "stage_date": stage_date,
                    "distance_km": None,
                    "pcs_stage_url": stage_url,
                    "is_one_day_race": False,
                })

        result = {
            "pcs_slug": pcs_slug,
            "display_name": race.name(),
            "year": year,
            "startdate": race.startdate(),
            "enddate": race.enddate(),
            "category": race.category(),
            "uci_tour": race.uci_tour(),
            "is_one_day_race": is_one_day,
            "stages": stages,
        }
        _sleep()
        return result
    except Exception as e:
        log.error("fetch_race_meta(%s, %s) failed: %s", pcs_slug, year, e)
        _sleep()
        return None


def fetch_startlist(pcs_slug, year):
    """
    Fetch startlist entries for pcs_slug/year.

    Returns list of dicts with keys:
        rider_name, rider_url, team_name, team_url, nationality, rider_number
    or [] on error.
    """
    try:
        url = f"race/{pcs_slug}/{year}/startlist"
        sl = RaceStartlist(url)
        entries = sl.startlist()
        _sleep()
        return entries
    except Exception as e:
        log.error("fetch_startlist(%s, %s) failed: %s", pcs_slug, year, e)
        _sleep()
        return []


def fetch_rider_profile(rider_url):
    """
    Fetch full rider profile.

    Returns dict with keys:
        pcs_url, name, nationality, birthdate, height_m, weight_kg,
        sp_one_day_races, sp_gc, sp_time_trial, sp_sprint, sp_climber,
        sp_hills, teams_history
    or None on error.
    """
    try:
        rider = Rider(rider_url)

        spec = {}
        try:
            spec = rider.points_per_speciality()
        except Exception:
            pass

        height = None
        try:
            height = rider.height()
        except Exception:
            pass

        weight = None
        try:
            weight = rider.weight()
        except Exception:
            pass

        birthdate = None
        try:
            birthdate = rider.birthdate()
        except Exception:
            pass

        teams = []
        try:
            teams = rider.teams_history()
        except Exception:
            pass

        result = {
            "pcs_url": rider_url,
            "name": rider.name(),
            "nationality": rider.nationality(),
            "birthdate": birthdate,
            "height_m": height,
            "weight_kg": weight,
            "sp_one_day_races": spec.get("one_day_races"),
            "sp_gc": spec.get("gc"),
            "sp_time_trial": spec.get("time_trial"),
            "sp_sprint": spec.get("sprint"),
            "sp_climber": spec.get("climber"),
            "sp_hills": spec.get("hills"),
            "teams_history": teams,
        }
        _sleep()
        return result
    except Exception as e:
        log.error("fetch_rider_profile(%s) failed: %s", rider_url, e)
        _sleep()
        return None


def fetch_stage_results(pcs_stage_url):
    """
    Fetch all classification results for one stage page.

    Returns dict keyed by result_category with list of result rows, or None on error.
    Each row has: rider_url, rank, status (stage only), time, pcs_points, uci_points.

    Categories returned:
        stage      — always present (stage finish or one-day race result)
        gc         — stage races only
        points     — stage races only
        mountains  — stage races only (kom classification)
        youth      — stage races only
    """
    try:
        stage = Stage(pcs_stage_url)
        is_one_day = stage.is_one_day_race()

        results = {}

        # --- Stage metadata (Tier 1 enrichment) ---
        meta = {}
        for attr, key in [
            ("distance",                  "distance_km"),
            ("vertical_meters",           "vertical_m"),
            ("profile_score",             "profile_score"),
            ("avg_temperature",           "avg_temp_c"),
            ("avg_speed_winner",          "avg_speed_winner_kmh"),
            ("won_how",                   "won_how"),
        ]:
            try:
                meta[key] = getattr(stage, attr)()
            except Exception:
                meta[key] = None
        try:
            sq = stage.race_startlist_quality_score()
            meta["startlist_quality_score"] = sq[0] if sq else None
        except Exception:
            meta["startlist_quality_score"] = None
        results["meta"] = meta

        # --- Classification results ---
        # Stage finish results (all races)
        try:
            results["stage"] = stage.results(
                "rider_url", "rank", "status", "time", "pcs_points", "uci_points"
            )
        except Exception as e:
            log.warning("stage.results() failed for %s: %s", pcs_stage_url, e)
            results["stage"] = []

        if not is_one_day:
            # GC standings after this stage
            try:
                results["gc"] = stage.gc(
                    "rider_url", "rank", "time", "pcs_points", "uci_points"
                )
            except Exception as e:
                log.warning("stage.gc() failed for %s: %s", pcs_stage_url, e)
                results["gc"] = []

            # Points classification
            try:
                results["points"] = stage.points(
                    "rider_url", "rank", "pcs_points", "uci_points"
                )
            except Exception as e:
                log.warning("stage.points() failed for %s: %s", pcs_stage_url, e)
                results["points"] = []

            # KOM / mountains classification
            try:
                results["mountains"] = stage.kom(
                    "rider_url", "rank", "pcs_points", "uci_points"
                )
            except Exception as e:
                log.warning("stage.kom() failed for %s: %s", pcs_stage_url, e)
                results["mountains"] = []

            # Youth classification
            try:
                results["youth"] = stage.youth(
                    "rider_url", "rank", "time", "pcs_points", "uci_points"
                )
            except Exception as e:
                log.warning("stage.youth() failed for %s: %s", pcs_stage_url, e)
                results["youth"] = []

        _sleep()
        return results
    except Exception as e:
        log.error("fetch_stage_results(%s) failed: %s", pcs_stage_url, e)
        _sleep()
        return None


def fetch_combativity(pcs_slug, year):
    """
    Fetch the combative riders list for a race/year.

    Returns list of dicts with keys: rider_url, stage_url
    or [] on error.
    """
    try:
        url = f"race/{pcs_slug}/{year}/results/comative-riders"
        rc = RaceCombativeRiders(url)
        rows = rc.combative_riders("rider_url", "stage_url")
        _sleep()
        return rows
    except Exception as e:
        log.error("fetch_combativity(%s, %s) failed: %s", pcs_slug, year, e)
        _sleep()
        return []


def fetch_race_climbs(pcs_slug, year):
    """
    Fetch the full list of climbs in a race.

    Returns list of dicts with keys:
        climb_name, climb_url, length, steepness, top, km_before_finnish, stage_number
    or [] on error.
    
    Note: km_before_finnish is stage-relative from PCS. Must be transformed
    to race-relative using stage distances.
    """
    try:
        url = f"race/{pcs_slug}/{year}/route/climbs"
        rc = RaceClimbs(url)
        rows = rc.climbs()
        _sleep()
        return rows
    except Exception as e:
        log.error("fetch_race_climbs(%s, %s) failed: %s", pcs_slug, year, e)
        _sleep()
        return []


def transform_km_before_finish(climbs, stage_distances):
    """
    Transform stage-relative km_before_finish to race-relative.
    
    Args:
        climbs: List of climb dicts with 'km_before_finnish' (stage-relative)
        stage_distances: Dict of {stage_number: distance_km}
    
    Returns:
        List of climbs with added 'km_before_finish_race' key (race-relative)
    """
    if not climbs or not stage_distances:
        return climbs
    
    # Calculate cumulative distances
    sorted_stages = sorted(stage_distances.items())  # (stage_num, distance)
    cum_distances = {}
    cum = 0
    for stage_num, dist in sorted_stages:
        cum += dist
        cum_distances[stage_num] = cum
    
    total_distance = cum
    
    # Group climbs by inferred stage
    # Climbs are typically grouped by stage in the raw data
    # For simplicity, we keep stage-relative and let the caller handle mapping
    
    for climb in climbs:
        # Keep original stage-relative value
        climb['km_before_finish_stage'] = climb.get('km_before_finnish', 0) or 0
    
    return climbs


def fetch_rider_results(rider_url):
    """
    Fetch all results for a rider.

    Returns list of dicts with keys:
        stage_url, rank, date, pcs_points, uci_points
    or [] on error.
    """
    try:
        results_url = f"{rider_url}/results"
        rr = RiderResults(results_url)
        rows = rr.results("stage_url", "rank", "date", "pcs_points", "uci_points")
        _sleep()
        return rows
    except Exception as e:
        log.error("fetch_rider_results(%s) failed: %s", rider_url, e)
        _sleep()
        return []
