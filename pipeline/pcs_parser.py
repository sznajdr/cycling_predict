import re

# Maps PCS URL suffixes to result_category values
_CATEGORY_MAP = {
    "gc": "gc",
    "points": "points",
    "mountains": "mountains",
    "kom": "mountains",
    "youth": "youth",
    "teams": "teams",
    "combativity": "combativity",
}

# Maps profile_icon + stage_name hints to stage_type
_PROFILE_TYPE = {
    "p1": "flat",
    "p2": "flat",
    "p3": "hilly",
    "p4": "hilly",
    "p5": "mountain",
    "p0": "flat",
}


def parse_stage_url(stage_url):
    """
    Parse a PCS stage URL into components.

    Returns dict with keys:
        race_slug, year, stage_number (int or None), result_category,
        is_one_day (bool), pcs_stage_url
    Returns None if the URL doesn't match expected format.
    """
    url = stage_url.strip("/")
    m = re.match(r"^race/([^/]+)/(\d{4})(?:/(.+))?$", url)
    if not m:
        return None

    race_slug = m.group(1)
    year = int(m.group(2))
    suffix = m.group(3)

    if suffix is None:
        # One-day race — no stage suffix
        return {
            "race_slug": race_slug,
            "year": year,
            "stage_number": None,
            "result_category": "stage",
            "is_one_day": True,
            "pcs_stage_url": url,
        }

    # Stage number: "stage-N"
    sm = re.match(r"^stage-(\d+)$", suffix)
    if sm:
        return {
            "race_slug": race_slug,
            "year": year,
            "stage_number": int(sm.group(1)),
            "result_category": "stage",
            "is_one_day": False,
            "pcs_stage_url": url,
        }

    # Prologue (stage 0)
    if suffix == "prologue":
        return {
            "race_slug": race_slug,
            "year": year,
            "stage_number": 0,
            "result_category": "stage",
            "is_one_day": False,
            "pcs_stage_url": url,
        }

    # Known classification suffixes
    if suffix in _CATEGORY_MAP:
        return {
            "race_slug": race_slug,
            "year": year,
            "stage_number": None,
            "result_category": _CATEGORY_MAP[suffix],
            "is_one_day": False,
            "pcs_stage_url": url,
        }

    # Unknown suffix — skip
    return None


def stage_type_from_name_and_icon(stage_name, profile_icon):
    """
    Infer stage_type from stage name and profile icon.

    stage_name: e.g. "Stage 2 (ITT)" or "Stage 3 (TTT)"
    profile_icon: e.g. "p1", "p2", ..., "p5"
    """
    if stage_name:
        name_upper = stage_name.upper()
        if "(ITT)" in name_upper or "ITT" in name_upper:
            return "itt"
        if "(TTT)" in name_upper or "TTT" in name_upper:
            return "ttt"
        if "PROLOGUE" in name_upper:
            return "prologue"
    return _PROFILE_TYPE.get(profile_icon, "road")


def parse_pcs_time(time_str):
    """
    Convert a PCS time string to total seconds.

    Handles formats: "H:MM:SS", "HH:MM:SS", "MM:SS", "+H:MM:SS".
    Returns None if the string cannot be parsed.
    """
    if not time_str:
        return None
    s = time_str.strip().lstrip("+")
    m = re.match(r"^(\d+):(\d{2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def normalize_rank(rank_raw):
    """
    Normalize a rank value to a string.

    Returns a numeric string ("1", "2", ...) for finishers,
    or "DNF"/"DNS"/"DSQ"/"OTL" for non-finishers. Returns None for missing.
    """
    if rank_raw is None:
        return None
    s = str(rank_raw).strip().upper()
    if s in ("DNF", "DNS", "DSQ", "OTL"):
        return s
    # "DF" = Did Finish (treat as valid finish; rank embedded in `rank` field)
    if s == "DF":
        return None
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s if s else None
