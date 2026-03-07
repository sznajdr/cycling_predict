import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "cycling.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pcs_slug TEXT NOT NULL,
    display_name TEXT,
    year INTEGER,
    startdate TEXT,
    enddate TEXT,
    category TEXT,
    uci_tour TEXT,
    is_one_day_race INTEGER,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pcs_slug, year)
);

CREATE TABLE IF NOT EXISTS riders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pcs_url TEXT UNIQUE NOT NULL,
    name TEXT,
    nationality TEXT,
    birthdate TEXT,
    height_m REAL,
    weight_kg REAL,
    sp_one_day_races INTEGER,
    sp_gc INTEGER,
    sp_time_trial INTEGER,
    sp_sprint INTEGER,
    sp_climber INTEGER,
    sp_hills INTEGER,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pcs_url TEXT UNIQUE NOT NULL,
    name TEXT,
    class TEXT,
    nationality TEXT
);

CREATE TABLE IF NOT EXISTS race_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    stage_number INTEGER,
    stage_type TEXT,
    stage_date TEXT,
    distance_km REAL,
    pcs_stage_url TEXT UNIQUE,
    is_one_day_race INTEGER DEFAULT 0,
    vertical_m INTEGER,
    profile_score INTEGER,
    avg_temp_c REAL,
    avg_speed_winner_kmh REAL,
    won_how TEXT,
    startlist_quality_score INTEGER,
    UNIQUE(race_id, stage_number)
);

CREATE TABLE IF NOT EXISTS race_climbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    climb_name TEXT,
    climb_url TEXT,
    length_km REAL,
    steepness_pct REAL,
    top_m INTEGER,
    km_before_finish INTEGER,
    UNIQUE(race_id, climb_name)
);

CREATE TABLE IF NOT EXISTS startlist_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL REFERENCES races(id),
    rider_id INTEGER NOT NULL REFERENCES riders(id),
    team_id INTEGER REFERENCES teams(id),
    rider_number INTEGER,
    UNIQUE(race_id, rider_id)
);

CREATE TABLE IF NOT EXISTS rider_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL REFERENCES riders(id),
    race_id INTEGER NOT NULL REFERENCES races(id),
    stage_id INTEGER NOT NULL REFERENCES race_stages(id),
    result_category TEXT NOT NULL CHECK(result_category IN (
        'stage','gc','points','mountains','youth','teams','combativity'
    )),
    rank TEXT,
    time_seconds INTEGER,
    time_behind_winner_seconds INTEGER,
    pcs_points REAL,
    uci_points REAL,
    team_id INTEGER REFERENCES teams(id),
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(rider_id, stage_id, result_category)
);

CREATE TABLE IF NOT EXISTS rider_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL REFERENCES riders(id),
    season INTEGER,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    team_class TEXT,
    since TEXT,
    until TEXT,
    UNIQUE(rider_id, season, team_id)
);

CREATE INDEX IF NOT EXISTS idx_results_rider_race
    ON rider_results(rider_id, race_id, result_category);
CREATE INDEX IF NOT EXISTS idx_results_stage_rank
    ON rider_results(stage_id, rank);
CREATE INDEX IF NOT EXISTS idx_results_rider_date
    ON rider_results(rider_id, fetched_at);
"""


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn):
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_db(conn)


def _migrate_db(conn):
    """Add columns introduced after the initial schema. Safe to run repeatedly."""
    new_cols = [
        ("race_stages", "vertical_m",            "INTEGER"),
        ("race_stages", "profile_score",          "INTEGER"),
        ("race_stages", "avg_temp_c",             "REAL"),
        ("race_stages", "avg_speed_winner_kmh",   "REAL"),
        ("race_stages", "won_how",                "TEXT"),
        ("race_stages", "startlist_quality_score","INTEGER"),
    ]
    for table, col, dtype in new_cols:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # already exists
    conn.commit()


# ---------------------------------------------------------------------------
# Race
# ---------------------------------------------------------------------------

def upsert_race(conn, d):
    conn.execute(
        """
        INSERT INTO races (pcs_slug, display_name, year, startdate, enddate,
            category, uci_tour, is_one_day_race, fetched_at)
        VALUES (:pcs_slug, :display_name, :year, :startdate, :enddate,
            :category, :uci_tour, :is_one_day_race, :fetched_at)
        ON CONFLICT(pcs_slug, year) DO UPDATE SET
            display_name=excluded.display_name,
            startdate=excluded.startdate,
            enddate=excluded.enddate,
            category=excluded.category,
            uci_tour=excluded.uci_tour,
            is_one_day_race=excluded.is_one_day_race,
            fetched_at=excluded.fetched_at
        """,
        {
            "pcs_slug": d.get("pcs_slug"),
            "display_name": d.get("display_name"),
            "year": d.get("year"),
            "startdate": d.get("startdate"),
            "enddate": d.get("enddate"),
            "category": d.get("category"),
            "uci_tour": d.get("uci_tour"),
            "is_one_day_race": 1 if d.get("is_one_day_race") else 0,
            "fetched_at": datetime.utcnow().isoformat(),
        },
    )
    conn.commit()


def get_race_id(conn, pcs_slug, year):
    row = conn.execute(
        "SELECT id FROM races WHERE pcs_slug=? AND year=?", (pcs_slug, year)
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Rider
# ---------------------------------------------------------------------------

def upsert_rider(conn, d):
    conn.execute(
        """
        INSERT INTO riders (pcs_url, name, nationality, birthdate,
            height_m, weight_kg, sp_one_day_races, sp_gc, sp_time_trial,
            sp_sprint, sp_climber, sp_hills, fetched_at)
        VALUES (:pcs_url, :name, :nationality, :birthdate,
            :height_m, :weight_kg, :sp_one_day_races, :sp_gc, :sp_time_trial,
            :sp_sprint, :sp_climber, :sp_hills, :fetched_at)
        ON CONFLICT(pcs_url) DO UPDATE SET
            name=excluded.name,
            nationality=excluded.nationality,
            birthdate=excluded.birthdate,
            height_m=excluded.height_m,
            weight_kg=excluded.weight_kg,
            sp_one_day_races=excluded.sp_one_day_races,
            sp_gc=excluded.sp_gc,
            sp_time_trial=excluded.sp_time_trial,
            sp_sprint=excluded.sp_sprint,
            sp_climber=excluded.sp_climber,
            sp_hills=excluded.sp_hills,
            fetched_at=excluded.fetched_at
        """,
        {
            "pcs_url": d.get("pcs_url"),
            "name": d.get("name"),
            "nationality": d.get("nationality"),
            "birthdate": d.get("birthdate"),
            "height_m": d.get("height_m"),
            "weight_kg": d.get("weight_kg"),
            "sp_one_day_races": d.get("sp_one_day_races"),
            "sp_gc": d.get("sp_gc"),
            "sp_time_trial": d.get("sp_time_trial"),
            "sp_sprint": d.get("sp_sprint"),
            "sp_climber": d.get("sp_climber"),
            "sp_hills": d.get("sp_hills"),
            "fetched_at": datetime.utcnow().isoformat(),
        },
    )
    conn.commit()


def get_rider_id(conn, pcs_url):
    row = conn.execute(
        "SELECT id FROM riders WHERE pcs_url=?", (pcs_url,)
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

def upsert_team(conn, d):
    conn.execute(
        """
        INSERT INTO teams (pcs_url, name, class, nationality)
        VALUES (:pcs_url, :name, :class, :nationality)
        ON CONFLICT(pcs_url) DO UPDATE SET
            name=excluded.name,
            class=excluded.class,
            nationality=excluded.nationality
        """,
        {
            "pcs_url": d.get("pcs_url"),
            "name": d.get("name"),
            "class": d.get("class"),
            "nationality": d.get("nationality"),
        },
    )
    conn.commit()


def get_team_id(conn, pcs_url):
    row = conn.execute(
        "SELECT id FROM teams WHERE pcs_url=?", (pcs_url,)
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def upsert_stage(conn, race_id, stage_number, stage_type=None,
                 stage_date=None, distance_km=None,
                 pcs_stage_url=None, is_one_day_race=False):
    """Insert or update a stage row; returns the stage id."""
    conn.execute(
        """
        INSERT INTO race_stages
            (race_id, stage_number, stage_type, stage_date, distance_km,
             pcs_stage_url, is_one_day_race)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pcs_stage_url) DO UPDATE SET
            stage_type=excluded.stage_type,
            stage_date=excluded.stage_date,
            distance_km=excluded.distance_km,
            is_one_day_race=excluded.is_one_day_race
        """,
        (
            race_id, stage_number, stage_type, stage_date,
            distance_km, pcs_stage_url, 1 if is_one_day_race else 0,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM race_stages WHERE pcs_stage_url=?", (pcs_stage_url,)
    ).fetchone()
    return row["id"] if row else None


def get_stage_id(conn, race_id, stage_number, pcs_stage_url=None):
    """Look up stage id. For one-day races use pcs_stage_url."""
    if pcs_stage_url:
        row = conn.execute(
            "SELECT id FROM race_stages WHERE pcs_stage_url=?",
            (pcs_stage_url,),
        ).fetchone()
        return row["id"] if row else None
    row = conn.execute(
        "SELECT id FROM race_stages WHERE race_id=? AND stage_number=?",
        (race_id, stage_number),
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Startlist
# ---------------------------------------------------------------------------

def upsert_startlist_entry(conn, race_id, rider_id, team_id, rider_number):
    conn.execute(
        """
        INSERT INTO startlist_entries (race_id, rider_id, team_id, rider_number)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(race_id, rider_id) DO UPDATE SET
            team_id=excluded.team_id,
            rider_number=excluded.rider_number
        """,
        (race_id, rider_id, team_id, rider_number),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def insert_rider_result(conn, rider_id, race_id, stage_id, result_category,
                        rank, time_seconds=None,
                        time_behind_winner_seconds=None,
                        pcs_points=None, uci_points=None, team_id=None):
    conn.execute(
        """
        INSERT INTO rider_results
            (rider_id, race_id, stage_id, result_category, rank,
             time_seconds, time_behind_winner_seconds,
             pcs_points, uci_points, team_id, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(rider_id, stage_id, result_category) DO UPDATE SET
            rank=excluded.rank,
            time_seconds=excluded.time_seconds,
            time_behind_winner_seconds=excluded.time_behind_winner_seconds,
            pcs_points=excluded.pcs_points,
            uci_points=excluded.uci_points,
            team_id=excluded.team_id,
            fetched_at=excluded.fetched_at
        """,
        (
            rider_id, race_id, stage_id, result_category, rank,
            time_seconds, time_behind_winner_seconds,
            pcs_points, uci_points, team_id,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Rider teams
# ---------------------------------------------------------------------------

def update_stage_meta(conn, pcs_stage_url, meta):
    """
    Patch enrichment fields onto an existing race_stages row.
    Uses COALESCE so existing non-NULL values are never overwritten with NULL.
    """
    conn.execute(
        """
        UPDATE race_stages SET
            distance_km             = COALESCE(?, distance_km),
            vertical_m              = COALESCE(?, vertical_m),
            profile_score           = COALESCE(?, profile_score),
            avg_temp_c              = COALESCE(?, avg_temp_c),
            avg_speed_winner_kmh    = COALESCE(?, avg_speed_winner_kmh),
            won_how                 = COALESCE(?, won_how),
            startlist_quality_score = COALESCE(?, startlist_quality_score)
        WHERE pcs_stage_url = ?
        """,
        (
            meta.get("distance_km"),
            meta.get("vertical_m"),
            meta.get("profile_score"),
            meta.get("avg_temp_c"),
            meta.get("avg_speed_winner_kmh"),
            meta.get("won_how"),
            meta.get("startlist_quality_score"),
            pcs_stage_url,
        ),
    )
    conn.commit()


def upsert_race_climb(conn, race_id, d):
    """d: dict from RaceClimbs.climbs() — keys: climb_name, climb_url,
    length, steepness, top, km_before_finnish."""
    conn.execute(
        """
        INSERT INTO race_climbs
            (race_id, climb_name, climb_url, length_km,
             steepness_pct, top_m, km_before_finish)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, climb_name) DO UPDATE SET
            climb_url=excluded.climb_url,
            length_km=excluded.length_km,
            steepness_pct=excluded.steepness_pct,
            top_m=excluded.top_m,
            km_before_finish=excluded.km_before_finish
        """,
        (
            race_id,
            d.get("climb_name"),
            d.get("climb_url"),
            d.get("length"),
            d.get("steepness"),
            d.get("top"),
            d.get("km_before_finnish"),  # PCS scraper has this typo
        ),
    )
    conn.commit()


def insert_rider_teams(conn, rider_id, rows):
    """rows: list of dicts with keys season, team_url, class, since, until."""
    for row in rows:
        team_url = row.get("team_url", "")
        if not team_url:
            continue
        team_id = get_team_id(conn, team_url)
        if team_id is None:
            upsert_team(conn, {
                "pcs_url": team_url,
                "name": row.get("team_name"),
                "class": row.get("class"),
                "nationality": None,
            })
            team_id = get_team_id(conn, team_url)
        conn.execute(
            """
            INSERT INTO rider_teams (rider_id, season, team_id, team_class, since, until)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(rider_id, season, team_id) DO UPDATE SET
                team_class=excluded.team_class,
                since=excluded.since,
                until=excluded.until
            """,
            (
                rider_id,
                row.get("season"),
                team_id,
                row.get("class"),
                row.get("since"),
                row.get("until"),
            ),
        )
    conn.commit()
