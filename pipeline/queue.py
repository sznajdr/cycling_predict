from datetime import datetime, timedelta

_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL CHECK(job_type IN (
        'race_meta','race_startlist','stage_results','combativity','race_climbs',
        'rider_profile','rider_results'
    )),
    pcs_slug TEXT NOT NULL,
    year INTEGER,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN (
        'pending','in_progress','completed','failed','permanent_fail'
    )),
    priority INTEGER DEFAULT 5,
    retries INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(job_type, pcs_slug, year)
);

CREATE INDEX IF NOT EXISTS idx_queue_status_prio
    ON fetch_queue(status, priority, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_queue_lookup
    ON fetch_queue(job_type, pcs_slug, year, status);

CREATE TABLE IF NOT EXISTS data_freshness (
    entity_type TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    last_fetched_at TIMESTAMP,
    data_hash TEXT,
    ttl_seconds INTEGER DEFAULT 86400,
    PRIMARY KEY(entity_type, entity_key)
);
"""


def init_queue(conn):
    conn.executescript(_QUEUE_SCHEMA)
    conn.commit()
    _migrate_queue_if_needed(conn)


def _migrate_queue_if_needed(conn):
    """
    If fetch_queue was created before 'stage_results' was added to the
    job_type CHECK constraint, recreate the table transparently.
    The queue is transient so data loss is acceptable.
    """
    try:
        conn.execute(
            "INSERT INTO fetch_queue (job_type, pcs_slug, year) "
            "VALUES ('race_climbs', '__probe__', 0)"
        )
        conn.execute("DELETE FROM fetch_queue WHERE pcs_slug='__probe__'")
        conn.commit()
    except Exception:
        # Old schema — recreate with updated CHECK
        conn.executescript("""
            ALTER TABLE fetch_queue RENAME TO _fetch_queue_old;
            CREATE TABLE fetch_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL CHECK(job_type IN (
                    'race_meta','race_startlist','stage_results','combativity','race_climbs',
                    'rider_profile','rider_results'
                )),
                pcs_slug TEXT NOT NULL,
                year INTEGER,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN (
                    'pending','in_progress','completed','failed','permanent_fail'
                )),
                priority INTEGER DEFAULT 5,
                retries INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                next_attempt_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                UNIQUE(job_type, pcs_slug, year)
            );
            INSERT INTO fetch_queue SELECT * FROM _fetch_queue_old;
            DROP TABLE _fetch_queue_old;
        """)
        conn.commit()


def add_job(conn, job_type, pcs_slug, year=None, priority=5):
    """Idempotent: inserts only if (job_type, pcs_slug, year) not already queued."""
    conn.execute(
        """
        INSERT OR IGNORE INTO fetch_queue
            (job_type, pcs_slug, year, priority, status)
        VALUES (?, ?, ?, ?, 'pending')
        """,
        (job_type, pcs_slug, year, priority),
    )
    conn.commit()


def claim_next_job(conn):
    """
    Atomically claim the highest-priority pending job whose next_attempt_at
    has passed. Returns a dict or None if queue is empty / all jobs waiting.
    """
    now = datetime.utcnow().isoformat()
    row = conn.execute(
        """
        SELECT id, job_type, pcs_slug, year, retries, max_retries
        FROM fetch_queue
        WHERE status = 'pending' AND next_attempt_at <= ?
        ORDER BY priority ASC, next_attempt_at ASC
        LIMIT 1
        """,
        (now,),
    ).fetchone()
    if not row:
        return None

    conn.execute(
        "UPDATE fetch_queue SET status='in_progress' WHERE id=?", (row["id"],)
    )
    conn.commit()
    return dict(row)


def complete_job(conn, job_id, success, retries=0, error_msg=None):
    """Mark a job as completed or failed with exponential backoff."""
    now = datetime.utcnow().isoformat()
    if success:
        conn.execute(
            """
            UPDATE fetch_queue
            SET status='completed', completed_at=?, last_error=NULL
            WHERE id=?
            """,
            (now, job_id),
        )
    else:
        new_retries = retries + 1
        row = conn.execute(
            "SELECT max_retries FROM fetch_queue WHERE id=?", (job_id,)
        ).fetchone()
        max_retries = row["max_retries"] if row else 3

        if new_retries >= max_retries:
            conn.execute(
                """
                UPDATE fetch_queue
                SET status='permanent_fail', retries=?, last_error=?
                WHERE id=?
                """,
                (new_retries, error_msg, job_id),
            )
        else:
            backoff_secs = 2 ** new_retries
            next_attempt = (
                datetime.utcnow() + timedelta(seconds=backoff_secs)
            ).isoformat()
            conn.execute(
                """
                UPDATE fetch_queue
                SET status='pending', retries=?, last_error=?, next_attempt_at=?
                WHERE id=?
                """,
                (new_retries, error_msg, next_attempt, job_id),
            )
    conn.commit()


def is_fresh(conn, entity_type, entity_key):
    """Return True if the entity was fetched within its TTL."""
    row = conn.execute(
        """
        SELECT last_fetched_at, ttl_seconds
        FROM data_freshness
        WHERE entity_type=? AND entity_key=?
        """,
        (entity_type, entity_key),
    ).fetchone()
    if not row or not row["last_fetched_at"]:
        return False
    last = datetime.fromisoformat(row["last_fetched_at"])
    ttl = row["ttl_seconds"] or 86400
    return (datetime.utcnow() - last).total_seconds() < ttl


def mark_fresh(conn, entity_type, entity_key, data_hash=None):
    """Upsert a data_freshness row for the given entity."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO data_freshness (entity_type, entity_key, last_fetched_at, data_hash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(entity_type, entity_key) DO UPDATE SET
            last_fetched_at=excluded.last_fetched_at,
            data_hash=excluded.data_hash
        """,
        (entity_type, entity_key, now, data_hash),
    )
    conn.commit()


def seed_queue(conn, config):
    """
    Seed the queue with race_meta (priority 1) and race_startlist (priority 2)
    jobs for every race×year in the config. Idempotent.
    """
    for race in config.get("races", []):
        pcs_slug = race["pcs_slug"]
        years = race.get("history_years", [])
        for year in years:
            entity_key = f"{pcs_slug}/{year}"
            if not is_fresh(conn, "race_meta", entity_key):
                add_job(conn, "race_meta", pcs_slug, year=year, priority=1)
            if not is_fresh(conn, "race_startlist", entity_key):
                add_job(conn, "race_startlist", pcs_slug, year=year, priority=2)


def is_empty(conn):
    """Return True if there are no pending jobs ready to run now."""
    now = datetime.utcnow().isoformat()
    row = conn.execute(
        """
        SELECT COUNT(*) as n FROM fetch_queue
        WHERE status='pending' AND next_attempt_at <= ?
        """,
        (now,),
    ).fetchone()
    return row["n"] == 0
