"""
One-time fix: the pipeline previously ran without stage_results/combativity/race_climbs
support. This script:
  1. Creates the race_climbs table if missing
  2. Clears data_freshness for race_meta so seed_queue marks them stale
  3. Resets completed race_meta jobs to pending so they actually run again
  4. The next pipeline run will chain stage_results, combativity, race_climbs jobs

Run ONCE, then re-run: python -m pipeline.runner
"""
from pipeline.db import get_connection, init_db

conn = get_connection()

# 1. Apply current schema (adds race_climbs table via CREATE TABLE IF NOT EXISTS)
init_db(conn)
print("Schema updated (race_climbs table created if missing).")

# 2. Clear race_meta freshness
deleted = conn.execute(
    "DELETE FROM data_freshness WHERE entity_type='race_meta'"
).rowcount
conn.commit()
print(f"Cleared {deleted} race_meta freshness entries.")

# 3. Reset completed race_meta jobs to pending so they run again
reset = conn.execute(
    """
    UPDATE fetch_queue
    SET status='pending', retries=0, last_error=NULL,
        next_attempt_at=CURRENT_TIMESTAMP, completed_at=NULL
    WHERE job_type='race_meta' AND status='completed'
    """
).rowcount
conn.commit()
print(f"Reset {reset} race_meta jobs to pending.")

# 4. Show queue
rows = conn.execute(
    "SELECT job_type, status, COUNT(*) n FROM fetch_queue GROUP BY 1, 2 ORDER BY 1, 2"
).fetchall()
print("\nCurrent queue:")
for r in rows:
    print(f"  {r['job_type']:<20} {r['status']:<15} {r['n']}")

conn.close()
print("\nDone. Now run: python -m pipeline.runner")
