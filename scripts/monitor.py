import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
db = PROJECT_ROOT / "data" / "cycling.db"
if not db.exists():
    print("cycling.db not found — has the runner started yet?")
    raise SystemExit

conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

print("\n=== Queue status ===")
for row in conn.execute(
    "SELECT job_type, status, COUNT(*) as n FROM fetch_queue GROUP BY 1, 2 ORDER BY 1, 2"
):
    print(f"  {row['job_type']:<20} {row['status']:<15} {row['n']}")

print("\n=== Counts ===")
for table in ("races", "riders", "teams", "race_stages", "startlist_entries", "rider_results"):
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<25} {n}")

conn.close()
