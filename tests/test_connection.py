"""Smoke test: verify we can reach procyclingstats.com."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from procyclingstats import TodayRaces


def test_today_races():
    tr = TodayRaces()
    races = tr.finished_races()
    assert isinstance(races, list), f"expected list, got {type(races)}"
    print(f"PASS: TodayRaces.finished_races() returned list ({len(races)} items)")


if __name__ == "__main__":
    failures = 0
    for name, fn in [("today_races", test_today_races)]:
        try:
            fn()
        except Exception as e:
            print(f"FAIL [{name}]: {e}")
            failures += 1
    sys.exit(failures)
