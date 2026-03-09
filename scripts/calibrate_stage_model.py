"""
Platt calibration for stage ranking model.

Runs the full StageRankingModel on all historical stages that have startlist
data (year < current year), collects raw_scores vs actual stage winners, then
fits two calibration parameters:

  Phase 1 — Temperature MLE:
    Find T* = argmax_T sum_stages log(softmax(T * raw_scores)[winner]).
    T* replaces the target-range binary search heuristic with a data-driven
    optimal value from historical outcomes.

  Phase 2 — Platt sigmoid:
    Using T*, compute softmax probabilities for all historical stages.
    Fit logistic regression: P(win) = sigmoid(a * logit(p) + b)
    on (logit(p), is_winner) pairs. After fitting, apply:
      raw_calib[i] = sigmoid(a * logit(softmax(T* * scores)[i]) + b)
      model_prob[i] = raw_calib[i] / sum(raw_calib)

Results are stored in the platt_calibration table and loaded automatically
by StageRankingModel._calibrate_temperature() and _apply_platt().

Usage:
    python scripts/calibrate_stage_model.py               # fit all stage types
    python scripts/calibrate_stage_model.py --stage-type flat
    python scripts/calibrate_stage_model.py --dry-run     # print only, no save
    python scripts/calibrate_stage_model.py --db data/cycling.db
"""

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db import DB_PATH, get_connection, init_betting_schema, init_db
from genqirue.models.stage_ranker import StageRankingModel, _STAGE_TYPE_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Minimum field size to include a historical stage in calibration
_MIN_FIELD = 80
# Minimum model_prob for Platt fitting (excludes floored non-contenders)
_MIN_PROB_PLATT = 1e-4
# Stage types to calibrate (map canonical key -> DB stage_type list)
_CANONICAL_TYPES = ['flat', 'hilly', 'mountain', 'itt']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(scores: np.ndarray, T: float) -> np.ndarray:
    x = T * scores
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def get_historical_races(conn: sqlite3.Connection, max_year: int) -> list:
    """Return (race_slug, year, stage_number, stage_type) for all stages with
    startlists and known winners, ordered chronologically."""
    rows = conn.execute("""
        SELECT r.pcs_slug, r.year, rs.stage_number, rs.stage_type
        FROM race_stages rs
        JOIN races r ON rs.race_id = r.id
        WHERE r.year < ?
          AND EXISTS (
              SELECT 1 FROM startlist_entries se
              JOIN races r2 ON se.race_id = r2.id
              WHERE r2.pcs_slug = r.pcs_slug AND r2.year = r.year
          )
          AND EXISTS (
              SELECT 1 FROM rider_results rr
              WHERE rr.stage_id = rs.id
                AND rr.result_category = 'stage'
                AND CAST(rr.rank AS INTEGER) = 1
          )
        ORDER BY r.year, r.pcs_slug, rs.stage_number
    """, (max_year,)).fetchall()
    return [(r['pcs_slug'], r['year'], r['stage_number'], r['stage_type'])
            for r in rows]


def get_actual_winner(conn: sqlite3.Connection, stage_id: int) -> int | None:
    """Return rider_id of the actual stage winner (rank=1), or None."""
    row = conn.execute("""
        SELECT rider_id FROM rider_results
        WHERE stage_id = ? AND result_category = 'stage'
          AND CAST(rank AS INTEGER) = 1
        LIMIT 1
    """, (stage_id,)).fetchone()
    return row['rider_id'] if row else None


# ---------------------------------------------------------------------------
# Phase 1: temperature MLE
# ---------------------------------------------------------------------------

def fit_temperature_mle(stage_data: list) -> dict:
    """
    Find T* = argmax_T sum_i log(softmax(T * scores_i)[winner_i]).

    `stage_data` is a list of (raw_scores_array, winner_array_idx) tuples,
    where raw_scores_array is the 1-D array of per-rider raw_scores from the
    full StageRankingModel._compute_raw_score() pipeline.
    """
    if not stage_data:
        return {}

    def neg_ll(T):
        total = 0.0
        for scores, widx in stage_data:
            probs = _softmax(scores, T)
            total += np.log(max(float(probs[widx]), 1e-15))
        return -total

    # Grid search first for robustness, then refine with minimize_scalar
    T_grid = np.concatenate([
        np.linspace(0.1, 2.0, 20),
        np.linspace(2.0, 30.0, 30),
        np.linspace(30.0, 100.0, 15),
    ])
    grid_ll = [neg_ll(T) for T in T_grid]
    best_grid_T = T_grid[int(np.argmin(grid_ll))]

    # Refine around the grid minimum
    lo = max(0.1, best_grid_T * 0.5)
    hi = min(200.0, best_grid_T * 2.0)
    res = minimize_scalar(neg_ll, bounds=(lo, hi), method='bounded')
    T_star = float(res.x)
    ll = -float(res.fun)

    return {
        'T_star': T_star,
        'n_stages': len(stage_data),
        'log_ll_per_stage': ll / len(stage_data),
        'grid_best_T': float(best_grid_T),
    }


# ---------------------------------------------------------------------------
# Phase 2: Platt sigmoid
# ---------------------------------------------------------------------------

def fit_platt_sigmoid(stage_data: list, T_star: float) -> dict:
    """
    Fit logistic regression on (logit(model_prob), is_winner) pairs.
    Only riders with model_prob >= _MIN_PROB_PLATT are included.
    """
    X, y = [], []
    for scores, widx in stage_data:
        probs = _softmax(scores, T_star)
        for i, p in enumerate(probs):
            if float(p) >= _MIN_PROB_PLATT:
                lp = float(logit(np.clip(p, 1e-7, 1 - 1e-7)))
                X.append(lp)
                y.append(1 if i == widx else 0)

    n_winners = sum(y)
    if not X or n_winners < 10:
        logger.warning("Only %d winner samples for Platt fit — skipping.", n_winners)
        return {}

    X_arr = np.array(X).reshape(-1, 1)
    y_arr = np.array(y)

    # C=0.1 regularisation to prevent extreme coefficients.
    # Do NOT use class_weight='balanced' — the 1:170 win imbalance is real and
    # the intercept must encode the true base rate (~0.6%). Balanced weights
    # would inflate b to ~5, making sigmoid output 99%+ for all contenders
    # before normalisation, and corrupt edge calculations.
    clf = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs')
    clf.fit(X_arr, y_arr)

    proba = clf.predict_proba(X_arr)[:, 1]
    return {
        'platt_a': float(clf.coef_[0][0]),
        'platt_b': float(clf.intercept_[0]),
        'n_samples': len(y),
        'n_winners': n_winners,
        'log_loss': float(log_loss(y_arr, proba)),
        'brier_score': float(brier_score_loss(y_arr, proba)),
    }


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def save_calibration(conn: sqlite3.Connection, stage_type: str,
                     phase1: dict, phase2: dict) -> None:
    fitted_at = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO platt_calibration
            (stage_type, T_star, platt_a, platt_b,
             n_stages, n_samples, log_loss, brier_score, fitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stage_type,
        phase1.get('T_star'),
        phase2.get('platt_a'),
        phase2.get('platt_b'),
        phase1.get('n_stages'),
        phase2.get('n_samples'),
        phase2.get('log_loss'),
        phase2.get('brier_score'),
        fitted_at,
    ))
    conn.commit()


def load_calibration(conn: sqlite3.Connection, stage_type: str) -> dict:
    row = conn.execute("""
        SELECT T_star, platt_a, platt_b, n_stages, n_samples, log_loss,
               brier_score, fitted_at
        FROM platt_calibration
        WHERE stage_type = ?
        ORDER BY fitted_at DESC
        LIMIT 1
    """, (stage_type,)).fetchone()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(stage_type: str, phase1: dict, phase2: dict,
                 existing: dict, n_skipped: int) -> None:
    prev_T = existing.get('T_star')
    T_star = phase1.get('T_star', 0)
    print(f"\n{'='*62}")
    print(f"  {stage_type.upper()}  ({phase1.get('n_stages',0)} stages, {n_skipped} skipped/errored)")
    print(f"{'='*62}")
    print(f"  Phase 1 — Temperature MLE")
    print(f"    T*              = {T_star:.3f}" +
          (f"  (was {prev_T:.3f})" if prev_T else "  (first fit)"))
    print(f"    log-LL / stage  = {phase1.get('log_ll_per_stage', 0):.4f}"
          f"  (random ~ {np.log(1/150):.2f})")

    if phase2:
        a = phase2.get('platt_a', 0)
        b = phase2.get('platt_b', 0)
        print(f"  Phase 2 — Platt sigmoid  (n_samples={phase2.get('n_samples',0)}, "
              f"winners={phase2.get('n_winners',0)})")
        print(f"    a (logit weight) = {a:.4f}  "
              f"({'sharpens' if a > 1 else 'flattens'} distribution)")
        print(f"    b (intercept)    = {b:.4f}")
        print(f"    log-loss         = {phase2.get('log_loss', 0):.4f}")
        print(f"    Brier score      = {phase2.get('brier_score', 0):.4f}")
        print(f"  Transformation preview (raw softmax prob -> calibrated):")
        for p_raw in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
            lp = float(logit(np.clip(p_raw, 1e-7, 1 - 1e-7)))
            p_calib = float(expit(a * lp + b))
            arrow = " ==" if abs(p_calib - p_raw) < 0.005 else (">>" if p_calib > p_raw else "<<")
            print(f"    {p_raw*100:5.1f}%  {arrow}  {p_calib*100:5.1f}%")
    else:
        print("  Phase 2 — Platt sigmoid: skipped (insufficient winner samples)")


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def calibrate_type(
    model: StageRankingModel,
    conn: sqlite3.Connection,
    stage_type_key: str,
    historical_races: list,
    dry_run: bool,
) -> None:
    """Run both calibration phases for one stage type and optionally save."""
    # Filter to matching stage types
    matching_db_types = {
        'flat': {'flat', 'cobbles', 'road', 'transit'},
        'hilly': {'hilly'},
        'mountain': {'mountain'},
        'itt': {'itt', 'prologue'},
    }.get(stage_type_key, set())

    stages_for_type = [
        (slug, year, stage_num)
        for slug, year, stage_num, db_type in historical_races
        if db_type in matching_db_types
    ]

    if not stages_for_type:
        print(f"\n{stage_type_key.upper()}: no historical stages found.")
        return

    logger.info("Running model on %d %s stages...", len(stages_for_type), stage_type_key)

    stage_data = []  # (raw_scores_array, winner_idx) per stage
    n_skipped = 0

    for slug, year, stage_num in stages_for_type:
        try:
            result = model._rank(conn, slug, year, stage_num)
        except (ValueError, Exception) as e:
            logger.debug("Skipping %s %d S%d: %s", slug, year, stage_num, e)
            n_skipped += 1
            continue

        if result.field_size < _MIN_FIELD:
            n_skipped += 1
            continue

        # Find actual winner among model riders
        stage_id = result.stage_id
        actual_winner_id = get_actual_winner(conn, stage_id)
        if actual_winner_id is None:
            n_skipped += 1
            continue

        rider_ids = [rs.rider_id for rs in result.riders]
        if actual_winner_id not in rider_ids:
            # Winner not in startlist (DNS? different ID?) — skip
            n_skipped += 1
            continue

        winner_idx = rider_ids.index(actual_winner_id)
        raw_scores = np.array([rs.raw_score for rs in result.riders])
        stage_data.append((raw_scores, winner_idx))

    logger.info("Collected %d usable stages (%d skipped).",
                len(stage_data), n_skipped)

    if not stage_data:
        print(f"\n{stage_type_key.upper()}: no usable data after filtering.")
        return

    # Phase 1: temperature MLE
    logger.info("Fitting temperature MLE...")
    phase1 = fit_temperature_mle(stage_data)

    # Phase 2: Platt sigmoid
    T_star = phase1.get('T_star', 1.0)
    logger.info("Fitting Platt sigmoid (T*=%.3f)...", T_star)
    phase2 = fit_platt_sigmoid(stage_data, T_star)

    existing = load_calibration(conn, stage_type_key)
    print_report(stage_type_key, phase1, phase2, existing, n_skipped)

    if not dry_run:
        save_calibration(conn, stage_type_key, phase1, phase2)
        print(f"  Saved to platt_calibration.")
    else:
        print(f"  [dry-run] Not saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Platt calibration for stage ranking model")
    p.add_argument("--stage-type",
                   choices=['flat', 'hilly', 'mountain', 'itt', 'all'],
                   default='all',
                   help="Stage type to calibrate (default: all)")
    p.add_argument("--max-year", type=int, default=2026,
                   help="Only use stages from years < max-year (default: 2026)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print calibration without saving to DB")
    p.add_argument("--db", default=None, metavar="PATH",
                   help="Override DB path (default: data/cycling.db)")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = args.db or DB_PATH

    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    init_betting_schema(conn)

    model = StageRankingModel(db_path)

    logger.info("Loading historical races with startlists...")
    historical = get_historical_races(conn, args.max_year)
    logger.info("Found %d historical stages across %d race-years.",
                len(historical),
                len({(s, y) for s, y, *_ in historical}))

    types_to_fit = _CANONICAL_TYPES if args.stage_type == 'all' \
        else [args.stage_type]

    for st in types_to_fit:
        calibrate_type(model, conn, st, historical, args.dry_run)

    conn.close()
    print()


if __name__ == "__main__":
    main()
