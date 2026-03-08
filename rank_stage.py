"""
Pre-race stage ranking CLI.

Usage:
    python rank_stage.py paris-nice 2026 1
    python rank_stage.py paris-nice 2026 3 --top 20
    python rank_stage.py paris-nice 2026 1 --run-models
    python rank_stage.py paris-nice 2026 1 --save
"""
import argparse
import logging
import sqlite3
import sys
from datetime import datetime

import numpy as np

from pipeline.db import get_connection, init_db, init_betting_schema, DB_PATH
from genqirue.models.stage_ranker import StageRankingModel, StageRankingResult, WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Probability assignments for each SimpleTacticalDetector state
_STATE_PROBS = {
    'CONTESTING': (0.85, 0.15),
    'PRESERVING': (0.15, 0.85),
    'RECOVERING': (0.50, 0.25),
    'GRUPPETTO':  (0.10, 0.80),
}


# ---------------------------------------------------------------------------
# --run-models: fit frailty + tactical on historical data, persist to DB
# ---------------------------------------------------------------------------

def _run_models(conn: sqlite3.Connection, race_slug: str, year: int) -> None:
    from example_betting_workflow import load_survival_data_from_db, load_tactical_data_from_db
    from genqirue.models import FastFrailtyEstimator, SimpleTacticalDetector

    hist_years = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT year FROM races WHERE pcs_slug = ? AND year < ? ORDER BY year",
            (race_slug, year),
        ).fetchall()
    ]

    if not hist_years:
        print(f"  No historical data found for {race_slug} before {year}")
        return

    print(f"  Loading data from years: {hist_years}")

    all_survival = []
    all_tactical = []
    for y in hist_years:
        surv = load_survival_data_from_db(conn, race_slug, y)
        all_survival.extend(surv.get('survival_data', []))
        tact = load_tactical_data_from_db(conn, race_slug, y)
        all_tactical.extend(tact.get('observations', []))

    computed_at = datetime.utcnow().isoformat()

    # ---- Frailty ----
    if len(all_survival) >= 10:
        print(f"  Fitting frailty model on {len(all_survival)} records...")
        estimator = FastFrailtyEstimator()
        estimator.fit(all_survival)

        fe = estimator.frailty_estimates
        if fe:
            vals = np.array(list(fe.values()), dtype=float)
            mean_f = vals.mean()
            std_f = max(vals.std(), 1e-8)
            for rider_id, frailty in fe.items():
                z = (float(frailty) - mean_f) / std_f
                hidden_form_prob = 1.0 / (1.0 + np.exp(-z)) if z > 1.5 else 0.0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rider_frailty
                        (rider_id, frailty_estimate, hidden_form_prob, computed_at, model_version)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (rider_id, float(frailty), float(hidden_form_prob), computed_at, 'fast_1.0'),
                )
            conn.commit()
            print(f"  Inserted frailty for {len(fe)} riders")
    else:
        print(f"  Insufficient survival records ({len(all_survival)}) - skipping frailty")

    # ---- Tactical ----
    if len(all_tactical) >= 10:
        print(f"  Fitting tactical detector on {len(all_tactical)} observations...")
        detector = SimpleTacticalDetector()
        inserted = 0
        for obs in all_tactical:
            state = detector.update(obs)
            cp, pp = _STATE_PROBS.get(state.name, (0.50, 0.50))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tactical_states
                        (rider_id, stage_id, contesting_prob, preserving_prob,
                         decoded_state, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (obs.rider_id, obs.stage_id, cp, pp, state.name, computed_at),
                )
                inserted += 1
            except Exception:
                pass
        conn.commit()
        print(f"  Inserted {inserted} tactical state rows")
    else:
        print(f"  Insufficient tactical observations ({len(all_tactical)}) - skipping tactical")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_result(result: StageRankingResult, top_n: int) -> None:
    race_display = result.race_slug.replace('-', ' ').title()
    dist_str = f" ({result.distance_km:.1f}km)" if result.distance_km else ""
    stage_label = result.stage_type.upper()
    if result.is_uphill_finish:
        stage_label += "/UPHILL FINISH"
    print(f"\n{race_display} {result.year} Stage {result.stage_number}"
          f" - {stage_label}{dist_str}")

    # Signals header: show active weights renormalized, inactive as [x: no data]
    weights = result.weights_used
    active = result.signals_active
    w_sum = sum(weights.get(s, 0.0) for s in active)
    all_signals = ['specialty', 'historical', 'form', 'frailty', 'tactical', 'gc_relevance']
    sig_parts = []
    for s in all_signals:
        label = s.replace('_', ' ')
        if s in active and w_sum > 0:
            sig_parts.append(f"{label}({weights[s] / w_sum:.2f})")
        else:
            sig_parts.append(f"[{label}: no data]")
    print(f"Signals: {' '.join(sig_parts)}")
    print(
        f"Field: {result.field_size} riders"
        f" | Temperature: {result.temperature:.2f}"
        f" | Edge threshold: 50bps"
    )

    # Table
    print()
    hdr = (
        f"{'Rank':>5}  {'Rider':<24}  {'Spec':>6}  {'Hist':>7}  {'Form':>6}  "
        f"{'ModelProb':>9}  {'BkOdds':>7}  {'Edge(bps)':>9}  {'Kelly%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    shown = result.riders[:top_n] if top_n else result.riders
    history_flag_used = False
    form_none_shown = False

    for i, rs in enumerate(shown, 1):
        spec_str = f"{rs.specialty_signal:.2f}" if rs.specialty_signal is not None else "  -  "
        hist_str = f"{rs.historical_signal:.2f}" if rs.historical_signal is not None else "  -  "
        if rs.no_history_flag and rs.historical_signal is not None:
            hist_str += "*"
            history_flag_used = True

        if rs.form_signal is not None:
            form_str = f"{rs.form_signal:.2f}"
        else:
            form_str = "None*"
            form_none_shown = True

        prob_str = f"{rs.model_prob:.1%}"
        odds_str = f"{rs.back_odds:.1f}" if rs.back_odds else "  -  "

        if rs.edge_bps is not None:
            edge_str = f"+{rs.edge_bps:.0f}" if rs.edge_bps >= 0 else f"{rs.edge_bps:.0f}"
        else:
            edge_str = "  -  "

        kelly_str = f"{rs.kelly_pct:.1f}%" if rs.kelly_pct else ""

        # Safe encoding for Windows console
        name = rs.rider_name[:24].encode('ascii', 'replace').decode('ascii')
        print(
            f"{i:>5}  {name:<24}  {spec_str:>6}  {hist_str:>8}  {form_str:>6}  "
            f"{prob_str:>9}  {odds_str:>7}  {edge_str:>9}  {kelly_str:>6}"
        )

    if history_flag_used:
        print("* no race history (median used)")
    if form_none_shown:
        print("* no results in last 90 days")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-race stage ranking model")
    p.add_argument("race_slug", help="Race PCS slug (e.g. paris-nice)")
    p.add_argument("year", type=int, help="Race year (e.g. 2026)")
    p.add_argument("stage", type=int, help="Stage number (e.g. 1)")
    p.add_argument("--top", type=int, default=0, metavar="N",
                   help="Show only top N riders (default: all)")
    p.add_argument("--run-models", action="store_true",
                   help="Fit frailty and tactical models before ranking")
    p.add_argument("--save", action="store_true",
                   help="Persist ranking to strategy_outputs table")
    p.add_argument("--db", default=None, metavar="PATH",
                   help="Override DB path (default: data/cycling.db)")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = args.db or DB_PATH
    conn = get_connection(db_path)
    init_db(conn)
    init_betting_schema(conn)

    if args.run_models:
        print(f"Running models for {args.race_slug} {args.year}...")
        _run_models(conn, args.race_slug, args.year)

    model = StageRankingModel(db_path)
    try:
        result = model.rank(args.race_slug, args.year, args.stage)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _format_result(result, args.top)

    if args.save:
        n = result.save_to_db(conn)
        print(f"\nSaved {n} rows to strategy_outputs.")

    conn.close()


if __name__ == "__main__":
    main()
