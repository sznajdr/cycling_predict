"""
Stage Ranking Model.

Combines six signals (specialty, historical, frailty, tactical, gc_relevance,
form) into softmax probabilities, computes edge vs. live Betclic odds, and
sizes stakes via half-Kelly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight matrix and softmax temperature targets
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, Dict[str, float]] = {
    'flat':     {'specialty': 0.30, 'historical': 0.25, 'frailty': 0.15, 'tactical': 0.10, 'gc_relevance': 0.05, 'form': 0.15},
    'hilly':    {'specialty': 0.25, 'historical': 0.25, 'frailty': 0.15, 'tactical': 0.10, 'gc_relevance': 0.10, 'form': 0.15},
    'mountain': {'specialty': 0.20, 'historical': 0.20, 'frailty': 0.15, 'tactical': 0.10, 'gc_relevance': 0.20, 'form': 0.15},
    'itt':      {'specialty': 0.40, 'historical': 0.25, 'frailty': 0.15, 'tactical': 0.00, 'gc_relevance': 0.05, 'form': 0.15},
    'ttt':      {'specialty': 0.30, 'historical': 0.25, 'frailty': 0.15, 'tactical': 0.00, 'gc_relevance': 0.10, 'form': 0.20},
}

# Power-to-weight adjustment proxy for mountain stages
MEDIAN_WEIGHT_KG = 65.0

# (min_top_prob, max_top_prob) — bisect T until max(softmax) hits midpoint
TEMPERATURE_TARGET: Dict[str, tuple] = {
    'flat':     (0.12, 0.22),
    'hilly':    (0.10, 0.18),
    'mountain': (0.10, 0.18),
    'itt':      (0.15, 0.25),
    'ttt':      (0.05, 0.15),
}

# PCS specialty column per normalized stage type
_SPECIALTY_COL: Dict[str, str] = {
    'flat':     'sp_sprint',
    'hilly':    'sp_hills',
    'mountain': 'sp_climber',
    'itt':      'sp_time_trial',
    'ttt':      'sp_time_trial',
}

# DB stage_type strings → normalized key used in WEIGHTS / _SPECIALTY_COL
_STAGE_TYPE_MAP: Dict[str, str] = {
    'flat':      'flat',
    'hilly':     'hilly',
    'mountain':  'mountain',
    'itt':       'itt',
    'ttt':       'ttt',
    'road':      'flat',
    'prologue':  'itt',
    'cobbles':   'flat',
    'transit':   'flat',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiderSignals:
    rider_id: int
    rider_name: str
    specialty_signal: Optional[float]    # None = no specialty score
    historical_signal: Optional[float]   # None = no race history
    frailty_signal: Optional[float]      # None = rider_frailty table empty
    tactical_signal: Optional[float]     # None = Stage 1 or tactical_states empty
    gc_relevance_signal: float           # always set (0.5 = neutral/no data)
    form_signal: Optional[float] = None  # None = no races in last 90 days
    form_n_races: int = 0                # how many recent stages contributed
    no_history_flag: bool = False        # True = median fallback used
    raw_score: float = 0.0
    model_prob: float = 0.0
    implied_prob: Optional[float] = None
    back_odds: Optional[float] = None
    edge_bps: Optional[float] = None
    kelly_pct: Optional[float] = None


@dataclass
class StageRankingResult:
    race_slug: str
    year: int
    stage_number: int
    stage_type: str
    distance_km: Optional[float]
    stage_id: int
    field_size: int
    temperature: float
    weights_used: Dict[str, float]
    signals_active: List[str]
    riders: List[RiderSignals]
    is_uphill_finish: bool = False
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def save_to_db(self, conn: sqlite3.Connection) -> int:
        """Persist ranking to strategy_outputs. Returns number of rows inserted."""
        computed_at_str = self.computed_at.isoformat()
        rows = []
        for rs in self.riders:
            latent = {
                'specialty': rs.specialty_signal,
                'historical': rs.historical_signal,
                'frailty': rs.frailty_signal,
                'tactical': rs.tactical_signal,
                'gc_relevance': rs.gc_relevance_signal,
                'form': rs.form_signal,
                'form_n_races': rs.form_n_races,
                'raw_score': rs.raw_score,
                'no_history_flag': rs.no_history_flag,
                'temperature': self.temperature,
                'stage_type': self.stage_type,
                'is_uphill_finish': self.is_uphill_finish,
            }
            rows.append((
                'stage_ranking',
                rs.rider_id,
                self.stage_id,
                rs.model_prob,
                0.0,
                rs.edge_bps or 0.0,
                0.0,
                json.dumps(latent),
                computed_at_str,
            ))
        conn.executemany(
            """
            INSERT OR IGNORE INTO strategy_outputs
                (strategy_name, rider_id, stage_id, win_prob, win_prob_std,
                 edge_bps, expected_value, latent_states_json, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip accents and return ASCII lowercase for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ascii", "ignore").decode().strip().lower()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StageRankingModel:
    """
    Pre-race stage ranking model.

    Combines five signals into softmax probabilities calibrated to a
    realistic top-probability range, then layers on edge and Kelly sizing.
    """

    def __init__(self, db_path: str = 'data/cycling.db'):
        self.db_path = db_path

    def rank(self, race_slug: str, year: int, stage_number: int) -> StageRankingResult:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return self._rank(conn, race_slug, year, stage_number)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Core ranking pipeline
    # ------------------------------------------------------------------

    def _rank(self, conn, race_slug: str, year: int, stage_number: int) -> StageRankingResult:
        stage = self._get_stage(conn, race_slug, year, stage_number)
        if stage is None:
            raise ValueError(f"Stage not found: {race_slug} {year} stage {stage_number}")

        stage_type = _STAGE_TYPE_MAP.get((stage['stage_type'] or '').lower(), 'flat')
        stage_id = stage['stage_id']
        distance_km = stage['distance_km']

        riders = self._get_startlist(conn, race_slug, year)
        if not riders:
            raise ValueError(f"No startlist found for {race_slug} {year}")
        field_size = len(riders)

        is_uphill_finish, _ = self._get_finish_type(conn, stage_id)

        # Compute all six signals
        specialty_scores  = self._compute_specialty_signals(riders, stage_type, is_uphill_finish)
        historical_scores = self._compute_historical_signals(conn, riders, race_slug, year, stage_type)
        frailty_scores    = self._compute_frailty_signals(conn, riders)
        tactical_scores   = self._compute_tactical_signals(conn, riders, race_slug, year, stage_number, stage_type)
        gc_scores         = self._compute_gc_signals(conn, riders, race_slug, year, stage_number, stage_type)
        rider_ids         = [r['rider_id'] for r in riders]
        form_scores       = self._compute_form_signals(conn, rider_ids, race_slug, year)

        weights = WEIGHTS.get(stage_type, WEIGHTS['flat'])
        rider_signals: List[RiderSignals] = []
        for r in riders:
            rid = r['rider_id']
            hist = historical_scores.get(rid, {})
            form_entry = form_scores.get(rid, {})
            rs = RiderSignals(
                rider_id=rid,
                rider_name=r['rider_name'],
                specialty_signal=specialty_scores.get(rid),
                historical_signal=hist.get('signal'),
                frailty_signal=frailty_scores.get(rid),
                tactical_signal=tactical_scores.get(rid),
                gc_relevance_signal=gc_scores.get(rid, 0.5),
                form_signal=form_entry.get('signal'),
                form_n_races=form_entry.get('n_races', 0),
                no_history_flag=hist.get('fallback', False),
            )
            rs.raw_score = self._compute_raw_score(rs, weights)
            rider_signals.append(rs)

        # Softmax with temperature calibration
        scores = np.array([r.raw_score for r in rider_signals])
        T = self._calibrate_temperature(scores, stage_type)
        probs = self._softmax(T * scores)
        for i, rs in enumerate(rider_signals):
            rs.model_prob = float(probs[i])

        self._join_odds(conn, rider_signals)
        self._compute_kelly(rider_signals)

        rider_signals.sort(key=lambda r: r.model_prob, reverse=True)

        # Signals with at least one non-None value across the field
        signals_active: List[str] = []
        for sig in ('specialty', 'historical', 'frailty', 'tactical', 'gc_relevance', 'form'):
            attr = f'{sig}_signal'
            if any(getattr(r, attr, None) is not None for r in rider_signals):
                signals_active.append(sig)

        return StageRankingResult(
            race_slug=race_slug,
            year=year,
            stage_number=stage_number,
            stage_type=stage_type,
            distance_km=distance_km,
            stage_id=stage_id,
            field_size=field_size,
            temperature=T,
            weights_used=weights,
            signals_active=signals_active,
            riders=rider_signals,
            is_uphill_finish=is_uphill_finish,
        )

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_stage(self, conn, race_slug: str, year: int, stage_number: int):
        return conn.execute("""
            SELECT rs.id AS stage_id, rs.stage_type, rs.distance_km, rs.avg_temp_c
            FROM race_stages rs
            JOIN races r ON rs.race_id = r.id
            WHERE r.pcs_slug = ? AND r.year = ? AND rs.stage_number = ?
        """, (race_slug, year, stage_number)).fetchone()

    def _get_startlist(self, conn, race_slug: str, year: int) -> list:
        return conn.execute("""
            SELECT se.rider_id, ri.name AS rider_name,
                ri.sp_sprint, ri.sp_hills, ri.sp_climber,
                ri.sp_time_trial, ri.sp_gc, ri.sp_one_day_races,
                ri.weight_kg
            FROM startlist_entries se
            JOIN riders ri ON se.rider_id = ri.id
            JOIN races r ON se.race_id = r.id
            WHERE r.pcs_slug = ? AND r.year = ?
        """, (race_slug, year)).fetchall()

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_specialty_signals(
        self, riders: list, stage_type: str, is_uphill_finish: bool = False
    ) -> Dict[int, Optional[float]]:
        # --- Blending: determine which columns to mix and at what weights ---
        if stage_type == 'flat' and is_uphill_finish:
            blend = [('sp_sprint', 0.40), ('sp_climber', 0.60)]
        elif stage_type == 'hilly' and is_uphill_finish:
            blend = [('sp_hills', 0.50), ('sp_climber', 0.50)]
        else:
            blend = [(_SPECIALTY_COL.get(stage_type, 'sp_sprint'), 1.0)]

        # --- Compute blended raw score per rider ---
        raw: Dict[int, float] = {}
        for r in riders:
            rid = r['rider_id']
            score = 0.0
            total_alpha = 0.0
            for col, alpha in blend:
                val = r[col]
                if val is not None:
                    score += alpha * float(val)
                    total_alpha += alpha
            if total_alpha > 0:
                raw[rid] = score / total_alpha  # renormalise if a col was missing

        if not raw:
            return {r['rider_id']: None for r in riders}

        # --- Power-to-weight adjustment for mountain/uphill-hilly stages ---
        apply_ptw = stage_type == 'mountain' or (stage_type == 'hilly' and is_uphill_finish)
        if apply_ptw:
            adjusted: Dict[int, float] = {}
            for r in riders:
                rid = r['rider_id']
                if rid not in raw:
                    continue
                wkg = r['weight_kg']
                if wkg:
                    factor = MEDIAN_WEIGHT_KG / float(wkg)
                    factor = max(0.80, min(1.30, factor))
                    adjusted[rid] = raw[rid] * factor
                else:
                    adjusted[rid] = raw[rid]
            raw = adjusted

        # --- Min-max normalise across the field ---
        vals = list(raw.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn

        result: Dict[int, Optional[float]] = {}
        for r in riders:
            rid = r['rider_id']
            if rid not in raw:
                result[rid] = None
            elif rng > 0:
                result[rid] = (raw[rid] - mn) / rng
            else:
                result[rid] = 0.5  # all riders tied
        return result

    def _compute_historical_signals(
        self, conn, riders: list, race_slug: str, year: int, stage_type: str
    ) -> Dict[int, dict]:
        rows = conn.execute("""
            SELECT rr.rider_id,
                AVG(1.0 - CAST(rr.rank AS REAL) / field_sizes.field_size) AS avg_rank_pct
            FROM rider_results rr
            JOIN race_stages rs ON rr.stage_id = rs.id
            JOIN races r ON rs.race_id = r.id
            JOIN (
                SELECT rs2.id AS stage_id, COUNT(*) AS field_size
                FROM rider_results rr2
                JOIN race_stages rs2 ON rr2.stage_id = rs2.id
                JOIN races r2 ON rs2.race_id = r2.id
                WHERE r2.pcs_slug = ? AND r2.year < ?
                  AND rr2.result_category = 'stage'
                  AND CAST(rr2.rank AS INTEGER) > 0
                GROUP BY rs2.id
            ) field_sizes ON field_sizes.stage_id = rr.stage_id
            WHERE r.pcs_slug = ? AND r.year < ?
              AND rr.result_category = 'stage'
              AND rs.stage_type = ?
              AND CAST(rr.rank AS INTEGER) > 0
              AND rr.rider_id IN (
                  SELECT se.rider_id FROM startlist_entries se
                  JOIN races r3 ON se.race_id = r3.id
                  WHERE r3.pcs_slug = ? AND r3.year = ?
              )
            GROUP BY rr.rider_id
        """, (race_slug, year, race_slug, year, stage_type, race_slug, year)).fetchall()

        has_history: Dict[int, float] = {
            r['rider_id']: float(r['avg_rank_pct'])
            for r in rows
            if r['avg_rank_pct'] is not None
        }
        if not has_history:
            return {}  # all riders → None historical signal

        median_val = float(np.median(list(has_history.values())))
        result: Dict[int, dict] = {}
        for r in riders:
            rid = r['rider_id']
            if rid in has_history:
                result[rid] = {'signal': has_history[rid], 'fallback': False}
            else:
                result[rid] = {'signal': median_val, 'fallback': True}
        return result

    def _compute_frailty_signals(self, conn, riders: list) -> Dict[int, float]:
        rows = conn.execute("""
            SELECT rider_id, hidden_form_prob
            FROM rider_frailty
            WHERE computed_at = (SELECT MAX(computed_at) FROM rider_frailty)
        """).fetchall()

        if not rows:
            return {}  # all riders → None frailty signal

        db_vals = {r['rider_id']: float(r['hidden_form_prob']) for r in rows}
        rider_ids = {r['rider_id'] for r in riders}
        return {rid: db_vals[rid] for rid in rider_ids if rid in db_vals}

    def _compute_tactical_signals(
        self, conn, riders: list, race_slug: str, year: int,
        stage_number: int, stage_type: str
    ) -> Dict[int, float]:
        if stage_number <= 1:
            return {}  # all riders → None tactical signal

        rows = conn.execute("""
            SELECT ts.rider_id, ts.preserving_prob
            FROM tactical_states ts
            JOIN race_stages rs ON ts.stage_id = rs.id
            JOIN races r ON rs.race_id = r.id
            WHERE r.pcs_slug = ? AND r.year = ? AND rs.stage_number = ?
        """, (race_slug, year, stage_number - 1)).fetchall()

        if not rows:
            return {}  # all riders → None tactical signal

        pres_map = {r['rider_id']: float(r['preserving_prob']) for r in rows}
        rider_ids = {r['rider_id'] for r in riders}

        result: Dict[int, float] = {}
        for rid in rider_ids:
            if rid not in pres_map:
                continue
            p = pres_map[rid]
            if stage_type in ('flat', 'hilly'):
                result[rid] = p          # was preserving on mountain → better on flat
            elif stage_type == 'mountain':
                result[rid] = 1.0 - p   # was contesting → likely to contest again
            else:
                result[rid] = 0.5
        return result

    def _compute_gc_signals(
        self, conn, riders: list, race_slug: str, year: int,
        stage_number: int, stage_type: str
    ) -> Dict[int, float]:
        if stage_number <= 1:
            return {r['rider_id']: 0.5 for r in riders}

        rows = conn.execute("""
            SELECT rr.rider_id, CAST(rr.rank AS INTEGER) AS gc_rank
            FROM rider_results rr
            JOIN race_stages rs ON rr.stage_id = rs.id
            JOIN races r ON rs.race_id = r.id
            WHERE r.pcs_slug = ? AND r.year = ? AND rs.stage_number = ?
              AND rr.result_category = 'gc'
              AND CAST(rr.rank AS INTEGER) > 0
        """, (race_slug, year, stage_number - 1)).fetchall()

        gc_map = {r['rider_id']: r['gc_rank'] for r in rows}
        result: Dict[int, float] = {}

        for r in riders:
            rid = r['rider_id']
            gc_rank = gc_map.get(rid)

            if gc_rank is None:
                # No GC standing — assume non-contender
                if stage_type in ('flat', 'hilly'):
                    result[rid] = 0.90
                elif stage_type == 'mountain':
                    result[rid] = 0.20
                else:
                    result[rid] = 0.50
            elif stage_type in ('flat', 'hilly'):
                if gc_rank <= 10:
                    result[rid] = 0.10   # GC man won't risk on flat
                elif gc_rank <= 30:
                    result[rid] = 0.60
                else:
                    result[rid] = 0.90
            elif stage_type == 'mountain':
                if gc_rank <= 10:
                    result[rid] = 0.90   # GC men go all out on mountains
                elif gc_rank <= 20:
                    result[rid] = 0.60
                else:
                    result[rid] = 0.20
            else:
                result[rid] = 0.50
        return result

    def _get_finish_type(self, conn, stage_id: int):
        """Return (is_uphill, dist_km) based on closest climb to this stage's finish.

        race_climbs.km_before_finish is relative to the race finish (not stage
        finish). We map climbs to stages using cumulative stage distances, then
        check whether any climb ends within 2 km of this stage's finish line.
        """
        # Get this stage's race + cumulative distances via window functions
        stage_info = conn.execute("""
            WITH stage_cumulative AS (
                SELECT
                    id,
                    race_id,
                    stage_number,
                    SUM(COALESCE(distance_km, 0)) OVER (
                        PARTITION BY race_id
                        ORDER BY stage_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cum_dist_end,
                    COALESCE(SUM(COALESCE(distance_km, 0)) OVER (
                        PARTITION BY race_id
                        ORDER BY stage_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ), 0) AS cum_dist_start,
                    SUM(COALESCE(distance_km, 0)) OVER (
                        PARTITION BY race_id
                    ) AS total_race_dist
                FROM race_stages
            )
            SELECT race_id, total_race_dist, cum_dist_end, cum_dist_start
            FROM stage_cumulative
            WHERE id = ?
        """, (stage_id,)).fetchone()

        if stage_info is None:
            return (False, None)

        race_id = stage_info['race_id']
        total = stage_info['total_race_dist'] or 0
        if total == 0:
            return (False, None)

        # km_before_finish is measured from the race finish.
        # Climbs in this stage have km_before_finish in:
        #   [(total - cum_dist_end), (total - cum_dist_start)]
        # The closest climb to this stage's finish line has the
        # smallest km_before_finish in that range.
        stage_finish_kbf = total - (stage_info['cum_dist_end'] or 0)
        stage_start_kbf  = total - (stage_info['cum_dist_start'] or 0)

        row = conn.execute("""
            SELECT km_before_finish, climb_name, steepness_pct
            FROM race_climbs
            WHERE race_id = ?
              AND km_before_finish IS NOT NULL
              AND km_before_finish >= ?
              AND km_before_finish <= ?
            ORDER BY km_before_finish ASC
            LIMIT 1
        """, (race_id, stage_finish_kbf, stage_start_kbf)).fetchone()

        if row is None:
            return (False, None)

        # Distance from that climb's top to this stage's finish line
        dist_from_finish = float(row['km_before_finish']) - stage_finish_kbf
        if dist_from_finish <= 2.0:
            return (True, round(dist_from_finish, 2))
        return (False, None)

    def _compute_form_signals(
        self, conn, rider_ids: list, race_slug: str, year: int
    ) -> Dict[int, dict]:
        """Exponentially-decayed rank percentile across all races in last 90 days."""
        if not rider_ids:
            return {}

        placeholders = ','.join('?' * len(rider_ids))
        rows = conn.execute(f"""
            WITH field_sizes AS (
                SELECT rr2.stage_id, COUNT(*) AS field_size
                FROM rider_results rr2
                WHERE rr2.result_category = 'stage'
                  AND CAST(rr2.rank AS INTEGER) > 0
                GROUP BY rr2.stage_id
            ),
            decayed AS (
                SELECT
                    rr.rider_id,
                    (1.0 - CAST(rr.rank AS REAL) / fs.field_size)
                        * EXP(-CAST(JULIANDAY('now') - JULIANDAY(rs.stage_date) AS REAL) / 30.0) AS weighted_pct,
                    EXP(-CAST(JULIANDAY('now') - JULIANDAY(rs.stage_date) AS REAL) / 30.0) AS weight,
                    COUNT(*) OVER (PARTITION BY rr.rider_id) AS n_obs
                FROM rider_results rr
                JOIN race_stages rs ON rr.stage_id = rs.id
                JOIN field_sizes fs ON fs.stage_id = rr.stage_id
                WHERE rr.result_category = 'stage'
                  AND CAST(rr.rank AS INTEGER) > 0
                  AND rs.stage_date >= DATE('now', '-90 days')
                  AND rr.rider_id IN ({placeholders})
            )
            SELECT rider_id,
                   SUM(weighted_pct) / SUM(weight) AS form_score,
                   MAX(n_obs) AS n_races
            FROM decayed
            GROUP BY rider_id
        """, rider_ids).fetchall()

        result: Dict[int, dict] = {}
        for row in rows:
            score = row['form_score']
            n = row['n_races'] or 0
            if score is not None and n > 0:
                result[row['rider_id']] = {
                    'signal': max(0.0, min(1.0, float(score))),
                    'n_races': int(n),
                }
        return result

    # ------------------------------------------------------------------
    # Score, softmax, temperature calibration
    # ------------------------------------------------------------------

    def _compute_raw_score(self, rs: RiderSignals, weights: Dict[str, float]) -> float:
        signal_map = {
            'specialty':    rs.specialty_signal,
            'historical':   rs.historical_signal,
            'frailty':      rs.frailty_signal,
            'tactical':     rs.tactical_signal,
            'gc_relevance': rs.gc_relevance_signal,
            'form':         rs.form_signal,
        }
        active = {k: v for k, v in signal_map.items() if v is not None}
        if not active:
            return 0.0
        w_sum = sum(weights.get(k, 0.0) for k in active)
        if w_sum == 0:
            return 0.0
        return sum(v * weights.get(k, 0.0) / w_sum for k, v in active.items())

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - x.max()
        e = np.exp(x_shifted)
        return e / e.sum()

    def _calibrate_temperature(self, scores: np.ndarray, stage_type: str) -> float:
        """Binary search for T so that max(softmax(T*scores)) ≈ midpoint of target range."""
        if scores.std() < 1e-8:
            return 1.0
        lo_prob, hi_prob = TEMPERATURE_TARGET.get(stage_type, (0.10, 0.20))
        target = (lo_prob + hi_prob) / 2.0

        T_lo, T_hi = 0.1, 100.0
        for _ in range(60):
            T_mid = (T_lo + T_hi) / 2.0
            top_prob = self._softmax(T_mid * scores).max()
            if top_prob < target:
                T_lo = T_mid
            else:
                T_hi = T_mid
        return (T_lo + T_hi) / 2.0

    # ------------------------------------------------------------------
    # Odds join and Kelly sizing
    # ------------------------------------------------------------------

    def _join_odds(self, conn, rider_signals: List[RiderSignals]) -> None:
        rows = conn.execute("""
            SELECT participant_name, participant_name_norm, back_odds
            FROM bookmaker_odds_latest
            WHERE market_type = 'winner'
        """).fetchall()

        exact_lookup: Dict[str, float] = {}
        norm_lookup: Dict[str, float] = {}
        for row in rows:
            name = (row['participant_name'] or '').lower()
            norm = (row['participant_name_norm'] or '').lower()
            odds = row['back_odds']
            if odds and float(odds) > 1.0:
                exact_lookup[name] = float(odds)
                if norm:
                    norm_lookup[norm] = float(odds)

        for rs in rider_signals:
            odds = exact_lookup.get(rs.rider_name.lower())
            if odds is None:
                odds = norm_lookup.get(_normalize_name(rs.rider_name))
            if odds is not None and odds > 1.0:
                rs.back_odds = odds
                rs.implied_prob = 1.0 / odds

    def _compute_kelly(self, rider_signals: List[RiderSignals]) -> None:
        for rs in rider_signals:
            if rs.back_odds is None or rs.back_odds <= 1.0 or rs.implied_prob is None:
                continue
            rs.edge_bps = (rs.model_prob - rs.implied_prob) * 10_000.0
            if rs.edge_bps > 50.0:
                b = rs.back_odds - 1.0
                full_kelly = max(0.0, (b * rs.model_prob - (1.0 - rs.model_prob)) / b)
                rs.kelly_pct = round(full_kelly * 0.5 * 100.0, 1)
