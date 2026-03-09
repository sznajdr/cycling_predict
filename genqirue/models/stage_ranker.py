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
    'itt':      {'specialty': 0.85, 'historical': 0.05, 'frailty': 0.05, 'tactical': 0.00, 'gc_relevance': 0.00, 'form': 0.05},
    'ttt':      {'specialty': 0.75, 'historical': 0.10, 'frailty': 0.05, 'tactical': 0.00, 'gc_relevance': 0.05, 'form': 0.05},
}

# Power-to-weight adjustment proxy for mountain stages
MEDIAN_WEIGHT_KG = 65.0

# (min_top_prob, max_top_prob) — bisect T until max(softmax) hits midpoint
TEMPERATURE_TARGET: Dict[str, tuple] = {
    'flat':     (0.20, 0.35),  # raised: field reduction shrinks effective field
    'hilly':    (0.15, 0.28),  # raised: field reduction shrinks effective field
    'mountain': (0.10, 0.18),
    'itt':      (0.25, 0.45),  # ITT: clear favorites can have higher win probs
    'ttt':      (0.10, 0.20),  # TTT: slightly higher than before
}

# Field reduction: on bunch-sprint / puncher stages, only the top-N riders
# by specialty are treated as realistic contenders. Riders outside the
# contention pool have their raw_score floored to _CONTENTION_FLOOR so they
# retain a small non-zero probability (surprise breakaways exist) without
# diluting mass away from real sprint candidates.
# 0 = no field reduction (mountain / ITT / TTT).
CONTENTION_TOP_N: Dict[str, int] = {
    'flat':     35,
    'hilly':    30,
    'mountain': 0,
    'itt':      0,
    'ttt':      0,
}
_CONTENTION_FLOOR = 0.0  # non-contenders floored to this raw_score

# ---------------------------------------------------------------------------
# Race quality weights for specialty signal calibration
# ---------------------------------------------------------------------------

# Maps races.uci_tour → quality multiplier for weighted rank percentile
UCI_CATEGORY_WEIGHTS: Dict[str, float] = {
    '2.UWT': 1.00,   # World Tour stage race (TdF, Paris-Nice, Tirreno...)
    '1.UWT': 0.95,   # World Tour one-day (Roubaix, MSR, Flanders...)
    '2.Pro': 0.50,   # ProSeries stage race
    '1.Pro': 0.45,   # ProSeries one-day
    '2.HC':  0.30,   # HC stage race (legacy UCI code)
    '1.HC':  0.25,   # HC one-day
    '2.1':   0.18,   # Continental stage race
    '1.1':   0.15,   # Continental one-day
    '2.2':   0.08,   # Sub-continental stage race
    '1.2':   0.06,   # Sub-continental one-day
    'NC':    0.12,   # National championship
    'CC':    0.08,   # Continental championship
}
_DEFAULT_UCI_WEIGHT: float = 0.05   # unrecognised or missing UCI code

# Grand Tour slugs receive a 1.3× premium on top of their base UCI weight
_GRAND_TOUR_SLUGS: frozenset = frozenset({
    'tour-de-france',
    'giro-d-italia',
    'vuelta-a-espana',
})

# Only results where the rider finished in the top fraction of the field count
# toward quality specialty. This ensures GC riders who merely survive flat stages
# (rank 50/150 = 67th percentile) contribute nothing to sprint specialty, while
# actual sprint contenders (rank 1-30) score highly.
# 0.20 = top 20% of field (≈ top 30 in a 150-rider field, matching CONTENTION_TOP_N).
_QUALITY_TOP_FRACTION: float = 0.20

# Minimum total quality weight a rider must accumulate from top-fraction finishes
# to use quality specialty. Set to 2.0 so a rider needs the equivalent of at least
# ~2 UWT-level top-20% finishes across different races before quality specialty
# activates. This prevents riders with 1-3 top results at a single race from
# dominating the normalized score (e.g., a Giro top-5 + one small-race result).
_MIN_QUALITY_WEIGHT: float = 2.0
_MIN_DISTINCT_SLUGS: int = 2

# Stage types to query for each model stage type when computing quality specialty
_QUALITY_STAGE_TYPES: Dict[str, tuple] = {
    'flat':     ('flat', 'cobbles', 'road', 'transit'),
    'hilly':    ('hilly',),
    'mountain': ('mountain',),
    'itt':      ('itt', 'prologue'),
    'ttt':      ('ttt',),
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
    is_contender: bool = True            # False = floored by field reduction
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
        
        # Delete existing data for this stage/strategy to avoid stale data accumulation
        conn.execute(
            """
            DELETE FROM strategy_outputs
            WHERE strategy_name = 'stage_ranking' AND stage_id = ?
            """,
            (self.stage_id,),
        )
        
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
                'is_contender': rs.is_contender,
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
            INSERT OR REPLACE INTO strategy_outputs
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
        # Quality-weighted specialty (from actual DB results, weighted by race UCI category).
        # Falls back to static PCS specialty for riders with insufficient quality data.
        quality_specialty = self._compute_quality_specialty_signals(conn, riders, stage_type, is_uphill_finish)
        static_specialty  = self._compute_specialty_signals(riders, stage_type, is_uphill_finish)
        specialty_scores: Dict[int, Optional[float]] = {}
        for r in riders:
            rid = r['rider_id']
            qs = quality_specialty.get(rid)
            specialty_scores[rid] = qs if qs is not None else static_specialty.get(rid)
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

        # Field reduction: floor non-contenders on sprint/puncher stages so
        # probability mass concentrates on realistic winners. Non-contenders
        # keep a small non-zero share — breakaways can still produce surprises.
        self._apply_field_reduction(rider_signals, stage_type)

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

    def _compute_quality_specialty_signals(
        self, conn, riders: list, stage_type: str, is_uphill_finish: bool
    ) -> Dict[int, Optional[float]]:
        """Quality-weighted specialty from actual DB stage results.

        For each rider, computes a weighted-average rank percentile across all
        historical stages of the matching type. Results are weighted by the UCI
        category of the race (2.UWT > 1.UWT > 2.Pro > ...) with an additional
        1.3× premium for Grand Tour stages.

        REQUIRES GRAND TOUR DATA: With only WT stage races (Paris-Nice, Tirreno)
        in the DB, quality specialty degenerates into a race-specific historical
        signal — the same information already captured by the historical signal —
        and cannot distinguish a world-class sprinter from a consistent finisher
        at one specific race. Returns None for all riders (triggering static
        PCS specialty fallback) until at least 200 Grand Tour stage results are
        present in the DB. Run scripts/fetch_calibration_data.py to seed the
        scraping queue, then python -m pipeline.runner to collect the data.

        Riders with total quality weight < _MIN_QUALITY_WEIGHT or results from
        fewer than _MIN_DISTINCT_SLUGS race circuits also fall back to static.
        """
        from collections import defaultdict

        # Gate: require substantial Grand Tour data before activating.
        gt_row = conn.execute("""
            SELECT COUNT(*) AS n FROM rider_results rr
            JOIN race_stages rs ON rr.stage_id = rs.id
            JOIN races r ON rs.race_id = r.id
            WHERE r.pcs_slug IN ('tour-de-france', 'giro-d-italia', 'vuelta-a-espana')
              AND rr.result_category = 'stage'
              AND CAST(rr.rank AS INTEGER) > 0
        """).fetchone()
        if (gt_row['n'] if gt_row else 0) < 200:
            logger.debug(
                "Quality specialty inactive: only %d Grand Tour stage results in DB "
                "(need 200+). Falling back to static PCS specialty. "
                "Run scripts/fetch_calibration_data.py to seed the scraping queue.",
                gt_row['n'] if gt_row else 0,
            )
            return {r['rider_id']: None for r in riders}

        query_types = _QUALITY_STAGE_TYPES.get(stage_type, ('flat',))
        # For uphill-finish flat stages also include hilly results
        if stage_type == 'flat' and is_uphill_finish:
            query_types = query_types + ('hilly',)

        rider_ids = [r['rider_id'] for r in riders]
        if not rider_ids:
            return {}

        type_ph = ','.join('?' * len(query_types))
        rider_ph = ','.join('?' * len(rider_ids))

        # Only count finishes in the top QUALITY_TOP_FRACTION of the field.
        # rank_pct is rescaled so that 1st place = ~1.0 and the threshold
        # position = 0.0. Results outside the threshold are excluded entirely
        # so that GC riders surviving flat stages don't accrue sprint specialty.
        top_frac = _QUALITY_TOP_FRACTION
        rows = conn.execute(f"""
            WITH field_sizes AS (
                SELECT rr2.stage_id, COUNT(*) AS field_size
                FROM rider_results rr2
                WHERE rr2.result_category = 'stage'
                  AND CAST(rr2.rank AS INTEGER) > 0
                GROUP BY rr2.stage_id
            )
            SELECT
                rr.rider_id,
                r.pcs_slug,
                r.uci_tour,
                1.0 - CAST(rr.rank AS REAL) / (fs.field_size * {top_frac}) AS rank_pct
            FROM rider_results rr
            JOIN race_stages rs ON rr.stage_id = rs.id
            JOIN races r       ON rs.race_id  = r.id
            JOIN field_sizes fs ON fs.stage_id = rr.stage_id
            WHERE rs.stage_type IN ({type_ph})
              AND rr.result_category = 'stage'
              AND CAST(rr.rank AS INTEGER) > 0
              AND CAST(rr.rank AS INTEGER) <= CAST(fs.field_size * {top_frac} AS INTEGER)
              AND rr.rider_id IN ({rider_ph})
        """, list(query_types) + rider_ids).fetchall()

        weighted_sum: Dict[int, float] = defaultdict(float)
        weight_total: Dict[int, float] = defaultdict(float)
        distinct_slugs: Dict[int, set] = defaultdict(set)

        for row in rows:
            uci = row['uci_tour'] or ''
            w = UCI_CATEGORY_WEIGHTS.get(uci, _DEFAULT_UCI_WEIGHT)
            if row['pcs_slug'] in _GRAND_TOUR_SLUGS:
                w *= 1.3
            rp = float(row['rank_pct'])
            rid = row['rider_id']
            weighted_sum[rid] += rp * w
            weight_total[rid] += w
            distinct_slugs[rid].add(row['pcs_slug'])

        quality_scores: Dict[int, float] = {
            rid: weighted_sum[rid] / tw
            for rid, tw in weight_total.items()
            if tw >= _MIN_QUALITY_WEIGHT
            and len(distinct_slugs.get(rid, set())) >= _MIN_DISTINCT_SLUGS
        }

        if not quality_scores:
            return {r['rider_id']: None for r in riders}

        vals = list(quality_scores.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn

        result: Dict[int, Optional[float]] = {}
        for r in riders:
            rid = r['rider_id']
            if rid not in quality_scores:
                result[rid] = None
            elif rng > 0:
                result[rid] = (quality_scores[rid] - mn) / rng
            else:
                result[rid] = 0.5
        return result

    def _compute_specialty_signals(
        self, riders: list, stage_type: str, is_uphill_finish: bool = False
    ) -> Dict[int, Optional[float]]:
        # --- Blending: determine which columns to mix and at what weights ---
        if stage_type == 'flat' and is_uphill_finish:
            blend = [('sp_sprint', 0.55), ('sp_climber', 0.45)]
        elif stage_type == 'hilly' and is_uphill_finish:
            blend = [('sp_hills', 0.50), ('sp_climber', 0.50)]
        elif stage_type == 'flat':
            # Blend sprint and one-day-race specialty for flat stages.
            # PCS sp_sprint is career-accumulated on pure bunch-sprint stages, which
            # systematically undervalues puncher/classics riders (e.g. Girmay, Trentin)
            # who win flat stages but accumulate points via sp_one_day_races instead.
            blend = [('sp_sprint', 0.65), ('sp_one_day_races', 0.35)]
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

        normalized: Dict[int, float] = {}
        for rid, score in raw.items():
            if rng > 0:
                normalized[rid] = (score - mn) / rng
            else:
                normalized[rid] = 0.5
        
        # --- For ITT/TTT: apply power transformation to amplify top-end differences ---
        # Squaring the normalized scores gives more separation to elite specialists
        if stage_type in ('itt', 'ttt'):
            for rid in normalized:
                normalized[rid] = normalized[rid] ** 2
        
        result: Dict[int, Optional[float]] = {}
        for r in riders:
            rid = r['rider_id']
            if rid not in raw:
                result[rid] = None
            else:
                result[rid] = normalized[rid]
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

        # Validate this stage has distance data; without it we can't map climbs correctly
        stage_dist = stage_info['cum_dist_end'] - stage_info['cum_dist_start']
        if stage_dist <= 0:
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

    def _apply_field_reduction(
        self, rider_signals: List[RiderSignals], stage_type: str
    ) -> None:
        """Floor non-contenders' raw scores on sprint/puncher stages.

        For flat and hilly stages, only the top CONTENTION_TOP_N riders by
        specialty signal are treated as realistic stage winners. All others
        receive raw_score = _CONTENTION_FLOOR so they retain a tiny non-zero
        probability (surprise breakaways happen) without diluting probability
        mass away from real contenders.

        Mountain / ITT / TTT: no filtering — any GC rider can attack.
        """
        top_n = CONTENTION_TOP_N.get(stage_type, 0)
        if top_n <= 0 or len(rider_signals) <= top_n:
            return

        sorted_by_spec = sorted(
            rider_signals,
            key=lambda r: r.specialty_signal if r.specialty_signal is not None else 0.0,
            reverse=True,
        )
        contender_ids = {r.rider_id for r in sorted_by_spec[:top_n]}

        for rs in rider_signals:
            if rs.rider_id not in contender_ids:
                rs.is_contender = False
                rs.raw_score = _CONTENTION_FLOOR

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
            # Pass 1: exact lowercase match
            odds = exact_lookup.get(rs.rider_name.lower())
            # Pass 2: NFKD accent-stripped match
            if odds is None:
                odds = norm_lookup.get(_normalize_name(rs.rider_name))
            # Pass 3: reversed-words match for "LASTNAME Firstname" PCS format.
            # PCS now returns rider names as "VLASOV Aleksandr"; Betclic stores
            # them as "Alexander Vlasov". Reversing gives "Aleksandr VLASOV" →
            # normalised "aleksandr vlasov" which may match the Betclic entry.
            if odds is None:
                parts = rs.rider_name.split()
                if len(parts) >= 2:
                    reversed_name = ' '.join(parts[1:] + [parts[0]])
                    odds = exact_lookup.get(reversed_name.lower())
                    if odds is None:
                        odds = norm_lookup.get(_normalize_name(reversed_name))
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
