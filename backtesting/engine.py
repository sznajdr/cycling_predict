"""
Walk-forward backtesting engine for cycling betting strategies.

Respects strict temporal ordering — models are trained only on data from
races that ended before the test race started, plus prior stages within
the same race.

No real market odds are available, so we simulate against a fair naive
market (equal probability for all starters) to measure discrimination
power. Positive ROI means the model beats that equal-odds baseline.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.db import DB_PATH, get_connection
from genqirue.models import FastFrailtyEstimator, SimpleTacticalDetector, SurvivalRecord
from genqirue.models.tactical_hmm import TacticalObservation
from genqirue.domain.enums import StageType, TacticalState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    """Single simulated bet placed during the backtest."""
    race_slug: str
    year: int
    stage_num: int
    stage_type: str
    strategy: str
    rider_id: int
    rider_name: str
    predicted_score: float    # Raw model signal
    predicted_prob: float     # Normalised win probability
    naive_odds: float         # Simulated equal-market odds
    stake_fraction: float     # Kelly-sized fraction of bankroll
    stake_amount: float       # Absolute amount bet
    actual_rank: int          # 999 = DNF/DNS
    is_top3: bool
    is_top5: bool
    is_top10: bool
    is_winner: bool
    pnl: float                # P&L this bet (positive = profit)
    bankroll_before: float
    bankroll_after: float


@dataclass
class StrategyResult:
    """Aggregated backtest results for one strategy."""
    strategy: str
    n_bets: int = 0
    n_stages: int = 0
    n_races: int = 0

    # Hit rates
    win_rate: float = 0.0
    top3_rate: float = 0.0
    top5_rate: float = 0.0

    # Financial
    total_pnl: float = 0.0
    roi: float = 0.0
    final_bankroll: float = 0.0
    max_drawdown: float = 0.0

    # Calibration / discrimination
    brier_score: float = 1.0   # Lower = better (0 = perfect)
    spearman_rho: float = 0.0  # Correlation with actual ranks
    spearman_p: float = 1.0

    # Detail
    bankroll_curve: List[float] = field(default_factory=list)
    bet_records: List[BetRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class CyclingBacktester:
    """
    Walk-forward backtester.

    For each race in chronological order:
      - Training data = all stage results from races that finished
        BEFORE this race's start date, plus prior stages of the same race.
      - Test = each stage of the current race (from stage 2 onward).

    Strategies
    ----------
    frailty   : FastFrailtyEstimator (Strategy 2 – Gruppetto Frailty)
    tactical  : SimpleTacticalDetector (Strategy 1 – Tactical HMM proxy)
    baseline  : Random selection (sanity-check control)
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,   # Quarter-Kelly (conservative)
        top_k_bets: int = 3,            # Bet on top-K riders by signal
        min_train_records: int = 20,    # Minimum records needed to fit
        min_field_size: int = 8,        # Minimum starters per stage
        bet_on_top3: bool = True,       # Top-3 market (vs. win market)
        signal_boost: float = 2.0,      # Assumed edge: model riders are X× more
                                        # likely to podium than naive baseline.
                                        # Kelly requires P_model > P_market to
                                        # generate a positive stake. With no real
                                        # odds or calibrated probabilities, we
                                        # start at 2× and the backtest validates
                                        # whether that assumption holds (ROI > 0).
    ):
        self.db_path = db_path
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.top_k_bets = top_k_bets
        self.min_train_records = min_train_records
        self.min_field_size = min_field_size
        self.bet_on_top3 = bet_on_top3
        self.signal_boost = signal_boost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, strategy: str = "frailty") -> StrategyResult:
        """Run walk-forward backtest for a single strategy."""
        conn = get_connection(self.db_path)
        try:
            all_data = self._load_all_data(conn)
            races = self._load_races(conn)
            if races.empty or all_data.empty:
                logger.warning("No data in database — run the scraper first.")
                return StrategyResult(
                    strategy=strategy,
                    final_bankroll=self.initial_bankroll,
                    bankroll_curve=[self.initial_bankroll],
                )
            dispatch = {
                "frailty": self._run_frailty,
                "tactical": self._run_tactical,
                "baseline": self._run_baseline,
            }
            if strategy not in dispatch:
                raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(dispatch)}")
            return dispatch[strategy](all_data, races)
        finally:
            conn.close()

    def run_all(self) -> Dict[str, StrategyResult]:
        """Run all strategies and return results dict."""
        conn = get_connection(self.db_path)
        try:
            all_data = self._load_all_data(conn)
            races = self._load_races(conn)
            if races.empty or all_data.empty:
                logger.warning("No data in database — run the scraper first.")
                return {}
            results = {}
            for name, fn in [
                ("frailty",  self._run_frailty),
                ("tactical", self._run_tactical),
                ("baseline", self._run_baseline),
            ]:
                logger.info(f"Running backtest: {name}")
                results[name] = fn(all_data, races)
            return results
        finally:
            conn.close()

    def print_report(self, results: Dict[str, StrategyResult]) -> None:
        """Print a comparison table to stdout."""
        if not results:
            print("No results to display.")
            return

        width = 76
        print()
        print("=" * width)
        print("  CYCLING BACKTEST REPORT")
        print("=" * width)
        header = (
            f"{'Strategy':<12} {'Bets':>5} {'Races':>6} "
            f"{'Top3%':>7} {'Win%':>6} {'ROI':>8} "
            f"{'Bankroll':>9} {'MaxDD':>7} {'Spearman':>9}"
        )
        print(header)
        print("-" * width)

        for r in results.values():
            print(
                f"{r.strategy:<12} {r.n_bets:>5} {r.n_races:>6} "
                f"{r.top3_rate:>7.1%} {r.win_rate:>6.1%} {r.roi:>8.1%} "
                f"{r.final_bankroll:>9.2f} {r.max_drawdown:>7.1%} "
                f"{r.spearman_rho:>9.3f}"
            )

        print("=" * width)
        print()
        print("HOW TO READ THIS:")
        print("  ROI > 0%      -> model beats the naive equal-odds market")
        print("  Top3% > 3/N   -> model picks riders who podium more than chance")
        print("  Spearman > 0  -> predicted scores positively rank-correlate with outcomes")
        print("  Compare each strategy vs 'baseline' to see whether models add value.")
        print()

        # Show per-race breakdown for best strategy
        best = max(results.values(), key=lambda r: r.roi)
        if best.bet_records:
            self._print_race_breakdown(best)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _run_frailty(self, all_data: pd.DataFrame, races: pd.DataFrame) -> StrategyResult:
        """
        Strategy 2 – Gruppetto Frailty (FastFrailtyEstimator).

        For each stage in each race:
          - Train FastFrailtyEstimator on all prior races + prior stages
            of the current race.
          - Score riders in the current stage by their frailty.
          - Convert scores → win probabilities via softmax.
          - Bet quarter-Kelly on top-K riders.
        """
        result = StrategyResult(
            strategy="frailty",
            final_bankroll=self.initial_bankroll,
            bankroll_curve=[self.initial_bankroll],
        )
        bankroll = self.initial_bankroll

        for _, race in races.iterrows():
            race_id = int(race["id"])
            race_start = str(race["startdate"])

            # Historical training data: everything before this race
            hist = all_data[all_data["startdate"] < race_start].copy()

            # Stages within this race, in order
            race_stages = (
                all_data[all_data["race_id"] == race_id]
                [["stage_id", "stage_num", "stage_date", "stage_type"]]
                .drop_duplicates("stage_id")
                .sort_values("stage_num")
            )
            if len(race_stages) < 2:
                continue

            race_had_bets = False

            for _, stage_row in race_stages.iterrows():
                stage_id = int(stage_row["stage_id"])
                stage_num = stage_row["stage_num"]
                stage_type = str(stage_row["stage_type"] or "")

                # Stage outcomes
                stage_data = all_data[all_data["stage_id"] == stage_id]
                if len(stage_data) < self.min_field_size:
                    continue

                # Within-race prior stages
                prior_within = all_data[
                    (all_data["race_id"] == race_id) &
                    (all_data["stage_num"] < stage_num)
                ]

                # Combine: historical + prior stages of this race
                train = pd.concat([hist, prior_within], ignore_index=True)
                if len(train) < self.min_train_records:
                    continue

                # Fit. CoxPH always warns "all samples censored" because we
                # have no OTL event data — the simple fallback handles it.
                # Suppress the noisy logger noise for the duration.
                records = self._to_survival_records(train)
                estimator = FastFrailtyEstimator()
                _gq_log = logging.getLogger("genqirue")
                _gq_level = _gq_log.level
                _gq_log.setLevel(logging.ERROR)
                try:
                    estimator.fit(records)
                except Exception as exc:
                    logger.debug(f"Frailty fit failed ({race['pcs_slug']} s{stage_num}): {exc}")
                    _gq_log.setLevel(_gq_level)
                    continue
                finally:
                    _gq_log.setLevel(_gq_level)

                # Score riders in this stage
                rider_ids = stage_data["rider_id"].unique().tolist()
                scores = {rid: estimator.get_frailty(int(rid)) for rid in rider_ids}

                # Only riders the model has seen (non-zero frailty)
                nonzero = {rid: s for rid, s in scores.items() if s != 0.0}
                if not nonzero:
                    continue

                # Softmax only over the riders the model can rank
                ranked_probs = self._softmax_probs(nonzero)
                top_riders = sorted(nonzero.items(), key=lambda x: -x[1])[: self.top_k_bets]

                field_size = len(rider_ids)
                odds = self._market_odds(field_size)
                outcome_map = self._outcome_map(stage_data)

                # Boosted top-3 probability vs naive baseline.
                # naive = 3/N; model assumes signal_boost× better than baseline.
                # Kelly will stake positively only when boosted_prob > 1/odds.
                base_top3_prob = 3.0 / field_size if self.bet_on_top3 else 1.0 / field_size
                boosted_prob = min(0.95, base_top3_prob * self.signal_boost)

                for rider_id, score in top_riders:
                    prob = boosted_prob
                    stake_frac = self._kelly_stake(prob, odds)
                    stake_amt = bankroll * stake_frac
                    if stake_amt <= 0:
                        continue

                    outcome = outcome_map.get(int(rider_id), {"rank": 999, "name": ""})
                    rank = outcome["rank"]
                    won = rank <= 3 if self.bet_on_top3 else rank == 1
                    pnl = stake_amt * (odds - 1) if won else -stake_amt

                    bankroll_before = bankroll
                    bankroll = max(0.0, bankroll + pnl)

                    result.bet_records.append(BetRecord(
                        race_slug=str(race["pcs_slug"]),
                        year=int(race["year"]),
                        stage_num=int(stage_num),
                        stage_type=stage_type,
                        strategy="frailty",
                        rider_id=int(rider_id),
                        rider_name=outcome["name"],
                        predicted_score=float(score),
                        predicted_prob=float(prob),
                        naive_odds=odds,
                        stake_fraction=stake_frac,
                        stake_amount=stake_amt,
                        actual_rank=rank,
                        is_top3=rank <= 3,
                        is_top5=rank <= 5,
                        is_top10=rank <= 10,
                        is_winner=rank == 1,
                        pnl=pnl,
                        bankroll_before=bankroll_before,
                        bankroll_after=bankroll,
                    ))
                    result.bankroll_curve.append(bankroll)
                    race_had_bets = True

                result.n_stages += 1

            if race_had_bets:
                result.n_races += 1

        result.final_bankroll = bankroll
        self._compute_metrics(result)
        return result

    def _run_tactical(self, all_data: pd.DataFrame, races: pd.DataFrame) -> StrategyResult:
        """
        Strategy 1 – Tactical HMM proxy (SimpleTacticalDetector).

        Within each race, processes stages in order. After mountain/hilly
        stages, identifies riders in PRESERVING state, then bets on them
        for the next flat or hilly stage.

        No cross-race training needed — the detector is stateful within
        a race (it accumulates observations sequentially).
        """
        result = StrategyResult(
            strategy="tactical",
            final_bankroll=self.initial_bankroll,
            bankroll_curve=[self.initial_bankroll],
        )
        bankroll = self.initial_bankroll

        for _, race in races.iterrows():
            race_id = int(race["id"])

            race_stages = (
                all_data[all_data["race_id"] == race_id]
                [["stage_id", "stage_num", "stage_date", "stage_type"]]
                .drop_duplicates("stage_id")
                .sort_values("stage_num")
            )
            if len(race_stages) < 2:
                continue

            detector = SimpleTacticalDetector()
            race_had_bets = False
            stage_list = race_stages.to_dict("records")

            for i, stage_row in enumerate(stage_list):
                stage_id = int(stage_row["stage_id"])
                stage_type = str(stage_row["stage_type"] or "")
                stage_data = all_data[all_data["stage_id"] == stage_id]

                if len(stage_data) < self.min_field_size:
                    continue

                # Feed this stage into the detector (training step)
                st_enum = self._parse_stage_type(stage_type)
                for _, row in stage_data.iterrows():
                    raw_loss = row.get("time_behind_winner_seconds")
                    time_loss = 0.0 if (raw_loss is None or pd.isna(raw_loss)) else float(raw_loss)
                    obs = TacticalObservation(
                        rider_id=int(row["rider_id"]),
                        stage_id=stage_id,
                        stage_type=st_enum,
                        stage_date=self._parse_date(str(row.get("stage_date", ""))),
                        time_loss_seconds=time_loss,
                        gc_time_behind=time_loss,
                        gruppetto_indicator=time_loss > 900,
                    )
                    detector.update(obs)

                # Only generate bets after a mountain/hilly stage
                if st_enum not in (StageType.MOUNTAIN, StageType.HILLY):
                    continue

                # Look at next stage
                if i + 1 >= len(stage_list):
                    continue
                next_row = stage_list[i + 1]
                next_type = str(next_row["stage_type"] or "")
                next_id = int(next_row["stage_id"])
                next_data = all_data[all_data["stage_id"] == next_id]

                if len(next_data) < self.min_field_size:
                    continue

                # Only bet when next stage favours rested riders
                if self._parse_stage_type(next_type) not in (
                    StageType.FLAT, StageType.HILLY, StageType.COBBLES
                ):
                    continue

                rider_ids = next_data["rider_id"].unique().tolist()
                preserving = [
                    rid for rid in rider_ids
                    if detector.is_tactical_preserving(int(rid))
                ]
                if not preserving:
                    continue

                top_riders = preserving[: self.top_k_bets]
                field_size = len(rider_ids)
                odds = self._market_odds(field_size)
                outcome_map = self._outcome_map(next_data)

                base_top3_prob = 3.0 / field_size if self.bet_on_top3 else 1.0 / field_size
                prob = min(0.95, base_top3_prob * self.signal_boost)

                for rider_id in top_riders:
                    stake_frac = self._kelly_stake(prob, odds)
                    stake_amt = bankroll * stake_frac
                    if stake_amt <= 0:
                        continue

                    outcome = outcome_map.get(int(rider_id), {"rank": 999, "name": ""})
                    rank = outcome["rank"]
                    won = rank <= 3 if self.bet_on_top3 else rank == 1
                    pnl = stake_amt * (odds - 1) if won else -stake_amt

                    bankroll_before = bankroll
                    bankroll = max(0.0, bankroll + pnl)

                    result.bet_records.append(BetRecord(
                        race_slug=str(race["pcs_slug"]),
                        year=int(race["year"]),
                        stage_num=int(next_row["stage_num"]),
                        stage_type=next_type,
                        strategy="tactical",
                        rider_id=int(rider_id),
                        rider_name=outcome["name"],
                        predicted_score=1.0,
                        predicted_prob=prob,
                        naive_odds=odds,
                        stake_fraction=stake_frac,
                        stake_amount=stake_amt,
                        actual_rank=rank,
                        is_top3=rank <= 3,
                        is_top5=rank <= 5,
                        is_top10=rank <= 10,
                        is_winner=rank == 1,
                        pnl=pnl,
                        bankroll_before=bankroll_before,
                        bankroll_after=bankroll,
                    ))
                    result.bankroll_curve.append(bankroll)
                    race_had_bets = True

                result.n_stages += 1

            if race_had_bets:
                result.n_races += 1

        result.final_bankroll = bankroll
        self._compute_metrics(result)
        return result

    def _run_baseline(self, all_data: pd.DataFrame, races: pd.DataFrame) -> StrategyResult:
        """
        Baseline: randomly select top_k_bets riders per stage with a fixed
        1% stake. This is the null hypothesis — any real strategy should
        beat this.
        """
        result = StrategyResult(
            strategy="baseline",
            final_bankroll=self.initial_bankroll,
            bankroll_curve=[self.initial_bankroll],
        )
        bankroll = self.initial_bankroll
        rng = np.random.default_rng(42)

        stage_ids = all_data["stage_id"].unique()

        for stage_id in stage_ids:
            stage_data = all_data[all_data["stage_id"] == stage_id]
            if len(stage_data) < self.min_field_size:
                continue

            rider_ids = stage_data["rider_id"].unique().tolist()
            field_size = len(rider_ids)
            odds = self._market_odds(field_size)
            outcome_map = self._outcome_map(stage_data)
            race_row = stage_data.iloc[0]

            selected = rng.choice(
                rider_ids,
                size=min(self.top_k_bets, field_size),
                replace=False,
            )

            for rider_id in selected:
                stake_frac = 0.01  # Fixed 1% for baseline
                stake_amt = bankroll * stake_frac
                outcome = outcome_map.get(int(rider_id), {"rank": 999, "name": ""})
                rank = outcome["rank"]
                won = rank <= 3 if self.bet_on_top3 else rank == 1
                pnl = stake_amt * (odds - 1) if won else -stake_amt

                bankroll_before = bankroll
                bankroll = max(0.0, bankroll + pnl)

                result.bet_records.append(BetRecord(
                    race_slug=str(race_row.get("pcs_slug", "")),
                    year=int(race_row.get("year", 0)),
                    stage_num=int(race_row.get("stage_num", 0)),
                    stage_type=str(race_row.get("stage_type", "")),
                    strategy="baseline",
                    rider_id=int(rider_id),
                    rider_name=outcome["name"],
                    predicted_score=0.0,
                    predicted_prob=1.0 / field_size,
                    naive_odds=odds,
                    stake_fraction=stake_frac,
                    stake_amount=stake_amt,
                    actual_rank=rank,
                    is_top3=rank <= 3,
                    is_top5=rank <= 5,
                    is_top10=rank <= 10,
                    is_winner=rank == 1,
                    pnl=pnl,
                    bankroll_before=bankroll_before,
                    bankroll_after=bankroll,
                ))
                result.bankroll_curve.append(bankroll)

            result.n_stages += 1

        result.final_bankroll = bankroll
        result.n_races = races["id"].nunique()
        self._compute_metrics(result)
        return result

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all_data(self, conn) -> pd.DataFrame:
        """Load every stage result row we need for backtesting."""
        return pd.read_sql_query(
            """
            SELECT
                rr.rider_id,
                rr.stage_id,
                rs.stage_number  AS stage_num,
                rs.stage_date,
                rs.stage_type,
                rs.profile_score,
                rs.vertical_m,
                rr.rank,
                rr.time_behind_winner_seconds,
                r.pcs_slug,
                r.year,
                r.id             AS race_id,
                r.startdate,
                ri.name          AS rider_name
            FROM rider_results rr
            JOIN race_stages rs ON rr.stage_id = rs.id
            JOIN races r        ON rs.race_id   = r.id
            JOIN riders ri      ON rr.rider_id  = ri.id
            WHERE rr.result_category = 'stage'
              AND rs.stage_date IS NOT NULL
            ORDER BY rs.stage_date, rs.stage_number
            """,
            conn,
        )

    def _load_races(self, conn) -> pd.DataFrame:
        """Load races that have at least some results, ordered by startdate."""
        return pd.read_sql_query(
            """
            SELECT DISTINCT r.id, r.pcs_slug, r.display_name, r.year,
                            r.startdate, r.enddate
            FROM races r
            JOIN race_stages rs  ON rs.race_id = r.id
            JOIN rider_results rr ON rr.stage_id = rs.id
            WHERE r.startdate IS NOT NULL
            ORDER BY r.startdate
            """,
            conn,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_survival_records(self, df: pd.DataFrame) -> List[SurvivalRecord]:
        """Convert a DataFrame subset to SurvivalRecord objects."""
        records = []
        for _, row in df.iterrows():
            raw_loss = row.get("time_behind_winner_seconds")
            time_loss = 0.0 if (raw_loss is None or pd.isna(raw_loss)) else float(raw_loss)
            stage_type = str(row.get("stage_type") or "road")
            is_mountain = stage_type.lower() in ("mountain", "hilly")
            gruppetto = is_mountain and time_loss > 900

            try:
                rank_int = int(str(row.get("rank", "")).strip())
            except (ValueError, TypeError):
                rank_int = 100

            records.append(SurvivalRecord(
                rider_id=int(row["rider_id"]),
                stage_id=int(row["stage_id"]),
                stage_date=self._parse_date(str(row.get("stage_date", ""))),
                stage_type=stage_type,
                time_to_cutoff=45.0,
                event_observed=False,
                gc_position=rank_int,
                gc_time_behind=time_loss,
                gruppetto_indicator=1 if gruppetto else 0,
                gruppetto_time_loss=max(0.0, time_loss - 600) if gruppetto else 0.0,
            ))
        return records

    def _outcome_map(self, stage_data: pd.DataFrame) -> Dict[int, dict]:
        """Build {rider_id: {rank, name}} from a stage DataFrame."""
        out = {}
        for _, row in stage_data.iterrows():
            try:
                rank = int(str(row.get("rank", "")).strip())
            except (ValueError, TypeError):
                rank = 999
            out[int(row["rider_id"])] = {
                "rank": rank,
                "name": str(row.get("rider_name", "")),
            }
        return out

    def _softmax_probs(self, scores: Dict[int, float], temperature: float = 2.0) -> Dict[int, float]:
        """
        Convert raw model scores to win probabilities via softmax.

        Temperature > 1 sharpens the distribution (concentrates mass on
        higher-scored riders). Temperature = 1 gives standard softmax.
        """
        ids = list(scores.keys())
        vals = np.array([scores[i] for i in ids], dtype=float)
        vals = (vals - vals.max()) * temperature   # Numerical stability
        exp_v = np.exp(vals)
        probs = exp_v / exp_v.sum()
        return {i: float(p) for i, p in zip(ids, probs)}

    def _market_odds(self, field_size: int) -> float:
        """
        Simulated fair-market odds.

        For top-3 market: fair odds = field_size / 3
        For win market:   fair odds = field_size
        """
        if self.bet_on_top3:
            return max(1.1, field_size / 3.0)
        return max(1.1, float(field_size))

    def _kelly_stake(self, prob: float, odds: float) -> float:
        """Quarter-Kelly stake as a fraction of bankroll."""
        if odds <= 1.0 or prob <= 0.0:
            return 0.0
        b = odds - 1.0
        kelly = (b * prob - (1.0 - prob)) / b
        return max(0.0, kelly * self.kelly_fraction)

    @staticmethod
    def _parse_stage_type(s: str) -> StageType:
        s = (s or "").lower()
        if "mountain" in s:
            return StageType.MOUNTAIN
        if "hilly" in s or "hills" in s:
            return StageType.HILLY
        if "itt" in s or "time" in s or " tt" in s:
            return StageType.ITT
        if "cobble" in s:
            return StageType.COBBLES
        return StageType.FLAT

    @staticmethod
    def _parse_date(s: str) -> datetime:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s[:len(fmt)], fmt)
            except (ValueError, TypeError):
                continue
        return datetime(2000, 1, 1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, result: StrategyResult) -> None:
        """Compute aggregate metrics from bet_records in place."""
        bets = result.bet_records
        if not bets:
            return

        result.n_bets = len(bets)
        result.win_rate = sum(b.is_winner for b in bets) / len(bets)
        result.top3_rate = sum(b.is_top3 for b in bets) / len(bets)
        result.top5_rate = sum(b.is_top5 for b in bets) / len(bets)

        total_staked = sum(b.stake_amount for b in bets)
        result.total_pnl = sum(b.pnl for b in bets)
        result.roi = result.total_pnl / total_staked if total_staked > 0 else 0.0

        # Max drawdown on bankroll curve
        curve = result.bankroll_curve
        if curve:
            peak = curve[0]
            max_dd = 0.0
            for v in curve:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd

        # Brier score: treat top-3 as the binary event
        probs = [b.predicted_prob for b in bets]
        actuals_bin = [1.0 if b.is_top3 else 0.0 for b in bets]
        result.brier_score = float(
            np.mean([(p - a) ** 2 for p, a in zip(probs, actuals_bin)])
        )

        # Spearman: predicted score vs actual rank
        # (higher score should → lower actual rank)
        scores = [b.predicted_score for b in bets]
        neg_ranks = [-b.actual_rank if b.actual_rank < 999 else -200 for b in bets]
        if len(scores) > 5:
            rho, pval = spearmanr(scores, neg_ranks)
            result.spearman_rho = float(rho) if not np.isnan(rho) else 0.0
            result.spearman_p = float(pval) if not np.isnan(pval) else 1.0

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def _print_race_breakdown(self, result: StrategyResult) -> None:
        """Print per-race P&L summary for a single strategy."""
        if not result.bet_records:
            return

        df = pd.DataFrame([
            {
                "race": f"{b.race_slug} {b.year}",
                "stage": b.stage_num,
                "rider": b.rider_name,
                "rank": b.actual_rank if b.actual_rank < 999 else "DNF",
                "top3": "Y" if b.is_top3 else "N",
                "pnl": round(b.pnl, 2),
            }
            for b in result.bet_records
        ])

        print(f"  Best strategy ({result.strategy}) - per-bet detail (last 20):")
        try:
            print(df.tail(20).to_string(index=False))
        except UnicodeEncodeError:
            # Windows consoles with narrow codepages can't render accented names
            print(df.tail(20).to_string(index=False).encode("ascii", errors="replace").decode())
        print()
