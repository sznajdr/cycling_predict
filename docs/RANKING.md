# Stage Ranking Model

Pre-race probability ranking that synthesises six signals — specialty, historical, frailty, tactical, GC-relevance, and cross-race recent form — into calibrated win probabilities for every rider on the startlist, layers live Betclic odds on top, and sizes stakes via half-Kelly.

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [The Six Signals](#2-the-six-signals)
3. [Stage Topology Integration](#3-stage-topology-integration)
4. [Weight Matrix](#4-weight-matrix)
5. [Softmax Temperature Calibration](#5-softmax-temperature-calibration)
6. [Odds Join and Edge Calculation](#6-odds-join-and-edge-calculation)
7. [Kelly Sizing](#7-kelly-sizing)
8. [CLI Usage](#8-cli-usage)
9. [Head-to-Head Predictions](#9-head-to-head-predictions)
10. [Fitting Models First (`--run-models`)](#10-fitting-models-first---run-models)
11. [Database Persistence (`--save`)](#11-database-persistence---save)
12. [Graceful Degradation](#12-graceful-degradation)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Why This Exists

The individual strategies (frailty, tactical HMM) each produce a signal for one dimension of rider form. What was missing was a single ranked list that:

- Combines all available signals into one probability per rider
- Calibrates those probabilities to be realistic (the favourite shouldn't be 95% on a flat stage)
- Joins live Betclic odds to every rider
- Computes edge and Kelly stake in one pass

Previously the only way to get a pre-race view was a raw SQL join of specialty scores to market odds. This model makes that workflow a one-liner:

```bash
python scripts/rank_stage.py paris-nice 2026 1
```

---

## 2. The Six Signals

Each signal is a float in [0, 1] or `None` if data is unavailable.

---

### Signal 1: Specialty

**Primary source:** Quality-weighted rank percentile computed from `rider_results` in the DB, with each result weighted by the UCI category of the race it came from (`races.uci_tour`). Falls back to static PCS specialty scores (`riders.sp_sprint` etc.) when insufficient quality data is present.

**Quality weight by UCI category:**

| UCI code | Race type | Weight |
|---------|-----------|--------|
| `2.UWT` | World Tour stage race | 1.00 |
| `1.UWT` | World Tour one-day | 0.95 |
| Grand Tours (TdF/Giro/Vuelta) | — | 1.00 × 1.3 bonus |
| `2.Pro` | ProSeries stage race | 0.50 |
| `1.Pro` | ProSeries one-day | 0.45 |
| `2.1` / `1.1` | Continental | 0.18 / 0.15 |
| `2.2` / `1.2` | Sub-continental | 0.08 / 0.06 |
| `NC` | National championship | 0.12 |

**Top-20% filter:** Only stage results where the rider finished in the **top 20% of the field** contribute to quality specialty. A rider finishing 50th in a 150-rider field (33rd percentile) contributes nothing — this ensures that GC riders merely surviving flat stages do not accrue sprint specialty. The `rank_pct` within the qualifying window is:

```
rank_pct = 1 - rank / (field_size × 0.20)
```

So 1st place → rank_pct ≈ 0.97, while the threshold position (e.g. 30th in a 150-rider field) → rank_pct ≈ 0.0.

**Minimum evidence threshold:** A rider needs `_MIN_QUALITY_WEIGHT ≥ 2.0` (sum of UCI category weights across all qualifying top-20% finishes) **and** results from `_MIN_DISTINCT_SLUGS ≥ 2` different race circuits before quality specialty is used for them. A single Grand Tour top-20% finish scores ~1.3 weight; the equivalent of two UWT-level qualifying finishes across two different events is required. Riders below the threshold fall back to static PCS specialty.

**Activation gate:** Quality specialty only activates when at least 200 Grand Tour stage results are present in the DB. Without Grand Tour data, sprint-stage results from only Paris-Nice and Tirreno degenerate into a race-specific historical signal (the same information already captured by Signal 2), making it unable to distinguish Girmay from a consistent finisher at those two races. The gate ensures the signal only contributes when it carries cross-race, quality-differentiated information.

Run `python scripts/fetch_calibration_data.py` to see how much Grand Tour data is needed, then `python -m pipeline.runner` to collect it. Once active, this signal correctly ranks Girmay above Coquard on flat stages — their Grand Tour flat-stage win rates are the primary discriminator.

**Static fallback source:** `riders.sp_sprint / sp_hills / sp_climber / sp_time_trial` (PCS specialty scores, 0–100). Used when quality specialty is inactive or a rider has insufficient data (< `_MIN_QUALITY_WEIGHT = 2.0` total UCI-weighted evidence, or results from < `_MIN_DISTINCT_SLUGS = 2` race circuits). A single Grand Tour top-20% finish scores ~1.3 weight; a rider needs at least the equivalent of two UWT-level top-20% finishes across two or more different race events before quality specialty activates for them.

**Column used per stage type:**

| Stage type | Finish | Blend |
|-----------|--------|-------|
| `flat` | Flat finish | 65% `sp_sprint` + 35% `sp_one_day_races` |
| `flat` | Uphill finish | 55% `sp_sprint` + 45% `sp_climber` |
| `hilly` | Flat finish | 100% `sp_hills` |
| `hilly` | Uphill finish | 50% `sp_hills` + 50% `sp_climber` |
| `mountain` | any | 100% `sp_climber` |
| `itt` / `ttt` | any | 100% `sp_time_trial` |

**Why blend `sp_one_day_races` on flat stages?** PCS `sp_sprint` is accumulated exclusively from bunch-sprint stage wins and top placings. Puncher/classics riders — who regularly win flat stages in reduced groups or two-up finishes — accumulate most of their points under `sp_one_day_races` instead. Using `sp_sprint` alone would rank a rider like Girmay (sp_sprint ≈ 1500, sp_one_day ≈ 3300) far below pure bunch sprinters with identical or lower real-world win rates. The 65/35 blend corrects this without removing the primacy of raw sprint power.

**Finish-type blending.** When `race_climbs` data shows the nearest climb ends ≤ 2 km from the finish (uphill finish), the specialty columns are blended as shown above — `sp_sprint` is replaced with a climber component for flat/uphill finishes.

**Stage topology detection.** The model detects uphill finishes using stage distance data and climb positions:

1. **Data sources:** `race_stages.distance_km` + `race_climbs.km_before_finish`
2. **Transformation:** PCS provides stage-relative `km_before_finish` (0 = stage finish). The pipeline transforms this to race-relative using cumulative stage distances.
3. **Detection logic:** If any climb in the stage has `km_before_finish ≤ 2.0` from the stage's finish line, the stage is flagged as "uphill finish".
4. **Validation:** If stage `distance_km` is NULL, uphill detection is disabled (returns `False`) to prevent false positives from missing data.

Example: Paris-Nice Stage 4 (Uchon) has `km_before_finish=0` for the Uchon climb → detected as "HILLY/UPHILL FINISH" → uses 50/50 hills/climber blend.

**Power-to-weight adjustment (mountains / uphill-hilly).** For mountain stages and uphill-finish hilly stages, each rider's raw specialty score is multiplied by `MEDIAN_WEIGHT_KG / rider.weight_kg` (capped at [0.80, 1.30]). A 58 kg climber with `sp_climber = 75` scores ~12% higher than a 73 kg rouleur with the same specialty score. Riders with unknown weight are unadjusted.

**Normalisation:** min-max across the startlist after blending and P/W adjustment: `(score - field_min) / (field_max - field_min)`. A rider with no specialty score gets `None`.

---

### Signal 2: Historical

**Source:** `rider_results`, restricted to this race, this stage type, years prior to the current year.

**Query logic.** For each rider in the startlist, compute their average rank percentile across all matching historical stages:

```sql
AVG(1.0 - CAST(rank AS REAL) / field_size)  AS avg_rank_pct
```

`avg_rank_pct = 1.0` means the rider won every matching stage. `avg_rank_pct = 0.0` means last every time.

**Fallback.** Riders with no history in this race get the **median** of riders who do have history, and are flagged with an asterisk (`*`) in the output and `no_history_flag = True` in the DB.

**No history available.** If no rider in the startlist has prior results for this race and stage type, all historical signals are `None` and the signal is inactive.

---

### Signal 3: Frailty

**Source:** `rider_frailty.hidden_form_prob` at the latest `computed_at`.

`hidden_form_prob` is the probability that a rider is hiding good form based on their observed gruppetto pattern relative to what their observable characteristics predict. High value = sandbagging signal. See [`docs/MODELS.md`](MODELS.md) — Strategy 2.

Returns `None` for all riders if the `rider_frailty` table is empty. Run `--run-models` to populate it.

---

### Signal 4: Tactical

**Source:** `tactical_states.preserving_prob` for the **previous** stage (stage N-1).

A rider flagged PRESERVING on a mountain stage — dropping time deliberately rather than being genuinely distanced — is expected to perform better on the following flat or hilly stage. A rider who was actively CONTESTING on a prior mountain is expected to have spent more energy.

**Direction depends on the stage type being ranked:**

| Stage type | Signal value |
|-----------|-------------|
| `flat` / `hilly` | `preserving_prob` — sandbagging on a prior mountain suggests energy reserves |
| `mountain` | `1 - preserving_prob` — riders who contested prior stages have the form |
| `itt` / `ttt` | `0.5` (neutral — tactical state on road stages doesn't predict TT performance) |

Returns `None` for all riders if `stage_number = 1` (no prior stage) or `tactical_states` is empty. Run `--run-models` to populate it.

---

### Signal 5: GC Relevance


**Source:** `rider_results.rank` with `result_category = 'gc'` for the previous stage.

Unlike the other signals, this is always set — it defaults to `0.5` (neutral) when GC data is unavailable or it's Stage 1.

**GC rank thresholds:**

| GC rank | Flat / Hilly | Mountain |
|---------|-------------|---------|
| ≤ 10 | **0.10** — GC men won't sacrifice position for a sprint | **0.90** — GC men go all-out |
| ≤ 30 | 0.60 | 0.60 |
| > 30 / no data | **0.90** — out of GC contention, free to attack flat stages | **0.20** |
| ITT / TTT | 0.50 | 0.50 |

This encodes the basic strategic logic: on flat stages, GC relevance *reduces* win probability (protected riders don't sprint); on mountains it *increases* it.

---

### Signal 6: Cross-Race Recent Form

**Source:** `rider_results` across **all races in the DB** for the last 90 days.

**Method.** For each stage result in the 90-day window, compute a rank percentile and apply exponential time-decay with a 30-day constant (half-life ≈ 20 days). Aggregate per rider:

```
form_score = Σ (percentile_i × weight_i) / Σ weight_i
where weight_i = exp(-(days_ago_i) / 30)
```

`form_score = 1.0` means the rider won every race in the window at the most recent date. `form_score = 0.0` means last in every race.

**Fallback.** Riders with no results in the last 90 days → `form_signal = None`. The weight for this signal is redistributed to the remaining active signals (same graceful-degradation mechanism as other signals).

This signal directly addresses the *first-timer problem*: a rider making their Paris-Nice debut after winning Strade Bianche and finishing 2nd at Tirreno is correctly rated high rather than assigned the field median.

---

## 3. Stage Topology Integration

The model integrates detailed stage topology data to improve specialty signal accuracy and detect uphill finishes.

### Data Sources

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| `race_stages` | `stage_type`, `distance_km`, `vertical_m`, `profile_score` | Stage classification and cumulative distance calculation |
| `race_climbs` | `climb_name`, `length_km`, `steepness_pct`, `km_before_finish` | Climb characteristics and position relative to finish |

### km_before_finish Transformation

**The Problem:** PCS provides `km_before_finish` as **stage-relative** (0 = stage finish), but the model needs **race-relative** positions to correctly map climbs to stages.

**The Solution:** Transform using cumulative stage distances:

```
race_kbf = total_race_distance - cum_distance_at_stage_end + stage_kbf
```

Example: Paris-Nice 2026
- Stage 4 ends at km 576.4 (cumulative)
- Total race distance: 1229.9 km
- Uchon climb: stage_kbf = 0 km
- Transformed: 1229.9 - 576.4 + 0 = **653.5 km** (from race finish)

### Uphill Finish Detection

A stage is flagged as "uphill finish" when:

1. Stage has valid `distance_km` in `race_stages`
2. Any climb in `race_climbs` has `km_before_finish ≤ 2.0` km from stage finish
3. The climb belongs to that stage (verified via race-relative transformation)

**Detection Examples:**

| Stage | Race | Climb | kbf (stage) | Detected | Output Label |
|-------|------|-------|-------------|----------|--------------|
| 4 | Paris-Nice 2026 | Uchon | 0 km | ✅ Yes | HILLY/UPHILL FINISH |
| 7 | Paris-Nice 2026 | Auron | 0 km | ✅ Yes | MOUNTAIN/UPHILL FINISH |
| 2 | Paris-Nice 2026 | None | - | ❌ No | FLAT |

### Specialty Blending by Topology

| Stage Type | Flat Finish | Uphill Finish |
|-----------|-------------|---------------|
| `flat` | 65% `sp_sprint` + 35% `sp_one_day_races` | 55% `sp_sprint` + 45% `sp_climber` |
| `hilly` | 100% `sp_hills` | 50% `sp_hills` + 50% `sp_climber` |
| `mountain` | 100% `sp_climber` | 100% `sp_climber` (no change) |

### Power-to-Weight Adjustment

For mountain stages and uphill-finish hilly stages:

```
adjusted_score = raw_score × (MEDIAN_WEIGHT_KG / rider.weight_kg)
```

Capped at [0.80, 1.30]. A 58 kg climber gets ~12% boost vs a 73 kg rider with identical `sp_climber`.

### Data Quality Handling

| Scenario | Behavior |
|----------|----------|
| Missing `distance_km` | Uphill detection disabled (returns `False`) |
| Missing `race_climbs` | No uphill finish detection, standard specialty weights |
| Missing rider `weight_kg` | No P/W adjustment (multiplier = 1.0) |

---

## 3b. Field Reduction

On flat and hilly stages, the softmax runs over the full startlist (150+ riders). Without any filtering, climbers, pure GC domestiques, and time-trial specialists dilute probability mass away from sprint and puncher contenders — artificially compressing the favorites' model probabilities.

**Field reduction** floors non-contenders' raw scores so they retain only a negligible share of probability mass. This lets the temperature calibration concentrate realistic probability on the riders who actually contest the finish.

### Contention pools

| Stage type | Contenders (by specialty rank) | Non-contenders |
|-----------|-------------------------------|----------------|
| `flat` | Top 35 by blended `sp_sprint`/`sp_one_day_races` | floored to 0 raw score |
| `hilly` | Top 30 by `sp_hills` | floored to 0 raw score |
| `mountain` | All riders | — |
| `itt` / `ttt` | All riders | — |

Non-contenders are **not completely zeroed out** — they receive a small probability share via the softmax denominator. A surprise breakaway winner from outside the contention pool is possible; it is simply priced as a rare event rather than an equally-likely outcome.

### Effect on probability range

After field reduction and Platt calibration (T*≈5 for flat stages), the top flat-stage favourite typically reaches **5–8%** — matching the historical win rate for the strongest sprinter. Second and third choices sit at **3–6%**, reflecting that sprint finishes are genuinely unpredictable. Pre-calibration the top probability was an over-concentrated 27–30%.

### Contention flag in DB

When saved with `--save`, each rider's `latent_states_json` includes `"is_contender": true/false`, enabling post-hoc analysis of which riders were in the contention pool.

---

## 4. Weight Matrix

Each stage type has a fixed set of base weights that sum to 1.0:

| Stage type | Specialty | Historical | Frailty | Tactical | GC Relevance | Form |
|-----------|-----------|-----------|--------|---------|-------------|------|
| `flat` | 0.30 | 0.25 | 0.15 | 0.10 | 0.05 | 0.15 |
| `hilly` | 0.25 | 0.25 | 0.15 | 0.10 | 0.10 | 0.15 |
| `mountain` | 0.20 | 0.20 | 0.15 | 0.10 | 0.20 | 0.15 |
| `itt` | 0.40 | 0.25 | 0.15 | 0.00 | 0.05 | 0.15 |
| `ttt` | 0.30 | 0.25 | 0.15 | 0.00 | 0.10 | 0.20 |

Tactical weight is zero for ITT/TTT — tactical preserving on road stages is not predictive of TT performance. GC relevance is highest on mountains, where it carries real signal about who will race aggressively. Form gets a higher weight in TTT (0.20) because recent team coherence matters more than specialty scores.

When signals are missing (see [Section 12](#12-graceful-degradation)), weights are renormalised across available signals before scoring.

---

## 4. Softmax Temperature Calibration + Platt Scaling

Raw scores are converted to win probabilities via softmax:

```
prob_i = exp(T · score_i) / Σ_j exp(T · score_j)
```

`T` (temperature) controls how spread out the distribution is:
- `T` close to 0 → nearly uniform distribution
- `T` large → near-deterministic (winner takes almost all probability mass)

### Data-driven temperature (Platt calibration)

`T` is fitted via **maximum-likelihood estimation** on historical stage outcomes rather than a heuristic target range. The calibration script (`scripts/calibrate_stage_model.py`) runs the full model on all historical stages with startlist data, then finds the T* that maximises:

```
T* = argmax_T  Σ_stages  log( softmax(T · raw_scores_stage)[winner] )
```

Fitted values from 343 historical stages (2022–2025):

| Stage type | T* (MLE) | log-LL / stage | vs random | Notes |
|-----------|----------|---------------|-----------|-------|
| `flat`     | **4.90** | −4.05 | +23% | 153 stages used |
| `hilly`    | **3.53** | −4.49 | +11% | 63 stages |
| `mountain` | **9.30** | −3.53 | +42% | GC relevance very predictive |
| `itt`      | **12.74**| −2.71 | +83% | Specialty near-deterministic |

The pre-calibration heuristic used T≈20 for flat stages (binary search to hit 20–35% top probability). MLE calibration finds T*≈5, giving a much flatter, more realistic distribution — the favourite on a flat stage should win roughly 5–7% of the time, not 25–30%.

### Platt sigmoid (absolute probability correction)

After applying softmax(T* × scores), a **Platt sigmoid** adjusts absolute probabilities:

```
raw_calib_i = sigmoid(a · logit(prob_i) + b)
model_prob_i = raw_calib_i / Σ_j raw_calib_j     (renormalise)
```

Fitted values confirm the model's relative ordering is already well-calibrated (a ≈ 0.96, very close to 1), with a small negative intercept (b ≈ −0.15) correcting slight base-rate overconfidence. The sigmoid transform is nearly identity for most probability values.

### Fallback

If no calibration exists in the `platt_calibration` table (e.g. fresh DB), the model falls back to the legacy binary search targeting:

| Stage type | Target range | Notes |
|-----------|-------------|-------|
| `flat` | 20% – 35% | field reduction lowers effective pool |
| `hilly` | 15% – 28% | — |
| `mountain` | 10% – 18% | — |
| `itt` | 25% – 45% | ITT favorites clearly separated |

Run `python scripts/calibrate_stage_model.py` after adding new historical data to refresh the calibration.

---

## 5. Odds Join and Edge Calculation

After probabilities are assigned, the model queries `bookmaker_odds_latest` filtered to `market_type = 'winner'` and matches riders by name using two passes:

1. **Exact Unicode match** — `lower(participant_name) = lower(rider_name)`
2. **Accent-stripped fallback** — strips diacritics from both sides: `Pogačar` → `pogacar`

For matched riders:

```
implied_prob = 1 / back_odds
edge_bps     = (model_prob - implied_prob) × 10,000
```

Edge in basis points: 100 bps = 1 percentage point of probability above market. Positive edge = value bet; negative = the market has you priced tighter than your model.

---

## 6. Kelly Sizing

Half-Kelly is shown for any rider with `edge_bps > 50`:

```
b            = back_odds - 1
full_kelly   = max(0, (b × model_prob - (1 - model_prob)) / b)
half_kelly % = full_kelly × 0.5 × 100
```

Half-Kelly (rather than full) is the default because model miscalibration is almost certain at this stage of development. Halving the Kelly fraction cuts expected growth by ~15% while roughly halving variance. See [`docs/MODELS.md`](MODELS.md) — Portfolio: Robust Kelly for the full framework.

---

## 7. CLI Usage

```bash
python scripts/rank_stage.py <race-slug> <year> <stage-number> [options]
```

### Examples

```bash
# Rank all riders for Paris-Nice 2026 Stage 1
python scripts/rank_stage.py paris-nice 2026 1

# Top 20 only
python scripts/rank_stage.py paris-nice 2026 3 --top 20

# Fit frailty + tactical models from historical data, then rank
python scripts/rank_stage.py paris-nice 2026 1 --run-models

# Persist ranking to strategy_outputs table
python scripts/rank_stage.py paris-nice 2026 1 --save

# Combine all flags
python scripts/rank_stage.py paris-nice 2026 1 --run-models --top 30 --save

# Use a different DB
python scripts/rank_stage.py paris-nice 2026 1 --db /path/to/custom.db
```

### Sample output

```
Paris-Nice 2026 Stage 1 — FLAT (170.9km)
Signals: specialty(0.35) historical(0.29) form(0.18) [frailty: no data] [tactical: no data] gc_relevance(0.06)
Field: 154 riders | Temperature: 8.24 | Edge threshold: 50bps

 Rank  Rider                    Spec    Hist    Form  ModelProb  BkOdds  Edge(bps)  Kelly%
    1  Biniam Girmay            0.82    0.91    0.88     18.4%     5.0      +1383     4.2%
    2  Phil Bauhaus             0.98    0.00*  None*      8.7%    50.0       +669     2.1%
    3  Axel Zingle              0.74    0.78*   0.54     11.2%    10.0       +120     0.3%
   ...

* no race history (median used)
* no results in last 90 days
```

For a mountain stage with an uphill finish:

```
Paris-Nice 2026 Stage 5 — MOUNTAIN/UPHILL FINISH (142.0km)
```

The **Signals** line shows effective weights after renormalisation for the active signals only (frailty and tactical are marked `no data` when their tables are empty). The weights shown always sum to 1.0.

### Arguments

| Argument | Default | Description |
|---------|--------|-------------|
| `race_slug` | required | PCS slug, e.g. `paris-nice`, `tour-de-france` |
| `year` | required | Four-digit year |
| `stage` | required | Stage number (integer) |
| `--top N` | 0 (all) | Print only the top N riders |
| `--run-models` | off | Fit frailty + tactical models before ranking |
| `--save` | off | Persist to `strategy_outputs` |
| `--db PATH` | `data/cycling.db` | Override database path |

---

## 9. Head-to-Head Predictions

The `h2h.py` script computes head-to-head matchup probabilities from the stage ranking model. Useful for duel markets or comparing specific riders.

### How It Works

The script loads **saved** probabilities from `strategy_outputs` table and computes conditional probabilities:

```
P(A beats B) = P(A wins) / (P(A wins) + P(B wins))
P(A beats Field) = P(A wins) / (1 - P(A wins))
```

**⚠️ Important:** H2H does NOT compute fresh probabilities. It reads from the database. If you see:
```
WARNING: Data is 7.7 hours old. Run with --save to refresh:
  python scripts/rank_stage.py paris-nice 2026 2 --save
```

Run the rank command with `--save` to update the database, then re-run H2H.

### Usage

```bash
# Interactive mode
python scripts/h2h.py <race-slug> <year> <stage>

# File mode (batch matchups)
python scripts/h2h.py <race-slug> <year> <stage> -f matchups.txt
```

### Interactive Mode

```bash
python scripts/h2h.py paris-nice 2026 2

> Pogacar vs Vingegaard
Pogacar vs Vingegaard: 62.3% / 37.7% | Fair odds: @1.60 / @2.65

> Coquard vs The Field
Coquard vs The Field: 17.0% / 83.0% | Fair odds: @5.88 / @1.20

> quit
```

### File Mode

Create `matchups.txt`:

```
# Comments start with #
Bryan Coquard vs Pascal Ackermann
Jonas Vingegaard vs The Field
Wilco Kelderman vs Aleksandr Vlasov
```

Run:
```bash
python scripts/h2h.py paris-nice 2026 2 -f matchups.txt
```

Output:
```
======================================================================
Paris Nice 2026 Stage 2 - H2H Predictions
======================================================================
Rider A                   vs Rider B                   | Prob A | Fair Odds
----------------------------------------------------------------------
Bryan Coquard             vs Pascal Ackermann          |  73.9% | @1.35 / @3.82
Jonas Vingegaard          vs The Field                 |   1.8% | @56.52 / @1.02
Wilco Kelderman           vs Aleksandr Vlasov          |  66.2% | @1.51 / @2.96
======================================================================
```

### "The Field" / "Das Feld"

Special opponent representing all other riders combined:

```
P(A beats Field) = P(A wins) / (P(A wins) + P(all others win))
                 = P(A wins) / (1 - P(A wins))
```

Useful when a rider is priced against the entire peloton (common in breakaway vs peloton scenarios).

### Name Matching

Uses the same two-pass matching as the main ranking:
1. Exact Unicode match
2. NFKD normalization (strips accents: `č` → `c`, `ø` → `o`)

---

## 10. Fitting Models First (`--run-models`)

Without model output in `rider_frailty` and `tactical_states`, the frailty and tactical signals are inactive and the ranking runs on specialty + historical + GC relevance only. `--run-models` runs the fitting pipeline automatically before ranking.

**What it does:**

1. Queries the DB for all years with data for this race that are earlier than the target year
2. Loads survival records from `rider_results` for all those years (via `load_survival_data_from_db`)
3. Fits `FastFrailtyEstimator` on the combined records and inserts results to `rider_frailty`
4. Loads tactical observations for all those years (via `load_tactical_data_from_db`)
5. Runs `SimpleTacticalDetector` on each observation and inserts results to `tactical_states`

**Frailty persistence.** For each rider with a frailty estimate, `hidden_form_prob` is computed from the z-score:

```
z = (frailty - field_mean) / field_std
hidden_form_prob = logistic(z)  if z > 1.5 else 0.0
```

Riders more than 1.5 standard deviations above the mean frailty get a non-zero hidden form probability.

**Tactical persistence.** The `SimpleTacticalDetector` assigns a TacticalState to each rider-stage observation using heuristic rules (time loss threshold, GC gap). Each decoded state maps to probabilities:

| State | contesting_prob | preserving_prob |
|-------|----------------|----------------|
| CONTESTING | 0.85 | 0.15 |
| PRESERVING | 0.15 | 0.85 |
| RECOVERING | 0.50 | 0.25 |
| GRUPPETTO | 0.10 | 0.80 |

These are inserted to `tactical_states` with `INSERT OR REPLACE`, so re-running `--run-models` updates stale values.

**Data requirement.** At least 10 records are required in each case. If there are fewer, the step is skipped with a warning printed to stdout.

---

## 11. Database Persistence (`--save`)

`--save` writes one row per rider to `strategy_outputs`:

| Column | Value |
|--------|-------|
| `strategy_name` | `'stage_ranking'` |
| `rider_id` | rider primary key |
| `stage_id` | stage primary key |
| `win_prob` | `model_prob` |
| `edge_bps` | computed edge (0 if no odds) |
| `latent_states_json` | all five signal values, raw_score, temperature, stage_type |

The `latent_states_json` field stores the full signal breakdown for every rider, enabling later analysis of which signals drove which rankings.

To query rankings after saving:

```sql
SELECT r.name, so.win_prob, so.edge_bps,
       json_extract(so.latent_states_json, '$.specialty') AS specialty,
       json_extract(so.latent_states_json, '$.historical') AS historical
FROM strategy_outputs so
JOIN riders r ON so.rider_id = r.id
WHERE so.strategy_name = 'stage_ranking'
  AND so.stage_id = (
      SELECT rs.id FROM race_stages rs
      JOIN races ra ON rs.race_id = ra.id
      WHERE ra.pcs_slug = 'paris-nice' AND ra.year = 2026 AND rs.stage_number = 1
  )
ORDER BY so.win_prob DESC;
```

---

## 12. Graceful Degradation

The model never fails on missing data — it degrades gracefully:

| Condition | Effect |
|-----------|--------|
| Rider has no specialty score for this stage type | specialty signal = `None`; its weight redistributed to remaining signals |
| Race has no historical results in this stage type | all historical signals = `None`; signal inactive for all riders |
| `rider_frailty` table is empty | all frailty signals = `None`; signal inactive |
| Stage 1 or `tactical_states` empty | all tactical signals = `None`; signal inactive |
| Rider has no results in last 90 days | form signal = `None`; shown as `None*` in table |
| No `race_climbs` data for stage | `is_uphill_finish = False`; unblended specialty used |
| Rider `weight_kg` unknown | power-to-weight factor = 1.0 (no adjustment) |
| Rider not matched in `bookmaker_odds_latest` | `back_odds = None`; edge and Kelly not shown |
| All signals `None` for a rider | `raw_score = 0.0`; rider contributes uniformly to softmax |
| Field ≤ CONTENTION_TOP_N for this stage type | Field reduction skipped; all riders compete equally |
| Rider outside contention pool (flat/hilly) | `raw_score` floored; `is_contender = False` in DB JSON |

When signals are inactive, the Signals line in the output marks them explicitly:

```
Signals: specialty(0.54) historical(0.40) [frailty: no data] [tactical: no data] gc_relevance(0.06)
```

The displayed weights are the base weights renormalised over active signals only. They always sum to 1.0.

---

## 13. Troubleshooting

**`Error: Stage not found`**

The stage doesn't exist in `race_stages` for this race and year. Either the race hasn't been scraped or the stage number is wrong.

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
rows = conn.execute('''
    SELECT rs.stage_number, rs.stage_type, rs.stage_date, rs.distance_km
    FROM race_stages rs JOIN races r ON rs.race_id = r.id
    WHERE r.pcs_slug = 'paris-nice' AND r.year = 2026
    ORDER BY rs.stage_number
''').fetchall()
for r in rows: print(r)
"
```

---

**`Error: No startlist found`**

The startlist hasn't been scraped for this year. Run:

```bash
python -m pipeline.runner
```

Or manually seed the queue for 2026 if the config wasn't including the prediction year yet:

```bash
python -c "
import sqlite3, yaml
from pipeline.queue import init_queue, seed_queue
conn = sqlite3.connect('data/cycling.db')
conn.row_factory = sqlite3.Row
init_queue(conn)
with open('config/races.yaml') as f:
    config = yaml.safe_load(f)
seed_queue(conn, config)
print('Seeded 2026 jobs')
"
python -m pipeline.runner
```

Note: as of the current build, `seed_queue()` automatically seeds `config['year']` for every race — re-running the pipeline should pick up 2026 without manual intervention.

---

**`[frailty: no data]` / `[tactical: no data]` in signals line**

The `rider_frailty` or `tactical_states` tables are empty. Run with `--run-models` to populate them:

```bash
python scripts/rank_stage.py paris-nice 2026 1 --run-models
```

If `--run-models` prints "Insufficient records — skipping", there isn't enough historical data in the DB to fit the models. Scrape more years first.

---

**Riders not matching odds**

The name in `bookmaker_odds` doesn't match the name in `riders` exactly or after accent-stripping. Check both sides:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print('Betclic:', conn.execute(\"SELECT DISTINCT participant_name FROM bookmaker_odds WHERE participant_name LIKE '%Girmay%'\").fetchall())
print('DB:',     conn.execute(\"SELECT name FROM riders WHERE name LIKE '%Girmay%'\").fetchall())
"
```

---

**`market_type = 'unknown'` rows blocking odds join**

If Betclic's page was JS-rendered, the HTML label regex fails and falls back to `label = 'unknown'`. The URL-based classifier then kicks in: event URLs containing `/etape-N-` are classified as `winner`, URLs matching `/<race>-20YY-mXXX` as `gc_position`. If both fail, the row stays `unknown` and won't join to the winner market.

Clean stale unknown rows and re-scrape:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
conn.execute(\"DELETE FROM bookmaker_odds WHERE market_type='unknown'\")
conn.commit()
print('Deleted unknown rows')
"
python fetch_odds.py
```

Then verify:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/cycling.db')
print(conn.execute('SELECT market_type, COUNT(*) FROM bookmaker_odds GROUP BY market_type').fetchall())
"
```

---

**Softmax returns uniform distribution**

If all raw scores are identical (scores.std < 1e-8), temperature calibration returns T = 1.0 and all riders receive equal probability. This happens when:
- All active signals return the same value for every rider (e.g. all riders have null history and the median fallback collapses to one value)
- Only gc_relevance is active and all riders have no GC standing (all get 0.9 on flat — identical)

Add more historical data or run `--run-models` to activate additional signals.
