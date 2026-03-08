# Stage Topology Integration Plan

## Current State

### Data Available
| Source | Table | Fields | Status |
|--------|-------|--------|--------|
| PCS Stage | `race_stages` | `stage_type`, `distance_km`, `vertical_m`, `profile_score` | ✅ Complete |
| PCS Race | `race_climbs` | `climb_name`, `length_km`, `steepness_pct`, `km_before_finish` | ⚠️ Buggy |

### Current Issues
1. **Climb Mapping Bug**: `km_before_finish` is stage-relative (PCS), but code expects race-relative
2. **No Stage-Climb Link**: `race_climbs` has no `stage_id` foreign key
3. **Limited Topology Use**: Only "uphill finish" detection uses climbs; profile score/vertical meters ignored

### Current Specialty Weight Logic
```python
# Static weights by stage_type only
WEIGHTS = {
    'flat':     {'specialty': 0.30, 'historical': 0.25, ...},
    'hilly':    {'specialty': 0.25, 'historical': 0.25, ...},
    'mountain': {'specialty': 0.20, 'historical': 0.20, ...},
}
```

---

## Proposed Enhancements

### 1. Fix Climb Data Model (Priority: HIGH)

**Problem**: PCS provides `km_before_finish` relative to stage finish, but we store it as-is.

**Solution Options**:

#### Option A: Transform to Race-Relative (Recommended)
During scraping, convert stage-relative to race-relative:
```python
# In pipeline/fetcher.py
stage_relative_kbf = climb['km_before_finnish']
cumulative_to_stage_end = sum(prev_stage_distances) + current_stage_distance
race_relative_kbf = total_race_distance - cumulative_to_stage_end + stage_relative_kbf
```

#### Option B: Add stage_id to race_climbs
Store which stage each climb belongs to:
```sql
ALTER TABLE race_climbs ADD COLUMN stage_id INTEGER REFERENCES race_stages(id);
```
Then detect uphill finish per-stage without distance math.

### 2. Dynamic Specialty Weights by Profile Score (Priority: HIGH)

Replace static stage_type weights with continuous adjustment:

```python
def compute_dynamic_weights(profile_score: int, stage_type: str) -> dict:
    """
    Profile Score interpretation (PCS):
    - 0-50: Flat/sprinter
    - 50-100: Hilly/puncheur  
    - 100-150: Mountainous/classics
    - 150+: High mountain/GC
    """
    base = WEIGHTS[stage_type].copy()
    
    # Adjust specialty weight based on profile score
    if profile_score < 50:
        # Pure sprint - specialty matters less, form/tactical more
        base['specialty'] = 0.25
        base['form'] = 0.20
        base['tactical'] = 0.15
    elif profile_score > 150:
        # Mountain - specialty (climbing) matters more
        base['specialty'] = 0.30
        base['historical'] = 0.15
        base['gc_relevance'] = 0.25
    
    return base
```

### 3. Climb-Position Tactical Signals (Priority: MEDIUM)

Analyze where climbs occur within stage:

```python
@dataclass
class ClimbProfile:
    total_elevation: int      # vertical_m
    profile_score: int        # PCS difficulty
    n_climbs: int
    hardest_climb_position: float  # 0.0=start, 1.0=finish
    climbing_in_last_20km: int     # meters in finale
    
def compute_climb_profile(stage_id: int) -> ClimbProfile:
    """Analyze climb distribution for tactical insights."""
    ...
```

**Tactical Implications**:
| Profile | Prediction Impact |
|---------|------------------|
| Hard climb at 50% + flat finish | Breakaway sticks, reduce sprint weights |
| Multiple climbs in last 20km | Favor puncheurs over pure sprinters |
| Long flat then steep finish | Pure sprint, favor explosive riders |
| HC climb at start, downhill finish | Sprinter can rejoin, favor fast men |

### 4. Fatigue/Accumulation Model (Priority: MEDIUM)

Use cumulative stage data for multi-day races:

```python
def compute_fatigue_factor(rider_id: int, stage_number: int) -> float:
    """
    Reduce predicted performance based on:
    - Cumulative elevation previous 2 days
    - Whether rider is sprint train domestique
    - Days since last rest day
    """
    recent_elevation = get_elevation_last_n_days(rider_id, stage_number, n=2)
    if recent_elevation > 4000:  # Hard mountain stages
        return 0.85  # 15% reduction
    return 1.0
```

### 5. Specialty Column Selection Enhancement (Priority: HIGH)

Current:
```python
_SPECIALTY_COL = {
    'flat': 'sp_sprint',
    'hilly': 'sp_hills', 
    'mountain': 'sp_climber',
}
```

Proposed - Blend based on profile score:
```python
def get_specialty_blend(profile_score: int) -> List[Tuple[str, float]]:
    """Return (column, weight) tuples for specialty scoring."""
    if profile_score < 30:
        return [('sp_sprint', 1.0)]
    elif profile_score < 60:
        return [('sp_sprint', 0.7), ('sp_hills', 0.3)]
    elif profile_score < 100:
        return [('sp_hills', 0.6), ('sp_sprint', 0.2), ('sp_climber', 0.2)]
    elif profile_score < 150:
        return [('sp_hills', 0.4), ('sp_climber', 0.4), ('sp_sprint', 0.2)]
    else:
        return [('sp_climber', 0.7), ('sp_hills', 0.3)]
```

---

## Implementation Roadmap

### Phase 1: Data Fixes (Week 1)
- [ ] Fix `km_before_finish` transformation in pipeline
- [ ] Backfill Paris-Nice 2026 climb data with correct race-relative positions
- [ ] Add validation to ensure `distance_km` exists before uphill detection

### Phase 2: Profile Score Integration (Week 2)
- [ ] Add `profile_score` to StageContext
- [ ] Implement dynamic specialty weight adjustment
- [ ] A/B test vs static weights on historical races

### Phase 3: Climb Analysis (Week 3)
- [ ] Create ClimbProfile dataclass
- [ ] Add tactical signal for "breakaway-friendly" stages
- [ ] Implement position-weighted specialty blending

### Phase 4: Advanced Features (Week 4+)
- [ ] Fatigue accumulation model
- [ ] Team capacity analysis (sprint train strength)
- [ ] Weather + topology interaction (rain + climbs)

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Log-loss on flat stages | ~0.45 | <0.40 |
| Log-loss on mountain stages | ~0.38 | <0.35 |
| Top-3 accuracy | 52% | 60% |
| Value bet hit rate | 28% | 35% |

---

## Example: Paris-Nice 2026 Stage 4 (Uchon)

**Current Behavior**:
- Stage type: `hilly` → uses `sp_hills` at 25% weight
- No uphill finish detected (bug)
- Profile score 175 ignored

**Expected with Fix**:
- Profile score 175 → high mountain classification
- Blend: 60% climber / 30% hills / 10% sprint
- Specialty weight: 30% (increased from 25%)
- GC relevance: 25% (breakaway stage)
- Predicted: Vingegaard / Pogacar type riders

---

## Files to Modify

1. `pipeline/fetcher.py` - Transform km_before_finish
2. `pipeline/db.py` - Add stage_id to race_climbs (optional)
3. `genqirue/models/stage_ranker.py`:
   - Dynamic weight calculation
   - Profile score integration
   - Enhanced specialty blending
4. `docs/RANKING.md` - Update documentation
