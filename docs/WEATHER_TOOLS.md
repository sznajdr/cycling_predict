# Weather Analysis Tools for Cycling Betting

This directory contains sophisticated weather analysis tools for Strategy 6 (ITT Weather Arbitrage) and general race condition assessment.

---

## Overview

Weather conditions—particularly wind—can create significant performance deltas in Individual Time Trials (ITTs). A rider starting into a headwind may lose 10-30 seconds compared to a rider with a tailwind on the same course. The market often prices based on average conditions, creating value opportunities when forecasts shift after markets open.

---

## Tools Included

### 1. `weather_race_analyzer.py` - Quick ITT Analysis

Fast wind analysis for ITT stages. Fetches forecasts, maps against start times, calculates time deltas.

**Usage:**

```bash
# Basic analysis (requires OPENWEATHER_API_KEY env var)
export OPENWEATHER_API_KEY="your_key_here"
python scripts/weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# With explicit API key
python scripts/weather_race_analyzer.py --race paris-nice --year 2026 --stage 1 \
    --api-key YOUR_KEY_HERE

# Specify course bearing (0=North, 90=East, 180=South, 270=West)
python scripts/weather_race_analyzer.py --race tour-de-france --year 2025 --stage 21 \
    --bearing 135
```

**Sample Output:**

```
======================================================================
WEATHER RACE ANALYZER - TIRRENO-ADRIATICO 2026 Stage 1
======================================================================
Race: Tirreno-Adriatico (2026)
Stage: 1 - 11.5km
Type: itt
Route: Lido di Camaiore → Lido di Camaiore
Location: 43.867, 10.200

Fetching weather forecast...
Retrieved 40 forecast intervals
Startlist: 176 riders
Start window: 14:00 - 16:55

----------------------------------------------------------------------
WEATHER FORECAST - TIMELINE
----------------------------------------------------------------------
Time         Wind                 Temp       Rain       Riders Starting
----------------------------------------------------------------------
Mon 14:00    4.2 m/s @ 135°     12.5°C     0.0mm      Del Toro, Roglic...
Mon 15:00    5.8 m/s @ 180°     13.2°C     0.0mm      Van Aert, Ganna...
Mon 16:00    6.5 m/s @ 225°     13.8°C     0.0mm      Van der Poel...

----------------------------------------------------------------------
WIND IMPACT ANALYSIS
----------------------------------------------------------------------
Rank   Rider                     Start    Wind               Delta      Advantage
-------------------------------------------------------------------------------------
1      VAN AERT Wout             15:30    5.2m/s @ 175°    -2.3s      72/100 🟢 BEST
2      GANNA Filippo             15:45    5.0m/s @ 170°    -1.8s      68/100 🟢 GOOD
3      ROGLIC Primoz             14:15    3.8m/s @ 140°    -0.9s      61/100
...
172    DEL TORO Isaac            14:00    4.5m/s @ 90°     +3.2s      38/100 🔴 POOR

----------------------------------------------------------------------
STRATEGY RECOMMENDATIONS
======================================================================
Expected time spread: 5.5 seconds between best/worst conditions
Wind advantage variance: 34 points

🟡 MODERATE OPPORTUNITY
   Wind conditions create 5-15s spread - watch for line movements

BACK (Wind Advantage):
   • Van Aert (15:30) - 2.3s advantage
   • Ganna (15:45) - 1.8s advantage
   • Jorgenson (15:15) - 1.5s advantage

FADE (Wind Disadvantage):
   • Del Toro (14:00) - 3.2s disadvantage
   • Christen (14:05) - 2.8s disadvantage

MARKET TIMING:
   Early starters face stronger crosswinds (4.5 vs 5.2 m/s)
   → Late starters may be undervalued
```

---

### 2. `weather_advanced.py` - Physics-Based Analysis

Sophisticated physics engine with power-velocity modeling, air density calculations, and export for integration with `rank_stage.py`.

**Features:**
- Full power-duration modeling
- Air density based on temperature/altitude
- Wind field interpolation
- Export to CSV for probability adjustment
- JSON output for programmatic use

**Usage:**

```bash
# Table output
python scripts/weather_advanced.py --race tirreno-adriatico --year 2026 --stage 1

# JSON for integration
python scripts/weather_advanced.py --race paris-nice --year 2026 --stage 1 --json

# Export for rank_stage.py integration
python scripts/weather_advanced.py --race tirreno-adriatico --year 2026 --stage 1 \
    --export weather_adjustments.csv
```

**Integration with Rank Stage:**

```python
# After running weather analysis
python scripts/weather_advanced.py --race tirreno-adriatico --year 2026 --stage 1 \
    --export weather_multipliers.csv

# Then adjust rank_stage.py to load these multipliers
# and apply to base probabilities
```

---

## Physics Models

### Aerodynamic Time Impact

The tools use a simplified power-velocity model:

```
P = F_roll × v + F_aero × v
F_roll = Crr × m × g
F_aero = 0.5 × ρ × CdA × (v + headwind)²
```

Where:
- **P**: Rider power output (W)
- **v**: Ground speed (m/s)
- **ρ**: Air density (kg/m³) - varies with temperature/altitude
- **CdA**: Drag coefficient × area (~0.24 for TT position)
- **Crr**: Rolling resistance coefficient (~0.003)

### Wind Components

Wind is decomposed into:
- **Head/tail component**: Direct impact on speed
- **Crosswind component**: Handling difficulty (secondary effect)

A tailwind of 3 m/s (~11 km/h) can improve a 10km TT by 15-25 seconds.

---

## Strategy 6: ITT Weather Arbitrage

### The Edge

1. **Market Behavior**: Books price ITTs based on average expected conditions at market open
2. **Forecast Updates**: Weather models update 6-12 hours before race start
3. **Start Time Spread**: ITTs span 2-3 hours, wind can shift 90-180° during window
4. **Result**: Riders at opposite ends of start window face materially different conditions

### Execution

```bash
# 24 hours before: Get initial forecast
python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# 6 hours before: Check for forecast updates
python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# Compare: Has wind direction shifted?
# If yes → early/late starters may be mispriced
```

### Key Metrics

| Metric | Interpretation |
|--------|----------------|
| **Time Spread** | >15s = Strong opportunity, 5-15s = Moderate, <5s = Neutral |
| **Wind Variance** | High variance = high uncertainty = size down |
| **Gust Factor** | Gusts >30% above base wind = unpredictable = avoid |

---

## API Setup

### OpenWeatherMap (Free Tier)

1. Sign up at [openweathermap.org](https://openweathermap.org/api)
2. Get free API key (5-day/3-hour forecast, 60 calls/minute)
3. Set environment variable:

```bash
# Linux/macOS
export OPENWEATHER_API_KEY="your_key_here"

# Windows PowerShell
$env:OPENWEATHER_API_KEY="your_key_here"

# Windows CMD
set OPENWEATHER_API_KEY=your_key_here
```

### Without API Key

Both tools work without API keys using mock data (clearly marked). Use for testing only—do not bet on mock forecasts.

---

## Supported Races

Built-in configurations for:

| Race | Stage 1 Config |
|------|----------------|
| Tirreno-Adriatico | Lido di Camaiore ITT (11.5km) |
| Paris-Nice | Saint-Cyr-l'École ITT (~4km) |

Add new races by editing `RACE_WAYPOINTS` in `weather_advanced.py`:

```python
RACE_WAYPOINTS = {
    "your-race-slug": {
        "stage_1": {
            "name": "Stage Name",
            "type": "itt",
            "distance_km": 15.0,
            "locations": [
                {"name": "Start", "lat": 45.0, "lon": 3.0, "km": 0},
                {"name": "Finish", "lat": 45.1, "lon": 3.1, "km": 15.0},
            ],
            "course_bearing": 45,
        }
    }
}
```

---

## Integration with Betting Workflow

### Standalone Use

```bash
# Get weather analysis
python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# Use output to manually adjust rank_stage.py probabilities
# Riders with >2s advantage → increase probability 10-20%
# Riders with >2s disadvantage → decrease probability 10-20%
```

### Automated Integration

```python
# In your betting workflow
from weather_advanced import WeatherService, WeatherRaceAnalyzer

weather = WeatherService(api_key=os.getenv("OPENWEATHER_API_KEY"))
analyzer = WeatherRaceAnalyzer(weather)

analysis = analyzer.analyze_stage("tirreno-adriatico", 2026, 1)

# Apply weather multipliers to model probabilities
for profile in analysis["rider_profiles"]:
    delta = profile["impact"]["time_delta_s"]
    # Convert to probability multiplier
    multiplier = 1 + (-delta / 60) * 0.05  # ~5% per 60s
    
    # Adjust model probability
    base_prob = get_model_probability(profile["rider"]["name"])
    adjusted_prob = base_prob * multiplier
```

---

## Output Reference

### weather_race_analyzer.py Output

| Field | Description |
|-------|-------------|
| `Wind` | Speed (m/s) and direction (degrees). 0°=North wind, 90°=East wind |
| `Delta` | Estimated time difference vs neutral conditions. Negative=faster |
| `Advantage` | Score 0-100. >60=good, <40=poor |
| `Spread` | Max time difference between any two start times |

### weather_advanced.py Output

| Field | Description |
|-------|-------------|
| `time_delta_s` | Seconds gained/lost vs calm conditions |
| `impact_pct` | Time impact as percentage of total time |
| `power_adjustment_pct` | % more/less power required to match baseline |
| `required_power_w` | Absolute power needed to maintain base time |

---

## Limitations

1. **Forecast Accuracy**: 3-hour resolution from free APIs limits precision
2. **Microclimates**: Local terrain effects not captured
3. **Rider Adaptation**: Some riders handle wind better (not modeled)
4. **Tactical Pacing**: Model assumes constant power, not optimal pacing

**Mitigation**: Use quarter-Kelly sizing when weather edge >50bps to account for uncertainty.

---

## Examples

### Example 1: Tirreno-Adriatico 2026 Stage 1

```bash
export OPENWEATHER_API_KEY="your_key"

# Analyze
python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1

# Expected finding: Coastal location, variable winds
# Recommendation: Wait for forecast update 6 hours before
```

### Example 2: Tour de France Final TT

```bash
# Longer TT = more weather impact
python scripts/weather_advanced.py --race tour-de-france --year 2025 --stage 21 \
    --export tdf_weather.csv

# High impact expected due to:
# - Longer distance (40km+)
# - Late afternoon thunderstorms common
# - GC riders start reverse order (leaders last)
```

### Example 3: Script Integration

```bash
#!/bin/bash
# daily_weather_check.sh

RACE=$1
YEAR=$2
STAGE=$3

# Get weather
python weather_race_analyzer.py \
    --race $RACE --year $YEAR --stage $STAGE \
    > /tmp/weather_${RACE}_${STAGE}.txt

# Check spread
SPREAD=$(grep "Expected time spread" /tmp/weather_${RACE}_${STAGE}.txt | grep -oP '\d+\.\d+')

if (( $(echo "$SPREAD > 10" | bc -l) )); then
    echo "ALERT: High weather variance detected (${SPREAD}s)"
    # Send notification, trigger re-analysis
fi
```

---

## Further Reading

- [docs/ENGINE.md](docs/ENGINE.md) - Strategy 6 full specification
- [ONBOARDING.md](ONBOARDING.md) - Kelly sizing under uncertainty
- [EXAMPLE_TIRRENO.md](EXAMPLE_TIRRENO.md) - Complete race workflow example

---

## Support

For issues or race configuration additions, open a GitHub issue with:
1. Race name and stage number
2. Start/finish coordinates (lat/lon)
3. Expected start times
4. Weather API response (if error)
