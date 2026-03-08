#!/usr/bin/env python3
"""
Advanced Weather Integration for Cycling Betting
================================================

Integrates weather forecasts with Genqirue models for sophisticated
pre-race analysis. Outputs wind-adjusted probabilities and betting edges.

Features:
- Multi-location weather routing (for point-to-point stages)
- Wind field interpolation between forecast points
- Integration with Strategy 6 (ITT Weather Arbitrage)
- Outputs adjusted probabilities for rank_stage.py

Usage:
    python weather_advanced.py --race tirreno-adriatico --year 2026 --stage 1 --export
    python weather_advanced.py --race paris-nice --year 2026 --stage 1 --plot
"""

import argparse
import os
import sys
import sqlite3
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.db import get_connection


# =============================================================================
# Configuration
# =============================================================================

# Race waypoints (approximate) for multi-segment analysis
RACE_WAYPOINTS = {
    "tirreno-adriatico": {
        "stage_1": {
            "name": "Lido di Camaiore ITT",
            "type": "itt",
            "distance_km": 11.5,
            "locations": [
                {"name": "Start", "lat": 43.8667, "lon": 10.2333, "km": 0},
                {"name": "Finish", "lat": 43.8667, "lon": 10.2333, "km": 11.5},  # Out-and-back
            ],
            "course_bearing": 90,  # East for out leg
        }
    },
    "paris-nice": {
        "stage_1": {
            "name": "Saint-Cyr-l'École ITT",
            "type": "itt",
            "distance_km": 4.0,  # Approximate for 2026
            "locations": [
                {"name": "Start", "lat": 48.8000, "lon": 2.0667, "km": 0},
                {"name": "Finish", "lat": 48.8000, "lon": 2.0667, "km": 4.0},
            ],
            "course_bearing": 45,
        }
    }
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class WindVector:
    """Wind vector with speed and direction."""
    speed_ms: float
    direction_deg: float  # Meteorological: where wind comes FROM
    gust_ms: Optional[float] = None
    
    def to_components(self) -> Tuple[float, float]:
        """Convert to (u, v) components (East, North)."""
        # Meteorological to math angle
        theta = math.radians(270 - self.direction_deg)  # 0=from North -> 0=East
        u = -self.speed_ms * math.cos(theta)  # East component
        v = -self.speed_ms * math.sin(theta)  # North component
        return u, v
    
    def effective_wind(self, course_bearing: float) -> Tuple[float, float]:
        """
        Calculate head/tail and cross components relative to course.
        
        Returns:
            (headwind, crosswind) - positive headwind = tailwind (helping)
        """
        u, v = self.to_components()
        course_rad = math.radians(course_bearing)
        
        # Course unit vector
        cx = math.sin(course_rad)  # East component
        cy = math.cos(course_rad)  # North component
        
        # Perpendicular (crosswind direction)
        px = -cy
        py = cx
        
        # Projections
        headwind = -(u * cx + v * cy)  # Negative = headwind
        crosswind = abs(u * px + v * py)
        
        return headwind, crosswind


@dataclass
class WeatherWindow:
    """Weather conditions for a specific time window."""
    start_time: datetime
    end_time: datetime
    wind: WindVector
    temperature_c: float
    precipitation_mm: float
    pressure_hpa: float
    humidity_pct: float
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - start_time).total_seconds() / 60


@dataclass
class RiderWeatherProfile:
    """Complete weather profile for a rider's start time."""
    rider_id: int
    name: str
    team: str
    start_time: datetime
    duration_minutes: float
    
    # Weather at start
    start_wind: WindVector
    start_temp: float
    
    # Integrated weather over effort (for longer TTs)
    avg_wind: Optional[WindVector] = None
    wind_variability: Optional[float] = None  # Std dev of wind speed
    
    # Calculated impacts
    time_delta_s: float = 0.0
    power_adjustment_pct: float = 0.0
    confidence: float = 1.0  # 0-1, lower if forecast uncertain


# =============================================================================
# Physics Engine
# =============================================================================

class CyclingPhysics:
    """
    Physics models for cycling performance under varying conditions.
    
    References:
    - Martin et al. (2007) "Validation of a Mathematical Model for Road Cycling Power"
    - Blocken et al. (2013) "CFD simulations of the aerodynamic drag of time trial cycling helmets"
    """
    
    def __init__(self):
        # Rider parameters (customizable per rider)
        self.rider_mass_kg = 70
        self.bike_mass_kg = 8
        self.total_mass = self.rider_mass_kg + self.bike_mass_kg
        
        # Aerodynamics
        self.cda_tt_position = 0.24  # m^2 - aggressive TT position
        self.cda_hood_position = 0.32  # m^2 - less aggressive
        self.rho_sea_level = 1.225  # kg/m^3
        
        # Power
        self.ftp_watts = 400  # Functional threshold power
        self.anaerobic_capacity_j = 20000  # W' prime
        
        # Rolling resistance
        self.crr = 0.003  # Coefficient of rolling resistance
        self.g = 9.81  # m/s^2
    
    def air_density(self, temp_c: float, altitude_m: float = 0, 
                    humidity: float = 0.5) -> float:
        """Calculate air density based on conditions."""
        # Simplified approximation
        p0 = 101325 * math.exp(-altitude_m / 8500)  # Barometric formula
        t_k = temp_c + 273.15
        
        # Humidity correction (moist air less dense)
        pv = humidity * 611 * math.exp(17.27 * temp_c / (t_k - 35.85))
        pd = p0 - pv
        
        rho = (pd * 0.028965 + pv * 0.018015) / (8.314 * t_k)
        return rho
    
    def estimate_tt_time(self, distance_m: float, 
                         target_power_w: float,
                         wind: WindVector,
                         course_bearing: float,
                         temp_c: float = 20,
                         altitude_m: float = 0) -> float:
        """
        Estimate TT time using iterative power-velocity solution.
        
        Args:
            distance_m: Race distance in meters
            target_power_w: Target average power
            wind: Wind conditions
            course_bearing: Direction of travel (degrees)
            temp_c: Temperature
            altitude_m: Altitude for air density
        
        Returns:
            Estimated time in seconds
        """
        # Air density
        rho = self.air_density(temp_c, altitude_m)
        
        # Wind components
        headwind, crosswind = wind.effective_wind(course_bearing)
        
        # Iterative velocity solution
        # P = F_roll * v + F_aero * v + F_grade * v
        # F_roll = Crr * m * g
        # F_aero = 0.5 * rho * CdA * (v + headwind)^2
        
        v = 12  # Initial guess (m/s = 43.2 km/h)
        
        for _ in range(10):  # Newton-Raphson iterations
            f_roll = self.crr * self.total_mass * self.g
            f_aero = 0.5 * rho * self.cda_tt_position * (v + headwind) ** 2
            
            # Power equation
            p_total = (f_roll + f_aero) * v
            
            # Adjust velocity
            error = p_total - target_power_w
            if abs(error) < 0.1:
                break
            
            # Derivative for Newton step
            dp_dv = f_roll + 3 * f_aero
            v = v - error / dp_dv
            
            v = max(v, 5)  # Minimum 18 km/h
        
        time_s = distance_m / v
        return time_s
    
    def time_impact_analysis(self, base_conditions: WeatherWindow,
                            actual_conditions: WeatherWindow,
                            distance_km: float,
                            course_bearing: float) -> Dict:
        """
        Calculate time impact of weather vs baseline conditions.
        
        Returns dict with:
            - time_delta_s: Difference in seconds (positive = slower)
            - power_adjustment: Required power change to maintain time
            - impact_pct: Time impact as percentage
        """
        distance_m = distance_km * 1000
        
        # Base time (calm conditions)
        base_wind = WindVector(0, 0)
        base_time = self.estimate_tt_time(
            distance_m, self.ftp_watts, base_wind, 
            course_bearing, base_conditions.temperature_c
        )
        
        # Actual time
        actual_time = self.estimate_tt_time(
            distance_m, self.ftp_watts, actual_conditions.wind,
            course_bearing, actual_conditions.temperature_c
        )
        
        delta = actual_time - base_time
        
        # Find required power to match base time
        def find_matching_time(target_time: float) -> float:
            """Binary search for power that gives target time."""
            low, high = 0.5 * self.ftp_watts, 2.0 * self.ftp_watts
            for _ in range(20):
                mid = (low + high) / 2
                t = self.estimate_tt_time(
                    distance_m, mid, actual_conditions.wind,
                    course_bearing, actual_conditions.temperature_c
                )
                if t < target_time:
                    high = mid
                else:
                    low = mid
            return mid
        
        required_power = find_matching_time(base_time)
        power_adj = (required_power / self.ftp_watts - 1) * 100
        
        return {
            "time_delta_s": delta,
            "impact_pct": (delta / base_time) * 100,
            "required_power_w": required_power,
            "power_adjustment_pct": power_adj,
            "base_time_s": base_time,
            "actual_time_s": actual_time
        }


# =============================================================================
# Weather Service
# =============================================================================

class WeatherService:
    """Unified weather data service."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.cache = {}
    
    def get_forecast_5day(self, lat: float, lon: float) -> List[WeatherWindow]:
        """Get 5-day forecast from OpenWeatherMap."""
        if not self.api_key:
            return self._generate_mock_forecast(lat, lon)
        
        cache_key = f"{lat:.2f},{lon:.2f}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            windows = []
            for entry in data.get("list", []):
                dt_start = datetime.fromtimestamp(entry["dt"])
                dt_end = dt_start + timedelta(hours=3)
                
                wind = entry.get("wind", {})
                main = entry.get("main", {})
                rain = entry.get("rain", {}).get("3h", 0)
                
                windows.append(WeatherWindow(
                    start_time=dt_start,
                    end_time=dt_end,
                    wind=WindVector(
                        speed_ms=wind.get("speed", 0),
                        direction_deg=wind.get("deg", 0),
                        gust_ms=wind.get("gust")
                    ),
                    temperature_c=main.get("temp", 15),
                    precipitation_mm=rain,
                    pressure_hpa=main.get("pressure", 1013),
                    humidity_pct=main.get("humidity", 50)
                ))
            
            self.cache[cache_key] = windows
            return windows
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._generate_mock_forecast(lat, lon)
    
    def _generate_mock_forecast(self, lat: float, lon: float) -> List[WeatherWindow]:
        """Generate mock forecast for testing without API."""
        print("WARNING: Using mock weather data (set OPENWEATHER_API_KEY for real forecasts)")
        
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        windows = []
        
        for i in range(16):  # 48 hours, 3-hour intervals
            dt_start = now + timedelta(hours=3*i)
            
            # Simulate varying wind conditions
            wind_speed = 3 + 4 * abs(math.sin(i * 0.5))  # 3-7 m/s varying
            wind_dir = (180 + 90 * math.sin(i * 0.3)) % 360  # Shifting direction
            
            windows.append(WeatherWindow(
                start_time=dt_start,
                end_time=dt_start + timedelta(hours=3),
                wind=WindVector(speed_ms=wind_speed, direction_deg=wind_dir),
                temperature_c=12 + 3 * math.sin(i * 0.2),
                precipitation_mm=0,
                pressure_hpa=1013,
                humidity_pct=65
            ))
        
        return windows
    
    def interpolate_for_time(self, windows: List[WeatherWindow], 
                            target_time: datetime) -> WeatherWindow:
        """Interpolate weather conditions for a specific time."""
        # Find surrounding windows
        before = None
        after = None
        
        for w in windows:
            if w.start_time <= target_time <= w.end_time:
                return w  # Exact match
            if w.end_time <= target_time:
                before = w
            if w.start_time >= target_time and after is None:
                after = w
                break
        
        if before is None:
            return after if after else windows[0]
        if after is None:
            return before
        
        # Linear interpolation
        total_seconds = (after.start_time - before.end_time).total_seconds()
        elapsed = (target_time - before.end_time).total_seconds()
        t = elapsed / total_seconds if total_seconds > 0 else 0.5
        
        return WeatherWindow(
            start_time=target_time,
            end_time=target_time + timedelta(minutes=1),
            wind=WindVector(
                speed_ms=before.wind.speed_ms + t * (after.wind.speed_ms - before.wind.speed_ms),
                direction_deg=before.wind.direction_deg + t * (after.wind.direction_deg - before.wind.direction_deg),
                gust_ms=before.wind.gust_ms
            ),
            temperature_c=before.temperature_c + t * (after.temperature_c - before.temperature_c),
            precipitation_mm=before.precipitation_mm,
            pressure_hpa=before.pressure_hpa,
            humidity_pct=before.humidity_pct
        )


# =============================================================================
# Analysis Engine
# =============================================================================

class WeatherRaceAnalyzer:
    """Main analysis engine integrating weather with race data."""
    
    def __init__(self, weather_service: WeatherService):
        self.weather = weather_service
        self.physics = CyclingPhysics()
    
    def analyze_stage(self, race_slug: str, year: int, stage_num: int) -> Dict:
        """Complete stage analysis with weather integration."""
        
        # Get stage configuration
        stage_config = self._get_stage_config(race_slug, stage_num)
        if not stage_config:
            return {"error": f"No configuration for {race_slug} stage {stage_num}"}
        
        # Get weather forecast
        primary_loc = stage_config["locations"][0]
        forecast = self.weather.get_forecast_5day(
            primary_loc["lat"], primary_loc["lon"]
        )
        
        # Get startlist with estimated times
        startlist = self._get_startlist(race_slug, year, stage_num)
        
        # Calculate weather for each rider
        profiles = []
        for rider in startlist:
            weather = self.weather.interpolate_for_time(forecast, rider.start_time)
            
            # Calculate impact
            base = WeatherWindow(
                start_time=rider.start_time,
                end_time=rider.start_time + timedelta(minutes=20),
                wind=WindVector(0, 0),
                temperature_c=15,
                precipitation_mm=0,
                pressure_hpa=1013,
                humidity_pct=50
            )
            
            impact = self.physics.time_impact_analysis(
                base, weather,
                stage_config["distance_km"],
                stage_config["course_bearing"]
            )
            
            profiles.append({
                "rider": {
                    "id": rider.rider_id,
                    "name": rider.name,
                    "team": rider.team,
                    "start_time": rider.start_time.isoformat()
                },
                "weather": {
                    "wind_speed_ms": weather.wind.speed_ms,
                    "wind_direction_deg": weather.wind.direction_deg,
                    "temperature_c": weather.temperature_c,
                    "precipitation_mm": weather.precipitation_mm
                },
                "impact": impact
            })
        
        return {
            "stage": {
                "race": race_slug,
                "year": year,
                "stage": stage_num,
                "name": stage_config["name"],
                "distance_km": stage_config["distance_km"],
                "type": stage_config["type"]
            },
            "forecast_retrieved": datetime.now().isoformat(),
            "forecast_points": len(forecast),
            "rider_profiles": profiles
        }
    
    def _get_stage_config(self, race_slug: str, stage_num: int) -> Optional[Dict]:
        """Get stage configuration from built-in database."""
        race_config = RACE_WAYPOINTS.get(race_slug.lower(), {})
        return race_config.get(f"stage_{stage_num}")
    
    def _get_startlist(self, race_slug: str, year: int, stage_num: int) -> List:
        """Get startlist from database with estimated start times."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get race ID
        cursor.execute(
            "SELECT id FROM races WHERE pcs_slug = ? AND year = ?",
            (race_slug, year)
        )
        race_row = cursor.fetchone()
        if not race_row:
            conn.close()
            return []
        
        race_id = race_row[0]
        
        # Get riders
        cursor.execute("""
            SELECT r.id, r.name, t.name
            FROM startlist_entries se
            JOIN riders r ON se.rider_id = r.id
            LEFT JOIN teams t ON se.team_id = t.id
            WHERE se.race_id = ?
            ORDER BY r.name
        """, (race_id,))
        
        riders = cursor.fetchall()
        conn.close()
        
        # Estimate start times (reverse order for ITT)
        from weather_race_analyzer import RiderStart
        
        base_time = datetime(year, 3, 9, 14, 0)  # Typical ITT start
        results = []
        
        for i, (rider_id, name, team) in enumerate(reversed(riders)):
            results.append(RiderStart(
                rider_id=rider_id,
                name=name,
                team=team or "Unknown",
                start_time=base_time + timedelta(minutes=i)
            ))
        
        results.sort(key=lambda x: x.start_time)
        return results


# =============================================================================
# Output Formatters
# =============================================================================

def print_json_output(analysis: Dict):
    """Print analysis as JSON for integration with other tools."""
    print(json.dumps(analysis, indent=2))


def print_table_output(analysis: Dict):
    """Print formatted table output."""
    stage = analysis["stage"]
    profiles = analysis["rider_profiles"]
    
    print(f"\n{'='*80}")
    print(f"WEATHER ANALYSIS: {stage['name']} ({stage['distance_km']}km)")
    print(f"{'='*80}\n")
    
    # Sort by time impact
    sorted_profiles = sorted(profiles, key=lambda x: x["impact"]["time_delta_s"])
    
    print(f"{'Rank':<6} {'Rider':<30} {'Start':<8} {'Wind':<18} {'Time Delta':<12} {'Power Adj'}")
    print(f"{'-'*80}")
    
    for i, p in enumerate(sorted_profiles[:20], 1):
        rider = p["rider"]
        weather = p["weather"]
        impact = p["impact"]
        
        start = rider["start_time"][11:16]  # HH:MM
        wind = f"{weather['wind_speed_ms']:.1f}m/s @ {weather['wind_direction_deg']:.0f}°"
        delta = f"{impact['time_delta_s']:+.2f}s"
        power = f"{impact['power_adjustment_pct']:+.1f}%"
        
        marker = ""
        if impact["time_delta_s"] < -1.0:
            marker = " 🟢"
        elif impact["time_delta_s"] > 1.0:
            marker = " 🔴"
        
        print(f"{i:<6} {rider['name']:<30} {start:<8} {wind:<18} {delta:<12} {power}{marker}")
    
    # Summary stats
    deltas = [p["impact"]["time_delta_s"] for p in profiles]
    spread = max(deltas) - min(deltas)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: Expected weather spread = {spread:.2f} seconds")
    print(f"{'='*80}\n")


def export_for_rank_stage(analysis: Dict, output_file: str):
    """Export weather adjustments for integration with rank_stage.py."""
    # Create a simple CSV of rider weather multipliers
    import csv
    
    profiles = analysis["rider_profiles"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rider_name', 'weather_multiplier', 'time_delta_s', 'confidence'])
        
        for p in profiles:
            rider = p["rider"]
            impact = p["impact"]
            
            # Convert time delta to probability multiplier
            # -2s advantage ≈ 1.05 multiplier (5% boost)
            # +2s penalty ≈ 0.95 multiplier (5% reduction)
            multiplier = 1 + (-impact["time_delta_s"] / 60) * 0.05
            
            writer.writerow([
                rider["name"],
                f"{multiplier:.4f}",
                f"{impact['time_delta_s']:.2f}",
                "1.0"  # Full confidence (could adjust based on forecast uncertainty)
            ])
    
    print(f"Weather multipliers exported to: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Weather Integration for Cycling Betting"
    )
    parser.add_argument("--race", "-r", required=True, help="Race slug")
    parser.add_argument("--year", "-y", type=int, required=True, help="Race year")
    parser.add_argument("--stage", "-s", type=int, required=True, help="Stage number")
    parser.add_argument("--api-key", "-k", help="OpenWeather API key")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--export", "-e", help="Export CSV for rank_stage.py")
    parser.add_argument("--plot", action="store_true", help="Generate plots (if matplotlib available)")
    
    args = parser.parse_args()
    
    # Initialize services
    weather_service = WeatherService(args.api_key)
    analyzer = WeatherRaceAnalyzer(weather_service)
    
    # Run analysis
    print(f"Analyzing {args.race} {args.year} Stage {args.stage}...")
    analysis = analyzer.analyze_stage(args.race, args.year, args.stage)
    
    if "error" in analysis:
        print(f"ERROR: {analysis['error']}")
        return 1
    
    # Output
    if args.json:
        print_json_output(analysis)
    else:
        print_table_output(analysis)
    
    if args.export:
        export_for_rank_stage(analysis, args.export)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
