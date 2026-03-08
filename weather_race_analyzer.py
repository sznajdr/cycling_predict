#!/usr/bin/env python3
"""
Weather Race Analyzer - ITT Wind Arbitrage Tool
================================================

Fetches weather forecasts for race locations, maps conditions against
start times, and calculates expected time deltas between early/late starters.

Critical for Strategy 6 (ITT Weather Arbitrage) - coastal/windy stages
can create 10-30 second swings between early and late starters.

Usage:
    python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1
    python weather_race_analyzer.py --race paris-nice --year 2026 --stage 1 --itt
    python weather_race_analyzer.py --race tour-de-france --year 2025 --stage 21 --plot

Environment:
    Set OPENWEATHER_API_KEY environment variable for forecasts
    (Get free key at: https://openweathermap.org/api)
"""

import argparse
import os
import sys
import sqlite3
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import requests
import time

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.db import get_connection
from weather_free_providers import FreeWeatherClient, parse_manual_forecast


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WindCondition:
    """Wind condition at a specific time/location."""
    timestamp: datetime
    wind_speed_ms: float  # m/s
    wind_direction_deg: float  # degrees (meteorological: 0=N, 90=E, 180=S, 270=W)
    wind_gust_ms: Optional[float] = None
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    
    @property
    def wind_speed_kmh(self) -> float:
        return self.wind_speed_ms * 3.6
    
    @property
    def is_strong_wind(self) -> bool:
        return self.wind_speed_ms >= 8  # > 28.8 km/h


@dataclass
class RiderStart:
    """Rider with start time for ITT analysis."""
    rider_id: int
    name: str
    team: str
    start_time: datetime
    gc_position: Optional[int] = None
    itt_specialty: Optional[float] = None
    
    @property
    def start_hour(self) -> int:
        return self.start_time.hour


@dataclass  
class WindImpact:
    """Calculated impact of wind on a rider."""
    rider_name: str
    start_time: datetime
    wind_speed_ms: float
    wind_direction_deg: float
    # Impact metrics
    headwind_component: float  # negative = headwind, positive = tailwind
    crosswind_component: float  # lateral wind
    estimated_time_delta_s: float  # vs neutral conditions
    advantage_score: float  # 0-100, higher = better conditions
    

# =============================================================================
# Weather API Integration
# =============================================================================

class WeatherClient:
    """
    Weather client using free providers (no API key needed).
    
    Uses Open-Meteo, MET Norway, or NOAA depending on location.
    Falls back to mock data if all providers fail.
    """
    
    def __init__(self, manual_forecast: Optional[str] = None):
        """
        Args:
            manual_forecast: Optional manual forecast string 
                            (format: "HH:MM:speed@dir,HH:MM:speed@dir,...")
        """
        self.manual_forecast = manual_forecast
        self.free_client = FreeWeatherClient()
    
    def get_forecast(self, lat: float, lon: float) -> List[WindCondition]:
        """Get forecast using free providers or manual entry."""
        if self.manual_forecast:
            print(f"[Weather] Using manual forecast: {self.manual_forecast}")
            return parse_manual_forecast(self.manual_forecast)
        
        # Use free providers (Open-Meteo, MET Norway, NOAA)
        conditions = self.free_client.get_forecast(lat, lon)
        
        # Convert to our WindCondition format (ensure naive datetimes)
        result = []
        for c in conditions:
            ts = c.timestamp
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            result.append(WindCondition(
                timestamp=ts,
                wind_speed_ms=c.wind_speed_ms,
                wind_direction_deg=c.wind_direction_deg,
                wind_gust_ms=c.wind_gust_ms,
                temperature_c=c.temperature_c,
                precipitation_mm=c.precipitation_mm
            ))
        return result
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[WindCondition]:
        """Get current weather (uses first forecast hour as approximation)."""
        conditions = self.get_forecast(lat, lon)
        if conditions:
            # Find condition closest to now
            now = datetime.now()
            return min(conditions, key=lambda c: abs((c.timestamp - now).total_seconds()))
        return None


# =============================================================================
# Physics Models
# =============================================================================

class ITTAeroModel:
    """
    Simplified aerodynamic model for ITT time impact.
    
    Based on power equation: P = F_aero * v = 0.5 * rho * CdA * v^3
    where wind affects relative velocity.
    
    For Strategy 6 (Weather Arbitrage), we estimate time deltas between
    riders facing different wind conditions.
    """
    
    def __init__(self, 
                 cd_a: float = 0.24,  # Drag coefficient * area (typical TT position)
                 rho: float = 1.225,   # Air density kg/m^3 (sea level)
                 rider_power_w: float = 400,  # Sustainable TT power
                 distance_km: float = 11.5):  # Stage distance
        self.cd_a = cd_a
        self.rho = rho
        self.rider_power_w = rider_power_w
        self.distance_m = distance_km * 1000
    
    def calculate_wind_components(self, 
                                   wind_speed: float, 
                                   wind_direction: float,
                                   course_direction: float = 0) -> Tuple[float, float]:
        """
        Decompose wind into head/tail and cross components.
        
        Args:
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees (meteorological)
            course_direction: Course bearing in degrees (0=North, 90=East)
        
        Returns:
            (headwind_component, crosswind_component)
            Positive headwind = tailwind (helping)
            Negative headwind = headwind (hurting)
        """
        # Convert to math angles (0=East, increase counter-clockwise)
        wind_math = math.radians(90 - wind_direction)
        course_math = math.radians(90 - course_direction)
        
        # Wind vector components
        wind_x = wind_speed * math.cos(wind_math)
        wind_y = wind_speed * math.sin(wind_math)
        
        # Course unit vector
        course_x = math.cos(course_math)
        course_y = math.sin(course_math)
        
        # Project wind onto course direction
        headwind = -(wind_x * course_x + wind_y * course_y)  # Negative = headwind
        
        # Crosswind magnitude
        crosswind = abs(wind_x * (-course_y) + wind_y * course_x)
        
        return headwind, crosswind
    
    def estimate_time_delta(self, 
                           condition1: WindCondition, 
                           condition2: WindCondition,
                           course_direction: float = 0) -> float:
        """
        Estimate time delta (seconds) between two wind conditions.
        
        Returns: Time for condition1 minus time for condition2
                 Positive = condition1 is slower
        """
        # Simplified model: time proportional to relative wind power
        v_rider = self.rider_power_w / (0.5 * self.rho * self.cd_a)
        v_rider = (v_rider) ** (1/3)  # Approximate rider speed in calm air
        
        # Wind impact on effective speed
        def effective_speed(wind: WindCondition) -> float:
            hw, _ = self.calculate_wind_components(
                wind.wind_speed_ms, wind.wind_direction_deg, course_direction
            )
            # Tailwind increases speed, headwind decreases
            # Simplified: linear approximation for small wind speeds
            return v_rider + hw * 0.5  # 0.5 factor for partial wind benefit
        
        v1 = max(effective_speed(condition1), 5)  # Minimum speed
        v2 = max(effective_speed(condition2), 5)
        
        time1 = self.distance_m / v1
        time2 = self.distance_m / v2
        
        return time1 - time2
    
    def advantage_score(self, wind: WindCondition, course_direction: float = 0) -> float:
        """
        Calculate an advantage score (0-100) for given wind conditions.
        Higher = better for the rider.
        """
        hw, cw = self.calculate_wind_components(
            wind.wind_speed_ms, wind.wind_direction_deg, course_direction
        )
        
        # Tailwind is good, headwind is bad
        # Crosswind is neutral to slightly bad (handling risk)
        score = 50  # Neutral
        score += hw * 5  # +/- 5 points per m/s of tail/head wind
        score -= cw * 1  # Slight penalty for crosswind
        
        # Gust penalty
        if wind.wind_gust_ms and wind.wind_gust_ms > wind.wind_speed_ms * 1.3:
            score -= 5  # Unpredictable conditions
        
        return max(0, min(100, score))


# =============================================================================
# Database Integration
# =============================================================================

def get_stage_info(race_slug: str, year: int, stage_num: int) -> Optional[Dict]:
    """Get stage metadata from database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT rs.id, rs.stage_number, rs.distance_km, rs.stage_type,
               r.display_name, r.year, r.pcs_slug
        FROM race_stages rs
        JOIN races r ON rs.race_id = r.id
        WHERE r.pcs_slug = ? AND r.year = ? AND rs.stage_number = ?
    """, (race_slug, year, stage_num))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "stage_id": row[0],
        "stage_number": row[1],
        "distance_km": row[2],
        "stage_type": row[3],
        "race_name": row[4],
        "year": row[5],
        "race_slug": row[6],
        "start_location": "Unknown",
        "finish_location": "Unknown"
    }


def get_startlist_with_times(race_slug: str, year: int, stage_num: int) -> List[RiderStart]:
    """
    Get startlist with estimated start times for ITT.
    
    For ITTs, riders start in reverse GC order (last in GC starts first).
    We estimate times based on GC position.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get stage info
    stage_info = get_stage_info(race_slug, year, stage_num)
    if not stage_info:
        return []
    
    stage_id = stage_info["stage_id"]
    
    # Get all riders in startlist
    cursor.execute("""
        SELECT r.id, r.name, t.name as team_name, r.sp_time_trial
        FROM startlist_entries se
        JOIN riders r ON se.rider_id = r.id
        LEFT JOIN teams t ON se.team_id = t.id
        WHERE se.race_id = (
            SELECT id FROM races WHERE pcs_slug = ? AND year = ?
        )
        ORDER BY r.name
    """, (race_slug, year))
    
    riders = []
    for row in cursor.fetchall():
        riders.append({
            "id": row[0],
            "name": row[1],
            "team": row[2] or "Unknown",
            "itt_specialty": row[3]
        })
    
    # Estimate start times (reverse order = last rider first)
    # Typical ITT: 1 rider per minute, first rider at ~14:00
    base_time = datetime(year, 3, 9, 14, 0)  # Default start
    start_interval = timedelta(minutes=1)
    
    rider_starts = []
    # Reverse order: last rider in list starts first
    for i, rider in enumerate(reversed(riders)):
        start_time = base_time + (i * start_interval)
        rider_starts.append(RiderStart(
            rider_id=rider["id"],
            name=rider["name"],
            team=rider["team"],
            start_time=start_time,
            itt_specialty=rider["itt_specialty"]
        ))
    
    # Sort back by start time
    rider_starts.sort(key=lambda x: x.start_time)
    
    conn.close()
    return rider_starts


def get_race_location(race_slug: str, year: int) -> Optional[Tuple[float, float]]:
    """
    Get approximate race location coordinates.
    Returns (latitude, longitude) or None.
    """
    # Known race locations (approximate start/finish)
    LOCATIONS = {
        "tirreno-adriatico": (43.9, 10.2),    # Lido di Camaiore, Italy
        "paris-nice": (48.8566, 2.3522),       # Paris area, France
        "tour-de-france": (48.8566, 2.3522),   # Varies, default Paris
        "giro-d-italia": (45.4642, 9.1900),    # Milan area, Italy
        "vuelta-a-espana": (40.4168, -3.7038), # Madrid area, Spain
        "tour-de-suisse": (47.3769, 8.5417),   # Zurich area, Switzerland
        "dauphine": (45.7640, 4.8357),         # Lyon area, France
    }
    
    return LOCATIONS.get(race_slug.lower())


# =============================================================================
# Output Formatting
# =============================================================================

def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_weather_timeline(conditions: List[WindCondition], start_times: List[datetime]):
    """Print weather conditions aligned with start times."""
    print_header("WEATHER FORECAST - TIMELINE", "-")
    
    print(f"{'Time':<12} {'Wind':<20} {'Temp':<10} {'Rain':<10} {'Riders Starting'}")
    print("-" * 70)
    
    # Group riders by hour
    riders_by_hour = defaultdict(list)
    for rider in rider_starts:
        riders_by_hour[rider.start_hour].append(rider.name.split()[-1])  # Last name
    
    for cond in conditions[:12]:  # Show next 12 timepoints (36 hours)
        time_str = cond.timestamp.strftime("%a %H:%M")
        wind_str = f"{cond.wind_speed_ms:.1f} m/s @ {cond.wind_direction_deg:.0f} deg"
        temp_str = f"{cond.temperature_c:.1f}C" if cond.temperature_c else "N/A"
        rain_str = f"{cond.precipitation_mm:.1f}mm" if cond.precipitation_mm else "0mm"
        
        # Show riders starting this hour
        hour = cond.timestamp.hour
        riders = riders_by_hour.get(hour, [])
        rider_str = ", ".join(riders[:3]) + ("..." if len(riders) > 3 else "") if riders else "-"
        
        marker = " <<< NOW" if cond.timestamp <= datetime.now() + timedelta(hours=3) else ""
        print(f"{time_str:<12} {wind_str:<20} {temp_str:<10} {rain_str:<10} {rider_str}{marker}")


def print_wind_impact_analysis(rider_impacts: List[WindImpact], top_n: int = 15):
    """Print detailed wind impact analysis for each rider."""
    print_header("WIND IMPACT ANALYSIS", "-")
    
    # Sort by advantage score
    sorted_impacts = sorted(rider_impacts, key=lambda x: x.advantage_score, reverse=True)
    
    print(f"{'Rank':<6} {'Rider':<25} {'Start':<8} {'Wind':<18} {'Delta':<10} {'Advantage'}")
    print("-" * 85)
    
    for i, impact in enumerate(sorted_impacts[:top_n], 1):
        start_str = impact.start_time.strftime("%H:%M")
        wind_str = f"{impact.wind_speed_ms:.1f}m/s @ {impact.wind_direction_deg:.0f}deg"
        delta_str = f"{impact.estimated_time_delta_s:+.1f}s"
        adv_str = f"{impact.advantage_score:.0f}/100"
        
        # Marker for best/worst
        marker = ""
        if i == 1:
            marker = " [BEST]"
        elif i <= 3:
            marker = " [GOOD]"
        elif impact.advantage_score < 40:
            marker = " [POOR]"
        
        print(f"{i:<6} {impact.rider_name:<25} {start_str:<8} {wind_str:<18} {delta_str:<10} {adv_str}{marker}")
    
    # Worst conditions
    print("\n--- Most Disadvantaged ---")
    for impact in sorted_impacts[-5:]:
        start_str = impact.start_time.strftime("%H:%M")
        delta_str = f"{impact.estimated_time_delta_s:+.1f}s"
        print(f"  {impact.rider_name:<25} {start_str}  {delta_str}")


def print_strategy_recommendations(rider_impacts: List[WindImpact], 
                                   conditions: List[WindCondition]):
    """Print betting strategy recommendations."""
    print_header("STRATEGY RECOMMENDATIONS", "=")
    
    sorted_impacts = sorted(rider_impacts, key=lambda x: x.advantage_score, reverse=True)
    
    # Calculate overall spread
    best = sorted_impacts[0]
    worst = sorted_impacts[-1]
    spread = best.estimated_time_delta_s - worst.estimated_time_delta_s
    
    print(f"Expected time spread: {spread:.1f} seconds between best/worst conditions")
    print(f"Wind advantage variance: {best.advantage_score - worst.advantage_score:.0f} points")
    print()
    
    # Recommendations
    if spread > 15:
        print("[STRONG] ARBITRAGE OPPORTUNITY")
        print("   Wind conditions create >15s spread - significant edge possible")
    elif spread > 8:
        print("🟡 MODERATE OPPORTUNITY")  
        print("   Wind conditions create 8-15s spread - watch for line movements")
    else:
        print("[NEUTRAL] CONDITIONS")
        print("   Wind impact <8s - focus on rider quality over start time")
    
    print()
    
    # Back recommendations
    print("BACK (Wind Advantage):")
    for impact in sorted_impacts[:5]:
        delta = impact.estimated_time_delta_s
        if delta < -2:  # More than 2s advantage
            print(f"   - {impact.rider_name} ({impact.start_time.strftime('%H:%M')}) - "
                  f"{abs(delta):.1f}s advantage")
    
    print()
    
    # Lay/Fade recommendations
    print("FADE (Wind Disadvantage):")
    for impact in sorted_impacts[-5:]:
        delta = impact.estimated_time_delta_s
        if delta > 2:  # More than 2s disadvantage
            print(f"   - {impact.rider_name} ({impact.start_time.strftime('%H:%M')}) - "
                  f"{delta:.1f}s disadvantage")
    
    print()
    
    # Market timing
    print("MARKET TIMING:")
    early_avg = sum(r.wind_speed_ms for r in sorted_impacts[:len(sorted_impacts)//3]) / (len(sorted_impacts)//3)
    late_avg = sum(r.wind_speed_ms for r in sorted_impacts[-len(sorted_impacts)//3:]) / (len(sorted_impacts)//3)
    
    if early_avg > late_avg + 2:
        print(f"   Early starters face stronger winds ({early_avg:.1f} vs {late_avg:.1f} m/s)")
        print("   -> Late starters may be undervalued")
    elif late_avg > early_avg + 2:
        print(f"   Late starters face stronger winds ({late_avg:.1f} vs {early_avg:.1f} m/s)")
        print("   -> Early starters may be undervalued")
    else:
        print("   Wind conditions relatively stable throughout start window")


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_itt_weather(race_slug: str, year: int, stage_num: int,
                        manual_forecast: Optional[str] = None,
                        course_bearing: float = 0) -> None:
    """Complete ITT weather analysis workflow."""
    
    print_header(f"WEATHER RACE ANALYZER - {race_slug.upper()} {year} Stage {stage_num}", "=")
    
    # 1. Get stage info
    stage_info = get_stage_info(race_slug, year, stage_num)
    if not stage_info:
        print(f"ERROR: Stage {stage_num} not found in database for {race_slug} {year}")
        print("Run: python -m pipeline.runner")
        return
    
    print(f"Race: {stage_info['race_name']} ({stage_info['year']})")
    print(f"Stage: {stage_info['stage_number']} - {stage_info['distance_km']}km")
    print(f"Type: {stage_info['stage_type']}")
    print(f"Route: {stage_info['start_location']} -> {stage_info['finish_location']}")
    
    if stage_info['stage_type'] != 'itt':
        print(f"\n[!] Warning: Stage is '{stage_info['stage_type']}', not ITT")
        print("   Weather analysis most valuable for Individual Time Trials")
    
    # 2. Get location
    coords = get_race_location(race_slug, year)
    if not coords:
        print(f"ERROR: No coordinates known for {race_slug}")
        return
    
    lat, lon = coords
    print(f"Location: {lat:.3f}, {lon:.3f}")
    
    # 3. Fetch weather
    print("\nFetching weather forecast...")
    client = WeatherClient(manual_forecast=manual_forecast)
    conditions = client.get_forecast(lat, lon)
    
    if not conditions:
        print("ERROR: No weather data available")
        return
    
    print(f"Retrieved {len(conditions)} forecast intervals")
    
    # 4. Get startlist
    global rider_starts
    rider_starts = get_startlist_with_times(race_slug, year, stage_num)
    
    if not rider_starts:
        print(f"ERROR: No startlist found. Run pipeline to scrape race data.")
        return
    
    print(f"Startlist: {len(rider_starts)} riders")
    print(f"Start window: {rider_starts[0].start_time.strftime('%H:%M')} - "
          f"{rider_starts[-1].start_time.strftime('%H:%M')}")
    
    # 5. Calculate impacts
    aero_model = ITTAeroModel(distance_km=stage_info['distance_km'])
    
    # Interpolate wind conditions for each start time
    def interpolate_wind(target_time: datetime) -> WindCondition:
        """Interpolate wind conditions between forecast points."""
        # Find surrounding forecast points
        before = None
        after = None
        
        for c in conditions:
            if c.timestamp <= target_time:
                before = c
            if c.timestamp >= target_time and after is None:
                after = c
                break
        
        if before is None:
            return after if after else conditions[0]
        if after is None:
            return before
        if before.timestamp == after.timestamp:
            return before
        
        # Linear interpolation
        total = (after.timestamp - before.timestamp).total_seconds()
        elapsed = (target_time - before.timestamp).total_seconds()
        t = elapsed / total if total > 0 else 0.5
        
        # Handle wind direction wraparound
        dir_diff = after.wind_direction_deg - before.wind_direction_deg
        while dir_diff > 180:
            dir_diff -= 360
        while dir_diff < -180:
            dir_diff += 360
        
        interp_dir = (before.wind_direction_deg + t * dir_diff) % 360
        
        return WindCondition(
            timestamp=target_time,
            wind_speed_ms=before.wind_speed_ms + t * (after.wind_speed_ms - before.wind_speed_ms),
            wind_direction_deg=interp_dir,
            wind_gust_ms=before.wind_gust_ms + t * (after.wind_gust_ms - before.wind_gust_ms) if before.wind_gust_ms else None,
            temperature_c=before.temperature_c + t * (after.temperature_c - before.temperature_c) if before.temperature_c else None,
            precipitation_mm=before.precipitation_mm + t * (after.precipitation_mm - before.precipitation_mm)
        )
    
    rider_impacts = []
    for rider in rider_starts:
        # Interpolate wind for this rider's start time
        rider_wind = interpolate_wind(rider.start_time)
        
        # Calculate impact
        hw, cw = aero_model.calculate_wind_components(
            rider_wind.wind_speed_ms,
            rider_wind.wind_direction_deg,
            course_bearing
        )
        
        # Estimate delta vs neutral conditions
        neutral = WindCondition(
            timestamp=rider.start_time,
            wind_speed_ms=0,
            wind_direction_deg=0
        )
        delta = aero_model.estimate_time_delta(rider_wind, neutral, course_bearing)
        
        advantage = aero_model.advantage_score(rider_wind, course_bearing)
        
        rider_impacts.append(WindImpact(
            rider_name=rider.name,
            start_time=rider.start_time,
            wind_speed_ms=rider_wind.wind_speed_ms,
            wind_direction_deg=rider_wind.wind_direction_deg,
            headwind_component=hw,
            crosswind_component=cw,
            estimated_time_delta_s=delta,
            advantage_score=advantage
        ))
    
    # 6. Output results
    print_weather_timeline(conditions, rider_starts)
    print_wind_impact_analysis(rider_impacts)
    print_strategy_recommendations(rider_impacts, conditions)
    
    # 7. Summary
    print_header("EXECUTIVE SUMMARY", "=")
    
    sorted_impacts = sorted(rider_impacts, key=lambda x: x.advantage_score, reverse=True)
    best = sorted_impacts[0]
    worst = sorted_impacts[-1]
    
    spread = best.estimated_time_delta_s - worst.estimated_time_delta_s
    
    print(f"Expected weather impact spread: {spread:.1f} seconds")
    print(f"Best conditions:  {best.rider_name} at {best.start_time.strftime('%H:%M')} "
          f"(advantage: {abs(best.estimated_time_delta_s):.1f}s)")
    print(f"Worst conditions: {worst.rider_name} at {worst.start_time.strftime('%H:%M')} "
          f"(penalty: {worst.estimated_time_delta_s:.1f}s)")
    print()
    
    # Top picks
    print("Top 3 Riders by Wind Advantage:")
    for i, impact in enumerate(sorted_impacts[:3], 1):
        print(f"  {i}. {impact.rider_name} ({impact.start_time.strftime('%H:%M')}) - "
              f"Score: {impact.advantage_score:.0f}/100")
    
    print()
    print("=" * 70)
    print("Strategy 6 (ITT Weather Arbitrage) recommends:")
    print("  1. Monitor forecast updates close to race start")
    print("  2. Compare model edge vs market prices")
    print("  3. Size positions using Robust Kelly (uncertainty high)")
    print("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weather Race Analyzer - ITT Wind Arbitrage Tool (FREE - no API key needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis (uses free Open-Meteo API - no key needed!)
    python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1
    
    # Manual forecast entry (no internet needed)
    python weather_race_analyzer.py --race tirreno-adriatico --year 2026 --stage 1 \\
        --manual "14:00:5.2@180,15:00:6.8@200,16:00:4.1@220"
    
    # Specify course direction (0=North, 90=East, etc.)
    python weather_race_analyzer.py --race tour-de-france --year 2025 --stage 21 \\
        --bearing 180
        
Notes:
    - Uses free Open-Meteo API by default (global coverage, hourly data)
    - For European races, automatically uses MET Norway (very accurate)
    - Manual mode allows entry without any API calls
        """
    )
    
    parser.add_argument("--race", "-r", required=True,
                       help="Race PCS slug (e.g., tirreno-adriatico, paris-nice)")
    parser.add_argument("--year", "-y", type=int, required=True,
                       help="Race year (e.g., 2026)")
    parser.add_argument("--stage", "-s", type=int, required=True,
                       help="Stage number (e.g., 1)")
    parser.add_argument("--manual", "-m",
                       help="Manual forecast (format: 'HH:MM:speed@dir,HH:MM:speed@dir')")
    parser.add_argument("--bearing", "-b", type=float, default=0,
                       help="Course bearing in degrees (0=North, 90=East, 180=South, 270=West)")
    parser.add_argument("--itt", action="store_true",
                       help="Force ITT mode (skip stage type check)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization (requires matplotlib)")
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_itt_weather(
        race_slug=args.race,
        year=args.year,
        stage_num=args.stage,
        manual_forecast=args.manual,
        course_bearing=args.bearing
    )


if __name__ == "__main__":
    main()
