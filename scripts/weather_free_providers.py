#!/usr/bin/env python3
"""
Free Weather Providers - No API Keys Required
=============================================

Completely free weather data sources for Strategy 6 (ITT Weather Arbitrage).

Supported providers:
- Open-Meteo: Global, hourly, no registration
- MET Norway: Europe-focused, very accurate
- Manual entry: For quick analysis

Usage:
    from weather_free_providers import OpenMeteoClient
    
    client = OpenMeteoClient()
    forecast = client.get_forecast(43.9, 10.2)  # Lido di Camaiore
"""

import requests
import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class WindCondition:
    """Standardized wind condition across all providers."""
    timestamp: datetime
    wind_speed_ms: float
    wind_direction_deg: float
    wind_gust_ms: Optional[float] = None
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    source: str = "unknown"


# =============================================================================
# Open-Meteo (Global, No API Key)
# =============================================================================

class OpenMeteoClient:
    """
    Open-Meteo API client - completely free, no registration required.
    
    Open-Meteo is an open-source weather API using data from national
    weather services (NOAA, DWD, MeteoSwiss, etc.).
    
    Website: https://open-meteo.com/
    Limits: ~10,000 calls/day (no hard limit)
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def get_forecast(self, lat: float, lon: float, 
                     days: int = 3) -> List[WindCondition]:
        """
        Get hourly forecast for location.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast (1-16)
        
        Returns:
            List of hourly WindCondition
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "windspeed_10m",
                "winddirection_10m",
                "windgusts_10m",
                "temperature_2m",
                "precipitation"
            ],
            "timezone": "auto",
            "forecast_days": days,
            "models": "best_match"  # Uses best available model for location
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
            
        except requests.exceptions.RequestException as e:
            print(f"[Open-Meteo] Error: {e}")
            return []
    
    def _parse_response(self, data: Dict) -> List[WindCondition]:
        """Parse Open-Meteo response into WindConditions."""
        hourly = data.get("hourly", {})
        
        times = hourly.get("time", [])
        speeds = hourly.get("windspeed_10m", [])
        directions = hourly.get("winddirection_10m", [])
        gusts = hourly.get("windgusts_10m", [])
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        
        conditions = []
        for i in range(len(times)):
            # Open-Meteo wind speed is in km/h, convert to m/s
            speed_ms = speeds[i] / 3.6 if i < len(speeds) else 0
            gust_ms = gusts[i] / 3.6 if i < len(gusts) and gusts[i] else None
            
            conditions.append(WindCondition(
                timestamp=datetime.fromisoformat(times[i]),
                wind_speed_ms=speed_ms,
                wind_direction_deg=directions[i] if i < len(directions) else 0,
                wind_gust_ms=gust_ms,
                temperature_c=temps[i] if i < len(temps) else None,
                precipitation_mm=precip[i] if i < len(precip) else 0,
                source="open-meteo"
            ))
        
        return conditions


# =============================================================================
# MET Norway (Europe-focused, very accurate)
# =============================================================================

class MetNorwayClient:
    """
    Norwegian Meteorological Institute API - free, no registration.
    
    Excellent for European races (Paris-Nice, Tirreno, etc.).
    Uses the same models as yr.no (highly regarded for accuracy).
    
    Website: https://api.met.no/
    Limits: Be polite (add User-Agent), no hard limits
    """
    
    BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    USER_AGENT = "CyclingPredict/1.0 github.com/sznajdr/cycling_predict"
    
    def get_forecast(self, lat: float, lon: float) -> List[WindCondition]:
        """Get forecast from MET Norway."""
        headers = {"User-Agent": self.USER_AGENT}
        params = {"lat": lat, "lon": lon}
        
        try:
            response = requests.get(
                self.BASE_URL, 
                headers=headers, 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
            
        except requests.exceptions.RequestException as e:
            print(f"[MET Norway] Error: {e}")
            return []
    
    def _parse_response(self, data: Dict) -> List[WindCondition]:
        """Parse MET Norway response."""
        timeseries = data.get("properties", {}).get("timeseries", [])
        
        conditions = []
        for entry in timeseries:
            time_str = entry.get("time")
            details = entry.get("data", {}).get("instant", {}).get("details", {})
            
            # MET Norway uses m/s directly
            speed = details.get("wind_speed", 0)
            direction = details.get("wind_from_direction", 0)
            gust = details.get("wind_speed_of_gust")
            temp = details.get("air_temperature")
            
            conditions.append(WindCondition(
                timestamp=datetime.fromisoformat(time_str.replace('Z', '+00:00')),
                wind_speed_ms=speed,
                wind_direction_deg=direction,
                wind_gust_ms=gust,
                temperature_c=temp,
                precipitation_mm=0,  # Not in instant data
                source="met-norway"
            ))
        
        return conditions


# =============================================================================
# NOAA/NWS (USA-focused)
# =============================================================================

class NOAAClient:
    """
    National Weather Service API - free, USA only.
    
    Best for: Tour of California, Tour of Utah, Colorado Classic
    """
    
    BASE_URL = "https://api.weather.gov/gridpoints"
    
    def get_forecast(self, lat: float, lon: float) -> List[WindCondition]:
        """Get forecast from NOAA (USA only)."""
        # First, get grid point
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        
        try:
            response = requests.get(points_url, timeout=30)
            response.raise_for_status()
            points_data = response.json()
            
            forecast_url = points_data["properties"]["forecastHourly"]
            forecast_resp = requests.get(forecast_url, timeout=30)
            forecast_resp.raise_for_status()
            
            return self._parse_response(forecast_resp.json())
            
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"[NOAA] Error (USA only): {e}")
            return []
    
    def _parse_response(self, data: Dict) -> List[WindCondition]:
        """Parse NOAA response."""
        periods = data.get("properties", {}).get("periods", [])
        
        conditions = []
        for period in periods:
            # NOAA wind format: "10 mph" or "15 km/h"
            wind_str = period.get("windSpeed", "0 mph")
            speed = self._parse_speed(wind_str)
            
            # Direction like "SW", "NNE"
            dir_str = period.get("windDirection", "N")
            direction = self._parse_direction(dir_str)
            
            conditions.append(WindCondition(
                timestamp=datetime.fromisoformat(period["startTime"]),
                wind_speed_ms=speed,
                wind_direction_deg=direction,
                wind_gust_ms=None,  # Not always available
                temperature_c=(period.get("temperature", 0) - 32) * 5/9 
                              if period.get("temperatureUnit") == "F" 
                              else period.get("temperature"),
                precipitation_mm=0,
                source="noaa"
            ))
        
        return conditions
    
    def _parse_speed(self, speed_str: str) -> float:
        """Parse speed string like '10 mph' to m/s."""
        parts = speed_str.split()
        if len(parts) != 2:
            return 0
        
        value = float(parts[0])
        unit = parts[1].lower()
        
        if unit == "mph":
            return value * 0.44704
        elif unit in ["km/h", "kph"]:
            return value / 3.6
        else:
            return value
    
    def _parse_direction(self, dir_str: str) -> float:
        """Convert cardinal direction to degrees."""
        directions = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
        }
        return directions.get(dir_str.upper(), 0)


# =============================================================================
# Unified Client with Fallback
# =============================================================================

class FreeWeatherClient:
    """
    Unified client that tries multiple free sources.
    
    Priority:
    1. Open-Meteo (global, hourly)
    2. MET Norway (Europe, high accuracy)
    3. NOAA (USA only)
    """
    
    def __init__(self):
        self.providers = {
            "global": OpenMeteoClient(),
            "europe": MetNorwayClient(),
            "usa": NOAAClient()
        }
    
    def get_forecast(self, lat: float, lon: float, 
                     region: str = "auto") -> List[WindCondition]:
        """
        Get forecast using best available free provider.
        
        Args:
            lat, lon: Coordinates
            region: "auto", "europe", "usa", or "global"
        
        Returns:
            List of WindCondition
        """
        # Determine region if auto
        if region == "auto":
            # Europe rough bounding box
            if 35 < lat < 72 and -12 < lon < 40:
                region = "europe"
            # USA rough bounding box
            elif 25 < lat < 50 and -125 < lon < -65:
                region = "usa"
            else:
                region = "global"
        
        # Try region-specific first, then fall back
        providers_to_try = []
        if region == "europe":
            providers_to_try = ["europe", "global"]
        elif region == "usa":
            providers_to_try = ["usa", "global"]
        else:
            providers_to_try = ["global", "europe"]
        
        for provider_name in providers_to_try:
            provider = self.providers[provider_name]
            conditions = provider.get_forecast(lat, lon)
            
            if conditions:
                print(f"[FreeWeather] Using {provider_name} provider - "
                      f"got {len(conditions)} data points")
                return conditions
        
        print("[FreeWeather] All providers failed - using mock data")
        return self._generate_mock_data(lat, lon)
    
    def _generate_mock_data(self, lat: float, lon: float) -> List[WindCondition]:
        """Generate realistic mock data as final fallback."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        conditions = []
        
        for i in range(48):  # 48 hours
            dt = now + timedelta(hours=i)
            
            # Diurnal wind pattern (stronger afternoon)
            hour = dt.hour
            base_speed = 2 + 3 * math.sin((hour - 6) * math.pi / 12) ** 2
            
            # Some random variation
            speed = max(1, base_speed + math.sin(i * 0.5))
            direction = (180 + 90 * math.sin(i * 0.3)) % 360
            
            conditions.append(WindCondition(
                timestamp=dt,
                wind_speed_ms=speed,
                wind_direction_deg=direction,
                wind_gust_ms=speed * 1.2,
                temperature_c=15 + 5 * math.sin((hour - 6) * math.pi / 12),
                precipitation_mm=0,
                source="mock-fallback"
            ))
        
        return conditions


# =============================================================================
# Manual Entry (For Quick Analysis)
# =============================================================================

def parse_manual_forecast(forecast_str: str) -> List[WindCondition]:
    """
    Parse manually entered forecast.
    
    Format: "HH:MM:speed@dir,HH:MM:speed@dir,..."
    Example: "14:00:5.2@180,15:00:6.8@200,16:00:4.1@220"
    
    Args:
        forecast_str: Comma-separated time:wind entries
    
    Returns:
        List of WindCondition
    """
    conditions = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for entry in forecast_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        
        try:
            # Format: HH:MM:speed@dir or HH:MM:speed
            time_part, wind_part = entry.split(":", 2)[0], entry.split(":", 2)[1] + ":" + entry.split(":", 2)[2]
            hour, minute = int(time_part), int(wind_part.split(":")[0])
            
            wind_data = wind_part.split(":")[1]
            if "@" in wind_data:
                speed_str, dir_str = wind_data.split("@")
                speed = float(speed_str)
                direction = float(dir_str)
            else:
                speed = float(wind_data)
                direction = 0
            
            conditions.append(WindCondition(
                timestamp=today + timedelta(hours=hour, minutes=minute),
                wind_speed_ms=speed,
                wind_direction_deg=direction,
                wind_gust_ms=speed * 1.15,
                temperature_c=15,
                precipitation_mm=0,
                source="manual"
            ))
            
        except (ValueError, IndexError) as e:
            print(f"[Manual] Skipping invalid entry '{entry}': {e}")
            continue
    
    return conditions


# =============================================================================
# Demo / Testing
# =============================================================================

def demo():
    """Demo all free weather providers."""
    print("=" * 70)
    print("FREE WEATHER PROVIDERS DEMO")
    print("=" * 70)
    
    # Lido di Camaiore (Tirreno-Adriatico)
    lat, lon = 43.9, 10.2
    
    # Open-Meteo
    print("\n1. Open-Meteo (Global)")
    print("-" * 70)
    openmeteo = OpenMeteoClient()
    forecast = openmeteo.get_forecast(lat, lon, days=2)
    
    if forecast:
        for c in forecast[:5]:
            print(f"  {c.timestamp.strftime('%Y-%m-%d %H:%M')}: "
                  f"{c.wind_speed_ms:.1f} m/s @ {c.wind_direction_deg:.0f}° "
                  f"({c.temperature_c:.1f}°C)")
    
    # MET Norway
    print("\n2. MET Norway (Europe)")
    print("-" * 70)
    met = MetNorwayClient()
    forecast = met.get_forecast(lat, lon)
    
    if forecast:
        for c in forecast[:5]:
            print(f"  {c.timestamp.strftime('%Y-%m-%d %H:%M')}: "
                  f"{c.wind_speed_ms:.1f} m/s @ {c.wind_direction_deg:.0f}°")
    
    # Unified client
    print("\n3. Unified Client (Auto-selection)")
    print("-" * 70)
    unified = FreeWeatherClient()
    forecast = unified.get_forecast(lat, lon)
    
    if forecast:
        print(f"  Selected provider: {forecast[0].source}")
        print(f"  Data points: {len(forecast)}")
        print(f"  Time range: {forecast[0].timestamp} to {forecast[-1].timestamp}")
    
    # Manual entry
    print("\n4. Manual Entry")
    print("-" * 70)
    manual = "14:00:5.2@180,15:00:6.8@200,16:00:4.1@220"
    conditions = parse_manual_forecast(manual)
    for c in conditions:
        print(f"  {c.timestamp.strftime('%H:%M')}: "
              f"{c.wind_speed_ms:.1f} m/s @ {c.wind_direction_deg:.0f}°")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
