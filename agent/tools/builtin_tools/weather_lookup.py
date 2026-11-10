"""Weather lookup builtin tool — supports Open-Meteo, OpenWeatherMap and WeatherAPI."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# WMO weather code → human-readable condition
_WMO_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class WeatherLookupTool:
    """Look up current weather and short-term forecasts.

    Supported providers (selected via ``tool_config``):
    - ``open-meteo``   — free, no API key (default)
    - ``openweathermap`` — requires API key
    - ``weatherapi``   — requires API key
    """

    name = "weather_lookup"
    description = (
        "Get current weather and a short-term forecast for a location. "
        "Use 'location' (required, city name or 'lat,lon'), "
        "'forecast_days' (0-7, default 0), and 'units' ('metric' or 'imperial', default 'metric')."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or 'latitude,longitude' coordinates.",
                        },
                        "forecast_days": {
                            "type": "integer",
                            "description": "Number of forecast days to include (0 = current only, max 7).",
                            "default": 0,
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "Temperature units (default 'metric').",
                            "default": "metric",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

    # ------------------------------------------------------------------ #
    # Public execute
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        location: str,
        forecast_days: int = 0,
        units: str = "metric",
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = (context or {}).get("tool_config", {})
        provider = cfg.get("provider", "open-meteo").lower().strip()
        api_key = cfg.get("api_key", "")

        forecast_days = max(0, min(int(forecast_days), 7))

        if provider == "open-meteo":
            return await self._fetch_openmeteo(location, forecast_days, units)
        if provider == "openweathermap":
            return await self._fetch_openweathermap(location, forecast_days, units, api_key)
        if provider == "weatherapi":
            return await self._fetch_weatherapi(location, forecast_days, units, api_key)

        raise RuntimeError(f"Unknown weather provider: {provider}")

    # ------------------------------------------------------------------ #
    # Open-Meteo (free, no key)
    # ------------------------------------------------------------------ #

    async def _fetch_openmeteo(
        self, location: str, forecast_days: int, units: str
    ) -> str:
        lat, lon = await self._resolve_coordinates(location)

        unit_params = "temperature_unit=fahrenheit&wind_speed_unit=mph" if units == "imperial" else ""
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
            f"&daily=temperature_2m_max,temperature_2m_min,weather_code"
            f"&forecast_days={forecast_days + 1}&timezone=auto"
        )
        if unit_params:
            url += f"&{unit_params}"

        data = await self._http_json(url)
        current = data.get("current", {})
        daily = data.get("daily", {})

        temp_unit = "°F" if units == "imperial" else "°C"
        wind_unit = "mph" if units == "imperial" else "km/h"

        lines: List[str] = [
            f"Weather for {location} (Open-Meteo)",
            "",
            f"Current: {current.get('temperature_2m', 'N/A')}{temp_unit}",
            f"Feels like: {current.get('apparent_temperature', 'N/A')}{temp_unit}",
            f"Condition: {_WMO_CODES.get(current.get('weather_code', -1), 'Unknown')}",
            f"Humidity: {current.get('relative_humidity_2m', 'N/A')}%",
            f"Wind: {current.get('wind_speed_10m', 'N/A')} {wind_unit}",
        ]

        if forecast_days > 0:
            lines.append("")
            lines.append("Forecast:")
            dates = daily.get("time", [])
            maxs = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            codes = daily.get("weather_code", [])
            # Skip index 0 (today) if we only want future days
            for i in range(1, forecast_days + 1):
                if i < len(dates):
                    lines.append(
                        f"  {dates[i]}: {_WMO_CODES.get(codes[i], 'Unknown')} — "
                        f"High {maxs[i]}{temp_unit}, Low {mins[i]}{temp_unit}"
                    )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # OpenWeatherMap (API key required)
    # ------------------------------------------------------------------ #

    async def _fetch_openweathermap(
        self, location: str, forecast_days: int, units: str, api_key: str
    ) -> str:
        if not api_key:
            raise RuntimeError(
                "OpenWeatherMap requires an API key. "
                "Set it in the tool config (dashboard → Tools → weather_lookup)."
            )

        unit_param = "imperial" if units == "imperial" else "metric"
        temp_unit = "°F" if units == "imperial" else "°C"
        wind_unit = "mph" if units == "imperial" else "m/s"

        current_url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"q={location}&appid={api_key}&units={unit_param}"
        )
        current_data = await self._http_json(current_url)

        main = current_data.get("main", {})
        weather = current_data.get("weather", [{}])[0]
        wind = current_data.get("wind", {})

        lines: List[str] = [
            f"Weather for {location} (OpenWeatherMap)",
            "",
            f"Current: {main.get('temp', 'N/A')}{temp_unit}",
            f"Feels like: {main.get('feels_like', 'N/A')}{temp_unit}",
            f"Condition: {weather.get('main', 'Unknown')} — {weather.get('description', '')}",
            f"Humidity: {main.get('humidity', 'N/A')}%",
            f"Wind: {wind.get('speed', 'N/A')} {wind_unit}",
        ]

        if forecast_days > 0:
            forecast_url = (
                f"https://api.openweathermap.org/data/2.5/forecast?"
                f"q={location}&appid={api_key}&units={unit_param}&cnt={forecast_days * 8}"
            )
            forecast_data = await self._http_json(forecast_url)
            lines.append("")
            lines.append("Forecast (3-hour intervals):")
            for item in forecast_data.get("list", [])[:forecast_days * 8]:
                dt = item.get("dt_txt", "")
                temp = item.get("main", {}).get("temp", "N/A")
                desc = item.get("weather", [{}])[0].get("description", "")
                lines.append(f"  {dt}: {temp}{temp_unit} — {desc}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # WeatherAPI (API key required)
    # ------------------------------------------------------------------ #

    async def _fetch_weatherapi(
        self, location: str, forecast_days: int, units: str, api_key: str
    ) -> str:
        if not api_key:
            raise RuntimeError(
                "WeatherAPI requires an API key. "
                "Set it in the tool config (dashboard → Tools → weather_lookup)."
            )

        days = forecast_days + 1  # API includes today
        url = (
            f"https://api.weatherapi.com/v1/forecast.json?"
            f"key={api_key}&q={location}&days={days}&aqi=no"
        )
        data = await self._http_json(url)

        current = data.get("current", {})
        loc = data.get("location", {})
        forecast_days_list = data.get("forecast", {}).get("forecastday", [])

        is_imperial = units == "imperial"
        temp = current.get("temp_f" if is_imperial else "temp_c", "N/A")
        feels_like = current.get("feelslike_f" if is_imperial else "feelslike_c", "N/A")
        wind = current.get("wind_mph" if is_imperial else "wind_kph", "N/A")
        wind_unit = "mph" if is_imperial else "km/h"
        temp_unit = "°F" if is_imperial else "°C"

        lines: List[str] = [
            f"Weather for {loc.get('name', location)} (WeatherAPI)",
            "",
            f"Current: {temp}{temp_unit}",
            f"Feels like: {feels_like}{temp_unit}",
            f"Condition: {current.get('condition', {}).get('text', 'Unknown')}",
            f"Humidity: {current.get('humidity', 'N/A')}%",
            f"Wind: {wind} {wind_unit}",
        ]

        if forecast_days > 0 and len(forecast_days_list) > 1:
            lines.append("")
            lines.append("Forecast:")
            for day in forecast_days_list[1:forecast_days + 1]:
                date = day.get("date", "")
                day_data = day.get("day", {})
                max_temp = day_data.get("maxtemp_f" if is_imperial else "maxtemp_c", "N/A")
                min_temp = day_data.get("mintemp_f" if is_imperial else "mintemp_c", "N/A")
                condition = day_data.get("condition", {}).get("text", "Unknown")
                lines.append(
                    f"  {date}: {condition} — High {max_temp}{temp_unit}, Low {min_temp}{temp_unit}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _resolve_coordinates(self, location: str) -> tuple[float, float]:
        """Return (lat, lon) from a "lat,lon" string or geocode a city name."""
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", location)
        if m:
            return float(m.group(1)), float(m.group(2))

        url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        data = await self._http_json(url)
        results = data.get("results", [])
        if not results:
            raise RuntimeError(f"Could not geocode location: {location}")
        return float(results[0]["latitude"]), float(results[0]["longitude"])

    async def _http_json(self, url: str) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                return await resp.json()
