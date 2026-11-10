"""Tests for the built-in weather_lookup tool."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.builtin_tools.weather_lookup import WeatherLookupTool


@pytest.fixture
def tool() -> WeatherLookupTool:
    return WeatherLookupTool()


class TestOpenMeteo:
    @pytest.mark.asyncio
    async def test_current_weather_only(self, tool):
        """Open-Meteo: current weather with 0 forecast days."""
        geocode = {
            "results": [{"latitude": 52.52, "longitude": 13.41}],
        }
        weather = {
            "current": {
                "temperature_2m": 15.3,
                "relative_humidity_2m": 62,
                "apparent_temperature": 14.1,
                "weather_code": 1,
                "wind_speed_10m": 12.5,
            },
            "daily": {},
        }

        call_count = 0
        async def _fake_http_json(url: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if "geocoding-api" in url:
                return geocode
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("Berlin", forecast_days=0, units="metric")

        assert "15.3°C" in result
        assert "Mainly clear" in result
        assert "Humidity: 62%" in result
        assert "Wind: 12.5 km/h" in result
        assert "Forecast:" not in result

    @pytest.mark.asyncio
    async def test_with_forecast(self, tool):
        """Open-Meteo: current + 2-day forecast."""
        geocode = {"results": [{"latitude": 48.85, "longitude": 2.35}]}
        weather = {
            "current": {
                "temperature_2m": 22.0,
                "relative_humidity_2m": 55,
                "apparent_temperature": 23.0,
                "weather_code": 0,
                "wind_speed_10m": 8.0,
            },
            "daily": {
                "time": ["2024-06-16", "2024-06-17", "2024-06-18"],
                "temperature_2m_max": [24, 26, 28],
                "temperature_2m_min": [16, 18, 19],
                "weather_code": [0, 1, 2],
            },
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            if "geocoding-api" in url:
                return geocode
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("Paris", forecast_days=2, units="metric")

        assert "Clear sky" in result
        assert "Forecast:" in result
        assert "2024-06-17" in result
        assert "2024-06-18" in result

    @pytest.mark.asyncio
    async def test_lat_lon_input(self, tool):
        """Open-Meteo: location given as lat,lon skips geocoding."""
        weather = {
            "current": {
                "temperature_2m": -5.0,
                "relative_humidity_2m": 80,
                "apparent_temperature": -8.0,
                "weather_code": 71,
                "wind_speed_10m": 20.0,
            },
            "daily": {},
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            assert "geocoding-api" not in url
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("60.17,24.94")

        assert "-5.0°C" in result
        assert "Slight snow fall" in result

    @pytest.mark.asyncio
    async def test_imperial_units(self, tool):
        """Open-Meteo: imperial units."""
        geocode = {"results": [{"latitude": 40.71, "longitude": -74.01}]}
        weather = {
            "current": {
                "temperature_2m": 68.0,
                "relative_humidity_2m": 45,
                "apparent_temperature": 70.0,
                "weather_code": 0,
                "wind_speed_10m": 5.0,
            },
            "daily": {},
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            if "geocoding-api" in url:
                return geocode
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("New York", units="imperial")

        assert "68.0°F" in result
        assert "mph" in result

    @pytest.mark.asyncio
    async def test_geocode_failure(self, tool):
        """Open-Meteo: unknown location raises RuntimeError."""
        async def _fake_http_json(url: str) -> Dict[str, Any]:
            return {"results": []}

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            with pytest.raises(RuntimeError, match="Could not geocode"):
                await tool.execute("Xyzabc123")


class TestOpenWeatherMap:
    @pytest.mark.asyncio
    async def test_success_with_key(self, tool):
        """OpenWeatherMap: current + forecast with API key."""
        current = {
            "main": {"temp": 18.5, "feels_like": 17.0, "humidity": 60},
            "weather": [{"main": "Clouds", "description": "overcast clouds"}],
            "wind": {"speed": 3.5},
        }
        forecast = {
            "list": [
                {"dt_txt": "2024-06-17 09:00:00", "main": {"temp": 19}, "weather": [{"description": "light rain"}]},
                {"dt_txt": "2024-06-17 12:00:00", "main": {"temp": 21}, "weather": [{"description": "cloudy"}]},
            ],
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            if "forecast" in url:
                return forecast
            return current

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute(
                "London",
                forecast_days=1,
                context={"tool_config": {"provider": "openweathermap", "api_key": "test-key"}},
            )

        assert "18.5°C" in result
        assert "overcast clouds" in result
        assert "Forecast" in result
        assert "light rain" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, tool):
        """OpenWeatherMap: missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="OpenWeatherMap requires an API key"):
            await tool.execute(
                "London",
                context={"tool_config": {"provider": "openweathermap"}},
            )

    @pytest.mark.asyncio
    async def test_imperial_units(self, tool):
        """OpenWeatherMap: imperial units."""
        current = {
            "main": {"temp": 86.0, "feels_like": 88.0, "humidity": 40},
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "wind": {"speed": 10.0},
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            return current

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute(
                "Dallas",
                context={"tool_config": {"provider": "openweathermap", "api_key": "k"}},
                units="imperial",
            )

        assert "86.0°F" in result
        assert "mph" in result


class TestWeatherAPI:
    @pytest.mark.asyncio
    async def test_success_with_key(self, tool):
        """WeatherAPI: current + forecast with API key."""
        data = {
            "location": {"name": "Tokyo"},
            "current": {
                "temp_c": 25.0,
                "feelslike_c": 27.0,
                "humidity": 70,
                "wind_kph": 15.0,
                "condition": {"text": "Partly cloudy"},
            },
            "forecast": {
                "forecastday": [
                    {"date": "2024-06-16", "day": {"maxtemp_c": 28, "mintemp_c": 20, "condition": {"text": "Sunny"}}},
                    {"date": "2024-06-17", "day": {"maxtemp_c": 26, "mintemp_c": 19, "condition": {"text": "Rainy"}}},
                ],
            },
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            return data

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute(
                "Tokyo",
                forecast_days=1,
                context={"tool_config": {"provider": "weatherapi", "api_key": "test-key"}},
            )

        assert "25.0°C" in result
        assert "Partly cloudy" in result
        assert "Rainy" in result
        assert "2024-06-17" in result

    @pytest.mark.asyncio
    async def test_missing_api_key(self, tool):
        """WeatherAPI: missing API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="WeatherAPI requires an API key"):
            await tool.execute(
                "Tokyo",
                context={"tool_config": {"provider": "weatherapi"}},
            )

    @pytest.mark.asyncio
    async def test_imperial_units(self, tool):
        """WeatherAPI: imperial units."""
        data = {
            "location": {"name": "Miami"},
            "current": {
                "temp_f": 90.0,
                "feelslike_f": 95.0,
                "humidity": 85,
                "wind_mph": 8.0,
                "condition": {"text": "Hot"},
            },
            "forecast": {"forecastday": []},
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            return data

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute(
                "Miami",
                context={"tool_config": {"provider": "weatherapi", "api_key": "k"}},
                units="imperial",
            )

        assert "90.0°F" in result
        assert "mph" in result


class TestToolConfig:
    @pytest.mark.asyncio
    async def test_default_provider_is_openmeteo(self, tool):
        """Default provider (no context) falls back to open-meteo."""
        geocode = {"results": [{"latitude": 59.91, "longitude": 10.75}]}
        weather = {
            "current": {
                "temperature_2m": 10.0,
                "relative_humidity_2m": 50,
                "apparent_temperature": 9.0,
                "weather_code": 3,
                "wind_speed_10m": 5.0,
            },
            "daily": {},
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            if "geocoding-api" in url:
                return geocode
            assert "api.open-meteo.com" in url
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("Oslo")
            assert "Overcast" in result

    @pytest.mark.asyncio
    async def test_forecast_days_clamped(self, tool):
        """forecast_days > 7 is clamped to 7."""
        geocode = {"results": [{"latitude": 0.0, "longitude": 0.0}]}
        weather = {
            "current": {
                "temperature_2m": 20.0,
                "relative_humidity_2m": 50,
                "apparent_temperature": 20.0,
                "weather_code": 0,
                "wind_speed_10m": 5.0,
            },
            "daily": {
                "time": ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"],
                "temperature_2m_max": [1]*9,
                "temperature_2m_min": [1]*9,
                "weather_code": [0]*9,
            },
        }

        async def _fake_http_json(url: str) -> Dict[str, Any]:
            if "geocoding-api" in url:
                return geocode
            # Check that forecast_days=8 is passed in URL
            assert "forecast_days=8" in url
            return weather

        with patch.object(tool, "_http_json", side_effect=_fake_http_json):
            result = await tool.execute("City", forecast_days=99)
            assert "Forecast:" in result
            # Should not crash; clamping happens inside execute


class TestToolSchema:
    def test_schema_structure(self, tool):
        schema = tool.tool_schema
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "weather_lookup"
        params = func["parameters"]
        assert "location" in params["properties"]
        assert params["required"] == ["location"]


class TestDiscovery:
    def test_in_builtin_discovery(self):
        from agent.tools.builtin_tools import discover_builtin_tool_classes
        classes = discover_builtin_tool_classes()
        assert "weather_lookup" in classes
        assert classes["weather_lookup"] is WeatherLookupTool
