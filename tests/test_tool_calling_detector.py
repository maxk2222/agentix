"""Tests for app.tool_calling_detector."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.tool_calling_detector import detect_tool_calling_support


class FakeResponse:
    def __init__(self, data: Dict[str, Any], status: int = 200) -> None:
        self._data = data
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> Dict[str, Any]:
        return self._data


@pytest.mark.asyncio
async def test_detect_native_tool_calls() -> None:
    """Model returns non-empty tool_calls → supports_native=True."""
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Berlin"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=FakeResponse(response_data))
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="key", model="gpt-4"
        )

    assert result["supports_native"] is True
    assert result["used_content_json"] is False
    assert result["error"] is None


@pytest.mark.asyncio
async def test_detect_parsed_tool_calls() -> None:
    """Model returns JSON in content but empty tool_calls → supports_native=False."""
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"location": "Berlin"}',
                    "tool_calls": [],
                }
            }
        ]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=FakeResponse(response_data))
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="key", model="gpt-oss"
        )

    assert result["supports_native"] is False
    assert result["used_content_json"] is True
    assert result["error"] is None


@pytest.mark.asyncio
async def test_detect_http_error() -> None:
    """Provider returns HTTP error → error field populated."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            return_value=FakeResponse({"error": "unauthorized"}, status=401)
        )
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="bad", model="m"
        )

    assert result["supports_native"] is None
    assert result["error"] is not None
    assert "401" in result["error"]


@pytest.mark.asyncio
async def test_detect_jinja_template_error() -> None:
    """Provider returns HTTP 500 with Jinja template error → supports_native=False."""
    error_body = "Error rendering prompt with jinja template: got UndefinedValue"

    class _JinjaErrorResponse:
        status_code = 500
        text = error_body

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "http://test/v1/chat/completions"),
                response=httpx.Response(500, text=error_body),
            )

        def json(self) -> Dict[str, Any]:
            return {}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_JinjaErrorResponse())
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="key", model="gemma4"
        )

    assert result["supports_native"] is False
    assert result["error"] is not None
    assert "500" in result["error"]


@pytest.mark.asyncio
async def test_detect_content_json_array() -> None:
    """Model returns JSON array in content → supports_native=False, used_content_json=True."""
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '[{"get_weather": {"location": "Berlin"}}]',
                    "tool_calls": [],
                }
            }
        ]
    }
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=FakeResponse(response_data))
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="key", model="gpt-oss"
        )

    assert result["supports_native"] is False
    assert result["used_content_json"] is True


@pytest.mark.asyncio
async def test_detect_content_json_with_fences() -> None:
    """Model returns fenced JSON in content → supports_native=False, used_content_json=True."""
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"get_weather": {"location": "Berlin"}}\n```',
                    "tool_calls": [],
                }
            }
        ]
    }
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=FakeResponse(response_data))
        mock_client_cls.return_value = mock_client

        result = await detect_tool_calling_support(
            base_url="http://test", api_key="key", model="gpt-oss"
        )

    assert result["supports_native"] is False
    assert result["used_content_json"] is True


@pytest.mark.asyncio
async def test_ensure_detected_uses_per_model_cache() -> None:
    """ensure_detected_for_provider should use per-model cache, not provider-wide."""
    from app.tool_calling_detector import ensure_detected_for_provider

    class FakeProvider:
        def __init__(self) -> None:
            self.id = "p1"
            self.base_url = "http://test"
            self.timeout_seconds = 30
            self.supports_native_tool_calls = None
            self.enabled = True

    class FakeStore:
        def __init__(self) -> None:
            self._per_model: Dict[str, Optional[bool]] = {}
            self.provider = FakeProvider()

        async def get_provider(self, provider_id: str):
            return self.provider

        async def get_model_tool_calling_status(self, provider_id: str, model_id: str) -> Optional[bool]:
            return self._per_model.get(f"{provider_id}:{model_id}")

        async def update_model_tool_calling_status(
            self, provider_id: str, model_id: str, supports_native: Optional[bool], error: Optional[str] = None
        ) -> None:
            self._per_model[f"{provider_id}:{model_id}"] = supports_native

        async def get_api_key(self, provider_id: str) -> Optional[str]:
            return None

    store = FakeStore()
    # Pre-populate per-model cache for model-a
    store._per_model["p1:model-a"] = True

    result_a = await ensure_detected_for_provider(store, "p1", "model-a")
    assert result_a is True

    # model-b has no per-model cache entry → returns None (triggers background detection)
    result_b = await ensure_detected_for_provider(store, "p1", "model-b")
    assert result_b is None
