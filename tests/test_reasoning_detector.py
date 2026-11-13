"""Tests for app.reasoning_detector — live reasoning-capability probes."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.reasoning_detector import (
    _parse_supported_levels,
    detect_reasoning_support,
    ensure_detected_for_provider,
)


class FakeResponse:
    def __init__(self, data: Dict[str, Any] | None = None, status: int = 200, text: str = "") -> None:
        self._data = data or {"choices": [{"message": {"content": "ok"}}]}
        self.status_code = status
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "http://test/v1/chat/completions"),
                response=httpx.Response(self.status_code, text=self.text),
            )

    def json(self) -> Dict[str, Any]:
        return self._data


def _mock_client(responses: list[FakeResponse]):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=responses)
    cls = MagicMock(return_value=mock_client)
    return cls, mock_client


@pytest.mark.asyncio
async def test_detect_effort_and_toggle_supported() -> None:
    """200 on all probes → both capabilities True, levels empty."""
    # probe1 accepted silently → level-discovery probe (unhelpful error) → toggle
    cls, mock_client = _mock_client([
        FakeResponse(),
        FakeResponse(status=400, text="invalid request"),
        FakeResponse(),
    ])
    with patch("httpx.AsyncClient", cls):
        result = await detect_reasoning_support(base_url="http://test", model="m")

    assert result["supports_effort"] is True
    assert result["supports_thinking_toggle"] is True
    assert result["effort_levels"] == []
    assert result["error"] is None
    assert mock_client.post.await_count == 3


@pytest.mark.asyncio
async def test_detect_silent_accept_discovers_levels() -> None:
    """200 on effort → invalid-level probe reveals the supported set."""
    cls, mock_client = _mock_client([
        FakeResponse(),
        FakeResponse(
            status=400,
            text="Invalid reasoning effort. Supported types are xhigh, medium, low.",
        ),
        FakeResponse(),
    ])
    with patch("httpx.AsyncClient", cls):
        result = await detect_reasoning_support(base_url="http://test", model="m")

    assert result["supports_effort"] is True
    assert result["effort_levels"] == ["low", "medium", "xhigh"]
    assert result["supports_thinking_toggle"] is True
    assert mock_client.post.await_count == 3


@pytest.mark.asyncio
async def test_detect_value_reject_reveals_levels() -> None:
    """vLLM 400 'Supported types are…' → effort supported, levels parsed."""
    body = (
        "Unexpected reasoning effort medium. Supported types are "
        "xhigh (default), medium, and low."
    )
    cls, _ = _mock_client([
        FakeResponse(status=400, text=body),
        FakeResponse(),
    ])
    with patch("httpx.AsyncClient", cls):
        result = await detect_reasoning_support(base_url="http://test", model="qwen3.8")

    assert result["supports_effort"] is True
    assert result["effort_levels"] == ["low", "medium", "xhigh"]
    assert result["supports_thinking_toggle"] is True


@pytest.mark.asyncio
async def test_detect_param_rejected() -> None:
    """400 'unrecognized parameter' → effort unsupported, probe 2 still runs."""
    cls, mock_client = _mock_client([
        FakeResponse(status=400, text="Unrecognized parameter reasoning_effort"),
        FakeResponse(status=400, text="unrecognized parameter chat_template_kwargs"),
    ])
    with patch("httpx.AsyncClient", cls):
        result = await detect_reasoning_support(base_url="http://test", model="m")

    assert result["supports_effort"] is False
    assert result["supports_thinking_toggle"] is False
    assert mock_client.post.await_count == 2


@pytest.mark.asyncio
async def test_detect_network_error_inconclusive() -> None:
    """Transport error → inconclusive, second probe skipped, error set."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
    cls = MagicMock(return_value=mock_client)

    with patch("httpx.AsyncClient", cls):
        result = await detect_reasoning_support(base_url="http://test", model="m")

    assert result["supports_effort"] is None
    assert result["supports_thinking_toggle"] is None
    assert result["error"]
    assert mock_client.post.await_count == 1


def test_parse_supported_levels() -> None:
    text = "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
    assert _parse_supported_levels(text) == ["low", "medium", "xhigh"]
    assert _parse_supported_levels("unrecognized parameter reasoning_effort") == []
    # word-boundary: 'low' inside 'below' must not match
    assert "low" not in _parse_supported_levels("Supported types are below thresholds")


@pytest.mark.asyncio
async def test_ensure_detected_cache_and_schedule() -> None:
    """Cache hit returns status; miss returns None and schedules a probe."""

    class FakeProvider:
        id = "p1"
        base_url = "http://test"
        timeout_seconds = 30
        enabled = True

    class FakeStore:
        def __init__(self) -> None:
            self.rows: Dict[str, Optional[Dict[str, Any]]] = {}

        async def get_provider(self, provider_id: str):
            return FakeProvider()

        async def get_model_reasoning_status(self, provider_id: str, model_id: str):
            return self.rows.get(f"{provider_id}:{model_id}")

        async def update_model_reasoning_status(
            self, provider_id: str, model_id: str, capabilities, error: Optional[str] = None
        ) -> None:
            self.rows[f"{provider_id}:{model_id}"] = {"capabilities": capabilities, "check_error": error}

        async def get_api_key(self, provider_id: str):
            return None

    store = FakeStore()
    store.rows["p1:model-a"] = {"capabilities": {"supports_effort": True}, "check_error": None}

    hit = await ensure_detected_for_provider(store, "p1", "model-a")
    assert hit == {"capabilities": {"supports_effort": True}, "check_error": None}

    # Cache miss → returns None and schedules the probe; patch detection so no
    # real HTTP happens, then let the event loop run the task.
    with patch(
        "app.reasoning_detector.detect_reasoning_support",
        new=AsyncMock(
            return_value={
                "supports_effort": True,
                "effort_levels": ["low", "medium"],
                "supports_thinking_toggle": False,
                "error": None,
            }
        ),
    ):
        miss = await ensure_detected_for_provider(store, "p1", "model-b")
        assert miss is None
        await asyncio.sleep(0)  # let the scheduled probe task run

    cached = store.rows.get("p1:model-b")
    assert cached is not None
    assert cached["capabilities"]["supports_effort"] is True
    assert cached["check_error"] is None
