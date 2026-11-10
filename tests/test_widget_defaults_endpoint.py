"""Tests for the widget defaults endpoints (public GET + admin POST)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from fastapi import Response

from app.routers.admin import WidgetDefaultsPayload, widget_defaults_save, _sanitize_widget_defaults
from app.routers.widget import widget_defaults_get


class FakeWidgetDefaultsStore:
    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._defaults = defaults
        self.saved: Optional[Dict[str, Any]] = None
        self.saved_by: Optional[str] = None

    async def get(self):
        return self._defaults

    async def save(self, config: Dict[str, Any], created_by: str = "admin") -> None:
        self.saved = config
        self.saved_by = created_by


def _make_request(
    *,
    store: Optional[FakeWidgetDefaultsStore],
    cors_origins: str = "*",
    origin: Optional[str] = None,
):
    headers = {"origin": origin} if origin else {}
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                _widget_defaults_store=store,
                app_config=SimpleNamespace(widget_defaults_cors_origins=cors_origins),
            )
        ),
        headers=headers,
        cookies={},
    )


# ---------------------------------------------------------------------------
# Sanitize (pure function)
# ---------------------------------------------------------------------------


def test_sanitize_strips_secrets_and_callbacks():
    config = {
        "apiKey": "sk-secret",
        "apiSecret": "secret",
        "onOpen": "function(){}",
        "onInit": "function(){}",
        "onError": "function(){}",
        "title": "Bot",
        "language": "auto",
        "colors": {"primary": "#fff"},
    }
    cleaned = _sanitize_widget_defaults(config)
    assert cleaned == {"title": "Bot", "colors": {"primary": "#fff"}}


def test_sanitize_keeps_explicit_language():
    cleaned = _sanitize_widget_defaults({"language": "de", "title": "T"})
    assert cleaned["language"] == "de"


def test_sanitize_does_not_mutate_input():
    config = {"apiKey": "sk", "title": "T"}
    _sanitize_widget_defaults(config)
    assert config == {"apiKey": "sk", "title": "T"}


# ---------------------------------------------------------------------------
# GET /widget-defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_defaults_without_secrets():
    store = FakeWidgetDefaultsStore({"title": "Bot", "apiKey": "sk-leak", "apiSecret": "x"})
    request = _make_request(store=store)
    response = Response()
    result = await widget_defaults_get(request, response)
    assert result == {"title": "Bot"}
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_get_empty_store_returns_empty_dict():
    store = FakeWidgetDefaultsStore(None)
    request = _make_request(store=store)
    response = Response()
    result = await widget_defaults_get(request, response)
    assert result == {}


@pytest.mark.asyncio
async def test_get_cors_allowlist_echoes_matching_origin():
    store = FakeWidgetDefaultsStore({"title": "T"})
    request = _make_request(
        store=store,
        cors_origins="https://a.example.com, https://b.example.com",
        origin="https://a.example.com",
    )
    response = Response()
    await widget_defaults_get(request, response)
    assert response.headers["access-control-allow-origin"] == "https://a.example.com"


@pytest.mark.asyncio
async def test_get_cors_allowlist_blocks_unknown_origin():
    store = FakeWidgetDefaultsStore({"title": "T"})
    request = _make_request(
        store=store,
        cors_origins="https://a.example.com",
        origin="https://evil.example.com",
    )
    response = Response()
    await widget_defaults_get(request, response)
    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# POST /admin/widget-defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_strips_apikey_and_sets_created_by(monkeypatch):
    store = FakeWidgetDefaultsStore()

    async def _noop_permission(request, permission):
        return None

    monkeypatch.setattr("app.routers.admin.require_permission", _noop_permission)
    monkeypatch.setattr(
        "app.routers.admin.read_session_cookie",
        lambda req, cfg: {"username": "alice", "role": "owner"},
    )

    request = _make_request(store=store)
    request.app.state.app_config = SimpleNamespace(dashboard_session_secret="s")

    payload = WidgetDefaultsPayload(config={"title": "Bot", "apiKey": "sk-x", "language": "auto"})
    result = await widget_defaults_save(request, payload)

    assert result == {"success": True}
    assert store.saved == {"title": "Bot"}
    assert store.saved_by == "alice"


@pytest.mark.asyncio
async def test_save_requires_permission(monkeypatch):
    store = FakeWidgetDefaultsStore()

    async def _deny(request, permission):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Permission denied")

    monkeypatch.setattr("app.routers.admin.require_permission", _deny)

    request = _make_request(store=store)
    payload = WidgetDefaultsPayload(config={"title": "Bot"})

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await widget_defaults_save(request, payload)
    assert exc_info.value.status_code == 403
    assert store.saved is None
