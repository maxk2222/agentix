"""Tests for the branding endpoints (public GET /branding + admin POST)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException, Response

from app.routers.branding import (
    BrandingPayload,
    _sanitize_branding,
    _sanitize_theme,
    branding_get,
    branding_save,
)


class FakeBrandingStore:
    def __init__(self, branding: Optional[Dict[str, Any]] = None) -> None:
        self._branding = branding
        self.saved: Optional[Dict[str, Any]] = None
        self.saved_by: Optional[str] = None

    async def get(self):
        return self._branding

    async def save(self, config: Dict[str, Any], created_by: str = "admin") -> None:
        self.saved = config
        self.saved_by = created_by


def _make_request(
    *,
    store: Optional[FakeBrandingStore],
    cors_origins: str = "*",
    origin: Optional[str] = None,
):
    headers = {"origin": origin} if origin else {}
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                _branding_store=store,
                app_config=SimpleNamespace(widget_defaults_cors_origins=cors_origins),
            )
        ),
        headers=headers,
        cookies={},
    )


# ---------------------------------------------------------------------------
# Sanitize (pure functions)
# ---------------------------------------------------------------------------


def test_sanitize_theme_keeps_known_vars_and_colors():
    theme = _sanitize_theme(
        {
            "light": {"primary": "#123456", "bg": "rgba(1,2,3,0.5)", "evil": "#fff"},
            "dark": {"text": "white", "bogus_var": "red"},
        }
    )
    assert theme == {
        "light": {"primary": "#123456", "bg": "rgba(1,2,3,0.5)"},
        "dark": {"text": "white"},
    }


def test_sanitize_theme_rejects_non_color_values():
    theme = _sanitize_theme(
        {
            "light": {
                "primary": "url(javascript:alert(1))",
                "bg": "expression(alert(1))",
                "card": "#fff; position: fixed",
                "surface": "#aabbcc",
            }
        }
    )
    assert theme == {"light": {"surface": "#aabbcc"}}


def test_sanitize_theme_ignores_unknown_modes_and_non_dict():
    assert _sanitize_theme({"midnight": {"bg": "#000"}}) == {}
    assert _sanitize_theme("not-a-dict") == {}
    assert _sanitize_theme({"light": "nope"}) == {}


def test_sanitize_branding_drops_empty_theme_but_keeps_rest():
    cleaned = _sanitize_branding(
        {"app_name": "ACME", "theme": {"light": {"bogus": "x"}}, "logo_url": "https://x/l.png"}
    )
    assert cleaned == {"app_name": "ACME", "logo_url": "https://x/l.png"}
    assert "theme" not in cleaned


# ---------------------------------------------------------------------------
# GET /branding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_branding_publicly():
    store = FakeBrandingStore({"app_name": "ACME", "logo_url": "https://x/l.png"})
    request = _make_request(store=store)
    response = Response()
    result = await branding_get(request, response)
    assert result == {"app_name": "ACME", "logo_url": "https://x/l.png"}
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_get_empty_store_returns_empty_dict():
    store = FakeBrandingStore(None)
    request = _make_request(store=store)
    response = Response()
    result = await branding_get(request, response)
    assert result == {}


@pytest.mark.asyncio
async def test_get_cors_allowlist_echoes_matching_origin():
    store = FakeBrandingStore({"app_name": "T"})
    request = _make_request(
        store=store,
        cors_origins="https://a.example.com, https://b.example.com",
        origin="https://a.example.com",
    )
    response = Response()
    await branding_get(request, response)
    assert response.headers["access-control-allow-origin"] == "https://a.example.com"


@pytest.mark.asyncio
async def test_get_cors_allowlist_blocks_unknown_origin():
    store = FakeBrandingStore({"app_name": "T"})
    request = _make_request(
        store=store,
        cors_origins="https://a.example.com",
        origin="https://evil.example.com",
    )
    response = Response()
    await branding_get(request, response)
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_get_without_store_returns_503():
    request = _make_request(store=None)
    request.app.state._branding_store = None
    request.app.state.rag_application = None
    response = Response()
    with pytest.raises(HTTPException) as exc_info:
        await branding_get(request, response)
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# POST /admin/branding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_sanitizes_and_sets_created_by(monkeypatch):
    store = FakeBrandingStore()

    async def _noop_permission(request, permission):
        return None

    monkeypatch.setattr("app.routers.branding.require_permission", _noop_permission)
    monkeypatch.setattr(
        "app.routers.branding.read_session_cookie",
        lambda req, cfg: {"username": "alice", "role": "owner"},
    )

    request = _make_request(store=store)
    request.app.state.app_config = SimpleNamespace(dashboard_session_secret="s")

    payload = BrandingPayload(
        config={
            "app_name": "ACME",
            "theme": {"light": {"primary": "#112233", "hax": "red"}},
        }
    )
    result = await branding_save(request, payload)

    assert result == {"success": True}
    assert store.saved == {"app_name": "ACME", "theme": {"light": {"primary": "#112233"}}}
    assert store.saved_by == "alice"


@pytest.mark.asyncio
async def test_save_requires_permission(monkeypatch):
    store = FakeBrandingStore()

    async def _deny(request, permission):
        raise HTTPException(status_code=403, detail="Permission denied")

    monkeypatch.setattr("app.routers.branding.require_permission", _deny)

    request = _make_request(store=store)
    payload = BrandingPayload(config={"app_name": "ACME"})

    with pytest.raises(HTTPException) as exc_info:
        await branding_save(request, payload)
    assert exc_info.value.status_code == 403
    assert store.saved is None


@pytest.mark.asyncio
async def test_save_rejects_oversized_payload(monkeypatch):
    store = FakeBrandingStore()

    async def _noop_permission(request, permission):
        return None

    monkeypatch.setattr("app.routers.branding.require_permission", _noop_permission)
    monkeypatch.setattr("app.routers.branding.read_session_cookie", lambda req, cfg: None)

    request = _make_request(store=store)
    request.app.state.app_config = SimpleNamespace(dashboard_session_secret="s")

    # 3MB+ payload — a data-URL logo would realistically be ~1-2MB; this is abuse.
    payload = BrandingPayload(config={"logo_url": "data:image/png;base64," + "A" * (3 * 1024 * 1024)})
    with pytest.raises(HTTPException) as exc_info:
        await branding_save(request, payload)
    assert exc_info.value.status_code == 413
    assert store.saved is None
