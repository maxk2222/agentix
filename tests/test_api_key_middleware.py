"""API-key middleware dashboard-session exceptions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import create_session_cookie
from app.api_keys import AppApiKey, AppApiKeyStore
from app.middleware import APIKeyMiddleware


def _app_with_api_key_required() -> FastAPI:
    app = FastAPI()
    app.state.app_config = SimpleNamespace(
        dashboard_session_secret="test-dashboard-secret",
        dashboard_session_ttl_seconds=3600,
        api_keys_enabled=True,
    )
    key_store = AppApiKeyStore(None)
    key_store.list_keys = AsyncMock(
        return_value=[
            AppApiKey(
                id="key-1",
                name="configured",
                api_key_hash="unused",
            )
        ]
    )
    app.state._api_key_store = key_store
    # A legacy key makes API-key enforcement active even without a store.
    app.add_middleware(APIKeyMiddleware, legacy_key="configured-key")

    @app.post("/v1/chat/completions")
    async def chat():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"ok": True}

    @app.get("/branding")
    async def branding():
        return {"ok": True}

    @app.get("/public-chat/config")
    async def public_chat_config():
        return {"ok": True}

    return app


def test_owner_dashboard_session_can_use_chat_without_api_key():
    app = _app_with_api_key_required()
    cookie = create_session_cookie(
        app.state.app_config,
        {"id": "owner-1", "username": "owner", "role": "owner"},
    )

    with TestClient(app) as client:
        client.cookies.set("dashboard_session", cookie)
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_owner_dashboard_session_does_not_bypass_other_api_key_routes():
    app = _app_with_api_key_required()
    cookie = create_session_cookie(
        app.state.app_config,
        {"id": "owner-1", "username": "owner", "role": "owner"},
    )

    with TestClient(app) as client:
        client.cookies.set("dashboard_session", cookie)
        response = client.get("/v1/models")

    assert response.status_code == 401


def test_public_surfaces_skip_api_key_even_when_enforced():
    """/branding and /public-chat are public by design — the dashboard login
    page and embedded widgets read them without any credentials."""
    app = _app_with_api_key_required()

    with TestClient(app) as client:
        assert client.get("/branding").status_code == 200
        assert client.get("/public-chat/config").status_code == 200
        # Sanity: a protected route still rejects without a key.
        assert client.get("/v1/models").status_code == 401
