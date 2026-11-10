"""Tests for app.routers.public_chat."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from app.routers.public_chat import (
    _resolve_enabled_assistants,
    public_chat_config_get,
    public_chat_config_save,
)


class FakePublicChatConfigStore:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config
        self.saved: list[dict] = []

    async def get(self):
        return self._config

    async def save(self, config: dict, created_by: str = "admin") -> None:
        self.saved.append(config)


@pytest.mark.asyncio
async def test_public_chat_config_get_returns_empty_when_no_config(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore(None),
    )
    response = Response()
    result = await public_chat_config_get(request, response)
    # Even without stored config the injected capability keys are present —
    # the response must not short-circuit before branding/capability merges.
    assert result == {
        "enabled_assistants": [],
        "assistant_descriptions": {},
        "assistant_memory_enabled": {},
        "assistant_workspace_enabled": {},
    }


@pytest.mark.asyncio
async def test_public_chat_config_get_returns_config(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore({"enabled": True, "title": "Test"}),
    )
    response = Response()
    result = await public_chat_config_get(request, response)
    assert result == {
        "enabled": True,
        "title": "Test",
        "enabled_assistants": [],
        "assistant_descriptions": {},
        "assistant_memory_enabled": {},
        "assistant_workspace_enabled": {},
    }
    assert response.headers["cache-control"] == "private, no-store, no-cache, must-revalidate"
    assert response.headers["vary"] == "Cookie"


@pytest.mark.asyncio
async def test_public_chat_config_filters_deleted_assistants_from_role_mapping(monkeypatch):
    """Stale role rows must not re-expose assistants removed from instructions."""
    class FakeRoleStore:
        async def get_assistants_for_role(self, role: str):
            assert role == "standard"
            return ["DeletedAssistant", "AvailableAssistant", "AvailableAssistant"]

    async def _user(_request):
        return {"id": "user-1", "role": "standard", "dashboard_user_id": None}

    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda _req: FakePublicChatConfigStore({"enabled": True}),
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_current_chat_user", _user
    )
    monkeypatch.setattr(
        "app.chat_group_access.get_role_store",
        lambda _req: FakeRoleStore(),
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_all_assistant_names",
        lambda _req: ["AvailableAssistant"],
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_assistant_descriptions",
        lambda _req: {
            "DeletedAssistant": "stale description",
            "AvailableAssistant": "current description",
        },
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_assistant_memory_flags",
        lambda _req: {"AvailableAssistant": True},
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(app_config=None)),
        cookies={},
        headers={},
        query_params={},
        state=SimpleNamespace(api_key_id=None),
    )

    result = await public_chat_config_get(request, Response())

    assert result["enabled_assistants"] == ["AvailableAssistant"]
    assert result["assistant_descriptions"] == {
        "AvailableAssistant": "current description"
    }
    assert result["assistant_memory_enabled"] == {"AvailableAssistant": True}


@pytest.mark.asyncio
async def test_public_chat_stale_only_role_mapping_does_not_fall_back_to_all(monkeypatch):
    """A configured role with only deleted names must not gain broad access."""
    class FakeRoleStore:
        async def get_assistants_for_role(self, _role: str):
            return ["DeletedAssistant"]

    monkeypatch.setattr(
        "app.chat_group_access.get_role_store",
        lambda _req: FakeRoleStore(),
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_all_assistant_names",
        lambda _req: ["CurrentAssistant"],
    )
    result = await _resolve_enabled_assistants(
        SimpleNamespace(), {"enabled": True},
        {"id": "user-1", "role": "standard"},
    )
    assert result == []


@pytest.mark.asyncio
async def test_public_chat_config_issues_scoped_token_when_enabled(monkeypatch):
    """enabled=true + session secret → an access_token is issued and validates."""
    app_config = SimpleNamespace(dashboard_session_secret="test-secret")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(app_config=app_config)),
        cookies={},
        headers={},
        query_params={},
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore({"enabled": True, "title": "Test"}),
    )
    result = await public_chat_config_get(request, Response())
    token = result.get("access_token")
    assert isinstance(token, str) and token

    from app.routers.public_chat_auth import _read_public_chat_access_token

    request.headers["x-public-chat-token"] = token
    assert _read_public_chat_access_token(request, app_config)["role"] == "anonymous"
    # Same token via ?pct= query param (artifact <a href> URLs)
    request.headers.pop("x-public-chat-token")
    request.query_params["pct"] = token
    assert _read_public_chat_access_token(request, app_config)["role"] == "anonymous"


@pytest.mark.asyncio
async def test_public_chat_config_no_token_when_disabled(monkeypatch):
    """enabled=false → no token is issued at all."""
    app_config = SimpleNamespace(dashboard_session_secret="test-secret")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(app_config=app_config)),
        cookies={},
        headers={},
        query_params={},
    )
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore({"enabled": False, "title": "Test"}),
    )
    result = await public_chat_config_get(request, Response())
    assert "access_token" not in result


@pytest.mark.asyncio
async def test_public_chat_token_role_roundtrip_and_tamper(monkeypatch):
    """The token carries the caller's role; forged tokens are rejected."""
    from app.routers.public_chat_auth import (
        _create_public_chat_access_token,
        _read_public_chat_access_token,
    )

    app_config = SimpleNamespace(dashboard_session_secret="test-secret")
    token = _create_public_chat_access_token(app_config, "standard")
    request = SimpleNamespace(
        headers={"x-public-chat-token": token},
        query_params={},
    )
    assert _read_public_chat_access_token(request, app_config)["role"] == "standard"

    request.headers["x-public-chat-token"] = token[:-4] + "AAAA"
    assert _read_public_chat_access_token(request, app_config) is None

    # Different secret → invalid signature
    other = SimpleNamespace(dashboard_session_secret="other-secret")
    request.headers["x-public-chat-token"] = token
    assert _read_public_chat_access_token(request, other) is None


@pytest.mark.asyncio
async def test_public_chat_config_save_admin_only(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    store = FakePublicChatConfigStore()
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: store,
    )

    # Simulate non-admin by raising from require_admin
    def fake_require_admin(req):
        raise HTTPException(status_code=403, detail="Admin required")

    monkeypatch.setattr("app.routers.public_chat.require_admin", fake_require_admin)

    from pydantic import BaseModel

    class FakePayload(BaseModel):
        config: dict

    payload = FakePayload(config={"enabled": True})
    with pytest.raises(HTTPException) as exc_info:
        await public_chat_config_save(request, payload)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_public_chat_config_save_success(monkeypatch):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace()),
    )
    store = FakePublicChatConfigStore()
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: store,
    )

    # Simulate admin
    def fake_require_admin(req):
        pass

    monkeypatch.setattr("app.routers.public_chat.require_admin", fake_require_admin)

    from pydantic import BaseModel

    class FakePayload(BaseModel):
        config: dict

    payload = FakePayload(config={"enabled": True, "enabled_assistants": ["A"]})
    result = await public_chat_config_save(request, payload)
    assert result == {"success": True}
    assert len(store.saved) == 1
    assert store.saved[0]["enabled"] is True


class FakeConvStore:
    def __init__(self, conv: dict | None = None) -> None:
        self._conv = conv

    async def get_by_conversation_id(self, conversation_id: str):
        return self._conv


@pytest.mark.asyncio
async def test_public_chat_memory_requires_auth(monkeypatch):
    from app.routers.public_chat import public_chat_conversation_memory

    async def no_user(req):
        return None

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", no_user)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as exc_info:
        await public_chat_conversation_memory(request, "conv-1")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_public_chat_memory_denies_foreign_conversation(monkeypatch):
    from app.routers.public_chat import public_chat_conversation_memory

    async def user(req):
        return {"id": "chat-user-1", "dashboard_user_id": None}

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", user)
    monkeypatch.setattr(
        "app.routers.public_chat._get_chat_conversations_store",
        lambda req: FakeConvStore({"user_id": "chat-user-other"}),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as exc_info:
        await public_chat_conversation_memory(request, "conv-1")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_public_chat_memory_returns_entries(monkeypatch):
    from app.routers.public_chat import public_chat_conversation_memory
    from app.stores import AgentMemoryStore

    async def user(req):
        return {"id": "chat-user-1", "dashboard_user_id": None}

    async def fake_list(self, conversation_id):
        return [{"assistant_name": "A", "memory": "prefers German", "updated_at": "x"}]

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", user)
    monkeypatch.setattr(
        "app.routers.public_chat._get_chat_conversations_store",
        lambda req: FakeConvStore({"user_id": "chat-user-1"}),
    )
    monkeypatch.setattr(AgentMemoryStore, "list_memories_for_conversation", fake_list)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(rag_application=SimpleNamespace(engine=object())))
    )
    result = await public_chat_conversation_memory(request, "conv-1")
    assert result["conversation_id"] == "conv-1"
    assert result["memories"][0]["memory"] == "prefers German"


@pytest.mark.asyncio
async def test_public_chat_config_uploads_per_caller(monkeypatch):
    """uploads.files is caller-aware: anonymous visitors never get it (login
    required for disk writes), authenticated users get it when enabled."""
    app_config = SimpleNamespace(
        dashboard_session_secret="test-secret",
        chat_image_upload_enabled=True,
        chat_file_upload_enabled=True,
    )

    async def _no_user(req):
        return None

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", _no_user)
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore({"enabled": True}),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(app_config=app_config)),
        cookies={},
        headers={},
        query_params={},
        state=SimpleNamespace(api_key_id=None),
    )
    result = await public_chat_config_get(request, Response())
    assert result["uploads"]["images"] is True
    assert result["uploads"]["files"] is False  # anonymous → login required

    async def _user(req):
        return {"id": "chat-user-1", "role": "standard", "dashboard_user_id": None}

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", _user)
    result = await public_chat_config_get(request, Response())
    assert result["uploads"]["files"] is True


@pytest.mark.asyncio
async def test_public_chat_config_uploads_without_session_secret(monkeypatch):
    """uploads must be present even when no dashboard_session_secret is
    configured — the secret only gates access_token signing, not the
    capability flags. Otherwise the upload UI never appears."""
    app_config = SimpleNamespace(
        dashboard_session_secret=None,
        chat_image_upload_enabled=True,
        chat_file_upload_enabled=True,
    )

    async def _user(req):
        return {"id": "chat-user-1", "role": "standard", "dashboard_user_id": None}

    monkeypatch.setattr("app.routers.public_chat._get_current_chat_user", _user)
    monkeypatch.setattr(
        "app.routers.public_chat._get_public_chat_store",
        lambda req: FakePublicChatConfigStore({"enabled": True}),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(app_config=app_config)),
        cookies={},
        headers={},
        query_params={},
        state=SimpleNamespace(api_key_id=None),
    )
    result = await public_chat_config_get(request, Response())
    assert "access_token" not in result
    assert result["uploads"]["images"] is True
    assert result["uploads"]["files"] is True
