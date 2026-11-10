from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent.config import AppConfig
from app.routers.admin import (
    admin_login,
    admin_logout,
    admin_me,
    conversation_detail,
    conversation_metadata,
    conversations_export,
    conversations_list,
    conversations_models,
    conversations_users,
    list_api_keys,
    metrics_summary,
    put_runtime_config,
    vectorstore_summary,
    vectorstore_by_model,
    vectorstore_history,
    vectorstore_state,
)


class FakeDashboardUserStore:
    """Minimal fake store that authenticates against env credentials."""

    def __init__(self, config) -> None:
        self._config = config
        self._users: dict[str, Any] = {}

    async def authenticate(self, username: str, password: str) -> Any | None:
        pwd = self._config.dashboard_admin_password or "admin"
        if username == self._config.dashboard_admin_username and password == pwd:
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin",
                username=username,
                display_name="Environment Admin",
                password_hash="",
                role="admin",
                is_active=True,
            )
        return None

    async def get_by_username(self, username: str) -> Any | None:
        if username == self._config.dashboard_admin_username:
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin",
                username=username,
                display_name="Environment Admin",
                password_hash="",
                role="admin",
                is_active=True,
            )
        return None

    async def get_user(self, user_id: str) -> Any | None:
        if user_id == "env-admin":
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin",
                username=self._config.dashboard_admin_username,
                display_name="Environment Admin",
                password_hash="",
                role="admin",
                is_active=True,
            )
        return None

    async def record_login(self, user_id: str) -> None:
        pass

    async def list_users(self) -> list[Any]:
        return list(self._users.values())

    async def create_user(self, user: Any) -> None:
        self._users[user.id] = user

    async def update_user(self, user: Any) -> None:
        self._users[user.id] = user

    async def set_password(self, user_id: str, password: str) -> None:
        pass

    async def set_active(self, user_id: str, is_active: bool) -> None:
        pass

    async def bump_session_version(self, user_id: str) -> None:
        pass

    async def delete_user(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    async def count_active_admins(self) -> int:
        return 1

    async def count_active_owners(self) -> int:
        return 1


def _make_request(
    *,
    app_config: Any | None = None,
    cookies: dict[str, str] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Any:
    state: dict[str, Any] = {
        "app_config": app_config,
    }
    if app_config is not None:
        state["_dashboard_user_store"] = FakeDashboardUserStore(app_config)
    if extra_state:
        state.update(extra_state)
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(**state)
        ),
        cookies=cookies or {},
    )


def _enabled_config() -> Any:
    return SimpleNamespace(
        dashboard_enabled=True,
        dashboard_admin_username="admin",
        dashboard_admin_password="secret",
        dashboard_session_secret="super-secret-key-for-tests",
        dashboard_session_ttl_seconds=3600,
    )


@pytest.mark.asyncio
async def test_login_success():
    config = _enabled_config()
    request = _make_request(app_config=config)
    response = await admin_login(
        request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    assert response.status_code == 200
    assert "dashboard_session=" in response.headers["Set-Cookie"]


@pytest.mark.asyncio
async def test_login_wrong_password():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await admin_login(
            request,
            payload=SimpleNamespace(username="admin", password="wrong"),
        )
    assert "401" in str(exc_info.value) or "Invalid credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_login_dashboard_disabled():
    config = SimpleNamespace(
        dashboard_enabled=False,
        dashboard_admin_username="admin",
        dashboard_admin_password="secret",
        dashboard_session_secret="super-secret-key-for-tests",
        dashboard_session_ttl_seconds=3600,
    )
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await admin_login(
            request,
            payload=SimpleNamespace(username="admin", password="secret"),
        )
    assert "403" in str(exc_info.value)


@pytest.mark.asyncio
async def test_me_authenticated():
    config = _enabled_config()
    request = _make_request(app_config=config)
    # First login to get a valid cookie
    login_response = await admin_login(
        request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    cookie_header = login_response.headers["Set-Cookie"]
    token = cookie_header.split("dashboard_session=")[1].split(";")[0]
    # Now call me with the cookie
    request_with_cookie = _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
    )
    result = await admin_me(request_with_cookie)
    assert result.authenticated is True
    assert result.role == "admin"


class FakeDashboardUserStoreMissing:
    """Fake store that returns None so legacy fallback is triggered."""

    async def authenticate(self, username: str, password: str) -> Any | None:
        return None

    async def get_by_username(self, username: str) -> Any | None:
        return None

    async def get_user(self, user_id: str) -> Any | None:
        return None

    async def record_login(self, user_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_me_legacy_session_has_owner_role():
    """Legacy env-admin session without store user gets owner role and users:write."""
    config = _enabled_config()
    request = _make_request(
        app_config=config,
        extra_state={"_dashboard_user_store": FakeDashboardUserStoreMissing()},
    )
    login_response = await admin_login(
        request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    assert login_response.status_code == 200
    cookie_header = login_response.headers["Set-Cookie"]
    token = cookie_header.split("dashboard_session=")[1].split(";")[0]

    request_with_cookie = _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_dashboard_user_store": FakeDashboardUserStoreMissing()},
    )
    result = await admin_me(request_with_cookie)
    assert result.authenticated is True
    assert result.role == "owner"
    assert "users:write" in result.permissions


@pytest.mark.asyncio
async def test_me_unauthenticated():
    config = _enabled_config()
    request = _make_request(app_config=config)
    result = await admin_me(request)
    assert result.authenticated is False


@pytest.mark.asyncio
async def test_logout():
    response = await admin_logout()
    assert response.status_code == 200
    assert "Max-Age=0" in response.headers["Set-Cookie"]


@pytest.mark.asyncio
async def test_metrics_summary_requires_auth():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await metrics_summary(request, params=SimpleNamespace(range_hours=24, bucket="hour", model=None))
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_vectorstore_summary_requires_auth():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await vectorstore_summary(request, params=SimpleNamespace(range_hours=24, bucket="hour", model=None))
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_vectorstore_state_requires_auth():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await vectorstore_state(request)
    assert "401" in str(exc_info.value)


class FakeMessageStore:
    async def list_models(self, *, date_from=None, date_to=None):
        return ["CyberSecurityAssistant", "GameSummaryAssistant"]

    def __init__(self) -> None:
        self.last_list_kwargs: dict[str, Any] = {}
        self.last_export_kwargs: dict[str, Any] = {}

    async def list_conversations(self, **kwargs):
        self.last_list_kwargs = dict(kwargs)
        return {
            "total": 1,
            "page": 1,
            "page_size": 50,
            "items": [
                {
                    "conversation_id": "conv-1",
                    "message_count": 3,
                    "user_message_count": 2,
                    "assistant_message_count": 1,
                    "first_message_at": "2026-06-01T10:00:00+00:00",
                    "last_message_at": "2026-06-01T10:03:00+00:00",
                    "model_name": "CyberSecurityAssistant",
                    "retrieval_rows": 1,
                    "title": "Threat analysis",
                    "conv_model_name": "CyberSecurityAssistant",
                    "user_id": "u1",
                    "username": "alice",
                    "display_name": "Alice A.",
                    "used_api_key": False,
                }
            ],
        }

    async def list_conversation_users(self):
        return {
            "users": [
                {
                    "user_id": "u1",
                    "username": "alice",
                    "display_name": "Alice A.",
                    "conversation_count": 3,
                }
            ],
            "anonymous_count": 2,
        }

    async def get_conversation_users(self, conversation_ids):
        return {
            cid: {
                "conversation_id": cid,
                "user_id": "u1",
                "title": f"Title {cid}",
                "username": "alice",
                "display_name": "Alice A.",
            }
            for cid in conversation_ids
        }

    async def get_conversation_messages(self, conversation_id: str):
        if conversation_id == "missing":
            return []
        return [
            {
                "id": "row-1",
                "conversation_id": conversation_id,
                "user_content": "Hallo",
                "assistant_content": "Hi",
                "model_name": "CyberSecurityAssistant",
                "chat_history": None,
                "retrieval_info": None,
                "created_at": "2026-06-01T10:00:00+00:00",
            }
        ]

    async def get_conversation_metadata(self, conversation_id: str):
        if conversation_id == "missing":
            return None
        return {
            "conversation_id": conversation_id,
            "message_count": 1,
            "user_message_count": 1,
            "assistant_message_count": 1,
            "first_message_at": "2026-06-01T10:00:00+00:00",
            "last_message_at": "2026-06-01T10:00:10+00:00",
            "model_name": "CyberSecurityAssistant",
            "retrieval_rows": 0,
        }

    async def list_messages_for_export(self, **kwargs):
        self.last_export_kwargs = dict(kwargs)
        return [
            {
                "id": "m1",
                "conversation_id": "conv-1",
                "created_at": "2026-06-01T10:00:00+00:00",
                "model_name": "CyberSecurityAssistant",
                "user_content": "Hallo",
                "assistant_content": "Antwort",
                "chat_history": None,
                "retrieval_info": None,
            }
        ]


class FakeRuntimeConfigStore:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get_all(self) -> dict[str, Any]:
        return dict(self._data)

    async def set_many(self, values: dict[str, Any]) -> None:
        self._data.update(values)


class FakeApiKeyStore:
    def __init__(self) -> None:
        self._keys: list[Any] = []

    async def list_keys(self) -> list[Any]:
        return list(self._keys)

    async def create_key(self, key: Any) -> None:
        self._keys.append(key)


async def _authed_request_with_store() -> Any:
    config = _enabled_config()
    login_request = _make_request(app_config=config)
    login_response = await admin_login(
        login_request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    token = login_response.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]
    return _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_message_store": FakeMessageStore()},
    )


@pytest.mark.asyncio
async def test_conversations_list_requires_auth():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await conversations_list(request, params=SimpleNamespace(date_from=None, date_to=None, conversation_id=None, model=None, min_user_messages=0, user_id=None, query=None, page=1, page_size=50, sort_by="last_message_at", sort_order="desc"))
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_conversations_endpoints_success():
    request = await _authed_request_with_store()

    models = await conversations_models(request)
    assert "CyberSecurityAssistant" in models

    listed = await conversations_list(
        request,
        params=SimpleNamespace(
            date_from="2026-06-01T00:00:00Z",
            date_to="2026-06-01T23:59:59Z",
            conversation_id="conv",
            model="CyberSecurityAssistant",
            min_user_messages=1,
            user_id=None,
            query=None,
            page=1,
            page_size=50,
            sort_by="last_message_at",
            sort_order="desc",
        ),
    )
    assert listed["total"] == 1
    assert listed["items"][0]["conversation_id"] == "conv-1"
    assert listed["items"][0]["display_name"] == "Alice A."
    assert listed["items"][0]["title"] == "Threat analysis"

    detail = await conversation_detail(request, "conv-1")
    assert detail["conversation_id"] == "conv-1"
    assert len(detail["messages"]) == 1

    metadata = await conversation_metadata(request, "conv-1")
    assert metadata["message_count"] == 1


@pytest.mark.asyncio
async def test_conversation_detail_not_found():
    request = await _authed_request_with_store()
    with pytest.raises(Exception) as exc_info:
        await conversation_detail(request, "missing")
    assert "404" in str(exc_info.value) or "Conversation not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_conversations_export_csv_and_json():
    request = await _authed_request_with_store()

    csv_response = await conversations_export(
        request,
        params=SimpleNamespace(
            date_from="2026-06-01T00:00:00Z",
            date_to="2026-06-01T23:59:59Z",
            conversation_id=None,
            model=None,
            min_user_messages=0,
            user_id=None,
            query=None,
            format="csv",
        ),
    )
    csv_text = csv_response.body.decode("utf-8")
    assert "conversation_id" in csv_text
    assert "conv-1" in csv_text
    assert "Alice A." in csv_text
    assert "Title conv-1" in csv_text

    json_response = await conversations_export(
        request,
        params=SimpleNamespace(
            date_from="2026-06-01T00:00:00Z",
            date_to="2026-06-01T23:59:59Z",
            conversation_id=None,
            model=None,
            min_user_messages=0,
            user_id="u1",
            query=None,
            format="json",
        ),
    )
    payload = json.loads(json_response.body.decode("utf-8"))
    assert payload["total_messages"] == 1
    assert payload["total_conversations"] == 1
    conv = payload["conversations"][0]
    assert conv["display_name"] == "Alice A."
    assert conv["title"] == "Title conv-1"
    assert conv["user_id"] == "u1"


@pytest.mark.asyncio
async def test_conversations_users_endpoint():
    request = await _authed_request_with_store()
    result = await conversations_users(request)
    assert result["anonymous_count"] == 2
    assert result["users"][0]["user_id"] == "u1"
    assert result["users"][0]["display_name"] == "Alice A."


@pytest.mark.asyncio
async def test_conversations_list_passes_user_and_query_filters():
    request = await _authed_request_with_store()
    await conversations_list(
        request,
        params=SimpleNamespace(
            date_from=None,
            date_to=None,
            conversation_id=None,
            model=None,
            min_user_messages=0,
            user_id="anonymous",
            query="threat",
            page=1,
            page_size=25,
            sort_by="user",
            sort_order="asc",
        ),
    )
    store = request.app.state._message_store
    assert store.last_list_kwargs["user_id"] == "anonymous"
    assert store.last_list_kwargs["query"] == "threat"
    assert store.last_list_kwargs["sort_by"] == "user"


@pytest.mark.asyncio
async def test_conversations_export_passes_user_and_query_filters():
    request = await _authed_request_with_store()
    await conversations_export(
        request,
        params=SimpleNamespace(
            date_from="2026-06-01T00:00:00Z",
            date_to="2026-06-01T23:59:59Z",
            conversation_id=None,
            model=None,
            min_user_messages=0,
            user_id="u1",
            query="threat",
            format="json",
        ),
    )
    store = request.app.state._message_store
    assert store.last_export_kwargs["user_id"] == "u1"
    assert store.last_export_kwargs["query"] == "threat"


@pytest.mark.asyncio
async def test_conversations_users_requires_auth():
    config = _enabled_config()
    request = _make_request(app_config=config)
    with pytest.raises(Exception) as exc_info:
        await conversations_users(request)
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_put_runtime_config_api_keys_enabled_with_stored_value():
    """Saving api_keys_enabled must succeed when no env override exists,
    even if a stored DB value has already been applied to the live config."""
    config = AppConfig()
    config.dashboard_enabled = True
    config.dashboard_admin_password = "secret"
    config.dashboard_session_secret = "test-session-secret"
    config._env_backed_values = {"api_keys_enabled": True}  # same as default
    config.api_keys_enabled = False  # simulate stored DB value applied at startup

    login_request = _make_request(app_config=config)
    login_response = await admin_login(
        login_request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    token = login_response.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    store = FakeRuntimeConfigStore()
    store._data["api_keys_enabled"] = False  # existing stored value
    request = _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_runtime_config_store": store},
    )

    result = await put_runtime_config(
        request,
        payload=SimpleNamespace(values={"api_keys_enabled": True}),
    )
    assert result["success"] is True
    assert store._data["api_keys_enabled"] is True


@pytest.mark.asyncio
async def test_put_runtime_config_api_keys_enabled_rejects_env_override():
    """Saving api_keys_enabled must be rejected when an env override exists."""
    config = AppConfig()
    config.dashboard_enabled = True
    config.dashboard_admin_password = "secret"
    config.dashboard_session_secret = "test-session-secret"
    config._env_backed_values = {"api_keys_enabled": False}  # env overrides default True
    config.api_keys_enabled = False

    login_request = _make_request(app_config=config)
    login_response = await admin_login(
        login_request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    token = login_response.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    store = FakeRuntimeConfigStore()
    request = _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_runtime_config_store": store},
    )

    with pytest.raises(Exception) as exc_info:
        await put_runtime_config(
            request,
            payload=SimpleNamespace(values={"api_keys_enabled": True}),
        )
    assert "overridden by environment" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_api_keys_env_source():
    """Legacy API key must be returned with source='env', regular keys with source='db'."""
    config = _enabled_config()
    login_request = _make_request(app_config=config)
    login_response = await admin_login(
        login_request,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    token = login_response.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    from app.api_keys import AppApiKey
    store = FakeApiKeyStore()
    await store.create_key(AppApiKey(id="legacy", name="Environment API Key", api_key_hash="abc", allowed_models=[], enabled=True))
    await store.create_key(AppApiKey(id="ak-123", name="Test Key", api_key_hash="def", allowed_models=["model-a"], enabled=True))

    request = _make_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_api_key_store": store},
    )

    result = await list_api_keys(request)
    assert len(result) == 2
    legacy = next((k for k in result if k["id"] == "legacy"), None)
    regular = next((k for k in result if k["id"] == "ak-123"), None)
    assert legacy is not None
    assert legacy["source"] == "env"
    assert regular is not None
    assert regular["source"] == "db"
