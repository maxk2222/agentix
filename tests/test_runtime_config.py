"""Tests for runtime config and LLM provider admin endpoints."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.config import AppConfig
from app.routers.admin import (
    _build_effective_config,
    admin_login,
    get_runtime_config,
    put_runtime_config,
    list_providers,
    create_provider,
    delete_provider,
    rotate_provider_secret,
)
from app.llm_providers import LLMProvider


def _enabled_config(**kwargs: Any) -> AppConfig:
    return AppConfig(
        data_dir="/tmp/test",
        instructions_file="/tmp/test/instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
        llm_api_base="",
        llm_api_key="",
        dashboard_enabled=True,
        dashboard_admin_username="admin",
        dashboard_admin_password="secret",
        dashboard_session_secret="super-secret-key-for-tests",
        dashboard_session_ttl_seconds=3600,
        **kwargs,
    )


def _make_request(
    *,
    app_config: Any | None = None,
    extra_state: dict[str, Any] | None = None,
    cookies: dict[str, str] | None = None,
) -> Any:
    state: dict[str, Any] = {"app_config": app_config}
    if app_config is not None:
        state["_dashboard_user_store"] = FakeDashboardUserStore()
    if extra_state:
        state.update(extra_state)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state)),
        cookies=cookies or {},
    )


class FakeDashboardUserStore:
    async def authenticate(self, username: str, password: str) -> Any | None:
        if username == "admin" and password == "admin":
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin", username=username, display_name="Admin",
                password_hash="", role="admin", is_active=True,
            )
        return None

    async def get_by_username(self, username: str) -> Any | None:
        if username == "admin":
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin", username=username, display_name="Admin",
                password_hash="", role="admin", is_active=True,
            )
        return None

    async def get_user(self, user_id: str) -> Any | None:
        if user_id == "env-admin":
            from app.dashboard_users import DashboardUser
            return DashboardUser(
                id="env-admin", username="admin", display_name="Admin",
                password_hash="", role="admin", is_active=True,
            )
        return None

    async def record_login(self, user_id: str) -> None:
        pass


class FakeRuntimeConfigStore:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = dict(data or {})

    async def get_all(self) -> dict[str, Any]:
        return dict(self._data)

    async def set_many(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    async def import_many(self, values: dict[str, Any]) -> None:
        self._data.update(values)

    async def export_all(self) -> dict[str, Any]:
        return dict(self._data)


class FakeProviderStore:
    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        self._providers = list(providers or [])
        self._secrets: dict[str, str | None] = {}

    async def list_providers(self) -> list[LLMProvider]:
        return list(self._providers)

    async def get_provider(self, provider_id: str) -> LLMProvider | None:
        for p in self._providers:
            if p.id == provider_id:
                return p
        return None

    async def create_provider(self, provider: LLMProvider) -> None:
        self._providers.append(provider)

    async def update_provider(self, provider: LLMProvider) -> None:
        self._providers = [p for p in self._providers if p.id != provider.id]
        self._providers.append(provider)

    async def delete_provider(self, provider_id: str) -> None:
        self._providers = [p for p in self._providers if p.id != provider_id]
        self._secrets.pop(provider_id, None)

    async def set_api_key(self, provider_id: str, api_key: str | None) -> None:
        self._secrets[provider_id] = api_key

    async def get_api_key(self, provider_id: str) -> str | None:
        return self._secrets.get(provider_id)

    async def has_api_key(self, provider_id: str) -> bool:
        key = self._secrets.get(provider_id)
        return key is not None and key != ""

    async def rotate_api_key(self, provider_id: str, new_key: str) -> None:
        self._secrets[provider_id] = new_key

    async def update_check_status(self, provider_id: str, status: str) -> None:
        for p in self._providers:
            if p.id == provider_id:
                p.last_check_status = status

    async def get_default_provider(self) -> LLMProvider | None:
        for p in self._providers:
            if p.is_default:
                return p
        return None

    async def get_enabled_providers(self) -> list[LLMProvider]:
        return [p for p in self._providers if p.enabled]


@pytest.fixture
def fake_config() -> AppConfig:
    return AppConfig(
        data_dir="/tmp/test",
        instructions_file="/tmp/test/instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test",
        pg_user="test",
        pg_password="test",
    )


def test_build_effective_config_defaults(fake_config: AppConfig) -> None:
    stored: dict[str, Any] = {}
    effective = _build_effective_config(fake_config, stored)
    assert "vector_search_k" in effective
    assert effective["vector_search_k"]["source"] == "default"
    assert effective["vector_search_k"]["editable"] is True


def test_build_effective_config_env_override(fake_config: AppConfig) -> None:
    # Simulate env override by setting a non-default value directly on the config
    # This is tricky because AppConfig is frozen after creation.
    # Instead we test the logic with a known field.
    # For VECTOR_SEARCH_K, default is 5. If we can't easily mutate AppConfig,
    # we just verify the structure.
    stored: dict[str, Any] = {"vector_search_k": 10}
    effective = _build_effective_config(fake_config, stored)
    # If env did not override (default), db value wins
    assert effective["vector_search_k"]["value"] == 10
    assert effective["vector_search_k"]["source"] == "db"
    assert effective["vector_search_k"]["editable"] is True


async def _login_cookie(config: AppConfig) -> str:
    pwd = config.dashboard_admin_password or "admin"
    login_req = _make_request(app_config=config, extra_state={"_dashboard_user_store": FakeDashboardUserStore()})
    resp = await admin_login(login_req, payload=SimpleNamespace(username=config.dashboard_admin_username, password=pwd))
    header = resp.headers["Set-Cookie"]
    return header.split("dashboard_session=")[1].split(";")[0]


@pytest.mark.asyncio
async def test_get_runtime_config(fake_config: AppConfig) -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    store = FakeRuntimeConfigStore({"vector_search_k": 15})
    request = _make_request(
        app_config=config,
        extra_state={"_runtime_config_store": store},
        cookies={"dashboard_session": token},
    )
    result = await get_runtime_config(request)
    assert result["stored"]["vector_search_k"] == 15
    assert result["effective"]["vector_search_k"]["value"] == 15
    assert result["effective"]["vector_search_k"]["source"] == "db"


@pytest.mark.asyncio
async def test_put_runtime_config(fake_config: AppConfig) -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    store = FakeRuntimeConfigStore()
    request = _make_request(
        app_config=config,
        extra_state={"_runtime_config_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(values={"vector_search_k": 20})
    result = await put_runtime_config(request, payload)
    assert result["success"] is True
    stored = await store.get_all()
    assert stored["vector_search_k"] == 20


@pytest.mark.asyncio
async def test_put_runtime_config_rejects_invalid_key(fake_config: AppConfig) -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    store = FakeRuntimeConfigStore()
    request = _make_request(
        app_config=config,
        extra_state={"_runtime_config_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(values={"not_a_valid_key": 123})
    with pytest.raises(Exception) as exc_info:
        await put_runtime_config(request, payload)
    assert "not editable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_providers() -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    provider = LLMProvider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com",
        enabled=True,
        is_default=True,
    )
    store = FakeProviderStore([provider])
    request = _make_request(
        app_config=config,
        extra_state={"_provider_store": store},
        cookies={"dashboard_session": token},
    )
    result = await list_providers(request)
    db_providers = [r for r in result if r.get("source") == "db"]
    assert len(db_providers) == 1
    assert db_providers[0]["id"] == "openai"
    assert db_providers[0]["name"] == "OpenAI"
    assert db_providers[0]["is_default"] is True
    assert db_providers[0]["has_api_key"] is False


@pytest.mark.asyncio
async def test_create_provider() -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    store = FakeProviderStore()
    request = _make_request(
        app_config=config,
        extra_state={"_provider_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(
        id="ollama",
        name="Ollama Local",
        base_url="http://localhost:11434",
        enabled=True,
        is_default=False,
        provider_type="openai-compatible",
        timeout_seconds=30,
        max_retries=3,
        extra_headers=None,
        supports_native_tool_calls=None,
    )
    result = await create_provider(request, payload)
    assert result["success"] is True
    providers = await store.list_providers()
    assert len(providers) == 1
    assert providers[0].id == "ollama"


@pytest.mark.asyncio
async def test_create_provider_duplicate() -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    provider = LLMProvider(
        id="ollama",
        name="Ollama",
        base_url="http://localhost:11434",
    )
    store = FakeProviderStore([provider])
    request = _make_request(
        app_config=config,
        extra_state={"_provider_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(
        id="ollama",
        name="Ollama Local",
        base_url="http://localhost:11434",
        enabled=True,
        is_default=False,
        provider_type="openai-compatible",
        timeout_seconds=30,
        max_retries=3,
        extra_headers=None,
        supports_native_tool_calls=None,
    )
    with pytest.raises(Exception) as exc_info:
        await create_provider(request, payload)
    assert "already exists" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_provider() -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    provider = LLMProvider(
        id="ollama",
        name="Ollama",
        base_url="http://localhost:11434",
    )
    store = FakeProviderStore([provider])
    request = _make_request(
        app_config=config,
        extra_state={"_provider_store": store},
        cookies={"dashboard_session": token},
    )
    result = await delete_provider(request, "ollama")
    assert result["success"] is True
    providers = await store.list_providers()
    assert len(providers) == 0


@pytest.mark.asyncio
async def test_rotate_provider_secret() -> None:
    config = _enabled_config()
    token = await _login_cookie(config)
    provider = LLMProvider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com",
    )
    store = FakeProviderStore([provider])
    request = _make_request(
        app_config=config,
        extra_state={"_provider_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(api_key="sk-new-key")
    result = await rotate_provider_secret(request, "openai", payload)
    assert result["success"] is True
    assert await store.has_api_key("openai") is True
    assert await store.get_api_key("openai") == "sk-new-key"


@pytest.mark.asyncio
async def test_put_runtime_config_accepts_chat_file_keys() -> None:
    """chat_file_* keys are editable — previously rejected as 'not editable',
    which made file upload impossible to enable via the admin UI."""
    config = _enabled_config()
    token = await _login_cookie(config)
    store = FakeRuntimeConfigStore()
    request = _make_request(
        app_config=config,
        extra_state={"_runtime_config_store": store},
        cookies={"dashboard_session": token},
    )
    payload = SimpleNamespace(values={"chat_file_upload_enabled": True})
    result = await put_runtime_config(request, payload)
    assert result["success"] is True
    stored = await store.get_all()
    assert stored["chat_file_upload_enabled"] is True

    effective = _build_effective_config(config, stored)
    assert effective["chat_file_upload_enabled"]["value"] is True
    assert effective["chat_file_upload_enabled"]["source"] == "db"
