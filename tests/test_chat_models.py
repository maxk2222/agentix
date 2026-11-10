from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.routers.chat import list_models


class FakeChainEntry:
    def __init__(self, name: str, *, enabled: bool = True, api_accessible: bool = True) -> None:
        self.id = f"id-{name}"
        self.name = name
        self.description = f"{name} chain"
        self.enabled = enabled
        self.api_accessible = api_accessible
        self.flow_definition = {"nodes": [], "edges": []}
        self.execution_options = {}
        self.created_at = None
        self.updated_at = None


class FakeAgentChainsStore:
    def __init__(self, entries: list[FakeChainEntry]) -> None:
        self._entries = entries

    async def ensure_tables(self) -> None:
        pass

    async def list_chains(self):
        return list(self._entries)


class FakeChatRoleAssistantsStore:
    def __init__(self, assistants: list[str] | None = None) -> None:
        self._assistants = assistants or []

    async def get_assistants_for_role(self, role: str):
        return list(self._assistants)


class FakeRAGApplication:
    ready = True
    engine = object()
    instructions = {
        "PublicAssistant": {
            "model": "gpt-4o-mini",
            "description": "Public",
            "api_accessible": True,
        },
        "InternalAssistant": {
            "model": "gpt-4o-mini",
            "description": "Internal",
            "api_accessible": False,
        },
    }

    async def refresh_instructions_if_needed(self) -> bool:
        return False


def _make_request(*, app_config=None):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=FakeRAGApplication(),
                app_config=app_config,
            )
        ),
        headers={},
    )


def _patch_stores(monkeypatch, *, chains=None, anonymous=None):
    monkeypatch.setattr(
        "app.routers.chat.AgentChainsStore",
        lambda engine: FakeAgentChainsStore(chains or []),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(anonymous),
    )


@pytest.mark.asyncio
async def test_list_models_excludes_internal(monkeypatch):
    _patch_stores(monkeypatch)
    request = _make_request(app_config=None)
    result = await list_models(request)
    ids = [m["id"] for m in result["data"]]
    assert "PublicAssistant" in ids
    assert "InternalAssistant" not in ids


@pytest.mark.asyncio
async def test_list_models_no_dashboard_bypass(monkeypatch):
    """Internal assistants stay hidden even with a valid dashboard session."""
    _patch_stores(monkeypatch)
    config = SimpleNamespace(dashboard_session_secret="test-secret")
    request = _make_request(app_config=config)

    monkeypatch.setattr(
        "app.routers.chat.read_session_cookie",
        lambda req, cfg: {"user": "admin"},
    )

    result = await list_models(request)
    ids = [m["id"] for m in result["data"]]
    assert "PublicAssistant" in ids
    assert "InternalAssistant" not in ids


@pytest.mark.asyncio
async def test_list_models_ignores_anonymous_allowlist(monkeypatch):
    """Public-chat role allowlists do NOT affect /v1/models — only
    api_accessible decides what the raw API exposes."""
    _patch_stores(monkeypatch, anonymous=["InternalAssistant"])
    request = _make_request(app_config=None)
    result = await list_models(request)
    ids = [m["id"] for m in result["data"]]
    # InternalAssistant stays hidden (api_accessible=false), PublicAssistant
    # is listed even though it is not in the anonymous allowlist.
    assert ids == ["PublicAssistant"]


@pytest.mark.asyncio
async def test_list_models_includes_public_chains(monkeypatch):
    _patch_stores(
        monkeypatch,
        chains=[
            FakeChainEntry("PublicChain"),
            FakeChainEntry("DisabledChain", enabled=False),
            FakeChainEntry("InternalChain", api_accessible=False),
        ],
    )
    request = _make_request(app_config=None)
    result = await list_models(request)
    ids = [m["id"] for m in result["data"]]
    assert "chain:PublicChain" in ids
    assert "chain:DisabledChain" not in ids
    assert "chain:InternalChain" not in ids
    chain_entry = next(m for m in result["data"] if m["id"] == "chain:PublicChain")
    assert chain_entry["type"] == "chain"
    assistant_entry = next(m for m in result["data"] if m["id"] == "PublicAssistant")
    assert assistant_entry["type"] == "assistant"


@pytest.mark.asyncio
async def test_list_models_chain_ignores_allowlist(monkeypatch):
    """All enabled+api_accessible chains are listed regardless of allowlists."""
    _patch_stores(
        monkeypatch,
        chains=[FakeChainEntry("PublicChain"), FakeChainEntry("OtherChain")],
        anonymous=["PublicAssistant", "chain:PublicChain"],
    )
    request = _make_request(app_config=None)
    result = await list_models(request)
    ids = [m["id"] for m in result["data"]]
    assert ids == ["PublicAssistant", "chain:PublicChain", "chain:OtherChain"]
