"""Tests for the tools admin router."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.routers.tools import ToolCreatePayload, create_tool, delete_tool


class FakeToolsStore:
    def __init__(self, existing=None):
        self._tools = existing or {}

    async def get_tool_by_name(self, name):
        return self._tools.get(name)

    async def create_tool(self, tool):
        self._tools[tool.name] = tool

    async def get_tool(self, tool_id):
        return next((tool for tool in self._tools.values() if tool.id == tool_id), None)

    async def update_tool(self, tool):
        self._tools[tool.name] = tool
        return True

    async def list_tools(self, enabled_only=False):
        return list(self._tools.values())


class FakeRequest:
    def __init__(self, state):
        self.app = type("App", (), {"state": state})()


async def _mock_require_permission(req, perm):
    pass


@pytest.mark.asyncio
async def test_create_builtin_tool_rejected(monkeypatch):
    """Creating a new tool with kind='builtin' must be rejected."""
    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    store = FakeToolsStore()
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    payload = ToolCreatePayload(
        name="test_builtin",
        description="",
        enabled=True,
        kind="builtin",
    )
    with pytest.raises(HTTPException) as exc_info:
        await create_tool(request, payload)
    assert exc_info.value.status_code == 400
    assert "builtin" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_create_assistant_prefixed_tool_rejected(monkeypatch):
    """Creating a tool with name starting with 'assistant__' must be rejected."""
    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    store = FakeToolsStore()
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    payload = ToolCreatePayload(
        name="assistant__test",
        description="",
        enabled=True,
        kind="custom",
    )
    with pytest.raises(HTTPException) as exc_info:
        await create_tool(request, payload)
    assert exc_info.value.status_code == 400
    assert "reserved" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_create_runtime_subagent_tool_rejected(monkeypatch):
    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    store = FakeToolsStore()
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    payload = ToolCreatePayload(
        name="spawn_subagent",
        description="",
        enabled=True,
        kind="custom",
    )
    with pytest.raises(HTTPException) as exc_info:
        await create_tool(request, payload)
    assert exc_info.value.status_code == 400
    assert "reserved" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_builtin_wait_config_can_still_be_updated(monkeypatch):
    from app.routers.tools import ToolUpdatePayload, update_tool
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    builtin = ToolEntry(
        id="builtin-wait",
        name="wait",
        description="Bounded pause",
        enabled=True,
        kind="builtin",
        tool_schema={},
        config={"max_seconds": 120},
    )
    store = FakeToolsStore({"wait": builtin})
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    result = await update_tool(
        request,
        "builtin-wait",
        ToolUpdatePayload(
            name="wait",
            description="Bounded pause",
            enabled=True,
            kind="builtin",
            config={"max_seconds": 60},
        ),
    )
    assert result.name == "wait"


@pytest.mark.asyncio
async def test_list_tools_includes_virtual_assistant_tools(monkeypatch):
    """Virtual assistant tools from the RAG registry should appear in the list."""
    from app.routers.tools import list_tools
    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)

    store = FakeToolsStore()

    mock_tool = MagicMock()
    mock_tool.name = "assistant__my_assistant"
    mock_tool.assistant_name = "MyAssistant"
    mock_tool.tool_schema = {
        "type": "function",
        "function": {"name": "assistant__my_assistant", "description": "Desc"},
    }

    registry = MagicMock()
    registry._assistant = {"assistant__my_assistant": mock_tool}

    rag_app = MagicMock()
    rag_app._tool_registry = registry
    rag_app.instructions = {"MyAssistant": {"agent_enabled": True}}

    request = FakeRequest(type("State", (), {"_tools_store": store, "rag_application": rag_app})())
    result = await list_tools(request)
    names = [r.name for r in result]
    assert "assistant__my_assistant" in names

    # Verify agent_enabled is propagated for the assistant tool
    assistant_tools = [r for r in result if r.kind == "assistant"]
    assert len(assistant_tools) == 1
    assert assistant_tools[0].config.get("agent_enabled") is True


@pytest.mark.asyncio
async def test_virtual_assistant_tool_agent_disabled(monkeypatch):
    """Virtual assistant tools should report agent_enabled=False when the target assistant has it disabled."""
    from app.routers.tools import list_tools
    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)

    store = FakeToolsStore()

    mock_tool = MagicMock()
    mock_tool.name = "assistant__plain_assistant"
    mock_tool.assistant_name = "PlainAssistant"
    mock_tool.tool_schema = {
        "type": "function",
        "function": {"name": "assistant__plain_assistant", "description": "Desc"},
    }

    registry = MagicMock()
    registry._assistant = {"assistant__plain_assistant": mock_tool}

    rag_app = MagicMock()
    rag_app._tool_registry = registry
    rag_app.instructions = {"PlainAssistant": {"agent_enabled": False}}

    request = FakeRequest(type("State", (), {"_tools_store": store, "rag_application": rag_app})())
    result = await list_tools(request)
    assistant_tools = [r for r in result if r.kind == "assistant"]
    assert len(assistant_tools) == 1
    assert assistant_tools[0].config.get("agent_enabled") is False


@pytest.mark.asyncio
async def test_delete_enabled_mcp_tool_rejected(monkeypatch):
    """Enabled managed MCP tools must be disabled, not deleted (they
    re-materialize on the next server sync — delete would lie)."""
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    mcp_entry = ToolEntry(
        id="mcp-g-t", name="mcp__g__t", description="", enabled=True,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "t"},
    )
    store = FakeToolsStore({"mcp__g__t": mcp_entry})

    async def delete_tool_fn(tool_id):
        store._tools.pop(mcp_entry.name)
        return True

    store.delete_tool = delete_tool_fn
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    with pytest.raises(HTTPException) as exc_info:
        await delete_tool(request, "mcp-g-t")
    assert exc_info.value.status_code == 400
    assert "disable" in exc_info.value.detail.lower()
    assert "mcp__g__t" in store._tools  # untouched


@pytest.mark.asyncio
async def test_delete_disabled_mcp_orphan_allowed(monkeypatch):
    """Disabled (orphaned) MCP entries CAN be deleted for real cleanup."""
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    mcp_entry = ToolEntry(
        id="mcp-g-t", name="mcp__g__t", description="", enabled=False,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "t"},
    )
    store = FakeToolsStore({"mcp__g__t": mcp_entry})

    async def delete_tool_fn(tool_id):
        store._tools.pop(mcp_entry.name)
        return True

    store.delete_tool = delete_tool_fn
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    result = await delete_tool(request, "mcp-g-t")
    assert result == {"success": True}
    assert "mcp__g__t" not in store._tools


@pytest.mark.asyncio
async def test_delete_reserved_builtin_rejected(monkeypatch):
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    entry = ToolEntry(
        id="builtin-mcp_servers", name="mcp_servers", description="", enabled=True,
        kind="builtin", tool_schema={}, config={"servers": []},
    )
    store = FakeToolsStore({"mcp_servers": entry})
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    with pytest.raises(HTTPException) as exc_info:
        await delete_tool(request, "builtin-mcp_servers")
    assert exc_info.value.status_code == 400
    assert "reserved" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# config_schema + secret masking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builtin_response_carries_config_schema_and_masked_secret(monkeypatch):
    """Builtins expose a config_schema and never return stored secrets."""
    from app.routers.tools import get_tool
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    entry = ToolEntry(
        id="builtin-email_sender", name="email_sender", description="", enabled=True,
        kind="builtin", tool_schema={},
        config={"smtp_host": "smtp.example.org", "smtp_password": "s3cret"},
    )
    store = FakeToolsStore({"email_sender": entry})
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    result = await get_tool(request, "builtin-email_sender")
    assert result.config_schema is not None
    assert any(f["key"] == "smtp_password" for f in result.config_schema)
    assert result.config["smtp_password"] == "__agentix_secret_set__"
    assert result.config["smtp_host"] == "smtp.example.org"


@pytest.mark.asyncio
async def test_secret_sentinel_preserves_stored_value_on_update(monkeypatch):
    """PUT with the sentinel keeps the stored secret; '' clears it."""
    from app.routers.tools import ToolUpdatePayload, update_tool
    from app.tools_store import ToolEntry
    from app.tool_config_schemas import SECRET_SENTINEL

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    entry = ToolEntry(
        id="builtin-email_sender", name="email_sender", description="", enabled=True,
        kind="builtin", tool_schema={},
        config={"smtp_host": "smtp.example.org", "smtp_password": "s3cret"},
    )
    store = FakeToolsStore({"email_sender": entry})
    request = FakeRequest(type("State", (), {"_tools_store": store})())

    await update_tool(
        request, "builtin-email_sender",
        ToolUpdatePayload(
            name="email_sender", description="", enabled=True, kind="builtin",
            config={"smtp_host": "smtp.example.org", "smtp_password": SECRET_SENTINEL},
        ),
    )
    assert store._tools["email_sender"].config["smtp_password"] == "s3cret"

    await update_tool(
        request, "builtin-email_sender",
        ToolUpdatePayload(
            name="email_sender", description="", enabled=True, kind="builtin",
            config={"smtp_host": "smtp.example.org", "smtp_password": ""},
        ),
    )
    assert store._tools["email_sender"].config["smtp_password"] == ""


@pytest.mark.asyncio
async def test_invalid_egress_cidr_rejected_on_update(monkeypatch):
    """cidr_list fields validate server-side — garbage must not reach the
    store where it would silently weaken the egress policy."""
    from app.routers.tools import ToolUpdatePayload, update_tool
    from app.tools_store import ToolEntry

    monkeypatch.setattr("app.routers.tools.require_permission", _mock_require_permission)
    entry = ToolEntry(
        id="builtin-code_execution", name="code_execution", description="",
        enabled=True, kind="builtin", tool_schema={}, config={},
    )
    store = FakeToolsStore({"code_execution": entry})
    request = FakeRequest(type("State", (), {"_tools_store": store})())
    with pytest.raises(HTTPException) as exc_info:
        await update_tool(
            request, "builtin-code_execution",
            ToolUpdatePayload(
                name="code_execution", description="", enabled=True, kind="builtin",
                config={"egress_blocked": ["10.0.0.0/8", "not-a-cidr"]},
            ),
        )
    assert exc_info.value.status_code == 400
    assert "not-a-cidr" in exc_info.value.detail
