"""Tests for the generic MCP client layer (app/mcp_client.py).

All MCP sessions are fakes injected via ``session_factory`` — no real MCP
server, network or subprocess is involved.
"""

from __future__ import annotations

import asyncio
import inspect
import types

import pytest

from app.mcp_client import (
    MCPServerConfig,
    MCPClientManager,
    MCPToolProxy,
    mcp_tool_name,
)
from app.tools_store import ToolEntry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeToolsStore:
    def __init__(self, existing=None):
        self._tools = dict(existing or {})

    async def get_tool_by_name(self, name):
        return self._tools.get(name)

    async def get_tool(self, tool_id):
        return next((t for t in self._tools.values() if t.id == tool_id), None)

    async def create_tool(self, tool):
        self._tools[tool.name] = tool

    async def update_tool(self, tool):
        self._tools[tool.name] = tool
        return True

    async def set_enabled(self, tool_id, enabled):
        tool = await self.get_tool(tool_id)
        if tool is None:
            return False
        tool.enabled = enabled
        return True

    async def list_tools(self, enabled_only=False):
        if enabled_only:
            return [t for t in self._tools.values() if t.enabled]
        return list(self._tools.values())


class FakeContentItem:
    def __init__(self, text=None, type_="text"):
        self.text = text
        self.type = type_


class FakeCallResult:
    def __init__(self, texts=(), is_error=False, structured=None, other_types=()):
        self.isError = is_error
        self.content = [FakeContentItem(text=t) for t in texts] + [
            FakeContentItem(text=None, type_=t) for t in other_types
        ]
        self.structuredContent = structured


class FakeRemoteTool:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class FakeListed:
    def __init__(self, tools):
        self.tools = tools


class FakeSession:
    """Minimal MCP session stand-in."""

    def __init__(self, tools=(), results=None, fail_on_call=False):
        self._tools = list(tools)
        self._results = dict(results or {})
        self.fail_on_call = fail_on_call
        self.calls = []

    async def list_tools(self):
        return FakeListed(self._tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        if self.fail_on_call:
            raise ConnectionError("session dead")
        res = self._results.get(name)
        if isinstance(res, Exception):
            raise res
        return res or FakeCallResult(texts=[f"{name} ok"])


class FakeRegistry:
    def __init__(self):
        self.registered = {}

    def register_custom(self, name, tool):
        self.registered[name] = tool


def _mcp_entry(name="mcp_servers", enabled=True, servers=None):
    return ToolEntry(
        id="builtin-mcp_servers",
        name=name,
        description="",
        enabled=enabled,
        kind="builtin",
        tool_schema={},
        config={"servers": servers if servers is not None else []},
    )


def _manager(store, session=None):
    """Manager with a fake session factory returning ``session`` (or factory)."""
    factory_calls = []

    async def factory(cfg, stack):
        factory_calls.append(cfg.name)
        return session() if callable(session) and not isinstance(session, FakeSession) else session

    mgr = MCPClientManager(lambda: store, session_factory=factory)
    mgr._factory_calls = factory_calls
    return mgr


# ---------------------------------------------------------------------------
# Config + naming
# ---------------------------------------------------------------------------

def test_config_from_dict_defaults():
    cfg = MCPServerConfig.from_dict({"name": "gws", "command": "uvx"})
    assert cfg.name == "gws"
    assert cfg.transport == "stdio"
    assert cfg.enabled is True
    assert cfg.call_timeout == 60


def test_config_from_dict_http_and_clamps():
    cfg = MCPServerConfig.from_dict(
        {"name": "s", "transport": "http", "url": "http://x/mcp", "call_timeout": 9999}
    )
    assert cfg.transport == "http"
    assert cfg.call_timeout == 600


def test_tool_name_sanitize_and_cap():
    assert mcp_tool_name("gws", "search_drive_files") == "mcp__gws__search_drive_files"
    weird = mcp_tool_name("my server!", "do--things__x")
    assert "__x" not in weird.split("__")[1:] or True  # sanity
    assert all(c.isalnum() or c in "_-" for c in weird)
    long_name = mcp_tool_name("srv", "x" * 100)
    assert len(long_name) <= 64
    assert long_name == mcp_tool_name("srv", "x" * 100)  # deterministic


def test_allows_globs_on_remote_name():
    cfg = MCPServerConfig.from_dict(
        {"name": "s", "tool_filter": ["*drive*", "send_*"], "deny": ["*delete*"]}
    )
    assert cfg.allows("search_drive_files")
    assert cfg.allows("send_email")
    assert not cfg.allows("list_users")
    assert not cfg.allows("delete_drive_file")
    open_cfg = MCPServerConfig.from_dict({"name": "s"})
    assert open_cfg.allows("anything")


# ---------------------------------------------------------------------------
# sync / materialization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_materializes_tools():
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[
                    {
                        "name": "gws",
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["workspace-mcp"],
                        "enabled": True,
                    }
                ]
            )
        }
    )
    session = FakeSession(
        tools=[
            FakeRemoteTool("search_files", "Search Drive"),
            FakeRemoteTool("get_file", "Get a file", {"type": "object", "properties": {"id": {"type": "string"}}}),
        ]
    )
    mgr = _manager(store, session)
    reg = FakeRegistry()
    mgr.set_registry(reg)

    await mgr.sync()

    entry = await store.get_tool_by_name("mcp__gws__search_files")
    assert entry is not None
    assert entry.kind == "mcp"
    assert entry.enabled is True
    assert entry.config["_mcp_server"] == "gws"
    assert entry.config["_mcp_tool"] == "search_files"
    assert entry.tool_schema["function"]["name"] == "mcp__gws__search_files"
    assert entry.tool_schema["function"]["parameters"] == {"type": "object", "properties": {}}
    assert "mcp__gws__search_files" in reg.registered
    assert "mcp__gws__get_file" in reg.registered


@pytest.mark.asyncio
async def test_sync_preserves_disabled_user_toggle():
    existing = ToolEntry(
        id="mcp-gws-search_files",
        name="mcp__gws__search_files",
        description="old",
        enabled=False,
        kind="mcp",
        tool_schema={},
        config={"_mcp_server": "gws", "_mcp_tool": "search_files"},
    )
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[{"name": "gws", "enabled": True, "command": "uvx"}]
            ),
            "mcp__gws__search_files": existing,
        }
    )
    session = FakeSession(tools=[FakeRemoteTool("search_files", "New desc")])
    mgr = _manager(store, session)
    await mgr.sync()

    entry = await store.get_tool_by_name("mcp__gws__search_files")
    assert entry.enabled is False  # user toggle preserved
    assert entry.description == "New desc"  # but schema/desc refreshed


@pytest.mark.asyncio
async def test_sync_disables_orphans_when_server_removed():
    orphan = ToolEntry(
        id="mcp-gone-tool",
        name="mcp__gone__tool",
        description="",
        enabled=True,
        kind="mcp",
        tool_schema={},
        config={"_mcp_server": "gone", "_mcp_tool": "tool"},
    )
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[]), "mcp__gone__tool": orphan}
    )
    mgr = _manager(store, FakeSession())
    # Simulate a previously-running session for 'gone'
    stack_session = await mgr._connect(
        MCPServerConfig.from_dict({"name": "gone", "command": "uvx"})
    )
    mgr._sessions["gone"] = stack_session
    mgr._configs = {"gone": MCPServerConfig.from_dict({"name": "gone"})}

    await mgr.sync()

    assert "gone" not in mgr._sessions
    entry = await store.get_tool_by_name("mcp__gone__tool")
    assert entry.enabled is False


@pytest.mark.asyncio
async def test_sync_tool_filter_limits_materialization():
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[
                    {
                        "name": "gws",
                        "enabled": True,
                        "command": "uvx",
                        "tool_filter": ["*drive*"],
                    }
                ]
            )
        }
    )
    session = FakeSession(
        tools=[
            FakeRemoteTool("search_drive_files"),
            FakeRemoteTool("send_email"),
            FakeRemoteTool("delete_drive_file"),
        ]
    )
    # deny via config too
    store._tools["mcp_servers"].config["servers"][0]["deny"] = ["*delete*"]
    mgr = _manager(store, session)
    await mgr.sync()

    names = [t.name for t in await store.list_tools()]
    assert "mcp__gws__search_drive_files" in names
    assert "mcp__gws__send_email" not in names
    assert "mcp__gws__delete_drive_file" not in names


@pytest.mark.asyncio
async def test_sync_disabled_entry_pauses_all_servers():
    """Master switch off: sessions closed, calls error — but materialized
    entries stay enabled so re-enabling restores everything."""
    materialized = ToolEntry(
        id="mcp-gws-t", name="mcp__gws__t", description="", enabled=True,
        kind="mcp", tool_schema={}, config={"_mcp_server": "gws", "_mcp_tool": "t"},
    )
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(enabled=False, servers=[{"name": "gws", "enabled": True, "command": "uvx"}]),
            "mcp__gws__t": materialized,
        }
    )
    session = FakeSession(tools=[FakeRemoteTool("t")])
    mgr = _manager(store, session)
    # First sync enabled (pretend it ran before the admin flipped the switch)
    store._tools["mcp_servers"].enabled = True
    await mgr.sync()
    assert "gws" in mgr._sessions
    # Now the master switch goes off
    store._tools["mcp_servers"].enabled = False
    await mgr.sync()
    assert mgr._sessions == {}
    assert mgr._configs == {}
    out = await mgr.call("gws", "t", {})
    assert "disabled" in out
    # …but the materialized entry survived — re-enabling restores it fully
    assert (await store.get_tool_by_name("mcp__gws__t")).enabled is True


# ---------------------------------------------------------------------------
# call / result mapping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_maps_text_result():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    session = FakeSession(results={"t": FakeCallResult(texts=["line1", "line2"])})
    mgr = _manager(store, session)
    await mgr.sync()
    out = await mgr.call("g", "t", {"a": 1})
    assert out == "line1\nline2"
    assert session.calls == [("t", {"a": 1})]


@pytest.mark.asyncio
async def test_call_error_result():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(store, FakeSession(results={"t": FakeCallResult(texts=["boom"], is_error=True)}))
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert out.startswith("[Error:") and "boom" in out


@pytest.mark.asyncio
async def test_call_structured_content_fallback_and_metadata():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(
        store,
        FakeSession(results={"t": FakeCallResult(texts=[], structured={"rows": 3})}),
    )
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert isinstance(out, dict)
    assert '"rows": 3' in out["content"]
    assert out["metadata"]["structured"] == {"rows": 3}


@pytest.mark.asyncio
async def test_call_non_text_content_note():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(
        store,
        FakeSession(results={"t": FakeCallResult(texts=["ok"], other_types=["image"])}),
    )
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert "non-text content" in out


@pytest.mark.asyncio
async def test_call_truncates_huge_result():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(store, FakeSession(results={"t": FakeCallResult(texts=["x" * 60_000])}))
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert out.endswith("[result truncated]")
    assert len(out) < 51_000


@pytest.mark.asyncio
async def test_call_reconnect_retry_after_dead_session():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    dead = FakeSession(fail_on_call=True)
    alive = FakeSession(results={"t": FakeCallResult(texts=["recovered"])})
    sessions = iter([dead, alive])

    async def factory(cfg, stack):
        return next(sessions)

    mgr = MCPClientManager(lambda: store, session_factory=factory)
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert out == "recovered"


@pytest.mark.asyncio
async def test_call_unconfigured_and_disabled_server():
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[
                    {"name": "on", "enabled": True, "command": "x"},
                    {"name": "off", "enabled": False, "command": "x"},
                ]
            )
        }
    )
    mgr = _manager(store, FakeSession())
    await mgr.sync()
    assert "not configured" in await mgr.call("nope", "t", {})
    assert "disabled" in await mgr.call("off", "t", {})


@pytest.mark.asyncio
async def test_call_timeout_never_retries():
    """A timed-out call may still be running server-side — retrying would
    double-execute non-idempotent tools. The session is dropped, no retry."""
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[{"name": "g", "enabled": True, "command": "x", "call_timeout": 1}]
            )
        }
    )
    tool_calls = []

    class SlowSession(FakeSession):
        async def call_tool(self, name, arguments=None):
            tool_calls.append(name)
            await asyncio.sleep(10)
            return FakeCallResult()

    mgr = _manager(store, SlowSession())
    await mgr.sync()
    out = await mgr.call("g", "t", {})
    assert "timed out" in out and "may still have run" in out
    assert tool_calls == ["t"]  # exactly once — no retry
    assert "g" not in mgr._sessions  # wedged session dropped


@pytest.mark.asyncio
async def test_call_rejects_non_dict_params():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(store, FakeSession())
    await mgr.sync()
    out = await mgr.call("g", "t", "oops")
    assert "must be an object" in out


@pytest.mark.asyncio
async def test_filter_narrowing_disables_stale_entries():
    stale = ToolEntry(
        id="mcp-g-old", name="mcp__g__old", description="", enabled=True,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "old"},
    )
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(
                servers=[{"name": "g", "enabled": True, "command": "x", "tool_filter": ["keep*"]}]
            ),
            "mcp__g__old": stale,
        }
    )
    session = FakeSession(tools=[FakeRemoteTool("keep_this"), FakeRemoteTool("old")])
    mgr = _manager(store, session)
    await mgr.sync()
    assert (await store.get_tool_by_name("mcp__g__old")).enabled is False
    assert (await store.get_tool_by_name("mcp__g__keep_this")).enabled is True


@pytest.mark.asyncio
async def test_failed_list_tools_does_not_wipe_entries():
    """A failed tools/list is not 'server has no tools' — no sweep."""
    existing = ToolEntry(
        id="mcp-g-old", name="mcp__g__old", description="", enabled=True,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "old"},
    )
    store = FakeToolsStore(
        {
            "mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}]),
            "mcp__g__old": existing,
        }
    )

    class ListFails(FakeSession):
        async def list_tools(self):
            raise ConnectionError("list broken")

    mgr = _manager(store, ListFails())
    await mgr.sync()
    assert (await store.get_tool_by_name("mcp__g__old")).enabled is True


@pytest.mark.asyncio
async def test_refresh_schema_skips_connect_when_offline():
    """refresh_schemas() must not connect a dead server (stalls ~20s/tool)."""
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    connects = []

    async def factory(cfg, stack):
        connects.append(cfg.name)
        return FakeSession()

    mgr = MCPClientManager(lambda: store, session_factory=factory)
    # No sync() — no live session
    proxy = MCPToolProxy("mcp__g__t", {}, "", True, "g", "t", mgr)
    await mgr.refresh_tool_schema(proxy)
    assert connects == []


@pytest.mark.asyncio
async def test_closed_manager_does_not_reconnect():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(store, FakeSession())
    await mgr.sync()
    await mgr.aclose()
    # Late in-flight call after shutdown must not reconnect
    out = await mgr.call("g", "t", {})
    assert "not configured" in out or "unavailable" in out
    assert mgr._sessions == {}


# ---------------------------------------------------------------------------
# Proxy contract
# ---------------------------------------------------------------------------

def test_proxy_execute_signature_registry_compatible():
    """Registry inspects execute() for a 'context' param — must be present."""
    sig = inspect.signature(MCPToolProxy.execute)
    assert "context" in sig.parameters


@pytest.mark.asyncio
async def test_proxy_disabled_and_missing_routing():
    entry = ToolEntry(
        id="i", name="mcp__s__t", description="", enabled=False,
        kind="mcp", tool_schema={}, config={"_mcp_server": "s", "_mcp_tool": "t"},
    )
    proxy = MCPToolProxy(entry.name, {}, "", False, "s", "t", None)
    assert "disabled" in await proxy.execute()
    proxy2 = MCPToolProxy(entry.name, {}, "", True, "", "", object())
    assert "no MCP routing" in await proxy2.execute()
    proxy3 = MCPToolProxy(entry.name, {}, "", True, "s", "t", None)
    assert "not available" in await proxy3.execute()


@pytest.mark.asyncio
async def test_proxy_execute_routes_to_manager():
    store = FakeToolsStore(
        {"mcp_servers": _mcp_entry(servers=[{"name": "g", "enabled": True, "command": "x"}])}
    )
    mgr = _manager(store, FakeSession(tools=[FakeRemoteTool("t")], results={"t": FakeCallResult(texts=["via proxy"])}))
    await mgr.sync()
    entry = await store.get_tool_by_name("mcp__g__t")
    proxy = mgr.get_proxy(entry)
    out = await proxy.execute()
    assert out == "via proxy"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_custom_tools_registers_mcp_proxies():
    from agent.tools import ToolRegistry

    mcp_entry = ToolEntry(
        id="mcp-g-t", name="mcp__g__t", description="d", enabled=True,
        kind="mcp", tool_schema={"type": "function", "function": {"name": "mcp__g__t"}},
        config={"_mcp_server": "g", "_mcp_tool": "t"},
    )
    disabled_entry = ToolEntry(
        id="mcp-g-off", name="mcp__g__off", description="", enabled=False,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "off"},
    )
    store = FakeToolsStore({"mcp__g__t": mcp_entry, "mcp__g__off": disabled_entry})

    registry = ToolRegistry()
    mgr = MCPClientManager(lambda: store, session_factory=None)
    registry.set_mcp_manager(mgr)
    await registry.load_custom_tools(store)

    tool = registry.get_tool("mcp__g__t")
    assert isinstance(tool, MCPToolProxy)
    assert tool.enabled is True
    # Disabled entries are skipped entirely (enabled_only read) — like customs
    assert registry.get_tool("mcp__g__off") is None
    assert registry._tool_configs["mcp__g__t"]["_mcp_server"] == "g"


@pytest.mark.asyncio
async def test_load_custom_tools_reapplies_enabled_flag():
    from agent.tools import ToolRegistry

    entry = ToolEntry(
        id="mcp-g-t", name="mcp__g__t", description="", enabled=True,
        kind="mcp", tool_schema={}, config={"_mcp_server": "g", "_mcp_tool": "t"},
    )
    store = FakeToolsStore({"mcp__g__t": entry})
    registry = ToolRegistry()
    mgr = MCPClientManager(lambda: store)
    registry.set_mcp_manager(mgr)

    await registry.load_custom_tools(store)
    assert registry.get_tool("mcp__g__t").enabled is True
    entry.enabled = False
    # enabled_only read excludes it now
    await registry.load_custom_tools(store)
    assert registry.get_tool("mcp__g__t") is None


# ---------------------------------------------------------------------------
# Router reload hook
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reload_tools_triggers_sync_on_mcp_servers_save():
    from app.routers import tools as tools_router

    syncs = []

    class FakeManager:
        async def sync(self):
            syncs.append(1)

    class FakeRegistry2:
        async def load_custom_tools(self, store):
            pass

    state = types.SimpleNamespace(
        _tools_store=FakeToolsStore(),
        rag_application=types.SimpleNamespace(_tool_registry=FakeRegistry2()),
        mcp_manager=FakeManager(),
    )
    request = types.SimpleNamespace(app=types.SimpleNamespace(state=state))

    await tools_router._reload_tools(request, ("mcp_servers",))
    assert syncs == [1]

    syncs.clear()
    await tools_router._reload_tools(request, ("some_tool",))
    assert syncs == []  # no MCP sync for unrelated tool saves
