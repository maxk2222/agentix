"""Tests for app.tools_store."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools_store import ToolEntry, ToolsStore


@pytest.fixture
def mock_engine():
    """Create a mock async engine that returns a mock connection."""
    engine = MagicMock()
    conn = AsyncMock()
    engine.begin.return_value.__aenter__ = AsyncMock(return_value=conn)
    engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=conn)
    engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    engine._conn = conn
    return engine


@pytest.mark.asyncio
async def test_create_tool(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    tool = ToolEntry(
        id="tool-1",
        name="test-tool",
        description="A test tool",
        enabled=True,
        kind="custom",
        tool_schema={"type": "object"},
        config={"key": "val"},
        code="print('hello')",
        venv_config={"python_version": "3.11", "requirements": "requests"},
    )
    await store.create_tool(tool)

    conn = mock_engine._conn
    assert conn.execute.called
    call_args = conn.execute.call_args
    stmt = str(call_args[0][0])
    params = call_args[0][1]
    assert "INSERT" in stmt
    assert params["id"] == "tool-1"
    assert params["name"] == "test-tool"
    assert '"python_version": "3.11"' in params["venv_config"]


@pytest.mark.asyncio
async def test_get_tool_found(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchone.return_value = {
        "id": "tool-1",
        "name": "test-tool",
        "description": "desc",
        "enabled": True,
        "kind": "custom",
        "tool_schema": {},
        "config": {},
        "code": None,
        "venv_config": None,
        "runtime_mode": "venv",
        "created_at": None,
        "updated_at": None,
    }
    mock_engine._conn.execute.return_value = mock_result

    result = await store.get_tool("tool-1")
    assert result is not None
    assert result.id == "tool-1"
    assert result.name == "test-tool"


@pytest.mark.asyncio
async def test_get_tool_not_found(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.mappings.return_value.fetchone.return_value = None
    mock_engine._conn.execute.return_value = mock_result

    result = await store.get_tool("missing")
    assert result is None


@pytest.mark.asyncio
async def test_list_tools(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"id": "t1", "name": "a", "description": "", "enabled": True, "kind": "builtin", "tool_schema": {}, "config": {}, "code": None, "venv_config": None, "runtime_mode": "venv", "created_at": None, "updated_at": None},
        {"id": "t2", "name": "b", "description": "", "enabled": False, "kind": "custom", "tool_schema": {}, "config": {}, "code": None, "venv_config": None, "runtime_mode": "docker", "created_at": None, "updated_at": None},
    ]
    mock_engine._conn.execute.return_value = mock_result

    all_tools = await store.list_tools(enabled_only=False)
    assert len(all_tools) == 2


@pytest.mark.asyncio
async def test_update_tool(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_engine._conn.execute.return_value = mock_result

    tool = ToolEntry(
        id="tool-1",
        name="updated-name",
        description="updated",
        enabled=False,
        kind="custom",
        tool_schema={},
        config={},
        code="",
        venv_config={"requirements": "numpy"},
        created_at=None,
    )
    updated = await store.update_tool(tool)
    assert updated is True
    call_args = mock_engine._conn.execute.call_args
    stmt = str(call_args[0][0])
    params = call_args[0][1]
    assert "UPDATE" in stmt
    assert params["name"] == "updated-name"
    assert '"requirements": "numpy"' in params["venv_config"]


@pytest.mark.asyncio
async def test_delete_tool(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_engine._conn.execute.return_value = mock_result

    deleted = await store.delete_tool("tool-1")
    assert deleted is True
    call_args = mock_engine._conn.execute.call_args
    stmt = str(call_args[0][0])
    assert "DELETE" in stmt


@pytest.mark.asyncio
async def test_set_enabled(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_engine._conn.execute.return_value = mock_result

    updated = await store.set_enabled("tool-1", False)
    assert updated is True
    call_args = mock_engine._conn.execute.call_args
    params = call_args[0][1]
    assert params["enabled"] is False


@pytest.mark.asyncio
async def test_seed_builtin_tools_inserts_code_execution(mock_engine):
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    # Simulate get_tool_by_name returning None (nothing exists yet)
    mock_engine._conn.execute.return_value = MagicMock(
        mappings=MagicMock(return_value=MagicMock(
            fetchone=MagicMock(return_value=None)
        ))
    )

    await store.seed_builtin_tools()

    calls = mock_engine._conn.execute.call_args_list
    insert_stmts = [str(c[0][0]) for c in calls if "INSERT" in str(c[0][0])]
    names = [c[0][1].get("name") for c in calls if c[0][1] and "name" in c[0][1]]

    assert "code_execution" in names
    assert "weather_lookup" in names
    assert any("builtin-code_execution" in str(c[0][1].get("id", "")) for c in calls if c[0][1] and "id" in c[0][1])


@pytest.mark.asyncio
async def test_seed_builtin_tools_repairs_stale_description(mock_engine):
    """An existing builtin row with an outdated description/schema is updated —
    enabled/config/code stay untouched."""
    store = ToolsStore(mock_engine)
    store._tables_ensured = True

    def stale_row(name):
        return {
            "id": f"builtin-{name}",
            "name": name,
            "description": "STALE description",
            "enabled": False,  # user-disabled must be preserved
            "kind": "builtin",
            "tool_schema": {"type": "function", "function": {"name": name}},
            "config": {"user": "cfg"},
            "code": None,
            "venv_config": {},
            "runtime_mode": "venv",
            "created_at": None,
            "updated_at": None,
        }

    def fake_execute(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "SELECT" in sql:
            name = (params or {}).get("name", "")
            result.mappings.return_value.fetchone.return_value = stale_row(name)
        return result

    mock_engine._conn.execute = AsyncMock(side_effect=fake_execute)

    await store.seed_builtin_tools()

    calls = mock_engine._conn.execute.call_args_list
    updates = [c for c in calls if "UPDATE" in str(c[0][0])]
    doc_updates = [c for c in updates if c[0][1].get("name") == "document_generator"]
    assert doc_updates, "expected an UPDATE for the stale document_generator row"
    params = doc_updates[0][0][1]
    assert ".xlsx" in params["description"]
    assert params["enabled"] is False      # preserved
    assert params["config"] == '{"user": "cfg"}'  # preserved
    assert params["kind"] == "builtin"
