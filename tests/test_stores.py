"""Tests for app.stores."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.stores import AgentMemoryStore, InstructionsStore


class FakeRow:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class FakeMappings:
    def __init__(self, rows: List[dict[str, Any]]) -> None:
        self._rows = [FakeRow(**r) for r in rows]

    def first(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class FakeResult:
    def __init__(self, rows: List[dict[str, Any]] | None = None, scalar_value: Any = None, rowcount: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar_value
        self.rowcount = rowcount

    def scalar(self):
        return self._scalar

    def mappings(self):
        return FakeMappings(self._rows)


class FakeConnection:
    def __init__(self, results: List[FakeResult] | None = None) -> None:
        self.results = results or []
        self.executed: List[tuple[str, dict[str, Any]]] = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        if self.results:
            return self.results.pop(0)
        return FakeResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeEngine:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def begin(self):
        return self._conn

    def connect(self):
        return self._conn


@pytest.mark.asyncio
async def test_save_dedup():
    conn = FakeConnection()
    store = InstructionsStore(FakeEngine(conn))
    store._tables_ensured = True

    # First save: no current version exists
    conn.results = [
        FakeResult(scalar_value=None),   # get_current
        FakeResult(),                    # UPDATE is_current
        FakeResult(scalar_value=1),    # INSERT new version
    ]
    vid1 = await store.save("content: foo")
    assert vid1 == 1

    # Second save with identical content: should return existing id without INSERT
    conn.results = [
        FakeResult(scalar_value="content: foo"),  # get_current
        FakeResult(scalar_value=1),                # get_current_version_id
    ]
    vid2 = await store.save("content: foo")
    assert vid2 == 1
    # Only 2 new queries (get_current + get_current_version_id), no UPDATE/INSERT
    assert len(conn.executed) == 5


@pytest.mark.asyncio
async def test_agent_memory_store_get():
    conn = FakeConnection()
    store = AgentMemoryStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[{"memory": "test memory content"}]),
    ]
    result = await store.get_memory("conv-1", "AssistantA")
    assert result == "test memory content"


@pytest.mark.asyncio
async def test_agent_memory_store_get_empty():
    conn = FakeConnection()
    store = AgentMemoryStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[]),
    ]
    result = await store.get_memory("conv-1", "AssistantA")
    assert result is None


@pytest.mark.asyncio
async def test_agent_memory_store_set():
    conn = FakeConnection()
    store = AgentMemoryStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult()]
    result = await store.set_memory("conv-1", "AssistantA", "new memory")
    assert result == "new memory"
    assert len(conn.executed) == 1


@pytest.mark.asyncio
async def test_agent_memory_store_append():
    conn = FakeConnection()
    store = AgentMemoryStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[{"memory": "existing"}]),
        FakeResult(),
    ]
    result = await store.append_memory("conv-1", "AssistantA", "appended")
    assert result == "existing\nappended"


@pytest.mark.asyncio
async def test_agent_memory_store_delete():
    conn = FakeConnection()
    store = AgentMemoryStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult(rowcount=1)]
    result = await store.delete_memory("conv-1", "AssistantA")
    assert result is True
