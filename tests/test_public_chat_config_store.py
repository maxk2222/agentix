"""Tests for app.public_chat_config_store."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.public_chat_config_store import PublicChatConfigStore


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
async def test_get_empty():
    conn = FakeConnection()
    store = PublicChatConfigStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult(rows=[])]
    result = await store.get()
    assert result is None


@pytest.mark.asyncio
async def test_save_and_get():
    conn = FakeConnection()
    store = PublicChatConfigStore(FakeEngine(conn))
    store._tables_ensured = True

    config = {"enabled": True, "enabled_assistants": ["TestAssistant"]}
    conn.results = [FakeResult()]
    await store.save(config)
    assert len(conn.executed) == 1

    json_content = conn.executed[0][1].get("json_content")
    assert json_content is not None
    assert json.loads(json_content) == config


@pytest.mark.asyncio
async def test_get_returns_latest():
    conn = FakeConnection()
    store = PublicChatConfigStore(FakeEngine(conn))
    store._tables_ensured = True

    config_json = json.dumps({"enabled": True, "title": "Public Chat"})
    conn.results = [
        FakeResult(scalar_value=config_json),
    ]
    result = await store.get()
    assert result == {"enabled": True, "title": "Public Chat"}


@pytest.mark.asyncio
async def test_json_roundtrip_with_nested_colors():
    conn = FakeConnection()
    store = PublicChatConfigStore(FakeEngine(conn))
    store._tables_ensured = True

    config = {
        "enabled": True,
        "enabled_assistants": ["A", "B"],
        "colors": {"primary": "#123456", "background": "#ffffff"},
    }
    conn.results = [FakeResult()]
    await store.save(config)

    json_content = conn.executed[0][1].get("json_content")
    parsed = json.loads(json_content)
    assert parsed["colors"]["primary"] == "#123456"
