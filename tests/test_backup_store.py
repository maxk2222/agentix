"""Tests for app.backup_store."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock

import pytest

from app.backup_store import BackupConfigStore, BackupJobStore


class FakeRow:
    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()


class FakeMappings:
    def __init__(self, rows: List[dict]) -> None:
        self._rows = [FakeRow(**r) for r in rows]

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class FakeResult:
    def __init__(self, rows=None, scalar_value=None, rowcount=None, first_row=None) -> None:
        self._rows = rows or []
        self._scalar = scalar_value
        self._first_row = first_row
        self.rowcount = rowcount

    def scalar(self):
        return self._scalar

    def mappings(self):
        return FakeMappings(self._rows)

    def first(self):
        if self._first_row is not None:
            return self._first_row
        return FakeRow(**self._rows[0]) if self._rows else None


class FakeConnection:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.executed = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        if self.results:
            return self.results.pop(0)
        return FakeResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeEngine:
    def __init__(self, conn) -> None:
        self._conn = conn

    def begin(self):
        return self._conn

    def connect(self):
        return self._conn


def _cfg_row() -> dict:
    return {
        "id": 1, "enabled": True, "schedule_type": "cron",
        "schedule_minutes": 1440, "schedule_cron": "0 3 * * *",
        "timezone": "Europe/Berlin", "webdav_url": "https://sciebo.de/dav",
        "remote_path": "/agentix-backups", "username": "u",
        "password_encrypted": "enc", "retention_count": 5,
        "retention_days": 30, "next_run_at": None, "last_run_at": None,
        "created_at": None, "updated_at": None,
    }


@pytest.mark.asyncio
async def test_config_defaults_when_empty():
    conn = FakeConnection()
    store = BackupConfigStore(FakeEngine(conn))
    store._tables_ensured = True
    conn.results = [FakeResult(rows=[])]
    cfg = await store.get_config()
    assert cfg["enabled"] is False
    assert cfg["password_set"] is False
    assert cfg["retention_count"] == 10


@pytest.mark.asyncio
async def test_config_hides_password():
    conn = FakeConnection()
    store = BackupConfigStore(FakeEngine(conn))
    store._tables_ensured = True
    conn.results = [FakeResult(rows=[_cfg_row()])]
    cfg = await store.get_config()
    assert cfg["password_set"] is True
    assert "password_encrypted" not in cfg
    assert "password" not in cfg


@pytest.mark.asyncio
async def test_config_internal_decrypts_password():
    conn = FakeConnection()
    store = BackupConfigStore(FakeEngine(conn))
    store._tables_ensured = True
    row = _cfg_row()
    from app.secrets import _encrypt_password
    row["password_encrypted"] = _encrypt_password("s3cret")
    conn.results = [FakeResult(rows=[row])]
    cfg = await store.get_config_internal()
    assert cfg["password"] == "s3cret"
    assert "password_encrypted" not in cfg


@pytest.mark.asyncio
async def test_save_config_insert_and_update():
    conn = FakeConnection()
    store = BackupConfigStore(FakeEngine(conn))
    store._tables_ensured = True
    # insert path: select returns nothing, then get_config read
    conn.results = [
        FakeResult(rows=[]),                      # _select -> none
        FakeResult(),                             # insert
        FakeResult(rows=[_cfg_row()]),            # get_config
    ]
    cfg = await store.save_config(enabled=True, webdav_url="https://x", password="pw")
    sqls = [s for s, _ in conn.executed]
    assert any("INSERT" in s for s in sqls)
    assert cfg["enabled"] is True


@pytest.mark.asyncio
async def test_is_due():
    conn = FakeConnection()
    store = BackupConfigStore(FakeEngine(conn))
    store._tables_ensured = True
    conn.results = [FakeResult(first_row=FakeRow(**{"?column?": 1}))]
    assert await store.is_due() is True
    conn.results = [FakeResult(first_row=None)]
    assert await store.is_due() is False


@pytest.mark.asyncio
async def test_job_lifecycle():
    conn = FakeConnection()
    store = BackupJobStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult(scalar_value=7)]
    job_id = await store.create_job("manual", "dashboard")
    assert job_id == 7

    conn.results = [FakeResult()]
    await store.finish_job(
        job_id, "success", filename="f.enc", size_bytes=10, duration_ms=42
    )
    sql, params = conn.executed[-1]
    assert "UPDATE" in sql and params["status"] == "success"

    conn.results = [FakeResult(rows=[{"id": 7, "status": "success"}])]
    job = await store.get_job(7)
    assert job["status"] == "success"


@pytest.mark.asyncio
async def test_job_summary():
    conn = FakeConnection()
    store = BackupJobStore(FakeEngine(conn))
    store._tables_ensured = True
    conn.results = [FakeResult(rows=[{
        "last_success_at": "2026-01-01", "last_error_at": None, "total_jobs": 3,
    }])]
    summary = await store.summary()
    assert summary["total_jobs"] == 3
