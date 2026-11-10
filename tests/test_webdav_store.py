"""Tests for app.webdav_store."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.webdav_store import WebDAVFileMetaStore, WebDAVSourceStore, _encrypt_password, _decrypt_password


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
async def test_webdav_source_store_list():
    conn = FakeConnection()
    store = WebDAVSourceStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[{
            "id": 1, "name": "sciebo", "webdav_url": "https://sciebo.de/dav",
            "remote_path": "/docs", "username": "user", "enabled": True,
            "convert_to_okf": True,
            "last_sync_at": None, "last_sync_status": None, "last_sync_error": None,
            "created_at": None, "updated_at": None,
        }]),
    ]
    sources = await store.list_sources()
    assert len(sources) == 1
    assert sources[0]["name"] == "sciebo"
    assert sources[0]["convert_to_okf"] is True
    # Password should NOT be in list output
    assert "password_encrypted" not in sources[0]


@pytest.mark.asyncio
async def test_webdav_source_store_create_and_get():
    conn = FakeConnection()
    store = WebDAVSourceStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[{
            "id": 1, "name": "sciebo", "webdav_url": "https://sciebo.de/dav",
            "remote_path": "/docs", "username": "user", "enabled": True,
            "convert_to_okf": True,
            "last_sync_at": None, "last_sync_status": None, "last_sync_error": None,
            "created_at": None, "updated_at": None,
        }]),
    ]
    result = await store.create_source(
        name="sciebo",
        webdav_url="https://sciebo.de/dav",
        remote_path="/docs",
        username="user",
        password="secret",
        enabled=True,
        convert_to_okf=True,
    )
    assert result["name"] == "sciebo"
    assert result["convert_to_okf"] is True
    # Check that password was encrypted in the executed SQL
    executed_params = conn.executed[-1][1]
    encrypted = executed_params.get("password_encrypted")
    assert encrypted is not None
    assert encrypted != "secret"

    # Now get_source should return decrypted password
    conn.results = [
        FakeResult(rows=[{
            "id": 1, "name": "sciebo", "webdav_url": "https://sciebo.de/dav",
            "remote_path": "/docs", "username": "user",
            "password_encrypted": encrypted, "enabled": True, "convert_to_okf": True,
            "last_sync_at": None, "last_sync_status": None, "last_sync_error": None,
            "created_at": None, "updated_at": None,
        }]),
    ]
    source = await store.get_source("sciebo")
    assert source is not None
    assert source["password"] == "secret"
    assert source["convert_to_okf"] is True


@pytest.mark.asyncio
async def test_webdav_source_store_update():
    conn = FakeConnection()
    store = WebDAVSourceStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [
        FakeResult(rows=[{
            "id": 1, "name": "sciebo", "webdav_url": "https://new.url",
            "remote_path": "/", "username": "user", "enabled": True, "convert_to_okf": True,
            "last_sync_at": None, "last_sync_status": None, "last_sync_error": None,
            "created_at": None, "updated_at": None,
        }]),
    ]
    result = await store.update_source("sciebo", webdav_url="https://new.url")
    assert result is not None
    assert result["webdav_url"] == "https://new.url"
    assert result["convert_to_okf"] is True


@pytest.mark.asyncio
async def test_webdav_source_store_delete():
    conn = FakeConnection()
    store = WebDAVSourceStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult(rowcount=1)]
    deleted = await store.delete_source("sciebo")
    assert deleted is True


@pytest.mark.asyncio
async def test_webdav_source_store_sync_status():
    conn = FakeConnection()
    store = WebDAVSourceStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult()]
    await store.update_sync_status("sciebo", "running")
    assert "UPDATE" in conn.executed[-1][0]

    conn.results = [FakeResult()]
    await store.update_sync_status("sciebo", "success", error=None)
    assert "UPDATE" in conn.executed[-1][0]


@pytest.mark.asyncio
async def test_webdav_file_meta_store():
    conn = FakeConnection()
    store = WebDAVFileMetaStore(FakeEngine(conn))
    store._tables_ensured = True

    conn.results = [FakeResult()]
    await store.upsert_file("sciebo", "/doc.txt", "etag1", content_length=1024)
    assert len(conn.executed) == 1

    conn.results = [
        FakeResult(rows=[{
            "filename": "/doc.txt", "etag": "etag1",
            "content_length": 1024, "last_modified": None, "indexed_at": None,
            "index_attempt_source": None,
        }]),
    ]
    files = await store.list_files("sciebo")
    assert len(files) == 1
    assert files[0]["etag"] == "etag1"

    conn.results = [FakeResult()]
    await store.mark_indexed("sciebo", "/doc.txt", "docs/doc.txt")
    assert len(conn.executed) == 3
    assert conn.executed[-1][1]["index_attempt_source"] == "docs/doc.txt"

    conn.results = [FakeResult(rowcount=1)]
    deleted = await store.delete_file("sciebo", "/doc.txt")
    assert deleted is True

    conn.results = [FakeResult(rowcount=3)]
    count = await store.delete_source_files("sciebo")
    assert count == 3


def test_encryption_roundtrip():
    plain = "my-secret-password-123"
    encrypted = _encrypt_password(plain)
    assert encrypted != plain
    decrypted = _decrypt_password(encrypted)
    assert decrypted == plain
