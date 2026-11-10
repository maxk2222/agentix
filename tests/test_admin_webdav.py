"""Tests for WebDAV admin endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.routers.admin import (
    webdav_create_source,
    webdav_delete_source,
    webdav_list_sources,
    webdav_sync_source,
    webdav_update_source,
    WebDAVSourcePayload,
    WebDAVUpdatePayload,
)


class FakeSourceStore:
    def __init__(self, sources=None):
        self._sources = sources or []

    async def list_sources(self):
        return self._sources

    async def create_source(self, **kwargs):
        return {"id": 1, **{k: v for k, v in kwargs.items() if k != "password"}}

    async def get_source(self, name):
        for s in self._sources:
            if s.get("name") == name:
                return s
        return None

    async def update_source(self, name, **kwargs):
        return {"id": 1, "name": name, **kwargs}

    async def delete_source(self, name):
        return True

    async def update_sync_status(self, name, status, error=None):
        pass


class FakeMetaStore:
    async def list_files(self, source_name):
        return []

    async def delete_source_files(self, source_name):
        return 0


class FakeRAGApp:
    ready = True
    collection_paths = {}
    vector_managers = {}
    available_collections = set()

    async def delete_knowledge_collection(self, name, delete_folder=True):
        pass


def _make_request(store=None, meta_store=None, rag_app=None) -> Any:
    state = {
        "app_config": SimpleNamespace(data_dir="/tmp"),
    }
    if store:
        state["_webdav_source_store"] = store
    if meta_store:
        state["_webdav_meta_store"] = meta_store
    if rag_app:
        state["rag_application"] = rag_app
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(**state)
        ),
        cookies={},
    )


@pytest.fixture(autouse=True)
def mock_require_permission():
    with patch("app.routers.admin.require_permission", new_callable=AsyncMock) as m:
        yield m


@pytest.mark.asyncio
async def test_webdav_list_sources(mock_require_permission):
    store = FakeSourceStore(sources=[{"name": "sciebo", "webdav_url": "https://sciebo.de"}])
    request = _make_request(store=store)
    result = await webdav_list_sources(request)
    assert len(result) == 1
    assert result[0]["name"] == "sciebo"


@pytest.mark.asyncio
async def test_webdav_create_source(mock_require_permission):
    store = FakeSourceStore()
    request = _make_request(store=store)
    payload = WebDAVSourcePayload(
        name="sciebo",
        webdav_url="https://sciebo.de/dav",
        remote_path="/docs",
        username="user",
        password="secret",
        enabled=True,
    )
    result = await webdav_create_source(request, payload)
    assert result["name"] == "sciebo"
    assert result["convert_to_okf"] is True
    assert "password_encrypted" not in result


@pytest.mark.asyncio
async def test_webdav_create_source_invalid_url(mock_require_permission):
    store = FakeSourceStore()
    request = _make_request(store=store)
    payload = WebDAVSourcePayload(
        name="sciebo",
        webdav_url="http://sciebo.de/dav",
        remote_path="/",
        username="user",
        password="secret",
    )
    with pytest.raises(Exception) as exc_info:
        await webdav_create_source(request, payload)
    assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_webdav_update_source(mock_require_permission):
    store = FakeSourceStore()
    request = _make_request(store=store)
    payload = WebDAVUpdatePayload(webdav_url="https://new.url")
    result = await webdav_update_source(request, "sciebo", payload)
    assert result["webdav_url"] == "https://new.url"


@pytest.mark.asyncio
async def test_webdav_delete_source(mock_require_permission):
    store = FakeSourceStore()
    meta = FakeMetaStore()
    rag = FakeRAGApp()
    request = _make_request(store=store, meta_store=meta, rag_app=rag)
    result = await webdav_delete_source(request, "sciebo")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_webdav_sync_source_not_ready(mock_require_permission):
    store = FakeSourceStore()
    meta = FakeMetaStore()
    request = _make_request(store=store, meta_store=meta)
    with pytest.raises(Exception) as exc_info:
        await webdav_sync_source(request, "sciebo")
    assert "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_webdav_sync_source(mock_require_permission):
    store = FakeSourceStore(sources=[{
        "name": "sciebo", "webdav_url": "https://sciebo.de",
        "remote_path": "/", "username": "u", "password": "p", "enabled": True,
        "convert_to_okf": True,
    }])
    meta = FakeMetaStore()
    rag = FakeRAGApp()
    request = _make_request(store=store, meta_store=meta, rag_app=rag)

    with patch("app.routers.admin.WebDAVSyncEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine.sync_source = AsyncMock(return_value={"success": True, "downloaded": 2, "skipped": 0, "deleted": 0})
        mock_engine_cls.return_value = mock_engine
        result = await webdav_sync_source(request, "sciebo")
        assert result["success"] is True
        assert result["downloaded"] == 2
