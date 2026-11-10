"""Tests for app.webdav_sync."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.webdav_sync import WebDAVClient, WebDAVSyncEngine, _parse_propfind


SAMPLE_PROPFIND = """<?xml version="1.0" encoding="utf-8"?>
<multistatus xmlns="DAV:">
  <response>
    <href>/</href>
    <propstat>
      <prop>
        <resourcetype><collection/></resourcetype>
      </prop>
    </propstat>
  </response>
  <response>
    <href>/doc.txt</href>
    <propstat>
      <prop>
        <getetag>"abc123"</getetag>
        <getcontentlength>1024</getcontentlength>
        <getlastmodified>Mon, 01 Jan 2024 12:00:00 GMT</getlastmodified>
      </prop>
    </propstat>
  </response>
  <response>
    <href>/subdir/</href>
    <propstat>
      <prop>
        <resourcetype><collection/></resourcetype>
      </prop>
    </propstat>
  </response>
</multistatus>
"""


def test_parse_propfind():
    entries = _parse_propfind(SAMPLE_PROPFIND)
    assert len(entries) == 3  # includes collections
    file_entry = next(e for e in entries if not e["is_collection"])
    assert file_entry["href"].endswith("doc.txt")
    assert file_entry["etag"] == '"abc123"'
    assert file_entry["content_length"] == 1024
    assert file_entry["last_modified"] is not None


@pytest.mark.asyncio
async def test_webdav_client_list_files():
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    subdir_propfind = """<?xml version="1.0" encoding="utf-8"?>
<multistatus xmlns="DAV:">
  <response>
    <href>/subdir/</href>
    <propstat>
      <prop>
        <resourcetype><collection/></resourcetype>
      </prop>
    </propstat>
  </response>
  <response>
    <href>/subdir/nested.txt</href>
    <propstat>
      <prop>
        <getetag>"def456"</getetag>
        <getcontentlength>512</getcontentlength>
      </prop>
    </propstat>
  </response>
</multistatus>
"""

    call_count = 0
    async def fake_request(method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.status_code = 207
        # Distinguish root call from subdir call by checking if URL ends with subdir/
        if url.rstrip("/").endswith("subdir"):
            resp.text = subdir_propfind
        else:
            resp.text = SAMPLE_PROPFIND
        return resp

    client._client.request = fake_request
    files = await client.list_files("/")
    assert len(files) == 2
    assert any(f["href"].endswith("doc.txt") for f in files)
    assert any(f["href"].endswith("nested.txt") for f in files)


@pytest.mark.asyncio
async def test_webdav_client_list_files_with_remote_path():
    """Verify that a remote_path like /docs is appended to base_url, not the URL origin."""
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    subdir_propfind = """<?xml version="1.0" encoding="utf-8"?>
<multistatus xmlns="DAV:">
  <response>
    <href>/dav/docs/</href>
    <propstat>
      <prop>
        <resourcetype><collection/></resourcetype>
      </prop>
    </propstat>
  </response>
  <response>
    <href>/dav/docs/note.txt</href>
    <propstat>
      <prop>
        <getetag>"xyz"</getetag>
        <getcontentlength>256</getcontentlength>
      </prop>
    </propstat>
  </response>
</multistatus>
"""

    requested_urls = []
    async def fake_request(method, url, **kwargs):
        requested_urls.append(url)
        resp = MagicMock()
        resp.status_code = 207
        resp.text = subdir_propfind
        return resp

    client._client.request = fake_request
    files = await client.list_files("/docs")
    assert len(files) == 1
    assert files[0]["href"].endswith("note.txt")
    # URL should be base_url + "/" + "docs", not origin + "/docs"
    assert requested_urls[0] == "https://example.com/dav/docs"


@pytest.mark.asyncio
async def test_webdav_client_download():
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    async def fake_request(method, url, **kwargs):
        resp = MagicMock()
        resp.content = b"file contents"
        return resp

    client._client.request = fake_request
    data = await client.download("/doc.txt")
    assert data == b"file contents"


class FakeSourceStore:
    def __init__(self, sources: List[Dict[str, Any]] | None = None):
        self._sources = sources or []

    async def get_source(self, name: str):
        for s in self._sources:
            if s.get("name") == name:
                return s
        return None

    async def update_sync_status(self, name: str, status: str, error: str | None = None):
        pass


class FakeMetaStore:
    def __init__(self, files: List[Dict[str, Any]] | None = None):
        self._files = {f["filename"]: f for f in (files or [])}

    async def list_files(self, source_name: str):
        return list(self._files.values())

    async def upsert_file(self, source_name: str, filename: str, etag: str, **kwargs):
        self._files[filename] = {"filename": filename, "etag": etag, **kwargs}

    async def mark_indexed(
        self, source_name: str, filename: str, index_attempt_source: str | None = None
    ):
        if filename in self._files and index_attempt_source:
            self._files[filename]["index_attempt_source"] = index_attempt_source

    async def delete_file(self, source_name: str, filename: str):
        return self._files.pop(filename, None) is not None

    async def delete_source_files(self, source_name: str):
        count = len(self._files)
        self._files.clear()
        return count


class FakeVectorManager:
    def __init__(self, sources=None):
        self._sources = list(sources or [])
        self.deleted_sources: List[str] = []

    async def prepare_metadata_store(self):
        pass

    async def collection_exists(self):
        return False

    async def initialize(self, docs, embeddings, force=False):
        pass

    async def upsert_source_hash(self, source, source_hash, raw_file_hash=None):
        pass

    async def source_unchanged(self, source, docs):
        return None

    def hash_source_documents(self, docs):
        return "fake-hash"

    async def list_embedding_sources(self):
        return list(self._sources)

    async def delete_source(self, source):
        self.deleted_sources.append(source)
        return {"deleted_state": 1}


class FakeRAGApp:
    ready = True
    config = MagicMock()
    config.system_slow = False
    embeddings = MagicMock()

    def __init__(self, manager=None):
        self.vector_managers = {}
        self.available_collections = set()
        self._index_manager = MagicMock()
        self._index_manager.build_vector_manager = lambda name: manager or FakeVectorManager()
        self._index_manager.load_documents_for_collection = AsyncMock(return_value=[])
        self._index_manager.prepare_splits = MagicMock(return_value=[])
        self._index_manager.normalize_document_metadata = MagicMock(side_effect=lambda col, doc: doc)
        self.deleted_knowledge_sources: List[tuple] = []

    def _refresh_retriever_builder(self):
        pass

    async def delete_knowledge_source(self, collection: str, source: str, *, delete_file: bool = True):
        self.deleted_knowledge_sources.append((collection, source, delete_file))
        return {"deleted_state": 1}

    async def index_knowledge_files(self, collection: str, filenames: list):
        return {"initialized": True, "sources_count": len(filenames), "chunks": 1}


@pytest.mark.asyncio
async def test_sync_source_disabled():
    source_store = FakeSourceStore([{"name": "sciebo", "enabled": False}])
    meta_store = FakeMetaStore()
    engine = WebDAVSyncEngine(source_store, meta_store, MagicMock(data_dir="/tmp"))
    result = await engine.sync_source("sciebo", None)
    assert result["success"] is False
    assert result["error"] == "Source is disabled"


@pytest.mark.asyncio
async def test_sync_source_invalid_url():
    source_store = FakeSourceStore([{"name": "sciebo", "enabled": True, "webdav_url": "http://insecure.de", "username": "u", "password": "p"}])
    meta_store = FakeMetaStore()
    engine = WebDAVSyncEngine(source_store, meta_store, MagicMock(data_dir="/tmp"))
    with pytest.raises(ValueError, match="Only HTTPS"):
        await engine.sync_source("sciebo", None)


@pytest.mark.asyncio
async def test_sync_source_success():
    source_store = FakeSourceStore([{
        "name": "sciebo", "enabled": True,
        "webdav_url": "https://sciebo.de/dav", "remote_path": "/",
        "username": "u", "password": "p", "convert_to_okf": True,
    }])
    meta_store = FakeMetaStore()
    rag = FakeRAGApp()
    config = MagicMock()
    config.data_dir = "/tmp"

    engine = WebDAVSyncEngine(source_store, meta_store, config)

    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_files = AsyncMock(return_value=[
            {
                "href": "/doc.txt", "is_collection": False,
                "etag": "etag1", "content_length": 100,
                "last_modified": None,
            },
            {
                "href": "/slides.pptx", "is_collection": False,
                "etag": "etag2", "content_length": 2048,
                "last_modified": None,
            },
        ])
        mock_client.download = AsyncMock(return_value=b"content")
        mock_client_cls.return_value = mock_client

        with patch("agent.processing.clean_documents", return_value=[]):
            result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    assert result["downloaded"] == 2
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_sync_source_with_okf_conversion():
    source_store = FakeSourceStore([{
        "name": "sciebo", "enabled": True,
        "webdav_url": "https://sciebo.de/dav", "remote_path": "/",
        "username": "u", "password": "p", "convert_to_okf": True,
    }])
    meta_store = FakeMetaStore()
    rag = FakeRAGApp()
    # Make load_documents return a real document
    from langchain.schema import Document
    rag._index_manager.load_documents_for_collection = AsyncMock(return_value=[
        Document(page_content="hello world", metadata={"source": "/doc.txt"}),
    ])
    config = MagicMock()
    config.data_dir = "/tmp"

    engine = WebDAVSyncEngine(source_store, meta_store, config)

    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_files = AsyncMock(return_value=[
            {"href": "/doc.txt", "is_collection": False, "etag": "e1", "content_length": 100, "last_modified": None},
        ])
        mock_client.download = AsyncMock(return_value=b"hello world")
        mock_client_cls.return_value = mock_client

        with patch("agent.processing.clean_documents", side_effect=lambda docs: docs):
            result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    assert result["downloaded"] == 1
    # Verify that the document passed to prepare_splits has OKF metadata
    prepared_docs = rag._index_manager.prepare_splits.call_args[0][2]
    assert len(prepared_docs) == 1
    assert prepared_docs[0].metadata["okf_type"] == "text"
    assert prepared_docs[0].metadata["okf_concept_id"] == "/doc.txt"
    assert "okf_frontmatter" in prepared_docs[0].metadata


# ------------------------------------------------------------------------------
# _rel_href unit tests
# ------------------------------------------------------------------------------
def test_rel_href_strips_base_prefix_and_decodes():
    from app.webdav_sync import _rel_href
    base = "https://sciebo.de/remote.php/dav/files/user"
    assert _rel_href(base, "/remote.php/dav/files/user/Ordner/My%20File.pdf") == "Ordner/My File.pdf"
    assert _rel_href(base, "/remote.php/dav/files/user/doc.pdf") == "doc.pdf"


def test_rel_href_sanitizes_segments():
    from app.webdav_sync import _rel_href
    base = "https://sciebo.de/dav"
    assert _rel_href(base, "/dav/../etc/passwd") == "etc/passwd"
    assert _rel_href(base, "/dav/./a//b.pdf") == "a/b.pdf"


def test_rel_href_full_url_and_missing_prefix():
    from app.webdav_sync import _rel_href
    base = "https://sciebo.de/dav"
    assert _rel_href(base, "https://sciebo.de/dav/x/f.pdf") == "x/f.pdf"
    # href outside the base prefix keeps its (sanitized) path — still unique
    assert _rel_href(base, "/other/root/f.pdf") == "other/root/f.pdf"


# ------------------------------------------------------------------------------
# sync_source pipeline tests (nested downloads, rewrite, deletes, migration)
# ------------------------------------------------------------------------------
def _make_mock_client(remote_files):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.list_files = AsyncMock(return_value=remote_files)
    mock_client.download = AsyncMock(return_value=b"content")
    return mock_client


def _sciebo_source(**overrides):
    src = {
        "name": "sciebo", "enabled": True,
        "webdav_url": "https://sciebo.de/dav", "remote_path": "/",
        "username": "u", "password": "p", "convert_to_okf": True,
    }
    src.update(overrides)
    return src


@pytest.mark.asyncio
async def test_sync_download_preserves_nested_paths(tmp_path):
    """Same-named files in different remote dirs must not overwrite each other."""
    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore()
    rag = FakeRAGApp()
    config = MagicMock()
    config.data_dir = str(tmp_path)

    engine = WebDAVSyncEngine(source_store, meta_store, config)
    engine._index_downloaded = AsyncMock()

    remote = [
        {"href": "/dav/a/x.pdf", "is_collection": False, "etag": "e1",
         "content_length": 7, "last_modified": None},
        {"href": "/dav/b/x.pdf", "is_collection": False, "etag": "e2",
         "content_length": 7, "last_modified": None},
    ]
    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client_cls.return_value = _make_mock_client(remote)
        result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    downloaded = engine._index_downloaded.call_args[0][1]
    paths = sorted(lp for _, lp in downloaded)
    assert len(paths) == 2
    assert paths[0].endswith("a/x.pdf")
    assert paths[1].endswith("b/x.pdf")
    # both written into nested dirs (cache itself is cleaned up afterwards)
    assert (tmp_path / ".webdav_cache" / "sciebo").exists() is False


@pytest.mark.asyncio
async def test_index_downloaded_rewrites_source_to_rel(tmp_path):
    from langchain.schema import Document

    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore()
    rag = FakeRAGApp()
    config = MagicMock()
    config.data_dir = str(tmp_path)

    cache_dir = Path(tmp_path) / ".webdav_cache" / "sciebo"
    (cache_dir / "Ordner").mkdir(parents=True)
    local = cache_dir / "Ordner" / "Doc.pdf"
    local.write_bytes(b"pdf-bytes")

    doc = Document(page_content="text", metadata={"source": str(local.resolve())})
    rag._index_manager.load_documents_for_collection = AsyncMock(return_value=[doc])

    engine = WebDAVSyncEngine(source_store, meta_store, config)
    with patch("agent.processing.clean_documents", side_effect=lambda docs: docs):
        await engine._index_downloaded(
            "sciebo",
            [("/dav/Ordner/Doc.pdf", str(local))],
            rag,
            True,
            remote_meta_by_rel={"Ordner/Doc.pdf": {"etag": "e9", "content_length": 9}},
            rel_by_href={"/dav/Ordner/Doc.pdf": "Ordner/Doc.pdf"},
        )

    args = rag._index_manager.prepare_splits.call_args[0]
    assert args[1] == "Ordner/Doc.pdf"
    meta = args[2][0].metadata
    assert meta["source"] == "Ordner/Doc.pdf"
    assert meta["okf_concept_id"] == "Ordner/Doc.pdf"
    assert meta["okf_frontmatter"]["etag"] == "e9"
    assert meta["okf_frontmatter"]["file_size"] == 9


@pytest.mark.asyncio
async def test_sync_delete_uses_relative_source(tmp_path):
    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore(files=[
        {"filename": "/dav/old.pdf", "etag": "old1"},
        {"filename": "/dav/keep.pdf", "etag": "same"},
    ])
    rag = FakeRAGApp()
    config = MagicMock()
    config.data_dir = str(tmp_path)

    engine = WebDAVSyncEngine(source_store, meta_store, config)
    engine._index_downloaded = AsyncMock()

    remote = [{"href": "/dav/keep.pdf", "is_collection": False, "etag": "same",
               "content_length": 1, "last_modified": None}]
    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client_cls.return_value = _make_mock_client(remote)
        result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    # stored source values are relative paths, not raw hrefs
    assert ("sciebo", "old.pdf", False) in rag.deleted_knowledge_sources


@pytest.mark.asyncio
async def test_sync_forces_reindex_when_rel_not_stored(tmp_path):
    """etag-matched files with non-canonical stored sources are re-downloaded."""
    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore(files=[
        {"filename": "/dav/doc.pdf", "etag": "e1"},
    ])
    # manager reports only a legacy cache-path source → rel missing → reindex
    manager = FakeVectorManager(sources=[
        {"source": "/abs/data/.webdav_cache/sciebo/doc.pdf", "chunks": 3},
    ])
    rag = FakeRAGApp()
    rag.vector_managers["sciebo"] = manager
    config = MagicMock()
    config.data_dir = str(tmp_path)

    engine = WebDAVSyncEngine(source_store, meta_store, config)
    engine._index_downloaded = AsyncMock()

    remote = [{"href": "/dav/doc.pdf", "is_collection": False, "etag": "e1",
               "content_length": 1, "last_modified": None}]
    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client_cls.return_value = _make_mock_client(remote)
        result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    assert result["downloaded"] == 1  # forced despite etag match
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_sync_does_not_repeat_failed_migration_attempt(tmp_path):
    """A successful sync with no extracted documents should not redownload forever."""
    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore(files=[
        {"filename": "/dav/bad.pdf", "etag": "e1"},
    ])
    manager = FakeVectorManager(sources=[
        {"source": "/legacy/cache/bad.pdf", "chunks": 2},
    ])
    rag = FakeRAGApp()
    rag.vector_managers["sciebo"] = manager
    rag._index_manager.load_documents_for_collection = AsyncMock(return_value=[])
    config = MagicMock()
    config.data_dir = str(tmp_path)
    engine = WebDAVSyncEngine(source_store, meta_store, config)

    remote = [{
        "href": "/dav/bad.pdf", "is_collection": False, "etag": "e1",
        "content_length": 1, "last_modified": None,
    }]
    mock_client = _make_mock_client(remote)
    with patch("app.webdav_sync.WebDAVClient", return_value=mock_client):
        first = await engine.sync_source("sciebo", rag)
        second = await engine.sync_source("sciebo", rag)

    assert first["downloaded"] == 1
    assert second["downloaded"] == 0
    assert second["skipped"] == 1
    assert mock_client.download.await_count == 1
    assert meta_store._files["/dav/bad.pdf"]["index_attempt_source"] == "bad.pdf"


@pytest.mark.asyncio
async def test_sync_prunes_stale_sources_without_downloads(tmp_path):
    """Stored sources not matching a current remote rel are deleted even when
    every remote file is etag-skipped (downloaded == 0) — previously the prune
    lived inside _index_downloaded so stale cache-path entries survived
    forever once no file needed downloading."""
    source_store = FakeSourceStore([_sciebo_source()])
    meta_store = FakeMetaStore(files=[
        {"filename": "/dav/Doc.pdf", "etag": "e1",
         "index_attempt_source": "Doc.pdf"},
    ])
    manager = FakeVectorManager(sources=[
        {"source": "Doc.pdf", "chunks": 2},
        {"source": "/abs/data/.webdav_cache/sciebo/legacy.pdf", "chunks": 3},
    ])
    rag = FakeRAGApp()
    rag.vector_managers["sciebo"] = manager
    config = MagicMock()
    config.data_dir = str(tmp_path)

    engine = WebDAVSyncEngine(source_store, meta_store, config)
    engine._index_downloaded = AsyncMock()

    remote = [{"href": "/dav/Doc.pdf", "is_collection": False, "etag": "e1",
               "content_length": 1, "last_modified": None}]
    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client_cls.return_value = _make_mock_client(remote)
        result = await engine.sync_source("sciebo", rag)

    assert result["success"] is True
    assert result["downloaded"] == 0  # etag-skipped — prune must still run
    assert "/abs/data/.webdav_cache/sciebo/legacy.pdf" in manager.deleted_sources
    assert "Doc.pdf" not in manager.deleted_sources


@pytest.mark.asyncio
async def test_sync_source_cancelled():
    source_store = FakeSourceStore([{
        "name": "sciebo", "enabled": True,
        "webdav_url": "https://sciebo.de/dav", "remote_path": "/",
        "username": "u", "password": "p", "convert_to_okf": True,
    }])
    meta_store = FakeMetaStore()
    rag = FakeRAGApp()
    config = MagicMock()
    config.data_dir = "/tmp"

    cancel_store = {"sciebo": True}
    progress_store = {}
    engine = WebDAVSyncEngine(source_store, meta_store, config, progress_store=progress_store, cancel_store=cancel_store)

    with patch("app.webdav_sync.WebDAVClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_files = AsyncMock(return_value=[
            {"href": "/doc.txt", "is_collection": False, "etag": "e1", "content_length": 100, "last_modified": None},
        ])
        mock_client.download = AsyncMock(return_value=b"content")
        mock_client_cls.return_value = mock_client

        result = await engine.sync_source("sciebo", rag)

    assert result["success"] is False
    assert result["cancelled"] is True
    assert result["error"] == "Cancelled by user"
    assert progress_store["sciebo"]["phase"] == "cancelled"


# ------------------------------------------------------------------------------
# write-side client methods (used by the backup feature)
# ------------------------------------------------------------------------------
def test_webdav_client_url_origin_preserves_port():
    assert WebDAVClient("https://example.com:8443/dav", "u", "p")._url_origin() == "https://example.com:8443"
    assert WebDAVClient("https://sciebo.de/dav", "u", "p")._url_origin() == "https://sciebo.de"


@pytest.mark.asyncio
async def test_webdav_client_upload_sends_put(tmp_path):
    from app.webdav_sync import _aiter_file  # noqa: F401

    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    calls = []
    async def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        resp = MagicMock()
        resp.status_code = 201
        return resp

    client._client.request = fake_request
    f = tmp_path / "b.enc"
    f.write_bytes(b"archive")
    await client.upload("backups/b.enc", f)
    method, url, kwargs = calls[0]
    assert method == "PUT"
    assert url == "https://example.com/dav/backups/b.enc"
    assert kwargs["headers"]["Content-Length"] == "7"


@pytest.mark.asyncio
async def test_webdav_client_upload_error_raises():
    from app.webdav_sync import WebDAVError
    from pathlib import Path
    import tempfile

    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    async def fake_request(method, url, **kwargs):
        resp = MagicMock()
        resp.status_code = 507  # insufficient storage
        return resp

    client._client.request = fake_request
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"x")
        fname = Path(f.name)
    try:
        with pytest.raises(WebDAVError):
            await client.upload("b.enc", fname)
    finally:
        fname.unlink()


@pytest.mark.asyncio
async def test_webdav_client_ensure_collection_creates_nested():
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    calls = []
    async def fake_request(method, url, **kwargs):
        calls.append(url)
        resp = MagicMock()
        resp.status_code = 201  # created
        resp.text = ""
        return resp

    client._client.request = fake_request
    await client.ensure_collection("/backups/pre-restore")
    assert calls == [
        "https://example.com/dav/backups",
        "https://example.com/dav/backups/pre-restore",
    ]


@pytest.mark.asyncio
async def test_webdav_client_ensure_collection_existing_ok():
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    async def fake_request(method, url, **kwargs):
        resp = MagicMock()
        if method == "MKCOL":
            resp.status_code = 405  # already exists
        else:  # PROPFIND stat check
            resp.status_code = 207
            resp.text = """<?xml version="1.0"?>
<multistatus xmlns="DAV:"><response><href>/dav/backups</href>
<propstat><prop><resourcetype><collection/></resourcetype></prop></propstat>
</response></multistatus>"""
        return resp

    client._client.request = fake_request
    await client.ensure_collection("/backups")  # must not raise


@pytest.mark.asyncio
async def test_webdav_client_stat_and_delete():
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    async def fake_request(method, url, **kwargs):
        resp = MagicMock()
        if method == "PROPFIND":
            resp.status_code = 207
            resp.text = """<?xml version="1.0"?>
<multistatus xmlns="DAV:"><response><href>/dav/b.enc</href>
<propstat><prop><resourcetype/><getcontentlength>42</getcontentlength></prop></propstat>
</response></multistatus>"""
        else:
            resp.status_code = 204
        return resp

    client._client.request = fake_request
    info = await client.stat("b.enc")
    assert info["content_length"] == 42
    assert await client.delete("b.enc") is True


@pytest.mark.asyncio
async def test_webdav_client_download_to_file(tmp_path):
    client = WebDAVClient("https://example.com/dav", "user", "pass")
    client._client = MagicMock()

    chunks = [b"hello ", b"world"]

    class FakeStream:
        async def __aenter__(self):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()

            async def aiter(n):
                for c in chunks:
                    yield c
            resp.aiter_bytes = aiter
            return resp

        async def __aexit__(self, *a):
            return None

    client._client.stream = MagicMock(return_value=FakeStream())
    dest = tmp_path / "out.bin"
    written = await client.download_to_file("/b.enc", dest)
    assert written == 11
    assert dest.read_bytes() == b"hello world"
