"""WebDAV client and incremental sync engine for external knowledge sources."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

import httpx
from langchain.schema import Document

from app.webdav_store import WebDAVFileMetaStore, WebDAVSourceStore

logger = logging.getLogger(__name__)

# WebDAV namespace
DAV_NS = "{DAV:}"

class WebDAVError(Exception):
    """Raised when a WebDAV write operation fails (upload, mkdir, delete)."""


async def _aiter_file(path: Path, chunk_size: int = 4 * 1024 * 1024):
    """Async iterator over a local file for streaming PUT bodies."""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# Supported file extensions for knowledge indexing
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx",
    ".doc", ".odt",
    ".xls", ".xlsx",
    ".ppt", ".pptx",
    ".csv",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}


def _rel_href(base_url: str, href: str) -> str:
    """Map a server href to a remote-relative, URL-decoded path.

    ``/remote.php/dav/files/u/Ordner/My%20File.pdf`` with base_url
    ``https://host/remote.php/dav/files/u`` becomes ``Ordner/My File.pdf``.
    Path segments like ``..``/``.``/empty are dropped; the result is safe to
    join under a local directory.
    """
    h = href.strip()
    if h.startswith("http"):  # some servers return full URLs in hrefs
        with contextlib.suppress(Exception):
            h = httpx.URL(h).path
    with contextlib.suppress(Exception):
        prefix = httpx.URL(base_url).path.rstrip("/")
        if h == prefix:
            h = "/"
        elif prefix and h.startswith(prefix + "/"):
            h = h[len(prefix):]
    parts = [
        p for p in PurePosixPath(unquote(h)).parts
        if p not in ("", "/", ".", "..")
    ]
    return "/".join(parts)


class WebDAVClient:
    """Async WebDAV client using httpx."""

    def __init__(
        self, base_url: str, username: str, password: str, timeout: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "WebDAVClient":
        self._client = httpx.AsyncClient(
            auth=(self.username, self.password), timeout=self._timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _url_origin(self) -> str:
        """Return scheme://host[:port] from base_url, e.g. https://sciebo.de."""
        parsed = httpx.URL(self.base_url)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.host}{port}"

    def _relative_href(self, href: str) -> str:
        """Strip the base_url path prefix from an absolute href."""
        parsed = httpx.URL(self.base_url)
        prefix = parsed.path
        if href.startswith(prefix):
            rel = href[len(prefix):]
            return "/" + rel.lstrip("/")
        return href

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if self._client is None:
            raise RuntimeError("WebDAVClient not opened as context manager")
        # If path is absolute (starts with /), prepend only the URL origin
        # so server-root hrefs like /remote.php/dav/files/user/Notes/ map correctly.
        if path.startswith("/"):
            url = self._url_origin() + path
        else:
            url = self.base_url + "/" + path
        return await self._client.request(method, url, **kwargs)

    async def list_files(self, remote_path: str = "/") -> List[Dict[str, Any]]:
        """Recursively list all files under remote_path using PROPFIND."""
        # Strip leading / so the path is appended to base_url, not the URL origin
        remote_path = remote_path.lstrip("/")
        files: List[Dict[str, Any]] = []
        await self._propfind_recursive(remote_path, files)
        return files

    async def _propfind_recursive(self, path: str, result: List[Dict[str, Any]]) -> None:
        body = """<?xml version="1.0" encoding="utf-8"?>
<propfind xmlns="DAV:"><allprop/></propfind>"""
        resp = await self._request(
            "PROPFIND",
            path,
            content=body,
            headers={"Depth": "1"},
        )
        if resp.status_code not in (207, 200):
            logger.warning("PROPFIND %s returned %s", path, resp.status_code)
            return
        entries = _parse_propfind(resp.text)
        # Compute the relative version of the current path for self-skip
        if path.startswith("/"):
            current_rel = self._relative_href(path).rstrip("/")
        else:
            current_rel = "/" + path.rstrip("/")
        for entry in entries:
            href = entry.get("href", "")
            if not href:
                continue
            # Skip the directory itself by comparing relative hrefs
            rel_href = self._relative_href(href).rstrip("/")
            if rel_href == current_rel or rel_href == current_rel.lstrip("/"):
                continue
            if entry.get("is_collection", False):
                await self._propfind_recursive(href, result)
            else:
                result.append(entry)

    async def download(self, path: str) -> bytes:
        resp = await self._request("GET", path)
        resp.raise_for_status()
        return resp.content

    async def download_to_file(self, path: str, dest: Path) -> int:
        """Stream a remote file to disk. Returns bytes written."""
        if self._client is None:
            raise RuntimeError("WebDAVClient not opened as context manager")
        url = self._url_origin() + path if path.startswith("/") else self.base_url + "/" + path
        written = 0
        async with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
        return written

    async def stat(self, path: str) -> Optional[Dict[str, Any]]:
        """PROPFIND Depth:0 for a single resource. Returns None when absent."""
        body = """<?xml version="1.0" encoding="utf-8"?>
<propfind xmlns="DAV:"><allprop/></propfind>"""
        resp = await self._request(
            "PROPFIND",
            path.lstrip("/"),
            content=body,
            headers={"Depth": "0"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        entries = _parse_propfind(resp.text)
        return entries[0] if entries else None

    async def ensure_collection(self, remote_path: str) -> None:
        """Recursively create the collection (MKCOL) if it does not exist."""
        remote_path = remote_path.strip("/")
        if not remote_path:
            return
        current = ""
        for segment in remote_path.split("/"):
            current = f"{current}/{segment}"
            resp = await self._request("MKCOL", current.lstrip("/"))
            if resp.status_code in (201, 301):
                continue
            # 405 = collection already exists; some servers answer 409 on
            # the final existing segment as well — verify via stat.
            if resp.status_code in (405, 409):
                info = await self.stat(current)
                if info and info.get("is_collection"):
                    continue
            raise WebDAVError(
                f"MKCOL {current} failed with status {resp.status_code}"
            )

    async def upload(self, remote_path: str, local_path: Path) -> None:
        """PUT a local file to a remote path with Content-Length."""
        if self._client is None:
            raise RuntimeError("WebDAVClient not opened as context manager")
        size = local_path.stat().st_size
        resp = await self._request(
            "PUT",
            remote_path.lstrip("/"),
            content=_aiter_file(local_path),
            headers={"Content-Length": str(size)},
        )
        if resp.status_code not in (200, 201, 204):
            raise WebDAVError(
                f"Upload of '{local_path.name}' failed with status {resp.status_code}"
            )

    async def delete(self, path: str) -> bool:
        """DELETE a remote resource. Returns True when it was removed."""
        resp = await self._request("DELETE", path.lstrip("/"))
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True


def _parse_propfind(xml_text: str) -> List[Dict[str, Any]]:
    """Parse a WebDAV multistatus XML response into a list of entries."""
    root = ET.fromstring(xml_text.encode("utf-8"))
    entries: List[Dict[str, Any]] = []
    for response in root.findall(f"{DAV_NS}response"):
        href_elem = response.find(f"{DAV_NS}href")
        href = href_elem.text if href_elem is not None else ""
        propstat = response.find(f"{DAV_NS}propstat")
        if propstat is None:
            continue
        prop = propstat.find(f"{DAV_NS}prop")
        if prop is None:
            continue
        res_type = prop.find(f"{DAV_NS}resourcetype")
        is_collection = res_type is not None and res_type.find(f"{DAV_NS}collection") is not None
        etag_elem = prop.find(f"{DAV_NS}getetag")
        etag = etag_elem.text if etag_elem is not None else ""
        length_elem = prop.find(f"{DAV_NS}getcontentlength")
        length = int(length_elem.text) if length_elem is not None and length_elem.text else None
        modified_elem = prop.find(f"{DAV_NS}getlastmodified")
        last_modified = None
        if modified_elem is not None and modified_elem.text:
            try:
                last_modified = datetime.strptime(
                    modified_elem.text, "%a, %d %b %Y %H:%M:%S GMT"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        entries.append({
            "href": href,
            "is_collection": is_collection,
            "etag": etag or "",
            "content_length": length,
            "last_modified": last_modified,
        })
    return entries


class WebDAVSyncEngine:
    """Incremental sync from WebDAV to vector store."""

    def __init__(
        self,
        source_store: WebDAVSourceStore,
        meta_store: WebDAVFileMetaStore,
        config: Any,
        progress_store: Optional[Dict[str, Dict[str, Any]]] = None,
        cancel_store: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.source_store = source_store
        self.meta_store = meta_store
        self.config = config
        self._progress_store = progress_store
        self._cancel_store = cancel_store

    def _set_progress(self, name: str, phase: str, current: int = 0, total: int = 0, message: str = "") -> None:
        if self._progress_store is not None:
            self._progress_store[name] = {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
            }

    def _is_cancelled(self, name: str) -> bool:
        if self._cancel_store is not None:
            return self._cancel_store.get(name, False)
        return False

    async def sync_source(self, source_name: str, rag_app: Any) -> Dict[str, Any]:
        """Sync a single WebDAV source. Returns summary dict."""
        source = await self.source_store.get_source(source_name)
        if source is None:
            raise ValueError(f"WebDAV source '{source_name}' not found")
        if not source.get("enabled", True):
            return {"success": False, "error": "Source is disabled"}

        url = source["webdav_url"]
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS WebDAV URLs are allowed")

        await self.source_store.update_sync_status(source_name, "running")
        self._set_progress(source_name, "listing", message="Scanning remote...")

        try:
            async with WebDAVClient(
                url, source["username"], source["password"]
            ) as client:
                remote_files = await client.list_files(source.get("remote_path", "/"))
        except Exception as exc:
            logger.exception("Failed to list WebDAV files for '%s'", source_name)
            await self.source_store.update_sync_status(source_name, "error", str(exc))
            self._set_progress(source_name, "error", message=str(exc))
            return {"success": False, "error": str(exc)}

        # Filter supported extensions
        remote_files = [
            f for f in remote_files
            if Path(f["href"]).suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        existing_meta = await self.meta_store.list_files(source_name)
        existing_by_href = {m["filename"]: m for m in existing_meta}

        # Canonical remote-relative source paths (folder structure preserving)
        rel_by_href = {e["href"]: _rel_href(url, e["href"]) for e in remote_files}
        expected_rels = {r for r in rel_by_href.values() if r}

        cache_dir = Path(self.config.data_dir) / ".webdav_cache" / source_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[Tuple[str, str]] = []  # (href, local_path)
        skipped: List[str] = []
        to_delete: List[str] = []

        current_hrefs = {f["href"] for f in remote_files}
        for old_href in existing_by_href:
            if old_href not in current_hrefs:
                to_delete.append(old_href)

        # Sources already stored under canonical relative paths. Files whose
        # rel is missing are force-reindexed — but only while the index still
        # contains non-current sources (legacy cache-path entries). Once the
        # index is fully migrated (stored ⊆ expected), etag-skipping applies
        # again so permanently unloadable files are not retried every sync.
        stored_set: Set[str] = set()
        mgr = rag_app.vector_managers.get(source_name) if rag_app is not None else None
        if mgr is None and rag_app is not None and getattr(rag_app, "ready", False):
            with contextlib.suppress(Exception):
                mgr = rag_app._index_manager.build_vector_manager(source_name)
                rag_app.vector_managers[source_name] = mgr
                rag_app.available_collections.add(source_name)
        if mgr is not None:
            with contextlib.suppress(Exception):
                await mgr.prepare_metadata_store()
                stored_set = {
                    s["source"] for s in await mgr.list_embedding_sources() if s.get("source")
                }
        migration_mode = bool(stored_set - expected_rels)

        # Determine which files need downloading
        to_download = []
        forced_reindex = 0
        for entry in remote_files:
            href = entry["href"]
            rel = rel_by_href.get(href)
            old = existing_by_href.get(href)
            needs_reindex = (
                migration_mode
                and bool(rel)
                and rel not in stored_set
                and (old or {}).get("index_attempt_source") != rel
            )
            if old and old.get("etag") == entry.get("etag") and not needs_reindex:
                skipped.append(href)
                continue
            if needs_reindex and old and old.get("etag") == entry.get("etag"):
                forced_reindex += 1
            to_download.append(entry)
        if forced_reindex:
            logger.info(
                "Re-indexing %d unchanged file(s) of '%s' (canonical-source migration).",
                forced_reindex, source_name,
            )

        self._set_progress(source_name, "downloading", 0, len(to_download), "Downloading files...")

        # Download new or changed files
        async with WebDAVClient(
            url, source["username"], source["password"]
        ) as client:
            for i, entry in enumerate(to_download):
                if self._is_cancelled(source_name):
                    logger.info("Sync of '%s' cancelled by user", source_name)
                    self._set_progress(source_name, "cancelled", message="Cancelled by user")
                    await self.source_store.update_sync_status(source_name, "cancelled", "Cancelled by user")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    return {"success": False, "error": "Cancelled by user", "cancelled": True}
                href = entry["href"]
                try:
                    data = await client.download(href)
                    rel = rel_by_href.get(href) or Path(unquote(href)).name
                    local_path = cache_dir.joinpath(*PurePosixPath(rel).parts)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(data)
                    downloaded.append((href, str(local_path)))
                except Exception as exc:
                    logger.warning("Failed to download '%s': %s", href, exc)
                self._set_progress(source_name, "downloading", i + 1, len(to_download), "Downloading files...")

        # Index downloaded files
        indexing_completed = False
        if downloaded and rag_app is not None and getattr(rag_app, "ready", False):
            if self._is_cancelled(source_name):
                logger.info("Sync of '%s' cancelled by user before indexing", source_name)
                self._set_progress(source_name, "cancelled", message="Cancelled by user")
                await self.source_store.update_sync_status(source_name, "cancelled", "Cancelled by user")
                shutil.rmtree(cache_dir, ignore_errors=True)
                return {"success": False, "error": "Cancelled by user", "cancelled": True}
            self._set_progress(source_name, "indexing", 0, len(downloaded), "Indexing...")
            try:
                convert_to_okf = bool(source.get("convert_to_okf", True))
                remote_meta_by_rel = {
                    rel_by_href[e["href"]]: e for e in remote_files if rel_by_href.get(e["href"])
                }
                await self._index_downloaded(
                    source_name, downloaded, rag_app, convert_to_okf,
                    remote_meta_by_rel, rel_by_href,
                )
                indexing_completed = True
            except Exception as exc:
                logger.exception("Failed to index WebDAV files for '%s'", source_name)
                await self.source_store.update_sync_status(source_name, "error", str(exc))
                self._set_progress(source_name, "error", message=str(exc))
                return {"success": False, "error": str(exc)}

        # Update metadata
        for href, _ in downloaded:
            entry = next((e for e in remote_files if e["href"] == href), {})
            await self.meta_store.upsert_file(
                source_name,
                href,
                entry.get("etag", ""),
                content_length=entry.get("content_length"),
                last_modified=entry.get("last_modified"),
            )
            await self.meta_store.mark_indexed(
                source_name,
                href,
                rel_by_href.get(href) if indexing_completed else None,
            )

        # Delete removed files from metadata and vector store
        for href in to_delete:
            await self.meta_store.delete_file(source_name, href)
            if rag_app is not None and getattr(rag_app, "ready", False):
                rel = _rel_href(url, href)
                if rel:
                    try:
                        await rag_app.delete_knowledge_source(source_name, rel, delete_file=False)
                    except Exception as exc:
                        logger.warning("Failed to delete vector source '%s': %s", rel, exc)

        # Prune stored sources that no longer map to a current remote file
        # (legacy cache-path entries, leftovers of broken deletes). Must run
        # even when this sync downloaded nothing — otherwise etag-skipped runs
        # never reach _index_downloaded and stale entries persist forever.
        if mgr is not None:
            await self._prune_stale_sources(mgr, expected_rels)

        # Cleanup cache
        shutil.rmtree(cache_dir, ignore_errors=True)

        await self.source_store.update_sync_status(source_name, "success")
        self._set_progress(source_name, "done", len(downloaded), len(downloaded), "Sync complete")
        return {
            "success": True,
            "downloaded": len(downloaded),
            "skipped": len(skipped),
            "deleted": len(to_delete),
        }

    async def _prune_stale_sources(self, manager: Any, keep: Set[str]) -> None:
        """Delete embedding sources that no longer map to a current remote file."""
        try:
            for row in await manager.list_embedding_sources():
                stale = row.get("source")
                if stale and stale not in keep:
                    await manager.delete_source(stale)
        except Exception as exc:
            logger.warning("Failed to prune stale embedding sources: %s", exc)

    async def _index_downloaded(
        self,
        source_name: str,
        downloaded: List[Tuple[str, str]],
        rag_app: Any,
        convert_to_okf: bool = True,
        remote_meta_by_rel: Optional[Dict[str, Dict[str, Any]]] = None,
        rel_by_href: Optional[Dict[str, str]] = None,
    ) -> None:
        """Index downloaded files into the vector store."""
        from agent.indexing import IndexManager

        cache_dir = Path(self.config.data_dir) / ".webdav_cache" / source_name
        # Load documents from the temp folder, reporting per-file progress so the
        # UI doesn't sit at "Indexing 0/N" while parsers/OCR run.
        index_manager = rag_app._index_manager
        documents = await index_manager.load_documents_for_collection(
            source_name,
            cache_dir,
            progress_cb=lambda done, total, _path: self._set_progress(
                source_name, "indexing", done, total, "Parsing documents..."
            ),
        )
        if not documents:
            return

        # Rewrite local cache paths to canonical remote-relative sources so
        # folder structure survives in the index (source, source_path, filename).
        cache_resolved = cache_dir.resolve()
        rel_by_href = rel_by_href or {}
        local_to_rel = {
            str(Path(lp).resolve()): rel_by_href[h]
            for h, lp in downloaded if rel_by_href.get(h)
        }

        def _to_rel(raw: object) -> Optional[str]:
            if not raw:
                return None
            try:
                resolved = Path(str(raw)).resolve()
            except Exception:
                return None
            rel = local_to_rel.get(str(resolved))
            if rel is None:
                with contextlib.suppress(Exception):
                    rel = resolved.relative_to(cache_resolved).as_posix()
            return rel

        for doc in documents:
            meta = dict(doc.metadata or {})
            for key in ("source", "source_path", "filename"):
                rel = _to_rel(meta.get(key))
                if rel:
                    meta[key] = rel
            doc.metadata = meta

        if convert_to_okf:
            from app.okf_converter import OKFConverter

            remote_meta_by_rel = remote_meta_by_rel or {}
            okf_docs: List[Document] = []
            for doc in documents:
                src = str(doc.metadata.get("source", "")) if doc.metadata else ""
                meta = remote_meta_by_rel.get(src, {})
                okf_docs.append(OKFConverter.convert_document(doc, src, meta))
            documents = okf_docs

        from agent.processing import clean_documents
        cleaned = clean_documents(documents)
        cleaned = [
            index_manager.normalize_document_metadata(source_name, doc)
            for doc in cleaned
        ]

        # Group by source
        sources: Dict[str, List[Document]] = {}
        for doc in cleaned:
            meta = dict(doc.metadata or {})
            src = str(meta.get("source") or "unknown")
            sources.setdefault(src, []).append(doc)

        # Build or get vector manager
        manager = rag_app.vector_managers.get(source_name)
        if manager is None:
            manager = index_manager.build_vector_manager(source_name)
            rag_app.vector_managers[source_name] = manager
            rag_app.available_collections.add(source_name)

        embeddings = (
            rag_app.embeddings
            if not rag_app.config.system_slow
            else rag_app.embeddings  # Slow wrapper already applied elsewhere
        )
        # Actually we need the slow wrapper if configured
        from agent.embeddings import SlowSystemEmbeddingWrapper
        embeddings = (
            SlowSystemEmbeddingWrapper(rag_app.embeddings)
            if rag_app.config.system_slow
            else rag_app.embeddings
        )

        await manager.prepare_metadata_store()
        collection_exists = await manager.collection_exists()

        fingerprints: Dict[str, str] = {}
        for href, local_path in downloaded:
            rel = rel_by_href.get(href)
            path = Path(local_path)
            if rel and path.exists():
                data = path.read_bytes()
                fingerprints[rel] = hashlib.sha256(data).hexdigest()

        if not collection_exists:
            all_splits: List[Document] = []
            for i, (src, docs) in enumerate(sources.items()):
                all_splits.extend(index_manager.prepare_splits(source_name, src, docs))
                self._set_progress(source_name, "indexing", i + 1, len(sources), "Indexing...")
            self._set_progress(
                source_name, "indexing", len(sources), len(sources),
                f"Embedding {len(all_splits)} chunks..."
            )
            await manager.initialize(all_splits, embeddings, force=False)
            for src, docs in sources.items():
                raw_hash = fingerprints.get(src)
                await manager.upsert_source_hash(
                    src, manager.hash_source_documents(docs), raw_file_hash=raw_hash
                )
        else:
            await manager.initialize([], embeddings, force=False)
            for i, (src, docs) in enumerate(sources.items()):
                changed_hash = await manager.source_unchanged(src, docs)
                if changed_hash is None:
                    raw_hash = fingerprints.get(src)
                    if raw_hash is not None:
                        stored = await manager._get_source_hash(src)
                        if stored is not None:
                            await manager.upsert_source_hash(src, stored, raw_file_hash=raw_hash)
                    self._set_progress(source_name, "indexing", i + 1, len(sources), "Indexing...")
                    continue
                doc_splits = index_manager.prepare_splits(source_name, src, docs)
                await manager.upsert_documents(
                    src, doc_splits, embeddings,
                    source_hash=changed_hash,
                    raw_file_hash=fingerprints.get(src),
                )
                self._set_progress(source_name, "indexing", i + 1, len(sources), "Indexing...")

        # Refresh retriever builder
        rag_app._refresh_retriever_builder()
