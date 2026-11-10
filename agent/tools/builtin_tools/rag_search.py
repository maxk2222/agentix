"""RAG search builtin tool."""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import unquote

from langchain.schema import Document

from agent.tools.builtin_tools.common import normalize_collection_names, read_partial_text

logger = logging.getLogger(__name__)

_READ_HARD_LIMIT = 50_000
_BROWSE_DEFAULT_LIMIT = 15_000


def _pos_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
_OVERLAP_SCAN_MAX = 512
_MIN_OVERLAP = 16
_WEBDAV_CACHE_MARKER = ".webdav_cache/"


class RagSearchTool:
    """Search configured knowledge collections using the RAG retrieval pipeline."""

    name = "rag_search"
    description = (
        "Search configured knowledge collections using the RAG retrieval pipeline. "
        "Returns ranked document chunks with source citations, a list of all documents "
        "in a collection, or the full text of a single document (action='read'). "
        "Workflow: action='list' (depth=1 or folder= for large trees, pattern= to filter by "
        "path) → action='read' with pattern= to locate content inside a document → page "
        "with lines_start/lines_end. Read output is line-numbered; the limit parameter "
        "controls the window size."
    )

    def __init__(self, rag_app: Any) -> None:
        self._rag_app = rag_app
        self._collection_names: Optional[List[str]] = None
        self._max_top_k: int = 10
        self._read_limit: int = _READ_HARD_LIMIT
        self._limit_max: int = _READ_HARD_LIMIT
        self._vs_store: Optional[Any] = None

    def _vs_metrics_store(self) -> Optional[Any]:
        """Lazily build a VectorstoreMetricsStore on the app's engine."""
        if self._vs_store is not None:
            return self._vs_store
        engine = getattr(self._rag_app, "engine", None)
        if engine is None:
            return None
        from app.metrics import VectorstoreMetricsStore
        self._vs_store = VectorstoreMetricsStore(engine)
        return self._vs_store

    async def _record_retrieval(
        self,
        context: Optional[Dict[str, Any]],
        *,
        collection: str,
        retrieval_time_ms: float,
        chunks: int,
        status: str,
        error_type: Optional[str] = None,
        rag_tokens: Optional[int] = None,
    ) -> None:
        """Persist a vectorstore_metrics row for an explicit rag_search call.

        Automatic retrieval metrics already flow through ``retrieval_info`` in
        the chat router; this covers direct tool usage. Best-effort only —
        a metrics failure must never break the tool result.
        """
        conversation_id = (context or {}).get("conversation_id")
        if not conversation_id:
            return
        try:
            store = self._vs_metrics_store()
            if store is None:
                return
            from app.metrics import VectorstoreMetricRecord
            record = VectorstoreMetricRecord(
                conversation_id=str(conversation_id),
                model_name=str((context or {}).get("assistant_name") or "unknown"),
                collection_names=collection,
                retrieval_time_ms=float(retrieval_time_ms),
                chunks_retrieved=int(chunks),
                status=status,
                started_at=datetime.now(timezone.utc).isoformat(),
                error_type=error_type,
                rag_context_tokens=rag_tokens,
            )
            await store.save(record)
        except Exception:
            logger.warning("Failed to record rag_search metric for %s", conversation_id, exc_info=True)

    async def refresh_schema(self) -> None:
        """Reload configured collection names and limits from DB config."""
        try:
            self._collection_names = await self._get_collection_names()
        except Exception:
            self._collection_names = None
        try:
            config = await self._get_tool_config()
            max_top_k = config.get("max_top_k")
            if max_top_k is not None:
                self._max_top_k = max(1, int(max_top_k))
            self._read_limit = _pos_int(config.get("read_limit"), _READ_HARD_LIMIT)
            self._limit_max = _pos_int(config.get("limit_max"), _READ_HARD_LIMIT)
        except Exception:
            pass

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return self.schema_with_config({})

    def schema_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the tool schema using the given config (may contain assistant overrides)."""
        names_raw = config.get("collection_names")
        if names_raw is not None:
            collection_names: Optional[List[str]] = normalize_collection_names(names_raw) or None
        else:
            collection_names = self._collection_names

        max_top_k_raw = config.get("max_top_k")
        if max_top_k_raw is not None:
            max_top_k = max(1, int(max_top_k_raw))
        else:
            max_top_k = self._max_top_k

        read_limit = _pos_int(config.get("read_limit"), self._read_limit)
        limit_max = _pos_int(config.get("limit_max"), self._limit_max)
        schema_max = max(read_limit, limit_max)

        collection_prop: Dict[str, Any] = {
            "type": "string",
            "description": "The target knowledge collection to search.",
        }
        if collection_names:
            collection_prop["enum"] = collection_names

        desc = self.description
        if collection_names:
            desc += f" Configured collections: {', '.join(collection_names)}."

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "list", "read", "missing"],
                            "description": "The action to perform: 'search' for semantic query, 'list' to enumerate all documents, 'read' to fetch the full text of one document, 'missing' to check which files of the collection are not mentioned in a document (coverage diff).",
                            "default": "search",
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query (required when action='search').",
                        },
                        "collection": collection_prop,
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of top results to return for search (default 5, max {max_top_k}).",
                            "default": 5,
                            "maximum": max_top_k,
                        },
                        "source": {
                            "type": "string",
                            "description": "Document to read in full (required when action='read' or 'missing'). Pass the 'Source:' value from search results or the source/filename shown by action='list'. Basenames are accepted.",
                        },
                        "document_collection": {
                            "type": "string",
                            "description": "For action='missing': collection that contains the 'source' document, if different from 'collection'. Default: same collection.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Max characters of output to return (when omitted: {_BROWSE_DEFAULT_LIMIT} for search/list, {read_limit} for read; hard max {schema_max}). Applies to search, list and read results. Request a larger limit to read bigger windows in one call; use offset or lines_start/lines_end to page beyond it.",
                            "maximum": schema_max,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "CHARACTER offset into the output text (0-based; negative values count from end). Applies to search, list and read. This is a character position, NOT a page or chunk index. The output snaps to line boundaries; the footer shows the actual range returned and the next offset/lines_start to continue with.",
                            "default": 0,
                        },
                        "lines_start": {
                            "type": "integer",
                            "description": "First line of output to return (1-based, inclusive). Easier than offset for paging; one list line equals one file/folder entry. Takes priority over offset/limit.",
                        },
                        "lines_end": {
                            "type": "integer",
                            "description": "Last line of output to return (1-based, inclusive). Takes priority over offset/limit.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "For action='list': only list documents inside this subfolder (path relative to the collection root, e.g. 'reports/2024'). Use this to browse large collections folder by folder.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "For action='list': expand folders up to this level (1 = top level only). Folders at this depth are shown collapsed as 'folder/ (N files, M chunks)'. Combine with 'folder' to navigate deep trees.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Case-insensitive substring filter. For action='read': return only matching lines with surrounding context (locate content instead of paging). For action='list': only show documents whose path contains it.",
                        },
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Like 'pattern' but a list of needles checked in one call. For action='read': one labeled match group per pattern — verify many strings at once instead of separate calls. For action='list': documents matching any pattern.",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "For action='read' with 'pattern': context lines around each match (default 2, max 10).",
                            "default": 2,
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "For action='list': include hidden entries (paths with dotfile segments like '.trash-xxx/'). Default false.",
                            "default": False,
                        },
                    },
                    "required": ["collection"],
                },
            },
        }

    async def _get_collection_names(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Read the configured collection names from the tools DB or assistant override."""
        config = await self._get_tool_config(context)
        names = config.get("collection_names")
        return normalize_collection_names(names)

    async def _get_db_tool_config(self) -> Dict[str, Any]:
        """Read the full tool config from the tools DB."""
        tools_store = getattr(self._rag_app, "_tools_store", None)
        if tools_store is None:
            return {}
        try:
            entry = await tools_store.get_tool_by_name(self.name)
            if entry is not None:
                return entry.config or {}
        except Exception:
            pass
        return {}

    async def _get_tool_config(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return effective tool config: DB config merged with assistant override (assistant wins)."""
        base = await self._get_db_tool_config()
        override = (context or {}).get("tool_config", {})
        if override:
            return {**base, **override}
        return base

    @staticmethod
    def _format_results(docs: Sequence[Document]) -> str:
        """Format retrieved documents into a string with source citations."""
        parts: List[str] = []
        for doc in docs:
            meta = doc.metadata or {}
            source = meta.get("source", "")
            page = meta.get("page") or meta.get("page_number", "")
            chunk_idx = meta.get("chunk_index", "")
            content = getattr(doc, "page_content", "") or ""

            header_parts: List[str] = []
            if source:
                header_parts.append(f"Source: {source}")
            if page:
                header_parts.append(f"Page: {page}")
            if chunk_idx != "":
                header_parts.append(f"Chunk: {chunk_idx}")

            if header_parts:
                parts.append(f"{', '.join(header_parts)}\n{content}")
            else:
                parts.append(content)

        return "\n\n".join(parts)

    @staticmethod
    def _relative_source(raw: Any, collection: str = "") -> str:
        """Return the display path of a stored source relative to the collection root.

        Older index runs stored absolute cache paths like
        ``/app/data/.webdav_cache/<collection>/sub/file.pdf``; strip everything
        up to and including the collection folder so listings show
        ``sub/file.pdf``. Other source kinds (bare names, upload paths, URLs)
        are returned unchanged.
        """
        s = unquote(str(raw or ""))
        idx = s.find(_WEBDAV_CACHE_MARKER)
        if idx >= 0:
            s = s[idx + len(_WEBDAV_CACHE_MARKER):]
        if collection and s.startswith(collection + "/"):
            s = s[len(collection) + 1:]
        return s

    @classmethod
    def _normalize_list_entries(
        cls,
        items: List[Dict[str, Any]],
        collection: str,
        folder: Optional[str],
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """Map store entries to ``{rel, raw, chunks, item}`` and apply filters."""
        entries: List[Dict[str, Any]] = []
        for it in items:
            raw = str(it.get("source") or "unknown")
            rel = cls._relative_source(raw, collection)
            chunks_raw = it.get("chunks")
            entries.append(
                {
                    "rel": rel,
                    "raw": raw,
                    "chunks": int(chunks_raw) if chunks_raw is not None else None,
                    "item": it,
                }
            )
        if not include_hidden:
            entries = [
                e
                for e in entries
                if not any(seg.startswith(".") for seg in e["rel"].split("/"))
            ]
        if folder:
            f = unquote(str(folder).strip().lstrip("./").strip("/"))
            entries = [
                e
                for e in entries
                if e["rel"] == f or e["rel"].startswith(f + "/")
            ]
        pats = [p for p in ([pattern] if pattern else []) + (patterns or []) if p]
        if pats:
            needles = [unquote(str(p).strip()).casefold() for p in pats]
            needles = [n for n in needles if n]
            if needles:
                entries = [
                    e for e in entries if any(n in e["rel"].casefold() for n in needles)
                ]
        return entries

    @staticmethod
    def _build_listing(
        entries: List[Dict[str, Any]],
        collection: str,
        folder: Optional[str],
        depth: Optional[int],
        line_for_entry,
        total_entries: Optional[int] = None,
    ) -> str:
        """Render a folder-tree listing with optional folder collapsing by depth."""
        total_files = len(entries)
        chunks_known = [e["chunks"] for e in entries if e["chunks"] is not None]
        total_chunks = sum(chunks_known)
        header = (
            f"Documents in collection '{collection}'"
            if collection else "Documents in collection"
        )
        if folder:
            header += f" under '{folder.strip().strip('/')}/'"
        counts = (
            f"{total_files} of {total_entries} files"
            if total_entries is not None and total_entries != total_files
            else f"{total_files} files"
        )
        counts += f", {total_chunks} chunks" if chunks_known else ""
        lines = [f"{header} ({counts}):"]

        max_depth = int(depth) if depth is not None else None
        if max_depth is not None and max_depth < 1:
            max_depth = 1

        groups: Dict[str, List[Dict[str, Any]]] = {}
        collapsed: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            rel = e["rel"]
            parent_depth = rel.count("/")
            if max_depth is not None and parent_depth >= max_depth:
                ancestor = "/".join(rel.split("/")[:max_depth])
                collapsed.setdefault(ancestor, []).append(e)
            else:
                parent = str(PurePosixPath(rel).parent)
                groups.setdefault("" if parent in (".", "") else parent, []).append(e)

        def _folder_summary(ancestors: List[Dict[str, Any]]) -> str:
            n_files = len(ancestors)
            n_chunks = sum(e["chunks"] or 0 for e in ancestors)
            if any(e["chunks"] is None for e in ancestors):
                return f"{n_files} files"
            return f"{n_files} files, {n_chunks} chunks"

        root_entries = groups.pop("", [])
        if root_entries:
            lines.append("Root:")
            for e in sorted(root_entries, key=lambda x: x["rel"].lower()):
                lines.append(line_for_entry(e))
        for name in sorted(set(collapsed) | set(groups), key=str.lower):
            if name in collapsed:
                lines.append(f"{name}/  ({_folder_summary(collapsed[name])})")
            else:
                lines.append(f"{name}/")
                for e in sorted(groups[name], key=lambda x: x["rel"].lower()):
                    lines.append(line_for_entry(e))
        return "\n".join(lines)

    @classmethod
    def _format_file_list(
        cls,
        files: List[Dict[str, Any]],
        collection: str = "",
        folder: Optional[str] = None,
        depth: Optional[int] = None,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
    ) -> str:
        """Format file metadata into a folder tree; paths are relative to the collection."""
        if not files:
            return "No documents found in this collection."
        entries = cls._normalize_list_entries(
            files, collection, folder, pattern=pattern, patterns=patterns,
            include_hidden=include_hidden,
        )
        if not entries:
            scope = f" under '{folder}/'" if folder else ""
            filt = f" matching '{pattern}'" if pattern else ""
            return f"No documents found{scope}{filt} in this collection."

        def _line(e: Dict[str, Any]) -> str:
            f = e["item"]
            fname = f.get("filename") or PurePosixPath(e["rel"]).name or "unknown"
            updated = f.get("updated_at") or "unknown"
            content_type = f.get("content_type") or "unknown"
            file_hash = f.get("file_hash", "")
            hash_short = file_hash[:8] if file_hash else ""
            return (
                f"  - {fname}"
                f"  |  source: {e['rel']}"
                f"  |  type: {content_type}"
                f"  |  updated: {updated}"
                f"  |  hash: {hash_short}..."
            )

        return cls._build_listing(
            entries, collection, folder, depth, _line, total_entries=len(files)
        )

    @classmethod
    def _format_vector_sources(
        cls,
        sources: List[Dict[str, Any]],
        collection: str = "",
        folder: Optional[str] = None,
        depth: Optional[int] = None,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
    ) -> str:
        """Format vector store sources grouped by directory (folder tree)."""
        if not sources:
            return "No documents found in this collection."
        entries = cls._normalize_list_entries(
            sources, collection, folder, pattern=pattern, patterns=patterns,
            include_hidden=include_hidden,
        )
        if not entries:
            scope = f" under '{folder}/'" if folder else ""
            filt = f" matching '{pattern}'" if pattern else ""
            return f"No documents found{scope}{filt} in this collection."

        rel_counts = Counter(e["rel"] for e in entries)

        def _line(e: Dict[str, Any]) -> str:
            name = PurePosixPath(e["rel"]).name or e["rel"]
            line = f"  - {name}  ({e['chunks']} chunks)  |  source: {e['rel']}"
            # Physical duplicates share one rel — expose the raw path so the
            # agent has a copyable, unambiguous disambiguator.
            if rel_counts[e["rel"]] > 1:
                line += f"  |  path: {e['raw']}"
            return line

        return cls._build_listing(
            entries, collection, folder, depth, _line, total_entries=len(sources)
        )

    async def execute(
        self,
        query: Optional[str] = None,
        collection: str = "",
        top_k: int = 5,
        action: str = "search",
        source: str = "",
        folder: Optional[str] = None,
        depth: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        lines_start: Optional[int] = None,
        lines_end: Optional[int] = None,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        context_lines: Optional[int] = None,
        include_hidden: bool = False,
        document_collection: Optional[str] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        action = action.lower().strip()
        if action not in ("search", "list", "read", "missing"):
            return f"Unknown action '{action}'. Use 'search', 'list', 'read' or 'missing'."

        allowed = await self._get_collection_names(context)
        if not allowed:
            return (
                "No collections configured for the rag_search tool. "
                "Please assign collections in the tool settings."
            )
        if collection not in allowed:
            return (
                f"Collection '{collection}' is not allowed. "
                f"Allowed collections: {', '.join(allowed)}"
            )

        config = await self._get_tool_config(context)
        read_limit = _pos_int(config.get("read_limit"), self._read_limit)
        limit_max = _pos_int(config.get("limit_max"), self._limit_max)
        browse_limit = limit if limit is not None else _BROWSE_DEFAULT_LIMIT

        if action == "missing":
            return await self._execute_missing(
                collection,
                source,
                document_collection=document_collection,
            )

        if action == "list":
            return await self._execute_list(
                collection,
                folder=folder,
                depth=depth,
                limit=browse_limit,
                offset=offset,
                lines_start=lines_start,
                lines_end=lines_end,
                pattern=pattern,
                patterns=patterns,
                include_hidden=include_hidden,
                hard_limit=limit_max,
            )

        if action == "read":
            return await self._execute_read(
                collection,
                source,
                limit=limit,
                offset=offset,
                lines_start=lines_start,
                lines_end=lines_end,
                pattern=pattern,
                patterns=patterns,
                context_lines=context_lines,
                hard_limit=read_limit,
                context=context,
            )

        # action == "search"
        if not query:
            return "A query is required for action='search'."

        effective_top_k = max(1, min(int(top_k), self._max_top_k))

        retriever_builder = getattr(self._rag_app, "_retriever_builder", None)
        if retriever_builder is None:
            return "RAG retriever is not available."

        retriever = await retriever_builder.get_retriever_for_collections(
            [collection],
            retrieval_degraded=False,
            search_k=effective_top_k,
        )
        if retriever is None:
            return f"Retriever for collection '{collection}' is not available."

        retrieval_t0 = time.perf_counter()
        try:
            docs = await retriever.ainvoke(query)
        except Exception as exc:
            logger.warning("RAG search failed for collection '%s': %s", collection, exc)
            await self._record_retrieval(
                context,
                collection=collection,
                retrieval_time_ms=(time.perf_counter() - retrieval_t0) * 1000.0,
                chunks=0,
                status="error",
                error_type=type(exc).__name__,
            )
            return f"Search failed: {exc}"
        retrieval_ms = (time.perf_counter() - retrieval_t0) * 1000.0

        docs = docs[:effective_top_k]
        from app.chat_utils import estimate_rag_context_tokens
        rag_tokens, _ = estimate_rag_context_tokens(docs)
        await self._record_retrieval(
            context,
            collection=collection,
            retrieval_time_ms=retrieval_ms,
            chunks=len(docs),
            status="ok",
            rag_tokens=rag_tokens,
        )

        if not docs:
            return f"No results found for query '{query}' in collection '{collection}'."

        return read_partial_text(
            self._format_results(docs),
            limit=browse_limit,
            offset=offset,
            lines_start=lines_start,
            lines_end=lines_end,
            hard_limit=limit_max,
        )

    async def _execute_list(
        self,
        collection: str,
        *,
        folder: Optional[str] = None,
        depth: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        lines_start: Optional[int] = None,
        lines_end: Optional[int] = None,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
        hard_limit: int = _READ_HARD_LIMIT,
    ) -> str:
        """List all documents in a collection (RawFileStore -> VectorStore fallback).

        The formatted listing is paged with ``read_partial_text`` so large
        collections can be browsed via offset/limit or lines_start/lines_end
        instead of being silently capped at the hard limit.
        """
        engine = getattr(self._rag_app, "engine", None)
        if engine is None:
            return "Database engine not available."

        # Try RawFileStore first
        from app.stores import RawFileStore
        raw_store = RawFileStore(engine)
        try:
            files = await raw_store.list_by_collection(collection)
        except Exception as exc:
            logger.warning("Failed to list files for collection '%s': %s", collection, exc)
            return f"Failed to list documents: {exc}"

        if files:
            return read_partial_text(
                self._format_file_list(
                    files, collection, folder=folder, depth=depth,
                    pattern=pattern, patterns=patterns, include_hidden=include_hidden,
                ),
                limit=limit,
                offset=offset,
                lines_start=lines_start,
                lines_end=lines_end,
                hard_limit=hard_limit,
            )

        # Fallback: query vector store for sources + chunk counts
        manager = await self._get_vector_manager(collection)
        if manager is None:
            return "No documents found in this collection."

        try:
            sources = await manager.list_embedding_sources()
        except Exception as exc:
            logger.warning("Failed to list embedding sources for '%s': %s", collection, exc)
            return f"Failed to list documents: {exc}"

        if not sources:
            return "No documents found in this collection."

        return read_partial_text(
            self._format_vector_sources(
                sources, collection, folder=folder, depth=depth,
                pattern=pattern, patterns=patterns, include_hidden=include_hidden,
            ),
            limit=limit,
            offset=offset,
            lines_start=lines_start,
            lines_end=lines_end,
            hard_limit=hard_limit,
        )

    async def _get_vector_manager(self, collection: str) -> Optional[Any]:
        """Return a metadata-ready VectorStoreManager for the collection."""
        index_manager = getattr(self._rag_app, "_index_manager", None)
        if index_manager is None:
            return None

        manager = getattr(self._rag_app, "vector_managers", {}).get(collection)
        if manager is None:
            manager = index_manager.build_vector_manager(collection)

        try:
            await manager.prepare_metadata_store()
        except Exception as exc:
            logger.warning("Failed to prepare metadata store for '%s': %s", collection, exc)
            return None
        return manager

    async def _resolve_source(self, manager: Any, source_query: str, collection: str = "") -> Any:
        """Resolve a user-supplied source/filename to an exact source identifier.

        Returns ``(resolved_source, None)`` on success, ``(None, error_message)``
        when ambiguous, or ``(None, None)`` when nothing matches.
        """
        chunks = await manager.get_source_chunks(source_query)
        if chunks:
            return source_query, None

        try:
            sources = await manager.list_embedding_sources()
        except Exception:
            sources = []
        all_sources = [s.get("source", "") for s in sources if s.get("source")]
        # Compare URL-decoded basenames so 'My File.pdf' matches 'My%20File.pdf' hrefs,
        # and relative display paths so 'sub/file.pdf' matches stored absolute sources.
        query_base = PurePosixPath(unquote(source_query)).name
        query_rel = unquote(source_query).lstrip("./").strip("/")
        candidates = []
        for s in all_sources:
            rel = self._relative_source(s, collection)
            if (
                s == source_query
                or rel == query_rel
                or PurePosixPath(unquote(s)).name == query_base
                or s.endswith("/" + source_query)
                or unquote(s).endswith("/" + query_rel)
            ):
                candidates.append(s)
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            # Collapse physical duplicates: candidates that all map to the same
            # relative display path are copies of one logical document (e.g.
            # the same file synced into two .webdav_cache roots) — resolve
            # deterministically instead of erroring out.
            rels = {self._relative_source(c, collection) for c in candidates}
            if len(rels) == 1:
                return sorted(candidates)[0], None
            pairs = ", ".join(
                f"{self._relative_source(c, collection)} → {c}" for c in candidates[:10]
            )
            return None, (
                f"Ambiguous source '{source_query}'. Matching documents: {pairs}. "
                "Pass the full 'source:' value."
            )
        return None, None

    def _reassemble_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Reassemble ordered chunks into full text, stripping boundary overlap.

        ``RecursiveCharacterTextSplitter`` duplicates ``chunk_overlap`` chars at
        chunk boundaries; a duplicated prefix of a following chunk is removed —
        but only for overlaps of at least ``_MIN_OVERLAP`` chars so incidental
        single-char boundary matches never drop content. When no overlap is
        found, chunks join verbatim: never lose, never invent whitespace.
        """
        def _key(c: Dict[str, Any]) -> Any:
            idx = c.get("metadata", {}).get("chunk_index")
            try:
                return (0, int(idx)) if idx is not None else (1, 0)
            except (TypeError, ValueError):
                return (1, 0)

        ordered = sorted(chunks, key=_key)
        config = getattr(self._rag_app, "config", None)
        chunk_overlap = int(getattr(config, "chunk_overlap", 200) or 200)
        scan_max = min(_OVERLAP_SCAN_MAX, chunk_overlap + 64)

        text = ""
        for chunk in ordered:
            piece = (chunk.get("document") or "")
            if not piece:
                continue
            if not text:
                text = piece
                continue
            overlap = 0
            upper = min(scan_max, len(piece), len(text))
            for n in range(upper, _MIN_OVERLAP - 1, -1):
                if text.endswith(piece[:n]):
                    overlap = n
                    break
            text += piece[overlap:] if overlap else piece
        return text

    async def _read_raw_text(self, collection: str, source: str) -> Optional[str]:
        """Return the byte-exact UTF-8 text of a source from ``RawFileStore``.

        Tries the stored ``source`` spelling, its collection-relative form and
        the ``collection/rel`` form. Returns ``None`` when no row exists, the
        file is binary (strict decode fails), or the store is unavailable.
        """
        engine = getattr(self._rag_app, "engine", None)
        if engine is None:
            return None
        from app.stores import RawFileStore
        store = RawFileStore(engine)
        rel = self._relative_source(source, collection)
        for cand in dict.fromkeys((source, rel, f"{collection}/{rel}")):
            if not cand:
                continue
            try:
                row = await store.get(collection, cand)
            except Exception:
                row = None
            if not row:
                continue
            data = row.get("file_data")
            if isinstance(data, str):
                return data
            if isinstance(data, (bytes, bytearray)):
                try:
                    return bytes(data).decode("utf-8")
                except (UnicodeDecodeError, ValueError):
                    return None  # binary file exists — reconstruction needed
            return None
        return None

    async def _read_source_text(
        self,
        collection: str,
        resolved: str,
        manager: Any,
    ) -> tuple[str, str, int]:
        """Return ``(text, kind, chunks_used)`` for a resolved source.

        Prefers the byte-exact raw file for text-based documents; falls back
        to chunk reassembly for binary sources or stores without the file.
        """
        raw = await self._read_raw_text(collection, resolved)
        if raw is not None:
            return raw, "raw", 0
        chunks = await manager.get_source_chunks(resolved)
        return self._reassemble_chunks(chunks), "reconstructed", len(chunks)

    async def _execute_missing(
        self,
        collection: str,
        source: str,
        *,
        document_collection: Optional[str] = None,
    ) -> str:
        """Coverage diff: which files of *collection* are not named in *source*.

        Compares the collection's file listing against the full text of the
        overview document. A file counts as covered when its full filename or
        its stem (name without extension) appears in the document —
        case-insensitive; ``matched_by`` reports loose matches.
        """
        doc_collection = document_collection or collection
        if not (source or "").strip():
            return "A source (the overview document) is required for action='missing'."

        names: List[str] = []
        engine = getattr(self._rag_app, "engine", None)
        if engine is not None:
            from app.stores import RawFileStore
            try:
                files = await RawFileStore(engine).list_by_collection(collection)
                names = [
                    f.get("filename") or PurePosixPath(str(f.get("source") or "")).name
                    for f in files
                ]
            except Exception as exc:
                logger.warning("missing: failed to list '%s': %s", collection, exc)
        if not names:
            manager = await self._get_vector_manager(collection)
            if manager is not None:
                try:
                    sources = await manager.list_embedding_sources()
                    names = [
                        PurePosixPath(self._relative_source(s.get("source", ""), collection)).name
                        for s in sources
                        if s.get("source")
                    ]
                except Exception as exc:
                    logger.warning("missing: vector listing failed for '%s': %s", collection, exc)
        names = [n for n in names if n]
        if not names:
            return f"No documents found in collection '{collection}'."

        doc_text = await self._read_raw_text(doc_collection, source.strip())
        if doc_text is None:
            manager = await self._get_vector_manager(doc_collection)
            if manager is not None:
                resolved, resolve_error = await self._resolve_source(manager, source.strip(), doc_collection)
                if resolve_error is not None:
                    return resolve_error
                if resolved is not None:
                    doc_text, _kind, _chunks = await self._read_source_text(doc_collection, resolved, manager)
        if doc_text is None:
            return (
                f"Document '{source}' not found or unreadable in collection "
                f"'{doc_collection}'. Run action='list' to see available documents."
            )

        haystack = doc_text.casefold()
        missing: List[str] = []
        stem_only: List[str] = []
        for name in sorted(set(names)):
            lowered = name.casefold()
            stem = name.rsplit(".", 1)[0].casefold() if "." in name else lowered
            if lowered in haystack:
                continue
            # Stem matching needs a meaningful length — a 1-3 char stem would
            # substring-match almost any prose and hide real gaps.
            if len(stem) >= 4 and stem in haystack:
                stem_only.append(name)
            else:
                missing.append(name)

        total = len(set(names))
        covered = total - len(missing)
        lines = [
            f"Coverage check: '{collection}' ({total} files) vs "
            f"'{doc_collection}/{source}' ({len(doc_text)} chars)",
            "",
        ]
        if missing:
            lines.append(f"Missing ({len(missing)}):")
            lines.extend(f"- {n}" for n in missing)
        else:
            lines.append("Missing: none — all files are covered.")
        lines.append("")
        lines.append(f"Covered: {covered}/{total}")
        if stem_only:
            lines.append(f"Matched by stem only: {', '.join(stem_only)}")
        if missing:
            lines.append(
                "Update the document for each missing file "
                "(e.g. via collection_manager edit) if it should be listed."
            )
        return "\n".join(lines)

    async def _execute_read(
        self,
        collection: str,
        source: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        lines_start: Optional[int] = None,
        lines_end: Optional[int] = None,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        context_lines: Optional[int] = None,
        hard_limit: int = _READ_HARD_LIMIT,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the (byte-exact or reassembled) text of a single document."""
        source = (source or "").strip()
        if not source:
            return (
                "A source is required for action='read'. "
                "Use action='list' to see available documents."
            )

        retrieval_t0 = time.perf_counter()

        async def _rec(status: str, error_type: Optional[str] = None,
                       chunks: int = 0, rag_tokens: Optional[int] = None) -> None:
            await self._record_retrieval(
                context,
                collection=collection,
                retrieval_time_ms=(time.perf_counter() - retrieval_t0) * 1000.0,
                chunks=chunks,
                status=status,
                error_type=error_type,
                rag_tokens=rag_tokens,
            )

        manager = await self._get_vector_manager(collection)
        if manager is None:
            await _rec("error", "not_found")
            return "No documents found in this collection."

        resolved, resolve_error = await self._resolve_source(manager, source, collection)
        if resolve_error is not None:
            await _rec("error", "resolve_error")
            return resolve_error
        if resolved is None:
            raw = await self._read_raw_text(collection, source)
            if raw is None:
                await _rec("error", "not_found")
                return (
                    f"Document '{source}' not found in collection '{collection}'. "
                    "Run action='list' to see available documents."
                )
            full_text, kind, n_chunks = raw, "raw", 0
            resolved = source
        else:
            full_text, kind, n_chunks = await self._read_source_text(
                collection, resolved, manager
            )
        if not full_text.strip():
            await _rec("error", "empty_content")
            if kind == "reconstructed" and n_chunks == 0:
                return f"Document '{resolved}' has no indexed content in collection '{collection}'."
            return f"Document '{resolved}' has no readable text content."

        header_parts = [
            f"Source: {resolved}",
            f"{len(full_text)} chars",
            f"{full_text.count(chr(10)) + 1} lines",
            f"type: {kind}",
        ]
        if kind == "reconstructed":
            header_parts.append(f"chunks: {n_chunks}")
        header = "  |  ".join(header_parts) + "\n\n"
        result = header + read_partial_text(
            full_text,
            limit=limit,
            offset=offset,
            lines_start=lines_start,
            lines_end=lines_end,
            line_numbers=True,
            pattern=pattern,
            patterns=patterns,
            context_lines=context_lines if context_lines is not None else 2,
            hard_limit=hard_limit,
        )
        from app.chat_utils import count_tokens
        await _rec(
            "ok",
            chunks=n_chunks if kind == "reconstructed" else 1,
            rag_tokens=count_tokens(result),
        )
        return result

    async def read_document_text(
        self,
        collection: str,
        source: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Read the full text of a document in a collection.

        Shared entry point for tools that need document content (e.g.
        ``long_document_summarizer``). Tries the byte-exact raw file first —
        covering unindexed collections — then resolves the source against the
        vector store and reassembles chunks when no raw file exists.

        Returns ``(text, None)`` on success or ``(None, error_message)``.
        Records one ``vectorstore_metrics`` row per call when ``context``
        carries a ``conversation_id`` (same ``read`` semantics as
        ``_execute_read``: chunks = indexed chunk count, or 1 for raw files;
        ``rag_context_tokens`` counts the full loaded text).
        """
        source = (source or "").strip()
        if not source:
            return None, "A source is required. Use rag_search action='list' to see available documents."

        retrieval_t0 = time.perf_counter()

        async def _rec(status: str, error_type: Optional[str] = None,
                       chunks: int = 0, rag_tokens: Optional[int] = None) -> None:
            await self._record_retrieval(
                context,
                collection=collection,
                retrieval_time_ms=(time.perf_counter() - retrieval_t0) * 1000.0,
                chunks=chunks,
                status=status,
                error_type=error_type,
                rag_tokens=rag_tokens,
            )

        from app.chat_utils import count_tokens

        raw = await self._read_raw_text(collection, source)
        if raw is not None:
            await _rec("ok", chunks=1, rag_tokens=count_tokens(raw))
            return raw, None

        manager = await self._get_vector_manager(collection)
        if manager is None:
            await _rec("error", "not_found")
            return None, f"No documents found in collection '{collection}'."

        resolved, resolve_error = await self._resolve_source(manager, source, collection)
        if resolve_error is not None:
            await _rec("error", "resolve_error")
            return None, resolve_error
        if resolved is None:
            await _rec("error", "not_found")
            return None, (
                f"Document '{source}' not found in collection '{collection}'. "
                "Run rag_search action='list' to see available documents."
            )

        full_text, kind, n_chunks = await self._read_source_text(collection, resolved, manager)
        if not full_text.strip():
            await _rec("error", "empty_content")
            return None, f"Document '{resolved}' has no readable text content."

        await _rec(
            "ok",
            chunks=n_chunks if kind == "reconstructed" else 1,
            rag_tokens=count_tokens(full_text),
        )
        return full_text, None
