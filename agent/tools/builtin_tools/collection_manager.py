"""Collection manager builtin tool."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent.tools.builtin_tools.common import (
    apply_text_edits,
    normalize_collection_names,
    read_partial_text,
)

logger = logging.getLogger(__name__)


class CollectionManagerTool:
    """CRUD .md files in a fixed knowledge collection and trigger re-indexing."""

    name = "collection_manager"
    description = (
        "Manage .md files in configured knowledge collections. "
        "Actions: create, delete, list, read, edit. "
        "For every action provide the target collection name. "
        "create: needs filename + content (overwrites existing). "
        "delete: needs filename. "
        "list: only collection name needed; supports pattern, offset/limit and lines paging. "
        "read: needs filename. Output is line-numbered (NNNN│) — the same line numbers "
        "work directly for edit insert line= / remove lines_start+lines_end. "
        "Without paging args returns the FULL file (up to 15000 chars). "
        "Use pattern to locate content instead of paging. "
        "edit: needs filename + operation. "
        "insert needs content + line OR after_text. "
        "remove needs lines_start+lines_end OR text_snippet. "
        "replace needs old_text + content. "
        "Or pass 'edits': a batch of the same operations applied "
        "sequentially — later edits see the result of earlier ones — and "
        "saved atomically (one failure discards the whole batch). Do NOT "
        "combine 'edits' with top-level 'operation'."
    )

    _READ_HARD_LIMIT = 15_000
    _READ_FULL_LIMIT = 60_000

    def __init__(self, rag_app: Any) -> None:
        self._rag_app = rag_app
        self._collection_names: Optional[List[str]] = None
        self._pending_index: Dict[str, Set[str]] = {}
        self._index_task: Optional[asyncio.Task] = None
        self._index_lock = asyncio.Lock()
        self._index_debounce_seconds: int = 60
        self._file_locks: Dict[str, asyncio.Lock] = {}

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        """Serialize same-file mutations inside a parallel tool-call batch."""
        action = str(args.get("action") or "").strip().lower()
        if action in ("create", "delete", "edit"):
            return f"{args.get('collection')}/{args.get('filename')}"
        return None

    def _file_lock(self, key: str) -> asyncio.Lock:
        """Per-file mutex guarding read-modify-write against concurrent runs."""
        lock = self._file_locks.get(key)
        if lock is None:
            lock = self._file_locks[key] = asyncio.Lock()
        return lock

    async def refresh_schema(self) -> None:
        """Reload configured collection names and debounce timer from DB config."""
        try:
            self._collection_names = await self._get_collection_names()
        except Exception:
            self._collection_names = None
        try:
            config = await self._get_tool_config()
            debounce = config.get("index_debounce_seconds")
            if debounce is not None:
                self._index_debounce_seconds = max(1, int(debounce))
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

        collection_prop: Dict[str, Any] = {
            "type": "string",
            "description": "The target knowledge collection (must be one of the configured collections).",
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
                            "enum": ["create", "delete", "list", "read", "edit"],
                            "description": "The CRUD action to perform.",
                        },
                        "collection": collection_prop,
                        "filename": {
                            "type": "string",
                            "description": "The .md filename (required for create, delete, read, edit).",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text to write or insert/replace. Required for 'create', edit 'insert', and edit 'replace'.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max characters to return for read/list output. Default 15000. Omit to return the whole output (still capped at 15000).",
                            "default": 15000,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Character offset for read/list (0-based; negative values count from end). The output snaps to line boundaries; the footer shows the actual range and the next offset/lines_start.",
                            "default": 0,
                        },
                        "lines_start": {
                            "type": "integer",
                            "description": "First output line to return (1-based, inclusive) for read/list. Takes priority over offset/limit. Omit to start at the beginning.",
                        },
                        "lines_end": {
                            "type": "integer",
                            "description": "Last output line to return (1-based, inclusive) for read/list. Takes priority over offset/limit.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Case-insensitive substring filter. For 'read': return only matching lines with surrounding context (locate content instead of paging). For 'list': only show files whose name contains it.",
                        },
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Like 'pattern' but a list of needles checked in one call. For 'read': one labeled match group per pattern (verifies many strings at once). For 'list': files matching any pattern.",
                        },
                        "full": {
                            "type": "boolean",
                            "description": "For 'read': return up to 60000 chars instead of the 15000 default cap. Use when you intentionally need most of a file at once.",
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "For 'read' with 'pattern': context lines around each match (default 2, max 10).",
                            "default": 2,
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["insert", "remove", "replace"],
                            "description": "Edit operation. 'insert' requires 'content' plus 'line' or 'after_text'. 'remove' requires 'lines_start'+'lines_end' OR 'text_snippet'. 'replace' requires 'old_text' plus 'content'.",
                        },
                        "line": {
                            "type": "integer",
                            "description": "For edit insert only: line number (1-based) to insert BEFORE. -1 = append at end.",
                        },
                        "after_text": {
                            "type": "string",
                            "description": "For edit insert only: existing text snippet to insert AFTER.",
                        },
                        "text_snippet": {
                            "type": "string",
                            "description": "For edit remove only: exact text snippet to delete.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "For edit replace only: exact text to be replaced.",
                        },
                        "edits": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "For 'edit': ordered batch of operations applied sequentially "
                                "to the evolving file (max 50, all-or-nothing). Each item: "
                                "{operation: 'insert'|'remove'|'replace', ...same fields as "
                                "the single-op form — line/after_text/content for insert, "
                                "lines_start/lines_end/text_snippet for remove, "
                                "old_text/content for replace}. Line numbers refer to the "
                                "state left by previous edits — prefer text anchors or run "
                                "line-based edits bottom-up. Do NOT combine with top-level "
                                "'operation'."
                            ),
                        },
                    },
                    "required": ["action", "collection"],
                },
            },
        }

    def _validate_filename(self, filename: Optional[str]) -> str:
        if not filename:
            raise ValueError("filename is required for this action")
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError(f"Invalid filename: {filename}")
        if not filename.lower().endswith(".md"):
            raise ValueError(f"Only .md files are supported, got: {filename}")
        return filename

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

    def _get_raw_store(self) -> Any:
        engine = getattr(self._rag_app, "engine", None)
        if engine is None:
            raise RuntimeError("Database engine not available")
        from app.stores import RawFileStore
        return RawFileStore(engine)

    async def _read_file_content(
        self,
        raw_store: Any,
        collection: str,
        filename: str,
        *,
        strict: bool = False,
    ) -> str:
        # strict=True for read-modify-write paths — decoding a non-UTF-8 file
        # with errors="replace" then saving it back would corrupt the bytes.
        errors = "strict" if strict else "replace"
        source = f"{collection}/{filename}"
        db_file = await raw_store.get(collection, source)
        if db_file is not None:
            data = db_file.get("file_data")
            if isinstance(data, bytes):
                return data.decode("utf-8", errors=errors)
            return str(data)
        target = self._rag_app.config.knowledge_base_root / collection / filename
        if target.exists():
            return target.read_text(encoding="utf-8", errors=errors)
        raise FileNotFoundError(f"'{filename}' not found in collection '{collection}'")

    def _read_partial(
        self,
        text: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        lines_start: Optional[int] = None,
        lines_end: Optional[int] = None,
        line_numbers: bool = False,
        pattern: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        context_lines: Optional[int] = None,
        full: bool = False,
    ) -> str:
        return read_partial_text(
            text,
            limit=limit,
            offset=offset,
            lines_start=lines_start,
            lines_end=lines_end,
            hard_limit=self._READ_FULL_LIMIT if full else self._READ_HARD_LIMIT,
            line_numbers=line_numbers,
            pattern=pattern,
            patterns=patterns,
            context_lines=context_lines if context_lines is not None else 2,
        )

    async def _get_file_metadata(self, raw_store: Any, collection: str, filename: str) -> Dict[str, Any]:
        source = f"{collection}/{filename}"
        updated_at = None
        db_file = await raw_store.get(collection, source)
        if db_file is not None:
            updated_at = db_file.get("updated_at")
            data = db_file.get("file_data")
            if isinstance(data, bytes):
                text = data.decode("utf-8", errors="replace")
            else:
                text = str(data)
        else:
            target = self._rag_app.config.knowledge_base_root / collection / filename
            if target.exists():
                text = target.read_text(encoding="utf-8", errors="replace")
                stat = target.stat()
                updated_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            else:
                return {"total_chars": 0, "total_lines": 0, "updated_at": None}

        return {
            "total_chars": len(text),
            "total_lines": text.count("\n") + 1,
            "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
        }

    async def _save(self, raw_store: Any, collection: str, filename: str, content: str) -> None:
        source = f"{collection}/{filename}"
        content_bytes = content.encode("utf-8")
        await raw_store.save(
            collection_name=collection,
            source=source,
            filename=filename,
            content_type="text/markdown; charset=utf-8",
            data=content_bytes,
        )
        target_dir = self._rag_app.config.knowledge_base_root / collection
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        target.write_text(content, encoding="utf-8")

    def _schedule_index(self, collection: str, filename: str) -> None:
        if self._index_task is not None and not self._index_task.done():
            self._index_task.cancel()
        self._pending_index.setdefault(collection, set()).add(filename)
        self._index_task = asyncio.create_task(self._run_index_debounced())

    async def _run_index_debounced(self) -> None:
        try:
            await asyncio.sleep(self._index_debounce_seconds)
        except asyncio.CancelledError:
            return
        async with self._index_lock:
            pending = dict(self._pending_index)
            self._pending_index.clear()
        for col, filenames in pending.items():
            try:
                await self._rag_app.index_knowledge_files(col, sorted(filenames))
            except Exception as exc:
                logger.warning("Deferred indexing failed for %s: %s", col, exc)

    async def execute(
        self,
        action: str,
        collection: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        action = action.lower().strip()
        allowed = await self._get_collection_names(context)
        if not allowed:
            return (
                "No collections configured for the collection_manager tool. "
                "Please assign collections in the tool settings."
            )
        if collection not in allowed:
            return (
                f"Collection '{collection}' is not allowed. "
                f"Allowed collections: {', '.join(allowed)}"
            )
        raw_store = self._get_raw_store()

        if action == "list":
            files = await raw_store.list_by_collection(collection)
            md_files = sorted(
                [f for f in files if f.get("filename", "").lower().endswith(".md")],
                key=lambda f: f.get("filename", ""),
            )
            pattern = (kwargs.get("pattern") or "").strip().casefold()
            patterns = [p.casefold() for p in (kwargs.get("patterns") or []) if str(p).strip()]
            if pattern:
                patterns = patterns + [pattern]
            if patterns:
                md_files = [f for f in md_files if any(p in f.get("filename", "").casefold() for p in patterns)]
            if not md_files:
                scope = f" matching '{kwargs.get('pattern')}'" if pattern else ""
                return f"Collection '{collection}' contains no .md files{scope}."
            total = len(md_files)
            header = f"Files in '{collection}' ({total}):"
            if pattern:
                header = f"Files in '{collection}' ({total} matching '{kwargs.get('pattern')}'):"
            output_lines = [header]
            show_metadata = len(md_files) <= 50
            for file_entry in md_files:
                fname = file_entry.get("filename", "")
                if show_metadata:
                    meta = await self._get_file_metadata(raw_store, collection, fname)
                    updated = meta.get("updated_at") or "unknown"
                    total_chars = meta.get("total_chars", 0)
                    total_lines = meta.get("total_lines", 0)
                    output_lines.append(
                        f"- {fname}  |  updated: {updated}  |  {total_chars} chars  |  {total_lines} lines"
                    )
                else:
                    updated = file_entry.get("updated_at") or "unknown"
                    output_lines.append(f"- {fname}  |  updated: {updated}")
            return self._read_partial(
                "\n".join(output_lines),
                limit=kwargs.get("limit"),
                offset=kwargs.get("offset"),
                lines_start=kwargs.get("lines_start"),
                lines_end=kwargs.get("lines_end"),
            )

        filename = self._validate_filename(kwargs.get("filename"))
        content = kwargs.get("content")
        source = f"{collection}/{filename}"

        if action == "create":
            if content is None:
                raise ValueError("content is required for create")
            async with self._file_lock(source):
                await self._save(raw_store, collection, filename, content)
            self._schedule_index(collection, filename)
            return (
                f"Saved '{filename}' in collection "
                f"'{collection}'. Indexing scheduled (60s debounce)."
            )

        if action == "delete":
            async with self._file_lock(source):
                deleted = await raw_store.delete(collection, source)
                target = self._rag_app.config.knowledge_base_root / collection / filename
                if target.exists():
                    target.unlink()
                db_result = await self._rag_app.delete_knowledge_source(
                    collection, filename, delete_file=False
                )
            return (
                f"Deleted '{filename}' from collection '{collection}'. "
                f"DB deleted: {deleted}, Vectorstore cleanup: {db_result}"
            )

        if action == "read":
            text = await self._read_file_content(raw_store, collection, filename)
            header = (
                f"File: {collection}/{filename}  |  {len(text)} chars  |  "
                f"{text.count(chr(10)) + 1} lines\n\n"
            )
            return header + self._read_partial(
                text,
                limit=kwargs.get("limit"),
                offset=kwargs.get("offset"),
                lines_start=kwargs.get("lines_start"),
                lines_end=kwargs.get("lines_end"),
                line_numbers=True,
                pattern=kwargs.get("pattern"),
                patterns=kwargs.get("patterns"),
                context_lines=kwargs.get("context_lines"),
                full=bool(kwargs.get("full")),
            )

        if action == "edit":
            edits = kwargs.get("edits")
            if edits is not None:
                if kwargs.get("operation"):
                    raise ValueError(
                        "Pass either 'operation' + op fields OR 'edits' — not both."
                    )
                async with self._file_lock(source):
                    text = await self._read_file_content(
                        raw_store, collection, filename, strict=True
                    )
                    new_text, op_names = apply_text_edits(text, edits)
                    await self._save(raw_store, collection, filename, new_text)
                self._schedule_index(collection, filename)
                return (
                    f"Edited '{filename}' in collection '{collection}' — applied "
                    f"{len(op_names)} edit(s) ({', '.join(op_names)}). Now "
                    f"{new_text.count(chr(10)) + 1} lines / {len(new_text)} chars. "
                    "Indexing scheduled (60s debounce)."
                )
            operation = (kwargs.get("operation") or "").lower().strip()
            if operation not in ("insert", "remove", "replace"):
                raise ValueError(f"Unknown edit operation: {operation}")
            async with self._file_lock(source):
                text = await self._read_file_content(
                    raw_store, collection, filename, strict=True
                )
                new_text, _ = apply_text_edits(
                    text, [{"operation": operation, **{k: kwargs.get(k) for k in (
                        "line", "after_text", "text_snippet", "old_text",
                        "lines_start", "lines_end", "content",
                    )}}]
                )
                await self._save(raw_store, collection, filename, new_text)
            self._schedule_index(collection, filename)
            return (
                f"Edited '{filename}' in collection '{collection}' "
                f"(operation={operation}). Indexing scheduled (60s debounce)."
            )

        raise ValueError(f"Unknown action: {action}")
