"""Persistent per-assistant workspace folder tool.

Active when the assistant config sets ``workspace.enabled``. The run resolves
the scope directory once (root run) and hands it down via
``context["agent_workspace_dir"]`` so subagents share the same folder even
though they run under derived conversation ids.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.builtin_tools.common import apply_text_edits, read_partial_text

logger = logging.getLogger(__name__)

_READ_TEXT_CAP = 60_000
_LIST_CAP = 500

# Extensions where plain `read` produces garbage — point the model at the
# document-aware actions instead.
_BINARY_HINT_EXTS = {
    ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls",
    ".odt", ".odp", ".ods", ".rtf", ".pptm", ".docm", ".xlsm",
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
}


class AgentWorkspaceTool:
    name = "agent_workspace"

    def __init__(
        self,
        artifact_root: Optional[Path] = None,
        store: Any = None,
        attachment_store: Any = None,
    ) -> None:
        self._store = store
        if self._store is None:
            from app.agent_workspace import AgentWorkspaceStore

            self._store = AgentWorkspaceStore()
        self._artifact_store = None
        if artifact_root is not None:
            from app.artifacts import ArtifactStore

            self._artifact_store = ArtifactStore(Path(artifact_root))
        self._att_store = attachment_store
        self._lo_converter = None
        self._file_locks: Dict[str, asyncio.Lock] = {}

    def _converter(self):
        """LibreOffice converter for office rendering (shared with artifacts)."""
        if self._artifact_store is not None:
            return self._artifact_store.converter
        if self._lo_converter is None:
            from app.artifacts import LibreOfficeConverter

            self._lo_converter = LibreOfficeConverter()
        return self._lo_converter

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Your persistent workspace folder for this assistant. It survives across "
                    "messages (and conversations, depending on scope) — use it to keep working "
                    "files, notes, datasets and results. Paths are RELATIVE to the folder. "
                    "Actions: 'list' (default, path=dir or '.'), 'read' (text files), 'write' "
                    "(create/overwrite text file with 'content'), 'append', 'mkdir', 'delete', "
                    "'move' (path -> to_path), 'edit' (ordered batch of text edits "
                    "on the evolving file — see 'edits'; use 'read' with "
                    "line_numbers=true to get numbered lines first), "
                    "'offer' (attach the file to the chat as a "
                    "download for the user — the right way to hand over a result file), "
                    "'list_uploads' (show files the user uploaded to this chat), "
                    "'import_upload' (copy a user-uploaded file into the workspace — "
                    "'attachment' selects it by id or name, 'to_path' chooses the "
                    "destination, default 'uploads/<name>'), "
                    "'inspect' (structure of PDF/Office docs: slide shapes with "
                    "positions, headings, tables, sheets; 'extract_media'=true also "
                    "writes embedded images/media to extracted/<name>/), "
                    "'view' (SEE pages/slides of PDF/Office files or images as actual "
                    "pictures — 'pages' like '1-3' or '5', 'max_images' up to 8; "
                    "requires a vision-capable model). "
                    "The code sandbox also sees this folder mounted at /agent_ws."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "list", "read", "write", "append", "mkdir", "delete",
                                "move", "edit", "offer", "list_uploads", "import_upload",
                                "inspect", "view",
                            ],
                            "description": "Operation to perform (default 'list').",
                        },
                        "attachment": {
                            "type": "string",
                            "description": "Upload id or filename for 'import_upload'.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path inside the workspace (file or dir).",
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content for 'write'/'append'.",
                        },
                        "edits": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "For 'edit': ordered batch of text edits (max 50, "
                                "all-or-nothing). Each item: {operation: 'insert'|"
                                "'remove'|'replace'} plus its fields — insert: "
                                "content + line (1-based, -1 = append)|after_text; "
                                "remove: lines_start+lines_end|text_snippet; "
                                "replace: old_text + content. Indexes apply to the "
                                "evolving file state — later edits see earlier results."
                            ),
                        },
                        "line_numbers": {
                            "type": "boolean",
                            "description": "For 'read': prefix lines with 'NNNN│ ' numbers (needed for line-based 'edit' ops).",
                        },
                        "to_path": {
                            "type": "string",
                            "description": "Destination relative path for 'move'.",
                        },
                        "pages": {
                            "type": "string",
                            "description": "Pages/slides to render for 'view', e.g. '1-3', '5' or 'all' (default: first 4).",
                        },
                        "max_images": {
                            "type": "integer",
                            "description": "Max page images for 'view' (default 4, hard cap 8).",
                        },
                        "extract_media": {
                            "type": "boolean",
                            "description": "For 'inspect': also extract embedded media files into extracted/<name>/.",
                        },
                    },
                    "required": [],
                },
            },
        }

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _ctx(context: Optional[Dict[str, Any]]) -> tuple:
        ctx = context or {}
        scope_dir = ctx.get("agent_workspace_dir")
        if not scope_dir:
            raise RuntimeError("No agent workspace is configured for this run")
        return Path(scope_dir), ctx.get("agent_workspace_cfg") or {}

    @staticmethod
    def _upload_conversation(ctx: Dict[str, Any]) -> Optional[str]:
        """Conversation id whose uploads may be imported.

        Subagents run under derived ids (``root.sa_N``) — the root id is
        carried through ``agent_workspace_conv`` so uploads resolve to the
        real conversation regardless of run depth.
        """
        return ctx.get("agent_workspace_conv") or ctx.get("conversation_id")

    _MUTATING_ACTIONS = frozenset(
        {"write", "append", "mkdir", "delete", "move", "edit", "offer", "import_upload"}
    )

    def call_conflict_key(self, args: Dict[str, Any]) -> Optional[str]:
        """Serialize same-path mutations inside a parallel tool-call batch."""
        action = str(args.get("action") or "list").strip().lower()
        if action in self._MUTATING_ACTIONS:
            return str(args.get("path") or ".")
        return None

    def _file_lock(self, key: str) -> asyncio.Lock:
        lock = self._file_locks.get(key)
        if lock is None:
            lock = self._file_locks[key] = asyncio.Lock()
        return lock

    def _path_locks(self, scope_dir: Path, *rels: str) -> List[asyncio.Lock]:
        """Locks keyed by resolved workspace path, sorted to a deterministic
        order so multi-path actions (move) cannot deadlock (ABBA)."""
        keys = sorted({str(self._store.resolve(scope_dir, r)) for r in rels})
        return [self._file_lock(k) for k in keys]

    async def execute(
        self,
        action: str = "list",
        path: Optional[str] = None,
        content: Optional[str] = None,
        to_path: Optional[str] = None,
        attachment: Optional[str] = None,
        pages: Optional[str] = None,
        max_images: Optional[int] = None,
        extract_media: Optional[bool] = None,
        edits: Optional[List[Dict[str, Any]]] = None,
        line_numbers: Optional[bool] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        from app.agent_workspace import WorkspaceError

        try:
            scope_dir, cfg = self._ctx(context)
        except RuntimeError as exc:
            return f"[Error: {exc}]"
        action = (action or "list").strip().lower()
        rel = (path or "").strip() or "."

        try:
            if action == "list":
                entries = await _to_thread(self._store.list_entries, scope_dir, rel)
                if not entries:
                    return "(workspace empty)"
                lines = [f"Workspace ({len(entries)} entries{'+' if len(entries) >= _LIST_CAP else ''}):"]
                for e in entries[:_LIST_CAP]:
                    size = f" {e['size']}B" if e.get("size") is not None else ""
                    lines.append(f"  {e['path']}{'/' if e['type'] == 'dir' else ''}{size}")
                return "\n".join(lines)

            if action == "read":
                if Path(rel).suffix.lower() in _BINARY_HINT_EXTS:
                    return (
                        f"'{rel}' is a binary document — use 'inspect' to read its "
                        f"structure (headings, slide shapes, tables, embedded media) "
                        f"or 'view' to see rendered page images."
                    )
                data = await _to_thread(self._store.read, scope_dir, rel)
                text = data.decode("utf-8", errors="replace")
                if line_numbers:
                    return read_partial_text(
                        text, hard_limit=_READ_TEXT_CAP, line_numbers=True
                    )
                if len(text) > _READ_TEXT_CAP:
                    text = text[:_READ_TEXT_CAP] + f"\n[truncated — {len(text)} chars total]"
                return text or "(empty file)"

            if action in ("write", "append"):
                if content is None:
                    return "[Error: 'content' is required for write/append]"
                async with self._path_locks(scope_dir, rel)[0]:
                    entry = await _to_thread(
                        self._store.write, scope_dir, rel, content.encode("utf-8"), cfg, action == "append"
                    )
                verb = "Appended to" if action == "append" else "Wrote"
                return f"{verb} {entry['path']} ({entry['size']} bytes)"

            if action == "mkdir":
                async with self._path_locks(scope_dir, rel)[0]:
                    entry = await _to_thread(self._store.mkdir, scope_dir, rel)
                return f"Created directory {entry['path']}"

            if action == "delete":
                async with self._path_locks(scope_dir, rel)[0]:
                    entry = await _to_thread(self._store.delete, scope_dir, rel)
                return f"Deleted {entry['path']}"

            if action == "move":
                if not to_path:
                    return "[Error: 'to_path' is required for move]"
                locks = self._path_locks(scope_dir, rel, to_path)
                async with locks[0]:
                    async with locks[1]:
                        entry = await _to_thread(self._store.move, scope_dir, rel, to_path)
                return f"Moved to {entry['path']}"

            if action == "edit":
                if Path(rel).suffix.lower() in _BINARY_HINT_EXTS:
                    return (
                        f"'{rel}' is a binary document — 'edit' only works on "
                        "text files. Use 'inspect'/'view' for binary formats."
                    )
                if edits is None:
                    return "[Error: 'edits' is required for edit]"
                async with self._path_locks(scope_dir, rel)[0]:
                    data = await _to_thread(self._store.read, scope_dir, rel)
                    try:
                        # strict — errors='replace' would silently write
                        # U+FFFD back and corrupt non-UTF-8 files
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        return (
                            f"[Error: '{rel}' is not valid UTF-8 — 'edit' only "
                            "works on UTF-8 text files]"
                        )
                    new_text, op_names = apply_text_edits(text, edits)
                    entry = await _to_thread(
                        self._store.write, scope_dir, rel,
                        new_text.encode("utf-8"), cfg, False,
                    )
                return (
                    f"Edited {entry['path']} — applied {len(op_names)} edit(s) "
                    f"({', '.join(op_names)}); {entry['size']} bytes"
                )

            if action == "offer":
                ctx = context or {}
                conversation_id = ctx.get("conversation_id")
                if self._artifact_store is None or not conversation_id:
                    return "[Error: file offering is unavailable in this context]"
                async with self._path_locks(scope_dir, rel)[0]:
                    data = await _to_thread(self._store.read, scope_dir, rel)
                entry = await self._artifact_store.save(
                    conversation_id,
                    str(ctx.get("assistant_name") or ""),
                    Path(rel).name,
                    data,
                    "file",
                )
                return {
                    "ok": True,
                    "content": f"Offered '{entry['name']}' ({entry['size']} bytes) to the user as a download.",
                    "metadata": {"files": [entry]},
                }

            if action == "list_uploads":
                if self._att_store is None:
                    return "[Error: chat uploads are unavailable on this server]"
                conv = self._upload_conversation(context or {})
                if not conv:
                    return "[Error: no conversation context for uploads]"
                metas = await _to_thread(self._att_store.list, conv)
                if not metas:
                    return "(no uploaded files in this chat)"
                lines = [f"User uploads ({len(metas)}):"]
                for m in metas:
                    lines.append(
                        f"  {m.get('id')}  {m.get('name')}  {m.get('size', 0)}B"
                    )
                return "\n".join(lines)

            if action == "import_upload":
                if self._att_store is None:
                    return "[Error: chat uploads are unavailable on this server]"
                conv = self._upload_conversation(context or {})
                if not conv:
                    return "[Error: no conversation context for uploads]"
                meta = await _to_thread(
                    self._find_upload, conv, attachment or path
                )
                if meta is None:
                    return f"[Error: no upload matching '{attachment or path or ''}' — use list_uploads]"
                src = await _to_thread(self._att_store.original_path, conv, meta["id"])
                if src is None:
                    return f"[Error: original file for '{meta.get('name')}' not found]"
                data = await _to_thread(src.read_bytes)
                dest = (to_path or "").strip() or f"uploads/{meta.get('name') or meta['id']}"
                async with self._path_locks(scope_dir, dest)[0]:
                    entry = await _to_thread(self._store.write, scope_dir, dest, data, cfg)
                return (
                    f"Imported '{meta.get('name')}' → {entry['path']} ({entry['size']} bytes). "
                    f"The sandbox sees it at /agent_ws/{entry['path']}."
                )

            if action == "inspect":
                from app.doc_inspect import DocInspectError, extract_embedded_media, inspect_document

                resolved = self._store.resolve(scope_dir, rel)
                try:
                    text = await _to_thread(
                        inspect_document, resolved, converter=self._converter()
                    )
                except DocInspectError as exc:
                    return f"[Error: {exc}]"
                if not extract_media:
                    return text
                media = await _to_thread(extract_embedded_media, resolved)
                if not media:
                    return text + "\n\n(no embedded media found)"
                stem = Path(rel).stem or "doc"
                written: List[str] = []
                for fname, data in media:
                    try:
                        dest = f"extracted/{stem}/{fname}"
                        async with self._path_locks(scope_dir, dest)[0]:
                            entry = await _to_thread(
                                self._store.write, scope_dir, dest, data, cfg
                            )
                        written.append(entry["path"])
                    except Exception as exc:
                        logger.warning("media extract write %s failed: %s", fname, exc)
                if not written:
                    return text + "\n\n[Error: embedded media found but could not be written (quota?)]"
                listing = "\n".join(f"  - {p}" for p in written)
                return (
                    f"{text}\n\nExtracted {len(written)} media file(s) into the workspace "
                    f"(sandbox: /agent_ws/extracted/{stem}/):\n{listing}"
                )

            if action == "view":
                if not cfg.get("vision_read"):
                    return (
                        "[Error: 'view' is disabled for this assistant — the admin must "
                        "enable workspace.vision_read (only sensible for vision-capable "
                        "models). Use 'inspect' for text-level document structure.]"
                    )
                from app.doc_inspect import (
                    DEFAULT_RENDER_IMAGES,
                    DocInspectError,
                    default_cache_dir,
                    render_document,
                )

                resolved = self._store.resolve(scope_dir, rel)
                try:
                    res = await render_document(
                        resolved,
                        pages=pages,
                        converter=self._converter(),
                        cache_dir=default_cache_dir(self._store.root),
                        max_images=max_images or DEFAULT_RENDER_IMAGES,
                    )
                except DocInspectError as exc:
                    return f"[Error: {exc}]"
                kind = "image" if res.source == "image" else "page"
                pages_str = ", ".join(str(p) for p in res.pages_rendered)
                return {
                    "content": (
                        f"Rendered {len(res.images)} {kind} image(s) of '{rel}' "
                        f"({kind} {pages_str} of {res.total_pages} total). "
                        "The images are attached to this result."
                    ),
                    "images": res.images,
                }

            return f"[Error: unknown action '{action}']"
        except WorkspaceError as exc:
            return f"[Error: {exc}]"
        except Exception as exc:
            logger.warning("agent_workspace %s failed on %s: %s", action, rel, exc)
            return f"[Error: {exc}]"

    def _find_upload(self, conversation_id: str, query: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve an upload by id or (case-insensitive) filename."""
        q = (query or "").strip().lower()
        if not q:
            return None
        metas = self._att_store.list(conversation_id)
        for m in metas:
            if str(m.get("id", "")).lower() == q:
                return m
        for m in metas:
            if str(m.get("name", "")).lower() == q:
                return m
        for m in metas:
            name = str(m.get("name", "")).lower()
            if q in name:
                return m
        return None


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)
