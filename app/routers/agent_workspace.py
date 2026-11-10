"""Read-only API for per-assistant agent workspace folders.

Exposed only when the assistant config sets ``workspace.enabled`` AND
``workspace.api_enabled``. Access mirrors the parent assistant's own rules
(``_model_access_allowed``): privileged API keys, dashboard sessions and
public-chat roles pass first; anonymous callers fall back to the assistant's
``api_accessible`` flag. Scope resolution still applies — a ``user``-scoped
workspace needs a chat identity, a ``conversation``-scoped one needs the
conversation id (unguessable = bearer credential for anonymous convs).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.agent_workspace import (
    DEFAULT_MAX_FILE_BYTES,
    AgentWorkspaceStore,
    WorkspaceError,
    normalize_scope,
)
from app.artifacts import _TEXT_EXTS, mime_for
from app.routers.chat import _attachment_conv_owner_allowed, _model_access_allowed
from app.routers.public_chat_auth import _get_current_chat_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


def _workspace_store(request: Request) -> AgentWorkspaceStore:
    store = getattr(request.app.state, "agent_workspace_store", None)
    if store is None:
        store = AgentWorkspaceStore()
        request.app.state.agent_workspace_store = store
    return store


async def _resolve_workspace_dir(request: Request, assistant: str) -> tuple:
    """Return (store, scope_dir, scope) after full auth — or raise."""
    rag_application = request.app.state.rag_application
    entry: Dict[str, Any] = (rag_application.instructions or {}).get(assistant) or {}
    ws = entry.get("workspace") or {}
    if not ws.get("enabled"):
        raise HTTPException(status_code=404, detail="Workspace not found.")
    if not await _model_access_allowed(request, assistant):
        raise HTTPException(status_code=403, detail="Not allowed to access this workspace.")

    store = _workspace_store(request)
    scope = normalize_scope(ws.get("scope"))
    if scope == "assistant":
        return store, store.scope_dir(assistant, "assistant"), scope
    if scope == "user":
        chat_user = await _get_current_chat_user(request)
        if chat_user is None:
            raise HTTPException(
                status_code=401,
                detail="This workspace is user-scoped — log in to identify your folder.",
            )
        return store, store.scope_dir(assistant, "user", user_id=chat_user["id"]), scope
    # conversation scope — the conv id doubles as the capability for
    # anonymous conversations; owned conversations check the caller.
    conversation_id = request.query_params.get("conversation_id", "").strip()
    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail="conversation_id query parameter is required for conversation-scoped workspaces.",
        )
    if not await _attachment_conv_owner_allowed(request, conversation_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this workspace.")
    return store, store.scope_dir(assistant, "conversation", conversation_id), scope


@router.get("/chat/assistants/{assistant}/workspace/files")
async def workspace_list(assistant: str, request: Request, path: str = ".", limit: int = 200, offset: int = 0):
    """List entries of the assistant's workspace folder (when api_enabled)."""
    store, scope_dir, scope = await _resolve_workspace_dir(request, assistant)
    try:
        entries = store.list_entries(scope_dir, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    logger.info(
        "workspace list: assistant=%s scope=%s dir=%s entries=%d",
        assistant, scope, path, len(entries),
    )
    return {
        "assistant": assistant,
        "scope": scope,
        "path": path,
        "total": len(entries),
        "entries": entries[offset : offset + limit],
    }


@router.get("/chat/assistants/{assistant}/workspace/file")
async def workspace_file(assistant: str, request: Request, path: str):
    """Download a file from the assistant's workspace folder."""
    store, scope_dir, scope = await _resolve_workspace_dir(request, assistant)
    try:
        resolved = store.resolve(scope_dir, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = mime_for(resolved.name)
    logger.info(
        "workspace file: assistant=%s scope=%s path=%s size=%d",
        assistant, scope, path, resolved.stat().st_size,
    )
    return FileResponse(
        resolved,
        media_type=media_type,
        filename=resolved.name,
        content_disposition_type="inline" if media_type.startswith("image/") else "attachment",
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=60"},
    )


@router.get("/chat/assistants/{assistant}/workspace")
async def workspace_info(assistant: str, request: Request):
    """Expose whether the assistant's workspace browser is available."""
    rag_application = request.app.state.rag_application
    entry: Dict[str, Any] = (rag_application.instructions or {}).get(assistant) or {}
    ws = entry.get("workspace") or {}
    if not ws.get("enabled"):
        return {"enabled": False}
    if not await _model_access_allowed(request, assistant):
        raise HTTPException(status_code=403, detail="Not allowed to access this workspace.")
    return {"enabled": True, "scope": normalize_scope(ws.get("scope"))}


@router.get("/chat/assistants/{assistant}/workspace/file/preview")
async def workspace_file_preview(
    assistant: str,
    request: Request,
    path: str,
    pages: Optional[str] = None,
    max_images: int = 4,
):
    """Preview a workspace file like chat artifacts: rendered page images as
    data URIs for PDF/Office/image files, a text outline for text files."""
    store, scope_dir, _scope = await _resolve_workspace_dir(request, assistant)
    try:
        resolved = store.resolve(scope_dir, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    ext = resolved.suffix.lower()
    if ext in _TEXT_EXTS:
        try:
            text = await asyncio.to_thread(
                resolved.read_text, "utf-8", "replace"
            )
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        lines = [f"{i}: {line}" for i, line in enumerate(text.splitlines()[:60], 1)]
        return {
            "name": resolved.name,
            "mime": mime_for(resolved.name),
            "rendered": False,
            "page_count": 0,
            "outline": [{"title": resolved.name, "lines": lines}],
        }

    from app.doc_inspect import (
        DocInspectError,
        default_cache_dir,
        render_document,
    )
    from app.artifacts import LibreOfficeConverter

    try:
        result = await render_document(
            resolved,
            pages=pages,
            converter=LibreOfficeConverter(),
            cache_dir=default_cache_dir(store.root),
            max_images=max(1, min(max_images, 8)),
        )
    except DocInspectError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "name": resolved.name,
        "mime": mime_for(resolved.name),
        "rendered": True,
        "page_count": result.total_pages,
        "pages_rendered": result.pages_rendered,
        "images": result.images,
        "outline": None,
    }


@router.post("/chat/assistants/{assistant}/workspace/file")
async def workspace_file_upload(
    assistant: str,
    request: Request,
    path: str,
    file: UploadFile,
):
    """Upload a file into the resolved workspace folder (quota-enforced)."""
    store, scope_dir, _scope = await _resolve_workspace_dir(request, assistant)
    # Bound the body before buffering — a body beyond the default file cap is
    # rejected without reading it into memory.
    data = await file.read(DEFAULT_MAX_FILE_BYTES + 1)
    if len(data) > DEFAULT_MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {DEFAULT_MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
        )
    rel = f"{path.rstrip('/')}/{file.filename}" if path not in ("", ".", "/") else (file.filename or "upload.bin")
    try:
        entry = await asyncio.to_thread(store.write, scope_dir, rel, data)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info(
        "workspace upload: assistant=%s path=%s size=%d",
        assistant, rel, len(data),
    )
    return {"ok": True, "entry": entry}


@router.post("/chat/assistants/{assistant}/workspace/mkdir")
async def workspace_mkdir(assistant: str, request: Request, path: str):
    """Create a directory inside the resolved workspace folder."""
    store, scope_dir, _scope = await _resolve_workspace_dir(request, assistant)
    try:
        entry = await asyncio.to_thread(store.mkdir, scope_dir, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info("workspace mkdir: assistant=%s path=%s", assistant, path)
    return {"ok": True, "entry": entry}


@router.delete("/chat/assistants/{assistant}/workspace/file")
async def workspace_file_delete(assistant: str, request: Request, path: str):
    """Delete a file or directory (recursive) inside the workspace folder."""
    store, scope_dir, _scope = await _resolve_workspace_dir(request, assistant)
    try:
        entry = await asyncio.to_thread(store.delete, scope_dir, path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    logger.info("workspace delete: assistant=%s path=%s", assistant, path)
    return {"ok": True, "entry": entry}
