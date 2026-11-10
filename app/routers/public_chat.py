"""Public chat configuration router."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.public_chat_config_store import PublicChatConfigStore
from app.routers.branding import _get_branding_store
from app.chat_group_access import resolve_role_allowlist
from app.chat_conversations_store import ChatConversationsStore
from app.messages import MessageStore
from app.auth import require_admin, read_session_cookie
from app.config_access import effective_config_bool
from app.routers.public_chat_auth import (
    _create_public_chat_access_token,
    _get_config,
    _get_current_chat_user,
    _read_public_chat_access_token,
)
from app.chat_group_access import resolve_effective_allowlist

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Public Chat"])


class PublicChatConfigPayload(BaseModel):
    config: Dict[str, Any]


def _get_public_chat_store(request: Request) -> Optional[PublicChatConfigStore]:
    store = getattr(request.app.state, "_public_chat_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = PublicChatConfigStore(engine)
    request.app.state._public_chat_store = store
    return store


def _get_chat_conversations_store(request: Request) -> Optional[ChatConversationsStore]:
    store = getattr(request.app.state, "_chat_conversations_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = ChatConversationsStore(engine)
    request.app.state._chat_conversations_store = store
    return store


def _get_message_store(request: Request) -> Optional[MessageStore]:
    store = getattr(request.app.state, "_message_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = MessageStore(engine)
    request.app.state._message_store = store
    return store


def _get_all_assistant_names(request: Request) -> List[str]:
    """Load all assistant names from the current instructions."""
    try:
        config = getattr(request.app.state, "app_config", None)
        if config is None:
            return []
        import yaml
        path = config.instructions_file
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return [a.get("name") for a in data.get("assistants", []) if a.get("name")]
    except Exception:
        pass
    return []


def _get_assistant_descriptions(request: Request) -> Dict[str, str]:
    """Load assistant name -> description mapping from the current instructions."""
    try:
        config = getattr(request.app.state, "app_config", None)
        if config is None:
            return {}
        import yaml
        path = config.instructions_file
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                a["name"]: (a.get("description") or "").strip()
                for a in data.get("assistants", [])
                if a.get("name")
            }
    except Exception:
        pass
    return {}


def _allowed_user_ids(chat_user: Dict[str, Any]) -> List[str]:
    """Canonical chat user id plus any linked legacy identity.

    Conversations created before the dashboard-session fallback resolved the
    shadow chat user are stored under ``dashboard_user_id`` — keep them
    readable and deletable by accepting both ids for ownership checks.
    """
    ids = [chat_user["id"]]
    alt = chat_user.get("dashboard_user_id")
    if alt and alt not in ids:
        ids.append(alt)
    return ids


def _assistant_workspace_enabled(request: Request, assistant: str) -> bool:
    """workspace.enabled flag for one assistant (instructions live data)."""
    try:
        rag_application = getattr(request.app.state, "rag_application", None)
        entry = (getattr(rag_application, "instructions", None) or {}).get(assistant) or {}
        return bool((entry.get("workspace") or {}).get("enabled"))
    except Exception:
        return False


def _get_assistant_memory_flags(request: Request) -> Dict[str, bool]:
    """Load assistant name -> agent_memory_enabled mapping from instructions."""
    try:
        config = getattr(request.app.state, "app_config", None)
        if config is None:
            return {}
        import yaml
        path = config.instructions_file
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                a["name"]: bool(a.get("agent_memory_enabled"))
                for a in data.get("assistants", [])
                if a.get("name")
            }
    except Exception:
        pass
    return {}


async def _resolve_enabled_assistants(
    request: Request,
    config: Dict[str, Any],
    chat_user: Optional[Dict[str, Any]] = None,
    pct_data: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Assistants the caller may use — group-union, role mapping, or all.

    The effective allowlist comes from ``resolve_effective_allowlist``;
    a ``None`` result means unrestricted. Group/role names are intersected
    with available assistant names so stale entries are never exposed.
    """
    if chat_user is None:
        chat_user = await _get_current_chat_user(request)
    if pct_data is None:
        pct_data = _read_public_chat_access_token(request, _get_config(request))

    available_names = _get_all_assistant_names(request)

    allowlist = await resolve_effective_allowlist(
        request, pct_data=pct_data, chat_user=chat_user
    )
    if allowlist is None and chat_user is None and not (pct_data or {}).get("groups"):
        # Anonymous visitors without a session are governed by the pseudo-
        # role "anonymous" (legacy behavior — applies even before the page
        # holds a minted access token).
        allowlist = await resolve_role_allowlist(
            request, (pct_data or {}).get("role") or "anonymous"
        )
    if allowlist is not None:
        # Mappings can outlive an assistant deleted from instructions —
        # never expose stale names; importantly, do not fall back to all
        # assistants when a configured allowlist is now empty. Iterate the
        # instructions order so the listing is deterministic (the allowlist
        # is a set / union with no inherent order).
        return [name for name in available_names if name in allowlist]

    return available_names


@router.get("/public-chat/config")
async def public_chat_config_get(
    request: Request,
    response: Response,
) -> Dict[str, Any]:
    """Get the current public chat config. Filters assistants by user role."""
    # This response is role/session-specific and changes when assistants are
    # deleted or role mappings change. It must not be cached by browsers or
    # shared intermediaries.
    response.headers["Cache-Control"] = "private, no-store, no-cache, must-revalidate"
    response.headers["Vary"] = "Cookie"
    store = _get_public_chat_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Public chat config store unavailable")
    defaults = await store.get()
    if defaults is None:
        defaults = {}

    # Filter assistants based on current user's role. When the feature is
    # disabled the page is offered no assistants at all — anonymous API
    # callers are unaffected since they never call this endpoint.
    chat_user = await _get_current_chat_user(request)
    app_config = _get_config(request)
    # Read an incoming scoped token FIRST — a group-scoped widget link must
    # not be overwritten by the session role below.
    pct_data = _read_public_chat_access_token(request, app_config)
    feature_enabled = bool(defaults.get("enabled"))
    filtered_assistants = (
        await _resolve_enabled_assistants(request, defaults, chat_user, pct_data)
        if feature_enabled
        else []
    )
    result = dict(defaults)
    result["enabled_assistants"] = filtered_assistants
    descriptions = _get_assistant_descriptions(request)
    result["assistant_descriptions"] = {
        name: descriptions[name]
        for name in filtered_assistants
        if descriptions.get(name)
    }
    memory_flags = _get_assistant_memory_flags(request)
    result["assistant_memory_enabled"] = {
        name: bool(memory_flags.get(name))
        for name in filtered_assistants
    }
    result["assistant_workspace_enabled"] = {
        name: _assistant_workspace_enabled(request, name)
        for name in filtered_assistants
    }
    # Scoped access token — lets the page call /chat/* under public-chat rules
    # while the raw API stays governed by api_accessible. Only issued when the
    # public chat feature is enabled. The incoming scope is preserved: a
    # group-scoped widget link keeps its groups; otherwise we fall back to
    # the session role (or "anonymous").
    if defaults.get("enabled") and app_config is not None:
        if app_config.dashboard_session_secret:
            pct_groups = (pct_data or {}).get("groups") or []
            if pct_groups:
                role = (pct_data or {}).get("role") or (
                    chat_user["role"] if chat_user else "anonymous"
                )
            else:
                role = chat_user["role"] if chat_user else "anonymous"
            result["access_token"] = _create_public_chat_access_token(
                app_config,
                role,
                groups=pct_groups or None,
                link=(pct_data or {}).get("kind") == "link",
            )
    # Tell the page which upload affordances the runtime allows so it can
    # render image/document pickers only when they would be accepted.
    # File attachments require a real identity (anonymous visitors only
    # hold the free public-chat token and must not write files to disk).
    if defaults.get("enabled") and app_config is not None:
        authenticated = (
            chat_user is not None
            or read_session_cookie(request, app_config) is not None
            or getattr(getattr(request, "state", None), "api_key_id", None) is not None
        )
        result["uploads"] = {
            "images": await effective_config_bool(request, "chat_image_upload_enabled"),
            "files": bool(authenticated)
            and await effective_config_bool(request, "chat_file_upload_enabled"),
        }
    # Global branding acts as the default for unset surface-specific values —
    # the page falls back to these when its own logo/favicon/subtitle fields
    # are empty.
    branding_store = _get_branding_store(request)
    if branding_store is not None:
        branding = await branding_store.get() or {}
        result["branding"] = {
            key: branding[key]
            for key in ("app_name", "powered_by", "logo_url", "favicon_url")
            if branding.get(key)
        }
    return result


@router.get("/public-chat/manifest.json")
async def public_chat_manifest(request: Request) -> JSONResponse:
    """PWA manifest for the public chat. Built dynamically so the installed
    app carries the configured platform name and icon."""
    store = _get_public_chat_store(request)
    defaults = (await store.get()) if store is not None else None
    defaults = defaults or {}
    branding_store = _get_branding_store(request)
    branding = (await branding_store.get()) if branding_store is not None else {}
    branding = branding or {}

    name = defaults.get("title") or branding.get("app_name") or "Chat"
    description = defaults.get("subtitle") or branding.get("powered_by") or name
    manifest = {
        "name": name,
        "short_name": name,
        "description": description,
        "start_url": "/chat/",
        "scope": "/chat/",
        "display": "standalone",
        "background_color": "#131d2b",
        "theme_color": "#131d2b",
        "icons": [
            {"src": "/public-chat/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/public-chat/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {
                "src": "/public-chat/icons/icon-maskable-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "maskable",
            },
        ],
    }
    return JSONResponse(
        manifest,
        media_type="application/manifest+json",
        headers={"Cache-Control": "private, max-age=300"},
    )


# PWA app-icon renderer: sizes the configured branding favicon (PNG, SVG or
# remote URL) to the exact bitmap sizes install prompts and home screens
# expect. Results are cached under <data_dir>/cache/pwa-icons keyed by the
# source value, so LibreOffice is only invoked once per icon change.
_PWA_ICON_SPECS = {
    "icon-192.png": (192, False),
    "icon-512.png": (512, False),
    "apple-touch-icon.png": (180, False),
    "icon-maskable-512.png": (512, True),
}
_PWA_ICON_LOCK = asyncio.Lock()
_PWA_ICON_BG = (0x13, 0x1D, 0x2B, 0xFF)  # matches manifest background_color
_MAX_ICON_SRC_BYTES = 8 * 1024 * 1024


def _decode_icon_source(src: str) -> Optional[Tuple[bytes, str]]:
    """Return (bytes, mime) for data-URL sources. None otherwise."""
    if not src or not src.startswith("data:") or ";base64," not in src:
        return None
    mime = src[5 : src.index(";")].strip().lower()
    try:
        raw = base64.b64decode(src.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) > _MAX_ICON_SRC_BYTES:
        return None
    return raw, mime


async def _fetch_icon_source(src: str) -> Optional[Tuple[bytes, str]]:
    """Fetch http(s) icon sources server-side so remote favicons work too."""
    if not src.lower().startswith(("http://", "https://")):
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(src)
            if resp.status_code != 200 or len(resp.content) > _MAX_ICON_SRC_BYTES:
                return None
            mime = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            return resp.content, mime
    except Exception:
        return None


def _soffice_rasterize(svg: bytes) -> Optional[bytes]:
    """Rasterize SVG via the bundled LibreOffice (single instance via lock)."""
    with tempfile.TemporaryDirectory(prefix="pwa-icon-") as tmp:
        src = Path(tmp) / "icon.svg"
        src.write_bytes(svg)
        try:
            proc = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "-env:UserInstallation=file:///tmp/lo-pwa-profile",
                    "--convert-to",
                    'png:draw_png_Export:{"PixelWidth":{"type":"long","value":1024}}',
                    "--outdir",
                    tmp,
                    str(src),
                ],
                capture_output=True,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        out = Path(tmp) / "icon.png"
        if proc.returncode != 0 or not out.exists():
            return None
        return out.read_bytes()


def _compose_icon(raw: bytes, size: int, maskable: bool) -> Optional[bytes]:
    """Center the source on a square canvas and emit exact-size PNG."""
    from PIL import Image

    try:
        im = Image.open(BytesIO(raw)).convert("RGBA")
    except Exception:
        return None
    inner = int(size * 0.8) if maskable else size
    opaque = maskable
    canvas = Image.new("RGBA", (size, size), _PWA_ICON_BG if opaque else (0, 0, 0, 0))
    im.thumbnail((inner, inner), Image.LANCZOS)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


@router.get("/public-chat/icons/{name}")
async def public_chat_icon(request: Request, name: str) -> Response:
    """Branding-aware PWA/touch icons. Falls back to the bundled static icon."""
    spec = _PWA_ICON_SPECS.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown icon")
    size, maskable = spec

    store = _get_public_chat_store(request)
    defaults = (await store.get()) if store is not None else None
    defaults = defaults or {}
    branding_store = _get_branding_store(request)
    branding = (await branding_store.get()) if branding_store is not None else {}
    branding = branding or {}
    src = defaults.get("favicon_url") or branding.get("favicon_url") or ""

    app_config = _get_config(request)
    data_dir = getattr(app_config, "data_dir", None)
    cache_dir = Path(data_dir) / "cache" / "pwa-icons" if data_dir else None
    cache_file = None
    if src and cache_dir is not None:
        key = hashlib.sha1(f"{src}|{name}".encode()).hexdigest()[:24]
        cache_file = cache_dir / f"{key}.png"
        if cache_file.exists():
            return FileResponse(
                cache_file,
                media_type="image/png",
                headers={"Cache-Control": "private, max-age=300"},
            )

    png: Optional[bytes] = None
    decoded = _decode_icon_source(src) if src else None
    if decoded is None and src:
        decoded = await _fetch_icon_source(src)
    if decoded is not None:
        raw, mime = decoded
        head = raw.lstrip()[:512].lower()
        is_svg = "svg" in mime or head.startswith(b"<svg") or b"<svg" in head
        if is_svg:
            async with _PWA_ICON_LOCK:
                raw = await asyncio.to_thread(_soffice_rasterize, raw) or b""
        if raw:
            png = await asyncio.to_thread(_compose_icon, raw, size, maskable)

    if png is None:
        return RedirectResponse(f"/chat/icons/{name}", status_code=307)

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(png)
        except OSError:
            cache_file = None
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.post("/admin/public-chat/config")
async def public_chat_config_save(
    request: Request,
    payload: PublicChatConfigPayload,
) -> Dict[str, Any]:
    """Save public chat config. Admin only."""
    require_admin(request)
    store = _get_public_chat_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Public chat config store unavailable")
    config = dict(payload.config)
    # Strip keys that are injected server-side into the GET response — they
    # must never round-trip back into the stored config.
    for injected_key in (
        "enabled_assistants",
        "assistant_descriptions",
        "assistant_memory_enabled",
        "access_token",
        "uploads",
        "branding",
    ):
        config.pop(injected_key, None)
    await store.save(config)
    return {"success": True}


# ------------------------------------------------------------------------------
# Conversation endpoints for public chat users
# ------------------------------------------------------------------------------

@router.get("/public-chat/conversations")
async def public_chat_conversations(request: Request) -> Dict[str, Any]:
    """List conversations for the current public chat user."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    store = _get_chat_conversations_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")

    items = await store.list_for_user(_allowed_user_ids(chat_user))
    return {"items": items}


@router.get("/public-chat/conversations/{conversation_id}/messages")
async def public_chat_conversation_messages(
    request: Request,
    conversation_id: str,
) -> Dict[str, Any]:
    """Get messages for a specific conversation."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    conv_store = _get_chat_conversations_store(request)
    if conv_store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")

    conv = await conv_store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") not in _allowed_user_ids(chat_user):
        raise HTTPException(status_code=403, detail="Access denied")

    msg_store = _get_message_store(request)
    if msg_store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    messages = await msg_store.get_conversation_messages(conversation_id)
    return {"messages": messages}


@router.get("/public-chat/conversations/{conversation_id}/memory")
async def public_chat_conversation_memory(
    request: Request,
    conversation_id: str,
) -> Dict[str, Any]:
    """Read-only view of the agent memory entries for an owned conversation."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    conv_store = _get_chat_conversations_store(request)
    if conv_store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")

    conv = await conv_store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") not in _allowed_user_ids(chat_user):
        raise HTTPException(status_code=403, detail="Access denied")

    rag_application = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Memory store unavailable")

    from app.stores import AgentMemoryStore
    memories = await AgentMemoryStore(engine).list_memories_for_conversation(conversation_id)
    return {"conversation_id": conversation_id, "memories": memories}


async def _public_memory_access(request: Request, conversation_id: str):
    """Shared auth for public-chat memory writes: signed-in chat user owning
    the conversation, plus a usable memory store."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    conv_store = _get_chat_conversations_store(request)
    if conv_store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")
    conv = await conv_store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") not in _allowed_user_ids(chat_user):
        raise HTTPException(status_code=403, detail="Access denied")
    rag_application = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Memory store unavailable")
    from app.stores import AgentMemoryStore
    return AgentMemoryStore(engine)


@router.put("/public-chat/conversations/{conversation_id}/memory")
async def public_chat_conversation_memory_set(
    request: Request,
    conversation_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Set agent memory for an assistant in an owned public-chat conversation."""
    assistant_name = str(payload.get("assistant_name") or "").strip()
    if not assistant_name:
        raise HTTPException(status_code=400, detail="assistant_name required")
    flags = _get_assistant_memory_flags(request)
    if not flags.get(assistant_name):
        raise HTTPException(status_code=400, detail="Memory not enabled for this assistant")
    store = await _public_memory_access(request, conversation_id)
    await store.set_memory(
        conversation_id, assistant_name, str(payload.get("memory") or "")
    )
    return {"success": True, "conversation_id": conversation_id, "assistant_name": assistant_name}


@router.delete("/public-chat/conversations/{conversation_id}/memory")
async def public_chat_conversation_memory_delete(
    request: Request,
    conversation_id: str,
    assistant_name: str,
) -> Dict[str, Any]:
    """Delete agent memory for an assistant in an owned conversation."""
    store = await _public_memory_access(request, conversation_id)
    deleted = await store.delete_memory(conversation_id, assistant_name)
    return {"success": True, "deleted": deleted}


@router.delete("/public-chat/conversations/{conversation_id}")
async def public_chat_conversation_delete(
    request: Request,
    conversation_id: str,
) -> Dict[str, Any]:
    """Delete a conversation link for the current user."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    store = _get_chat_conversations_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")

    conv = await store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") not in _allowed_user_ids(chat_user):
        raise HTTPException(status_code=403, detail="Access denied")

    # Flag before deleting so a still-finalizing run cannot recreate the link.
    from app.chat_runs import get_run
    run = get_run(conversation_id)
    if run is not None:
        run.conversation_deleted = True
        run.cancel()

    from app.conversation_cleanup import cleanup_conversation_data
    await cleanup_conversation_data(request.app.state, conversation_id)
    return {"success": True}


@router.patch("/public-chat/conversations/{conversation_id}/title")
async def public_chat_conversation_update_title(
    request: Request,
    conversation_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Update the title of a conversation."""
    chat_user = await _get_current_chat_user(request)
    if chat_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    store = _get_chat_conversations_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Conversations store unavailable")

    conv = await store.get_by_conversation_id(conversation_id)
    if conv is None or conv.get("user_id") not in _allowed_user_ids(chat_user):
        raise HTTPException(status_code=403, detail="Access denied")

    title = payload.get("title", "")
    await store.update_title(conversation_id, title)
    return {"success": True}
