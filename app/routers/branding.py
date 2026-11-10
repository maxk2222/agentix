"""Branding router — public read, permission-gated write.

The dashboard (including the pre-login page), the public chat page and
externally embedded widgets all read branding via ``GET /branding``. Writes go
through ``POST /admin/branding`` and require the ``branding:write`` permission.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app.auth import read_session_cookie, require_permission
from app.branding_store import BrandingStore
from app.routers.widget import _resolve_acao

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Branding"])

# Reject payloads whose serialized form exceeds this size — branding can carry
# data-URL logos (~1-2MB), anything much bigger is a mistake or abuse.
_MAX_BRANDING_JSON_BYTES = 3 * 1024 * 1024

# Theme color values the frontend can apply as CSS custom properties.
_COLOR_VALUE_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|rgba?\([^)]{1,60}\)|transparent|[a-zA-Z]{3,20})$")

# CSS custom property names the theme editor is allowed to set per theme.
_ALLOWED_THEME_VARS = {
    "bg",
    "card",
    "surface",
    "surface_hover",
    "text",
    "text_secondary",
    "text_muted",
    "border",
    "primary",
    "danger",
    "success",
    "warning",
    "info",
}


class BrandingPayload(BaseModel):
    config: Dict[str, Any]


def _get_branding_store(request: Request) -> Optional[BrandingStore]:
    store = getattr(request.app.state, "_branding_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = BrandingStore(engine)
    request.app.state._branding_store = store
    return store


def _sanitize_theme(theme: Any) -> Dict[str, Dict[str, str]]:
    """Keep only known var names with plausible color values, per theme."""
    cleaned: Dict[str, Dict[str, str]] = {}
    if not isinstance(theme, dict):
        return cleaned
    for mode in ("light", "dark"):
        block = theme.get(mode)
        if not isinstance(block, dict):
            continue
        block_clean = {
            k: v.strip()
            for k, v in block.items()
            if k in _ALLOWED_THEME_VARS
            and isinstance(v, str)
            and _COLOR_VALUE_RE.match(v.strip())
        }
        if block_clean:
            cleaned[mode] = block_clean
    return cleaned


def _sanitize_branding(config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(config)
    if "theme" in cleaned:
        theme = _sanitize_theme(cleaned["theme"])
        if theme:
            cleaned["theme"] = theme
        else:
            cleaned.pop("theme", None)
    return cleaned


@router.get("/branding")
async def branding_get(request: Request, response: Response) -> Dict[str, Any]:
    """Get the current branding config. Public endpoint – no auth required."""
    acao = _resolve_acao(request)
    if acao:
        response.headers["Access-Control-Allow-Origin"] = acao
    response.headers["Cache-Control"] = "no-cache"
    store = _get_branding_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Branding store unavailable")
    branding = await store.get()
    if branding is None:
        return {}
    return branding


@router.post("/admin/branding")
async def branding_save(request: Request, payload: BrandingPayload) -> Dict[str, Any]:
    """Save branding config. Requires branding:write (admin/owner)."""
    await require_permission(request, "branding:write")
    store = _get_branding_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Branding store unavailable")
    cleaned = _sanitize_branding(payload.config)
    try:
        size = len(json.dumps(cleaned).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Branding config is not JSON-serializable") from exc
    if size > _MAX_BRANDING_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Branding config exceeds size limit")
    app_config = getattr(request.app.state, "app_config", None)
    session = read_session_cookie(request, app_config) if app_config else None
    created_by = (session or {}).get("username") or "admin"
    await store.save(cleaned, created_by=created_by)
    return {"success": True}
