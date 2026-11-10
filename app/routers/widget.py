"""Public widget router (no /admin prefix)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response

from app.widget_defaults_store import WidgetDefaultsStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget"])

# Keys that must never be exposed through the public widget defaults endpoint.
_WIDGET_DEFAULTS_HIDDEN_KEYS = {
    "apiKey",
    "apiSecret",
    "onOpen",
    "onInit",
    "onError",
}


def _get_widget_defaults_store(request: Request) -> Optional[WidgetDefaultsStore]:
    store = getattr(request.app.state, "_widget_defaults_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = WidgetDefaultsStore(engine)
    request.app.state._widget_defaults_store = store
    return store


def _resolve_acao(request: Request) -> str:
    """Resolve the Access-Control-Allow-Origin value for the public defaults endpoint.

    ``WIDGET_DEFAULTS_CORS_ORIGINS`` is a comma-separated list; ``*`` allows any
    origin (safe here: the endpoint is public, read-only and fetched without
    credentials). Otherwise the request Origin is echoed only on a list match.
    """
    app_config = getattr(request.app.state, "app_config", None)
    configured = getattr(app_config, "widget_defaults_cors_origins", "*") or "*"
    origins = [o.strip() for o in configured.split(",") if o.strip()]
    if "*" in origins:
        return "*"
    request_origin = request.headers.get("origin", "")
    return request_origin if request_origin in origins else ""


@router.get("/widget-defaults")
async def widget_defaults_get(request: Request, response: Response) -> Dict[str, Any]:
    """Get the current widget defaults. Public endpoint – no auth required."""
    acao = _resolve_acao(request)
    if acao:
        response.headers["Access-Control-Allow-Origin"] = acao
    response.headers["Cache-Control"] = "no-cache"
    store = _get_widget_defaults_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Widget defaults store unavailable")
    defaults = await store.get()
    if defaults is None:
        return {}
    return {k: v for k, v in defaults.items() if k not in _WIDGET_DEFAULTS_HIDDEN_KEYS}
