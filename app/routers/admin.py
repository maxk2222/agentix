"""Admin dashboard router: authentication and metrics aggregation endpoints.

All paths are prefixed with ``/admin`` by main.py.
"""
from __future__ import annotations

import csv
import io
import json
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.config import AppConfig
from app.auth import (
    AdminAuthRequired,
    clear_session_cookie,
    create_session_cookie,
    read_session_cookie,
    require_admin,
    require_permission,
    verify_admin_password,
)
from app import chat_runs
from app.metrics import MetricsStore, VectorstoreMetricsStore
from app.reasoning_params import (
    describe_for_model,
    descriptor_from_capabilities,
    match_family,
    normalize_reasoning_dict,
    probe_overrides_family,
)
from app.llm_providers import normalize_base_url
from app.messages import MessageStore
from app.stores import AgentMemoryStore, InstructionsStore, RawFileStore
from app.webdav_store import WebDAVFileMetaStore, WebDAVSourceStore
from app.webdav_sync import WebDAVSyncEngine, SUPPORTED_EXTENSIONS
from app.widget_defaults_store import WidgetDefaultsStore
from app.chat_groups_store import ChatGroup, ChatGroupsStore, normalize_domain

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)


class LoginPayload(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool


class UserInfo(BaseModel):
    id: str
    username: str
    display_name: str
    role: str


class MeResponse(BaseModel):
    authenticated: bool
    role: Optional[str] = None
    user: Optional[UserInfo] = None
    permissions: List[str] = []
    session_ttl_seconds: Optional[int] = None


class MetricsRangeParams(BaseModel):
    range_hours: int = 24
    bucket: str = "hour"  # hour | day
    model: Optional[str] = None


class SubagentsConfig(BaseModel):
    enabled: bool = False
    max_concurrent: int = Field(default=2, ge=1, le=10)
    max_depth: int = Field(default=1, ge=1, le=5)
    max_total: int = Field(default=8, ge=1, le=500)
    report_answer_chars: Optional[int] = Field(default=None, ge=200, le=200000)
    exclude_tools: Optional[List[str]] = None


class AgentWorkspaceConfig(BaseModel):
    """Per-assistant persistent folder the agent can read/write; optionally API-readable."""

    enabled: bool = False
    scope: Literal["assistant", "user", "conversation"] = "conversation"
    api_enabled: bool = False
    vision_read: bool = False
    max_file_mb: int = Field(default=50, ge=1, le=500)
    max_total_mb: int = Field(default=500, ge=10, le=10000)
    max_files: int = Field(default=5000, ge=1, le=100000)


class ObserverConfig(BaseModel):
    enabled: bool = False
    assistant: Optional[str] = None
    every_iterations: int = Field(default=4, ge=1, le=20)
    every_seconds: float = Field(default=180.0, ge=30, le=3600)
    max_checks: int = Field(default=8, ge=1, le=50)
    max_nudges: int = Field(default=4, ge=1, le=20)
    check_timeout_seconds: float = Field(default=60.0, ge=10, le=300)
    include_subagents: bool = True
    instructions: Optional[str] = None


class AssistantEntry(BaseModel):
    name: str
    description: Optional[str] = ""
    model: str
    temperature: float = 0.0
    collections: List[str] = []
    reasoning: Optional[Dict[str, Any]] = None
    system_instructions: str
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    rag_reserved_tokens: Optional[int] = None
    min_recent_turns: Optional[int] = None
    provider_id: Optional[str] = None
    extra_parameters: Optional[Dict[str, Any]] = None
    extra_body: Optional[Dict[str, Any]] = None
    enable_thinking: Optional[bool] = None
    agent_enabled: bool = False
    tools: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    max_iterations: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_parallel_tool_calls: Optional[int] = None
    agent_memory_enabled: bool = False
    subagents: Optional[SubagentsConfig] = None
    observer: Optional[ObserverConfig] = None
    workspace: Optional[AgentWorkspaceConfig] = None
    api_accessible: bool = True
    audio_stt_enabled: bool = False
    audio_tts_enabled: bool = False
    audio_phone_enabled: bool = False
    audio_tts_auto_read: bool = False
    audio_tts_streaming_enabled: bool = True
    audio_voice_id: Optional[str] = None
    emotion_detection_enabled: bool = False
    emotion_text_enabled: bool = True
    emotion_webcam_enabled: bool = False
    emotion_audio_enabled: bool = False
    emotion_prompt_steering_enabled: bool = True
    emotion_analytics_enabled: bool = True


class InstructionsPayload(BaseModel):
    assistants: List[AssistantEntry]


class SaveTextPayload(BaseModel):
    content: str


class WebDAVSourcePayload(BaseModel):
    name: str
    webdav_url: str
    remote_path: str = "/"
    username: str
    password: str
    enabled: bool = True
    convert_to_okf: bool = True


class WebDAVTestPayload(BaseModel):
    webdav_url: str
    remote_path: str = "/"
    username: str
    password: str = ""
    name: Optional[str] = None


class WebDAVUpdatePayload(BaseModel):
    webdav_url: Optional[str] = None
    remote_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: Optional[bool] = None
    convert_to_okf: Optional[bool] = None


def _validate_instructions_payload(payload: InstructionsPayload) -> None:
    for assistant in payload.assistants:
        if not assistant.name or not assistant.name.strip():
            raise HTTPException(status_code=400, detail="Assistant name is required")
        if not assistant.model or not assistant.model.strip():
            raise HTTPException(status_code=400, detail="Assistant model is required")
        if not assistant.system_instructions or not assistant.system_instructions.strip():
            raise HTTPException(status_code=400, detail="Assistant system_instructions is required")
        if assistant.workspace and assistant.workspace.api_enabled and not assistant.workspace.enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Assistant '{assistant.name}': workspace.api_enabled requires workspace.enabled",
            )
        if assistant.reasoning and normalize_reasoning_dict(assistant.reasoning) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Assistant '{assistant.name}': reasoning must contain canonical keys (effort, enabled, budget, level)",
            )


def _instructions_payload_to_yaml(payload: InstructionsPayload) -> str:
    output: Dict[str, Any] = {"assistants": []}
    for entry in payload.assistants:
        ad = entry.model_dump(exclude_none=False)
        if ad.get("collections") == []:
            ad.pop("collections", None)
        if ad.get("tools") == {}:
            ad.pop("tools", None)
        if not ad.get("extra_parameters"):
            ad.pop("extra_parameters", None)
        if not ad.get("extra_body"):
            ad.pop("extra_body", None)
        if ad.get("enable_thinking") is None:
            ad.pop("enable_thinking", None)
        normalized_reasoning = normalize_reasoning_dict(ad.get("reasoning"))
        if normalized_reasoning is None:
            ad.pop("reasoning", None)
        else:
            ad["reasoning"] = normalized_reasoning
        if not ad.get("agent_enabled"):
            ad.pop("agent_enabled", None)
        if not ad.get("agent_memory_enabled"):
            ad.pop("agent_memory_enabled", None)
        subagents = ad.get("subagents")
        if not subagents or not subagents.get("enabled"):
            ad.pop("subagents", None)
        else:
            ad["subagents"] = {key: value for key, value in subagents.items() if value is not None}
        observer = ad.get("observer")
        if not observer or not observer.get("enabled"):
            ad.pop("observer", None)
        else:
            ad["observer"] = {key: value for key, value in observer.items() if value is not None}
        workspace = ad.get("workspace")
        if not workspace or not workspace.get("enabled"):
            ad.pop("workspace", None)
        else:
            ad["workspace"] = {key: value for key, value in workspace.items() if value is not None}
        if ad.get("api_accessible"):
            ad.pop("api_accessible", None)
        if not ad.get("audio_stt_enabled"):
            ad.pop("audio_stt_enabled", None)
        if not ad.get("audio_tts_enabled"):
            ad.pop("audio_tts_enabled", None)
        if not ad.get("audio_phone_enabled"):
            ad.pop("audio_phone_enabled", None)
        if not ad.get("audio_tts_auto_read"):
            ad.pop("audio_tts_auto_read", None)
        if ad.get("audio_tts_streaming_enabled"):
            ad.pop("audio_tts_streaming_enabled", None)
        if not ad.get("audio_voice_id"):
            ad.pop("audio_voice_id", None)
        if not ad.get("emotion_detection_enabled"):
            ad.pop("emotion_detection_enabled", None)
        if ad.get("emotion_text_enabled"):
            ad.pop("emotion_text_enabled", None)
        if not ad.get("emotion_webcam_enabled"):
            ad.pop("emotion_webcam_enabled", None)
        if not ad.get("emotion_audio_enabled"):
            ad.pop("emotion_audio_enabled", None)
        if ad.get("emotion_prompt_steering_enabled"):
            ad.pop("emotion_prompt_steering_enabled", None)
        if ad.get("emotion_analytics_enabled"):
            ad.pop("emotion_analytics_enabled", None)
        output["assistants"].append(ad)

    return yaml.safe_dump(
        output,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=float("inf"),
    )


class CollectionInfo(BaseModel):
    name: str
    files: List[str]
    subfolders: List["CollectionInfo"] = []
    indexed_files: List[str] = []
    db_files: List[str] = []
    db_only: bool = False
    updated_at: Optional[datetime] = None
    vector_doc_count: int = 0
    file_updated_at: Dict[str, Optional[datetime]] = {}
    file_vector_doc_count: Dict[str, int] = {}


class ConversationListParams(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    min_user_messages: int = Field(default=0, ge=0)
    user_id: Optional[str] = None
    query: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)
    sort_by: str = Field(default="last_message_at")
    sort_order: str = Field(default="desc")


class ConversationExportParams(BaseModel):
    date_from: str
    date_to: str
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    min_user_messages: int = Field(default=0, ge=0)
    user_id: Optional[str] = None
    query: Optional[str] = None
    format: str = "json"


def _get_metrics_store(request: Request) -> Optional[MetricsStore]:
    return getattr(request.app.state, "_metrics_store", None)


def _get_blocked_store(request: Request) -> Optional[Any]:
    return getattr(request.app.state, "_blocked_store", None)


def _get_vectorstore_metrics_store(request: Request) -> Optional[VectorstoreMetricsStore]:
    return getattr(request.app.state, "_vectorstore_metrics_store", None)


def _get_message_store(request: Request) -> Optional[MessageStore]:
    store = getattr(request.app.state, "_message_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = MessageStore(engine)
    request.app.state._message_store = store
    return store


def _get_agent_memory_store(request: Request) -> Optional[AgentMemoryStore]:
    store = getattr(request.app.state, "_agent_memory_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = AgentMemoryStore(engine)
    request.app.state._agent_memory_store = store
    return store


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


def _get_instructions_store(request: Request) -> Optional[InstructionsStore]:
    store = getattr(request.app.state, "_instructions_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = InstructionsStore(engine)
    request.app.state._instructions_store = store
    return store


def _get_raw_file_store(request: Request) -> Optional[RawFileStore]:
    store = getattr(request.app.state, "_raw_file_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = RawFileStore(engine)
    request.app.state._raw_file_store = store
    return store


def _get_webdav_source_store(request: Request) -> Optional[WebDAVSourceStore]:
    store = getattr(request.app.state, "_webdav_source_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = WebDAVSourceStore(engine)
    request.app.state._webdav_source_store = store
    return store


def _get_webdav_meta_store(request: Request) -> Optional[WebDAVFileMetaStore]:
    store = getattr(request.app.state, "_webdav_meta_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = WebDAVFileMetaStore(engine)
    request.app.state._webdav_meta_store = store
    return store


def _parse_iso_datetime(value: str, field_name: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} must not be empty")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Use ISO-8601 datetime.") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_iso(value) for key, value in row.items()}


@router.post("/auth/login", response_model=LoginResponse)
async def admin_login(
    request: Request,
    payload: LoginPayload,
) -> Response:
    config = getattr(request.app.state, "app_config", None)
    if not config or not config.dashboard_enabled:
        raise HTTPException(status_code=403, detail="Dashboard is disabled")
    if not config.dashboard_session_secret:
        raise HTTPException(status_code=403, detail="Dashboard not configured")

    from app.dashboard_users import DashboardUserStore, permissions_for_role

    store: Optional[DashboardUserStore] = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    user = await store.authenticate(payload.username, payload.password)
    if user is None:
        # Fallback to env credentials for breakglass / legacy login
        env_ok = (
            payload.username == config.dashboard_admin_username
            and verify_admin_password(config, payload.password)
        )
        if not env_ok:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = await store.get_by_username(config.dashboard_admin_username)
        if user is None:
            # Pure legacy session (no store user yet)
            token = create_session_cookie(config)
            response = Response(
                content='{"success": true}',
                media_type="application/json",
            )
            secure = getattr(config, "_is_https", False)
            cookie_parts = [
                f"dashboard_session={token}",
                "Path=/",
                "HttpOnly",
                "SameSite=Strict",
            ]
            if secure:
                cookie_parts.append("Secure")
            max_age = config.dashboard_session_ttl_seconds
            cookie_parts.append(f"Max-Age={max_age}")
            response.headers["Set-Cookie"] = "; ".join(cookie_parts)
            return response

    await store.record_login(user.id)

    token = create_session_cookie(
        config,
        user={
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "session_version": user.session_version,
        },
    )
    response = Response(
        content='{"success": true}',
        media_type="application/json",
    )
    secure = getattr(config, "_is_https", False)
    cookie_parts = [
        f"dashboard_session={token}",
        "Path=/",
        "HttpOnly",
        "SameSite=Strict",
    ]
    if secure:
        cookie_parts.append("Secure")
    max_age = config.dashboard_session_ttl_seconds
    cookie_parts.append(f"Max-Age={max_age}")
    response.headers["Set-Cookie"] = "; ".join(cookie_parts)
    return response


@router.post("/auth/logout")
async def admin_logout() -> Response:
    response = Response(content='{"success": true}', media_type="application/json")
    key, value = clear_session_cookie()
    response.headers[key] = value
    return response


@router.get("/auth/me", response_model=MeResponse)
async def admin_me(request: Request) -> MeResponse:
    config = getattr(request.app.state, "app_config", None)
    if not config or not config.dashboard_enabled:
        return MeResponse(authenticated=False)

    from app.auth import get_current_dashboard_user, permissions_for_role

    user = await get_current_dashboard_user(request)
    if user is None:
        return MeResponse(authenticated=False)

    perms = permissions_for_role(user.get("role", ""))
    return MeResponse(
        authenticated=True,
        role=user.get("role"),
        user=UserInfo(
            id=user.get("id", ""),
            username=user.get("username", ""),
            display_name=user.get("display_name", ""),
            role=user.get("role", ""),
        ),
        permissions=perms,
        session_ttl_seconds=int(getattr(config, "dashboard_session_ttl_seconds", 3600) or 3600),
    )


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


@router.post("/auth/change-password")
async def admin_change_password(
    request: Request,
    payload: ChangePasswordPayload,
) -> Dict[str, Any]:
    """Allow the current dashboard user to change their own password."""
    from app.auth import get_current_dashboard_user

    user = await get_current_dashboard_user(request)
    if user is None:
        raise AdminAuthRequired()

    user_id = user.get("id", "")
    if not user_id or user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot change password for this account")

    store: Optional[DashboardUserStore] = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    db_user = await store.get_user(user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not store._verify_password(payload.current_password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    await store.set_password(user_id, payload.new_password)
    return {"success": True}


@router.get("/metrics/summary")
async def metrics_summary(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    return await store.get_summary(params.range_hours)


@router.get("/metrics/by-model")
async def metrics_by_model(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    return await store.get_by_model(params.range_hours)


@router.get("/metrics/history")
async def metrics_history(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    return await store.get_history(params.range_hours, params.bucket, params.model)


@router.get("/metrics/history/api-key")
async def metrics_history_by_api_key(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    return await store.get_history_by_api_key(params.range_hours, params.bucket)


@router.get("/metrics/usage")
async def metrics_usage(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    return await store.get_usage_stats(params.range_hours)


@router.get("/metrics/models")
async def metrics_models(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[str]:
    await require_permission(request, "metrics:read")
    store = _get_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    by_model = await store.get_by_model(params.range_hours)
    return [row["model_name"] for row in by_model]


@router.get("/metrics/blocked-summary")
async def blocked_summary(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    store = _get_blocked_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Blocked request store unavailable")
    return await store.get_summary(params.range_hours)


@router.get("/metrics/blocked-history")
async def blocked_history(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "metrics:read")
    store = _get_blocked_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Blocked request store unavailable")
    return await store.get_history(params.range_hours, params.bucket)


# ------------------------------------------------------------------------------
# Dashboard aggregate endpoints
# ------------------------------------------------------------------------------

@router.get("/dashboard/overview")
async def dashboard_overview(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
    page_size: int = 200,
) -> dict:
    await require_permission(request, "metrics:read")
    metrics_store = _get_metrics_store(request)
    vs_metrics_store = _get_vectorstore_metrics_store(request)
    msg_store = _get_message_store(request)
    if metrics_store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    if vs_metrics_store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")
    if msg_store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(hours=params.range_hours)).isoformat()
    date_to = now.isoformat()

    metrics_summary, usage_stats, vs_summary, conv_result = await asyncio.gather(
        metrics_store.get_summary(params.range_hours),
        metrics_store.get_usage_stats(params.range_hours),
        vs_metrics_store.get_summary(params.range_hours),
        msg_store.list_conversations(
            date_from=datetime.fromisoformat(date_from),
            date_to=datetime.fromisoformat(date_to),
            page=1,
            page_size=min(page_size, 200),
            sort_by="last_message_at",
            sort_order="desc",
        ),
    )

    # Vectorstore state (same logic as /vectorstore/state)
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG application unavailable")
    collections = []
    total_documents = 0
    for name, manager in getattr(rag_app, "vector_managers", {}).items():
        doc_count = await manager._get_document_count()
        health = await manager.health_check()
        collections.append({
            "name": name,
            "documents": doc_count,
            "healthy": health,
        })
        total_documents += doc_count
    vs_state = {
        "collections": collections,
        "total_collections": len(collections),
        "total_documents": total_documents,
    }

    total = int(conv_result["total"])
    conv_page_size = int(conv_result["page_size"])
    total_pages = max(1, (total + conv_page_size - 1) // conv_page_size)
    return {
        "metrics_summary": metrics_summary,
        "usage_stats": usage_stats,
        "vectorstore_summary": vs_summary,
        "vectorstore_state": vs_state,
        "conversations": {
            "total": total,
            "page": int(conv_result["page"]),
            "page_size": conv_page_size,
            "total_pages": total_pages,
            "items": [_normalize_row(item) for item in conv_result["items"]],
        },
    }


def _live_run_payload(run: Any, now_monotonic: float) -> Dict[str, Any]:
    """Normalize one ActiveChatRun into the overview live-activity shape."""
    model = getattr(run, "model_name", "") or ""
    if model.startswith("chain:"):
        kind = "chain"
    elif getattr(run, "is_agent_run", False):
        kind = "agent"
    else:
        kind = "chat"
    done = bool(getattr(run, "done", False))
    if not done:
        state = "running"
    elif getattr(run, "cancelled", False):
        state = "cancelled"
    elif getattr(run, "stream_error", None) is not None:
        state = "error"
    else:
        state = "done"
    end_monotonic = getattr(run, "finished_monotonic", None) or now_monotonic
    elapsed_s = max(0.0, end_monotonic - (getattr(run, "created_monotonic", None) or end_monotonic))
    live_usage = getattr(run, "live_usage", None) or {}
    subagents = getattr(run, "subagent_events", None) or []
    subagents_running = sum(
        1
        for s in subagents
        if isinstance(s, dict) and s.get("status") == "running"
    )
    todo = getattr(run, "todo_list", None)
    todos_total = todos_open = None
    todo_items: List[Any] = []
    if isinstance(todo, dict):
        todo_items = todo.get("items") or []
    elif isinstance(todo, list):
        todo_items = todo
    if todo_items:
        todos_total = len(todo_items)
        todos_open = sum(
            1
            for item in todo_items
            if isinstance(item, dict) and item.get("status") not in ("done", "cancelled")
        )
    user_message = (getattr(run, "user_message", "") or "")[:140]
    started_ts = getattr(run, "created_ts", None)
    return {
        "conversation_id": getattr(run, "conversation_id", ""),
        "model": model,
        "kind": kind,
        "source": getattr(run, "source", "") or "",
        "state": state,
        "started_ts": (
            datetime.fromtimestamp(started_ts, tz=timezone.utc).isoformat()
            if isinstance(started_ts, (int, float)) else None
        ),
        "elapsed_s": round(elapsed_s, 1),
        "user_message": user_message,
        "phase": getattr(run, "activity_phase", "") or "",
        "phase_detail": getattr(run, "activity_detail", "") or "",
        "tokens": live_usage.get("total_tokens"),
        "llm_calls": live_usage.get("llm_calls"),
        "estimated": bool(live_usage.get("estimated")),
        "tool_calls": len(getattr(run, "tool_calls_buffer", None) or []),
        "subagents_running": subagents_running,
        "subagents_total": len(subagents),
        "todos_open": todos_open,
        "todos_total": todos_total,
    }


@router.get("/dashboard/live-activity")
async def dashboard_live_activity(request: Request) -> dict:
    """Active chat/agent runs plus automation activity for the overview strip.

    Read-only admin snapshot: in-memory run registry (per-process, same
    caveat as run resume) plus lightweight store reads for automation.
    """
    await require_permission(request, "metrics:read")
    now_monotonic = time.monotonic()

    runs: List[Dict[str, Any]] = []
    try:
        raw_runs = []
        for cid in chat_runs.active_conversation_ids():
            run = chat_runs.get_run(cid)
            if run is not None:
                raw_runs.append(run)
        # Running first (longest-running on top), finished by recency.
        raw_runs.sort(
            key=lambda r: (
                bool(getattr(r, "done", False)),
                -(
                    getattr(r, "finished_monotonic", None)
                    or getattr(r, "created_monotonic", 0.0)
                    or 0.0
                )
                if getattr(r, "done", False)
                else getattr(r, "created_monotonic", 0.0),
            )
        )
        runs = [_live_run_payload(r, now_monotonic) for r in raw_runs]
    except Exception:
        logger.exception("live-activity: failed to enumerate chat runs")

    triggers: List[Dict[str, Any]] = []
    try:
        scheduler = getattr(request.app.state, "_triggers_scheduler", None)
        running_since = scheduler.running_now() if scheduler is not None else {}
        triggers_store = getattr(request.app.state, "_triggers_store", None)
        for trigger_id, since in running_since.items():
            name = ""
            if triggers_store is not None:
                try:
                    entry = await triggers_store.get_trigger(trigger_id)
                    name = getattr(entry, "name", "") or ""
                except Exception:
                    pass
            triggers.append({
                "id": trigger_id,
                "name": name,
                "elapsed_s": round(max(0.0, now_monotonic - since), 1),
            })
    except Exception:
        logger.exception("live-activity: failed to collect trigger activity")

    backups: List[Dict[str, Any]] = []
    try:
        job_store = getattr(request.app.state, "_backup_job_store", None)
        if job_store is not None:
            for job in await job_store.list_jobs(limit=10):
                if job.get("status") != "running":
                    continue
                backups.append({
                    "id": job.get("id"),
                    "kind": job.get("kind") or "manual",
                    "triggered_by": job.get("triggered_by") or "",
                    "started_at": (
                        job["started_at"].isoformat()
                        if isinstance(job.get("started_at"), datetime)
                        else job.get("started_at")
                    ),
                })
    except Exception:
        logger.exception("live-activity: failed to collect backup jobs")

    syncs: List[Dict[str, Any]] = []
    try:
        webdav_store = _get_webdav_source_store(request)
        if webdav_store is not None:
            for src in await webdav_store.list_sources():
                if src.get("last_sync_status") != "running":
                    continue
                since = src.get("last_sync_at") or src.get("updated_at")
                syncs.append({
                    "name": src.get("name") or "",
                    "since": (
                        since.isoformat() if isinstance(since, datetime) else since
                    ),
                })
    except Exception:
        logger.exception("live-activity: failed to collect webdav syncs")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "automation": {
            "triggers": triggers,
            "backups": backups,
            "syncs": syncs,
        },
    }


@router.get("/dashboard/models")
async def dashboard_models(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    metrics_store = _get_metrics_store(request)
    vs_metrics_store = _get_vectorstore_metrics_store(request)
    if metrics_store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    if vs_metrics_store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")

    metrics_by_model, vs_by_model = await asyncio.gather(
        metrics_store.get_by_model(params.range_hours),
        vs_metrics_store.get_by_model(params.range_hours),
    )
    return {
        "metrics_by_model": metrics_by_model,
        "vectorstore_by_model": vs_by_model,
    }


@router.get("/dashboard/history")
async def dashboard_history(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    metrics_store = _get_metrics_store(request)
    vs_metrics_store = _get_vectorstore_metrics_store(request)
    if metrics_store is None:
        raise HTTPException(status_code=503, detail="Metrics store unavailable")
    if vs_metrics_store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")

    metrics_history, vs_history, api_key_history, by_model = await asyncio.gather(
        metrics_store.get_history(params.range_hours, params.bucket, params.model),
        vs_metrics_store.get_history(params.range_hours, params.bucket, params.model),
        metrics_store.get_history_by_api_key(params.range_hours, params.bucket),
        metrics_store.get_by_model(params.range_hours),
    )
    return {
        "metrics_history": metrics_history,
        "vectorstore_history": vs_history,
        "api_key_history": api_key_history,
        "models": [row["model_name"] for row in by_model],
    }


@router.get("/dashboard/blocked")
async def dashboard_blocked(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "metrics:read")
    store = _get_blocked_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Blocked request store unavailable")

    summary, history = await asyncio.gather(
        store.get_summary(params.range_hours),
        store.get_history(params.range_hours, params.bucket),
    )
    return {
        "blocked_summary": summary,
        "blocked_history": history,
    }


# ------------------------------------------------------------------------------
# Configuration / Instructions endpoints
# ------------------------------------------------------------------------------

def _normalize_tools_field(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy tools list format to dict format in-place."""
    for assistant in data.get("assistants", []):
        tools = assistant.get("tools")
        if isinstance(tools, list):
            assistant["tools"] = {
                str(name).strip(): {} for name in tools if str(name).strip()
            }
        elif not isinstance(tools, dict):
            assistant["tools"] = {}
    return data


@router.get("/config/instructions")
async def get_instructions(request: Request) -> dict:
    """Return the current instructions from DB (or file as fallback)."""
    await require_permission(request, "instructions:read")
    store = _get_instructions_store(request)
    if store is not None:
        yaml_text = await store.get_current()
        if yaml_text is not None:
            data = yaml.safe_load(yaml_text) or {}
            return _normalize_tools_field(data)
    # Fallback to file
    config = getattr(request.app.state, "app_config", None)
    path = config.instructions_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Instructions file not found")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _normalize_tools_field(data)


@router.put("/config/instructions")
async def put_instructions(request: Request, payload: InstructionsPayload) -> dict:
    """Save instructions to DB (primary) and mirror to file."""
    await require_permission(request, "instructions:write")
    config = getattr(request.app.state, "app_config", None)

    _validate_instructions_payload(payload)
    yaml_text = _instructions_payload_to_yaml(payload)

    store = _get_instructions_store(request)

    # Rename lifecycle: if exactly one assistant name disappears and one new
    # name appears AT THE SAME LIST POSITION, treat it as a rename and move
    # its agent-workspace tree. The position check prevents a "delete A +
    # create B" save from leaking A's files into B's folder. Deletions are
    # NOT handled here — the sweeper removes orphan dirs after the TTL.
    try:
        old_names_ordered = []
        old_yaml = await store.get_current() if store is not None else None
        if old_yaml is None and config is not None and config.instructions_file.exists():
            old_yaml = config.instructions_file.read_text(encoding="utf-8")
        if old_yaml:
            old_names_ordered = [
                a.get("name")
                for a in (yaml.safe_load(old_yaml) or {}).get("assistants", [])
                if a.get("name")
            ]
        old_names = set(old_names_ordered)
        new_names_ordered = [a.name for a in payload.assistants]
        removed, added = old_names - set(new_names_ordered), set(new_names_ordered) - old_names
        if len(removed) == 1 and len(added) == 1:
            removed_name, added_name = next(iter(removed)), next(iter(added))
            # Renames keep the slot; a delete+create pair usually does not.
            same_slot = (
                removed_name in old_names_ordered
                and added_name in new_names_ordered
                and old_names_ordered.index(removed_name) == new_names_ordered.index(added_name)
            )
            if same_slot:
                from app.agent_workspace import AgentWorkspaceStore

                AgentWorkspaceStore().rename_assistant(removed_name, added_name)
    except Exception:
        logger.warning("Assistant workspace rename check failed", exc_info=True)

    if store is not None:
        await store.save(yaml_text, created_by="admin")

    path = config.instructions_file
    path.write_text(yaml_text, encoding="utf-8")

    # Force reload so the backend picks up the new assistant immediately
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is not None:
        try:
            await rag_app.force_refresh_instructions()
        except Exception:
            logger.warning("Failed to force-reload instructions after save", exc_info=True)

    return {"success": True}


@router.get("/config/instructions/download")
async def download_instructions(request: Request) -> Response:
    """Download current instructions YAML from DB (or file fallback)."""
    await require_permission(request, "instructions:read")
    store = _get_instructions_store(request)
    yaml_text = None
    if store is not None:
        yaml_text = await store.get_current()
    if yaml_text is None:
        config = getattr(request.app.state, "app_config", None)
        path = config.instructions_file
        if not path.exists():
            raise HTTPException(status_code=404, detail="Instructions not found")
        yaml_text = path.read_text(encoding="utf-8")
    return Response(
        content=yaml_text,
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="instructions.yaml"'},
    )


@router.post("/config/instructions/upload")
async def upload_instructions(request: Request, file: UploadFile = File(...)) -> dict:
    await require_permission(request, "instructions:write")
    config = getattr(request.app.state, "app_config", None)

    filename = file.filename or ""
    if filename and not filename.lower().endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Only .yaml or .yml files are allowed")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Instructions file must be UTF-8 encoded")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Instructions YAML must contain a mapping")

    try:
        payload = InstructionsPayload.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid instructions schema: {exc}")

    _validate_instructions_payload(payload)
    yaml_text = _instructions_payload_to_yaml(payload)

    store = _get_instructions_store(request)
    if store is not None:
        await store.save(yaml_text, created_by="admin")

    config.instructions_file.write_text(yaml_text, encoding="utf-8")
    return {"success": True}


@router.post("/config/apply")
async def config_apply(request: Request) -> dict:
    """Hot-reload assistant instructions without reinitialising the vector store."""
    await require_permission(request, "instructions:apply")
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG application not available")
    reloaded = await rag_app.force_refresh_instructions()
    return {"message": "Configuration initialized.", "reloaded": reloaded}


@router.get("/config/instructions/history")
async def get_instructions_history(request: Request, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the last N versions of instructions (without yaml_content)."""
    await require_permission(request, "instructions:read")
    store = _get_instructions_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Instructions store unavailable")
    return await store.get_history(limit=limit)


@router.post("/config/instructions/rollback/{version_id}")
async def rollback_instructions(request: Request, version_id: int) -> dict:
    """Set a previous version as current and mirror it to the file."""
    await require_permission(request, "instructions:rollback")
    store = _get_instructions_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Instructions store unavailable")
    ok = await store.rollback(version_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Version not found")
    # Mirror rolled-back content to file
    yaml_text = await store.get_current()
    if yaml_text is not None:
        config = getattr(request.app.state, "app_config", None)
        config.instructions_file.write_text(yaml_text, encoding="utf-8")
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is not None:
        try:
            await rag_app.force_refresh_instructions()
        except Exception:
            logger.warning("Failed to force-reload instructions after rollback", exc_info=True)
    return {"success": True}


@router.get("/config/instructions/version/{version_id}")
async def get_instructions_version(request: Request, version_id: int) -> Dict[str, Any]:
    """Return a specific instructions version including its YAML content."""
    await require_permission(request, "instructions:read")
    store = _get_instructions_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Instructions store unavailable")
    version = await store.get_version(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


class DiffPayload(BaseModel):
    from_version_id: int
    to_version_id: int


@router.post("/config/instructions/diff")
async def diff_instructions(request: Request, payload: DiffPayload) -> Dict[str, Any]:
    """Return a unified diff between two instructions versions."""
    await require_permission(request, "instructions:read")
    store = _get_instructions_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Instructions store unavailable")

    from_version = await store.get_version(payload.from_version_id)
    to_version = await store.get_version(payload.to_version_id)
    if from_version is None or to_version is None:
        raise HTTPException(status_code=404, detail="One or both versions not found")

    import difflib

    from_lines = from_version["yaml_content"].splitlines(keepends=True)
    to_lines = to_version["yaml_content"].splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"version_{payload.from_version_id}",
            tofile=f"version_{payload.to_version_id}",
        )
    )

    structured_changes = _build_structured_diff(
        from_version["yaml_content"], to_version["yaml_content"]
    )

    return {
        "diff": "".join(diff),
        "structured_changes": structured_changes,
        "from": {
            "id": from_version["id"],
            "created_at": from_version["created_at"],
            "created_by": from_version["created_by"],
            "is_current": from_version["is_current"],
        },
        "to": {
            "id": to_version["id"],
            "created_at": to_version["created_at"],
            "created_by": to_version["created_by"],
            "is_current": to_version["is_current"],
        },
    }


def _build_structured_diff(from_yaml: str, to_yaml: str) -> List[Dict[str, Any]]:
    """Parse two instructions YAMLs and return structured per-assistant changes."""
    try:
        from_data = yaml.safe_load(from_yaml) or {}
        to_data = yaml.safe_load(to_yaml) or {}
    except Exception:
        return []

    from_assistants = from_data.get("assistants", []) or []
    to_assistants = to_data.get("assistants", []) or []

    from_by_name = {a.get("name", f"__idx_{i}"): a for i, a in enumerate(from_assistants)}
    to_by_name = {a.get("name", f"__idx_{i}"): a for i, a in enumerate(to_assistants)}

    all_names = set(from_by_name.keys()) | set(to_by_name.keys())
    changes: List[Dict[str, Any]] = []

    for name in sorted(all_names):
        old_a = from_by_name.get(name)
        new_a = to_by_name.get(name)

        if old_a is None and new_a is not None:
            changes.append({"type": "added", "assistant": name, "fields": []})
            continue
        if old_a is not None and new_a is None:
            changes.append({"type": "removed", "assistant": name, "fields": []})
            continue

        field_changes: List[Dict[str, Any]] = []
        all_keys = set(old_a.keys()) | set(new_a.keys())  # type: ignore[union-attr]
        for key in sorted(all_keys):
            old_val = old_a.get(key)  # type: ignore[union-attr]
            new_val = new_a.get(key)  # type: ignore[union-attr]
            if old_val != new_val:
                entry: Dict[str, Any] = {
                    "field": key,
                    "old_value": old_val,
                    "new_value": new_val,
                }
                if key == "system_instructions" and isinstance(old_val, str) and isinstance(new_val, str):
                    entry["text_diff"] = _text_diff(old_val, new_val)
                field_changes.append(entry)

        if field_changes:
            changes.append({"type": "modified", "assistant": name, "fields": field_changes})

    return changes


def _text_diff(old_text: str, new_text: str) -> str:
    """Return a unified diff of two multi-line strings."""
    import difflib

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="system_instructions_old",
            tofile="system_instructions_new",
        )
    )
    return "".join(diff)


@router.get("/config/models")
async def config_models(request: Request) -> List[str]:
    """Fetch available models from the configured upstream LLM API."""
    await require_permission(request, "providers:read")
    cfg = getattr(request.app.state, "app_config", None)
    base = cfg.llm_api_base
    api_key = cfg.llm_api_key
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base}/v1/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            names = [m["id"] for m in models if "id" in m]
            return sorted(names)
    except Exception:
        logger.exception("Failed to fetch models from upstream LLM API")
        raise HTTPException(status_code=502, detail="Could not fetch models from upstream LLM API")


# ------------------------------------------------------------------------------
# Knowledge database endpoints
# ------------------------------------------------------------------------------

def _scan_folder(path: Path, rel_prefix: str = "") -> CollectionInfo:
    """Recursively scan a folder and return a CollectionInfo tree."""
    files: List[str] = []
    subfolders: List[CollectionInfo] = []
    for item in sorted(path.iterdir()):
        if item.is_file():
            files.append(item.name)
        elif item.is_dir():
            subfolders.append(_scan_folder(item, rel_prefix=""))
    name = rel_prefix if rel_prefix else path.name
    return CollectionInfo(name=name, files=sorted(files), subfolders=subfolders)


def _to_relative_filename(collection_name: str, source: str) -> Optional[str]:
    """Strip collection prefix from a DB source to get the visible filename."""
    prefix = collection_name + "/"
    if source.startswith(prefix):
        rel = source[len(prefix):]
        if "/" not in rel:
            return rel
    return None


@router.get("/knowledge/collections")
async def knowledge_collections(request: Request, include_counts: bool = False) -> List[CollectionInfo]:
    """List all knowledge collections recursively with files and subfolders, merged with DB state."""
    await require_permission(request, "knowledge:read")
    config = getattr(request.app.state, "app_config", None)
    root = config.knowledge_base_root
    result: List[CollectionInfo] = []

    # Scan filesystem
    fs_collections: Dict[str, CollectionInfo] = {}
    if root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                info = _scan_folder(child)
                fs_collections[info.name] = info

    # Merge with DB sources in a single query
    rag_app = getattr(request.app.state, "rag_application", None)
    source_dates: Dict[str, Dict[str, datetime]] = {}
    if rag_app is not None:
        try:
            engine = getattr(rag_app, "engine", None)
            if engine is not None:
                from agent.utils.pg_helpers import SOURCE_STATE_TABLE
                from sqlalchemy import text as sa_text

                db_sources_by_collection: Dict[str, List[str]] = {}
                async with engine.connect() as conn:
                    try:
                        res = await conn.execute(
                            sa_text(
                                f"SELECT collection_name, source, updated_at FROM {SOURCE_STATE_TABLE}"
                            )
                        )
                        for row in res:
                            col_name = str(row.collection_name)
                            source = str(row.source)
                            db_sources_by_collection.setdefault(col_name, []).append(source)
                            if row.updated_at is not None:
                                source_dates.setdefault(col_name, {})[source] = row.updated_at
                    except Exception:
                        pass

                for col_name, db_sources in db_sources_by_collection.items():
                    relative_names = [_to_relative_filename(col_name, s) for s in db_sources]
                    collection_level = [n for n in relative_names if n is not None]
                    if col_name in fs_collections:
                        info = fs_collections[col_name]
                        fs_files = set(info.files)
                        info.indexed_files = sorted(n for n in collection_level if n in fs_files)
                        info.db_files = sorted(n for n in collection_level if n not in fs_files)
                        info.db_only = False
                    else:
                        result.append(CollectionInfo(
                            name=col_name,
                            files=[],
                            subfolders=[],
                            indexed_files=[],
                            db_files=sorted(collection_level),
                            db_only=True,
                        ))

                # Also merge raw files that may not be indexed yet
                raw_store = _get_raw_file_store(request)
                if raw_store is not None:
                    raw_collections = await raw_store.list_collections()
                    for col_name in raw_collections:
                        raw_files = await raw_store.list_by_collection(col_name)
                        raw_names = [f["filename"] for f in raw_files]
                        existing = next((i for i in result if i.name == col_name), None)
                        if existing is not None:
                            already_present = set(existing.files) | set(existing.indexed_files)
                            new_db = sorted(set(raw_names) - already_present)
                            existing.db_files = sorted(set(existing.db_files) | set(new_db))
                        elif col_name in fs_collections:
                            fs_info = fs_collections[col_name]
                            already_present = set(fs_info.files) | set(fs_info.indexed_files)
                            new_db = sorted(set(raw_names) - already_present)
                            fs_info.db_files = sorted(set(fs_info.db_files) | set(new_db))
                            fs_info.db_only = False
                        else:
                            result.append(CollectionInfo(
                                name=col_name,
                                files=[],
                                subfolders=[],
                                indexed_files=[],
                                db_files=sorted(raw_names),
                                db_only=True,
                            ))
        except Exception as exc:
            logger.warning("Failed to merge DB sources: %s", exc)

    for info in fs_collections.values():
        result.append(info)
    result.sort(key=lambda x: x.name)

    # Populate updated_at and optionally vector_doc_count fields
    for info in result:
        col_name = info.name
        dates = source_dates.get(col_name, {})
        if dates:
            info.updated_at = max(dates.values())
        for filename, dt in dates.items():
            rel = _to_relative_filename(col_name, filename)
            if rel is not None:
                info.file_updated_at[rel] = dt

        if include_counts:
            manager = getattr(rag_app, "vector_managers", {}).get(col_name) if rag_app is not None else None
            if manager is not None:
                try:
                    info.vector_doc_count = await manager.get_document_count()
                    if dates:
                        counts = await manager.count_documents_for_sources(list(dates.keys()))
                        for source, count in counts.items():
                            rel = _to_relative_filename(col_name, source)
                            if rel is not None:
                                info.file_vector_doc_count[rel] = count
                except Exception:
                    pass

    return result


@router.get("/knowledge/collections/{name}/stats")
async def knowledge_collection_stats(request: Request, name: str) -> dict:
    """Return vector document counts for a specific collection."""
    await require_permission(request, "knowledge:read")
    result = {"vector_doc_count": 0, "file_vector_doc_count": {}}

    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        return result

    manager = getattr(rag_app, "vector_managers", {}).get(name)
    if manager is None:
        return result

    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return result

    try:
        from agent.utils.pg_helpers import SOURCE_STATE_TABLE
        from sqlalchemy import text as sa_text

        sources: List[str] = []
        async with engine.connect() as conn:
            res = await conn.execute(
                sa_text(f"SELECT source FROM {SOURCE_STATE_TABLE} WHERE collection_name = :name"),
                {"name": name},
            )
            sources = [str(row.source) for row in res]

        if sources:
            counts = await manager.count_documents_for_sources(sources)
            for source, count in counts.items():
                rel = _to_relative_filename(name, source)
                if rel is not None:
                    result["file_vector_doc_count"][rel] = count

        result["vector_doc_count"] = await manager.get_document_count()
    except Exception:
        pass

    return result


@router.post("/knowledge/collections/{name}")
async def knowledge_create_collection(
    request: Request,
    name: str,
) -> dict:
    """Create a new knowledge collection folder."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid collection name")
    target = config.knowledge_base_root / name
    target.mkdir(parents=True, exist_ok=True)
    return {"success": True, "name": name}


@router.delete("/knowledge/collections/{name}")
async def knowledge_delete_collection(
    request: Request,
    name: str,
) -> dict:
    """Delete a knowledge collection folder and all its contents, also clean up DB."""
    await require_permission(request, "knowledge:delete")
    config = getattr(request.app.state, "app_config", None)
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid collection name")
    target = config.knowledge_base_root / name
    exists_fs = target.exists()

    rag_app = getattr(request.app.state, "rag_application", None)
    db_result = None
    if rag_app is not None and getattr(rag_app, "ready", False):
        try:
            db_result = await rag_app.delete_knowledge_collection(
                name, delete_folder=exists_fs
            )
        except Exception as exc:
            logger.exception("Failed to delete collection '%s' from DB: %s", name, exc)
            db_result = {"error": str(exc)}
    elif exists_fs:
        import shutil
        shutil.rmtree(target)

    raw_store = _get_raw_file_store(request)
    raw_deleted = 0
    if raw_store is not None:
        raw_deleted = await raw_store.delete_collection(name)

    if not exists_fs and raw_deleted == 0 and (db_result is None or "error" in db_result):
        if db_result and "error" in db_result:
            raise HTTPException(status_code=500, detail=f"DB cleanup failed: {db_result['error']}")
        raise HTTPException(status_code=404, detail="Collection not found")

    # Remove deleted collection from assistant instructions
    store = _get_instructions_store(request)
    if store is not None and config is not None:
        try:
            yaml_text = await store.get_current()
            if yaml_text:
                data = yaml.safe_load(yaml_text) or {}
                assistants = data.get("assistants", [])
                changed = False
                for assistant in assistants:
                    cols = assistant.get("collections", [])
                    if name in cols:
                        assistant["collections"] = [c for c in cols if c != name]
                        if not assistant["collections"]:
                            assistant.pop("collections", None)
                        changed = True
                if changed:
                    new_yaml = _instructions_payload_to_yaml(
                        InstructionsPayload.model_validate(data)
                    )
                    await store.save(new_yaml, created_by="admin")
                    config.instructions_file.write_text(new_yaml, encoding="utf-8")
        except Exception as exc:
            logger.exception(
                "Failed to remove collection '%s' from instructions: %s", name, exc
            )

    return {"success": True, "db_cleanup": db_result, "raw_deleted": raw_deleted}


@router.put("/knowledge/collections/{name}/rename")
async def knowledge_rename_collection(
    request: Request,
    name: str,
    new_name: str = Form(...),
) -> dict:
    """Rename a knowledge collection folder."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)
    for n in (name, new_name):
        if ".." in n or "/" in n or "\\" in n:
            raise HTTPException(status_code=400, detail="Invalid collection name")
    source = config.knowledge_base_root / name
    if not source.exists():
        raise HTTPException(status_code=404, detail="Collection not found")
    target = config.knowledge_base_root / new_name
    if target.exists():
        raise HTTPException(status_code=409, detail="A collection with that name already exists")
    source.rename(target)
    return {"success": True, "old_name": name, "new_name": new_name}


@router.post("/knowledge/collections/{name}/rebuild")
async def knowledge_rebuild_collection(
    request: Request,
    name: str,
) -> dict:
    """Rebuild the full vector index for a collection."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid collection name")

    target = config.knowledge_base_root / name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Collection not found")

    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None or not getattr(rag_app, "ready", False):
        raise HTTPException(status_code=503, detail="RAG application not available")

    all_files = [f.name for f in target.iterdir() if f.is_file()]
    if not all_files:
        return {"success": True, "message": "No files to index"}

    # Hydrate from DB first, then index
    raw_store = _get_raw_file_store(request)
    if raw_store is not None:
        for fn in all_files:
            source = f"{name}/{fn}"
            db_file = await raw_store.get(name, source)
            if db_file is not None:
                data = db_file["file_data"]
                if isinstance(data, bytes):
                    (target / fn).write_bytes(data)

    index_result = await rag_app.index_knowledge_files(name, all_files)
    return {"success": True, "collection": name, "files": all_files, "indexing": index_result}


@router.post("/knowledge/upload")
async def knowledge_upload(
    request: Request,
    collection: str = Form(...),
    file: List[UploadFile] = File(..., alias="file"),
) -> dict:
    """Upload one or more files into a knowledge collection (DB primary, FS mirror)."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)

    if ".." in collection:
        raise HTTPException(status_code=400, detail="Invalid path")

    target_dir = config.knowledge_base_root / collection
    target_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[str] = []

    normalized_files = [f for f in file if f is not None]
    if not normalized_files:
        raise HTTPException(status_code=400, detail="At least one file must be provided")

    for entry in normalized_files:
        filename = (entry.filename or "").strip()
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"File type not allowed. Supported: {allowed}")
        if ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid path")

    raw_store = _get_raw_file_store(request)

    for entry in normalized_files:
        filename = (entry.filename or "").strip()
        content = await entry.read()
        # Save to filesystem (legacy mirror)
        target_path = target_dir / filename
        target_path.write_bytes(content)
        uploaded.append(filename)
        # Save to DB (primary)
        if raw_store is not None:
            source = f"{collection}/{filename}"
            await raw_store.save(
                collection_name=collection,
                source=source,
                filename=filename,
                content_type=entry.content_type,
                data=content,
            )

    index_result: Optional[dict] = None
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is not None and getattr(rag_app, "ready", False):
        try:
            index_result = await rag_app.index_knowledge_files(collection, uploaded)
        except Exception as exc:
            logger.exception(
                "Failed to index uploaded files for collection '%s': %s",
                collection,
                exc,
            )
            index_result = {"error": str(exc)}
    else:
        index_result = {"error": "RAG application not available"}

    return {
        "success": True,
        "collection": collection,
        "filenames": uploaded,
        "indexing": index_result,
    }


@router.get("/knowledge/preview/{collection:path}/{filename}")
async def knowledge_preview(
    request: Request,
    collection: str,
    filename: str,
) -> Response:
    """Serve a file from DB (primary) or filesystem (fallback)."""
    await require_permission(request, "knowledge:read")
    config = getattr(request.app.state, "app_config", None)
    if ".." in collection or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    raw_store = _get_raw_file_store(request)
    source = f"{collection}/{filename}"
    if raw_store is not None:
        db_file = await raw_store.get(collection, source)
        if db_file is not None:
            media_type = db_file.get("content_type")
            if not media_type:
                lower_name = filename.lower()
                if lower_name.endswith(".pdf"):
                    media_type = "application/pdf"
                elif lower_name.endswith(".txt"):
                    media_type = "text/plain; charset=utf-8"
                elif lower_name.endswith(".doc"):
                    media_type = "application/msword"
                elif lower_name.endswith(".docx"):
                    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif lower_name.endswith(".md"):
                    media_type = "text/markdown; charset=utf-8"
            headers: Optional[dict] = None
            if filename.lower().endswith(".pdf"):
                headers = {"Content-Disposition": f'inline; filename="{filename}"'}
            data = db_file["file_data"]
            if not isinstance(data, bytes):
                data = bytes(data)
            if len(data) == 0:
                raise HTTPException(status_code=404, detail="File is empty in database")
            if headers is None:
                headers = {}
            headers["Content-Length"] = str(len(data))
            return Response(
                content=data,
                media_type=media_type,
                headers=headers,
            )

    target = config.knowledge_base_root / collection / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@router.delete("/knowledge/{collection:path}/{filename}")
async def knowledge_delete(
    request: Request,
    collection: str,
    filename: str,
) -> dict:
    """Delete a file from a knowledge collection and clean up DB."""
    await require_permission(request, "knowledge:delete")
    config = getattr(request.app.state, "app_config", None)
    if ".." in collection or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    target = config.knowledge_base_root / collection / filename
    exists_fs = target.exists()

    # Delete from raw file store
    raw_store = _get_raw_file_store(request)
    raw_deleted = False
    if raw_store is not None:
        source = f"{collection}/{filename}"
        raw_deleted = await raw_store.delete(collection, source)

    rag_app = getattr(request.app.state, "rag_application", None)
    db_result = None
    if rag_app is not None and getattr(rag_app, "ready", False):
        try:
            db_result = await rag_app.delete_knowledge_source(
                collection, filename, delete_file=exists_fs
            )
        except Exception as exc:
            logger.exception("Failed to delete source '%s' from DB: %s", filename, exc)
            db_result = {"error": str(exc)}
    elif exists_fs:
        target.unlink()

    if not exists_fs and not raw_deleted and (db_result is None or "error" in db_result):
        if db_result and "error" in db_result:
            raise HTTPException(status_code=500, detail=f"DB cleanup failed: {db_result['error']}")
        raise HTTPException(status_code=404, detail="File not found")

    return {"success": True, "db_cleanup": db_result}


@router.get("/knowledge/text/{collection:path}/{filename}")
async def knowledge_read_text(
    request: Request,
    collection: str,
    filename: str,
) -> dict:
    """Read a .md or .txt file from DB (primary) or filesystem (fallback)."""
    await require_permission(request, "knowledge:read")
    config = getattr(request.app.state, "app_config", None)
    if ".." in collection or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filename.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt files can be edited")

    raw_store = _get_raw_file_store(request)
    source = f"{collection}/{filename}"
    if raw_store is not None:
        db_file = await raw_store.get(collection, source)
        if db_file is not None:
            data = db_file["file_data"]
            if not isinstance(data, bytes):
                data = bytes(data)
            return {"content": data.decode("utf-8", errors="replace")}

    target = config.knowledge_base_root / collection / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    content = target.read_text(encoding="utf-8")
    return {"content": content}


@router.put("/knowledge/text/{collection:path}/{filename}")
async def knowledge_save_text(
    request: Request,
    collection: str,
    filename: str,
    payload: SaveTextPayload,
) -> dict:
    """Save a .md or .txt file to DB (primary) and filesystem (mirror), then re-index."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)
    if ".." in collection or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not filename.lower().endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt files can be saved")

    content_bytes = payload.content.encode("utf-8")
    source = f"{collection}/{filename}"

    raw_store = _get_raw_file_store(request)
    if raw_store is not None:
        await raw_store.save(
            collection_name=collection,
            source=source,
            filename=filename,
            content_type="text/plain; charset=utf-8",
            data=content_bytes,
        )

    target_dir = config.knowledge_base_root / collection
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    target.write_text(payload.content, encoding="utf-8")

    index_result: Optional[dict] = None
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is not None and getattr(rag_app, "ready", False):
        try:
            index_result = await rag_app.index_knowledge_files(collection, [filename])
        except Exception as exc:
            logger.exception(
                "Failed to index saved text file for collection '%s': %s",
                collection,
                exc,
            )
            index_result = {"error": str(exc)}
    else:
        index_result = {"error": "RAG application not available"}

    return {
        "success": True,
        "collection": collection,
        "filename": filename,
        "indexing": index_result,
    }


# ------------------------------------------------------------------------------
# WebDAV source endpoints
# ------------------------------------------------------------------------------

@router.get("/knowledge/webdav")
async def webdav_list_sources(request: Request) -> List[Dict[str, Any]]:
    """List all WebDAV sources (without passwords)."""
    await require_permission(request, "knowledge:read")
    store = _get_webdav_source_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    sources = await store.list_sources()
    return [_normalize_row(s) for s in sources]


@router.post("/knowledge/webdav")
async def webdav_create_source(
    request: Request,
    payload: WebDAVSourcePayload,
) -> Dict[str, Any]:
    """Create a new WebDAV source."""
    await require_permission(request, "knowledge:write")
    if ".." in payload.name or "/" in payload.name or "\\" in payload.name:
        raise HTTPException(status_code=400, detail="Invalid source name")
    if not payload.webdav_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS WebDAV URLs are allowed")
    store = _get_webdav_source_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        result = await store.create_source(
            name=payload.name,
            webdav_url=payload.webdav_url,
            remote_path=payload.remote_path,
            username=payload.username,
            password=payload.password,
            enabled=payload.enabled,
            convert_to_okf=payload.convert_to_okf,
        )
    except Exception as exc:
        logger.exception("Failed to create WebDAV source: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _normalize_row(result)


@router.put("/knowledge/webdav/{name}")
async def webdav_update_source(
    request: Request,
    name: str,
    payload: WebDAVUpdatePayload,
) -> Dict[str, Any]:
    """Update a WebDAV source."""
    await require_permission(request, "knowledge:write")
    store = _get_webdav_source_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    result = await store.update_source(
        name=name,
        webdav_url=payload.webdav_url,
        remote_path=payload.remote_path,
        username=payload.username,
        password=payload.password,
        enabled=payload.enabled,
        convert_to_okf=payload.convert_to_okf,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="WebDAV source not found")
    return _normalize_row(result)


@router.delete("/knowledge/webdav/{name}")
async def webdav_delete_source(
    request: Request,
    name: str,
) -> dict:
    """Delete a WebDAV source and its vectors."""
    await require_permission(request, "knowledge:delete")
    store = _get_webdav_source_store(request)
    meta_store = _get_webdav_meta_store(request)
    if store is None or meta_store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Delete vectors first
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is not None and getattr(rag_app, "ready", False):
        try:
            await rag_app.delete_knowledge_collection(name, delete_folder=False)
        except Exception as exc:
            logger.warning("Failed to delete vector collection '%s': %s", name, exc)
        # Clean up registration
        rag_app.collection_paths.pop(name, None)
        rag_app.vector_managers.pop(name, None)
        rag_app.available_collections.discard(name)

    await meta_store.delete_source_files(name)
    deleted = await store.delete_source(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="WebDAV source not found")
    return {"success": True}


@router.post("/knowledge/webdav/{name}/sync")
async def webdav_sync_source(
    request: Request,
    name: str,
) -> dict:
    """Manually trigger a WebDAV sync."""
    await require_permission(request, "knowledge:write")
    config = getattr(request.app.state, "app_config", None)
    rag_app = getattr(request.app.state, "rag_application", None)
    store = _get_webdav_source_store(request)
    meta_store = _get_webdav_meta_store(request)
    if store is None or meta_store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if rag_app is None or not getattr(rag_app, "ready", False):
        raise HTTPException(status_code=503, detail="RAG application not ready")

    progress_store = getattr(request.app.state, "_webdav_sync_progress", None)
    if progress_store is None:
        progress_store = {}
        request.app.state._webdav_sync_progress = progress_store

    cancel_store = getattr(request.app.state, "_webdav_sync_cancel", None)
    if cancel_store is None:
        cancel_store = {}
        request.app.state._webdav_sync_cancel = cancel_store
    cancel_store.pop(name, None)

    engine = WebDAVSyncEngine(source_store=store, meta_store=meta_store, config=config, progress_store=progress_store, cancel_store=cancel_store)
    result = await engine.sync_source(name, rag_app)
    cancel_store.pop(name, None)
    return result


@router.post("/knowledge/webdav/{name}/cancel")
async def webdav_cancel_sync(
    request: Request,
    name: str,
) -> dict:
    """Cancel a running WebDAV sync."""
    await require_permission(request, "knowledge:write")
    cancel_store = getattr(request.app.state, "_webdav_sync_cancel", None)
    if cancel_store is None:
        cancel_store = {}
        request.app.state._webdav_sync_cancel = cancel_store
    cancel_store[name] = True
    return {"success": True}


@router.post("/knowledge/webdav/test")
async def webdav_test_connection(
    request: Request,
    payload: WebDAVTestPayload,
) -> dict:
    """Test WebDAV connection by doing a PROPFIND on the remote path."""
    await require_permission(request, "knowledge:write")
    if not payload.webdav_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS WebDAV URLs are allowed")

    password = payload.password
    if not password and payload.name:
        store = _get_webdav_source_store(request)
        if store is not None:
            source = await store.get_source(payload.name)
            if source is not None:
                password = source.get("password", "")

    if not password:
        raise HTTPException(status_code=400, detail="Password is required for testing")

    from app.webdav_sync import WebDAVClient

    try:
        async with WebDAVClient(
            payload.webdav_url, payload.username, password
        ) as client:
            files = await client.list_files(payload.remote_path)
        return {"success": True, "file_count": len(files)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/knowledge/webdav/{name}/status")
async def webdav_source_status(
    request: Request,
    name: str,
) -> dict:
    """Return sync status and file count for a WebDAV source."""
    await require_permission(request, "knowledge:read")
    store = _get_webdav_source_store(request)
    meta_store = _get_webdav_meta_store(request)
    if store is None or meta_store is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    source = await store.get_source(name)
    if source is None:
        raise HTTPException(status_code=404, detail="WebDAV source not found")
    files = await meta_store.list_files(name)
    progress_store = getattr(request.app.state, "_webdav_sync_progress", {})
    sync_progress = progress_store.get(name)
    return {
        "name": name,
        "last_sync_at": source.get("last_sync_at").isoformat() if source.get("last_sync_at") else None,
        "last_sync_status": source.get("last_sync_status"),
        "last_sync_error": source.get("last_sync_error"),
        "file_count": len(files),
        "sync_progress": sync_progress,
    }


# ------------------------------------------------------------------------------
# Conversations endpoints
# ------------------------------------------------------------------------------

@router.get("/conversations/models")
async def conversations_models(
    request: Request,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[str]:
    await require_permission(request, "conversations:read")
    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    parsed_from = _parse_iso_datetime(date_from, "date_from") if date_from else None
    parsed_to = _parse_iso_datetime(date_to, "date_to") if date_to else None
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise HTTPException(status_code=400, detail="date_from must be less than or equal to date_to")

    return await store.list_models(date_from=parsed_from, date_to=parsed_to)


@router.get("/conversations/users")
async def conversations_users(request: Request) -> dict:
    """Users owning conversations — feeds the user filter dropdown."""
    await require_permission(request, "conversations:read")
    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")
    return await store.list_conversation_users()


@router.get("/conversations")
async def conversations_list(
    request: Request,
    params: Annotated[ConversationListParams, Depends()],
) -> dict:
    await require_permission(request, "conversations:read")
    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    parsed_from = _parse_iso_datetime(params.date_from, "date_from") if params.date_from else None
    parsed_to = _parse_iso_datetime(params.date_to, "date_to") if params.date_to else None
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise HTTPException(status_code=400, detail="date_from must be less than or equal to date_to")

    result = await store.list_conversations(
        date_from=parsed_from,
        date_to=parsed_to,
        conversation_query=params.conversation_id,
        model_name=params.model,
        min_user_messages=params.min_user_messages,
        user_id=params.user_id,
        query=params.query,
        page=params.page,
        page_size=params.page_size,
        sort_by=params.sort_by,
        sort_order=params.sort_order,
    )

    total = int(result["total"])
    page_size = int(result["page_size"])
    total_pages = max(1, (total + page_size - 1) // page_size)
    return {
        "total": total,
        "page": int(result["page"]),
        "page_size": page_size,
        "total_pages": total_pages,
        "items": [_normalize_row(item) for item in result["items"]],
    }


@router.get("/conversations/export")
async def conversations_export(
    request: Request,
    params: Annotated[ConversationExportParams, Depends()],
) -> Response:
    await require_permission(request, "conversations:export")
    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    export_format = params.format.lower().strip()
    if export_format not in {"json", "csv", "txt"}:
        raise HTTPException(status_code=400, detail="format must be one of: json, csv, txt")

    parsed_from = _parse_iso_datetime(params.date_from, "date_from")
    parsed_to = _parse_iso_datetime(params.date_to, "date_to")
    if parsed_from > parsed_to:
        raise HTTPException(status_code=400, detail="date_from must be less than or equal to date_to")
    if parsed_to - parsed_from > timedelta(days=31):
        raise HTTPException(status_code=400, detail="Export date range must not exceed 31 days")

    rows = await store.list_messages_for_export(
        date_from=parsed_from,
        date_to=parsed_to,
        conversation_query=params.conversation_id,
        model_name=params.model,
        min_user_messages=params.min_user_messages,
        user_id=params.user_id,
        query=params.query,
    )
    normalized_rows = [_normalize_row(row) for row in rows]

    # Enrich grouped conversations with title + owner info
    get_conv_users = getattr(store, "get_conversation_users", None)
    user_lookup = (
        await get_conv_users(
            list({str(row.get("conversation_id") or "") for row in normalized_rows if row.get("conversation_id")})
        )
        if get_conv_users
        else {}
    )

    # Build grouped conversations (one entry per conversation)
    grouped: dict[str, dict[str, Any]] = {}
    for row in normalized_rows:
        cid = str(row.get("conversation_id") or "")
        if cid not in grouped:
            info = user_lookup.get(cid) or {}
            grouped[cid] = {
                "conversation_id": cid,
                "title": info.get("title"),
                "user_id": info.get("user_id") or row.get("user_id"),
                "username": info.get("username"),
                "display_name": info.get("display_name"),
                "message_count": 0,
                "user_message_count": 0,
                "assistant_message_count": 0,
                "first_message_at": row.get("created_at"),
                "last_message_at": row.get("created_at"),
                "models": set(),
                "messages": [],
            }
        bucket = grouped[cid]
        bucket["message_count"] += 1
        if row.get("user_content"):
            bucket["user_message_count"] += 1
        if row.get("assistant_content"):
            bucket["assistant_message_count"] += 1
        if row.get("model_name"):
            bucket["models"].add(row["model_name"])
        if row.get("created_at") and (bucket["first_message_at"] is None or row["created_at"] < bucket["first_message_at"]):
            bucket["first_message_at"] = row["created_at"]
        if row.get("created_at") and (bucket["last_message_at"] is None or row["created_at"] > bucket["last_message_at"]):
            bucket["last_message_at"] = row["created_at"]
        bucket["messages"].append(row)

    conversations: list[dict[str, Any]] = []
    for conversation in grouped.values():
        msgs = conversation.pop("messages", [])
        lines: list[str] = []
        for msg in msgs:
            ts = msg.get("created_at", "")
            if msg.get("user_content"):
                lines.append(f"User [{ts}]:\n{msg['user_content']}")
            if msg.get("assistant_content"):
                lines.append(f"Assistant [{ts}]:\n{msg['assistant_content']}")
        conversation["conversation_text"] = "\n\n".join(lines)
        conversation["models"] = sorted(list(conversation["models"]))
        conversations.append(conversation)

    conversations = sorted(conversations, key=lambda item: item.get("last_message_at") or "", reverse=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if export_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "conversation_id",
                "title",
                "user",
                "message_count",
                "user_message_count",
                "assistant_message_count",
                "first_message_at",
                "last_message_at",
                "models",
                "conversation_text",
            ]
        )
        for conv in conversations:
            writer.writerow(
                [
                    conv.get("conversation_id"),
                    conv.get("title") or "",
                    conv.get("display_name") or conv.get("username") or conv.get("user_id") or "",
                    conv.get("message_count"),
                    conv.get("user_message_count"),
                    conv.get("assistant_message_count"),
                    conv.get("first_message_at") or "",
                    conv.get("last_message_at") or "",
                    ",".join(conv.get("models", [])),
                    conv.get("conversation_text") or "",
                ]
            )
        content = output.getvalue()
        headers = {
            "Content-Disposition": f'attachment; filename="conversations_{stamp}.csv"',
        }
        return Response(content=content, media_type="text/csv", headers=headers)

    if export_format == "txt":
        lines: list[str] = []
        for i, conv in enumerate(conversations, start=1):
            lines.append(f"=== Conversation {i} ===")
            lines.append(f"ID: {conv.get('conversation_id')}")
            if conv.get("title"):
                lines.append(f"Title: {conv.get('title')}")
            owner = conv.get("display_name") or conv.get("username") or conv.get("user_id")
            if owner:
                lines.append(f"User: {owner}")
            lines.append(f"Messages: {conv.get('message_count')} (User: {conv.get('user_message_count')}, Assistant: {conv.get('assistant_message_count')})")
            lines.append(f"Models: {', '.join(conv.get('models', []) or ['—'])}")
            lines.append(f"First: {conv.get('first_message_at') or '—'}")
            lines.append(f"Last: {conv.get('last_message_at') or '—'}")
            lines.append("")
            lines.append(conv.get("conversation_text", ""))
            lines.append("")
            lines.append("=" * 50)
            lines.append("")
        content = "\n".join(lines)
        headers = {
            "Content-Disposition": f'attachment; filename="conversations_{stamp}.txt"',
        }
        return Response(content=content, media_type="text/plain; charset=utf-8", headers=headers)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "date_from": parsed_from.isoformat(),
            "date_to": parsed_to.isoformat(),
            "conversation_id": params.conversation_id,
            "model": params.model,
            "min_user_messages": params.min_user_messages,
            "user_id": params.user_id,
        },
        "total_conversations": len(conversations),
        "total_messages": len(normalized_rows),
        "conversations": conversations,
    }
    headers = {
        "Content-Disposition": f'attachment; filename="conversations_{stamp}.json"',
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json",
        headers=headers,
    )


@router.get("/conversations/{conversation_id}")
async def conversation_detail(request: Request, conversation_id: str) -> dict:
    await require_permission(request, "conversations:read")
    cid = conversation_id.strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id must not be empty")

    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    messages = await store.get_conversation_messages(cid)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": cid,
        "messages": [_normalize_row(row) for row in messages],
    }


@router.get("/conversations/{conversation_id}/metadata")
async def conversation_metadata(request: Request, conversation_id: str) -> dict:
    await require_permission(request, "conversations:read")
    cid = conversation_id.strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id must not be empty")

    store = _get_message_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Message store unavailable")

    metadata = await store.get_conversation_metadata(cid)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = _normalize_row(metadata)

    # Chat metrics
    metrics_store = _get_metrics_store(request)
    if metrics_store is not None:
        chat_metrics = await metrics_store.get_conversation_metrics(cid)
        result["chat_metrics"] = chat_metrics
    else:
        result["chat_metrics"] = None

    # Vectorstore metrics
    vs_store = _get_vectorstore_metrics_store(request)
    if vs_store is not None:
        vs_metrics = await vs_store.get_conversation_metrics(cid)
        result["vectorstore_metrics"] = vs_metrics
    else:
        result["vectorstore_metrics"] = None

    # Emotion summary
    messages = await store.get_conversation_messages(cid)
    emotion_counts: dict[str, int] = {}
    text_count = 0
    webcam_count = 0
    for msg in messages:
        emotion = msg.get("emotion")
        if emotion and isinstance(emotion, dict):
            primary = emotion.get("primary_emotion", "unknown")
            emotion_counts[primary] = emotion_counts.get(primary, 0) + 1
            source = emotion.get("source", "")
            if "text" in source:
                text_count += 1
            elif "webcam" in source:
                webcam_count += 1
    dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "—"
    result["emotion_summary"] = {
        "dominant": dominant,
        "text_count": text_count,
        "webcam_count": webcam_count,
    }

    return result


# ------------------------------------------------------------------------------
# Vectorstore metrics endpoints
# ------------------------------------------------------------------------------

@router.get("/vectorstore/summary")
async def vectorstore_summary(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> dict:
    await require_permission(request, "vectorstore:read")
    store = _get_vectorstore_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")
    return await store.get_summary(params.range_hours)


@router.get("/vectorstore/by-model")
async def vectorstore_by_model(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "vectorstore:read")
    store = _get_vectorstore_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")
    return await store.get_by_model(params.range_hours)


@router.get("/vectorstore/history")
async def vectorstore_history(
    request: Request,
    params: Annotated[MetricsRangeParams, Depends()],
) -> list[dict]:
    await require_permission(request, "vectorstore:read")
    store = _get_vectorstore_metrics_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Vectorstore metrics store unavailable")
    return await store.get_history(params.range_hours, params.bucket, params.model)


@router.get("/vectorstore/state")
async def vectorstore_state(request: Request) -> dict:
    """Return live vectorstore state: collections, document counts, health."""
    await require_permission(request, "vectorstore:read")
    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG application unavailable")
    collections = []
    total_documents = 0
    for name, manager in getattr(rag_app, "vector_managers", {}).items():
        doc_count = await manager._get_document_count()
        health = await manager.health_check()
        collections.append({
            "name": name,
            "documents": doc_count,
            "healthy": health,
        })
        total_documents += doc_count
    return {
        "collections": collections,
        "total_collections": len(collections),
        "total_documents": total_documents,
    }


# ------------------------------------------------------------------------------
# Runtime configuration endpoints
# ------------------------------------------------------------------------------

class RuntimeConfigItemPayload(BaseModel):
    key: str
    value: Any


class RuntimeConfigPayload(BaseModel):
    values: Dict[str, Any]


class ProviderPayload(BaseModel):
    id: str
    name: str
    base_url: str
    enabled: bool = True
    is_default: bool = False
    provider_type: str = "openai-compatible"
    timeout_seconds: int = 30
    max_retries: int = 3
    extra_headers: Optional[Dict[str, str]] = None
    supports_native_tool_calls: Optional[bool] = None


class SecretRotatePayload(BaseModel):
    api_key: str


def _get_runtime_config_store(request: Request):
    store = getattr(request.app.state, "_runtime_config_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Runtime config store unavailable")
    return store


def _get_provider_store(request: Request):
    store = getattr(request.app.state, "_provider_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Provider store unavailable")
    return store


def _schedule_provider_model_refresh(request: Request) -> None:
    """Kick a background rebuild of the model->provider membership map.

    Fire-and-forget: the refresh does a ``/v1/models`` fetch per provider
    and must never block the admin request. Endpoint details (URL, key,
    headers, timeouts) propagate without it — only membership needs it.
    """
    rag = getattr(request.app.state, "rag_application", None)
    schedule = getattr(rag, "schedule_provider_model_refresh", None)
    if callable(schedule):
        try:
            schedule()
        except Exception:
            logger.debug("Could not schedule provider model refresh", exc_info=True)


def _get_api_key_store(request: Request):
    store = getattr(request.app.state, "_api_key_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="API key store unavailable")
    return store


# --- Config helpers ---

_EDITABLE_CONFIG_KEYS = {
    # Vector / embeddings
    "vector_search_k",
    "vector_batch_size",
    "vector_force_rebuild",
    "embedding_model",
    "embeddings_request_timeout",
    "embeddings_max_retries",
    "embeddings_backoff_factor",
    "embeddings_api_url",
    # Tuning
    "system_slow",
    "api_reload",
    "auto_initialize",
    "user_agent",
    "stream_idle_timeout_seconds",
    "run_idle_timeout_seconds",
    "retriever_timeout_seconds",
    # Logging
    "log_level",
    "log_file_max_bytes",
    # OCR
    "tesseract_cmd",
    "ocr_language",
    "ocr_mode",
    "ocr_llm_prompt",
    "ocr_llm_concurrency",
    "pdf_image_extraction_enabled",
    "pdf_image_extraction_mode",
    "pptx_slide_analysis_enabled",
    "pptx_slide_analysis_mode",
    # Chat image upload
    "chat_image_upload_enabled",
    "chat_image_max_size_bytes",
    "chat_image_max_count",
    "chat_image_allowed_types",
    # Chat file attachments
    "chat_file_upload_enabled",
    "chat_file_max_size_bytes",
    "chat_file_max_count",
    "chat_file_allowed_exts",
    "chat_file_max_extract_chars",
    "chat_file_max_total_chars",
    "chat_file_extract_timeout_seconds",
    # Chat file lifecycle (storage sweeper)
    "chat_files_orphan_ttl_days",
    "chat_files_retention_days",
    # Validation / abuse prevention
    "max_request_bytes",
    "max_messages",
    "max_message_chars",
    "max_total_chars",
    "require_last_user",
    "allow_non_string_content",
    "validation_control_chars_mode",
    "validation_enforce",
    # History
    "max_history_messages",
    "rolling_history_enabled",
    "rolling_history_context_window",
    "rolling_history_max_output_tokens",
    "rolling_history_rag_reserved_tokens",
    "rolling_history_min_recent_turns",
    "rolling_history_safety_margin_ratio",
    "rolling_history_summary_timeout_seconds",
    # Loader
    "loader_max_concurrency",
    "chunk_size",
    "chunk_overlap",
    # Retrieval
    "retrieval_hybrid_enabled",
    "hybrid_vector_candidates",
    "hybrid_keyword_candidates",
    "hybrid_rrf_k",
    "hybrid_max_chunks_per_source",
    "hybrid_query_expansion_enabled",
    "hybrid_max_expanded_queries",
    "hybrid_rerank_enabled",
    "hybrid_min_semantic_similarity",
    "retrieval_history_enabled",
    "retrieval_history_user_messages",
    "retrieval_history_max_chars",
    "retrieval_history_standalone_detection",
    # Contextualization
    "contextualize_question_enabled",
    "contextualize_question_model",
    "contextualize_question_temperature",
    "contextualize_question_disable_reasoning",
    # Indexing
    "indexing_incremental_enabled",
    # API key enforcement
    "api_keys_enabled",
    # Audio
    "audio_stt_enabled",
    "audio_tts_enabled",
    "audio_tts_voices_dir",
    "audio_tts_default_voice",
    "audio_tts_piper_binary",
    "audio_tts_streaming_enabled",
    "audio_tts_provider",
    "audio_tts_kokoro_voice",
    "audio_tts_kokoro_model_dir",
    "audio_stt_provider",
    "audio_stt_model",
    "audio_stt_language",
    "hf_cache_dir",
    "hf_token",
    # Emotion Detection
    "emotion_detection_enabled",
    "emotion_text_model_enabled",
    "emotion_text_model_url",
    "emotion_text_model_download_url",
    "emotion_text_tokenizer_url",
    "emotion_text_tokenizer_download_url",
    "emotion_webcam_enabled",
    "emotion_prompt_steering_enabled",
    "emotion_analytics_enabled",
    "emotion_min_confidence",
    "emotion_webcam_runtime",
    "emotion_webcam_execution_providers",
    "emotion_webcam_face_detector",
    "emotion_webcam_face_model_url",
    "emotion_webcam_face_model_download_url",
    "emotion_webcam_expression_model_url",
    "emotion_webcam_expression_model_download_url",
    "emotion_webcam_expression_model_type",
    "emotion_webcam_expression_input_size",
    "emotion_webcam_snippet_window_ms",
    "emotion_webcam_sample_fps",
    "emotion_webcam_max_frames_per_snippet",
    "emotion_webcam_min_valid_face_frames",
    "emotion_webcam_frame_upload_enabled",
    "emotion_audio_enabled",
    "emotion_audio_model_url",
    "emotion_audio_model_download_url",
    "emotion_fusion_source_weights",
    "emotion_fusion_min_confidence",
}


_CONFIG_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    # Vector & Embeddings
    "vector_search_k": {
        "label": "Vector Search K",
        "description": "Number of top-k chunks retrieved per collection from the vector store.",
        "type": "integer",
        "default": 5,
        "category": "Vector & Embeddings",
    },
    "vector_batch_size": {
        "label": "Vector Batch Size",
        "description": "Batch size for PGVector embedding inserts during ingestion.",
        "type": "integer",
        "default": 200,
        "category": "Vector & Embeddings",
    },
    "vector_force_rebuild": {
        "label": "Force Vector Rebuild",
        "description": "When true, re-ingests all documents into PGVector on every startup.",
        "type": "boolean",
        "default": False,
        "category": "Vector & Embeddings",
    },
    "embedding_model": {
        "label": "Embedding Model",
        "description": "Model name passed to the embeddings endpoint (e.g. nomic-embed-text, text-embedding-multilingual-e5-base).",
        "type": "string",
        "default": "nomic-embed-text",
        "category": "Vector & Embeddings",
    },
    "embeddings_request_timeout": {
        "label": "Embeddings Request Timeout",
        "description": "Timeout in seconds for each embedding request.",
        "type": "integer",
        "default": 30,
        "category": "Vector & Embeddings",
    },
    "embeddings_max_retries": {
        "label": "Embeddings Max Retries",
        "description": "Number of retry attempts for failed embedding calls.",
        "type": "integer",
        "default": 3,
        "category": "Vector & Embeddings",
    },
    "embeddings_backoff_factor": {
        "label": "Embeddings Backoff Factor",
        "description": "Exponential backoff factor between embedding retries.",
        "type": "float",
        "default": 0.5,
        "category": "Vector & Embeddings",
    },
    "embeddings_api_url": {
        "label": "Embeddings API URL",
        "description": "Base URL of the OpenAI-compatible embeddings API. When unset, falls back to the default LLM API base.",
        "type": "string",
        "default": None,
        "category": "Vector & Embeddings",
    },
    # Tuning / Runtime
    "system_slow": {
        "label": "System Slow",
        "description": "Set to true if the upstream LLM is slow; adjusts default timeout values upward.",
        "type": "boolean",
        "default": True,
        "category": "Runtime & API",
    },
    "api_reload": {
        "label": "API Reload",
        "description": "Enable FastAPI auto-reload in development. Disable in production.",
        "type": "boolean",
        "default": False,
        "category": "Runtime & API",
    },
    "auto_initialize": {
        "label": "Auto Initialize",
        "description": "Automatically load the RAG pipeline at startup. Disable for lazy/manual initialization.",
        "type": "boolean",
        "default": True,
        "category": "Runtime & API",
    },
    "user_agent": {
        "label": "User Agent",
        "description": "HTTP User-Agent string used by internal HTTP clients.",
        "type": "string",
        "default": "Agentix-RAG/1.0",
        "category": "Runtime & API",
    },
    "stream_idle_timeout_seconds": {
        "label": "Stream Idle Timeout",
        "description": "Seconds without a token from an in-flight LLM stream before that call is aborted (per model call inside a run).",
        "type": "float",
        "default": 300.0,
        "category": "Runtime & API",
    },
    "run_idle_timeout_seconds": {
        "label": "Run Idle Timeout",
        "description": "Seconds without real stream activity (token, reasoning, tool or subagent event) before a run is aborted; synthetic tool keepalives do not count. No fixed total-run limit. Should stay above the sandbox install+exec budgets so healthy code_execution jobs fit one silent window.",
        "type": "float",
        "default": 900.0,
        "category": "Runtime & API",
    },
    "retriever_timeout_seconds": {
        "label": "Retriever Timeout",
        "description": "Timeout in seconds for per-collection async retriever calls.",
        "type": "float",
        "default": 10.0,
        "category": "Runtime & API",
    },
    # Logging
    "log_level": {
        "label": "Log Level",
        "description": "Logging verbosity: DEBUG, INFO, WARNING, or ERROR.",
        "type": "string",
        "default": "INFO",
        "category": "Logging",
    },
    "log_file_max_bytes": {
        "label": "Log File Max Bytes",
        "description": "Maximum size of a single log file before rotation occurs.",
        "type": "integer",
        "default": 5_242_880,
        "category": "Logging",
    },
    # OCR
    "tesseract_cmd": {
        "label": "Tesseract Command",
        "description": "Absolute path to the Tesseract binary. Leave empty to use PATH lookup.",
        "type": "string",
        "default": None,
        "category": "OCR",
    },
    "ocr_language": {
        "label": "OCR Language",
        "description": "Tesseract language pack code (e.g. deu, deu+eng, eng).",
        "type": "string",
        "default": "deu",
        "category": "OCR",
    },
    "ocr_mode": {
        "label": "OCR Mode",
        "description": "Controls how standalone images are processed: tesseract (classic OCR), llm (LLM vision analysis), auto (LLM with Tesseract fallback).",
        "type": "select",
        "default": "tesseract",
        "options": [
            {"value": "tesseract", "label": "Tesseract"},
            {"value": "llm", "label": "LLM Vision"},
            {"value": "auto", "label": "Auto (LLM + Tesseract fallback)"},
        ],
        "category": "OCR",
    },
    "ocr_llm_prompt": {
        "label": "OCR LLM Prompt",
        "description": "Prompt sent to the LLM when analyzing images during ingestion.",
        "type": "string",
        "default": "Extract all text and relevant information from this image.",
        "category": "OCR",
    },
    "ocr_llm_concurrency": {
        "label": "OCR LLM Concurrency",
        "description": "Number of parallel LLM vision calls during ingestion. Separate semaphore so text loading is not blocked.",
        "type": "integer",
        "default": 4,
        "category": "OCR",
    },
    "pdf_image_extraction_enabled": {
        "label": "PDF Image Extraction",
        "description": "When enabled, embedded images in PDFs are extracted and analyzed via LLM or Tesseract.",
        "type": "boolean",
        "default": False,
        "category": "OCR",
    },
    "pdf_image_extraction_mode": {
        "label": "PDF Image Extraction Mode",
        "description": "How extracted PDF images are processed: llm (LLM vision analysis) or tesseract (classic OCR).",
        "type": "select",
        "default": "llm",
        "options": [
            {"value": "llm", "label": "LLM Vision"},
            {"value": "tesseract", "label": "Tesseract"},
        ],
        "category": "OCR",
    },
    "pptx_slide_analysis_enabled": {
        "label": "PowerPoint Slide Analysis",
        "description": "When enabled, PPTX slides are rendered as images and analyzed via LLM or Tesseract. Requires LibreOffice for rendering.",
        "type": "boolean",
        "default": False,
        "category": "OCR",
    },
    "pptx_slide_analysis_mode": {
        "label": "PPTX Slide Analysis Mode",
        "description": "How rendered PPTX slides are processed: llm (LLM vision analysis) or tesseract (classic OCR).",
        "type": "select",
        "default": "llm",
        "options": [
            {"value": "llm", "label": "LLM Vision"},
            {"value": "tesseract", "label": "Tesseract"},
        ],
        "category": "OCR",
    },
    # Chat image upload
    "chat_image_upload_enabled": {
        "label": "Chat Image Upload",
        "description": "When enabled, users can upload images in chat. Images are sent directly to the LLM as multimodal input (requires vision-capable provider).",
        "type": "boolean",
        "default": False,
        "category": "Chat Images",
    },
    "chat_image_max_size_bytes": {
        "label": "Chat Image Max Size (bytes)",
        "description": "Maximum size per uploaded image in bytes. Default: 5 MB.",
        "type": "integer",
        "default": 5_242_880,
        "category": "Chat Images",
    },
    "chat_image_max_count": {
        "label": "Chat Image Max Count",
        "description": "Maximum number of images allowed per chat message.",
        "type": "integer",
        "default": 4,
        "category": "Chat Images",
    },
    "chat_image_allowed_types": {
        "label": "Chat Image Allowed Types",
        "description": "Comma-separated list of allowed MIME types for uploaded chat images.",
        "type": "string",
        "default": "image/png,image/jpeg,image/webp,image/gif",
        "category": "Chat Images",
    },
    # Chat file attachments
    "chat_file_upload_enabled": {
        "label": "Chat File Upload",
        "description": "When enabled, authenticated users can attach documents (docx, xlsx, pptx, pdf, …) in chat. Extracted text is injected into subsequent turns.",
        "type": "boolean",
        "default": False,
        "category": "Chat Files",
    },
    "chat_file_max_size_bytes": {
        "label": "Chat File Max Size (bytes)",
        "description": "Maximum size per uploaded attachment in bytes. Default: 50 MB.",
        "type": "integer",
        "default": 52_428_800,
        "category": "Chat Files",
    },
    "chat_file_max_count": {
        "label": "Chat File Max Count",
        "description": "Maximum number of attachments stored per conversation.",
        "type": "integer",
        "default": 5,
        "category": "Chat Files",
    },
    "chat_file_allowed_exts": {
        "label": "Chat File Allowed Extensions",
        "description": "Comma-separated list of allowed file extensions for chat attachments. Images are extracted via OCR/vision.",
        "type": "string",
        "default": ".docx,.xlsx,.pptx,.pdf,.odt,.csv,.txt,.md,.doc,.xls,.ppt,.png,.jpg,.jpeg,.webp,.gif,.bmp,.tiff",
        "category": "Chat Files",
    },
    "chat_file_max_extract_chars": {
        "label": "Chat File Max Extract Chars",
        "description": "Maximum extracted text characters kept per attachment; longer content is truncated.",
        "type": "integer",
        "default": 50_000,
        "category": "Chat Files",
    },
    "chat_file_max_total_chars": {
        "label": "Chat File Max Total Chars",
        "description": "Maximum total extracted text characters injected per turn across all attachments.",
        "type": "integer",
        "default": 120_000,
        "category": "Chat Files",
    },
    "chat_file_extract_timeout_seconds": {
        "label": "Chat File Extract Timeout (s)",
        "description": "Timeout in seconds for extracting text from an uploaded attachment. Raise for large decks/documents.",
        "type": "number",
        "default": 120.0,
        "category": "Chat Files",
    },
    "chat_files_orphan_ttl_days": {
        "label": "Chat Files Orphan TTL (days)",
        "description": "File dirs (artifacts, uploads) whose conversation has no record are deleted once older than this. Covers uploads to never-finished conversations and client-forgotten resets.",
        "type": "integer",
        "default": 7,
        "category": "Chat Files",
    },
    "chat_files_retention_days": {
        "label": "Chat Files Retention (days)",
        "description": "Optional: also delete the FILES of conversations whose record is older than this. Conversation and message history are kept — files are not. 0 = keep files forever.",
        "type": "integer",
        "default": 0,
        "category": "Chat Files",
    },
    # Validation
    "max_request_bytes": {
        "label": "Max Request Bytes",
        "description": "Hard cap for incoming request body size in bytes. Set 0 to disable (not recommended).",
        "type": "integer",
        "default": 1_048_576,
        "category": "Request Validation",
    },
    "max_messages": {
        "label": "Max Messages",
        "description": "Maximum number of messages allowed in a single chat/completions request.",
        "type": "integer",
        "default": 100,
        "category": "Request Validation",
    },
    "max_message_chars": {
        "label": "Max Message Characters",
        "description": "Maximum characters allowed per individual message content.",
        "type": "integer",
        "default": 20_000,
        "category": "Request Validation",
    },
    "max_total_chars": {
        "label": "Max Total Characters",
        "description": "Maximum characters allowed across all messages in a single request.",
        "type": "integer",
        "default": 120_000,
        "category": "Request Validation",
    },
    "require_last_user": {
        "label": "Require Last User",
        "description": "Require the final message in a request to be from the user (OpenAI-style semantics).",
        "type": "boolean",
        "default": True,
        "category": "Request Validation",
    },
    "allow_non_string_content": {
        "label": "Allow Non-String Content",
        "description": "When true, allows list/dict message content (e.g. tool blocks). When false, rejects them.",
        "type": "boolean",
        "default": False,
        "category": "Request Validation",
    },
    "validation_control_chars_mode": {
        "label": "Control Character Mode",
        "description": "Policy for control characters in message content: allow or reject.",
        "type": "string",
        "default": "reject",
        "category": "Request Validation",
    },
    "validation_enforce": {
        "label": "Validation Enforce",
        "description": "When true, blocks requests that fail validation. When false, only logs the issue.",
        "type": "boolean",
        "default": True,
        "category": "Request Validation",
    },
    # History
    "max_history_messages": {
        "label": "Max History Messages",
        "description": "Maximum prior messages (excluding the latest user message) forwarded to the LLM. Set 0 to disable truncation.",
        "type": "integer",
        "default": 50,
        "category": "History",
    },
    "rolling_history_enabled": {
        "label": "Rolling History Enabled",
        "description": "Enable token-budget summarization to keep conversation history within the context window.",
        "type": "boolean",
        "default": True,
        "category": "History",
    },
    "rolling_history_context_window": {
        "label": "Rolling Context Window",
        "description": "Token budget for the entire conversation history (including system prompt and RAG context).",
        "type": "integer",
        "default": 32_768,
        "category": "History",
    },
    "rolling_history_max_output_tokens": {
        "label": "Rolling Max Output Tokens",
        "description": "Expected maximum tokens for the LLM response, reserved from the context window.",
        "type": "integer",
        "default": 2_048,
        "category": "History",
    },
    "rolling_history_rag_reserved_tokens": {
        "label": "Rolling RAG Reserved Tokens",
        "description": "Tokens reserved for RAG-retrieved document chunks in the context window.",
        "type": "integer",
        "default": 0,
        "category": "History",
    },
    "rolling_history_min_recent_turns": {
        "label": "Rolling Min Recent Turns",
        "description": "Minimum number of recent conversation turns always kept (not summarized away).",
        "type": "integer",
        "default": 4,
        "category": "History",
    },
    "rolling_history_safety_margin_ratio": {
        "label": "Rolling Safety Margin Ratio",
        "description": "Safety margin ratio applied to token budget calculations to avoid overflow.",
        "type": "float",
        "default": 0.08,
        "category": "History",
    },
    "rolling_history_summary_timeout_seconds": {
        "label": "Rolling Summary Timeout",
        "description": "Timeout in seconds for conversation summary generation.",
        "type": "float",
        "default": 8.0,
        "category": "History",
    },
    # Loader
    "loader_max_concurrency": {
        "label": "Loader Max Concurrency",
        "description": "Maximum concurrent OCR/parsing tasks during document ingestion.",
        "type": "integer",
        "default": 8,
        "category": "Indexing & Processing",
    },
    "chunk_size": {
        "label": "Chunk Size",
        "description": "Character size per document chunk before embedding.",
        "type": "integer",
        "default": 1500,
        "category": "Indexing & Processing",
    },
    "chunk_overlap": {
        "label": "Chunk Overlap",
        "description": "Character overlap between consecutive chunks to preserve context across splits.",
        "type": "integer",
        "default": 200,
        "category": "Indexing & Processing",
    },
    "indexing_incremental_enabled": {
        "label": "Incremental Indexing",
        "description": "When true, re-embeds only changed/new documents. Recommended when docs rarely change.",
        "type": "boolean",
        "default": True,
        "category": "Indexing & Processing",
    },
    # Retrieval
    "retrieval_hybrid_enabled": {
        "label": "Hybrid Retrieval Enabled",
        "description": "Enable hybrid retrieval combining PGVector similarity with Postgres FTS + pg_trgm.",
        "type": "boolean",
        "default": True,
        "category": "Retrieval",
    },
    "hybrid_vector_candidates": {
        "label": "Hybrid Vector Candidates",
        "description": "Number of candidates to fetch from vector similarity search before fusion.",
        "type": "integer",
        "default": 20,
        "category": "Retrieval",
    },
    "hybrid_keyword_candidates": {
        "label": "Hybrid Keyword Candidates",
        "description": "Number of candidates to fetch from keyword (FTS) search before fusion.",
        "type": "integer",
        "default": 20,
        "category": "Retrieval",
    },
    "hybrid_rrf_k": {
        "label": "Hybrid RRF K",
        "description": "Reciprocal Rank Fusion constant K. Higher values reduce the impact of rank position.",
        "type": "integer",
        "default": 60,
        "category": "Retrieval",
    },
    "hybrid_max_chunks_per_source": {
        "label": "Max Chunks Per Source",
        "description": "Maximum chunks returned from the same source document to avoid duplication.",
        "type": "integer",
        "default": 3,
        "category": "Retrieval",
    },
    "hybrid_query_expansion_enabled": {
        "label": "Query Expansion Enabled",
        "description": "Enable LLM-based query expansion to generate additional search terms.",
        "type": "boolean",
        "default": True,
        "category": "Retrieval",
    },
    "hybrid_max_expanded_queries": {
        "label": "Max Expanded Queries",
        "description": "Maximum number of expanded query variants generated per user question.",
        "type": "integer",
        "default": 2,
        "category": "Retrieval",
    },
    "hybrid_rerank_enabled": {
        "label": "Hybrid Rerank Enabled",
        "description": "Enable semantic re-ranking of fused retrieval results for better relevance ordering.",
        "type": "boolean",
        "default": True,
        "category": "Retrieval",
    },
    "hybrid_min_semantic_similarity": {
        "label": "Min Semantic Similarity",
        "description": "Minimum cosine similarity threshold for a chunk to be included in RAG context. 0.0 disables filtering.",
        "type": "float",
        "default": 0.4,
        "category": "Retrieval",
    },
    "retrieval_history_enabled": {
        "label": "Retrieval History Enabled",
        "description": "Lightweight history-aware retrieval: build query from recent user messages without an extra LLM call.",
        "type": "boolean",
        "default": True,
        "category": "Retrieval",
    },
    "retrieval_history_user_messages": {
        "label": "Retrieval History User Messages",
        "description": "Number of recent user messages to include in the history-aware retrieval query.",
        "type": "integer",
        "default": 2,
        "category": "Retrieval",
    },
    "retrieval_history_max_chars": {
        "label": "Retrieval History Max Chars",
        "description": "Character cap for the combined history query string passed to the retriever.",
        "type": "integer",
        "default": 800,
        "category": "Retrieval",
    },
    "retrieval_history_standalone_detection": {
        "label": "Standalone Detection",
        "description": "Detect standalone definition/explanation queries and skip history mixing for them.",
        "type": "boolean",
        "default": True,
        "category": "Retrieval",
    },
    # Contextualization
    "contextualize_question_enabled": {
        "label": "Contextualize Question",
        "description": "Use an extra LLM call to rewrite follow-up questions into standalone queries. Adds latency but improves retrieval.",
        "type": "boolean",
        "default": False,
        "category": "Contextualization",
    },
    "contextualize_question_model": {
        "label": "Contextualization Model",
        "description": "Optional model override for the contextualization step (e.g. a smaller/faster model).",
        "type": "string",
        "default": None,
        "category": "Contextualization",
    },
    "contextualize_question_temperature": {
        "label": "Contextualization Temperature",
        "description": "Sampling temperature for the contextualization rewrite. Usually kept at 0.0 for determinism.",
        "type": "float",
        "default": 0.0,
        "category": "Contextualization",
    },
    "contextualize_question_disable_reasoning": {
        "label": "Disable Reasoning for Contextualization",
        "description": "Disable reasoning mode for the rewrite step to reduce latency.",
        "type": "boolean",
        "default": True,
        "category": "Contextualization",
    },
    # API keys
    "api_keys_enabled": {
        "label": "API Keys Enabled",
        "description": "Enforce API key authentication for /v1/chat/completions requests.",
        "type": "boolean",
        "default": True,
        "category": "API Keys",
    },
    # Audio
    "audio_stt_enabled": {
        "label": "STT Enabled",
        "description": "Enable speech-to-text (faster-whisper) globally.",
        "type": "boolean",
        "default": False,
        "category": "Audio",
    },
    "audio_tts_enabled": {
        "label": "TTS Enabled",
        "description": "Enable text-to-speech (Piper) globally.",
        "type": "boolean",
        "default": False,
        "category": "Audio",
    },
    "audio_tts_voices_dir": {
        "label": "TTS Voices Directory",
        "description": "Path where Piper voice ONNX files are stored.",
        "type": "string",
        "default": "data/voices/tts_voices",
        "category": "Audio",
    },
    "audio_tts_default_voice": {
        "label": "TTS Default Voice",
        "description": "Default voice ID used when no assistant-specific voice is set.",
        "type": "string",
        "default": None,
        "category": "Audio",
    },
    "audio_tts_piper_binary": {
        "label": "Piper Binary Path",
        "description": "Path or name of the piper executable. Use full path if not in PATH (e.g. /venv/bin/piper).",
        "type": "string",
        "default": "piper",
        "category": "Audio",
    },
    "audio_tts_streaming_enabled": {
        "label": "TTS Streaming Enabled",
        "description": "Stream TTS audio during assistant response generation.",
        "type": "boolean",
        "default": True,
        "category": "Audio",
    },
    "audio_tts_provider": {
        "label": "TTS Provider",
        "description": "Primary TTS engine. 'kokoro' uses Kokoro ONNX (better German). 'piper' uses Piper CLI.",
        "type": "string",
        "default": "kokoro",
        "category": "Audio",
    },
    "audio_tts_kokoro_voice": {
        "label": "Kokoro Voice",
        "description": "Voice ID for Kokoro TTS (German model uses 'martin'). Only used when provider is kokoro.",
        "type": "string",
        "default": "martin",
        "category": "Audio",
    },
    "audio_tts_kokoro_model_dir": {
        "label": "Kokoro Model Directory",
        "description": "Directory where Kokoro model caches are stored.",
        "type": "string",
        "default": "data/models/voices/kokoro_models",
        "category": "Audio",
    },
    "audio_stt_provider": {
        "label": "STT Provider",
        "description": "Speech-to-text backend (faster_whisper or distil_whisper). distil_whisper requires transformers+torch.",
        "type": "string",
        "default": "faster_whisper",
        "category": "Audio",
    },
    "audio_stt_model": {
        "label": "STT Model",
        "description": "Model size/ID. faster_whisper: tiny, base, small, medium, large. distil_whisper: distil-whisper/distil-large-v3, distil-whisper/distil-medium.en, etc.",
        "type": "string",
        "default": "small",
        "category": "Audio",
    },
    "audio_stt_language": {
        "label": "STT Language",
        "description": "Language code for speech-to-text (e.g. de, en, fr).",
        "type": "string",
        "default": "de",
        "category": "Audio",
    },
    "hf_cache_dir": {
        "label": "HF Cache Directory",
        "description": "Directory for caching HuggingFace models (e.g. /app/cache/hf). Leave empty for default.",
        "type": "string",
        "default": None,
        "category": "Audio",
    },
    "hf_token": {
        "label": "HF Token",
        "description": "HuggingFace API token for authenticated (faster) model downloads.",
        "type": "string",
        "default": None,
        "category": "Audio",
    },
    # Emotion Detection
    "emotion_detection_enabled": {
        "label": "Emotion Detection Enabled",
        "description": "Master switch for emotion detection (text + webcam).",
        "type": "boolean",
        "default": False,
        "category": "Emotion Detection",
    },
    "emotion_text_model_enabled": {
        "label": "Text Emotion Model Enabled",
        "description": "Enable ONNX-based text emotion detection from user messages (German GoEmotions).",
        "type": "boolean",
        "default": True,
        "category": "Emotion Detection",
    },
    "emotion_text_model_url": {
        "label": "Text Emotion Model URL",
        "description": "URL/path to the ONNX text emotion model.",
        "type": "string",
        "default": "/models/emotion/text-emotion.onnx",
        "category": "Emotion Detection",
    },
    "emotion_text_model_download_url": {
        "label": "Text Emotion Model Download URL",
        "description": "External URL to download the text emotion ONNX model if missing.",
        "type": "string",
        "default": "https://huggingface.co/tojohere/goemotions-de-xlm-roberta-base/resolve/main/goemotions.onnx",
        "category": "Emotion Detection",
    },
    "emotion_text_tokenizer_url": {
        "label": "Text Emotion Tokenizer URL",
        "description": "URL/path to the tokenizer.json for the text emotion model.",
        "type": "string",
        "default": "/models/emotion/text-emotion-tokenizer.json",
        "category": "Emotion Detection",
    },
    "emotion_text_tokenizer_download_url": {
        "label": "Text Emotion Tokenizer Download URL",
        "description": "External URL to download the tokenizer.json if missing.",
        "type": "string",
        "default": "https://huggingface.co/tojohere/goemotions-de-xlm-roberta-base/resolve/main/tokenizer.json",
        "category": "Emotion Detection",
    },
    "emotion_webcam_enabled": {
        "label": "Webcam Emotion Enabled",
        "description": "Allow browser-local webcam emotion detection when user opts in.",
        "type": "boolean",
        "default": False,
        "category": "Emotion Detection",
    },
    "emotion_prompt_steering_enabled": {
        "label": "Emotion Prompt Steering Enabled",
        "description": "Allow detected emotions to gently steer assistant tone.",
        "type": "boolean",
        "default": True,
        "category": "Emotion Detection",
    },
    "emotion_analytics_enabled": {
        "label": "Emotion Analytics Enabled",
        "description": "Persist emotion metadata per message for analytics.",
        "type": "boolean",
        "default": True,
        "category": "Emotion Detection",
    },
    "emotion_min_confidence": {
        "label": "Emotion Min Confidence",
        "description": "Minimum confidence threshold before an emotion signal is considered for steering.",
        "type": "float",
        "default": 0.55,
        "category": "Emotion Detection",
    },
    "emotion_fusion_source_weights": {
        "label": "Emotion Fusion Source Weights",
        "description": "JSON dict of source weights for emotion fusion. Keys: text, visual_pre, audio_prosody, phase_typing, phase_waiting_for_response, phase_reading.",
        "type": "json",
        "default": {
            "text": 1.0,
            "visual_pre": 1.0,
            "audio_prosody": 0.9,
            "phase_typing": 0.8,
            "phase_waiting_for_response": 0.6,
            "phase_reading": 0.4,
        },
        "category": "Emotion Detection",
    },
    "emotion_fusion_min_confidence": {
        "label": "Emotion Fusion Min Confidence",
        "description": "JSON dict of per-source min confidence thresholds. Keys: text, visual_pre, audio_prosody, phase.",
        "type": "json",
        "default": {
            "text": 0.55,
            "visual_pre": 0.55,
            "audio_prosody": 0.55,
            "phase": 0.55,
        },
        "category": "Emotion Detection",
    },
    "emotion_webcam_runtime": {
        "label": "Webcam Runtime",
        "description": "Browser runtime for webcam emotion inference (e.g. onnxruntime-web).",
        "type": "string",
        "default": "onnxruntime-web",
        "category": "Emotion Detection",
    },
    "emotion_webcam_execution_providers": {
        "label": "Execution Providers",
        "description": "Comma-separated preferred ONNX execution providers (webgpu,wasm).",
        "type": "string",
        "default": "webgpu,wasm",
        "category": "Emotion Detection",
    },
    "emotion_webcam_face_detector": {
        "label": "Face Detector",
        "description": "Face detection backend: onnx, browser, or mediapipe.",
        "type": "string",
        "default": "onnx",
        "category": "Emotion Detection",
    },
    "emotion_webcam_face_model_url": {
        "label": "Face Model URL",
        "description": "URL/path to the browser ONNX face detection model.",
        "type": "string",
        "default": "/models/emotion/face-detector.onnx",
        "category": "Emotion Detection",
    },
    "emotion_webcam_face_model_download_url": {
        "label": "Face Model Download URL",
        "description": "External URL to download the face detector ONNX model if missing (e.g. GitHub).",
        "type": "string",
        "default": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=",
        "category": "Emotion Detection",
    },
    "emotion_webcam_expression_model_url": {
        "label": "Expression Model URL",
        "description": "URL/path to the browser ONNX expression classification model.",
        "type": "string",
        "default": "/models/emotion/expression-affectnet.onnx",
        "category": "Emotion Detection",
    },
    "emotion_webcam_expression_model_download_url": {
        "label": "Expression Model Download URL",
        "description": "External URL to download the expression ONNX model if missing (e.g. GitHub).",
        "type": "string",
        "default": "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
        "category": "Emotion Detection",
    },
    "emotion_webcam_expression_model_type": {
        "label": "Expression Model Type",
        "description": "Model architecture type: fer (64x64 grayscale) or affectnet (224x224 RGB ImageNet).",
        "type": "string",
        "default": "affectnet",
        "category": "Emotion Detection",
    },
    "emotion_webcam_expression_input_size": {
        "label": "Expression Model Input Size",
        "description": "Input dimension (pixels) for the expression classifier (64 for FER, 224 for AffectNet).",
        "type": "integer",
        "default": 224,
        "category": "Emotion Detection",
    },
    "emotion_webcam_snippet_window_ms": {
        "label": "Snippet Window (ms)",
        "description": "Max duration of a webcam snippet batch in milliseconds.",
        "type": "integer",
        "default": 3000,
        "category": "Emotion Detection",
    },
    "emotion_webcam_sample_fps": {
        "label": "Sample FPS",
        "description": "Frames per second sampled from webcam for analysis.",
        "type": "integer",
        "default": 3,
        "category": "Emotion Detection",
    },
    "emotion_webcam_max_frames_per_snippet": {
        "label": "Max Frames Per Snippet",
        "description": "Maximum frames to analyze in one snippet batch.",
        "type": "integer",
        "default": 12,
        "category": "Emotion Detection",
    },
    "emotion_webcam_min_valid_face_frames": {
        "label": "Min Valid Face Frames",
        "description": "Minimum frames with a detected face required to trust a visual emotion result.",
        "type": "integer",
        "default": 3,
        "category": "Emotion Detection",
    },
    "emotion_webcam_frame_upload_enabled": {
        "label": "Frame Upload Enabled",
        "description": "Allow sending raw frames to the backend for server-side analysis (privacy-sensitive).",
        "type": "boolean",
        "default": False,
        "category": "Emotion Detection",
    },
    "emotion_audio_enabled": {
        "label": "Audio Emotion Enabled",
        "description": "Allow browser-local audio emotion detection when user opts in.",
        "type": "boolean",
        "default": False,
        "category": "Emotion Detection",
    },
    "emotion_audio_model_url": {
        "label": "Audio Model URL",
        "description": "URL/path to the browser ONNX audio emotion model.",
        "type": "string",
        "default": "/models/emotion/audio-emotion.onnx",
        "category": "Emotion Detection",
    },
    "emotion_audio_model_download_url": {
        "label": "Audio Model Download URL",
        "description": "External URL to download the audio emotion ONNX model if missing (e.g. HuggingFace).",
        "type": "string",
        "default": "https://huggingface.co/onnx-community/wav2vec2-emotion-recognition-ONNX/resolve/main/onnx/model_q4f16.onnx?download=true",
        "category": "Emotion Detection",
    },
}


def _build_effective_config(base: AppConfig, stored: Dict[str, Any]) -> Dict[str, Any]:
    """Merge stored config with env-backed AppConfig, marking overrides."""
    effective: Dict[str, Any] = {}
    env_backed = getattr(base, "_env_backed_values", {})
    for key in _EDITABLE_CONFIG_KEYS:
        base_val = env_backed.get(key, getattr(base, key, None))
        stored_val = stored.get(key)
        default_val = AppConfig.model_fields[key].default
        env_override = base_val != default_val
        effective[key] = {
            "value": base_val if env_override else (stored_val if stored_val is not None else base_val),
            "source": "env" if env_override else ("db" if stored_val is not None else "default"),
            "editable": not env_override,
            "default_value": default_val,
            "env_value": base_val if env_override else None,
        }
    return effective


@router.get("/config/runtime")
async def get_runtime_config(request: Request) -> dict:
    """Return stored and effective runtime config with source info."""
    await require_permission(request, "runtime_config:read")
    base = getattr(request.app.state, "app_config", None)
    if base is None:
        raise HTTPException(status_code=503, detail="App config not available")
    store = _get_runtime_config_store(request)
    stored = await store.get_all()
    effective = _build_effective_config(base, stored)
    return {
        "stored": stored,
        "effective": effective,
    }


@router.put("/config/runtime")
async def put_runtime_config(request: Request, payload: RuntimeConfigPayload) -> dict:
    """Persist runtime config values that are not overridden by env."""
    await require_permission(request, "runtime_config:write")
    base = getattr(request.app.state, "app_config", None)
    if base is None:
        raise HTTPException(status_code=503, detail="App config not available")
    store = _get_runtime_config_store(request)
    env_backed = getattr(base, "_env_backed_values", {})
    for key in payload.values:
        if key not in _EDITABLE_CONFIG_KEYS:
            raise HTTPException(status_code=400, detail=f"Key '{key}' is not editable")
        base_val = env_backed.get(key, getattr(base, key, None))
        default_val = AppConfig.model_fields[key].default
        if base_val != default_val:
            raise HTTPException(status_code=400, detail=f"Key '{key}' is overridden by environment")
    await store.set_many(payload.values)
    return {"success": True}


@router.post("/config/runtime/apply")
async def apply_runtime_config(request: Request) -> dict:
    """Apply stored runtime config by hot-reloading the RAG application."""
    await require_permission(request, "runtime_config:apply")
    from app.main import reload_rag_application
    await reload_rag_application(request.app)
    return {"message": "Runtime configuration applied."}


@router.get("/config/runtime/export")
async def export_runtime_config(request: Request) -> Response:
    """Export stored runtime config as JSON (secrets excluded)."""
    await require_permission(request, "runtime_config:import_export")
    store = _get_runtime_config_store(request)
    data = await store.export_all()
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="runtime_config.json"'},
    )


@router.post("/config/runtime/import")
async def import_runtime_config(request: Request, payload: RuntimeConfigPayload) -> dict:
    """Import runtime config values from JSON (skipping env-overridden keys)."""
    await require_permission(request, "runtime_config:import_export")
    base = getattr(request.app.state, "app_config", None)
    if base is None:
        raise HTTPException(status_code=503, detail="App config not available")
    store = _get_runtime_config_store(request)
    filtered = {}
    for key, value in payload.values.items():
        if key not in _EDITABLE_CONFIG_KEYS:
            continue
        base_val = getattr(base, key, None)
        default_val = AppConfig.model_fields[key].default
        if base_val != default_val:
            continue
        filtered[key] = value
    await store.import_many(filtered)
    return {"success": True, "imported_keys": list(filtered.keys())}


@router.get("/config/descriptions")
async def get_config_descriptions(request: Request) -> dict:
    """Return human-readable descriptions, types, defaults and categories for all editable config keys."""
    await require_permission(request, "runtime_config:read")
    return _CONFIG_DESCRIPTIONS


# ------------------------------------------------------------------------------
# LLM Provider endpoints
# ------------------------------------------------------------------------------

@router.get("/config/providers")
async def list_providers(request: Request) -> list[dict]:
    """Return all LLM providers without secrets, including env-based fallback."""
    await require_permission(request, "providers:read")
    store = _get_provider_store(request)
    providers = await store.list_providers()
    result = [
        {
            "id": p.id,
            "name": p.name,
            "base_url": p.base_url,
            "enabled": p.enabled,
            "is_default": p.is_default,
            "provider_type": p.provider_type,
            "timeout_seconds": p.timeout_seconds,
            "max_retries": p.max_retries,
            "extra_headers": p.extra_headers,
            "last_checked_at": p.last_checked_at,
            "last_check_status": p.last_check_status,
            "supports_native_tool_calls": p.supports_native_tool_calls,
            "tool_calling_checked_at": p.tool_calling_checked_at,
            "tool_calling_check_error": p.tool_calling_check_error,
            "has_api_key": await store.has_api_key(p.id),
            "source": "db",
        }
        for p in providers
    ]
    cfg = getattr(request.app.state, "app_config", None)
    if cfg and getattr(cfg, "llm_api_base", None):
        has_db_default = any(p["is_default"] for p in result)
        result.append({
            "id": "env-default",
            "name": "Environment LLM Provider",
            "base_url": cfg.llm_api_base,
            "enabled": True,
            "is_default": not has_db_default,
            "provider_type": "openai-compatible",
            "timeout_seconds": 30,
            "max_retries": 3,
            "extra_headers": None,
            "last_checked_at": None,
            "last_check_status": None,
            "supports_native_tool_calls": None,
            "tool_calling_checked_at": None,
            "tool_calling_check_error": None,
            "has_api_key": bool(getattr(cfg, "llm_api_key", None)),
            "source": "env",
        })
    return result


@router.get("/config/providers/{provider_id}")
async def get_provider(request: Request, provider_id: str) -> dict:
    """Return a single provider without secrets."""
    await require_permission(request, "providers:read")
    store = _get_provider_store(request)
    p = await store.get_provider(provider_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {
        "id": p.id,
        "name": p.name,
        "base_url": p.base_url,
        "enabled": p.enabled,
        "is_default": p.is_default,
        "provider_type": p.provider_type,
        "timeout_seconds": p.timeout_seconds,
        "max_retries": p.max_retries,
        "extra_headers": p.extra_headers,
        "last_checked_at": p.last_checked_at,
        "last_check_status": p.last_check_status,
        "supports_native_tool_calls": p.supports_native_tool_calls,
        "tool_calling_checked_at": p.tool_calling_checked_at,
        "tool_calling_check_error": p.tool_calling_check_error,
        "has_api_key": await store.has_api_key(p.id),
    }


@router.post("/config/providers")
async def create_provider(request: Request, payload: ProviderPayload) -> dict:
    """Create a new LLM provider."""
    await require_permission(request, "providers:write")
    if not payload.id or not payload.id.strip():
        raise HTTPException(status_code=400, detail="Provider ID is required")
    if not payload.name or not payload.name.strip():
        raise HTTPException(status_code=400, detail="Provider name is required")
    if not payload.base_url or not payload.base_url.strip():
        raise HTTPException(status_code=400, detail="Provider base_url is required")
    store = _get_provider_store(request)
    existing = await store.get_provider(payload.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Provider ID already exists")
    from app.llm_providers import LLMProvider
    provider = LLMProvider(
        id=payload.id,
        name=payload.name,
        base_url=payload.base_url,
        enabled=payload.enabled,
        is_default=payload.is_default,
        provider_type=payload.provider_type,
        timeout_seconds=payload.timeout_seconds,
        max_retries=payload.max_retries,
        extra_headers=payload.extra_headers,
        supports_native_tool_calls=payload.supports_native_tool_calls,
    )
    await store.create_provider(provider)
    _schedule_provider_model_refresh(request)
    return {"success": True}


@router.put("/config/providers/{provider_id}")
async def update_provider(request: Request, provider_id: str, payload: ProviderPayload) -> dict:
    """Update an existing LLM provider."""
    await require_permission(request, "providers:write")
    if not payload.name or not payload.name.strip():
        raise HTTPException(status_code=400, detail="Provider name is required")
    if not payload.base_url or not payload.base_url.strip():
        raise HTTPException(status_code=400, detail="Provider base_url is required")
    store = _get_provider_store(request)
    existing = await store.get_provider(provider_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    from app.llm_providers import LLMProvider
    provider = LLMProvider(
        id=provider_id,
        name=payload.name,
        base_url=payload.base_url,
        enabled=payload.enabled,
        is_default=payload.is_default,
        provider_type=payload.provider_type,
        timeout_seconds=payload.timeout_seconds,
        max_retries=payload.max_retries,
        extra_headers=payload.extra_headers,
        supports_native_tool_calls=payload.supports_native_tool_calls,
    )
    await store.update_provider(provider)
    _schedule_provider_model_refresh(request)
    return {"success": True}


@router.delete("/config/providers/{provider_id}")
async def delete_provider(request: Request, provider_id: str) -> dict:
    """Delete a provider and its secret."""
    await require_permission(request, "providers:write")
    store = _get_provider_store(request)
    existing = await store.get_provider(provider_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    await store.delete_provider(provider_id)
    _schedule_provider_model_refresh(request)
    return {"success": True}


@router.post("/config/providers/{provider_id}/secret/rotate")
async def rotate_provider_secret(
    request: Request, provider_id: str, payload: SecretRotatePayload
) -> dict:
    """Rotate the API key for a provider."""
    await require_permission(request, "providers:secrets")
    store = _get_provider_store(request)
    existing = await store.get_provider(provider_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    await store.rotate_api_key(provider_id, payload.api_key)
    _schedule_provider_model_refresh(request)
    return {"success": True}


@router.post("/config/providers/{provider_id}/check")
async def check_provider(request: Request, provider_id: str) -> dict:
    """Check provider connectivity by fetching models."""
    await require_permission(request, "providers:check")
    store = _get_provider_store(request)
    provider = await store.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    api_key = await store.get_api_key(provider_id)
    headers = dict(provider.extra_headers or {})
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
            resp = await client.get(f"{provider.base_url}/v1/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", []) if "id" in m]
            await store.update_check_status(provider_id, "ok")
            return {"status": "ok", "models": sorted(models)}
    except Exception as exc:
        logger.exception("Provider check failed for %s", provider_id)
        await store.update_check_status(provider_id, f"error: {exc}")
        raise HTTPException(status_code=502, detail=f"Provider check failed: {exc}")


@router.post("/config/providers/{provider_id}/detect-tool-calling")
async def detect_provider_tool_calling(request: Request, provider_id: str, payload: dict = {}) -> dict:
    """Manually trigger tool-calling detection for a provider."""
    await require_permission(request, "providers:check")
    store = _get_provider_store(request)
    provider = await store.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    model = payload.get("model")
    if not model:
        # Try to fetch first available model from provider
        models = await store.fetch_models(provider_id)
        if models:
            model = models[0]
    if not model:
        raise HTTPException(status_code=400, detail="No model specified and could not auto-detect one from provider")
    from app.tool_calling_detector import detect_tool_calling_support
    api_key = await store.get_api_key(provider_id)
    result = await detect_tool_calling_support(
        base_url=provider.base_url,
        api_key=api_key,
        model=model,
        timeout=float(provider.timeout_seconds),
        extra_headers=provider.extra_headers,
    )
    await store.update_tool_calling_status(
        provider_id=provider_id,
        supports_native=result.get("supports_native"),
        error=result.get("error"),
    )
    return {
        "success": True,
        "model": model,
        "supports_native": result.get("supports_native"),
        "used_content_json": result.get("used_content_json"),
        "error": result.get("error"),
    }


@router.post("/config/providers/{provider_id}/detect-reasoning")
async def detect_provider_reasoning(request: Request, provider_id: str, payload: dict = {}) -> dict:
    """Manually trigger live reasoning-capability detection for a provider model."""
    await require_permission(request, "providers:check")
    store = _get_provider_store(request)
    provider = await store.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    model = payload.get("model")
    if not model:
        models = await store.fetch_models(provider_id)
        if models:
            model = models[0]
    if not model:
        raise HTTPException(status_code=400, detail="No model specified and could not auto-detect one from provider")
    from app.reasoning_detector import detect_reasoning_support

    api_key = await store.get_api_key(provider_id)
    result = await detect_reasoning_support(
        base_url=provider.base_url,
        api_key=api_key,
        model=model,
        timeout=float(provider.timeout_seconds),
        extra_headers=provider.extra_headers,
    )
    persistable = (
        result.get("supports_effort") is not None
        or result.get("supports_thinking_toggle") is not None
    )
    await store.update_model_reasoning_status(
        provider_id=provider_id,
        model_id=model,
        capabilities=result if persistable else None,
        error=None if persistable else (result.get("error") or "inconclusive"),
    )
    return {
        "success": True,
        "model": model,
        "supports_effort": result.get("supports_effort"),
        "effort_levels": result.get("effort_levels"),
        "supports_thinking_toggle": result.get("supports_thinking_toggle"),
        "error": result.get("error"),
    }


@router.get("/config/reasoning-schema")
async def reasoning_schema(request: Request, model: str = "", provider_id: str = "") -> dict:
    """Describe the reasoning controls available for a given model.

    Returns a capability descriptor (family, wire format, UI controls) used by
    the assistant config editor to render model-aware reasoning options.
    """
    await require_permission(request, "providers:read")
    base_url = None
    provider = None
    store = None
    if provider_id:
        try:
            store = _get_provider_store(request)
            provider = await store.get_provider(provider_id)
        except HTTPException:
            provider = None
        if provider is not None:
            base_url = provider.base_url

    # Prefer live-probed capabilities when a provider/model pair was checked —
    # but only for families the probe dimensions can actually describe.
    if provider is not None and store is not None and model:
        try:
            from app.reasoning_detector import ensure_detected_for_provider

            status = await ensure_detected_for_provider(store, provider_id, model)
            if status and status.get("capabilities"):
                if probe_overrides_family(match_family(model, base_url)):
                    descriptor = descriptor_from_capabilities(status["capabilities"], model)
                    if descriptor is not None:
                        return descriptor
        except Exception:
            logger.exception("Reasoning capability lookup failed for %s/%s", provider_id, model)

    return describe_for_model(model or None, base_url)


@router.get("/config/models")
async def config_models(request: Request) -> List[str]:
    """Fetch available models from all enabled upstream LLM APIs."""
    await require_permission(request, "providers:read")
    store = _get_provider_store(request)
    providers = await store.get_enabled_providers()
    if not providers:
        cfg = getattr(request.app.state, "app_config", None)
        if cfg is None:
            raise HTTPException(status_code=503, detail="App config not available")
        base = normalize_base_url(cfg.llm_api_base)
        api_key = cfg.llm_api_key
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", []) if "id" in m]
                return sorted(models)
        except Exception:
            logger.exception("Failed to fetch models from upstream LLM API")
            raise HTTPException(status_code=502, detail="Could not fetch models from upstream LLM API")

    all_models: set[str] = set()
    for provider in providers:
        api_key = await store.get_api_key(provider.id)
        headers = dict(provider.extra_headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
                resp = await client.get(f"{provider.base_url}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", []) if "id" in m]
                all_models.update(models)
        except Exception:
            logger.warning("Failed to fetch models from provider %s", provider.id)
    return sorted(all_models)


@router.get("/config/provider-models")
async def config_provider_models(request: Request) -> Dict[str, Dict[str, Any]]:
    """Fetch available models per provider from all enabled upstream LLM APIs."""
    await require_permission(request, "providers:read")
    store = _get_provider_store(request)
    providers = await store.get_enabled_providers()
    result: Dict[str, Dict[str, Any]] = {}

    for provider in providers:
        api_key = await store.get_api_key(provider.id)
        headers = dict(provider.extra_headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
                resp = await client.get(f"{provider.base_url}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", []) if "id" in m]
                result[provider.id] = {"name": provider.name, "models": sorted(models)}
        except Exception:
            logger.warning("Failed to fetch models from provider %s", provider.id)
            result[provider.id] = {"name": provider.name, "models": []}

    cfg = getattr(request.app.state, "app_config", None)
    if cfg and getattr(cfg, "llm_api_base", None):
        api_key = cfg.llm_api_key
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{normalize_base_url(cfg.llm_api_base)}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", []) if "id" in m]
                result["env-default"] = {"name": "Environment LLM Provider", "models": sorted(models)}
        except Exception:
            logger.exception("Failed to fetch models from env-default upstream LLM API")
            result["env-default"] = {"name": "Environment LLM Provider", "models": []}
    return result


@router.get("/config/providers/export")
async def export_providers(request: Request) -> Response:
    """Export providers without secrets as JSON."""
    await require_permission(request, "providers:import_export")
    store = _get_provider_store(request)
    data = await store.export_providers()
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="providers.json"'},
    )


@router.post("/config/providers/import")
async def import_providers(request: Request, payload: ProviderPayload) -> dict:
    """Import providers from JSON."""
    await require_permission(request, "providers:import_export")
    store = _get_provider_store(request)
    from app.llm_providers import LLMProvider
    for item in payload.providers if hasattr(payload, "providers") else [payload]:
        provider = LLMProvider(
            id=item.id,
            name=item.name,
            base_url=item.base_url,
            enabled=item.enabled,
            is_default=item.is_default,
            provider_type=item.provider_type,
            timeout_seconds=item.timeout_seconds,
            max_retries=item.max_retries,
            extra_headers=item.extra_headers,
        )
        existing = await store.get_provider(provider.id)
        if existing:
            await store.update_provider(provider)
        else:
            await store.create_provider(provider)
    _schedule_provider_model_refresh(request)
    return {"success": True}


# ==============================================================================
# API Key Management
# ==============================================================================

class ApiKeyCreatePayload(BaseModel):
    name: str
    allowed_models: List[str] = []
    enabled: bool = True
    rate_limit_rpm: int = Field(default=0, ge=0)
    rate_limit_rpd: int = Field(default=0, ge=0)


class ApiKeyUpdatePayload(BaseModel):
    name: str
    allowed_models: List[str] = []
    enabled: bool = True
    rate_limit_rpm: int = Field(default=0, ge=0)
    rate_limit_rpd: int = Field(default=0, ge=0)


@router.get("/api-keys")
async def list_api_keys(request: Request) -> List[Dict[str, Any]]:
    """List all API keys (without hashes)."""
    await require_permission(request, "api_keys:read")
    store = _get_api_key_store(request)
    keys = await store.list_keys()
    return [
        {
            "id": k.id,
            "name": k.name,
            "allowed_models": k.allowed_models,
            "enabled": k.enabled,
            "rate_limit_rpm": k.rate_limit_rpm,
            "rate_limit_rpd": k.rate_limit_rpd,
            "created_at": k.created_at,
            "updated_at": k.updated_at,
            "source": "env" if k.id == "legacy" else "db",
        }
        for k in keys
    ]


@router.post("/api-keys")
async def create_api_key(request: Request, payload: ApiKeyCreatePayload) -> Dict[str, Any]:
    """Create a new API key. The raw key is returned once and never stored in plain text."""
    await require_permission(request, "api_keys:write")
    store = _get_api_key_store(request)
    raw_key = store.generate_key()
    from app.api_keys import AppApiKey
    key = AppApiKey(
        id=raw_key.replace("ak_", "")[:16],
        name=payload.name,
        api_key_hash=store.hash_key(raw_key),
        allowed_models=payload.allowed_models,
        enabled=payload.enabled,
        rate_limit_rpm=payload.rate_limit_rpm,
        rate_limit_rpd=payload.rate_limit_rpd,
    )
    await store.create_key(key)
    return {
        "id": key.id,
        "name": key.name,
        "api_key": raw_key,
        "allowed_models": key.allowed_models,
        "enabled": key.enabled,
        "rate_limit_rpm": key.rate_limit_rpm,
        "rate_limit_rpd": key.rate_limit_rpd,
    }


@router.put("/api-keys/{key_id}")
async def update_api_key(request: Request, key_id: str, payload: ApiKeyUpdatePayload) -> Dict[str, Any]:
    """Update an existing API key (name, allowed models, enabled)."""
    await require_permission(request, "api_keys:write")
    store = _get_api_key_store(request)
    existing = await store.get_key(key_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="API key not found")
    from app.api_keys import AppApiKey
    key = AppApiKey(
        id=key_id,
        name=payload.name,
        api_key_hash=existing.api_key_hash,
        allowed_models=payload.allowed_models,
        enabled=payload.enabled,
        rate_limit_rpm=payload.rate_limit_rpm,
        rate_limit_rpd=payload.rate_limit_rpd,
    )
    await store.update_key(key)
    return {
        "id": key.id,
        "name": key.name,
        "allowed_models": key.allowed_models,
        "enabled": key.enabled,
        "rate_limit_rpm": key.rate_limit_rpm,
        "rate_limit_rpd": key.rate_limit_rpd,
    }


@router.delete("/api-keys/{key_id}")
async def delete_api_key(request: Request, key_id: str) -> Dict[str, Any]:
    """Delete an API key."""
    await require_permission(request, "api_keys:write")
    store = _get_api_key_store(request)
    existing = await store.get_key(key_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="API key not found")
    await store.delete_key(key_id)
    return {"success": True}


@router.get("/api-keys/models")
async def list_api_key_models(request: Request) -> List[str]:
    """List available assistant/model names for API key whitelisting."""
    await require_permission(request, "api_keys:read")
    rag_application = getattr(request.app.state, "rag_application", None)
    if rag_application is None:
        raise HTTPException(status_code=503, detail="RAG application not available")
    instructions = getattr(rag_application, "instructions", {})
    return sorted(instructions.keys())


# ------------------------------------------------------------------------------
# User management endpoints
# ------------------------------------------------------------------------------

class UserCreatePayload(BaseModel):
    username: str
    display_name: str
    password: str
    role: str


class UserUpdatePayload(BaseModel):
    username: str
    display_name: str
    role: str
    is_active: bool = True


class SetPasswordPayload(BaseModel):
    password: str


def _env_admin_dict(config) -> Optional[Dict[str, Any]]:
    """Return a virtual user dict for the env admin if credentials are configured."""
    password = getattr(config, "dashboard_admin_password", "")
    if not password:
        return None
    return {
        "id": "env-admin",
        "username": getattr(config, "dashboard_admin_username", "admin"),
        "display_name": "Environment Admin",
        "role": "owner",
        "is_active": True,
        "created_at": None,
        "updated_at": None,
        "last_login_at": None,
    }


@router.get("/users")
async def list_users(request: Request) -> List[Dict[str, Any]]:
    """List all dashboard users (without password hashes)."""
    await require_permission(request, "users:read")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")
    users = await store.list_users()
    result = [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "updated_at": u.updated_at.isoformat() if u.updated_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        }
        for u in users
    ]
    config = getattr(request.app.state, "app_config", None)
    env_admin = _env_admin_dict(config) if config else None
    if env_admin is not None:
        usernames = {u["username"] for u in result}
        if env_admin["username"] not in usernames:
            result.append(env_admin)
    return result


@router.post("/users")
async def create_user(request: Request, payload: UserCreatePayload) -> Dict[str, Any]:
    """Create a new dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    import secrets
    from app.dashboard_users import DashboardUser

    existing = await store.get_by_username(payload.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    if payload.role not in ("viewer", "operator", "editor", "admin", "owner"):
        raise HTTPException(status_code=400, detail="Invalid role")

    user = DashboardUser(
        id=secrets.token_hex(8),
        username=payload.username,
        display_name=payload.display_name,
        password_hash=store._hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    await store.create_user(user)
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "is_active": user.is_active,
    }


@router.put("/users/{user_id}")
async def update_user(
    request: Request, user_id: str, payload: UserUpdatePayload
) -> Dict[str, Any]:
    """Update a dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot modify environment admin")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.role not in ("viewer", "operator", "editor", "admin", "owner"):
        raise HTTPException(status_code=400, detail="Invalid role")

    # Prevent removing the last owner
    if user.role == "owner" and payload.role != "owner":
        owner_count = await store.count_active_owners()
        env_admin = _env_admin_dict(getattr(request.app.state, "app_config", None))
        if owner_count <= 1 and env_admin is None:
            raise HTTPException(status_code=400, detail="Cannot remove the last owner")

    # Prevent deactivating the last admin/owner
    if user.is_active and not payload.is_active and user.role in ("owner", "admin"):
        admin_count = await store.count_active_admins()
        env_admin = _env_admin_dict(getattr(request.app.state, "app_config", None))
        if admin_count <= 1 and env_admin is None:
            raise HTTPException(status_code=400, detail="Cannot deactivate the last admin")

    # If role changed, bump session version to invalidate existing sessions
    role_changed = user.role != payload.role

    user.username = payload.username
    user.display_name = payload.display_name
    user.role = payload.role
    user.is_active = payload.is_active
    await store.update_user(user)

    if role_changed:
        await store.bump_session_version(user_id)

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "is_active": user.is_active,
    }


@router.post("/users/{user_id}/password")
async def set_user_password(
    request: Request, user_id: str, payload: SetPasswordPayload
) -> Dict[str, Any]:
    """Set a new password for a dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot modify environment admin")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await store.set_password(user_id, payload.password)
    return {"success": True}


@router.post("/users/{user_id}/disable")
async def disable_user(request: Request, user_id: str) -> Dict[str, Any]:
    """Disable a dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot modify environment admin")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role in ("owner", "admin"):
        admin_count = await store.count_active_admins()
        env_admin = _env_admin_dict(getattr(request.app.state, "app_config", None))
        if admin_count <= 1 and env_admin is None:
            raise HTTPException(status_code=400, detail="Cannot deactivate the last admin")

    await store.set_active(user_id, False)
    return {"success": True}


@router.post("/users/{user_id}/enable")
async def enable_user(request: Request, user_id: str) -> Dict[str, Any]:
    """Enable a dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot modify environment admin")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await store.set_active(user_id, True)
    return {"success": True}


@router.delete("/users/{user_id}")
async def delete_user(request: Request, user_id: str) -> Dict[str, Any]:
    """Delete a dashboard user."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="User store unavailable")

    if user_id == "env-admin":
        raise HTTPException(status_code=403, detail="Cannot modify environment admin")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role == "owner":
        owner_count = await store.count_active_owners()
        env_admin = _env_admin_dict(getattr(request.app.state, "app_config", None))
        if owner_count <= 1 and env_admin is None:
            raise HTTPException(status_code=400, detail="Cannot delete the last owner")

    await store.delete_user(user_id)

    # A linked public-chat identity must not outlive the dashboard account —
    # it would keep logging in with its own stored password.
    chat_store = getattr(request.app.state, "_chat_user_store", None)
    if chat_store is not None:
        try:
            linked = await chat_store.get_by_dashboard_user_id(user_id)
            if linked is not None:
                await _purge_chat_user(request, linked.id)
        except Exception:
            logger.exception(
                "Linked chat user purge failed for dashboard user %s", user_id
            )
    return {"success": True}


# ------------------------------------------------------------------------------
# Agent Memory endpoints
# ------------------------------------------------------------------------------

class AgentMemoryCheckPayload(BaseModel):
    conversation_ids: List[str]


class AgentMemoryCheckResponse(BaseModel):
    conversation_ids_with_memory: List[str]


class AgentMemorySetPayload(BaseModel):
    assistant_name: str
    memory: str


@router.post("/agent-memory/check")
async def agent_memory_check(
    request: Request,
    payload: AgentMemoryCheckPayload,
) -> AgentMemoryCheckResponse:
    """Return the subset of conversation_ids that have agent memory entries."""
    await require_permission(request, "conversations:read")
    store = _get_agent_memory_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent memory store unavailable")
    result = await store.check_conversations_with_memory(payload.conversation_ids)
    return AgentMemoryCheckResponse(conversation_ids_with_memory=result)


@router.get("/agent-memory/{conversation_id}")
async def agent_memory_get(
    request: Request,
    conversation_id: str,
) -> Dict[str, Any]:
    """Get all agent memory entries for a conversation."""
    await require_permission(request, "conversations:read")
    store = _get_agent_memory_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent memory store unavailable")
    memories = await store.list_memories_for_conversation(conversation_id)
    return {"conversation_id": conversation_id, "memories": memories}


@router.put("/agent-memory/{conversation_id}")
async def agent_memory_set(
    request: Request,
    conversation_id: str,
    payload: AgentMemorySetPayload,
) -> Dict[str, Any]:
    """Set agent memory for a specific assistant in a conversation."""
    await require_permission(request, "conversations:write")
    store = _get_agent_memory_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent memory store unavailable")
    await store.set_memory(conversation_id, payload.assistant_name, payload.memory)
    return {"success": True, "conversation_id": conversation_id, "assistant_name": payload.assistant_name}


@router.delete("/agent-memory/{conversation_id}")
async def agent_memory_delete(
    request: Request,
    conversation_id: str,
    assistant_name: str,
) -> Dict[str, Any]:
    """Delete agent memory for a specific assistant in a conversation."""
    await require_permission(request, "conversations:write")
    store = _get_agent_memory_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent memory store unavailable")
    deleted = await store.delete_memory(conversation_id, assistant_name)
    return {"success": True, "deleted": deleted, "conversation_id": conversation_id, "assistant_name": assistant_name}


# --- Widget Defaults ---

# Keys that must never be persisted into the publicly readable widget defaults.
_WIDGET_DEFAULTS_STRIP_KEYS = {
    "apiKey",
    "apiSecret",
    "onOpen",
    "onInit",
    "onError",
}


def _sanitize_widget_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop secrets and non-serializable callbacks before storing as public defaults."""
    cleaned = {k: v for k, v in dict(config).items() if k not in _WIDGET_DEFAULTS_STRIP_KEYS}
    if cleaned.get("language") == "auto":
        cleaned.pop("language", None)
    return cleaned


class WidgetDefaultsPayload(BaseModel):
    config: Dict[str, Any]


@router.post("/widget-defaults")
async def widget_defaults_save(
    request: Request,
    payload: WidgetDefaultsPayload,
) -> Dict[str, Any]:
    """Save new widget defaults. Requires widget_defaults:write (admin/owner)."""
    await require_permission(request, "widget_defaults:write")
    store = _get_widget_defaults_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Widget defaults store unavailable")
    app_config = getattr(request.app.state, "app_config", None)
    session = read_session_cookie(request, app_config) if app_config else None
    created_by = (session or {}).get("username") or "admin"
    await store.save(_sanitize_widget_defaults(payload.config), created_by=created_by)
    return {"success": True}


# ------------------------------------------------------------------------------
# Chat Users (Public Chat)
# ------------------------------------------------------------------------------

def _get_groups_store(request: Request) -> Optional[ChatGroupsStore]:
    """Shared ChatGroupsStore on app.state (lazy — None without a DB engine)."""
    store = getattr(request.app.state, "_chat_groups_store", None)
    if store is not None:
        return store
    rag_application = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is None or not hasattr(engine, "begin"):
        return None
    store = ChatGroupsStore(engine)
    request.app.state._chat_groups_store = store
    return store


def _group_to_dict(g: ChatGroup) -> Dict[str, Any]:
    return {
        "id": g.id,
        "name": g.name,
        "description": g.description,
        "color": g.color,
        "allowed_models": g.allowed_models,
        "auto_domains": g.auto_domains,
        "member_count": g.member_count,
        "created_at": g.created_at.isoformat() if g.created_at else None,
        "updated_at": g.updated_at.isoformat() if g.updated_at else None,
    }


@router.get("/chat-users")
async def list_chat_users(request: Request) -> List[Dict[str, Any]]:
    """List all public chat users."""
    await require_permission(request, "users:read")
    store = getattr(request.app.state, "_chat_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")
    users = await store.list_users()
    groups_store = _get_groups_store(request)
    groups_per_user: Dict[str, List[Dict[str, str]]] = {}
    if groups_store is not None:
        groups_per_user = await groups_store.group_ids_for_users(
            [u.id for u in users]
        )
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "email": u.email,
            "role": u.role,
            "is_active": u.is_active,
            "dashboard_user_id": u.dashboard_user_id,
            "source": "dashboard" if u.dashboard_user_id else "public",
            "groups": groups_per_user.get(u.id, []),
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "updated_at": u.updated_at.isoformat() if u.updated_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        }
        for u in users
    ]


class ChatUserUpdatePayload(BaseModel):
    display_name: str
    role: str
    is_active: bool = True
    email: Optional[str] = None


@router.put("/chat-users/{user_id}")
async def update_chat_user(
    request: Request, user_id: str, payload: ChatUserUpdatePayload
) -> Dict[str, Any]:
    """Update a public chat user (role, display name, active status)."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_chat_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Chat user not found")

    user.display_name = payload.display_name
    user.role = payload.role
    user.is_active = payload.is_active
    if payload.email is not None:
        new_email = payload.email.strip() or None
        email_changed = new_email != user.email
        user.email = new_email
    else:
        email_changed = False
    await store.update_user(user)

    if email_changed:
        groups_store = _get_groups_store(request)
        if groups_store is not None:
            await groups_store.auto_assign_user(user)

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
    }


class ChatUserPasswordPayload(BaseModel):
    password: str = Field(..., min_length=6, max_length=100)


@router.post("/chat-users/{user_id}/password")
async def reset_chat_user_password(
    request: Request, user_id: str, payload: ChatUserPasswordPayload
) -> Dict[str, Any]:
    """Reset a public chat user's password."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_chat_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Chat user not found")

    await store.set_password(user_id, payload.password)
    return {"success": True}


async def _purge_chat_user(request: Request, chat_user_id: str) -> None:
    """Delete a chat user plus everything they own.

    Removes every conversation the user is linked to (link row, artifacts,
    uploads, agent memory, sandbox, workspace dirs AND the message content
    itself), then the ``chat_users`` row — group memberships cascade via FK.
    """
    app_state = request.app.state
    conv_store = getattr(app_state, "_chat_conversations_store", None)
    if conv_store is not None:
        try:
            links = await conv_store.list_for_user(chat_user_id)
        except Exception:
            logger.exception("Conversation link lookup failed for %s", chat_user_id)
            links = []
        rag_app = getattr(app_state, "rag_application", None)
        engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
        from app.conversation_cleanup import cleanup_conversation_data
        from sqlalchemy import text as sa_text

        for link in links:
            cid = link.get("conversation_id")
            if not cid:
                continue
            await cleanup_conversation_data(app_state, cid)
            if engine is not None:
                try:
                    async with engine.begin() as conn:
                        await conn.execute(
                            sa_text(
                                "DELETE FROM messages WHERE conversation_id = :c"
                            ),
                            {"c": cid},
                        )
                except Exception:
                    logger.exception("Message purge failed for %s", cid)
    store = getattr(app_state, "_chat_user_store", None)
    if store is not None:
        await store.delete_user(chat_user_id)


@router.delete("/chat-users/{user_id}")
async def delete_chat_user(request: Request, user_id: str) -> Dict[str, Any]:
    """Delete a public chat user and all their data."""
    await require_permission(request, "users:write")
    store = getattr(request.app.state, "_chat_user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    user = await store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Chat user not found")

    await _purge_chat_user(request, user_id)
    return {"success": True}


# ------------------------------------------------------------------------------
# Chat Role Assistants
# ------------------------------------------------------------------------------

class ChatRoleAssistantsPayload(BaseModel):
    role: str
    assistant_names: List[str]


class ChatRoleAssistantsBatchPayload(BaseModel):
    mappings: Dict[str, List[str]]


@router.get("/chat-roles")
async def list_chat_roles(request: Request) -> Dict[str, Any]:
    """Get all role-to-assistant mappings."""
    require_admin(request)
    store = getattr(request.app.state, "_chat_role_assistants_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Role assistants store unavailable")
    mappings = await store.get_all_mappings()
    return {"mappings": mappings}


@router.post("/chat-roles")
async def save_chat_roles(
    request: Request,
    payload: ChatRoleAssistantsBatchPayload,
) -> Dict[str, Any]:
    """Save role-to-assistant mappings."""
    require_admin(request)
    store = getattr(request.app.state, "_chat_role_assistants_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Role assistants store unavailable")
    for role, assistant_names in payload.mappings.items():
        await store.set_assistants_for_role(role, assistant_names)
    return {"success": True}


# ------------------------------------------------------------------------------
# Chat Groups
# ------------------------------------------------------------------------------

class ChatGroupPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    color: str = ""
    # None = unrestricted; [] = the group grants access to nothing.
    allowed_models: Optional[List[str]] = None
    auto_domains: List[str] = []


class ChatGroupMembersPayload(BaseModel):
    user_ids: List[str]


def _normalize_group_payload(payload: ChatGroupPayload) -> ChatGroup:
    return ChatGroup(
        id="",
        name=payload.name.strip(),
        description=payload.description.strip(),
        color=payload.color.strip(),
        allowed_models=(
            None if payload.allowed_models is None
            else [str(m).strip() for m in payload.allowed_models if str(m).strip()]
        ),
        auto_domains=sorted({
            d for d in (normalize_domain(x) for x in payload.auto_domains) if d
        }),
    )


@router.get("/chat-groups")
async def list_chat_groups(request: Request) -> List[Dict[str, Any]]:
    """List all public-chat groups (with member counts)."""
    await require_permission(request, "users:read")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    return [_group_to_dict(g) for g in await store.list_groups()]


@router.post("/chat-groups")
async def create_chat_group(
    request: Request, payload: ChatGroupPayload
) -> Dict[str, Any]:
    """Create a public-chat group."""
    await require_permission(request, "users:write")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    group = _normalize_group_payload(payload)
    if not group.name:
        raise HTTPException(status_code=422, detail="Group name must not be empty")
    existing = await store.get_group_by_name(group.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Group name already exists")
    import secrets

    group.id = f"chatgrp-{secrets.token_hex(8)}"
    await store.create_group(group)
    return _group_to_dict(group)


@router.get("/chat-groups/{group_id}/widget-link")
async def chat_group_widget_link(request: Request, group_id: str) -> Dict[str, Any]:
    """Mint a group-scoped public-chat link (``/chat?pct=<signed token>``)."""
    await require_permission(request, "users:read")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    if await store.get_group(group_id) is None:
        raise HTTPException(status_code=404, detail="Group not found")
    app_config = getattr(request.app.state, "app_config", None)
    if app_config is None or not getattr(app_config, "dashboard_session_secret", None):
        raise HTTPException(status_code=503, detail="Token signing unavailable")
    from app.routers.public_chat_auth import _create_public_chat_access_token

    token = _create_public_chat_access_token(
        app_config, "anonymous", groups=[group_id], link=True
    )
    from urllib.parse import quote

    return {"token": token, "path": f"/chat?pct={quote(token, safe='')}"}


@router.post("/chat-groups/reapply-domains")
async def reapply_group_domains(request: Request) -> Dict[str, Any]:
    """Re-run domain auto-assignment for every chat user (bulk)."""
    await require_permission(request, "users:write")
    store = _get_groups_store(request)
    chat_users = getattr(request.app.state, "_chat_user_store", None)
    if store is None or chat_users is None:
        raise HTTPException(status_code=503, detail="Store unavailable")
    return await store.reapply_domain_assignments(chat_users)


@router.get("/chat-groups/{group_id}")
async def get_chat_group(request: Request, group_id: str) -> Dict[str, Any]:
    """Group detail incl. member list."""
    await require_permission(request, "users:read")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    group = await store.get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")
    members = await store.members_of(group_id)
    result = _group_to_dict(group)
    result["members"] = [
        {"user_id": m.user_id, "assigned_via": m.assigned_via} for m in members
    ]
    return result


@router.put("/chat-groups/{group_id}")
async def update_chat_group(
    request: Request, group_id: str, payload: ChatGroupPayload
) -> Dict[str, Any]:
    """Update a public-chat group."""
    await require_permission(request, "users:write")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    existing = await store.get_group(group_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Group not found")
    group = _normalize_group_payload(payload)
    if not group.name:
        raise HTTPException(status_code=422, detail="Group name must not be empty")
    group.id = group_id
    if group.name != existing.name:
        dup = await store.get_group_by_name(group.name)
        if dup is not None and dup.id != group_id:
            raise HTTPException(status_code=409, detail="Group name already exists")
    await store.update_group(group)
    return _group_to_dict(await store.get_group(group_id) or group)


@router.delete("/chat-groups/{group_id}")
async def delete_chat_group(request: Request, group_id: str) -> Dict[str, Any]:
    """Delete a group (memberships cascade)."""
    await require_permission(request, "users:write")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    if not await store.delete_group(group_id):
        raise HTTPException(status_code=404, detail="Group not found")
    return {"success": True}


@router.put("/chat-groups/{group_id}/members")
async def set_chat_group_members(
    request: Request, group_id: str, payload: ChatGroupMembersPayload
) -> Dict[str, Any]:
    """Replace the member list of a group (manual assignments)."""
    await require_permission(request, "users:write")
    store = _get_groups_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat groups store unavailable")
    if await store.get_group(group_id) is None:
        raise HTTPException(status_code=404, detail="Group not found")
    chat_users = getattr(request.app.state, "_chat_user_store", None)
    if chat_users is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")
    valid_ids: List[str] = []
    for uid in payload.user_ids:
        user = await chat_users.get_user(uid) if chat_users is not None else None
        if user is not None:
            valid_ids.append(uid)
    current = await store.members_of(group_id)
    keep = set(valid_ids)
    for m in current:
        if m.user_id not in keep:
            await store.remove_user_from_group(group_id, m.user_id)
    for uid in valid_ids:
        await store.add_user_to_group(uid, group_id, assigned_via="manual")
    return {"success": True, "member_count": len(valid_ids)}
