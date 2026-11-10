"""Admin API endpoints for scheduled triggers."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import require_permission
from app.triggers_store import TriggerEntry, TriggersStore
from app.triggers_scheduler import TriggersScheduler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/triggers")


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class TriggerCreatePayload(BaseModel):
    name: str
    assistant_name: str
    prompt: str
    schedule_type: str = "interval"  # "interval" | "cron" | "message_threshold"
    schedule_config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    include_new_messages: bool = False
    include_trigger_history: bool = False
    max_history_turns: int = 10
    watch_assistants: List[str] = Field(default_factory=list)
    target_type: str = "assistant"  # "assistant" | "chain"
    target_name: str = ""
    timeout_seconds: int = Field(default=600, ge=10, le=3600)


class TriggerUpdatePayload(BaseModel):
    name: str
    assistant_name: str
    prompt: str
    schedule_type: str = "interval"
    schedule_config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    include_new_messages: bool = False
    include_trigger_history: bool = False
    max_history_turns: int = 10
    watch_assistants: List[str] = Field(default_factory=list)
    target_type: str = "assistant"
    target_name: str = ""
    timeout_seconds: int = Field(default=600, ge=10, le=3600)


class TriggerResponse(BaseModel):
    id: str
    name: str
    assistant_name: str
    prompt: str
    schedule_type: str
    schedule_config: Dict[str, Any]
    enabled: bool
    include_new_messages: bool = False
    include_trigger_history: bool = False
    max_history_turns: int = 10
    watch_assistants: List[str] = Field(default_factory=list)
    target_type: str = "assistant"
    target_name: str = ""
    timeout_seconds: int = 600
    is_running: bool = False
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TriggerRunResponse(BaseModel):
    id: int
    trigger_id: str
    started_at: str
    finished_at: Optional[str] = None
    status: str
    result_text: Optional[str] = None
    error_text: Optional[str] = None
    conversation_id: Optional[str] = None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_store(request: Request) -> Optional[TriggersStore]:
    return getattr(request.app.state, "_triggers_store", None)


def _get_scheduler(request: Request) -> Optional[TriggersScheduler]:
    return getattr(request.app.state, "_triggers_scheduler", None)


def _trigger_to_response(entry: TriggerEntry, running_ids: set = None) -> TriggerResponse:
    running = running_ids or set()
    return TriggerResponse(
        id=entry.id,
        name=entry.name,
        assistant_name=entry.assistant_name,
        prompt=entry.prompt,
        schedule_type=entry.schedule_type,
        schedule_config=entry.schedule_config,
        enabled=entry.enabled,
        include_new_messages=entry.include_new_messages,
        include_trigger_history=entry.include_trigger_history,
        max_history_turns=entry.max_history_turns,
        watch_assistants=entry.watch_assistants,
        target_type=entry.target_type,
        target_name=entry.target_name,
        timeout_seconds=entry.timeout_seconds,
        is_running=entry.id in running,
        last_run_at=entry.last_run_at.isoformat() if entry.last_run_at else None,
        next_run_at=entry.next_run_at.isoformat() if entry.next_run_at else None,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
        updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
    )


def _run_to_response(entry) -> TriggerRunResponse:
    return TriggerRunResponse(
        id=entry.id,
        trigger_id=entry.trigger_id,
        started_at=entry.started_at.isoformat() if entry.started_at else "",
        finished_at=entry.finished_at.isoformat() if entry.finished_at else None,
        status=entry.status,
        result_text=entry.result_text,
        error_text=entry.error_text,
        conversation_id=entry.conversation_id,
    )


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@router.get("", response_model=List[TriggerResponse])
async def list_triggers(request: Request):
    await require_permission(request, "triggers:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")
    scheduler = _get_scheduler(request)
    running_ids = scheduler._running if scheduler else set()
    triggers = await store.list_triggers()
    return [_trigger_to_response(t, running_ids) for t in triggers]


@router.post("", response_model=TriggerResponse)
async def create_trigger(request: Request, payload: TriggerCreatePayload):
    await require_permission(request, "triggers:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")

    # Compute next_run_at on creation
    from app.triggers_scheduler import _compute_next_run

    entry = TriggerEntry(
        id=str(uuid.uuid4()),
        name=payload.name,
        assistant_name=payload.assistant_name,
        prompt=payload.prompt,
        schedule_type=payload.schedule_type,
        schedule_config=payload.schedule_config,
        enabled=payload.enabled,
        include_new_messages=payload.include_new_messages,
        include_trigger_history=payload.include_trigger_history,
        max_history_turns=payload.max_history_turns,
        watch_assistants=payload.watch_assistants,
        target_type=payload.target_type,
        target_name=payload.target_name or payload.assistant_name,
        timeout_seconds=payload.timeout_seconds,
    )
    entry.next_run_at = _compute_next_run(entry)

    await store.create_trigger(entry)
    logger.info("Created trigger '%s' (%s)", entry.name, entry.id)
    return _trigger_to_response(entry)


@router.get("/{trigger_id}", response_model=TriggerResponse)
async def get_trigger(request: Request, trigger_id: str):
    await require_permission(request, "triggers:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")
    trigger = await store.get_trigger(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    scheduler = _get_scheduler(request)
    running_ids = scheduler._running if scheduler else set()
    return _trigger_to_response(trigger, running_ids)


@router.put("/{trigger_id}", response_model=TriggerResponse)
async def update_trigger(request: Request, trigger_id: str, payload: TriggerUpdatePayload):
    await require_permission(request, "triggers:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")

    existing = await store.get_trigger(trigger_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # If schedule changed, recompute next_run_at
    schedule_changed = (
        payload.schedule_type != existing.schedule_type
        or payload.schedule_config != existing.schedule_config
    )

    entry = TriggerEntry(
        id=trigger_id,
        name=payload.name,
        assistant_name=payload.assistant_name,
        prompt=payload.prompt,
        schedule_type=payload.schedule_type,
        schedule_config=payload.schedule_config,
        enabled=payload.enabled,
        include_new_messages=payload.include_new_messages,
        include_trigger_history=payload.include_trigger_history,
        max_history_turns=payload.max_history_turns,
        watch_assistants=payload.watch_assistants,
        target_type=payload.target_type,
        target_name=payload.target_name or payload.assistant_name,
        timeout_seconds=payload.timeout_seconds,
        last_run_at=existing.last_run_at,
        next_run_at=existing.next_run_at,
        created_at=existing.created_at,
        updated_at=existing.updated_at,
    )

    if schedule_changed:
        from app.triggers_scheduler import _compute_next_run

        entry.next_run_at = _compute_next_run(entry)

    ok = await store.update_trigger(entry)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update trigger")
    logger.info("Updated trigger '%s' (%s)", entry.name, entry.id)
    return _trigger_to_response(entry)


@router.delete("/{trigger_id}")
async def delete_trigger(request: Request, trigger_id: str):
    await require_permission(request, "triggers:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")
    ok = await store.delete_trigger(trigger_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trigger not found")
    logger.info("Deleted trigger %s", trigger_id)
    return {"success": True}


@router.post("/{trigger_id}/run")
async def run_trigger_now(request: Request, trigger_id: str):
    await require_permission(request, "triggers:run")
    scheduler = _get_scheduler(request)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Trigger scheduler not available")
    try:
        run_id = await scheduler.run_trigger_now(trigger_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"success": True, "run_id": run_id}


@router.get("/{trigger_id}/runs", response_model=List[TriggerRunResponse])
async def list_trigger_runs(request: Request, trigger_id: str, limit: int = 50, status: Optional[str] = None, sort_order: str = "desc"):
    await require_permission(request, "triggers:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Triggers store not available")
    runs = await store.list_runs(trigger_id, limit=limit, status=status, sort_order=sort_order)
    return [_run_to_response(r) for r in runs]
