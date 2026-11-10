"""Admin API endpoints for agent chains."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.auth import get_current_dashboard_user, require_permission
from app.agent_chains_store import AgentChainEntry, AgentChainsStore
from app.agent_chains_validation import validate_flow
from app.chain_graph import check_subchain_references
from app.chain_runner import ChainRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/agent-chains")


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------


class AgentChainCreatePayload(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    api_accessible: bool = True
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)


class AgentChainUpdatePayload(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    api_accessible: bool = True
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)


class AgentChainResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    api_accessible: bool = True
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ValidatePayload(BaseModel):
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _get_store(request: Request) -> Optional[AgentChainsStore]:
    store = getattr(request.app.state, "_agent_chains_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None)
    if engine is None:
        return None
    store = AgentChainsStore(engine)
    request.app.state._agent_chains_store = store
    return store


async def _check_chain_refs(
    store: AgentChainsStore,
    flow_definition: Dict[str, Any],
    self_id: Optional[str] = None,
    self_name: Optional[str] = None,
) -> List[str]:
    """Async sub-chain reference check: existence, enabled, no cycles."""

    async def _resolve(ref: str):
        return await store.get_chain(ref) or await store.get_chain_by_name(ref)

    return await check_subchain_references(
        flow_definition,
        self_id=self_id,
        self_name=self_name,
        resolve=_resolve,
    )


def _entry_to_response(entry: AgentChainEntry) -> AgentChainResponse:
    return AgentChainResponse(
        id=entry.id,
        name=entry.name,
        description=entry.description,
        enabled=entry.enabled,
        api_accessible=entry.api_accessible,
        flow_definition=entry.flow_definition,
        execution_options=entry.execution_options,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
        updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
    )


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------


@router.get("", response_model=List[AgentChainResponse])
async def list_agent_chains(request: Request):
    await require_permission(request, "agent_chains:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    chains = await store.list_chains()
    return [_entry_to_response(c) for c in chains]


@router.post("", response_model=AgentChainResponse)
async def create_agent_chain(request: Request, payload: AgentChainCreatePayload):
    await require_permission(request, "agent_chains:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")

    # Validate flow before creation
    validation = validate_flow(payload.flow_definition)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow: {'; '.join(validation['errors'])}"
        )
    ref_errors = await _check_chain_refs(store, payload.flow_definition, self_name=payload.name)
    if ref_errors:
        raise HTTPException(status_code=400, detail=f"Invalid flow: {'; '.join(ref_errors)}")

    entry = AgentChainEntry(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        api_accessible=payload.api_accessible,
        flow_definition=payload.flow_definition,
        execution_options=payload.execution_options,
    )
    await store.create_chain(entry)
    logger.info("Created agent chain '%s' (%s)", entry.name, entry.id)
    return _entry_to_response(entry)


@router.get("/{chain_id}", response_model=AgentChainResponse)
async def get_agent_chain(request: Request, chain_id: str):
    await require_permission(request, "agent_chains:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    chain = await store.get_chain(chain_id)
    if chain is None:
        raise HTTPException(status_code=404, detail="Agent chain not found")
    return _entry_to_response(chain)


@router.put("/{chain_id}", response_model=AgentChainResponse)
async def update_agent_chain(
    request: Request, chain_id: str, payload: AgentChainUpdatePayload
):
    await require_permission(request, "agent_chains:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")

    existing = await store.get_chain(chain_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent chain not found")

    # Validate flow before update
    validation = validate_flow(payload.flow_definition)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow: {'; '.join(validation['errors'])}"
        )
    ref_errors = await _check_chain_refs(
        store, payload.flow_definition, self_id=chain_id, self_name=payload.name
    )
    if ref_errors:
        raise HTTPException(status_code=400, detail=f"Invalid flow: {'; '.join(ref_errors)}")

    entry = AgentChainEntry(
        id=chain_id,
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        api_accessible=payload.api_accessible,
        flow_definition=payload.flow_definition,
        execution_options=payload.execution_options,
        created_at=existing.created_at,
        updated_at=existing.updated_at,
    )
    ok = await store.update_chain(entry)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update agent chain")
    logger.info("Updated agent chain '%s' (%s)", entry.name, entry.id)
    return _entry_to_response(entry)


@router.delete("/{chain_id}")
async def delete_agent_chain(request: Request, chain_id: str):
    await require_permission(request, "agent_chains:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    ok = await store.delete_chain(chain_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent chain not found")
    logger.info("Deleted agent chain %s", chain_id)
    return {"success": True}


@router.post("/validate", response_model=ValidateResponse)
async def validate_agent_chain(request: Request, payload: ValidatePayload):
    await require_permission(request, "agent_chains:read")
    validation = validate_flow(payload.flow_definition)
    errors = list(validation["errors"])
    if validation["valid"]:
        store = _get_store(request)
        if store is not None:
            errors.extend(await _check_chain_refs(store, payload.flow_definition))
    return ValidateResponse(
        valid=not errors,
        errors=errors,
    )


class TestRunPayload(BaseModel):
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    input: str = ""
    execution_options: Dict[str, Any] = Field(default_factory=dict)


@router.post("/test-run")
async def test_run_agent_chain(request: Request, payload: TestRunPayload):
    """Execute an ad-hoc flow definition (unsaved drafts allowed) and return
    the final answer plus per-node outputs. Requires write permission since
    running a chain performs real LLM calls."""
    await require_permission(request, "agent_chains:write")

    validation = validate_flow(payload.flow_definition)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow: {'; '.join(validation['errors'])}",
        )

    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG application not available")

    user = await get_current_dashboard_user(request) or {}

    try:
        runner = ChainRunner(
            payload.flow_definition,
            rag_app,
            execution_options=payload.execution_options,
            run_context={"chain_id": "", "chain_name": "(test)", "source": "test"},
        )
        result = await runner.run(
            input_text=payload.input,
            chat_user_id=user.get("id") or None,
        )
    except Exception as exc:
        logger.exception("Chain test-run failed")
        raise HTTPException(status_code=500, detail=f"Test run failed: {exc}")

    return {
        "answer": result.get("answer", ""),
        "reasoning": result.get("reasoning", ""),
        "node_outputs": result.get("node_outputs", {}),
        "node_status": result.get("node_status", {}),
        "tool_calls": result.get("tool_calls", []),
    }


@router.post("/test-run/stream")
async def test_run_agent_chain_stream(request: Request, payload: TestRunPayload):
    """Same as /test-run but streams per-node lifecycle events over SSE.

    Event sequence: node_start / node_end per executed node (with status +
    duration), then a final run_end event carrying the full result payload.
    """
    await require_permission(request, "agent_chains:write")

    validation = validate_flow(payload.flow_definition)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow: {'; '.join(validation['errors'])}",
        )

    rag_app = getattr(request.app.state, "rag_application", None)
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG application not available")

    user = await get_current_dashboard_user(request) or {}
    queue: asyncio.Queue = asyncio.Queue()

    async def _on_event(event: Dict[str, Any]) -> None:
        await queue.put(event)

    runner = ChainRunner(
        payload.flow_definition,
        rag_app,
        execution_options=payload.execution_options,
        on_event=_on_event,
        run_context={"chain_id": "", "chain_name": "(test)", "source": "test"},
    )

    async def _execute() -> None:
        try:
            result = await runner.run(
                input_text=payload.input,
                chat_user_id=user.get("id") or None,
            )
            await queue.put({"type": "run_result", "result": result})
        except Exception as exc:
            logger.exception("Chain test-run stream failed")
            await queue.put({"type": "run_error", "error": str(exc)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(_execute())

    async def _stream():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(_stream(), media_type="text/event-stream")


class ChainRunResponse(BaseModel):
    id: str
    chain_id: str
    chain_name: str = ""
    source: str = "run"
    conversation_id: Optional[str] = None
    started_at: Optional[str] = None
    duration_ms: int = 0
    status: str = "done"
    input: str = ""
    answer: str = ""
    node_status: Dict[str, Any] = Field(default_factory=dict)
    node_outputs: Dict[str, Any] = Field(default_factory=dict)


def _run_to_response(run) -> ChainRunResponse:
    return ChainRunResponse(
        id=run.id,
        chain_id=run.chain_id,
        chain_name=run.chain_name,
        source=run.source,
        conversation_id=run.conversation_id,
        started_at=run.started_at.isoformat() if run.started_at else None,
        duration_ms=run.duration_ms,
        status=run.status,
        input=run.input_text,
        answer=run.answer,
        node_status=run.node_status or {},
        node_outputs=run.node_outputs or {},
    )


@router.get("/{chain_id}/runs", response_model=List[ChainRunResponse])
async def list_chain_runs(request: Request, chain_id: str, limit: int = 50):
    """Run history for one chain (newest first)."""
    await require_permission(request, "agent_chains:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    runs = await store.list_runs(chain_id, limit=limit)
    return [_run_to_response(r) for r in runs]


@router.get("/runs/{run_id}", response_model=ChainRunResponse)
async def get_chain_run(request: Request, run_id: str):
    """Single recorded run with node statuses + outputs (replay payload)."""
    await require_permission(request, "agent_chains:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(run)


@router.get("/{chain_id}/export")
async def export_agent_chain(request: Request, chain_id: str):
    """Export a chain as JSON (without internal id and timestamps)."""
    await require_permission(request, "agent_chains:read")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    entry = await store.get_chain(chain_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Agent chain not found")
    payload = {
        "name": entry.name,
        "description": entry.description,
        "enabled": entry.enabled,
        "api_accessible": entry.api_accessible,
        "flow_definition": entry.flow_definition,
        "execution_options": entry.execution_options,
    }
    filename = f"chain_{entry.name.replace(' ', '_')}.json"
    return PlainTextResponse(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class ImportPayload(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    api_accessible: bool = True
    flow_definition: Dict[str, Any] = Field(default_factory=dict)
    execution_options: Dict[str, Any] = Field(default_factory=dict)


@router.post("/import", response_model=AgentChainResponse)
async def import_agent_chain(request: Request, payload: ImportPayload):
    """Import a chain from JSON export."""
    await require_permission(request, "agent_chains:write")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Agent chains store not available")
    validation = validate_flow(payload.flow_definition)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow: {'; '.join(validation['errors'])}"
        )
    ref_errors = await _check_chain_refs(store, payload.flow_definition, self_name=payload.name)
    if ref_errors:
        raise HTTPException(status_code=400, detail=f"Invalid flow: {'; '.join(ref_errors)}")
    entry = AgentChainEntry(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        api_accessible=payload.api_accessible,
        flow_definition=payload.flow_definition,
        execution_options=payload.execution_options,
    )
    await store.create_chain(entry)
    logger.info("Imported agent chain '%s' (%s)", entry.name, entry.id)
    return _entry_to_response(entry)
