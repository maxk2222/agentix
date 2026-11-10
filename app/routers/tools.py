"""Admin API endpoints for agent tools management."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import require_permission
from app.tools_store import ToolEntry, ToolsStore
from app import tool_config_schemas
from agent.tools.builtin_tools.assistant_caller import ASSISTANT_TOOL_PREFIX
from agent.tools.venv_executor import VenvExecutor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/tools")
_RESERVED_BUILTIN_NAMES = {
    "agent_todo",
    "agent_workspace",
    "spawn_subagent",
    "wait_subagent",
    "subagent_status",
    "wait",
    "mcp_servers",
}


# ------------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------------

class ToolCreatePayload(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    kind: str = "custom"
    tool_schema: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    code: Optional[str] = None
    venv_config: Dict[str, Any] = Field(default_factory=dict)
    runtime_mode: str = "venv"


class ToolUpdatePayload(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    kind: str = "custom"
    tool_schema: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    code: Optional[str] = None
    venv_config: Dict[str, Any] = Field(default_factory=dict)
    runtime_mode: str = "venv"


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    kind: str
    tool_schema: Dict[str, Any]
    config: Dict[str, Any]
    config_schema: Optional[List[Dict[str, Any]]] = None
    code: Optional[str] = None
    venv_config: Dict[str, Any] = Field(default_factory=dict)
    runtime_mode: str = "venv"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_tools_store(request: Request) -> Optional[ToolsStore]:
    return getattr(request.app.state, "_tools_store", None)


def _get_rag_app(request: Request) -> Any:
    return getattr(request.app.state, "rag_application", None)


async def _reload_tools(request: Request, touched_names: tuple = ()) -> None:
    """Reload tool registrations after store mutations so changes apply
    immediately (no restart). When ``mcp_servers`` was touched, reconcile the
    MCP client first so freshly materialized tools get registered in the same
    pass."""
    manager = getattr(request.app.state, "mcp_manager", None)
    if manager is not None and "mcp_servers" in touched_names:
        try:
            await manager.sync()
        except Exception as exc:
            logger.warning("MCP server sync failed: %s", exc)
    store = _get_tools_store(request)
    rag_app = _get_rag_app(request)
    registry = getattr(rag_app, "_tool_registry", None) if rag_app is not None else None
    if registry is not None and store is not None:
        try:
            await registry.load_custom_tools(store)
        except Exception as exc:
            logger.warning("Tool registry reload failed: %s", exc)


def _tool_to_response(tool: ToolEntry) -> ToolResponse:
    schema = tool_config_schemas.schema_for(tool.name)
    return ToolResponse(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        enabled=tool.enabled,
        kind=tool.kind,
        tool_schema=tool.tool_schema,
        # Secret fields are masked so a tools:read role never receives
        # stored credentials in cleartext.
        config=tool_config_schemas.mask_config(tool.name, tool.config),
        config_schema=schema or None,
        code=tool.code,
        venv_config=tool.venv_config or {},
        runtime_mode=tool.runtime_mode,
        created_at=tool.created_at.isoformat() if tool.created_at else None,
        updated_at=tool.updated_at.isoformat() if tool.updated_at else None,
    )


def _virtual_assistant_tools(rag_app: Any) -> List[ToolResponse]:
    """Return synthetic ToolResponse entries for assistant-derived tools."""
    responses: List[ToolResponse] = []
    if rag_app is None:
        logger.debug("_virtual_assistant_tools: rag_app is None")
        return responses
    registry = getattr(rag_app, "_tool_registry", None)
    if registry is None:
        logger.debug("_virtual_assistant_tools: registry is None")
        return responses
    instructions = getattr(rag_app, "instructions", {}) or {}
    logger.debug("_virtual_assistant_tools: instructions keys=%s", list(instructions.keys()))
    for tool in getattr(registry, "_assistant", {}).values():
        schema = getattr(tool, "tool_schema", {}) or {}
        func = schema.get("function", {})
        assistant_name = getattr(tool, "assistant_name", "")
        assistant_entry = instructions.get(assistant_name, {})
        agent_enabled = bool(assistant_entry.get("agent_enabled", False))
        logger.debug(
            "_virtual_assistant_tools: assistant_name=%s agent_enabled=%s entry_keys=%s",
            assistant_name,
            agent_enabled,
            list(assistant_entry.keys()) if isinstance(assistant_entry, dict) else "n/a",
        )
        responses.append(
            ToolResponse(
                id=f"virtual-{tool.name}",
                name=tool.name,
                description=func.get("description", ""),
                enabled=True,
                kind="assistant",
                tool_schema=schema,
                config={"agent_enabled": agent_enabled},
                code=None,
                venv_config={},
                runtime_mode="venv",
                created_at=None,
                updated_at=None,
            )
        )
    return responses


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@router.get("", response_model=List[ToolResponse])
async def list_tools(request: Request, enabled_only: bool = False):
    await require_permission(request, "tools:read")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    tools = await store.list_tools(enabled_only=enabled_only)
    responses = [_tool_to_response(t) for t in tools]
    # Append virtual assistant tools
    responses.extend(_virtual_assistant_tools(_get_rag_app(request)))
    return responses


@router.post("", response_model=ToolResponse)
async def create_tool(request: Request, payload: ToolCreatePayload):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")

    if payload.kind == "builtin":
        raise HTTPException(status_code=400, detail="Cannot create new builtin tools")

    if payload.name.startswith(ASSISTANT_TOOL_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"Tool names starting with '{ASSISTANT_TOOL_PREFIX}' are reserved for assistant tools",
        )
    if payload.name in _RESERVED_BUILTIN_NAMES:
        raise HTTPException(status_code=400, detail=f"Tool name '{payload.name}' is reserved")

    existing = await store.get_tool_by_name(payload.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Tool '{payload.name}' already exists")

    tool = ToolEntry(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        kind=payload.kind,
        tool_schema=payload.tool_schema,
        config=payload.config,
        code=payload.code,
        venv_config=payload.venv_config,
        runtime_mode=payload.runtime_mode,
    )
    await store.create_tool(tool)
    logger.info("Created tool '%s' (%s)", tool.name, tool.id)
    await _reload_tools(request, (payload.name,))
    return _tool_to_response(tool)


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(request: Request, tool_id: str):
    await require_permission(request, "tools:read")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    tool = await store.get_tool(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _tool_to_response(tool)


@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(request: Request, tool_id: str, payload: ToolUpdatePayload):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")

    existing = await store.get_tool(tool_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Tool not found")

    if payload.name in _RESERVED_BUILTIN_NAMES and not (
        existing.kind == "builtin" and payload.name == existing.name
    ):
        raise HTTPException(status_code=400, detail=f"Tool name '{payload.name}' is reserved")

    # Check name collision with other tools
    if payload.name != existing.name:
        other = await store.get_tool_by_name(payload.name)
        if other is not None and other.id != tool_id:
            raise HTTPException(status_code=409, detail=f"Tool '{payload.name}' already exists")

    errors = tool_config_schemas.validate_schema_config(payload.name, payload.config)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    config = tool_config_schemas.merge_secret_preserves(
        payload.name, payload.config, existing.config
    )

    tool = ToolEntry(
        id=tool_id,
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        kind=payload.kind,
        tool_schema=payload.tool_schema,
        config=config,
        code=payload.code,
        venv_config=payload.venv_config,
        runtime_mode=payload.runtime_mode,
        created_at=existing.created_at,
    )
    updated = await store.update_tool(tool)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update tool")
    logger.info("Updated tool '%s' (%s)", tool.name, tool.id)
    await _reload_tools(request, (existing.name, payload.name))
    return _tool_to_response(tool)


@router.post("/{tool_id}/venv/rebuild")
async def rebuild_venv(request: Request, tool_id: str):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    tool = await store.get_tool(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    if tool.kind != "custom" or not tool.venv_config:
        raise HTTPException(status_code=400, detail="Tool has no venv configuration")
    try:
        executor = VenvExecutor(tool_id, tool.venv_config)
        await executor.rebuild()
        return {"success": True, "venv_path": str(executor._venv_path)}
    except Exception as exc:
        logger.exception("Venv rebuild failed for tool %s", tool_id)
        raise HTTPException(status_code=500, detail=f"Venv rebuild failed: {exc}") from exc


@router.delete("/{tool_id}")
async def delete_tool(request: Request, tool_id: str):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    existing = await store.get_tool(tool_id)
    if existing is not None:
        if existing.name in _RESERVED_BUILTIN_NAMES:
            raise HTTPException(status_code=400, detail=f"Tool '{existing.name}' is reserved")
        if existing.kind == "mcp" and existing.enabled:
            # Enabled MCP entries are re-materialized by the next server sync —
            # deleting looks successful, then the tool comes back. Disabled
            # (orphaned) entries may be deleted for real.
            raise HTTPException(
                status_code=400,
                detail="Managed MCP tool — disable it instead of deleting "
                       "(it re-appears as long as the server advertises it).",
            )
    deleted = await store.delete_tool(tool_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")
    logger.info("Deleted tool %s", tool_id)
    await _reload_tools(request, ((existing.name,) if existing else ()))
    return {"success": True}


@router.post("/{tool_id}/enable")
async def enable_tool(request: Request, tool_id: str):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    entry = await store.get_tool(tool_id)
    updated = await store.set_enabled(tool_id, True)
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    await _reload_tools(request, ((entry.name,) if entry else ()))
    return {"success": True}


@router.post("/{tool_id}/disable")
async def disable_tool(request: Request, tool_id: str):
    await require_permission(request, "tools:write")
    store = _get_tools_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Tools store not available")
    entry = await store.get_tool(tool_id)
    updated = await store.set_enabled(tool_id, False)
    if not updated:
        raise HTTPException(status_code=404, detail="Tool not found")
    await _reload_tools(request, ((entry.name,) if entry else ()))
    return {"success": True}
