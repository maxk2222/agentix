"""Postgres-backed store for agent chains."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

AGENT_CHAINS_TABLE = "agentix_agent_chains"
AGENT_RUNS_TABLE = "agentix_agent_chain_runs"
_MAX_TEXT_LEN = 65536


@dataclass
class AgentChainEntry:
    """Represents a configured agent chain with its flow definition."""

    id: str
    name: str
    description: str = ""
    enabled: bool = True
    api_accessible: bool = True
    flow_definition: Dict[str, Any] = field(default_factory=dict)
    execution_options: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class AgentChainRunEntry:
    """One recorded chain execution (test run, chat run, trigger run, …)."""

    id: str
    chain_id: str
    chain_name: str = ""
    source: str = "run"
    conversation_id: Optional[str] = None
    started_at: Optional[datetime] = None
    duration_ms: int = 0
    status: str = "done"
    input_text: str = ""
    answer: str = ""
    node_status: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)


class AgentChainsStore:
    """Async store for agent chain definitions."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {AGENT_CHAINS_TABLE} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL DEFAULT '',
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        api_accessible BOOLEAN NOT NULL DEFAULT true,
                        flow_definition JSONB NOT NULL DEFAULT '{{}}',
                        execution_options JSONB NOT NULL DEFAULT '{{}}',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_agent_chains_enabled
                    ON {AGENT_CHAINS_TABLE} (enabled)
                    WHERE enabled = true
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {AGENT_RUNS_TABLE} (
                        id TEXT PRIMARY KEY,
                        chain_id TEXT NOT NULL DEFAULT '',
                        chain_name TEXT NOT NULL DEFAULT '',
                        source TEXT NOT NULL DEFAULT 'run',
                        conversation_id TEXT,
                        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        duration_ms INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'done',
                        input TEXT NOT NULL DEFAULT '',
                        answer TEXT NOT NULL DEFAULT '',
                        node_status JSONB NOT NULL DEFAULT '{{}}',
                        node_outputs JSONB NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_agent_chain_runs_chain
                    ON {AGENT_RUNS_TABLE} (chain_id, started_at DESC)
                    """
                )
            )
        self._tables_ensured = True

    async def list_chains(self) -> List[AgentChainEntry]:
        await self.ensure_tables()
        query = f"""
            SELECT id, name, description, enabled, api_accessible,
                   flow_definition, execution_options, created_at, updated_at
            FROM {AGENT_CHAINS_TABLE}
            ORDER BY name
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            rows = result.mappings().all()
            return [_row_to_entry(row) for row in rows]

    async def get_chain(self, chain_id: str) -> Optional[AgentChainEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, enabled, api_accessible,
                           flow_definition, execution_options, created_at, updated_at
                    FROM {AGENT_CHAINS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": chain_id},
            )
            row = result.mappings().first()
            return _row_to_entry(row) if row else None

    async def get_chain_by_name(self, name: str) -> Optional[AgentChainEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, enabled, api_accessible,
                           flow_definition, execution_options, created_at, updated_at
                    FROM {AGENT_CHAINS_TABLE}
                    WHERE name = :name
                    """
                ),
                {"name": name},
            )
            row = result.mappings().first()
            return _row_to_entry(row) if row else None

    async def create_chain(self, entry: AgentChainEntry) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {AGENT_CHAINS_TABLE}
                        (id, name, description, enabled, api_accessible,
                         flow_definition, execution_options, created_at, updated_at)
                    VALUES (:id, :name, :description, :enabled, :api_accessible,
                            CAST(:flow_definition AS jsonb), CAST(:execution_options AS jsonb),
                            NOW(), NOW())
                    """
                ),
                {
                    "id": entry.id,
                    "name": entry.name,
                    "description": entry.description,
                    "enabled": entry.enabled,
                    "api_accessible": entry.api_accessible,
                    "flow_definition": _json_dumps(entry.flow_definition),
                    "execution_options": _json_dumps(entry.execution_options),
                },
            )

    async def update_chain(self, entry: AgentChainEntry) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE {AGENT_CHAINS_TABLE}
                    SET name = :name,
                        description = :description,
                        enabled = :enabled,
                        api_accessible = :api_accessible,
                        flow_definition = CAST(:flow_definition AS jsonb),
                        execution_options = CAST(:execution_options AS jsonb),
                        updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": entry.id,
                    "name": entry.name,
                    "description": entry.description,
                    "enabled": entry.enabled,
                    "api_accessible": entry.api_accessible,
                    "flow_definition": _json_dumps(entry.flow_definition),
                    "execution_options": _json_dumps(entry.execution_options),
                },
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def delete_chain(self, chain_id: str) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f"DELETE FROM {AGENT_CHAINS_TABLE} WHERE id = :id"),
                {"id": chain_id},
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    async def record_run(
        self,
        *,
        chain_id: str,
        chain_name: str,
        source: str,
        conversation_id: Optional[str],
        duration_ms: int,
        status: str,
        input_text: str,
        answer: str,
        node_status: Dict[str, Any],
        node_outputs: Dict[str, Any],
    ) -> Optional[str]:
        """Persist one execution; returns the run id (best-effort)."""
        await self.ensure_tables()
        run_id = uuid.uuid4().hex
        truncated_outputs = {
            str(k): str(v)[:_MAX_TEXT_LEN] for k, v in (node_outputs or {}).items()
        }
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {AGENT_RUNS_TABLE}
                        (id, chain_id, chain_name, source, conversation_id,
                         duration_ms, status, input, answer, node_status, node_outputs)
                    VALUES (:id, :chain_id, :chain_name, :source, :conversation_id,
                            :duration_ms, :status, :input, :answer,
                            CAST(:node_status AS jsonb), CAST(:node_outputs AS jsonb))
                    """
                ),
                {
                    "id": run_id,
                    "chain_id": chain_id,
                    "chain_name": chain_name,
                    "source": source,
                    "conversation_id": conversation_id,
                    "duration_ms": int(duration_ms),
                    "status": status,
                    "input": (input_text or "")[:_MAX_TEXT_LEN],
                    "answer": (answer or "")[:_MAX_TEXT_LEN],
                    "node_status": _json_dumps(node_status or {}),
                    "node_outputs": _json_dumps(truncated_outputs),
                },
            )
        return run_id

    async def list_runs(
        self,
        chain_id: Optional[str] = None,
        *,
        limit: int = 50,
    ) -> List[AgentChainRunEntry]:
        await self.ensure_tables()
        limit = max(1, min(int(limit), 200))
        if chain_id:
            query = f"""
                SELECT id, chain_id, chain_name, source, conversation_id,
                       started_at, duration_ms, status, input, answer,
                       node_status, node_outputs
                FROM {AGENT_RUNS_TABLE}
                WHERE chain_id = :chain_id
                ORDER BY started_at DESC
                LIMIT :lim
            """
            params = {"chain_id": chain_id, "lim": limit}
        else:
            query = f"""
                SELECT id, chain_id, chain_name, source, conversation_id,
                       started_at, duration_ms, status, input, answer,
                       node_status, node_outputs
                FROM {AGENT_RUNS_TABLE}
                ORDER BY started_at DESC
                LIMIT :lim
            """
            params = {"lim": limit}
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query), params)
            return [_row_to_run(row) for row in result.mappings().all()]

    async def get_run(self, run_id: str) -> Optional[AgentChainRunEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, chain_id, chain_name, source, conversation_id,
                           started_at, duration_ms, status, input, answer,
                           node_status, node_outputs
                    FROM {AGENT_RUNS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": run_id},
            )
            row = result.mappings().first()
            return _row_to_run(row) if row else None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {}


def _row_to_entry(row) -> AgentChainEntry:
    return AgentChainEntry(
        id=row["id"],
        name=row["name"],
        description=row.get("description") or "",
        enabled=bool(row["enabled"]),
        api_accessible=bool(row.get("api_accessible", True)),
        flow_definition=_json_loads(row["flow_definition"]),
        execution_options=_json_loads(row["execution_options"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_run(row) -> AgentChainRunEntry:
    return AgentChainRunEntry(
        id=row["id"],
        chain_id=row["chain_id"],
        chain_name=row.get("chain_name") or "",
        source=row.get("source") or "run",
        conversation_id=row.get("conversation_id"),
        started_at=row["started_at"],
        duration_ms=int(row.get("duration_ms") or 0),
        status=row.get("status") or "done",
        input_text=row.get("input") or "",
        answer=row.get("answer") or "",
        node_status=_json_loads(row.get("node_status")),
        node_outputs=_json_loads(row.get("node_outputs")),
    )
