"""Postgres-backed store for scheduled assistant triggers."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

TRIGGERS_TABLE = "agentix_triggers"
TRIGGER_RUNS_TABLE = "agentix_trigger_runs"


@dataclass
class TriggerEntry:
    id: str
    name: str
    assistant_name: str
    prompt: str
    schedule_type: str  # "interval" | "cron" | "message_threshold"
    schedule_config: Dict[str, Any]
    enabled: bool = True
    include_new_messages: bool = False
    include_trigger_history: bool = False
    max_history_turns: int = 10
    watch_assistants: List[str] = field(default_factory=list)
    target_type: str = "assistant"  # "assistant" | "chain"
    target_name: str = ""
    timeout_seconds: int = 600
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TriggerRunEntry:
    id: int
    trigger_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str = "pending"  # pending | success | error
    result_text: Optional[str] = None
    error_text: Optional[str] = None
    conversation_id: Optional[str] = None


class TriggersStore:
    """Async store for triggers and their execution history."""

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
                    CREATE TABLE IF NOT EXISTS {TRIGGERS_TABLE} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        assistant_name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        schedule_type TEXT NOT NULL,
                        schedule_config JSONB NOT NULL DEFAULT '{{}}',
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        include_new_messages BOOLEAN NOT NULL DEFAULT false,
                        include_trigger_history BOOLEAN NOT NULL DEFAULT false,
                        max_history_turns INTEGER NOT NULL DEFAULT 10,
                        watch_assistants JSONB NOT NULL DEFAULT '[]',
                        target_type TEXT NOT NULL DEFAULT 'assistant',
                        target_name TEXT NOT NULL DEFAULT '',
                        timeout_seconds INTEGER NOT NULL DEFAULT 600,
                        last_run_at TIMESTAMPTZ,
                        next_run_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_triggers_enabled_next_run
                    ON {TRIGGERS_TABLE} (enabled, next_run_at)
                    WHERE enabled = true
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {TRIGGERS_TABLE}
                    ADD COLUMN IF NOT EXISTS include_new_messages BOOLEAN NOT NULL DEFAULT false,
                    ADD COLUMN IF NOT EXISTS include_trigger_history BOOLEAN NOT NULL DEFAULT false,
                    ADD COLUMN IF NOT EXISTS max_history_turns INTEGER NOT NULL DEFAULT 10,
                    ADD COLUMN IF NOT EXISTS watch_assistants JSONB NOT NULL DEFAULT '[]',
                    ADD COLUMN IF NOT EXISTS target_type TEXT NOT NULL DEFAULT 'assistant',
                    ADD COLUMN IF NOT EXISTS target_name TEXT NOT NULL DEFAULT '',
                    ADD COLUMN IF NOT EXISTS timeout_seconds INTEGER NOT NULL DEFAULT 600
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TRIGGER_RUNS_TABLE} (
                        id SERIAL PRIMARY KEY,
                        trigger_id TEXT NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        finished_at TIMESTAMPTZ,
                        status TEXT NOT NULL DEFAULT 'pending',
                        result_text TEXT,
                        error_text TEXT,
                        conversation_id TEXT
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_trigger_runs_trigger_id
                    ON {TRIGGER_RUNS_TABLE} (trigger_id)
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_trigger_runs_started_at
                    ON {TRIGGER_RUNS_TABLE} (started_at DESC)
                    """
                )
            )
        self._tables_ensured = True

    async def list_triggers(self) -> List[TriggerEntry]:
        await self.ensure_tables()
        query = f"""
            SELECT id, name, assistant_name, prompt, schedule_type, schedule_config,
                   enabled, include_new_messages, include_trigger_history, max_history_turns, watch_assistants,
                   target_type, target_name, timeout_seconds,
                   last_run_at, next_run_at, created_at, updated_at
            FROM {TRIGGERS_TABLE}
            ORDER BY name
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            rows = result.mappings().all()
            return [_row_to_trigger(row) for row in rows]

    async def get_trigger(self, trigger_id: str) -> Optional[TriggerEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, assistant_name, prompt, schedule_type, schedule_config,
                           enabled, include_new_messages, include_trigger_history, max_history_turns, watch_assistants,
                           target_type, target_name, timeout_seconds,
                           last_run_at, next_run_at, created_at, updated_at
                    FROM {TRIGGERS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": trigger_id},
            )
            row = result.mappings().first()
            return _row_to_trigger(row) if row else None

    async def create_trigger(self, entry: TriggerEntry) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {TRIGGERS_TABLE}
                        (id, name, assistant_name, prompt, schedule_type, schedule_config,
                         enabled, include_new_messages, include_trigger_history, max_history_turns, watch_assistants,
                         target_type, target_name, timeout_seconds, last_run_at, next_run_at, created_at, updated_at)
                    VALUES (:id, :name, :assistant_name, :prompt, :schedule_type,
                            CAST(:schedule_config AS jsonb), :enabled, :include_new_messages,
                            :include_trigger_history, :max_history_turns,
                            CAST(:watch_assistants AS jsonb),
                            :target_type, :target_name, :timeout_seconds, :last_run_at,
                            :next_run_at, NOW(), NOW())
                    """
                ),
                {
                    "id": entry.id,
                    "name": entry.name,
                    "assistant_name": entry.assistant_name,
                    "prompt": entry.prompt,
                    "schedule_type": entry.schedule_type,
                    "schedule_config": _json_dumps(entry.schedule_config),
                    "enabled": entry.enabled,
                    "include_new_messages": entry.include_new_messages,
                    "include_trigger_history": entry.include_trigger_history,
                    "max_history_turns": entry.max_history_turns,
                    "watch_assistants": _json_dumps(entry.watch_assistants),
                    "target_type": entry.target_type,
                    "target_name": entry.target_name or entry.assistant_name,
                    "timeout_seconds": entry.timeout_seconds,
                    "last_run_at": entry.last_run_at,
                    "next_run_at": entry.next_run_at,
                },
            )

    async def update_trigger(self, entry: TriggerEntry) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE {TRIGGERS_TABLE}
                    SET name = :name,
                        assistant_name = :assistant_name,
                        prompt = :prompt,
                        schedule_type = :schedule_type,
                        schedule_config = CAST(:schedule_config AS jsonb),
                        enabled = :enabled,
                        include_new_messages = :include_new_messages,
                        include_trigger_history = :include_trigger_history,
                        max_history_turns = :max_history_turns,
                        watch_assistants = CAST(:watch_assistants AS jsonb),
                        target_type = :target_type,
                        target_name = :target_name,
                        timeout_seconds = :timeout_seconds,
                        last_run_at = :last_run_at,
                        next_run_at = :next_run_at,
                        updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": entry.id,
                    "name": entry.name,
                    "assistant_name": entry.assistant_name,
                    "prompt": entry.prompt,
                    "schedule_type": entry.schedule_type,
                    "schedule_config": _json_dumps(entry.schedule_config),
                    "enabled": entry.enabled,
                    "include_new_messages": entry.include_new_messages,
                    "include_trigger_history": entry.include_trigger_history,
                    "max_history_turns": entry.max_history_turns,
                    "watch_assistants": _json_dumps(entry.watch_assistants),
                    "target_type": entry.target_type,
                    "target_name": entry.target_name or entry.assistant_name,
                    "timeout_seconds": entry.timeout_seconds,
                    "last_run_at": entry.last_run_at,
                    "next_run_at": entry.next_run_at,
                },
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def delete_trigger(self, trigger_id: str) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            # Delete runs first
            await conn.execute(
                text(f"DELETE FROM {TRIGGER_RUNS_TABLE} WHERE trigger_id = :id"),
                {"id": trigger_id},
            )
            result = await conn.execute(
                text(f"DELETE FROM {TRIGGERS_TABLE} WHERE id = :id"),
                {"id": trigger_id},
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def update_next_run(self, trigger_id: str, next_run_at: Optional[datetime]) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {TRIGGERS_TABLE}
                    SET next_run_at = :next_run_at, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": trigger_id, "next_run_at": next_run_at},
            )

    async def update_last_run(self, trigger_id: str, last_run_at: datetime) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {TRIGGERS_TABLE}
                    SET last_run_at = :last_run_at, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": trigger_id, "last_run_at": last_run_at},
            )

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def create_run(
        self,
        trigger_id: str,
        status: str = "pending",
        conversation_id: Optional[str] = None,
    ) -> int:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    INSERT INTO {TRIGGER_RUNS_TABLE}
                        (trigger_id, started_at, status, conversation_id)
                    VALUES (:trigger_id, NOW(), :status, :conversation_id)
                    RETURNING id
                    """
                ),
                {
                    "trigger_id": trigger_id,
                    "status": status,
                    "conversation_id": conversation_id,
                },
            )
            return int(result.scalar())

    async def finish_run(
        self,
        run_id: int,
        status: str,
        result_text: Optional[str] = None,
        error_text: Optional[str] = None,
    ) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {TRIGGER_RUNS_TABLE}
                    SET finished_at = NOW(),
                        status = :status,
                        result_text = :result_text,
                        error_text = :error_text
                    WHERE id = :id
                    """
                ),
                {
                    "id": run_id,
                    "status": status,
                    "result_text": result_text,
                    "error_text": error_text,
                },
            )

    async def list_runs(
        self, trigger_id: str, limit: int = 50, status: Optional[str] = None, sort_order: str = "desc"
    ) -> List[TriggerRunEntry]:
        await self.ensure_tables()
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        async with self._engine.connect() as conn:
            if status:
                result = await conn.execute(
                    text(
                        f"""
                        SELECT id, trigger_id, started_at, finished_at,
                               status, result_text, error_text, conversation_id
                        FROM {TRIGGER_RUNS_TABLE}
                        WHERE trigger_id = :trigger_id AND status = :status
                        ORDER BY started_at {order}
                        LIMIT :limit
                        """
                    ),
                    {"trigger_id": trigger_id, "status": status, "limit": limit},
                )
            else:
                result = await conn.execute(
                    text(
                        f"""
                        SELECT id, trigger_id, started_at, finished_at,
                               status, result_text, error_text, conversation_id
                        FROM {TRIGGER_RUNS_TABLE}
                        WHERE trigger_id = :trigger_id
                        ORDER BY started_at {order}
                        LIMIT :limit
                        """
                    ),
                    {"trigger_id": trigger_id, "limit": limit},
                )
            rows = result.mappings().all()
            return [_row_to_run(row) for row in rows]

    async def get_due_triggers(self) -> List[TriggerEntry]:
        """Return enabled triggers whose next_run_at is due (or NULL)."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, assistant_name, prompt, schedule_type, schedule_config,
                           enabled, include_new_messages, include_trigger_history, max_history_turns, watch_assistants,
                           target_type, target_name, timeout_seconds,
                           last_run_at, next_run_at, created_at, updated_at
                    FROM {TRIGGERS_TABLE}
                    WHERE enabled = true
                      AND (next_run_at IS NULL OR next_run_at <= NOW())
                    ORDER BY next_run_at ASC NULLS FIRST
                    """
                )
            )
            rows = result.mappings().all()
            return [_row_to_trigger(row) for row in rows]


def _json_dumps(value: Any) -> Optional[str]:
    import json
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _row_to_trigger(row) -> TriggerEntry:
    import json
    raw_sc = row["schedule_config"]
    if isinstance(raw_sc, dict):
        schedule_config = raw_sc
    elif isinstance(raw_sc, str):
        schedule_config = json.loads(raw_sc)
    else:
        schedule_config = {}
    import json
    raw_wa = row.get("watch_assistants")
    if isinstance(raw_wa, list):
        watch_assistants = raw_wa
    elif isinstance(raw_wa, str):
        watch_assistants = json.loads(raw_wa)
    else:
        watch_assistants = []
    return TriggerEntry(
        id=row["id"],
        name=row["name"],
        assistant_name=row["assistant_name"],
        prompt=row["prompt"],
        schedule_type=row["schedule_type"],
        schedule_config=schedule_config,
        enabled=bool(row["enabled"]),
        include_new_messages=bool(row.get("include_new_messages", False)),
        include_trigger_history=bool(row.get("include_trigger_history", False)),
        max_history_turns=int(row.get("max_history_turns", 10)),
        watch_assistants=watch_assistants,
        target_type=str(row.get("target_type", "assistant")),
        target_name=str(row.get("target_name", "")),
        timeout_seconds=int(row.get("timeout_seconds", 600) or 600),
        last_run_at=row["last_run_at"],
        next_run_at=row["next_run_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_run(row) -> TriggerRunEntry:
    return TriggerRunEntry(
        id=row["id"],
        trigger_id=row["trigger_id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        status=row["status"],
        result_text=row["result_text"],
        error_text=row["error_text"],
        conversation_id=row["conversation_id"],
    )
