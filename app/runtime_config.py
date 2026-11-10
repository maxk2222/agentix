"""Async runtime configuration store backed by Postgres.

Provides persistent key-value storage for application settings, with merge
logic layered on top of :class:`AppConfig` so environment variables always
win.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import _quote_identifier

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfigEntry:
    key: str
    value: Any
    source: str = "db"
    updated_at: Optional[datetime] = None
    overridden_by_env: bool = False


class RuntimeConfigStore:
    """Async store for editable runtime settings."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
        self._tables_ensured = True

    async def get(self, key: str, default: Any = None) -> Any:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("SELECT value FROM runtime_config WHERE key = :key"),
                {"key": key},
            )
            row = result.mappings().fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]

    async def get_all(self) -> Dict[str, Any]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("SELECT key, value FROM runtime_config ORDER BY key")
            )
            rows = result.mappings().all()
            out: Dict[str, Any] = {}
            for row in rows:
                raw = row["value"]
                try:
                    out[row["key"]] = json.loads(raw)
                except json.JSONDecodeError:
                    out[row["key"]] = raw
            return out

    async def set(self, key: str, value: Any) -> None:
        await self.ensure_tables()
        payload = json.dumps(value) if not isinstance(value, str) else value
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO runtime_config (key, value, updated_at)
                    VALUES (:key, :value, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {"key": key, "value": payload},
            )

    async def set_many(self, values: Dict[str, Any]) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            for key, value in values.items():
                payload = json.dumps(value) if not isinstance(value, str) else value
                await conn.execute(
                    text(
                        """
                        INSERT INTO runtime_config (key, value, updated_at)
                        VALUES (:key, :value, now())
                        ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {"key": key, "value": payload},
                )

    async def delete(self, key: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM runtime_config WHERE key = :key"),
                {"key": key},
            )

    async def export_all(self) -> Dict[str, Any]:
        """Export every stored key-value pair."""
        return await self.get_all()

    async def import_many(self, values: Dict[str, Any]) -> None:
        await self.set_many(values)
