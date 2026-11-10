"""Postgres-backed store for widget default configuration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import WIDGET_DEFAULTS_TABLE

logger = logging.getLogger(__name__)


class WidgetDefaultsStore:
    """CRUD for widget default settings persisted as JSON."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def _ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {WIDGET_DEFAULTS_TABLE} (
                        id SERIAL PRIMARY KEY,
                        json_content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        created_by TEXT
                    )
                    """
                )
            )
        self._tables_ensured = True

    async def get(self) -> Optional[Dict[str, Any]]:
        """Return the latest widget defaults as a dict, or None if none exist."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT json_content
                    FROM {WIDGET_DEFAULTS_TABLE}
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
            )
            row = res.scalar()
            if row is None:
                return None
            import json
            return json.loads(str(row))

    async def save(self, config: Dict[str, Any], created_by: str = "admin") -> None:
        """Save new widget defaults, replacing any previous row."""
        await self._ensure_tables()
        import json
        json_content = json.dumps(config)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {WIDGET_DEFAULTS_TABLE}
                        (json_content, created_at, updated_at, created_by)
                    VALUES (:json_content, NOW(), NOW(), :created_by)
                    """
                ),
                {"json_content": json_content, "created_by": created_by},
            )
