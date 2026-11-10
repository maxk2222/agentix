"""Postgres-backed store for role-based assistant access in public chat."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


class ChatRoleAssistantsStore:
    """CRUD for mapping roles to allowed assistants in public chat."""

    TABLE = "chat_role_assistants"

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
                    CREATE TABLE IF NOT EXISTS {self.TABLE} (
                        role TEXT NOT NULL,
                        assistant_name TEXT NOT NULL,
                        PRIMARY KEY (role, assistant_name)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_chat_role_assistants_role
                    ON {self.TABLE} (role)
                    """
                )
            )
        self._tables_ensured = True

    async def get_assistants_for_role(self, role: str) -> List[str]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT assistant_name FROM {self.TABLE}
                    WHERE role = :role
                    ORDER BY assistant_name
                    """
                ),
                {"role": role},
            )
            rows = result.mappings().all()
            return [row["assistant_name"] for row in rows]

    async def get_all_mappings(self) -> Dict[str, List[str]]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT role, assistant_name FROM {self.TABLE}
                    ORDER BY role, assistant_name
                    """
                )
            )
            rows = result.mappings().all()
            mappings: Dict[str, List[str]] = {}
            for row in rows:
                mappings.setdefault(row["role"], []).append(row["assistant_name"])
            return mappings

    async def set_assistants_for_role(self, role: str, assistant_names: List[str]) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE role = :role"),
                {"role": role},
            )
            for name in assistant_names:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self.TABLE} (role, assistant_name)
                        VALUES (:role, :assistant_name)
                        """
                    ),
                    {"role": role, "assistant_name": name},
                )

    async def delete_role(self, role: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE role = :role"),
                {"role": role},
            )
