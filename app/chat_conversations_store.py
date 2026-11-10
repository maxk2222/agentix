"""Async store for chat user conversations backed by Postgres."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# user_id is NOT NULL in chat_conversations — anonymous conversations get a
# row under this sentinel so a generated title has somewhere to live. It can
# never collide with a real id, never matches list_for_user queries and is
# treated like NULL (bearer semantics) by the artifact access check.
ANON_USER_ID = "_anonymous"


class ChatConversationsStore:
    """Async store linking conversations to chat users."""

    TABLE = "chat_user_conversations"

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
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        title TEXT,
                        model_name TEXT,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_updated
                    ON {self.TABLE} (user_id, updated_at DESC)
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_chat_conversations_conversation_id
                    ON {self.TABLE} (conversation_id)
                    """
                )
            )
            # title_source marks who set the title ('auto' | 'manual') so the
            # auto-namer may refresh its own titles but never user edits;
            # auto_title_count records the message count at the last
            # auto-generated title for spaced re-titling.
            for col, col_type in (("title_source", "TEXT"), ("auto_title_count", "INT")):
                await conn.execute(
                    text(f"ALTER TABLE {self.TABLE} ADD COLUMN IF NOT EXISTS {col} {col_type}")
                )
        self._tables_ensured = True

    async def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        title: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        await self.ensure_tables()
        record_id = str(uuid.uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (id, user_id, conversation_id, title, model_name, created_at, updated_at)
                    VALUES (:id, :user_id, :conversation_id, :title, :model_name, now(), now())
                    """
                ),
                {
                    "id": record_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "title": title,
                    "model_name": model_name,
                },
            )
        return record_id

    async def ensure_exists(
        self,
        *,
        user_id: str,
        conversation_id: str,
        title: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id FROM {self.TABLE}
                    WHERE user_id = :user_id AND conversation_id = :conversation_id
                    """
                ),
                {"user_id": user_id, "conversation_id": conversation_id},
            )
            if result.scalar() is not None:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {self.TABLE}
                        SET updated_at = now()
                        WHERE user_id = :user_id AND conversation_id = :conversation_id
                        """
                    ),
                    {"user_id": user_id, "conversation_id": conversation_id},
                )
                return
            record_id = str(uuid.uuid4())
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (id, user_id, conversation_id, title, model_name, created_at, updated_at)
                    VALUES (:id, :user_id, :conversation_id, :title, :model_name, now(), now())
                    """
                ),
                {
                    "id": record_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "title": title,
                    "model_name": model_name,
                },
            )

    async def list_for_user(self, user_id: Any) -> List[dict[str, Any]]:
        """List conversations for a user.

        ``user_id`` may be a single id or a list of ids (e.g. canonical chat
        user id plus a linked dashboard user id) so legacy rows created under
        either identity stay reachable.
        """
        await self.ensure_tables()
        ids = list(user_id) if isinstance(user_id, (list, tuple, set)) else [user_id]
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        c.id,
                        c.conversation_id,
                        c.title,
                        c.model_name,
                        c.created_at,
                        c.updated_at,
                        COALESCE(m.message_count, 0) AS message_count
                    FROM {self.TABLE} c
                    LEFT JOIN (
                        SELECT conversation_id, COUNT(*) AS message_count
                        FROM messages
                        GROUP BY conversation_id
                    ) m ON m.conversation_id = c.conversation_id
                    WHERE c.user_id = ANY(:user_ids)
                    ORDER BY c.updated_at DESC
                    """
                ),
                {"user_ids": ids},
            )
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[dict[str, Any]]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            # Prefer a real-user row when an anonymous sentinel row exists for
            # the same conversation_id — titles belong on the row a signed-in
            # user's sidebar actually reads.
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, user_id, conversation_id, title, model_name,
                           title_source, auto_title_count, created_at, updated_at
                    FROM {self.TABLE}
                    WHERE conversation_id = :conversation_id
                    ORDER BY (user_id = '{ANON_USER_ID}'), updated_at DESC
                    """
                ),
                {"conversation_id": conversation_id},
            )
            row = result.mappings().fetchone()
            return dict(row) if row else None

    async def update_title(self, conversation_id: str, title: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET title = :title, title_source = 'manual', updated_at = now()
                    WHERE conversation_id = :conversation_id
                    """
                ),
                {"conversation_id": conversation_id, "title": title},
            )

    async def set_title(self, conversation_id: str, title: str, message_count: int = 0) -> None:
        """Auto-namer title write: marks ``title_source='auto'`` and records
        the message count so titles can be refreshed as conversations grow.
        Never touches ``updated_at`` — the sidebar sorts by it and a
        background rename must not reorder it."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET title = :title, title_source = 'auto', auto_title_count = :count
                    WHERE conversation_id = :conversation_id
                    """
                ),
                {"conversation_id": conversation_id, "title": title, "count": message_count},
            )

    async def list_conversation_ids(self) -> List[str]:
        """All conversation_ids — used by the storage sweeper to match
        on-disk dirs (named via ``sanitize_filename``) against live rows."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT conversation_id FROM {self.TABLE}")
            )
            return [str(row[0]) for row in result.fetchall() if row[0]]

    async def get_updated_at_map(self) -> Dict[str, Any]:
        """conversation_id -> updated_at for retention checks."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT conversation_id, updated_at FROM {self.TABLE}")
            )
            return {str(row[0]): row[1] for row in result.fetchall() if row[0]}

    async def delete(self, conversation_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE conversation_id = :conversation_id"),
                {"conversation_id": conversation_id},
            )
