"""Lightweight async message persistence to Postgres.

Stores individual chat messages without conversation wrapping.  Re-uses the
shared async engine created by :class:`agent.application.RAGApplication`.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import _quote_identifier


class MessageStore:
    """Async message store backed by a single ``messages`` table."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def save_turn(
        self,
        *,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        assistant_reasoning: Optional[str] = None,
        model_name: Optional[str] = None,
        chat_history: Optional[list[dict[str, Any]]] = None,
        retrieval_info: Optional[dict[str, Any]] = None,
        api_key_id: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        emotion: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if not self._tables_ensured:
            await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO messages (
                        id, conversation_id, user_content, assistant_content, assistant_reasoning,
                        model_name, chat_history, retrieval_info, api_key_id, tool_calls, emotion, user_id, created_at
                    )
                    VALUES (
                        :id, :cid, :user_content, :assistant_content, :assistant_reasoning,
                        :model, CAST(:chat_history AS jsonb), CAST(:retrieval_info AS jsonb), :api_key_id,
                        CAST(:tool_calls AS jsonb), CAST(:emotion AS jsonb), :user_id, now()
                    )
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "cid": conversation_id,
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "assistant_reasoning": assistant_reasoning,
                    "model": model_name,
                    "chat_history": self._json_dumps(chat_history),
                    "retrieval_info": self._json_dumps(retrieval_info),
                    "api_key_id": api_key_id,
                    "tool_calls": self._json_dumps(tool_calls),
                    "emotion": self._json_dumps(emotion),
                    "user_id": user_id,
                },
            )

    async def _ensure_tables(self) -> None:
        async with self._engine.begin() as conn:
            # Create table if it does not exist yet.
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT,
                        user_content TEXT,
                        assistant_content TEXT,
                        model_name TEXT,
                        user_id TEXT,
                        chat_history JSONB,
                        retrieval_info JSONB,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            # Add missing column if the table was created before model_name existed.
            await conn.execute(
                text(
                    """
                    ALTER TABLE messages
                    ADD COLUMN IF NOT EXISTS model_name TEXT
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    ALTER TABLE messages
                    ADD COLUMN IF NOT EXISTS user_content TEXT,
                    ADD COLUMN IF NOT EXISTS assistant_content TEXT,
                    ADD COLUMN IF NOT EXISTS chat_history JSONB,
                    ADD COLUMN IF NOT EXISTS retrieval_info JSONB,
                    ADD COLUMN IF NOT EXISTS api_key_id TEXT,
                    ADD COLUMN IF NOT EXISTS assistant_reasoning TEXT,
                    ADD COLUMN IF NOT EXISTS tool_calls JSONB,
                    ADD COLUMN IF NOT EXISTS emotion JSONB,
                    ADD COLUMN IF NOT EXISTS user_id TEXT,
                    DROP COLUMN IF EXISTS role,
                    DROP COLUMN IF EXISTS content,
                    DROP COLUMN IF EXISTS retrieval_query,
                    DROP COLUMN IF EXISTS rag_context
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages (conversation_id)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_created_at
                    ON messages (created_at DESC)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_model_name
                    ON messages (model_name)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_api_key_id
                    ON messages (api_key_id)
                    """
                )
            )
            # Drop any stale FK to the old conversations table (conversations are gone).
            try:
                fk_rows = await conn.execute(
                    text(
                        """
                        SELECT tc.constraint_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_name = 'messages'
                          AND tc.constraint_type = 'FOREIGN KEY'
                          AND kcu.column_name = 'conversation_id'
                        """
                    )
                )
                for row in fk_rows.mappings().all():
                    fk_name = row.get("constraint_name")
                    if fk_name:
                        await conn.execute(
                            text(f"ALTER TABLE messages DROP CONSTRAINT {_quote_identifier(str(fk_name))}")
                        )
            except Exception:
                pass  # no stale FK or table does not exist yet
        self._tables_ensured = True

    async def list_conversations(
        self,
        *,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        conversation_query: Optional[str] = None,
        model_name: Optional[str] = None,
        min_user_messages: int = 0,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "last_message_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        if not self._tables_ensured:
            await self._ensure_tables()

        # Sort keys map to full SQL expressions over the `enriched` CTE —
        # never interpolate raw user input into the ORDER BY.
        sort_expressions = {
            "last_message_at": "e.last_message_at",
            "first_message_at": "e.first_message_at",
            "message_count": "e.message_count",
            "user_message_count": "e.user_message_count",
            "assistant_message_count": "e.assistant_message_count",
            "conversation_id": "e.conversation_id",
            "model_name": "e.model_name",
            "user": "COALESCE(e.display_name, e.username, '')",
            "title": "COALESCE(e.title, '')",
        }
        sort_expr = sort_expressions.get(sort_by, "e.last_message_at")
        sort_dir = "DESC" if sort_order.lower().strip() == "desc" else "ASC"
        nulls = " NULLS LAST" if sort_by in ("user", "title") else ""

        where_clauses: list[str] = ["conversation_id IS NOT NULL", "conversation_id <> ''"]
        params: dict[str, Any] = {
            "min_user_messages": max(0, min_user_messages),
            "limit": max(1, page_size),
            "offset": max(0, (page - 1) * page_size),
        }
        if date_from is not None:
            where_clauses.append("created_at >= :date_from")
            params["date_from"] = date_from
        if date_to is not None:
            where_clauses.append("created_at <= :date_to")
            params["date_to"] = date_to
        if conversation_query:
            where_clauses.append("conversation_id ILIKE :conversation_like")
            params["conversation_like"] = f"%{conversation_query.strip()}%"
        if model_name:
            where_clauses.append("model_name = :model_name")
            params["model_name"] = model_name

        # Post-join filters (apply to `enriched` — they reference joined tables)
        post_clauses: list[str] = []
        if user_id:
            if user_id.strip().lower() == "anonymous":
                post_clauses.append("COALESCE(cm.user_id, g.message_user_id) IS NULL")
            else:
                post_clauses.append("COALESCE(cm.user_id, g.message_user_id) = :user_filter")
                params["user_filter"] = user_id.strip()
        if query:
            post_clauses.append("(g.conversation_id ILIKE :search_q OR cm.title ILIKE :search_q)")
            params["search_q"] = f"%{query.strip()}%"
        post_where = (" WHERE " + " AND ".join(post_clauses)) if post_clauses else ""

        where_sql = " AND ".join(where_clauses)
        cte = f"""
            WITH filtered AS (
                SELECT conversation_id, user_content, assistant_content, model_name, created_at,
                       retrieval_info, user_id, api_key_id
                FROM messages
                WHERE {where_sql}
            ),
            grouped AS (
                SELECT
                    conversation_id,
                    COUNT(*) AS message_count,
                    COUNT(*) FILTER (WHERE COALESCE(user_content, '') <> '') AS user_message_count,
                    COUNT(*) FILTER (WHERE COALESCE(assistant_content, '') <> '') AS assistant_message_count,
                    MIN(created_at) AS first_message_at,
                    MAX(created_at) AS last_message_at,
                    MAX(model_name) AS model_name,
                    COUNT(*) FILTER (WHERE retrieval_info IS NOT NULL) AS retrieval_rows,
                    MAX(user_id) AS message_user_id,
                    BOOL_OR(api_key_id IS NOT NULL) AS used_api_key
                FROM filtered
                GROUP BY conversation_id
                HAVING COUNT(*) FILTER (WHERE COALESCE(user_content, '') <> '') >= :min_user_messages
            ),
            conversation_map AS (
                SELECT DISTINCT ON (conversation_id)
                    conversation_id, user_id, title, model_name AS conv_model_name
                FROM chat_user_conversations
                ORDER BY conversation_id, updated_at DESC
            ),
            enriched AS (
                SELECT
                    g.*,
                    cm.title,
                    cm.conv_model_name,
                    COALESCE(cm.user_id, g.message_user_id) AS resolved_user_id,
                    cu.username,
                    cu.display_name
                FROM grouped g
                LEFT JOIN conversation_map cm ON cm.conversation_id = g.conversation_id
                LEFT JOIN chat_users cu ON cu.id = COALESCE(cm.user_id, g.message_user_id)
                {post_where}
            )
        """

        async with self._engine.begin() as conn:
            total_res = await conn.execute(
                text(
                    cte
                    + """
                    SELECT COUNT(*) AS total
                    FROM enriched e
                    """
                ),
                params,
            )
            total = int(total_res.mappings().one()["total"])

            rows_res = await conn.execute(
                text(
                    cte
                    + f"""
                    SELECT
                        e.conversation_id,
                        e.message_count,
                        e.user_message_count,
                        e.assistant_message_count,
                        e.first_message_at,
                        e.last_message_at,
                        e.model_name,
                        e.retrieval_rows,
                        e.title,
                        e.conv_model_name,
                        e.resolved_user_id AS user_id,
                        e.username,
                        e.display_name,
                        e.used_api_key
                    FROM enriched e
                    ORDER BY {sort_expr} {sort_dir}{nulls}
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            )
            items = [dict(row) for row in rows_res.mappings().all()]
            return {
                "total": total,
                "page": max(1, page),
                "page_size": max(1, page_size),
                "items": items,
            }

    async def list_conversation_users(self) -> dict[str, Any]:
        """Users that own conversations, plus a count of anonymous ones.

        Drives the user filter dropdown on the conversations page.
        """
        if not self._tables_ensured:
            await self._ensure_tables()
        async with self._engine.begin() as conn:
            # Users linked through chat_user_conversations plus users that only
            # appear via messages.user_id (no conversation row exists).
            users_res = await conn.execute(
                text(
                    """
                    WITH conv_owners AS (
                        SELECT conversation_id, user_id
                        FROM (
                            SELECT DISTINCT ON (conversation_id) conversation_id, user_id
                            FROM chat_user_conversations
                            ORDER BY conversation_id, updated_at DESC
                        ) cm
                        UNION
                        SELECT DISTINCT m.conversation_id, m.user_id
                        FROM messages m
                        WHERE m.conversation_id IS NOT NULL AND m.conversation_id <> ''
                          AND m.user_id IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM chat_user_conversations c
                              WHERE c.conversation_id = m.conversation_id
                          )
                    )
                    SELECT u.id AS user_id, u.username, u.display_name,
                           COUNT(*) AS conversation_count
                    FROM conv_owners co
                    JOIN chat_users u ON u.id = co.user_id
                    GROUP BY u.id, u.username, u.display_name
                    ORDER BY u.display_name NULLS LAST, u.username
                    """
                )
            )
            users = [dict(row) for row in users_res.mappings().all()]

            # Anonymous = no resolved owner (same semantics as list_conversations).
            anon_res = await conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS n FROM (
                        SELECT m.conversation_id
                        FROM messages m
                        WHERE m.conversation_id IS NOT NULL AND m.conversation_id <> ''
                        GROUP BY m.conversation_id
                        HAVING COALESCE(
                            (SELECT c.user_id FROM chat_user_conversations c
                             WHERE c.conversation_id = m.conversation_id
                             ORDER BY c.updated_at DESC LIMIT 1),
                            MAX(m.user_id)
                        ) IS NULL
                    ) t
                    """
                )
            )
            anonymous_count = int(anon_res.mappings().one()["n"])
            return {"users": users, "anonymous_count": anonymous_count}

    async def get_conversation_users(
        self, conversation_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Batch-lookup title/user for a set of conversation ids (export)."""
        if not conversation_ids:
            return {}
        if not self._tables_ensured:
            await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    (
                        SELECT DISTINCT ON (cm.conversation_id)
                            cm.conversation_id, cm.user_id, cm.title,
                            u.username, u.display_name
                        FROM chat_user_conversations cm
                        LEFT JOIN chat_users u ON u.id = cm.user_id
                        WHERE cm.conversation_id = ANY(:cids)
                        ORDER BY cm.conversation_id, cm.updated_at DESC
                    )
                    UNION ALL
                    -- Conversations owned only via messages.user_id (no
                    -- chat_user_conversations row exists)
                    (
                        SELECT DISTINCT ON (m.conversation_id)
                            m.conversation_id, m.user_id, NULL AS title,
                            u.username, u.display_name
                        FROM messages m
                        JOIN chat_users u ON u.id = m.user_id
                        WHERE m.conversation_id = ANY(:cids)
                          AND m.user_id IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM chat_user_conversations c
                              WHERE c.conversation_id = m.conversation_id
                          )
                        ORDER BY m.conversation_id, m.created_at DESC
                    )
                    """
                ),
                {"cids": list(conversation_ids)},
            )
            return {row["conversation_id"]: dict(row) for row in res.mappings().all()}

    async def get_conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        if not self._tables_ensured:
            await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    SELECT
                        id,
                        conversation_id,
                        user_content,
                        assistant_content,
                        assistant_reasoning,
                        model_name,
                        chat_history,
                        retrieval_info,
                        tool_calls,
                        emotion,
                        created_at
                    FROM messages
                    WHERE conversation_id = :conversation_id
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"conversation_id": conversation_id},
            )
            return [dict(row) for row in res.mappings().all()]

    async def get_conversation_metadata(self, conversation_id: str) -> Optional[dict[str, Any]]:
        if not self._tables_ensured:
            await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    WITH grouped AS (
                        SELECT
                            conversation_id,
                            COUNT(*) AS message_count,
                            COUNT(*) FILTER (WHERE COALESCE(user_content, '') <> '') AS user_message_count,
                            COUNT(*) FILTER (WHERE COALESCE(assistant_content, '') <> '') AS assistant_message_count,
                            MIN(created_at) AS first_message_at,
                            MAX(created_at) AS last_message_at,
                            MAX(model_name) AS model_name,
                            COUNT(*) FILTER (WHERE retrieval_info IS NOT NULL) AS retrieval_rows,
                            MAX(user_id) AS message_user_id,
                            BOOL_OR(api_key_id IS NOT NULL) AS used_api_key
                        FROM messages
                        WHERE conversation_id = :conversation_id
                        GROUP BY conversation_id
                    ),
                    conversation_map AS (
                        SELECT DISTINCT ON (conversation_id)
                            conversation_id, user_id, title, model_name AS conv_model_name
                        FROM chat_user_conversations
                        WHERE conversation_id = :conversation_id
                        ORDER BY conversation_id, updated_at DESC
                    )
                    SELECT
                        g.*,
                        cm.title,
                        cm.conv_model_name,
                        COALESCE(cm.user_id, g.message_user_id) AS resolved_user_id,
                        cu.username,
                        cu.display_name
                    FROM grouped g
                    LEFT JOIN conversation_map cm ON cm.conversation_id = g.conversation_id
                    LEFT JOIN chat_users cu ON cu.id = COALESCE(cm.user_id, g.message_user_id)
                    """
                ),
                {"conversation_id": conversation_id},
            )
            row = res.mappings().first()
            if row is None:
                return None
            data = dict(row)
            data["user_id"] = data.pop("resolved_user_id", None)
            data.pop("message_user_id", None)
            return data

    async def list_models(
        self,
        *,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[str]:
        if not self._tables_ensured:
            await self._ensure_tables()

        where_clauses = ["model_name IS NOT NULL", "model_name <> ''"]
        params: dict[str, Any] = {}
        if date_from is not None:
            where_clauses.append("created_at >= :date_from")
            params["date_from"] = date_from
        if date_to is not None:
            where_clauses.append("created_at <= :date_to")
            params["date_to"] = date_to

        where_sql = " AND ".join(where_clauses)
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT DISTINCT model_name
                    FROM messages
                    WHERE {where_sql}
                    ORDER BY model_name ASC
                    """
                ),
                params,
            )
            return [str(row[0]) for row in res.fetchall()]

    async def count_messages_for_assistants(
        self,
        *,
        assistant_names: list[str],
        since: datetime,
    ) -> int:
        if not self._tables_ensured:
            await self._ensure_tables()
        if not assistant_names:
            return 0
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM messages
                    WHERE model_name = ANY(:assistant_names)
                      AND created_at > :since
                      AND conversation_id NOT LIKE 'trigger-%'
                    """
                ),
                {
                    "assistant_names": assistant_names,
                    "since": since,
                },
            )
            row = res.mappings().first()
            return int(row["cnt"]) if row else 0

    async def get_messages_for_assistants(
        self,
        *,
        assistant_names: list[str],
        since: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._tables_ensured:
            await self._ensure_tables()
        if not assistant_names:
            return []
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    """
                    SELECT
                        id,
                        conversation_id,
                        user_content,
                        assistant_content,
                        assistant_reasoning,
                        model_name,
                        tool_calls,
                        emotion,
                        created_at
                    FROM messages
                    WHERE model_name = ANY(:assistant_names)
                      AND created_at > :since
                      AND conversation_id NOT LIKE 'trigger-%'
                    ORDER BY created_at ASC
                    LIMIT :limit
                    """
                ),
                {
                    "assistant_names": assistant_names,
                    "since": since,
                    "limit": limit,
                },
            )
            return [dict(row) for row in res.mappings().all()]

    async def list_messages_for_export(
        self,
        *,
        date_from: datetime,
        date_to: datetime,
        conversation_query: Optional[str] = None,
        model_name: Optional[str] = None,
        min_user_messages: int = 0,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self._tables_ensured:
            await self._ensure_tables()

        where_clauses: list[str] = [
            "created_at >= :date_from",
            "created_at <= :date_to",
            "conversation_id IS NOT NULL",
            "conversation_id <> ''",
        ]
        params: dict[str, Any] = {
            "date_from": date_from,
            "date_to": date_to,
            "min_user_messages": max(0, min_user_messages),
        }
        if conversation_query:
            where_clauses.append("conversation_id ILIKE :conversation_like")
            params["conversation_like"] = f"%{conversation_query.strip()}%"
        if model_name:
            where_clauses.append("model_name = :model_name")
            params["model_name"] = model_name
        # Resolved owner = latest chat_user_conversations mapping, falling back
        # to messages.user_id — must match list_conversations' enriched CTE.
        resolved_owner_sql = (
            "COALESCE((SELECT c.user_id FROM chat_user_conversations c "
            "WHERE c.conversation_id = messages.conversation_id "
            "ORDER BY c.updated_at DESC LIMIT 1), messages.user_id)"
        )
        if user_id:
            if user_id.strip().lower() == "anonymous":
                where_clauses.append(f"{resolved_owner_sql} IS NULL")
            else:
                where_clauses.append(f"{resolved_owner_sql} = :user_filter")
                params["user_filter"] = user_id.strip()
        if query:
            where_clauses.append(
                "(messages.conversation_id ILIKE :search_q OR (SELECT c.title "
                "FROM chat_user_conversations c "
                "WHERE c.conversation_id = messages.conversation_id "
                "ORDER BY c.updated_at DESC LIMIT 1) ILIKE :search_q)"
            )
            params["search_q"] = f"%{query.strip()}%"
        where_sql = " AND ".join(where_clauses)

        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    WITH filtered AS (
                        SELECT *
                        FROM messages
                        WHERE {where_sql}
                    ),
                    allowed_conversations AS (
                        SELECT conversation_id
                        FROM filtered
                        GROUP BY conversation_id
                        HAVING COUNT(*) FILTER (WHERE COALESCE(user_content, '') <> '') >= :min_user_messages
                    )
                    SELECT
                        f.id,
                        f.conversation_id,
                        f.user_content,
                        f.assistant_content,
                        f.assistant_reasoning,
                        f.model_name,
                        f.chat_history,
                        f.retrieval_info,
                        f.tool_calls,
                        f.emotion,
                        f.user_id,
                        f.created_at
                    FROM filtered f
                    INNER JOIN allowed_conversations ac
                      ON ac.conversation_id = f.conversation_id
                    ORDER BY f.conversation_id ASC, f.created_at ASC, f.id ASC
                    """
                ),
                params,
            )
            return [dict(row) for row in res.mappings().all()]

    @staticmethod
    def _json_dumps(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)
