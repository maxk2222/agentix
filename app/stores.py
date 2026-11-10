"""Postgres-backed stores for instructions versions and raw knowledge files."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import (
    AGENT_MEMORIES_TABLE,
    INSTRUCTIONS_VERSIONS_TABLE,
    RAW_FILES_TABLE,
)

logger = logging.getLogger(__name__)


class InstructionsStore:
    """CRUD + versioning for assistant instructions YAML."""

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
                    CREATE TABLE IF NOT EXISTS {INSTRUCTIONS_VERSIONS_TABLE} (
                        id SERIAL PRIMARY KEY,
                        yaml_content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        created_by TEXT,
                        is_current BOOLEAN NOT NULL DEFAULT FALSE
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_instructions_current
                    ON {INSTRUCTIONS_VERSIONS_TABLE} (is_current)
                    WHERE is_current = TRUE
                    """
                )
            )
        self._tables_ensured = True

    async def get_current(self) -> Optional[str]:
        """Return the current YAML content, or None if no version exists."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT yaml_content
                    FROM {INSTRUCTIONS_VERSIONS_TABLE}
                    WHERE is_current = TRUE
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
            )
            row = res.scalar()
            return str(row) if row else None

    async def save(self, yaml_content: str, created_by: str = "admin") -> int:
        """Save a new version and mark it as current. Returns the new version id."""
        await self._ensure_tables()
        current = await self.get_current()
        if current == yaml_content:
            return await self.get_current_version_id()
        async with self._engine.begin() as conn:
            # Unmark previous current
            await conn.execute(
                text(
                    f"""
                    UPDATE {INSTRUCTIONS_VERSIONS_TABLE}
                    SET is_current = FALSE
                    WHERE is_current = TRUE
                    """
                )
            )
            # Insert new version
            res = await conn.execute(
                text(
                    f"""
                    INSERT INTO {INSTRUCTIONS_VERSIONS_TABLE}
                        (yaml_content, created_at, created_by, is_current)
                    VALUES (:yaml_content, NOW(), :created_by, TRUE)
                    RETURNING id
                    """
                ),
                {"yaml_content": yaml_content, "created_by": created_by},
            )
            version_id = res.scalar()
            return int(version_id)

    async def get_current_version_id(self) -> int:
        """Return the id of the current version, or 0 if none."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT id FROM {INSTRUCTIONS_VERSIONS_TABLE}
                    WHERE is_current = TRUE
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
            )
            row = res.scalar()
            return int(row) if row else 0

    async def get_version(self, version_id: int) -> Optional[Dict[str, Any]]:
        """Return a specific version by id (including yaml_content)."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT id, yaml_content, created_at, created_by, is_current
                    FROM {INSTRUCTIONS_VERSIONS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": version_id},
            )
            row = res.mappings().first()
            if row is None:
                return None
            return {
                "id": row.id,
                "yaml_content": row.yaml_content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "created_by": row.created_by,
                "is_current": row.is_current,
            }

    async def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the last N versions (newest first)."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT id, yaml_content, created_at, created_by, is_current
                    FROM {INSTRUCTIONS_VERSIONS_TABLE}
                    ORDER BY id DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "created_by": row.created_by,
                    "is_current": row.is_current,
                }
                for row in res.mappings()
            ]

    async def rollback(self, version_id: int) -> bool:
        """Set the given version as current. Returns True if found."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            # Verify version exists
            res = await conn.execute(
                text(
                    f"""
                    SELECT id FROM {INSTRUCTIONS_VERSIONS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": version_id},
            )
            if res.scalar() is None:
                return False
            # Unmark all
            await conn.execute(
                text(
                    f"""
                    UPDATE {INSTRUCTIONS_VERSIONS_TABLE}
                    SET is_current = FALSE
                    """
                )
            )
            # Mark selected
            await conn.execute(
                text(
                    f"""
                    UPDATE {INSTRUCTIONS_VERSIONS_TABLE}
                    SET is_current = TRUE
                    WHERE id = :id
                    """
                ),
                {"id": version_id},
            )
            return True

    async def bootstrap_from_file(self, filepath: str) -> bool:
        """Import file contents as the first DB version if DB is empty."""
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            return False
        current = await self.get_current()
        if current is not None:
            return False
        yaml_text = path.read_text(encoding="utf-8")
        await self.save(yaml_text, created_by="bootstrap")
        logger.info("Bootstrapped instructions DB from %s", filepath)
        return True


class RawFileStore:
    """Store / retrieve raw knowledge files (PDF, Word, TXT) as BYTEA."""

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
                    CREATE TABLE IF NOT EXISTS {RAW_FILES_TABLE} (
                        id SERIAL PRIMARY KEY,
                        collection_name TEXT NOT NULL,
                        source TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        content_type TEXT,
                        file_data BYTEA NOT NULL,
                        file_hash TEXT NOT NULL,
                        uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        indexed_at TIMESTAMPTZ,
                        UNIQUE (collection_name, source)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_raw_files_collection
                    ON {RAW_FILES_TABLE} (collection_name)
                    """
                )
            )
        self._tables_ensured = True

    async def save(
        self,
        collection_name: str,
        source: str,
        filename: str,
        content_type: Optional[str],
        data: bytes,
    ) -> None:
        """Insert or update a raw file."""
        await self._ensure_tables()
        file_hash = hashlib.sha256(data).hexdigest()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {RAW_FILES_TABLE}
                        (collection_name, source, filename, content_type, file_data, file_hash, uploaded_at, updated_at)
                    VALUES (:collection_name, :source, :filename, :content_type, :file_data, :file_hash, NOW(), NOW())
                    ON CONFLICT (collection_name, source) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        content_type = EXCLUDED.content_type,
                        file_data = EXCLUDED.file_data,
                        file_hash = EXCLUDED.file_hash,
                        updated_at = NOW()
                    """
                ),
                {
                    "collection_name": collection_name,
                    "source": source,
                    "filename": filename,
                    "content_type": content_type,
                    "file_data": data,
                    "file_hash": file_hash,
                },
            )

    async def get(self, collection_name: str, source: str) -> Optional[Dict[str, Any]]:
        """Return a dict with file metadata + bytes, or None."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT filename, content_type, file_data, file_hash, uploaded_at, updated_at
                    FROM {RAW_FILES_TABLE}
                    WHERE collection_name = :collection_name AND source = :source
                    """
                ),
                {"collection_name": collection_name, "source": source},
            )
            row = res.mappings().first()
            if row is None:
                return None
            return dict(row)

    async def delete(self, collection_name: str, source: str) -> bool:
        """Delete a raw file. Returns True if a row was deleted."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    DELETE FROM {RAW_FILES_TABLE}
                    WHERE collection_name = :collection_name AND source = :source
                    """
                ),
                {"collection_name": collection_name, "source": source},
            )
            return int(getattr(res, "rowcount", 0) or 0) > 0

    async def list_by_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """List all files in a collection (without file_data bytes)."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT source, filename, content_type, file_hash, uploaded_at, updated_at, indexed_at
                    FROM {RAW_FILES_TABLE}
                    WHERE collection_name = :collection_name
                    ORDER BY source
                    """
                ),
                {"collection_name": collection_name},
            )
            return [dict(row) for row in res.mappings()]

    async def list_collections(self) -> List[str]:
        """Return distinct collection names that have at least one raw file."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT DISTINCT collection_name
                    FROM {RAW_FILES_TABLE}
                    ORDER BY collection_name
                    """
                )
            )
            return [str(row.collection_name) for row in res]

    async def mark_indexed(self, collection_name: str, source: str) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {RAW_FILES_TABLE}
                    SET indexed_at = NOW()
                    WHERE collection_name = :collection_name AND source = :source
                    """
                ),
                {"collection_name": collection_name, "source": source},
            )

    async def delete_collection(self, collection_name: str) -> int:
        """Delete all files for a collection. Returns number of rows deleted."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    DELETE FROM {RAW_FILES_TABLE}
                    WHERE collection_name = :collection_name
                    """
                ),
                {"collection_name": collection_name},
            )
            return int(getattr(res, "rowcount", 0) or 0)


class AgentMemoryStore:
    """Persistent conversation memory for agent assistants."""

    MAX_MEMORY_CHARS = 8000

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
                    CREATE TABLE IF NOT EXISTS {AGENT_MEMORIES_TABLE} (
                        conversation_id TEXT NOT NULL,
                        assistant_name TEXT NOT NULL,
                        memory TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (conversation_id, assistant_name)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_agent_memories_updated
                    ON {AGENT_MEMORIES_TABLE} (updated_at DESC)
                    """
                )
            )
        self._tables_ensured = True

    async def get_memory(self, conversation_id: str, assistant_name: str) -> Optional[str]:
        """Return current memory or None if not exists."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT memory FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id = :conversation_id AND assistant_name = :assistant_name
                    """
                ),
                {"conversation_id": conversation_id, "assistant_name": assistant_name},
            )
            row = res.mappings().first()
            return row["memory"] if row else None

    async def set_memory(self, conversation_id: str, assistant_name: str, memory: str) -> str:
        """Overwrite memory, truncating to max size. Returns the stored value."""
        await self._ensure_tables()
        truncated = memory[: self.MAX_MEMORY_CHARS]
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {AGENT_MEMORIES_TABLE}
                        (conversation_id, assistant_name, memory, updated_at)
                    VALUES
                        (:conversation_id, :assistant_name, :memory, NOW())
                    ON CONFLICT (conversation_id, assistant_name)
                    DO UPDATE SET memory = EXCLUDED.memory, updated_at = NOW()
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "assistant_name": assistant_name,
                    "memory": truncated,
                },
            )
        return truncated

    async def append_memory(self, conversation_id: str, assistant_name: str, content: str) -> str:
        """Append content to existing memory, truncating to max size."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            # Read current memory first
            res = await conn.execute(
                text(
                    f"""
                    SELECT memory FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id = :conversation_id AND assistant_name = :assistant_name
                    """
                ),
                {"conversation_id": conversation_id, "assistant_name": assistant_name},
            )
            row = res.mappings().first()
            current = row["memory"] if row else ""
            new_memory = (current + "\n" + content).strip() if current else content
            truncated = new_memory[: self.MAX_MEMORY_CHARS]
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {AGENT_MEMORIES_TABLE}
                        (conversation_id, assistant_name, memory, updated_at)
                    VALUES
                        (:conversation_id, :assistant_name, :memory, NOW())
                    ON CONFLICT (conversation_id, assistant_name)
                    DO UPDATE SET memory = EXCLUDED.memory, updated_at = NOW()
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "assistant_name": assistant_name,
                    "memory": truncated,
                },
            )
        return truncated

    async def delete_memory(self, conversation_id: str, assistant_name: str) -> bool:
        """Delete memory for a conversation. Returns True if something was deleted."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    DELETE FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id = :conversation_id AND assistant_name = :assistant_name
                    """
                ),
                {"conversation_id": conversation_id, "assistant_name": assistant_name},
            )
            return int(getattr(res, "rowcount", 0) or 0) > 0

    async def delete_memories_for_conversation(self, conversation_id: str) -> int:
        """Delete ALL memory entries for a conversation. Returns rows deleted."""
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    DELETE FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id = :conversation_id
                    """
                ),
                {"conversation_id": conversation_id},
            )
            return int(res.rowcount or 0)

    async def memory_conversation_index(self) -> Dict[str, Any]:
        """Map of conversation_id → latest updated_at across its memory rows."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT conversation_id, MAX(updated_at) AS latest
                    FROM {AGENT_MEMORIES_TABLE}
                    GROUP BY conversation_id
                    """
                )
            )
            return {row[0]: row[1] for row in res.all()}

    async def list_memories_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return all memory entries for a given conversation_id."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT assistant_name, memory, created_at, updated_at
                    FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id = :conversation_id
                    ORDER BY updated_at DESC
                    """
                ),
                {"conversation_id": conversation_id},
            )
            rows = res.mappings().all()
            return [
                {
                    "assistant_name": row["assistant_name"],
                    "memory": row["memory"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
                for row in rows
            ]

    async def check_conversations_with_memory(self, conversation_ids: List[str]) -> List[str]:
        """Return the subset of conversation_ids that have at least one memory entry."""
        if not conversation_ids:
            return []
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            placeholders = ", ".join(f":id{i}" for i in range(len(conversation_ids)))
            params = {f"id{i}": cid for i, cid in enumerate(conversation_ids)}
            res = await conn.execute(
                text(
                    f"""
                    SELECT DISTINCT conversation_id FROM {AGENT_MEMORIES_TABLE}
                    WHERE conversation_id IN ({placeholders})
                    """
                ),
                params,
            )
            return [row[0] for row in res.all()]
