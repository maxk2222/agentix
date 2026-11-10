"""Postgres-backed stores for WebDAV source configurations and file metadata."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import WEBDAV_SOURCE_FILES_TABLE, WEBDAV_SOURCES_TABLE
from app.secrets import _decrypt_password, _encrypt_password

logger = logging.getLogger(__name__)

__all__ = ["WebDAVSourceStore", "WebDAVFileMetaStore", "_encrypt_password", "_decrypt_password"]


class WebDAVSourceStore:
    """CRUD for WebDAV source configurations with encrypted passwords."""

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
                    CREATE TABLE IF NOT EXISTS {WEBDAV_SOURCES_TABLE} (
                        id SERIAL PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        webdav_url TEXT NOT NULL,
                        remote_path TEXT NOT NULL DEFAULT '/',
                        username TEXT NOT NULL,
                        password_encrypted TEXT NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        convert_to_okf BOOLEAN NOT NULL DEFAULT TRUE,
                        last_sync_at TIMESTAMPTZ,
                        last_sync_status TEXT,
                        last_sync_error TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {WEBDAV_SOURCES_TABLE}
                    ADD COLUMN IF NOT EXISTS convert_to_okf BOOLEAN NOT NULL DEFAULT TRUE
                    """
                )
            )
        self._tables_ensured = True

    async def list_sources(self) -> List[Dict[str, Any]]:
        """Return all sources without passwords."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT id, name, webdav_url, remote_path, username, enabled,
                           convert_to_okf,
                           last_sync_at, last_sync_status, last_sync_error,
                           created_at, updated_at
                    FROM {WEBDAV_SOURCES_TABLE}
                    ORDER BY name
                    """
                )
            )
            return [dict(row) for row in res.mappings()]

    async def get_source(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a single source including decrypted password."""
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT id, name, webdav_url, remote_path, username,
                           password_encrypted, enabled, convert_to_okf,
                           last_sync_at, last_sync_status, last_sync_error,
                           created_at, updated_at
                    FROM {WEBDAV_SOURCES_TABLE}
                    WHERE name = :name
                    """
                ),
                {"name": name},
            )
            row = res.mappings().first()
            if row is None:
                return None
            data = dict(row)
            try:
                data["password"] = _decrypt_password(data["password_encrypted"])
            except Exception as exc:
                logger.warning("Failed to decrypt password for WebDAV source '%s': %s", name, exc)
                data["password"] = ""
            data.pop("password_encrypted", None)
            return data

    async def create_source(
        self,
        name: str,
        webdav_url: str,
        remote_path: str,
        username: str,
        password: str,
        enabled: bool = True,
        convert_to_okf: bool = True,
    ) -> Dict[str, Any]:
        await self._ensure_tables()
        encrypted = _encrypt_password(password)
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    INSERT INTO {WEBDAV_SOURCES_TABLE}
                        (name, webdav_url, remote_path, username, password_encrypted, enabled, convert_to_okf, updated_at)
                    VALUES (:name, :webdav_url, :remote_path, :username, :password_encrypted, :enabled, :convert_to_okf, NOW())
                    RETURNING id, name, webdav_url, remote_path, username, enabled, convert_to_okf,
                              last_sync_at, last_sync_status, last_sync_error,
                              created_at, updated_at
                    """
                ),
                {
                    "name": name,
                    "webdav_url": webdav_url,
                    "remote_path": remote_path,
                    "username": username,
                    "password_encrypted": encrypted,
                    "enabled": enabled,
                    "convert_to_okf": convert_to_okf,
                },
            )
            row = res.mappings().first()
            return dict(row) if row else {}

    async def update_source(
        self,
        name: str,
        webdav_url: Optional[str] = None,
        remote_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        enabled: Optional[bool] = None,
        convert_to_okf: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        fields: List[str] = []
        params: Dict[str, Any] = {"name": name}
        if webdav_url is not None:
            fields.append("webdav_url = :webdav_url")
            params["webdav_url"] = webdav_url
        if remote_path is not None:
            fields.append("remote_path = :remote_path")
            params["remote_path"] = remote_path
        if username is not None:
            fields.append("username = :username")
            params["username"] = username
        if password is not None:
            fields.append("password_encrypted = :password_encrypted")
            params["password_encrypted"] = _encrypt_password(password)
        if enabled is not None:
            fields.append("enabled = :enabled")
            params["enabled"] = enabled
        if convert_to_okf is not None:
            fields.append("convert_to_okf = :convert_to_okf")
            params["convert_to_okf"] = convert_to_okf
        if not fields:
            return await self.get_source(name)
        fields.append("updated_at = NOW()")
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    UPDATE {WEBDAV_SOURCES_TABLE}
                    SET {', '.join(fields)}
                    WHERE name = :name
                    RETURNING id, name, webdav_url, remote_path, username, enabled, convert_to_okf,
                              last_sync_at, last_sync_status, last_sync_error,
                              created_at, updated_at
                    """
                ),
                params,
            )
            row = res.mappings().first()
            return dict(row) if row else None

    async def delete_source(self, name: str) -> bool:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(f"DELETE FROM {WEBDAV_SOURCES_TABLE} WHERE name = :name"),
                {"name": name},
            )
            return int(getattr(res, "rowcount", 0) or 0) > 0

    async def update_sync_status(
        self,
        name: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            if status == "running":
                await conn.execute(
                    text(
                        f"""
                        UPDATE {WEBDAV_SOURCES_TABLE}
                        SET last_sync_status = :status, updated_at = NOW()
                        WHERE name = :name
                        """
                    ),
                    {"name": name, "status": status},
                )
            else:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {WEBDAV_SOURCES_TABLE}
                        SET last_sync_at = NOW(), last_sync_status = :status,
                            last_sync_error = :error, updated_at = NOW()
                        WHERE name = :name
                        """
                    ),
                    {"name": name, "status": status, "error": error},
                )


class WebDAVFileMetaStore:
    """Store ETag and metadata for incremental WebDAV syncing."""

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
                    CREATE TABLE IF NOT EXISTS {WEBDAV_SOURCE_FILES_TABLE} (
                        source_name TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        etag TEXT NOT NULL,
                        content_length BIGINT,
                        last_modified TIMESTAMPTZ,
                        indexed_at TIMESTAMPTZ,
                        index_attempt_source TEXT,
                        PRIMARY KEY (source_name, filename)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {WEBDAV_SOURCE_FILES_TABLE}
                    ADD COLUMN IF NOT EXISTS index_attempt_source TEXT
                    """
                )
            )
        self._tables_ensured = True

    async def list_files(self, source_name: str) -> List[Dict[str, Any]]:
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT filename, etag, content_length, last_modified, indexed_at,
                           index_attempt_source
                    FROM {WEBDAV_SOURCE_FILES_TABLE}
                    WHERE source_name = :source_name
                    ORDER BY filename
                    """
                ),
                {"source_name": source_name},
            )
            return [dict(row) for row in res.mappings()]

    async def get_file(self, source_name: str, filename: str) -> Optional[Dict[str, Any]]:
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT filename, etag, content_length, last_modified, indexed_at,
                           index_attempt_source
                    FROM {WEBDAV_SOURCE_FILES_TABLE}
                    WHERE source_name = :source_name AND filename = :filename
                    """
                ),
                {"source_name": source_name, "filename": filename},
            )
            row = res.mappings().first()
            return dict(row) if row else None

    async def upsert_file(
        self,
        source_name: str,
        filename: str,
        etag: str,
        content_length: Optional[int] = None,
        last_modified: Optional[datetime] = None,
    ) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {WEBDAV_SOURCE_FILES_TABLE}
                        (source_name, filename, etag, content_length, last_modified)
                    VALUES (:source_name, :filename, :etag, :content_length, :last_modified)
                    ON CONFLICT (source_name, filename) DO UPDATE SET
                        etag = EXCLUDED.etag,
                        content_length = EXCLUDED.content_length,
                        last_modified = EXCLUDED.last_modified
                    """
                ),
                {
                    "source_name": source_name,
                    "filename": filename,
                    "etag": etag,
                    "content_length": content_length,
                    "last_modified": last_modified,
                },
            )

    async def mark_indexed(
        self,
        source_name: str,
        filename: str,
        index_attempt_source: Optional[str] = None,
    ) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {WEBDAV_SOURCE_FILES_TABLE}
                    SET indexed_at = NOW(),
                        index_attempt_source = COALESCE(
                            :index_attempt_source, index_attempt_source
                        )
                    WHERE source_name = :source_name AND filename = :filename
                    """
                ),
                {
                    "source_name": source_name,
                    "filename": filename,
                    "index_attempt_source": index_attempt_source,
                },
            )

    async def delete_file(self, source_name: str, filename: str) -> bool:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    DELETE FROM {WEBDAV_SOURCE_FILES_TABLE}
                    WHERE source_name = :source_name AND filename = :filename
                    """
                ),
                {"source_name": source_name, "filename": filename},
            )
            return int(getattr(res, "rowcount", 0) or 0) > 0

    async def delete_source_files(self, source_name: str) -> int:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"DELETE FROM {WEBDAV_SOURCE_FILES_TABLE} WHERE source_name = :source_name"
                ),
                {"source_name": source_name},
            )
            return int(getattr(res, "rowcount", 0) or 0)
