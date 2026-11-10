"""Postgres-backed stores for backup configuration and backup job history.

Follows the same ``_ensure_tables`` pattern as the other ``*_store.py``
modules. The WebDAV password of the backup target is stored Fernet-encrypted
via :mod:`app.secrets`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import BACKUP_CONFIG_TABLE, BACKUP_JOBS_TABLE
from app.secrets import _decrypt_password, _encrypt_password

logger = logging.getLogger(__name__)

_CONFIG_ID = 1


class BackupConfigStore:
    """Single-row configuration for automated database backups."""

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
                    CREATE TABLE IF NOT EXISTS {BACKUP_CONFIG_TABLE} (
                        id INTEGER PRIMARY KEY,
                        enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        schedule_type TEXT NOT NULL DEFAULT 'interval',
                        schedule_minutes INTEGER NOT NULL DEFAULT 1440,
                        schedule_cron TEXT,
                        timezone TEXT NOT NULL DEFAULT 'UTC',
                        webdav_url TEXT,
                        remote_path TEXT NOT NULL DEFAULT '/agentix-backups',
                        username TEXT,
                        password_encrypted TEXT,
                        retention_count INTEGER NOT NULL DEFAULT 10,
                        retention_days INTEGER,
                        next_run_at TIMESTAMPTZ,
                        last_run_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
        self._tables_ensured = True

    async def _select(self, conn) -> Optional[Dict[str, Any]]:
        res = await conn.execute(
            text(f"SELECT * FROM {BACKUP_CONFIG_TABLE} WHERE id = :id"),
            {"id": _CONFIG_ID},
        )
        row = res.mappings().first()
        return dict(row) if row else None

    async def get_config(self) -> Dict[str, Any]:
        """Return the backup config without the WebDAV password."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            row = await self._select(conn)
        if row is None:
            return {
                "enabled": False,
                "schedule_type": "interval",
                "schedule_minutes": 1440,
                "schedule_cron": None,
                "timezone": "UTC",
                "webdav_url": None,
                "remote_path": "/agentix-backups",
                "username": None,
                "password_set": False,
                "retention_count": 10,
                "retention_days": None,
                "next_run_at": None,
                "last_run_at": None,
            }
        row["password_set"] = bool(row.get("password_encrypted"))
        row.pop("password_encrypted", None)
        return row

    async def get_config_internal(self) -> Optional[Dict[str, Any]]:
        """Return the config including the decrypted WebDAV password.

        Internal use only (backup service / connection test). Never expose
        the result through API responses.
        """
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            row = await self._select(conn)
        if row is None:
            return None
        try:
            row["password"] = (
                _decrypt_password(row["password_encrypted"])
                if row.get("password_encrypted")
                else ""
            )
        except Exception as exc:
            logger.warning("Failed to decrypt backup WebDAV password: %s", exc)
            row["password"] = ""
        row.pop("password_encrypted", None)
        return row

    async def save_config(
        self,
        *,
        enabled: Optional[bool] = None,
        schedule_type: Optional[str] = None,
        schedule_minutes: Optional[int] = None,
        schedule_cron: Optional[str] = None,
        timezone_name: Optional[str] = None,
        webdav_url: Optional[str] = None,
        remote_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        retention_count: Optional[int] = None,
        retention_days: Optional[int] = None,
        next_run_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Upsert the single config row. ``password=None`` keeps the stored one."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            existing = await self._select(conn)
            if existing is None:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {BACKUP_CONFIG_TABLE}
                            (id, enabled, schedule_type, schedule_minutes, schedule_cron,
                             timezone, webdav_url, remote_path, username, password_encrypted,
                             retention_count, retention_days, next_run_at, updated_at)
                        VALUES (:id, :enabled, :schedule_type, :schedule_minutes, :schedule_cron,
                                :timezone, :webdav_url, :remote_path, :username, :pw,
                                :retention_count, :retention_days, :next_run_at, NOW())
                        """
                    ),
                    {
                        "id": _CONFIG_ID,
                        "enabled": bool(enabled) if enabled is not None else False,
                        "schedule_type": schedule_type or "interval",
                        "schedule_minutes": schedule_minutes if schedule_minutes is not None else 1440,
                        "schedule_cron": schedule_cron,
                        "timezone": timezone_name or "UTC",
                        "webdav_url": webdav_url,
                        "remote_path": remote_path or "/agentix-backups",
                        "username": username,
                        "pw": _encrypt_password(password) if password else None,
                        "retention_count": retention_count if retention_count is not None else 10,
                        "retention_days": retention_days,
                        "next_run_at": next_run_at,
                    },
                )
            else:
                fields: List[str] = []
                params: Dict[str, Any] = {"id": _CONFIG_ID}
                mapping = {
                    "enabled": enabled,
                    "schedule_type": schedule_type,
                    "schedule_minutes": schedule_minutes,
                    "schedule_cron": schedule_cron,
                    "timezone": timezone_name,
                    "webdav_url": webdav_url,
                    "remote_path": remote_path,
                    "username": username,
                    "retention_count": retention_count,
                    "retention_days": retention_days,
                    "next_run_at": next_run_at,
                }
                for col, val in mapping.items():
                    if val is not None:
                        fields.append(f"{col} = :{col}")
                        params[col] = val
                if password is not None:
                    fields.append("password_encrypted = :pw")
                    params["pw"] = _encrypt_password(password)
                if not fields:
                    return await self.get_config()
                fields.append("updated_at = NOW()")
                await conn.execute(
                    text(
                        f"UPDATE {BACKUP_CONFIG_TABLE} SET {', '.join(fields)} WHERE id = :id"
                    ),
                    params,
                )
        return await self.get_config()

    async def update_next_run(self, next_run_at: Optional[datetime]) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"UPDATE {BACKUP_CONFIG_TABLE} SET next_run_at = :ts, updated_at = NOW() WHERE id = :id"
                ),
                {"ts": next_run_at, "id": _CONFIG_ID},
            )

    async def update_last_run(self, ts: Optional[datetime] = None) -> None:
        await self.ensure_tables()
        ts = ts or datetime.now(timezone.utc)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"UPDATE {BACKUP_CONFIG_TABLE} SET last_run_at = :ts, updated_at = NOW() WHERE id = :id"
                ),
                {"ts": ts, "id": _CONFIG_ID},
            )

    async def is_due(self) -> bool:
        """Return True when automated backups are enabled and the next run is due."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"SELECT 1 FROM {BACKUP_CONFIG_TABLE} "
                    "WHERE id = :id AND enabled AND next_run_at IS NOT NULL AND next_run_at <= NOW()"
                ),
                {"id": _CONFIG_ID},
            )
            return res.first() is not None


class BackupJobStore:
    """History/audit trail of backup and restore operations."""

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
                    CREATE TABLE IF NOT EXISTS {BACKUP_JOBS_TABLE} (
                        id SERIAL PRIMARY KEY,
                        kind TEXT NOT NULL DEFAULT 'manual',
                        status TEXT NOT NULL DEFAULT 'running',
                        filename TEXT,
                        remote_path TEXT,
                        size_bytes BIGINT,
                        sha256_plain TEXT,
                        sha256_enc TEXT,
                        duration_ms INTEGER,
                        error TEXT,
                        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        finished_at TIMESTAMPTZ,
                        triggered_by TEXT
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {BACKUP_JOBS_TABLE}_started_idx "
                    f"ON {BACKUP_JOBS_TABLE} (started_at DESC)"
                )
            )
        self._tables_ensured = True

    async def create_job(self, kind: str, triggered_by: Optional[str] = None) -> int:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"INSERT INTO {BACKUP_JOBS_TABLE} (kind, status, triggered_by, started_at) "
                    "VALUES (:kind, 'running', :by, NOW()) RETURNING id"
                ),
                {"kind": kind, "by": triggered_by},
            )
            return int(res.scalar())

    async def finish_job(
        self,
        job_id: int,
        status: str,
        *,
        filename: Optional[str] = None,
        remote_path: Optional[str] = None,
        size_bytes: Optional[int] = None,
        sha256_plain: Optional[str] = None,
        sha256_enc: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {BACKUP_JOBS_TABLE}
                    SET status = :status, filename = :filename, remote_path = :remote_path,
                        size_bytes = :size_bytes, sha256_plain = :sha256_plain,
                        sha256_enc = :sha256_enc, duration_ms = :duration_ms,
                        error = :error, finished_at = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": job_id,
                    "status": status,
                    "filename": filename,
                    "remote_path": remote_path,
                    "size_bytes": size_bytes,
                    "sha256_plain": sha256_plain,
                    "sha256_enc": sha256_enc,
                    "duration_ms": duration_ms,
                    "error": error,
                },
            )

    async def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(f"SELECT * FROM {BACKUP_JOBS_TABLE} WHERE id = :id"),
                {"id": job_id},
            )
            row = res.mappings().first()
            return dict(row) if row else None

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"SELECT * FROM {BACKUP_JOBS_TABLE} "
                    "ORDER BY started_at DESC LIMIT :limit OFFSET :offset"
                ),
                {"limit": limit, "offset": offset},
            )
            return [dict(r) for r in res.mappings()]

    async def summary(self) -> Dict[str, Any]:
        """Last successful/failed backup timestamps for the dashboard header."""
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT
                        MAX(CASE WHEN status = 'success' AND kind IN ('manual','scheduled')
                                 THEN finished_at END) AS last_success_at,
                        MAX(CASE WHEN status = 'error'
                                 THEN finished_at END) AS last_error_at,
                        COUNT(*) AS total_jobs
                    FROM {BACKUP_JOBS_TABLE}
                    """
                )
            )
            row = res.mappings().first()
            return dict(row) if row else {}
