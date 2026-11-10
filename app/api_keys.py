"""Async application API key store backed by Postgres.

Manages multiple API keys for the chat endpoint, each with an allowed
model whitelist so different consumers can be restricted to specific
assistants.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


@dataclass
class AppApiKey:
    id: str
    name: str
    api_key_hash: str
    allowed_models: List[str] = field(default_factory=list)
    enabled: bool = True
    rate_limit_rpm: int = 0
    rate_limit_rpd: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AppApiKeyStore:
    """Async store for application-level API keys."""

    TABLE = "app_api_keys"

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
                        name TEXT NOT NULL,
                        api_key_hash TEXT NOT NULL,
                        allowed_models TEXT[] NOT NULL DEFAULT '{{}}',
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {self.TABLE}
                        ADD COLUMN IF NOT EXISTS rate_limit_rpm INTEGER NOT NULL DEFAULT 0,
                        ADD COLUMN IF NOT EXISTS rate_limit_rpd INTEGER NOT NULL DEFAULT 0
                    """
                )
            )
        self._tables_ensured = True

    # ------------------------------------------------------------------
    # Key generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key() -> str:
        """Generate a new random API key (raw value)."""
        return "ak_" + secrets.token_urlsafe(32)

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list_keys(self) -> List[AppApiKey]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, api_key_hash, allowed_models,
                           enabled, rate_limit_rpm, rate_limit_rpd,
                           created_at, updated_at
                    FROM {self.TABLE}
                    ORDER BY name
                    """
                )
            )
            rows = result.mappings().all()
            return [_row_to_api_key(row) for row in rows]

    async def get_key(self, key_id: str) -> Optional[AppApiKey]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, api_key_hash, allowed_models,
                           enabled, rate_limit_rpm, rate_limit_rpd,
                           created_at, updated_at
                    FROM {self.TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": key_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_api_key(row)

    async def create_key(self, key: AppApiKey) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (
                        id, name, api_key_hash, allowed_models,
                        enabled, rate_limit_rpm, rate_limit_rpd,
                        created_at, updated_at
                    )
                    VALUES (
                        :id, :name, :api_key_hash, :allowed_models,
                        :enabled, :rate_limit_rpm, :rate_limit_rpd,
                        now(), now()
                    )
                    """
                ),
                {
                    "id": key.id,
                    "name": key.name,
                    "api_key_hash": key.api_key_hash,
                    "allowed_models": key.allowed_models,
                    "enabled": key.enabled,
                    "rate_limit_rpm": key.rate_limit_rpm,
                    "rate_limit_rpd": key.rate_limit_rpd,
                },
            )

    async def update_key(self, key: AppApiKey) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET name = :name,
                        allowed_models = :allowed_models,
                        enabled = :enabled,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {
                    "id": key.id,
                    "name": key.name,
                    "allowed_models": key.allowed_models,
                    "enabled": key.enabled,
                    "rate_limit_rpm": key.rate_limit_rpm,
                    "rate_limit_rpd": key.rate_limit_rpd,
                },
            )

    async def delete_key(self, key_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE id = :id"),
                {"id": key_id},
            )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify_key(self, raw_key: str) -> Optional[AppApiKey]:
        """Verify a raw API key and return the matching record if valid."""
        await self.ensure_tables()
        hashed = self.hash_key(raw_key)
        keys = await self.list_keys()
        for k in keys:
            if k.enabled and hmac.compare_digest(hashed, k.api_key_hash):
                return k
        return None

    async def is_model_allowed(self, raw_key: str, model_name: str) -> bool:
        key = await self.verify_key(raw_key)
        if key is None:
            return False
        if not key.allowed_models:
            return True
        return model_name in key.allowed_models

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    async def migrate_legacy_key(self, legacy_key: str) -> None:
        """If no keys exist, migrate the legacy single API_KEY into the store."""
        await self.ensure_tables()
        existing = await self.list_keys()
        if existing:
            return
        if not legacy_key or not legacy_key.strip():
            return
        key = AppApiKey(
            id="legacy",
            name="Environment API Key",
            api_key_hash=self.hash_key(legacy_key.strip()),
            allowed_models=[],
            enabled=True,
        )
        await self.create_key(key)
        logger.info("Migrated legacy API_KEY into app_api_keys table")


def _row_to_api_key(row) -> AppApiKey:
    models = row.get("allowed_models") or []
    return AppApiKey(
        id=row["id"],
        name=row["name"],
        api_key_hash=row["api_key_hash"],
        allowed_models=list(models) if models else [],
        enabled=bool(row["enabled"]),
        rate_limit_rpm=int(row.get("rate_limit_rpm") or 0),
        rate_limit_rpd=int(row.get("rate_limit_rpd") or 0),
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
    )
