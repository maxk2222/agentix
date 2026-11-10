"""Async LLM provider store backed by Postgres.

Manages multiple OpenAI-compatible upstreams so different assistants can
reference different providers or use a shared default.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.utils.pg_helpers import _quote_identifier

logger = logging.getLogger(__name__)


@dataclass
class LLMProvider:
    id: str
    name: str
    base_url: str
    enabled: bool = True
    is_default: bool = False
    provider_type: str = "openai-compatible"
    timeout_seconds: int = 30
    max_retries: int = 3
    extra_headers: Optional[Dict[str, str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_checked_at: Optional[str] = None
    last_check_status: Optional[str] = None
    supports_native_tool_calls: Optional[bool] = None
    tool_calling_checked_at: Optional[str] = None
    tool_calling_check_error: Optional[str] = None


class LLMProviderStore:
    """Async store for LLM provider definitions."""

    # Table name for provider metadata (secrets are stored separately)
    TABLE = "llm_providers"
    SECRET_TABLE = "llm_provider_secrets"

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
                        base_url TEXT NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        is_default BOOLEAN NOT NULL DEFAULT false,
                        provider_type TEXT NOT NULL DEFAULT 'openai-compatible',
                        timeout_seconds INTEGER NOT NULL DEFAULT 30,
                        max_retries INTEGER NOT NULL DEFAULT 3,
                        extra_headers TEXT,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        last_checked_at timestamptz,
                        last_check_status TEXT,
                        supports_native_tool_calls BOOLEAN,
                        tool_calling_checked_at timestamptz,
                        tool_calling_check_error TEXT
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.SECRET_TABLE} (
                        provider_id TEXT PRIMARY KEY REFERENCES {self.TABLE}(id) ON DELETE CASCADE,
                        api_key TEXT,
                        last_rotated_at timestamptz
                    )
                    """
                )
            )
            # Unique constraint so only one default provider exists
            await conn.execute(
                text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_providers_default
                    ON {self.TABLE} (is_default)
                    WHERE is_default = true;
                    """
                )
            )
            # Migrate existing tables
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {self.TABLE}
                    ADD COLUMN IF NOT EXISTS supports_native_tool_calls BOOLEAN,
                    ADD COLUMN IF NOT EXISTS tool_calling_checked_at timestamptz,
                    ADD COLUMN IF NOT EXISTS tool_calling_check_error TEXT
                    """
                )
            )
            # Per-model tool calling status (replaces provider-wide column for new entries)
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS provider_model_tool_calling (
                        provider_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        supports_native_tool_calls BOOLEAN,
                        checked_at timestamptz,
                        check_error TEXT,
                        PRIMARY KEY (provider_id, model_id)
                    )
                    """
                )
            )
            # Per-model reasoning capabilities from live probes
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS provider_model_reasoning (
                        provider_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        capabilities JSONB,
                        checked_at timestamptz,
                        check_error TEXT,
                        PRIMARY KEY (provider_id, model_id)
                    )
                    """
                )
            )
        self._tables_ensured = True

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list_providers(self) -> List[LLMProvider]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, base_url, enabled, is_default,
                           provider_type, timeout_seconds, max_retries,
                           extra_headers, created_at, updated_at,
                           last_checked_at, last_check_status,
                           supports_native_tool_calls, tool_calling_checked_at, tool_calling_check_error
                    FROM {self.TABLE}
                    ORDER BY name
                    """
                )
            )
            rows = result.mappings().all()
            return [_row_to_provider(row) for row in rows]

    async def get_provider(self, provider_id: str) -> Optional[LLMProvider]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, base_url, enabled, is_default,
                           provider_type, timeout_seconds, max_retries,
                           extra_headers, created_at, updated_at,
                           last_checked_at, last_check_status,
                           supports_native_tool_calls, tool_calling_checked_at, tool_calling_check_error
                    FROM {self.TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": provider_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_provider(row)

    async def create_provider(self, provider: LLMProvider) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            if provider.is_default:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {self.TABLE}
                        SET is_default = false
                        WHERE is_default = true
                        """
                    )
                )
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (
                        id, name, base_url, enabled, is_default,
                        provider_type, timeout_seconds, max_retries,
                        extra_headers, supports_native_tool_calls,
                        created_at, updated_at
                    )
                    VALUES (
                        :id, :name, :base_url, :enabled, :is_default,
                        :provider_type, :timeout_seconds, :max_retries,
                        :extra_headers, :supports_native_tool_calls,
                        now(), now()
                    )
                    """
                ),
                {
                    "id": provider.id,
                    "name": provider.name,
                    "base_url": provider.base_url,
                    "enabled": provider.enabled,
                    "is_default": provider.is_default,
                    "provider_type": provider.provider_type,
                    "timeout_seconds": provider.timeout_seconds,
                    "max_retries": provider.max_retries,
                    "extra_headers": json.dumps(provider.extra_headers)
                    if provider.extra_headers
                    else None,
                    "supports_native_tool_calls": provider.supports_native_tool_calls,
                },
            )

    async def update_provider(self, provider: LLMProvider) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            if provider.is_default:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {self.TABLE}
                        SET is_default = false
                        WHERE is_default = true AND id != :id
                        """
                    ),
                    {"id": provider.id},
                )
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET name = :name,
                        base_url = :base_url,
                        enabled = :enabled,
                        is_default = :is_default,
                        provider_type = :provider_type,
                        timeout_seconds = :timeout_seconds,
                        max_retries = :max_retries,
                        extra_headers = :extra_headers,
                        supports_native_tool_calls = :supports_native_tool_calls,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {
                    "id": provider.id,
                    "name": provider.name,
                    "base_url": provider.base_url,
                    "enabled": provider.enabled,
                    "is_default": provider.is_default,
                    "provider_type": provider.provider_type,
                    "timeout_seconds": provider.timeout_seconds,
                    "max_retries": provider.max_retries,
                    "extra_headers": json.dumps(provider.extra_headers)
                    if provider.extra_headers
                    else None,
                    "supports_native_tool_calls": provider.supports_native_tool_calls,
                },
            )

    async def delete_provider(self, provider_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE id = :id"),
                {"id": provider_id},
            )

    async def fetch_models(self, provider_id: str) -> List[str]:
        """Fetch available model IDs from a provider's /v1/models endpoint."""
        provider = await self.get_provider(provider_id)
        if provider is None:
            return []
        api_key = await self.get_api_key(provider_id)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{provider.base_url}/v1/models", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data", [])
                return [m["id"] for m in models if "id" in m]
        except Exception:
            logger.exception("Failed to fetch models from provider %s", provider_id)
            return []

    # ------------------------------------------------------------------
    # Secrets
    # ------------------------------------------------------------------

    async def get_api_key(self, provider_id: str) -> Optional[str]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"SELECT api_key FROM {self.SECRET_TABLE} WHERE provider_id = :id"
                ),
                {"id": provider_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return row["api_key"]

    async def set_api_key(self, provider_id: str, api_key: Optional[str]) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.SECRET_TABLE} (provider_id, api_key, last_rotated_at)
                    VALUES (:id, :key, now())
                    ON CONFLICT (provider_id) DO UPDATE
                    SET api_key = EXCLUDED.api_key,
                        last_rotated_at = EXCLUDED.last_rotated_at
                    """
                ),
                {"id": provider_id, "key": api_key},
            )

    async def rotate_api_key(self, provider_id: str, new_key: str) -> None:
        await self.set_api_key(provider_id, new_key)

    async def has_api_key(self, provider_id: str) -> bool:
        key = await self.get_api_key(provider_id)
        return key is not None and key != ""

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    async def update_check_status(
        self, provider_id: str, status: str
    ) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET last_checked_at = now(),
                        last_check_status = :status
                    WHERE id = :id
                    """
                ),
                {"id": provider_id, "status": status},
            )

    async def update_tool_calling_status(
        self,
        provider_id: str,
        supports_native: Optional[bool],
        error: Optional[str] = None,
    ) -> None:
        """Deprecated: updates provider-wide status. Use update_model_tool_calling_status instead."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET supports_native_tool_calls = :supports_native,
                        tool_calling_checked_at = now(),
                        tool_calling_check_error = :error
                    WHERE id = :id
                    """
                ),
                {
                    "id": provider_id,
                    "supports_native": supports_native,
                    "error": error,
                },
            )

    async def get_model_tool_calling_status(
        self,
        provider_id: str,
        model_id: str,
    ) -> Optional[bool]:
        """Return cached tool-calling support for a specific (provider, model) pair."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT supports_native_tool_calls
                    FROM provider_model_tool_calling
                    WHERE provider_id = :provider_id AND model_id = :model_id
                    """
                ),
                {"provider_id": provider_id, "model_id": model_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return row["supports_native_tool_calls"]

    async def update_model_tool_calling_status(
        self,
        provider_id: str,
        model_id: str,
        supports_native: Optional[bool],
        error: Optional[str] = None,
    ) -> None:
        """Persist tool-calling support for a specific (provider, model) pair."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO provider_model_tool_calling (
                        provider_id, model_id, supports_native_tool_calls, checked_at, check_error
                    )
                    VALUES (:provider_id, :model_id, :supports_native, now(), :error)
                    ON CONFLICT (provider_id, model_id) DO UPDATE
                    SET supports_native_tool_calls = EXCLUDED.supports_native_tool_calls,
                        checked_at = EXCLUDED.checked_at,
                        check_error = EXCLUDED.check_error
                    """
                ),
                {
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "supports_native": supports_native,
                    "error": error,
                },
            )

    async def get_model_reasoning_status(
        self,
        provider_id: str,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return cached reasoning capabilities for a (provider, model) pair.

        Returns ``None`` when the pair was never probed OR the last probe was
        inconclusive (``capabilities`` NULL) — error rows must not suppress
        retries forever, so callers treat them like a cache miss.
        """
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT capabilities, check_error
                    FROM provider_model_reasoning
                    WHERE provider_id = :provider_id AND model_id = :model_id
                    """
                ),
                {"provider_id": provider_id, "model_id": model_id},
            )
            row = result.mappings().fetchone()
            if row is None or row["capabilities"] is None:
                return None
            return {"capabilities": row["capabilities"], "check_error": row["check_error"]}

    async def update_model_reasoning_status(
        self,
        provider_id: str,
        model_id: str,
        capabilities: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        """Persist reasoning capabilities for a (provider, model) pair."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO provider_model_reasoning (
                        provider_id, model_id, capabilities, checked_at, check_error
                    )
                    VALUES (:provider_id, :model_id, CAST(:capabilities AS jsonb), now(), :error)
                    ON CONFLICT (provider_id, model_id) DO UPDATE
                    SET capabilities = EXCLUDED.capabilities,
                        checked_at = EXCLUDED.checked_at,
                        check_error = EXCLUDED.check_error
                    """
                ),
                {
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "capabilities": json.dumps(capabilities) if capabilities is not None else None,
                    "error": error,
                },
            )

    async def get_default_provider(self) -> Optional[LLMProvider]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, base_url, enabled, is_default,
                           provider_type, timeout_seconds, max_retries,
                           extra_headers, created_at, updated_at,
                           last_checked_at, last_check_status,
                           supports_native_tool_calls, tool_calling_checked_at, tool_calling_check_error
                    FROM {self.TABLE}
                    WHERE is_default = true
                    LIMIT 1
                    """
                )
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_provider(row)

    async def get_enabled_providers(self) -> List[LLMProvider]:
        providers = await self.list_providers()
        return [p for p in providers if p.enabled]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    async def export_providers(self) -> List[Dict[str, Any]]:
        providers = await self.list_providers()
        out: List[Dict[str, Any]] = []
        for p in providers:
            d = asdict(p)
            d["has_api_key"] = await self.has_api_key(p.id)
            # Never include the actual key
            del d["created_at"]
            del d["updated_at"]
            out.append(d)
        return out

    async def get_provider_by_base_url(self, base_url: str) -> Optional[LLMProvider]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, base_url, enabled, is_default,
                           provider_type, timeout_seconds, max_retries,
                           extra_headers, created_at, updated_at,
                           last_checked_at, last_check_status,
                           supports_native_tool_calls, tool_calling_checked_at, tool_calling_check_error
                    FROM {self.TABLE}
                    WHERE base_url = :base_url
                    LIMIT 1
                    """
                ),
                {"base_url": base_url},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_provider(row)

    async def import_providers(self, data: List[Dict[str, Any]]) -> None:
        for item in data:
            provider = LLMProvider(
                id=item["id"],
                name=item["name"],
                base_url=item["base_url"],
                enabled=item.get("enabled", True),
                is_default=item.get("is_default", False),
                provider_type=item.get("provider_type", "openai-compatible"),
                timeout_seconds=item.get("timeout_seconds", 30),
                max_retries=item.get("max_retries", 3),
                extra_headers=item.get("extra_headers"),
                supports_native_tool_calls=item.get("supports_native_tool_calls"),
            )
            if provider.is_default:
                async with self._engine.begin() as conn:
                    await conn.execute(
                        text(
                            f"""
                            UPDATE {self.TABLE}
                            SET is_default = false
                            WHERE is_default = true AND id != :id
                            """
                        ),
                        {"id": provider.id},
                    )
            existing = await self.get_provider(provider.id)
            if existing:
                await self.update_provider(provider)
            else:
                await self.create_provider(provider)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_provider(row) -> LLMProvider:
    extra = row.get("extra_headers")
    return LLMProvider(
        id=row["id"],
        name=row["name"],
        base_url=row["base_url"],
        enabled=bool(row["enabled"]),
        is_default=bool(row["is_default"]),
        provider_type=row["provider_type"],
        timeout_seconds=int(row["timeout_seconds"]),
        max_retries=int(row["max_retries"]),
        extra_headers=json.loads(extra) if extra else None,
        created_at=row["created_at"].isoformat() if row["created_at"] else None,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        last_checked_at=row["last_checked_at"].isoformat() if row["last_checked_at"] else None,
        last_check_status=row["last_check_status"],
        supports_native_tool_calls=row["supports_native_tool_calls"],
        tool_calling_checked_at=row["tool_calling_checked_at"].isoformat() if row["tool_calling_checked_at"] else None,
        tool_calling_check_error=row["tool_calling_check_error"],
    )
