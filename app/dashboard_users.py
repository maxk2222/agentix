"""Async dashboard user store backed by Postgres.

Manages multi-user access to the admin dashboard with role-based permissions.
"""
from __future__ import annotations

import bcrypt
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Role-to-permissions mapping
_PERMISSIONS: Dict[str, List[str]] = {
    "viewer": [
        "dashboard:view",
        "metrics:read",
        "vectorstore:read",
        "conversations:read",
        "conversations:export",
        "instructions:read",
        "knowledge:read",
        "runtime_config:read",
        "providers:read",
        "api_keys:read",
        "tools:read",
        "triggers:read",
        "agent_chains:read",
    ],
    "operator": [
        "dashboard:view",
        "metrics:read",
        "vectorstore:read",
        "conversations:read",
        "conversations:export",
        "instructions:read",
        "knowledge:read",
        "runtime_config:read",
        "providers:read",
        "api_keys:read",
        "tools:read",
        "triggers:read",
        "agent_chains:read",
    ],
    "editor": [
        "dashboard:view",
        "metrics:read",
        "vectorstore:read",
        "conversations:read",
        "conversations:export",
        "instructions:read",
        "instructions:write",
        "instructions:rollback",
        "instructions:apply",
        "knowledge:read",
        "knowledge:write",
        "runtime_config:read",
        "providers:read",
        "api_keys:read",
        "tools:read",
        "tools:write",
        "triggers:read",
        "triggers:write",
        "triggers:run",
        "agent_chains:read",
        "agent_chains:write",
        "backup:read",
    ],
    "admin": [
        "dashboard:view",
        "metrics:read",
        "vectorstore:read",
        "conversations:read",
        "conversations:export",
        "instructions:read",
        "instructions:write",
        "instructions:rollback",
        "instructions:apply",
        "knowledge:read",
        "knowledge:write",
        "knowledge:delete",
        "runtime_config:read",
        "runtime_config:write",
        "runtime_config:apply",
        "runtime_config:import_export",
        "providers:read",
        "providers:write",
        "providers:secrets",
        "providers:check",
        "providers:import_export",
        "api_keys:read",
        "api_keys:write",
        "users:read",
        "users:write",
        "audit:read",
        "tools:read",
        "tools:write",
        "triggers:read",
        "triggers:write",
        "triggers:run",
        "agent_chains:read",
        "agent_chains:write",
        "backup:read",
        "backup:write",
        "backup:restore",
        "widget_defaults:write",
        "branding:write",
    ],
    "owner": [
        "dashboard:view",
        "metrics:read",
        "vectorstore:read",
        "conversations:read",
        "conversations:export",
        "instructions:read",
        "instructions:write",
        "instructions:rollback",
        "instructions:apply",
        "knowledge:read",
        "knowledge:write",
        "knowledge:delete",
        "runtime_config:read",
        "runtime_config:write",
        "runtime_config:apply",
        "runtime_config:import_export",
        "providers:read",
        "providers:write",
        "providers:secrets",
        "providers:check",
        "providers:import_export",
        "api_keys:read",
        "api_keys:write",
        "users:read",
        "users:write",
        "audit:read",
        "tools:read",
        "tools:write",
        "triggers:read",
        "triggers:write",
        "triggers:run",
        "agent_chains:read",
        "agent_chains:write",
        "backup:read",
        "backup:write",
        "backup:restore",
        "widget_defaults:write",
        "branding:write",
    ],
}


@dataclass
class DashboardUser:
    id: str
    username: str
    display_name: str
    password_hash: str
    role: str
    is_active: bool = True
    session_version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


def permissions_for_role(role: str) -> List[str]:
    return _PERMISSIONS.get(role, [])


def has_permission(role: str, permission: str) -> bool:
    return permission in _PERMISSIONS.get(role, [])


class DashboardUserStore:
    """Async store for dashboard users."""

    TABLE = "dashboard_users"

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
                        username TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'viewer',
                        is_active BOOLEAN NOT NULL DEFAULT true,
                        session_version INTEGER NOT NULL DEFAULT 1,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        last_login_at timestamptz
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_dashboard_users_username
                    ON {self.TABLE} (username)
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_dashboard_users_role
                    ON {self.TABLE} (role)
                    """
                )
            )
        self._tables_ensured = True

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap_env_admin(self, config) -> None:
        """Create an initial owner from env credentials if no active owner exists."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id FROM {self.TABLE}
                    WHERE is_active = true AND role IN ('owner', 'admin')
                    LIMIT 1
                    """
                )
            )
            if result.scalar() is not None:
                return

        username = getattr(config, "dashboard_admin_username", "admin")
        password = getattr(config, "dashboard_admin_password", "")
        if not password:
            logger.warning("Dashboard admin password not set; skipping bootstrap.")
            return

        password_hash = self._hash_password(password)
        user = DashboardUser(
            id="env-admin",
            username=username,
            display_name="Environment Admin",
            password_hash=password_hash,
            role="owner",
            is_active=True,
        )
        await self.create_user(user)
        logger.info("Bootstrapped dashboard owner from environment config.")

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against a bcrypt hash."""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def authenticate(self, username: str, password: str) -> Optional[DashboardUser]:
        """Verify credentials and return the user if valid and active."""
        await self.ensure_tables()
        user = await self.get_by_username(username)
        if user is None or not user.is_active:
            return None
        if not self._verify_password(password, user.password_hash):
            return None
        return user

    async def get_user(self, user_id: str) -> Optional[DashboardUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, session_version, created_at, updated_at, last_login_at
                    FROM {self.TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": user_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    async def get_by_username(self, username: str) -> Optional[DashboardUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, session_version, created_at, updated_at, last_login_at
                    FROM {self.TABLE}
                    WHERE username = :username
                    """
                ),
                {"username": username},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    async def list_users(self) -> List[DashboardUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, session_version, created_at, updated_at, last_login_at
                    FROM {self.TABLE}
                    ORDER BY username
                    """
                )
            )
            rows = result.mappings().all()
            return [_row_to_user(row) for row in rows]

    async def create_user(self, user: DashboardUser) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (
                        id, username, display_name, password_hash, role,
                        is_active, session_version, created_at, updated_at
                    )
                    VALUES (
                        :id, :username, :display_name, :password_hash, :role,
                        :is_active, :session_version, now(), now()
                    )
                    """
                ),
                {
                    "id": user.id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "password_hash": user.password_hash,
                    "role": user.role,
                    "is_active": user.is_active,
                    "session_version": user.session_version,
                },
            )

    async def update_user(self, user: DashboardUser) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET username = :username,
                        display_name = :display_name,
                        role = :role,
                        is_active = :is_active,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {
                    "id": user.id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "role": user.role,
                    "is_active": user.is_active,
                },
            )

    async def set_password(self, user_id: str, password: str) -> None:
        await self.ensure_tables()
        password_hash = self._hash_password(password)
        # Increment session_version to invalidate existing sessions
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET password_hash = :password_hash,
                        session_version = session_version + 1,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": user_id, "password_hash": password_hash},
            )

    async def set_active(self, user_id: str, is_active: bool) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET is_active = :is_active,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": user_id, "is_active": is_active},
            )

    async def bump_session_version(self, user_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET session_version = session_version + 1,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": user_id},
            )

    async def record_login(self, user_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET last_login_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": user_id},
            )

    async def delete_user(self, user_id: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self.TABLE} WHERE id = :id"),
                {"id": user_id},
            )

    async def count_active_admins(self) -> int:
        """Count active users with admin or owner role."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) FROM {self.TABLE}
                    WHERE is_active = true AND role IN ('owner', 'admin')
                    """
                )
            )
            return int(result.scalar() or 0)

    async def count_active_owners(self) -> int:
        """Count active users with owner role."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) FROM {self.TABLE}
                    WHERE is_active = true AND role = 'owner'
                    """
                )
            )
            return int(result.scalar() or 0)


def _row_to_user(row) -> DashboardUser:
    return DashboardUser(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        password_hash=row["password_hash"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        session_version=int(row["session_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_login_at=row["last_login_at"],
    )
