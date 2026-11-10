"""Async chat user store backed by Postgres.

Manages self-registered public chat users with role-based assistant access.
Dashboard users can also log in via the public chat auth flow.
"""
from __future__ import annotations

import bcrypt
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Role mapping from dashboard roles to chat roles
DASHBOARD_TO_CHAT_ROLE: Dict[str, str] = {
    "viewer": "standard",
    "operator": "pro",
    "editor": "pro",
    "admin": "admin",
    "owner": "admin",
}


@dataclass
class ChatUser:
    id: str
    username: str
    display_name: str
    password_hash: str
    role: str
    is_active: bool = True
    dashboard_user_id: Optional[str] = None
    email: Optional[str] = None
    session_version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class ChatUserStore:
    """Async store for public chat users."""

    TABLE = "chat_users"

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
                        role TEXT NOT NULL DEFAULT 'standard',
                        is_active BOOLEAN NOT NULL DEFAULT true,
                        dashboard_user_id TEXT,
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
                    CREATE INDEX IF NOT EXISTS idx_chat_users_username
                    ON {self.TABLE} (username)
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_chat_users_dashboard_user_id
                    ON {self.TABLE} (dashboard_user_id)
                    """
                )
            )
            await conn.execute(
                text(f"ALTER TABLE {self.TABLE} ADD COLUMN IF NOT EXISTS email TEXT")
            )
        self._tables_ensured = True

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    async def authenticate(self, username: str, password: str) -> Optional[ChatUser]:
        user = await self.get_by_username(username)
        if user is None or not user.is_active:
            return None
        if not self._verify_password(password, user.password_hash):
            return None
        return user

    async def get_user(self, user_id: str) -> Optional[ChatUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, dashboard_user_id, session_version,
                           created_at, updated_at, last_login_at, email
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

    async def get_by_username(self, username: str) -> Optional[ChatUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, dashboard_user_id, session_version,
                           created_at, updated_at, last_login_at, email
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

    async def get_by_dashboard_user_id(self, dashboard_user_id: str) -> Optional[ChatUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, dashboard_user_id, session_version,
                           created_at, updated_at, last_login_at, email
                    FROM {self.TABLE}
                    WHERE dashboard_user_id = :dashboard_user_id
                    """
                ),
                {"dashboard_user_id": dashboard_user_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_user(row)

    async def list_users(self) -> List[ChatUser]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, username, display_name, password_hash, role,
                           is_active, dashboard_user_id, session_version,
                           created_at, updated_at, last_login_at, email
                    FROM {self.TABLE}
                    ORDER BY username
                    """
                )
            )
            rows = result.mappings().all()
            return [_row_to_user(row) for row in rows]

    async def create_user(self, user: ChatUser) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.TABLE} (
                        id, username, display_name, password_hash, role,
                        is_active, dashboard_user_id, session_version, email,
                        created_at, updated_at
                    )
                    VALUES (
                        :id, :username, :display_name, :password_hash, :role,
                        :is_active, :dashboard_user_id, :session_version, :email,
                        now(), now()
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
                    "dashboard_user_id": user.dashboard_user_id,
                    "session_version": user.session_version,
                    "email": user.email,
                },
            )

    async def update_user(self, user: ChatUser) -> None:
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
                        email = :email,
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
                    "email": user.email,
                },
            )

    async def set_password(self, user_id: str, password: str) -> None:
        await self.ensure_tables()
        password_hash = self._hash_password(password)
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

    async def set_role(self, user_id: str, role: str) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self.TABLE}
                    SET role = :role,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": user_id, "role": role},
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


def _row_to_user(row) -> ChatUser:
    return ChatUser(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        password_hash=row["password_hash"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        dashboard_user_id=row["dashboard_user_id"],
        session_version=int(row["session_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_login_at=row["last_login_at"],
        email=row.get("email"),
    )
