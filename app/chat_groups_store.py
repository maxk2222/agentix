"""Postgres-backed store for public-chat user groups.

Groups are an additive m:n layer on top of the legacy ``role`` string:
a user may belong to any number of groups; each group grants an
``allowed_models`` list (NULL = unrestricted, [] = grants nothing).
The effective allowlist of a user is the union of their groups' lists;
users without any membership fall back to the role mapping.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


def normalize_domain(raw: str) -> str:
    """Normalize a domain entry: lowercase, strip whitespace and a leading '@'."""
    return str(raw or "").strip().lower().lstrip("@")


def extract_domain(value: str) -> str:
    """Domain part of an email-like value ('' when it has no '@')."""
    v = str(value or "").strip().lower()
    if "@" not in v:
        return ""
    return v.rsplit("@", 1)[1]


@dataclass
class ChatGroup:
    id: str
    name: str
    description: str = ""
    color: str = ""
    allowed_models: Optional[List[str]] = None  # None = unrestricted
    auto_domains: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    member_count: int = 0


@dataclass
class ChatGroupMembership:
    user_id: str
    group_id: str
    assigned_via: str = "manual"  # 'manual' | 'domain'


class ChatGroupsStore:
    """CRUD + membership + effective-allowlist resolution for chat groups."""

    GROUPS_TABLE = "chat_groups"
    MEMBERS_TABLE = "chat_user_groups"

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
                    CREATE TABLE IF NOT EXISTS {self.GROUPS_TABLE} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL DEFAULT '',
                        color TEXT NOT NULL DEFAULT '',
                        allowed_models JSONB,
                        auto_domains JSONB NOT NULL DEFAULT '[]',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.MEMBERS_TABLE} (
                        user_id TEXT NOT NULL REFERENCES chat_users(id) ON DELETE CASCADE,
                        group_id TEXT NOT NULL REFERENCES chat_groups(id) ON DELETE CASCADE,
                        assigned_via TEXT NOT NULL DEFAULT 'manual',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (user_id, group_id)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_chat_user_groups_group
                    ON {self.MEMBERS_TABLE} (group_id)
                    """
                )
            )
            # chat_users.email is additive — idempotent for existing databases.
            await conn.execute(
                text("ALTER TABLE chat_users ADD COLUMN IF NOT EXISTS email TEXT")
            )
        self._tables_ensured = True

    # ------------------------------------------------------------------
    # group CRUD
    # ------------------------------------------------------------------

    async def create_group(self, group: ChatGroup) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.GROUPS_TABLE}
                        (id, name, description, color, allowed_models, auto_domains,
                         created_at, updated_at)
                    VALUES
                        (:id, :name, :description, :color,
                         CAST(:allowed_models AS jsonb), CAST(:auto_domains AS jsonb),
                         now(), now())
                    """
                ),
                {
                    "id": group.id,
                    "name": group.name,
                    "description": group.description,
                    "color": group.color,
                    "allowed_models": _dumps_or_none(group.allowed_models),
                    "auto_domains": _dumps(group.auto_domains or []),
                },
            )

    async def update_group(self, group: ChatGroup) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE {self.GROUPS_TABLE}
                    SET name = :name,
                        description = :description,
                        color = :color,
                        allowed_models = CAST(:allowed_models AS jsonb),
                        auto_domains = CAST(:auto_domains AS jsonb),
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {
                    "id": group.id,
                    "name": group.name,
                    "description": group.description,
                    "color": group.color,
                    "allowed_models": _dumps_or_none(group.allowed_models),
                    "auto_domains": _dumps(group.auto_domains or []),
                },
            )
            return (result.rowcount or 0) > 0

    async def delete_group(self, group_id: str) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f"DELETE FROM {self.GROUPS_TABLE} WHERE id = :id"),
                {"id": group_id},
            )
            return (result.rowcount or 0) > 0

    async def get_group(self, group_id: str) -> Optional[ChatGroup]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, color, allowed_models, auto_domains,
                           created_at, updated_at
                    FROM {self.GROUPS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": group_id},
            )
            row = result.mappings().fetchone()
            return _row_to_group(row) if row else None

    async def get_group_by_name(self, name: str) -> Optional[ChatGroup]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, color, allowed_models, auto_domains,
                           created_at, updated_at
                    FROM {self.GROUPS_TABLE}
                    WHERE name = :name
                    """
                ),
                {"name": name},
            )
            row = result.mappings().fetchone()
            return _row_to_group(row) if row else None

    async def list_groups(self) -> List[ChatGroup]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT g.id, g.name, g.description, g.color, g.allowed_models,
                           g.auto_domains, g.created_at, g.updated_at,
                           (SELECT COUNT(*) FROM {self.MEMBERS_TABLE} m
                            WHERE m.group_id = g.id) AS member_count
                    FROM {self.GROUPS_TABLE} g
                    ORDER BY g.name
                    """
                )
            )
            return [_row_to_group(row) for row in result.mappings().all()]

    # ------------------------------------------------------------------
    # membership
    # ------------------------------------------------------------------

    async def set_user_groups(
        self, user_id: str, group_ids: List[str], assigned_via: str = "manual"
    ) -> None:
        """Replace a user's memberships (manual assignments only — domain
        auto-assignments are additive and use ``add_user_to_group``)."""
        await self.ensure_tables()
        keep = {str(g) for g in group_ids}
        async with self._engine.begin() as conn:
            rows = await conn.execute(
                text(f"SELECT group_id FROM {self.MEMBERS_TABLE} WHERE user_id = :u"),
                {"u": user_id},
            )
            existing = {r["group_id"] for r in rows.mappings().all()}
            for gid in existing - keep:
                await conn.execute(
                    text(
                        f"DELETE FROM {self.MEMBERS_TABLE} "
                        "WHERE user_id = :u AND group_id = :g"
                    ),
                    {"u": user_id, "g": gid},
                )
            for gid in keep - existing:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self.MEMBERS_TABLE} (user_id, group_id, assigned_via)
                        VALUES (:u, :g, :via)
                        """
                    ),
                    {"u": user_id, "g": gid, "via": assigned_via},
                )

    async def add_user_to_group(
        self, user_id: str, group_id: str, assigned_via: str = "manual"
    ) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self.MEMBERS_TABLE} (user_id, group_id, assigned_via)
                    VALUES (:u, :g, :via)
                    ON CONFLICT (user_id, group_id) DO NOTHING
                    """
                ),
                {"u": user_id, "g": group_id, "via": assigned_via},
            )

    async def remove_user_from_group(self, group_id: str, user_id: str) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"DELETE FROM {self.MEMBERS_TABLE} "
                    "WHERE group_id = :g AND user_id = :u"
                ),
                {"g": group_id, "u": user_id},
            )
            return (result.rowcount or 0) > 0

    async def groups_for_user(self, user_id: str) -> List[ChatGroup]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT g.id, g.name, g.description, g.color, g.allowed_models,
                           g.auto_domains, g.created_at, g.updated_at
                    FROM {self.GROUPS_TABLE} g
                    JOIN {self.MEMBERS_TABLE} m ON m.group_id = g.id
                    WHERE m.user_id = :u
                    ORDER BY g.name
                    """
                ),
                {"u": user_id},
            )
            return [_row_to_group(row) for row in result.mappings().all()]

    async def memberships_for_user(self, user_id: str) -> List[ChatGroupMembership]:
        """Membership rows incl. assigned_via for the members UI."""
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT user_id, group_id, assigned_via
                    FROM {self.MEMBERS_TABLE}
                    WHERE user_id = :u
                    """
                ),
                {"u": user_id},
            )
            return [
                ChatGroupMembership(
                    user_id=row["user_id"],
                    group_id=row["group_id"],
                    assigned_via=row["assigned_via"],
                )
                for row in result.mappings().all()
            ]

    async def members_of(self, group_id: str) -> List[ChatGroupMembership]:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT user_id, group_id, assigned_via
                    FROM {self.MEMBERS_TABLE}
                    WHERE group_id = :g
                    """
                ),
                {"g": group_id},
            )
            return [
                ChatGroupMembership(
                    user_id=row["user_id"],
                    group_id=row["group_id"],
                    assigned_via=row["assigned_via"],
                )
                for row in result.mappings().all()
            ]

    async def group_ids_for_users(self, user_ids: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """user_id -> [{id, name, color, assigned_via}] for list views."""
        await self.ensure_tables()
        if not user_ids:
            return {}
        out: Dict[str, List[Dict[str, str]]] = {u: [] for u in user_ids}
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT m.user_id, m.assigned_via, g.id, g.name, g.color
                    FROM {self.MEMBERS_TABLE} m
                    JOIN {self.GROUPS_TABLE} g ON g.id = m.group_id
                    WHERE m.user_id = ANY(:uids)
                    ORDER BY g.name
                    """
                ),
                {"uids": list(user_ids)},
            )
            for row in result.mappings().all():
                out.setdefault(row["user_id"], []).append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "color": row["color"],
                        "assigned_via": row["assigned_via"],
                    }
                )
        return out

    # ------------------------------------------------------------------
    # effective allowlist + domain matching
    # ------------------------------------------------------------------

    async def effective_allowed_models(
        self, group_ids: Optional[List[str]] = None, *, user_id: Optional[str] = None
    ) -> Optional[Set[str]]:
        """Union of ``allowed_models`` over the given groups / the user's
        memberships. ``None`` = unrestricted; an empty set = access to nothing.
        """
        await self.ensure_tables()
        if user_id is not None:
            groups = await self.groups_for_user(user_id)
        else:
            groups = [await self.get_group(g) for g in (group_ids or [])]
            groups = [g for g in groups if g is not None]
        if user_id is not None and not groups:
            return None  # no memberships → caller falls back to role mapping
        union: Set[str] = set()
        for g in groups:
            if g.allowed_models is None:
                return None  # one unrestricted group wins
            union.update(str(m) for m in g.allowed_models)
        return union

    async def groups_for_domain(self, domain: str) -> List[ChatGroup]:
        """Groups whose ``auto_domains`` contains the (normalized) domain."""
        await self.ensure_tables()
        domain = normalize_domain(domain)
        if not domain:
            return []
        matches: List[ChatGroup] = []
        for g in await self.list_groups():
            if domain in {normalize_domain(d) for d in (g.auto_domains or [])}:
                matches.append(g)
        return matches

    async def auto_assign_user(self, user) -> List[str]:
        """Add the user to every group matching their email/username domain.

        Returns the group ids that were newly assigned. Additive — never
        removes existing memberships.
        """
        await self.ensure_tables()
        domain = extract_domain(getattr(user, "email", None) or "") or extract_domain(
            getattr(user, "username", "") or ""
        )
        if not domain:
            return []
        before = {m.group_id for m in await self.memberships_for_user(user.id)}
        assigned: List[str] = []
        for group in await self.groups_for_domain(domain):
            await self.add_user_to_group(user.id, group.id, assigned_via="domain")
            if group.id not in before:
                assigned.append(group.id)
        return assigned

    async def reapply_domain_assignments(self, user_store) -> Dict[str, int]:
        """Re-run domain matching for every chat user (bulk 'Reapply' op).

        Additive and idempotent — only missing memberships are added, manual
        assignments are never touched. Returns ``{"users": n, "assigned": m}``.
        """
        await self.ensure_tables()
        users = await user_store.list_users()
        assigned = 0
        for user in users:
            assigned += len(await self.auto_assign_user(user))
        return {"users": len(users), "assigned": assigned}


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _dumps_or_none(value: Optional[Any]) -> Optional[str]:
    return None if value is None else _dumps(value)


def _loads(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return None


def _row_to_group(row) -> ChatGroup:
    return ChatGroup(
        id=row["id"],
        name=row["name"],
        description=row.get("description") or "",
        color=row.get("color") or "",
        allowed_models=_loads(row.get("allowed_models")),
        auto_domains=_loads(row.get("auto_domains")) or [],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        member_count=int(row.get("member_count") or 0),
    )
