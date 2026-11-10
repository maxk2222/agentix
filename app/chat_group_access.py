"""Shared access resolution for public-chat model gating.

Precedence (documented in the groups plan):
  1. ``pct`` token carrying ``groups`` → union of those groups' allowed_models
  2. Session user with ≥1 group membership → union of their groups
  3. ``pct`` token ``role`` (legacy) / session user role → role allowlist
  4. no context → ``None`` (unrestricted / handled by api_accessible gate)

Union semantics on the store: a group with ``allowed_models is None`` is
unrestricted; ``[]`` contributes nothing. The result here: ``None`` =
unrestricted, a set = the effective allowlist (possibly empty = no access).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from fastapi import Request

import logging

from app.chat_role_assistants_store import ChatRoleAssistantsStore

logger = logging.getLogger(__name__)


def _app_state(request: Request) -> Any:
    return getattr(getattr(request, "app", None), "state", None)


def get_groups_store(request: Request):
    """Shared ChatGroupsStore on app.state (lazy, None without engine)."""
    state = _app_state(request)
    if state is None:
        return None
    store = getattr(state, "_chat_groups_store", None)
    if store is not None:
        return store
    rag_application = getattr(state, "rag_application", None)
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is None or not hasattr(engine, "begin"):
        return None
    from app.chat_groups_store import ChatGroupsStore

    store = ChatGroupsStore(engine)
    state._chat_groups_store = store
    return store


def get_role_store(request: Request):
    """Shared ChatRoleAssistantsStore on app.state (lazy, None without engine)."""
    state = _app_state(request)
    if state is None:
        return None
    store = getattr(state, "_chat_role_assistants_store", None)
    if store is not None:
        return store
    rag_application = getattr(state, "rag_application", None)
    engine = getattr(rag_application, "engine", None) or getattr(rag_application, "_engine", None)
    if engine is None:
        return None
    store = ChatRoleAssistantsStore(engine)
    state._chat_role_assistants_store = store
    return store


async def resolve_role_allowlist(request: Request, role: str) -> Optional[Set[str]]:
    """Allowlist configured for a public-chat role, or None (unrestricted)."""
    store = get_role_store(request)
    if store is None:
        return None
    names = await store.get_assistants_for_role(role)
    return set(names) if names else None


async def resolve_groups_allowlist(
    request: Request, group_ids: List[str]
) -> Optional[Set[str]]:
    """Union of ``allowed_models`` over the given group ids.

    ``None`` = unrestricted (any group without a list, or empty id list);
    an empty set = the groups grant access to nothing. A caller that
    explicitly presents group scope must never degrade to unrestricted —
    when the store cannot answer, the result is an empty set (deny all).
    """
    if not group_ids:
        return None
    store = get_groups_store(request)
    if store is None:
        return set()
    try:
        return await store.effective_allowed_models(group_ids=group_ids)
    except Exception:
        logger.warning("chat-groups allowlist resolution failed", exc_info=True)
        return set()


async def resolve_effective_allowlist(
    request: Request,
    *,
    pct_data: Optional[Dict[str, Any]],
    chat_user: Optional[Dict[str, Any]],
) -> Optional[Set[str]]:
    """Effective model allowlist for a public-chat caller (see module doc)."""
    pct_groups = (pct_data or {}).get("groups") or []
    if pct_groups:
        return await resolve_groups_allowlist(request, pct_groups)

    store = get_groups_store(request)
    if chat_user is not None and store is not None:
        try:
            groups = await store.groups_for_user(chat_user["id"])
            if groups:
                return await store.effective_allowed_models(user_id=chat_user["id"])
        except Exception:
            # A broken groups layer must not lock users out — degrade to the
            # legacy role mapping below.
            logger.warning("chat-groups lookup failed", exc_info=True)
    role = chat_user["role"] if chat_user else (pct_data or {}).get("role")
    if role:
        return await resolve_role_allowlist(request, role)
    return None
