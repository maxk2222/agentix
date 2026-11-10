"""Public chat authentication router.

Handles self-registration, login, logout, and current user info for public chat users.
Dashboard users can also authenticate via this router (with role mapping).
"""
from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from agent.config import AppConfig
from app.auth import create_session_cookie, read_session_cookie
from app.chat_users import ChatUser, ChatUserStore, DASHBOARD_TO_CHAT_ROLE
from app.dashboard_users import DashboardUserStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Public Chat Auth"])


class RegisterPayload(BaseModel):
    username: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = None


class LoginPayload(BaseModel):
    username: str
    password: str


class ChatUserInfo(BaseModel):
    id: str
    username: str
    display_name: str
    role: str
    email: Optional[str] = None
    groups: List[str] = []


class MeResponse(BaseModel):
    authenticated: bool
    user: Optional[ChatUserInfo] = None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_chat_user_store(request: Request) -> Optional[ChatUserStore]:
    store = getattr(request.app.state, "_chat_user_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = ChatUserStore(engine)
    request.app.state._chat_user_store = store
    return store


def _get_dashboard_user_store(request: Request) -> Optional[DashboardUserStore]:
    store = getattr(request.app.state, "_dashboard_user_store", None)
    if store is not None:
        return store
    rag_app = getattr(request.app.state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return None
    store = DashboardUserStore(engine)
    request.app.state._dashboard_user_store = store
    return store


async def _auto_assign_groups(request: Request, user) -> None:
    """Domain-based group assignment for a freshly created chat user."""
    try:
        from app.chat_group_access import get_groups_store

        store = get_groups_store(request)
        if store is not None:
            await store.auto_assign_user(user)
    except Exception:
        logger.warning("group auto-assign failed for %s", user.id, exc_info=True)


def _get_config(request: Request) -> Optional[AppConfig]:
    return getattr(getattr(getattr(request, "app", None), "state", None), "app_config", None)


def _create_public_chat_session_cookie(config: AppConfig, user: Dict[str, Any]) -> str:
    secret = config.dashboard_session_secret or ""
    if not secret:
        raise RuntimeError("DASHBOARD_SESSION_SECRET is required for public chat sessions")
    from itsdangerous import URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(secret, salt="public-chat-session")
    payload = {
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "session_version": user.get("session_version", 1),
    }
    return serializer.dumps(payload)


def _read_public_chat_session_cookie(request: Request, config: AppConfig) -> Optional[dict]:
    secret = config.dashboard_session_secret or ""
    if not secret:
        return None
    raw = request.cookies.get("public_chat_session")
    if not raw:
        return None
    from itsdangerous import BadSignature, URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(secret, salt="public-chat-session")
    try:
        data = serializer.loads(raw, max_age=config.dashboard_session_ttl_seconds)
        return data if isinstance(data, dict) else None
    except BadSignature:
        return None


# -- Scoped public-chat access token -----------------------------------------
# Lets the public chat page reach /chat/* endpoints under public-chat rules
# (enabled flag + role allowlists) while anonymous API callers are governed
# solely by ``api_accessible``. Sent as X-Public-Chat-Token header or ?pct=.

_PUBLIC_CHAT_ACCESS_TTL = 12 * 3600
# Admin-minted share links (group-scoped widget links) must outlive a single
# session — a link that dies after 12h is useless for distribution. They are
# marked kind="link" and get a longer TTL (30 days).
_PUBLIC_CHAT_LINK_TTL = 30 * 24 * 3600


def _create_public_chat_access_token(
    config: AppConfig,
    role: str,
    groups: Optional[List[str]] = None,
    link: bool = False,
) -> str:
    """Mint the scoped public-chat token.

    ``role`` is kept for backward compatibility (legacy tokens + role
    allowlists). ``groups`` carries chat-group ids for group-scoped widget
    links — when present it takes precedence over the role in allowlist
    resolution. ``link=True`` marks the token as a long-lived share link
    (kind="link", 30-day TTL instead of the session-scoped 12h).
    """
    secret = config.dashboard_session_secret or ""
    if not secret:
        raise RuntimeError("DASHBOARD_SESSION_SECRET is required for public chat access tokens")
    from itsdangerous import URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(secret, salt="public-chat-access")
    payload: Dict[str, Any] = {"role": role}
    if groups:
        payload["groups"] = [str(g) for g in groups]
    if link:
        payload["kind"] = "link"
    return serializer.dumps(payload)


def _read_public_chat_access_token(
    request: Request, config: Optional[AppConfig]
) -> Optional[Dict[str, Any]]:
    """Decode the scoped token → ``{"role": str|None, "groups": [ids]}``.

    Returns ``None`` for missing/invalid/expired tokens. Legacy tokens that
    only carry ``role`` decode to ``{"role": r, "groups": []}``.
    """
    if config is None:
        return None
    secret = config.dashboard_session_secret or ""
    headers = getattr(request, "headers", None) or {}
    query_params = getattr(request, "query_params", None) or {}
    raw = headers.get("x-public-chat-token") or query_params.get("pct")
    if not raw or not secret:
        return None
    from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(secret, salt="public-chat-access")
    try:
        data = serializer.loads(raw, max_age=_PUBLIC_CHAT_ACCESS_TTL)
        link_token = False
    except SignatureExpired:
        # Long-lived share links carry kind="link" — retry with the extended
        # TTL and reject the token if the claim is absent.
        try:
            data = serializer.loads(raw, max_age=_PUBLIC_CHAT_LINK_TTL)
        except BadSignature:
            return None
        link_token = True
    except BadSignature:
        return None
    if not isinstance(data, dict):
        return None
    if link_token and data.get("kind") != "link":
        return None
    role = data.get("role")
    groups = data.get("groups") or []
    return {
        "role": str(role) if role else None,
        "groups": [str(g) for g in groups if g],
        "kind": data.get("kind"),
    }


async def _get_current_chat_user(request: Request) -> Optional[Dict[str, Any]]:
    config = _get_config(request)
    if not config:
        return None

    # Try public chat session first
    session = _read_public_chat_session_cookie(request, config)
    if session:
        user_id = session.get("user_id")
        if user_id:
            store = _get_chat_user_store(request)
            if store is not None:
                user = await store.get_user(user_id)
                if user is not None and user.is_active:
                    session_version = session.get("session_version", 1)
                    if user.session_version == session_version:
                        return {
                            "id": user.id,
                            "username": user.username,
                            "display_name": user.display_name,
                            "role": user.role,
                            "session_version": user.session_version,
                            "dashboard_user_id": user.dashboard_user_id,
                        }

    # Fall back to dashboard session — resolve through the linked shadow chat
    # user so conversations/deletes stay under ONE identity regardless of
    # which session cookie is present (the login endpoint uses the same upsert).
    dashboard_session = read_session_cookie(request, config)
    if dashboard_session:
        user_id = dashboard_session.get("user_id")
        if user_id:
            dash_store = _get_dashboard_user_store(request)
            if dash_store is not None:
                dash_user = await dash_store.get_user(user_id)
                if dash_user is not None and dash_user.is_active:
                    chat_role = DASHBOARD_TO_CHAT_ROLE.get(dash_user.role, "standard")
                    chat_store = _get_chat_user_store(request)
                    linked = None
                    if chat_store is not None:
                        try:
                            linked = await chat_store.get_by_dashboard_user_id(dash_user.id)
                            if linked is None:
                                linked = ChatUser(
                                    id=f"chat-user-{secrets.token_hex(8)}",
                                    username=dash_user.username,
                                    display_name=dash_user.display_name,
                                    password_hash=dash_user.password_hash,
                                    role=chat_role,
                                    is_active=dash_user.is_active,
                                    dashboard_user_id=dash_user.id,
                                    session_version=dash_user.session_version,
                                )
                                await chat_store.create_user(linked)
                            elif linked.role != chat_role:
                                await chat_store.set_role(linked.id, chat_role)
                                linked = await chat_store.get_user(linked.id) or linked
                        except Exception:
                            linked = None
                    if linked is not None:
                        return {
                            "id": linked.id,
                            "username": linked.username,
                            "display_name": linked.display_name,
                            "role": linked.role,
                            "session_version": linked.session_version,
                            "dashboard_user_id": dash_user.id,
                        }
                    return {
                        "id": dash_user.id,
                        "username": dash_user.username,
                        "display_name": dash_user.display_name,
                        "role": chat_role,
                        "session_version": dash_user.session_version,
                        "dashboard_user_id": dash_user.id,
                    }

    return None


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@router.post("/public-chat/auth/register")
async def public_chat_register(
    request: Request,
    payload: RegisterPayload,
) -> Dict[str, Any]:
    """Self-register a new public chat user."""
    store = _get_chat_user_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    existing = await store.get_by_username(payload.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = ChatUser(
        id=f"chat-user-{secrets.token_hex(8)}",
        username=payload.username.strip(),
        display_name=payload.display_name.strip(),
        email=(payload.email or "").strip() or None,
        password_hash=ChatUserStore._hash_password(payload.password),
        role="standard",
        is_active=True,
    )
    await store.create_user(user)
    await _auto_assign_groups(request, user)
    return {"success": True, "message": "User registered successfully"}


@router.post("/public-chat/auth/login")
async def public_chat_login(
    request: Request,
    response: Response,
    payload: LoginPayload,
) -> Dict[str, Any]:
    """Log in a public chat user. Also accepts dashboard credentials with role mapping."""
    config = _get_config(request)
    if not config:
        raise HTTPException(status_code=503, detail="App config unavailable")

    chat_store = _get_chat_user_store(request)
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    # First try chat_users table
    chat_user = await chat_store.authenticate(payload.username, payload.password)

    if chat_user is None:
        # Fallback to dashboard_users table
        dash_store = _get_dashboard_user_store(request)
        if dash_store is not None:
            dash_user = await dash_store.authenticate(payload.username, payload.password)
            if dash_user is not None:
                # Upsert a shadow chat user for this dashboard user
                existing_chat = await chat_store.get_by_dashboard_user_id(dash_user.id)
                chat_role = DASHBOARD_TO_CHAT_ROLE.get(dash_user.role, "standard")
                if existing_chat is not None:
                    # Update role if it changed
                    if existing_chat.role != chat_role:
                        await chat_store.set_role(existing_chat.id, chat_role)
                    chat_user = await chat_store.get_user(existing_chat.id)
                else:
                    new_chat_user = ChatUser(
                        id=f"chat-user-{secrets.token_hex(8)}",
                        username=dash_user.username,
                        display_name=dash_user.display_name,
                        password_hash=dash_user.password_hash,
                        role=chat_role,
                        is_active=dash_user.is_active,
                        dashboard_user_id=dash_user.id,
                        session_version=dash_user.session_version,
                    )
                    await chat_store.create_user(new_chat_user)
                    await _auto_assign_groups(request, new_chat_user)
                    chat_user = new_chat_user

    if chat_user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    await chat_store.record_login(chat_user.id)

    session_cookie = _create_public_chat_session_cookie(config, {
        "id": chat_user.id,
        "username": chat_user.username,
        "role": chat_user.role,
        "session_version": chat_user.session_version,
    })
    response.set_cookie(
        key="public_chat_session",
        value=session_cookie,
        httponly=True,
        samesite="Lax",
        secure=False,
        path="/",
    )

    return {
        "success": True,
        "user": {
            "id": chat_user.id,
            "username": chat_user.username,
            "display_name": chat_user.display_name,
            "role": chat_user.role,
        },
    }


@router.post("/public-chat/auth/logout")
async def public_chat_logout(response: Response) -> Dict[str, Any]:
    """Log out the current public chat user."""
    response.set_cookie(
        key="public_chat_session",
        value="",
        httponly=True,
        samesite="Lax",
        secure=False,
        path="/",
        max_age=0,
    )
    # Also clear dashboard session so dashboard users are fully logged out
    response.set_cookie(
        key="dashboard_session",
        value="",
        httponly=True,
        samesite="Strict",
        secure=False,
        path="/",
        max_age=0,
    )
    return {"success": True}


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


@router.post("/public-chat/auth/change-password")
async def public_chat_change_password(
    request: Request,
    payload: ChangePasswordPayload,
) -> Dict[str, Any]:
    """Allow the current public chat user to change their own password."""
    user = await _get_current_chat_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    chat_store = _get_chat_user_store(request)
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat user store unavailable")

    chat_user = await chat_store.get_user(user["id"])
    if chat_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not ChatUserStore._verify_password(payload.current_password, chat_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    await chat_store.set_password(user["id"], payload.new_password)

    # If this chat user is linked to a dashboard user, sync the password there too
    if chat_user.dashboard_user_id:
        dash_store = _get_dashboard_user_store(request)
        if dash_store is not None:
            await dash_store.set_password(chat_user.dashboard_user_id, payload.new_password)

    return {"success": True}


@router.get("/public-chat/auth/me")
async def public_chat_me(request: Request) -> MeResponse:
    """Get the current public chat user info."""
    user = await _get_current_chat_user(request)
    if user is None:
        return MeResponse(authenticated=False)
    # Resolve email + group memberships for the response (dicts from the
    # dashboard-session fallback carry neither).
    email: Optional[str] = None
    group_ids: List[str] = []
    chat_store = _get_chat_user_store(request)
    chat_user = None
    if chat_store is not None and str(user["id"]).startswith("chat-user-"):
        chat_user = await chat_store.get_user(user["id"])
        if chat_user is not None:
            email = chat_user.email
    try:
        from app.chat_group_access import get_groups_store

        groups_store = get_groups_store(request)
        if groups_store is not None and chat_user is not None:
            group_ids = [
                g.id for g in await groups_store.groups_for_user(chat_user.id)
            ]
    except Exception:
        logger.warning("group lookup failed for %s", user["id"], exc_info=True)
    return MeResponse(
        authenticated=True,
        user=ChatUserInfo(
            id=user["id"],
            username=user["username"],
            display_name=user["display_name"],
            role=user["role"],
            email=email,
            groups=group_ids,
        ),
    )
