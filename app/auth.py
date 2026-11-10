"""Session-based authentication and permission enforcement for the dashboard.

Uses itsdangerous URL-safe timed serializer for signed session cookies and
integrates with DashboardUserStore for multi-user role-based access.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from itsdangerous import BadSignature, URLSafeTimedSerializer

from agent.config import AppConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Session helpers
# ------------------------------------------------------------------------------

def get_serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="dashboard-session")


def create_session_cookie(config: AppConfig, user: Optional[Dict[str, Any]] = None) -> str:
    """Sign a session payload with user context."""
    secret = config.dashboard_session_secret or ""
    if not secret:
        raise RuntimeError("DASHBOARD_SESSION_SECRET is required for admin sessions")
    serializer = get_serializer(secret)
    if user is not None:
        payload = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "session_version": user.get("session_version", 1),
        }
    else:
        # Legacy fallback (env admin is the owner)
        payload = {"role": "owner"}
    return serializer.dumps(payload)


def read_session_cookie(request: Request, config: AppConfig) -> Optional[dict]:
    """Read and verify the dashboard session cookie."""
    secret = config.dashboard_session_secret or ""
    if not secret:
        return None
    raw = request.cookies.get("dashboard_session")
    if not raw:
        return None
    serializer = get_serializer(secret)
    try:
        data = serializer.loads(
            raw,
            max_age=config.dashboard_session_ttl_seconds,
        )
        return data if isinstance(data, dict) else None
    except BadSignature:
        return None


def clear_session_cookie() -> tuple[str, str]:
    """Return a cookie deletion directive for the response."""
    return (
        "Set-Cookie",
        "dashboard_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
    )


# ------------------------------------------------------------------------------
# User context retrieval
# ------------------------------------------------------------------------------

async def get_current_dashboard_user(request: Request) -> Optional[Dict[str, Any]]:
    """Load the current dashboard user from the session and DB."""
    from app.dashboard_users import DashboardUserStore

    config: Optional[AppConfig] = getattr(request.app.state, "app_config", None)
    if not config or not config.dashboard_enabled:
        return None

    session = read_session_cookie(request, config)
    if not session:
        return None

    user_id = session.get("user_id")
    if not user_id:
        # Legacy session fallback
        return {"id": "", "username": "", "role": session.get("role", "admin"), "session_version": 1}

    store: Optional[DashboardUserStore] = getattr(request.app.state, "_dashboard_user_store", None)
    if store is None:
        return None

    user = await store.get_user(user_id)
    if user is None or not user.is_active:
        return None

    # Validate session version (invalidate on password/role change)
    session_version = session.get("session_version", 1)
    if user.session_version != session_version:
        return None

    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "session_version": user.session_version,
    }


# ------------------------------------------------------------------------------
# Permission helpers
# ------------------------------------------------------------------------------

def permissions_for_role(role: str) -> List[str]:
    """Return the permission list for a given role."""
    from app.dashboard_users import _PERMISSIONS
    return _PERMISSIONS.get(role, [])


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    return permission in permissions_for_role(role)


class PermissionDenied(HTTPException):
    def __init__(self, detail: str = "Permission denied") -> None:
        super().__init__(status_code=403, detail=detail)


class AdminAuthRequired(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=401, detail="Admin session required")


async def require_permission(request: Request, permission: str) -> None:
    """Enforce that the current user has a specific permission."""
    user = await get_current_dashboard_user(request)
    if user is None:
        raise AdminAuthRequired()
    if not has_permission(user.get("role", ""), permission):
        raise PermissionDenied(f"Missing permission: {permission}")


async def require_any_permission(request: Request, permissions: List[str]) -> None:
    """Enforce that the current user has at least one of the given permissions."""
    user = await get_current_dashboard_user(request)
    if user is None:
        raise AdminAuthRequired()
    role = user.get("role", "")
    if not any(has_permission(role, p) for p in permissions):
        raise PermissionDenied()


def require_admin(request: Request) -> None:
    """Legacy dependency: enforce any valid admin session."""
    # This is a synchronous wrapper for the async permission check.
    # FastAPI dependencies can be async, but to keep compatibility with
    # existing code that calls this synchronously, we fall back to
    # checking only the cookie without DB lookup.
    config: Optional[AppConfig] = getattr(request.app.state, "app_config", None)
    if not config or not config.dashboard_enabled:
        raise AdminAuthRequired()
    session = read_session_cookie(request, config)
    if not session or session.get("role") not in ("admin", "owner"):
        raise AdminAuthRequired()


# ------------------------------------------------------------------------------
# Legacy helpers (kept for compatibility)
# ------------------------------------------------------------------------------

def _hash_password(password: str, secret: str) -> str:
    """Hash a password for comparison/storage using a simple HMAC."""
    return hashlib.sha256(f"{password}::{secret}".encode()).hexdigest()


def verify_admin_password(config: AppConfig, password: str) -> bool:
    """Timing-safe comparison of the provided password against the configured one."""
    expected = config.dashboard_admin_password
    if not expected:
        return False
    expected_hash = _hash_password(expected, config.dashboard_session_secret or "")
    provided_hash = _hash_password(password, config.dashboard_session_secret or "")
    return hmac.compare_digest(expected_hash, provided_hash)
