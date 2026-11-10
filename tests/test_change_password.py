"""Tests for self-service change-password endpoints (dashboard + public chat)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.routers.admin import admin_change_password, ChangePasswordPayload
from app.routers.public_chat_auth import public_chat_change_password


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _hash_pw(pw: str) -> str:
    import bcrypt
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=4)).decode("utf-8")


def _verify_pw(pw: str, pw_hash: str) -> bool:
    import bcrypt
    return bcrypt.checkpw(pw.encode("utf-8"), pw_hash.encode("utf-8"))


def _enabled_config() -> Any:
    return SimpleNamespace(
        dashboard_enabled=True,
        dashboard_admin_username="admin",
        dashboard_admin_password="secret",
        dashboard_session_secret="super-secret-key-for-tests",
        dashboard_session_ttl_seconds=3600,
    )


# ------------------------------------------------------------------------------
# Dashboard change-password tests
# ------------------------------------------------------------------------------

class FakeDashboardStoreWithUser:
    """Fake store that holds a real user with a bcrypt password hash."""

    def __init__(self) -> None:
        from app.dashboard_users import DashboardUser
        self._user = DashboardUser(
            id="user-001",
            username="testuser",
            display_name="Test User",
            password_hash=_hash_pw("oldpass123"),
            role="viewer",
            is_active=True,
            session_version=1,
        )
        self.set_password_calls: list[tuple[str, str]] = []

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        return _verify_pw(password, password_hash)

    async def get_user(self, user_id: str) -> Any | None:
        if user_id == self._user.id:
            return self._user
        return None

    async def get_by_username(self, username: str) -> Any | None:
        if username == self._user.username:
            return self._user
        return None

    async def authenticate(self, username: str, password: str) -> Any | None:
        u = await self.get_by_username(username)
        if u and self._verify_password(password, u.password_hash):
            return u
        return None

    async def record_login(self, user_id: str) -> None:
        pass

    async def set_password(self, user_id: str, password: str) -> None:
        self.set_password_calls.append((user_id, password))
        self._user.password_hash = _hash_pw(password)
        self._user.session_version += 1


def _make_dashboard_request(
    *,
    app_config: Any | None = None,
    cookies: dict[str, str] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Any:
    state: dict[str, Any] = {"app_config": app_config}
    if extra_state:
        state.update(extra_state)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state)),
        cookies=cookies or {},
    )


async def _login_dashboard_user(store: FakeDashboardStoreWithUser) -> tuple[Any, str]:
    """Login as testuser and return (request_with_cookie, user_id)."""
    from app.routers.admin import admin_login
    config = _enabled_config()
    request = _make_dashboard_request(
        app_config=config,
        extra_state={"_dashboard_user_store": store},
    )
    response = await admin_login(
        request,
        payload=SimpleNamespace(username="testuser", password="oldpass123"),
    )
    token = response.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]
    return request, token


@pytest.mark.asyncio
async def test_dashboard_change_password_success():
    store = FakeDashboardStoreWithUser()
    config = _enabled_config()

    # Login to get a valid session cookie
    from app.routers.admin import admin_login
    login_req = _make_dashboard_request(
        app_config=config,
        extra_state={"_dashboard_user_store": store},
    )
    login_resp = await admin_login(
        login_req,
        payload=SimpleNamespace(username="testuser", password="oldpass123"),
    )
    token = login_resp.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    # Change password
    request = _make_dashboard_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_dashboard_user_store": store},
    )
    result = await admin_change_password(
        request,
        payload=ChangePasswordPayload(current_password="oldpass123", new_password="newpass456"),
    )
    assert result["success"] is True
    assert len(store.set_password_calls) == 1
    assert store.set_password_calls[0] == ("user-001", "newpass456")


@pytest.mark.asyncio
async def test_dashboard_change_password_wrong_current():
    store = FakeDashboardStoreWithUser()
    config = _enabled_config()

    from app.routers.admin import admin_login
    login_req = _make_dashboard_request(
        app_config=config,
        extra_state={"_dashboard_user_store": store},
    )
    login_resp = await admin_login(
        login_req,
        payload=SimpleNamespace(username="testuser", password="oldpass123"),
    )
    token = login_resp.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    request = _make_dashboard_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_dashboard_user_store": store},
    )
    with pytest.raises(Exception) as exc_info:
        await admin_change_password(
            request,
            payload=ChangePasswordPayload(current_password="wrongpassword", new_password="newpass456"),
        )
    assert "400" in str(exc_info.value) or "incorrect" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_dashboard_change_password_not_authenticated():
    config = _enabled_config()
    request = _make_dashboard_request(
        app_config=config,
        extra_state={"_dashboard_user_store": FakeDashboardStoreWithUser()},
    )
    with pytest.raises(Exception) as exc_info:
        await admin_change_password(
            request,
            payload=ChangePasswordPayload(current_password="oldpass123", new_password="newpass456"),
        )
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dashboard_change_password_env_admin_rejected():
    """env-admin should not be able to change password via this endpoint."""
    from app.routers.admin import admin_login

    class FakeEnvAdminStore:
        @staticmethod
        def _verify_password(password: str, password_hash: str) -> bool:
            return True

        async def get_user(self, user_id: str) -> Any | None:
            from app.dashboard_users import DashboardUser
            if user_id == "env-admin":
                return DashboardUser(
                    id="env-admin",
                    username="admin",
                    display_name="Env Admin",
                    password_hash="",
                    role="owner",
                    is_active=True,
                    session_version=1,
                )
            return None

        async def get_by_username(self, username: str) -> Any | None:
            return await self.get_user("env-admin")

        async def authenticate(self, username: str, password: str) -> Any | None:
            return await self.get_user("env-admin")

        async def record_login(self, user_id: str) -> None:
            pass

        async def set_password(self, user_id: str, password: str) -> None:
            pass

    config = _enabled_config()
    store = FakeEnvAdminStore()
    login_req = _make_dashboard_request(
        app_config=config,
        extra_state={"_dashboard_user_store": store},
    )
    login_resp = await admin_login(
        login_req,
        payload=SimpleNamespace(username="admin", password="secret"),
    )
    token = login_resp.headers["Set-Cookie"].split("dashboard_session=")[1].split(";")[0]

    request = _make_dashboard_request(
        app_config=config,
        cookies={"dashboard_session": token},
        extra_state={"_dashboard_user_store": store},
    )
    with pytest.raises(Exception) as exc_info:
        await admin_change_password(
            request,
            payload=ChangePasswordPayload(current_password="secret", new_password="newpass456"),
        )
    assert "403" in str(exc_info.value)


# ------------------------------------------------------------------------------
# Public chat change-password tests
# ------------------------------------------------------------------------------

class FakeChatUserStore:
    """Fake chat user store with a real bcrypt hash."""

    def __init__(self) -> None:
        from app.chat_users import ChatUser
        self._user = ChatUser(
            id="chat-user-001",
            username="chatuser",
            display_name="Chat User",
            password_hash=_hash_pw("oldpass123"),
            role="standard",
            is_active=True,
            session_version=1,
        )
        self.set_password_calls: list[tuple[str, str]] = []

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        return _verify_pw(password, password_hash)

    @staticmethod
    def _hash_password(password: str) -> str:
        return _hash_pw(password)

    async def get_user(self, user_id: str) -> Any | None:
        if user_id == self._user.id:
            return self._user
        return None

    async def get_by_username(self, username: str) -> Any | None:
        if username == self._user.username:
            return self._user
        return None

    async def get_by_dashboard_user_id(self, dashboard_user_id: str) -> Any | None:
        return None

    async def authenticate(self, username: str, password: str) -> Any | None:
        u = await self.get_by_username(username)
        if u and self._verify_password(password, u.password_hash):
            return u
        return None

    async def record_login(self, user_id: str) -> None:
        pass

    async def set_password(self, user_id: str, password: str) -> None:
        self.set_password_calls.append((user_id, password))
        self._user.password_hash = _hash_pw(password)
        self._user.session_version += 1

    async def set_role(self, user_id: str, role: str) -> None:
        pass

    async def create_user(self, user: Any) -> None:
        pass

    async def list_users(self) -> list[Any]:
        return [self._user]


class FakeDashboardStoreForSync:
    """Minimal dashboard store to verify password sync for linked users."""

    def __init__(self) -> None:
        self.set_password_calls: list[tuple[str, str]] = []

    async def set_password(self, user_id: str, password: str) -> None:
        self.set_password_calls.append((user_id, password))

    async def get_user(self, user_id: str) -> Any | None:
        return None


def _make_chat_request(
    *,
    app_config: Any | None = None,
    cookies: dict[str, str] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Any:
    state: dict[str, Any] = {"app_config": app_config}
    if extra_state:
        state.update(extra_state)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state)),
        cookies=cookies or {},
    )


async def _login_chat_user(chat_store: FakeChatUserStore) -> str:
    """Login as chat user and return the session cookie token."""
    from app.routers.public_chat_auth import public_chat_login
    config = _enabled_config()
    request = _make_chat_request(
        app_config=config,
        extra_state={"_chat_user_store": chat_store},
    )
    response = await public_chat_login(
        request,
        response=SimpleNamespace(set_cookie=lambda **kw: None),
        payload=SimpleNamespace(username="chatuser", password="oldpass123"),
    )
    # The login endpoint sets cookie on the response object; we need to
    # extract the cookie from the response headers instead.
    # Since our fake response doesn't capture, we'll build the cookie manually.
    from app.routers.public_chat_auth import _create_public_chat_session_cookie
    return _create_public_chat_session_cookie(config, {
        "id": "chat-user-001",
        "username": "chatuser",
        "role": "standard",
        "session_version": 1,
    })


@pytest.mark.asyncio
async def test_public_chat_change_password_success():
    chat_store = FakeChatUserStore()
    config = _enabled_config()
    token = await _login_chat_user(chat_store)

    request = _make_chat_request(
        app_config=config,
        cookies={"public_chat_session": token},
        extra_state={"_chat_user_store": chat_store},
    )
    result = await public_chat_change_password(
        request,
        payload=SimpleNamespace(current_password="oldpass123", new_password="newpass456"),
    )
    assert result["success"] is True
    assert len(chat_store.set_password_calls) == 1
    assert chat_store.set_password_calls[0] == ("chat-user-001", "newpass456")


@pytest.mark.asyncio
async def test_public_chat_change_password_wrong_current():
    chat_store = FakeChatUserStore()
    config = _enabled_config()
    token = await _login_chat_user(chat_store)

    request = _make_chat_request(
        app_config=config,
        cookies={"public_chat_session": token},
        extra_state={"_chat_user_store": chat_store},
    )
    with pytest.raises(Exception) as exc_info:
        await public_chat_change_password(
            request,
            payload=SimpleNamespace(current_password="wrongpassword", new_password="newpass456"),
        )
    assert "400" in str(exc_info.value) or "incorrect" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_public_chat_change_password_not_authenticated():
    config = _enabled_config()
    request = _make_chat_request(
        app_config=config,
        extra_state={"_chat_user_store": FakeChatUserStore()},
    )
    with pytest.raises(Exception) as exc_info:
        await public_chat_change_password(
            request,
            payload=SimpleNamespace(current_password="oldpass123", new_password="newpass456"),
        )
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_public_chat_change_password_syncs_dashboard_linked_user():
    """When chat user has dashboard_user_id, password should be synced to dashboard store."""
    from app.chat_users import ChatUser

    chat_store = FakeChatUserStore()
    chat_store._user.dashboard_user_id = "dash-001"
    dash_store = FakeDashboardStoreForSync()

    config = _enabled_config()
    token = await _login_chat_user(chat_store)

    request = _make_chat_request(
        app_config=config,
        cookies={"public_chat_session": token},
        extra_state={
            "_chat_user_store": chat_store,
            "_dashboard_user_store": dash_store,
        },
    )
    result = await public_chat_change_password(
        request,
        payload=SimpleNamespace(current_password="oldpass123", new_password="newpass456"),
    )
    assert result["success"] is True
    assert len(chat_store.set_password_calls) == 1
    assert len(dash_store.set_password_calls) == 1
    assert dash_store.set_password_calls[0] == ("dash-001", "newpass456")
