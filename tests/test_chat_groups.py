"""Tests for chat user groups: store logic, domain matching, token scope,
and effective-allowlist precedence."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import pytest

from app.chat_groups_store import (
    ChatGroupsStore,
    extract_domain,
    normalize_domain,
)
from app.routers.public_chat_auth import (
    _create_public_chat_access_token,
    _read_public_chat_access_token,
)


# ------------------------------------------------------------------------------
# fakes
# ------------------------------------------------------------------------------

class FakeRow:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class FakeMappings:
    def __init__(self, rows: List[dict]) -> None:
        self._rows = [FakeRow(**r) for r in rows]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class FakeResult:
    def __init__(self, rows: Optional[List[dict]] = None, rowcount: int = 0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def mappings(self):
        return FakeMappings(self._rows)


class FakeConnection:
    def __init__(self, results: Optional[List[FakeResult]] = None) -> None:
        self.results = list(results or [])
        self.executed: List[tuple] = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params or {}))
        if self.results:
            return self.results.pop(0)
        return FakeResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


class FakeEngine:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def begin(self):
        return self._conn


def _group_row(**kw) -> dict:
    row = {
        "id": "g1",
        "name": "G",
        "description": "",
        "color": "",
        "allowed_models": '["a","b"]',
        "auto_domains": "[]",
        "created_at": None,
        "updated_at": None,
    }
    row.update(kw)
    return row


# ------------------------------------------------------------------------------
# pure helpers
# ------------------------------------------------------------------------------

class TestDomainHelpers:
    def test_normalize_domain(self):
        assert normalize_domain("  Firma.DE ") == "firma.de"
        assert normalize_domain("@firma.de") == "firma.de"
        assert normalize_domain("") == ""

    def test_extract_domain(self):
        assert extract_domain("a.b@Firma.DE") == "firma.de"
        assert extract_domain("not-an-email") == ""
        assert extract_domain("") == ""


# ------------------------------------------------------------------------------
# effective_allowed_models union semantics
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEffectiveAllowlist:
    async def test_union_of_groups(self):
        # groups_for_user → single JOIN query returning two groups
        conn = FakeConnection([
            FakeResult([_group_row(allowed_models='["a","b"]'),
                        _group_row(id="g2", allowed_models='["c"]')]),
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        result = await store.effective_allowed_models(user_id="u1")
        assert result == {"a", "b", "c"}

    async def test_null_group_is_unrestricted(self):
        conn = FakeConnection([
            FakeResult([_group_row(allowed_models=None),
                        _group_row(id="g2", allowed_models='["a"]')]),
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.effective_allowed_models(user_id="u1") is None

    async def test_empty_list_grants_nothing(self):
        conn = FakeConnection([
            FakeResult([_group_row(allowed_models="[]")]),
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.effective_allowed_models(user_id="u1") == set()

    async def test_no_memberships_falls_back(self):
        conn = FakeConnection([FakeResult([])])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.effective_allowed_models(user_id="u1") is None

    async def test_union_by_group_ids(self):
        # get_group called per id → one SELECT each
        conn = FakeConnection([
            FakeResult([_group_row(id="g1", allowed_models='["a"]')]),
            FakeResult([_group_row(id="g2", allowed_models='["b"]')]),
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        result = await store.effective_allowed_models(group_ids=["g1", "g2"])
        assert result == {"a", "b"}


# ------------------------------------------------------------------------------
# auto_assign_user
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAutoAssign:
    async def test_assigns_matching_domain(self):
        user = SimpleNamespace(id="u1", email="x@firma.de", username="x")
        # memberships_for_user → [] ; list_groups → g1 with matching domain;
        # add_user_to_group → insert
        conn = FakeConnection([
            FakeResult([]),  # memberships_for_user
            FakeResult([_group_row(auto_domains='["firma.de"]'),
                        _group_row(id="g2", auto_domains='["andere.de"]')]),
            FakeResult(rowcount=1),  # insert
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assigned = await store.auto_assign_user(user)
        assert assigned == ["g1"]
        assert any("INSERT" in sql for sql, _ in conn.executed)

    async def test_username_fallback_domain(self):
        user = SimpleNamespace(id="u1", email=None, username="x@firma.de")
        conn = FakeConnection([
            FakeResult([]),
            FakeResult([_group_row(auto_domains='["firma.de"]')]),
            FakeResult(rowcount=1),
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.auto_assign_user(user) == ["g1"]

    async def test_no_email_no_assign(self):
        user = SimpleNamespace(id="u1", email=None, username="plain")
        conn = FakeConnection()
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.auto_assign_user(user) == []
        assert conn.executed == []

    async def test_idempotent(self):
        # already a member → no new assignment
        user = SimpleNamespace(id="u1", email="x@firma.de", username="x")
        conn = FakeConnection([
            FakeResult([{"user_id": "u1", "group_id": "g1", "assigned_via": "domain"}]),
            FakeResult([_group_row(auto_domains='["firma.de"]')]),
            FakeResult(rowcount=0),  # ON CONFLICT DO NOTHING
        ])
        store = ChatGroupsStore(FakeEngine(conn))
        store._tables_ensured = True
        assert await store.auto_assign_user(user) == []


# ------------------------------------------------------------------------------
# token round-trip
# ------------------------------------------------------------------------------

class TestTokenScope:
    def _cfg(self):
        return SimpleNamespace(dashboard_session_secret="test-secret-123")

    def _req(self, token: Optional[str] = None):
        return SimpleNamespace(
            headers={"x-public-chat-token": token} if token else {},
            query_params={},
        )

    def test_role_only_round_trip(self):
        token = _create_public_chat_access_token(self._cfg(), "standard")
        data = _read_public_chat_access_token(self._req(token), self._cfg())
        assert data == {"role": "standard", "groups": [], "kind": None}

    def test_groups_round_trip(self):
        token = _create_public_chat_access_token(
            self._cfg(), "anonymous", groups=["g1", "g2"]
        )
        data = _read_public_chat_access_token(self._req(token), self._cfg())
        assert data == {"role": "anonymous", "groups": ["g1", "g2"], "kind": None}

    def test_link_kind_round_trip(self):
        token = _create_public_chat_access_token(
            self._cfg(), "anonymous", groups=["g1"], link=True
        )
        data = _read_public_chat_access_token(self._req(token), self._cfg())
        assert data == {"role": "anonymous", "groups": ["g1"], "kind": "link"}

    def test_no_token(self):
        assert _read_public_chat_access_token(self._req(), self._cfg()) is None

    def test_bad_token(self):
        assert _read_public_chat_access_token(
            self._req("garbage"), self._cfg()
        ) is None


# ------------------------------------------------------------------------------
# resolve_effective_allowlist precedence
# ------------------------------------------------------------------------------

class _FakeGroupsStore:
    def __init__(self, groups=None, allowed=None):
        self._groups = groups or []
        self._allowed = allowed

    async def groups_for_user(self, user_id):
        return self._groups

    async def effective_allowed_models(self, group_ids=None, *, user_id=None):
        return self._allowed


def _make_request(groups_store=None, engine=object()):
    app_state = SimpleNamespace(
        _chat_groups_store=groups_store,
        rag_application=SimpleNamespace(engine=engine),
    )
    return SimpleNamespace(app=SimpleNamespace(state=app_state))


@pytest.mark.asyncio
async def test_resolver_pct_groups_win(monkeypatch):
    from app import chat_group_access

    store = _FakeGroupsStore(allowed={"m1"})
    monkeypatch.setattr(chat_group_access, "get_groups_store", lambda r: store)
    role_calls = []
    async def _role(req, role):
        role_calls.append(role)
        return {"role-model"}
    monkeypatch.setattr(chat_group_access, "resolve_role_allowlist", _role)

    req = _make_request(groups_store=store)
    result = await chat_group_access.resolve_effective_allowlist(
        req,
        pct_data={"role": "standard", "groups": ["g1"]},
        chat_user={"id": "u1", "role": "standard"},
    )
    assert result == {"m1"}
    assert role_calls == []  # role must NOT be consulted


@pytest.mark.asyncio
async def test_resolver_session_groups(monkeypatch):
    from app import chat_group_access

    store = _FakeGroupsStore(groups=[object()], allowed={"g-model"})
    monkeypatch.setattr(chat_group_access, "get_groups_store", lambda r: store)
    req = _make_request(groups_store=store)
    result = await chat_group_access.resolve_effective_allowlist(
        req, pct_data=None,
        chat_user={"id": "u1", "role": "standard"},
    )
    assert result == {"g-model"}


@pytest.mark.asyncio
async def test_resolver_role_fallback(monkeypatch):
    from app import chat_group_access

    store = _FakeGroupsStore(groups=[])
    monkeypatch.setattr(chat_group_access, "get_groups_store", lambda r: store)
    async def _role(req, role):
        return {"r-model"} if role == "pro" else None
    monkeypatch.setattr(chat_group_access, "resolve_role_allowlist", _role)

    req = _make_request(groups_store=store)
    result = await chat_group_access.resolve_effective_allowlist(
        req, pct_data={"role": "pro", "groups": []}, chat_user=None,
    )
    assert result == {"r-model"}


@pytest.mark.asyncio
async def test_resolver_anonymous_unrestricted(monkeypatch):
    from app import chat_group_access

    monkeypatch.setattr(chat_group_access, "get_groups_store", lambda r: None)
    async def _role(req, role):
        return None
    monkeypatch.setattr(chat_group_access, "resolve_role_allowlist", _role)
    req = _make_request()
    assert await chat_group_access.resolve_effective_allowlist(
        req, pct_data=None, chat_user=None
    ) is None
