"""Tests for the chat file storage sweeper (app/storage_sweeper.py)."""

from __future__ import annotations

import os
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.storage_sweeper import sweep_chat_storage


class FakeConvStore:
    def __init__(self, rows=None):
        # rows: {raw_conversation_id: updated_at}
        self.rows = dict(rows or {})

    async def get_updated_at_map(self):
        return dict(self.rows)


def _state(tmp_path, conv_store=None, data_dir=None, cfg=None):
    """app.state stub: data_dir + optional conv store + optional cfg values."""
    data_dir = data_dir or tmp_path / "data"
    store_cfg = cfg or {}

    class FakeRuntimeStore:
        async def get(self, key):
            return store_cfg.get(key)

    class FakeConfig:
        _env_backed_values = {}
        # Mirror AppConfig field defaults so the runtime store is consulted.
        chat_files_orphan_ttl_days = 7
        chat_files_retention_days = 0

        def __init__(self):
            self.data_dir = data_dir

    return types.SimpleNamespace(
        app_config=FakeConfig(),
        _chat_conversations_store=conv_store,
        _runtime_config_store=FakeRuntimeStore(),
        rag_application=None,
    )


def _mk_conv_dir(root: Path, conv_id: str, *, age_days: float = 0) -> Path:
    d = root / conv_id
    d.mkdir(parents=True)
    (d / "file.bin").write_bytes(b"x")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(d, (old, old))
    return d


def _no_runs(monkeypatch):
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: [])


@pytest.mark.asyncio
async def test_orphan_dir_older_than_ttl_deleted(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "artifacts"
    orphan = _mk_conv_dir(root, "gone-conv", age_days=10)
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert not orphan.exists()
    assert stats["deleted_orphan"] == 1


@pytest.mark.asyncio
async def test_fresh_orphan_dir_kept_by_grace(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "chat_uploads"
    fresh = _mk_conv_dir(root, "pending-upload", age_days=1)
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert fresh.exists()
    assert stats["deleted_orphan"] == 0


@pytest.mark.asyncio
async def test_dir_with_row_never_deleted_by_orphan_rule(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "artifacts"
    live = _mk_conv_dir(root, "live-conv", age_days=365)
    old = datetime.now(timezone.utc) - timedelta(days=365)
    state = _state(tmp_path, FakeConvStore({"live-conv": old}))

    stats = await sweep_chat_storage(state)

    assert live.exists()
    assert stats["deleted_orphan"] == 0
    assert stats["deleted_retention"] == 0


@pytest.mark.asyncio
async def test_sanitized_dirname_matches_raw_row(tmp_path, monkeypatch):
    """Dirs are named sanitize_filename(conv_id) — a raw-id comparison would
    orphan live conversations whose ids contain special chars."""
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "artifacts"
    from app.artifacts import sanitize_filename

    raw_id = "conv/with space&chars"
    live = _mk_conv_dir(root, sanitize_filename(raw_id), age_days=365)
    state = _state(tmp_path, FakeConvStore({raw_id: datetime.now(timezone.utc)}))

    await sweep_chat_storage(state)

    assert live.exists()


@pytest.mark.asyncio
async def test_retention_deletes_files_but_not_for_live_by_default(
    tmp_path, monkeypatch
):
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "chat_uploads"
    live = _mk_conv_dir(root, "old-live", age_days=365)
    old = datetime.now(timezone.utc) - timedelta(days=365)
    state = _state(
        tmp_path,
        FakeConvStore({"old-live": old}),
        cfg={"chat_files_retention_days": 90},
    )

    stats = await sweep_chat_storage(state)

    assert not live.exists()
    assert stats["deleted_retention"] == 1


@pytest.mark.asyncio
async def test_retention_zero_keeps_live_dirs(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "chat_uploads"
    live = _mk_conv_dir(root, "old-live", age_days=365)
    old = datetime.now(timezone.utc) - timedelta(days=365)
    state = _state(tmp_path, FakeConvStore({"old-live": old}), cfg={"chat_files_retention_days": 0})

    stats = await sweep_chat_storage(state)

    assert live.exists()
    assert stats["deleted_retention"] == 0


@pytest.mark.asyncio
async def test_active_run_dir_skipped(tmp_path, monkeypatch):
    root = tmp_path / "data" / "artifacts"
    busy = _mk_conv_dir(root, "running-conv", age_days=100)
    monkeypatch.setattr(
        "app.chat_runs.active_conversation_ids",
        lambda: ["running-conv"],
    )
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert busy.exists()
    assert stats["skipped_active"] == 1


@pytest.mark.asyncio
async def test_no_store_no_deletes(tmp_path, monkeypatch):
    """Without a conversations store the sweeper must never delete."""
    _no_runs(monkeypatch)
    root = tmp_path / "data" / "artifacts"
    orphan = _mk_conv_dir(root, "orphan", age_days=100)
    state = _state(tmp_path, conv_store=None)

    stats = await sweep_chat_storage(state)

    assert orphan.exists()
    assert stats["deleted_orphan"] == 0
    assert stats["errors"] == 1  # lookup failure counted, nothing touched


@pytest.mark.asyncio
async def test_non_dir_entries_ignored_and_both_roots_swept(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    art = tmp_path / "data" / "artifacts"
    upl = tmp_path / "data" / "chat_uploads"
    art.mkdir(parents=True)
    upl.mkdir(parents=True)
    (art / "stray-file.txt").write_text("not a dir")
    o1 = _mk_conv_dir(art, "o1", age_days=10)
    o2 = _mk_conv_dir(upl, "o2", age_days=10)
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert not o1.exists() and not o2.exists()
    assert (art / "stray-file.txt").exists()
    assert stats["deleted_orphan"] == 2
    assert stats["scanned"] == 2


# ---------------------------------------------------------------------------
# DELETE /chat/conversations/{id}
# ---------------------------------------------------------------------------

class _StoreSpy:
    def __init__(self):
        self.deleted = []

    async def delete_conversation(self, cid):
        self.deleted.append(cid)


class _ConvStoreSpy(FakeConvStore):
    async def delete(self, cid):
        self.deleted_row = cid


def _req(state_attrs):
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(**state_attrs)),
        state=types.SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_delete_conversation_cleans_all_stores(monkeypatch):
    from app.routers import chat as chat_router

    monkeypatch.setattr(chat_router, "_attachment_access_allowed", lambda r: _ait(True))
    monkeypatch.setattr(chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(True))
    conv_store = _ConvStoreSpy()
    art, upl = _StoreSpy(), _StoreSpy()
    req = _req(
        {
            "_chat_conversations_store": conv_store,
            "artifact_store": art,
            "chat_attachment_store": upl,
        }
    )

    out = await chat_router.chat_conversation_delete("conv-1", req)

    assert out == {"success": True}
    assert conv_store.deleted_row == "conv-1"
    assert art.deleted == ["conv-1"] and upl.deleted == ["conv-1"]


async def _ait(v):
    return v


@pytest.mark.asyncio
async def test_delete_conversation_denied(monkeypatch):
    from app.routers import chat as chat_router

    monkeypatch.setattr(chat_router, "_attachment_access_allowed", lambda r: _ait(True))
    monkeypatch.setattr(chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(False))
    req = _req({})

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await chat_router.chat_conversation_delete("conv-1", req)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_conversation_cancels_active_run(monkeypatch):
    from app.routers import chat as chat_router
    from app.chat_runs import ActiveChatRun

    monkeypatch.setattr(chat_router, "_attachment_access_allowed", lambda r: _ait(True))
    monkeypatch.setattr(chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(True))
    run = ActiveChatRun(conversation_id="conv-1", model_name="m", user_message="hi")
    monkeypatch.setattr("app.chat_runs.get_run", lambda cid: run)
    req = _req({"_chat_conversations_store": _ConvStoreSpy()})

    await chat_router.chat_conversation_delete("conv-1", req)

    assert run.conversation_deleted is True
    assert run.cancel_event.is_set()


@pytest.mark.asyncio
async def test_delete_conversation_tolerates_missing_stores(monkeypatch):
    from app.routers import chat as chat_router

    monkeypatch.setattr(chat_router, "_attachment_access_allowed", lambda r: _ait(True))
    monkeypatch.setattr(chat_router, "_attachment_conv_owner_allowed", lambda r, cid: _ait(True))
    monkeypatch.setattr("app.chat_runs.get_run", lambda cid: None)
    req = _req({})  # no stores at all

    out = await chat_router.chat_conversation_delete("conv-1", req)
    assert out == {"success": True}


# ---------------------------------------------------------------------------
# v2 sweeper: sandbox_workspaces root, container + memory orphans
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sandbox_workspaces_orphan_swept(tmp_path, monkeypatch):
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: [])
    monkeypatch.setenv("SANDBOX_WORKSPACE_ROOT", str(tmp_path / "data" / "sandbox_workspaces"))
    root = tmp_path / "data" / "sandbox_workspaces"
    orphan = _mk_conv_dir(root, "gone-ws", age_days=30)
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert not orphan.exists()
    assert stats["deleted_orphan"] == 1


@pytest.mark.asyncio
async def test_container_orphan_destroyed_by_label(tmp_path, monkeypatch):
    """Containers labeled for a conv with no row + older than TTL get destroyed."""
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: [])
    destroyed = []

    class _FakeContainer:
        attrs = {
            "Labels": {"agentix.sandbox.conversation_id": "dead-conv", "agentix.sandbox.managed": "true"},
            "Created": "2020-01-01T00:00:00.000000000Z",
        }

    class _FakeContainers:
        def list(self, all=False, filters=None):
            assert filters == {"label": "agentix.sandbox.managed=true"}
            return [_FakeContainer()]

    class _Mgr:
        _client = types.SimpleNamespace(containers=_FakeContainers())

        async def _ensure_client(self):
            return self._client

        async def destroy(self, cid):
            destroyed.append(cid)

    import agent.tools.docker_sandbox as ds

    monkeypatch.setattr(ds, "ConversationSandboxManager", _Mgr)
    state = _state(tmp_path, FakeConvStore())

    stats = await sweep_chat_storage(state)

    assert destroyed == ["dead-conv"]
    assert stats["deleted_sandbox_container"] == 1


@pytest.mark.asyncio
async def test_container_with_live_row_survives(tmp_path, monkeypatch):
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: [])
    destroyed = []

    class _FakeContainer:
        attrs = {
            "Labels": {"agentix.sandbox.conversation_id": "live-conv", "agentix.sandbox.managed": "true"},
            "Created": "2020-01-01T00:00:00.000000000Z",
        }

    class _FakeContainers:
        def list(self, all=False, filters=None):
            return [_FakeContainer()]

    class _Mgr:
        _client = types.SimpleNamespace(containers=_FakeContainers())

        async def _ensure_client(self):
            return self._client

        async def destroy(self, cid):
            destroyed.append(cid)

    import agent.tools.docker_sandbox as ds

    monkeypatch.setattr(ds, "ConversationSandboxManager", _Mgr)
    state = _state(
        tmp_path, FakeConvStore({"live-conv": datetime.now(timezone.utc)})
    )

    stats = await sweep_chat_storage(state)

    assert destroyed == []
    assert stats["deleted_sandbox_container"] == 0


@pytest.mark.asyncio
async def test_memory_orphan_deleted_after_grace(tmp_path, monkeypatch):
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: [])
    deleted = []

    class _FakeMemoryStore:
        def __init__(self, engine):
            pass

        async def memory_conversation_index(self):
            old = datetime.now(timezone.utc) - timedelta(days=30)
            return {"dead-conv": old, "live-conv": old}

        async def delete_memories_for_conversation(self, cid):
            deleted.append(cid)
            return 2

    monkeypatch.setattr("app.stores.AgentMemoryStore", _FakeMemoryStore)
    conv_store = FakeConvStore({"live-conv": datetime.now(timezone.utc)})
    state = _state(tmp_path, conv_store)
    state.rag_application = types.SimpleNamespace(engine=object(), _engine=None)

    stats = await sweep_chat_storage(state)

    assert deleted == ["dead-conv"]
    assert stats["deleted_memory_orphans"] == 2


@pytest.mark.asyncio
async def test_memory_orphan_fresh_kept_by_grace(tmp_path, monkeypatch):
    """Memory written mid-first-run (row not yet created) must not be swept."""
    monkeypatch.setattr("app.chat_runs.active_conversation_ids", lambda: ["new-conv"])
    deleted = []

    class _FakeMemoryStore:
        def __init__(self, engine):
            pass

        async def memory_conversation_index(self):
            return {"new-conv": datetime.now(timezone.utc)}

        async def delete_memories_for_conversation(self, cid):
            deleted.append(cid)
            return 1

    monkeypatch.setattr("app.stores.AgentMemoryStore", _FakeMemoryStore)
    state = _state(tmp_path, FakeConvStore())
    state.rag_application = types.SimpleNamespace(engine=object(), _engine=None)

    stats = await sweep_chat_storage(state)

    assert deleted == []
    assert stats["deleted_memory_orphans"] == 0


# ---------------------------------------------------------------------------
# Agent workspace sweep (data/agent_workspaces/<assistant>/<scope>/)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_workspace_conv_dir_live_sanitized_id_kept(tmp_path, monkeypatch):
    """c_<sanitized> dirs must be compared against SANITIZED live ids —
    a conv id that needs sanitizing must not be orphan-deleted while live."""
    _no_runs(monkeypatch)
    monkeypatch.setenv(
        "AGENT_WORKSPACE_APP_ROOT", str(tmp_path / "data" / "agent_workspaces")
    )
    raw_conv = "conv:a/b"  # sanitize_filename -> 'b'
    ws_root = tmp_path / "data" / "agent_workspaces"
    live_dir = _mk_conv_dir(ws_root / "MyBot", "c_b", age_days=365)
    state = _state(
        tmp_path, FakeConvStore({raw_conv: datetime.now(timezone.utc)})
    )
    state.rag_application = types.SimpleNamespace(
        instructions={"MyBot": {}}, engine=None, _engine=None
    )

    stats = await sweep_chat_storage(state)

    assert live_dir.exists()
    assert stats["deleted_workspace_orphans"] == 0


@pytest.mark.asyncio
async def test_agent_workspace_orphan_assistant_deleted(tmp_path, monkeypatch):
    _no_runs(monkeypatch)
    monkeypatch.setenv(
        "AGENT_WORKSPACE_APP_ROOT", str(tmp_path / "data" / "agent_workspaces")
    )
    ws_root = tmp_path / "data" / "agent_workspaces"
    orphan = _mk_conv_dir(ws_root / "GoneBot", "shared", age_days=30)
    live = _mk_conv_dir(ws_root / "MyBot", "shared", age_days=30)
    # The sweep ages the ASSISTANT dir, not the leaf inside it.
    old = time.time() - 30 * 86400
    os.utime(ws_root / "GoneBot", (old, old))
    state = _state(tmp_path, FakeConvStore())
    state.rag_application = types.SimpleNamespace(
        instructions={"MyBot": {}}, engine=None, _engine=None
    )

    stats = await sweep_chat_storage(state)

    assert not orphan.exists()
    assert live.exists()
    assert stats["deleted_workspace_orphans"] == 1


@pytest.mark.asyncio
async def test_agent_workspace_shared_and_user_never_swept(tmp_path, monkeypatch):
    """shared/ and u_*/ under a LIVE assistant are user data — never deleted."""
    _no_runs(monkeypatch)
    monkeypatch.setenv(
        "AGENT_WORKSPACE_APP_ROOT", str(tmp_path / "data" / "agent_workspaces")
    )
    ws_root = tmp_path / "data" / "agent_workspaces"
    shared = _mk_conv_dir(ws_root / "MyBot", "shared", age_days=365)
    user = _mk_conv_dir(ws_root / "MyBot", "u_someone", age_days=365)
    state = _state(tmp_path, FakeConvStore())
    state.rag_application = types.SimpleNamespace(
        instructions={"MyBot": {}}, engine=None, _engine=None
    )

    stats = await sweep_chat_storage(state)

    assert shared.exists()
    assert user.exists()
    assert stats["deleted_workspace_orphans"] == 0
