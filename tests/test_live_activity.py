"""Tests for GET /admin/dashboard/live-activity and its backing state.

Covers the ActiveChatRun live-activity fields (source/phase), the
trigger scheduler's running_now() accessor, and the endpoint payload:
shape, auth gate, run normalization, ordering, and automation sources.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.chat_runs import ActiveChatRun, _runs, get_run, register_run, unregister_run
from app.routers import admin as admin_module
from app.routers.admin import dashboard_live_activity, _live_run_payload
from app.triggers_scheduler import TriggersScheduler


@pytest.fixture(autouse=True)
def clear_runs():
    _runs.clear()
    yield
    _runs.clear()


def _make_run(conv_id: str = "conv-1", **kwargs: Any) -> ActiveChatRun:
    run = ActiveChatRun(
        conversation_id=conv_id,
        model_name=kwargs.pop("model_name", "TestAssistant"),
        user_message=kwargs.pop("user_message", "hello"),
        retrieval_info={},
    )
    for key, value in kwargs.items():
        setattr(run, key, value)
    return run


def _make_request(**state: Any) -> Any:
    defaults = {
        "app_config": None,
        "_triggers_scheduler": None,
        "_triggers_store": None,
        "_backup_job_store": None,
        "_webdav_source_store": None,
    }
    defaults.update(state)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**defaults)),
        cookies={},
    )


def _patch_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin_module, "require_permission", AsyncMock())


class _FakeTrigger:
    def __init__(self, tid: str, name: str) -> None:
        self.id = tid
        self.name = name
        self.schedule_type = "interval"
        self.schedule_config = {}


class _FakeTriggersStore:
    def __init__(self, names: dict[str, str] | None = None) -> None:
        self._names = names or {}

    async def get_trigger(self, trigger_id: str):
        name = self._names.get(trigger_id)
        if name is None:
            return None
        return _FakeTrigger(trigger_id, name)


class _FakeBackupJobStore:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self._jobs = jobs

    async def list_jobs(self, limit: int = 50, offset: int = 0):
        return self._jobs[:limit]


class _FakeWebDAVStore:
    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self._sources = sources

    async def list_sources(self):
        return self._sources


class _FakeScheduler:
    def __init__(self, running: dict[str, float]) -> None:
        self._running = running

    def running_now(self) -> dict[str, float]:
        return dict(self._running)


# ---------------------------------------------------------------------------
# ActiveChatRun live-activity fields
# ---------------------------------------------------------------------------

def test_run_defaults():
    run = _make_run()
    assert run.source == ""
    assert run.activity_phase == "starting"
    assert run.activity_detail == ""


def test_record_token_phase_mapping():
    run = _make_run()
    run.record_token({"reasoning": "thinking..."})
    assert run.activity_phase == "reasoning"
    run.record_token({"content": "answer"})
    assert run.activity_phase == "generating"
    run.record_token({"tool_progress": {"id": "t1", "tool_name": "web_search", "stage": "started"}})
    assert run.activity_phase == "tool"
    assert run.activity_detail == "web_search"
    run.record_token({"tool_execution": {"tool": "code_execution", "stage": "started"}})
    assert run.activity_phase == "tool"
    assert run.activity_detail == "code_execution"
    run.record_token({"tool_calls": [{"id": "t1", "name": "web_search", "output": "res", "ok": True}]})
    assert run.activity_phase == "results"
    run.record_token({"observer": {"action": "nudge"}})
    assert run.activity_phase == "observer"


def test_record_token_tool_calls_request_sets_tool_phase():
    run = _make_run()
    run.record_token({"content": "answer"})
    # A bare tool-call request (no result yet) shows the tool phase — and
    # must not masquerade as results.
    run.record_token({"tool_calls": [{"id": "t1", "name": "web_search"}]})
    assert run.activity_phase == "tool"
    assert run.activity_detail == "web_search"


# ---------------------------------------------------------------------------
# _live_run_payload normalization
# ---------------------------------------------------------------------------

def test_payload_running_agent():
    run = _make_run(
        model_name="Agent A",
        is_agent_run=True,
        source="dashboard",
        activity_phase="tool",
        activity_detail="rag_search",
        live_usage={"total_tokens": 1234, "llm_calls": 3, "estimated": True},
        subagent_events=[{"id": "s1", "status": "running"}, {"id": "s2", "status": "done"}],
        todo_list={"items": [{"status": "done"}, {"status": "open"}, {"status": "pending"}]},
        tool_calls_buffer=[{"id": "t1"}],
        user_message="x" * 300,
    )
    payload = _live_run_payload(run, run.created_monotonic + 5.0)
    assert payload["kind"] == "agent"
    assert payload["state"] == "running"
    assert payload["source"] == "dashboard"
    assert payload["phase"] == "tool"
    assert payload["phase_detail"] == "rag_search"
    assert payload["tokens"] == 1234
    assert payload["llm_calls"] == 3
    assert payload["estimated"] is True
    assert payload["tool_calls"] == 1
    assert payload["subagents_running"] == 1
    assert payload["subagents_total"] == 2
    assert payload["todos_open"] == 2
    assert payload["todos_total"] == 3
    assert len(payload["user_message"]) == 140
    assert payload["elapsed_s"] == pytest.approx(5.0, abs=0.2)


def test_payload_chain_and_states():
    chain = _make_run(model_name="chain:MyChain")
    assert _live_run_payload(chain, chain.created_monotonic)["kind"] == "chain"
    chat = _make_run(model_name="Plain")
    assert _live_run_payload(chat, chat.created_monotonic)["kind"] == "chat"

    run = _make_run()
    run.done = True
    run.finished_monotonic = run.created_monotonic + 2.0
    assert _live_run_payload(run, run.created_monotonic + 99)["state"] == "done"
    assert _live_run_payload(run, run.created_monotonic + 99)["elapsed_s"] == pytest.approx(2.0, abs=0.2)

    cancelled = _make_run()
    cancelled.done = True
    cancelled.cancelled = True
    assert _live_run_payload(cancelled, cancelled.created_monotonic)["state"] == "cancelled"

    errored = _make_run()
    errored.done = True
    errored.stream_error = RuntimeError("boom")
    assert _live_run_payload(errored, errored.created_monotonic)["state"] == "error"


# ---------------------------------------------------------------------------
# TriggerScheduler.running_now()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scheduler_running_now_tracks_and_cleans(monkeypatch):
    store = SimpleNamespace(get_due_triggers=AsyncMock(return_value=[]))
    scheduler = TriggersScheduler(store, rag_app=None)
    trigger = _FakeTrigger("t-1", "Nightly")
    seen_during_run: list[dict] = []

    async def fake_run(t):
        seen_during_run.append(scheduler.running_now())

    monkeypatch.setattr(scheduler, "_run_trigger", AsyncMock(side_effect=fake_run))
    store.get_due_triggers = AsyncMock(return_value=[trigger])
    await scheduler.check_and_run()
    assert seen_during_run and "t-1" in seen_during_run[0]
    assert isinstance(seen_during_run[0]["t-1"], float)
    assert scheduler.running_now() == {}
    assert "t-1" not in scheduler._running


@pytest.mark.asyncio
async def test_scheduler_running_now_skips_duplicates():
    store = SimpleNamespace(get_due_triggers=AsyncMock(return_value=[]))
    scheduler = TriggersScheduler(store, rag_app=None)
    scheduler._running.add("t-1")
    scheduler._running_since["t-1"] = 0.0
    scheduler._run_trigger = AsyncMock()  # must not be invoked for a dup

    store.get_due_triggers = AsyncMock(return_value=[_FakeTrigger("t-1", "Nightly")])
    await scheduler.check_and_run()
    scheduler._run_trigger.assert_not_called()
    assert scheduler.running_now() == {"t-1": 0.0}


# ---------------------------------------------------------------------------
# Endpoint: auth gate + payload assembly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_activity_requires_permission():
    request = _make_request(
        app_config=SimpleNamespace(
            dashboard_enabled=True,
            dashboard_admin_username="admin",
            dashboard_admin_password="secret",
            dashboard_session_secret="x" * 32,
            dashboard_session_ttl_seconds=3600,
        )
    )
    with pytest.raises(Exception) as exc_info:
        await dashboard_live_activity(request)
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_live_activity_empty(monkeypatch):
    _patch_permission(monkeypatch)
    res = await dashboard_live_activity(_make_request())
    assert res["runs"] == []
    assert res["automation"] == {"triggers": [], "backups": [], "syncs": []}
    assert res["generated_at"]


@pytest.mark.asyncio
async def test_live_activity_runs_shape_and_order(monkeypatch):
    _patch_permission(monkeypatch)
    running_old = _make_run("c-old", model_name="Agent A", is_agent_run=True)
    running_old.created_monotonic -= 60.0
    running_new = _make_run("c-new", model_name="chain:X")
    done_run = _make_run("c-done")
    done_run.done = True
    done_run.finished_monotonic = done_run.created_monotonic + 3.0
    for r in (running_old, running_new, done_run):
        register_run(r)

    res = await dashboard_live_activity(_make_request())
    ids = [r["conversation_id"] for r in res["runs"]]
    assert ids[:2] == ["c-old", "c-new"]  # running first, oldest on top
    assert ids[2] == "c-done"
    assert res["runs"][0]["kind"] == "agent"
    assert res["runs"][1]["kind"] == "chain"
    assert res["runs"][2]["state"] == "done"


@pytest.mark.asyncio
async def test_live_activity_automation(monkeypatch):
    _patch_permission(monkeypatch)
    import time as _t

    scheduler = _FakeScheduler({"t-1": _t.monotonic() - 30.0})
    request = _make_request(
        _triggers_scheduler=scheduler,
        _triggers_store=_FakeTriggersStore({"t-1": "Nightly Ingest"}),
        _backup_job_store=_FakeBackupJobStore([
            {"id": 7, "kind": "scheduled", "status": "running", "triggered_by": "scheduler", "started_at": None},
            {"id": 6, "kind": "manual", "status": "success", "triggered_by": "admin", "started_at": None},
        ]),
        _webdav_source_store=_FakeWebDAVStore([
            {"name": "CCL", "last_sync_status": "running", "updated_at": None},
            {"name": "Other", "last_sync_status": "success", "updated_at": None},
        ]),
    )
    res = await dashboard_live_activity(request)
    assert res["automation"]["triggers"] == [
        {"id": "t-1", "name": "Nightly Ingest", "elapsed_s": pytest.approx(30.0, abs=2.0)}
    ]
    assert res["automation"]["backups"] == [
        {"id": 7, "kind": "scheduled", "triggered_by": "scheduler", "started_at": None}
    ]
    assert res["automation"]["syncs"] == [{"name": "CCL", "since": None}]


@pytest.mark.asyncio
async def test_live_activity_source_failures_isolated(monkeypatch):
    """A broken automation source must not 500 the whole endpoint."""
    _patch_permission(monkeypatch)

    class _BoomStore:
        async def list_jobs(self, limit: int = 50, offset: int = 0):
            raise RuntimeError("db down")

    register_run(_make_run("c-1", is_agent_run=True))
    request = _make_request(_backup_job_store=_BoomStore())
    res = await dashboard_live_activity(request)
    assert [r["conversation_id"] for r in res["runs"]] == ["c-1"]
    assert res["automation"]["backups"] == []


@pytest.mark.asyncio
async def test_live_activity_unregister_removes_run(monkeypatch):
    _patch_permission(monkeypatch)
    run = _make_run("c-1")
    register_run(run)
    unregister_run("c-1", run)
    res = await dashboard_live_activity(_make_request())
    assert res["runs"] == []
    assert get_run("c-1") is None
