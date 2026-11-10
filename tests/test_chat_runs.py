"""Tests for the detached chat-run registry and the resumable SSE view.

These cover the reload-during-answer scenario: the run's producer must keep
going after its SSE subscriber disconnects, and a resuming subscriber must
receive the accumulated snapshot plus the remaining live tokens.
"""

import asyncio
import json
import time

import pytest

from app.chat_runs import ActiveChatRun, _runs, get_run, register_run, unregister_run
from app.routers.chat import _stream_run_events


@pytest.fixture(autouse=True)
def clear_runs():
    _runs.clear()
    yield
    _runs.clear()


def _make_run(conv_id: str = "conv-1") -> ActiveChatRun:
    run = ActiveChatRun(
        conversation_id=conv_id,
        model_name="TestAssistant",
        user_message="hello",
        retrieval_info={},
    )
    register_run(run)
    return run


def _sse_payloads(chunks):
    """Parse SSE `data:` lines from collected chunks into JSON payloads."""
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                payloads.append(json.loads(line[6:]))
    return payloads


# ---------------------------------------------------------------------------
# ActiveChatRun bookkeeping
# ---------------------------------------------------------------------------


def test_record_token_accumulates_buffers():
    run = _make_run()
    run.record_token({"content": "Hel"})
    run.record_token({"content": "lo", "reasoning": "r1"})
    run.record_token({"tool_progress": {"id": "t1", "tool_name": "rag_search", "stage": "started"}})
    run.record_token({"tool_calls": [{"id": "t1", "tool_name": "rag_search"}]})
    run.record_token({"tool_calls": [{"id": "t1", "ok": True, "output": "out"}]})
    run.record_token({"tool_progress": {"id": "t1", "tool_name": "rag_search", "stage": "started"}})
    run.record_token({"subagent": {"id": "s1", "status": "spawned"}})
    run.record_token({"todo_list": {"items": []}})

    snap = run.snapshot()
    assert snap["resumed"] is True
    assert snap["user_message"] == "hello"
    assert snap["content"] == "Hello"
    assert snap["reasoning_content"] == "r1"
    assert snap["tool_calls"] == [
        {"id": "t1", "tool_name": "rag_search", "ok": True, "output": "out"}
    ]
    # duplicate progress events collapse to one buffer entry
    assert snap["tool_progress"] == [
        {"id": "t1", "tool_name": "rag_search", "stage": "started"}
    ]
    assert snap["subagent_events"] == [{"id": "s1", "status": "spawned"}]
    assert snap["todo_list"] == {"items": []}
    assert run.produced_any and run.produced_content


def test_record_token_upserts_subagent_events_by_id():
    """Every event carries the full record dict — the snapshot buffer must
    keep only the latest per subagent, not an append-only log."""
    run = _make_run()
    run.record_token({"subagent": {"id": "s1", "status": "running", "iterations": 1}})
    run.record_token({"subagent": {"id": "s2", "status": "running", "iterations": 1}})
    run.record_token({"subagent": {"id": "s1", "status": "running", "iterations": 4}})
    run.record_token({"subagent": {"id": "s1", "status": "done", "iterations": 6}})

    snap = run.snapshot()
    assert snap["subagent_events"] == [
        {"id": "s1", "status": "done", "iterations": 6},
        {"id": "s2", "status": "running", "iterations": 1},
    ]


def test_record_token_ignores_non_dict():
    run = _make_run()
    run.record_token("plain string")
    assert not run.produced_any
    assert run.assistant_buffer == []


def test_registry_lifecycle_and_ttl(monkeypatch):
    run = _make_run("conv-ttl")
    assert get_run("conv-ttl") is run

    # Not finished -> stays registered
    assert get_run("conv-ttl") is run

    run.finish()
    assert get_run("conv-ttl") is run  # still resumable right after finish

    monkeypatch.setattr("app.chat_runs._RUN_TTL_SECONDS", -1)
    assert get_run("conv-ttl") is None
    assert "conv-ttl" not in _runs

    run2 = _make_run("conv-2")
    unregister_run("conv-2")
    assert get_run("conv-2") is None
    _ = run2


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_marks_run_and_is_idempotent():
    run = _make_run()
    assert not run.cancelled
    assert not run.cancel_event.is_set()
    run.cancel()
    assert run.cancelled
    assert run.cancel_event.is_set()
    run.cancel()  # second cancel is a no-op


def test_unregister_identity_preserves_newer_run():
    """A lagging producer must not delete a newer run that re-registered the
    same conversation id (send-now/cancel races make this common)."""
    run_a = _make_run("conv-race")
    run_b = ActiveChatRun(
        conversation_id="conv-race",
        model_name="TestAssistant",
        user_message="next",
        retrieval_info={},
    )
    register_run(run_b)  # new run wins the registry slot
    unregister_run("conv-race", run_a)  # stale producer finishing late
    assert get_run("conv-race") is run_b
    unregister_run("conv-race", run_b)
    assert get_run("conv-race") is None


def test_unregister_without_run_still_removes():
    run = _make_run("conv-plain")
    unregister_run("conv-plain")
    assert get_run("conv-plain") is None


# ---------------------------------------------------------------------------
# Fan-out / subscriber behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_subscribers_each_receive_all_tokens():
    run = _make_run()
    q1 = run.subscribe()
    q2 = run.subscribe()
    run.broadcast({"content": "a"})
    run.finish()
    for q in (q1, q2):
        assert await q.get() == {"content": "a"}
        assert await q.get() is None


@pytest.mark.asyncio
async def test_subscribe_after_finish_releases_immediately():
    run = _make_run()
    run.finish()
    queue = run.subscribe()
    assert await queue.get() is None


@pytest.mark.asyncio
async def test_unsubscribe_removes_queue():
    run = _make_run()
    queue = run.subscribe()
    run.unsubscribe(queue)
    run.broadcast({"content": "x"})
    run.finish()
    assert queue.empty()


# ---------------------------------------------------------------------------
# SSE view / detachment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_survives_subscriber_disconnect():
    """Cancelling the SSE generator must not stop or lose the run."""
    run = _make_run("conv-disc")

    async def produce():
        for token in ("Hel", "lo"):
            run.broadcast({"content": token})
            await asyncio.sleep(0)
        run.finish()

    producer = asyncio.create_task(produce())

    gen = _stream_run_events(run, resume=False)
    await gen.__anext__()  # role chunk
    await gen.aclose()  # client disconnect / page reload

    await asyncio.wait_for(producer, timeout=5)

    assert run.done
    assert "".join(run.assistant_buffer) == "Hello"
    # Run remains resumable (still registered within the TTL window)
    assert get_run("conv-disc") is run


@pytest.mark.asyncio
async def test_resume_after_finish_replays_snapshot_then_done():
    run = _make_run()
    run.broadcast({"content": "partial answer"})
    run.broadcast({"reasoning": "why"})
    run.finish()

    chunks = [c async for c in _stream_run_events(run, resume=True)]
    text = "".join(chunks)
    payloads = _sse_payloads(chunks)

    deltas = [p["choices"][0]["delta"] for p in payloads]
    snapshot = next(d for d in deltas if d.get("resumed"))
    assert snapshot["content"] == "partial answer"
    assert snapshot["reasoning_content"] == "why"
    assert snapshot["user_message"] == "hello"
    finish = payloads[-1]["choices"][0]["finish_reason"]
    assert finish == "stop"
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_resume_mid_stream_replays_buffer_then_streams_live():
    run = _make_run()
    run.broadcast({"content": "A"})

    gen = _stream_run_events(run, resume=True)
    await gen.__anext__()  # role
    snapshot_chunk = await gen.__anext__()
    snapshot = json.loads(snapshot_chunk[6:])["choices"][0]["delta"]
    assert snapshot["resumed"] is True
    assert snapshot["content"] == "A"

    run.broadcast({"content": "B"})
    live_chunk = await gen.__anext__()
    live_delta = json.loads(live_chunk[6:])["choices"][0]["delta"]
    assert live_delta["content"] == "B"

    run.finish()
    rest = [c async for c in gen]
    assert "data: [DONE]" in "".join(rest)


@pytest.mark.asyncio
async def test_snapshot_and_subscribe_are_gap_free():
    """A token broadcast between subscribe() and the first queue.get must not
    be dropped or duplicated."""
    run = _make_run()
    gen = _stream_run_events(run, resume=True)
    await gen.__anext__()  # role -> subscribe+snapshot happened synchronously
    # Token produced while the subscriber is active but has consumed nothing yet
    run.broadcast({"content": "X"})
    first = await gen.__anext__()
    first_delta = json.loads(first[6:])["choices"][0]["delta"]
    # snapshot (empty at subscribe time) — the live token follows via queue
    if first_delta.get("resumed"):
        nxt = await gen.__anext__()
        nxt_delta = json.loads(nxt[6:])["choices"][0]["delta"]
        assert nxt_delta.get("content") == "X"
    else:
        assert first_delta.get("content") == "X"
    run.finish()
    await gen.aclose()


@pytest.mark.asyncio
async def test_finished_run_emits_trailer_chunks():
    run = _make_run()
    run.retrieval_info["status"] = "ok"
    run.retrieval_info["chunks_retrieved"] = 3
    run.retrieval_info["rag_context_tokens"] = 42
    run.retrieval_info["rag_context"] = [{"content": "ctx"}]
    run.broadcast({"content": "answer"})
    run.finish()

    chunks = [c async for c in _stream_run_events(run, resume=True)]
    payloads = _sse_payloads(chunks)
    deltas = [p["choices"][0]["delta"] for p in payloads]
    retrieval = next(d for d in deltas if "retrieval_info" in d)["retrieval_info"]
    assert retrieval["status"] == "ok"
    assert retrieval["chunks_retrieved"] == 3
    metrics = next(d for d in deltas if "metrics" in d)["metrics"]
    assert metrics["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_producer_error_marks_run_but_stream_completes():
    run = _make_run()
    run.finish(RuntimeError("upstream blew up"))
    assert run.done
    assert isinstance(run.stream_error, RuntimeError)
    chunks = [c async for c in _stream_run_events(run, resume=True)]
    assert "data: [DONE]" in "".join(chunks)


# ---------------------------------------------------------------------------
# Mid-run input injection ("steer")
# ---------------------------------------------------------------------------


def test_submit_input_requires_agent_run():
    run = _make_run()
    assert run.submit_input({"text": "hi"}) is False  # not an agent run
    run.is_agent_run = True
    assert run.submit_input({"text": "hi"}) is True
    assert run.pending_inputs.qsize() == 1


def test_submit_input_rejects_finished_run():
    run = _make_run()
    run.is_agent_run = True
    run.finish()
    assert run.submit_input({"text": "late"}) is False


def test_drain_pending_inputs_empties_queue():
    run = _make_run()
    run.is_agent_run = True
    run.submit_input({"text": "a"})
    run.submit_input({"text": "b"})
    assert run.drain_pending_inputs() == [{"text": "a"}, {"text": "b"}]
    assert run.drain_pending_inputs() == []


def test_record_token_collects_injected_messages_and_snapshot():
    run = _make_run()
    run.record_token({"content": "", "reasoning": "", "injected_user_message": "steer me"})
    run.record_token({"injected_user_message": "again", "injected_source": "code_execution"})
    assert run.injected_messages == [
        {"text": "steer me", "source": ""},
        {"text": "again", "source": "code_execution"},
    ]
    assert run.assistant_buffer == []  # empty content is not buffered
    snap = run.snapshot()
    assert snap["injected_messages"] == run.injected_messages


def test_subagent_manager_idle_seconds_config():
    """The manager takes the stream idle limit from the caller, not from
    per-assistant config — there is no fixed subagent total-run timeout."""
    from agent.subagents import SubagentManager

    mgr = SubagentManager(config={}, conversation_id="c", rag_app=None, idle_seconds=42.0)
    assert mgr.idle_seconds == 42.0

    mgr2 = SubagentManager(config={"timeout_seconds": 2400}, conversation_id="c", rag_app=None)
    # legacy config key is ignored; the default idle window applies
    assert mgr2.idle_seconds == 300.0


def test_record_token_collects_observer_events_and_caps():
    """Observer deltas land in a bounded buffer and survive into the
    resume snapshot; non-dict observer payloads are ignored."""
    run = _make_run()
    run.record_token({"observer": {"kind": "check_started", "check_id": 1, "trigger": ["periodic_iterations"]}})
    run.record_token({"observer": "not-a-dict"})
    run.record_token({"observer": {"kind": "action", "check_id": 1, "nudge": "keep going"}})
    for i in range(60):
        run.record_token({"observer": {"kind": "assessment", "check_id": i + 2}})

    snap = run.snapshot()
    assert len(snap["observer_events"]) == 50
    # oldest events are dropped first; the earliest surviving check_id is 12
    assert snap["observer_events"][0]["check_id"] == 12
    assert snap["observer_events"][-1]["check_id"] == 61
    assert all(isinstance(ev, dict) for ev in snap["observer_events"])


def test_submit_input_sets_input_event():
    run = _make_run()
    run.is_agent_run = True
    assert run.submit_input({"text": "hi"}) is True
    assert run.input_event.is_set()


def test_submit_subagent_input_routes_to_record_queue():
    import asyncio
    from types import SimpleNamespace

    run = _make_run()
    assert run.submit_subagent_input("sub-1", {"text": "hi"}) == "not_agent"
    run.is_agent_run = True
    assert run.submit_subagent_input("sub-1", {"text": "hi"}) == "no_subagent"

    record = SimpleNamespace(
        status="running",
        nudge_queue=asyncio.Queue(),
        input_event=asyncio.Event(),
    )
    run.subagent_channels["sub-1"] = record

    assert run.submit_subagent_input("sub-1", {"text": "hi"}) is None
    assert record.input_event.is_set()
    item = record.nudge_queue.get_nowait()
    assert item["source"] == "user"
    assert item["text"] == "hi"
    assert item["content"].startswith("[User message]")

    record.status = "done"
    assert run.submit_subagent_input("sub-1", {"text": "hi"}) == "subagent_finished"

    run.finish()
    record.status = "running"
    assert run.submit_subagent_input("sub-1", {"text": "hi"}) == "done"


def test_run_chunk_forwards_injected_source():
    from app.routers.chat import _run_chunk_from_token

    run = _make_run()
    chunk = _run_chunk_from_token(
        run,
        {
            "content": "",
            "reasoning": "",
            "injected_user_message": "task done",
            "injected_source": "code_execution",
        },
    )
    delta = json.loads(chunk.splitlines()[0][6:])["choices"][0]["delta"]
    assert delta["injected_user_message"] == "task done"
    assert delta["injected_source"] == "code_execution"
    # user steering (no source) keeps the field absent
    chunk = _run_chunk_from_token(run, {"injected_user_message": "user steer"})
    delta = json.loads(chunk.splitlines()[0][6:])["choices"][0]["delta"]
    assert "injected_source" not in delta


# ---------------------------------------------------------------------------
# live_usage — cumulative token snapshots for the streaming chip
# ---------------------------------------------------------------------------


def test_run_chunk_forwards_live_usage():
    from app.routers.chat import _run_chunk_from_token

    run = _make_run()
    payload = {"llm_calls": 2, "prompt_tokens": 900, "completion_tokens": 120,
               "reasoning_tokens": 30, "total_tokens": 1020, "per_call_completion": [80, 40]}
    chunk = _run_chunk_from_token(run, {"live_usage": payload})
    delta = json.loads(chunk.splitlines()[0][6:])["choices"][0]["delta"]
    assert delta["live_usage"] == payload


def test_record_token_keeps_latest_live_usage_in_snapshot():
    run = _make_run()
    run.record_token({"live_usage": {"llm_calls": 1, "total_tokens": 100}})
    run.record_token({"live_usage": {"llm_calls": 2, "total_tokens": 250}})
    assert run.live_usage == {"llm_calls": 2, "total_tokens": 250}
    snap = run.snapshot()
    assert snap["live_usage"] == {"llm_calls": 2, "total_tokens": 250}


def test_live_usage_alone_produces_a_chunk():
    from app.routers.chat import _run_chunk_from_token

    run = _make_run()
    # A token carrying only live_usage must not be dropped by the early-return guard.
    chunk = _run_chunk_from_token(run, {"live_usage": {"llm_calls": 1}})
    assert chunk is not None
    delta = json.loads(chunk.splitlines()[0][6:])["choices"][0]["delta"]
    assert delta["live_usage"] == {"llm_calls": 1}
