"""Tests for agent.observer (RunObserver) and its AgentRunner integration."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.observer import (
    FIRST_CHECK_ITERATION,
    OBSERVER_PROMPT,
    ObserverConfig,
    ObserverDecision,
    ObserverRunSnapshot,
    RunObserver,
)
from agent.tools import ToolRegistry
from agent.tools.builtin_tools.todo_list import TodoListState
from agent.tools.runner import AgentRunner


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _snap(**overrides: Any) -> ObserverRunSnapshot:
    defaults: Dict[str, Any] = dict(
        iteration=2,
        max_iterations=10,
        tool_calls_executed=1,
        max_tool_calls=10,
        elapsed_s=20.0,
        recent_records=[],
        todo_version=0,
        todo_render="",
        open_tasks=0,
        last_ai_text="",
        reasoning_tail="",
        input_text="do something",
        system_excerpt="you are helpful",
        injected_since_last=0,
        todo_available=True,
    )
    defaults.update(overrides)
    return ObserverRunSnapshot(**defaults)


class _ObsResponse:
    def __init__(self, content: str, usage: Optional[Dict[str, Any]] = None) -> None:
        self.content = content
        self.response_metadata = {"token_usage": usage or {}}


class _ObsLLM:
    """Observer LLM stub: returns scripted contents via ainvoke."""

    def __init__(self, contents: Optional[List[str]] = None) -> None:
        self.contents = list(contents or [])
        self.calls: List[Any] = []
        self.model_name = "obs-model"

    async def ainvoke(self, messages: List[Any]) -> _ObsResponse:
        self.calls.append(messages)
        await asyncio.sleep(0)  # let the task scheduling behave like real IO
        content = self.contents.pop(0) if self.contents else '{"assessment": "ok"}'
        return _ObsResponse(content, {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})


def _observer(cfg: Optional[Dict[str, Any]] = None, llm: Optional[_ObsLLM] = None) -> RunObserver:
    obs_llm = llm or _ObsLLM()
    return RunObserver(
        ObserverConfig.from_dict(cfg or {"enabled": True}),
        lambda: asyncio.sleep(0, result=obs_llm),
        assistant_name="TestAssistant",
    )


# ---------------------------------------------------------------------------
# ObserverConfig
# ---------------------------------------------------------------------------


def test_config_defaults_and_bounds():
    cfg = ObserverConfig.from_dict({"enabled": True})
    assert cfg.enabled is True
    assert cfg.every_iterations == 4
    assert cfg.every_seconds == 180.0
    assert cfg.max_checks == 8
    assert cfg.max_nudges == 4
    assert cfg.check_timeout_seconds == 60.0
    assert cfg.include_subagents is True
    assert cfg.assistant is None

    clamped = ObserverConfig.from_dict({
        "enabled": True,
        "every_iterations": 999,
        "every_seconds": 1,
        "max_checks": 0,
        "max_nudges": 5000,
        "check_timeout_seconds": 99999,
        "assistant": "  obs-helper ",
    })
    assert clamped.every_iterations == 20
    assert clamped.every_seconds == 30.0
    assert clamped.max_checks == 1
    assert clamped.max_nudges == 20
    assert clamped.check_timeout_seconds == 300.0
    assert clamped.assistant == "obs-helper"


def test_config_from_none_gives_disabled():
    cfg = ObserverConfig.from_dict(None)
    assert cfg.enabled is False


# ---------------------------------------------------------------------------
# evaluate_triggers
# ---------------------------------------------------------------------------


def test_trigger_periodic_iterations():
    obs = _observer({"enabled": True, "every_iterations": 3})
    reasons = obs.evaluate_triggers(_snap(iteration=3, tool_calls_executed=0))
    assert "periodic_iterations" in reasons


def test_trigger_not_before_first_check_iteration():
    obs = _observer({"enabled": True, "every_iterations": 1})
    # tick() gates on FIRST_CHECK_ITERATION even when evaluate would fire
    assert obs.tick(_snap(iteration=FIRST_CHECK_ITERATION - 1)) is None


@pytest.mark.asyncio
async def test_trigger_repeat_tool_calls_and_latch():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    records = [{"tool_name": "search", "args": {"q": "x"}, "ok": True}] * 3
    snap = _snap(iteration=5, recent_records=records)
    assert "repeat_tool_calls" in obs.evaluate_triggers(snap)

    ev = obs.tick(snap)
    assert ev is not None and "repeat_tool_calls" in ev["trigger"]
    await obs.aclose()  # pending task is gone but the latch remains
    # same condition still holds -> latched, does not refire
    assert "repeat_tool_calls" not in obs.evaluate_triggers(snap)
    # different call breaks the streak -> latch re-arms
    snap2 = _snap(iteration=6, recent_records=[{"tool_name": "other", "args": {}, "ok": True}])
    obs.evaluate_triggers(snap2)
    assert "repeat_tool_calls" not in obs._latches


def test_repeat_tool_calls_ignores_coordination_tools():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    records = [{"tool_name": "wait", "args": {"seconds": 5}, "ok": True}] * 5
    snap = _snap(iteration=5, recent_records=records, tool_calls_executed=0)
    assert "repeat_tool_calls" not in obs.evaluate_triggers(snap)


def test_trigger_error_streak():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    records = [{"tool_name": "t", "args": {}, "ok": False}] * 3
    snap = _snap(iteration=4, recent_records=records)
    assert "error_streak" in obs.evaluate_triggers(snap)
    # two errors + success at the tail -> no streak
    snap_ok = _snap(iteration=4, recent_records=records[:2] + [{"tool_name": "t", "args": {}, "ok": True}])
    obs._latches.clear()
    assert "error_streak" not in obs.evaluate_triggers(snap_ok)


@pytest.mark.asyncio
async def test_trigger_heuristic_cooldown():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    records = [{"tool_name": "t", "args": {}, "ok": False}] * 3
    snap = _snap(iteration=4, recent_records=records)
    obs.tick(snap)
    await obs.aclose()
    obs._latches.clear()  # force re-fire eligibility
    # cooldown blocks immediate re-fire of heuristic reasons
    assert "error_streak" not in obs.evaluate_triggers(snap)
    obs._last_heuristic_at = 0.0  # simulate elapsed cooldown
    assert "error_streak" in obs.evaluate_triggers(snap)


def test_trigger_no_todo_and_stale_todo():
    obs = _observer({"enabled": True, "every_iterations": 2, "every_seconds": 9999})
    snap = _snap(iteration=4, tool_calls_executed=2, todo_version=0)
    assert "no_todo" in obs.evaluate_triggers(snap)

    obs2 = _observer({"enabled": True, "every_iterations": 2, "every_seconds": 9999})
    snap_v1 = _snap(iteration=3, tool_calls_executed=3, todo_version=1, open_tasks=2)
    obs2.evaluate_triggers(snap_v1)  # registers version 1 at iteration 3
    snap_stale = _snap(iteration=6, tool_calls_executed=5, todo_version=1, open_tasks=2)
    assert "stale_todo" in obs2.evaluate_triggers(snap_stale)


def test_trigger_budget_pressure():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    snap = _snap(iteration=9, max_iterations=10, tool_calls_executed=1)
    assert "budget_pressure" in obs.evaluate_triggers(snap)


def test_trigger_user_input_injected():
    obs = _observer({"enabled": True, "every_iterations": 99, "every_seconds": 9999})
    snap = _snap(iteration=4, injected_since_last=1, tool_calls_executed=0)
    assert "user_input_injected" in obs.evaluate_triggers(snap)


@pytest.mark.asyncio
async def test_tick_respects_max_checks_and_pending():
    obs = _observer({"enabled": True, "every_iterations": 1, "max_checks": 1})
    ev = obs.tick(_snap(iteration=3))
    assert ev is not None
    await obs.aclose()
    assert obs.tick(_snap(iteration=4)) is None  # budget exhausted


# ---------------------------------------------------------------------------
# decision parsing / applying
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_check_parses_strict_json():
    payload = {
        "assessment": "looping",
        "todo_ops": [{"action": "add", "title": "t fix"}],
        "nudge": "stop repeating",
    }
    llm = _ObsLLM([json.dumps(payload)])
    obs = _observer(llm=llm)
    dec = await obs._run_check(_snap(), ["periodic_iterations"])
    assert dec.assessment == "looping"
    assert dec.nudge == "stop repeating"
    assert dec.todo_ops[0]["action"] == "add"
    assert obs.usage["total_tokens"] == 5
    # prompt carries the supervisor system prompt + digest human message
    assert OBSERVER_PROMPT.splitlines()[0] in llm.calls[0][0].content


@pytest.mark.asyncio
async def test_run_check_tolerates_garbage_and_fenced_json():
    llm = _ObsLLM(["not json at all", '```json\n{"assessment": "fenced ok", "nudge": null}\n```'])
    obs = _observer(llm=llm)
    dec = await obs._run_check(_snap(), [])
    assert dec.todo_ops == [] and dec.nudge is None and dec.assessment
    dec2 = await obs._run_check(_snap(), [])
    assert dec2.assessment == "fenced ok"


@pytest.mark.asyncio
async def test_run_check_failure_is_contained():
    class FailingLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("observer llm down")

    obs = RunObserver(
        ObserverConfig.from_dict({"enabled": True}),
        lambda: asyncio.sleep(0, result=FailingLLM()),
        "A",
    )
    dec = await obs._run_check(_snap(), ["repeat_tool_calls"])
    assert "observer check failed" in dec.assessment
    assert dec.nudge is None


def test_apply_decision_todo_ops_and_nudge():
    obs = _observer({"enabled": True})
    todo = TodoListState()
    todo.add("first task")
    dec = ObserverDecision(
        assessment="needs adjustments",
        todo_ops=[
            {"action": "update", "task_id": "t1", "status": "completed"},
            {"action": "add", "title": "new task", "note": "why"},
            {"action": "remove", "task_id": "t99"},          # unknown id -> skipped
            {"action": "frobnicate", "task_id": "t1"},        # unknown action -> skipped
        ],
        nudge="try narrower queries",
        trigger_reasons=["error_streak"],
        iteration=3,
        check_ms=42.0,
        model="obs-model",
    )
    events, nudge = obs.apply_decision(dec, todo)
    assert nudge == "[Observer note #1] try narrower queries"
    assert len(events) == 1 and events[0]["kind"] == "action"
    ops = events[0]["todo_ops_applied"]
    assert ops[0]["ok"] and ops[0]["task_id"] == "t1"
    assert ops[1]["ok"] and ops[1]["title"] == "new task"
    assert ops[2]["ok"] is False  # unknown id
    assert ops[3]["ok"] is False  # unknown action
    tasks = {t["id"]: t for t in todo.tasks}
    assert tasks["t1"]["status"] == "completed"
    assert "new task" in {t["title"] for t in todo.tasks}


def test_apply_decision_nudge_cap_and_assessment_kind():
    obs = _observer({"enabled": True, "max_nudges": 1})
    dec = ObserverDecision(assessment="a", nudge="hint one")
    _, nudge1 = obs.apply_decision(dec, None)
    assert nudge1 is not None
    _, nudge2 = obs.apply_decision(dec, None)
    assert nudge2 is None  # cap reached
    # assessment-only decision yields kind="assessment"
    events, _ = obs.apply_decision(ObserverDecision(assessment="quiet"), None)
    assert events[0]["kind"] == "assessment"


@pytest.mark.asyncio
async def test_collect_nonblocking_and_aclose_cancels():
    obs = _observer({"enabled": True})
    assert obs.collect() is None  # nothing pending
    obs._pending = asyncio.ensure_future(asyncio.sleep(60))
    assert obs.collect() is None  # still running -> None, non-blocking
    await obs.aclose()
    assert obs._pending is None


# ---------------------------------------------------------------------------
# AgentRunner integration
# ---------------------------------------------------------------------------


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    tool = MagicMock()
    tool.execute = AsyncMock(return_value="tool result")
    tool.enabled = True
    tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "mock_tool",
            "description": "mock",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
    reg.register_builtin("mock_tool", tool)
    return reg


class _ScriptedLLM:
    def __init__(self, responses: List[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: List[List[Any]] = []

    def bind_tools(self, schemas):
        return self

    async def astream(self, messages):
        self.calls.append(messages)
        yield self.responses.pop(0)

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="final")


def _call(name: str, call_id: str) -> Dict[str, Any]:
    return {"name": name, "args": {"query": call_id}, "id": call_id}


@pytest.mark.asyncio
async def test_runner_applies_observer_decision_and_streams_events():
    """At an iteration boundary the observer nudge is injected as a
    HumanMessage, todo ops mutate the run list, and observer events stream."""
    reg = _registry()
    llm = _ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[_call("mock_tool", "c1")]),
            AIMessage(content="", tool_calls=[_call("mock_tool", "c2")]),
            AIMessage(content="", tool_calls=[_call("mock_tool", "c3")]),
            AIMessage(content="", tool_calls=[_call("mock_tool", "c4")]),
            AIMessage(content="all done"),
            # The observer-added open todo blocks the first finalize attempt;
            # the last iteration (max) is allowed to end anyway.
            AIMessage(content="all done"),
        ]
    )
    obs_llm = _ObsLLM(
        [
            json.dumps(
                {
                    "assessment": "progressing but plan missing",
                    "todo_ops": [{"action": "add", "title": "verify result"}],
                    "nudge": "summarize findings before finishing",
                }
            )
        ]
    )
    observer = _observer({"enabled": True, "every_iterations": 2}, llm=obs_llm)
    runner = AgentRunner(reg, ["mock_tool", "agent_todo"], max_iterations=6, observer=observer)

    chunks = []
    async for chunk in runner.run_streaming(llm, "task", [], system_instructions="sys"):
        chunks.append(chunk)

    observer_chunks = [c["observer"] for c in chunks if c.get("observer")]
    assert any(ev["kind"] == "check_started" for ev in observer_chunks)
    action_events = [ev for ev in observer_chunks if ev["kind"] == "action"]
    assert action_events and action_events[0]["nudge"] == "summarize findings before finishing"
    # todo_list streamed because the observer mutated it
    todo_chunks = [c["todo_list"] for c in chunks if c.get("todo_list")]
    assert todo_chunks and any(
        t["title"] == "verify result" for t in todo_chunks[-1]["tasks"]
    )
    # The open task held the first finalize attempt back (todo gate)
    assert any(
        "Final answer held back" in (c.get("reasoning") or "") for c in chunks
    )
    # the nudge reached the LLM as a HumanMessage in a later iteration
    nudge_seen = any(
        isinstance(m, HumanMessage) and str(m.content).startswith("[Observer note")
        for call_msgs in llm.calls
        for m in call_msgs
    )
    assert nudge_seen
    # system prompt announces the observer to the agent
    first_system = next(
        str(m.content) for call_msgs in llm.calls for m in call_msgs if m.__class__.__name__ == "SystemMessage"
    )
    assert "[Run observer]" in first_system
    # trace captured the events
    trace = runner._last_trace.to_dict()
    assert len(trace["observer_events"]) >= 2


@pytest.mark.asyncio
async def test_runner_without_observer_is_unchanged():
    reg = _registry()
    llm = _ScriptedLLM([AIMessage(content="answer")])
    runner = AgentRunner(reg, ["mock_tool"])
    chunks = [c async for c in runner.run_streaming(llm, "hi", [], system_instructions="sys")]
    assert not any(c.get("observer") for c in chunks)
    first_system = next(
        str(m.content) for call_msgs in llm.calls for m in call_msgs if m.__class__.__name__ == "SystemMessage"
    )
    assert "[Run observer]" not in first_system


@pytest.mark.asyncio
async def test_runner_observer_llm_failure_does_not_break_run():
    reg = _registry()
    llm = _ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[_call("mock_tool", "c1")]),
            AIMessage(content="", tool_calls=[_call("mock_tool", "c2")]),
            AIMessage(content="done"),
        ]
    )

    class FailingObsLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("down")

    observer = RunObserver(
        ObserverConfig.from_dict({"enabled": True, "every_iterations": 2}),
        lambda: asyncio.sleep(0, result=FailingObsLLM()),
        "A",
    )
    runner = AgentRunner(reg, ["mock_tool"], max_iterations=4, observer=observer)
    chunks = [c async for c in runner.run_streaming(llm, "task", [], system_instructions="sys")]
    content = "".join(c.get("content", "") for c in chunks)
    assert "done" in content


def test_trigger_no_todo_waits_for_first_periodic_cadence():
    """A "no plan" nudge before the agent had a fair chance is noise — the
    no_todo heuristic only activates once `every_iterations` is reached,
    while hard stalls (error_streak) may fire earlier."""
    obs = _observer({"enabled": True, "every_iterations": 4, "every_seconds": 9999})
    early = _snap(iteration=2, tool_calls_executed=1, todo_version=0)
    assert "no_todo" not in obs.evaluate_triggers(early)
    early3 = _snap(iteration=3, tool_calls_executed=2, todo_version=0)
    assert "no_todo" not in obs.evaluate_triggers(early3)
    due = _snap(iteration=4, tool_calls_executed=2, todo_version=0)
    assert "no_todo" in obs.evaluate_triggers(due)

    # hard stalls unaffected: three consecutive errors still fire early
    obs2 = _observer({"enabled": True, "every_iterations": 4, "every_seconds": 9999})
    errs = [
        {"tool_name": "x", "ok": False, "error": "e", "args": {}}
        for _ in range(3)
    ]
    snap_err = _snap(iteration=2, tool_calls_executed=3, recent_records=errs)
    assert "error_streak" in obs2.evaluate_triggers(snap_err)


# ---------------------------------------------------------------------------
# Final review (end-of-run plan check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_streaming_emits_final_todo_snapshot_after_sweep():
    """End-to-end: a run with an open todo emits a final snapshot whose
    leftover task is 'skipped'."""
    runner = AgentRunner(
        _registry(),
        ["mock_tool", "agent_todo"],
        max_iterations=3,
    )
    add_call = AIMessage(
        content="",
        tool_calls=[{
            "id": "call-1", "name": "agent_todo",
            "args": {"action": "add", "title": "never finished"},
        }],
    )
    # iter2 "done" is blocked by the todo gate; iter3 == max_iterations is
    # allowed to finalize and the leftover task is swept to 'skipped'.
    chunks = [
        chunk async for chunk in runner.run_streaming(
            _ScriptedLLM([
                add_call,
                AIMessage(content="done"),
                AIMessage(content="done"),
            ]),
            "plan something",
            [],
            "Instructions",
        )
    ]
    snapshots = [c["todo_list"] for c in chunks if c.get("todo_list")]
    assert snapshots, "expected todo_list chunks"
    final = snapshots[-1]
    assert final["tasks"][0]["status"] == "skipped"
