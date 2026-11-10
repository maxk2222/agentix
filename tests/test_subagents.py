from __future__ import annotations

import asyncio
import time

import pytest

from agent.subagents import SubagentManager, SubagentRecord, make_subagent_state
from agent.tools.builtin_tools.wait import WaitTool
from agent.tools.builtin_tools.nudge_subagent import NudgeSubagentTool
from agent.tools import ToolRegistry
from agent.tools.runner import AgentRunner
from agent.tools.builtin_tools.spawn_subagent import SpawnSubagentTool
from agent.tools.builtin_tools.wait_subagent import WaitSubagentTool
from agent.tools.builtin_tools.subagent_status import SubagentStatusTool
from langchain_core.messages import AIMessage, HumanMessage
from agent.application import DEFAULT_SUBAGENT_EXCLUDE_TOOLS, _resolve_allowed_tools
from agent.subagents import SubagentState


class FakeRAG:
    ready = True

    async def run_streaming(
        self, user_input, *, chat_history, model_name, retrieval_info,
        conversation_id, subagent_state,
        injected_input_queue=None, input_event=None,
    ):
        if user_input == "parent":
            await subagent_state.manager.spawn(
                task="grandchild", label="Grandchild", owner=subagent_state
            )
            await subagent_state.manager.wait(None, subagent_state)
        else:
            await asyncio.sleep(0.01)
        yield {"content": f"answer: {user_input}", "reasoning": ""}
        yield {
            "content": "",
            "reasoning": "",
            "tool_calls": [{
                "tool_name": "rag_search",
                "args": {"query": user_input},
                "ok": True,
                "duration_ms": 12,
                "output": "found",
            }],
        }
        retrieval_info["_llm_usage_metadata"] = {"total_tokens": 3}
        retrieval_info["_agent_trace"] = {"tool_calls": []}


@pytest.mark.asyncio
async def test_subagent_spawn_wait_returns_answer_and_activity():
    manager = SubagentManager(
        {"max_concurrent": 2, "max_depth": 1, "max_total": 2},
        "conversation",
        FakeRAG(),
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="investigate", label="Investigation", owner=state)

    result = await manager.wait([record.id], state)

    assert record.status == "done"
    assert "answer: investigate" in result["content"]
    assert "rag_search" in result["content"]
    assert result["metadata"]["subagent_results"][0]["id"] == record.id
    assert record.collected
    assert record.nested_usage == {"total_tokens": 3}


@pytest.mark.asyncio
async def test_subagent_todo_tool_activity_is_not_exposed_to_parent():
    class TodoRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            call = {
                "tool_name": "agent_todo",
                "args": {"action": "add", "title": "Private child task"},
                "ok": True,
                "output": "Todo list - 1 open, 0 done",
            }
            yield {"tool_calls": [call]}
            retrieval_info["_agent_trace"] = {"tool_calls": [call]}
            yield {"content": "Child answer"}

    manager = SubagentManager({}, "conversation", TodoRAG())
    root = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="child task", label="Child", owner=root)
    await manager.wait([record.id], root)

    assert record.status == "done"
    assert all(item["tool_name"] != "agent_todo" for item in record.activity)
    assert all(call["tool_name"] != "agent_todo" for call in record.nested_tool_calls)


@pytest.mark.asyncio
async def test_subagent_slot_is_released_while_waiting_for_nested_subagent():
    manager = SubagentManager(
        {"max_concurrent": 1, "max_depth": 2, "max_total": 3},
        "conversation",
        FakeRAG(),
    )
    state = make_subagent_state(manager, "Analyst")
    parent = await manager.spawn(task="parent", label="Parent", owner=state)

    result = await manager.wait([parent.id], state)

    assert parent.status == "done"
    assert "answer: parent" in result["content"]
    assert any(record.parent_id == parent.id and record.status == "done" for record in manager.records.values())


@pytest.mark.asyncio
async def test_spawn_enforces_depth_and_total_budget():
    manager = SubagentManager(
        {"max_depth": 1, "max_total": 1},
        "conversation",
        FakeRAG(),
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="one", label=None, owner=state)

    with pytest.raises(ValueError, match="budget"):
        await manager.spawn(task="two", label=None, owner=state)
    child_state = state.child(record.id)
    with pytest.raises(ValueError, match="depth"):
        await manager.spawn(task="nested", label=None, owner=child_state)
    await manager.wait(None, state)


@pytest.mark.asyncio
async def test_wait_tool_clamps_delay():
    tool = WaitTool()
    result = await tool.execute(seconds=0, context={"tool_config": {"max_seconds": 2}})
    assert result == "Waited 0.1s."


def test_status_is_scoped_to_own_subagents():
    manager = SubagentManager({}, None, FakeRAG())
    root = make_subagent_state(manager, "Analyst")
    other = root.child("sub-other")
    assert manager.status(None, root)["content"] == "No matching subagents were spawned by this agent."
    assert manager.status(None, other)["metadata"]["subagent_results"] == []


def test_subagents_inherit_full_parent_tool_set():
    """Subagents must get every tool the parent has (incl. web_search,
    web_scraper and mcp__*) minus the small default exclusion list."""
    tools = {
        "web_search": {}, "web_scraper": {}, "rag_search": {},
        "mcp__drive__read": {}, "code_execution": {}, "email_sender": {},
        "agent_memory": {},
    }
    parent = _resolve_allowed_tools(tools, {}, None)
    assert "agent_memory" in parent  # root keeps everything

    child_state = SubagentState(
        manager=None, depth=1, is_root=False,
        assistant_name="Analyst", owner_key="sub-1", parent_subagent_id="sub-1",
    )
    child = _resolve_allowed_tools(tools, {}, child_state)
    assert set(child) == set(tools) - DEFAULT_SUBAGENT_EXCLUDE_TOOLS
    assert "agent_memory" not in child  # child-conv memory is unreadable anyway
    assert "mcp__drive__read" in child
    assert "code_execution" in child


def test_subagent_exclude_tools_override():
    child_state = SubagentState(
        manager=None, depth=1, is_root=False,
        assistant_name="Analyst", owner_key="sub-1", parent_subagent_id="sub-1",
    )
    tools = {"web_search": {}, "email_sender": {}, "agent_memory": {}}
    # Explicit list replaces the defaults entirely.
    child = _resolve_allowed_tools(
        tools, {"exclude_tools": ["email_sender"]}, child_state
    )
    assert child == ["web_search", "agent_memory"]
    # An explicitly empty list means "exclude nothing".
    child = _resolve_allowed_tools(tools, {"exclude_tools": []}, child_state)
    assert set(child) == set(tools)


@pytest.mark.asyncio
async def test_wait_first_mode_returns_after_earliest_completion():
    class SlowFastRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            delay = 0.4 if user_input == "slow" else 0.01
            await asyncio.sleep(delay)
            yield {"content": f"answer: {user_input}", "reasoning": ""}

    manager = SubagentManager({"max_concurrent": 2}, "conversation", SlowFastRAG())
    state = make_subagent_state(manager, "Analyst")
    slow = await manager.spawn(task="slow", label="Slow", owner=state)
    fast = await manager.spawn(task="fast", label="Fast", owner=state)

    result = await manager.wait(None, state, mode="first")

    assert fast.status == "done"
    assert "answer: fast" in result["content"]
    assert "answer: slow" not in result["content"]
    assert slow.status in {"queued", "running"}
    assert slow.id in result["content"]  # listed under "Still running"
    assert fast.collected and not slow.collected

    # Subsequent wait collects the rest.
    result = await manager.wait(None, state, mode="all")
    assert "answer: slow" in result["content"]


@pytest.mark.asyncio
async def test_wait_first_mode_already_finished_returns_immediately():
    manager = SubagentManager({}, "conversation", FakeRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="done", label=None, owner=state)
    await asyncio.wait_for(manager.tasks[record.id], timeout=2)

    result = await manager.wait(None, state, mode="first")
    assert record.id in result["content"] or "answer: done" in result["content"]
    assert record.collected


@pytest.mark.asyncio
async def test_agent_iteration_field_beats_prose_marker():
    class IterRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            # Model writes "Iteration 2025..." (a year) — must not be parsed.
            yield {"content": "", "reasoning": "thinking about **Iteration 2025...**"}
            # Backend's structured field is authoritative.
            yield {"content": "", "reasoning": "", "agent_iteration": 6}
            yield {"content": "done", "reasoning": ""}

    manager = SubagentManager({}, "conversation", IterRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)
    await manager.wait([record.id], state)

    assert record.iterations == 6


@pytest.mark.asyncio
async def test_prose_marker_fallback_parses_only_small_counters():
    class MarkerRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            yield {"content": "", "reasoning": "\n\n**Iteration 3...**\n\n"}
            # 4-digit "2025" (year in model text) is rejected by the pattern.
            yield {"content": "", "reasoning": "**Iteration 2025...** more"}
            # A marker mid-chunk (model echo) is not the backend's own chunk.
            yield {"content": "", "reasoning": "note to self **Iteration 5...**"}
            yield {"content": "done", "reasoning": ""}

    manager = SubagentManager({}, "conversation", MarkerRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)
    await manager.wait([record.id], state)

    assert record.iterations == 3


def test_to_dict_caps_activity_log():
    record = SubagentRecord(
        id="s1", parent_id=None, owner_key="root", task="t", label="t",
        assistant="a", depth=1,
    )
    record.activity = [{"tool_name": f"t{i}"} for i in range(150)]
    data = record.to_dict()
    assert len(data["activity"]) == 100
    assert data["activity_total"] == 150
    assert data["activity"][-1]["tool_name"] == "t149"


def test_to_dict_includes_and_caps_observer_events():
    record = SubagentRecord(
        id="s1", parent_id=None, owner_key="root", task="t", label="t",
        assistant="a", depth=1,
    )
    record.observer_events = [{"kind": "assessment", "check_id": i} for i in range(30)]
    data = record.to_dict()
    assert len(data["observer_events"]) == 20
    assert data["observer_events"][0]["check_id"] == 10
    assert data["observer_events"][-1]["check_id"] == 29


@pytest.mark.asyncio
async def test_root_runner_multiplexes_subagent_events():
    manager = SubagentManager({}, None, FakeRAG())
    state = make_subagent_state(manager, "Analyst")
    runner = AgentRunner(ToolRegistry(), [], subagent_state=state)

    async def fake_inner(*args, **kwargs):
        await asyncio.sleep(0.01)
        yield {"content": "answer", "reasoning": ""}

    runner._run_streaming_inner = fake_inner
    await manager.event_queue.put({"id": "sub-1", "stage": "started", "status": "running"})
    chunks = [
        chunk async for chunk in runner.run_streaming(None, "question", [], "instructions")
    ]
    assert any(chunk.get("subagent", {}).get("id") == "sub-1" for chunk in chunks)
    assert any(chunk.get("content") == "answer" for chunk in chunks)


@pytest.mark.asyncio
async def test_agent_runner_spawn_wait_and_persist_subagent_result():
    class ScriptedLLM:
        def __init__(self):
            self.responses = [
                AIMessage(content="", tool_calls=[{
                    "name": "spawn_subagent",
                    "args": {"task": "research topic", "label": "Research"},
                    "id": "spawn-call",
                }]),
                AIMessage(content="", tool_calls=[{
                    "name": "wait_subagent",
                    "args": {},
                    "id": "wait-call",
                }]),
                AIMessage(content="Synthesized answer."),
            ]

        def bind_tools(self, schemas):
            return self

        async def astream(self, messages):
            yield self.responses.pop(0)

    rag = FakeRAG()
    manager = SubagentManager({"max_depth": 1}, "conversation", rag)
    state = make_subagent_state(manager, "Analyst")
    registry = ToolRegistry()
    registry.register_builtin("spawn_subagent", SpawnSubagentTool(rag))
    registry.register_builtin("wait_subagent", WaitSubagentTool())
    registry.register_builtin("subagent_status", SubagentStatusTool())
    runner = AgentRunner(
        registry,
        ["spawn_subagent", "wait_subagent", "subagent_status"],
        supports_native_tool_calls=True,
        assistant_name="Analyst",
        subagent_state=state,
    )

    result = await runner.run(ScriptedLLM(), "question", [], "instructions")

    assert result["answer"] == "Synthesized answer."
    spawn_record = next(
        call for call in result["agent_trace"]["tool_calls"]
        if call["tool_name"] == "spawn_subagent"
    )
    assert spawn_record["metadata"]["subagent"]["status"] == "done"
    assert "answer: research topic" in spawn_record["output"]


@pytest.mark.asyncio
async def test_agent_runner_auto_collects_before_final_response():
    class ScriptedLLM:
        def __init__(self):
            self.responses = [
                AIMessage(content="", tool_calls=[{
                    "name": "spawn_subagent",
                    "args": {"task": "research topic"},
                    "id": "spawn-call",
                }]),
                AIMessage(content="Premature answer."),
                AIMessage(content="Final answer with subagent results."),
            ]
            self.messages = []

        def bind_tools(self, schemas):
            return self

        async def astream(self, messages):
            self.messages.append(messages)
            yield self.responses.pop(0)

    rag = FakeRAG()
    manager = SubagentManager({"max_depth": 1}, "conversation", rag)
    state = make_subagent_state(manager, "Analyst")
    registry = ToolRegistry()
    registry.register_builtin("spawn_subagent", SpawnSubagentTool(rag))
    runner = AgentRunner(
        registry,
        ["spawn_subagent"],
        supports_native_tool_calls=True,
        assistant_name="Analyst",
        subagent_state=state,
    )

    result = await runner.run(ScriptedLLM(), "question", [], "instructions")

    assert result["answer"] == "Final answer with subagent results."
    assert "Premature answer." not in result["answer"]


@pytest.mark.asyncio
async def test_max_iteration_finalization_collects_spawned_subagents():
    class ScriptedLLM:
        def __init__(self):
            self.responses = [
                AIMessage(content="", tool_calls=[{
                    "name": "spawn_subagent",
                    "args": {"task": "research topic"},
                    "id": "spawn-call",
                }]),
                AIMessage(content="Forced final answer."),
            ]
            self.messages = []

        def bind_tools(self, schemas):
            return self

        async def astream(self, messages):
            self.messages.append(messages)
            yield self.responses.pop(0)

    rag = FakeRAG()
    manager = SubagentManager({"max_depth": 1}, "conversation", rag)
    state = make_subagent_state(manager, "Analyst")
    registry = ToolRegistry()
    registry.register_builtin("spawn_subagent", SpawnSubagentTool(rag))
    runner = AgentRunner(
        registry,
        ["spawn_subagent"],
        max_iterations=1,
        supports_native_tool_calls=True,
        assistant_name="Analyst",
        subagent_state=state,
    )

    result = await runner.run(ScriptedLLM(), "question", [], "instructions")

    assert result["answer"] == "Forced final answer."
    assert "answer: research topic" in str(runner._last_trace.to_dict())


class LongAnswerRAG:
    ready = True
    ANSWER = "x" * 10_000

    async def run_streaming(
        self, user_input, *, chat_history, model_name, retrieval_info,
        conversation_id, subagent_state,
        injected_input_queue=None, input_event=None,
    ):
        yield {"content": self.ANSWER, "reasoning": ""}


@pytest.mark.asyncio
async def test_wait_report_includes_full_long_answer():
    manager = SubagentManager({}, "conversation", LongAnswerRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="task", label=None, owner=state)

    result = await manager.wait([record.id], state)

    assert LongAnswerRAG.ANSWER in result["content"]
    assert "answer truncated" not in result["content"]


@pytest.mark.asyncio
async def test_truncated_report_points_to_full_answer_fetch():
    manager = SubagentManager(
        {"report_answer_chars": 5_000}, "conversation", LongAnswerRAG()
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="task", label=None, owner=state)

    result = await manager.wait([record.id], state)

    assert LongAnswerRAG.ANSWER not in result["content"]
    assert (
        f'subagent_status(subagent_id="{record.id}", full_answer=true)'
        in result["content"]
    )

    tool = SubagentStatusTool()
    preview = await tool.execute(
        subagent_id=record.id, context={"subagent_state": state}
    )
    assert "answer truncated" in preview["content"]
    assert LongAnswerRAG.ANSWER not in preview["content"]

    full = await tool.execute(
        subagent_id=record.id,
        full_answer=True,
        context={"subagent_state": state},
    )
    assert LongAnswerRAG.ANSWER in full["content"]


@pytest.mark.asyncio
async def test_wait_tool_wakes_on_input_event():
    event = asyncio.Event()
    queue: asyncio.Queue = asyncio.Queue()
    tool = WaitTool()

    async def set_later() -> None:
        await asyncio.sleep(0.1)
        event.set()

    asyncio.create_task(set_later())
    started = time.monotonic()
    result = await tool.execute(
        seconds=5,
        context={"injected_input_queue": queue, "input_event": event},
    )
    assert time.monotonic() - started < 2
    assert result.startswith("Interrupted by new input")
    assert "planned 5s" in result


@pytest.mark.asyncio
async def test_wait_tool_returns_immediately_on_prefilled_queue():
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"text": "hi"})
    event = asyncio.Event()
    result = await WaitTool().execute(
        seconds=5,
        context={"injected_input_queue": queue, "input_event": event},
    )
    assert result.startswith("Interrupted by pending input")


@pytest.mark.asyncio
async def test_wait_tool_plain_sleep_without_context():
    started = time.monotonic()
    result = await WaitTool().execute(seconds=0.1)
    assert result == "Waited 0.1s."
    assert time.monotonic() - started >= 0.09


@pytest.mark.asyncio
async def test_wait_tool_polls_queue_without_event():
    queue: asyncio.Queue = asyncio.Queue()

    async def fill_later() -> None:
        await asyncio.sleep(0.2)
        queue.put_nowait({"text": "late"})

    asyncio.create_task(fill_later())
    started = time.monotonic()
    result = await WaitTool().execute(
        seconds=5, context={"injected_input_queue": queue}
    )
    assert time.monotonic() - started < 2
    assert result.startswith("Interrupted by new input")


@pytest.mark.asyncio
async def test_manager_wait_interrupted_by_input_event():
    class SlowRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            await asyncio.sleep(0.4)
            yield {"content": "done", "reasoning": ""}

    manager = SubagentManager(
        {"max_concurrent": 1, "max_total": 5}, "c", SlowRAG()
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)
    event = asyncio.Event()

    async def set_later() -> None:
        await asyncio.sleep(0.05)
        event.set()

    asyncio.create_task(set_later())
    started = time.monotonic()
    result = await manager.wait(None, state, input_event=event)
    assert time.monotonic() - started < 0.35
    assert result["metadata"]["interrupted_by_input"] is True
    assert "Still running" in result["content"]

    # A follow-up wait without fresh input collects the finished child.
    result2 = await manager.wait(None, state)
    assert record.status == "done"
    assert not result2["metadata"].get("interrupted_by_input")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_wait_all_returns_when_children_done_without_input():
    """Regression: with an input_event attached, mode='all' must return once
    every child finished — the pending event waiter must not keep waiting."""

    class FastRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            yield {"content": "quick answer", "reasoning": ""}

    manager = SubagentManager(
        {"max_concurrent": 1, "max_total": 5}, "c", FastRAG()
    )
    state = make_subagent_state(manager, "Analyst")
    await manager.spawn(task="t", label=None, owner=state)
    event = asyncio.Event()  # never set — must NOT block the wait

    result = await asyncio.wait_for(
        manager.wait(None, state, input_event=event), timeout=5.0
    )
    assert not result["metadata"].get("interrupted_by_input")
    assert "quick answer" in result["content"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_nudge_queues_message_and_wakes_child():
    registry: dict = {}
    manager = SubagentManager(
        {"max_concurrent": 1, "max_total": 5}, "c", FakeRAG(),
        channel_registry=registry,
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label="T", owner=state)
    assert registry[record.id] is record

    result = await manager.nudge(record.id, "focus on X", state)

    assert "Nudge delivered" in result["content"]
    item = record.nudge_queue.get_nowait()
    assert item["source"] == "parent"
    assert item["text"] == "focus on X"
    assert item["content"].startswith("[Message from parent agent]")
    assert record.input_event.is_set()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_nudge_scoped_to_owner_and_rejects_finished():
    manager = SubagentManager({}, "c", FakeRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)

    other = state.child("sub-other")  # different owner_key
    result = await manager.nudge(record.id, "hi", other)
    assert "no subagent" in result["content"]

    await manager.wait(None, state)
    assert record.status == "done"
    result = await manager.nudge(record.id, "hi", state)
    assert "already finished" in result["content"]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_execute_record_receives_nudge_channel_and_captures_injected():
    class NudgeRAG:
        ready = True

        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state,
            injected_input_queue=None, input_event=None,
        ):
            # The child's own queue/event must be wired through.
            assert injected_input_queue is not None
            assert input_event is not None
            yield {
                "content": "", "reasoning": "",
                "injected_user_message": "focus on X",
                "injected_source": "parent",
            }
            yield {"content": "ok", "reasoning": ""}

    manager = SubagentManager({}, "c", NudgeRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)
    await manager.wait(None, state)

    assert record.status == "done"
    assert len(record.nudges) == 1
    assert record.nudges[0]["source"] == "parent"
    assert record.nudges[0]["text"] == "focus on X"
    assert record.to_dict()["nudges"][0]["text"] == "focus on X"
    assert "Nudges received: 1" in record.format_dossier()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_nudge_subagent_tool_enqueues_via_registry():
    registry = ToolRegistry()
    registry.register_builtin("nudge_subagent", NudgeSubagentTool())
    manager = SubagentManager({}, "c", FakeRAG())
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="t", label=None, owner=state)

    result = await registry.execute(
        "nudge_subagent",
        {"subagent_id": record.id, "message": "hurry"},
        context={"subagent_state": state},
    )

    assert "Nudge delivered" in result["content"]
    assert record.nudge_queue.qsize() == 1
    assert record.input_event.is_set()
    await manager.shutdown()


def test_drain_injected_clears_event_and_keeps_source():
    runner = AgentRunner.__new__(AgentRunner)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"text": "hi", "content": "[Message from parent agent]: hi", "source": "parent"})
    queue.put_nowait({"text": "steer", "content": "steer"})
    event = asyncio.Event()
    event.set()
    runner._injected_input_queue = queue
    runner._input_event = event

    messages: list = []
    drained = runner._drain_injected(messages)

    assert drained == [
        {"text": "hi", "source": "parent"},
        {"text": "steer", "source": None},
    ]
    assert not event.is_set()
    assert [type(m) for m in messages] == [HumanMessage, HumanMessage]
    assert messages[0].content == "[Message from parent agent]: hi"


def test_usage_totals_includes_cache_tokens():
    manager = SubagentManager({}, None, FakeRAG())
    manager.records["s1"] = SubagentRecord(
        id="s1", parent_id=None, owner_key="s1", task="t", label="l",
        assistant="a", depth=1, status="done",
        nested_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cache_tokens": 7},
    )
    totals = manager.usage_totals()
    assert totals["cache_tokens"] == 7
    assert totals["total_tokens"] == 15


@pytest.mark.asyncio
async def test_child_keepalive_events_propagate_flag():
    """A child's synthetic keepalives must carry the flag into the parent's
    event stream so they don't reset the run-level idle watchdog."""

    class KeepaliveRAG(FakeRAG):
        async def run_streaming(
            self, user_input, *, chat_history, model_name, retrieval_info,
            conversation_id, subagent_state, **kwargs,
        ):
            yield {
                "content": "",
                "reasoning": "",
                "tool_progress": {
                    "id": "call_1",
                    "tool_name": "code_execution",
                    "stage": "progress",
                    "keepalive": True,
                },
            }
            yield {
                "content": "",
                "reasoning": "",
                "tool_progress": {
                    "id": "call_1",
                    "tool_name": "code_execution",
                    "stage": "started",
                },
            }
            yield {"content": "done", "reasoning": ""}
            retrieval_info["_llm_usage_metadata"] = {"total_tokens": 1}

    manager = SubagentManager(
        {"max_concurrent": 2, "max_depth": 1, "max_total": 2},
        "conversation",
        KeepaliveRAG(),
    )
    state = make_subagent_state(manager, "Analyst")
    record = await manager.spawn(task="work", label="Work", owner=state)
    await manager.wait([record.id], state)

    events = []
    while not manager.event_queue.empty():
        events.append(manager.event_queue.get_nowait())

    flagged = [e for e in events if e.get("keepalive") is True]
    assert flagged, "expected the child's keepalive to arrive flagged"
    assert flagged[0]["activity_item"]["tool_name"] == "code_execution"

    unflagged_progress = [
        e for e in events
        if e.get("activity_item", {}).get("stage") == "started"
    ]
    assert unflagged_progress
    assert all("keepalive" not in e for e in unflagged_progress)


def test_usage_totals_includes_estimated_iterations():
    manager = SubagentManager({}, None, FakeRAG())
    manager.records["s1"] = SubagentRecord(
        id="s1", parent_id=None, owner_key="s1", task="t", label="l",
        assistant="a", depth=1, status="done",
        nested_usage={"prompt_tokens": 10, "total_tokens": 15, "estimated_iterations": 2},
    )
    manager.records["s2"] = SubagentRecord(
        id="s2", parent_id="s1", owner_key="s1", task="t", label="l",
        assistant="a", depth=2, status="done",
        nested_usage={"prompt_tokens": 5, "total_tokens": 8, "estimated_iterations": 1},
    )
    totals = manager.usage_totals()
    assert totals["estimated_iterations"] == 3


def test_usage_totals_includes_rag_tool_tokens():
    manager = SubagentManager({}, None, FakeRAG())
    manager.records["s1"] = SubagentRecord(
        id="s1", parent_id=None, owner_key="s1", task="t", label="l",
        assistant="a", depth=1, status="done",
        nested_usage={"prompt_tokens": 10, "rag_tool_tokens": 40},
    )
    manager.records["s2"] = SubagentRecord(
        id="s2", parent_id="s1", owner_key="s1", task="t", label="l",
        assistant="a", depth=2, status="done",
        nested_usage={"prompt_tokens": 5, "rag_tool_tokens": 12},
    )
    totals = manager.usage_totals()
    assert totals["rag_tool_tokens"] == 52
