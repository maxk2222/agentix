from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent.application import RAGApplication
from agent.tools import ToolRegistry
from agent.tools.builtin_tools.todo_list import AgentTodoTool, TodoListState
from agent.tools.runner import AgentRunner


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, schemas):
        self.schemas = schemas
        return self

    async def astream(self, messages):
        self.calls.append(list(messages))
        yield self.responses.pop(0)


def _add_response(title: str, call_id: str = "todo-add"):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "agent_todo",
            "args": {"action": "add", "title": title},
            "id": call_id,
        }],
    )


@pytest.mark.asyncio
async def test_todo_tool_supports_crud_and_returns_authoritative_snapshots():
    tool = AgentTodoTool()
    state = TodoListState()
    context = {"todo_state": state}

    first = await tool.execute(action="add", title="Read the source", context=context)
    assert "t1" in first["content"]
    assert state.snapshot() == {
        "version": 1,
        "tasks": [{"id": "t1", "title": "Read the source", "status": "pending"}],
    }

    await tool.execute(
        action="update",
        task_id="1",
        status="in_progress",
        note="Check the full document",
        context=context,
    )
    assert state.snapshot()["tasks"][0]["status"] == "in_progress"
    await tool.execute(action="complete", task_id="t1", context=context)
    assert state.snapshot()["tasks"][0]["status"] == "completed"

    await tool.execute(action="add", title="Write a summary", context=context)
    await tool.execute(action="remove", task_id="t2", context=context)
    assert [task["id"] for task in state.snapshot()["tasks"]] == ["t1"]
    await tool.execute(action="clear", context=context)
    assert state.snapshot()["tasks"] == []
    assert state.snapshot()["version"] == 6


@pytest.mark.asyncio
async def test_todo_tool_rejects_invalid_operations_and_bounds_task_count():
    tool = AgentTodoTool()
    state = TodoListState()
    context = {"todo_state": state, "tool_config": {"max_tasks": 1}}
    with pytest.raises(ValueError, match="title"):
        await tool.execute(action="add", title=" ", context=context)
    await tool.execute(action="add", title="First", context=context)
    with pytest.raises(ValueError, match="full"):
        await tool.execute(action="add", title="Second", context=context)
    with pytest.raises(ValueError, match="Unknown task id"):
        await tool.execute(action="complete", task_id="t99", context=context)
    with pytest.raises(ValueError, match="Unknown action"):
        await tool.execute(action="invalid", context=context)


@pytest.mark.asyncio
async def test_agent_runner_streams_todo_snapshot_and_resets_state_for_each_run():
    registry = ToolRegistry()
    runner = AgentRunner(
        registry,
        ["agent_todo"],
        supports_native_tool_calls=True,
        max_iterations=3,
    )

    # Iteration 2 finalize attempts are held back by the open todo (gate);
    # iteration 3 == max_iterations is allowed to end and sweeps the task.
    first_chunks = [
        chunk async for chunk in runner.run_streaming(
            ScriptedLLM([
                _add_response("First run"),
                AIMessage(content="Finished."),
                AIMessage(content="Finished."),
            ]),
            "multi-step request",
            [],
            "Instructions",
        )
    ]
    first_delta = next(chunk["todo_list"] for chunk in first_chunks if chunk.get("todo_list"))
    assert first_delta["tasks"][0]["title"] == "First run"

    second_result = await runner.run(
        ScriptedLLM([
            _add_response("Second run", "todo-add-2"),
            AIMessage(content="Finished again."),
            AIMessage(content="Finished again."),
        ]),
        "another multi-step request",
        [],
        "Instructions",
    )
    # The run-end sweep marks tasks the agent left open as "skipped" — the
    # run is over, they will never be worked on.
    assert second_result["todo_list"]["tasks"] == [
        {"id": "t1", "title": "Second run", "status": "skipped"}
    ]


def test_registry_discovers_todo_tool_automatically():
    registry = ToolRegistry()
    assert "agent_todo" in registry.list_tools()
    assert registry.get_tool("agent_todo") is not None


@pytest.mark.asyncio
async def test_agent_application_auto_enables_todo_without_configured_tools(monkeypatch):
    snapshot = {"version": 1, "tasks": [{"id": "t1", "title": "Plan", "status": "pending"}]}
    observed_tools = []

    class FakeChainBuilder:
        async def build_chains(self, *args, **kwargs):
            return None, None, None

        async def _get_agent_llm(self, *args, **kwargs):
            return object()

    class FakeRunner:
        def __init__(self, registry, tool_names, **kwargs):
            observed_tools.extend(tool_names)
            self._last_trace = None
            self._iteration_usages = []

        async def run(self, *args, **kwargs):
            return {"answer": "Done", "todo_list": snapshot}

        async def run_streaming(self, *args, **kwargs):
            yield {"content": "", "reasoning": "", "todo_list": snapshot}

        def todo_list_snapshot(self):
            return snapshot

        def _aggregate_iteration_usages(self, usages):
            return None

    app = RAGApplication.__new__(RAGApplication)
    app._ready = True
    app.config = type("Config", (), {
        "retrieval_history_enabled": False,
        "retrieval_history_user_messages": 0,
        "retrieval_history_max_chars": 0,
        "retrieval_history_standalone_detection": False,
        "contextualize_question_enabled": False,
    })()
    app.instructions = {"Analyst": {"agent_enabled": True, "tools": {}}}
    app._recovery_manager = None
    app._retriever_builder = None
    app._chain_builder = FakeChainBuilder()
    app._tool_registry = ToolRegistry()
    app._agent_memory_store = None
    monkeypatch.setattr("agent.application.AgentRunner", FakeRunner)
    monkeypatch.setattr("agent.application.resolve_rolling_history_settings", lambda *args, **kwargs: type("Settings", (), {"enabled": False})())
    monkeypatch.setattr("agent.application.render_system_instructions", lambda instructions, **kwargs: instructions)
    monkeypatch.setattr(app, "_get_provider_tool_calling_hint", lambda *args: _async_value(True))

    retrieval_info = {}
    result = await app.run("question", chat_history=[], model_name="Analyst", retrieval_info=retrieval_info)

    assert "agent_todo" in observed_tools
    assert result["answer"] == "Done"
    assert retrieval_info["todo_list"] == snapshot

    streaming_info = {}
    chunks = [
        chunk async for chunk in app.run_streaming(
            "question",
            chat_history=[],
            model_name="Analyst",
            retrieval_info=streaming_info,
        )
    ]
    assert observed_tools.count("agent_todo") == 2
    assert any(chunk.get("todo_list") == snapshot for chunk in chunks)
    assert streaming_info["todo_list"] == snapshot


async def _async_value(value):
    return value


def _todo_call(action: str, task_id: str, call_id: str, **extra_args) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "agent_todo",
            "args": {"action": action, "task_id": task_id, **extra_args},
            "id": call_id,
        }],
    )


@pytest.mark.asyncio
async def test_todo_gate_blocks_finalization_until_tasks_closed():
    """An open todo holds the run open: the finalize attempt becomes a
    steering HumanMessage listing the task, and the run only ends once the
    task is completed."""
    runner = AgentRunner(
        ToolRegistry(),
        ["agent_todo"],
        supports_native_tool_calls=True,
        max_iterations=6,
    )
    llm = ScriptedLLM([
        _add_response("finish the report"),
        AIMessage(content="All done, here is the answer."),
        _todo_call("complete", "t1", "todo-complete"),
        AIMessage(content="Now really done."),
    ])
    chunks = [
        chunk async for chunk in runner.run_streaming(
            llm, "do a multi-step thing", [], "Instructions",
        )
    ]
    # Gate marker reached the stream once (at the blocked attempt)
    markers = [
        c for c in chunks
        if "Final answer held back" in (c.get("reasoning") or "")
    ]
    assert len(markers) == 1
    assert "1 open todo task" in markers[0]["reasoning"]
    # The block notice went to the model as a HumanMessage naming the task
    gate_msgs = [
        m for call in llm.calls for m in call
        if m.__class__.__name__ == "HumanMessage"
        and "Finalization blocked" in str(m.content)
    ]
    assert gate_msgs, "expected a gate HumanMessage in a later LLM call"
    assert "t1" in gate_msgs[0].content
    assert "finish the report" in gate_msgs[0].content
    # Final answer was produced only after the task was completed
    assert "Now really done." in "".join(c.get("content", "") for c in chunks)
    snapshot = runner.todo_list_snapshot()
    assert snapshot["tasks"][0]["status"] == "completed"
    assert runner._last_trace.todo_gate_blocks == 1


@pytest.mark.asyncio
async def test_todo_gate_skipped_task_counts_as_done():
    """A task explicitly marked 'skipped' is settled — no gate block."""
    runner = AgentRunner(
        ToolRegistry(),
        ["agent_todo"],
        supports_native_tool_calls=True,
        max_iterations=5,
    )
    llm = ScriptedLLM([
        _add_response("obsolete subtask"),
        _todo_call("update", "t1", "todo-skip", status="skipped"),
        AIMessage(content="done"),
    ])
    chunks = [
        chunk async for chunk in runner.run_streaming(
            llm, "request", [], "Instructions",
        )
    ]
    assert not any(
        "Final answer held back" in (c.get("reasoning") or "") for c in chunks
    )
    assert runner._last_trace.todo_gate_blocks == 0


@pytest.mark.asyncio
async def test_todo_gate_caps_reminders_then_forces_finalize():
    """After MAX_TODO_FINALIZE_REMINDERS blocked attempts the run finalizes
    anyway and the leftover task is swept to 'skipped'."""
    runner = AgentRunner(
        ToolRegistry(),
        ["agent_todo"],
        supports_native_tool_calls=True,
        max_iterations=5,
    )
    llm = ScriptedLLM([
        _add_response("never finished"),
        AIMessage(content="try to end 1"),
        AIMessage(content="try to end 2"),
        AIMessage(content="try to end 3"),
        AIMessage(content="try to end 4"),
    ])
    chunks = [
        chunk async for chunk in runner.run_streaming(
            llm, "request", [], "Instructions",
        )
    ]
    markers = [
        c for c in chunks
        if "Final answer held back" in (c.get("reasoning") or "")
    ]
    # iterations 2,3,4 blocked; iteration 5 == max_iterations is allowed
    assert len(markers) == 3
    assert runner._todo_gate_reminders == 3
    assert runner._last_trace.todo_gate_blocks == 3
    snapshot = runner.todo_list_snapshot()
    assert snapshot["tasks"][0]["status"] == "skipped"


@pytest.mark.asyncio
async def test_todo_gate_inactive_without_todo_tool():
    """Without agent_todo in the run's tools the gate never blocks."""
    registry = ToolRegistry()
    runner = AgentRunner(
        registry,
        ["mock_tool"],  # agent_todo absent
        supports_native_tool_calls=True,
        max_iterations=3,
    )
    llm = ScriptedLLM([AIMessage(content="plain answer")])
    chunks = [
        chunk async for chunk in runner.run_streaming(
            llm, "request", [], "Instructions",
        )
    ]
    assert not any(
        "Final answer held back" in (c.get("reasoning") or "") for c in chunks
    )
    assert "plain answer" in "".join(c.get("content", "") for c in chunks)


@pytest.mark.asyncio
async def test_final_answer_tool_call_reenters_loop():
    """A tool call detected inside the streamed final answer is executed and
    the loop continues (reenter_loop was dead code before the fix)."""
    runner = AgentRunner(
        ToolRegistry(),
        ["wait", "agent_todo"],
        supports_native_tool_calls=False,  # JSON fallback mode
        max_iterations=6,
    )
    llm = ScriptedLLM([
        # iter1: clean text -> fallback -> path B -> separate final-answer
        # stream -> the JSON tool call is detected there
        AIMessage(content="Here is my answer."),
        AIMessage(content='{"wait": {"seconds": 0}}'),
        # iter2 (after the reentered tool call): clean text -> final gen again
        AIMessage(content="done"),
        AIMessage(content="final answer"),
    ])
    chunks = [
        chunk async for chunk in runner.run_streaming(
            llm, "request", [], "Instructions",
        )
    ]
    assert "final answer" in "".join(c.get("content", "") for c in chunks)
    # The wait call parsed out of the final answer really executed
    wait_records = [
        rec for rec in runner._last_trace.tool_calls if rec.tool_name == "wait"
    ]
    assert wait_records and wait_records[0].ok
