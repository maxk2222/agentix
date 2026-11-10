"""Tests for agent.tools.runner (AgentRunner)."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.tools import ToolRegistry
from agent.tools.runner import AgentRunner, _try_parse_tool_calls


class FakeLLM:
    """Mock LLM that supports tool binding and invocation."""

    def __init__(self, responses: list = None) -> None:
        self.responses = responses or []
        self.calls: list = []
        self._bound_tools: list = []

    def bind_tools(self, tools: list) -> "FakeLLM":
        self._bound_tools = tools
        return self

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="Final answer.")

    async def astream(self, messages: list, **kwargs: Any):
        """Yield the final answer as a single chunk (no real streaming)."""
        response = await self.ainvoke(messages, **kwargs)
        yield response


class FakeLLMNoBind:
    """Mock LLM that does NOT support bind_tools (triggers JSON fallback mode)."""

    def __init__(self, responses: list = None) -> None:
        self.responses = responses or []
        self.calls: list = []

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="Final answer.")

    async def astream(self, messages: list, **kwargs: Any):
        """Yield the final answer as a single chunk (no real streaming)."""
        response = await self.ainvoke(messages, **kwargs)
        yield response


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.execute = AsyncMock(return_value="tool result")
    mock_tool.enabled = True
    mock_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "mock_tool",
            "description": "A mock tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
            },
        },
    }
    reg.register_builtin("mock_tool", mock_tool)

    mock_tool2 = MagicMock()
    mock_tool2.execute = AsyncMock(return_value="tool2 result")
    mock_tool2.enabled = True
    mock_tool2.tool_schema = {
        "type": "function",
        "function": {
            "name": "mock_tool2",
            "description": "Another mock tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
            },
        },
    }
    reg.register_builtin("mock_tool2", mock_tool2)
    return reg


@pytest.mark.asyncio
async def test_agent_no_tool_calls(registry: ToolRegistry):
    """If the LLM returns no tool calls, the answer is returned directly."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(
        llm,
        "Hello",
        [],
        system_instructions="You are a helpful assistant.",
    )

    assert result["answer"] == "Final answer."
    assert "reasoning" in result


@pytest.mark.asyncio
async def test_agent_single_tool_call(registry: ToolRegistry):
    """If the LLM requests a tool, it should be executed and result fed back."""
    # First response: tool call
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="The result is: tool result")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(
        llm,
        "Search for something",
        [],
        system_instructions="You are helpful.",
    )

    assert "tool result" in result["answer"]
    assert "reasoning" in result
    # Verify the tool was called
    tool = registry.get_tool("mock_tool")
    assert tool is not None
    tool.execute.assert_called_once_with(query="test")


@pytest.mark.asyncio
async def test_agent_stream_no_tool_calls(registry: ToolRegistry):
    """Streaming without tool calls yields content directly."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm,
        "Hello",
        [],
        system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    # run_streaming now yields response.content in chunks of 8 chars
    full_content = "".join(c.get("content", "") for c in chunks)
    assert "Final answer." in full_content
    assert any(c.get("reasoning") for c in chunks)


@pytest.mark.asyncio
async def test_agent_max_iterations_limit(registry: ToolRegistry):
    """The agent should stop after max_iterations."""
    # LLM always responds with a tool call (never terminates)
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}],
    )
    llm = FakeLLM(responses=[msg] * 10)
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=2)

    result = await runner.run(llm, "test", [], "")

    # After max_iterations, it should force a final answer with up to 3 retry attempts
    # The LLM should have been called at least max_iterations + 1 times
    assert len(llm.calls) >= 3
    assert "reasoning" in result


@pytest.mark.asyncio
async def test_agent_reasoning_accumulated(registry: ToolRegistry):
    """Reasoning from multiple iterations is accumulated in the result."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}],
        additional_kwargs={"reasoning_content": "I need to search."},
    )
    second_msg = AIMessage(
        content="Done.",
        additional_kwargs={"reasoning_content": "I have the answer."},
    )

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    assert "reasoning" in result
    assert "I need to search." in result["reasoning"]
    assert "I have the answer." in result["reasoning"]


@pytest.mark.asyncio
async def test_agent_disabled_tool_not_executed(registry: ToolRegistry):
    """Tools not in the allowed list should not be called."""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "forbidden_tool", "args": {}, "id": "call_1"}],
    )
    llm = FakeLLM(responses=[msg])
    runner = AgentRunner(registry, ["mock_tool"])  # forbidden_tool not allowed

    result = await runner.run(llm, "test", [], "")

    # The runner should not try to execute the forbidden tool
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_agent_fallback_json_tool_call(registry: ToolRegistry):
    """If the LLM returns JSON text as content (no native tool_calls), parse it as a tool call."""
    first_msg = AIMessage(content='{"mock_tool": {"query": "fallback test"}}')
    second_msg = AIMessage(content="Fallback tool result processed.")

    llm = FakeLLMNoBind(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Search", [], "")

    assert "fallback test" in result["answer"].lower() or "tool result" in result["answer"].lower()
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="fallback test")


def test_try_parse_tool_calls_with_tool_name():
    """Form 1: tool name as top-level key."""
    result, _ = _try_parse_tool_calls('{"web_search": {"query": "test"}}', ["web_search"])
    assert result is not None
    assert result[0]["name"] == "web_search"
    assert result[0]["args"] == {"query": "test"}


def test_try_parse_tool_calls_with_markdown_fences():
    """Markdown-fenced JSON should be stripped and parsed."""
    text = "```json\n{\"web_search\": {\"query\": \"test\"}}\n```"
    result, _ = _try_parse_tool_calls(text, ["web_search"])
    assert result is not None
    assert result[0]["name"] == "web_search"


def test_try_parse_tool_calls_unknown_tool():
    """Unknown tool name should return None."""
    result, _ = _try_parse_tool_calls('{"unknown_tool": {"query": "test"}}', ["web_search"])
    assert result is None


def test_try_parse_tool_calls_non_json():
    """Non-JSON content should return None."""
    result, _ = _try_parse_tool_calls("This is just text.", ["web_search"])
    assert result is None


@pytest.mark.asyncio
async def test_agent_forced_fallback_via_hint(registry: ToolRegistry):
    """If supports_native_tool_calls=False is passed, runner should use JSON fallback
    even when the LLM advertises bind_tools."""
    first_msg = AIMessage(content='{"mock_tool": {"query": "forced fallback"}}')
    second_msg = AIMessage(content="Forced fallback processed.")

    llm = FakeLLM(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(
        registry, ["mock_tool"], supports_native_tool_calls=False
    )

    result = await runner.run(llm, "Search", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="forced fallback")
    # Fallback mode should have been active
    assert runner.supports_native_tool_calls is False


@pytest.mark.asyncio
async def test_agent_native_mode_via_hint(registry: ToolRegistry):
    """If supports_native_tool_calls=True is passed, runner should use native mode."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "native hint"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Native hint processed.")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(
        registry, ["mock_tool"], supports_native_tool_calls=True
    )

    result = await runner.run(llm, "Search", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="native hint")
    assert runner.supports_native_tool_calls is True


def test_try_parse_tool_calls_from_messy_text():
    """JSON embedded in surrounding text should be extracted via regex."""
    messy = (
        "The user asks about price. I need to search. "
        '{"web_search": {"query": "MinisForum N5 Pro price"}} '
        "Let's execute the tool."
    )
    result, _ = _try_parse_tool_calls(messy, ["web_search"])
    assert result is not None
    assert result[0]["name"] == "web_search"
    assert result[0]["args"] == {"query": "MinisForum N5 Pro price"}


def test_try_parse_tool_calls_flat_dict():
    """Flat dict with 'query' key should be matched to web_search."""
    mapping = {"query": ["web_search"]}
    result, _ = _try_parse_tool_calls('{"query": "test query", "results": 10}', ["web_search"], flat_dict_mapping=mapping)
    assert result is not None
    assert result[0]["name"] == "web_search"
    assert result[0]["args"] == {"query": "test query", "results": 10}


def test_try_parse_tool_calls_multiple_json():
    """Multiple JSON objects in one response should find the first valid tool call."""
    text = '{"unknown": 1} {"web_search": {"query": "test"}} {"other": 2}'
    result, _ = _try_parse_tool_calls(text, ["web_search"])
    assert result is not None
    assert result[0]["name"] == "web_search"
    assert result[0]["args"] == {"query": "test"}


@pytest.mark.asyncio
async def test_agent_retry_on_invalid_tool_format(registry: ToolRegistry):
    """If the model mentions a tool but produces invalid JSON, it should retry."""
    # First response: mentions tool name but no valid JSON
    first_msg = AIMessage(content="I will use mock_tool to search for something.")
    # Second response: valid JSON after retry prompt
    second_msg = AIMessage(content='{"mock_tool": {"query": "retry test"}}')
    third_msg = AIMessage(content="Retry result processed.")

    llm = FakeLLMNoBind(responses=[first_msg, second_msg, third_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Search", [], "")

    # The model should have been called at least twice (first invalid, second valid)
    assert len(llm.calls) >= 2
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="retry test")


@pytest.mark.asyncio
async def test_agent_deduplication_allows_repeat(registry: ToolRegistry):
    """Duplicate tool calls with identical arguments are now executed (no blocking)."""
    first_msg = AIMessage(content='{"mock_tool": {"query": "dedup test"}}')
    second_msg = AIMessage(content='{"mock_tool": {"query": "dedup test"}}')
    third_msg = AIMessage(content="Based on the results, the answer is 42.")

    llm = FakeLLMNoBind(responses=[first_msg, second_msg, third_msg, third_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Search", [], "")

    # Tool should be executed TWICE (no deduplication)
    tool = registry.get_tool("mock_tool")
    assert tool.execute.call_count == 2
    assert "42" in result["answer"]


class SlowFakeLLM:
    """Mock LLM that never returns (simulates a hung request)."""

    def bind_tools(self, tools: list) -> "SlowFakeLLM":
        return self

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        import asyncio
        await asyncio.sleep(9999)
        return AIMessage(content="never reached")

    async def astream(self, messages: list, **kwargs: Any):
        import asyncio
        await asyncio.sleep(9999)
        yield AIMessage(content="never reached")


class SlowFakeLLMNoBind:
    """Mock LLM without bind_tools that never returns."""

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        import asyncio
        await asyncio.sleep(9999)
        return AIMessage(content="never reached")

    async def astream(self, messages: list, **kwargs: Any):
        import asyncio
        await asyncio.sleep(9999)
        yield AIMessage(content="never reached")


@pytest.mark.asyncio
async def test_agent_timeout():
    """The runner should abort once the stream idles beyond the idle limit."""
    reg = ToolRegistry()
    runner = AgentRunner(reg, [], stream_idle_seconds=0.1)
    llm = SlowFakeLLM()
    result = await runner.run(llm, "test", [], "")
    assert "stopped responding" in result["answer"].lower()


@pytest.mark.asyncio
async def test_agent_stream_timeout():
    """The streaming runner should yield an idle-abort message when ainvoke hangs."""
    reg = ToolRegistry()
    runner = AgentRunner(reg, [], stream_idle_seconds=0.1)
    llm = SlowFakeLLM()
    chunks = []
    async for chunk in runner.run_streaming(llm, "test", [], ""):
        chunks.append(chunk)
    assert any("stopped responding" in (c.get("content") or "").lower() for c in chunks)


@pytest.mark.asyncio
async def test_agent_trace_present(registry: ToolRegistry):
    """The result should include an agent_trace when running in agent mode."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])
    result = await runner.run(llm, "Hello", [], "")
    assert "agent_trace" in result
    trace = result["agent_trace"]
    assert trace["iterations"] >= 1
    assert trace["fallback_mode"] is False


@pytest.mark.asyncio
async def test_agent_fallback_mode_detected(registry: ToolRegistry):
    """If LLM has no bind_tools, fallback_mode should be True in the trace."""
    llm = FakeLLMNoBind()
    runner = AgentRunner(registry, ["mock_tool"])
    result = await runner.run(llm, "Hello", [], "")
    assert result["agent_trace"]["fallback_mode"] is True


@pytest.mark.asyncio
async def test_agent_trace_includes_tool_call_list(registry: ToolRegistry):
    """AgentTrace.to_dict() should include full tool call records, not just a count."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Done.")
    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])
    result = await runner.run(llm, "Hello", [], "")
    trace = result["agent_trace"]
    assert "tool_call_count" in trace
    assert "tool_calls" in trace
    assert isinstance(trace["tool_calls"], list)
    assert len(trace["tool_calls"]) == trace["tool_call_count"]
    if trace["tool_calls"]:
        call = trace["tool_calls"][0]
        assert call["tool_name"] == "mock_tool"
        assert call["args"] == {"query": "test"}
        assert call["ok"] is True


@pytest.mark.asyncio
async def test_agent_validation_error_returned(registry: ToolRegistry):
    """If tool arguments are invalid, a validation error should be returned to the model."""
    # Create a tool with a schema that requires 'query'
    mock_tool = MagicMock()
    mock_tool.execute = AsyncMock(return_value="tool result")
    mock_tool.enabled = True
    mock_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "schema_tool",
            "description": "A tool with schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query."},
                },
                "required": ["query"],
            },
        },
    }
    reg = ToolRegistry()
    reg.register_builtin("schema_tool", mock_tool)

    # LLM calls the tool without the required 'query' argument
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "schema_tool", "args": {}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Got an error.")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(reg, ["schema_tool"])
    result = await runner.run(llm, "test", [], "")

    # execute should NOT have been called because validation failed first
    mock_tool.execute.assert_not_called()
    assert "error" in result["answer"].lower() or "Got an error" in result["answer"]


@pytest.mark.asyncio
async def test_agent_bind_tools_but_json_output(registry: ToolRegistry):
    """LLM has bind_tools but returns JSON in content instead of native tool_calls.

    This is the regression for models that advertise tool support but still
    emit JSON text (e.g. GPT-OSS 120B via OpenAI-compatible API).
    """
    first_msg = AIMessage(content='{"mock_tool": {"query": "json test"}}')
    second_msg = AIMessage(content="Result from json tool.")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])
    result = await runner.run(llm, "Search", [], "")

    # The tool should have been executed via JSON parsing
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="json test")
    assert "Result from json tool" in result["answer"]


@pytest.mark.asyncio
async def test_agent_rag_context_replaces_placeholder(registry: ToolRegistry):
    """If system_instructions contains {{context}}, rag_context replaces it."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])

    await runner.run(
        llm,
        "Hello",
        [],
        system_instructions="Use this context: {{context}}",
        rag_context="Doc 1 content",
    )

    first_call_messages = llm.calls[0]
    sys_msg = first_call_messages[0]
    assert "Use this context: Doc 1 content" in sys_msg.content
    assert "{{context}}" not in sys_msg.content


@pytest.mark.asyncio
async def test_agent_rag_context_appends_when_no_placeholder(registry: ToolRegistry):
    """If system_instructions has no {{context}}, rag_context is appended."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])

    await runner.run(
        llm,
        "Hello",
        [],
        system_instructions="You are helpful.",
        rag_context="Doc 1 content",
    )

    first_call_messages = llm.calls[0]
    sys_msg = first_call_messages[0]
    assert "You are helpful." in sys_msg.content
    assert "Relevant context:" in sys_msg.content
    assert "Doc 1 content" in sys_msg.content


@pytest.mark.asyncio
async def test_agent_streaming_rag_context_replaces_placeholder(registry: ToolRegistry):
    """Streaming mode also injects rag_context via {{context}} replacement."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm,
        "Hello",
        [],
        system_instructions="Context: {{context}}",
        rag_context="Streaming doc",
    ):
        chunks.append(chunk)

    first_call_messages = llm.calls[0]
    sys_msg = first_call_messages[0]
    assert "Context: Streaming doc" in sys_msg.content
    assert "{{context}}" not in sys_msg.content


class FakeLLMWithReasoning:
    """Mock LLM that yields reasoning content in streaming chunks."""

    def __init__(self, content: str = "Final answer.", reasoning: str = "") -> None:
        self.content = content
        self.reasoning = reasoning
        self.calls: list = []

    def bind_tools(self, tools: list) -> "FakeLLMWithReasoning":
        return self

    async def astream(self, messages: list, **kwargs: Any):
        self.calls.append(messages)
        if self.reasoning:
            yield AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": self.reasoning},
            )
        chunk_size = 8
        for i in range(0, len(self.content), chunk_size):
            yield AIMessageChunk(content=self.content[i:i + chunk_size])

    async def ainvoke(self, messages: list, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(
            content=self.content,
            additional_kwargs={"reasoning_content": self.reasoning} if self.reasoning else {},
        )


@pytest.mark.asyncio
async def test_agent_streaming_reasoning_emitted_in_real_time(registry: ToolRegistry):
    """Streaming should yield reasoning content as it arrives, before the full response is buffered."""
    llm = FakeLLMWithReasoning(content="The answer is 42.", reasoning="Let me think about this...")
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm,
        "What is the answer?",
        [],
        system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    reasoning_parts = [c.get("reasoning", "") for c in chunks if c.get("reasoning")]
    content_parts = [c.get("content", "") for c in chunks if c.get("content")]

    assert any("Let me think about this..." in r for r in reasoning_parts)
    assert "The answer is 42." in "".join(content_parts)


@pytest.mark.asyncio
async def test_agent_streaming_tool_call_from_buffered_content(registry: ToolRegistry):
    """Tool calls should still be parsed from buffered content after streaming completes."""
    first_msg = AIMessage(content='{"mock_tool": {"query": "streaming test"}}')
    second_msg = AIMessage(content="Result from streaming tool.")

    # Three responses needed: tool call, intermediate "no tool calls" round,
    # and final answer streaming via llm.astream(final_messages).
    llm = FakeLLMNoBind(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm,
        "Search for something",
        [],
        system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    full_content = "".join(c.get("content", "") for c in chunks)
    assert "Result from streaming tool." in full_content

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="streaming test")


def test_try_parse_nested_json_tool_call():
    """Nested JSON objects should be extracted by the balanced-brace scanner."""
    text = '{"openalex_search": {"query": "hydrogen storage", "filters": {"publication_year": "2023-2024"}}}'
    result, _ = _try_parse_tool_calls(text, ["openalex_search"])
    assert result is not None
    assert result[0]["name"] == "openalex_search"
    assert result[0]["args"]["query"] == "hydrogen storage"
    assert result[0]["args"]["filters"]["publication_year"] == "2023-2024"


@pytest.mark.asyncio
async def test_agent_final_answer_uses_clean_system_prompt(registry: ToolRegistry):
    """After max_iterations the final LLM call must receive a clean system prompt without tool instructions."""
    # LLM always returns a tool call, forcing max_iterations exit
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}],
    )
    # Final answer response
    final_msg = AIMessage(content="Here is the final answer.")
    llm = FakeLLM(responses=[tool_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=1)

    result = await runner.run(llm, "test", [], "You are a helpful assistant.")

    # There should be 2 LLM calls: 1 tool iteration + 1 final answer
    assert len(llm.calls) == 2

    # The final call should use a clean system prompt
    final_call_messages = llm.calls[1]
    sys_msg = final_call_messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert "You are a helpful assistant." in sys_msg.content
    # Must NOT contain tool prompts or descriptions
    assert "Available tools" not in sys_msg.content
    assert "tool_calls" not in sys_msg.content
    assert "JSON object" not in sys_msg.content
    assert result["answer"] == "Here is the final answer."


@pytest.mark.asyncio
async def test_agent_final_answer_warns_at_max_iterations(registry: ToolRegistry):
    """If the LLM emits a tool call in the forced final answer after max_iterations,
    the raw JSON is streamed and a warning is included in reasoning."""
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}],
    )
    # Final answer erroneously contains a JSON tool call
    bad_final = AIMessage(content='{"mock_tool": {"query": "should not appear"}}')
    llm = FakeLLM(responses=[tool_msg, bad_final])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=1)

    result = await runner.run(llm, "test", [], "You are helpful.")

    # Max iterations reached – raw JSON remains in answer, warning in reasoning
    assert "mock_tool" in result["answer"]
    assert "Max iterations reached" in result["reasoning"]


@pytest.mark.asyncio
async def test_agent_empty_response_prompts_continue(registry: ToolRegistry):
    """An empty response with no tool calls should prompt the model to continue instead of ending early."""
    # Iteration 1: empty content, no tool calls (simulates a model that hasn't decided yet)
    empty_msg = AIMessage(content="")
    # Iteration 2: model now provides a real tool call after the continue prompt
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "continue test"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="Done after continue.")

    llm = FakeLLM(responses=[empty_msg, tool_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=5)

    result = await runner.run(llm, "test", [], "You are helpful.")

    # The tool should have been executed (iteration 2), not treated as final answer in iteration 1
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="continue test")
    assert "Done after continue." in result["answer"]


@pytest.mark.asyncio
async def test_agent_empty_response_last_iteration_allows_final(registry: ToolRegistry):
    """An empty response on the LAST iteration may still be treated as final (no more retries)."""
    empty_msg = AIMessage(content="")
    llm = FakeLLM(responses=[empty_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=1)

    result = await runner.run(llm, "test", [], "You are helpful.")

    # On the final iteration, empty content is treated as final answer (fallback)
    assert "reasoning" in result


@pytest.mark.asyncio
async def test_agent_non_empty_response_no_tools_terminates(registry: ToolRegistry):
    """If the model returns non-empty content without ever calling a tool, treat as genuine final answer."""
    final_msg = AIMessage(content="I don't need any tools for this.")
    llm = FakeLLM(responses=[final_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=5)

    result = await runner.run(llm, "test", [], "You are helpful.")

    # No tools should have been called, and it should terminate immediately
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_not_called()
    assert "I don't need any tools for this." in result["answer"]


@pytest.mark.asyncio
async def test_agent_final_answer_reenters_loop_on_tool_call(registry: ToolRegistry):
    """If the model sneaks a tool call into the streamed final answer and max_iterations is not reached,
    the runner should detect it, execute the tool, and continue iterating."""
    # FakeLLM always returns the next queued response for EVERY astream/ainvoke call
    bad_final = AIMessage(content='{"mock_tool": {"query": "reentry test"}}')
    good_final = AIMessage(content="The tool result is: reentry test processed.")

    llm = FakeLLM(responses=[bad_final, good_final])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=5)

    result = await runner.run(llm, "test", [], "You are helpful.")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="reentry test")
    assert "reentry test processed" in result["answer"]


def test_try_parse_tool_calls_skips_non_tool_first():
    """If the first JSON object is not a tool call, keep scanning remaining candidates."""
    text = '{"unknown": 1} {"mock_tool": {"query": "skip test"}}'
    result, _ = _try_parse_tool_calls(text, ["mock_tool"])
    assert result is not None
    assert result[0]["name"] == "mock_tool"


def test_try_parse_tool_calls_with_surrounding_text():
    """JSON tool call embedded in surrounding prose should be extracted."""
    text = 'The user asks about price. I need to search. {"mock_tool": {"query": "MinisForum N5 Pro price"}} Let\'s execute the tool.'
    result, _ = _try_parse_tool_calls(text, ["mock_tool"])
    assert result is not None
    assert result[0]["name"] == "mock_tool"
    assert result[0]["args"] == {"query": "MinisForum N5 Pro price"}


@pytest.mark.asyncio
async def test_agent_parallel_native_tool_calls(registry: ToolRegistry):
    """Multiple tool_calls in one AIMessage should be executed in parallel."""
    first_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": "parallel1"}, "id": "call_1"},
            {"name": "mock_tool2", "args": {"url": "http://example.com"}, "id": "call_2"},
        ],
    )
    second_msg = AIMessage(content="Combined result.")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool", "mock_tool2"])

    result = await runner.run(llm, "Search and fetch", [], "")

    assert "Combined result." in result["answer"]
    tool1 = registry.get_tool("mock_tool")
    tool2 = registry.get_tool("mock_tool2")
    tool1.execute.assert_called_once_with(query="parallel1")
    tool2.execute.assert_called_once_with(url="http://example.com")


@pytest.mark.asyncio
async def test_agent_parallel_fallback_json_array(registry: ToolRegistry):
    """JSON array with multiple tool calls should be parsed and executed in parallel."""
    first_msg = AIMessage(
        content='[{"mock_tool": {"query": "arr1"}}, {"mock_tool2": {"url": "http://a.com"}}]'
    )
    second_msg = AIMessage(content="Array result.")

    llm = FakeLLMNoBind(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool", "mock_tool2"])

    result = await runner.run(llm, "Search", [], "")

    assert "Array result." in result["answer"]
    tool1 = registry.get_tool("mock_tool")
    tool2 = registry.get_tool("mock_tool2")
    tool1.execute.assert_called_once_with(query="arr1")
    tool2.execute.assert_called_once_with(url="http://a.com")


@pytest.mark.asyncio
async def test_agent_parallel_deduplication(registry: ToolRegistry):
    """Duplicate calls within a parallel batch are no longer deduplicated – all execute."""
    first_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": "dedup"}, "id": "call_1"},
            {"name": "mock_tool", "args": {"query": "dedup"}, "id": "call_2"},
        ],
    )
    second_msg = AIMessage(content="Dedup result.")

    llm = FakeLLM(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    tool1 = registry.get_tool("mock_tool")
    assert tool1.execute.call_count == 2


@pytest.mark.asyncio
async def test_agent_parallel_respects_max_tool_calls(registry: ToolRegistry):
    """If the batch exceeds max_tool_calls, only the allowed number are executed."""
    first_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": "1"}, "id": "call_1"},
            {"name": "mock_tool", "args": {"query": "2"}, "id": "call_2"},
            {"name": "mock_tool", "args": {"query": "3"}, "id": "call_3"},
        ],
    )
    second_msg = AIMessage(content="Truncated.")

    llm = FakeLLM(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_tool_calls=2)

    result = await runner.run(llm, "test", [], "")

    tool1 = registry.get_tool("mock_tool")
    assert tool1.execute.call_count == 2


def test_try_parse_tool_calls_json_array():
    """JSON array of tool calls should produce multiple entries."""
    text = '[{"web_search": {"query": "a"}}, {"web_scraper": {"url": "b"}}]'
    result, _ = _try_parse_tool_calls(text, ["web_search", "web_scraper"])
    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "web_search"
    assert result[1]["name"] == "web_scraper"


@pytest.mark.asyncio
async def test_agent_parallel_respects_max_parallel_tool_calls(registry: ToolRegistry):
    """If the batch exceeds max_parallel_tool_calls, only the allowed number are executed."""
    first_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": "1"}, "id": "call_1"},
            {"name": "mock_tool", "args": {"query": "2"}, "id": "call_2"},
            {"name": "mock_tool", "args": {"query": "3"}, "id": "call_3"},
        ],
    )
    second_msg = AIMessage(content="Truncated parallel.")

    llm = FakeLLM(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_tool_calls=10, max_parallel_tool_calls=2)

    result = await runner.run(llm, "test", [], "")

    tool1 = registry.get_tool("mock_tool")
    assert tool1.execute.call_count == 2
    tool_messages = [
        message for message in llm.calls[1]
        if isinstance(message, ToolMessage)
    ]
    assert {message.tool_call_id for message in tool_messages} == {"call_1", "call_2", "call_3"}


def test_build_tool_prompt_contains_limits():
    """The dynamic tool prompt should include max_total and max_parallel."""
    from agent.tools.runner import _build_tool_prompt

    prompt = _build_tool_prompt(max_total=7, max_parallel=3, fallback=False)
    assert "up to 7 regular (non-coordination) tools in total" in prompt
    assert "up to 3 tools in parallel" in prompt

    fallback_prompt = _build_tool_prompt(max_total=4, max_parallel=2, fallback=True)
    assert "up to 4 regular (non-coordination) tools in total" in fallback_prompt
    assert "up to 2 tools in parallel" in fallback_prompt


@pytest.mark.asyncio
async def test_agent_max_iterations_independent_from_max_tool_calls(registry: ToolRegistry):
    """max_iterations and max_tool_calls should be independent limits."""
    # 3 tool calls per iteration, but max_tool_calls=5 → only 5 total calls
    tool_msg1 = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": f"q{i}"}, "id": f"call_{i}"}
            for i in range(3)
        ],
    )
    tool_msg2 = AIMessage(
        content="",
        tool_calls=[
            {"name": "mock_tool", "args": {"query": f"q{i+3}"}, "id": f"call_{i+3}"}
            for i in range(3)
        ],
    )
    final_msg = AIMessage(content="Done.")

    llm = FakeLLM(responses=[tool_msg1, tool_msg2, final_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=10, max_tool_calls=5)

    result = await runner.run(llm, "test", [], "")

    tool1 = registry.get_tool("mock_tool")
    assert tool1.execute.call_count == 5
    assert "Done." in result["answer"]


@pytest.mark.asyncio
async def test_agent_streaming_tool_execution_events_in_fallback_mode():
    """In fallback mode, tool_execution started and result events must both be emitted."""
    reg = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.execute = AsyncMock(return_value="print(hello)")
    mock_tool.enabled = True
    mock_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute Python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    }
    reg.register_builtin("code_execution", mock_tool)

    first_msg = AIMessage(
        content='{"code_execution": {"code": "print(hello)"}}'
    )
    second_msg = AIMessage(content="Execution complete.")

    llm = FakeLLMNoBind(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(reg, ["code_execution"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm,
        "Run some code",
        [],
        system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    started_events = [
        c for c in chunks
        if c.get("tool_execution") and c["tool_execution"].get("stage") == "started"
    ]
    result_events = [
        c for c in chunks
        if c.get("tool_execution") and c["tool_execution"].get("stage") == "result"
    ]

    assert len(started_events) == 1
    assert started_events[0]["tool_execution"]["code"] == "print(hello)"
    assert len(result_events) == 1
    assert result_events[0]["tool_execution"]["output"] == "print(hello)"


def test_try_parse_tool_calls_ambiguous_flat_dict():
    """If multiple tools share the same required param, flat dict should not guess."""
    mapping = {"query": ["web_search", "duckduckgo_search"]}
    result, _ = _try_parse_tool_calls(
        '{"query": "test"}', ["web_search", "duckduckgo_search"], flat_dict_mapping=mapping
    )
    # Should NOT return a tool call when ambiguous
    assert result is None


@pytest.mark.asyncio
async def test_agent_normalize_openai_raw_tool_call(registry: ToolRegistry):
    """LLM returns OpenAI raw tool_call shape with function.arguments as JSON string."""
    raw_tool_call = {
        "type": "function",
        "function": {
            "name": "mock_tool",
            "arguments": '{"query": "raw test"}',
        },
    }
    first_msg = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
    )
    second_msg = AIMessage(content="Raw tool processed.")

    llm = FakeLLM(responses=[first_msg, second_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Search", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="raw test")
    assert "Raw tool processed" in result["answer"]


def test_try_parse_tool_calls_json_array_in_text():
    """JSON array embedded in text should be parsed."""
    text = 'Some text before [{"mock_tool": {"query": "array test"}}] and after'
    result, _ = _try_parse_tool_calls(text, ["mock_tool"])
    assert result is not None
    assert result[0]["name"] == "mock_tool"
    assert result[0]["args"] == {"query": "array test"}


def test_try_parse_tool_calls_mixed_text_and_json():
    """Mixed text with multiple JSON blocks should extract all tool calls."""
    text = 'First {"mock_tool": {"query": "q1"}} then some text {"mock_tool2": {"url": "u1"}}'
    result, _ = _try_parse_tool_calls(text, ["mock_tool", "mock_tool2"])
    assert result is not None
    assert len(result) == 2


def test_build_tool_prompt_fallback_clear_phase_separation():
    """Fallback prompt should clearly separate tool step from final step."""
    from agent.tools.runner import _build_tool_prompt

    prompt = _build_tool_prompt(max_total=5, max_parallel=2, fallback=True)
    assert "JSON" in prompt
    assert "natural language" in prompt.lower()


@pytest.mark.asyncio
async def test_agent_native_mode_has_minimal_tool_prompt(registry: ToolRegistry):
    """In native mode, the system message contains behavioural instructions + limits,
    but no JSON fallback instructions."""
    llm = FakeLLM()
    runner = AgentRunner(registry, ["mock_tool"], supports_native_tool_calls=True)

    await runner.run(
        llm,
        "Hello",
        [],
        system_instructions="You are a helpful assistant.",
    )

    first_call_messages = llm.calls[0]
    sys_msg = first_call_messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert "You are a helpful assistant." in sys_msg.content
    # Must contain behavioural instructions and limits
    assert "You have access to tools" in sys_msg.content
    assert "up to 10 regular (non-coordination) tools in total" in sys_msg.content
    assert "answer in natural language" in sys_msg.content
    assert "Do not explain your plans before acting" in sys_msg.content
    assert "Do not use XML tags" in sys_msg.content
    # Must NOT contain fallback-specific text
    assert "Available tools:" not in sys_msg.content
    assert "JSON" not in sys_msg.content
    assert "tool_calls mechanism" not in sys_msg.content


@pytest.mark.asyncio
async def test_agent_fallback_mode_has_tool_prompt_in_system(registry: ToolRegistry):
    """In fallback mode, the system message MUST contain tool prompt and tool descriptions."""
    llm = FakeLLMNoBind()
    runner = AgentRunner(registry, ["mock_tool"])

    await runner.run(
        llm,
        "Hello",
        [],
        system_instructions="You are a helpful assistant.",
    )

    first_call_messages = llm.calls[0]
    sys_msg = first_call_messages[0]
    assert isinstance(sys_msg, SystemMessage)
    assert "You are a helpful assistant." in sys_msg.content
    assert "You have access to tools" in sys_msg.content
    assert "Available tools:" in sys_msg.content


@pytest.mark.asyncio
async def test_agent_native_mode_final_answer_sees_tool_results_at_max_iterations(registry: ToolRegistry):
    """When max_iterations is reached in native mode, the final answer must include tool results."""
    # Force max_iterations by having the LLM always return a tool call
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "q1"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="Final answer with results.")

    llm = FakeLLM(responses=[tool_msg, tool_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], supports_native_tool_calls=True, max_iterations=2)

    result = await runner.run(llm, "test", [], "")

    # The tool should have been called
    tool = registry.get_tool("mock_tool")
    assert tool.execute.call_count >= 1
    # The final answer should contain the text from the final LLM call
    assert "Final answer" in result["answer"]


@pytest.mark.asyncio
async def test_agent_native_mode_keeps_tool_messages_in_history(registry: ToolRegistry):
    """In native mode, ToolMessage objects must remain in the conversation history."""
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "q1"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="Done.")

    llm = FakeLLM(responses=[tool_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], supports_native_tool_calls=True)

    await runner.run(llm, "test", [], "")

    # Second LLM call should include a ToolMessage with the tool result
    assert len(llm.calls) >= 2
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    assert tool_messages[0].content == "tool result"


def test_light_compress_messages_removes_clutter_keeps_tools():
    """_light_compress_messages removes empty AIMessages and internal SystemMessages
    but preserves ToolMessages and AIMessages with content."""
    from agent.tools.runner import AgentRunner
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello"),
        AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": "t1"}]),
        ToolMessage(content="result", tool_call_id="t1"),
        SystemMessage(content="Answer the user, or call another tool if needed."),
        AIMessage(content="Thanks for the result."),
    ]

    compressed = AgentRunner._light_compress_messages(messages)

    # System prompt kept
    assert any(isinstance(m, SystemMessage) and "helpful assistant" in m.content for m in compressed)
    # HumanMessage kept
    assert any(isinstance(m, HumanMessage) for m in compressed)
    # Empty AIMessage removed
    assert not any(isinstance(m, AIMessage) and not m.content.strip() and not getattr(m, "tool_calls", None) for m in compressed)
    # ToolMessage kept
    assert any(isinstance(m, ToolMessage) and m.content == "result" for m in compressed)
    # Internal SystemMessage removed
    assert not any(isinstance(m, SystemMessage) and m.content == "Answer the user, or call another tool if needed." for m in compressed)
    # AIMessage with content kept
    assert any(isinstance(m, AIMessage) and "Thanks" in m.content for m in compressed)


@pytest.mark.asyncio
async def test_agent_fallback_mode_compresses_tool_messages(registry: ToolRegistry):
    """In fallback mode, ToolMessages are compressed into a SystemMessage."""
    first_msg = AIMessage(content='{"mock_tool": {"query": "q1"}}')
    final_msg = AIMessage(content="Done.")

    llm = FakeLLMNoBind(responses=[first_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    await runner.run(llm, "test", [], "")

    # Second LLM call should NOT contain raw ToolMessages
    assert len(llm.calls) >= 2
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 0
    # Instead, tool results should be present somewhere in the messages
    all_content = " ".join(m.content or "" for m in second_call_messages)
    assert "tool result" in all_content


def test_try_parse_tool_calls_duplicate_blocks():
    """Identical JSON blocks should each be parsed and removed from remaining text."""
    from agent.tools.runner import _try_parse_tool_calls

    text = '{"mock_tool": {"query": "dup"}} some text {"mock_tool": {"query": "dup"}}'
    result, remaining = _try_parse_tool_calls(text, ["mock_tool"])
    assert result is not None
    assert len(result) == 2
    assert "some text" in remaining


@pytest.mark.asyncio
async def test_agent_final_answer_retries_when_empty_after_successful_tool(registry: ToolRegistry):
    """If the final answer is empty after successful tool calls, retry and use the answer from retry."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "retry test"}, "id": "call_1"}],
    )
    trigger_final = AIMessage(content="Here is my answer.")  # Triggers finalization path
    empty_final = AIMessage(content="")  # First final answer attempt is empty
    good_final = AIMessage(content="The answer from retry.")

    llm = FakeLLMNoBind(responses=[first_msg, trigger_final, empty_final, good_final])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="retry test")
    assert "The answer from retry." in result["answer"]
    # Should NOT have the generic "search tools unreachable" fallback
    assert "nicht erreichbar" not in result["answer"].lower()
    assert "Bitte versuche es sp\u00e4ter" not in result["answer"]


@pytest.mark.asyncio
async def test_agent_final_answer_fallback_after_three_empty_attempts(registry: ToolRegistry):
    """After 3 empty final answer attempts, use the appropriate fallback when tools were successful."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "fallback test"}, "id": "call_1"}],
    )
    trigger_final = AIMessage(content="Here is my answer.")
    empty1 = AIMessage(content="")
    empty2 = AIMessage(content="")
    empty3 = AIMessage(content="")

    llm = FakeLLMNoBind(responses=[first_msg, trigger_final, empty1, empty2, empty3])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="fallback test")
    # In fallback mode tool results are stored as HumanMessages, so _prepare_final_messages
    # cannot distinguish successful vs failed results. The generic fallback is expected.
    assert "nicht erreichbar" in result["answer"].lower()
    assert "Bitte versuche es sp\u00e4ter" in result["answer"]


@pytest.mark.asyncio
async def test_agent_final_answer_uses_search_error_fallback_when_all_tools_failed():
    """If all tools fail and final answer is empty, use the search tools unreachable fallback."""

    class FailingTool:
        enabled = True
        tool_schema = {
            "type": "function",
            "function": {
                "name": "failing_tool",
                "description": "A tool that fails",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            },
        }

        async def execute(self, **kwargs):
            raise RuntimeError("Connection refused")

    reg = ToolRegistry()
    reg.register_builtin("failing_tool", FailingTool())

    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "failing_tool", "args": {"query": "fail test"}, "id": "call_1"}],
    )
    trigger_final = AIMessage(content="Here is my answer.")
    empty1 = AIMessage(content="")
    empty2 = AIMessage(content="")
    empty3 = AIMessage(content="")

    llm = FakeLLMNoBind(responses=[first_msg, trigger_final, empty1, empty2, empty3])
    runner = AgentRunner(reg, ["failing_tool"])

    result = await runner.run(llm, "test", [], "")

    # Debug: check what the trace says about the tool call
    trace = result.get("agent_trace", {})
    assert trace.get("tool_calls"), f"Expected tool_calls in trace, got: {trace!r}"
    call = trace["tool_calls"][0]
    assert call.get("ok") is False, f"Expected ok=False, got: {call!r}"
    assert "[Error:" in (call.get("output") or ""), f"Expected error output, got: {call!r}"
    # Should have the "search tools unreachable" fallback since all tools failed
    assert "nicht erreichbar" in result["answer"].lower()


@pytest.mark.asyncio
async def test_agent_max_iterations_final_answer_retries_when_empty(registry: ToolRegistry):
    """When max_iterations is reached and final answer is empty, retry should be attempted."""
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "max_iter test"}, "id": "call_1"}],
    )
    empty_final = AIMessage(content="")
    good_final = AIMessage(content="Max iterations answer.")

    llm = FakeLLM(responses=[tool_msg, empty_final, good_final])
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=1)

    result = await runner.run(llm, "test", [], "")

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="max_iter test")
    assert "Max iterations answer." in result["answer"]
    assert "nicht erreichbar" not in result["answer"].lower()


@pytest.mark.asyncio
async def test_agent_usage_metadata_aggregation(registry: ToolRegistry):
    """Agent usage metadata should sum tokens across iterations and average TPS."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "usage test"}, "id": "call_1"}],
    )
    first_msg.usage_metadata = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "reasoning_tokens": 5,
    }

    second_msg = AIMessage(content="Done.")
    second_msg.usage_metadata = {
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "total_tokens": 250,
        "reasoning_tokens": 10,
    }

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    usage = result.get("agent_usage_metadata")
    assert usage is not None
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 70
    assert usage["total_tokens"] == 370
    assert usage["reasoning_tokens"] == 15
    # FakeLLM returns instantly (<1ms LLM-active time) — tokens_per_second
    # is correctly omitted; verify the rate path with realistic timing.
    for u in runner._iteration_usages:
        u["_response_time_ms"] = 2000.0
    agg = runner._aggregate_iteration_usages(runner._iteration_usages)
    assert agg["tokens_per_second"] == pytest.approx(70 / 4.0, rel=0.01)


@pytest.mark.asyncio
async def test_agent_memory_not_auto_appended_when_already_injected(registry: ToolRegistry):
    """With {{agent_memory}} in the instructions (already_injected=True), the
    runner substitutes the placeholder per iteration instead of appending a
    second [Persistent Memory] block — so mid-run writes stay visible."""
    memory_store = AsyncMock()
    memory_store.get_memory = AsyncMock(return_value="user likes tea")
    llm = FakeLLM(responses=[AIMessage(content="Done.")])
    runner = AgentRunner(
        registry, ["mock_tool"],
        agent_memory_enabled=True,
        agent_memory_store=memory_store,
        conversation_id="conv-1",
        agent_memory_already_injected=True,
    )
    async for _ in runner.run_streaming(llm, "test", [], "Instructions with {{agent_memory}}."):
        pass
    first_call_messages = llm.calls[0]
    first_system_msg = first_call_messages[0]
    assert isinstance(first_system_msg, SystemMessage)
    assert "[Persistent Memory]" not in first_system_msg.content
    assert "user likes tea" in first_system_msg.content
    assert "{{agent_memory}}" not in first_system_msg.content
    memory_store.get_memory.assert_called()


@pytest.mark.asyncio
async def test_agent_memory_auto_appended_when_not_injected(registry: ToolRegistry):
    """When agent_memory_already_injected=False, the runner must append the memory snippet."""
    memory_store = AsyncMock()
    memory_store.get_memory = AsyncMock(return_value="user likes tea")
    llm = FakeLLM(responses=[AIMessage(content="Done.")])
    runner = AgentRunner(
        registry, ["mock_tool"],
        agent_memory_enabled=True,
        agent_memory_store=memory_store,
        conversation_id="conv-1",
        agent_memory_already_injected=False,
    )
    async for _ in runner.run_streaming(llm, "test", [], "Instructions without placeholder."):
        pass
    first_call_messages = llm.calls[0]
    first_system_msg = first_call_messages[0]
    assert isinstance(first_system_msg, SystemMessage)
    assert "[Persistent Memory]" in first_system_msg.content
    assert "user likes tea" in first_system_msg.content
    memory_store.get_memory.assert_awaited_once_with("conv-1", "default")


@pytest.mark.asyncio
async def test_agent_native_mode_direct_output_no_second_call(registry: ToolRegistry):
    """In native mode, a direct answer without tool calls must be emitted immediately
    without triggering a second LLM call for final-answer generation."""
    direct = AIMessage(content="Hello user, I can answer this directly.")
    llm = FakeLLM(responses=[direct])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Simple question", [], "You are helpful.")

    # Only one LLM call should have been made
    assert len(llm.calls) == 1
    # The answer must contain the direct response
    assert "Hello user, I can answer this directly." in result["answer"]
    assert result["agent_trace"]["fallback_mode"] is False
    assert result["agent_trace"]["finalization"] == "no_tool_calls"


@pytest.mark.asyncio
async def test_agent_native_mode_tool_name_in_answer_no_retry(registry: ToolRegistry):
    """In native mode, mentioning a tool name in natural language must NOT trigger
    a JSON-format retry (that guard is fallback-only)."""
    direct = AIMessage(content="I don't need mock_tool for this simple question.")
    llm = FakeLLM(responses=[direct])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "Simple question", [], "You are helpful.")

    # Only one LLM call – no retry
    assert len(llm.calls) == 1
    assert "I don't need mock_tool for this simple question." in result["answer"]
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_agent_streaming_native_mode_direct_output(registry: ToolRegistry):
    """Streaming in native mode should yield content directly when no tool calls are present."""
    llm = FakeLLMWithReasoning(content="Streaming answer.", reasoning="Let me think...")
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm, "What is up?", [], system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    full_content = "".join(c.get("content", "") for c in chunks)
    full_reasoning = "".join(c.get("reasoning", "") for c in chunks)

    assert "Streaming answer." in full_content
    assert "Let me think..." in full_reasoning
    # Only one LLM call in native mode for a direct answer
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agent_streaming_native_mode_no_duplicate_output(registry: ToolRegistry):
    """In native mode, content streamed during the LLM call must not be yielded again at the end."""
    llm = FakeLLMWithReasoning(content="Hello world.", reasoning="")
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = []
    async for chunk in runner.run_streaming(
        llm, "Say hello", [], system_instructions="You are helpful.",
    ):
        chunks.append(chunk)

    content_chunks = [c for c in chunks if c.get("content")]
    total_content = "".join(c.get("content", "") for c in content_chunks)

    # Content must appear exactly once, not duplicated by a final block yield
    assert total_content == "Hello world."
    # There should be at least one content chunk (i.e. streamed in real time)
    assert len(content_chunks) >= 1


def test_refresh_system_prompt_strips_old_and_prepends_fresh(registry: ToolRegistry):
    """_refresh_system_prompt removes stale SystemMessages and puts a fresh one at index 0."""
    runner = AgentRunner(registry, ["mock_tool"])
    messages = [
        SystemMessage(content="old system"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
        SystemMessage(content="internal guidance"),
        ToolMessage(content="result", tool_call_id="call_1"),
    ]
    refreshed = runner._refresh_system_prompt(
        messages,
        system_instructions="You are helpful.",
        fallback_mode=False,
        tool_schemas=runner.registry.tool_schemas(["mock_tool"]),
    )
    # Only one SystemMessage at the very beginning
    sys_msgs = [m for m in refreshed if isinstance(m, SystemMessage)]
    assert len(sys_msgs) == 1
    assert isinstance(refreshed[0], SystemMessage)
    # All non-system messages preserved in order (internal guidance SystemMessage is stripped)
    non_system = [m for m in refreshed if not isinstance(m, SystemMessage)]
    assert len(non_system) == 3
    assert isinstance(non_system[0], HumanMessage)
    assert isinstance(non_system[1], AIMessage)
    assert isinstance(non_system[2], ToolMessage)


@pytest.mark.asyncio
async def test_agent_empty_tool_call_message_replaced_in_final_answer(registry: ToolRegistry):
    """Empty AIMessages with tool_calls are replaced by descriptive text
    in the final answer so non-reasoning models (e.g. Qwen via LM Studio)
    can see the tool calls as normal conversation text."""
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "empty test"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="The result is: tool result")

    # max_iterations=1 forces the runner into the final-answer path
    # where _prepare_final_messages is called and the fix applies.
    llm = FakeLLM(responses=[tool_msg, final_msg])
    runner = AgentRunner(registry, ["mock_tool"], supports_native_tool_calls=True, max_iterations=1)

    result = await runner.run(llm, "Search", [], "")

    # Tool should have been executed
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="empty test")
    assert "The result is: tool result" in result["answer"]

    # The final-answer LLM call (call index 1) must contain the
    # empty AIMessage replaced by descriptive text
    assert len(llm.calls) == 2
    final_call_messages = llm.calls[1]
    ai_msgs = [m for m in final_call_messages if isinstance(m, AIMessage)]
    descriptive_ai = [
        m for m in ai_msgs
        if "I called the following tools" in (getattr(m, "content", "") or "")
    ]
    assert len(descriptive_ai) >= 1


@pytest.mark.asyncio
async def test_execute_tool_surfaces_ignored_arguments(registry: ToolRegistry):
    """Unknown tool arguments must be reported to the model, not silently dropped.

    A model that guesses parameter names (e.g. ``start`` instead of ``offset``)
    would otherwise get default-arg results with no feedback about why repeated
    calls return identical output.
    """
    from agent.tools.runner import AgentTrace

    runner = AgentRunner(registry, ["mock_tool"])
    trace = AgentTrace()
    runner._trace_id = "testtrace"

    _display, tool_msg, _imgs = await runner._execute_tool(
        "mock_tool",
        {"query": "hello", "start": 5, "char_offset": 10},
        "call_1",
        iteration=1,
        trace=trace,
    )

    assert "Tool argument note" in tool_msg.content
    assert "start" in tool_msg.content
    assert "char_offset" in tool_msg.content
    # The valid arg must still have been executed
    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once()
    assert tool.execute.call_args.kwargs.get("query") == "hello" or \
        tool.execute.call_args[1].get("query") == "hello"


@pytest.mark.asyncio
async def test_execute_tool_notes_mistyped_argument(registry: ToolRegistry):
    """An argument with an un-coercible type is dropped and reported."""
    from agent.tools.validation import validate_and_normalize_args

    schema = {
        "type": "function",
        "function": {
            "name": "reader",
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "default": 0},
                    "source": {"type": "string"},
                },
                "required": ["source"],
            },
        },
    }
    normalized, error, notes = validate_and_normalize_args(
        "reader", {"source": "doc.md", "offset": "middle"}, schema
    )
    assert error is None
    assert normalized["offset"] == 0  # fell back to schema default
    assert any("offset" in n for n in notes)


@pytest.mark.asyncio
async def test_context_compression_preserves_tool_call_ids(registry: ToolRegistry):
    """Native-mode compression must keep real tool_call_ids and spare the fresh batch.

    Previously all ToolMessages were replaced by a single message with the
    fabricated id ``compressed_summary``, which left the AIMessage's tool_calls
    unanswered (protocol violation on strict APIs) and could hide fresh results.
    """
    from agent.tools.runner import RollingHistorySettings

    runner = AgentRunner(
        registry,
        ["mock_tool"],
        rolling_history_settings=RollingHistorySettings(
            enabled=True,
            context_window=2_000,
            max_output_tokens=256,
            rag_reserved_tokens=0,
            min_recent_turns=1,
            safety_margin_ratio=0.05,
            summary_timeout_seconds=1.0,
            model="gpt-4o-mini",
            system_instructions="",
        ),
    )

    old_output = "lorem ipsum dolor sit amet consectetur " * 150
    fresh_output = "FRESH_RESULT_" + ("data " * 40)
    messages: list[BaseMessage] = [
        SystemMessage(content="system " * 50),
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[
            {"name": "mock_tool", "args": {}, "id": "call_old_1"},
            {"name": "mock_tool", "args": {}, "id": "call_old_2"},
        ]),
        ToolMessage(content=old_output, tool_call_id="call_old_1"),
        ToolMessage(content=old_output, tool_call_id="call_old_2"),
        AIMessage(content="", tool_calls=[
            {"name": "mock_tool", "args": {}, "id": "call_fresh"},
        ]),
        ToolMessage(content=fresh_output, tool_call_id="call_fresh"),
    ]

    compressed = await runner._check_and_compress_context(messages, use_fallback=False)

    tool_msgs = [m for m in compressed if isinstance(m, ToolMessage)]
    ids = {m.tool_call_id for m in tool_msgs}
    assert "compressed_summary" not in ids
    assert "call_old_1" in ids
    assert "call_old_2" in ids
    assert "call_fresh" in ids

    # Old outputs were truncated in place; the fresh batch is untouched.
    fresh = next(m for m in tool_msgs if m.tool_call_id == "call_fresh")
    assert fresh.content == fresh_output
    old1 = next(m for m in tool_msgs if m.tool_call_id == "call_old_1")
    assert len(old1.content) < len(old_output)


@pytest.mark.asyncio
async def test_context_compression_fallback_merges_into_human_message(registry: ToolRegistry):
    """Fallback mode may merge old outputs (no tool protocol to preserve)."""
    from agent.tools.runner import RollingHistorySettings

    runner = AgentRunner(
        registry,
        ["mock_tool"],
        rolling_history_settings=RollingHistorySettings(
            enabled=True,
            context_window=2_000,
            max_output_tokens=256,
            rag_reserved_tokens=0,
            min_recent_turns=1,
            safety_margin_ratio=0.05,
            summary_timeout_seconds=1.0,
            model="gpt-4o-mini",
            system_instructions="",
        ),
    )
    old_output = "lorem ipsum dolor sit amet consectetur " * 150
    messages: list[BaseMessage] = [
        SystemMessage(content="system " * 50),
        HumanMessage(content="question"),
        AIMessage(content="called tools"),
        ToolMessage(content=old_output, tool_call_id="call_old_1"),
        ToolMessage(content=old_output, tool_call_id="call_old_2"),
        AIMessage(content="next"),
        ToolMessage(content="fresh", tool_call_id="call_fresh"),
    ]

    compressed = await runner._check_and_compress_context(messages, use_fallback=True)
    human = [m for m in compressed if isinstance(m, HumanMessage)]
    assert any("Summarized tool results" in (m.content or "") for m in human)
    # fresh batch untouched
    tool_msgs = [m for m in compressed if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["call_fresh"]


@pytest.mark.asyncio
async def test_execute_tool_keeps_normal_output_verbatim(registry: ToolRegistry):
    """Normal-sized tool outputs must reach the model verbatim, not LLM-summarized.

    With a 32k context window a ~4k-token result must pass through unchanged;
    condensing fresh outputs made repeated paginated reads look identical.
    """
    from agent.tools.runner import AgentTrace, RollingHistorySettings

    big_output = "verifiable output block " * 300  # ~1200 tokens, below ctx//4
    tool = registry.get_tool("mock_tool")
    tool.execute = AsyncMock(return_value=big_output)

    def _exploding_factory():
        raise AssertionError("summary LLM must not be invoked for normal outputs")

    runner = AgentRunner(
        registry,
        ["mock_tool"],
        rolling_history_settings=RollingHistorySettings(
            enabled=True,
            context_window=32_768,
            max_output_tokens=2_048,
            rag_reserved_tokens=0,
            min_recent_turns=1,
            safety_margin_ratio=0.05,
            summary_timeout_seconds=1.0,
            model="gpt-4o-mini",
            system_instructions="",
        ),
        summary_llm_factory=_exploding_factory,
    )
    trace = AgentTrace()
    runner._trace_id = "testtrace"

    _display, tool_msg, _imgs = await runner._execute_tool(
        "mock_tool", {"query": "read"}, "call_1", iteration=1, trace=trace
    )
    assert tool_msg.content == big_output
    assert trace.tool_calls[-1].summarized is False


class FakeLLMTruncatedThenTool:
    """First stream ends with finish_reason='length' and no tool calls —
    simulating an output-limit truncation mid-turn. Later calls behave normally."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "FakeLLMTruncatedThenTool":
        return self

    async def astream(self, messages: list, **kwargs: Any):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="Ich erstelle jetzt die Datei. ",
                response_metadata={"finish_reason": "length"},
            )
        elif self.calls == 2:
            yield AIMessageChunk(
                content="",
                tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "c1"}],
            )
        else:
            yield AIMessageChunk(content="Done: tool result")


class FakeLLMAlwaysTruncated:
    """Every stream ends truncated — the retry guard must still terminate."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "FakeLLMAlwaysTruncated":
        return self

    async def astream(self, messages: list, **kwargs: Any):
        self.calls += 1
        yield AIMessageChunk(
            content="partial text ",
            response_metadata={"finish_reason": "length"},
        )


@pytest.mark.asyncio
async def test_agent_truncated_response_retries_instead_of_finalizing(registry: ToolRegistry):
    """finish_reason='length' with no tool calls must NOT be treated as a
    final answer — the runner nudges the model and continues the loop."""
    llm = FakeLLMTruncatedThenTool()
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(
        llm,
        "Erstelle die Übersichtsdatei",
        [],
        system_instructions="You are helpful.",
    )

    tool = registry.get_tool("mock_tool")
    tool.execute.assert_called_once_with(query="x")
    assert "Done: tool result" in result["answer"]
    # 3 LLM calls: truncated turn → tool call → final answer
    assert llm.calls == 3
    assert result["agent_trace"]["finalization"] == "no_tool_calls"


@pytest.mark.asyncio
async def test_agent_truncated_response_eventually_terminates(registry: ToolRegistry):
    """A pathological model that keeps truncating still ends at max_iterations —
    the retry consumes iterations and never loops forever."""
    llm = FakeLLMAlwaysTruncated()
    runner = AgentRunner(registry, ["mock_tool"], max_iterations=3)

    result = await runner.run(
        llm,
        "Do something",
        [],
        system_instructions="You are helpful.",
    )

    # The last truncated turn finalizes via no_tool_calls — the run terminates
    # even though every iteration hit the output limit.
    assert result["agent_trace"]["finalization"] == "no_tool_calls"
    assert "partial text" in result["answer"]
    # 3 iteration calls + up to 3 final-answer attempts are bounded
    assert llm.calls <= 7
    # 3 truncated iterations + up to 3 forced-finalization attempts
    assert llm.calls <= 6


@pytest.mark.asyncio
async def test_execute_tool_repairs_json_string_array():
    """Array-typed args sent as JSON strings are parsed, not wrapped.

    Models frequently serialize nested arrays (e.g. document sections) as a
    JSON string; before the repair this crashed tools that iterated dict items.
    """
    from agent.tools.validation import validate_and_normalize_args

    schema = {
        "type": "function",
        "function": {
            "name": "docgen",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["sections"],
            },
        },
    }
    normalized, error, notes = validate_and_normalize_args(
        "docgen", {"sections": '[{"heading": "A", "paragraphs": ["x"]}]'}, schema
    )
    assert error is None
    assert normalized["sections"] == [{"heading": "A", "paragraphs": ["x"]}]
    assert any("parsed from a JSON string" in n for n in notes)


@pytest.mark.asyncio
async def test_execute_tool_invalid_json_string_wrapped():
    """An invalid JSON string for an array arg keeps the wrap-into-list behavior."""
    from agent.tools.validation import validate_and_normalize_args

    schema = {
        "type": "function",
        "function": {
            "name": "docgen",
            "parameters": {
                "type": "object",
                "properties": {"tags": {"type": "array"}},
                "required": ["tags"],
            },
        },
    }
    normalized, error, notes = validate_and_normalize_args(
        "docgen", {"tags": "not-json["}, schema
    )
    assert error is None
    assert normalized["tags"] == ["not-json["]
    assert any("wrapped into an array" in n for n in notes)


# -- rendered-image injection --------------------------------------------------

def _img_tool_registry() -> ToolRegistry:
    """Registry whose mock tool returns a text result plus image data URIs."""
    reg = ToolRegistry()
    img_tool = MagicMock()
    img_tool.execute = AsyncMock(return_value={
        "content": "Rendered 1 page image of 'doc.pdf'",
        "images": ["data:image/png;base64,AAA=", "data:image/png;base64,BBB="],
    })
    img_tool.enabled = True
    img_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "img_tool",
            "description": "renders",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    reg.register_builtin("img_tool", img_tool)
    return reg


@pytest.mark.asyncio
async def test_tool_images_become_human_message(registry):
    """Tool results carrying `images` must reach the model as image_url blocks
    in a follow-up HumanMessage (not inside the ToolMessage)."""
    reg = _img_tool_registry()
    first = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "doc.pdf"}, "id": "c1"}],
    )
    llm = FakeLLM(responses=[first, AIMessage(content="I see the page.")])
    runner = AgentRunner(reg, ["img_tool"])
    result = await runner.run(llm, "look", [], system_instructions="sys")

    assert result["answer"] == "I see the page."
    # second LLM call received: tool message + follow-up image HumanMessage
    sent = llm.calls[1]
    img_msgs = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m.content)
    ]
    assert len(img_msgs) == 1
    blocks = [p for p in img_msgs[0].content if p.get("type") == "image_url"]
    assert [b["image_url"]["url"] for b in blocks] == [
        "data:image/png;base64,AAA=", "data:image/png;base64,BBB=",
    ]
    # tagged for pruning; user uploads would NOT carry this marker
    assert img_msgs[0].additional_kwargs.get("workspace_images") is True
    # base64 must never land in the ToolMessage / trace
    tool_msgs = [m for m in sent if isinstance(m, ToolMessage)]
    assert all("base64" not in str(m.content) for m in tool_msgs)


@pytest.mark.asyncio
async def test_prune_keeps_only_latest_image_message(registry):
    """On a later iteration only the most recent injected image message keeps
    its image blocks; older ones are replaced by an omission marker."""
    reg = _img_tool_registry()
    call1 = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "a.pdf"}, "id": "c1"}],
    )
    call2 = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "b.pdf"}, "id": "c2"}],
    )
    llm = FakeLLM(responses=[call1, call2, AIMessage(content="done")])
    runner = AgentRunner(reg, ["img_tool"])
    await runner.run(llm, "look twice", [], system_instructions="sys")

    sent = llm.calls[2]
    img_msgs = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and (m.additional_kwargs or {}).get("workspace_images")
    ]
    assert len(img_msgs) == 2
    older, newer = img_msgs
    assert not any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in older.content
    ), "older image message should be pruned"
    assert any("omitted" in (p.get("text") or "") for p in older.content if isinstance(p, dict))
    assert any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in newer.content
    ), "newest image message keeps its images"


@pytest.mark.asyncio
async def test_prune_never_touches_user_images(registry):
    """multimodal_input user uploads are not tagged and must never be pruned."""
    reg = _img_tool_registry()
    call1 = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "a.pdf"}, "id": "c1"}],
    )
    call2 = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "b.pdf"}, "id": "c2"}],
    )
    llm = FakeLLM(responses=[call1, call2, AIMessage(content="done")])
    runner = AgentRunner(reg, ["img_tool"])
    user_mm = HumanMessage(content=[
        {"type": "text", "text": "see my upload"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,USER="}},
    ])
    await runner.run(llm, "look", [], system_instructions="sys", multimodal_input=user_mm)

    sent = llm.calls[2]
    user_msgs = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and not (m.additional_kwargs or {}).get("workspace_images")
    ]
    assert any(
        isinstance(p, dict) and p.get("type") == "image_url" and "USER=" in p["image_url"]["url"]
        for m in user_msgs for p in m.content
    ), "user-uploaded image must survive pruning"


@pytest.mark.asyncio
async def test_registry_images_passthrough():
    """registry.execute must pass `images` through and filter to data URIs."""
    reg = ToolRegistry()
    t = MagicMock()
    t.execute = AsyncMock(return_value={
        "content": "ok",
        "images": [
            "data:image/png;base64,AAA=",
            "http://evil/img",
            42,
            "data:image/jpeg;base64,CCC=",
        ],
    })
    t.enabled = True
    reg.register_builtin("t", t)
    out = await reg.execute("t", {})
    assert out["ok"] is True
    assert out["images"] == [
        "data:image/png;base64,AAA=", "data:image/jpeg;base64,CCC=",
    ]


@pytest.mark.asyncio
async def test_fallback_mode_images_merge_into_result_message(registry):
    """In JSON-fallback mode (no tool protocol), images merge into the same
    HumanMessage that carries the tool result text."""
    reg = _img_tool_registry()
    llm = FakeLLM(responses=[
        AIMessage(content='{"img_tool": {"path": "doc.pdf"}}'),
        AIMessage(content="I see it"),
        AIMessage(content="fallback done"),
    ])
    runner = AgentRunner(reg, ["img_tool"], supports_native_tool_calls=False)
    result = await runner.run(llm, "look", [], system_instructions="sys")
    # fallback mode ends with a dedicated final-answer call — any of the
    # post-tool responses proves the loop completed without errors
    assert result["answer"] in ("I see it", "fallback done")
    sent = llm.calls[1]
    merged = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and any("img_tool" in (p.get("text") or "") for p in m.content if isinstance(p, dict))
        and any(p.get("type") == "image_url" for p in m.content if isinstance(p, dict))
    ]
    assert merged, "fallback mode should merge result text + images in one message"


@pytest.mark.asyncio
async def test_fallback_multibatch_does_not_resurrect_old_images(registry):
    """Regression: when a second fallback batch squashes old 'Tool' messages,
    it must not re-attach images from earlier batches — only its own."""
    reg = _img_tool_registry()
    batch1 = AIMessage(content='{"img_tool": {"path": "a.pdf"}, "mock_tool": {"query": "x"}}')
    batch2 = AIMessage(content='{"img_tool": {"path": "b.pdf"}, "mock_tool": {"query": "y"}}')
    llm = FakeLLM(responses=[batch1, batch2, AIMessage(content="mid"), AIMessage(content="final")])
    runner = AgentRunner(reg, ["img_tool", "mock_tool"], supports_native_tool_calls=False)
    await runner.run(llm, "two rounds", [], system_instructions="sys")

    # The final LLM call: the batch-1 aggregated message was pruned at the
    # iteration boundary — its base64 is gone (squash + prune), only the
    # newest batch's images remain.
    sent = llm.calls[-1]
    tagged = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and (m.additional_kwargs or {}).get("workspace_images")
    ]
    assert len(tagged) == 2
    older, newer = tagged
    older_imgs = [p for p in older.content if isinstance(p, dict) and p.get("type") == "image_url"]
    newer_imgs = [p for p in newer.content if isinstance(p, dict) and p.get("type") == "image_url"]
    assert not older_imgs, "batch-1 images must not be resurrected into batch-2's message"
    assert newer_imgs, "batch-2 images must be present"


# -- run-content dedup (exactly-once + budget) ---------------------------------

def _img_tool_registry_seq(uris_per_call: int = 2) -> ToolRegistry:
    """Registry whose image tool returns UNIQUE data URIs on every call."""
    reg = ToolRegistry()
    img_tool = MagicMock()
    calls = {"n": 0}

    async def _exec(**kw: Any) -> Dict[str, Any]:
        calls["n"] += 1
        i = calls["n"]
        return {
            "content": f"rendered batch {i}",
            "images": [
                f"data:image/png;base64,IM{i}_{k}=" for k in range(uris_per_call)
            ],
        }

    img_tool.execute = AsyncMock(side_effect=_exec)
    img_tool.enabled = True
    img_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "img_tool",
            "description": "renders",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    reg.register_builtin("img_tool", img_tool)
    return reg


def _img_parts(msg: HumanMessage):
    return [
        p for p in msg.content
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]


def _stub_texts(msg: HumanMessage):
    return [
        p["text"] for p in msg.content
        if isinstance(p, dict) and p.get("type") == "text" and "#img-" in (p.get("text") or "")
    ]


@pytest.mark.asyncio
async def test_dedup_keeps_distinct_images():
    """Different images are NOT deduped — the old positional pruning is gone:
    a later view call must not blank out an earlier message's unique pages."""
    reg = _img_tool_registry_seq()
    calls = [
        AIMessage(content="", tool_calls=[{"name": "img_tool", "args": {"path": "a.pdf"}, "id": f"c{i}"}])
        for i in range(1, 3)
    ]
    llm = FakeLLM(responses=[*calls, AIMessage(content="done")])
    runner = AgentRunner(reg, ["img_tool"])
    await runner.run(llm, "look", [], system_instructions="sys")

    sent = llm.calls[-1]
    tagged = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and (m.additional_kwargs or {}).get("workspace_images")
    ]
    assert len(tagged) == 2
    for m in tagged:
        assert _img_parts(m), "distinct image sets must all stay live"


@pytest.mark.asyncio
async def test_dedup_upload_render_collision():
    """The same image arriving as a user upload AND a workspace render must
    appear exactly once — the latest occurrence (the render) keeps it."""
    reg = _img_tool_registry()
    call = AIMessage(
        content="",
        tool_calls=[{"name": "img_tool", "args": {"path": "x.png"}, "id": "c1"}],
    )
    llm = FakeLLM(responses=[call, AIMessage(content="done")])
    runner = AgentRunner(reg, ["img_tool"])
    user_mm = HumanMessage(content=[
        {"type": "text", "text": "my upload"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
    ])
    await runner.run(llm, "look", [], system_instructions="sys", multimodal_input=user_mm)

    sent = llm.calls[-1]
    aaa = [
        (m, p) for m in sent if isinstance(m, HumanMessage) and isinstance(m.content, list)
        for p in m.content
        if isinstance(p, dict) and p.get("image_url", {}).get("url") == "data:image/png;base64,AAA="
    ]
    assert len(aaa) == 1, "identical image must appear exactly once"
    kept_msg, _ = aaa[0]
    assert (kept_msg.additional_kwargs or {}).get("workspace_images") is True
    # the earlier user part became a stub
    assert any(
        "#img-" in (p.get("text") or "")
        for p in user_mm.content if isinstance(p, dict)
    )


@pytest.mark.asyncio
async def test_dedup_budget_evicts_oldest_tagged_first():
    """More than _MAX_CTX_IMAGES unique images → oldest-positioned evicted,
    tagged tool renders before untagged user uploads."""
    reg = _img_tool_registry_seq(uris_per_call=6)
    calls = [
        AIMessage(content="", tool_calls=[{"name": "img_tool", "args": {"path": "d.pdf"}, "id": f"c{i}"}])
        for i in range(1, 4)
    ]
    llm = FakeLLM(responses=[*calls, AIMessage(content="done")])
    runner = AgentRunner(reg, ["img_tool"])
    user_mm = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,USER="}},
    ])
    await runner.run(llm, "look", [], system_instructions="sys", multimodal_input=user_mm)

    sent = llm.calls[-1]
    live_imgs = sum(
        1 for m in sent if isinstance(m, HumanMessage) and isinstance(m.content, list)
        for p in m.content if isinstance(p, dict) and p.get("type") == "image_url"
    )
    # 18 tool images + 1 user image = 19 unique > 16 → 3 evicted, user kept
    assert live_imgs == 16
    assert any(
        p.get("image_url", {}).get("url") == "data:image/png;base64,USER="
        for m in sent if isinstance(m, HumanMessage) and isinstance(m.content, list)
        for p in m.content if isinstance(p, dict)
    ), "user upload must be evicted last"
    # the first tool batch lost its images to the budget
    tagged = [
        m for m in sent
        if isinstance(m, HumanMessage) and isinstance(m.content, list)
        and (m.additional_kwargs or {}).get("workspace_images")
    ]
    assert any("to save context" in t for t in _stub_texts(tagged[0]))


@pytest.mark.asyncio
async def test_dedup_identical_tool_output_becomes_stub(registry):
    """A tool returning the same >1000-char output twice → the earlier
    ToolMessage collapses to a #res-N stub naming the tool."""
    big = "result " + "x" * 2000
    tool = registry.get_tool("mock_tool")
    tool.execute = AsyncMock(return_value=big)
    calls = [
        AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {"query": "same"}, "id": f"c{i}"}])
        for i in range(1, 3)
    ]
    llm = FakeLLM(responses=[*calls, AIMessage(content="done")])
    runner = AgentRunner(registry, ["mock_tool"])
    await runner.run(llm, "go", [], system_instructions="sys")

    sent = llm.calls[-1]
    tool_msgs = [m for m in sent if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert "#res-1" in tool_msgs[0].content and "mock_tool" in tool_msgs[0].content
    assert tool_msgs[1].content == big


@pytest.mark.asyncio
async def test_dedup_distinct_outputs_untouched(registry):
    """Different outputs are never deduped."""
    tool = registry.get_tool("mock_tool")
    tool.execute = AsyncMock(side_effect=["a" * 2000, "b" * 2000])
    calls = [
        AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {"query": i}, "id": f"c{i}"}])
        for i in range(1, 3)
    ]
    llm = FakeLLM(responses=[*calls, AIMessage(content="done")])
    runner = AgentRunner(registry, ["mock_tool"])
    await runner.run(llm, "go", [], system_instructions="sys")

    tool_msgs = [m for m in llm.calls[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == "a" * 2000
    assert tool_msgs[1].content == "b" * 2000


def test_dedup_run_content_idempotent(registry):
    """Running the dedup pass twice must be a no-op — stubs never re-match."""
    runner = AgentRunner(registry, ["mock_tool"])
    uri = "data:image/png;base64,DUP="
    msgs = [
        HumanMessage(content=[
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": uri}},
        ], additional_kwargs={"workspace_images": True}),
        HumanMessage(content=[
            {"type": "text", "text": "second"},
            {"type": "image_url", "image_url": {"url": uri}},
        ], additional_kwargs={"workspace_images": True}),
    ]
    runner._dedup_run_content(msgs)
    after_first = [str(m.content) for m in msgs]
    runner._dedup_run_content(msgs)
    assert [str(m.content) for m in msgs] == after_first
    assert not _img_parts(msgs[0]) and _img_parts(msgs[1])
    assert "img-1" in _stub_texts(msgs[0])[0]


def test_dedup_stable_ids_across_passes(registry):
    """A re-viewed image keeps its #img-N id; new images get fresh numbers."""
    runner = AgentRunner(registry, ["mock_tool"])
    msgs = [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,A="}},
        ], additional_kwargs={"workspace_images": True}),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,B="}},
        ], additional_kwargs={"workspace_images": True}),
    ]
    runner._dedup_run_content(msgs)
    msgs.append(HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,A="}},
    ], additional_kwargs={"workspace_images": True}))
    runner._dedup_run_content(msgs)
    stub = _stub_texts(msgs[0])
    assert stub and "img-1" in stub[0], "re-viewed image keeps its original id"


def test_dedup_never_touches_real_user_text(registry):
    """A user message that happens to start with 'Tool '' is NOT a tool
    result — only messages carrying the internal run_tool_result tag are
    eligible for output dedup."""
    runner = AgentRunner(registry, ["mock_tool"])
    user_text = "Tool 'x' says: " + "y" * 2000
    msgs = [
        HumanMessage(content=user_text),
        HumanMessage(content=user_text),
        # tagged fallback result — identical twice → deduped
        HumanMessage(content="Tool 'mock_tool' result:\n" + "z" * 2000,
                     additional_kwargs={"run_tool_result": True}),
        HumanMessage(content="Tool 'mock_tool' result:\n" + "z" * 2000,
                     additional_kwargs={"run_tool_result": True}),
    ]
    runner._dedup_run_content(msgs)
    assert msgs[0].content == user_text and msgs[1].content == user_text
    assert "#res-" in msgs[2].content and msgs[3].content.endswith("z" * 2000)


class _OrderingTool:
    """Fake tool that records execution order and serializes per 'target'."""

    name = "ordering_tool"
    enabled = True
    tool_schema = {
        "type": "function",
        "function": {
            "name": "ordering_tool",
            "description": "records execution order",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "tag": {"type": "string"},
                    "delay": {"type": "number"},
                },
            },
        },
    }

    def __init__(self) -> None:
        self.events: list = []

    def call_conflict_key(self, args):
        target = args.get("target")
        return target if target else None

    async def execute(self, target=None, tag=None, delay=None, **kwargs):
        self.events.append(("start", tag))
        import asyncio
        await asyncio.sleep(float(delay or 0))
        self.events.append(("end", tag))
        return f"done {tag}"


@pytest.mark.asyncio
async def test_batch_calls_same_conflict_key_run_sequentially(registry):
    """Calls mutating the same target serialize in call order — a second
    same-key call must see the first call's result (read-modify-write
    races otherwise lose every write but the last)."""
    import asyncio  # noqa: F401 — used by the tool
    tool = _OrderingTool()
    registry.register_builtin("ordering_tool", tool)
    llm = FakeLLM(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "ordering_tool", "args": {"target": "f.md", "tag": "a", "delay": 0.05}, "id": "c1"},
            {"name": "ordering_tool", "args": {"target": "f.md", "tag": "b", "delay": 0.0}, "id": "c2"},
            {"name": "ordering_tool", "args": {"target": "g.md", "tag": "c", "delay": 0.0}, "id": "c3"},
        ]),
        AIMessage(content="done"),
    ])
    runner = AgentRunner(registry, ["ordering_tool"])
    result = await runner.run(llm, "go", [], system_instructions="sys")
    assert result["answer"] == "done"
    # 'b' shares a's conflict key → must not start before 'a' ended.
    assert tool.events.index(("start", "b")) > tool.events.index(("end", "a"))
    assert sum(1 for e in tool.events if e[0] == "end") == 3


@pytest.mark.asyncio
async def test_batch_calls_without_conflict_key_stay_parallel(registry):
    """Calls without a conflict key keep the old parallel behaviour."""
    tool = _OrderingTool()
    registry.register_builtin("ordering_tool", tool)
    llm = FakeLLM(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "ordering_tool", "args": {"tag": "a", "delay": 0.05}, "id": "c1"},
            {"name": "ordering_tool", "args": {"tag": "b", "delay": 0.0}, "id": "c2"},
        ]),
        AIMessage(content="done"),
    ])
    runner = AgentRunner(registry, ["ordering_tool"])
    await runner.run(llm, "go", [], system_instructions="sys")
    # No keys → flat parallel: both start before 'a' ends.
    assert tool.events.index(("start", "b")) < tool.events.index(("end", "a"))


def test_code_execution_arg_limit_raised():
    """code_execution.code now allows 100k chars (was 10k) before truncation."""
    from agent.tools.validation import validate_and_normalize_args

    schema = {
        "type": "function",
        "function": {
            "name": "code_execution",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
    normalized, error, notes = validate_and_normalize_args(
        "code_execution", {"code": "x" * 50_000}, schema
    )
    assert error is None
    assert normalized["code"] == "x" * 50_000
    assert not any("truncated" in n for n in notes)

    normalized, error, notes = validate_and_normalize_args(
        "code_execution", {"code": "x" * 150_000}, schema
    )
    assert error is None
    assert len(normalized["code"]) == 100_000
    assert any("truncated" in n for n in notes)


def test_aggregate_iteration_usages_merges_observer_compression_cache():
    """Run total must cover every LLM call — observer and compression usage
    merge into the summed fields and surface as breakdown keys."""
    from agent.tools.runner import AgentRunner

    runner = AgentRunner(registry, ["mock_tool"])
    runner._compression_usage = [
        {"prompt_tokens": 400, "completion_tokens": 60, "total_tokens": 460},
    ]
    runner._observer = type("Obs", (), {"usage": {
        "prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240, "reasoning_tokens": 10,
    }})()

    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
         "reasoning_tokens": 50, "cache_tokens": 80},
        {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    ])

    assert result["iterations"] == 2
    assert result["prompt_tokens"] == 1000 + 500 + 400 + 200
    assert result["completion_tokens"] == 200 + 100 + 60 + 40
    assert result["total_tokens"] == 1200 + 600 + 460 + 240
    assert result["reasoning_tokens"] == 50 + 10
    assert result["cache_tokens"] == 80
    assert result["observer_tokens"] == 240
    assert result["compression_tokens"] == 460


def test_aggregate_iteration_usages_without_observer_or_compression():
    from agent.tools.runner import AgentRunner

    runner = AgentRunner(registry, ["mock_tool"])
    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    ])
    assert result["prompt_tokens"] == 10
    assert "observer_tokens" not in result
    assert "compression_tokens" not in result
    assert "cache_tokens" not in result


@pytest.mark.asyncio
async def test_tool_keepalive_events_carry_flag(monkeypatch, registry):
    """Long tool executions emit periodic tool_progress keepalives flagged as
    synthetic — they keep the wire alive but must not reset the run watchdog."""
    import asyncio

    import agent.tools.runner as runner_module

    monkeypatch.setattr(runner_module, "TOOL_KEEPALIVE_SECONDS", 0.05)

    slow_tool = MagicMock()

    async def _slow(*args, **kwargs):
        await asyncio.sleep(0.3)
        return "slow result"

    slow_tool.execute = AsyncMock(side_effect=_slow)
    slow_tool.enabled = True
    slow_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "slow_tool",
            "description": "A slow tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry.register_builtin("slow_tool", slow_tool)

    llm = FakeLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "slow_tool", "args": {}, "id": "call_1"}],
            ),
            AIMessage(content="Final answer."),
        ]
    )
    runner = AgentRunner(registry, ["slow_tool"])

    keepalives = []
    async for chunk in runner.run_streaming(llm, "test", [], ""):
        if chunk.get("keepalive") is True:
            keepalives.append(chunk)

    assert keepalives, "expected at least one keepalive during the slow tool call"
    for item in keepalives:
        assert item["tool_progress"]["keepalive"] is True
        assert item["tool_progress"]["stage"] == "progress"


@pytest.mark.asyncio
async def test_usage_estimated_when_provider_reports_none(registry: ToolRegistry):
    """Calls without provider usage still count: prompt is estimated from the
    messages actually sent (including tool outputs), completion from the
    generated text — and flagged via estimated_iterations."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Final answer.")

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    usage = result.get("agent_usage_metadata")
    assert usage is not None
    assert usage["iterations"] == 2
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] >= usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["estimated_iterations"] == 2
    # The tool result rode in the second call's prompt — the estimate must
    # exceed a bare-bones prompt.
    assert usage["tool_result_tokens"] > 0
    assert usage["prompt_tokens"] > usage["tool_result_tokens"]
    # FakeLLM calls take <1ms — the rate is omitted, not garbage.
    assert "tokens_per_second" not in usage
    for u in runner._iteration_usages:
        u["_response_time_ms"] = 1000.0
    agg = runner._aggregate_iteration_usages(runner._iteration_usages)
    assert agg["tokens_per_second"] > 0


@pytest.mark.asyncio
async def test_usage_estimation_mixed_provider_and_estimated(registry: ToolRegistry):
    """Only the usage-less call is flagged estimated; provider values win."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Done.")
    second_msg.usage_metadata = {
        "prompt_tokens": 500,
        "completion_tokens": 50,
        "total_tokens": 550,
    }

    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")

    usage = result.get("agent_usage_metadata")
    assert usage is not None
    assert usage["estimated_iterations"] == 1
    # Provider prompt (500) plus estimated first-call prompt (>0).
    assert usage["prompt_tokens"] > 500
    assert usage["completion_tokens"] >= 50


def test_estimate_prompt_tokens_counts_tool_output(registry: ToolRegistry):
    """ToolMessages (tool results inside the prompt) are part of the estimate."""
    from langchain_core.messages import SystemMessage, ToolMessage
    from app.chat_utils import count_tokens

    runner = AgentRunner(registry, ["mock_tool"])
    tool_text = "x " * 500
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        ToolMessage(content=tool_text, tool_call_id="1"),
    ]
    est = runner._estimate_prompt_tokens(msgs)
    assert est > count_tokens(tool_text)
    assert est > count_tokens("sys") + count_tokens("hi") + count_tokens(tool_text)


def test_aggregate_tps_completion_over_llm_time(registry: ToolRegistry):
    """tokens_per_second is output generation rate over LLM-active time —
    observer/compression completions (untimed) must not inflate it."""
    runner = AgentRunner(registry, ["mock_tool"])
    runner._compression_usage = [
        {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    ]
    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 1000.0},
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 2000.0},
    ])
    # 100 iteration-completion tokens over 3s — compression's 500 excluded.
    assert result["tokens_per_second"] == pytest.approx(100 / 3.0, rel=0.01)
    # …but it IS in the totals.
    assert result["completion_tokens"] == 600
    assert "estimated_iterations" not in result


def test_aggregate_estimated_iterations_flag(registry: ToolRegistry):
    runner = AgentRunner(registry, ["mock_tool"])
    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 1000.0, "_estimated": True},
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 1000.0},
    ])
    assert result["estimated_iterations"] == 1


def test_aggregate_reports_llm_generation_ms(registry: ToolRegistry):
    """The aggregate exposes total LLM-active time for metrics consumers."""
    runner = AgentRunner(registry, ["mock_tool"])
    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 1000.0},
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "_response_time_ms": 2500.0},
    ])
    assert result["llm_generation_ms"] == 3500.0


@pytest.mark.asyncio
async def test_rag_search_result_counts_as_rag_tokens(registry: ToolRegistry):
    """rag_search results feed rag_tool_tokens; other tools do not."""
    rag_tool = MagicMock()
    rag_tool.execute = AsyncMock(return_value="retrieved context chunk")
    rag_tool.enabled = True
    rag_tool.tool_schema = {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Vectorstore search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    registry.register_builtin("rag_search", rag_tool)

    first_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "rag_search", "args": {"q": "x"}, "id": "call_1"},
            {"name": "mock_tool", "args": {"query": "y"}, "id": "call_2"},
        ],
    )
    second_msg = AIMessage(content="Done.")
    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["rag_search", "mock_tool"])

    result = await runner.run(llm, "test", [], "")

    usage = result.get("agent_usage_metadata")
    assert usage is not None
    assert usage["rag_tool_tokens"] > 0
    # Only the rag_search result counts — mock_tool's output is excluded.
    assert usage["rag_tool_tokens"] < usage["tool_result_tokens"]


@pytest.mark.asyncio
async def test_no_rag_search_means_zero_rag_tool_tokens(registry: ToolRegistry):
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "call_1"}],
    )
    llm = FakeLLM(responses=[first_msg, AIMessage(content="Done.")])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")
    usage = result.get("agent_usage_metadata")
    assert usage is not None
    assert usage["rag_tool_tokens"] == 0


@pytest.mark.asyncio
async def test_run_streaming_emits_live_usage_per_call(registry: ToolRegistry):
    """Each completed LLM call emits a cumulative live_usage snapshot."""
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "call_1"}],
    )
    second_msg = AIMessage(content="Done.")
    llm = FakeLLM(responses=[first_msg, second_msg])
    runner = AgentRunner(registry, ["mock_tool"])

    chunks = [c async for c in runner.run_streaming(llm, "test", [], system_instructions="sys")]
    live = [c["live_usage"] for c in chunks if c.get("live_usage")]
    assert len(live) >= 2  # one after the tool-call iteration, one after the final answer
    # cumulative: call counts grow monotonically
    calls = [lv["llm_calls"] for lv in live]
    assert calls == sorted(calls) and calls[-1] >= 2
    last = live[-1]
    assert last["prompt_tokens"] > 0
    assert last["completion_tokens"] > 0
    assert last["total_tokens"] >= last["completion_tokens"]
    expected_len = min(calls[-1], 40)
    assert len(last["per_call_completion"]) == expected_len


@pytest.mark.asyncio
async def test_aggregate_includes_per_call_series(registry: ToolRegistry):
    first_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"query": "x"}, "id": "call_1"}],
    )
    llm = FakeLLM(responses=[first_msg, AIMessage(content="Done.")])
    runner = AgentRunner(registry, ["mock_tool"])

    result = await runner.run(llm, "test", [], "")
    usage = result.get("agent_usage_metadata")
    assert usage is not None
    per_call = usage.get("per_call")
    assert isinstance(per_call, list) and len(per_call) >= 2
    for entry in per_call:
        assert set(entry.keys()) >= {"p", "c", "r", "ms"}
        assert entry["c"] >= 0


@pytest.mark.asyncio
async def test_live_usage_includes_finished_subagent_totals(registry: ToolRegistry):
    """The chip counts subagent consumption the moment a subagent reports —
    matching the totals the final metrics badge merges in afterwards."""
    from agent.subagents import SubagentManager, SubagentRecord, make_subagent_state

    runner = AgentRunner(registry, ["mock_tool"])
    manager = SubagentManager({}, "conv-live-usage", None)
    record = SubagentRecord(
        id="sub-1", parent_id=None, owner_key="root", task="t", label="t",
        assistant="a", depth=1, status="done", collected=True,
    )
    record.nested_usage = {
        "prompt_tokens": 5000,
        "completion_tokens": 1200,
        "total_tokens": 6200,
        "reasoning_tokens": 400,
    }
    manager.records["sub-1"] = record
    runner._subagent_state = make_subagent_state(manager, "test")

    llm = FakeLLM(responses=[AIMessage(content="Done.")])
    chunks = [c async for c in runner.run_streaming(llm, "test", [], system_instructions="sys")]
    live = [c["live_usage"] for c in chunks if c.get("live_usage")]
    assert live, "expected at least one live_usage event"
    last = live[-1]
    own = runner._aggregate_iteration_usages(runner._iteration_usages) or {}
    assert last["prompt_tokens"] == (own.get("prompt_tokens") or 0) + 5000
    assert last["completion_tokens"] == (own.get("completion_tokens") or 0) + 1200
    assert last["reasoning_tokens"] == (own.get("reasoning_tokens") or 0) + 400


def test_live_usage_ignores_non_root_subagent_state(registry: ToolRegistry):
    """Nested runners must not double-count: only the root manager owns the
    global subagent usage picture."""
    from types import SimpleNamespace

    runner = AgentRunner(registry, ["mock_tool"])
    runner._iteration_usages = [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}]
    manager = SimpleNamespace(usage_totals=lambda: {"prompt_tokens": 9999, "total_tokens": 9999})
    runner._subagent_state = SimpleNamespace(is_root=False, manager=manager)

    live = runner._live_usage_event()["live_usage"]
    assert live["prompt_tokens"] == 10
    assert live["total_tokens"] == 15


@pytest.mark.asyncio
async def test_aggregate_includes_observer_calls(registry: ToolRegistry):
    """Observer checks are real LLM calls — llm_calls (and the live chip)
    count them, so the aggregate must expose observer_calls too."""
    from types import SimpleNamespace

    runner = AgentRunner(registry, ["mock_tool"])
    runner._observer = SimpleNamespace(
        checks_used=2,
        usage={"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380, "reasoning_tokens": 0},
    )
    runner._iteration_usages = [
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "_response_time_ms": 10.0}
    ]

    usage = runner._aggregate_iteration_usages(runner._iteration_usages)
    assert usage["observer_calls"] == 2
    assert usage["observer_tokens"] == 380

    # …and the live event counts the checks as LLM calls.
    live = runner._live_usage_event()["live_usage"]
    assert live["llm_calls"] == 1 + 2


def test_aggregate_omits_tps_below_1ms(registry: ToolRegistry):
    """Sub-millisecond aggregate LLM time yields no rate — a 0.25ms
    generation window would produce hundreds of thousands of tokens/s."""
    runner = AgentRunner(registry, ["mock_tool"])
    result = runner._aggregate_iteration_usages([
        {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80,
         "_response_time_ms": 0.5},
    ])
    assert "tokens_per_second" not in result
    assert "llm_generation_ms" not in result
