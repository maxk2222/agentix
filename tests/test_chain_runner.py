"""Tests for ChainRunner DAG execution."""

from __future__ import annotations

import pytest
from app.chain_runner import ChainRunner, _topological_sort, _get_predecessors, _get_successors


class _FakeLLM:
    """Stand-in for a non-streaming LangChain chat model."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return {"answer": self.reply}


class FakeRAGApp:
    """Mock RAG application that echoes back inputs with prefixes."""

    def __init__(
        self,
        responses: dict | None = None,
        streaming_tool_calls: dict | None = None,
        streaming_subagents: dict | None = None,
        metadata_llms: dict | None = None,
    ) -> None:
        self._responses = responses or {}
        self._streaming_tool_calls = streaming_tool_calls or {}
        self._streaming_subagents = streaming_subagents or {}
        self._metadata_llms = {name: _FakeLLM(reply) for name, reply in (metadata_llms or {}).items()}
        self.calls: list[dict] = []

    async def get_metadata_llm(self, model_name: str = "", purpose: str = "metadata"):
        self.calls.append({"method": "get_metadata_llm", "model": model_name})
        if model_name in self._metadata_llms:
            return self._metadata_llms[model_name]
        return _FakeLLM("")

    async def run(self, input_text: str, *, model_name: str = "", chat_history=None, **kwargs):
        self.calls.append({"method": "run", "input": input_text, "model": model_name})
        if model_name in self._responses:
            resp = self._responses[model_name]
            if isinstance(resp, dict):
                return resp
            return {"answer": resp, "reasoning": f"thinking about {model_name}"}
        return {"answer": f"[{model_name}:{input_text}]", "reasoning": f"thinking about {model_name}"}

    async def run_streaming(self, user_input: str, *, model_name: str = "", chat_history=None, **kwargs):
        self.calls.append({"method": "run_streaming", "input": user_input, "model": model_name})
        tcs = self._streaming_tool_calls.get(model_name, [])
        for event in self._streaming_subagents.get(model_name, []):
            yield {"content": "", "reasoning": "", "subagent": event}
        if model_name in self._responses:
            resp = self._responses[model_name]
            if isinstance(resp, dict):
                yield {"content": resp.get("answer", ""), "reasoning": resp.get("reasoning", ""), "tool_calls": tcs}
            else:
                yield {"content": resp, "reasoning": f"thinking about {model_name}", "tool_calls": tcs}
        else:
            yield {"content": f"[{model_name}:{user_input}]", "reasoning": f"thinking about {model_name}", "tool_calls": tcs}


class TestTopologicalSort:
    def test_linear_chain(self):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}]
        result = _topological_sort(nodes, edges)
        assert result == ["a", "b", "c"]

    def test_diamond(self):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "c"},
            {"source": "b", "target": "d"},
            {"source": "c", "target": "d"},
        ]
        result = _topological_sort(nodes, edges)
        assert result.index("a") < result.index("b")
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")

    def test_cycle_raises(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]
        with pytest.raises(ValueError, match="Cycle"):
            _topological_sort(nodes, edges)


class TestChainRunner:
    @pytest.fixture
    def simple_flow(self):
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input", "label": "Input"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output", "label": "Output"}},
            ],
            "edges": [{"source": "input", "target": "out"}],
        }

    @pytest.fixture
    def assistant_flow(self):
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input", "label": "Input"}},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "label": "A1", "assistant_name": "Alpha"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output", "label": "Output"}},
            ],
            "edges": [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
            ],
        }

    @pytest.fixture
    def merge_flow(self):
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "label": "A1", "assistant_name": "Alpha"}},
                {"id": "a2", "type": "assistant", "data": {"nodeType": "assistant", "label": "A2", "assistant_name": "Beta"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "a1"},
                {"source": "input", "target": "a2"},
                {"source": "a1", "target": "out"},
                {"source": "a2", "target": "out"},
            ],
        }

    @pytest.fixture
    def reactflow_assistant_flow(self):
        """ReactFlow stores type='chainNode' and logical type in data.nodeType."""
        return {
            "nodes": [
                {"id": "input-1", "type": "chainNode", "data": {"nodeType": "input", "label": "Input"}},
                {"id": "assistant-1", "type": "chainNode", "data": {"nodeType": "assistant", "label": "Assistant", "assistant_name": "Alpha"}},
                {"id": "output-1", "type": "chainNode", "data": {"nodeType": "output", "label": "Output"}},
            ],
            "edges": [
                {"source": "input-1", "target": "assistant-1"},
                {"source": "assistant-1", "target": "output-1"},
            ],
        }

    @pytest.fixture
    def preparatory_flow(self):
        """Chain with a preparatory assistant and a final streaming assistant."""
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "label": "A1", "assistant_name": "Alpha"}},
                {"id": "a2", "type": "assistant", "data": {"nodeType": "assistant", "label": "A2", "assistant_name": "Beta"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "a2"},
                {"source": "a2", "target": "out"},
            ],
        }

    @pytest.mark.asyncio
    async def test_input_to_output_passes_through(self, simple_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(simple_flow, rag)
        result = await runner.run("hello")
        assert result["answer"] == "hello"

    @pytest.mark.asyncio
    async def test_single_assistant_chain(self, assistant_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(assistant_flow, rag)
        result = await runner.run("hello")
        assert result["answer"] == "[Alpha:hello]"
        assert len(rag.calls) == 1
        assert rag.calls[0]["model"] == "Alpha"

    @pytest.mark.asyncio
    async def test_reactflow_chainNode_assistant_invoked(self, reactflow_assistant_flow):
        """When type='chainNode' and data.nodeType='assistant', the assistant must be called."""
        rag = FakeRAGApp()
        runner = ChainRunner(reactflow_assistant_flow, rag)
        result = await runner.run("hello")
        # If node_type resolving is broken (type first), this would echo "hello".
        # With the fix (data.nodeType first), the assistant is invoked.
        assert result["answer"] == "[Alpha:hello]"
        assert len(rag.calls) == 1
        assert rag.calls[0]["model"] == "Alpha"

    @pytest.mark.asyncio
    async def test_merge_node_concatenates(self, merge_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(merge_flow, rag)
        result = await runner.run("hello")
        # Alpha and Beta both receive "hello", merge concatenates their outputs with labels
        assert "Alpha:" in result["answer"]
        assert "Beta:" in result["answer"]
        assert "[Alpha:hello]" in result["answer"]
        assert "[Beta:hello]" in result["answer"]
        # Both assistants were invoked (parallel execution)
        assert len(rag.calls) == 2
        models = {c["model"] for c in rag.calls}
        assert models == {"Alpha", "Beta"}
        # Reasoning from both assistants is aggregated
        assert "thinking about Alpha" in result.get("reasoning", "")
        assert "thinking about Beta" in result.get("reasoning", "")

    @pytest.mark.asyncio
    async def test_missing_assistant_name_returns_error(self, assistant_flow):
        assistant_flow["nodes"][1]["data"]["assistant_name"] = ""
        rag = FakeRAGApp()
        runner = ChainRunner(assistant_flow, rag)
        result = await runner.run("hello")
        assert "Error" in result["answer"]
        assert "has no assistant_name" in result["answer"]

    @pytest.mark.asyncio
    async def test_run_streaming_output(self, assistant_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(assistant_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "[Alpha:hello]"
        assert "thinking about Alpha" in chunks[0].get("reasoning", "")

    @pytest.mark.asyncio
    async def test_run_streaming_no_output_node(self, simple_flow):
        simple_flow["nodes"] = simple_flow["nodes"][:1]  # remove output node
        simple_flow["edges"] = []
        rag = FakeRAGApp()
        runner = ChainRunner(simple_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_streaming_missing_assistant_name_returns_error(self, assistant_flow):
        assistant_flow["nodes"][1]["data"]["assistant_name"] = ""
        rag = FakeRAGApp()
        runner = ChainRunner(assistant_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "Error" in chunks[0]["content"]
        assert "has no assistant_name" in chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_parallel_assistants_no_echo_on_missing_name(self, merge_flow):
        """Parallel assistants with empty names should error, not echo input."""
        merge_flow["nodes"][1]["data"]["assistant_name"] = ""
        merge_flow["nodes"][2]["data"]["assistant_name"] = ""
        rag = FakeRAGApp()
        runner = ChainRunner(merge_flow, rag)
        result = await runner.run("hello")
        # Should contain error messages, not "hello" echoed twice
        assert "has no assistant_name" in result["answer"]
        assert "hello" != result["answer"]
        # Errors are labeled with node labels
        assert "A1:" in result["answer"] or "A2:" in result["answer"]

    @pytest.mark.asyncio
    async def test_parallel_assistants_streaming(self, merge_flow):
        """Parallel assistants in streaming mode should invoke both and merge outputs."""
        rag = FakeRAGApp()
        runner = ChainRunner(merge_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        # Parallel streaming falls back to single chunk with merged content
        assert len(chunks) == 1
        merged = chunks[0]["content"]
        assert "Alpha:" in merged
        assert "Beta:" in merged
        assert "[Alpha:hello]" in merged
        assert "[Beta:hello]" in merged
        # Both assistants were invoked (parallel execution)
        assert len(rag.calls) == 2
        models = {c["model"] for c in rag.calls}
        assert models == {"Alpha", "Beta"}
        # Reasoning from both assistants is present
        reasoning = chunks[0].get("reasoning", "")
        assert "thinking about Alpha" in reasoning
        assert "thinking about Beta" in reasoning

    @pytest.mark.asyncio
    async def test_single_assistant_chain_tool_calls(self, assistant_flow):
        """Non-streaming run should aggregate tool_calls from the assistant node."""
        rag = FakeRAGApp(streaming_tool_calls={
            "Alpha": [{"tool_name": "web_search", "args": {"q": "hello"}, "ok": True}],
        })
        runner = ChainRunner(assistant_flow, rag)
        result = await runner.run("hello")
        assert result["answer"] == "[Alpha:hello]"
        assert len(result.get("tool_calls", [])) == 1
        assert result["tool_calls"][0]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_run_streaming_output_tool_calls(self, assistant_flow):
        """Streaming a single assistant should pass through tool_calls."""
        rag = FakeRAGApp(streaming_tool_calls={
            "Alpha": [{"tool_name": "web_search", "args": {"q": "hello"}, "ok": True}],
        })
        runner = ChainRunner(assistant_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "[Alpha:hello]"
        assert len(chunks[0].get("tool_calls", [])) == 1
        assert chunks[0]["tool_calls"][0]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_streaming_preparatory_tool_calls(self, preparatory_flow):
        """Preparatory node tool_calls should be emitted before the final stream."""
        rag = FakeRAGApp(streaming_tool_calls={
            "Alpha": [{"tool_name": "web_search", "args": {"q": "hello"}, "ok": True}],
            "Beta": [{"tool_name": "calculator", "args": {"expr": "1+1"}, "ok": True}],
        })
        runner = ChainRunner(preparatory_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        # Should have: preparatory tool_calls chunk + final content chunk(s)
        tool_call_chunks = [c for c in chunks if c.get("tool_calls")]
        assert len(tool_call_chunks) == 2
        # First chunk comes from preparatory node (Alpha)
        assert tool_call_chunks[0]["tool_calls"][0]["tool_name"] == "web_search"
        # Second chunk comes from the final stream (Beta)
        assert tool_call_chunks[1]["tool_calls"][0]["tool_name"] == "calculator"

    @pytest.mark.asyncio
    async def test_preparatory_subagent_events_are_forwarded(self, preparatory_flow):
        event = {"id": "sub-run-1", "stage": "started", "status": "running"}
        rag = FakeRAGApp(streaming_subagents={"Alpha": [event]})
        runner = ChainRunner(preparatory_flow, rag)

        chunks = [chunk async for chunk in runner.run_streaming("hello")]

        assert any(chunk.get("subagent", {}).get("id") == event["id"] for chunk in chunks)

    @pytest.mark.asyncio
    async def test_parallel_assistants_streaming_tool_calls(self, merge_flow):
        """Parallel assistants should aggregate tool_calls in the fallback chunks."""
        rag = FakeRAGApp(streaming_tool_calls={
            "Alpha": [{"tool_name": "web_search", "args": {"q": "hello"}, "ok": True}],
            "Beta": [{"tool_name": "calculator", "args": {"expr": "1+1"}, "ok": True}],
        })
        runner = ChainRunner(merge_flow, rag)
        chunks = []
        async for chunk in runner.run_streaming("hello"):
            chunks.append(chunk)
        # Both assistants are preparatory; tool_calls emitted first, then merged content
        assert len(chunks) == 2
        # First chunk is the preparatory tool_calls
        assert len(chunks[0].get("tool_calls", [])) == 2
        tool_names = {tc["tool_name"] for tc in chunks[0]["tool_calls"]}
        assert tool_names == {"web_search", "calculator"}
        # Second chunk is the merged content; tool_calls already emitted in prep chunk
        assert "Alpha:" in chunks[1]["content"]
        assert chunks[1].get("tool_calls") is None


from types import SimpleNamespace


class TestRouterNode:
    @pytest.fixture
    def router_flow(self):
        # input -> router --[code]--> a1 --> out
        #                --[chat]--> a2 --> out
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "r", "type": "router", "data": {
                    "nodeType": "router", "label": "Decide", "assistant_name": "Decider",
                    "routes": [
                        {"id": "1", "name": "code", "description": "coding questions"},
                        {"id": "2", "name": "chat", "description": "general chat"},
                    ],
                }},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "label": "Code", "assistant_name": "CodeBot"}},
                {"id": "a2", "type": "assistant", "data": {"nodeType": "assistant", "label": "Chat", "assistant_name": "ChatBot"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "r"},
                {"id": "e2", "source": "r", "sourceHandle": "route-1", "target": "a1"},
                {"id": "e3", "source": "r", "sourceHandle": "route-2", "target": "a2"},
                {"id": "e4", "source": "a1", "target": "out"},
                {"id": "e5", "source": "a2", "target": "out"},
            ],
        }

    @pytest.mark.asyncio
    async def test_router_routes_to_chosen_branch(self, router_flow):
        rag = FakeRAGApp(metadata_llms={"Decider": '{"route": "code"}'})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("write a function")
        assert "[CodeBot:write a function]" in result["answer"]
        models = {c.get("model") for c in rag.calls if c["method"] == "run_streaming"}
        assert models == {"CodeBot"}
        assert result["node_status"]["a2"]["status"] == "skipped"
        assert result["node_status"]["r"]["chosen_route"] == "code"
        assert result["node_status"]["out"]["status"] == "done"

    @pytest.mark.asyncio
    async def test_router_plain_name_reply(self, router_flow):
        rag = FakeRAGApp(metadata_llms={"Decider": "chat"})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("hi there")
        assert "[ChatBot:hi there]" in result["answer"]
        assert result["node_status"]["a1"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_router_substring_match(self, router_flow):
        rag = FakeRAGApp(metadata_llms={"Decider": "I would pick CODE for this."})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("hello")
        assert "[CodeBot:hello]" in result["answer"]

    @pytest.mark.asyncio
    async def test_router_fallback_route_on_garbage(self, router_flow):
        router_flow["nodes"][1]["data"]["fallback_route"] = "chat"
        rag = FakeRAGApp(metadata_llms={"Decider": "??? not a route ???"})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("hello")
        assert "[ChatBot:hello]" in result["answer"]
        assert result["node_status"]["r"]["chosen_route"] == "chat"

    @pytest.mark.asyncio
    async def test_router_no_fallback_uses_first_route(self, router_flow):
        rag = FakeRAGApp(metadata_llms={"Decider": "total garbage"})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("hello")
        assert "[CodeBot:hello]" in result["answer"]

    @pytest.mark.asyncio
    async def test_skip_propagates_through_downstream(self, router_flow):
        # extend: a2 -> a3 -> out ; dead route must skip a2 AND a3
        router_flow["nodes"].append(
            {"id": "a3", "type": "assistant", "data": {"nodeType": "assistant", "label": "A3", "assistant_name": "Deep"}}
        )
        router_flow["edges"][4] = {"id": "e5", "source": "a3", "target": "out"}
        router_flow["edges"].append({"id": "e6", "source": "a2", "target": "a3"})
        rag = FakeRAGApp(metadata_llms={"Decider": "code"})
        runner = ChainRunner(router_flow, rag)
        result = await runner.run("hello")
        assert result["node_status"]["a2"]["status"] == "skipped"
        assert result["node_status"]["a3"]["status"] == "skipped"
        assert "[CodeBot:hello]" in result["answer"]

    @pytest.mark.asyncio
    async def test_nested_routers(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "r1", "type": "router", "data": {
                    "nodeType": "router", "assistant_name": "D1",
                    "routes": [{"id": "1", "name": "x", "description": ""}, {"id": "2", "name": "y", "description": ""}],
                }},
                {"id": "r2", "type": "router", "data": {
                    "nodeType": "router", "assistant_name": "D2",
                    "routes": [{"id": "a", "name": "p", "description": ""}, {"id": "b", "name": "q", "description": ""}],
                }},
                {"id": "ax", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "AX"}},
                {"id": "by", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "BY"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "r1"},
                {"id": "e2", "source": "r1", "sourceHandle": "route-1", "target": "r2"},
                {"id": "e3", "source": "r1", "sourceHandle": "route-2", "target": "by"},
                {"id": "e4", "source": "r2", "sourceHandle": "route-a", "target": "ax"},
                {"id": "e5", "source": "r2", "sourceHandle": "route-b", "target": "by"},
                {"id": "e6", "source": "ax", "target": "out"},
                {"id": "e7", "source": "by", "target": "out"},
            ],
        }
        rag = FakeRAGApp(metadata_llms={"D1": "x", "D2": "q"})
        runner = ChainRunner(flow, rag)
        result = await runner.run("hello")
        # r1 chose x -> r2; r2 chose q -> by. ax skipped, but merges on 'by' (live).
        assert "[BY:" in result["answer"]
        assert result["node_status"]["ax"]["status"] == "skipped"
        assert result["node_status"]["r2"]["status"] == "done"


class TestConditionNode:
    @pytest.fixture
    def cond_flow(self):
        return {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "c", "type": "condition", "data": {
                    "nodeType": "condition", "rule": {"op": "contains", "value": "urgent"},
                }},
                {"id": "t1", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "True"}},
                {"id": "f1", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "False"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "c"},
                {"id": "e2", "source": "c", "sourceHandle": "cond-true", "target": "t1"},
                {"id": "e3", "source": "c", "sourceHandle": "cond-false", "target": "f1"},
                {"id": "e4", "source": "t1", "target": "out"},
                {"id": "e5", "source": "f1", "target": "out"},
            ],
        }

    @pytest.mark.asyncio
    async def test_condition_true_branch(self, cond_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(cond_flow, rag)
        result = await runner.run("this is URGENT please")
        assert "[True:this is URGENT please]" in result["answer"]
        assert result["node_status"]["f1"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_condition_false_branch(self, cond_flow):
        rag = FakeRAGApp()
        runner = ChainRunner(cond_flow, rag)
        result = await runner.run("nothing special")
        assert "[False:nothing special]" in result["answer"]
        assert result["node_status"]["t1"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_condition_regex(self, cond_flow):
        cond_flow["nodes"][1]["data"]["rule"] = {"op": "regex", "value": r"\d{4}"}
        rag = FakeRAGApp()
        runner = ChainRunner(cond_flow, rag)
        result = await runner.run("order 1234 done")
        assert "[True:" in result["answer"]


class TestExecutorAndLLMNodes:
    @pytest.mark.asyncio
    async def test_executor_assistant_target(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "ex", "type": "executor", "data": {
                    "nodeType": "executor", "label": "Run Alpha",
                    "target_kind": "assistant", "target": "Alpha",
                }},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "ex"},
                {"source": "ex", "target": "out"},
            ],
        }
        rag = FakeRAGApp()
        runner = ChainRunner(flow, rag)
        result = await runner.run("hello")
        assert result["answer"] == "[Alpha:hello]"
        assert rag.calls[0]["model"] == "Alpha"

    @pytest.mark.asyncio
    async def test_executor_subchain_target(self):
        sub_flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "sa", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "Sub"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "sa"},
                {"source": "sa", "target": "out"},
            ],
        }
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "ex", "type": "executor", "data": {
                    "nodeType": "executor", "target_kind": "chain", "target": "sub-chain",
                }},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "ex"},
                {"source": "ex", "target": "out"},
            ],
        }
        rag = FakeRAGApp()
        runner = ChainRunner(flow, rag)
        entry = SimpleNamespace(flow_definition=sub_flow, enabled=True, id="c1", name="sub-chain")
        runner._resolve_chain = lambda ref: asyncio_sleep0(entry)

        result = await runner.run("hello")
        assert result["answer"] == "[Sub:hello]"

    @pytest.mark.asyncio
    async def test_executor_subchain_not_found(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "ex", "type": "executor", "data": {
                    "nodeType": "executor", "target_kind": "chain", "target": "missing",
                }},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "ex"},
                {"source": "ex", "target": "out"},
            ],
        }
        rag = FakeRAGApp()  # no engine -> resolves to None
        runner = ChainRunner(flow, rag)
        result = await runner.run("hello")
        assert "not found" in result["answer"]

    @pytest.mark.asyncio
    async def test_executor_subchain_depth_cap(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "ex", "type": "executor", "data": {
                    "nodeType": "executor", "target_kind": "chain", "target": "self",
                }},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "ex"},
                {"source": "ex", "target": "out"},
            ],
        }
        rag = FakeRAGApp()
        runner = ChainRunner(flow, rag, depth=3)  # already at cap
        result = await runner.run("hello")
        assert "max sub-chain depth" in result["answer"]

    @pytest.mark.asyncio
    async def test_llm_node_renders_template(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "l", "type": "llm", "data": {
                    "nodeType": "llm", "assistant_name": "Writer",
                    "prompt_template": "Summarize in one word: {{input}}",
                }},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "l"},
                {"source": "l", "target": "out"},
            ],
        }
        rag = FakeRAGApp(metadata_llms={"Writer": "SUMMARY"})
        runner = ChainRunner(flow, rag)
        result = await runner.run("a long text")
        assert result["answer"] == "SUMMARY"
        llm = rag._metadata_llms["Writer"]
        prompt = llm.prompts[0]
        assert "a long text" in str(prompt)


class TestNoteAndStatus:
    @pytest.mark.asyncio
    async def test_note_node_not_executed(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "n1", "type": "note", "data": {"nodeType": "note", "label": "Docs"}},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "Alpha"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
            ],
        }
        rag = FakeRAGApp()
        runner = ChainRunner(flow, rag)
        result = await runner.run("hello")
        assert result["answer"] == "[Alpha:hello]"
        assert "n1" not in result["node_outputs"]
        assert "n1" not in result["node_status"]

    @pytest.mark.asyncio
    async def test_node_status_reported(self):
        flow = {
            "nodes": [
                {"id": "input", "type": "input", "data": {"nodeType": "input"}},
                {"id": "a1", "type": "assistant", "data": {"nodeType": "assistant", "assistant_name": "Alpha"}},
                {"id": "out", "type": "output", "data": {"nodeType": "output"}},
            ],
            "edges": [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
            ],
        }
        rag = FakeRAGApp()
        runner = ChainRunner(flow, rag)
        result = await runner.run("hello")
        statuses = result["node_status"]
        assert statuses["input"]["status"] == "done"
        assert statuses["a1"]["status"] == "done"
        assert statuses["out"]["status"] == "done"


async def asyncio_sleep0(value):
    return value


# ---------------------------------------------------------------------------
# V3 extensions: error edges, utility nodes, templates, race merge, tools
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Minimal tool-registry stand-in (get + execute)."""

    def __init__(self, tools: dict | None = None, results: dict | None = None) -> None:
        self._tools = tools or {}
        self._results = results or {}
        self.calls: list[dict] = []

    def get(self, name):
        return self._tools.get(name)

    async def execute(self, name, params):
        self.calls.append({"tool": name, "params": params})
        if name in self._results:
            return self._results[name]
        return {"output": f"tool:{name}:{params}"}


def _node(nid, ntype, **data):
    return {"id": nid, "type": "chainNode", "data": {"nodeType": ntype, "label": nid, **data}}


def _flow(nodes, edges):
    return {"nodes": nodes, "edges": edges}


class TestErrorEdges:
    @pytest.mark.asyncio
    async def test_error_edge_delivers_error_to_fallback(self):
        """a1 fails → normal edge dies, on_error edge delivers to fallback."""
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name=""),
                _node("fb", "assistant", assistant_name="Beta"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
                {"source": "a1", "target": "fb", "sourceHandle": "on_error"},
                {"source": "fb", "target": "out"},
            ],
        )
        rag = FakeRAGApp()
        result = await ChainRunner(flow, rag).run("hello")
        assert "Beta" in result["answer"]
        assert result["node_status"]["a1"]["status"] == "error"
        assert result["node_status"]["fb"]["status"] == "done"
        # fb received the error text as input
        assert "no assistant_name" in result["node_outputs"]["fb"]

    @pytest.mark.asyncio
    async def test_error_edge_dies_on_success(self):
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("fb", "assistant", assistant_name="Beta"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
                {"source": "a1", "target": "fb", "sourceHandle": "on_error"},
                {"source": "fb", "target": "out"},
            ],
        )
        rag = FakeRAGApp()
        result = await ChainRunner(flow, rag).run("hello")
        assert result["answer"] == "[Alpha:hello]"
        assert result["node_status"]["fb"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_failure_kills_normal_edges(self):
        """Failure without error edge → downstream skipped, leaf fallback."""
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name=""),
                _node("b1", "assistant", assistant_name="Beta"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "b1"},
                {"source": "b1", "target": "out"},
            ],
        )
        rag = FakeRAGApp()
        result = await ChainRunner(flow, rag).run("hello")
        # b1 never ran (no wasted call), error surfaced via leaf fallback
        assert result["node_status"]["b1"]["status"] == "skipped"
        assert result["node_status"]["out"]["status"] == "skipped"
        assert "no assistant_name" in result["answer"]


class TestSwitchNode:
    def _switch_flow(self, **case_overrides):
        cases = [
            {"id": "1", "name": "german", "op": "contains", "value": "hallo"},
            {"id": "2", "name": "code", "op": "starts_with", "value": "```"},
        ]
        return _flow(
            [
                _node("input", "input"),
                _node("sw", "switch", cases=cases, **case_overrides),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("a2", "assistant", assistant_name="Beta"),
                _node("ae", "assistant", assistant_name="Gamma"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "sw"},
                {"source": "sw", "target": "a1", "sourceHandle": "route-1"},
                {"source": "sw", "target": "a2", "sourceHandle": "route-2"},
                {"source": "sw", "target": "ae", "sourceHandle": "else"},
                {"source": "a1", "target": "out"},
                {"source": "a2", "target": "out"},
                {"source": "ae", "target": "out"},
            ],
        )

    @pytest.mark.asyncio
    async def test_first_matching_case_wins(self):
        result = await ChainRunner(self._switch_flow(), FakeRAGApp()).run("hallo welt")
        assert result["answer"] == "[Alpha:hallo welt]"
        assert result["node_status"]["a2"]["status"] == "skipped"
        assert result["node_status"]["ae"]["status"] == "skipped"
        assert result["node_status"]["sw"]["chosen_route"] == "german"

    @pytest.mark.asyncio
    async def test_else_port_on_no_match(self):
        result = await ChainRunner(self._switch_flow(), FakeRAGApp()).run("good day")
        assert result["answer"] == "[Gamma:good day]"
        assert result["node_status"]["sw"]["chosen_route"] == "else"

    @pytest.mark.asyncio
    async def test_regex_case(self):
        flow = self._switch_flow()
        flow["nodes"][1]["data"]["cases"] = [
            {"id": "1", "name": "digits", "op": "regex", "value": r"\d+"},
        ]
        result = await ChainRunner(flow, FakeRAGApp()).run("order 42 please")
        assert result["answer"] == "[Alpha:order 42 please]"


class TestDelayNode:
    @pytest.mark.asyncio
    async def test_delay_passthrough_and_cap(self):
        flow = _flow(
            [
                _node("input", "input"),
                _node("d", "delay", ms=999_999),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "d"},
                {"source": "d", "target": "out"},
            ],
        )
        # cap at 5ms so the test stays instant
        runner = ChainRunner(flow, FakeRAGApp(), execution_options={"max_delay_ms": 5})
        result = await runner.run("hello")
        assert result["answer"] == "hello"
        assert result["node_status"]["d"]["status"] == "done"


class TestJsonNode:
    def _json_flow(self, **data):
        base = {"assistant_name": "JsonLLM", "prompt_template": "make json of {{input}}"}
        base.update(data)
        return _flow(
            [
                _node("input", "input"),
                _node("j", "json", **base),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "j"},
                {"source": "j", "target": "out"},
            ],
        )

    @pytest.mark.asyncio
    async def test_valid_json_extracted_from_fences(self):
        rag = FakeRAGApp(metadata_llms={"JsonLLM": '```json\n{"a": 1}\n```'})
        result = await ChainRunner(self._json_flow(), rag).run("hi")
        assert result["node_status"]["j"]["status"] == "done"
        assert '{"a": 1}' in result["answer"]

    @pytest.mark.asyncio
    async def test_invalid_json_fails_then_error_status(self):
        rag = FakeRAGApp(metadata_llms={"JsonLLM": "not json at all"})
        result = await ChainRunner(self._json_flow(max_retries=1), rag).run("hi")
        assert result["node_status"]["j"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_schema_violation_triggers_retry(self):
        llm_replies = iter(['{"a": 1}', '{"b": 2}'])
        rag = FakeRAGApp()
        llm = _FakeLLM("")
        llm.ainvoke = lambda messages: _ainvoke_seq(llm, llm_replies, messages)
        rag._metadata_llms["JsonLLM"] = llm
        schema = {"type": "object", "required": ["b"]}
        result = await ChainRunner(
            self._json_flow(json_schema=schema, max_retries=1), rag
        ).run("hi")
        assert result["node_status"]["j"]["status"] == "done"
        assert '"b": 2' in result["answer"]


async def _ainvoke_seq(llm, replies, messages):
    llm.prompts.append(messages)
    return {"answer": next(replies)}


class TestToolNode:
    def _tool_flow(self, **data):
        base = {"tool_name": "rag_search"}
        base.update(data)
        return _flow(
            [
                _node("input", "input"),
                _node("t", "tool", **base),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "t"},
                {"source": "t", "target": "out"},
            ],
        )

    @pytest.mark.asyncio
    async def test_tool_executes_with_params(self):
        rag = FakeRAGApp()
        rag._tool_registry = _FakeRegistry(
            tools={"rag_search": object()},
            results={"rag_search": {"output": "found docs"}},
        )
        flow = self._tool_flow(params={"q": "{{input}}", "k": 3})
        result = await ChainRunner(flow, rag).run("search this")
        assert result["answer"] == "found docs"
        assert rag._tool_registry.calls[0]["params"] == {"q": "search this", "k": 3}

    @pytest.mark.asyncio
    async def test_unknown_tool_errors(self):
        rag = FakeRAGApp()
        rag._tool_registry = _FakeRegistry()
        result = await ChainRunner(self._tool_flow(), rag).run("hi")
        assert result["node_status"]["t"]["status"] == "error"
        assert "unknown tool" in result["node_outputs"]["t"]

    @pytest.mark.asyncio
    async def test_side_effect_tool_gated(self):
        rag = FakeRAGApp()
        rag._tool_registry = _FakeRegistry(tools={"email_sender": object()})
        flow = self._tool_flow(tool_name="email_sender")
        result = await ChainRunner(flow, rag).run("hi")
        assert result["node_status"]["t"]["status"] == "error"
        assert "side effects" in result["node_outputs"]["t"]
        assert not rag._tool_registry.calls

    @pytest.mark.asyncio
    async def test_side_effect_tool_allowed_by_optin(self):
        rag = FakeRAGApp()
        rag._tool_registry = _FakeRegistry(tools={"email_sender": object()})
        flow = self._tool_flow(tool_name="email_sender", allow_side_effects=True)
        result = await ChainRunner(flow, rag).run("hi")
        assert result["node_status"]["t"]["status"] == "done"


class TestTemplatesAndRaceMerge:
    @pytest.mark.asyncio
    async def test_node_ref_in_template(self):
        """{{node.a1}} resolves to a1's content."""
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("t1", "transform", template="summary of {{node.a1}}"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "t1"},
                {"source": "t1", "target": "out"},
            ],
        )
        result = await ChainRunner(flow, FakeRAGApp()).run("hi")
        assert result["answer"] == "summary of [Alpha:hi]"

    @pytest.mark.asyncio
    async def test_node_route_ref(self):
        """{{node.r1.route}} exposes the router's chosen route name."""
        flow = _flow(
            [
                _node("input", "input"),
                _node(
                    "r1", "router",
                    assistant_name="Decider",
                    routes=[
                        {"id": "1", "name": "path_a"},
                        {"id": "2", "name": "path_b"},
                    ],
                ),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("t1", "transform", template="route was {{node.r1.route}}"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "r1"},
                {"source": "r1", "target": "a1", "sourceHandle": "route-1"},
                {"source": "r1", "target": "t1", "sourceHandle": "route-2"},
                {"source": "a1", "target": "out"},
                {"source": "t1", "target": "out"},
            ],
        )
        rag = FakeRAGApp(metadata_llms={"Decider": '{"route": "path_b"}'})
        result = await ChainRunner(flow, rag).run("x")
        assert result["answer"] == "route was path_b"

    @pytest.mark.asyncio
    async def test_input_template_overrides_merge(self):
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("a2", "assistant", assistant_name="Beta"),
                _node("e", "executor", target_kind="assistant", target="Gamma",
                      input_template="combine {{node.a1}} and {{node.a2}}"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "input", "target": "a2"},
                {"source": "a1", "target": "e"},
                {"source": "a2", "target": "e"},
                {"source": "e", "target": "out"},
            ],
        )
        rag = FakeRAGApp()
        result = await ChainRunner(flow, rag).run("hi")
        gamma = [c for c in rag.calls if c["model"] == "Gamma"][0]
        assert gamma["input"] == "combine [Alpha:hi] and [Beta:hi]"
        assert result["answer"] == "[Gamma:combine [Alpha:hi] and [Beta:hi]]"

    @pytest.mark.asyncio
    async def test_merge_mode_first_picks_winner(self):
        """Race merge: first delivered predecessor wins."""
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("m", "merge", merge_mode="first"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "input", "target": "m"},
                {"source": "a1", "target": "m"},
                {"source": "m", "target": "out"},
            ],
        )
        result = await ChainRunner(flow, FakeRAGApp()).run("hi")
        # input executes first in topo order → it wins the race
        assert result["answer"] == "hi"
        assert result["node_status"]["m"]["winning_pred"] == "input"


class TestLiveEvents:
    @pytest.mark.asyncio
    async def test_on_event_emits_lifecycle(self):
        flow = _flow(
            [
                _node("input", "input"),
                _node("a1", "assistant", assistant_name="Alpha"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a1"},
                {"source": "a1", "target": "out"},
            ],
        )
        events: list[dict] = []

        async def on_event(ev):
            events.append(ev)

        runner = ChainRunner(flow, FakeRAGApp(), on_event=on_event)
        await runner.run("hi")
        types = [e["type"] for e in events]
        assert types[0] == "run_start"
        assert "node_start" in types and "node_end" in types
        assert types[-1] == "run_end"
        starts = [e["node_id"] for e in events if e["type"] == "node_start"]
        assert set(starts) == {"input", "a1", "out"}
        ends = {e["node_id"]: e for e in events if e["type"] == "node_end"}
        assert ends["a1"]["status"] == "done"
        assert ends["a1"]["duration_ms"] >= 0



class TestRaceMergeOrder:
    @pytest.mark.asyncio
    async def test_merge_first_uses_completion_order(self):
        """The earlier-*finishing* normal pred wins — not topo order.

        'a_delay' sorts before 'z_asst' topologically, but the delay finishes
        later → the instant assistant must win the race.
        """
        flow = _flow(
            [
                _node("input", "input"),
                _node("a_delay", "delay", ms=80),
                _node("z_asst", "assistant", assistant_name="Alpha"),
                _node("m", "merge", merge_mode="first"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "a_delay"},
                {"source": "input", "target": "z_asst"},
                {"source": "a_delay", "target": "m"},
                {"source": "z_asst", "target": "m"},
                {"source": "m", "target": "out"},
            ],
        )
        result = await ChainRunner(flow, FakeRAGApp()).run("hi")
        assert result["node_status"]["m"]["winning_pred"] == "z_asst"
        assert result["answer"] == "[Alpha:hi]"

    @pytest.mark.asyncio
    async def test_merge_first_error_pred_never_wins(self):
        """An on_error contingency delivered early must not win the race."""
        flow = _flow(
            [
                _node("input", "input"),
                _node("bad", "assistant", assistant_name=""),  # fails instantly
                _node("good", "delay", ms=60),
                _node("m", "merge", merge_mode="first"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "bad"},
                {"source": "input", "target": "good"},
                {"source": "bad", "target": "m", "sourceHandle": "on_error"},
                {"source": "good", "target": "m"},
                {"source": "m", "target": "out"},
            ],
        )
        result = await ChainRunner(flow, FakeRAGApp()).run("hi")
        assert result["node_status"]["m"]["winning_pred"] == "good"
        assert result["answer"] == "hi"


class TestToolResultPropagation:
    @pytest.mark.asyncio
    async def test_tool_error_result_sets_error_status(self):
        """Registry dict {ok: False, error: …} must mark the node as error."""
        rag = FakeRAGApp()
        rag._tool_registry = _FakeRegistry(
            tools={"rag_search": object()},
            results={"rag_search": {"ok": False, "error": "quota exceeded"}},
        )
        flow = _flow(
            [
                _node("input", "input"),
                _node("t", "tool", tool_name="rag_search"),
                _node("out", "output"),
            ],
            [
                {"source": "input", "target": "t"},
                {"source": "t", "target": "out"},
            ],
        )
        result = await ChainRunner(flow, rag).run("hi")
        assert result["node_status"]["t"]["status"] == "error"
        assert "quota exceeded" in result["node_outputs"]["t"]
