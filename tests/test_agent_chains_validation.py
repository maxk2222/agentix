"""Unit tests for agent chain flow validation (app/agent_chains_validation.py)."""

from __future__ import annotations

from app.agent_chains_validation import validate_flow


def _node(node_id, node_type, **data):
    return {"id": node_id, "type": "chainNode", "data": {"nodeType": node_type, **data}}


def _edge(source, target):
    return {"source": source, "target": target}


def _flow(nodes, edges):
    return {"nodes": nodes, "edges": edges}


class TestValidTypes:
    def test_transform_node_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("t1", "transform", template="Say: {{input}}"),
                _node("out", "output"),
            ],
            [_edge("in", "t1"), _edge("t1", "out")],
        )
        assert validate_flow(flow) == {"valid": True, "errors": []}

    def test_transform_without_template_valid(self):
        flow = _flow(
            [_node("in", "input"), _node("t1", "transform"), _node("out", "output")],
            [_edge("in", "t1"), _edge("t1", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is True

    def test_merge_node_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("a1", "assistant", assistant_name="A"),
                _node("a2", "assistant", assistant_name="B"),
                _node("m", "merge"),
                _node("out", "output"),
            ],
            [_edge("in", "a1"), _edge("in", "a2"), _edge("a1", "m"), _edge("a2", "m"), _edge("m", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_parallel_node_valid(self):
        flow = _flow(
            [_node("in", "input"), _node("p", "parallel"), _node("out", "output")],
            [_edge("in", "p"), _edge("p", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_agent_node_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("g1", "agent", assistant_name="AgentX"),
                _node("out", "output"),
            ],
            [_edge("in", "g1"), _edge("g1", "out")],
        )
        assert validate_flow(flow)["valid"] is True


class TestErrors:
    def test_unknown_type_rejected(self):
        flow = _flow(
            [_node("in", "input"), _node("x", "loop"), _node("out", "output")],
            [_edge("in", "x"), _edge("x", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("invalid type" in e for e in result["errors"])

    def test_assistant_missing_name_rejected(self):
        flow = _flow(
            [_node("in", "input"), _node("a1", "assistant"), _node("out", "output")],
            [_edge("in", "a1"), _edge("a1", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("assistant_name" in e for e in result["errors"])

    def test_transform_does_not_require_assistant_name(self):
        flow = _flow(
            [_node("in", "input"), _node("t1", "transform"), _node("out", "output")],
            [_edge("in", "t1"), _edge("t1", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_two_inputs_rejected(self):
        flow = _flow(
            [_node("in1", "input"), _node("in2", "input"), _node("out", "output")],
            [_edge("in1", "out"), _edge("in2", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("exactly one 'input' node" in e for e in result["errors"])

    def test_cycle_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("a", "transform"),
                _node("b", "transform"),
                _node("out", "output"),
            ],
            [_edge("in", "a"), _edge("a", "b"), _edge("b", "a"), _edge("b", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("cycle" in e.lower() for e in result["errors"])

    def test_unreachable_node_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("a", "transform"),
                _node("orphan", "transform"),
                _node("out", "output"),
            ],
            [_edge("in", "a"), _edge("a", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("unreachable" in e for e in result["errors"])


class TestV2Types:
    def test_executor_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("ex", "executor", target_kind="assistant", target="Alpha"),
                _node("out", "output"),
            ],
            [_edge("in", "ex"), _edge("ex", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_executor_missing_target_rejected(self):
        flow = _flow(
            [_node("in", "input"), _node("ex", "executor", target_kind="assistant"), _node("out", "output")],
            [_edge("in", "ex"), _edge("ex", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("target" in e for e in result["errors"])

    def test_executor_invalid_kind_rejected(self):
        flow = _flow(
            [_node("in", "input"), _node("ex", "executor", target_kind="webhook", target="x"), _node("out", "output")],
            [_edge("in", "ex"), _edge("ex", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("target_kind" in e for e in result["errors"])

    def test_router_valid_with_route_handles(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", assistant_name="D", routes=[
                    {"id": "1", "name": "a", "description": "route a"},
                    {"id": "2", "name": "b", "description": "route b"},
                ]),
                _node("a1", "assistant", assistant_name="A"),
                _node("a2", "assistant", assistant_name="B"),
                _node("out", "output"),
            ],
            [
                _edge("in", "r"),
                {"source": "r", "target": "a1", "sourceHandle": "route-1"},
                {"source": "r", "target": "a2", "sourceHandle": "route-2"},
                _edge("a1", "out"),
                _edge("a2", "out"),
            ],
        )
        assert validate_flow(flow)["valid"] is True

    def test_router_edge_without_handle_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", assistant_name="D", routes=[{"id": "1", "name": "a"}]),
                _node("a1", "assistant", assistant_name="A"),
                _node("out", "output"),
            ],
            [_edge("in", "r"), _edge("r", "a1"), _edge("a1", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("sourceHandle" in e for e in result["errors"])

    def test_router_missing_llm_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", routes=[{"id": "1", "name": "a"}]),
                _node("out", "output"),
            ],
            [_edge("in", "r"), {"source": "r", "target": "out", "sourceHandle": "route-1"}],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("assistant_name" in e for e in result["errors"])

    def test_condition_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("c", "condition", rule={"op": "contains", "value": "x"}),
                _node("a1", "assistant", assistant_name="A"),
                _node("a2", "assistant", assistant_name="B"),
                _node("out", "output"),
            ],
            [
                _edge("in", "c"),
                {"source": "c", "target": "a1", "sourceHandle": "cond-true"},
                {"source": "c", "target": "a2", "sourceHandle": "cond-false"},
                _edge("a1", "out"),
                _edge("a2", "out"),
            ],
        )
        assert validate_flow(flow)["valid"] is True

    def test_condition_missing_rule_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("c", "condition"),
                _node("out", "output"),
            ],
            [_edge("in", "c"), {"source": "c", "target": "out", "sourceHandle": "cond-true"}],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("rule" in e for e in result["errors"])

    def test_llm_node_valid(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("l", "llm", assistant_name="M", prompt_template="Say {{input}}"),
                _node("out", "output"),
            ],
            [_edge("in", "l"), _edge("l", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_note_node_needs_no_edges(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("note", "note", label="documentation"),
                _node("a1", "assistant", assistant_name="A"),
                _node("out", "output"),
            ],
            [_edge("in", "a1"), _edge("a1", "out")],
        )
        assert validate_flow(flow)["valid"] is True

    def test_note_node_edges_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("note", "note"),
                _node("out", "output"),
            ],
            [_edge("in", "note"), _edge("note", "out")],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("Note" in e for e in result["errors"])

    def test_router_duplicate_route_ids_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", assistant_name="D", routes=[
                    {"id": "1", "name": "a"},
                    {"id": "1", "name": "b"},
                ]),
                _node("a1", "assistant", assistant_name="A"),
                _node("out", "output"),
            ],
            [
                _edge("in", "r"),
                {"source": "r", "target": "a1", "sourceHandle": "route-1"},
                _edge("a1", "out"),
            ],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("duplicate route ids" in e for e in result["errors"])

    def test_router_duplicate_route_names_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", assistant_name="D", routes=[
                    {"id": "1", "name": "same"},
                    {"id": "2", "name": "same"},
                ]),
                _node("a1", "assistant", assistant_name="A"),
                _node("out", "output"),
            ],
            [
                _edge("in", "r"),
                {"source": "r", "target": "a1", "sourceHandle": "route-1"},
                _edge("a1", "out"),
            ],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("duplicate route names" in e for e in result["errors"])

    def test_router_route_without_id_rejected(self):
        flow = _flow(
            [
                _node("in", "input"),
                _node("r", "router", assistant_name="D", routes=[
                    {"name": "a"},
                    {"id": "2", "name": "b"},
                ]),
                _node("a1", "assistant", assistant_name="A"),
                _node("out", "output"),
            ],
            [
                _edge("in", "r"),
                {"source": "r", "target": "a1", "sourceHandle": "route-2"},
                _edge("a1", "out"),
            ],
        )
        result = validate_flow(flow)
        assert result["valid"] is False
        assert any("route without an 'id'" in e for e in result["errors"])
