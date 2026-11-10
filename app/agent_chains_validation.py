"""Validation logic for agent chain flow definitions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

_VALID_RULE_OPS = {"contains", "not_contains", "equals", "starts_with", "ends_with", "regex"}
ERROR_HANDLE = "on_error"
ELSE_HANDLE = "else"
_ERROR_PORT_TYPES = {
    "executor", "router", "condition", "switch", "llm", "transform",
    "tool", "delay", "json", "merge", "parallel", "assistant", "agent",
}
_TEMPLATE_INPUT_TYPES = _ERROR_PORT_TYPES - {"merge", "parallel"}
_NODE_REF_RE = re.compile(r"\{\{\s*node\.([A-Za-z0-9_.:-]+)\s*\}\}")


def validate_flow(flow_definition: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a flow definition and return {valid: bool, errors: List[str]}."""
    errors: List[str] = []

    if not isinstance(flow_definition, dict):
        errors.append("Flow definition must be a JSON object")
        return {"valid": False, "errors": errors}

    nodes = flow_definition.get("nodes")
    edges = flow_definition.get("edges")

    if not isinstance(nodes, list):
        errors.append("'nodes' must be a list")
        return {"valid": False, "errors": errors}
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        return {"valid": False, "errors": errors}

    if not nodes:
        errors.append("Flow must contain at least one node")

    if not edges:
        errors.append("Flow must contain at least one edge")

    # Validate node structure
    node_ids: Set[str] = set()
    input_nodes: List[str] = []
    output_nodes: List[str] = []
    assistant_nodes: List[str] = []

    for idx, node in enumerate(nodes):
        node_id = node.get("id")
        node_type = node.get("data", {}).get("nodeType") or node.get("type", "")
        if not node_id:
            errors.append(f"Node at index {idx} is missing 'id'")
            continue
        if node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)

        if not node_type:
            errors.append(f"Node {node_id} is missing 'type'")
            continue

        valid_types = {
            "input", "output",
            # legacy types — still supported
            "assistant", "agent", "transform", "merge", "parallel",
            # V2 types
            "executor", "router", "condition", "llm", "note",
            # V3 types
            "switch", "delay", "json", "tool",
        }
        if node_type not in valid_types:
            errors.append(f"Node {node_id} has invalid type '{node_type}'")
            continue

        data = node.get("data", {}) or {}
        if node_type == "input":
            input_nodes.append(node_id)
        elif node_type == "output":
            output_nodes.append(node_id)
        elif node_type in ("assistant", "agent"):
            assistant_nodes.append(node_id)
            # Validate assistant reference
            assistant_name = data.get("assistant_name")
            if not assistant_name:
                errors.append(f"Assistant node {node_id} is missing 'assistant_name'")
        elif node_type == "executor":
            kind = data.get("target_kind") or "assistant"
            if kind not in ("assistant", "agent", "chain"):
                errors.append(f"Executor node {node_id} has invalid target_kind '{kind}'")
            if not (data.get("target") or data.get("assistant_name")):
                errors.append(f"Executor node {node_id} is missing 'target'")
        elif node_type == "router":
            if not data.get("assistant_name"):
                errors.append(f"Router node {node_id} is missing 'assistant_name' (LLM)")
            routes = data.get("routes") or []
            if not routes:
                errors.append(f"Router node {node_id} needs at least one route")
            route_ids: List[str] = []
            route_names: List[str] = []
            for r in routes:
                rid = r.get("id")
                if rid is None or str(rid) == "":
                    errors.append(f"Router node {node_id} has a route without an 'id'")
                else:
                    route_ids.append(str(rid))
                if not r.get("name") or not str(r.get("name")).strip():
                    errors.append(f"Router node {node_id} has a route without a name")
                else:
                    route_names.append(str(r.get("name")).strip())
            if len(route_ids) != len(set(route_ids)):
                errors.append(f"Router node {node_id} has duplicate route ids")
            if len(route_names) != len(set(route_names)):
                errors.append(f"Router node {node_id} has duplicate route names")
        elif node_type == "condition":
            rule = data.get("rule") or {}
            if not rule.get("op") or "value" not in rule:
                errors.append(f"Condition node {node_id} is missing rule.op/rule.value")
        elif node_type == "llm":
            if not data.get("assistant_name"):
                errors.append(f"LLM node {node_id} is missing 'assistant_name' (model)")
            if not data.get("prompt_template"):
                errors.append(f"LLM node {node_id} is missing 'prompt_template'")
        elif node_type == "switch":
            cases = data.get("cases") or []
            if not cases:
                errors.append(f"Switch node {node_id} needs at least one case")
            case_ids: List[str] = []
            for c in cases:
                if not isinstance(c, dict):
                    errors.append(f"Switch node {node_id} has a malformed case")
                    continue
                if c.get("id") is None or str(c.get("id")) == "":
                    errors.append(f"Switch node {node_id} has a case without an 'id'")
                else:
                    case_ids.append(str(c.get("id")))
                if c.get("op") and str(c.get("op")) not in _VALID_RULE_OPS:
                    errors.append(f"Switch node {node_id} case '{c.get('id')}' has invalid op '{c.get('op')}'")
            if len(case_ids) != len(set(case_ids)):
                errors.append(f"Switch node {node_id} has duplicate case ids")
        elif node_type == "delay":
            ms = data.get("ms")
            if ms is None or not isinstance(ms, (int, float)) or isinstance(ms, bool) or ms < 0:
                errors.append(f"Delay node {node_id} needs a non-negative 'ms' value")
        elif node_type == "json":
            if not data.get("assistant_name"):
                errors.append(f"JSON node {node_id} is missing 'assistant_name' (model)")
            schema = data.get("json_schema")
            if isinstance(schema, str) and schema.strip():
                try:
                    json.loads(schema)
                except Exception:
                    errors.append(f"JSON node {node_id} has invalid 'json_schema' (not parseable)")
            elif schema is not None and not isinstance(schema, dict):
                errors.append(f"JSON node {node_id} has invalid 'json_schema' (must be object)")
        elif node_type == "tool":
            if not data.get("tool_name"):
                errors.append(f"Tool node {node_id} is missing 'tool_name'")
            params = data.get("params")
            if isinstance(params, str) and params.strip():
                try:
                    json.loads(params)
                except Exception:
                    errors.append(f"Tool node {node_id} has invalid 'params' (not parseable JSON)")
            elif params is not None and not isinstance(params, dict):
                errors.append(f"Tool node {node_id} 'params' must be an object")
        # transform/merge/parallel have no required fields; a missing
        # transform 'template' degrades to pass-through (see ChainRunner).

    # Validate edge structure
    edge_pairs: List[tuple[str, str]] = []
    for idx, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        if not source:
            errors.append(f"Edge at index {idx} is missing 'source'")
            continue
        if not target:
            errors.append(f"Edge at index {idx} is missing 'target'")
            continue
        if source not in node_ids:
            errors.append(f"Edge references unknown source node: {source}")
            continue
        if target not in node_ids:
            errors.append(f"Edge references unknown target node: {target}")
            continue
        if source == target:
            errors.append(f"Self-loop detected on node {source}")
            continue
        edge_pairs.append((source, target))

    # Check connectivity constraints
    if len(input_nodes) == 0:
        errors.append("Flow must contain exactly one 'input' node (found 0)")
    elif len(input_nodes) > 1:
        errors.append(f"Flow must contain exactly one 'input' node (found {len(input_nodes)})")

    if len(output_nodes) == 0:
        errors.append("Flow must contain at least one 'output' node")

    # Check that input has outgoing edges and output has incoming edges
    if input_nodes:
        input_outgoing = [e for e in edges if e.get("source") in input_nodes]
        if not input_outgoing:
            errors.append("Input node must have at least one outgoing edge")
    if output_nodes:
        output_incoming = [e for e in edges if e.get("target") in output_nodes]
        if not output_incoming:
            errors.append("Output node must have at least one incoming edge")

    # Per-type edge rules: router edges must leave through a declared route
    # port, condition edges through cond-true/cond-false, note nodes have no
    # handles at all.
    node_index = {n.get("id"): n for n in nodes if n.get("id")}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        src_type = (node_index.get(src) or {}).get("data", {}).get("nodeType") or (node_index.get(src) or {}).get("type", "")
        tgt_type = (node_index.get(tgt) or {}).get("data", {}).get("nodeType") or (node_index.get(tgt) or {}).get("type", "")
        if src_type == "note" or tgt_type == "note":
            errors.append("Note nodes cannot be connected by edges")
            continue
        handle = edge.get("sourceHandle") or ""
        if src_type == "router":
            routes = (node_index.get(src) or {}).get("data", {}).get("routes") or []
            route_ids = {f"route-{r.get('id')}" for r in routes}
            if handle != ERROR_HANDLE and handle not in route_ids:
                errors.append(
                    f"Router edge {src}→{tgt} needs sourceHandle matching a route port ({sorted(route_ids)})"
                )
        elif src_type == "switch":
            cases = (node_index.get(src) or {}).get("data", {}).get("cases") or []
            case_ids = {f"route-{c.get('id')}" for c in cases if isinstance(c, dict)}
            if handle not in case_ids and handle not in (ERROR_HANDLE, ELSE_HANDLE):
                errors.append(
                    f"Switch edge {src}→{tgt} needs sourceHandle matching a case port or '{ELSE_HANDLE}'"
                )
        elif src_type == "condition":
            if handle not in ("cond-true", "cond-false", ERROR_HANDLE):
                errors.append(f"Condition edge {src}→{tgt} needs sourceHandle 'cond-true' or 'cond-false'")
        elif handle == ERROR_HANDLE and src_type not in _ERROR_PORT_TYPES:
            errors.append(f"on_error edge not allowed from node type '{src_type}' ({src})")

        if (edge.get("sourceHandle") or "") == ERROR_HANDLE and tgt_type in ("input", "output"):
            errors.append(f"on_error edge cannot target an '{tgt_type}' node ({src}→{tgt})")

    # Check for cycles (DFS)
    if node_ids and edge_pairs:
        adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for src, tgt in edge_pairs:
            adjacency[src].append(tgt)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if _has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for nid in node_ids:
            if nid not in visited:
                if _has_cycle(nid):
                    errors.append("Flow contains a cycle (cycles are not allowed in V1)")
                    break

    # Check that all nodes are reachable from input
    if input_nodes and edge_pairs:
        reachable: Set[str] = set()
        queue = [input_nodes[0]]
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in adjacency.get(current, []):
                if neighbor not in reachable:
                    queue.append(neighbor)

        # 'note' nodes are annotation-only (no edges) — exempt from reachability
        note_ids = {
            n.get("id")
            for n in nodes
            if ((n.get("data", {}) or {}).get("nodeType") or n.get("type", "")) == "note"
        }
        unreachable = node_ids - reachable - note_ids
        if unreachable:
            errors.append(f"Nodes unreachable from input: {', '.join(sorted(unreachable))}")

    # ---- template reference checks -----------------------------------------
    # {{node.<id>}} / {{node.<id>.route}} may only reference strict transitive
    # ancestors — anything else would render empty at runtime.
    ancestors: Dict[str, Set[str]] = {}

    def _ancestors(nid: str) -> Set[str]:
        if nid in ancestors:
            return ancestors[nid]
        seen: Set[str] = set()
        stack = [src for src, tgt in edge_pairs if tgt == nid]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(s for s, t in edge_pairs if t == cur)
        ancestors[nid] = seen
        return seen

    def _check_template_refs(node_id: str, field: str, value: Any) -> None:
        for m in _NODE_REF_RE.finditer(str(value)):
            ref = m.group(1)
            if ref.endswith(".route"):
                ref = ref[: -len(".route")]
            if ref == node_id:
                errors.append(f"Node {node_id}: '{field}' references itself")
            elif ref not in node_ids:
                errors.append(f"Node {node_id}: '{field}' references unknown node '{ref}'")
            elif ref not in _ancestors(node_id):
                errors.append(
                    f"Node {node_id}: '{field}' references '{ref}' which is not an "
                    "upstream ancestor (its output would always be empty)"
                )

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        data = node.get("data") or {}
        node_type = data.get("nodeType") or node.get("type", "")
        if data.get("input_template") and node_type not in _TEMPLATE_INPUT_TYPES:
            errors.append(
                f"Node {node_id}: 'input_template' is not supported on type '{node_type}'"
            )
        for field in ("input_template", "prompt_template", "template"):
            if data.get(field):
                _check_template_refs(node_id, field, data[field])
        params = data.get("params")
        if isinstance(params, dict):
            leaves = json.dumps(params, ensure_ascii=False)
            _check_template_refs(node_id, "params", leaves)
        elif isinstance(params, str) and params.strip():
            _check_template_refs(node_id, "params", params)

    return {"valid": len(errors) == 0, "errors": errors}
