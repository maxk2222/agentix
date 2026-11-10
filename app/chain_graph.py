"""Shared graph helpers for agent-chain flows.

Used by ``chain_runner`` (execution) and ``agent_chains_validation`` /
``routers/agent_chains`` (validation). Centralizing the traversal logic keeps
skip-propagation and reference checks consistent everywhere.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set


def edge_uid(edge: Dict[str, Any]) -> str:
    """Stable identifier for an edge (id preferred, else derived key)."""
    eid = edge.get("id")
    if eid:
        return str(eid)
    return f"{edge.get('source')}->{edge.get('target')}:{edge.get('sourceHandle') or ''}"


def node_id_of(node: Dict[str, Any]) -> Optional[str]:
    nid = node.get("id")
    return str(nid) if nid else None


def node_type_of(node: Dict[str, Any]) -> str:
    return node.get("data", {}).get("nodeType", "") or node.get("type", "") or ""


def predecessors(node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
    return [e["source"] for e in edges if e.get("target") == node_id]


def successors(node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
    return [e["target"] for e in edges if e.get("source") == node_id]


def incoming_edges(node_id: str, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in edges if e.get("target") == node_id]


def topological_sort(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    """Node IDs in topological order (Kahn). Raises ValueError on cycles."""
    node_ids = {node_id_of(n) for n in nodes}
    node_ids.discard(None)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src in node_ids and tgt in node_ids:
            adjacency[src].append(tgt)
            in_degree[tgt] += 1
    queue = deque(sorted(nid for nid in node_ids if in_degree[nid] == 0))
    result: List[str] = []
    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != len(node_ids):
        raise ValueError("Cycle detected in flow graph")
    return result


def propagate_dead(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    dead_edges: Set[str],
) -> Set[str]:
    """Fixpoint propagation of dead routes into skipped nodes.

    A node is *skipped* when every incoming edge is dead or originates from a
    skipped node — i.e. no live path can reach it. The input node and nodes
    without any incoming edges are never skipped (a stray node still receives
    the initial input, matching legacy behaviour). Called after each routing
    decision (router/condition) until a fixpoint is reached.
    """
    executable_ids = {
        node_id_of(n)
        for n in nodes
        if node_id_of(n) and node_type_of(n) != "note"
    }
    skipped: Set[str] = set()
    incoming: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in executable_ids}
    for e in edges:
        tgt = e.get("target")
        if tgt in executable_ids:
            incoming[tgt].append(e)

    changed = True
    while changed:
        changed = False
        for nid in executable_ids - skipped:
            preds = incoming.get(nid) or []
            if not preds:
                continue  # rootless node stays alive (legacy semantics)
            if node_type_of(_node_by_id(nodes, nid)) == "input":
                continue
            if all(edge_uid(e) in dead_edges or e.get("source") in skipped for e in preds):
                skipped.add(nid)
                changed = True
    return skipped


def _node_by_id(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if node_id_of(n) == node_id:
            return n
    return None


def live_predecessors(
    node_id: str,
    edges: List[Dict[str, Any]],
    dead_edges: Set[str],
    skipped: Set[str],
) -> List[str]:
    """Predecessor ids whose connecting edge is alive and who aren't skipped."""
    out: List[str] = []
    for e in edges:
        if e.get("target") != node_id:
            continue
        if edge_uid(e) in dead_edges:
            continue
        src = e.get("source")
        if src and src not in skipped:
            out.append(src)
    return out


def collect_chain_refs(flow_definition: Dict[str, Any]) -> Set[str]:
    """All chain ids/names referenced by ``executor`` nodes (target_kind=chain)."""
    refs: Set[str] = set()
    for node in (flow_definition or {}).get("nodes", []) or []:
        data = node.get("data", {}) or {}
        if node_type_of(node) != "executor":
            continue
        if data.get("target_kind") == "chain" and data.get("target"):
            refs.add(str(data["target"]))
    return refs


async def check_subchain_references(
    flow_definition: Dict[str, Any],
    *,
    self_id: Optional[str],
    self_name: Optional[str],
    resolve: Callable[[str], Awaitable[Optional[Any]]],
    max_depth: int = 5,
) -> List[str]:
    """Validate executor ``chain`` references transitively.

    ``resolve(ref)`` must return a chain entry (attributes/keys ``id``, ``name``,
    ``enabled``, ``flow_definition``) or ``None``. Reports missing/disabled
    references and any cycle that loops back into an ancestor (including a
    direct or indirect self-reference of the chain being saved).
    """
    errors: List[str] = []
    roots = {v for v in (self_id, self_name) if v}

    async def _expand(flow_def: Dict[str, Any], path: List[str], depth: int) -> None:
        if depth > max_depth:
            errors.append("Sub-chain reference depth exceeded (possible cycle)")
            return
        for ref in sorted(collect_chain_refs(flow_def)):
            entry = await resolve(ref)
            if entry is None:
                errors.append(f"Referenced chain '{ref}' not found")
                continue
            eid = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
            ename = getattr(entry, "name", None) or (entry.get("name") if isinstance(entry, dict) else None)
            eflow = getattr(entry, "flow_definition", None) or (
                entry.get("flow_definition") if isinstance(entry, dict) else {}
            )
            enabled = getattr(entry, "enabled", True) if not isinstance(entry, dict) else entry.get("enabled", True)
            if not enabled:
                errors.append(f"Referenced chain '{ref}' is disabled")
            if (roots and (eid in roots or ename in roots)) or eid in path or ename in path:
                errors.append(f"Chain reference cycle detected at '{ref}'")
                continue
            await _expand(eflow or {}, [*path, eid or ename or ref], depth + 1)

    await _expand(flow_definition or {}, [], 0)
    return errors
