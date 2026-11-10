"""ChainRunner: executes agent-chain DAGs with sequential/parallel support.

Execution model (V2):

* Fan-out and fan-in are purely topological — more than one outgoing edge
  runs successors in parallel, more than one live incoming edge merges the
  predecessor outputs into a labelled block.
* ``router`` and ``condition`` nodes decide which outgoing route is live;
  edges leaving through any other port are *dead* and ``propagate_dead``
  marks downstream nodes as skipped until merges converge on live branches.
* ``executor`` nodes run an assistant/agent directly or a nested sub-chain
  (depth-capped), ``llm`` performs a bare non-streaming LLM call, ``note``
  nodes are never executed. Legacy types (``assistant``, ``agent``,
  ``merge``, ``parallel``, ``transform``) keep their V1 behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.chain_graph import (
    edge_uid,
    live_predecessors,
    node_id_of,
    node_type_of,
    propagate_dead,
    topological_sort as _topological_sort_base,
)
from app.chat_utils import extract_answer_text, extract_reasoning_text

logger = logging.getLogger(__name__)

DEFAULT_MAX_SUBCHAIN_DEPTH = 3
DEFAULT_MAX_DELAY_MS = 30_000
DEFAULT_TOOL_TIMEOUT_S = 60.0
ERROR_HANDLE = "on_error"
ELSE_HANDLE = "else"

# Node types that may carry an ``on_error`` output port.
ERROR_PORT_TYPES = {
    "executor", "router", "condition", "switch", "llm", "transform",
    "tool", "delay", "json", "merge", "parallel", "assistant", "agent",
}

# Tools with real-world side effects — require an explicit opt-in flag.
SIDE_EFFECT_TOOLS = {
    "email_sender", "webhook_post", "ssh_exec",
    "spawn_subagent", "nudge_subagent", "wait", "wait_subagent",
}

# Node types where ``data.input_template`` may override the merged input.
TEMPLATE_INPUT_TYPES = {
    "executor", "router", "condition", "switch", "llm", "transform",
    "tool", "delay", "json", "assistant", "agent",
}

_TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.:-]+)\s*\}\}")


def _topological_sort(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    return _topological_sort_base(nodes, edges)


def _get_node_by_id(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if node_id_of(n) == node_id:
            return n
    return None


def _get_predecessors(node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
    return [e["source"] for e in edges if e.get("target") == node_id]


def _get_successors(node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
    return [e["target"] for e in edges if e.get("source") == node_id]


def _parse_router_choice(text: str, routes: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the chosen route name from an LLM reply (tolerant)."""
    names = [str(r.get("name", "")).strip() for r in routes if r.get("name")]
    if not text or not names:
        return None
    # 1) JSON object containing "route"
    for match in re.findall(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(match)
        except Exception:
            continue
        route = obj.get("route") or obj.get("choice") or obj.get("selected")
        if isinstance(route, str) and route.strip() in names:
            return route.strip()
    # 2) exact name token in the reply
    stripped = text.strip().strip('"`\' .')
    if stripped in names:
        return stripped
    # 3) case-insensitive substring match on names
    lowered = text.lower()
    hits = [n for n in names if n.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    return None


def _eval_rule(op: str, value: str, text: str) -> bool:
    """Deterministic predicate shared by ``condition`` and ``switch`` nodes."""
    op = op or "contains"
    try:
        if op == "contains":
            return value.lower() in text.lower()
        if op == "not_contains":
            return value.lower() not in text.lower()
        if op == "equals":
            return text.strip().lower() == value.lower()
        if op == "starts_with":
            return text.strip().lower().startswith(value.lower())
        if op == "ends_with":
            return text.strip().lower().endswith(value.lower())
        if op == "regex":
            return bool(re.search(value, text, re.IGNORECASE | re.DOTALL))
    except re.error:
        return False
    return False


def _extract_json(text: str) -> Any:
    """Pull the first JSON value out of an LLM reply (fence-tolerant)."""
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
    raise ValueError("no JSON value found in LLM reply")


def _validate_schema(obj: Any, schema: Dict[str, Any]) -> Optional[str]:
    """Minimal JSON-schema check (type/required/properties-type/items-type).

    Returns an error string or ``None`` when valid — deliberately keeps the
    common cases instead of pulling a full jsonschema dependency.
    """
    if not isinstance(schema, dict):
        return "schema must be an object"
    type_map = {
        "object": dict, "array": list, "string": str,
        "boolean": bool, "null": type(None),
    }
    t = schema.get("type")
    if t:
        if t == "number":
            if not isinstance(obj, (int, float)) or isinstance(obj, bool):
                return "expected type 'number'"
        elif t == "integer":
            if not isinstance(obj, int) or isinstance(obj, bool):
                return "expected type 'integer'"
        elif t in type_map and not isinstance(obj, type_map[t]):
            return f"expected type '{t}'"
    if isinstance(obj, dict):
        for key in schema.get("required") or []:
            if key not in obj:
                return f"missing required key '{key}'"
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                if key in obj and isinstance(sub, dict) and sub.get("type"):
                    err = _validate_schema(obj[key], {"type": sub["type"]})
                    if err:
                        return f"property '{key}': {err}"
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        sub = schema["items"]
        if sub.get("type"):
            for i, item in enumerate(obj):
                err = _validate_schema(item, {"type": sub["type"]})
                if err:
                    return f"item {i}: {err}"
    return None


def resolve_template_text(
    template: str,
    *,
    node_input: str,
    initial_input: str,
    node_outputs: Dict[str, Dict[str, Any]],
    node_status: Dict[str, Dict[str, Any]],
) -> str:
    """Resolve ``{{input}}``, ``{{user}}``, ``{{node.<id>}}`` and
    ``{{node.<id>.route}}`` placeholders against runner state.

    Unknown placeholders are left untouched so literal ``{{...}}`` text in
    prompts survives.
    """
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        if key == "input":
            return node_input or ""
        if key == "user":
            return initial_input or ""
        if key.startswith("node."):
            rest = key[5:]
            if rest.endswith(".route"):
                nid = rest[: -len(".route")]
                return str((node_status.get(nid) or {}).get("chosen_route") or "")
            out = node_outputs.get(rest)
            return str(out.get("content", "")) if out else ""
        return match.group(0)

    return _TEMPLATE_RE.sub(_repl, template or "")


class ChainRunner:
    """Executes a configured agent chain flow definition."""

    def __init__(
        self,
        flow_definition: Dict[str, Any],
        rag_application: Any,
        *,
        depth: int = 0,
        execution_options: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._flow = flow_definition
        self._nodes = flow_definition.get("nodes", [])
        self._edges = flow_definition.get("edges", [])
        self._rag = rag_application
        self._depth = depth
        self._exec_opts = execution_options or {}
        self._on_event = on_event
        self._run_ctx = run_context or {}
        self._node_outputs: Dict[str, Dict[str, Any]] = {}
        self._node_status: Dict[str, Dict[str, Any]] = {}
        self._dead_edges: Set[str] = set()
        self._skipped: Set[str] = set()
        self._execution_order: List[str] = []
        self._subagent_event_queue: Optional[asyncio.Queue[Dict[str, Any]]] = None
        self._chain_store: Any = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_node_label(self, node_id: str) -> str:
        """Return a human-readable label for a node (assistant_name or label or id)."""
        node = _get_node_by_id(self._nodes, node_id)
        if node is None:
            return node_id
        data = node.get("data", {})
        return data.get("assistant_name", "") or data.get("label", "") or node_id

    def _set_status(self, node_id: str, status: str, **extra: Any) -> None:
        self._node_status[node_id] = {"status": status, **extra}

    def _live_preds(self, node_id: str) -> List[str]:
        return live_predecessors(node_id, self._edges, self._dead_edges, self._skipped)

    def _refresh_skipped(self) -> None:
        self._skipped |= propagate_dead(self._nodes, self._edges, self._dead_edges)

    async def _resolve_chain(self, ref: str) -> Optional[Any]:
        """Lazy sub-chain lookup by id or name via AgentChainsStore."""
        engine = getattr(self._rag, "engine", None)
        if engine is None:
            return None
        if self._chain_store is None:
            from app.agent_chains_store import AgentChainsStore

            self._chain_store = AgentChainsStore(engine)
            await self._chain_store.ensure_tables()
        return await self._chain_store.get_chain(ref) or await self._chain_store.get_chain_by_name(ref)

    async def _emit(self, type_: str, **data: Any) -> None:
        """Fire a lifecycle event (node_start/node_end) when a listener is set."""
        cb = self._on_event
        if cb is None:
            return
        try:
            await cb({"type": type_, **data})
        except Exception:
            logger.debug("chain on_event callback failed", exc_info=True)

    def _resolve_template(self, template: str, node_input: str, initial_input: str) -> str:
        return resolve_template_text(
            template,
            node_input=node_input,
            initial_input=initial_input,
            node_outputs=self._node_outputs,
            node_status=self._node_status,
        )

    def _resolve_params(self, value: Any, node_input: str, initial_input: str) -> Any:
        """Recursively resolve template placeholders inside tool param leaves."""
        if isinstance(value, str):
            return self._resolve_template(value, node_input, initial_input)
        if isinstance(value, dict):
            return {k: self._resolve_params(v, node_input, initial_input) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_params(v, node_input, initial_input) for v in value]
        return value

    def _resolve_out_edges(self, node_id: str, output: Dict[str, Any]) -> None:
        """Decide which outgoing edges survive, based on the node's outcome.

        Success: ``on_error`` edges die plus any router/condition/switch route
        edges that were not chosen. Failure: every *normal* out-edge dies so
        error text never flows downstream — only ``on_error`` edges deliver.
        """
        ok = bool(output.get("ok", True))
        chosen = output.get("chosen_handle") if ok else None
        for e in self._edges:
            if e.get("source") != node_id:
                continue
            handle = e.get("sourceHandle")
            if ok:
                if handle == ERROR_HANDLE or (chosen and handle != chosen):
                    self._dead_edges.add(edge_uid(e))
            elif handle != ERROR_HANDLE:
                self._dead_edges.add(edge_uid(e))
        self._refresh_skipped()

    # ------------------------------------------------------------------
    # node executors
    # ------------------------------------------------------------------

    async def _execute_assistant_node(
        self,
        node: Dict[str, Any],
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single assistant/agent node via RAGApplication.

        Returns a dict with ``content`` (answer text) and ``reasoning``.
        """
        data = node.get("data", {})
        assistant_name = data.get("assistant_name", "") or (
            data.get("target", "") if data.get("target_kind") != "chain" else ""
        )
        if not assistant_name:
            logger.error("Assistant node %s has no assistant_name", node.get("id"))
            return {
                "content": f"[Error: Assistant node {node.get('id')} has no assistant_name]",
                "reasoning": "",
                "tool_calls": [],
            }

        try:
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_parts: list[Dict[str, Any]] = []
            agent_agen = self._rag.run_streaming(
                user_input=input_text,
                chat_history=chat_history,
                model_name=assistant_name,
                conversation_id=conversation_id,
                chat_user_id=chat_user_id,
            )
            try:
                async for chunk in agent_agen:
                    content_parts.append(chunk.get("content", ""))
                    reasoning_parts.append(chunk.get("reasoning", ""))
                    subagent_event = chunk.get("subagent")
                    if subagent_event and self._subagent_event_queue is not None:
                        await self._subagent_event_queue.put(subagent_event)
                    tc = chunk.get("tool_calls")
                    if tc:
                        tool_calls_parts.extend(tc)
            finally:
                await agent_agen.aclose()
            answer = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            return {"content": answer, "reasoning": reasoning, "tool_calls": tool_calls_parts}
        except Exception as exc:
            logger.exception("Assistant node %s failed", node.get("id"))
            return {
                "content": f"[Error in {assistant_name}: {exc}]",
                "reasoning": "",
                "tool_calls": [],
            }

    async def _execute_subchain_node(
        self,
        node: Dict[str, Any],
        target: str,
        input_text: str,
        chat_history: Optional[List[BaseMessage]],
        conversation_id: Optional[str],
        chat_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Run a referenced chain as a nested ChainRunner (depth-capped)."""
        max_depth = int(self._exec_opts.get("max_subchain_depth") or DEFAULT_MAX_SUBCHAIN_DEPTH)
        if self._depth >= max_depth:
            return {
                "content": f"[Error: max sub-chain depth ({max_depth}) exceeded]",
                "reasoning": "",
                "tool_calls": [],
            }
        try:
            entry = await self._resolve_chain(target)
            if entry is None:
                return {"content": f"[Error: chain '{target}' not found]", "reasoning": "", "tool_calls": []}
            if not getattr(entry, "enabled", True):
                return {"content": f"[Error: chain '{target}' is disabled]", "reasoning": "", "tool_calls": []}
            nested = ChainRunner(
                entry.flow_definition,
                self._rag,
                depth=self._depth + 1,
                execution_options=self._exec_opts,
                run_context={
                    "chain_id": getattr(entry, "id", None) or target,
                    "chain_name": getattr(entry, "name", "") or target,
                    "source": "subchain",
                    "conversation_id": conversation_id,
                },
            )
            result = await nested.run(
                input_text=input_text,
                chat_history=chat_history,
                conversation_id=conversation_id,
                chat_user_id=chat_user_id,
            )
            reasoning = result.get("reasoning", "")
            label = self._get_node_label(node_id_of(node) or "")
            return {
                "content": result.get("answer", ""),
                "reasoning": f"[{label}]:\n{reasoning}" if reasoning else "",
                "tool_calls": result.get("tool_calls") or [],
            }
        except Exception as exc:
            logger.exception("Sub-chain node %s failed", node.get("id"))
            return {"content": f"[Error in sub-chain '{target}': {exc}]", "reasoning": "", "tool_calls": []}

    async def _execute_router_node(self, node: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """Ask an LLM which of the configured routes to take."""
        data = node.get("data", {})
        assistant_name = data.get("assistant_name", "")
        routes = data.get("routes") or []
        if not assistant_name or not routes:
            return {
                "content": f"[Error: router node {node.get('id')} needs assistant_name and routes]",
                "reasoning": "",
                "tool_calls": [],
            }
        route_lines = "\n".join(
            f"{i + 1}. {r.get('name')}: {r.get('description', '')}" for i, r in enumerate(routes)
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a routing decider. Given a numbered list of routes "
                    "and an input, choose exactly one route that best matches the "
                    "input. Respond ONLY with a JSON object {\"route\": \"<name>\"}."
                )
            ),
            HumanMessage(content=f"Routes:\n{route_lines}\n\nInput:\n{input_text}"),
        ]
        try:
            llm = await self._rag.get_metadata_llm(model_name=assistant_name)
            resp = await llm.ainvoke(messages)
            reply = extract_answer_text(resp)
        except Exception as exc:
            logger.exception("Router node %s decision failed", node.get("id"))
            reply = ""
            llm_error = str(exc)
        else:
            llm_error = ""

        chosen = _parse_router_choice(reply, routes)
        if chosen is None:
            fallback = data.get("fallback_route")
            valid = {str(r.get("name")) for r in routes}
            chosen = fallback if fallback in valid else str(routes[0].get("name"))
            note = "fallback" if llm_error or not reply else "no match"
        else:
            note = "llm"
        chosen_id = next(
            (str(r.get("id")) for r in routes if str(r.get("name")) == chosen),
            str(routes[0].get("id", "1")),
        )
        reasoning = f"[router:{assistant_name}] chose '{chosen}' ({note})"
        if reply:
            reasoning += f" — raw: {reply[:200]}"
        if llm_error:
            reasoning += f" — llm error: {llm_error[:200]}"
        return {
            "content": input_text,
            "reasoning": reasoning,
            "tool_calls": [],
            "chosen_route": chosen,
            "chosen_handle": f"route-{chosen_id}",
        }

    async def _execute_condition_node(self, node: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """Evaluate a deterministic rule against the input → true/false port."""
        data = node.get("data", {})
        rule = data.get("rule") or {}
        op = rule.get("op", "contains")
        value = str(rule.get("value", ""))
        text = input_text or ""
        known_ops = {"contains", "not_contains", "equals", "starts_with", "ends_with", "regex"}
        if op not in known_ops:
            result = False
            op = f"unknown:{op}"
        elif op == "regex":
            try:
                result = bool(re.search(value, text, re.IGNORECASE | re.DOTALL))
            except re.error as exc:
                result = False
                op = f"invalid-regex:{exc}"
        else:
            result = _eval_rule(op, value, text)
        route = "true" if result else "false"
        return {
            "content": text,
            "reasoning": f"[condition] {op}({value!r}) → {route}",
            "tool_calls": [],
            "chosen_route": route,
            "chosen_handle": f"cond-{route}",
        }

    async def _execute_llm_node(self, node: Dict[str, Any], input_text: str, initial_input: str) -> Dict[str, Any]:
        """Bare non-streaming LLM call with a user-defined prompt template."""
        data = node.get("data", {})
        assistant_name = data.get("assistant_name", "")
        template = data.get("prompt_template") or "{{input}}"
        if not assistant_name:
            return {
                "content": f"[Error: llm node {node.get('id')} needs assistant_name]",
                "reasoning": "",
                "tool_calls": [],
            }
        prompt = self._resolve_template(template, input_text, initial_input)
        try:
            llm = await self._rag.get_metadata_llm(model_name=assistant_name)
            resp = await llm.ainvoke(prompt)
            return {
                "content": extract_answer_text(resp),
                "reasoning": extract_reasoning_text(resp),
                "tool_calls": [],
            }
        except Exception as exc:
            logger.exception("LLM node %s failed", node.get("id"))
            return {"content": f"[Error in llm node: {exc}]", "reasoning": "", "tool_calls": []}

    def _execute_switch_node(self, node: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """Evaluate ordered cases against the input; first match wins, else
        the ``else`` port. Deterministic (no LLM) — the router's cheap twin."""
        data = node.get("data", {})
        cases = data.get("cases") or []
        for case in cases:
            if not isinstance(case, dict):
                continue
            op = str(case.get("op") or "contains")
            value = str(case.get("value") or "")
            if op != "regex" and not value:
                continue
            if _eval_rule(op, value, input_text):
                name = str(case.get("name") or case.get("id") or "")
                return {
                    "content": input_text,
                    "reasoning": f"switch matched case '{name}' ({op} {value!r})",
                    "tool_calls": [],
                    "chosen_route": name or str(case.get("id")),
                    "chosen_handle": f"route-{case.get('id')}",
                }
        return {
            "content": input_text,
            "reasoning": "switch matched no case → else",
            "tool_calls": [],
            "chosen_route": "else",
            "chosen_handle": ELSE_HANDLE,
        }

    async def _execute_json_node(self, node: Dict[str, Any], input_text: str, initial_input: str) -> Dict[str, Any]:
        """LLM call whose reply must be valid JSON, optionally schema-checked.

        Retries with an appended repair hint until ``max_retries`` is reached.
        """
        node_id = node.get("id", "?")
        data = node.get("data", {})
        assistant_name = data.get("assistant_name", "")
        if not assistant_name:
            return {"content": f"[Error: json node {node_id} needs assistant_name]", "reasoning": "", "tool_calls": []}

        try:
            schema = data.get("json_schema")
            if isinstance(schema, str):
                schema = json.loads(schema) if schema.strip() else None
        except Exception:
            return {"content": f"[Error: json node {node_id} has invalid json_schema]", "reasoning": "", "tool_calls": []}

        max_retries = max(0, int(data.get("max_retries") or 0))
        base_prompt = self._resolve_template(
            data.get("prompt_template") or "{{input}}", input_text, initial_input
        )
        repair_hint = (
            "\n\nYour previous reply was not valid JSON matching the required "
            "schema. Respond with ONLY the JSON value — no markdown, no prose."
        )
        errors: List[str] = []
        try:
            llm = await self._rag.get_metadata_llm(model_name=assistant_name)
        except Exception as exc:
            return {"content": f"[Error in json node: {exc}]", "reasoning": "", "tool_calls": []}

        for attempt in range(max_retries + 1):
            prompt = base_prompt + (repair_hint * min(attempt, 1))
            try:
                resp = await llm.ainvoke(prompt)
                raw = extract_answer_text(resp)
                parsed = _extract_json(raw)
            except Exception as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                continue
            schema_err = _validate_schema(parsed, schema) if schema else None
            if schema_err is None:
                reasoning_text = extract_reasoning_text(resp)
                reasoning_parts = [reasoning_text] if reasoning_text else []
                if errors:
                    reasoning_parts.append(f"recovered after {len(errors)} failed attempt(s)")
                return {
                    "content": json.dumps(parsed, ensure_ascii=False),
                    "reasoning": "\n".join(reasoning_parts),
                    "tool_calls": [],
                    "json": parsed,
                }
            errors.append(f"attempt {attempt + 1}: schema mismatch ({schema_err})")

        return {
            "content": f"[Error in json node: {'; '.join(errors)}]",
            "reasoning": "",
            "tool_calls": [],
        }

    def _tool_registry(self) -> Any:
        return getattr(self._rag, "_tool_registry", None)

    async def _execute_tool_node(
        self,
        node: Dict[str, Any],
        input_text: str,
        initial_input: str,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a registry tool with template-resolved params.

        Side-effect tools require ``execution_options.allow_side_effects``
        (or per-node ``data.allow_side_effects``); timeout comes from
        ``data.timeout_s`` capped by ``execution_options.tool_timeout_s``.
        """
        node_id = node.get("id", "?")
        data = node.get("data", {})
        tool_name = str(data.get("tool_name") or "")
        if not tool_name:
            return {"content": f"[Error: tool node {node_id} needs tool_name]", "reasoning": "", "tool_calls": []}

        registry = self._tool_registry()
        if registry is None:
            return {"content": "[Error: tool registry unavailable]", "reasoning": "", "tool_calls": []}

        if tool_name in SIDE_EFFECT_TOOLS and not (
            data.get("allow_side_effects") or self._exec_opts.get("allow_side_effects")
        ):
            return {
                "content": f"[Error: tool '{tool_name}' has side effects — enable allow_side_effects]",
                "reasoning": "",
                "tool_calls": [],
            }

        tool = None
        if hasattr(registry, "get_tool"):
            tool = registry.get_tool(tool_name)
        elif hasattr(registry, "get"):
            tool = registry.get(tool_name)
        if tool is None:
            return {"content": f"[Error: unknown tool '{tool_name}']", "reasoning": "", "tool_calls": []}

        raw_params = data.get("params")
        if isinstance(raw_params, str):
            try:
                raw_params = json.loads(raw_params) if raw_params.strip() else {}
            except Exception:
                return {"content": f"[Error: tool node {node_id} has invalid params JSON]", "reasoning": "", "tool_calls": []}
        params = self._resolve_params(raw_params if isinstance(raw_params, dict) else {}, input_text, initial_input)
        if not params and data.get("pass_input") is not False:
            params = {"input": input_text}

        timeout_s = float(data.get("timeout_s") or self._exec_opts.get("tool_timeout_s") or DEFAULT_TOOL_TIMEOUT_S)
        try:
            if hasattr(registry, "execute"):
                import inspect

                exec_sig = inspect.signature(registry.execute)
                call_kwargs: Dict[str, Any] = {}
                if "context" in exec_sig.parameters:
                    call_kwargs["context"] = {
                        "source": "chain",
                        "conversation_id": conversation_id,
                        "chat_user_id": chat_user_id,
                    }
                result = await asyncio.wait_for(
                    registry.execute(tool_name, params, **call_kwargs),
                    timeout=timeout_s,
                )
            elif hasattr(tool, "ainvoke"):
                result = await asyncio.wait_for(tool.ainvoke(params), timeout=timeout_s)
            elif hasattr(tool, "invoke"):
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, params), timeout=timeout_s
                )
            else:
                return {"content": f"[Error: tool '{tool_name}' is not callable]", "reasoning": "", "tool_calls": []}
        except asyncio.TimeoutError:
            return {"content": f"[Error: tool '{tool_name}' timed out after {timeout_s:.0f}s]", "reasoning": "", "tool_calls": []}
        except Exception as exc:
            logger.exception("Tool node %s (%s) failed", node_id, tool_name)
            return {"content": f"[Error in tool '{tool_name}': {exc}]", "reasoning": "", "tool_calls": []}

        if isinstance(result, dict):
            ok = bool(result.get("ok", True))
            err = result.get("error")
            content = (
                result.get("output") or result.get("result") or result.get("content")
                or ("" if not ok else json.dumps(result, ensure_ascii=False, default=str))
            )
            if not ok:
                return {
                    "content": f"[Error in tool '{tool_name}': {err or 'execution failed'}]",
                    "reasoning": "",
                    "tool_calls": [{"tool": tool_name, "params": params}],
                    "ok": False,
                }
            return {
                "content": str(content),
                "reasoning": str(result.get("reasoning") or ""),
                "tool_calls": [{"tool": tool_name, "params": params}],
            }
        return {"content": str(result), "reasoning": "", "tool_calls": [{"tool": tool_name, "params": params}]}

    def _pred_content(self, nid: str) -> str:
        out = self._node_outputs.get(nid)
        return str(out["content"]) if out else ""

    def _compute_node_input(
        self,
        node: Dict[str, Any],
        node_type: str,
        preds: List[str],
        initial_input: str,
    ) -> tuple[str, Optional[str]]:
        """Determine the node's input text; returns (text, winning_pred|None).

        Default rules: input node → initial input; single pred → its content;
        multiple preds → labeled merge (or first-delivered content when
        ``data.merge_mode == 'first'``). ``data.input_template`` overrides the
        merged text on every executable type.
        """
        data = node.get("data", {})
        winner: Optional[str] = None
        if node_type == "input":
            node_input = initial_input
        elif len(preds) == 1:
            node_input = self._pred_content(preds[0])
        elif len(preds) > 1 and data.get("merge_mode") == "first":
            # Race semantics: the earliest *completed* normal-edge predecessor
            # wins — _node_outputs insertion order tracks completion order.
            # on_error predecessors are contingencies and never win the race.
            normal_preds = {
                e.get("source")
                for e in self._edges
                if e.get("target") == node.get("id")
                and e.get("sourceHandle") != ERROR_HANDLE
                and edge_uid(e) not in self._dead_edges
                and e.get("source") not in self._skipped
            }
            for nid in self._node_outputs:
                if nid in normal_preds:
                    winner = nid
                    break
            node_input = self._pred_content(winner) if winner else ""
        elif len(preds) > 1:
            parts = [f"user message:\n{initial_input}"]
            for p in preds:
                text = self._pred_content(p)
                if text:
                    parts.append(f"{self._get_node_label(p)}:\n{text}")
            node_input = "\n\n".join(parts)
        else:
            node_input = initial_input

        if node_type in TEMPLATE_INPUT_TYPES:
            template = str(data.get("input_template") or "")
            if template:
                node_input = self._resolve_template(template, node_input, initial_input)
        return node_input, winner

    async def _execute_node(
        self,
        node_id: str,
        initial_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single node and return its output dict (content + reasoning)."""
        node = _get_node_by_id(self._nodes, node_id)
        if node is None:
            return {"content": "", "reasoning": "", "tool_calls": [], "ok": True}

        node_type = node_type_of(node)
        data = node.get("data", {})
        preds = self._live_preds(node_id) or _get_predecessors(node_id, self._edges)

        await self._emit("node_start", node_id=node_id, node_type=node_type)
        t0 = time.monotonic()
        node_input, winner = self._compute_node_input(node, node_type, preds, initial_input)

        # Execute based on type
        if node_type in ("assistant", "agent"):
            output = await self._execute_assistant_node(node, node_input, chat_history, conversation_id, chat_user_id)
        elif node_type == "executor":
            kind = data.get("target_kind") or "assistant"
            target = str(data.get("target") or data.get("assistant_name") or "")
            if kind == "chain":
                output = await self._execute_subchain_node(node, target, node_input, chat_history, conversation_id, chat_user_id)
            else:
                output = await self._execute_assistant_node(node, node_input, chat_history, conversation_id, chat_user_id)
        elif node_type == "router":
            output = await self._execute_router_node(node, node_input)
        elif node_type == "condition":
            output = await self._execute_condition_node(node, node_input)
        elif node_type == "switch":
            output = self._execute_switch_node(node, node_input)
        elif node_type == "llm":
            output = await self._execute_llm_node(node, node_input, initial_input)
        elif node_type == "json":
            output = await self._execute_json_node(node, node_input, initial_input)
        elif node_type == "tool":
            output = await self._execute_tool_node(node, node_input, initial_input, conversation_id, chat_user_id)
        elif node_type == "delay":
            ms = data.get("ms")
            delay_ms = int(ms) if isinstance(ms, (int, float)) else 0
            delay_ms = max(0, min(delay_ms, int(self._exec_opts.get("max_delay_ms", DEFAULT_MAX_DELAY_MS))))
            await asyncio.sleep(delay_ms / 1000.0)
            output = {"content": node_input, "reasoning": "", "tool_calls": []}
        elif node_type == "transform":
            transform_template = str(data.get("template") or "")
            out_text = (
                self._resolve_template(transform_template, node_input, initial_input)
                if transform_template else node_input
            )
            output = {"content": out_text, "reasoning": "", "tool_calls": []}
        elif node_type == "merge" and data.get("merge_mode") == "first":
            # Race merge — node_input is already the winner's content.
            win_out = self._node_outputs.get(winner) if winner else None
            output = {
                "content": node_input,
                "reasoning": (win_out or {}).get("reasoning", ""),
                "tool_calls": (win_out or {}).get("tool_calls") or [],
            }
        elif node_type == "merge":
            labeled_parts = []
            for p in preds:
                text = self._pred_content(p)
                if text:
                    labeled_parts.append(f"[{self._get_node_label(p)}]:\n{text}")
            reasoning_parts = []
            tool_calls_parts: List[Dict[str, Any]] = []
            for p in preds:
                out = self._node_outputs.get(p)
                if out and out.get("reasoning"):
                    reasoning_parts.append(f"[{self._get_node_label(p)}]:\n{out['reasoning']}")
                if out and out.get("tool_calls"):
                    tool_calls_parts.extend(out["tool_calls"])
            output = {
                "content": "\n\n".join(labeled_parts),
                "reasoning": "\n\n".join(reasoning_parts),
                "tool_calls": tool_calls_parts,
            }
        elif node_type in ("parallel", "output", "input", "note"):
            output = {"content": node_input, "reasoning": "", "tool_calls": []}
        else:
            output = {"content": node_input, "reasoning": "", "tool_calls": []}

        # Outcome bookkeeping: explicit ok flag, legacy [Error prefix as fallback
        ok = output.get("ok")
        if ok is None:
            ok = not str(output.get("content", "")).startswith("[Error")
        output["ok"] = bool(ok)
        if not ok and "error" not in output:
            output["error"] = str(output.get("content", ""))[:400]

        self._node_outputs[node_id] = output
        status = "done" if ok else "error"
        extra: Dict[str, Any] = {}
        if output.get("chosen_route"):
            extra["chosen_route"] = output["chosen_route"]
        if not ok:
            extra["error"] = output.get("error")
        if winner:
            extra["winning_pred"] = winner
        self._set_status(node_id, status, **extra)
        self._resolve_out_edges(node_id, output)
        await self._emit(
            "node_end",
            node_id=node_id,
            status=status,
            chosen_route=output.get("chosen_route"),
            error=output.get("error") if not ok else None,
            duration_ms=int((time.monotonic() - t0) * 1000),
            preview=str(output.get("content", ""))[:1000],
        )
        return output

    def _node_type(self, node_id: str) -> str:
        node = _get_node_by_id(self._nodes, node_id)
        return node_type_of(node) if node else ""

    def _leaf_content_fallback(self) -> str:
        """Aggregate outputs of non-skipped leaf nodes (used when the output
        node itself ended up skipped or is missing). A 'leaf' here means a
        node with no *live* outgoing edge — dead-edge endpoints (e.g. behind
        a failed node or an unchosen route) surface through the last live
        leaf instead of silently dropping their content."""
        live_out = {
            e.get("source")
            for e in self._edges
            if edge_uid(e) not in self._dead_edges
        }
        leaf_nodes = [
            nid
            for nid in self._execution_order
            if nid not in live_out and nid not in self._skipped and self._node_type(nid) != "note"
        ]
        if not leaf_nodes and self._execution_order:
            leaf_nodes = [self._execution_order[-1]]
        if len(leaf_nodes) == 1:
            out = self._node_outputs.get(leaf_nodes[0])
            return out.get("content", "") if out else ""
        leaf_contents = []
        for nid in leaf_nodes:
            out = self._node_outputs.get(nid)
            if out and out.get("content"):
                label = self._get_node_label(nid)
                leaf_contents.append(f"{label}:\n{out['content']}")
        return "\n\n".join(leaf_contents)

    async def _run_levelwise(
        self,
        node_ids: List[str],
        input_text: str,
        chat_history: Optional[List[BaseMessage]],
        conversation_id: Optional[str],
        chat_user_id: Optional[str],
    ) -> None:
        """Level-wise execution with dead-route propagation.

        Ready nodes run in parallel via gather; after each level the dead-edge
        set is re-evaluated so branches behind unchosen routes are skipped.
        """
        pending = set(node_ids)
        executed: Set[str] = set()
        while pending:
            self._refresh_skipped()
            skipped_now = pending & self._skipped
            for nid in skipped_now:
                pending.discard(nid)
                self._set_status(nid, "skipped")
                await self._emit("node_end", node_id=nid, status="skipped")
            if not pending:
                break
            ready = [
                nid
                for nid in self._execution_order
                if nid in pending and self._node_ready(nid, pending, executed)
            ]
            if not ready:
                break  # no live path can resolve remaining nodes
            await asyncio.gather(
                *(
                    self._execute_node(nid, input_text, chat_history, conversation_id, chat_user_id)
                    for nid in ready
                )
            )
            for nid in ready:
                executed.add(nid)
                pending.discard(nid)
        # Anything still pending has no live path — mark skipped explicitly
        for nid in pending:
            self._set_status(nid, "skipped")
            self._skipped.add(nid)
            await self._emit("node_end", node_id=nid, status="skipped")

    def _node_ready(self, nid: str, pending: Set[str], executed: Set[str]) -> bool:
        """Whether all live inbound edges have resolved their sources.

        Normal merge: every live predecessor must have delivered (or skipped).
        ``merge_mode='first'`` (race): a single delivered normal-edge
        predecessor suffices — but armed ``on_error`` edges must still be
        resolved first, since error edges are contingencies, never raced.
        """
        node = _get_node_by_id(self._nodes, nid)
        data = (node or {}).get("data", {})
        merge_first = data.get("merge_mode") == "first"

        resolved = lambda s: (
            s in executed or s in self._skipped or s in self._node_outputs
        )
        delivered = lambda s: s in self._node_outputs
        live_in = [
            (e.get("source"), e.get("sourceHandle"))
            for e in self._edges
            if e.get("target") == nid
            and edge_uid(e) not in self._dead_edges
            and e.get("source") not in self._skipped
        ]
        if merge_first:
            armed_err = [s for s, h in live_in if h == ERROR_HANDLE]
            normal = [s for s, h in live_in if h != ERROR_HANDLE]
            return all(resolved(s) for s in armed_err) and any(delivered(s) for s in normal)
        return all(resolved(s) for s, _ in live_in)

    def _reset_run_state(self) -> None:
        self._node_outputs = {}
        self._node_status = {}
        self._dead_edges = set()
        self._skipped = set()
        self._execution_order = _topological_sort(self._nodes, self._edges)

    async def _record_run(
        self,
        input_text: str,
        result: Dict[str, Any],
        duration_ms: int,
        status: str,
    ) -> None:
        """Best-effort run-history persistence (chat/trigger/test runs)."""
        if not self._run_ctx or self._exec_opts.get("record_runs") is False:
            return
        try:
            engine = getattr(self._rag, "engine", None)
            if engine is None:
                return
            if self._chain_store is None:
                from app.agent_chains_store import AgentChainsStore
                self._chain_store = AgentChainsStore(engine)
                await self._chain_store.ensure_tables()
            await self._chain_store.record_run(
                chain_id=str(self._run_ctx.get("chain_id") or ""),
                chain_name=str(self._run_ctx.get("chain_name") or ""),
                source=str(self._run_ctx.get("source") or "run"),
                conversation_id=self._run_ctx.get("conversation_id") or None,
                duration_ms=duration_ms,
                status=status,
                input_text=input_text,
                answer=str(result.get("answer") or ""),
                node_status=self._node_status,
                node_outputs={k: v.get("content", "") for k, v in self._node_outputs.items()},
            )
        except Exception:
            logger.debug("chain run recording failed", exc_info=True)

    async def run(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the chain DAG non-streaming and return the final result.

        Independent nodes are executed in parallel via asyncio.gather.
        """
        self._reset_run_state()
        started = time.monotonic()
        await self._emit("run_start", source=self._run_ctx.get("source"))
        result: Dict[str, Any] = {}
        status = "done"
        try:
            executable = [nid for nid in self._execution_order if self._node_type(nid) != "note"]

            await self._run_levelwise(executable, input_text, chat_history, conversation_id, chat_user_id)

            # Find output node(s)
            output_nodes = [
                n for n in self._nodes
                if node_type_of(n) == "output"
            ]

            final_content = ""
            if not output_nodes:
                last = self._node_outputs.get(self._execution_order[-1], {})
                final_content = last.get("content", "")
            else:
                final_node_id = output_nodes[0]["id"]
                if final_node_id in self._skipped:
                    final_content = self._leaf_content_fallback()
                else:
                    final = self._node_outputs.get(final_node_id, {})
                    final_content = final.get("content", "")

            # Aggregate reasoning and tool_calls from all executed nodes
            all_reasoning: List[str] = []
            all_tool_calls: List[Dict[str, Any]] = []
            for nid in self._execution_order:
                if nid in self._skipped:
                    continue
                out = self._node_outputs.get(nid)
                if out and out.get("reasoning"):
                    label = self._get_node_label(nid)
                    all_reasoning.append(f"[{label}]:\n{out['reasoning']}")
                if out and out.get("tool_calls"):
                    all_tool_calls.extend(out["tool_calls"])
            aggregated_reasoning = "\n\n".join(all_reasoning)

            result = {
                "answer": final_content,
                "reasoning": aggregated_reasoning,
                "node_outputs": {k: v["content"] for k, v in self._node_outputs.items()},
                "node_status": self._node_status,
                "tool_calls": all_tool_calls,
            }
            if any(s.get("status") == "error" for s in self._node_status.values()):
                status = "partial_error"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            await self._emit(
                "run_end",
                status=status,
                duration_ms=duration_ms,
                answer=str(result.get("answer") or "")[:2000],
            )
            await self._record_run(input_text, result, duration_ms, status)

    async def run_streaming(
        self,
        input_text: str,
        chat_history: Optional[List[BaseMessage]] = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the chain and stream the final output node."""
        self._reset_run_state()
        started = time.monotonic()
        await self._emit("run_start", source=self._run_ctx.get("source"))
        completed = False
        final_output_id: Optional[str] = None
        try:
            # Find the final output node
            output_nodes = [
                n for n in self._nodes
                if node_type_of(n) == "output"
            ]
            if not output_nodes:
                # No output node defined; execute all nodes and concatenate outputs from all leaf nodes
                executable = [nid for nid in self._execution_order if self._node_type(nid) != "note"]
                await self._run_levelwise(executable, input_text, chat_history, conversation_id, chat_user_id)
                final_content = self._leaf_content_fallback()
                # Collect tool_calls from all executed nodes
                all_tool_calls: List[Dict[str, Any]] = []
                for nid in self._execution_order:
                    if nid in self._skipped:
                        continue
                    out = self._node_outputs.get(nid)
                    if out and out.get("tool_calls"):
                        all_tool_calls.extend(out["tool_calls"])
                yield {"content": final_content, "reasoning": "", "tool_calls": all_tool_calls or None}
                completed = True
                return

            final_output_id = output_nodes[0]["id"]

            # Determine which nodes are "preparatory" (not on the path to the final output)
            # Build reverse reachability from output
            reachable_from_output: Set[str] = set()
            queue = deque([final_output_id])
            while queue:
                current = queue.popleft()
                reachable_from_output.add(current)
                for pred in _get_predecessors(current, self._edges):
                    if pred not in reachable_from_output:
                        queue.append(pred)

            # Identify the direct predecessor of the output node (the node that produces the stream)
            output_preds = _get_predecessors(final_output_id, self._edges)
            stream_source_id: Optional[str] = None
            if len(output_preds) == 1:
                pred = _get_node_by_id(self._nodes, output_preds[0])
                pred_type = node_type_of(pred) if pred else ""
                if pred_type in ("assistant", "agent", "executor"):
                    stream_source_id = output_preds[0]

            # Execute preparatory nodes level-wise (router decisions are only
            # known at runtime — a single gather over the whole set would run
            # branches that the decision may kill).
            preparatory_ids = [
                nid for nid in self._execution_order
                if nid != final_output_id
                and nid != stream_source_id
                and nid in reachable_from_output
                and self._node_type(nid) != "note"
            ]
            if preparatory_ids:
                self._subagent_event_queue = asyncio.Queue()

                async def _run_preparatory_nodes() -> None:
                    await self._run_levelwise(preparatory_ids, input_text, chat_history, conversation_id, chat_user_id)

                prep_task = asyncio.create_task(_run_preparatory_nodes())
                try:
                    while not prep_task.done():
                        event_task = asyncio.create_task(self._subagent_event_queue.get())
                        done, pending = await asyncio.wait(
                            {prep_task, event_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        if event_task in done:
                            yield {"content": "", "reasoning": "", "subagent": event_task.result()}
                    await prep_task
                    while not self._subagent_event_queue.empty():
                        yield {
                            "content": "",
                            "reasoning": "",
                            "subagent": self._subagent_event_queue.get_nowait(),
                        }
                finally:
                    self._subagent_event_queue = None
                    if not prep_task.done():
                        prep_task.cancel()
                        await asyncio.gather(prep_task, return_exceptions=True)

            # Collect tool_calls from preparatory nodes and emit before final stream
            prep_tool_calls: List[Dict[str, Any]] = []
            for nid in preparatory_ids:
                if nid in self._skipped:
                    continue
                out = self._node_outputs.get(nid)
                if out and out.get("tool_calls"):
                    prep_tool_calls.extend(out["tool_calls"])
            if prep_tool_calls:
                yield {"content": "", "reasoning": "", "tool_calls": prep_tool_calls}

            # Compute the input for the stream source / output
            def _content(node_id: str) -> str:
                out = self._node_outputs.get(node_id)
                return out["content"] if out else ""

            if final_output_id in self._skipped:
                # The route to the output was not taken — aggregate live leaves.
                yield {"content": self._leaf_content_fallback(), "reasoning": "", "tool_calls": None}
                completed = True
                return

            if len(output_preds) == 1:
                pred_id = output_preds[0]
                if pred_id == stream_source_id:
                    # Compute input for the assistant node (same logic as _execute_node)
                    pred_preds = live_predecessors(pred_id, self._edges, self._dead_edges, self._skipped)
                    if len(pred_preds) == 1:
                        final_input = _content(pred_preds[0])
                    elif len(pred_preds) > 1:
                        parts = [f"user message:\n{input_text}"]
                        for p in pred_preds:
                            label = self._get_node_label(p)
                            text = _content(p)
                            if text:
                                parts.append(f"{label}:\n{text}")
                        final_input = "\n\n".join(parts)
                    else:
                        final_input = input_text
                else:
                    final_input = _content(pred_id)
            elif len(output_preds) > 1:
                live = live_predecessors(final_output_id, self._edges, self._dead_edges, self._skipped)
                parts = [f"user message:\n{input_text}"]
                for p in (live or output_preds):
                    label = self._get_node_label(p)
                    text = _content(p)
                    if text:
                        parts.append(f"{label}:\n{text}")
                final_input = "\n\n".join(parts)
            else:
                final_input = input_text

            # Collect reasoning from preparatory nodes for streaming
            prep_reasoning: List[str] = []
            for nid in self._execution_order:
                if nid == final_output_id or nid == stream_source_id or nid in self._skipped:
                    continue
                out = self._node_outputs.get(nid)
                if out and out.get("reasoning"):
                    label = self._get_node_label(nid)
                    prep_reasoning.append(f"[{label}]:\n{out['reasoning']}")
            prep_reasoning_text = "\n\n".join(prep_reasoning)

            # Stream from assistant if the predecessor is an assistant node
            if stream_source_id and stream_source_id not in self._skipped:
                pred_node = _get_node_by_id(self._nodes, stream_source_id)
                pred_data = pred_node.get("data", {}) if pred_node else {}
                pred_type = node_type_of(pred_node) if pred_node else ""
                assistant_name = pred_data.get("assistant_name", "")
                if pred_type == "executor":
                    if (pred_data.get("target_kind") or "assistant") != "chain":
                        assistant_name = assistant_name or str(pred_data.get("target") or "")
                    else:
                        assistant_name = ""  # sub-chains are not streamable
                if not assistant_name:
                    # Executor pointing at a sub-chain — run nested, emit once.
                    if pred_type == "executor" and (pred_data.get("target_kind") or "") == "chain":
                        out = await self._execute_subchain_node(
                            pred_node, str(pred_data.get("target") or ""), final_input,
                            chat_history, conversation_id, chat_user_id,
                        )
                        self._node_outputs[stream_source_id] = out
                        ok = not str(out.get("content", "")).startswith("[Error")
                        self._set_status(stream_source_id, "done" if ok else "error")
                        yield {"content": out.get("content", ""), "reasoning": out.get("reasoning", ""), "tool_calls": out.get("tool_calls") or None}
                        completed = True
                        return
                    logger.error("Streaming assistant node %s has no assistant_name", stream_source_id)
                    self._set_status(stream_source_id, "error")
                    yield {"content": f"[Error: Assistant node {stream_source_id} has no assistant_name]", "reasoning": "", "tool_calls": None}
                    completed = True
                    return
                agent_agen = None
                streamed_parts: List[str] = []
                await self._emit("node_start", node_id=stream_source_id, node_type=pred_type)
                self._set_status(stream_source_id, "running")
                try:
                    prep_reasoning_emitted = False
                    stream_source_label_emitted = False
                    agent_agen = self._rag.run_streaming(
                        user_input=final_input,
                        chat_history=chat_history,
                        model_name=assistant_name,
                        conversation_id=conversation_id,
                        chat_user_id=chat_user_id,
                    )
                    async for chunk in agent_agen:
                        streamed_parts.append(str(chunk.get("content") or ""))
                        # Merge preparatory reasoning with assistant's own reasoning
                        chunk_reasoning = chunk.get("reasoning", "")
                        if prep_reasoning_text and not prep_reasoning_emitted:
                            if chunk_reasoning:
                                merged_reasoning = f"{prep_reasoning_text}\n\n[{self._get_node_label(stream_source_id)}]:\n{chunk_reasoning}"
                                stream_source_label_emitted = True
                            else:
                                merged_reasoning = prep_reasoning_text
                            prep_reasoning_emitted = True
                        elif prep_reasoning_text:
                            if chunk_reasoning:
                                if not stream_source_label_emitted:
                                    merged_reasoning = f"[{self._get_node_label(stream_source_id)}]:\n{chunk_reasoning}"
                                    stream_source_label_emitted = True
                                else:
                                    merged_reasoning = chunk_reasoning
                            else:
                                merged_reasoning = ""
                        else:
                            merged_reasoning = chunk_reasoning
                        if merged_reasoning:
                            yield {
                                "content": chunk.get("content", ""),
                                "reasoning": merged_reasoning,
                                "tool_calls": chunk.get("tool_calls"),
                                "subagent": chunk.get("subagent"),
                                "observer": chunk.get("observer"),
                                "live_usage": chunk.get("live_usage"),
                            }
                        else:
                            yield {
                                "content": chunk.get("content", ""),
                                "reasoning": "",
                                "tool_calls": chunk.get("tool_calls"),
                                "subagent": chunk.get("subagent"),
                                "observer": chunk.get("observer"),
                                "live_usage": chunk.get("live_usage"),
                            }
                    # If stream ended without ever emitting prep reasoning, yield it now
                    if prep_reasoning_text and not prep_reasoning_emitted:
                        yield {"content": "", "reasoning": prep_reasoning_text, "tool_calls": None}
                    streamed = "".join(streamed_parts)
                    self._node_outputs[stream_source_id] = {
                        "content": streamed,
                        "reasoning": prep_reasoning_text,
                        "tool_calls": [],
                    }
                    if final_output_id:
                        self._node_outputs[final_output_id] = {
                            "content": streamed,
                            "reasoning": "",
                            "tool_calls": [],
                        }
                    self._set_status(stream_source_id, "done")
                    await self._emit("node_end", node_id=stream_source_id, status="done")
                    completed = True
                    return
                except Exception as exc:
                    logger.exception("Streaming final assistant node failed")
                    self._set_status(stream_source_id, "error", error=str(exc))
                    await self._emit("node_end", node_id=stream_source_id, status="error")
                    yield {"content": f"[Error: {exc}]", "reasoning": "", "tool_calls": None}
                    completed = True
                    return
                finally:
                    # Deterministic close of the inner generator (subagent
                    # shutdown lives in its finally) instead of asyncgen GC.
                    if agent_agen is not None:
                        await agent_agen.aclose()

            # Fallback: just output the final result as single chunk
            # Aggregate reasoning and tool_calls from all preparatory nodes
            all_reasoning: List[str] = []
            all_tool_calls: List[Dict[str, Any]] = []
            for nid in self._execution_order:
                if nid == final_output_id or nid in self._skipped:
                    continue
                out = self._node_outputs.get(nid)
                if out and out.get("reasoning"):
                    label = self._get_node_label(nid)
                    all_reasoning.append(f"[{label}]:\n{out['reasoning']}")
                if out and out.get("tool_calls"):
                    all_tool_calls.extend(out["tool_calls"])
            aggregated_reasoning = "\n\n".join(all_reasoning)

            self._node_outputs[final_output_id] = {"content": final_input, "reasoning": aggregated_reasoning, "tool_calls": []}
            yield {"content": final_input, "reasoning": aggregated_reasoning, "tool_calls": None}
            completed = True
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            status = "done" if completed else "aborted"
            if completed and any(
                s.get("status") == "error" for s in self._node_status.values()
            ):
                status = "partial_error"
            await self._emit(
                "run_end",
                status=status,
                duration_ms=duration_ms,
                answer=str(
                    (self._node_outputs.get(final_output_id) or {}).get("content", "")
                )[:2000],
            )
            await self._record_run(
                input_text,
                {
                    "answer": str(
                        (self._node_outputs.get(final_output_id) or {}).get("content", "")
                    )[:65536]
                },
                duration_ms,
                status,
            )
