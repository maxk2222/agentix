"""Agent runner that enables multi-step tool use for assistants."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from uuid import uuid4

try:
    from openai import BadRequestError as _OpenAIBadRequestError
except ImportError:
    _OpenAIBadRequestError = type("BadRequestError", (Exception,), {})
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.tools import ToolRegistry
from agent.tools.validation import validate_and_normalize_args, truncate_tool_output
from app.chat_utils import render_system_instructions, _format_current_datetime_german, extract_usage_metadata, count_tokens, stringify_message_content
from app.context_summarizer import summarize_context_if_needed, _hard_truncate
from app.history_budget import RollingHistorySettings, count_text_tokens, count_messages_tokens
from agent.subagents import SubagentState
from agent.stream_watchdog import DEFAULT_STREAM_IDLE_SECONDS, StreamIdleTimeout, idle_bounded_agen
from agent.tools.builtin_tools.todo_list import (
    OPEN_STATUSES,
    TODO_TOOL_NAME,
    TodoListState,
)
from agent.observer import ObserverRunSnapshot

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 10
MAX_TOOL_CALLS = 10
# Cadence of synthetic tool_progress keepalives while a tool batch executes —
# they keep the SSE wire and the dashboard progress display alive but carry a
# `keepalive` flag and do not count as run activity (module-level for tests).
TOOL_KEEPALIVE_SECONDS = 45.0
# How often a run may attempt to finalize while todo tasks are still open
# before finalization is forced anyway (leftovers are swept to 'skipped').
MAX_TODO_FINALIZE_REMINDERS = 3
SUBAGENT_COORDINATION_TOOLS = {
    "spawn_subagent",
    "wait_subagent",
    "subagent_status",
    "nudge_subagent",
    "wait",
}
# Tools that coordinate the agent's own work and never consume the regular
# tool-call budget: subagent management, waits, and the run-local todo list.
INTERNAL_TOOLS = SUBAGENT_COORDINATION_TOOLS | {TODO_TOOL_NAME}

# Tool whose results count as RAG context: its output tokens feed the
# rag_context_tokens metric so retrieval cost stays visible in dashboards.
RAG_TOOL_NAME = "rag_search"

async def _invoke_with_idle_timeout(
    llm: Any,
    messages: List[BaseMessage],
    idle_timeout: float = DEFAULT_STREAM_IDLE_SECONDS,
) -> AIMessage:
    """Invoke LLM with activity-based idle timeout using streaming.

    If no token arrives within ``idle_timeout``, the call is cancelled.
    There is intentionally no fixed total limit — a model that keeps
    producing tokens may take as long as it needs.
    """
    chunks: List[Any] = []
    last_activity = time.time()
    stream_done = asyncio.Event()

    async def _consume() -> None:
        nonlocal last_activity
        try:
            async for chunk in llm.astream(messages):
                last_activity = time.time()
                chunks.append(chunk)
        finally:
            stream_done.set()

    task = asyncio.create_task(_consume())

    # Monitor with idle timeout
    while not task.done():
        elapsed = time.time() - last_activity
        remaining = max(0.1, idle_timeout - elapsed)
        try:
            await asyncio.wait_for(stream_done.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            if time.time() - last_activity >= idle_timeout:
                logger.warning("LLM idle timeout after %.1fs", idle_timeout)
                task.cancel()
                break

    try:
        await task
    except asyncio.CancelledError:
        pass

    if not chunks:
        raise asyncio.TimeoutError("No response within idle timeout")

    # Combine chunks into AIMessage
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined = combined + chunk

    if isinstance(combined, AIMessage):
        return combined

    content = getattr(combined, "content", str(combined)) or ""
    tool_calls = list(getattr(combined, "tool_calls", []))
    return AIMessage(content=content, tool_calls=tool_calls)


def _build_tool_prompt(max_total: int, max_parallel: int, fallback: bool) -> str:
    """Build the tool-calling system prompt with the current limits."""
    limit_text = (
        f"You may call up to {max_total} regular (non-coordination) tools in total, "
        f"and up to {max_parallel} tools in parallel in a single response. "
        "Coordination tools such as wait and subagent management have separate limits. "
    )
    if fallback:
        return (
            "You have access to tools. You MUST use them whenever they help answer the user's question. "
            + limit_text +
            "At EACH step, decide: do you still need information from a tool, or can you answer?\n"
            "  - If you need a tool, output ONLY a JSON object or array of JSON objects. No other text. No markdown. No explanations.\n"
            '    Single tool: {"tool_name": {"arg1": "value1"}}\n'
            '    Multiple tools: [{"tool_name1": {"arg1": "value1"}}, {"tool_name2": {"arg2": "value2"}}]\n'
            "  - If you already have ALL necessary information, answer ONLY in natural language. Do NOT output JSON."
        )
    return (
        "You have access to tools. You MUST use them whenever they help answer the user's question. "
        + limit_text +
        "If tool calling is available, use the tool_calls mechanism. "
        "If you cannot use tool_calls, output ONLY a single JSON object like: "
        '{"tool_name": {"arg1": "value1"}} — no other text, no markdown, no explanations. '
        "If you already have ALL necessary information, answer in natural language."
    )


def _build_native_tool_prompt(max_total: int, max_parallel: int) -> str:
    """Build a concise tool prompt for native tool-calling mode.

    Omits JSON fallback instructions since the LLM receives tool schemas via
    ``bind_tools``, but keeps the behavioural instructions so the model knows
    *when* to call tools and *when* to stop.
    """
    return (
        "You have access to tools. Use them whenever they help answer the user's question. "
        f"You may call up to {max_total} regular (non-coordination) tools in total, "
        f"and up to {max_parallel} tools in parallel in a single response. "
        "Coordination tools such as wait and subagent management have separate limits. "
        "If you already have ALL necessary information, answer in natural language. "
        "Do not explain your plans before acting. "
        "When answering, output only natural language. Do not use XML tags, code blocks, or structured formats."
    )


def _has_bind_tools(llm: Any) -> bool:
    """Check whether the LLM instance advertises tool binding."""
    return callable(getattr(llm, "bind_tools", None))


def _build_tool_description(tool_schemas: List[Dict[str, Any]]) -> str:
    """Build a human-readable description of available tools for the system prompt."""
    if not tool_schemas:
        return ""

    def _example_value(param_type: str) -> str:
        """Return a concrete example value for a given JSON schema type."""
        examples = {
            "string": '"<value>"',
            "integer": "1",
            "number": "1.0",
            "boolean": "true",
            "array": '["item1"]',
            "object": "{}",
        }
        return examples.get(param_type, '"..."')

    lines: List[str] = ["\n\nAvailable tools:"]
    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        lines.append(f"\n- {name}: {desc}")
        if props:
            lines.append("  Allowed parameters (use ONLY these - do NOT add others):")
            for param_name, param_info in props.items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                req_flag = " (required)" if param_name in required else ""
                lines.append(f'    - {param_name} ({param_type}){req_flag}: {param_desc}')
        # Build exact call format with concrete example values
        if props:
            allowed = ", ".join(f'"{p}": {_example_value(param_info.get("type", "any"))}' for p, param_info in props.items())
            lines.append(f'  Call format: {{"{name}": {{{allowed}}}}}')
        else:
            lines.append(f'  Call format: {{"{name}": {{}} }}')
    return "\n".join(lines)


def _extract_json_spans(text: str) -> List[Tuple[str, int, int]]:
    """Extract all top-level JSON values with their start/end indices.

    Returns list of (json_text, start_index, end_index) using balanced bracket
    scanner with string-literal awareness.
    """
    spans: List[Tuple[str, int, int]] = []
    brace_depth = 0
    bracket_depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if brace_depth == 0 and bracket_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and bracket_depth == 0 and start != -1:
                spans.append((text[start:i + 1], start, i + 1))
                start = -1
        elif ch == "[":
            if bracket_depth == 0 and brace_depth == 0:
                start = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and brace_depth == 0 and start != -1:
                spans.append((text[start:i + 1], start, i + 1))
                start = -1

    return spans


def _build_flat_dict_mapping(
    registry: Any, tool_names: List[str]
) -> Dict[str, List[str]]:
    """Build a mapping from parameter name to list of tool names that require it.

    Inspects each tool's ``tool_schema`` (OpenAI function-calling format) and
    extracts ``parameters.required`` so the fallback parser can match flat dicts
    to the correct tool dynamically.
    """
    mapping: Dict[str, List[str]] = {}
    for name in tool_names:
        tool = registry.get_tool(name)
        if tool is None:
            continue
        schema = getattr(tool, "tool_schema", None)
        if not isinstance(schema, dict):
            continue
        # OpenAI schema: {"type": "function", "function": {"parameters": {"required": [...]}}}
        params = schema.get("function", {}).get("parameters", {})
        if not isinstance(params, dict):
            continue
        required = params.get("required", [])
        if isinstance(required, list):
            for param in required:
                mapping.setdefault(str(param), []).append(name)
    return mapping


def _try_parse_tool_calls(
    content: Optional[str],
    tool_names: List[str],
    flat_dict_mapping: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Try to extract tool calls from raw JSON text and return remaining text.

    Supports two forms:
    1. Tool-name as top-level key: {"web_search": {"query": "..."}}
    2. Flat args dict: {"query": "...", "results": 10}
       (only when ``flat_dict_mapping`` is provided)
    3. JSON arrays with multiple tool calls: [{"tool1": {...}}, {"tool2": {...}}]

    Returns (tool_calls, remaining_text) where tool_calls is a list of synthetic
    tool call dicts or None if parsing fails, and remaining_text is the original
    text with successfully parsed JSON blocks removed.
    """
    if not content:
        return None, ""
    candidates: List[Dict[str, Any]] = []
    consumed_spans: List[Tuple[int, int]] = []

    # Helper: try to parse a string fragment into dicts or lists
    def _try_parse(text: str, start_idx: int = -1, end_idx: int = -1) -> None:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                candidates.append(parsed)
                if start_idx >= 0 and end_idx >= 0:
                    consumed_spans.append((start_idx, end_idx))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        candidates.append(item)
                if start_idx >= 0 and end_idx >= 0:
                    consumed_spans.append((start_idx, end_idx))
        except (json.JSONDecodeError, ValueError):
            pass

    # 1. Find fenced JSON blocks and extract values inside
    for fence_match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE):
        fence_start = fence_match.start(1)
        for val, _, _ in _extract_json_spans(fence_match.group(1)):
            _try_parse(val)

    # 2. Find ALL JSON values in the full text, tracking their exact positions
    for val, start_idx, end_idx in _extract_json_spans(content):
        _try_parse(val, start_idx, end_idx)

    # Collect candidate tool calls from ALL parsed candidates
    results: List[Dict[str, Any]] = []
    for parsed in candidates:
        # Form 1: tool name is a top-level key
        matched_tools = []
        for name in tool_names:
            if name in parsed:
                args = parsed[name]
                if not isinstance(args, dict):
                    args = {}
                matched_tools.append({"name": name, "args": args, "id": f"fallback_{len(results) + len(matched_tools) + 1}"})

        if matched_tools:
            results.extend(matched_tools)
            continue

        # Form 2: flat dict – match by required parameters
        if flat_dict_mapping:
            keys = set(parsed.keys())
            tool_scores: Dict[str, int] = {}
            for param, tnames in flat_dict_mapping.items():
                if param in keys:
                    for tname in tnames:
                        tool_scores[tname] = tool_scores.get(tname, 0) + 1
            if tool_scores:
                max_score = max(tool_scores.values())
                best_tools = [t for t, s in tool_scores.items() if s == max_score]
                # Only accept if there is a single unambiguous best match
                if len(best_tools) == 1:
                    best_tool = best_tools[0]
                    results.append({"name": best_tool, "args": parsed, "id": f"fallback_{len(results) + 1}"})

    # Build remaining text by removing consumed spans
    if consumed_spans:
        consumed_spans.sort()
        remaining_parts: List[str] = []
        last_end = 0
        for start, end in consumed_spans:
            if start > last_end:
                remaining_parts.append(content[last_end:start])
            last_end = max(last_end, end)
        if last_end < len(content):
            remaining_parts.append(content[last_end:])
        remaining_text = " ".join(part.strip() for part in remaining_parts if part.strip())
    else:
        remaining_text = content.strip()

    return (results if results else None), remaining_text


def _extract_reasoning_from_message(msg: Any) -> str:
    """Pull reasoning_content from a LangChain message's additional_kwargs."""
    if hasattr(msg, "additional_kwargs"):
        ak = msg.additional_kwargs or {}
        return ak.get("reasoning_content", "") or ak.get("reasoning", "")
    return ""


def _normalize_tool_call(raw_call: Dict[str, Any], allowed_tool_names: List[str]) -> Optional[Dict[str, Any]]:
    """Normalize a raw tool call into the internal format used by the runner.

    Supports:
    - LangChain format: {"name": ..., "args": ..., "id": ...}
    - OpenAI raw format: {"type":"function", "function":{"name": ..., "arguments": ...}}
    - Flat fallback format: {"name": ..., "args": ..., "id": ...}

    Returns None if the tool name is not in allowed_tool_names or the format is invalid.
    """
    if not isinstance(raw_call, dict):
        return None

    # Try LangChain / flat fallback format first
    name = raw_call.get("name")
    args = raw_call.get("args")
    tid = raw_call.get("id")

    # Try OpenAI raw format
    if name is None:
        func = raw_call.get("function")
        if isinstance(func, dict):
            name = func.get("name")
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = None
            tid = raw_call.get("id")

    if not name or name not in allowed_tool_names:
        return None
    if not isinstance(args, dict):
        args = {}
    if not tid:
        tid = f"fallback_{name}"

    return {"name": name, "args": args, "id": tid}


@dataclass
class ToolCallRecord:
    """Structured record of a single tool call for diagnostics."""
    iteration: int
    tool_name: str
    args: Dict[str, Any]
    started_at: float
    ended_at: float = 0.0
    ok: bool = True
    error: Optional[str] = None
    output_chars: int = 0
    truncated: bool = False
    summarized: bool = False
    output: Optional[str] = None
    kind: str = "tool"
    metadata: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None


@dataclass
class AgentTrace:
    """Complete trace of an agent run for observability."""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    iterations: int = 0
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    finalization: str = ""
    error: Optional[str] = None
    fallback_mode: bool = False
    observer_events: List[Dict[str, Any]] = field(default_factory=list)
    todo_gate_blocks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round((self.ended_at - self.started_at) * 1000, 1) if self.ended_at else None,
            "iterations": self.iterations,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [
                {
                    "iteration": rec.iteration,
                    "tool_name": rec.tool_name,
                    "args": rec.args,
                    "ok": rec.ok,
                    "error": rec.error,
                    "duration_ms": round((rec.ended_at - rec.started_at) * 1000, 1) if rec.ended_at else None,
                    "output": rec.output,
                    "summarized": rec.summarized,
                    "kind": rec.kind,
                    "metadata": rec.metadata,
                    "id": rec.call_id,
                }
                for rec in self.tool_calls
            ],
            "finalization": self.finalization,
            "error": self.error,
            "fallback_mode": self.fallback_mode,
            "observer_events": list(self.observer_events),
            "todo_gate_blocks": self.todo_gate_blocks,
        }


class AgentRunner:
    """Run an agent loop for a given assistant with tool access.

    The runner builds a LangChain-compatible conversation, binds available tools to
    the LLM, and iterates until the model produces a final answer or a limit is
    reached.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        tool_names: List[str],
        max_iterations: int = MAX_AGENT_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_parallel_tool_calls: int = 5,
        supports_native_tool_calls: Optional[bool] = None,
        conversation_id: Optional[str] = None,
        agent_memory_enabled: bool = False,
        assistant_name: Optional[str] = None,
        agent_memory_store: Optional[Any] = None,
        agent_memory_already_injected: bool = False,
        tool_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        rolling_history_settings: Optional[RollingHistorySettings] = None,
        summary_llm_factory: Optional[Any] = None,
        subagent_state: Optional[SubagentState] = None,
        stream_idle_seconds: float = DEFAULT_STREAM_IDLE_SECONDS,
        injected_input_queue: Optional[asyncio.Queue] = None,
        input_event: Optional[asyncio.Event] = None,
        observer: Optional[Any] = None,
        workspace_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._idle_seconds = stream_idle_seconds
        self.registry = registry
        self.tool_names = tool_names
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.supports_native_tool_calls = supports_native_tool_calls
        self.conversation_id = conversation_id
        self.agent_memory_enabled = agent_memory_enabled
        self.assistant_name = assistant_name or "default"
        self.agent_memory_store = agent_memory_store
        self.agent_memory_already_injected = agent_memory_already_injected
        self.tool_overrides = tool_overrides or {}
        self._rolling_history_settings = rolling_history_settings
        self._summary_llm_factory = summary_llm_factory
        self._subagent_state = subagent_state
        # User messages queued for mid-run injection (steer). Drained at the
        # top of every agent iteration and appended as HumanMessages.
        self._injected_input_queue = injected_input_queue
        # Set alongside every queued input so mid-iteration blocking tools
        # (`wait`, `wait_subagent`, auto-collect) can wake early.
        self._input_event = input_event
        # Optional run observer: analyses progress at iteration boundaries,
        # maintains the todo list and injects [Observer note] guidance.
        self._observer = observer
        # Resolved agent-workspace {dir, cfg} for this run (or inherited via
        # subagent_state); tools read it from the execution context.
        self._workspace_ctx = workspace_ctx or (
            subagent_state.workspace_ctx if subagent_state is not None else None
        )
        # Stable per-run ids for content dedup: identical images (by data URI)
        # and identical large tool outputs (by sha256) appear exactly once in
        # the model payload; earlier occurrences become reference stubs.
        self._media_ids: Dict[str, str] = {}
        self._result_ids: Dict[str, str] = {}
        # Run-local todo list, (re)created at the start of every run so each
        # run — including each spawned subagent — gets an isolated list.
        self._todo_state: Optional[TodoListState] = None
        self._emitted_todo_version = -1
        self._todo_gate_reminders = 0
        self._last_ai_text = ""
        self._last_reasoning_tail = ""
        self._injected_since_observer = 0
        logger.debug(
            "AgentRunner initialized: tools=%s, max_iter=%d, max_calls=%d, max_parallel=%d, supports_native=%s, conversation_id=%s, agent_memory=%s, already_injected=%s",
            tool_names, max_iterations, max_tool_calls, max_parallel_tool_calls, supports_native_tool_calls, conversation_id, agent_memory_enabled, agent_memory_already_injected,
        )

    def _build_messages(
        self,
        system_instructions: str,
        chat_history: Sequence[BaseMessage],
        input_text: str,
        fallback_mode: bool,
        tool_schemas: List[Dict[str, Any]],
        rag_context: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
    ) -> List[BaseMessage]:
        """Build the initial message list with system prompt, history and user input."""
        if system_instructions.strip():
            sys_text = system_instructions.strip()
        else:
            sys_text = ""
        if rag_context:
            if "{{context}}" in sys_text:
                sys_text = sys_text.replace("{{context}}", rag_context)
            else:
                sys_text += f"\n\nRelevant context:\n{rag_context}"
        subagent_state = getattr(self, "_subagent_state", None)
        if subagent_state and subagent_state.depth < subagent_state.manager.max_depth:
            sys_text += (
                "\n\n[Subagents]\n"
                "You can delegate independent subtasks to background clones of yourself using "
                "`spawn_subagent(task, label)`. A clone has the same instructions and read-only research/retrieval tools, "
                "but no access to this conversation, so include all needed context in its task. "
                f"Up to {subagent_state.manager.max_concurrent} run concurrently; total spawn "
                f"budget is {subagent_state.manager.max_total}. Spawn independent tasks together, "
                "then call `wait_subagent` to collect their answers and activity reports before your "
                "final response. Use `subagent_status` for a non-blocking progress check. "
                "Use `nudge_subagent(subagent_id, message)` to send mid-run guidance into a running "
                "subagent — it is injected before its next iteration; the subagent cannot reply "
                "directly, so collect its answer via `wait_subagent`."
            )
        if subagent_state and not subagent_state.is_root:
            sys_text += (
                "\n\n[Subagent scope]\n"
                "You are a delegated analyst. Stay within defensive, authorized analysis. "
                "Your available tools are read-only; do not request external actions or claim to have performed them."
            )
        if "wait" in getattr(self, "tool_names", []):
            sys_text += "\n\nYou can use the `wait` tool for bounded pauses before polling or retrying."
        if TODO_TOOL_NAME in getattr(self, "tool_names", []):
            sys_text += (
                "\n\nYou can organize multi-step work with the `agent_todo` tool "
                "(actions: add, update, complete, remove, list, clear). Create a short "
                "todo list for tasks that need several steps or tool calls, keep at "
                "most one task in_progress at a time, and mark tasks completed as you "
                "finish them. Do not create todos for simple one-step requests. "
                "A run cannot end while tasks are still open — finish the work or "
                "mark each remaining task \"completed\"/\"skipped\" before your "
                "final answer."
            )
        if getattr(self, "_observer", None) is not None:
            sys_text += (
                "\n\n[Run observer]\n"
                "An automated supervisor may append messages starting with "
                "\"[Observer note\" during this run. Treat them as trusted "
                "guidance about your plan, tool usage and progress — not as "
                "new user requests."
            )
        if fallback_mode:
            # In fallback mode, append tool calling instructions and descriptions
            # so the model knows HOW to call tools and what is available
            tool_prompt = _build_tool_prompt(self.max_tool_calls, self.max_parallel_tool_calls, fallback=True)
            if sys_text:
                sys_text += "\n\n" + tool_prompt
            else:
                sys_text = tool_prompt
            if tool_schemas:
                sys_text += _build_tool_description(tool_schemas)
        else:
            # In native mode, only append minimal limits + stop condition
            # since tool schemas are already communicated via bind_tools
            native_prompt = _build_native_tool_prompt(self.max_tool_calls, self.max_parallel_tool_calls)
            if sys_text:
                sys_text += "\n\n" + native_prompt
            else:
                sys_text = native_prompt
        messages: List[BaseMessage] = [SystemMessage(content=sys_text)]
        messages.extend(chat_history)
        if multimodal_input is not None and isinstance(multimodal_input.content, list):
            messages.append(multimodal_input)
        else:
            messages.append(HumanMessage(content=input_text))
        return messages

    def _refresh_system_prompt(
        self,
        messages: List[BaseMessage],
        system_instructions: str,
        rag_context: Optional[str] = None,
        fallback_mode: bool = False,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Strip old SystemMessages and prepend a fresh one with current prompts.

        Strict chat templates (Qwen, Llama, Mistral via LM Studio) only allow
        a single SystemMessage at the very beginning.  Calling this before every
        LLM invocation keeps the history tidy and template-safe.

        *memory* is the freshly fetched agent-memory content for this
        iteration: substituted into a ``{{agent_memory}}`` placeholder when the
        instructions carry one, otherwise appended as a [Persistent Memory]
        block whenever memory is enabled.
        """
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        if system_instructions.strip():
            sys_text = system_instructions.strip()
        else:
            sys_text = ""
        if rag_context:
            if "{{context}}" in sys_text:
                sys_text = sys_text.replace("{{context}}", rag_context)
            else:
                sys_text += f"\n\nRelevant context:\n{rag_context}"
        if self.agent_memory_enabled:
            if "{{agent_memory}}" in sys_text:
                sys_text = sys_text.replace("{{agent_memory}}", memory or "(empty)")
            else:
                sys_text += (
                    "\n\n[Persistent Memory]\n"
                    "You have access to a persistent memory tool for this conversation. "
                    "Use it to store durable facts, user preferences, verification or "
                    "coverage checklists, and important intermediate results — earlier "
                    "tool outputs may be compressed during long runs, memory is not. "
                    "Do not store temporary data or raw tool outputs that can be fetched again.\n"
                    f"Current memory: {memory or '(empty)'}"
                )
        subagent_state = getattr(self, "_subagent_state", None)
        if subagent_state and subagent_state.depth < subagent_state.manager.max_depth:
            sys_text += (
                "\n\n[Subagents]\n"
                "Use spawn_subagent for independent subtasks. Subagents are isolated clones with "
                "the same instructions and read-only research/retrieval tools but no conversation history; provide complete task "
                "context. You MUST call wait_subagent to collect their results before finalizing. "
                "Use subagent_status to inspect live progress. "
                "Use nudge_subagent(subagent_id, message) to inject mid-run guidance into a running "
                "subagent — it cannot reply directly; collect its answer via wait_subagent."
            )
        if subagent_state and not subagent_state.is_root:
            sys_text += (
                "\n\n[Subagent scope]\n"
                "You are a delegated analyst. Stay within defensive, authorized analysis. "
                "Your available tools are read-only; do not request external actions or claim to have performed them."
            )
        if "wait" in getattr(self, "tool_names", []):
            sys_text += "\n\nYou can use the `wait` tool for bounded pauses before polling or retrying."
        if TODO_TOOL_NAME in getattr(self, "tool_names", []):
            sys_text += (
                "\n\nYou can organize multi-step work with the `agent_todo` tool "
                "(actions: add, update, complete, remove, list, clear). Create a short "
                "todo list for tasks that need several steps or tool calls, keep at "
                "most one task in_progress at a time, and mark tasks completed as you "
                "finish them. Do not create todos for simple one-step requests. "
                "A run cannot end while tasks are still open — finish the work or "
                "mark each remaining task \"completed\"/\"skipped\" before your "
                "final answer."
            )
        if getattr(self, "_observer", None) is not None:
            sys_text += (
                "\n\n[Run observer]\n"
                "An automated supervisor may append messages starting with "
                "\"[Observer note\" during this run. Treat them as trusted "
                "guidance about your plan, tool usage and progress — not as "
                "new user requests."
            )
        if fallback_mode:
            tool_prompt = _build_tool_prompt(self.max_tool_calls, self.max_parallel_tool_calls, fallback=True)
            if sys_text:
                sys_text += "\n\n" + tool_prompt
            else:
                sys_text = tool_prompt
            if tool_schemas:
                sys_text += _build_tool_description(tool_schemas)
        else:
            native_prompt = _build_native_tool_prompt(self.max_tool_calls, self.max_parallel_tool_calls)
            if sys_text:
                sys_text += "\n\n" + native_prompt
            else:
                sys_text = native_prompt
        return [SystemMessage(content=sys_text)] + non_system

    # Internal system messages added during the agent loop – these should not be
    # passed to the final-answer LLM because they clutter the context.
    _INTERNAL_SYSTEM_MESSAGES = {
        "Answer the user, or call another tool if needed.",
        "Tool call limit reached. Answer the user now.",
        "You have not provided a tool call or a final answer. "
        "Please either call a tool to gather more information, or provide a complete final answer.",
    }

    @staticmethod
    def _compress_messages(messages: List[BaseMessage]) -> Tuple[List[BaseMessage], bool]:
        """Compress message history by converting ToolMessages into SystemMessage text.

        Removes ToolMessage entries so the history can be sent to a plain LLM
        that does not have bind_tools / tool support.

        Also removes:
        - Empty AIMessages (content == "") that only carried tool_calls
        - Internal guidance SystemMessages added by the loop

        Returns (compressed_messages, all_tools_failed).
        """
        compressed: List[BaseMessage] = []
        tool_results: List[str] = []
        error_count = 0
        total_tool_results = 0
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content or ""
                total_tool_results += 1
                if content.startswith("[Error:") or content.startswith("[Tool argument error:"):
                    error_count += 1
                tool_results.append(content)
            elif isinstance(msg, AIMessage):
                # Keep AIMessages that have content OR carried tool_calls
                if (msg.content and msg.content.strip()) or msg.tool_calls:
                    compressed.append(msg)
            elif isinstance(msg, SystemMessage):
                # Skip internal loop guidance messages
                if msg.content not in AgentRunner._INTERNAL_SYSTEM_MESSAGES:
                    compressed.append(msg)
            else:
                compressed.append(msg)
        all_failed = total_tool_results > 0 and error_count == total_tool_results
        if tool_results:
            results_text = "\n---\n".join(tool_results)
            compressed.append(
                HumanMessage(
                    content=f"Below are the results from tools that were called:\n\n{results_text}"
                )
            )
        return compressed, all_failed

    @staticmethod
    def _light_compress_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Lightweight cleanup for native tool-calling mode.

        Keeps ToolMessages intact (needed for bind_tools correlation) while
        removing clutter that makes the LLM repeat itself:
        - Empty AIMessages (content == "") that only carried tool_calls
        - Internal guidance SystemMessages added by the loop
        """
        compressed: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Keep AIMessages that have content OR carried tool_calls
                if (msg.content and msg.content.strip()) or msg.tool_calls:
                    compressed.append(msg)
            elif isinstance(msg, SystemMessage):
                # Skip internal loop guidance messages
                if msg.content not in AgentRunner._INTERNAL_SYSTEM_MESSAGES:
                    compressed.append(msg)
            else:
                compressed.append(msg)
        return compressed

    # The most recent tool outputs always stay verbatim during context
    # compression — they are the model's current working set.
    _VERBATIM_TOOL_MESSAGES = 10

    async def _summarized_or_truncated_shares(
        self,
        pairs: List[Tuple[int, str]],
        *,
        target_budget: int,
        model: str,
        timeout_seconds: float,
        context_window: int,
        min_share: int,
    ) -> Dict[int, str]:
        """Compress old tool outputs, preferring an LLM summary over blind cuts.

        When a summary LLM is available the combined outputs are condensed into
        a single labeled summary that lands in the FIRST compressed slot; the
        remaining slots get short re-invocation stubs.  Without a summary LLM
        each output is truncated to a proportional token share, visibly marked
        so the model knows it can re-invoke the same call to restore the data.
        """
        combined = "\n\n---\n\n".join(content for _, content in pairs)
        summarized: Optional[str] = None
        if self._summary_llm_factory is not None:
            try:
                candidate, ok = await summarize_context_if_needed(
                    combined,
                    token_budget=target_budget,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    summary_llm_factory=self._summary_llm_factory,
                    summary_context_window=context_window,
                    usage_sink=self._compression_usage,
                )
                if ok:
                    summarized = candidate
            except Exception as exc:
                logger.warning("Tool-output summary failed; truncating instead: %r", exc)
        result: Dict[int, str] = {}
        if summarized:
            for position, (idx, content) in enumerate(pairs):
                if position == 0:
                    result[idx] = (
                        f"[Compressed working set — LLM summary of {len(pairs)} earlier tool outputs. "
                        f"Re-invoke a call to restore its full output.]\n\n{summarized}"
                    )
                else:
                    result[idx] = (
                        f"[compressed — re-invoke this tool call to restore "
                        f"the full output ({len(content)} chars)]"
                    )
            return result
        combined_tokens = max(1, count_text_tokens(combined, model))
        for idx, content in pairs:
            share = max(min_share, int(target_budget * count_text_tokens(content, model) / combined_tokens))
            kept = _hard_truncate(content, share, model)
            result[idx] = (
                kept
                + (
                    f"\n\n… [compressed — {len(content)} chars → {len(kept)} shown; "
                    f"re-invoke the same tool call to restore the full output]"
                )
                if len(kept) < len(content)
                else content
            )
        return result

    async def _check_and_compress_context(
        self,
        messages: List[BaseMessage],
        *,
        use_fallback: bool,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BaseMessage]:
        """Proactively compress tool outputs if total message tokens approach the context window.

        When the accumulated messages exceed a safe threshold (accounting for
        tool schema tokens and output reservation), older ToolMessages are
        bulk-summarized to free up space.  This prevents 400 errors from the
        LLM when tool outputs accumulate across iterations.
        """
        if self._rolling_history_settings is None:
            return messages

        rh = self._rolling_history_settings
        model = rh.model or "gpt-4o-mini"
        total_tokens = count_messages_tokens(messages, model)

        # Estimate tool schema overhead (bind_tools adds these to the API payload)
        tool_schema_tokens = 0
        if tool_schemas:
            tool_schema_tokens = count_text_tokens(json.dumps(tool_schemas), model)

        safety_margin = int(rh.context_window * rh.safety_margin_ratio)
        threshold = rh.context_window - rh.max_output_tokens - tool_schema_tokens - safety_margin
        threshold = max(threshold, int(rh.context_window * 0.3))  # never go below 30%

        if total_tokens <= threshold:
            return messages

        logger.info(
            "Context compression triggered: %d tokens > %d threshold (context_window=%d)",
            total_tokens, threshold, rh.context_window,
        )

        # Collect ToolMessages belonging to EARLIER iterations only. The most
        # recent batch (after the last AIMessage) just arrived and must reach
        # the model verbatim — compressing it beforehand would silently swap
        # fresh results for truncated stubs.
        last_ai_idx = -1
        for idx, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                last_ai_idx = idx

        tool_indices: List[int] = []
        tool_contents: List[str] = []
        for idx, msg in enumerate(messages):
            if last_ai_idx >= 0 and idx >= last_ai_idx:
                break
            if isinstance(msg, ToolMessage):
                content = msg.content or ""
                if content and not content.startswith("[Error:"):
                    tool_indices.append(idx)
                    tool_contents.append(content)

        if not tool_contents:
            return messages

        # Compress the accumulated old tool outputs down to a shared budget
        combined = "\n\n---\n\n".join(tool_contents)
        combined_tokens = count_text_tokens(combined, model)
        target_budget = max(512, (rh.context_window - rh.max_output_tokens) // 4)

        if combined_tokens <= target_budget:
            return messages

        if use_fallback:
            # Fallback mode has no tool-call protocol to preserve: merge all
            # old tool outputs into a single HumanMessage.
            summarized: Optional[str] = None
            if self._summary_llm_factory is not None:
                try:
                    summarized, _ = await summarize_context_if_needed(
                        combined,
                        token_budget=target_budget,
                        model=model,
                        timeout_seconds=rh.summary_timeout_seconds,
                        summary_llm_factory=self._summary_llm_factory,
                        summary_context_window=rh.context_window,
                        usage_sink=self._compression_usage,
                    )
                except Exception as exc:
                    logger.warning("Context compression failed: %r", exc)
                    summarized = None
            if summarized is None:
                summarized = _hard_truncate(combined, target_budget, model)
            logger.info("Context compression: %d → %d tokens", combined_tokens, count_text_tokens(summarized, model))
            new_messages = []
            replaced = False
            for idx, msg in enumerate(messages):
                if idx in set(tool_indices):
                    if not replaced:
                        new_messages.append(HumanMessage(
                            content=f"[Summarized tool results from earlier iterations]:\n{summarized}"
                        ))
                        replaced = True
                else:
                    new_messages.append(msg)
            return new_messages

        # Native mode: compress each old ToolMessage in place, preserving its
        # real tool_call_id.  Emitting a single ToolMessage with a fabricated
        # id would leave the AIMessage's other tool_calls unanswered and
        # violate the tool-calling protocol on strict APIs.
        # The most recent outputs stay verbatim — they are the model's current
        # working set, and truncating them is what sends agents into re-read
        # loops.  Only older outputs get compressed.
        protected = (
            set(tool_indices[-self._VERBATIM_TOOL_MESSAGES:])
            if len(tool_indices) > self._VERBATIM_TOOL_MESSAGES
            else set()
        )
        compressible: List[Tuple[int, str]] = [
            (idx, content)
            for idx, content in zip(tool_indices, tool_contents)
            if idx not in protected
        ]
        if not compressible:
            return messages
        logger.info(
            "Context compression: compressing %d tool messages in place (%d → ~%d tokens, %d protected)",
            len(compressible), combined_tokens, target_budget, len(protected),
        )
        new_contents = await self._summarized_or_truncated_shares(
            compressible,
            target_budget=target_budget,
            model=model,
            timeout_seconds=rh.summary_timeout_seconds,
            context_window=rh.context_window,
            min_share=96,
        )
        new_messages = []
        for idx, msg in enumerate(messages):
            if idx in new_contents:
                new_messages.append(ToolMessage(
                    content=new_contents[idx],
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                ))
            else:
                new_messages.append(msg)
        return new_messages

    async def _emergency_compress(
        self,
        messages: List[BaseMessage],
        *,
        use_fallback: bool,
    ) -> Optional[List[BaseMessage]]:
        """Emergency compression when a 400 context-length error occurs.

        Aggressively summarizes ALL tool outputs and hard-truncates if needed.
        Returns compressed messages or None if compression is not possible.
        """
        if self._rolling_history_settings is None:
            return None

        rh = self._rolling_history_settings
        model = rh.model or "gpt-4o-mini"

        tool_indices: List[int] = []
        tool_contents: List[str] = []
        for idx, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                content = msg.content or ""
                if content:
                    tool_indices.append(idx)
                    tool_contents.append(content)

        if not tool_contents:
            return None

        combined = "\n\n---\n\n".join(tool_contents)
        combined_tokens = max(1, count_text_tokens(combined, model))
        target_budget = max(256, rh.context_window // 8)

        if not use_fallback:
            # Keep the tool-call protocol intact: shrink every ToolMessage in
            # place instead of emitting one message with a fabricated
            # tool_call_id that matches no tool_call.
            new_contents = await self._summarized_or_truncated_shares(
                list(zip(tool_indices, tool_contents)),
                target_budget=target_budget,
                model=model,
                timeout_seconds=rh.summary_timeout_seconds,
                context_window=rh.context_window,
                min_share=64,
            )
            new_messages: List[BaseMessage] = []
            for idx, msg in enumerate(messages):
                if idx in new_contents:
                    new_messages.append(ToolMessage(
                        content=new_contents[idx],
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, "name", None),
                    ))
                else:
                    new_messages.append(msg)
            logger.info("Emergency compression: compressed %d tool messages in place", len(new_contents))
            return new_messages

        if self._summary_llm_factory is not None:
            try:
                summarized, _ = await summarize_context_if_needed(
                    combined,
                    token_budget=target_budget,
                    model=model,
                    timeout_seconds=rh.summary_timeout_seconds,
                    summary_llm_factory=self._summary_llm_factory,
                    summary_context_window=rh.context_window,
                    usage_sink=self._compression_usage,
                )
            except Exception as exc:
                logger.warning("Emergency compression summarization failed: %r", exc)
                summarized = _hard_truncate(combined, target_budget, model)
        else:
            summarized = _hard_truncate(combined, target_budget, model)

        logger.info("Emergency compression: %d → %d tokens", combined_tokens, count_text_tokens(summarized, model))

        new_messages = []
        replaced = False
        for idx, msg in enumerate(messages):
            if idx in set(tool_indices):
                if not replaced:
                    new_messages.append(HumanMessage(
                        content=f"[Emergency summarized tool results]:\n{summarized}"
                    ))
                    replaced = True
            else:
                new_messages.append(msg)
        return new_messages if replaced else None

    @staticmethod
    def _prepare_final_messages(
        messages: List[BaseMessage],
        raw_system_instructions: str,
        rag_context: Optional[str] = None,
    ) -> Tuple[List[BaseMessage], bool, bool]:
        """Rebuild the conversation for the final answer with a clean system prompt.

        Replaces the first SystemMessage (which contains tool prompts and
        descriptions) with the raw system instructions only.  Keeps user and
        assistant history, compresses tool results, and strips internal loop
        guidance.

        Returns (final_messages, all_tools_failed, has_successful_results).
        """
        compressed: List[BaseMessage] = []
        tool_results: List[str] = []
        error_count = 0
        total_tool_results = 0
        successful_count = 0
        first_system_seen = False

        for msg in messages:
            if isinstance(msg, SystemMessage) and not first_system_seen:
                # Replace the very first system message with the raw instructions
                # Safety: re-render in case placeholders were not replaced upstream
                sys_text = render_system_instructions(raw_system_instructions.strip(), rag_context=rag_context or "")
                # Safety: also replace escaped variant that may come from Markdown editors
                sys_text = sys_text.replace("{{current\\_datetime}}", _format_current_datetime_german())
                if rag_context:
                    if "{{context}}" in sys_text:
                        sys_text = sys_text.replace("{{context}}", rag_context)
                    else:
                        sys_text += f"\n\nRelevant context:\n{rag_context}"
                if sys_text:
                    compressed.append(SystemMessage(content=sys_text))
                first_system_seen = True
                continue

            if isinstance(msg, ToolMessage):
                content = msg.content or ""
                total_tool_results += 1
                if content.startswith("[Error:") or content.startswith("[Tool argument error:"):
                    error_count += 1
                elif content.strip():
                    successful_count += 1
                tool_results.append(content)
            elif isinstance(msg, AIMessage):
                # Keep AIMessages that have content OR carried tool_calls
                if (msg.content and msg.content.strip()) or msg.tool_calls:
                    # For final answer generation, empty tool-call-only
                    # AIMessages can confuse non-reasoning models (e.g. Qwen
                    # via LM Studio). Replace them with a short summary so
                    # the model sees the tool calls as normal text.
                    if not msg.content and msg.tool_calls:
                        tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                        compressed.append(
                            AIMessage(content=f"I called the following tools: {', '.join(tool_names)}")
                        )
                    else:
                        compressed.append(msg)
            elif isinstance(msg, SystemMessage):
                # Skip internal loop guidance messages
                if msg.content not in AgentRunner._INTERNAL_SYSTEM_MESSAGES:
                    compressed.append(msg)
            else:
                compressed.append(msg)

        all_failed = total_tool_results > 0 and error_count == total_tool_results
        has_successful_results = successful_count > 0
        if tool_results:
            results_text = "\n---\n".join(tool_results)
            compressed.append(
                HumanMessage(
                    content=f"Below are the results from tools that were called:\n\n{results_text}"
                )
            )
        return compressed, all_failed, has_successful_results

    def _call_conflict_key(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Optional[Any]:
        """Serialization key for batch calls that mutate the same target.

        A tool may implement ``call_conflict_key(args) -> hashable | None``.
        Calls in one batch sharing ``(tool_name, key)`` run sequentially in
        call order; all other calls run in parallel as before.
        """
        tool_obj = self.registry.get_tool(tool_name)
        fn = getattr(tool_obj, "call_conflict_key", None)
        if not callable(fn):
            return None
        try:
            key = fn(tool_args)
            if key is not None:
                hash(key)  # unhashable keys would break group lookup
        except Exception:
            return None
        return (tool_name, key) if key is not None else None

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
        iteration: int,
        trace: AgentTrace,
    ) -> Tuple[str, ToolMessage, List[str]]:
        """Execute a single tool with validation and structured error handling.

        Returns (display_content, tool_message_for_history, image_data_uris).
        The third element is non-empty for vision tools that render pages —
        the caller appends them as a follow-up HumanMessage with image_url
        blocks (OpenAI-compatible providers reject non-text tool messages).
        """
        record = ToolCallRecord(iteration=iteration, tool_name=tool_name, args=tool_args, started_at=time.time())
        record.call_id = f"{self._trace_id}:{tool_id}"
        if any(existing.call_id == record.call_id for existing in trace.tool_calls):
            record.call_id = f"{record.call_id}:{iteration}:{len(trace.tool_calls) + 1}"
        record.kind = (
            "subagent" if tool_name == "spawn_subagent"
            else "subagent_control" if tool_name in SUBAGENT_COORDINATION_TOOLS
            else "tool"
        )
        trace.tool_calls.append(record)

        # 1. Validation
        tool_schema = None
        tool_obj = self.registry.get_tool(tool_name)
        if tool_obj is not None:
            tool_schema = getattr(tool_obj, "tool_schema", None)

        normalized, validation_error, arg_notes = validate_and_normalize_args(tool_name, tool_args, tool_schema)
        if validation_error:
            if arg_notes:
                validation_error = f"{validation_error} ({'; '.join(arg_notes)})"
            record.ok = False
            record.error = validation_error
            record.ended_at = time.time()
            error_text = f"[Tool argument error: {validation_error}]"
            return error_text, ToolMessage(content=error_text, tool_call_id=tool_id), []

        # 2. Execution — no fixed wall-clock limit. The periodic tool_progress
        # keepalives emitted while this runs carry a `keepalive` flag and do NOT
        # reset the run-level idle watchdog — a genuinely stuck tool that
        # produces no real items is aborted after run_idle_timeout_seconds.
        context: Dict[str, Any] = {}
        if self.conversation_id:
            context["conversation_id"] = self.conversation_id
        if self.agent_memory_store:
            context["agent_memory_store"] = self.agent_memory_store
        context["assistant_name"] = self.assistant_name
        if self._subagent_state is not None:
            context["subagent_state"] = self._subagent_state
            context["tool_call_id"] = tool_id
        if self._todo_state is not None:
            context["todo_state"] = self._todo_state
        if self._workspace_ctx is not None:
            context["agent_workspace_dir"] = str(self._workspace_ctx.get("dir"))
            context["agent_workspace_cfg"] = self._workspace_ctx.get("cfg") or {}
            if self._workspace_ctx.get("conv"):
                context["agent_workspace_conv"] = self._workspace_ctx["conv"]
        injected_queue = getattr(self, "_injected_input_queue", None)
        if injected_queue is not None:
            context["injected_input_queue"] = injected_queue
        input_event = getattr(self, "_input_event", None)
        if input_event is not None:
            context["input_event"] = input_event
        try:
            result = await self.registry.execute(
                tool_name, normalized,
                context=context if context else None,
                assistant_override=self.tool_overrides.get(tool_name),
            )
        except Exception as exc:
            record.ok = False
            record.error = str(exc)
            record.ended_at = time.time()
            error_text = f"[Error: {record.error}]"
            return error_text, ToolMessage(content=error_text, tool_call_id=tool_id), []

        # 3. Result processing
        content = result.get("content", "")
        if result.get("error"):
            record.ok = False
            record.error = result["error"]
            content = f"[Error: {result['error']}]"

        # Surface ignored/mistyped arguments so the model can self-correct;
        # silently dropped args otherwise yield identical results with no clue.
        if arg_notes:
            note_text = f"[Tool argument note: {'; '.join(arg_notes)}]"
            content = f"{content}\n\n{note_text}" if content.strip() else note_text

        truncated = truncate_tool_output(content)
        record.output_chars = len(truncated)
        record.truncated = truncated != content

        # LLM-based summarization for very large tool outputs when rolling
        # history is active.  Only outputs that are a large fraction of the
        # context window are condensed — normal-sized results must reach the
        # model verbatim (e.g. paginated document reads), and the bulk
        # compressor handles accumulated old outputs separately.
        if (
            self._rolling_history_settings is not None
            and self._summary_llm_factory is not None
            and record.ok
            and not truncated.startswith("[Error:")
        ):
            rh = self._rolling_history_settings
            tool_budget = max(512, min(16_000, (rh.context_window - rh.max_output_tokens) // 4))
            token_count = count_text_tokens(truncated, rh.model or "gpt-4o-mini")
            if token_count > tool_budget:
                try:
                    summarized, was_summarized = await summarize_context_if_needed(
                        truncated,
                        token_budget=tool_budget,
                        model=rh.model or "gpt-4o-mini",
                        timeout_seconds=rh.summary_timeout_seconds,
                        summary_llm_factory=self._summary_llm_factory,
                        summary_context_window=rh.context_window,
                        usage_sink=self._compression_usage,
                    )
                    if was_summarized:
                        truncated = summarized + (
                            "\n\n[Note: this tool output was condensed to fit the context window. "
                            "Call the tool again with narrower parameters (smaller range/offset) "
                            "if you need the verbatim text.]"
                        )
                        record.summarized = True
                        record.output_chars = len(truncated)
                except Exception as exc:
                    logger.warning("Tool output summarization failed for %s: %s", tool_name, exc)

        record.output = truncated
        record.metadata = result.get("metadata")
        record.ended_at = time.time()
        # Rendered page images travel separately from the text result — they
        # never enter record.output / traces / SSE payloads.
        images = [u for u in (result.get("images") or []) if isinstance(u, str)][:8]
        return truncated, ToolMessage(content=truncated, tool_call_id=tool_id), images

    async def _execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        iteration: int,
        trace: AgentTrace,
        use_fallback: bool,
    ) -> Tuple[List[BaseMessage], int, List[ToolMessage]]:
        """Execute a batch of tool calls in parallel and return messages to append.

        Returns (messages_to_append, num_executed, tool_messages_for_observability).
        """
        calls_to_run: List[Tuple[str, Dict[str, Any], str]] = []
        messages: List[BaseMessage] = []
        tool_messages: List[ToolMessage] = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})
            tool_id = call.get("id", f"call_{len(calls_to_run)}")

            if tool_name not in self.tool_names:
                error_text = f"[Error: Tool '{tool_name}' not allowed]"
                if use_fallback:
                    messages.append(HumanMessage(content=error_text))
                else:
                    messages.append(ToolMessage(content=error_text, tool_call_id=tool_id))
                tool_messages.append(ToolMessage(content=error_text, tool_call_id=tool_id))
                record = ToolCallRecord(
                    iteration=iteration, tool_name=tool_name, args=tool_args,
                    started_at=time.time(), ended_at=time.time(), ok=False,
                    error=error_text,
                )
                trace.tool_calls.append(record)
                continue

            calls_to_run.append((tool_name, tool_args, tool_id))

        if not calls_to_run:
            return messages, 0, tool_messages

        # Execute calls in parallel — except calls that mutate the same
        # target (e.g. two edits on the same file): those serialize in call
        # order so each sees the previous call's result instead of racing a
        # read-modify-write cycle where only the last write survives.
        async def _run_one(name: str, args: Dict[str, Any], tid: str) -> Tuple[str, ToolMessage, List[str]]:
            logger.info("Calling tool: %s(args=%r)", name, args)
            return await self._execute_tool(name, args, tid, iteration, trace)

        groups: List[List[int]] = []
        group_slots: Dict[Any, List[int]] = {}
        for idx, (name, args, _tid) in enumerate(calls_to_run):
            key = self._call_conflict_key(name, args)
            if key is None:
                groups.append([idx])
            else:
                slot = group_slots.get(key)
                if slot is None:
                    slot = []
                    group_slots[key] = slot
                    groups.append(slot)
                slot.append(idx)

        async def _run_group(indices: List[int]) -> Dict[int, Any]:
            out: Dict[int, Any] = {}
            for i in indices:
                gname, gargs, gtid = calls_to_run[i]
                try:
                    out[i] = await _run_one(gname, gargs, gtid)
                except Exception as exc:
                    out[i] = exc
            return out

        group_results = await asyncio.gather(
            *[_run_group(g) for g in groups], return_exceptions=True
        )
        results: List[Any] = [None] * len(calls_to_run)
        for gres in group_results:
            if isinstance(gres, dict):
                for i, val in gres.items():
                    results[i] = val
        for idx, res in enumerate(results):
            if res is None:
                results[idx] = RuntimeError("tool call did not produce a result")

        result_texts: List[str] = []
        batch_images: List[str] = []  # rendered images produced by THIS batch only
        for idx, (name, args, tid) in enumerate(calls_to_run):
            res = results[idx]
            if isinstance(res, Exception):
                error_text = f"[Error: {res}]"
                result_texts.append(f"Tool '{name}': {error_text}")
                tool_messages.append(ToolMessage(content=error_text, tool_call_id=tid))
                if not use_fallback:
                    messages.append(ToolMessage(content=error_text, tool_call_id=tid))
                continue
            display_content, tool_message, images = res
            if name == RAG_TOOL_NAME:
                self._rag_tool_tokens += count_tokens(
                    stringify_message_content(display_content) or "")
            if images:
                batch_images.extend(images)
            result_texts.append(f"Tool '{name}':\n{display_content}")
            tool_messages.append(tool_message)
            if use_fallback:
                # Fallback mode has no tool protocol — the result rides in a
                # HumanMessage, and rendered images merge into the same message
                # as content blocks.
                if images:
                    messages.append(self._image_result_message(
                        name, args, images, prefix=f"Tool '{name}' result:\n{display_content}"))
                else:
                    messages.append(HumanMessage(
                        content=f"Tool '{name}' result:\n{display_content}",
                        additional_kwargs={"run_tool_result": True}))
            else:
                messages.append(tool_message)
                if images:
                    # Images cannot ride inside a ToolMessage on strict OpenAI
                    # wire formats — a follow-up HumanMessage carries them.
                    messages.append(self._image_result_message(name, args, images))

        if use_fallback and len(calls_to_run) > 1:
            # Aggregate fallback results into a single HumanMessage. The filter
            # must inspect stringified content — image-carrying results have
            # list content where str.startswith would crash. Old "Tool '"
            # messages (incl. their images) are intentionally squashed; only
            # THIS batch's images are re-attached to the aggregated message.
            aggregated = "\n\n".join(result_texts)
            messages = [msg for msg in messages if not (
                isinstance(msg, HumanMessage)
                and stringify_message_content(msg.content).startswith("Tool '")
            )]
            if batch_images:
                content: Any = [{"type": "text", "text": f"Tool results:\n\n{aggregated}"}]
                content.extend(
                    {"type": "image_url", "image_url": {"url": u}} for u in batch_images[:8]
                )
                messages.append(HumanMessage(
                    content=content,
                    additional_kwargs={"workspace_images": True, "run_tool_result": True}))
            else:
                messages.append(HumanMessage(
                    content=f"Tool results:\n\n{aggregated}",
                    additional_kwargs={"run_tool_result": True}))

        return messages, len(calls_to_run), tool_messages

    @staticmethod
    def _image_result_message(
        tool_name: str,
        args: Dict[str, Any],
        images: List[str],
        *,
        prefix: Optional[str] = None,
    ) -> HumanMessage:
        """HumanMessage carrying rendered page-image blocks.

        ToolMessages cannot hold image content on strict OpenAI-compatible
        wire formats, so rendered pages are injected as a follow-up user
        message. ``additional_kwargs['workspace_images']`` tags the message
        for ``_dedup_run_content`` — it is not serialized to the wire.
        """
        path_hint = str((args or {}).get("path") or "file")
        content: List[Dict[str, Any]] = []
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.append({
            "type": "text",
            "text": (
                f"[{len(images)} rendered page image(s) of '{path_hint}' via tool "
                f"'{tool_name}'. Base your answer on these images; call the tool "
                "again with a different 'pages' range to see other pages.]"
            ),
        })
        content.extend({"type": "image_url", "image_url": {"url": u}} for u in images)
        return HumanMessage(content=content, additional_kwargs={"workspace_images": True})

    # At most this many unique images are kept live in the run context; older
    # ones are replaced by reference stubs (tagged tool renders first, user
    # uploads last).
    _MAX_CTX_IMAGES = 16
    # Only tool outputs longer than this are candidates for identical-content
    # dedup — short outputs cost less than a stub roundtrip.
    _DEDUP_TEXT_MIN_CHARS = 1000

    def _dedup_run_content(self, messages: List[BaseMessage]) -> None:
        """Exactly-once semantics for identical images and large tool outputs.

        Per unique image (keyed by data URI) and per unique large tool output
        (keyed by sha256 of stripped text) only the LAST occurrence keeps its
        payload; earlier duplicates become ``#img-N``/``#res-N`` reference
        stubs pointing at that latest copy. Beyond ``_MAX_CTX_IMAGES`` unique
        live images the oldest are evicted — tagged tool renders before
        untagged user uploads.

        Called at the top of every iteration. Idempotent: stubs are plain
        text below the dedup threshold and contain no ``image_url`` parts, so
        a second pass is a no-op. Kept copies are never labelled — mutating
        them would change their hash and break future dedup; references
        resolve positionally ("the latest result").
        """
        # Pass 1+2 — collect image occurrences, dedup identical URIs.
        occurrences: Dict[str, List[Tuple[int, int, bool]]] = {}
        for mi, m in enumerate(messages):
            if not (isinstance(m, HumanMessage) and isinstance(m.content, list)):
                continue
            tagged = bool((m.additional_kwargs or {}).get("workspace_images"))
            for pi, part in enumerate(m.content):
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    continue
                iu = part.get("image_url")
                uri = str(iu.get("url") or "") if isinstance(iu, dict) else ""
                if not uri:
                    continue
                img_id = self._media_ids.setdefault(uri, f"img-{len(self._media_ids) + 1}")
                occurrences.setdefault(img_id, []).append((mi, pi, tagged))

        live: List[Tuple[int, int, str, bool]] = []
        for img_id, occs in occurrences.items():
            keep = max(occs, key=lambda o: (o[0], o[1]))
            live.append((keep[0], keep[1], img_id, keep[2]))
            for mi, pi, _ in occs:
                if (mi, pi) != (keep[0], keep[1]):
                    messages[mi].content[pi] = {
                        "type": "text",
                        "text": (
                            f"[#{img_id} — identical image omitted; it is shown "
                            "once in the latest image result below]"
                        ),
                    }

        # Pass 3 — budget: evict oldest unique images, tagged renders first.
        if len(live) > self._MAX_CTX_IMAGES:
            live.sort(key=lambda t: (not t[3], t[0], t[1]))
            for mi, pi, img_id, _ in live[: len(live) - self._MAX_CTX_IMAGES]:
                messages[mi].content[pi] = {
                    "type": "text",
                    "text": (
                        f"[#{img_id} — image omitted to save context; call the "
                        "tool again with a 'pages' range to re-see it]"
                    ),
                }

        # Pass 4 — identical large tool outputs. ToolMessages carry no `name`
        # here, so resolve the tool via the AIMessage tool_calls mapping.
        call_names: Dict[str, str] = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        call_names[tc["id"]] = tc.get("name") or "tool"
        results: Dict[str, List[Tuple[int, str]]] = {}
        for mi, m in enumerate(messages):
            name = "tool"
            if isinstance(m, ToolMessage) and isinstance(m.content, str):
                text = m.content
                name = call_names.get(m.tool_call_id) or name
            elif (
                isinstance(m, HumanMessage)
                and isinstance(m.content, str)
                and (m.additional_kwargs or {}).get("run_tool_result")
            ):
                text = m.content
            else:
                continue
            if len(text) <= self._DEDUP_TEXT_MIN_CHARS:
                continue
            h = hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()
            res_id = self._result_ids.setdefault(h, f"res-{len(self._result_ids) + 1}")
            results.setdefault(res_id, []).append((mi, name))
        for res_id, occs in results.items():
            if len(occs) < 2:
                continue
            keep_name = occs[-1][1]
            for mi, _ in occs[:-1]:
                messages[mi].content = (
                    f"[#{res_id} — identical output omitted; see the latest "
                    f"'{keep_name}' result below]"
                )

    def _observer_snapshot(
        self,
        iteration: int,
        trace: "AgentTrace",
        tool_calls_executed: int,
        input_text: str,
    ) -> ObserverRunSnapshot:
        """Build a bounded point-in-time view of the run for the observer."""
        recent: List[Dict[str, Any]] = []
        for rec in trace.tool_calls[-8:]:
            recent.append(
                {
                    "iteration": rec.iteration,
                    "tool_name": rec.tool_name,
                    "args": rec.args,
                    "ok": rec.ok,
                    "error": rec.error,
                    "output_preview": (rec.output or "")[:200],
                    "duration_ms": round((rec.ended_at - rec.started_at) * 1000, 1) if rec.ended_at else None,
                    "skipped": bool(rec.call_id and str(rec.call_id).endswith(":skipped")),
                }
            )
        todo_state = self._todo_state
        tasks = todo_state.tasks if todo_state is not None else []
        return ObserverRunSnapshot(
            iteration=iteration,
            max_iterations=self.max_iterations,
            tool_calls_executed=tool_calls_executed,
            max_tool_calls=self.max_tool_calls,
            elapsed_s=time.time() - trace.started_at,
            recent_records=recent,
            todo_version=todo_state.version if todo_state is not None else 0,
            todo_render=todo_state.render() if todo_state is not None else "",
            open_tasks=sum(1 for t in tasks if t.get("status") in OPEN_STATUSES),
            last_ai_text=self._last_ai_text,
            reasoning_tail=self._last_reasoning_tail,
            input_text=input_text,
            system_excerpt=(self._raw_system_instructions or "")[:1500],
            injected_since_last=self._injected_since_observer,
            todo_available=TODO_TOOL_NAME in self.tool_names,
        )

    def todo_list_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the current todo list snapshot, or None if no list was created."""
        state = getattr(self, "_todo_state", None)
        if state is None or state.version == 0:
            return None
        return state.snapshot()

    def _open_todo_tasks(self) -> List[Dict[str, Any]]:
        """Todo tasks that are not in a terminal state (completed/skipped/removed)."""
        state = self._todo_state
        if state is None:
            return []
        return [t for t in state.tasks if t.get("status") in OPEN_STATUSES]

    def _todo_gate_open_tasks(self, iteration: int) -> List[Dict[str, Any]]:
        """Open todo tasks blocking finalization — empty list when allowed.

        A run may only end with a fully worked-through plan: every task must
        be completed, skipped or removed. Finalization stays allowed when the
        list cannot be fixed (tool unavailable), the reminder budget is spent
        (forced finalization, leftovers swept to ``skipped``), or no further
        iteration exists (``max_iterations`` reached — the forced-answer path
        below handles it).
        """
        if TODO_TOOL_NAME not in self.tool_names:
            return []
        if self._todo_gate_reminders >= MAX_TODO_FINALIZE_REMINDERS:
            return []
        if iteration >= self.max_iterations:
            return []
        return self._open_todo_tasks()

    @staticmethod
    def _todo_gate_message(open_tasks: List[Dict[str, Any]]) -> str:
        """HumanMessage telling the agent which tasks still block finalization."""
        lines = [f"[Finalization blocked — {len(open_tasks)} open todo task(s)]"]
        for task in open_tasks:
            lines.append(f"- {task['id']} [{task['status']}] {task['title']}")
        lines.append(
            "This run cannot end while todo tasks are open. Finish the remaining "
            "work now, or mark each task \"completed\"/\"skipped\" via agent_todo "
            "(or remove it). Only then give your final answer."
        )
        return "\n".join(lines)

    @staticmethod
    def _estimate_prompt_tokens(messages: Sequence[BaseMessage]) -> int:
        """Estimate prompt tokens for a call whose provider reported no usage.

        Counts every message actually sent — including ToolMessages carrying
        tool outputs — plus tool_call arguments and a small per-message
        format overhead. Image parts are skipped by ``stringify_message_content``.
        """
        total = 0
        for m in messages:
            total += count_tokens(stringify_message_content(getattr(m, "content", "") or ""))
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                try:
                    total += count_tokens(json.dumps(tool_calls, default=str))
                except Exception:
                    pass
            total += 4  # per-message format overhead
        return total

    def _record_iteration_usage(
        self,
        result: Any,
        sent_messages: Sequence[BaseMessage],
        response_ms: float,
    ) -> None:
        """Record one LLM call's usage, estimating fields the provider omitted.

        Providers that stream without usage reporting (Ollama, vLLM, OpenAI
        without ``stream_options.include_usage``) would otherwise vanish from
        the totals entirely — including prompts that carried tool outputs.
        Always appends so ``iterations`` and ``_final_answer_usage`` stay exact.
        """
        usage = extract_usage_metadata(result) or {}
        estimated = False
        if not usage.get("prompt_tokens"):
            usage["prompt_tokens"] = self._estimate_prompt_tokens(sent_messages)
            estimated = True
        if not usage.get("completion_tokens"):
            generated = stringify_message_content(getattr(result, "content", "") or "")
            tool_calls = getattr(result, "tool_calls", None)
            if tool_calls:
                try:
                    generated += json.dumps(tool_calls, default=str)
                except Exception:
                    pass
            reasoning = _extract_reasoning_from_message(result) or ""
            usage["completion_tokens"] = count_tokens((generated or "") + (reasoning or ""))
            estimated = True
        if not usage.get("total_tokens"):
            usage["total_tokens"] = (
                (usage.get("prompt_tokens") or 0)
                + (usage.get("completion_tokens") or 0)
                + (usage.get("cache_tokens") or 0)
            )
        if estimated:
            usage["_estimated"] = True
        usage["_response_time_ms"] = response_ms
        self._iteration_usages.append(usage)

    def _live_usage_event(self) -> Dict[str, Any]:
        """Cumulative usage snapshot for the live token counter — emitted
        after each completed LLM call so the UI can show tokens growing
        mid-run. Covers iteration calls plus compression and observer usage
        so the counter matches the final totals as closely as possible.
        """
        prompt = completion = total = reasoning = 0
        estimated = False
        per_call: List[int] = []
        for u in getattr(self, "_iteration_usages", None) or []:
            prompt += u.get("prompt_tokens") or 0
            completion += u.get("completion_tokens") or 0
            total += u.get("total_tokens") or 0
            reasoning += u.get("reasoning_tokens") or 0
            estimated = estimated or bool(u.get("_estimated"))
            per_call.append(int(u.get("completion_tokens") or 0))
        compression_calls = 0
        for u in getattr(self, "_compression_usage", None) or []:
            prompt += u.get("prompt_tokens") or 0
            completion += u.get("completion_tokens") or 0
            total += u.get("total_tokens") or 0
            reasoning += u.get("reasoning_tokens") or 0
            compression_calls += 1
        observer = getattr(self, "_observer", None)
        obs = getattr(observer, "usage", None) or {}
        if obs:
            prompt += int(obs.get("prompt_tokens") or 0)
            completion += int(obs.get("completion_tokens") or 0)
            total += int(obs.get("total_tokens") or 0)
            reasoning += int(obs.get("reasoning_tokens") or 0)
        obs_calls = int(getattr(observer, "checks_used", 0) or 0)
        # Finished subagents report their usage onto the shared manager —
        # include it so the counter tracks the same totals the final badge
        # will show instead of jumping when the trailer lands.
        sub_state = getattr(self, "_subagent_state", None)
        manager = (
            getattr(sub_state, "manager", None)
            if sub_state is not None and getattr(sub_state, "is_root", False)
            else None
        )
        if manager is not None:
            try:
                nested = manager.usage_totals()
                if isinstance(nested, dict):
                    prompt += int(nested.get("prompt_tokens") or 0)
                    completion += int(nested.get("completion_tokens") or 0)
                    total += int(nested.get("total_tokens") or 0)
                    reasoning += int(nested.get("reasoning_tokens") or 0)
                    estimated = estimated or bool(nested.get("estimated_iterations"))
            except Exception:
                pass
        payload: Dict[str, Any] = {
            "llm_calls": len(getattr(self, "_iteration_usages", None) or []) + compression_calls + obs_calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "total_tokens": total,
            "per_call_completion": per_call[-40:],
        }
        if estimated:
            payload["estimated"] = True
        return {"live_usage": payload}

    def _aggregate_iteration_usages(
        self, iteration_usages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Aggregate per-iteration usage metadata into a single summary.

        Sums prompt/completion/total/reasoning tokens across all iterations.
        Averages tokens_per_second per iteration to exclude tool execution time.
        """
        if not iteration_usages:
            return None

        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        total_reasoning = 0
        total_cache = 0
        iter_completion = 0
        gen_ms = 0.0
        est_count = 0

        for usage in iteration_usages:
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            rt = usage.get("reasoning_tokens")
            ct_cache = usage.get("cache_tokens")
            resp_ms = usage.get("_response_time_ms", 0)

            if pt is not None:
                total_prompt += pt
            if ct is not None:
                total_completion += ct
                iter_completion += ct
            if tt is not None:
                total_tokens += tt
            if rt is not None:
                total_reasoning += rt
            if ct_cache is not None:
                total_cache += ct_cache
            if resp_ms:
                gen_ms += resp_ms
            if usage.get("_estimated"):
                est_count += 1

        # Observer and context-compression LLM calls are part of this run's
        # consumption but never land in _iteration_usages — merge their usage
        # into the totals so "total" really covers everything the agent did.
        observer_usage = getattr(getattr(self, "_observer", None), "usage", None) or {}
        for usage in getattr(self, "_compression_usage", None) or []:
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            tt = usage.get("total_tokens") or 0
            rt = usage.get("reasoning_tokens") or 0
            cc = usage.get("cache_tokens") or 0
            total_prompt += pt
            total_completion += ct
            total_tokens += tt
            total_reasoning += rt
            total_cache += cc
        compression_tokens = sum(
            (u.get("total_tokens") or ((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)))
            for u in (getattr(self, "_compression_usage", None) or [])
        )
        compression_calls = len(getattr(self, "_compression_usage", None) or [])
        obs_prompt = int(observer_usage.get("prompt_tokens") or 0)
        obs_completion = int(observer_usage.get("completion_tokens") or 0)
        obs_total = int(observer_usage.get("total_tokens") or 0)
        obs_reasoning = int(observer_usage.get("reasoning_tokens") or 0)
        total_prompt += obs_prompt
        total_completion += obs_completion
        total_tokens += obs_total
        total_reasoning += obs_reasoning
        observer_tokens = obs_prompt + obs_completion if (obs_prompt or obs_completion) else obs_total

        result: Dict[str, Any] = {"iterations": len(iteration_usages)}
        if total_prompt > 0:
            result["prompt_tokens"] = total_prompt
        if total_completion > 0:
            result["completion_tokens"] = total_completion
        if total_tokens > 0:
            result["total_tokens"] = total_tokens
        if total_reasoning > 0:
            result["reasoning_tokens"] = total_reasoning
        if total_cache > 0:
            result["cache_tokens"] = total_cache
        if observer_tokens:
            result["observer_tokens"] = observer_tokens
        obs_call_count = int(getattr(getattr(self, "_observer", None), "checks_used", 0) or 0)
        if obs_call_count:
            result["observer_calls"] = obs_call_count
        if compression_tokens:
            result["compression_tokens"] = compression_tokens
        if compression_calls:
            result["compression_calls"] = compression_calls
        # Output generation rate over LLM-active time only — tool execution,
        # waiting between calls and untimed observer/compression calls must
        # not dilute the reported speed. The >=1ms floor guards against
        # latency-free iteration records producing absurd rates.
        if iter_completion > 0 and gen_ms >= 1:
            result["tokens_per_second"] = iter_completion / (gen_ms / 1000.0)
        if gen_ms >= 1:
            result["llm_generation_ms"] = gen_ms
        if est_count:
            result["estimated_iterations"] = est_count
        # Per-call series for the metrics badge's mini chart — bounded so the
        # persisted payload stays small on very long runs.
        if iteration_usages:
            result["per_call"] = [
                {
                    "p": u.get("prompt_tokens") or 0,
                    "c": u.get("completion_tokens") or 0,
                    "r": u.get("reasoning_tokens") or 0,
                    "ms": round(u.get("_response_time_ms") or 0, 1),
                }
                for u in iteration_usages
            ][-40:]

        return result if result else None

    async def run(
        self,
        llm: Any,
        input_text: str,
        chat_history: Sequence[BaseMessage],
        system_instructions: str,
        rag_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
    ) -> Dict[str, Any]:
        """Execute the agent loop and return the final result dict.

        Thin wrapper around ``run_streaming`` that aggregates chunks so the
        non-streaming and streaming paths share identical logic.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        agent_agen = self.run_streaming(
            llm, input_text, chat_history, system_instructions, rag_context=rag_context,
            conversation_id=conversation_id, multimodal_input=multimodal_input,
        )
        try:
            async for chunk in agent_agen:
                content_parts.append(chunk.get("content", ""))
                reasoning_parts.append(chunk.get("reasoning", ""))
        finally:
            await agent_agen.aclose()

        answer = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        trace = getattr(self, "_last_trace", None)
        trace_dict = trace.to_dict() if trace is not None else {}
        usage_meta = self._aggregate_iteration_usages(self._iteration_usages)
        if usage_meta is not None:
            usage_meta["tool_result_tokens"] = self._tool_result_tokens
            usage_meta["rag_tool_tokens"] = self._rag_tool_tokens
            if self._final_answer_usage is not None:
                usage_meta["final_answer_usage"] = self._final_answer_usage
        return {
            "answer": answer,
            "agent_trace": trace_dict,
            "reasoning": reasoning,
            "agent_usage_metadata": usage_meta,
            "todo_list": self.todo_list_snapshot(),
        }

    async def run_streaming(
        self,
        llm: Any,
        input_text: str,
        chat_history: Sequence[BaseMessage],
        system_instructions: str,
        rag_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        state = self._subagent_state
        if state is None or not state.is_root:
            inner_agen = self._run_streaming_inner(
                llm, input_text, chat_history, system_instructions,
                rag_context=rag_context, conversation_id=conversation_id,
                multimodal_input=multimodal_input,
            )
            try:
                async for chunk in inner_agen:
                    yield chunk
            finally:
                # Close the inner generator in-band — its finally blocks do
                # cleanup that must not depend on asyncgen finalization.
                try:
                    await inner_agen.aclose()
                finally:
                    observer = getattr(self, "_observer", None)
                    if observer is not None:
                        await observer.aclose()
                    trace = getattr(self, "_last_trace", None)
                    if trace is not None and state is not None:
                        state.manager.backfill_tool_call_records(trace.tool_calls)
            return

        output_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def pump() -> None:
            inner_agen = self._run_streaming_inner(
                llm, input_text, chat_history, system_instructions,
                rag_context=rag_context, conversation_id=conversation_id,
                multimodal_input=multimodal_input,
            )
            try:
                async for item in inner_agen:
                    await output_queue.put(("chunk", item))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await output_queue.put(("error", exc))
            finally:
                try:
                    # Close the abandoned inner generator in-band (cancel of
                    # pump_task leaves it suspended at a yield otherwise).
                    await inner_agen.aclose()
                finally:
                    await output_queue.put(("end", None))

        pump_task = asyncio.create_task(pump())
        ended = False
        try:
            while not ended:
                chunk_get = asyncio.create_task(output_queue.get())
                event_get = asyncio.create_task(state.manager.event_queue.get())
                done, pending = await asyncio.wait(
                    {chunk_get, event_get},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    source = "chunk" if task is chunk_get else "event"
                    kind, value = task.result() if source == "chunk" else ("event", task.result())
                    if kind == "end":
                        ended = True
                    elif kind == "error":
                        raise value
                    elif source == "event":
                        yield {
                            "content": "",
                            "reasoning": "",
                            "subagent": value,
                            **(
                                {"keepalive": True}
                                if isinstance(value, dict) and value.get("keepalive")
                                else {}
                            ),
                        }
                    else:
                        yield value
            await state.manager.shutdown()
            trace = getattr(self, "_last_trace", None)
            if trace is not None:
                state.manager.backfill_tool_call_records(trace.tool_calls)
                spawn_records = [
                    rec for rec in trace.tool_calls
                    if rec.tool_name == "spawn_subagent"
                ]
                if spawn_records:
                    yield {
                        "content": "",
                        "reasoning": "",
                        "tool_calls": [
                            {
                                "id": rec.call_id,
                                "tool_name": rec.tool_name,
                                "iteration": rec.iteration,
                                "kind": rec.kind,
                                "args": rec.args,
                                "ok": rec.ok,
                                "error": rec.error,
                                "duration_ms": round((rec.ended_at - rec.started_at) * 1000, 1) if rec.ended_at else None,
                                "output": rec.output,
                                "metadata": rec.metadata,
                            }
                            for rec in spawn_records
                        ],
                    }
            while not state.manager.event_queue.empty():
                event = state.manager.event_queue.get_nowait()
                yield {"content": "", "reasoning": "", "subagent": event}
        finally:
            if not pump_task.done():
                pump_task.cancel()
            await asyncio.gather(pump_task, return_exceptions=True)
            observer = getattr(self, "_observer", None)
            if observer is not None:
                await observer.aclose()
            await state.manager.shutdown()

    def _drain_injected(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Pop queued user inputs and append them to the running message list.

        Returns ``{"text", "source"}`` dicts so the caller can announce them
        on the stream and tag their origin (user steer, parent/user nudge).
        The input event is cleared BEFORE draining: an input racing between
        the clear and a get_nowait re-sets it, so a subsequent wait wakes
        spuriously at worst — a lost wake is impossible. Items are only
        popped when an LLM iteration actually follows — the finalization
        peek checks ``.empty()`` first, so messages left in the queue after
        the run ends surface as ``unconsumed_inputs``.
        """
        drained: List[Dict[str, Any]] = []
        queue = self._injected_input_queue
        if queue is None:
            return drained
        input_event = getattr(self, "_input_event", None)
        if input_event is not None:
            input_event.clear()
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not isinstance(item, dict):
                continue
            content = item.get("content", item.get("text", ""))
            if content is None:
                continue
            messages.append(HumanMessage(content=content))
            text = item.get("text")
            drained.append(
                {
                    "text": text if isinstance(text, str) and text else str(content),
                    "source": item.get("source"),
                }
            )
        return drained

    async def _run_streaming_inner(
        self,
        llm: Any,
        input_text: str,
        chat_history: Sequence[BaseMessage],
        system_instructions: str,
        rag_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Wrapper around the iteration loop that runs the end-of-run epilogue.

        On a NORMAL finish the epilogue still yields (the swept todo
        snapshot). On aclose/cancel no chunks may be emitted anymore, so the
        plan is swept silently — ``todo_list_snapshot`` still reflects it for
        the run result.
        """
        completed_normally = False
        try:
            async for chunk in self._agent_iteration_loop(
                llm, input_text, chat_history, system_instructions,
                rag_context=rag_context, conversation_id=conversation_id,
                multimodal_input=multimodal_input,
            ):
                yield chunk
            completed_normally = True
        finally:
            if completed_normally:
                async for event in self._run_final_epilogue():
                    yield event
            else:
                await self._sweep_unfinished_todos()

    async def _run_final_epilogue(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Sweep leftover open todos and emit the final list snapshot."""
        await self._sweep_unfinished_todos()
        if (
            self._todo_state is not None
            and self._todo_state.version != self._emitted_todo_version
        ):
            self._emitted_todo_version = self._todo_state.version
            yield {"content": "", "reasoning": "", "todo_list": self._todo_state.snapshot()}

    async def _sweep_unfinished_todos(self) -> None:
        """Mark leftover open tasks skipped — no yields (safe on cancel)."""
        state = getattr(self, "_todo_state", None)
        if state is not None:
            try:
                state.mark_unfinished_skipped()
            except Exception:
                logger.debug("Todo sweep failed", exc_info=True)

    async def _agent_iteration_loop(
        self,
        llm: Any,
        input_text: str,
        chat_history: Sequence[BaseMessage],
        system_instructions: str,
        rag_context: Optional[str] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming agent loop.

        Each iteration uses ``astream`` so reasoning content is emitted in
        real time while the actual response content is buffered.  Tool calls
        are parsed only after the stream of an iteration completes.
        """
        # Override instance conversation_id if explicitly passed
        if conversation_id is not None:
            self.conversation_id = conversation_id
        trace = AgentTrace()
        self._last_trace = trace
        self._trace_id = uuid4().hex[:8]
        self._auto_collected_once = False
        self._todo_state = TodoListState()
        self._emitted_todo_version = -1
        self._todo_gate_reminders = 0
        self._last_ai_text = ""
        self._last_reasoning_tail = ""
        self._injected_since_observer = 0
        if self.supports_native_tool_calls is not None:
            use_fallback = not self.supports_native_tool_calls
        else:
            use_fallback = not _has_bind_tools(llm)
        trace.fallback_mode = use_fallback

        await self.registry.refresh_schemas()
        # Agent memory is injected per-iteration in _refresh_system_prompt so
        # mid-run writes become visible on the very next turn — see below.
        enhanced_instructions = system_instructions
        tool_schemas = self.registry.tool_schemas(self.tool_names, tool_overrides=self.tool_overrides)
        messages = self._build_messages(enhanced_instructions, chat_history, input_text, fallback_mode=use_fallback, tool_schemas=tool_schemas, rag_context=rag_context, multimodal_input=multimodal_input)
        self._raw_system_instructions = system_instructions
        self._rag_context = rag_context
        llm_with_tools = llm.bind_tools(tool_schemas) if (tool_schemas and not use_fallback) else llm
        tool_calls_executed = 0
        flat_dict_mapping = _build_flat_dict_mapping(self.registry, self.tool_names)
        self._iteration_usages: List[Dict[str, Any]] = []
        self._compression_usage: List[Dict[str, Any]] = []
        self._tool_result_tokens = 0
        self._rag_tool_tokens = 0
        self._final_answer_usage: Optional[Dict[str, Any]] = None

        for iteration in range(1, self.max_iterations + 1):
            trace.iterations = iteration
            self._dedup_run_content(messages)
            logger.debug("Agent streaming iteration %d/%d (fallback=%s)", iteration, self.max_iterations, use_fallback)

            # User messages injected mid-run become visible to the very next
            # LLM call — the agent answers them as part of this turn.
            drained_inputs = self._drain_injected(messages)
            self._injected_since_observer += len(drained_inputs)
            for injected in drained_inputs:
                yield {"content": "", "reasoning": "", "injected_user_message": injected["text"], "injected_source": injected.get("source")}

            # Run observer: apply a finished check (todo ops + optional nudge)
            # and maybe start the next asynchronous check. The observer can
            # only act at iteration boundaries — inside a long tool call it
            # merely watches.
            observer = self._observer
            if observer is not None:
                try:
                    decision = observer.collect()
                except Exception:
                    decision = None
                if decision is not None:
                    try:
                        obs_events, nudge = observer.apply_decision(decision, self._todo_state)
                    except Exception:
                        obs_events, nudge = [], None
                    for ev in obs_events:
                        trace.observer_events.append(ev)
                        yield {"content": "", "reasoning": "", "observer": ev}
                    if nudge:
                        messages.append(HumanMessage(content=nudge))
                    if (
                        self._todo_state is not None
                        and self._todo_state.version != self._emitted_todo_version
                    ):
                        self._emitted_todo_version = self._todo_state.version
                        yield {"content": "", "reasoning": "", "todo_list": self._todo_state.snapshot()}
                try:
                    started = observer.tick(
                        self._observer_snapshot(iteration, trace, tool_calls_executed, input_text)
                    )
                except Exception:
                    started = None
                if started is not None:
                    self._injected_since_observer = 0
                    trace.observer_events.append(started)
                    yield {"content": "", "reasoning": "", "observer": started}

            # Clean up SystemMessages before every LLM call so strict chat
            # templates (Qwen, Llama, Mistral via LM Studio) never see
            # a SystemMessage after HumanMessage / AIMessage.
            memory_text: Optional[str] = None
            if self.agent_memory_enabled and self.agent_memory_store and self.conversation_id:
                try:
                    memory_text = await self.agent_memory_store.get_memory(
                        self.conversation_id, self.assistant_name
                    )
                except Exception:
                    logger.debug(
                        "Failed to load agent memory for conversation %s",
                        self.conversation_id, exc_info=True,
                    )
            messages = self._refresh_system_prompt(
                messages,
                system_instructions=enhanced_instructions,
                rag_context=rag_context,
                fallback_mode=use_fallback,
                tool_schemas=tool_schemas,
                memory=memory_text,
            )

            # Proactively compress tool outputs if approaching context window limit
            messages = await self._check_and_compress_context(messages, use_fallback=use_fallback, tool_schemas=tool_schemas if not use_fallback else None)

            # Diagnose-Logging: messages-Liste vor dieser Iteration
            _msg_summary = []
            for _m in messages:
                _type = type(_m).__name__
                _content_preview = (getattr(_m, "content", "") or "")[:60]
                if isinstance(_m, ToolMessage):
                    _msg_summary.append(f"{_type}(tool_call_id={getattr(_m, 'tool_call_id', 'none')}, content={_content_preview!r})")
                elif isinstance(_m, AIMessage):
                    _tc = getattr(_m, "tool_calls", [])
                    _msg_summary.append(f"{_type}(content={_content_preview!r}, tool_calls={len(_tc)})")
                else:
                    _msg_summary.append(f"{_type}(content={_content_preview!r})")
            logger.debug("Agent messages before iteration %d: %s", iteration, " | ".join(_msg_summary))

            buffered_content = ""
            accumulated_reasoning = ""
            full_chunk = None

            # Notify client that a new iteration started. ``agent_iteration``
            # carries the counter as structured data so consumers don't have
            # to parse the prose marker (which model reasoning can spoof).
            yield {
                "content": "",
                "reasoning": f"\n\n**Iteration {iteration}...**\n\n",
                "agent_iteration": iteration,
            }

            iteration_start = time.perf_counter()
            try:
                async for chunk in idle_bounded_agen(llm_with_tools.astream(messages), self._idle_seconds):
                    # Accumulate chunks for tool call reconstruction
                    if full_chunk is None:
                        full_chunk = chunk
                    else:
                        full_chunk += chunk

                    # Stream reasoning in real time as it arrives
                    reasoning = _extract_reasoning_from_message(chunk)
                    if reasoning:
                        accumulated_reasoning += reasoning
                        yield {"content": "", "reasoning": reasoning}

                    # Buffer content; in native mode stream it immediately
                    text = chunk.content or ""
                    buffered_content += text
                    should_hold_for_subagents = bool(
                        self._subagent_state
                        and self._subagent_state.manager.uncollected_records(self._subagent_state)
                    )
                    if not use_fallback and text and not should_hold_for_subagents:
                        yield {"content": text, "reasoning": ""}
            except StreamIdleTimeout:
                logger.warning("Agent streaming iteration %d idle for over %.1fs", iteration, self._idle_seconds)
                trace.error = f"LLM stream idle for over {self._idle_seconds:g}s"
                trace.ended_at = time.time()
                yield {"content": "(Agent stopped responding – please try again.)", "reasoning": ""}
                return
            except _OpenAIBadRequestError as exc:
                if "context length" not in str(exc).lower():
                    raise
                logger.warning("Context length exceeded in iteration %d; attempting emergency compression", iteration)
                messages = await self._emergency_compress(messages, use_fallback=use_fallback)
                if messages is None:
                    raise
                # Retry the LLM call with compressed messages
                full_chunk = None
                buffered_content = ""
                try:
                    async for chunk in idle_bounded_agen(llm_with_tools.astream(messages), self._idle_seconds):
                        if full_chunk is None:
                            full_chunk = chunk
                        else:
                            full_chunk += chunk
                        reasoning = _extract_reasoning_from_message(chunk)
                        if reasoning:
                            accumulated_reasoning += reasoning
                            yield {"content": "", "reasoning": reasoning}
                        text = chunk.content or ""
                        buffered_content += text
                        should_hold_for_subagents = bool(
                            self._subagent_state
                            and self._subagent_state.manager.uncollected_records(self._subagent_state)
                        )
                        if not use_fallback and text and not should_hold_for_subagents:
                            yield {"content": text, "reasoning": ""}
                except Exception:
                    logger.error("Emergency compression retry also failed; re-raising original error")
                    raise

            iteration_end = time.perf_counter()
            iteration_response_time_ms = (iteration_end - iteration_start) * 1000
            self._record_iteration_usage(full_chunk, messages, iteration_response_time_ms)
            yield self._live_usage_event()

            # Reconstruct an AIMessage from the buffered stream
            raw_tool_calls = []
            if full_chunk is not None:
                if hasattr(full_chunk, "tool_calls") and full_chunk.tool_calls:
                    raw_tool_calls = full_chunk.tool_calls
                elif hasattr(full_chunk, "additional_kwargs"):
                    ak = full_chunk.additional_kwargs or {}
                    if "tool_calls" in ak:
                        raw_tool_calls = ak["tool_calls"]

            # Normalize all tool calls into a uniform internal format
            tool_calls = []
            for raw in raw_tool_calls:
                normalized = _normalize_tool_call(raw, self.tool_names)
                if normalized:
                    tool_calls.append(normalized)

            response = AIMessage(
                content=buffered_content,
                tool_calls=tool_calls,
            )
            self._last_ai_text = (buffered_content or "")[-1200:]
            self._last_reasoning_tail = (accumulated_reasoning or "")[-800:]

            if not response.tool_calls:
                # Always try JSON fallback, even in native mode
                fallback, remaining_text = _try_parse_tool_calls(response.content, self.tool_names, flat_dict_mapping=flat_dict_mapping)
                if fallback:
                    logger.debug("Parsed tool calls from content (streaming): %r", fallback)
                    response.tool_calls = fallback
                    response.content = remaining_text
                elif use_fallback and any(name in (response.content or "") for name in self.tool_names):
                    logger.debug("Tool name detected but invalid JSON format. Retrying.")
                    messages.append(response)
                    messages.append(
                        HumanMessage(
                            content='Invalid format. Use the exact JSON format: {"tool_name": {"arg": "value"}}. '
                                    'No markdown, no explanations. Only the JSON object.'
                        )
                    )
                    continue
                else:
                    # An output-token-limit truncation looks exactly like a clean
                    # "text without tool calls" reply: the model was mid-turn (often
                    # right before emitting the tool call) when the provider cut it
                    # off. Detect finish_reason=length/max_tokens and retry instead
                    # of persisting a half-finished status line as the final answer.
                    finish_reason = (
                        (getattr(full_chunk, "response_metadata", None) or {}).get("finish_reason") or ""
                    ).lower()
                    if (
                        ("length" in finish_reason or "max_token" in finish_reason)
                        and iteration < self.max_iterations
                    ):
                        logger.warning(
                            "Iteration %d/%d truncated by output limit (finish_reason=%r); nudging model to continue.",
                            iteration, self.max_iterations, finish_reason,
                        )
                        yield {"content": "", "reasoning": "\n\n**Output token limit reached – continuing.**\n\n"}
                        messages.append(response)
                        messages.append(
                            HumanMessage(
                                content="Your previous reply was cut off by the output token limit before a tool call could be emitted. "
                                "Keep your reply SHORT and either call the needed tool now or provide a concise final answer."
                            )
                        )
                        continue
                    content_stripped = (response.content or "").strip()
                    # If the model returned empty/whitespace content with no tool calls
                    # and we still have iterations left, prompt it to continue rather
                    # than treating this as a premature final answer.
                    if not content_stripped and iteration < self.max_iterations:
                        # Check if reasoning contained a tool call intent
                        reasoning_tool_calls, _ = _try_parse_tool_calls(accumulated_reasoning, self.tool_names, flat_dict_mapping=flat_dict_mapping)
                        if reasoning_tool_calls:
                            logger.debug("Tool calls found in reasoning on iteration %d/%d.", iteration, self.max_iterations)
                            response.tool_calls = reasoning_tool_calls
                            response.content = ""
                            # Fall through to tool execution below
                        else:
                            logger.debug(
                                "Empty response with no tool calls on iteration %d/%d; prompting model to continue.",
                                iteration, self.max_iterations,
                            )
                            messages.append(response)
                            messages.append(
                                HumanMessage(
                                    content="You have not provided a tool call or a final answer. "
                                    "Please either call a tool to gather more information, or provide a complete final answer."
                                )
                            )
                            continue

                    # In native mode, content was already streamed in real time;
                    # just mark completion without emitting buffered_content again.
                    # However, if nothing was actually streamed, fall through to
                    # final answer generation so the user doesn't get an empty message.
                    if (
                        self._subagent_state is not None
                        and not self._auto_collected_once
                        and self._subagent_state.manager.uncollected_records(self._subagent_state)
                    ):
                        if buffered_content.strip():
                            messages.append(response)
                        collected = await self._subagent_state.manager.wait(
                            None, self._subagent_state,
                            input_event=getattr(self, "_input_event", None),
                        )
                        collected_msg = HumanMessage(
                            content="[Automatically collected subagent results]\n\n"
                            + collected["content"]
                        )
                        messages.append(collected_msg)
                        self._tool_result_tokens += count_tokens(str(collected_msg.content))
                        # New input may have interrupted the wait early — only
                        # mark auto-collect done when it genuinely completed,
                        # otherwise still-running children would never get
                        # their results collected on a later attempt.
                        if not (collected.get("metadata") or {}).get("interrupted_by_input"):
                            self._auto_collected_once = True
                        continue

                    if not use_fallback and buffered_content.strip():
                        queue = self._injected_input_queue
                        if iteration < self.max_iterations and queue is not None and not queue.empty():
                            messages.append(response)
                            drained_inputs = self._drain_injected(messages)
                            self._injected_since_observer += len(drained_inputs)
                            for injected in drained_inputs:
                                yield {"content": "", "reasoning": "", "injected_user_message": injected["text"], "injected_source": injected.get("source")}
                            continue
                        blocked = self._todo_gate_open_tasks(iteration)
                        if blocked:
                            self._todo_gate_reminders += 1
                            trace.todo_gate_blocks += 1
                            messages.append(response)
                            messages.append(HumanMessage(content=self._todo_gate_message(blocked)))
                            logger.info(
                                "Finalization held back: %d open todo task(s) (reminder %d/%d)",
                                len(blocked), self._todo_gate_reminders, MAX_TODO_FINALIZE_REMINDERS,
                            )
                            yield {
                                "content": "",
                                "reasoning": (
                                    f"\n\n**Final answer held back — {len(blocked)} open todo task(s). "
                                    f"(reminder {self._todo_gate_reminders}/{MAX_TODO_FINALIZE_REMINDERS})**\n\n"
                                ),
                            }
                            continue
                        logger.debug("No tool calls found (native mode). Content already streamed.")
                        trace.finalization = "no_tool_calls"
                        trace.ended_at = time.time()
                        self._final_answer_usage = self._iteration_usages[-1] if self._iteration_usages else None
                        return

                    if use_fallback:
                        logger.debug("No tool calls found (fallback mode). Generating final answer.")
                    else:
                        logger.debug("No tool calls found (native mode). Content empty – generating final answer.")
                    queue = self._injected_input_queue
                    if iteration < self.max_iterations and queue is not None and not queue.empty():
                        drained_inputs = self._drain_injected(messages)
                        self._injected_since_observer += len(drained_inputs)
                        for injected in drained_inputs:
                            yield {"content": "", "reasoning": "", "injected_user_message": injected["text"], "injected_source": injected.get("source")}
                        continue
                    # Prepare clean messages for final answer (no tool prompts)
                    blocked = self._todo_gate_open_tasks(iteration)
                    if blocked:
                        self._todo_gate_reminders += 1
                        trace.todo_gate_blocks += 1
                        messages.append(response)
                        messages.append(HumanMessage(content=self._todo_gate_message(blocked)))
                        logger.info(
                            "Finalization held back: %d open todo task(s) (reminder %d/%d)",
                            len(blocked), self._todo_gate_reminders, MAX_TODO_FINALIZE_REMINDERS,
                        )
                        yield {
                            "content": "",
                            "reasoning": (
                                f"\n\n**Final answer held back — {len(blocked)} open todo task(s). "
                                f"(reminder {self._todo_gate_reminders}/{MAX_TODO_FINALIZE_REMINDERS})**\n\n"
                            ),
                        }
                        continue
                    final_messages, all_failed, has_successful_results = self._prepare_final_messages(
                        messages, self._raw_system_instructions, rag_context=self._rag_context
                    )
                    if all_failed:
                        final_messages.append(
                            HumanMessage(content="The search tools could not retrieve any results due to errors. Provide a helpful answer based on your knowledge, or honestly say you cannot answer this.")
                        )
                    else:
                        if use_fallback:
                            final_messages.append(
                                HumanMessage(content="Provide a final answer to the user based on the conversation above. Do NOT output JSON or tool calls. Answer in natural language only.")
                            )
                        else:
                            final_messages.append(
                                HumanMessage(content="Provide a final answer to the user based on the conversation above.")
                            )

                    # Try up to 3 times for a non-empty final answer
                    final_text = ""
                    reenter_loop = False
                    for attempt in range(1, 4):
                        total_text_len = 0
                        attempt_text = ""
                        final_full_chunk = None
                        try:
                            iteration_start = time.perf_counter()
                            async for chunk in llm.astream(final_messages):
                                if final_full_chunk is None:
                                    final_full_chunk = chunk
                                else:
                                    final_full_chunk += chunk
                                text = chunk.content or ""
                                reasoning = _extract_reasoning_from_message(chunk)
                                total_text_len += len(text)
                                attempt_text += text
                                if text:
                                    yield {"content": text, "reasoning": ""}
                                if reasoning:
                                    yield {"content": "", "reasoning": reasoning}
                            iteration_end = time.perf_counter()
                            if final_full_chunk is not None:
                                self._record_iteration_usage(
                                    final_full_chunk, final_messages,
                                    (iteration_end - iteration_start) * 1000)
                                yield self._live_usage_event()
                        except Exception:
                            logger.warning("Final answer streaming failed on attempt %d, falling back to non-streaming.", attempt, exc_info=True)
                            try:
                                iteration_start = time.perf_counter()
                                final_response = await _invoke_with_idle_timeout(
                                    llm, final_messages, idle_timeout=self._idle_seconds
                                )
                                iteration_end = time.perf_counter()
                                self._record_iteration_usage(
                                    final_response, final_messages,
                                    (iteration_end - iteration_start) * 1000)
                                yield self._live_usage_event()
                                content = final_response.content or ""
                                total_text_len += len(content)
                                attempt_text += content
                                if content:
                                    yield {"content": content, "reasoning": ""}
                            except Exception:
                                logger.error("Final answer fallback also failed on attempt %d.", attempt, exc_info=True)

                        final_text += attempt_text

                        # If a tool call sneaked into the final answer, re-enter the loop
                        fallback, _ = _try_parse_tool_calls(attempt_text, self.tool_names, flat_dict_mapping=flat_dict_mapping)
                        if fallback and iteration < self.max_iterations:
                            logger.debug("Tool call detected in streamed final answer on iteration %d, re-entering loop.", iteration)
                            response.tool_calls = fallback
                            response.content = ""
                            reenter_loop = True
                            break

                        if total_text_len > 0:
                            trace.finalization = "no_tool_calls"
                            trace.ended_at = time.time()
                            self._final_answer_usage = self._iteration_usages[-1] if self._iteration_usages else None
                            return

                        # Add retry instruction for next attempt
                        if attempt < 3:
                            if use_fallback:
                                final_messages.append(
                                    HumanMessage(content="Your previous answer attempt produced no text. Please provide a final answer to the user based on the tool results above. Do NOT output JSON or tool calls.")
                                )
                            else:
                                final_messages.append(
                                    HumanMessage(content="Your previous answer attempt was empty. Please provide a final answer to the user based on the conversation and tool results above.")
                                )

                    if not reenter_loop:
                        trace.finalization = "no_tool_calls"
                        trace.ended_at = time.time()
                        if has_successful_results:
                            yield {"content": "Die finale Antwortgenerierung ist fehlgeschlagen, obwohl Tool-Ergebnisse vorlagen. Bitte versuche es später noch einmal.", "reasoning": ""}
                        else:
                            yield {"content": "Ich konnte leider keine aktuellen Informationen zu dieser Anfrage finden, da die Suchtools nicht erreichbar waren. Bitte versuche es später noch einmal.", "reasoning": ""}
                        return
                    # reenter_loop: a tool call was parsed out of the final
                    # answer — fall through so it is appended and executed.

            messages.append(response)
            logger.debug(
                "Agent streaming response content=%r tool_calls=%r",
                getattr(response, "content", None),
                getattr(response, "tool_calls", None),
            )

            # Execute tool calls in parallel batch
            batch_calls = response.tool_calls
            if self._auto_collected_once:
                self._auto_collected_once = False
            remaining = self.max_tool_calls - tool_calls_executed
            selected_calls: List[Dict[str, Any]] = []
            skipped_calls: List[Dict[str, Any]] = []
            regular_selected = 0
            for call in batch_calls:
                if call.get("name") in INTERNAL_TOOLS:
                    selected_calls.append(call)
                elif regular_selected < max(0, remaining):
                    selected_calls.append(call)
                    regular_selected += 1
                else:
                    skipped_calls.append(call)
            coordination_calls = [call for call in batch_calls if call.get("name") in INTERNAL_TOOLS]
            if remaining <= 0 and not coordination_calls:
                yield {"content": "", "reasoning": f"\n\n**Tool call limit reached ({self.max_tool_calls}).**\n\n"}
            if len(selected_calls) > self.max_parallel_tool_calls:
                skipped_calls.extend(selected_calls[self.max_parallel_tool_calls:])
                selected_calls = selected_calls[:self.max_parallel_tool_calls]
            batch_calls = selected_calls
            pre_tool_count = len(trace.tool_calls)
            skipped_messages: List[BaseMessage] = []
            for call in skipped_calls:
                call_name = call.get("name", "unknown")
                call_id = call.get("id", "")
                reason = (
                    "Tool call limit reached"
                    if call_name not in INTERNAL_TOOLS and remaining <= 0
                    else f"Parallel tool call limit reached ({self.max_parallel_tool_calls})"
                )
                error_text = f"[Error: {reason}; '{call_name}' was not executed.]"
                record = ToolCallRecord(
                    iteration=iteration,
                    tool_name=call_name,
                    args=call.get("args") or {},
                    started_at=time.time(),
                    ended_at=time.time(),
                    ok=False,
                    error=error_text,
                    output=error_text,
                    kind="subagent" if call_name == "spawn_subagent" else (
                        "subagent_control" if call_name in SUBAGENT_COORDINATION_TOOLS else "tool"
                    ),
                    call_id=f"{self._trace_id}:{call_id}:skipped",
                )
                trace.tool_calls.append(record)
                if use_fallback:
                    skipped_messages.append(HumanMessage(content=error_text))
                else:
                    skipped_messages.append(ToolMessage(content=error_text, tool_call_id=call_id))

            tool_messages: List[ToolMessage] = []
            if batch_calls:
                yield {"content": "", "reasoning": f"\n**Executing {len(batch_calls)} tool(s) in parallel...**\n"}
                for call in batch_calls:
                    yield {
                        "content": "",
                        "reasoning": "",
                        "tool_progress": {
                            "id": call.get("id"),
                            "tool_name": call.get("name", "unknown"),
                            "stage": "started",
                            "iteration": iteration,
                            "args_preview": str(call.get("args") or "")[:120],
                        },
                    }
                # Emit code_execution started events before parallel run
                for call in batch_calls:
                    if call.get("name") == "code_execution":
                        yield {
                            "content": "",
                            "reasoning": "",
                            "tool_execution": {
                                "tool": "code_execution",
                                "stage": "started",
                                "code": call.get("args", {}).get("code", ""),
                                "packages": call.get("args", {}).get("packages") or [],
                            },
                        }

                # Long-running tools (code_execution installs/computation)
                # emit no chunks — re-emit progress events periodically so the
                # SSE wire and the dashboard progress display stay alive. The
                # `keepalive` flag marks them as synthetic: they do NOT reset
                # the run-level idle watchdog, so a stuck tool still dies.
                exec_task = asyncio.create_task(
                    self._execute_tools_parallel(
                        batch_calls, iteration, trace, use_fallback=use_fallback,
                    )
                )
                try:
                    while True:
                        done, _pending = await asyncio.wait(
                            {exec_task}, timeout=TOOL_KEEPALIVE_SECONDS
                        )
                        if done:
                            break
                        for call in batch_calls:
                            yield {
                                "content": "",
                                "reasoning": "",
                                "tool_progress": {
                                    "id": call.get("id"),
                                    "tool_name": call.get("name", "unknown"),
                                    "stage": "progress",
                                    "iteration": iteration,
                                    "args_preview": str(call.get("args") or "")[:120],
                                    "keepalive": True,
                                },
                                "keepalive": True,
                            }
                finally:
                    # GeneratorExit/Cancel mid-wait: don't let the tool batch
                    # keep running orphaned — cancellation reaches the gather
                    # and aborts in-flight calls (e.g. aiohttp requests).
                    if not exec_task.done():
                        exec_task.cancel()
                new_messages, executed, tool_messages = exec_task.result()
                messages.extend(new_messages)
                self._tool_result_tokens += sum(
                    count_tokens(stringify_message_content(getattr(m, "content", "")) or "")
                    for m in new_messages
                )
                tool_calls_executed += sum(
                    1 for call in batch_calls
                    if call.get("name") not in INTERNAL_TOOLS
                )
            messages.extend(skipped_messages)
            self._tool_result_tokens += sum(
                count_tokens(str(getattr(m, "content", "") or ""))
                for m in skipped_messages
            )

            # Stream tool call results to the client
            new_records = trace.tool_calls[pre_tool_count:]
            if new_records:
                yield {
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        {
                            "tool_name": rec.tool_name,
                            "iteration": rec.iteration,
                            "kind": rec.kind,
                            "id": rec.call_id,
                            "args": rec.args,
                            "ok": rec.ok,
                            "error": rec.error,
                            "duration_ms": round((rec.ended_at - rec.started_at) * 1000, 1) if rec.ended_at else None,
                            "output": rec.output,
                            "metadata": rec.metadata,
                        }
                        for rec in new_records
                    ],
                }

            # Stream the updated todo list when the agent_todo tool mutated it
            if (
                self._todo_state is not None
                and self._todo_state.version > 0
                and self._todo_state.version != self._emitted_todo_version
            ):
                self._emitted_todo_version = self._todo_state.version
                yield {"content": "", "reasoning": "", "todo_list": self._todo_state.snapshot()}

            # Compress messages before next iteration to keep context bounded.
            if use_fallback:
                messages, _ = self._compress_messages(messages)
            else:
                # In native mode, do light cleanup: remove empty AIMessages
                # and internal guidance SystemMessages, but keep ToolMessages
                # so the LLM can correlate them with prior tool_calls.
                messages = self._light_compress_messages(messages)

            # Emit code_execution result events after parallel run
            for msg in tool_messages:
                if msg.tool_call_id:
                    # Find matching call to check if it was code_execution
                    for call in batch_calls:
                        if call.get("id") == msg.tool_call_id and call.get("name") == "code_execution":
                            yield {
                                "content": "",
                                "reasoning": "",
                                "tool_execution": {
                                    "tool": "code_execution",
                                    "stage": "result",
                                    "output": msg.content,
                                },
                            }

            if batch_calls:
                yield {"content": "", "reasoning": f"**All {len(batch_calls)} tool result(s) received.**\n\n"}

            # After tool execution, guide the model
            if use_fallback:
                if tool_calls_executed >= self.max_tool_calls:
                    messages.append(
                        HumanMessage(content="Tool call limit reached. You must answer the user now. Do not call any more tools.")
                    )
                else:
                    messages.append(
                        HumanMessage(content="Tool result is above. Answer the user now, or call another tool with different arguments if you still need more info.")
                    )
            else:
                if tool_calls_executed >= self.max_tool_calls:
                    messages.append(
                        SystemMessage(content="Tool call limit reached. Answer the user now.")
                    )
                else:
                    messages.append(
                        SystemMessage(content="Answer the user, or call another tool if needed.")
                    )

        # Force final answer with one more LLM call (no tools) – streamed
        trace.finalization = "max_iterations"
        if (
            self._subagent_state is not None
            and self._subagent_state.manager.uncollected_records(self._subagent_state)
        ):
            collected = await self._subagent_state.manager.wait(
                None, self._subagent_state
            )
            collected_msg = HumanMessage(
                content="[Automatically collected subagent results before forced finalization]\n\n"
                + collected["content"]
            )
            messages.append(collected_msg)
            self._tool_result_tokens += count_tokens(str(collected_msg.content))
        # Prepare clean messages for final answer (no tool prompts)
        final_messages, all_failed, has_successful_results = self._prepare_final_messages(
            messages, self._raw_system_instructions, rag_context=self._rag_context
        )
        if all_failed:
            final_messages.append(
                HumanMessage(content="The search tools could not retrieve any results due to errors. Provide a helpful answer based on your knowledge, or honestly say you cannot answer this.")
            )
        else:
            if use_fallback:
                final_messages.append(
                    HumanMessage(content="You have reached the maximum number of tool calls. Provide a final answer to the user based on the conversation above. Do NOT output JSON or tool calls. Answer in natural language only.")
                )
            else:
                final_messages.append(
                    HumanMessage(content="Provide a final answer to the user based on the conversation above.")
                )

        # Try up to 3 times for a non-empty final answer
        final_text = ""
        for attempt in range(1, 4):
            total_text_len = 0
            attempt_text = ""
            final_full_chunk = None
            try:
                iteration_start = time.perf_counter()
                chunk_count = 0
                async for chunk in llm.astream(final_messages):
                    if final_full_chunk is None:
                        final_full_chunk = chunk
                    else:
                        final_full_chunk += chunk
                    text = chunk.content or ""
                    reasoning = _extract_reasoning_from_message(chunk)
                    chunk_count += 1
                    total_text_len += len(text)
                    attempt_text += text
                    if text:
                        yield {"content": text, "reasoning": ""}
                    if reasoning:
                        yield {"content": "", "reasoning": reasoning}
                iteration_end = time.perf_counter()
                logger.debug("Final answer attempt %d streaming complete: %d chunks, %d chars", attempt, chunk_count, total_text_len)
                if final_full_chunk is not None:
                    self._record_iteration_usage(
                        final_full_chunk, final_messages,
                        (iteration_end - iteration_start) * 1000)
                    yield self._live_usage_event()
            except Exception:
                logger.warning("Final answer streaming failed on attempt %d, falling back to non-streaming.", attempt, exc_info=True)
                try:
                    iteration_start = time.perf_counter()
                    final_response = await _invoke_with_idle_timeout(
                        llm, final_messages, idle_timeout=self._idle_seconds
                    )
                    iteration_end = time.perf_counter()
                    trace.ended_at = time.time()
                    self._record_iteration_usage(
                        final_response, final_messages,
                        (iteration_end - iteration_start) * 1000)
                    yield self._live_usage_event()
                    content = final_response.content or ""
                    total_text_len += len(content)
                    attempt_text += content
                    if content:
                        yield {"content": content, "reasoning": ""}
                except Exception:
                    logger.error("Final answer fallback also failed on attempt %d.", attempt, exc_info=True)
                    trace.error = "streaming_final_answer_failed"
                    trace.ended_at = time.time()

            final_text += attempt_text

            # max_iterations already reached – cannot re-enter loop, but warn if a tool call was detected
            fallback, _ = _try_parse_tool_calls(attempt_text, self.tool_names, flat_dict_mapping=flat_dict_mapping)
            if fallback:
                logger.debug("Tool call detected in forced final answer after max_iterations; cannot re-enter loop.")
                yield {"content": "", "reasoning": "(Max iterations reached – tool call detected but could not be executed.)"}

            if total_text_len > 0:
                trace.ended_at = time.time()
                self._final_answer_usage = self._iteration_usages[-1] if self._iteration_usages else None
                return

            # Add retry instruction for next attempt
            if attempt < 3:
                if use_fallback:
                    final_messages.append(
                        HumanMessage(content="Your previous answer attempt produced no text. Please provide a final answer to the user based on the tool results above. Do NOT output JSON or tool calls.")
                    )
                else:
                    final_messages.append(
                        HumanMessage(content="Your previous answer attempt was empty. Please provide a final answer to the user based on the conversation and tool results above.")
                    )

        # All attempts failed
        trace.ended_at = time.time()
        if has_successful_results:
            yield {"content": "Die finale Antwortgenerierung ist fehlgeschlagen, obwohl Tool-Ergebnisse vorlagen. Bitte versuche es später noch einmal.", "reasoning": ""}
        else:
            yield {"content": "Ich konnte leider keine aktuellen Informationen zu dieser Anfrage finden, da die Suchtools nicht erreichbar waren. Bitte versuche es später noch einmal.", "reasoning": ""}
