"""Run observer: an optional, per-assistant supervisor for agent runs.

The observer watches a running ``AgentRunner`` from the sidelines. At the top
of every agent iteration the runner calls ``collect()`` to apply a finished
check (todo-list updates plus an optional ``[Observer note]`` nudge injected
as a HumanMessage) and ``tick()`` to decide whether a new asynchronous check
should start. The actual LLM analysis runs as a background asyncio task so
the agent loop is never blocked — a decision lands one iteration later at
the earliest.

Triggering is hybrid: periodic checks (every N iterations / every N seconds)
plus semantic heuristics (identical repeated tool calls, error streaks,
stale or missing todo lists, budget pressure, mid-run user input). Each
heuristic latches once fired and only re-arms after its condition clears,
and a shared cooldown prevents observer storms.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools.builtin_tools.todo_list import TodoListState
from app.chat_utils import extract_usage_metadata

logger = logging.getLogger(__name__)

# First possible check: the agent needs at least one finished iteration of
# context before an assessment makes sense.
FIRST_CHECK_ITERATION = 2
# Heuristic thresholds
REPEAT_CALL_THRESHOLD = 3
ERROR_STREAK = 3
BUDGET_PRESSURE_RATIO = 0.8
HEURISTIC_COOLDOWN_SECONDS = 45.0
# Output caps
NUDGE_MAX_CHARS = 1000
MAX_TODO_OPS_PER_CHECK = 10
ASSESSMENT_MAX_CHARS = 2000
# Digest caps
_DIGEST_TOOL_RECORDS = 8
_DIGEST_ARG_PREVIEW = 120
_DIGEST_OUTPUT_PREVIEW = 200
_DIGEST_AI_TEXT = 1200
_DIGEST_REASONING_TAIL = 800
_DIGEST_USER_REQUEST = 2000
_DIGEST_SYSTEM_EXCERPT = 1500

# Coordination/todo tools poll legitimately — identical repeats are expected
# and must not count as a stuck loop. Mirrors runner.INTERNAL_TOOLS plus the
# spawn call (kept local so this module stays importable without the runner).
_OBSERVER_EXCLUDED_TOOLS = {
    "spawn_subagent",
    "wait_subagent",
    "subagent_status",
    "wait",
    "agent_todo",
}

_HEURISTIC_REASONS = {
    "repeat_tool_calls",
    "error_streak",
    "stale_todo",
    "no_todo",
    "budget_pressure",
}


@dataclass
class ObserverConfig:
    """Tunable observer behaviour, sourced from the assistant ``observer:`` block."""

    enabled: bool = False
    assistant: Optional[str] = None
    every_iterations: int = 4
    every_seconds: float = 180.0
    max_checks: int = 8
    max_nudges: int = 4
    check_timeout_seconds: float = 60.0
    include_subagents: bool = True
    instructions: str = ""

    @staticmethod
    def _int(raw: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _float(raw: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, value))

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ObserverConfig":
        data = raw if isinstance(raw, dict) else {}
        assistant = data.get("assistant")
        instructions = data.get("instructions")
        return cls(
            enabled=bool(data.get("enabled", False)),
            assistant=str(assistant).strip() if assistant else None,
            every_iterations=cls._int(data.get("every_iterations"), 4, 1, 20),
            every_seconds=cls._float(data.get("every_seconds"), 180.0, 30.0, 3600.0),
            max_checks=cls._int(data.get("max_checks"), 8, 1, 50),
            max_nudges=cls._int(data.get("max_nudges"), 4, 1, 20),
            check_timeout_seconds=cls._float(data.get("check_timeout_seconds"), 60.0, 10.0, 300.0),
            include_subagents=bool(data.get("include_subagents", True)),
            instructions=str(instructions).strip()[:4000] if instructions else "",
        )


@dataclass
class ObserverRunSnapshot:
    """Point-in-time view of the run, built by the runner at each iteration."""

    iteration: int
    max_iterations: int
    tool_calls_executed: int
    max_tool_calls: int
    elapsed_s: float
    recent_records: List[Dict[str, Any]] = field(default_factory=list)
    todo_version: int = 0
    todo_render: str = ""
    open_tasks: int = 0
    last_ai_text: str = ""
    reasoning_tail: str = ""
    input_text: str = ""
    system_excerpt: str = ""
    injected_since_last: int = 0
    todo_available: bool = True


@dataclass
class ObserverDecision:
    """Result of one observer check."""

    assessment: str = ""
    todo_ops: List[Dict[str, Any]] = field(default_factory=list)
    nudge: Optional[str] = None
    trigger_reasons: List[str] = field(default_factory=list)
    iteration: int = 0
    check_ms: float = 0.0
    model: str = ""


OBSERVER_PROMPT = """You are a run observer supervising an AI agent that is currently executing a task.
You are NOT the agent and NOT a chat partner — you analyze the run digest below and decide whether intervention helps.

You receive: the agent's role/system excerpt, the user request, run statistics, trigger reasons, the current todo list, recent tool calls, the last assistant text and a reasoning tail.

Respond with ONE strict JSON object (no markdown, no prose around it):
{
  "assessment": "1-3 sentences: is the run on track? what is wrong or missing?",
  "todo_ops": [
    {"action": "add", "title": "short task", "note": "optional detail", "status": "pending|in_progress"},
    {"action": "update", "task_id": "t1", "title": "new title", "note": "new note", "status": "pending|in_progress|completed|skipped"},
    {"action": "complete", "task_id": "t1"},
    {"action": "remove", "task_id": "t1"}
  ],
  "nudge": "short actionable hint for the agent" | null
}

Rules:
- If the run is healthy and on track: empty todo_ops and "nudge": null.
- Nudges must be short (1-3 sentences) and actionable: correct a wrong tool argument, refocus on the user request, break a loop, or push toward finalization when the budget is nearly exhausted.
- Keep the todo list realistic: complete finished tasks, split only when genuinely useful, do not rewrite the whole plan without cause.
- Do not restart or re-plan work that already produced successful tool results.
- Write assessment and nudge in the same language as the user request.
- Reference existing task ids exactly (e.g. "t1"); unknown ids are ignored.
"""

def _norm_args(args: Any) -> str:
    try:
        return json.dumps(args or {}, sort_keys=True, default=str)[:500]
    except Exception:
        return str(args)[:500]


def _extract_json_objects(text: str) -> List[str]:
    """Return all top-level ``{...}`` spans via a balanced-brace scan."""
    spans: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start:i + 1])
                start = -1
    return spans


class RunObserver:
    """Per-run observer state machine. All public methods are exception-safe."""

    def __init__(
        self,
        config: ObserverConfig,
        llm_factory: Callable[[], Awaitable[Any]],
        assistant_name: str = "",
    ) -> None:
        self.config = config
        self._llm_factory = llm_factory
        self.assistant_name = assistant_name
        self.last_check_iter = 0
        # Anchors: periodic gates count from observer creation (~run start),
        # heuristic cooldown from the last heuristic-triggered check.
        self.last_check_at = time.monotonic()
        self._last_heuristic_at = 0.0
        self.checks_used = 0
        self.nudges_used = 0
        self._pending: Optional[asyncio.Task] = None
        self._latches: set[str] = set()
        self._seen_todo_version = -1
        self._todo_change_iter = 0
        self.usage: Dict[str, int] = {}

    # -- trigger evaluation ---------------------------------------------

    def evaluate_triggers(self, snap: ObserverRunSnapshot) -> List[str]:
        """Return the list of reasons a check should fire for this snapshot."""
        now = time.monotonic()
        reasons: List[str] = []
        if snap.iteration - self.last_check_iter >= self.config.every_iterations:
            reasons.append("periodic_iterations")
        if now - self.last_check_at >= self.config.every_seconds:
            reasons.append("periodic_seconds")

        # -- active heuristic conditions --------------------------------
        active: set[str] = set()
        candidates = [
            r for r in snap.recent_records
            if r.get("tool_name") not in _OBSERVER_EXCLUDED_TOOLS and not r.get("skipped")
        ]
        if len(candidates) >= REPEAT_CALL_THRESHOLD:
            tail = candidates[-REPEAT_CALL_THRESHOLD:]
            keys = {(r.get("tool_name"), _norm_args(r.get("args"))) for r in tail}
            if len(keys) == 1:
                active.add("repeat_tool_calls")
        streak = 0
        for rec in reversed(snap.recent_records):
            if rec.get("skipped") or rec.get("ok"):
                break
            streak += 1
        if streak >= ERROR_STREAK:
            active.add("error_streak")
        if snap.todo_version != self._seen_todo_version:
            self._seen_todo_version = snap.todo_version
            self._todo_change_iter = snap.iteration
        elif (
            snap.todo_version > 0
            and snap.open_tasks > 0
            and snap.iteration - self._todo_change_iter >= self.config.every_iterations
        ):
            active.add("stale_todo")
        if (
            # "no plan" is a soft signal — unlike hard stalls it is only
            # meaningful once the agent had a fair chance (one periodic
            # cadence) to produce a todo.
            snap.todo_available
            and snap.todo_version == 0
            and snap.tool_calls_executed > 0
            and snap.iteration >= self.config.every_iterations
        ):
            active.add("no_todo")
        if (
            snap.iteration >= BUDGET_PRESSURE_RATIO * snap.max_iterations
            or snap.tool_calls_executed >= BUDGET_PRESSURE_RATIO * snap.max_tool_calls
        ):
            active.add("budget_pressure")

        # Re-arm latches whose condition cleared; latched reasons don't refire.
        self._latches &= active
        heuristic_allowed = (
            self._last_heuristic_at == 0.0
            or now - self._last_heuristic_at >= HEURISTIC_COOLDOWN_SECONDS
        )
        if heuristic_allowed:
            reasons.extend(sorted(active - self._latches))

        if snap.injected_since_last > 0:
            reasons.append("user_input_injected")
        return reasons

    # -- check lifecycle -------------------------------------------------

    def tick(self, snap: ObserverRunSnapshot) -> Optional[Dict[str, Any]]:
        """Maybe start a background check. Returns a ``check_started`` event."""
        if not self.config.enabled or snap.iteration < FIRST_CHECK_ITERATION:
            return None
        if self._pending is not None or self.checks_used >= self.config.max_checks:
            return None
        try:
            reasons = self.evaluate_triggers(snap)
        except Exception:
            logger.warning("Observer trigger evaluation failed", exc_info=True)
            return None
        if not reasons:
            return None
        now = time.monotonic()
        self.last_check_at = now
        self.last_check_iter = snap.iteration
        self.checks_used += 1
        self._latches |= {r for r in reasons if r in _HEURISTIC_REASONS}
        if any(r in _HEURISTIC_REASONS for r in reasons):
            self._last_heuristic_at = now
        try:
            self._pending = asyncio.create_task(self._run_check(snap, reasons))
        except Exception:
            logger.warning("Observer could not start check task", exc_info=True)
            return None
        return {
            "kind": "check_started",
            "iteration": snap.iteration,
            "trigger": reasons,
            "check": self.checks_used,
            "ts": time.time(),
        }

    def collect(self) -> Optional[ObserverDecision]:
        """Pick up a finished check result (sync, non-blocking)."""
        task = self._pending
        if task is None:
            return None
        if not task.done():
            return None
        self._pending = None
        try:
            return task.result()
        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.warning("Observer check task failed: %s", exc)
            return None

    def apply_decision(
        self, decision: ObserverDecision, todo_state: Optional[TodoListState]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Apply todo ops and produce stream events + the nudge text."""
        ops_applied: List[Dict[str, Any]] = []
        if todo_state is not None:
            for op in decision.todo_ops[:MAX_TODO_OPS_PER_CHECK]:
                if not isinstance(op, dict):
                    continue
                action = str(op.get("action") or "").strip().lower()
                entry: Dict[str, Any] = {"action": action or "?", "ok": False}
                task_id = op.get("task_id")
                if task_id:
                    entry["task_id"] = str(task_id)
                try:
                    if action == "add":
                        task = todo_state.add(
                            str(op.get("title") or ""),
                            note=op.get("note"),
                            status=op.get("status") or "pending",
                        )
                        entry.update(ok=True, task_id=task["id"], title=task["title"])
                    elif action in {"update", "complete"}:
                        kwargs: Dict[str, Any] = {}
                        if op.get("title") is not None:
                            kwargs["title"] = str(op["title"])
                        if op.get("note") is not None:
                            kwargs["note"] = str(op["note"])
                        kwargs["status"] = (
                            "completed" if action == "complete" else op.get("status")
                        )
                        task = todo_state.update(str(task_id or ""), **kwargs)
                        entry.update(ok=True, task_id=task["id"], title=task["title"])
                    elif action == "remove":
                        task = todo_state.remove(str(task_id or ""))
                        entry.update(ok=True, task_id=task["id"], title=task["title"])
                    else:
                        entry["detail"] = "unknown action"
                except Exception as exc:
                    entry["detail"] = str(exc)[:200]
                ops_applied.append(entry)

        nudge_text: Optional[str] = None
        if decision.nudge and self.nudges_used < self.config.max_nudges:
            self.nudges_used += 1
            nudge_text = (
                f"[Observer note #{self.nudges_used}] "
                f"{decision.nudge[:NUDGE_MAX_CHARS]}"
            )

        event: Dict[str, Any] = {
            "kind": "action" if (ops_applied or nudge_text) else "assessment",
            "iteration": decision.iteration,
            "trigger": list(decision.trigger_reasons),
            "assessment": decision.assessment[:ASSESSMENT_MAX_CHARS],
            "check_ms": round(decision.check_ms, 1),
            "model": decision.model,
            "check": self.checks_used,
            "ts": time.time(),
        }
        if ops_applied:
            event["todo_ops_applied"] = ops_applied
        if nudge_text:
            event["nudge"] = decision.nudge[:NUDGE_MAX_CHARS]
        elif decision.nudge:
            event["nudge_dropped"] = "cap"
        return [event], nudge_text

    async def aclose(self) -> None:
        """Cancel and await a still-pending check task."""
        task = self._pending
        self._pending = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # -- internals -------------------------------------------------------

    async def _run_check(
        self, snap: ObserverRunSnapshot, reasons: List[str]
    ) -> ObserverDecision:
        started = time.perf_counter()
        decision = ObserverDecision(trigger_reasons=list(reasons), iteration=snap.iteration)
        try:
            llm = await self._llm_factory()
            decision.model = (
                getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""
            )
            digest = self._build_digest(snap, reasons)
            prompt = OBSERVER_PROMPT
            if self.config.instructions:
                prompt += f"\nAdditional operator instructions:\n{self.config.instructions}\n"
            resp = await asyncio.wait_for(
                llm.ainvoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=digest),
                    ]
                ),
                timeout=self.config.check_timeout_seconds,
            )
            usage = extract_usage_metadata(resp)
            if usage:
                self._merge_usage(usage)
            content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
            parsed = self._parse_decision(content)
            decision.assessment = parsed.assessment
            decision.todo_ops = parsed.todo_ops
            decision.nudge = parsed.nudge
        except Exception as exc:
            logger.warning("Observer check %d failed: %s", self.checks_used, exc)
            decision.assessment = f"(observer check failed: {exc})"
        decision.check_ms = (time.perf_counter() - started) * 1000.0
        return decision

    def _merge_usage(self, usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and value:
                self.usage[key] = int(self.usage.get(key, 0) + value)

    def _parse_decision(self, text: str) -> ObserverDecision:
        """Parse the observer LLM reply; degrade to assessment-only on garbage."""
        decision = ObserverDecision()
        payload: Optional[Dict[str, Any]] = None
        for span in _extract_json_objects(text):
            try:
                candidate = json.loads(span)
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict) and (
                "assessment" in candidate or "todo_ops" in candidate or "nudge" in candidate
            ):
                payload = candidate
                break
        if payload is None:
            stripped = (text or "").strip()
            if stripped:
                decision.assessment = stripped[:ASSESSMENT_MAX_CHARS]
            return decision
        assessment = payload.get("assessment")
        if isinstance(assessment, str):
            decision.assessment = assessment.strip()
        raw_ops = payload.get("todo_ops")
        if isinstance(raw_ops, list):
            decision.todo_ops = [op for op in raw_ops if isinstance(op, dict)]
        nudge = payload.get("nudge")
        if isinstance(nudge, str) and nudge.strip():
            decision.nudge = nudge.strip()
        return decision

    def _build_digest(self, snap: ObserverRunSnapshot, reasons: List[str]) -> str:
        lines: List[str] = [
            f"Assistant: {self.assistant_name or 'unknown'}",
            "",
            "## Agent role (excerpt)",
            (snap.system_excerpt or "(none)")[:_DIGEST_SYSTEM_EXCERPT],
            "",
            "## User request",
            (snap.input_text or "(unknown)")[:_DIGEST_USER_REQUEST],
            "",
            "## Run status",
            f"- Iteration: {snap.iteration}/{snap.max_iterations}",
            f"- Tool calls used: {snap.tool_calls_executed}/{snap.max_tool_calls}",
            f"- Elapsed: {snap.elapsed_s:.0f}s",
            f"- Observer check: {self.checks_used}/{self.config.max_checks}",
            f"- Trigger reasons: {', '.join(reasons)}",
        ]
        if snap.injected_since_last:
            lines.append(f"- New user messages injected since last check: {snap.injected_since_last}")
        lines += [
            "",
            "## Todo list",
            snap.todo_render or "(no todo list created)",
            "",
            "## Recent tool calls",
        ]
        if snap.recent_records:
            for rec in snap.recent_records[-_DIGEST_TOOL_RECORDS:]:
                status = "ok" if rec.get("ok") else "error"
                duration = rec.get("duration_ms")
                dur_text = f" {duration}ms" if duration is not None else ""
                args_preview = _norm_args(rec.get("args"))[:_DIGEST_ARG_PREVIEW]
                lines.append(
                    f"- [iter {rec.get('iteration', '?')}] {rec.get('tool_name', '?')}"
                    f"({args_preview}) -> {status}{dur_text}"
                )
                preview = (rec.get("error") if not rec.get("ok") else rec.get("output_preview")) or ""
                preview = str(preview).strip()
                if preview:
                    lines.append(f"  output: {preview[:_DIGEST_OUTPUT_PREVIEW]}")
        else:
            lines.append("(none yet)")
        if snap.last_ai_text:
            lines += [
                "",
                "## Last assistant text",
                snap.last_ai_text[-_DIGEST_AI_TEXT:],
            ]
        if snap.reasoning_tail:
            lines += [
                "",
                "## Reasoning tail",
                snap.reasoning_tail[-_DIGEST_REASONING_TAIL:],
            ]
        return "\n".join(lines)
