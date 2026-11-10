"""Per-request lifecycle management for asynchronously running subagents."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.stream_watchdog import DEFAULT_STREAM_IDLE_SECONDS, StreamIdleTimeout, idle_bounded_agen


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _compact_nested_calls(calls: List[Dict[str, Any]], depth: int = 0) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for call in calls[:30]:
        args = call.get("args") or {}
        if isinstance(args, dict):
            args = {
                str(key): str(value)[:240]
                for key, value in list(args.items())[:12]
            }
        else:
            args = {}
        item = {
            key: call.get(key)
            for key in ("id", "iteration", "tool_name", "kind", "ok", "error", "duration_ms")
            if call.get(key) is not None
        }
        item["args"] = args
        if call.get("output"):
            item["output"] = str(call["output"])[:2000]
        child = (call.get("metadata") or {}).get("subagent")
        if isinstance(child, dict):
            child_meta = {
                key: child.get(key)
                for key in (
                    "id", "parent_id", "label", "task", "assistant", "depth",
                    "status", "duration_ms", "iterations", "current_step", "error",
                )
                if child.get(key) is not None
            }
            child_meta["answer"] = str(child.get("answer") or "")[:4000]
            child_meta["activity"] = list(child.get("activity") or [])[:20]
            child_meta["nested_usage"] = dict(child.get("nested_usage") or {})
            if depth < 2:
                child_meta["nested_tool_calls"] = _compact_nested_calls(
                    child.get("nested_tool_calls") or [], depth + 1
                )
            item["metadata"] = {"subagent": child_meta}
        compact.append(item)
    return compact


@dataclass
class SubagentRecord:
    id: str
    parent_id: Optional[str]
    owner_key: str
    task: str
    label: str
    assistant: str
    depth: int
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    ended_at: float = 0.0
    iterations: int = 0
    current_step: str = "Queued"
    reasoning_preview: str = ""
    answer: str = ""
    error: Optional[str] = None
    activity: List[Dict[str, Any]] = field(default_factory=list)
    nested_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    nested_usage: Dict[str, Any] = field(default_factory=dict)
    observer_events: List[Dict[str, Any]] = field(default_factory=list)
    collected: bool = False
    slot_held: bool = False
    # Mid-run nudges into this subagent (parent agent or user): consumed by
    # the child's injected-input drain; input_event wakes blocking waits.
    nudge_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    input_event: asyncio.Event = field(default_factory=asyncio.Event)
    nudges: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        end = self.ended_at or time.time()
        start = self.started_at or self.created_at
        return round((end - start) * 1000, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "task": self.task,
            "label": self.label,
            "assistant": self.assistant,
            "depth": self.depth,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "iterations": self.iterations,
            "current_step": self.current_step,
            "reasoning_preview": self.reasoning_preview,
            "answer": self.answer[:25_000],
            "error": self.error,
            # Bound the serialized activity log: every progress event carries
            # the full record dict, and an unbounded list would make the SSE
            # stream grow quadratically. The complete list stays on the
            # record for format_dossier.
            "activity": list(self.activity[-100:]),
            "activity_total": len(self.activity),
            "nested_tool_calls": _compact_nested_calls(self.nested_tool_calls),
            "nested_usage": dict(self.nested_usage),
            "observer_events": list(self.observer_events[-20:]),
            "nudges": [dict(entry) for entry in self.nudges[-10:]],
        }

    def format_dossier(self, max_answer_chars: int = 6000) -> str:
        elapsed = self.duration_ms or 0
        lines = [
            f'=== Subagent {self.id} ("{self.label}") ===',
            f"Status: {self.status} | Duration: {elapsed / 1000:.1f}s | Iterations: {self.iterations} | Tool calls: {len(self.activity)}",
        ]
        if self.current_step and self.status in {"queued", "running"}:
            lines.append(f"Current step: {self.current_step}")
        if self.reasoning_preview and self.status == "running":
            lines.append(f"Reasoning preview: {self.reasoning_preview}")
        if self.activity:
            lines.append("Activity log:")
            for idx, item in enumerate(self.activity, 1):
                duration = f" ({item['duration_ms']}ms)" if item.get("duration_ms") is not None else ""
                args = f"({item['args_preview']})" if item.get("args_preview") else ""
                result = item.get("error") or item.get("output_preview") or item["status"]
                lines.append(
                    f"  {idx}. [iter {item.get('iteration', '?')}] "
                    f"{item['tool_name']}{args} -> {result}{duration}"
                )
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.nudges:
            lines.append(f"Nudges received: {len(self.nudges)}")
        if self.answer:
            answer = self.answer[:max_answer_chars]
            if len(self.answer) > max_answer_chars:
                answer += (
                    f"\n[answer truncated — {max_answer_chars} of {len(self.answer)} chars shown; "
                    f"call subagent_status(subagent_id=\"{self.id}\", full_answer=true) "
                    "for the complete answer]"
                )
            lines.extend(["Result:", answer])
        return "\n".join(lines)


@dataclass
class SubagentState:
    manager: "SubagentManager"
    depth: int
    is_root: bool
    assistant_name: str
    owner_key: str
    parent_subagent_id: Optional[str] = None
    # Resolved agent-workspace context from the root run — children share the
    # same folder although they execute under derived conversation ids.
    workspace_ctx: Optional[Dict[str, Any]] = None

    def child(self, subagent_id: str) -> "SubagentState":
        return SubagentState(
            manager=self.manager,
            depth=self.depth + 1,
            is_root=False,
            assistant_name=self.assistant_name,
            owner_key=subagent_id,
            parent_subagent_id=subagent_id,
            workspace_ctx=self.workspace_ctx,
        )


class SubagentManager:
    """Coordinates background tasks and their lifecycle events for one chat run."""

    def __init__(
        self,
        config: Dict[str, Any],
        conversation_id: Optional[str],
        rag_app: Any,
        idle_seconds: float = DEFAULT_STREAM_IDLE_SECONDS,
        channel_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_concurrent = _bounded_int(config.get("max_concurrent"), 2, 1, 10)
        self.max_depth = _bounded_int(config.get("max_depth"), 1, 1, 5)
        self.idle_seconds = idle_seconds
        self.max_total = _bounded_int(config.get("max_total"), 8, 1, 500)
        # Per-subagent answer budget in wait() dossiers — larger than a tool
        # page; the parent's context compression handles the aggregate.
        self.report_answer_chars = _bounded_int(
            config.get("report_answer_chars"), 20_000, 1_000, 60_000
        )
        self.root_conversation_id = conversation_id
        self.rag_app = rag_app
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.records: Dict[str, SubagentRecord] = {}
        self.tasks: Dict[str, asyncio.Task[None]] = {}
        self._counter = 0
        self._spawn_lock = asyncio.Lock()
        self._run_id = uuid4().hex[:8]
        # sid -> record registry shared with the owning chat run so user-side
        # nudges can address any subagent in the tree (sids are unique across
        # the whole tree: root `sub-<run>-<n>`, nested `<parent>.<counter>`).
        self._channel_registry = channel_registry

    async def spawn(
        self,
        *,
        task: str,
        label: Optional[str],
        owner: SubagentState,
    ) -> SubagentRecord:
        if owner.depth >= self.max_depth:
            raise ValueError(f"Maximum subagent depth ({self.max_depth}) reached")
        async with self._spawn_lock:
            if self._counter >= self.max_total:
                raise ValueError(f"Subagent budget ({self.max_total}) exhausted")
            self._counter += 1
            sid = (
                f"{owner.parent_subagent_id}.{self._counter}"
                if owner.parent_subagent_id
                else f"sub-{self._run_id}-{self._counter}"
            )
            record = SubagentRecord(
                id=sid,
                parent_id=owner.parent_subagent_id,
                owner_key=owner.owner_key,
                task=task,
                label=(label or "").strip()[:80] or task.strip()[:80] or sid,
                assistant=owner.assistant_name,
                depth=owner.depth + 1,
            )
            self.records[sid] = record
            if self._channel_registry is not None:
                self._channel_registry[sid] = record
            await self.event_queue.put({**record.to_dict(), "stage": "queued"})
            self.tasks[sid] = asyncio.create_task(self._run(record, owner))
            return record

    async def _run(self, record: SubagentRecord, owner: SubagentState) -> None:
        child_state = owner.child(record.id)
        try:
            await self.semaphore.acquire()
            record.slot_held = True
            record.status = "running"
            record.current_step = "Starting"
            record.started_at = time.time()
            await self.event_queue.put({**record.to_dict(), "stage": "started"})
            await self._execute_record(record, child_state)
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "Cancelled"
            raise
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
        finally:
            if record.slot_held:
                record.slot_held = False
                self.semaphore.release()
            if record.status not in {"done", "error", "cancelled"}:
                record.status = "error"
            record.ended_at = time.time()
            record.current_step = record.status.capitalize()
            await self.event_queue.put(
                {
                    **record.to_dict(),
                    "stage": "cancelled" if record.status == "cancelled" else "finished",
                    "ok": record.status == "done",
                }
            )

    async def _execute_record(self, record: SubagentRecord, child_state: SubagentState) -> None:
        retrieval_info: Dict[str, Any] = {}
        last_progress_at = 0.0
        stream: Any = None
        try:
            content: List[str] = []
            stream = idle_bounded_agen(
                self.rag_app.run_streaming(
                    record.task,
                    chat_history=[],
                    model_name=record.assistant,
                    retrieval_info=retrieval_info,
                    conversation_id=f"{self.root_conversation_id or 'chat'}.{record.id}",
                    subagent_state=child_state,
                    injected_input_queue=record.nudge_queue,
                    input_event=record.input_event,
                ),
                self.idle_seconds,
            )
            async for chunk in stream:
                text = chunk.get("content", "")
                if text:
                    content.append(text)
                    if record.current_step != "Writing answer":
                        record.current_step = "Writing answer"
                        await self.event_queue.put(
                            {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                        )
                iteration = chunk.get("agent_iteration")
                if iteration is not None:
                    try:
                        record.iterations = max(record.iterations, int(iteration))
                    except (TypeError, ValueError):
                        pass
                reasoning = chunk.get("reasoning", "")
                if reasoning:
                    record.reasoning_preview = (record.reasoning_preview + reasoning)[-240:]
                    # Fallback only: the backend emits "**Iteration N...**" as
                    # the sole content of its own chunk, so only a marker at
                    # the very start counts — model reasoning embedding the
                    # phrase mid-text (e.g. "Iteration 2025...") is rejected.
                    # agent_iteration above is authoritative when present.
                    if iteration is None:
                        match = re.match(
                            r"\*\*Iteration (\d{1,3})\.\.\.", reasoning.lstrip()
                        )
                        if match:
                            record.iterations = max(record.iterations, int(match.group(1)))
                    record.current_step = f"Iteration {record.iterations or 1}: thinking"
                    if time.time() - last_progress_at >= 1.0:
                        last_progress_at = time.time()
                        await self.event_queue.put(
                            {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                        )
                tool_calls = chunk.get("tool_calls") or []
                for call in tool_calls:
                    if call.get("tool_name") == "agent_todo":
                        continue
                    item = {
                        "iteration": call.get("iteration", record.iterations),
                        "tool_name": call.get("tool_name", "unknown"),
                        "args_preview": str(call.get("args") or "")[:120],
                        "status": "ok" if call.get("ok", True) else "error",
                        "duration_ms": call.get("duration_ms"),
                        "error": call.get("error"),
                        "output_preview": str(call.get("output") or "")[:200],
                    }
                    record.activity.append(item)
                    record.nested_tool_calls.append(call)
                    record.current_step = f"Tool {item['tool_name']}"
                    await self.event_queue.put(
                        {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                    )
                event = chunk.get("subagent")
                if event:
                    record.current_step = f"Subagent {event.get('id')} {event.get('stage')}"
                    await self.event_queue.put(
                        {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                    )
                tool_progress = chunk.get("tool_progress")
                if tool_progress:
                    record.current_step = f"Executing {tool_progress.get('tool_name', 'tool')}"
                    progress_payload: Dict[str, Any] = {
                        **record.to_dict(),
                        "stage": "progress",
                        "detail": record.current_step,
                        "activity_item": tool_progress,
                    }
                    # Forward a child's synthetic keepalives with the flag so
                    # they don't reset the parent's run-level idle watchdog.
                    if tool_progress.get("keepalive"):
                        progress_payload["keepalive"] = True
                    await self.event_queue.put(progress_payload)
                obs_event = chunk.get("observer")
                if isinstance(obs_event, dict):
                    record.observer_events.append(obs_event)
                    if len(record.observer_events) > 20:
                        record.observer_events = record.observer_events[-20:]
                    if obs_event.get("kind") == "action":
                        record.current_step = "Observer guidance"
                        await self.event_queue.put(
                            {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                        )
                injected = chunk.get("injected_user_message")
                if injected:
                    record.nudges.append(
                        {
                            "source": chunk.get("injected_source") or "user",
                            "text": str(injected)[:2000],
                            "at": time.time(),
                        }
                    )
                    if len(record.nudges) > 10:
                        record.nudges = record.nudges[-10:]
                    record.current_step = "Nudge received"
                    await self.event_queue.put(
                        {**record.to_dict(), "stage": "progress", "detail": record.current_step}
                    )
            record.answer = "".join(content).strip()
            record.status = "done"
            record.nested_usage = retrieval_info.get("_llm_usage_metadata") or {}
            trace = retrieval_info.get("_agent_trace") or {}
            if trace.get("tool_calls"):
                record.nested_tool_calls = [
                    call for call in trace["tool_calls"]
                    if call.get("tool_name") != "agent_todo"
                ]
        except StreamIdleTimeout:
            record.status = "error"
            record.error = f"No stream activity for {self.idle_seconds:g}s"
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "Cancelled"
            raise
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
        finally:
            # Close the inner generator deterministically (its finally runs
            # the child runner's cleanup) instead of relying on asyncgen GC.
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.aclose()

    def _owned_records(self, owner: SubagentState, ids: Optional[List[str]] = None) -> List[SubagentRecord]:
        selected = set(ids or [])
        return sorted(
            (
                record for record in self.records.values()
                if record.owner_key == owner.owner_key and (not selected or record.id in selected)
            ),
            key=lambda record: record.created_at,
        )

    async def wait(
        self,
        ids: Optional[List[str]],
        owner: SubagentState,
        mode: str = "all",
        input_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        records = self._owned_records(owner, ids)
        if not records:
            return {"content": "No matching subagents were spawned by this agent.", "metadata": {"subagent_results": []}}
        tasks = [self.tasks[r.id] for r in records if r.id in self.tasks and not self.tasks[r.id].done()]
        interrupted = False
        if tasks:
            owner_record = self.records.get(owner.parent_subagent_id or "")
            released_slot = bool(owner_record and owner_record.slot_held)
            if released_slot:
                owner_record.slot_held = False
                self.semaphore.release()
            event_waiter = (
                asyncio.create_task(input_event.wait())
                if input_event is not None
                else None
            )
            try:
                # Unbounded: every subagent is bounded by its own stream idle
                # watchdog, so this always terminates. mode="first" returns as
                # soon as the next selected task finishes; the rest keeps
                # running and is reported under "Still running". A set
                # input_event (new user or parent input) interrupts early so
                # the agent can react to it on its next iteration.
                pending_tasks: set = set(tasks)
                while pending_tasks:
                    wait_set = set(pending_tasks)
                    if event_waiter is not None:
                        wait_set.add(event_waiter)
                    done, pending_tasks = await asyncio.wait(
                        wait_set, return_when=asyncio.FIRST_COMPLETED
                    )
                    # The loop condition tracks CHILD tasks only — the still-
                    # pending event_waiter must not keep mode="all" waiting
                    # after every child has already finished.
                    pending_tasks.discard(event_waiter)
                    if event_waiter is not None and event_waiter in done:
                        interrupted = True
                        break
                    if mode == "first":
                        break
            except asyncio.CancelledError:
                raise
            else:
                if released_slot and owner_record and owner_record.status == "running":
                    await self.semaphore.acquire()
                    owner_record.slot_held = True
            finally:
                if event_waiter is not None:
                    event_waiter.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await event_waiter
        terminal = [r for r in records if r.status in {"done", "error", "cancelled"}]
        for record in terminal:
            record.collected = True
        report = "\n\n".join(
            record.format_dossier(max_answer_chars=self.report_answer_chars)
            for record in terminal
        )
        pending = [r.id for r in records if r.status in {"queued", "running"}]
        if pending:
            report += ("\n\n" if report else "") + "Still running: " + ", ".join(pending)
        metadata: Dict[str, Any] = {"subagent_results": [r.to_dict() for r in records]}
        if interrupted:
            report += "\n\n[Wait interrupted — new input was queued into your context.]"
            metadata["interrupted_by_input"] = True
        return {"content": report or "No subagent has finished yet.", "metadata": metadata}

    async def nudge(
        self,
        subagent_id: str,
        message: str,
        owner: SubagentState,
    ) -> Dict[str, Any]:
        """Queue mid-run guidance into one of the caller's subagents.

        The message lands in the child's injected-input queue and is delivered
        as a HumanMessage before its next iteration, also waking interruptible
        waits (``wait``/``wait_subagent``) inside that child. One-way only —
        the child cannot reply; the parent collects results via ``wait``.
        """
        records = self._owned_records(owner, [subagent_id])
        if not records:
            return {"content": f"Error: no subagent '{subagent_id}' was spawned by this agent."}
        record = records[0]
        if record.status in {"done", "error", "cancelled"}:
            return {
                "content": f"Subagent {record.id} already finished ({record.status}) — it cannot receive nudges.",
                "metadata": {"subagent": record.to_dict()},
            }
        record.nudge_queue.put_nowait(
            {
                "text": message,
                "content": f"[Message from parent agent]: {message}",
                "source": "parent",
            }
        )
        record.input_event.set()
        await self.event_queue.put(
            {**record.to_dict(), "stage": "progress", "detail": "Nudge received"}
        )
        return {
            "content": f"Nudge delivered to {record.id} ({record.label}).",
            "metadata": {"subagent": record.to_dict()},
        }

    def status(
        self,
        subagent_id: Optional[str],
        owner: SubagentState,
        full_answer: bool = False,
    ) -> Dict[str, Any]:
        ids = [subagent_id] if subagent_id else None
        records = self._owned_records(owner, ids)
        # Compact preview by default; full_answer returns the complete stored
        # answer (same cap as the serialized record metadata).
        answer_cap = 25_000 if full_answer else min(self.report_answer_chars, 1_500)
        report = "\n\n".join(record.format_dossier(max_answer_chars=answer_cap) for record in records)
        return {"content": report or "No matching subagents were spawned by this agent.",
                "metadata": {"subagent_results": [r.to_dict() for r in records]}}

    def uncollected_records(self, owner: SubagentState) -> List[SubagentRecord]:
        return [
            record for record in self._owned_records(owner)
            if record.status in {"queued", "running"} or not record.collected
        ]

    def backfill_tool_call_records(self, records: List[Any]) -> None:
        for call in records:
            if getattr(call, "tool_name", None) != "spawn_subagent":
                continue
            sid = (getattr(call, "metadata", None) or {}).get("subagent", {}).get("id")
            record = self.records.get(sid)
            if record is None:
                continue
            call.metadata = {"subagent": record.to_dict()}
            call.output = record.format_dossier(max_answer_chars=self.report_answer_chars)
            call.ok = record.status == "done"
            call.error = record.error
            call.ended_at = record.ended_at or time.time()

    def usage_totals(self) -> Dict[str, int]:
        totals = {
            key: 0
            for key in (
                "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
                "tool_result_tokens", "rag_tool_tokens", "cache_tokens", "observer_tokens",
                "compression_tokens", "compression_calls", "estimated_iterations",
                "llm_generation_ms",
            )
        }
        for record in self.records.values():
            for key in totals:
                try:
                    totals[key] += int(record.nested_usage.get(key) or 0)
                except (TypeError, ValueError):
                    pass
        return {key: value for key, value in totals.items() if value}

    async def shutdown(self) -> None:
        pending = [task for task in self.tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def make_subagent_state(
    manager: SubagentManager,
    assistant_name: str,
    workspace_ctx: Optional[Dict[str, Any]] = None,
) -> SubagentState:
    return SubagentState(
        manager, depth=0, is_root=True, assistant_name=assistant_name,
        owner_key="root", workspace_ctx=workspace_ctx,
    )
