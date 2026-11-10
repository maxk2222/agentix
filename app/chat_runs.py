"""Registry for in-flight chat runs so streams can outlive their connection.

A run is produced by ``/v1/chat/completions`` streaming requests. The producer
task broadcasts tokens to every subscribed queue, so the original SSE
generator is just one consumer — when the client disconnects, the run keeps
going, finishes, and persists its turn. Clients reconnecting via
``GET /v1/chat/runs/{id}/stream`` receive a snapshot of what they missed and
then the remaining live tokens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import orjson

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# How long a finished run stays resumable before it is dropped.
_RUN_TTL_SECONDS = 120.0
# Redis state outlives the local TTL so cross-replica subscribers can still
# resume a just-finished run.
_STATE_TTL_SECONDS = int(_RUN_TTL_SECONDS + 900)
# Persist the resumable snapshot into Redis on every Nth token (and always on
# structural events) — full per-token writes would be wasteful.
_SNAPSHOT_EVERY_TOKENS = 50

_runs: Dict[str, "ActiveChatRun"] = {}


def _events_channel(conversation_id: str) -> str:
    return f"chatrun:{conversation_id}:events"


def _state_key(conversation_id: str) -> str:
    return f"chatrun:{conversation_id}:state"


def _done_key(conversation_id: str) -> str:
    return f"chatrun:{conversation_id}:done"


def _steer_channel(conversation_id: str) -> str:
    return f"chatrun:{conversation_id}:steer"


class ActiveChatRun:
    """One in-flight (or recently finished) streamed chat turn."""

    def __init__(
        self,
        *,
        conversation_id: str,
        model_name: str,
        user_message: str,
        retrieval_info: Optional[Dict[str, Any]] = None,
        emotion_data: Any = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.user_message = user_message
        self.retrieval_info: Dict[str, Any] = retrieval_info if retrieval_info is not None else {}
        self.emotion_data = emotion_data

        self.created_ts = int(time.time())
        self.started_perf = time.perf_counter()
        # Context needed to (re-)build trailer chunks for late subscribers.
        self.sys_instructions_rendered = ""
        self.history_for_metrics: List[Any] = []

        self.created_monotonic = time.monotonic()
        self.done = False
        self.finished_monotonic: Optional[float] = None
        self.stream_error: Optional[Exception] = None

        # Accumulated state — drives both persistence and resume snapshots.
        self.assistant_buffer: List[str] = []
        self.reasoning_buffer: List[str] = []
        self.tool_calls_buffer: List[Dict[str, Any]] = []
        self.tool_progress_buffer: List[Dict[str, Any]] = []
        self.subagent_events: List[Any] = []
        self.todo_list: Any = None
        # Run-observer events (check_started / assessment / action); capped so
        # a long run cannot grow the resume snapshot without bound.
        self.observer_events: List[Dict[str, Any]] = []

        self.produced_any = False
        self.produced_content = False
        self.first_token_at: Optional[float] = None
        self.first_answer_token_at: Optional[float] = None
        self.first_reasoning_token_at: Optional[float] = None
        self.stream_start = time.monotonic()

        # Mid-run user input injection ("steer"): messages queued here are
        # appended to the agent's message list at the next loop iteration.
        self.pending_inputs: asyncio.Queue = asyncio.Queue()
        # Set alongside every queued input so tools that block mid-iteration
        # (`wait`, `wait_subagent`) can race the event and return early.
        self.input_event: asyncio.Event = asyncio.Event()
        # sid -> SubagentRecord registry (filled by the run's SubagentManager)
        # so users can nudge any subagent in the tree via the /input endpoint.
        self.subagent_channels: Dict[str, Any] = {}
        self.injected_messages: List[Dict[str, str]] = []
        # Inputs left over when the run ends — broadcast live AND stashed here
        # so a client that missed the broadcast (disconnect/reload/abort) can
        # still recover them via GET /chat/runs/{id}.
        self.unconsumed_inputs: List[Dict[str, Any]] = []
        # Latest cumulative token snapshot emitted by the agent runner after
        # each completed LLM call — forwarded live and replayed on resume.
        self.live_usage: Optional[Dict[str, Any]] = None
        self.is_agent_run = False
        # Live-activity snapshot for the dashboard overview: ``source`` is set
        # once at registration, ``activity_phase``/``activity_detail`` track
        # the last observed produced token (see record_token).
        self.source: str = ""
        self.activity_phase: str = "starting"
        self.activity_detail: str = ""
        # Set when the conversation was deleted mid-run — _finalize must not
        # recreate the chat_user_conversations link after the delete.
        self.conversation_deleted = False

        self._subscribers: List[asyncio.Queue] = []
        self.done_event = asyncio.Event()
        self.cancelled = False
        self.cancel_event = asyncio.Event()
        # Redis bookkeeping — cross-replica fanout state.
        self._token_count = 0
        self._steer_task: Optional[asyncio.Task] = None

    # -- producer side -------------------------------------------------

    def cancel(self) -> None:
        """Request the producer to stop the run early (graceful stream close)."""
        self.cancelled = True
        self.cancel_event.set()

    def submit_input(self, item: Dict[str, Any]) -> bool:
        """Queue a user message for injection at the next agent iteration."""
        if self.done or not self.is_agent_run:
            return False
        self.pending_inputs.put_nowait(item)
        self.input_event.set()
        return True

    def submit_subagent_input(self, subagent_id: str, item: Dict[str, Any]) -> Optional[str]:
        """Queue a user nudge for one subagent of this run.

        Returns a reject reason string (``done``, ``not_agent``,
        ``no_subagent``, ``subagent_finished``) or ``None`` when accepted.
        """
        if self.done:
            return "done"
        if not self.is_agent_run:
            return "not_agent"
        record = self.subagent_channels.get(subagent_id)
        if record is None:
            return "no_subagent"
        if getattr(record, "status", "") in {"done", "error", "cancelled"}:
            return "subagent_finished"
        text = item.get("text") or ""
        nudge_queue = getattr(record, "nudge_queue", None)
        input_event = getattr(record, "input_event", None)
        if not isinstance(nudge_queue, asyncio.Queue) or not isinstance(
            input_event, asyncio.Event
        ):
            return "no_subagent"
        nudge_queue.put_nowait(
            {
                "text": text,
                "content": f"[User message]: {text}",
                "source": "user",
            }
        )
        input_event.set()
        return None

    def drain_pending_inputs(self) -> List[Dict[str, Any]]:
        """Pop every still-queued input (used to report unconsumed items)."""
        items: List[Dict[str, Any]] = []
        while True:
            try:
                items.append(self.pending_inputs.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def record_token(self, token: Any) -> None:
        """Accumulate buffers from one produced token (called by the producer)."""
        if not isinstance(token, dict):
            return
        self.produced_any = True
        content_text = token.get("content", "")
        reasoning_text = token.get("reasoning", "")
        if content_text:
            self.assistant_buffer.append(content_text)
            self.produced_content = True
            if self.first_answer_token_at is None:
                self.first_answer_token_at = time.monotonic()
        if reasoning_text:
            self.reasoning_buffer.append(reasoning_text)
            if self.first_reasoning_token_at is None:
                self.first_reasoning_token_at = time.monotonic()
        if (content_text or reasoning_text) and self.first_token_at is None:
            self.first_token_at = time.monotonic()
        for call in token.get("tool_calls") or []:
            call_id = call.get("id") if isinstance(call, dict) else None
            existing_idx = next(
                (
                    idx
                    for idx, existing in enumerate(self.tool_calls_buffer)
                    if call_id and existing.get("id") == call_id
                ),
                None,
            )
            if existing_idx is None:
                self.tool_calls_buffer.append(call)
            else:
                self.tool_calls_buffer[existing_idx] = {
                    **self.tool_calls_buffer[existing_idx],
                    **call,
                }
        tool_progress = token.get("tool_progress")
        if isinstance(tool_progress, dict):
            progress_id = tool_progress.get("id")
            if not any(
                progress_id and item.get("id") == progress_id
                for item in self.tool_progress_buffer
            ):
                self.tool_progress_buffer.append(tool_progress)
        subagent = token.get("subagent")
        if subagent:
            # Upsert by id — every event carries the full record dict, so an
            # append-only log would grow quadratically and flood resume
            # snapshots with stale copies. Latest state per subagent suffices.
            sid = subagent.get("id")
            for i, prev in enumerate(self.subagent_events):
                if isinstance(prev, dict) and prev.get("id") == sid:
                    self.subagent_events[i] = subagent
                    break
            else:
                self.subagent_events.append(subagent)
        if token.get("todo_list") is not None:
            self.todo_list = token["todo_list"]
        observer_event = token.get("observer")
        if isinstance(observer_event, dict):
            self.observer_events.append(observer_event)
            if len(self.observer_events) > 50:
                self.observer_events = self.observer_events[-50:]
        live_usage = token.get("live_usage")
        if isinstance(live_usage, dict):
            self.live_usage = live_usage
        injected = token.get("injected_user_message")
        if isinstance(injected, str) and injected:
            source = token.get("injected_source")
            self.injected_messages.append(
                {"text": injected, "source": source if isinstance(source, str) else ""}
            )
        # Live-activity phase for the dashboard overview — last observed
        # activity wins; token arrival order IS the run's chronology. Ordered
        # so the most telling signal wins within one token too.
        if reasoning_text:
            self.activity_phase, self.activity_detail = "reasoning", ""
        if content_text:
            self.activity_phase, self.activity_detail = "generating", ""
        for call in token.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            if call.get("output") is not None or call.get("ok") is not None:
                self.activity_phase, self.activity_detail = "results", ""
                break
            self.activity_phase = "tool"
            self.activity_detail = str(call.get("name") or "")
        if isinstance(tool_progress, dict):
            self.activity_phase = "tool"
            self.activity_detail = str(tool_progress.get("tool_name") or "")
        tool_execution = token.get("tool_execution")
        if isinstance(tool_execution, dict):
            self.activity_phase = "tool"
            self.activity_detail = str(tool_execution.get("tool") or "")
        if isinstance(observer_event, dict):
            self.activity_phase, self.activity_detail = "observer", ""

    def broadcast(self, token: Any) -> None:
        """Record a token and fan it out to every subscriber queue."""
        if token is not None:
            self.record_token(token)
        for queue in list(self._subscribers):
            queue.put_nowait(token)
        self._redis_broadcast(token)

    # -- Redis fanout ---------------------------------------------------

    def _redis_broadcast(self, token: Any) -> None:
        """Fire-and-forget publish — never blocks the local stream."""
        if get_redis() is None:
            return  # no task churn when Redis is off
        try:
            asyncio.get_running_loop().create_task(self._redis_publish(token))
        except RuntimeError:
            return

    async def _redis_publish(self, token: Any) -> None:
        r = get_redis()
        if r is None:
            return
        cid = self.conversation_id
        try:
            if token is not None:
                self._token_count += 1
                await r.publish(_events_channel(cid), orjson.dumps(token))
                # Persist the snapshot on structural events and periodically —
                # remote subscribers read it as the resume payload.
                structural = not isinstance(token, dict) or bool(
                    set(token) - {"content", "reasoning"}
                )
                if structural or self._token_count % _SNAPSHOT_EVERY_TOKENS == 0:
                    await self._redis_save_state(r)
        except Exception:
            logger.debug("redis publish failed for %s", cid, exc_info=True)

    async def _redis_save_state(self, r: Any) -> None:
        cid = self.conversation_id
        state = self.snapshot()
        await r.hset(
            _state_key(cid),
            mapping={
                "model_name": self.model_name or "",
                "done": "1" if self.done else "0",
                "data": orjson.dumps(state),
            },
        )
        await r.expire(_state_key(cid), _STATE_TTL_SECONDS)

    async def _redis_mark_done(self) -> None:
        r = get_redis()
        if r is None:
            return
        cid = self.conversation_id
        try:
            await self._redis_save_state(r)
            await r.set(_done_key(cid), "1", ex=_STATE_TTL_SECONDS)
            await r.publish(
                _events_channel(cid), orjson.dumps({"__done__": True})
            )
        except Exception:
            logger.debug("redis done-marker failed for %s", cid, exc_info=True)

    def finish(self, error: Optional[Exception] = None) -> None:
        """Mark the run finished and release all subscribers."""
        if self.done:
            return
        self.done = True
        self.finished_monotonic = time.monotonic()
        self.stream_error = error
        self.broadcast(None)
        if self._steer_task is not None:
            self._steer_task.cancel()
            self._steer_task = None
        try:
            asyncio.get_running_loop().create_task(self._redis_mark_done())
        except RuntimeError:
            pass
        self.done_event.set()

    # -- subscriber side -----------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        if self.done:
            queue.put_nowait(None)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    def snapshot(self) -> Dict[str, Any]:
        """Accumulated state for a resuming client."""
        return {
            "resumed": True,
            "user_message": self.user_message,
            "content": "".join(self.assistant_buffer),
            "reasoning_content": "".join(self.reasoning_buffer),
            "tool_calls": list(self.tool_calls_buffer),
            "tool_progress": list(self.tool_progress_buffer),
            "subagent_events": list(self.subagent_events),
            "todo_list": self.todo_list,
            "injected_messages": list(self.injected_messages),
            "observer_events": list(self.observer_events),
            "live_usage": self.live_usage,
        }


def _start_steer_listener(run: ActiveChatRun) -> None:
    """Subscribe this run's steer channel — steering messages published by any
    replica land in the local ``pending_inputs`` queue of the producing node.

    The /input endpoint publishes only when the run is not local, so a single
    message never double-delivers: local pushes go straight into the queue,
    remote ones arrive via this listener exactly once."""
    if get_redis() is None:
        return

    async def _listen() -> None:
        r = get_redis()
        if r is None:
            return
        pubsub = r.pubsub()
        try:
            await pubsub.subscribe(_steer_channel(run.conversation_id))
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    item = orjson.loads(message["data"])
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                subagent_id = item.get("subagent_id")
                if subagent_id:
                    run.submit_subagent_input(str(subagent_id), item)
                else:
                    run.submit_input(item)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "steer listener died for %s", run.conversation_id, exc_info=True
            )
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                pass

    run._steer_task = asyncio.create_task(_listen())


def register_run(run: ActiveChatRun) -> None:
    _runs[run.conversation_id] = run
    _start_steer_listener(run)


def get_run(conversation_id: str) -> Optional[ActiveChatRun]:
    run = _runs.get(conversation_id)
    if run is None:
        return None
    if run.done and run.finished_monotonic is not None:
        if time.monotonic() - run.finished_monotonic > _RUN_TTL_SECONDS:
            _runs.pop(conversation_id, None)
            return None
    return run


def unregister_run(conversation_id: str, run: Optional["ActiveChatRun"] = None) -> None:
    """Drop a run from the registry. When ``run`` is given, only remove the
    entry if it still points at that exact run — a lagging producer must not
    delete a newer run that re-registered the same conversation id."""
    if run is not None and _runs.get(conversation_id) is not run:
        return
    removed = _runs.pop(conversation_id, None)
    if removed is not None and removed._steer_task is not None:
        removed._steer_task.cancel()
        removed._steer_task = None


def active_conversation_ids() -> List[str]:
    """Raw conversation ids with a live (or within-TTL) run."""
    return [cid for cid in list(_runs) if get_run(cid) is not None]
