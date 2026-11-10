"""Async scheduler that runs triggers at their scheduled times."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from croniter import croniter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.messages import MessageStore
from app.triggers_store import TriggerEntry, TriggersStore

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30.0


class TriggersScheduler:
    """Checks for due triggers and executes them using the shared RAGApplication."""

    def __init__(
        self,
        store: TriggersStore,
        rag_app: Any,
        message_store: Optional[MessageStore] = None,
    ) -> None:
        self.store = store
        self.rag_app = rag_app
        self.message_store = message_store
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._running: Set[str] = set()
        # Monotonic start timestamps per running trigger — powers the
        # dashboard live-activity strip without guessing from run records.
        self._running_since: Dict[str, float] = {}
        # Optional callable returning True while destructive maintenance
        # (e.g. a database restore) is in progress; trigger runs are skipped.
        self.maintenance_check: Optional[Any] = None

    async def start(self) -> None:
        if self._task is not None:
            return
        logger.info("Starting trigger scheduler loop.")
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        logger.info("Stopping trigger scheduler loop.")
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.check_and_run()
            except Exception:
                logger.exception("Trigger scheduler check_and_run failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=CHECK_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass

    async def check_and_run(self) -> None:
        if callable(self.maintenance_check) and self.maintenance_check():
            logger.debug("Skipping trigger check: maintenance mode active")
            return
        from app.redis_locks import acquire_leader

        if not await acquire_leader("triggers"):
            return
        due = await self.store.get_due_triggers()
        for trigger in due:
            if trigger.id in self._running:
                logger.debug("Trigger %s is already running, skipping", trigger.id)
                continue

            # Message-threshold triggers: check if enough new messages exist
            if trigger.schedule_type == "message_threshold":
                if not await self._threshold_reached(trigger):
                    # Not enough messages yet; just bump next_run_at
                    config = trigger.schedule_config or {}
                    check_min = int(config.get("check_interval_minutes", 5))
                    next_run = datetime.now(timezone.utc) + timedelta(minutes=check_min)
                    await self.store.update_next_run(trigger.id, next_run)
                    logger.debug(
                        "Trigger %s threshold not reached, rescheduled to %s",
                        trigger.id, next_run,
                    )
                    continue

            self._running.add(trigger.id)
            self._running_since[trigger.id] = time.monotonic()
            try:
                await self._run_trigger(trigger)
            finally:
                self._running.discard(trigger.id)
                self._running_since.pop(trigger.id, None)

    async def _threshold_reached(self, trigger: TriggerEntry) -> bool:
        if not self.message_store or not trigger.watch_assistants:
            return True
        config = trigger.schedule_config or {}
        threshold = int(config.get("message_threshold", 1))
        since = trigger.last_run_at or trigger.created_at or datetime.min.replace(tzinfo=timezone.utc)
        count = await self.message_store.count_messages_for_assistants(
            assistant_names=trigger.watch_assistants,
            since=since,
        )
        logger.debug(
            "Trigger %s message count: %d/%d since %s",
            trigger.id, count, threshold, since,
        )
        return count >= threshold

    async def _run_trigger(self, trigger: TriggerEntry) -> None:
        logger.info("Running trigger '%s' (%s)", trigger.name, trigger.id)
        conversation_id = f"trigger-{trigger.name}"
        run_id = await self.store.create_run(
            trigger_id=trigger.id,
            status="pending",
            conversation_id=conversation_id,
        )
        result_text: Optional[str] = None
        error_text: Optional[str] = None
        status = "success"
        retrieval_info: Dict[str, Any] = {}

        try:
            if not self.rag_app or not getattr(self.rag_app, "ready", False):
                raise RuntimeError("RAG application not ready")

            chat_history: List[BaseMessage] = []
            if (
                trigger.include_trigger_history
                and self.message_store is not None
            ):
                prev_messages = await self.message_store.get_conversation_messages(
                    conversation_id
                )
                for msg in prev_messages:
                    user_content = msg.get("user_content") or ""
                    assistant_content = msg.get("assistant_content") or ""
                    # Skip turns where the assistant produced no meaningful output
                    # (e.g. failed runs that were saved with empty content)
                    if not assistant_content.strip():
                        continue
                    if user_content:
                        chat_history.append(HumanMessage(content=user_content))
                    chat_history.append(AIMessage(content=assistant_content))
                # Respect max_history_turns: each turn = user + assistant message pair
                if trigger.max_history_turns > 0:
                    max_messages = trigger.max_history_turns * 2
                    if len(chat_history) > max_messages:
                        chat_history = chat_history[-max_messages:]
                logger.info(
                    "Trigger '%s' loaded %d previous turns as chat history",
                    trigger.name,
                    len(chat_history) // 2,
                )

            input_text = trigger.prompt
            if trigger.include_new_messages and trigger.watch_assistants and self.message_store:
                since = trigger.last_run_at or trigger.created_at or datetime.min.replace(tzinfo=timezone.utc)
                messages = await self.message_store.get_messages_for_assistants(
                    assistant_names=trigger.watch_assistants,
                    since=since,
                    limit=100,
                )
                if messages:
                    lines = ["Neue Nachrichten seit letztem Lauf:"]
                    # Group by assistant, then by conversation
                    from collections import OrderedDict
                    by_assistant: OrderedDict[str, OrderedDict[str, List[dict]]] = OrderedDict()
                    for msg in messages:
                        assistant = msg.get("model_name", "?")
                        conv_id = msg.get("conversation_id", "?")
                        if assistant not in by_assistant:
                            by_assistant[assistant] = OrderedDict()
                        if conv_id not in by_assistant[assistant]:
                            by_assistant[assistant][conv_id] = []
                        by_assistant[assistant][conv_id].append(msg)

                    for assistant, conversations in by_assistant.items():
                        lines.append(f"\nAssistent: {assistant}")
                        for conv_id, msgs in conversations.items():
                            lines.append(f"  Konversation: {conv_id}")
                            for msg in msgs:
                                ts = msg.get("created_at", "")
                                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                                lines.append(f"    [{ts_str}]")
                                if msg.get("user_content"):
                                    lines.append(f"      User: {msg['user_content']}")
                                if msg.get("assistant_content"):
                                    lines.append(f"      Assistant: {msg['assistant_content']}")
                                lines.append("      ---")
                    input_text = trigger.prompt + "\n\n" + "\n".join(lines)
                    logger.info(
                        "Trigger '%s' appended %d new messages to prompt",
                        trigger.name, len(messages),
                    )

            run_timeout = float(trigger.timeout_seconds or 600)
            if trigger.target_type == "chain":
                from app.agent_chains_store import AgentChainsStore
                engine = getattr(self.rag_app, "engine", None)
                if engine is None:
                    raise RuntimeError("Database unavailable for chain lookup")
                chain_store = AgentChainsStore(engine)
                await chain_store.ensure_tables()
                chain_entry = await chain_store.get_chain(trigger.target_name or trigger.assistant_name)
                if chain_entry is None:
                    raise ValueError(f"Chain '{trigger.target_name or trigger.assistant_name}' not found")
                if not chain_entry.enabled:
                    raise ValueError(f"Chain '{trigger.target_name or trigger.assistant_name}' is disabled")
                response = await asyncio.wait_for(
                    self.rag_app.run_chain(
                        chain_definition=chain_entry.flow_definition,
                        input_text=input_text,
                        chat_history=chat_history,
                        conversation_id=conversation_id,
                        execution_options=chain_entry.execution_options,
                        run_context={
                            "chain_id": chain_entry.id,
                            "chain_name": chain_entry.name,
                            "source": "trigger",
                            "conversation_id": conversation_id,
                        },
                    ),
                    timeout=run_timeout,
                )
            else:
                response = await asyncio.wait_for(
                    self.rag_app.run(
                        input_text=input_text,
                        chat_history=chat_history,
                        model_name=trigger.assistant_name,
                        retrieval_info=retrieval_info,
                        conversation_id=conversation_id,
                    ),
                    timeout=run_timeout,
                )

            # Extract answer text from response
            from app.chat_utils import extract_answer_text, extract_reasoning_text

            result_text = extract_answer_text(response)
            reasoning_text = extract_reasoning_text(response)

            # Fallback: if answer is empty but we have reasoning or agent trace
            tool_calls = None
            if isinstance(response, dict):
                trace = response.get("agent_trace")
                if trace:
                    tool_calls = trace.get("tool_calls")
                if not result_text:
                    result_text = response.get("reasoning") or ""
                    if not result_text:
                        if trace:
                            result_text = (
                                f"Agent completed with {trace.get('tool_call_count', 0)} tool calls. "
                                f"Finalization: {trace.get('finalization', 'unknown')}."
                            )

            if self.message_store is not None:
                await self.message_store.save_turn(
                    conversation_id=conversation_id,
                    user_content=trigger.prompt,
                    assistant_content=result_text or "",
                    assistant_reasoning=reasoning_text,
                    model_name=trigger.assistant_name,
                    retrieval_info=retrieval_info,
                    tool_calls=tool_calls,
                )

            # Only update last_run_at on success so that failed runs
            # don't cause new messages to be skipped.
            await self.store.update_last_run(
                trigger.id, datetime.now(timezone.utc)
            )
            next_run = _compute_next_run(trigger)
            await self.store.update_next_run(trigger.id, next_run)
            logger.info(
                "Trigger '%s' finished with status=%s, next_run=%s",
                trigger.name,
                status,
                next_run,
            )

        except asyncio.TimeoutError:
            timeout_s = int(float(trigger.timeout_seconds or 600))
            logger.error("Trigger '%s' timed out after %ss", trigger.name, timeout_s)
            status = "error"
            error_text = f"Trigger run timed out after {timeout_s}s"
            result_text = None
        except Exception as exc:
            logger.exception("Trigger '%s' execution failed", trigger.name)
            status = "error"
            error_text = str(exc)
            result_text = None

        await self.store.finish_run(
            run_id=run_id,
            status=status,
            result_text=result_text,
            error_text=error_text,
        )

    def running_now(self) -> Dict[str, float]:
        """Currently executing trigger ids -> monotonic start timestamps."""
        return dict(self._running_since)

    async def run_trigger_now(self, trigger_id: str) -> int:
        """Manually execute a trigger immediately. Returns run_id."""
        trigger = await self.store.get_trigger(trigger_id)
        if trigger is None:
            raise ValueError(f"Trigger {trigger_id} not found")
        self._running.add(trigger.id)
        self._running_since[trigger.id] = time.monotonic()
        try:
            await self._run_trigger(trigger)
        finally:
            self._running.discard(trigger.id)
            self._running_since.pop(trigger.id, None)
        # Fetch the run we just created to return its id
        runs = await self.store.list_runs(trigger_id, limit=1)
        return runs[0].id if runs else 0


def _compute_next_run(trigger: TriggerEntry, base: Optional[datetime] = None) -> Optional[datetime]:
    if base is None:
        base = trigger.last_run_at or trigger.created_at or datetime.now(timezone.utc)

    if trigger.schedule_type == "interval":
        config = trigger.schedule_config or {}
        minutes = int(config.get("minutes", 0))
        hours = int(config.get("hours", 0))
        days = int(config.get("days", 0))
        delta = timedelta(minutes=minutes, hours=hours, days=days)
        if delta.total_seconds() <= 0:
            delta = timedelta(minutes=5)
        return base + delta

    if trigger.schedule_type == "cron":
        cron_expr = trigger.schedule_config.get("cron", "")
        if not cron_expr:
            return None
        try:
            itr = croniter(cron_expr, base)
            return itr.get_next(datetime)
        except Exception:
            logger.warning("Invalid cron expression for trigger %s: %s", trigger.id, cron_expr)
            return None

    if trigger.schedule_type == "message_threshold":
        config = trigger.schedule_config or {}
        check_min = int(config.get("check_interval_minutes", 5))
        return base + timedelta(minutes=check_min)

    return None
