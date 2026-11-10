"""Tests for trigger scheduler logic including message_threshold and prompt appending."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.triggers_scheduler import TriggersScheduler, _compute_next_run
from app.triggers_store import TriggerEntry


class FakeStore:
    """Minimal fake store for scheduler tests."""

    def __init__(self):
        self.triggers: List[TriggerEntry] = []
        self.runs: List[Dict[str, Any]] = []

    async def get_due_triggers(self) -> List[TriggerEntry]:
        now = datetime.now(timezone.utc)
        return [
            t for t in self.triggers
            if t.enabled and t.next_run_at and t.next_run_at <= now
        ]

    async def create_run(self, trigger_id: str, status: str, conversation_id: str) -> int:
        run_id = len(self.runs) + 1
        self.runs.append({
            "id": run_id,
            "trigger_id": trigger_id,
            "status": status,
            "conversation_id": conversation_id,
        })
        return run_id

    async def update_run(self, run_id: int, *, status: str, result_text: Optional[str] = None,
                       error_text: Optional[str] = None, finished_at: Optional[datetime] = None) -> None:
        for run in self.runs:
            if run["id"] == run_id:
                run["status"] = status
                run["result_text"] = result_text
                run["error_text"] = error_text
                run["finished_at"] = finished_at
                break

    async def update_next_run(self, trigger_id: str, next_run_at: datetime) -> None:
        for t in self.triggers:
            if t.id == trigger_id:
                t.next_run_at = next_run_at
                break

    async def finish_run(self, run_id: int, status: str, result_text: Optional[str] = None,
                         error_text: Optional[str] = None) -> None:
        for run in self.runs:
            if run["id"] == run_id:
                run["status"] = status
                run["result_text"] = result_text
                run["error_text"] = error_text
                break

    async def update_last_run(self, trigger_id: str, last_run_at: datetime) -> None:
        for t in self.triggers:
            if t.id == trigger_id:
                t.last_run_at = last_run_at
                break


class FakeMessageStore:
    """Fake message store for counting/fetching messages."""

    def __init__(self, messages: List[Dict[str, Any]] = None, conversation_messages: List[Dict[str, Any]] = None):
        self.messages = messages or []
        self._conversation_messages = conversation_messages or []

    async def count_messages_for_assistants(
        self, *, assistant_names: List[str], since: datetime
    ) -> int:
        return sum(
            1 for m in self.messages
            if m.get("model_name") in assistant_names
            and m.get("created_at", datetime.min) > since
            and not str(m.get("conversation_id", "")).startswith("trigger-")
        )

    async def get_messages_for_assistants(
        self, *, assistant_names: List[str], since: datetime, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return [
            m for m in self.messages
            if m.get("model_name") in assistant_names
            and m.get("created_at", datetime.min) > since
            and not str(m.get("conversation_id", "")).startswith("trigger-")
        ][:limit]

    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self._conversation_messages

    async def save_turn(self, **kwargs: Any) -> None:
        pass


class FakeRagApp:
    """Fake RAG application that records inputs."""

    def __init__(self, raise_error: Exception | None = None):
        self.ready = True
        self.calls: List[Dict[str, Any]] = []
        self._raise_error = raise_error

    async def run(self, *, input_text: str, chat_history: List[Any], model_name: str,
                  retrieval_info: Dict[str, Any], conversation_id: Optional[str] = None) -> Any:
        if self._raise_error:
            raise self._raise_error
        self.calls.append({
            "input_text": input_text,
            "chat_history": chat_history,
            "model_name": model_name,
            "conversation_id": conversation_id,
        })
        return {"answer": "test response"}

    async def run_chain(self, *, chain_definition: Dict[str, Any], input_text: str,
                        chat_history: List[Any], conversation_id: Optional[str] = None) -> Any:
        if self._raise_error:
            raise self._raise_error
        self.calls.append({
            "input_text": input_text,
            "chat_history": chat_history,
            "conversation_id": conversation_id,
            "chain_definition": chain_definition,
        })
        return {"answer": "chain response"}


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def fake_rag():
    return FakeRagApp()


@pytest.fixture
def scheduler(fake_store, fake_rag):
    return TriggersScheduler(
        store=fake_store,
        rag_app=fake_rag,
        message_store=None,
    )


# ------------------------------------------------------------------------------
# _compute_next_run tests
# ------------------------------------------------------------------------------

def test_compute_next_run_interval():
    t = TriggerEntry(
        id="t1", name="test", assistant_name="a", prompt="p",
        schedule_type="interval", schedule_config={"minutes": 10}
    )
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = _compute_next_run(t, base)
    assert result == base + timedelta(minutes=10)


def test_compute_next_run_cron():
    t = TriggerEntry(
        id="t1", name="test", assistant_name="a", prompt="p",
        schedule_type="cron", schedule_config={"cron": "0 9 * * *"}
    )
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = _compute_next_run(t, base)
    # Next 09:00 after 12:00 is next day
    assert result == datetime(2025, 1, 2, 9, 0, 0, tzinfo=timezone.utc)


def test_compute_next_run_message_threshold():
    t = TriggerEntry(
        id="t1", name="test", assistant_name="a", prompt="p",
        schedule_type="message_threshold", schedule_config={"check_interval_minutes": 7}
    )
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = _compute_next_run(t, base)
    assert result == base + timedelta(minutes=7)


# ------------------------------------------------------------------------------
# Scheduler check_and_run with message_threshold
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_threshold_not_reached_reschedules_only(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="threshold-trigger", assistant_name="a", prompt="p",
            schedule_type="message_threshold",
            schedule_config={"check_interval_minutes": 5, "message_threshold": 3},
            next_run_at=now,
            watch_assistants=["a"],
        )
    ]
    msg_store = FakeMessageStore(messages=[])  # no messages
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 0  # Did not run
    assert fake_store.triggers[0].next_run_at > now  # Rescheduled


@pytest.mark.asyncio
async def test_message_threshold_reached_runs_trigger(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="threshold-trigger", assistant_name="a", prompt="p",
            schedule_type="message_threshold",
            schedule_config={"check_interval_minutes": 5, "message_threshold": 2},
            next_run_at=now,
            watch_assistants=["a"],
        )
    ]
    messages = [
        {"model_name": "a", "conversation_id": "conv-1", "created_at": now - timedelta(minutes=1), "user_content": "hi"},
        {"model_name": "a", "conversation_id": "conv-1", "created_at": now, "user_content": "hello"},
    ]
    msg_store = FakeMessageStore(messages=messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1  # Ran
    assert fake_store.runs[0]["status"] == "success"


# ------------------------------------------------------------------------------
# Prompt appending logic
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_include_new_messages_appended_to_prompt(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="msg-trigger", assistant_name="a", prompt="Original prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_new_messages=True,
            watch_assistants=["a"],
        )
    ]
    messages = [
        {"model_name": "a", "conversation_id": "conv-1", "created_at": now - timedelta(minutes=1),
         "user_content": "question", "assistant_content": "answer"},
    ]
    msg_store = FakeMessageStore(messages=messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    assert "Original prompt" in fake_rag.calls[0]["input_text"]
    assert "Neue Nachrichten seit letztem Lauf" in fake_rag.calls[0]["input_text"]
    assert "question" in fake_rag.calls[0]["input_text"]


@pytest.mark.asyncio
async def test_include_new_messages_excludes_trigger_turns(fake_store, fake_rag):
    """Messages from trigger conversation IDs must not be counted/appended."""
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="msg-trigger", assistant_name="a", prompt="Original prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_new_messages=True,
            watch_assistants=["a"],
        )
    ]
    messages = [
        {"model_name": "a", "conversation_id": "conv-1", "created_at": now - timedelta(minutes=2),
         "user_content": "real user", "assistant_content": "real answer"},
        {"model_name": "a", "conversation_id": "trigger-t1", "created_at": now - timedelta(minutes=1),
         "user_content": "trigger prompt", "assistant_content": "trigger answer"},
    ]
    msg_store = FakeMessageStore(messages=messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    text = fake_rag.calls[0]["input_text"]
    assert "real user" in text
    assert "trigger prompt" not in text


@pytest.mark.asyncio
async def test_include_new_messages_groups_by_assistant_and_conversation(fake_store, fake_rag):
    """New messages should be grouped by assistant then by conversation."""
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="msg-trigger", assistant_name="a", prompt="Original prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_new_messages=True,
            watch_assistants=["a", "b"],
        )
    ]
    messages = [
        {"model_name": "a", "conversation_id": "conv-a1", "created_at": now - timedelta(minutes=3),
         "user_content": "a1 user", "assistant_content": "a1 answer"},
        {"model_name": "a", "conversation_id": "conv-a2", "created_at": now - timedelta(minutes=2),
         "user_content": "a2 user", "assistant_content": "a2 answer"},
        {"model_name": "b", "conversation_id": "conv-b1", "created_at": now - timedelta(minutes=1),
         "user_content": "b1 user", "assistant_content": "b1 answer"},
    ]
    msg_store = FakeMessageStore(messages=messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    text = fake_rag.calls[0]["input_text"]
    assert "Assistent: a" in text
    assert "Assistent: b" in text
    assert "Konversation: conv-a1" in text
    assert "Konversation: conv-a2" in text
    assert "Konversation: conv-b1" in text
    assert "a1 user" in text
    assert "a2 user" in text
    assert "b1 user" in text


@pytest.mark.asyncio
async def test_include_new_messages_no_messages_prompt_unchanged(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="msg-trigger", assistant_name="a", prompt="Original prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_new_messages=True,
            watch_assistants=["a"],
        )
    ]
    msg_store = FakeMessageStore(messages=[])
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    assert fake_rag.calls[0]["input_text"] == "Original prompt"


@pytest.mark.asyncio
async def test_conversation_id_passed_to_rag_app(fake_store, fake_rag):
    """Regression test: conversation_id must be forwarded to rag_app.run()."""
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="memory-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
        )
    ]
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=None,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    assert fake_rag.calls[0]["conversation_id"] == "trigger-memory-trigger"


@pytest.mark.asyncio
async def test_include_trigger_history_passes_chat_history(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="history-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_trigger_history=True,
        )
    ]
    conversation_messages = [
        {"user_content": "old user msg", "assistant_content": "old assistant msg"},
        {"user_content": "another user", "assistant_content": "another assistant"},
    ]
    msg_store = FakeMessageStore(conversation_messages=conversation_messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    chat_history = fake_rag.calls[0]["chat_history"]
    assert len(chat_history) == 4  # 2 user + 2 assistant messages
    assert chat_history[0].content == "old user msg"
    assert chat_history[1].content == "old assistant msg"
    assert chat_history[2].content == "another user"
    assert chat_history[3].content == "another assistant"


@pytest.mark.asyncio
async def test_include_trigger_history_false_no_chat_history(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="history-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_trigger_history=False,
        )
    ]
    conversation_messages = [
        {"user_content": "old user msg", "assistant_content": "old assistant msg"},
    ]
    msg_store = FakeMessageStore(conversation_messages=conversation_messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    assert fake_rag.calls[0]["chat_history"] == []


@pytest.mark.asyncio
async def test_last_run_at_not_updated_on_error(fake_store):
    """Failed runs should not bump last_run_at so messages are re-evaluated next time."""
    now = datetime.now(timezone.utc)
    original_last_run = now - timedelta(hours=1)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="error-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            last_run_at=original_last_run,
        )
    ]
    fake_rag = FakeRagApp(raise_error=RuntimeError("boom"))
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=None,
    )

    await sched.check_and_run()

    assert fake_store.triggers[0].last_run_at == original_last_run
    assert fake_store.runs[0]["status"] == "error"


@pytest.mark.asyncio
async def test_trigger_history_skips_empty_assistant_content(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="history-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_trigger_history=True,
        )
    ]
    conversation_messages = [
        {"user_content": "good user", "assistant_content": "good assistant"},
        {"user_content": "failed run user", "assistant_content": ""},
        {"user_content": "another good user", "assistant_content": "another good assistant"},
    ]
    msg_store = FakeMessageStore(conversation_messages=conversation_messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    chat_history = fake_rag.calls[0]["chat_history"]
    # Only 2 good turns = 4 messages
    assert len(chat_history) == 4
    assert chat_history[0].content == "good user"
    assert chat_history[1].content == "good assistant"
    assert chat_history[2].content == "another good user"
    assert chat_history[3].content == "another good assistant"


@pytest.mark.asyncio
async def test_trigger_history_respects_max_history_turns(fake_store, fake_rag):
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="history-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
            include_trigger_history=True,
            max_history_turns=2,
        )
    ]
    conversation_messages = [
        {"user_content": "old user 1", "assistant_content": "old assistant 1"},
        {"user_content": "old user 2", "assistant_content": "old assistant 2"},
        {"user_content": "old user 3", "assistant_content": "old assistant 3"},
        {"user_content": "old user 4", "assistant_content": "old assistant 4"},
    ]
    msg_store = FakeMessageStore(conversation_messages=conversation_messages)
    sched = TriggersScheduler(
        store=fake_store, rag_app=fake_rag, message_store=msg_store,
    )

    await sched.check_and_run()

    assert len(fake_rag.calls) == 1
    chat_history = fake_rag.calls[0]["chat_history"]
    # max_history_turns=2 → only last 2 turns = 4 messages
    assert len(chat_history) == 4
    assert chat_history[0].content == "old user 3"
    assert chat_history[1].content == "old assistant 3"
    assert chat_history[2].content == "old user 4"
    assert chat_history[3].content == "old assistant 4"


@pytest.mark.asyncio
async def test_scheduler_rag_app_reference_updated_after_reload(fake_store):
    """Regression: after a hot-reload the scheduler must use the new RAGApplication."""
    now = datetime.now(timezone.utc)
    fake_store.triggers = [
        TriggerEntry(
            id="t1", name="reload-trigger", assistant_name="a", prompt="prompt",
            schedule_type="interval", schedule_config={"minutes": 10},
            next_run_at=now,
        )
    ]
    old_rag = FakeRagApp()
    sched = TriggersScheduler(store=fake_store, rag_app=old_rag, message_store=None)

    # Simulate shutdown of old RAG (as reload_rag_application does)
    old_rag.ready = False

    # Create new RAG and update scheduler reference (the fix under test)
    new_rag = FakeRagApp()
    sched.rag_app = new_rag

    await sched.check_and_run()

    assert len(new_rag.calls) == 1
    assert new_rag.calls[0]["input_text"] == "prompt"
    assert fake_store.runs[0]["status"] == "success"
