"""Run-level idle watchdog: only *real* stream items reset the deadline.

Synthetic tool_progress keepalives keep the SSE wire and the dashboard
progress display alive but must not shield a stuck run forever — after
``run_idle_timeout_seconds`` without real activity the producer aborts.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.routers.chat import _is_stream_keepalive, chat_endpoint
from app.schemas import ChatRequest, Message


class FakeMessageStore:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def save_turn(self, **kwargs) -> None:
        self.saved.append(kwargs)


class FakePublicChatConfigStore:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config

    async def get(self):
        return self._config

    async def save(self, config: dict, created_by: str = "admin") -> None:
        pass


class FakeChatRoleAssistantsStore:
    def __init__(self, assistants: list[str] | None = None) -> None:
        self._assistants = assistants or []

    async def get_assistants_for_role(self, role: str):
        return list(self._assistants)


class KeepaliveOnlyRAG:
    """Fake app whose stream yields only flagged keepalives forever."""

    ready = True
    instructions = {"CyberSecurityAssistant": {"model": "gpt-4o-mini"}}
    engine = object()

    async def refresh_instructions_if_needed(self) -> bool:
        return False

    async def run(self, user_message, *, chat_history, model_name, retrieval_info=None, **kwargs):
        return {"answer": f"Echo: {user_message}"}

    async def get_metadata_llm(self, model_name=None, purpose=None):
        return None

    async def run_streaming(self, user_message, *, chat_history, model_name, retrieval_info=None, **kwargs):
        while True:
            yield {
                "content": "",
                "reasoning": "",
                "tool_progress": {
                    "id": "call_1",
                    "tool_name": "code_execution",
                    "stage": "progress",
                    "keepalive": True,
                },
                "keepalive": True,
            }
            await asyncio.sleep(0.01)


class MixedRAG(KeepaliveOnlyRAG):
    """Keepalives with periodic real content — the run must survive."""

    async def run_streaming(self, user_message, *, chat_history, model_name, retrieval_info=None, **kwargs):
        for _ in range(6):
            yield {
                "content": "",
                "reasoning": "",
                "tool_progress": {"id": "call_1", "tool_name": "code_execution", "stage": "progress", "keepalive": True},
                "keepalive": True,
            }
            await asyncio.sleep(0.01)
            yield {"content": "real", "reasoning": ""}
            await asyncio.sleep(0.05)
        retrieval_info["_llm_usage_metadata"] = {"total_tokens": 3}


def _request(rag_app, run_idle: float):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rag_application=rag_app,
                app_config=SimpleNamespace(
                    dashboard_session_secret="test",
                    stream_idle_timeout_seconds=30.0,
                    run_idle_timeout_seconds=run_idle,
                ),
                _message_store=FakeMessageStore(),
            )
        ),
        state=SimpleNamespace(),
        headers={"content-type": "application/json"},
        cookies={},
    )


def _patch_stores(monkeypatch):
    monkeypatch.setattr(
        "app.routers.chat.PublicChatConfigStore",
        lambda engine: FakePublicChatConfigStore(
            {"enabled": True, "enabled_assistants": ["CyberSecurityAssistant"]}
        ),
    )
    monkeypatch.setattr(
        "app.chat_group_access.ChatRoleAssistantsStore",
        lambda engine: FakeChatRoleAssistantsStore(["CyberSecurityAssistant"]),
    )


def test_is_stream_keepalive_predicate():
    assert _is_stream_keepalive({"keepalive": True, "tool_progress": {"stage": "progress"}})
    # Real items never carry the flag.
    assert not _is_stream_keepalive({"content": "hi"})
    assert not _is_stream_keepalive({"reasoning": "..."})
    assert not _is_stream_keepalive({"tool_progress": {"stage": "started"}})
    assert not _is_stream_keepalive({"tool_progress": {"stage": "progress"}})
    assert not _is_stream_keepalive({"tool_execution": {"tool": "code_execution", "stage": "result"}})
    assert not _is_stream_keepalive({"subagent": {"id": "sub-1", "stage": "progress"}})
    assert not _is_stream_keepalive({"observer": {"kind": "nudge"}})
    assert not _is_stream_keepalive({"todo_list": []})
    assert not _is_stream_keepalive("not-a-dict")
    assert not _is_stream_keepalive(None)


@pytest.mark.asyncio
async def test_producer_aborts_keepalive_only_stream(monkeypatch):
    """A run producing nothing but keepalives must die after run_idle — the
    keepalives keep the wire alive but don't reset the watchdog."""
    _patch_stores(monkeypatch)
    request = _request(KeepaliveOnlyRAG(), run_idle=0.3)
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
    )

    response = await chat_endpoint(chat_request, request)

    body = ""
    async def _consume():
        nonlocal body
        async for raw in response.body_iterator:
            body += raw if isinstance(raw, str) else raw.decode()

    await asyncio.wait_for(_consume(), timeout=20)

    assert "data: [DONE]" in body
    # Keepalives still reached the client (wire stayed alive until abort).
    assert '"tool_progress"' in body
    # No real content was produced → the run-level fallback answered instead.
    assert "Echo: Hallo" in body


@pytest.mark.asyncio
async def test_real_items_reset_run_idle(monkeypatch):
    """Keepalives plus periodic real tokens keep the run alive past idle."""
    _patch_stores(monkeypatch)
    request = _request(MixedRAG(), run_idle=0.15)
    chat_request = ChatRequest(
        model="CyberSecurityAssistant",
        messages=[Message(role="user", content="Hallo")],
        stream=True,
    )

    response = await chat_endpoint(chat_request, request)

    body = ""
    async def _consume():
        nonlocal body
        async for raw in response.body_iterator:
            body += raw if isinstance(raw, str) else raw.decode()

    await asyncio.wait_for(_consume(), timeout=20)
    # The run took ~0.4s total (> idle 0.15) yet completed because real items
    # reset the deadline each time.
    assert '"finish_reason":"stop"' in body or '"finish_reason": "stop"' in body
