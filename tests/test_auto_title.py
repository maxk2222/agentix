"""Tests for the background auto-title task."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.auto_title import auto_title_if_needed


def _history(user_turns: int = 3):
    msgs = []
    for i in range(user_turns):
        msgs.append(HumanMessage(content=f"q{i}"))
        msgs.append(AIMessage(content=f"a{i}"))
    return msgs


def _app(conv):
    store = MagicMock()
    store.get_by_conversation_id = AsyncMock(return_value=conv)
    store.set_title = AsyncMock()
    store.ensure_exists = AsyncMock()
    app = SimpleNamespace(_chat_conversations_store=store, rag_application=MagicMock())
    app.rag_application.get_metadata_llm = AsyncMock(return_value=MagicMock())
    return app, store


@pytest.mark.asyncio
async def test_anonymous_over_5_messages_titled():
    app, store = _app({"user_id": "_anonymous", "title": None})
    with patch(
        "app.auto_title.generate_title", new=AsyncMock(return_value="🔒 Incident")
    ):
        await auto_title_if_needed(app, "c1", _history())  # 6 messages
    store.set_title.assert_awaited_once_with("c1", "🔒 Incident", 6)


@pytest.mark.asyncio
async def test_anonymous_under_5_skipped():
    app, store = _app({"user_id": "_anonymous", "title": None})
    with patch("app.auto_title.generate_title") as gen:
        await auto_title_if_needed(app, "c1", _history(2))  # 4 messages
    gen.assert_not_called()
    store.set_title.assert_not_called()


@pytest.mark.asyncio
async def test_missing_row_created_anonymously():
    app, store = _app(None)
    store.get_by_conversation_id = AsyncMock(
        side_effect=[None, {"user_id": "_anonymous", "title": None}]
    )
    with patch(
        "app.auto_title.generate_title", new=AsyncMock(return_value="🔒 Chat")
    ):
        await auto_title_if_needed(app, "c1", _history())
    store.ensure_exists.assert_awaited_once_with(
        user_id="_anonymous", conversation_id="c1"
    )
    store.set_title.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_conversation_static_title():
    app, store = _app(
        {"user_id": "_anonymous", "title": None, "model_name": "SecAgent"}
    )
    with patch("app.auto_title.generate_title") as gen:
        await auto_title_if_needed(app, "trigger-nightly-scan", _history(1))
    gen.assert_not_called()
    store.set_title.assert_awaited_once_with(
        "trigger-nightly-scan", "TRIGGER - nightly-scan/SecAgent", 2
    )


@pytest.mark.asyncio
async def test_manual_title_never_overwritten():
    app, store = _app({"user_id": "u1", "title": "Manual", "title_source": "manual"})
    with patch("app.auto_title.generate_title") as gen:
        await auto_title_if_needed(app, "c1", _history(8))
    gen.assert_not_called()
    store.set_title.assert_not_called()


@pytest.mark.asyncio
async def test_auto_title_regenerates_after_10_new_messages():
    app, store = _app(
        {"user_id": "u1", "title": "Old", "title_source": "auto", "auto_title_count": 4}
    )
    with patch(
        "app.auto_title.generate_title", new=AsyncMock(return_value="🔒 New Title")
    ):
        await auto_title_if_needed(app, "c1", _history(8))  # 16 > 4 + 10
    store.set_title.assert_awaited_once_with("c1", "🔒 New Title", 16)


@pytest.mark.asyncio
async def test_auto_title_stable_below_delta():
    app, store = _app(
        {"user_id": "u1", "title": "Old", "title_source": "auto", "auto_title_count": 10}
    )
    with patch("app.auto_title.generate_title") as gen:
        await auto_title_if_needed(app, "c1", _history(8))  # 16 < 10 + 10
    gen.assert_not_called()
    store.set_title.assert_not_called()


@pytest.mark.asyncio
async def test_first_title_for_user_conversation():
    app, store = _app({"user_id": "u1", "title": None})
    with patch(
        "app.auto_title.generate_title", new=AsyncMock(return_value=" 🔒 Incident")
    ):
        await auto_title_if_needed(app, "c1", _history())
    store.set_title.assert_awaited_once_with("c1", "🔒 Incident", 6)


@pytest.mark.asyncio
async def test_errors_are_swallowed():
    app, store = _app({"user_id": "u1", "title": None})
    store.get_by_conversation_id = AsyncMock(side_effect=RuntimeError("db down"))
    await auto_title_if_needed(app, "c1", _history())  # must not raise
