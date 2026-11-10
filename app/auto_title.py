"""Automatic conversation titles for logged-in users.

Scheduled fire-and-forget after a chat run finishes. Rules:

- Trigger conversations (``trigger-{name}`` ids) get a static
  ``TRIGGER - {name}/{model}`` title — never an LLM call.
- Conversations with a real ``user_id`` are titled once they have enough
  content, and auto-generated titles are refreshed once the conversation
  gained 10+ new messages since the last auto-title.
- Anonymous conversations (sentinel ``_anonymous`` row) are titled once
  they exceed 5 messages so long-lived anonymous chats stay findable.
- Manual titles (``title_source='manual'`` or any title without an auto
  source marker) are never overwritten.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from app.chat_conversations_store import ANON_USER_ID
from app.chat_utils import generate_title

logger = logging.getLogger(__name__)

# Minimum user turns before a title is worth generating (mirrors the
# "too thin" gate inside generate_title).
_MIN_USER_TURNS = 2
# Anonymous conversations are named only past this many messages.
_ANON_MIN_MESSAGES = 5
# Auto titles are refreshed once this many new messages arrived.
_REGEN_DELTA = 10
# Only the tail of the history is sent to the metadata LLM.
_HISTORY_TAIL = 10
_TITLE_MAX_LEN = 200

_TRIGGER_PREFIX = "trigger-"

# create_task keeps only a weak reference to the task — without a strong one
# the task can be garbage-collected before it ever runs.
_PENDING: set[asyncio.Task] = set()


def schedule_auto_title(
    app_state: Any,
    conversation_id: str,
    history: Sequence[BaseMessage],
) -> None:
    """Fire-and-forget entry point — never raises, never blocks the caller."""
    try:
        task = asyncio.create_task(
            auto_title_if_needed(app_state, conversation_id, list(history))
        )
    except Exception:
        logger.warning("auto-title scheduling failed for %s", conversation_id, exc_info=True)
        return
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def auto_title_if_needed(
    app_state: Any,
    conversation_id: str,
    history: Sequence[BaseMessage],
) -> None:
    try:
        conv_store = getattr(app_state, "_chat_conversations_store", None)
        rag_application = getattr(app_state, "rag_application", None)
        if conv_store is None or rag_application is None:
            return
        conv = await conv_store.get_by_conversation_id(conversation_id)
        if conv is None:
            await conv_store.ensure_exists(
                user_id=ANON_USER_ID, conversation_id=conversation_id
            )
            conv = await conv_store.get_by_conversation_id(conversation_id) or {}

        message_count = len(history)

        # Trigger conversations carry a deterministic name — no LLM needed.
        if conversation_id.startswith(_TRIGGER_PREFIX):
            trigger_name = conversation_id[len(_TRIGGER_PREFIX):]
            model_name = conv.get("model_name") or ""
            static = f"TRIGGER - {trigger_name}" + (f"/{model_name}" if model_name else "")
            if conv.get("title") != static[:_TITLE_MAX_LEN]:
                await conv_store.set_title(
                    conversation_id, static[:_TITLE_MAX_LEN], message_count
                )
            return

        existing = (conv.get("title") or "").strip()
        source = conv.get("title_source")
        if existing and source != "auto":
            return  # manual or externally set titles are never overwritten

        user_turns = sum(1 for m in history if isinstance(m, HumanMessage))
        anonymous = conv.get("user_id") in (None, ANON_USER_ID)

        if not existing:
            if anonymous:
                if message_count <= _ANON_MIN_MESSAGES:
                    return
            elif user_turns < _MIN_USER_TURNS and message_count < 4:
                return
        else:
            # Refresh an auto-generated title once the conversation grew.
            last_count = conv.get("auto_title_count") or 0
            if message_count <= _REGEN_DELTA or message_count - last_count < _REGEN_DELTA:
                return

        metadata_llm = await rag_application.get_metadata_llm(purpose="title")
        title = await generate_title(metadata_llm, list(history)[-_HISTORY_TAIL:])
        title = (title or "").strip()[:_TITLE_MAX_LEN]
        if title:
            await conv_store.set_title(conversation_id, title, message_count)
    except Exception:
        logger.warning("auto-title failed for %s", conversation_id, exc_info=True)
