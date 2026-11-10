"""Shared per-conversation data cleanup.

Used by every conversation-delete path (admin chat, public chat) so no store
can silently be forgotten. Each step is isolated — one failing store never
aborts the rest.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def cleanup_conversation_data(app_state: Any, conversation_id: str) -> Dict[str, Any]:
    """Delete every persisted trace of *conversation_id*.

    Covers: conversation row, artifact dirs, chat uploads, agent memory rows,
    the live Docker sandbox container, and its writable-workspace directory.
    Returns a small report dict; failures are logged, never raised.
    """
    results: Dict[str, Any] = {}

    conv_store = getattr(app_state, "_chat_conversations_store", None)
    if conv_store is not None:
        try:
            await conv_store.delete(conversation_id)
            results["conversation_row"] = True
        except Exception:
            logger.exception("Conversation row delete failed for %s", conversation_id)

    for attr in ("artifact_store", "chat_attachment_store"):
        store = getattr(app_state, attr, None)
        if store is None:
            continue
        try:
            await store.delete_conversation(conversation_id)
            results[attr] = True
        except Exception:
            logger.exception("%s cleanup failed for %s", attr, conversation_id)

    # Agent memory rows keyed by (conversation_id, assistant_name).
    try:
        rag_app = getattr(app_state, "rag_application", None)
        engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
        if engine is not None:
            from app.stores import AgentMemoryStore

            deleted = await AgentMemoryStore(engine).delete_memories_for_conversation(conversation_id)
            results["agent_memory"] = deleted
    except Exception:
        logger.exception("Agent memory cleanup failed for %s", conversation_id)

    # Live sandbox container + per-conversation workspace dir.
    try:
        from agent.tools.docker_sandbox import (
            ConversationSandboxManager,
            remove_conversation_workspace,
        )
    except Exception:
        ConversationSandboxManager = None  # type: ignore[assignment]
        remove_conversation_workspace = None  # type: ignore[assignment]

    if ConversationSandboxManager is not None:
        try:
            await ConversationSandboxManager().destroy(conversation_id)
            results["sandbox_container"] = True
        except Exception:
            logger.debug("Sandbox destroy skipped/failed for %s", conversation_id, exc_info=True)
    if remove_conversation_workspace is not None:
        try:
            await asyncio.to_thread(remove_conversation_workspace, conversation_id)
            results["sandbox_workspace"] = True
        except Exception:
            logger.debug("Workspace cleanup failed for %s", conversation_id, exc_info=True)

    # Conversation-scoped agent workspace dirs — they live under
    # <assistant>/c_<conv_id>/ inside data/agent_workspaces, so every
    # assistant dir is checked (a conversation can involve several
    # assistants over its lifetime). shared/ and u_*/ dirs are unaffected.
    try:
        from app.agent_workspace import AgentWorkspaceStore
        from app.artifacts import sanitize_filename

        store = AgentWorkspaceStore()
        conv_leaf = f"c_{sanitize_filename(conversation_id)}"
        deleted = 0
        if store.root.is_dir():
            for assistant_dir in store.root.iterdir():
                candidate = assistant_dir / conv_leaf
                if assistant_dir.is_dir() and candidate.is_dir():
                    await asyncio.to_thread(store.delete_tree, candidate)
                    deleted += 1
        results["agent_workspace_dirs"] = deleted
    except Exception:
        logger.debug("Agent workspace cleanup failed for %s", conversation_id, exc_info=True)

    return results
