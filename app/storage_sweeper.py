"""Periodic cleanup for chat-scoped file storage.

Two roots hold per-conversation directories:

- ``data/artifacts``           — generated documents, previews, drafts (ArtifactStore)
- ``data/chat_uploads``        — user attachments + extracted text (ChatAttachmentStore)
- ``data/sandbox_workspaces``  — per-conversation ``writable_workspace`` dirs

Additionally: orphan Docker sandbox containers (labeled ``agentix.sandbox.*``)
whose conversation row is gone, and ``agent_memories`` rows without a live
conversation (grace via ``updated_at`` — memory can be written mid-first-run
before the lazy row exists).

Conversations deleted through the API clean their dirs via
``delete_conversation`` — but several paths leave orphans: admin-chat
conversation resets used to drop only the client-side id, uploads to
conversations that never completed a first message (rows are created lazily
after the first message), failed runs, and pre-feature leftovers.

Sweep rules:

- A directory **with** a ``chat_conversations`` row is never touched —
  files of live conversations are user data.
- A directory **without** a row is deleted once it is older than
  ``chat_files_orphan_ttl_days`` (default 7). The grace period covers
  uploads that happen before the first message creates the row.
- Optionally, ``chat_files_retention_days`` > 0 additionally deletes the
  *files* of conversations whose row is older (row + message history are
  kept — files are not). Default 0 = off.
- A conversation with an active ``ActiveChatRun`` is always skipped.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ORPHAN_TTL_DAYS = 7
_SWEEP_INTERVAL_SECONDS = 24 * 60 * 60


def _dir_age_days(path: Path) -> float:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return 0.0
    return max(0.0, (time.time() - mtime) / 86400.0)


def _row_age_days(value: Any) -> Optional[float]:
    """Age of a conversation row in days from its ``updated_at`` value."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


async def _delete_dir(path: Path) -> bool:
    try:
        await asyncio.to_thread(shutil.rmtree, path, True)
        return True
    except Exception as exc:
        logger.warning("Storage sweep failed to remove %s: %s", path, exc)
        return False


async def _sweep_root(
    root: Path,
    live_ids: Optional[set],
    updated_at_map: Dict[str, Any],
    active_ids: set,
    *,
    orphan_ttl_days: float,
    retention_days: float,
    stats: Dict[str, int],
) -> None:
    if not root.is_dir():
        return
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("Storage sweep cannot list %s: %s", root, exc)
        return
    for entry in entries:
        if not entry.is_dir():
            continue
        conv_id = entry.name
        stats["scanned"] += 1
        if conv_id in active_ids:
            stats["skipped_active"] += 1
            continue
        if live_ids is None:
            stats["errors"] += 1  # no row data — never delete on doubt
            continue
        if conv_id in live_ids:
            # Live conversation: only an explicit retention policy may purge
            # its files — the row and message history stay regardless.
            if retention_days > 0:
                age = _row_age_days(updated_at_map.get(conv_id))
                if age is not None and age > retention_days:
                    if await _delete_dir(entry):
                        stats["deleted_retention"] += 1
            continue
        # Row-less dir: candidate once it outlives the grace period. The TTL
        # covers the upload-before-first-message window (rows are created
        # lazily on the first completed message).
        if _dir_age_days(entry) > orphan_ttl_days:
            if await _delete_dir(entry):
                stats["deleted_orphan"] += 1


async def _sweep_sandbox_containers(
    live_ids_raw: Optional[set],
    active_ids_raw: set,
    *,
    stats: Dict[str, int],
) -> None:
    """Destroy orphan sandbox containers whose conversation row is gone.

    Uses the container labels — the manager's in-memory cache misses
    containers that outlived a backend restart. Grace: only containers older
    than MAX_TTL are touched, covering the create-before-first-row window.
    """
    if live_ids_raw is None:
        return
    try:
        from agent.tools.docker_sandbox import (
            ConversationSandboxManager,
            MAX_TTL_SECONDS,
        )
    except Exception:
        return
    try:
        manager = ConversationSandboxManager()
        client = await manager._ensure_client()
        containers = await asyncio.to_thread(
            client.containers.list, all=True, filters={"label": "agentix.sandbox.managed=true"}
        )
    except Exception as exc:
        logger.debug("Sandbox orphan scan skipped: %s", exc)
        return
    now = time.time()
    for container in containers:
        # NOTE: container.labels reads attrs['Config']['Labels'] and RAISES for
        # sparse list() objects — list responses carry Labels at top level.
        attrs = container.attrs or {}
        labels = attrs.get("Labels") or (attrs.get("Config") or {}).get("Labels") or {}
        conv_id = labels.get("agentix.sandbox.conversation_id") or ""
        if not conv_id or conv_id in live_ids_raw or conv_id in active_ids_raw:
            continue
        # containers.list() reports "Created" as epoch int; inspect gives ISO.
        created = attrs.get("Created")
        if isinstance(created, (int, float)):
            age = max(0.0, (now - float(created)) / 86400.0)
        else:
            created_at = str(created or "").replace("Z", "+00:00")
            age = _row_age_days(created_at)
        if age is None or age * 86400.0 <= MAX_TTL_SECONDS:
            continue  # too young or unparseable — never kill on doubt
        try:
            await manager.destroy(conv_id)
            stats["deleted_sandbox_container"] += 1
        except Exception as exc:
            logger.debug("Orphan sandbox destroy failed for %s: %s", conv_id, exc)


async def _sweep_memory_orphans(
    app_state: Any,
    live_ids_raw: Optional[set],
    active_ids_raw: set,
    *,
    orphan_ttl_days: float,
    stats: Dict[str, int],
) -> None:
    """Delete agent_memories rows for conversations that no longer exist."""
    if live_ids_raw is None:
        return
    rag_app = getattr(app_state, "rag_application", None)
    engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
    if engine is None:
        return
    try:
        from app.stores import AgentMemoryStore

        store = AgentMemoryStore(engine)
        index = await store.memory_conversation_index()
    except Exception as exc:
        logger.debug("Memory orphan scan skipped: %s", exc)
        return
    for conv_id, latest in index.items():
        if conv_id in live_ids_raw or conv_id in active_ids_raw:
            continue
        age = _row_age_days(latest)
        if age is None or age <= orphan_ttl_days:
            continue  # grace: mid-first-run memory before the lazy row exists
        try:
            deleted = await store.delete_memories_for_conversation(conv_id)
            stats["deleted_memory_orphans"] += int(deleted or 0)
        except Exception as exc:
            logger.debug("Memory orphan delete failed for %s: %s", conv_id, exc)


async def _sweep_agent_workspaces(
    app_state: Any,
    live_ids: Optional[set],
    *,
    orphan_ttl_days: float,
    stats: Dict[str, int],
) -> None:
    """Sweep ``data/agent_workspaces`` (two-level: <assistant>/<scope>/).

    - An assistant dir whose assistant no longer exists in the instructions
      is an orphan — deleted after the TTL (covers renames/removals).
    - Inside a live assistant dir, ``c_<conv>`` dirs follow the conversation
      rule: deleted once the conv row is gone AND the dir outlives the TTL.
      ``live_ids`` must hold SANITIZED ids — dir names come from
      ``sanitize_filename``, raw-id lookup would orphan live dirs.
    - ``shared/`` and ``u_*/`` dirs are persistent user data — never swept
      while the assistant lives.
    """
    try:
        from app.agent_workspace import AgentWorkspaceStore
    except Exception:
        return
    root = AgentWorkspaceStore().root
    if not root.is_dir():
        return
    rag_app = getattr(app_state, "rag_application", None)
    live_slugs = {
        AgentWorkspaceStore.assistant_slug(name)
        for name in (getattr(rag_app, "instructions", None) or {})
    }
    try:
        assistant_dirs = list(root.iterdir())
    except OSError:
        return
    for assistant_dir in assistant_dirs:
        if not assistant_dir.is_dir():
            continue
        stats["scanned"] += 1
        if assistant_dir.name not in live_slugs:
            if _dir_age_days(assistant_dir) > orphan_ttl_days:
                if await _delete_dir(assistant_dir):
                    stats["deleted_workspace_orphans"] += 1
            continue
        if live_ids is None:
            continue
        try:
            scope_dirs = list(assistant_dir.iterdir())
        except OSError:
            continue
        for scope_dir in scope_dirs:
            if not scope_dir.is_dir() or not scope_dir.name.startswith("c_"):
                continue
            stats["scanned"] += 1
            conv_id = scope_dir.name[2:]
            if conv_id in live_ids:
                continue  # live conversation — user data
            if _dir_age_days(scope_dir) > orphan_ttl_days:
                if await _delete_dir(scope_dir):
                    stats["deleted_workspace_orphans"] += 1


async def sweep_chat_storage(app_state: Any) -> Dict[str, int]:
    """Sweep ``artifacts`` and ``chat_uploads`` roots for orphaned dirs.

    Resolves stores and runtime config from ``app.state``; safe to call any
    time — every directory decision is conservative (never deletes on doubt).
    """
    from app.config_access import effective_config_value_for

    orphan_ttl_days = float(
        await effective_config_value_for(
            app_state, "chat_files_orphan_ttl_days", _DEFAULT_ORPHAN_TTL_DAYS
        )
        or _DEFAULT_ORPHAN_TTL_DAYS
    )
    retention_days = float(
        await effective_config_value_for(app_state, "chat_files_retention_days", 0) or 0
    )
    conv_store = getattr(app_state, "_chat_conversations_store", None)
    if conv_store is None:
        # Same lazy construction as the public-chat route — the store only
        # needs the DB engine and may not exist yet at first startup sweep.
        rag_app = getattr(app_state, "rag_application", None)
        engine = getattr(rag_app, "engine", None) or getattr(rag_app, "_engine", None)
        if engine is not None:
            try:
                from app.chat_conversations_store import ChatConversationsStore

                conv_store = ChatConversationsStore(engine)
                app_state._chat_conversations_store = conv_store
            except Exception as exc:
                logger.warning("Storage sweep: cannot create conversations store: %s", exc)
    stats: Dict[str, int] = {
        "scanned": 0,
        "deleted_orphan": 0,
        "deleted_retention": 0,
        "deleted_sandbox_container": 0,
        "deleted_memory_orphans": 0,
        "deleted_workspace_orphans": 0,
        "skipped_active": 0,
        "errors": 0,
    }
    from app.artifacts import sanitize_filename
    from app.chat_runs import active_conversation_ids

    # Active runs are keyed by RAW conversation id; dir names are sanitized —
    # keep both views so exotic ids are still protected.
    active_ids_raw = set(active_conversation_ids())
    active_ids = {sanitize_filename(cid) for cid in active_ids_raw}
    # One query: set of SANITIZED live conversation ids — directory names are
    # produced by sanitize_filename(), so the comparison must apply the same
    # transform (a raw-id lookup would miss sanitized ids and orphan live dirs).
    live_ids: Optional[set] = None
    live_ids_raw: Optional[set] = None
    updated_at_map: Dict[str, Any] = {}
    if conv_store is not None:
        try:
            rows = await conv_store.get_updated_at_map()
            live_ids_raw = set(rows)
            updated_at_map = {
                sanitize_filename(cid): ts for cid, ts in rows.items()
            }
            live_ids = set(updated_at_map)
        except Exception as exc:
            logger.warning("Storage sweep: conversation lookup failed — skipping all deletes: %s", exc)
    else:
        logger.warning("Storage sweep: no conversations store — skipping all deletes")
    # Roots mirror main.py's store construction (stores keep _root private —
    # deriving from config is stable across their internals).
    app_config = getattr(app_state, "app_config", None)
    data_dir = getattr(app_config, "data_dir", None)
    if data_dir is not None:
        roots = [Path(data_dir) / sub for sub in ("artifacts", "chat_uploads")]
        try:
            from agent.tools.docker_sandbox import _workspace_root

            roots.append(_workspace_root())
        except Exception:
            pass
        for root in roots:
            await _sweep_root(
                root,
                live_ids,
                updated_at_map,
                active_ids,
                orphan_ttl_days=orphan_ttl_days,
                retention_days=retention_days,
                stats=stats,
            )
    await _sweep_sandbox_containers(live_ids_raw, active_ids_raw, stats=stats)
    await _sweep_agent_workspaces(
        app_state, live_ids, orphan_ttl_days=orphan_ttl_days, stats=stats
    )
    await _sweep_memory_orphans(
        app_state, live_ids_raw, active_ids_raw, orphan_ttl_days=orphan_ttl_days, stats=stats
    )
    if any(stats[k] for k in ("deleted_orphan", "deleted_retention", "deleted_sandbox_container", "deleted_memory_orphans", "deleted_workspace_orphans")):
        logger.info("Chat storage sweep: %s", stats)
    return stats


async def storage_sweeper_loop(app_state: Any) -> None:
    """Background loop: sweep on start, then daily. Cancelled on shutdown."""
    while True:
        try:
            await sweep_chat_storage(app_state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Chat storage sweep failed: %s", exc)
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
