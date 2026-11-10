"""Generic MCP (Model Context Protocol) client layer.

Connects agentix to external MCP servers (e.g. ``workspace-mcp`` for Google
Workspace) over stdio subprocesses or streamable-HTTP endpoints. Remote tools
discovered via ``tools/list`` are materialized into the tools store as
``kind="mcp"`` entries, so they appear in the dashboard Tools page and can be
enabled/assigned to assistants like any other tool.

Design: :class:`MCPToolProxy` is a dumb proxy built from a *store entry* (no
live session needed — the persisted schema survives restarts and outages).
:class:`MCPClientManager` owns sessions, discovery, materialization and call
routing, with lazy reconnect and a TTL-based store resync.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_TOOL_NAME = 64  # OpenAI function-name limit
_DEFAULT_CALL_TIMEOUT = 60
_CONNECT_TIMEOUT = 20
_RESYNC_TTL_SECONDS = 30.0
_MAX_RESULT_CHARS = 50_000
_MCP_SERVERS_ENTRY = "mcp_servers"


def _sanitize_part(value: str) -> str:
    """Make a name fragment function-name-safe (no ``__`` sequences)."""
    clean = _NAME_SAFE.sub("_", str(value or "")).replace("__", "_").strip("_")
    return clean or "x"


def mcp_tool_name(server: str, remote: str) -> str:
    """Build the agentix tool name for a remote tool (≤64 chars, hash-suffixed)."""
    name = f"mcp__{_sanitize_part(server)}__{_sanitize_part(remote)}"
    if len(name) > _MAX_TOOL_NAME:
        # Non-cryptographic use: name-shortening only — sha256 keeps SAST quiet.
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        name = f"{name[: _MAX_TOOL_NAME - 9]}_{digest}"
    return name


@dataclass
class MCPServerConfig:
    """One entry in the ``servers`` list of the ``mcp_servers`` tool config."""

    name: str
    transport: str = "stdio"  # "stdio" | "http"
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    tool_filter: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)
    call_timeout: int = _DEFAULT_CALL_TIMEOUT

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MCPServerConfig":
        def _str_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(v) for v in value]

        def _str_dict(value: Any) -> Dict[str, str]:
            if not isinstance(value, dict):
                return {}
            return {str(k): str(v) for k, v in value.items()}

        try:
            timeout = int(raw.get("call_timeout") or _DEFAULT_CALL_TIMEOUT)
        except (TypeError, ValueError):
            timeout = _DEFAULT_CALL_TIMEOUT
        return cls(
            name=str(raw.get("name") or "").strip(),
            transport=str(raw.get("transport") or "stdio").strip().lower(),
            command=str(raw.get("command") or "").strip(),
            args=_str_list(raw.get("args")),
            env=_str_dict(raw.get("env")),
            url=str(raw.get("url") or "").strip(),
            headers=_str_dict(raw.get("headers")),
            enabled=bool(raw.get("enabled", True)),
            tool_filter=_str_list(raw.get("tool_filter")),
            deny=_str_list(raw.get("deny")),
            call_timeout=max(5, min(600, timeout)),
        )

    def fingerprint(self) -> str:
        blob = {
            "transport": self.transport, "command": self.command,
            "args": self.args, "env": self.env, "url": self.url,
            "headers": self.headers,
        }
        return hashlib.sha256(
            json.dumps(blob, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def allows(self, remote_name: str) -> bool:
        """Filter on the *original* remote tool name (admin-intuitive)."""
        if self.tool_filter and not any(
            fnmatch.fnmatch(remote_name, p) for p in self.tool_filter
        ):
            return False
        if any(fnmatch.fnmatch(remote_name, p) for p in self.deny):
            return False
        return True


async def _default_session_factory(
    cfg: MCPServerConfig, stack: AsyncExitStack
) -> Any:
    """Open an MCP ``ClientSession`` on the stack; caller keeps stack alive."""
    from mcp import ClientSession

    if cfg.transport == "http":
        from mcp.client.streamable_http import streamable_http_client

        streams = await stack.enter_async_context(
            streamable_http_client(cfg.url, headers=cfg.headers or None)
        )
        session = await stack.enter_async_context(
            ClientSession(streams[0], streams[1])
        )
    else:
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        # StdioServerParameters.env REPLACES the child env when set — a partial
        # admin map (just secrets) would wipe PATH and break e.g. `uvx`. Merge
        # over the SDK's safe minimal environment instead.
        merged_env = {**get_default_environment(), **cfg.env}
        rw = await stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=merged_env,
                )
            )
        )
        session = await stack.enter_async_context(ClientSession(rw[0], rw[1]))
    await session.initialize()
    return session


class _ServerSession:
    __slots__ = ("cfg_hash", "stack", "session", "lock", "tools", "listed_ok")

    def __init__(self, cfg_hash: str, stack: AsyncExitStack, session: Any) -> None:
        self.cfg_hash = cfg_hash
        self.stack = stack
        self.session = session
        self.lock = asyncio.Lock()
        self.tools: List[Any] = []
        # Whether the last tools/list succeeded — a failed list must not
        # trigger the filter-sweep (empty list ≠ "server removed all tools").
        self.listed_ok = False


class MCPToolProxy:
    """Registry-compatible proxy for one remote MCP tool.

    Built from a tools-store entry (schema/config persisted at discovery) so it
    registers cleanly even while the server is unreachable — ``execute`` fails
    gracefully instead of the tool disappearing from the registry.
    """

    def __init__(
        self,
        name: str,
        schema: Optional[Dict[str, Any]],
        description: str,
        enabled: bool,
        server: str,
        remote_tool: str,
        manager: Optional["MCPClientManager"],
    ) -> None:
        self.name = name
        self._schema = schema or {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": {"type": "object"}},
        }
        self.description = description
        self.enabled = enabled
        self._server = server
        self._remote = remote_tool
        self._manager = manager

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return self._schema

    async def refresh_schema(self) -> None:
        if self._manager is not None:
            await self._manager.refresh_tool_schema(self)

    async def execute(self, context: Optional[Dict[str, Any]] = None, **params: Any) -> Any:
        if not self.enabled:
            return f"[Error: tool '{self.name}' is disabled.]"
        if not self._server or not self._remote:
            return f"[Error: tool '{self.name}' has no MCP routing info — re-sync the MCP server.]"
        if self._manager is None:
            return "[Error: MCP client manager is not available.]"
        return await self._manager.call(self._server, self._remote, params)


class MCPClientManager:
    """Owns MCP server sessions, tool discovery/materialization and call routing."""

    def __init__(
        self,
        store_getter: Callable[[], Any],
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        # ``store_getter`` is resolved lazily — the tools store is injected
        # onto RAGApplication AFTER construction but before initialize().
        self._store_getter = store_getter
        self._session_factory = session_factory or _default_session_factory
        self._registry: Any = None
        self._sessions: Dict[str, _ServerSession] = {}
        self._configs: Dict[str, MCPServerConfig] = {}
        self._proxies: Dict[str, MCPToolProxy] = {}
        self._ensure_locks: Dict[str, asyncio.Lock] = {}
        self._sync_lock = asyncio.Lock()
        self._last_store_read = 0.0
        self._closed = False
        self._master_off = False

    def set_registry(self, registry: Any) -> None:
        self._registry = registry

    # -- config ----------------------------------------------------------

    async def _read_configs(self, store: Any) -> Dict[str, MCPServerConfig]:
        """Parse the server list. ``_master_off`` is set separately — it pauses
        all servers without sweeping the materialized tool entries."""
        entry = await store.get_tool_by_name(_MCP_SERVERS_ENTRY)
        self._master_off = entry is not None and not bool(getattr(entry, "enabled", True))
        raw = (entry.config if entry is not None else None) or {}
        servers = raw.get("servers") or []
        out: Dict[str, MCPServerConfig] = {}
        for item in servers:
            if not isinstance(item, dict):
                continue
            cfg = MCPServerConfig.from_dict(item)
            if cfg.name:
                out[cfg.name] = cfg
        return out

    # -- session lifecycle ------------------------------------------------

    async def _connect(self, cfg: MCPServerConfig) -> Optional[_ServerSession]:
        stack = AsyncExitStack()
        try:
            session = await asyncio.wait_for(
                self._session_factory(cfg, stack), timeout=_CONNECT_TIMEOUT
            )
            srv = _ServerSession(cfg.fingerprint(), stack, session)
            try:
                listed = await asyncio.wait_for(session.list_tools(), timeout=15)
                srv.tools = list(getattr(listed, "tools", None) or [])
                srv.listed_ok = True
            except Exception as exc:
                logger.warning("MCP '%s': list_tools failed: %s", cfg.name, exc)
            return srv
        except Exception as exc:
            logger.warning("MCP server '%s' connect failed: %s", cfg.name, exc)
            try:
                await stack.aclose()
            except Exception as close_exc:
                logger.debug("MCP '%s' stack close after failed connect: %s", cfg.name, close_exc)
            return None

    async def _close_session(self, name: str) -> None:
        sess = self._sessions.pop(name, None)
        if sess is not None:
            try:
                await sess.stack.aclose()
            except Exception as exc:
                logger.debug("MCP '%s' close failed: %s", name, exc)

    async def _ensure_session(self, cfg: MCPServerConfig) -> Optional[_ServerSession]:
        # Per-server lock: sync() and call() may both reach this concurrently —
        # only one connect per server at a time (separate from _sync_lock to
        # avoid deadlock: sync holds it while calling this).
        lock = self._ensure_locks.setdefault(cfg.name, asyncio.Lock())
        async with lock:
            if self._closed:
                return None
            sess = self._sessions.get(cfg.name)
            if sess is not None and sess.cfg_hash == cfg.fingerprint():
                return sess
            await self._close_session(cfg.name)
            sess = await self._connect(cfg)
            if self._closed and sess is not None:
                # Shutdown raced our connect — aclose() already swept _sessions,
                # so close the fresh stack here rather than leak the subprocess.
                try:
                    await sess.stack.aclose()
                except Exception as close_exc:
                    logger.debug("MCP '%s' raced-connect close: %s", cfg.name, close_exc)
                return None
            if sess is not None:
                self._sessions[cfg.name] = sess
                await self._materialize(cfg, sess.tools, sess.listed_ok)
            return sess

    async def aclose(self) -> None:
        async with self._sync_lock:
            self._closed = True
            for name in list(self._sessions):
                await self._close_session(name)
            self._configs.clear()

    # -- sync / materialization -------------------------------------------

    async def sync(self) -> None:
        """Reconcile running sessions + materialized tools with the store config."""
        store = self._store_getter() if callable(self._store_getter) else self._store_getter
        if store is None:
            return
        self._last_store_read = time.monotonic()
        try:
            desired = await self._read_configs(store)
        except Exception as exc:
            logger.warning("MCP config read failed: %s", exc)
            return
        async with self._sync_lock:
            self._configs = {} if self._master_off else desired
            for name in list(self._sessions):
                cfg = desired.get(name)
                if self._master_off or cfg is None or not cfg.enabled:
                    removed = cfg is None and not self._master_off
                    await self._close_session(name)
                    if removed:
                        await self._disable_orphaned_entries(name, store)
            if self._master_off:
                return
            for cfg in desired.values():
                if not cfg.enabled:
                    continue
                sess = await self._ensure_session(cfg)
                if sess is not None:
                    await self._materialize(cfg, sess.tools, sess.listed_ok)

    async def _maybe_resync(self) -> None:
        if time.monotonic() - self._last_store_read > _RESYNC_TTL_SECONDS:
            try:
                await self.sync()
            except Exception as exc:
                logger.debug("MCP lazy resync failed: %s", exc)

    async def _materialize(
        self, cfg: MCPServerConfig, remote_tools: List[Any], listed_ok: bool = True
    ) -> None:
        store = self._store_getter() if callable(self._store_getter) else self._store_getter
        if store is None:
            return
        from app.tools_store import ToolEntry

        allowed_remotes = set()
        for rt in remote_tools:
            remote_name = str(getattr(rt, "name", "") or "")
            if not remote_name or not cfg.allows(remote_name):
                continue
            allowed_remotes.add(remote_name)
            local = mcp_tool_name(cfg.name, remote_name)
            desc = str(getattr(rt, "description", "") or "") or (
                f"MCP tool '{remote_name}' (server '{cfg.name}')"
            )
            schema = getattr(rt, "inputSchema", None)
            wrapped = {
                "type": "function",
                "function": {
                    "name": local,
                    "description": desc,
                    "parameters": schema if isinstance(schema, dict) else {"type": "object"},
                },
            }
            try:
                existing = await store.get_tool_by_name(local)
                if existing is not None:
                    cfg_dict = dict(existing.config or {})
                    cfg_dict.setdefault("_mcp_server", cfg.name)
                    cfg_dict.setdefault("_mcp_tool", remote_name)
                    needs_write = (
                        cfg_dict != (existing.config or {})
                        or existing.tool_schema != wrapped
                        or existing.description != desc
                    )
                    if needs_write:
                        updated = ToolEntry(
                            id=existing.id, name=existing.name,
                            description=desc, enabled=existing.enabled,
                            kind="mcp", tool_schema=wrapped, config=cfg_dict,
                            code=None, venv_config={},
                            runtime_mode=existing.runtime_mode or "venv",
                            created_at=existing.created_at,
                        )
                        await store.update_tool(updated)
                        existing = updated
                    entry = existing
                else:
                    entry = ToolEntry(
                        id=f"mcp-{_sanitize_part(cfg.name)}-{_sanitize_part(remote_name)}",
                        name=local, description=desc, enabled=True, kind="mcp",
                        tool_schema=wrapped,
                        config={"_mcp_server": cfg.name, "_mcp_tool": remote_name},
                        code=None, venv_config={}, runtime_mode="venv",
                    )
                    await store.create_tool(entry)
                proxy = self.get_proxy(entry)
                if self._registry is not None and proxy is not None:
                    self._registry.register_custom(entry.name, proxy)
            except Exception as exc:
                logger.warning(
                    "MCP materialize failed for '%s/%s': %s", cfg.name, remote_name, exc
                )
        # A narrowed tool_filter leaves previously materialized entries enabled
        # — disable them (kept, not deleted, so widening restores them). Only
        # when the tool list is authoritative — a failed tools/list must not
        # wipe every entry of the server.
        if listed_ok:
            await self._disable_disallowed(cfg.name, allowed_remotes, store)

    async def _disable_disallowed(
        self, server_name: str, allowed_remotes: set, store: Any
    ) -> None:
        try:
            entries = await store.list_tools()
        except Exception:
            return
        from app.tools_store import ToolEntry

        for entry in entries:
            if entry.kind != "mcp" or not entry.enabled:
                continue
            cfg_dict = entry.config or {}
            remote = str(cfg_dict.get("_mcp_tool") or "")
            # Match on the REMOTE name — a renamed entry must not self-disable.
            if (cfg_dict.get("_mcp_server") != server_name
                    or remote in allowed_remotes):
                continue
            try:
                await store.update_tool(
                    ToolEntry(
                        id=entry.id, name=entry.name, description=entry.description,
                        enabled=False, kind="mcp", tool_schema=entry.tool_schema,
                        config=entry.config, code=entry.code,
                        venv_config=entry.venv_config,
                        runtime_mode=entry.runtime_mode, created_at=entry.created_at,
                    )
                )
            except Exception as exc:
                logger.debug("MCP filter-sweep failed for %s/%s: %s", server_name, remote, exc)

    async def _disable_orphaned_entries(self, server_name: str, store: Any) -> None:
        """Disable materialized tools of a removed server (keep, don't delete)."""
        try:
            entries = await store.list_tools()
        except Exception as exc:
            logger.warning("MCP orphan sweep failed: %s", exc)
            return
        for entry in entries:
            if entry.kind != "mcp" or not entry.enabled:
                continue
            if (entry.config or {}).get("_mcp_server") != server_name:
                continue
            from app.tools_store import ToolEntry

            try:
                await store.update_tool(
                    ToolEntry(
                        id=entry.id, name=entry.name, description=entry.description,
                        enabled=False, kind="mcp", tool_schema=entry.tool_schema,
                        config=entry.config, code=entry.code,
                        venv_config=entry.venv_config,
                        runtime_mode=entry.runtime_mode, created_at=entry.created_at,
                    )
                )
            except Exception as exc:
                logger.debug("Failed to disable orphaned MCP tool %s: %s", entry.name, exc)

    # -- proxies ----------------------------------------------------------

    def get_proxy(self, entry: Any) -> Optional[MCPToolProxy]:
        """Return (create-on-demand) the proxy for a materialized tool entry."""
        routing = entry.config or {}
        server = str(routing.get("_mcp_server") or "")
        remote = str(routing.get("_mcp_tool") or "")
        proxy = self._proxies.get(entry.name)
        if proxy is None:
            proxy = MCPToolProxy(
                entry.name, entry.tool_schema, entry.description,
                bool(entry.enabled), server, remote, self,
            )
            self._proxies[entry.name] = proxy
        else:
            proxy.enabled = bool(entry.enabled)
            if entry.tool_schema:
                proxy._schema = entry.tool_schema
            if server:
                proxy._server = server
            if remote:
                proxy._remote = remote
        return proxy

    async def refresh_tool_schema(self, proxy: MCPToolProxy) -> None:
        # Live sessions only — connecting just to refresh a schema would make
        # refresh_schemas() stall ~20s per tool on a dead server.
        sess = self._sessions.get(proxy._server)
        if sess is None:
            return
        try:
            listed = await asyncio.wait_for(sess.session.list_tools(), timeout=15)
            sess.tools = list(getattr(listed, "tools", None) or [])
            for rt in sess.tools:
                if getattr(rt, "name", "") == proxy._remote:
                    schema = getattr(rt, "inputSchema", None)
                    if isinstance(schema, dict):
                        proxy._schema = {
                            "type": "function",
                            "function": {
                                "name": proxy.name,
                                "description": str(getattr(rt, "description", "") or proxy.description),
                                "parameters": schema,
                            },
                        }
            return
        except Exception as exc:
            logger.debug("MCP schema refresh failed for %s: %s", proxy.name, exc)

    # -- calls ------------------------------------------------------------

    async def call(self, server_name: str, remote_tool: str, params: Dict[str, Any]) -> Any:
        await self._maybe_resync()
        if self._master_off:
            return "[Error: MCP integrations are disabled (the 'mcp_servers' entry is off).]"
        cfg = self._configs.get(server_name)
        if cfg is None:
            return f"[Error: MCP server '{server_name}' is not configured.]"
        if not cfg.enabled:
            return f"[Error: MCP server '{server_name}' is disabled.]"
        if not isinstance(params, dict):
            return f"[Error: tool parameters must be an object, got {type(params).__name__}.]"
        result = await self._call_once(cfg, remote_tool, params)
        if result is _CALL_TIMED_OUT:
            # NO retry: the call may still be executing server-side — retrying
            # would double-execute non-idempotent actions (send_email twice).
            # The session is dropped: a late response would corrupt its stream.
            await self._close_session(cfg.name)
            return (f"[Error: MCP tool '{remote_tool}' timed out after "
                    f"{cfg.call_timeout}s — it may still have run server-side.]")
        if result is _CALL_FAILED:
            # Session likely dead — drop, reconnect once, retry once.
            await self._close_session(cfg.name)
            result = await self._call_once(cfg, remote_tool, params)
            if result in (_CALL_FAILED, _CALL_TIMED_OUT):
                return f"[Error: MCP server '{server_name}' is unavailable — reconnect failed.]"
        return self._map_result(result)

    async def _call_once(
        self, cfg: MCPServerConfig, remote_tool: str, params: Dict[str, Any]
    ) -> Any:
        sess = await self._ensure_session(cfg)
        if sess is None:
            return _CALL_FAILED
        try:
            async with sess.lock:
                return await asyncio.wait_for(
                    sess.session.call_tool(remote_tool, arguments=dict(params)),
                    timeout=cfg.call_timeout,
                )
        except asyncio.TimeoutError:
            logger.warning("MCP call '%s/%s' timed out after %ss", cfg.name, remote_tool, cfg.call_timeout)
            return _CALL_TIMED_OUT
        except Exception as exc:
            logger.warning("MCP call '%s/%s' failed: %s", cfg.name, remote_tool, exc)
            return _CALL_FAILED

    @staticmethod
    def _map_result(result: Any) -> Any:
        """Map a ``CallToolResult`` to the registry result contract."""
        is_error = bool(
            getattr(result, "isError", False) or getattr(result, "is_error", False)
        )
        texts: List[str] = []
        other_types: List[str] = []
        for item in list(getattr(result, "content", None) or []):
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(str(text))
            else:
                other_types.append(str(getattr(item, "type", "unknown")))
        structured = getattr(result, "structuredContent", None)
        body = "\n".join(t for t in texts if t)
        if not body and structured is not None:
            body = json.dumps(structured, ensure_ascii=False, default=str)
        if other_types:
            note = (
                f"[{len(other_types)} non-text content item(s)"
                f" ({', '.join(sorted(set(other_types)))}) — not rendered]"
            )
            body = f"{body}\n{note}" if body else note
        if len(body) > _MAX_RESULT_CHARS:
            body = body[:_MAX_RESULT_CHARS] + "\n...[result truncated]"
        if is_error:
            return f"[Error: {body or 'MCP tool reported an error'}]"
        meta = {"structured": structured} if structured is not None else None
        if meta:
            return {"ok": True, "content": body or "(no output)", "metadata": meta}
        return body or "(no output)"


class _CallFailed:
    """Sentinel: transport-level failure → caller may reconnect + retry."""


class _CallTimedOut:
    """Sentinel: deadline hit — the request may still run; never retried."""


_CALL_FAILED = _CallFailed()
_CALL_TIMED_OUT = _CallTimedOut()
