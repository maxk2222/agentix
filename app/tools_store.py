"""Postgres-backed store for agent tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

TOOLS_TABLE = "agentix_tools"


@dataclass
class ToolEntry:
    id: str
    name: str
    description: str
    enabled: bool
    kind: str  # "builtin" | "custom"
    tool_schema: Dict[str, Any]
    config: Dict[str, Any]
    code: Optional[str] = None
    venv_config: Dict[str, Any] = None
    runtime_mode: str = "venv"  # "venv" | "docker"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ToolsStore:
    """Async store for agent tools."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._tables_ensured = False

    async def ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TOOLS_TABLE} (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL DEFAULT '',
                        enabled BOOLEAN NOT NULL DEFAULT true,
                        kind TEXT NOT NULL DEFAULT 'custom',
                        tool_schema JSONB NOT NULL DEFAULT '{{}}',
                        config JSONB NOT NULL DEFAULT '{{}}',
                        code TEXT,
                        venv_config JSONB NOT NULL DEFAULT '{{}}',
                        runtime_mode TEXT NOT NULL DEFAULT 'venv',
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {TOOLS_TABLE}
                    ADD COLUMN IF NOT EXISTS venv_config JSONB NOT NULL DEFAULT '{{}}'
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE {TOOLS_TABLE}
                    ADD COLUMN IF NOT EXISTS runtime_mode TEXT NOT NULL DEFAULT 'venv'
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{TOOLS_TABLE}' AND column_name = 'schema'
                        ) THEN
                            ALTER TABLE {TOOLS_TABLE} RENAME COLUMN schema TO tool_schema;
                        END IF;
                    END $$;
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_tools_enabled
                    ON {TOOLS_TABLE} (enabled)
                    """
                )
            )
        self._tables_ensured = True

    async def list_tools(self, enabled_only: bool = False) -> List[ToolEntry]:
        await self.ensure_tables()
        query = f"""
            SELECT id, name, description, enabled, kind, tool_schema, config, code, venv_config, runtime_mode, created_at, updated_at
            FROM {TOOLS_TABLE}
        """
        if enabled_only:
            query += " WHERE enabled = true"
        query += " ORDER BY name"
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            rows = result.mappings().all()
            return [_row_to_tool(row) for row in rows]

    async def get_tool(self, tool_id: str) -> Optional[ToolEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, enabled, kind, tool_schema, config, code, venv_config, runtime_mode, created_at, updated_at
                    FROM {TOOLS_TABLE}
                    WHERE id = :id
                    """
                ),
                {"id": tool_id},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_tool(row)

    async def get_tool_by_name(self, name: str) -> Optional[ToolEntry]:
        await self.ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, name, description, enabled, kind, tool_schema, config, code, venv_config, runtime_mode, created_at, updated_at
                    FROM {TOOLS_TABLE}
                    WHERE name = :name
                    """
                ),
                {"name": name},
            )
            row = result.mappings().fetchone()
            if row is None:
                return None
            return _row_to_tool(row)

    async def create_tool(self, tool: ToolEntry) -> None:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {TOOLS_TABLE} (
                        id, name, description, enabled, kind, tool_schema, config, code, venv_config, runtime_mode, created_at, updated_at
                    )
                    VALUES (
                        :id, :name, :description, :enabled, :kind,
                        CAST(:tool_schema AS jsonb), CAST(:config AS jsonb), :code, CAST(:venv_config AS jsonb), :runtime_mode, now(), now()
                    )
                    """
                ),
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "enabled": tool.enabled,
                    "kind": tool.kind,
                    "tool_schema": _json_dumps(tool.tool_schema),
                    "config": _json_dumps(tool.config),
                    "code": tool.code,
                    "venv_config": _json_dumps(tool.venv_config or {}),
                    "runtime_mode": tool.runtime_mode,
                },
            )

    async def update_tool(self, tool: ToolEntry) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE {TOOLS_TABLE}
                    SET name = :name,
                        description = :description,
                        enabled = :enabled,
                        kind = :kind,
                        tool_schema = CAST(:tool_schema AS jsonb),
                        config = CAST(:config AS jsonb),
                        code = :code,
                        venv_config = CAST(:venv_config AS jsonb),
                        runtime_mode = :runtime_mode,
                        updated_at = now()
                    WHERE id = :id
                    """
                ),
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "enabled": tool.enabled,
                    "kind": tool.kind,
                    "tool_schema": _json_dumps(tool.tool_schema),
                    "config": _json_dumps(tool.config),
                    "code": tool.code,
                    "venv_config": _json_dumps(tool.venv_config or {}),
                    "runtime_mode": tool.runtime_mode,
                },
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def delete_tool(self, tool_id: str) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(f"DELETE FROM {TOOLS_TABLE} WHERE id = :id"),
                {"id": tool_id},
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def set_enabled(self, tool_id: str, enabled: bool) -> bool:
        await self.ensure_tables()
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE {TOOLS_TABLE}
                    SET enabled = :enabled, updated_at = now()
                    WHERE id = :id
                    """
                ),
                {"id": tool_id, "enabled": enabled},
            )
            return int(getattr(result, "rowcount", 0) or 0) > 0

    async def seed_builtin_tools(self) -> None:
        """Insert built-in tools into DB if they don't already exist."""
        await self.ensure_tables()
        builtins = [
            ToolEntry(
                id="builtin-web_scraper",
                name="web_scraper",
                description="Fetch the content of a specific URL and return the readable text.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "web_scraper",
                        "description": "Fetch the content of a specific URL and return the readable text.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The URL to fetch and read."},
                                "render_js": {"type": "boolean", "description": "Render JavaScript before extracting text (default false).", "default": False},
                            },
                            "required": ["url"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-web_search",
                name="web_search",
                description="Search the web for a query. Returns a list of search results with titles, snippets and URLs. Engine and region are configured in the tool settings.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for a query. Returns a list of search results with titles, snippets and URLs.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query."},
                                "results": {"type": "integer", "description": "Number of result links to return (default 15, max 30).", "default": 15},
                                "read_pages": {"type": "integer", "description": "How many of the top result pages to fetch and read (default 0).", "default": 0},
                            },
                            "required": ["query"],
                        },
                    },
                },
                config={"engine": "auto", "region": "", "safe_search": True},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-serpapi_search",
                name="serpapi_search",
                description="Search the web using SerpAPI. Requires a SerpAPI API key in tool config.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "serpapi_search",
                        "description": "Search the web using SerpAPI. Requires a SerpAPI API key in tool config.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query."},
                                "num": {"type": "integer", "description": "Number of results (default 15).", "default": 15},
                            },
                            "required": ["query"],
                        },
                    },
                },
                config={"api_key": ""},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-openalex_search",
                name="openalex_search",
                description="Search the OpenAlex academic graph for scholarly works, authors, institutions, concepts, venues, funders, or publishers. Optionally fetches and extracts text from public PDFs for open-access works. An optional API key can be configured in the tool settings for premium access.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "openalex_search",
                        "description": "Search the OpenAlex academic graph for scholarly works, authors, institutions, concepts, venues, funders, or publishers. Optionally fetches and extracts text from public PDFs for open-access works.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query."},
                                "entity": {
                                    "type": "string",
                                    "description": "Entity type to search: works, authors, institutions, concepts, venues, funders, publishers. Default: works.",
                                    "default": "works",
                                },
                                "filters": {
                                    "type": "object",
                                    "description": "Optional structured filters as key-value pairs, e.g. {'publication_year': '2020-2024', 'is_oa': true}.",
                                },
                                "per_page": {
                                    "type": "integer",
                                    "description": "Number of results per page (default 5, max 10).",
                                    "default": 5,
                                },
                                "sort": {
                                    "type": "string",
                                    "description": "Sort key, e.g. 'cited_by_count:desc' or 'relevance_score:desc'.",
                                },
                                "fetch_pdfs": {
                                    "type": "boolean",
                                    "description": "If true, automatically download and extract text from public PDFs for open-access works. Default: false.",
                                    "default": False,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-collection_manager",
                name="collection_manager",
                description="Manage .md files in configured knowledge collections. Actions: create, delete, list, read, edit. Edit supports a single operation or an 'edits' batch applied sequentially and saved atomically.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "collection_manager",
                        "description": "Manage .md files in configured knowledge collections. Actions: create, delete, list, read, edit. Edit supports a single operation or an 'edits' batch applied sequentially and saved atomically.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["create", "delete", "list", "read", "edit"],
                                    "description": "The CRUD action to perform.",
                                },
                                "collection": {
                                    "type": "string",
                                    "description": "The target knowledge collection (must be one of the configured collections).",
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "The .md filename (required for create, delete, read, edit).",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The text to write or insert/replace. Required for 'create', edit 'insert', and edit 'replace'.",
                                },
                                "limit": {"type": "integer", "default": 15000},
                                "offset": {"type": "integer", "default": 0},
                                "lines_start": {"type": "integer"},
                                "lines_end": {"type": "integer"},
                                "pattern": {"type": "string"},
                                "patterns": {"type": "array", "items": {"type": "string"}},
                                "full": {"type": "boolean"},
                                "context_lines": {"type": "integer", "default": 2},
                                "operation": {
                                    "type": "string",
                                    "enum": ["insert", "remove", "replace"],
                                    "description": "Single edit operation (alternative to 'edits').",
                                },
                                "line": {"type": "integer"},
                                "after_text": {"type": "string"},
                                "text_snippet": {"type": "string"},
                                "old_text": {"type": "string"},
                                "edits": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                    "description": "Ordered batch of edit operations applied sequentially to the evolving file (max 50, all-or-nothing). Each: {operation, ...fields}.",
                                },
                            },
                            "required": ["action", "collection"],
                        },
                    },
                },
                config={"collection_names": [], "index_debounce_seconds": 60},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-rag_search",
                name="rag_search",
                description="Search configured knowledge collections using the RAG retrieval pipeline. Returns ranked document chunks with source citations.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "rag_search",
                        "description": "Search configured knowledge collections using the RAG retrieval pipeline. Returns ranked document chunks with source citations, or a list of all documents in a collection.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["search", "list"],
                                    "description": "The action to perform: 'search' for semantic query, 'list' to enumerate all documents.",
                                    "default": "search",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "The search query (required when action='search').",
                                },
                                "collection": {
                                    "type": "string",
                                    "description": "The target knowledge collection to search or list.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of top results to return for search (default 5, max 10).",
                                    "default": 5,
                                    "maximum": 10,
                                },
                            },
                            "required": ["collection"],
                        },
                    },
                },
                config={"collection_names": [], "max_top_k": 10},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-rss_feed_reader",
                name="rss_feed_reader",
                description="Fetch and parse an RSS or Atom feed. Returns recent entries with title, link, published date, and summary. Use ONLY 'url' (required), 'entries' (number of items, default 10), and 'fetch_full_content' (scrape full article text, default false).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "rss_feed_reader",
                        "description": "Fetch and parse an RSS or Atom feed. Returns recent entries with title, link, published date, and summary. Use ONLY 'url' (required), 'entries' (number of items, default 10), and 'fetch_full_content' (scrape full article text, default false).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The RSS or Atom feed URL.",
                                },
                                "entries": {
                                    "type": "integer",
                                    "description": "Number of entries to return (default 10).",
                                    "default": 10,
                                },
                                "fetch_full_content": {
                                    "type": "boolean",
                                    "description": "Fetch the full article text for each entry (default false).",
                                    "default": False,
                                },
                            },
                            "required": ["url"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-code_execution",
                name="code_execution",
                description="Execute Python code in an isolated sandbox container that persists across calls. action='exec' (default) runs 'code'; job actions run persistent background processes: 'shell_start' (returns job_id), 'jobs', 'output', 'write', 'wait', 'kill', plus one-shot 'shell'.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "code_execution",
                        "description": "Execute Python code in an isolated sandbox container that persists across calls. action='exec' (default) runs 'code'; job actions run persistent background processes: 'shell_start' (returns job_id), 'jobs', 'output', 'write', 'wait', 'kill', plus one-shot 'shell'.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["exec", "shell", "shell_start", "jobs", "output", "write", "kill", "wait"],
                                },
                                "code": {
                                    "type": "string",
                                    "description": "The Python source code to execute (required for action='exec').",
                                },
                                "cmd": {"type": "string", "description": "Shell command for shell_start/shell."},
                                "job_id": {"type": "string", "description": "Job id for output/write/wait/kill."},
                                "tail": {"type": "integer"},
                                "stream": {"type": "string", "enum": ["stdout", "stderr", "both"]},
                                "text": {"type": "string"},
                                "cwd": {"type": "string"},
                                "packages": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of pip-installable packages to install before execution.",
                                },
                                "timeout": {
                                    "type": "integer",
                                    "description": "Max execution/wait time in seconds (default 30).",
                                    "default": 30,
                                },
                                "output_files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Absolute paths (max 10) of sandbox files attached to the chat.",
                                },
                            },
                            "required": [],
                        },
                    },
                },
                config={
                    "sandbox_mode": "read_only",
                    "egress_internet": True,
                    "egress_local_network": False,
                    "egress_blocked": [],
                },
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-agent_memory",
                name="agent_memory",
                description="A persistent scratchpad for THIS conversation. Use it to store durable facts, user preferences, and important intermediate results that should persist across tool calls. Do NOT store temporary tool outputs, hallucinated information, or data that is better fetched again from a source tool. Actions: read (retrieve), set (overwrite), append (add), delete (clear).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "agent_memory",
                        "description": "A persistent scratchpad for THIS conversation. Use it to store durable facts, user preferences, and important intermediate results that should persist across tool calls. Do NOT store temporary tool outputs, hallucinated information, or data that is better fetched again from a source tool. Actions: read (retrieve), set (overwrite), append (add), delete (clear).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["read", "set", "append", "delete"],
                                    "description": "The action to perform on the persistent memory.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to store. Required for 'set' and 'append'.",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-webhook_post",
                name="webhook_post",
                description="POST a JSON payload to an HTTPS endpoint. Use ONLY 'url' (required), 'payload' (JSON object, default {}), 'headers' (optional dict), and 'timeout' (seconds, default 30, max 60). Only https:// is allowed.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "webhook_post",
                        "description": "POST a JSON payload to an HTTPS endpoint. Use ONLY 'url' (required), 'payload' (JSON object, default {}), 'headers' (optional dict), and 'timeout' (seconds, default 30, max 60). Only https:// is allowed.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The target HTTPS URL. Must start with https://"},
                                "payload": {"type": "object", "description": "JSON payload to send. Default is empty object.", "default": {}},
                                "headers": {"type": "object", "description": "Optional additional HTTP headers.", "default": {}},
                                "timeout": {"type": "integer", "description": "Request timeout in seconds (default 30, max 60).", "default": 30},
                            },
                            "required": ["url"],
                        },
                    },
                },
                config={"allowed_hosts": [], "allow_private_ips": True},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-email_sender",
                name="email_sender",
                description="Send an email via SMTP. Supports plain-text, HTML (multipart), CC, BCC, reply-to, and file attachments. Use 'to' (required), 'subject' (required), 'body' (required plain text), and optionally 'html_body', 'cc', 'bcc', 'reply_to', 'attachments'. Attachment items accept {'filename','content_base64'} or {'artifact':'<name>'} for generated conversation artifacts. SMTP credentials are configured in the tool settings (dashboard → Tools → email_sender).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "email_sender",
                        "description": "Send an email via SMTP. Supports plain-text, HTML (multipart), CC, BCC, reply-to, and file attachments. Use 'to' (required), 'subject' (required), 'body' (required plain text), and optionally 'html_body', 'cc', 'bcc', 'reply_to', 'attachments'.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string", "description": "Primary recipient email address."},
                                "subject": {"type": "string", "description": "Email subject line."},
                                "body": {"type": "string", "description": "Plain-text email body (always included for fallback)."},
                                "html_body": {"type": "string", "description": "Optional HTML email body. If provided, a multipart message is sent with both plain-text and HTML."},
                                "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional list of CC recipient email addresses."},
                                "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional list of BCC recipient email addresses."},
                                "reply_to": {"type": "string", "description": "Optional reply-to email address."},
                                "attachments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {"type": "string"},
                                            "content_base64": {"type": "string"},
                                            "mime_type": {"type": "string"},
                                            "artifact": {"type": "string", "description": "Name of a generated artifact in this conversation to attach (alternative to content_base64)."},
                                        },
                                    },
                                    "description": "Optional file attachments (max 3, max 5 MB total). Each item takes {'filename','content_base64','mime_type'?} or {'artifact':'<name>'}.",
                                },
                            },
                            "required": ["to", "subject", "body"],
                        },
                    },
                },
                config={"smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_password": "", "smtp_from": "", "smtp_use_ssl": False, "smtp_timeout": 30},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-email_reader",
                name="email_reader",
                description="Read emails from the configured IMAP inbox. action='list' returns recent envelopes [{uid, from, subject, date, seen}] — options: 'folder' (default INBOX), 'limit' (default 10, max 50), 'unseen_only', 'search' (matches sender or subject). action='read' fetches one message body by 'uid' ('mark_read' opt-in). IMAP credentials are configured in the tool settings (dashboard → Tools → email_reader).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "email_reader",
                        "description": "Read emails from the configured IMAP inbox (list envelopes / read bodies).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["list", "read"], "description": "'list' = recent envelopes, 'read' = one message body (requires 'uid')."},
                                "folder": {"type": "string", "description": "Mailbox folder (default 'INBOX')."},
                                "limit": {"type": "integer", "description": "Max envelopes for 'list' (default 10, max 50).", "default": 10},
                                "unseen_only": {"type": "boolean", "description": "Only unread messages for 'list' (default false).", "default": False},
                                "search": {"type": "string", "description": "Optional sender/subject filter for 'list'."},
                                "uid": {"type": "string", "description": "Message uid for action='read'."},
                                "mark_read": {"type": "boolean", "description": "Mark message as read on action='read' (default false).", "default": False},
                            },
                            "required": ["action"],
                        },
                    },
                },
                config={"imap_host": "", "imap_port": 993, "imap_user": "", "imap_password": "", "imap_use_ssl": True, "imap_timeout": 30},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-google_calendar",
                name="google_calendar",
                description="Manage events on a shared Google Calendar via a service account. action='list' (time_min/time_max, max_results), 'create' (summary, start, end, description, location, attendees, timezone), 'update' (event_id + changed fields), 'delete' (event_id). Configure service_account_json and calendar_id in the tool settings and share the calendar with the service account.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "google_calendar",
                        "description": "Manage events on a shared Google Calendar (service-account access).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["list", "create", "update", "delete"], "description": "Calendar operation."},
                                "event_id": {"type": "string", "description": "Event id (required for update/delete)."},
                                "summary": {"type": "string", "description": "Event title."},
                                "start": {"type": "string", "description": "Start ISO datetime or YYYY-MM-DD (all-day)."},
                                "end": {"type": "string", "description": "End ISO datetime or YYYY-MM-DD (all-day)."},
                                "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Europe/Berlin'."},
                                "description": {"type": "string", "description": "Event description/body."},
                                "location": {"type": "string", "description": "Event location."},
                                "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee email addresses."},
                                "time_min": {"type": "string", "description": "List window start (ISO)."},
                                "time_max": {"type": "string", "description": "List window end (ISO)."},
                                "max_results": {"type": "integer", "description": "Max events for 'list' (default 20, max 50).", "default": 20},
                            },
                            "required": ["action"],
                        },
                    },
                },
                config={"service_account_json": "", "calendar_id": "", "default_timezone": "Europe/Berlin"},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-mcp_servers",
                name="mcp_servers",
                description=(
                    "MCP server registry (config only — not a callable tool). "
                    "'config.servers' is a list of external MCP servers; discovered remote tools "
                    "are materialized as 'mcp__<server>__<tool>' entries that can be enabled and "
                    "assigned to assistants. Per server: name, transport ('stdio'|'http'), "
                    "stdio: command/args/env — http: url/headers, enabled, tool_filter/deny "
                    "(glob on remote tool names), call_timeout (seconds)."
                ),
                enabled=True,
                kind="builtin",
                tool_schema={},
                config={"servers": []},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-long_document_summarizer",
                name="long_document_summarizer",
                description="Summarize a long document provided as text, URL, or collection file path. Returns an extractive summary. Use 'text', 'url', or 'collection_file' (one required). Optional 'max_sentences' (default 5), 'max_words', 'min_sentences' (default 1), 'focus' (topic to emphasize), 'format' ('paragraph' or 'bullet_points'), and 'deduplicate' (default true).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "long_document_summarizer",
                        "description": "Summarize a long document provided as text, URL, or collection file path. Returns an extractive summary.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Raw text to summarize. Mutually exclusive with url and collection_file."},
                                "url": {"type": "string", "description": "URL to fetch and summarize. Mutually exclusive with text and collection_file."},
                                "collection_file": {"type": "string", "description": "Collection file path like 'collection/filename.md'. Mutually exclusive with text and url."},
                                "max_sentences": {"type": "integer", "description": "Maximum number of sentences in the summary (default 5, max 50).", "default": 5},
                                "max_words": {"type": "integer", "description": "Optional maximum total words in the summary. Acts as a hard limit alongside max_sentences."},
                                "min_sentences": {"type": "integer", "description": "Minimum number of sentences (default 1).", "default": 1},
                                "focus": {"type": "string", "description": "Optional topic or keyword to emphasize. Multi-word phrases supported."},
                                "format": {"type": "string", "enum": ["paragraph", "bullet_points"], "description": "Output format: paragraph or bullet_points.", "default": "paragraph"},
                                "deduplicate": {"type": "boolean", "description": "Remove near-duplicate sentences before scoring (default true).", "default": True},
                                "read_limit": {"type": "integer", "description": "Maximum characters loaded from collection_file (default 250000, max 1000000). Increase when the summary reports input truncation.", "default": 250000},
                            },
                            "required": [],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-plan_generator",
                name="plan_generator",
                description="Generate a structured plan for a given goal. Use 'goal' (required), optional 'context', 'constraints', 'detail_level' ('brief', 'standard', 'detailed'), and 'include_risks' (default true).",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "plan_generator",
                        "description": "Generate a structured plan for a given goal. Use 'goal' (required), optional 'context', 'constraints', 'detail_level' ('brief', 'standard', 'detailed'), and 'include_risks' (default true).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string", "description": "The main goal or objective of the plan."},
                                "context": {"type": "string", "description": "Additional context or background information."},
                                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Optional list of constraints or limitations to consider."},
                                "detail_level": {"type": "string", "enum": ["brief", "standard", "detailed"], "description": "Level of detail for the plan.", "default": "standard"},
                                "include_risks": {"type": "boolean", "description": "Include a risk assessment section.", "default": True},
                            },
                            "required": ["goal"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-arxiv_search",
                name="arxiv_search",
                description="Search arXiv for academic papers. No API key required. Use 'query' (required), optional 'max_results' (default 5, max 10), 'sort_by' ('relevance', 'lastUpdatedDate', 'submittedDate'), and 'sort_order' ('ascending', 'descending').",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "arxiv_search",
                        "description": "Search arXiv for academic papers. No API key required. Use 'query' (required), optional 'max_results' (default 5, max 10), 'sort_by' ('relevance', 'lastUpdatedDate', 'submittedDate'), and 'sort_order' ('ascending', 'descending').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query for arXiv papers."},
                                "max_results": {"type": "integer", "description": "Maximum number of results (default 5, max 10).", "default": 5},
                                "sort_by": {"type": "string", "enum": ["relevance", "lastUpdatedDate", "submittedDate"], "description": "Sort criterion.", "default": "relevance"},
                                "sort_order": {"type": "string", "enum": ["ascending", "descending"], "description": "Sort order.", "default": "descending"},
                            },
                            "required": ["query"],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-wikipedia_search",
                name="wikipedia_search",
                description="Search Wikipedia for articles or read a full article. No API key required. action='search' (default): 'query' (required), optional 'language' (default 'en'), 'limit' (default 5, max 10), 'fetch_extracts' (default true). action='read': 'title' (required, falls back to 'query') returns the complete plain-text article with '== Section ==' markers, paged via 'read_limit' (default 50000, max 250000), 'offset', 'lines_start'/'lines_end' and 'pattern'.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "wikipedia_search",
                        "description": "Search Wikipedia for articles or read a full article. No API key required. action='search' (default): 'query' (required), optional 'language' (default 'en'), 'limit' (default 5, max 10), 'fetch_extracts' (default true). action='read': 'title' (required, falls back to 'query') returns the complete plain-text article with '== Section ==' markers, paged via 'read_limit' (default 50000, max 250000), 'offset', 'lines_start'/'lines_end' and 'pattern'.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["search", "read"], "description": "'search' finds articles; 'read' returns the full plain-text article for 'title'.", "default": "search"},
                                "query": {"type": "string", "description": "Search query (required for action='search'; fallback title for action='read')."},
                                "title": {"type": "string", "description": "Article title to read in full (required for action='read')."},
                                "language": {"type": "string", "description": "Wikipedia language code (default 'en').", "default": "en"},
                                "limit": {"type": "integer", "description": "Maximum number of search results (default 5, max 10).", "default": 5},
                                "fetch_extracts": {"type": "boolean", "description": "Fetch 3-sentence article extracts in search results (default true).", "default": True},
                                "read_limit": {"type": "integer", "description": "Max characters of article text returned per read call (default 50000, max 250000). Page further with offset/lines_start.", "default": 50000},
                                "offset": {"type": "integer", "description": "action='read': character offset into the article text (negative counts from the end)."},
                                "lines_start": {"type": "integer", "description": "action='read': first line to return (1-based, takes priority over offset)."},
                                "lines_end": {"type": "integer", "description": "action='read': last line to return (inclusive)."},
                                "pattern": {"type": "string", "description": "action='read': case-insensitive substring filter — returns numbered match windows instead of a range."},
                                "context_lines": {"type": "integer", "description": "action='read': lines of context around each pattern match (default 2)."},
                            },
                            "required": [],
                        },
                    },
                },
                config={},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-weather_lookup",
                name="weather_lookup",
                description="Get current weather and a short-term forecast for a location. Use 'location' (required, city name or 'lat,lon'), 'forecast_days' (0-7, default 0), and 'units' ('metric' or 'imperial', default 'metric').",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "weather_lookup",
                        "description": "Get current weather and a short-term forecast for a location. Use 'location' (required, city name or 'lat,lon'), 'forecast_days' (0-7, default 0), and 'units' ('metric' or 'imperial', default 'metric').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name or 'latitude,longitude' coordinates.",
                                },
                                "forecast_days": {
                                    "type": "integer",
                                    "description": "Number of forecast days to include (0 = current only, max 7).",
                                    "default": 0,
                                },
                                "units": {
                                    "type": "string",
                                    "enum": ["metric", "imperial"],
                                    "description": "Temperature units (default 'metric').",
                                    "default": "metric",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                },
                config={"provider": "open-meteo", "api_key": ""},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-ssh_exec",
                name="ssh_exec",
                description="Execute a shell command on a preconfigured remote host via SSH. Use ONLY 'command' (required shell command) and optional 'timeout' (seconds, default 30, max 120). SSH credentials are configured in the tool settings.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "ssh_exec",
                        "description": "Execute a shell command on a preconfigured remote host via SSH. Use ONLY 'command' (required shell command) and optional 'timeout' (seconds, default 30, max 120). SSH credentials are configured in the tool settings.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The shell command to execute on the remote host.",
                                },
                                "timeout": {
                                    "type": "integer",
                                    "description": "Max execution time in seconds (default 30, max 120).",
                                    "default": 30,
                                },
                            },
                            "required": ["command"],
                        },
                    },
                },
                config={"host": "", "port": 22, "username": "", "auth_type": "password", "password": "", "private_key": "", "key_passphrase": ""},
                code=None,
                venv_config={},
                runtime_mode='venv',
            ),
            ToolEntry(
                id="builtin-wait",
                name="wait",
                description="Wait briefly before polling an external resource or retrying. Delay is capped at 120 seconds.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "description": "Wait before polling an external resource or retrying.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "seconds": {
                                    "type": "number",
                                    "description": "How many seconds to wait (0.1 to 120).",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Optional short reason for waiting.",
                                },
                            },
                            "required": ["seconds"],
                        },
                    },
                },
                config={"max_seconds": 120},
                code=None,
                venv_config={},
                runtime_mode="venv",
            ),
            ToolEntry(
                id="builtin-agent_todo",
                name="agent_todo",
                description="Maintain a temporary, read-only-in-chat todo list for the current agent run.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "agent_todo",
                        "description": (
                            "Maintain a scratch todo list for multi-step work in this run. "
                            "Actions: add, update, complete, remove, list, clear."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "update", "complete", "remove", "list", "clear"],
                                },
                                "task_id": {"type": "string"},
                                "title": {"type": "string"},
                                "note": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "skipped"],
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                config={"max_tasks": 20},
                code=None,
                venv_config={},
                runtime_mode="venv",
            ),
            ToolEntry(
                id="builtin-document_generator",
                name="document_generator",
                description="Generate downloadable documents and spreadsheets (.docx, .pptx, .pdf, .xlsx, .csv) with styled layouts, native charts and diagrams. Files appear as download cards in the chat.",
                enabled=True,
                kind="builtin",
                tool_schema={
                    "type": "function",
                    "function": {
                        "name": "document_generator",
                        "description": (
                            "Generate a downloadable file (.docx, .pptx, .pdf, .xlsx or .csv) from a structured spec. "
                            "docx/pdf use sections, pptx uses slides with layouts, charts and diagrams, "
                            "xlsx/csv use sheets (csv writes only the first sheet). "
                            "Small docs: omit 'action' and pass content in one call. "
                            "Long docs: action='create' -> 'add_sections'/'add_slides'/'add_sheets' -> 'finalize'; "
                            "action='edit' rewrites draft items via an ordered 'edits' batch (all-or-nothing)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["docx", "pptx", "pdf", "xlsx", "csv"]},
                                "title": {"type": "string"},
                                "filename": {"type": "string"},
                                "author": {"type": "string"},
                                "subject": {"type": "string"},
                                "sections": {"type": "array", "items": {"type": "object"}},
                                "slides": {"type": "array", "items": {"type": "object"}},
                                "sheets": {"type": "array", "items": {"type": "object"}},
                                "action": {
                                    "type": "string",
                                    "enum": ["create", "add_sections", "add_slides", "add_sheets", "edit", "finalize", "abort"],
                                },
                                "draft_id": {"type": "string"},
                                "edits": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                    "description": "Ordered batch of draft item ops for action='edit' (replace/insert/remove/move; max 50).",
                                },
                            },
                            "required": [],
                        },
                    },
                },
                config={"theme": {}},
                code=None,
                venv_config={},
                runtime_mode="venv",
            ),
        ]
        for tool in builtins:
            existing = await self.get_tool_by_name(tool.name)
            if existing is None:
                await self.create_tool(tool)
            elif existing.kind == "builtin" and (
                existing.description != tool.description
                or existing.tool_schema != tool.tool_schema
            ):
                # Repair stale builtin metadata (descriptions/schemas drift when
                # a tool gains features); never touch enabled/config/code.
                existing.description = tool.description
                existing.tool_schema = tool.tool_schema
                await self.update_tool(existing)


def _row_to_tool(row) -> ToolEntry:
    return ToolEntry(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        enabled=bool(row["enabled"]),
        kind=row["kind"],
        tool_schema=row["tool_schema"] or {},
        config=row["config"] or {},
        code=row["code"],
        venv_config=row["venv_config"] or {},
        runtime_mode=row.get("runtime_mode") or "venv",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
