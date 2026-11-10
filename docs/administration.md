# Administration

The admin dashboard is a React SPA served at `/dashboard` (requires `DASHBOARD_ENABLED=true`). Authentication uses signed session cookies; see [Security](security.md).

## Roles & permissions

| Role | Scope |
|---|---|
| `viewer` | Read-only: overview, models, history, chat, widget |
| `operator` | viewer + metrics, conversations read |
| `editor` | operator + assistants, knowledge, tools, triggers, agent chains write; `backup:read` |
| `admin` | editor + user management, runtime config, `backup:write`/`backup:restore` |
| `owner` | full access |

Permission checks run per endpoint via `require_permission(request, "<perm>")` (`app/auth.py`, role map in `app/dashboard_users.py`).

## Dashboard sections

| Page | Route | What it manages |
|---|---|---|
| Overview | `/overview` | Metrics summary, vectorstore state, recent conversations |
| Chat | `/chat` | Test conversations with any assistant; file attachments, artifact cards, mid-run steering |
| Conversations | `/conversations` | Archive, owner/title search, user/model/date filters, sortable columns, export (json/csv/txt), agent memories |
| Agents | `/config-assistants` | Assistants: model, model-aware reasoning, tools, RAG collections, subagents, observer, agent workspace (scope/quotas/`vision_read`/`api_enabled`), version history & diff/rollback |
| Knowledge | `/config-knowledge` | Collections, uploads, OCR options, WebDAV sources, OKF graph |
| Tools | `/config-tools` | Built-in tool settings (SMTP/IMAP/calendar/MCP `mcp_servers`) + custom tools (code, venv, schemas) — see [Tools](tools.md) |
| Agent Chains | `/agent-chains` | Multi-step flows (visual editor, validation, export) |
| Triggers | `/triggers` | Scheduled prompts (interval/cron), run history |
| Metrics | `/metrics` | Request/token/latency analytics |
| Backups | `/backups` | Backup schedule, WebDAV target, archives, restore — see [Backup & Recovery](backup-and-recovery.md) |
| Configuration ▸ Widget & Chat | `/widget` | Widget configurator + public chat config |
| Configuration ▸ Runtime & LLM APIs | `/config-runtime` | `runtime_config` table; env-overridden values are flagged non-editable |
| Configuration ▸ Users | `/users` | Dashboard accounts, roles, activation |

## Runtime configuration

`runtime_config` persists selected settings in Postgres. **Env vars always win** — the effective config endpoint marks each entry with its source (`env` vs `db`) and DB-only keys.

## WebDAV knowledge sources

Knowledge sources can sync from WebDAV (Sciebo) directories: recursive `PROPFIND` listing, ETag-based change detection, optional OKF conversion. Source passwords are stored Fernet-encrypted (`WEBDAV_ENCRYPTION_KEY`).

## Housekeeping notes

- Trigger scheduler and backup scheduler tick every 30 s; both use persisted `next_run_at` fields.
- During a database restore a maintenance flag pauses trigger execution.
- Audit surfaces: `agentix_backup_jobs` (backup/restore), trigger run history, instructions version history.

## Screenshot gallery

Screenshots and walkthrough GIFs are not shipped in the repository. Regenerate them
against a running instance with `AGENTIX_PASSWORD='…' docs/update_screenshots.sh`
(engine: `tests/e2e/capture_media.py`) — they land under `docs/assets/`.
