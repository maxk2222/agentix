# API Reference

Interactive docs are served by FastAPI at `/docs` (Swagger) and `/redoc`; the machine-readable schema is `/openapi.json`. This page groups every exposed endpoint by surface.

[[_TOC_]]

## Authentication

| Surface | Credential |
|---|---|
| `/v1/*` | `Authorization: Bearer <key>` or `X-API-Key: <key>` — managed keys in the dashboard (or `API_KEY` bootstrap); enforced when `API_KEYS_ENABLED=true` |
| `/admin/*`, `/dashboard` | Signed session cookie from `POST /admin/auth/login` |
| `/public-chat/*` | Public-chat account session / access token |
| `/chat/assistants/*/workspace/*`, `/v1/chat/*` | The **calling assistant's own access rules** — API key, dashboard session, public-chat role, or anonymous if `api_accessible` |

## Public agent API (`/v1`)

OpenAI-compatible.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat with an assistant — streaming (`stream: true`), tool calling, `attachments`, multimodal `content` blocks |
| `GET` | `/v1/models` | Publicly accessible assistants/agent chains (`api_accessible`; public-chat anonymous allowlist further restricts) |
| `GET` | `/v1/knowledge/okf-graph?collection=` | OKF concept graph for a collection |

```bash
curl -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  http://localhost:8000/v1/chat/completions \
  -d '{"model": "<assistant>", "messages": [{"role": "user", "content": "Hello"}]}'
```

Requests are validated against `MAX_*`/`VALIDATION_*` limits ([Configuration](configuration.md)); violations return `4xx` and count into the blocked-request metrics.

### Run control (`/v1/chat/runs`)

Long-running agent executions — status, live stream, cancel and **mid-run steering** ([details](agent.md#mid-run-steering-message-queue)):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/chat/runs/{conversation_id}` | Run status + tool activity snapshot |
| `GET` | `/v1/chat/runs/{conversation_id}/stream` | SSE stream (tokens, tool events, injected messages) |
| `POST` | `/v1/chat/runs/{conversation_id}/cancel` | Abort the run |
| `POST` | `/v1/chat/runs/{conversation_id}/input` | Queue a message into the running agent — applied at the next iteration |

```bash
curl -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  http://localhost:8000/v1/chat/runs/c_abc123/input \
  -d '{"message": "actually focus on the DNS part"}'
```

Queued input also wakes `wait`/`wait_subagent` calls that are currently
blocking the run. Optional `subagent_id` targets one of the run's subagents
instead of the main agent — the message lands in that child's nudge queue
as `[User message]` (text-only; reject reasons: `no_subagent`,
`subagent_finished`):

```bash
curl -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  http://localhost:8000/v1/chat/runs/c_abc123/input \
  -d '{"message": "skip the image analysis", "subagent_id": "sub-a1b2c3d4-1"}'
```

### Files & attachments (`/v1/chat/*`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/chat/files/{conversation_id}/{filename}` | Download an agent-produced artifact (docx/pptx/pdf/xlsx/csv/…) |
| `GET` | `…/{filename}/preview` | Inline preview metadata |
| `GET` | `…/{filename}/preview/page/{page}` | Rendered preview page |
| `GET` | `…/{filename}/export/{fmt}` | Export conversion (e.g. `pdf`) |
| `POST` | `/v1/chat/attachments` | Upload a chat attachment (image/document) |
| `GET` | `/v1/chat/attachments` | List own attachments |
| `GET` | `/v1/chat/attachments/{conversation_id}/{attachment_id}/file` | Download attachment bytes |
| `DELETE` | `/v1/chat/attachments/{conversation_id}/{attachment_id}` | Remove an attachment |
| `DELETE` | `/v1/chat/conversations/{conversation_id}` | Delete a conversation |
| `GET` | `/v1/chat/capabilities` | Feature flags for the client (attachments, TTS, …) |

Artifact/attachment access follows the conversation's bearer semantics — the conversation id is the capability for anonymous conversations, owned conversations check the caller.

### Agent workspace API

Read-only file access when `workspace.enabled` **and** `workspace.api_enabled` ([details](agent.md#read-only-api)):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/chat/assistants/{assistant}/workspace/files?path=&limit=&offset=` | List entries (paged) |
| `GET` | `/chat/assistants/{assistant}/workspace/file?path=` | Download a file (`inline` for images, `attachment` otherwise) |

```bash
curl -H "x-api-key: $KEY" \
  "http://localhost:8000/chat/assistants/CyberAnalyst/workspace/files?path=uploads"
```

Scope rules: `conversation`-scoped workspaces require `?conversation_id=` + owner check; `user`-scoped require a signed-in chat identity (401 anonymous); `assistant`-scoped is shared by everyone who passes the assistant access gate.

## Admin API (`/admin`)

Session-cookie auth (`POST /admin/auth/login`); every endpoint checks `require_permission` for a granular permission (see [Administration](administration.md#roles--permissions)).

### Auth

| Method | Path |
|---|---|
| `POST` | `/admin/auth/login` · `logout` · `change-password` |
| `GET` | `/admin/auth/me` |

### Metrics & dashboard

| Method | Path |
|---|---|
| `GET` | `/admin/metrics/summary` · `by-model` · `history` · `history/api-key` · `models` · `blocked-summary` · `blocked-history` |
| `GET` | `/admin/dashboard/overview` · `models` · `history` · `blocked` |
| `GET` | `/admin/vectorstore/summary` · `by-model` · `history` · `state` |

### Assistants & config

| Method | Path |
|---|---|
| `GET`/`PUT` | `/admin/config/instructions` (assistant definitions) |
| `GET`/`POST` | `…/instructions/download` · `upload` · `history` · `rollback/{version_id}` · `version/{version_id}` · `diff` |
| `POST` | `/admin/config/apply` — apply staged config |
| `GET`/`PUT` + `POST` | `/admin/config/runtime` · `runtime/apply` · `runtime/export` · `runtime/import` |
| `GET` | `/admin/config/descriptions` · `models` · `provider-models` |
| `GET` | `/admin/config/reasoning-schema?model=&provider_id=` — per-model reasoning control descriptor (used by the config editor) |
| `GET`/`POST`/`PUT`/`DELETE` | `/admin/config/providers` · `providers/{id}` + `…/secret/rotate` · `…/check` · `…/detect-tool-calling` · `…/detect-reasoning` · `providers/export` · `providers/import` |

### Knowledge

| Method | Path |
|---|---|
| `GET`/`POST`/`DELETE` | `/admin/knowledge/collections` · `collections/{name}` · `…/stats` · `…/rename` · `…/rebuild` |
| `POST`/`GET`/`PUT`/`DELETE` | `/admin/knowledge/upload` · `preview/{collection}/{file}` · `text/{collection}/{file}` · `{collection}/{file}` |
| `GET`/`POST`/`PUT`/`DELETE` | `/admin/knowledge/webdav` · `webdav/{name}` · `…/sync` · `…/cancel` · `webdav/test` · `…/status` |

### Conversations & memory

| Method | Path |
|---|---|
| `GET` | `/admin/conversations` · `conversations/models` · `conversations/users` · `conversations/export` · `conversations/{id}` · `…/metadata` — list/export accept `date_from`, `date_to`, `conversation_id`, `model`, `min_user_messages`, `user_id` (or `anonymous`), `query` (title/ID search); list additionally `sort_by`, `sort_order`, `page`, `page_size` |
| `GET`/`PUT`/`DELETE`/`POST` | `/admin/agent-memory/{conversation_id}` · `agent-memory/check` |

### Tools (custom + MCP)

`/admin/tools` — builtin/custom tool management incl. `mcp_servers` config ([Tools reference](tools.md#mcp-servers-mcp_servers)):

| Method | Path |
|---|---|
| `GET`/`POST` | `/admin/tools` — list / create |
| `GET`/`PUT`/`DELETE` | `/admin/tools/{tool_id}` |
| `POST` | `/admin/tools/{tool_id}/enable` · `/disable` · `/venv/rebuild` |

### Agent chains & triggers

| Method | Path |
|---|---|
| `GET`/`POST` | `/admin/agent-chains` |
| `GET`/`PUT`/`DELETE` | `/admin/agent-chains/{id}` |
| `POST`/`GET` | `/admin/agent-chains/validate` · `{id}/export` · `/import` |
| `GET`/`POST` | `/admin/triggers` |
| `GET`/`PUT`/`DELETE` | `/admin/triggers/{id}` |
| `POST`/`GET` | `/admin/triggers/{id}/run` · `{id}/runs` |

### Users & public chat

| Method | Path |
|---|---|
| `GET`/`POST`/`PUT`/`DELETE`/`POST` | `/admin/users` · `users/{id}` · `…/password` · `…/disable` · `…/enable` |
| `GET`/`PUT`/`POST`/`DELETE` | `/admin/chat-users` · `chat-users/{id}` · `…/password` |
| `GET`/`POST` | `/admin/chat-roles` |
| `GET`/`POST` | `/admin/widget-defaults` · `/admin/public-chat/config` |

### API keys (`/admin/api-keys`)

Manages the keys accepted on `/v1/*` and the workspace API ([Authentication](#authentication)). The raw key is returned **once** on create — only its hash is stored.

| Method | Path | Permission | Purpose |
|---|---|---|---|
| `GET` | `/admin/api-keys` | `api_keys:read` | List keys (no hashes; `source: env|db`) |
| `POST` | `/admin/api-keys` | `api_keys:write` | Create — returns `api_key` plaintext once |
| `PUT`/`DELETE` | `/admin/api-keys/{key_id}` | `api_keys:write` | Update name/models/enabled · delete |
| `GET` | `/admin/api-keys/models` | `api_keys:read` | Assistant names for `allowed_models` whitelisting |

### Backup (`/admin/backup`)

See the dedicated runbook: [Backup & Recovery](backup-and-recovery.md).

| Method | Path | Permission | Purpose |
|---|---|---|---|
| `GET`/`PUT` | `/admin/backup/config` | `backup:read`/`write` | Schedule, target, retention |
| `POST` | `…/test` · `…/run` · `…/cancel` | `backup:write` | Probe / start / cancel |
| `GET` | `…/jobs` · `…/jobs/{id}` · `…/remote` | `backup:read` | History, detail, remote archives |
| `POST`/`DELETE` | `…/verify/{filename}` · `…/remote/{filename}` | `backup:read`/`write` | Checksum-verify / delete |
| `POST` | `…/restore` | `backup:restore` | Restore — `confirm` must equal the filename |

Secrets are never returned (`password_set` flags instead). Errors: `400` invalid input, `404` unknown remote file, `409` busy/maintenance, `503` missing key or `pg_dump`.

### Audio & emotion admin

| Method | Path |
|---|---|
| `GET`/`POST`/`PUT`/`DELETE` | `/admin/audio/voices` · `…/upload` · `…/download` · `…/{voice_id}` · `…/activate` · `…/test` · `settings` · `voices/kokoro/*` |
| `GET`/`POST`/`DELETE` | `/admin/emotion/models` · `…/upload` · `…/{type}/activate` · `…/{type}/{file}` · `…/download` |

## Public chat (`/public-chat`)

Token-gated self-service surface for external users; configured via `/admin/public-chat/config`.

| Method | Path |
|---|---|
| `POST` | `/public-chat/auth/register` · `login` · `logout` · `change-password` |
| `GET` | `/public-chat/auth/me` · `/public-chat/config` |
| `GET`/`PATCH`/`DELETE` | `/public-chat/conversations` · `…/{id}/messages` · `…/{id}/memory` · `…/{id}/title` |

## Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET`/`POST` | `/audio/health` · `/audio/config` · `/audio/stt` · `/audio/tts` · `/audio/tts/segment` | STT/TTS |
| `GET`/`POST` | `/emotion/config` · `/emotion/detect-text` | Emotion detection |
| `GET` | `/widget-defaults` | Public widget defaults (`Cache-Control: no-cache`; CORS via `WIDGET_DEFAULTS_CORS_ORIGINS`, default `*`) |

## Error conventions

| Code | Meaning |
|---|---|
| `400` | Validation failed (`MAX_*`/`VALIDATION_*` limits), bad path, missing required param |
| `401` | Missing/invalid API key or session |
| `403` | Permission denied (role, allowlist, ownership) |
| `404` | Feature disabled (e.g. workspace `api_enabled` off), unknown resource |
| `409` | Conflict (backup busy, maintenance mode) |
| `503` | Dependency unavailable (LLM endpoint, `pg_dump`, missing encryption key) |
