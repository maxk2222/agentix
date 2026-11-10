# Security

## Authentication & authorization

| Surface | Mechanism |
|---|---|
| `/v1/*` | `Authorization: Bearer` / `X-API-Key` — keys stored hashed in Postgres (`api_keys` store); enforced when `API_KEYS_ENABLED=true` |
| `/dashboard`, `/admin/*` | Signed session cookies (`itsdangerous`), TTL `DASHBOARD_SESSION_TTL_SECONDS`; per-endpoint `require_permission()` against role permissions |
| `/public-chat/*` | Separate public-chat user accounts with their own sessions |
| `/v1/chat/files/*`, `/v1/chat/attachments/*` | Conversation bearer semantics — the conversation id is the capability; owned conversations check the caller |
| `/chat/assistants/*/workspace/*` | Assistant's own access cascade (API key → session → public-chat role → `api_accessible`); `conversation` scope additionally requires the conv id, `user` scope a signed-in identity |

Roles: `viewer` → `operator` → `editor` → `admin` → `owner` (see [Administration](administration.md)).

## Secrets inventory

| Secret | Where used | Storage |
|---|---|---|
| `LLM_API_KEY`, `EMBEDDINGS_API_KEY` | LLM/embeddings endpoints | env only |
| `PGPASSWORD` | Postgres | env; passed to `pg_dump`/`pg_restore` via `PGPASSWORD` env — never on argv |
| `DASHBOARD_SESSION_SECRET` | cookie signing | env only |
| `WEBDAV_ENCRYPTION_KEY` | encrypts stored WebDAV source passwords (Fernet) | env only |
| `BACKUP_ENCRYPTION_KEY` | encrypts backup archives (AES-256-GCM) | env only — never in DB |
| WebDAV passwords | knowledge sources + backup target | Fernet-encrypted in Postgres |

**Known caveat:** when `WEBDAV_ENCRYPTION_KEY` is unset, `webdav_store` falls back to a weak development key. Always set it in production. `BACKUP_ENCRYPTION_KEY` has **no** fallback — backups fail closed.

## Transport & input hardening

- WebDAV connections are HTTPS-only (enforced in `WebDAVClient` and the backup router).
- Remote paths are validated (`..` segments rejected).
- Agent workspace writes pass `AgentWorkspaceStore.resolve()` — path containment + symlink rejection + NUL-byte guard; quotas bound file size/count, and the render cache lives outside the workspace root so the orphan sweeper cannot delete it.
- `code_execution` runs in per-run Docker containers (TTL-bounded, token-authenticated). Egress is enforced packet-level inside the container netns by a one-shot sidecar installing a `AGENTIX_EGRESS` iptables chain — it runs as root with `cap_drop=ALL` + `cap_add=NET_ADMIN` only (uid-0 exec is required for the capability to enter the effective set), while the sandbox runs unprivileged with `cap_drop=ALL` and cannot modify the rules. Policy (tool config + per-assistant override): `egress_internet` (default on), `egress_local_network` (default off), `egress_blocked` CIDRs; `169.254.169.254` is always dropped, and `internet=off` also kills the embedded-DNS resolver. Residuals: the token remains readable via `/proc/self/environ` (scrubbed only from `os.environ`), and with `internet=on` DNS-tunnel exfiltration stays theoretically possible. Persistent background jobs (`code_execution` `shell_start`/`shell`) run inside the same container as `sandboxuser` with the same egress policy, the same scrubbed environment (no `SANDBOX_TOKEN`, pip packages via `PYTHONPATH`), `/tmp` write scope and own-process-group lifecycle — bounded to 20 jobs, ~4000-line per-stream scrollback and a 120 s wait cap.
- `RequestSizeLimitMiddleware` + `MAX_*`/`VALIDATION_*` settings bound request size and content.
- Restore requires a **typed confirmation** (filename retyped) plus the `backup:restore` permission; a safety dump is taken before `pg_restore --clean`.

## Logging & API hygiene

- Passwords/secrets are never returned by API responses (`password_set` booleans instead); tool-config secrets (SMTP/IMAP/SSH passwords, API keys, service-account JSON) are masked as `__agentix_secret_set__` in `GET /admin/tools` responses and preserved server-side on update.
- Subprocess stderr is captured for diagnostics; env-based credential passing avoids secrets in process listings.
- Failed validation attempts are counted in the blocked-request metrics (`/admin/metrics/blocked-summary`).

## CI security checks

`.gitlab-ci.yml` runs GitLab SAST, secret detection, container scanning (Trivy), Checkov and Bandit on non-prod branches.
