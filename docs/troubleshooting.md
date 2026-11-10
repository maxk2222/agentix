# Troubleshooting

Logs: `LOG_DIR` (default `logs/`, rotating, `LOG_FILE_MAX_BYTES`). Raise verbosity with `LOG_LEVEL=DEBUG`.

## Startup

| Symptom | Check |
|---|---|
| App fails to start / DB errors | `PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD` correct? Postgres reachable from the container? |
| `/dashboard` 404 | `DASHBOARD_ENABLED=true` and `dashboard/dist` built (`npm run build`)? |
| `/widget` 404 | `widget/dist` built? |
| RAG init failed in logs | LLM_API_URL/EMBEDDINGS endpoint reachable? The app still serves admin + backup endpoints. |

## OCR / ingestion

- `TESSERACT_CMD` must point to the binary if not on PATH; `OCR_LANG=deu` needs `tesseract-ocr-deu` (installed in the Docker image).
- Scanned PDFs/pptx are OCR'd via the VLM/OCR pipeline — slow on first ingest.

## Postgres notes

- Vector + FTS indexes live in the same DB; `pg_trgm` is used for keyword search.
- All app tables are created lazily on first use — a fresh DB needs no migrations.
- Connection string: asyncpg DSN assembled in `AppConfig.connection_string`.

## Backup & restore

See the failure-mode table in [Backup & Recovery](backup-and-recovery.md#failure-modes). Key diagnostics:

- `agentix_backup_jobs.error` holds the last failure message (per job).
- `GET /admin/backup/config` reports `encryption_configured` and `pg_dump_available` — both must be `true`.
- Verify an archive anytime: `POST /admin/backup/verify/{filename}`.
- Restore requires `pg_terminate_backend` privilege on the DB role; otherwise it aborts *before* touching data.

## Agent & tools

| Symptom | Check |
|---|---|
| Provider rejects requests with `image_url` / "unsupported content" | Model lacks vision — set `workspace.vision_read: false` on the assistant (renders stay available via `inspect`) |
| `view` fails on Office files | LibreOffice missing — install `libreoffice` or set `LIBREOFFICE_PATH`/`SOFFICE_PATH` (present in the Docker image) |
| `view`/`inspect` answers "[#img-N … omitted]" | Expected: run-content dedup — the image is shown once at its latest occurrence; the model can re-`view` any page (disk-cached) |
| `code_execution` errors about Docker | Docker socket reachable? `DOCKER_HOST`/`/var/run/docker.sock` mounted into the app container? |
| `code_execution` cannot reach a host / "network isolated" | Egress policy: `egress_internet=false` or `SANDBOX_NETWORK_INTERNAL=1` blocks all outbound incl. DNS; LAN needs `egress_local_network=true`; specific ranges may sit in `egress_blocked`/`SANDBOX_EGRESS_BLOCKED_CIDRS` |
| Sandbox fails to start with egress/sidecar errors | The sidecar needs `iptables` in the sandbox image + `NET_ADMIN` — rebuild via Dockerfile hash; a stale container/network is auto-recreated, check daemon logs for `iptables` errors |
| pip installs inside sandbox fail | Intentional when `egress_internet=false` (fail-fast error). Otherwise check `SANDBOX_PIP_INDEX_URL` mirror reachability — with `SANDBOX_PIP_ONLY_BINARY=1` pure-source packages are refused |
| Workspace 404 via API | `workspace.enabled` **and** `workspace.api_enabled` set? `conversation` scope additionally needs `?conversation_id=` |
| Renders seem stale | Clear `AGENT_WORKSPACE_RENDER_CACHE` (`<data>/ws_render_cache`, LRU 256 MB/24 h) — safe to delete anytime |

## Metrics / conversations empty

- Confirm traffic is flowing through `/v1/*` (widget `apiUrl` must point to the platform, not to OpenAI).
- Check `dashboard` overview endpoint returns data: `GET /admin/dashboard/overview`.

## Known pre-existing issues

- `AppConfig` kwargs like `dashboard_enabled=True` are ignored (alias-only fields) — affects some tests; use env vars.
- `WEBDAV_ENCRYPTION_KEY` has a weak dev fallback — set it in production (see [Security](security.md)).
