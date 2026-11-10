# Development

## Setup

```bash
uv sync && source .venv/bin/activate   # deps incl. dev group (pytest & co.)
```

Dashboard / widget:

```bash
cd dashboard && npm ci && npm run dev    # vite dev server
cd widget    && npm ci && npm run dev
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

- Backend tests use **fake async engines/stores** — no live Postgres needed.
- WebDAV paths are tested with mocked `httpx` clients.
- Backup coverage: `test_backup_crypto.py` (round-trip, tamper, wrong key), `test_backup_store.py`, `test_backup_service.py`, `test_backup_scheduler.py`, `test_backup_router.py`.
- **Known issue (pre-existing):** several admin/runtime-config tests fail with `403 Dashboard is disabled` because they construct `AppConfig` via field name while `dashboard_enabled` is declared with `alias="DASHBOARD_ENABLED"` only — the kwarg is silently ignored by pydantic-settings. Fails identically on a clean checkout.
- Restore drill (manual, needs Docker): `tests/e2e/restore_drill.sh` — see [Backup & Recovery](backup-and-recovery.md#restore-drill-acceptance-evidence).
- Documentation screenshots + walkthrough GIFs: regenerate everything with `AGENTIX_PASSWORD='…' docs/update_screenshots.sh` — the engine is `tests/e2e/capture_media.py` (Playwright + imageio-ffmpeg). Screenshots cover every dashboard page plus the public chat in light + dark at 2560×1440 (1440p, 1× scale) into `docs/assets/screenshots/` (`--pages`/`--themes`/`--mode` options). GIFs are recorded per page segment at the same resolution, head-trimmed so no "Loading…" splash ever ships, then concatenated and exported as 1920px-wide looping GIFs (12fps) in `docs/assets/walkthroughs/` — the tour includes read-only sub-interactions per page (expand an assistant, open a chain in the editor, drawers, tabs).

## Build

| Target | Command |
|---|---|
| Dashboard | `cd dashboard && npm run build` (tsc + vite) |
| Widget | `cd widget && npm run build` |
| Backend image | `docker build -t agentix-agent-platform .` |


## Commit conventions

- Configure your git author (`user.name` / `user.email`) as usual before committing.`
- Plain commit messages, no tool/agent attribution footers.
- Keep commits scoped; the backup feature was merged as `store → crypto → client → service/scheduler → router → UI → docs`.

## Code conventions

- Stores: `app/*_store.py`, lazy `ensure_tables()`, SQLAlchemy `text()` with bound params.
- Routers: `app/routers/*.py`, permission via `require_permission`.
- Dashboard: pages in `dashboard/src/components/*Page.tsx`, API wrappers in `src/api/client.ts`, strings in `src/i18n.ts` (de + en).
- Config: only via `AppConfig` fields with `alias="<ENV_NAME>"`.
- Agent runtime: `agent/tools/runner.py` (loop/dedup), `agent/observer.py`, `agent/subagents.py`; workspace `app/agent_workspace.py` + `app/doc_inspect.py`; reasoning families `app/reasoning_params.py`; MCP `app/mcp_client.py`.
- Internal message tags go in `additional_kwargs` (`workspace_images`, `run_tool_result`) — never serialized to the wire.
