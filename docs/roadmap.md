# Roadmap & Known Limitations

*(Translated and condensed from `FUTURE_UPDATES.md`.)*

## Planned: Playwright/Chromium in the Docker image for JavaScript rendering

**Goal:** make the web scraper tool (`fetch_with_js` in `agent/tools/builtin_tools/common.py`) render JavaScript-heavy pages.

**Current state:** `playwright` is in `pyproject.toml`; the function exists but aborts with `RuntimeError("Playwright not installed")` / `"Playwright browser not installed"`. The Docker image lacks the Chromium system libraries and browser binaries.

| Option | Trade-off |
|---|---|
| **A — bake Chromium into the image** (recommended for prod) | +200–400 MB image; instant at runtime; needs `--no-sandbox` (container runs as `appuser`) |
| **B — system libs only, browser on demand** | smaller image; first call downloads Chromium (needs internet) |
| **C — runtime install** | smallest image; fragile cold start |

**Open questions:** worth it only if agents frequently hit JS-only pages; `--no-sandbox` lowers browser sandbox security; ~50–200 MB extra RAM/CPU per request.

## Known limitations

- **Test stage in CI is empty** — pytest (~1800 tests) is not run in the pipeline; test deps live in the `dev` dependency group of `pyproject.toml`.
- **No Alembic migrations** — schema evolution relies on `CREATE TABLE IF NOT EXISTS` / lazy `ensure_tables()`.
- **Restore privilege dependency** — DB restores need `pg_terminate_backend` rights on the Postgres role.
- **`WEBDAV_ENCRYPTION_KEY` dev fallback** — weak hardcoded fallback when unset; must be configured in production.
- **pytest not pinned** — install `pytest`/`pytest-asyncio` manually for the test suite.
- **`scope: assistant` workspaces are shared** across all callers — a conscious trust model, not isolation.
- **Image budget** — at most 16 unique images stay live per run; older ones degrade to reference stubs (re-`view` on demand).
