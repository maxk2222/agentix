# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11 | Backend runtime |
| [uv](https://docs.astral.sh/uv/) | Package manager — `pip install uv` or the Astral installer |
| PostgreSQL 14+ with PGVector | **External database required** — the Compose example does not ship one |
| Node.js 20+ | Only for building `dashboard/` and `widget/` from source |
| Tesseract (optional) | OCR for scanned PDFs/images; set `TESSERACT_CMD` if not on PATH |
| LibreOffice (optional) | Headless rendering for `view`/`inspect` on Office docs + `document_generator` exports — already in the Docker image; otherwise set `LIBREOFFICE_PATH`/`SOFFICE_PATH` |
| Docker daemon (optional) | `code_execution` sandbox containers — mount `/var/run/docker.sock` or set `DOCKER_HOST` |
| `pg_dump` / `pg_restore` | Needed for backup/restore — already installed in the Docker image (`postgresql-client`) |

## Local quickstart

```bash
uv sync                       # creates .venv and installs deps incl. dev group
source .venv/bin/activate

cp .env.example.minimum .env
# edit .env: PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD, LLM_API_URL, LLM_API_KEY

python -m app.server
```

The API listens on `API_HOST:API_PORT` (default `127.0.0.1:8000`). Health check: `GET /health`. The HTTP server is `granian` by default; set `API_SERVER=uvicorn` to fall back.

## Dashboard

```bash
cd dashboard && npm ci && npm run build
```

The built SPA is served by the backend at `/dashboard`. Set `DASHBOARD_ENABLED=true` plus `DASHBOARD_ADMIN_PASSWORD` and `DASHBOARD_SESSION_SECRET` in `.env`.

## Widget

```bash
cd widget && npm ci && npm run build
```

The bundle is served at `/widget`. See [Widget Integration](widget.md).

## Docker Compose

```bash
cp .env.example.complete .env   # or .env.example.minimum
docker compose -f docker-compose.example.yml up -d --build
```

Notes:

- The compose file starts **only the application container**; point `PGHOST` at your PostgreSQL instance.
- `./data` is mounted at `/app/data` for caches/models — it does **not** replace database backups.
- The container runs as non-root `appuser`.
