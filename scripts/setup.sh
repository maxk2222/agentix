#!/usr/bin/env bash
# One-shot stack setup: generates .env with random secrets and brings the
# full compose stack up. Optional Postgres migration:
#   ./scripts/setup.sh --migrate-from <ext-host> <ext-user> <ext-db>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"
COMPOSE_FILE="$ROOT/docker-compose.example.yml"

rand() { openssl rand -hex 24; }

command -v docker >/dev/null || { echo "docker missing"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose missing"; exit 1; }

if [ ! -f "$ENV_FILE" ]; then
    echo "== Generating .env with random secrets"
    sed \
        -e "s|^PGPASSWORD=.*|PGPASSWORD=$(rand)|" \
        -e "s|^DASHBOARD_SESSION_SECRET=.*|DASHBOARD_SESSION_SECRET=$(rand)|" \
        -e "s|^DASHBOARD_ADMIN_PASSWORD=.*|DASHBOARD_ADMIN_PASSWORD=$(rand)|" \
        -e "s|^REDIS_PASSWORD=.*|REDIS_PASSWORD=$(rand)|" \
        "$ROOT/.env.example.complete" > "$ENV_FILE"
    echo "   .env written — secrets auto-generated. Edit LLM/API keys before use."
else
    echo "== .env exists — reusing (no secrets regenerated)"
    grep -q '^REDIS_PASSWORD=..*' "$ENV_FILE" || \
        sed -i.bak "s|^REDIS_PASSWORD=.*|REDIS_PASSWORD=$(rand)|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
fi

set -a; source "$ENV_FILE"; set +a
: "${PGHOST:=postgres}" "${PGPORT:=5432}" "${PGDATABASE:=agentix-db}" "${PGUSER:=agentix-user}"

echo "== Bringing stack up (postgres + redis + app)"
docker compose -f "$COMPOSE_FILE" up -d

if [ "${1:-}" = "--migrate-from" ]; then
    EXT_HOST="${2:?usage: setup.sh --migrate-from <host> <user> <db>}"
    EXT_USER="${3:?missing <user>}"
    EXT_DB="${4:?missing <db>}"
    echo "== Migrating external Postgres ${EXT_HOST}/${EXT_DB} → bundled"
    DUMP="/tmp/agentix-restore-$$.dump"
    # External password comes from PGPASSWORD_EXTERNAL (.env already holds the
    # NEW bundled credentials). Fall back to an interactive prompt when unset.
    if [ -n "${PGPASSWORD_EXTERNAL:-}" ]; then
        PGPASSWORD="$PGPASSWORD_EXTERNAL" pg_dump -h "$EXT_HOST" -U "$EXT_USER" -d "$EXT_DB" -Fc -f "$DUMP"
    else
        echo "   PGPASSWORD_EXTERNAL unset — prompting for the external password"
        pg_dump -W -h "$EXT_HOST" -U "$EXT_USER" -d "$EXT_DB" -Fc -f "$DUMP"
    fi
    docker cp "$DUMP" agentix-postgres:/tmp/restore.dump
    docker exec agentix-postgres pg_restore -U "$PGUSER" -d "$PGDATABASE" \
        --clean --if-exists --no-owner --no-privileges /tmp/restore.dump
    docker exec agentix-postgres psql -U "$PGUSER" -d "$PGDATABASE" \
        -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;"
    rm -f "$DUMP"
    echo "== Migration done — verify counts, then retire the old DB"
fi

echo "== Ready. App on :8000. Generated secrets live in .env"
