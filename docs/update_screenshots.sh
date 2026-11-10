#!/usr/bin/env bash
# Regenerate all documentation screenshots + walkthrough GIFs in one go.
#
# Captures every dashboard page plus the public chat at 2560x1440 (1440p,
# 1x scale), records a splash-free click-through tour GIF (1920x1080) and a
# few scenario GIFs — all via tests/e2e/capture_media.py.
#
# Prerequisites:
#   - a running Agentix instance (built dashboard)
#   - .venv with playwright + imageio-ffmpeg + `playwright install chromium`
#
# Usage:
#   AGENTIX_PASSWORD='...' docs/update_screenshots.sh
#   AGENTIX_BASE_URL=https://host AGENTIX_USER=admin docs/update_screenshots.sh
#   docs/update_screenshots.sh --screenshots-only --pages chat overview
#   docs/update_screenshots.sh --walkthroughs-only --scenarios demo-theme
#   docs/update_screenshots.sh --tour-only
#
# Env: AGENTIX_BASE_URL (default http://127.0.0.1:8000), AGENTIX_USER (default admin),
#      AGENTIX_PASSWORD (required), PYTHON_BIN (override interpreter).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL="${AGENTIX_BASE_URL:-http://127.0.0.1:8000}"
USER="${AGENTIX_USER:-admin}"
PASSWORD="${AGENTIX_PASSWORD:-}"
PY="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"

MODE=(screenshots tour scenarios)
EXTRA_PAGES=()
EXTRA_SCENARIOS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --screenshots-only) MODE=(screenshots) ;;
    --walkthroughs-only) MODE=(tour scenarios) ;;
    --tour-only) MODE=(tour) ;;
    --pages) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do EXTRA_PAGES+=("$1"); shift; done; continue ;;
    --scenarios) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do EXTRA_SCENARIOS+=("$1"); shift; done; continue ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$PASSWORD" ]; then
  echo "error: AGENTIX_PASSWORD is not set" >&2
  echo "usage: AGENTIX_PASSWORD='...' $0 [--screenshots-only|--walkthroughs-only|--tour-only] [--pages …] [--scenarios …]" >&2
  exit 2
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "error: no python interpreter found (tried $PY and python3)" >&2
  exit 2
fi

echo "==> preflight: $BASE_URL"
if ! curl -fsS -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
  echo "error: $BASE_URL/health unreachable — start the instance first" >&2
  exit 1
fi
if ! "$PY" -c "import playwright, imageio_ffmpeg" 2>/dev/null; then
  echo "error: playwright/imageio-ffmpeg not installed for $PY" >&2
  echo "       $PY -m pip install playwright imageio-ffmpeg && $PY -m playwright install chromium" >&2
  exit 1
fi

echo "==> capture_media (${MODE[*]}) → docs/assets"
"$PY" "$REPO_ROOT/tests/e2e/capture_media.py" \
  --base-url "$BASE_URL" --user "$USER" --password "$PASSWORD" \
  --shots-out "$REPO_ROOT/docs/assets/screenshots" \
  --gifs-out "$REPO_ROOT/docs/assets/walkthroughs" \
  --mode "${MODE[@]}" \
  ${EXTRA_PAGES[@]+--pages "${EXTRA_PAGES[@]}"} \
  ${EXTRA_SCENARIOS[@]+--scenarios "${EXTRA_SCENARIOS[@]}"}

echo "==> done. Review the regenerated assets before committing."
