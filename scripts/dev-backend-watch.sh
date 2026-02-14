#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_VENV="$ROOT_DIR/backend/venv"
EXPECTED_PY="$EXPECTED_VENV/bin/python"
PORT="${PORT:-17493}"

if [[ ! -x "$EXPECTED_PY" ]]; then
  echo "[dev-backend-watch] Missing virtualenv python: $EXPECTED_PY" >&2
  echo "[dev-backend-watch] Run: make -C $ROOT_DIR setup-python" >&2
  exit 1
fi

ACTIVE_VENV="${VIRTUAL_ENV:-}"
if [[ -n "$ACTIVE_VENV" && "$ACTIVE_VENV" != "$EXPECTED_VENV" ]]; then
  echo "[dev-backend-watch] Warning: active VIRTUAL_ENV is '$ACTIVE_VENV'" >&2
  echo "[dev-backend-watch] Using project venv instead: '$EXPECTED_VENV'" >&2
fi

echo "[dev-backend-watch] Using python: $EXPECTED_PY"
"$EXPECTED_PY" -c 'import sys; print(f"[dev-backend-watch] Python {sys.version.split()[0]}")'

cd "$ROOT_DIR"
exec "$EXPECTED_PY" -m uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --reload \
  --reload-dir backend
