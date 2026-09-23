#!/usr/bin/env bash
# One-command launcher (macOS / Linux): sets up a virtualenv, installs deps once,
# builds the React frontend if Node is available, and serves everything on :8000.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

if [ ! -d .venv ]; then
  echo "==> Creating virtual environment"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f .venv/.deps-installed ] || [ backend/requirements.txt -nt .venv/.deps-installed ]; then
  echo "==> Installing Python packages (first run takes a few minutes: torch is large)"
  python -m pip install --upgrade pip
  python -m pip install -r backend/requirements.txt
  touch .venv/.deps-installed
fi

if command -v npm >/dev/null 2>&1; then
  if [ ! -d frontend/node_modules ]; then
    echo "==> Installing frontend packages"
    (cd frontend && npm install)
  fi
  echo "==> Building frontend"
  (cd frontend && npm run build)
else
  echo "==> npm not found - using the prebuilt frontend in frontend/dist"
fi

echo ""
echo "  RL Rebalancing Lab is starting at  http://localhost:8000"
echo ""
( sleep 3; (command -v open >/dev/null && open http://localhost:8000) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8000) ) >/dev/null 2>&1 &
cd backend
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
