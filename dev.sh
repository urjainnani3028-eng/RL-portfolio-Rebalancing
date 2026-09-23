#!/usr/bin/env bash
# Development mode: backend with auto-reload on :8000 + Vite dev server with hot reload on :5173
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "Run ./run.sh once first to create the virtualenv"; exit 1; }
(cd backend && python -m uvicorn app.main:app --reload --port 8000) &
BACK=$!
trap 'kill $BACK' EXIT
cd frontend && npm run dev
