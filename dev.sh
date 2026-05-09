#!/usr/bin/env bash
# Start the full SafetyLens dev stack: Postgres + Backend + Frontend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# 1. Database
echo "Starting Postgres..."
docker compose -f "$SCRIPT_DIR/docker-compose.dev.yml" up -d --wait

# 2. Backend
echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
"$VENV/bin/python" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# 3. Wait for backend health
echo "Waiting for backend to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: Backend process exited unexpectedly."
    exit 1
  fi
  sleep 1
done

# 4. Frontend
echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "SafetyLens dev stack running:"
echo "  Frontend:  http://localhost:3030"
echo "  Backend:   http://localhost:8000"
echo "  Database:  localhost:5434"
echo ""
echo "Press Ctrl+C to stop all services."
wait
