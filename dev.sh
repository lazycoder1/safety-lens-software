#!/usr/bin/env bash
# Start the full Rakshak Lens dev stack: Postgres + Model Server + Backend + Frontend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
MODEL_SERVER_PID=""
BACKEND_PID=""
FRONTEND_PID=""
MODEL_SERVER_URL="${SAFETYLENS_MODEL_SERVER_URL:-http://127.0.0.1:8100}"
BACKEND_URL="${SAFETYLENS_BACKEND_URL:-http://127.0.0.1:8000}"

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
  [ -n "$MODEL_SERVER_PID" ] && kill "$MODEL_SERVER_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local max_attempts="$4"

  echo "Waiting for $name to be ready..."
  for _ in $(seq 1 "$max_attempts"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$name ready."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name process exited unexpectedly."
      return 1
    fi
    sleep 1
  done
  echo "ERROR: $name did not become ready at $url."
  return 1
}

# 1. Database
echo "Starting Postgres..."
docker compose -f "$SCRIPT_DIR/docker-compose.dev.yml" up -d --wait

# 2. Model server
echo "Starting model server..."
cd "$SCRIPT_DIR/backend"
"$VENV/bin/python" -m uvicorn model_server:app --host 0.0.0.0 --port 8100 --reload &
MODEL_SERVER_PID=$!
cd "$SCRIPT_DIR"
wait_for_url "Model server" "$MODEL_SERVER_URL/api/health" "$MODEL_SERVER_PID" 90

# 3. Backend
echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
SAFETYLENS_MODEL_SERVER_URL="$MODEL_SERVER_URL" \
DATABASE_URL="${DATABASE_URL:-postgresql://rakshak_lens:rakshak_lens@localhost:5434/rakshak_lens}" \
"$VENV/bin/python" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$SCRIPT_DIR"
wait_for_url "Backend" "$BACKEND_URL/api/health" "$BACKEND_PID" 60

# 4. Frontend
echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "Rakshak Lens dev stack running:"
echo "  Frontend:  http://localhost:3030"
echo "  Backend:   $BACKEND_URL"
echo "  Models:    $MODEL_SERVER_URL"
echo "  Database:  localhost:5434"
echo ""
echo "Debug check: $VENV/bin/python scripts/runtime_doctor.py"
echo ""
echo "Press Ctrl+C to stop all services."
wait
