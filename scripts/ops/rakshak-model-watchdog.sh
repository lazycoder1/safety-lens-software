#!/usr/bin/env bash
set -euo pipefail

MODEL_CONTAINER="${MODEL_CONTAINER:-rakshak-model-server}"
EDGE_CONTAINER="${EDGE_CONTAINER:-rakshak-edge}"
MODEL_HEALTH_URL="${MODEL_HEALTH_URL:-http://127.0.0.1:8100/api/health}"
BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-http://127.0.0.1:8000/api/health}"
HEALTH_CHECK_MODE="${HEALTH_CHECK_MODE:-container}"
CURL_TIMEOUT_SECONDS="${CURL_TIMEOUT_SECONDS:-5}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-2}"
FAIL_STATE_FILE="${FAIL_STATE_FILE:-/run/rakshak-model-watchdog.failures}"
RESTART_EDGE_ON_MODEL_RECOVERY="${RESTART_EDGE_ON_MODEL_RECOVERY:-0}"

log() {
  logger -t rakshak-model-watchdog -- "$*"
  printf '%s\n' "$*"
}

current_failures() {
  if [[ -f "$FAIL_STATE_FILE" ]]; then
    tr -dc '0-9' < "$FAIL_STATE_FILE"
  else
    printf '0'
  fi
}

write_failures() {
  local value="$1"
  install -d -m 0755 "$(dirname "$FAIL_STATE_FILE")"
  printf '%s\n' "$value" > "$FAIL_STATE_FILE"
}

container_http_healthy() {
  local container="$1"
  local url="$2"
  [[ "$(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null)" == "true" ]] || return 1
  docker exec "$container" python3 -c '
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])) as response:
    if not 200 <= response.status < 300:
        raise SystemExit(1)
' "$url" "$CURL_TIMEOUT_SECONDS" >/dev/null 2>&1
}

http_healthy() {
  local container="$1"
  local url="$2"
  if [[ "$HEALTH_CHECK_MODE" == "container" ]]; then
    container_http_healthy "$container" "$url"
  else
    curl -fsS --max-time "$CURL_TIMEOUT_SECONDS" "$url" >/dev/null
  fi
}

if http_healthy "$MODEL_CONTAINER" "$MODEL_HEALTH_URL"; then
  write_failures 0
  exit 0
fi

failures="$(current_failures)"
failures="$((failures + 1))"
write_failures "$failures"
log "model server health check failed (${failures}/${FAIL_THRESHOLD})"

if (( failures < FAIL_THRESHOLD )); then
  exit 0
fi

log "restarting ${MODEL_CONTAINER}"
docker restart "$MODEL_CONTAINER" >/dev/null
write_failures 0

for _ in $(seq 1 12); do
  if http_healthy "$MODEL_CONTAINER" "$MODEL_HEALTH_URL"; then
    log "model server recovered"
    if [[ "$RESTART_EDGE_ON_MODEL_RECOVERY" == "1" ]]; then
      if ! http_healthy "$EDGE_CONTAINER" "$BACKEND_HEALTH_URL"; then
        log "backend health check failed after model recovery; restarting ${EDGE_CONTAINER}"
        docker restart "$EDGE_CONTAINER" >/dev/null
      fi
    fi
    exit 0
  fi
  sleep 5
done

log "model server did not recover after restart"
exit 1
