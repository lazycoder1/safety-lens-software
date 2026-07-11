# Rakshak Lens Logging & Error Tracking Reference

Quick reference for debugging production issues. Every error source, where it's stored, and how to query it.

---

## Where Errors Are Stored

| Store | What | Retention | How to Query |
|-------|------|-----------|-------------|
| `error_log` table | Frontend crashes, API 500s, WebSocket errors, unhandled exceptions | 30 days | `GET /api/errors?source=frontend&limit=50` |
| `audit_log` table | Admin actions (user approve, alert ack, license upload, config changes) | Unlimited | `GET /api/audit` |
| `alerts` table | Detection alerts, violations, false positives | Active 24h, then auto-resolved | `GET /api/alerts?status=active` |
| `backend/logs/rakshak-lens.log` | INFO+ backend activity — requests, state changes, startup | 25MB x 8 files (200MB total) | `cat logs/rakshak-lens.log \| jq .` |
| `backend/logs/errors.log` | WARNING+ only — errors and warnings. Small, always actionable | 25MB x 8 files | `cat logs/errors.log \| jq .` |
| Console (stdout) | Same as main log; colorized in dev, JSON in prod | Docker default (json-file) | `docker logs rakshak-lens-backend` |

---

## Backend Files That Produce Logs

| File | Logger Name | What It Logs |
|------|-------------|-------------|
| `server.py` | `rakshak_lens` | Every HTTP request (method, route, status, duration_ms, request_id). 500s also written to `error_log` table |
| `error_store.py` | `rakshak_lens.errors` | Manages `error_log` PostgreSQL table. Frontend + backend errors persisted here |
| `audit_store.py` | `rakshak_lens.audit` | Manages `audit_log` table. Who did what, when (user mgmt, alert actions, config) |
| `alert_store.py` | `rakshak_lens.alerts` | Alert creation, state transitions, auto-resolve (24h), snapshot cleanup |
| `video_processing.py` | `rakshak_lens` | Detection loop, VLM calls, Telegram failures, frame capture |
| `licensing.py` | `rakshak_lens.licensing` | License state (VALID/WARNING/GRACE/SUSPENDED), heartbeat refresh |
| `model_manager.py` | `rakshak_lens.models` | Model download, installation jobs, warmup, failures |
| `diagnostics.py` | `rakshak_lens.diagnostics` | Retention cleanup loop, diagnostics bundle generation |
| `telegram_notifier.py` | `rakshak_lens.telegram` | Alert notification delivery success/failure |
| `db.py` | `rakshak_lens.db` | PostgreSQL connection pool init, health check failures |
| `auth_store.py` | `rakshak_lens.auth` | JWT secret generation, persistence warnings |
| `logging_config.py` | — | Configures all of the above (format, rotation, levels) |

---

## Frontend Files That Report Errors

| File | Catches | Reported As |
|------|---------|-------------|
| `lib/api.ts` | HTTP 500s, timeouts (30s), network errors | POSTs to `/api/errors` with `request_id` from `X-Request-ID` header |
| `components/ErrorBoundary.tsx` | React component crashes | POSTs to `/api/errors` with component stack trace |
| `components/AlertProvider.tsx` | WebSocket connection errors | POSTs to `/api/errors` with WebSocket URL |
| `main.tsx` | Unhandled promise rejections (global) | POSTs to `/api/errors` with `type: "unhandledrejection"` |
| `lib/errorReporter.ts` | — | The reporter itself. Fire-and-forget POST to `/api/errors`, never throws |

---

## PostgreSQL Table Schemas

### `error_log` — Error persistence
```
id          TEXT PRIMARY KEY        -- 8-char UUID
timestamp   TEXT NOT NULL           -- ISO 8601 UTC
source      TEXT NOT NULL           -- "frontend" or "backend"
message     TEXT NOT NULL           -- Error message
stack       TEXT                    -- Stack trace (if available)
url         TEXT                    -- Page URL or "GET /api/route"
request_id  TEXT                    -- Correlates to backend logs
context     JSONB DEFAULT '{}'      -- Arbitrary extra data
```

### `audit_log` — Admin action trail
```
id              TEXT PRIMARY KEY
timestamp       TEXT NOT NULL
actor_id        TEXT               -- User ID who performed action
actor_username  TEXT
actor_role      TEXT
action          TEXT NOT NULL      -- e.g. "alert.acknowledge", "user.approve"
target_type     TEXT               -- e.g. "alert", "user", "camera"
target_id       TEXT
outcome         TEXT DEFAULT 'success'
request_id      TEXT
details         JSONB DEFAULT '{}'
```

---

## Log Files

| File | Level | Purpose |
|------|-------|---------|
| `logs/rakshak-lens.log` | INFO+ | Operational log — requests, state changes, startup events |
| `logs/errors.log` | WARNING+ | Errors only — small file, always actionable. Check this first. |
| Console (stdout) | INFO+ | Same as rakshak-lens.log; colorized in dev, JSON in prod |

**Start debugging with `errors.log`** — it only contains warnings and errors, no noise.

## Log File Format

**Files:** `backend/logs/rakshak-lens.log` and `backend/logs/errors.log` (JSON-line, one object per line)

```json
{
  "timestamp": "2026-04-20T10:23:45.123456+00:00",
  "level": "ERROR",
  "service": "rakshak_lens",
  "environment": "prod",
  "host": "edge-server-01",
  "logger": "rakshak_lens",
  "message": "HTTP 500 on /api/alerts",
  "request_id": "a1b2c3d4",
  "method": "POST",
  "route": "/api/alerts",
  "status_code": 500,
  "duration_ms": 245.2,
  "user_id": "60101739",
  "exception": "Traceback (most recent call last):\n  ..."
}
```

Every log line includes `service`, `environment`, and `host` — standard fields for ELK/Datadog/Loki queryability.

Parse with: `cat logs/errors.log | jq .` (start here — small file, only actionable entries)

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAFETYLENS_ENV` | `dev` | `dev` = colorized console, `prod` = JSON console |
| `SAFETYLENS_LOG_LEVEL` | `INFO` | Console log level (DEBUG, INFO, WARNING, ERROR) |
| `SAFETYLENS_LOG_DIR` | `backend/logs` | Log file directory |
| `SAFETYLENS_LOG_MAX_BYTES` | `26214400` (25MB) | Max size before rotation |
| `SAFETYLENS_LOG_BACKUP_COUNT` | `8` | Number of rotated log files kept |

---

## Request-ID Correlation

Every API response includes an `X-Request-ID` header (8-char UUID). Use it to trace:

1. **Frontend error** → `error_log.request_id` → matches backend log entry
2. **Backend log** → `request_id` field in JSON log line
3. **Audit event** → `audit_log.request_id`

Example: User reports "something went wrong" → find the error in `GET /api/errors` → use `request_id` to grep the log file:
```bash
cat logs/rakshak-lens.log | jq 'select(.request_id == "a1b2c3d4")'
```

---

## Docker Log Persistence

```yaml
# docker-compose.yml
volumes:
  - ./logs:/app/backend/logs      # Survives container restarts
  - snapshots:/app/backend/snapshots
```

Without the `./logs` mount, log files are lost on container restart.

---

## Common Debug Queries

```bash
# Recent frontend errors
curl -H "Authorization: Bearer $TOKEN" "localhost:8000/api/errors?source=frontend&limit=20"

# Recent backend errors
curl -H "Authorization: Bearer $TOKEN" "localhost:8000/api/errors?source=backend&limit=20"

# Errors since a specific time
curl -H "Authorization: Bearer $TOKEN" "localhost:8000/api/errors?since=2026-04-20T00:00:00"

# All 500s in the log file
cat logs/rakshak-lens.log | jq 'select(.status_code >= 500)'

# Errors for a specific camera
cat logs/rakshak-lens.log | jq 'select(.camera_id == "cam1" and .level == "ERROR")'

# Audit trail for a user
curl -H "Authorization: Bearer $TOKEN" "localhost:8000/api/audit?actor=admin"
```

---

## What Gets Logged at Each Level

| Level | What | Goes to file? | Examples |
|-------|------|--------------|---------|
| ERROR | Operations that failed | `errors.log` + `rakshak-lens.log` | HTTP 500, DB down, model load failure, unhandled exception |
| WARNING | Recoverable issues | `errors.log` + `rakshak-lens.log` | HTTP 4xx, license expiring, heartbeat refresh failed, slow query |
| INFO | State changes only | `rakshak-lens.log` only | Startup, login, config change, POST/PUT/DELETE requests, slow GET (>2s) |
| DEBUG | Routine operations | Console only (not in any file) | Every GET request, alert creation, frame processing, health checks |

**Rule of thumb:** If it happens more than once per minute during normal operation, it's DEBUG. If it's a state change or something a human would care about, it's INFO.

---

## Best Practices We Follow

Based on Datadog, 12-Factor App, Better Stack, and SigNoz guidelines:

- **JSON structured logging** — every line is parseable by `jq`, ELK, Datadog
- **Standard fields on every line** — `timestamp`, `level`, `service`, `environment`, `host`
- **request_id correlation** — trace frontend error → backend log → audit event
- **Two log files** — `errors.log` (check first) + `rakshak-lens.log` (full operational log)
- **DEBUG disabled in files** — never written to disk, only console during development
- **No sensitive data** — no passwords, tokens, PII in logs
- **Rotating files** — 25MB x 8 backups, auto-rotated
- **Routine GETs at DEBUG** — polling/reads don't fill the log
- **State changes at INFO** — logins, config changes, POST/PUT/DELETE always logged
- **`logger.exception()` for errors** — full stack trace captured, not just message
