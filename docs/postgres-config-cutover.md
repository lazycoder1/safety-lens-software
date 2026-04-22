# Postgres Config Cutover

This runbook moves the live `video-analytics` backend from JSON config storage to Postgres-backed `app_config`.

Current state as of `2026-04-22`:

- `DATABASE_URL` is already present on the Vast demo box.
- `SAFETYLENS_CONFIG_STORE` still resolves to `json`.
- The live camera config is therefore still persisted in `/opt/safetylens/video-analytics/backend/config.json`.

## Files

- Migration script: `scripts/migrate_config_to_postgres.py`
- Live repo path on Vast: `/opt/safetylens/video-analytics`
- Live backend path on Vast: `/opt/safetylens/video-analytics/backend`

## Pre-Checks

1. SSH into the current Vast box.

```bash
ssh -i /Users/gauthamgsabahit/workspace/techser/.project-keys/vast_techser_ed25519 \
  -p 25757 \
  root@ssh3.vast.ai
```

2. Confirm the backend is healthy before the cutover.

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

3. Confirm the current config mode.

```bash
cd /opt/safetylens/video-analytics/backend
set -a
. /opt/safetylens/video-analytics/.env
set +a
/opt/safetylens/video-analytics/.venv/bin/python - <<'PY'
from config_manager import _resolve_config_store
print(_resolve_config_store())
PY
```

Expected output before cutover: `json`

## Cutover Steps

1. Back up the current JSON config.

```bash
cp /opt/safetylens/video-analytics/backend/config.json \
  /opt/safetylens/video-analytics/.deploy/config.backup.$(date +%Y%m%d-%H%M%S).json
```

2. Import the current JSON config into Postgres.

```bash
cd /opt/safetylens/video-analytics
set -a
. ./.env
set +a
./.venv/bin/python ./scripts/migrate_config_to_postgres.py \
  --backup-existing ./.deploy/app_config.backup.$(date +%Y%m%d-%H%M%S).json
```

3. Add this line to `/opt/safetylens/video-analytics/.env`:

```bash
SAFETYLENS_CONFIG_STORE=postgres
```

4. Restart the backend.

```bash
cd /opt/safetylens/video-analytics
./scripts/deploy_backend.sh --skip-pip --skip-systemd
tmux kill-session -t safetylens || true
tmux new-session -d -s safetylens \
  "cd /opt/safetylens/video-analytics/backend && set -a && . /opt/safetylens/video-analytics/.env && set +a && export CUDA_VISIBLE_DEVICES=0 && /opt/safetylens/video-analytics/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000"
sleep 8
curl -fsS http://127.0.0.1:8000/api/health
```

5. Confirm the backend now resolves config storage to Postgres.

```bash
cd /opt/safetylens/video-analytics/backend
set -a
. /opt/safetylens/video-analytics/.env
set +a
/opt/safetylens/video-analytics/.venv/bin/python - <<'PY'
from config_manager import _resolve_config_store
print(_resolve_config_store())
PY
```

Expected output after cutover: `postgres`

## Verification

- Remove a detection from `cam5`, save, reopen edit screen, confirm it stays removed.
- Create or edit a zone, save, reopen camera, confirm the zone still exists.
- Confirm `app_config` row exists:

```bash
cd /opt/safetylens/video-analytics/backend
set -a
. /opt/safetylens/video-analytics/.env
set +a
./.venv/bin/python - <<'PY'
import os
import psycopg2

with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT id, updated_at FROM app_config")
        print(cur.fetchall())
PY
```

## Rollback

1. Remove or comment out `SAFETYLENS_CONFIG_STORE=postgres`.
2. Restart the backend.
3. If needed, restore the JSON backup to `/opt/safetylens/video-analytics/backend/config.json`.

The backend will fall back to JSON storage again on restart.
