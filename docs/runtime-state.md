# Runtime State Boundary

Production images contain application code and the public license-verification
key only. Customer configuration, credentials, license/heartbeat files,
diagnostics, and downloaded models must not be baked into an image.

## Persistent paths

The checked-in Compose files use PostgreSQL for application configuration and
project-scoped named volumes for mutable files:

| Logical volume | Container path | Contents |
|---|---|---|
| `runtime_state` | `/var/lib/safetylens` | JWT signing secret, license, heartbeat, diagnostics |
| `models` | `/app/models` | Downloaded model artifacts and TensorRT engines |

Set a stable, device-unique `COMPOSE_PROJECT_NAME` in the production `.env`
before the first deployment. For example, `safetylens-plant1-jetson1` produces
`safetylens-plant1-jetson1_runtime_state` and
`safetylens-plant1-jetson1_models`. Reusing a project name across installations
would share the license and JWT identity; changing it later would select empty
volumes. Never run `docker compose down -v` against production.

`SAFETYLENS_STATE_DIR` can override the runtime path for a non-Compose
deployment, but it must be absolute and writable by the service account. The
backend creates private directories as mode `0700` and files as mode `0600`.
A JWT secret is generated once inside the runtime-state volume unless a
32-byte-or-longer `JWT_SECRET` or an existing `JWT_SECRET_FILE` is provided.
On first upgraded startup, a matching legacy `auth.jwt_secret` is copied to the
state volume and removed with a targeted config update, preserving existing
tokens. A mismatch fails startup instead of silently changing deployment
identity.

## Upgrade preflight

Do not recreate the current backend until all checks below pass.

1. Pin the production Compose identity and identify the existing containers:

   ```bash
   export COMPOSE_PROJECT_NAME=safetylens-plant1-jetson1
   export CURRENT_BACKEND=rakshak-lens-backend
   export CURRENT_MODEL_CONTAINER=rakshak-lens-model-server
   export STATE_VOLUME="${COMPOSE_PROJECT_NAME}_runtime_state"
   export MODELS_VOLUME="${COMPOSE_PROJECT_NAME}_models"
   ```

2. Verify PostgreSQL is authoritative. The query must return `default`:

   ```bash
   docker compose exec -T db \
     psql -U safetylens -d safetylens -Atc \
     "SELECT id FROM app_config WHERE id = 'default'"
   ```

   If it does not, run `scripts/migrate_config_to_postgres.py` against the
   current `backend/config.json` and verify the row before removing the old file
   bind. Back up PostgreSQL with `pg_dump`; a named volume is not a backup.

3. Copy current file state and models to a private host staging directory:

   ```bash
   install -d -m 700 /opt/rakshak-lens/state-migration/license
   install -d -m 700 /opt/rakshak-lens/state-migration/heartbeat
   install -d -m 700 /opt/rakshak-lens/state-migration/models
   docker cp "$CURRENT_BACKEND":/app/backend/license/current.lic \
     /opt/rakshak-lens/state-migration/license/current.lic
   chmod 600 /opt/rakshak-lens/state-migration/license/current.lic
   if docker exec "$CURRENT_BACKEND" test -f /app/backend/heartbeat/current.json; then
     docker cp "$CURRENT_BACKEND":/app/backend/heartbeat/current.json \
       /opt/rakshak-lens/state-migration/heartbeat/current.json
     chmod 600 /opt/rakshak-lens/state-migration/heartbeat/current.json
   fi
   docker cp "$CURRENT_MODEL_CONTAINER":/app/models/. \
     /opt/rakshak-lens/state-migration/models/
   ```

   If the current deployment already bind-mounts a durable host directory at
   `/app/models`, preserve that bind instead of switching to an empty volume.

4. Hash the staging copy without printing any secret contents:

   ```bash
   sha256sum /opt/rakshak-lens/state-migration/license/current.lic
   if [ -f /opt/rakshak-lens/state-migration/heartbeat/current.json ]; then
     sha256sum /opt/rakshak-lens/state-migration/heartbeat/current.json
   fi
   find /opt/rakshak-lens/state-migration/models -type f -exec sha256sum {} \; \
     | sort > /opt/rakshak-lens/state-migration/models.sha256
   ```

5. If no heartbeat exists, inspect the signed license issue date before
   cutover. The initial deadline is `issued_at + 35 days`, followed by 14 days
   of grace. A license more than 49 days old without a heartbeat will suspend
   immediately by design; obtain a fresh signed heartbeat before upgrading.

## Populate and verify volumes

Create volumes under the pinned project name and copy the staged files:

```bash
docker volume create "$STATE_VOLUME"
docker volume create "$MODELS_VOLUME"
docker run --name safetylens-state-copy \
  -v "$STATE_VOLUME":/state \
  -v "$MODELS_VOLUME":/models \
  -v /opt/rakshak-lens/state-migration:/source:ro \
  alpine:3.20 sh -c \
  'install -d -m 700 /state/license /state/heartbeat &&
   install -m 600 /source/license/current.lic /state/license/current.lic &&
   if [ -f /source/heartbeat/current.json ]; then
     install -m 600 /source/heartbeat/current.json /state/heartbeat/current.json;
   fi &&
   cp -a /source/models/. /models/'
docker rm safetylens-state-copy
```

Recompute hashes from read-only mounts and compare them with the staging
manifest before starting camera workers:

```bash
docker run --rm -v "$STATE_VOLUME":/state:ro alpine:3.20 \
  sha256sum /state/license/current.lic
docker run --rm -v "$MODELS_VOLUME":/models:ro alpine:3.20 sh -c \
  'find /models -type f -exec sha256sum {} \; | sort'
```

After `docker compose up --force-recreate`, verify license status, issue a login
token, recreate once more, and confirm both the status and existing token remain
valid. Keep the PostgreSQL dump and staging copy until a restore drill passes.

## Candidate isolation and build-context gate

Always give an isolated candidate a unique project name, for example
`docker compose -p safetylens-candidate-<commit> ...`. Project-scoped volumes
then cannot mount production JWT, license, database, or model state.

Before publishing an image, seed unique canaries in ignored runtime paths
(`backend/config.json`, license, heartbeat, face media, diagnostics, `.env`, a
private key, and root model files), build from that context, and inspect the
resulting filesystem. No canary or forbidden runtime path may exist in the
final image.
