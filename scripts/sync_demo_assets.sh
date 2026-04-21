#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
IDENTITY_FILE="$WORKSPACE_DIR/.project-keys/vast_techser_ed25519"
REMOTE_USER="root"
REMOTE_HOST=""
REMOTE_PORT="22"
REMOTE_DIR="/opt/safetylens/video-analytics"
SYNC_CONFIG="yes"
SYNC_ENV="no"

log() {
  printf '[sync-assets] %s\n' "$*"
}

warn() {
  printf '[sync-assets][warn] %s\n' "$*" >&2
}

die() {
  printf '[sync-assets][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./scripts/sync_demo_assets.sh --host HOST [options]

Uploads gitignored demo assets to the Vast backend host:
- backend/config.json
- test-videos/
- models/
- root-level model files such as yolo26n.pt and yoloe-11s-seg.pt

Options:
  --host HOST            Vast public IP or hostname
  --port PORT            SSH port. Default: 22
  --user USER            SSH user. Default: root
  --identity PATH        SSH private key. Default: ../.project-keys/vast_techser_ed25519
  --remote-dir PATH      Remote project directory. Default: /opt/safetylens/video-analytics
  --skip-config          Do not upload backend/config.json
  --with-env             Upload .env as well
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value"
      REMOTE_HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      REMOTE_PORT="$2"
      shift 2
      ;;
    --user)
      [[ $# -ge 2 ]] || die "--user requires a value"
      REMOTE_USER="$2"
      shift 2
      ;;
    --identity)
      [[ $# -ge 2 ]] || die "--identity requires a value"
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || die "--remote-dir requires a value"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --skip-config)
      SYNC_CONFIG="no"
      shift
      ;;
    --with-env)
      SYNC_ENV="yes"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -n "$REMOTE_HOST" ]] || die "--host is required"
[[ -f "$IDENTITY_FILE" ]] || die "Missing SSH key at $IDENTITY_FILE"
command -v ssh >/dev/null 2>&1 || die "ssh is required"
command -v rsync >/dev/null 2>&1 || die "rsync is required"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
RSYNC_RSH="ssh -i $IDENTITY_FILE -p $REMOTE_PORT -o ServerAliveInterval=60 -o ServerAliveCountMax=30"

ensure_remote_dirs() {
  ssh -i "$IDENTITY_FILE" -p "$REMOTE_PORT" "$SSH_TARGET" \
    "mkdir -p '$REMOTE_DIR' '$REMOTE_DIR/backend' '$REMOTE_DIR/test-videos' '$REMOTE_DIR/models'"
}

sync_file() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ -f "$src" ]]; then
    log "Uploading $label"
    rsync -az --info=progress2 -e "$RSYNC_RSH" "$src" "$SSH_TARGET:$dst"
  else
    warn "Skipping missing $label: $src"
  fi
}

sync_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ -d "$src" ]]; then
    log "Uploading $label"
    rsync -az --info=progress2 -e "$RSYNC_RSH" "$src/" "$SSH_TARGET:$dst/"
  else
    warn "Skipping missing $label: $src"
  fi
}

ensure_remote_dirs

if [[ "$SYNC_CONFIG" == "yes" ]]; then
  sync_file "$PROJECT_DIR/backend/config.json" "$REMOTE_DIR/backend/config.json" "backend/config.json"
fi

if [[ "$SYNC_ENV" == "yes" ]]; then
  sync_file "$PROJECT_DIR/.env" "$REMOTE_DIR/.env" ".env"
fi

sync_dir "$PROJECT_DIR/test-videos" "$REMOTE_DIR/test-videos" "test-videos"
sync_dir "$PROJECT_DIR/models" "$REMOTE_DIR/models" "models"

sync_file "$PROJECT_DIR/yolo26n.pt" "$REMOTE_DIR/yolo26n.pt" "yolo26n.pt"
sync_file "$PROJECT_DIR/yoloe-11s-seg.pt" "$REMOTE_DIR/yoloe-11s-seg.pt" "yoloe-11s-seg.pt"
sync_file "$PROJECT_DIR/mobileclip_blt.ts" "$REMOTE_DIR/mobileclip_blt.ts" "mobileclip_blt.ts"

cat <<EOF

Demo asset sync complete.

Remote backend path:
  $REMOTE_DIR

If the backend code is already present on the host, finish with:
  ssh -i $IDENTITY_FILE -p $REMOTE_PORT $SSH_TARGET "cd '$REMOTE_DIR' && bash ./scripts/deploy_backend.sh --install-systemd --env-file '$REMOTE_DIR/.env'"
EOF
