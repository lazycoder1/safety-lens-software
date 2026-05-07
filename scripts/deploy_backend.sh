#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
ENV_FILE="$PROJECT_DIR/.env"
VENV_DIR="$PROJECT_DIR/.venv"
SERVICE_NAME="safetylens-backend"
SYSTEMD_MODE="auto"
INSTALL_PACKAGES="yes"
DEPLOY_DIR="$PROJECT_DIR/.deploy"

log() {
  printf '[deploy] %s\n' "$*"
}

warn() {
  printf '[deploy][warn] %s\n' "$*" >&2
}

die() {
  printf '[deploy][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./scripts/deploy_backend.sh [options]

Sets up the SafetyLens backend in a Python virtualenv and optionally installs
or updates a systemd service on the target Linux host.

Options:
  --env-file PATH         Path to the backend env file. Default: ./video-analytics/.env
  --venv-dir PATH         Virtualenv directory. Default: ./video-analytics/.venv
  --service-name NAME     systemd service name. Default: safetylens-backend
  --install-systemd       Fail if the service cannot be installed into systemd
  --skip-systemd          Skip systemd installation and print the manual start command
  --skip-pip              Reuse the existing virtualenv without reinstalling packages
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || die "--env-file requires a value"
      ENV_FILE="$2"
      shift 2
      ;;
    --venv-dir)
      [[ $# -ge 2 ]] || die "--venv-dir requires a value"
      VENV_DIR="$2"
      shift 2
      ;;
    --service-name)
      [[ $# -ge 2 ]] || die "--service-name requires a value"
      SERVICE_NAME="$2"
      shift 2
      ;;
    --install-systemd)
      SYSTEMD_MODE="install"
      shift
      ;;
    --skip-systemd)
      SYSTEMD_MODE="skip"
      shift
      ;;
    --skip-pip)
      INSTALL_PACKAGES="no"
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

[[ -f "$REQUIREMENTS_FILE" ]] || die "Missing requirements file at $REQUIREMENTS_FILE"
[[ -f "$BACKEND_DIR/server.py" ]] || die "Missing backend entrypoint at $BACKEND_DIR/server.py"
command -v python3 >/dev/null 2>&1 || die "python3 is required on the target host"

mkdir -p "$DEPLOY_DIR" "$BACKEND_DIR/logs" "$BACKEND_DIR/snapshots" "$PROJECT_DIR/models"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$PROJECT_DIR/.env.example" ]]; then
    cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
    log "Created $ENV_FILE from .env.example"
  else
    warn "No env file found. The backend will fall back to in-code defaults."
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [[ "$INSTALL_PACKAGES" == "yes" ]]; then
  log "Installing backend dependencies"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

BACKEND_PORT="${SAFETYLENS_PORT:-8000}"

if [[ ! -f "$BACKEND_DIR/config.json" && -f "$BACKEND_DIR/config.example.json" ]]; then
  cp "$BACKEND_DIR/config.example.json" "$BACKEND_DIR/config.json"
  log "Created $BACKEND_DIR/config.json from config.example.json"
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  warn "DATABASE_URL is not set in $ENV_FILE. The backend will default to localhost PostgreSQL."
fi

if [[ -z "${JWT_SECRET:-}" ]]; then
  warn "JWT_SECRET is empty. Tokens will invalidate whenever the backend restarts."
fi

if [[ ! -f "$PROJECT_DIR/yolo26m.pt" && ! -f "$PROJECT_DIR/models/coco_primary/yolo26m.pt" ]]; then
  warn "COCO model missing (yolo26m.pt). Cameras will stay paused until it is installed."
fi

if [[ ! -f "$PROJECT_DIR/yoloe-11s-seg.pt" && ! -f "$PROJECT_DIR/models/yoloe_open_vocab/yoloe-11s-seg.pt" ]]; then
  warn "YOLOE model missing (yoloe-11s-seg.pt). Open-vocabulary detections will stay paused until it is installed."
fi

log "Validating backend imports"
(
  cd "$BACKEND_DIR"
  "$VENV_DIR/bin/python" -c "import server; print(server.app.title)"
) >/dev/null

SERVICE_FILE="$DEPLOY_DIR/${SERVICE_NAME}.service"
cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=SafetyLens backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BACKEND_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/python -m uvicorn server:app --host 0.0.0.0 --port $BACKEND_PORT
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF

log "Generated systemd unit at $SERVICE_FILE"

install_service() {
  local target="/etc/systemd/system/${SERVICE_NAME}.service"
  if [[ "$(id -u)" -eq 0 ]]; then
    install -m 0644 "$SERVICE_FILE" "$target"
    systemctl daemon-reload
    systemctl enable --now "$SERVICE_NAME"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo install -m 0644 "$SERVICE_FILE" "$target"
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$SERVICE_NAME"
    return 0
  fi
  return 1
}

if [[ "$SYSTEMD_MODE" != "skip" ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    if install_service; then
      log "systemd service $SERVICE_NAME is installed and running"
    elif [[ "$SYSTEMD_MODE" == "install" ]]; then
      die "systemd installation requested, but root or sudo access is unavailable"
    else
      warn "Could not install the systemd service automatically. Install $SERVICE_FILE manually."
    fi
  elif [[ "$SYSTEMD_MODE" == "install" ]]; then
    die "systemctl is not available on this host"
  else
    warn "systemctl not found. Skipping service installation."
  fi
fi

cat <<EOF

Backend deployment complete.

Manual start:
  cd $BACKEND_DIR
  $VENV_DIR/bin/python -m uvicorn server:app --host 0.0.0.0 --port $BACKEND_PORT

If you host the frontend separately, point it at:
  VITE_API_URL=https://your-backend-domain
  VITE_WS_URL=wss://your-backend-domain
EOF
