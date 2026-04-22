#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "[frontend-check] frontend/package.json not found" >&2
  exit 1
fi

echo "[frontend-check] Running production frontend build"
cd "$FRONTEND_DIR"
npm run build
