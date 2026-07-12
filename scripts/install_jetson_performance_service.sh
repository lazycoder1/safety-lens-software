#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer as root." >&2
  exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
unit_name=rakshak-lens-jetson-performance.service
unit_source="$repo_root/deploy/systemd/$unit_name"
unit_target="/etc/systemd/system/$unit_name"

if [ ! -f "$unit_source" ]; then
  echo "Missing systemd unit: $unit_source" >&2
  exit 1
fi

install -m 0644 "$unit_source" "$unit_target"
systemctl daemon-reload
systemctl enable --now "$unit_name"
systemctl --no-pager --full status "$unit_name"
