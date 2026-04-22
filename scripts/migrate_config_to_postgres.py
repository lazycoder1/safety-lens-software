#!/usr/bin/env python3
"""Import the current JSON backend config into Postgres app_config storage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json


PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from camera_config_utils import normalize_config  # noqa: E402
from config_manager import PG_CONFIG_ID  # noqa: E402


APP_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_config (
    id TEXT PRIMARY KEY,
    config JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy backend/config.json into Postgres app_config storage."
    )
    parser.add_argument(
        "--config",
        default=str(BACKEND_DIR / "config.json"),
        help="Path to the JSON config file to import.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="Postgres DSN. Defaults to DATABASE_URL from the environment.",
    )
    parser.add_argument(
        "--config-id",
        default=PG_CONFIG_ID,
        help="Row id to write inside app_config. Default: %(default)s",
    )
    parser.add_argument(
        "--backup-existing",
        default="",
        help="Optional path to write the existing app_config row before overwrite.",
    )
    return parser.parse_args()


def load_json_config(config_path: Path) -> tuple[dict, bool]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text())
    normalized, changed = normalize_config(payload)
    return normalized, changed


def ensure_app_config_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(APP_CONFIG_TABLE_SQL)
    conn.commit()


def fetch_existing_config(conn, config_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT config FROM app_config WHERE id = %s", (config_id,))
        row = cur.fetchone()
    if not row:
        return None
    payload = row[0]
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def upsert_config(conn, config_id: str, config: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app_config (id, config, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (id) DO UPDATE
            SET config = EXCLUDED.config,
                updated_at = NOW()
            """,
            (config_id, Json(config)),
        )
    conn.commit()


def write_backup(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    database_url = args.database_url.strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required. Pass --database-url or export DATABASE_URL.")

    config, normalized_changed = load_json_config(config_path)
    camera_count = len(config.get("cameras", {}))

    with psycopg2.connect(database_url) as conn:
        ensure_app_config_table(conn)
        existing = fetch_existing_config(conn, args.config_id)
        if existing is not None and args.backup_existing:
            write_backup(Path(args.backup_existing).expanduser().resolve(), existing)
        upsert_config(conn, args.config_id, config)
        verified = fetch_existing_config(conn, args.config_id)

    if verified != config:
        raise SystemExit("Postgres verification failed: app_config payload did not round-trip cleanly.")

    summary = {
        "config_path": str(config_path),
        "config_id": args.config_id,
        "camera_count": camera_count,
        "normalized_before_import": normalized_changed,
        "overwrote_existing_row": existing is not None,
        "backup_written": bool(existing is not None and args.backup_existing),
        "next_step": "Set SAFETYLENS_CONFIG_STORE=postgres and restart the backend.",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
