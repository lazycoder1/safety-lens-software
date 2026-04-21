"""
Config manager for SafetyLens backend.
Thread-safe config loading, saving, and updating with atomic writes.
"""

import json
import os
import threading
from copy import deepcopy
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

from camera_connection import redact_camera_secrets
from camera_config_utils import normalize_config

CONFIG_PATH = Path(__file__).parent / "config.json"
CONFIG_STORE_ENV = "SAFETYLENS_CONFIG_STORE"
PG_CONFIG_ID = "default"

_lock = threading.Lock()
_config: dict | None = None

DEFAULT_CONFIG = {
    "database": {
        "url": "postgresql://localhost:5432/safetylens",
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
        "severities": ["P1", "P2"],
    },
    "global": {
        "target_fps": 6,
        "yolo_conf": 0.35,
        "jpeg_quality": 60,
        "inference_width": 640,
        "device": "mps",
        "alert_cooldown": 60,
    },
    "vlm": {
        "enabled": True,
        "interval": 45,
        "model": "qwen3-vl:8b",
        "prompt": (
            "You are a factory safety inspector AI. Analyze this warehouse/factory image and answer:\n"
            "1. Is the aisle/gangway clear and unobstructed? Is there any equipment, forklift, or material blocking the path?\n"
            "2. Are all visible workers wearing proper PPE (helmet, vest, goggles)?\n"
            "3. Is there safe distance between workers and any forklifts or heavy equipment?\n"
            "4. Are there any other safety hazards visible?\n"
            "Be specific and concise (3-4 sentences max). If there are violations, state them clearly."
        ),
        "temperature": 0.1,
        "max_tokens": 300,
        "violation_keywords": [
            "not wearing", "missing", "blocked", "obstructed", "hazard",
            "violation", "unsafe", "no helmet", "no vest", "forklift",
            "too close", "proximity", "clearance",
        ],
    },
    "retention": {
        "cleanup_interval_hours": 24,
        "snapshot_retention_days": 30,
        "snapshot_max_bytes": 10 * 1024 * 1024 * 1024,
        "orphan_grace_hours": 24,
        "diagnostics_retention_days": 14,
        "diagnostics_max_bytes": 1 * 1024 * 1024 * 1024,
    },
    "cameras": {
        "cam2": {
            "name": "Warehouse Aisle",
            "video": "warehouse-forklift-aisle.mp4",
            "zone": "Warehouse",
            "demo": "yolo+vlm",
            "rules": ["Person Detection", "Forklift Detection", "Gangway Blockage (VLM)"],
            "enabled": True,
            "fps": 6,
        },
    },
}


def _merge_defaults(current, default):
    """Deep-merge default values into an existing config without overwriting user settings."""
    if isinstance(default, dict):
        merged = {}
        current_dict = current if isinstance(current, dict) else {}
        for key, default_value in default.items():
            merged[key] = _merge_defaults(current_dict.get(key), default_value)
        for key, value in current_dict.items():
            if key not in merged:
                merged[key] = value
        return merged
    if isinstance(default, list):
        return deepcopy(current) if isinstance(current, list) else deepcopy(default)
    return default if current is None else current


def load_config() -> dict:
    """Read config from disk. Creates default config if file is missing."""
    global _config
    with _lock:
        loaded = _load_raw_config_unlocked()
        _config = _merge_defaults(loaded, DEFAULT_CONFIG)
        _config, normalized = normalize_config(_config)
        if normalized or _config != loaded:
            _save_unlocked(_config)
        return _config


def save_config(config: dict) -> None:
    """Atomic write: write to .tmp then rename."""
    with _lock:
        _save_unlocked(config)


def _save_unlocked(config: dict) -> None:
    """Write config to disk (caller must hold _lock)."""
    global _config
    if _resolve_config_store() == "postgres":
        _save_to_postgres(config)
    else:
        _save_to_disk(config)
    _config = config


def get_config() -> dict:
    """Return current in-memory config, loading from disk on first call."""
    global _config
    if _config is None:
        return load_config()
    with _lock:
        return _config


def update_config(path: str, value) -> dict:
    """Update a nested key (dot-separated path like 'global.target_fps') and save."""
    config = get_config()
    keys = path.split(".")
    obj = config
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = value
    save_config(config)
    return config


def get_redacted_config() -> dict:
    """Return a copy of config with secrets and credentials masked."""
    config = json.loads(json.dumps(get_config()))

    telegram = config.get("telegram", {})
    if telegram.get("bot_token"):
        telegram["bot_token"] = "***redacted***"

    auth = config.get("auth", {})
    if auth.get("jwt_secret"):
        auth["jwt_secret"] = "***redacted***"

    database = config.get("database", {})
    db_url = database.get("url")
    if db_url:
        database["url"] = _redact_database_url(db_url)

    cameras = config.get("cameras", {})
    for camera_id, camera in list(cameras.items()):
        cameras[camera_id] = redact_camera_secrets(camera)

    return config


def get_public_config() -> dict:
    """Return config data safe for normal API responses."""
    config = json.loads(json.dumps(get_config()))
    cameras = config.get("cameras", {})
    for camera_id, camera in list(cameras.items()):
        cameras[camera_id] = redact_camera_secrets(camera)
    return config


def _redact_database_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, remainder = url.split("://", 1)
    credentials, host = remainder.split("@", 1)
    if ":" in credentials:
        username, _password = credentials.split(":", 1)
        return f"{scheme}://{username}:***redacted***@{host}"
    return f"{scheme}://***redacted***@{host}"


def _resolve_config_store() -> str:
    raw_value = os.environ.get(CONFIG_STORE_ENV, "json").strip().lower()
    if raw_value == "auto":
        return "postgres" if os.environ.get("DATABASE_URL") else "json"
    return "postgres" if raw_value in {"postgres", "pg"} else "json"


def _load_raw_config_unlocked() -> dict:
    if _resolve_config_store() == "postgres":
        loaded = _load_from_postgres()
        if loaded is not None:
            return loaded
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _save_to_disk(config: dict) -> None:
    tmp_path = str(CONFIG_PATH) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    os.rename(tmp_path, CONFIG_PATH)


def _get_postgres_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required when SAFETYLENS_CONFIG_STORE=postgres")
    return dsn


def _ensure_pg_config_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_config (
                id TEXT PRIMARY KEY,
                config JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    conn.commit()


def _load_from_postgres() -> dict | None:
    with psycopg2.connect(_get_postgres_dsn()) as conn:
        _ensure_pg_config_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT config FROM app_config WHERE id = %s", (PG_CONFIG_ID,))
            row = cur.fetchone()
            if not row:
                return None
            payload = row[0]
            if isinstance(payload, str):
                return json.loads(payload)
            return payload


def _save_to_postgres(config: dict) -> None:
    with psycopg2.connect(_get_postgres_dsn()) as conn:
        _ensure_pg_config_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_config (id, config, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                SET config = EXCLUDED.config,
                    updated_at = NOW()
                """,
                (PG_CONFIG_ID, Json(config)),
            )
        conn.commit()
