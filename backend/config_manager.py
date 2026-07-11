"""
Config manager for Rakshak Lens backend.
Thread-safe config loading, saving, and updating with atomic writes.
"""

import json
import os
import tempfile
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _default_model_server_config() -> dict:
    url = os.environ.get("SAFETYLENS_MODEL_SERVER_URL", "").strip().rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    return {
        "enabled": bool(url),
        "url": url,
        "token": os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", ""),
        "timeout_seconds": _env_float("SAFETYLENS_MODEL_SERVER_TIMEOUT_SECONDS", 30.0),
    }


DEFAULT_ALERT_OUTPUTS = [
    {
        "id": "in_app",
        "name": "In-App Alerts",
        "type": "in_app",
        "enabled": True,
        "severities": ["P1", "P2", "P3", "P4"],
        "zones": [],
        "mode": "websocket",
        "status": "ready",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {},
    },
    {
        "id": "browser_sound",
        "name": "Browser Sound",
        "type": "browser_sound",
        "enabled": True,
        "severities": ["P1"],
        "zones": [],
        "mode": "local_browser",
        "status": "simulated",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"sound": "critical", "repeatSeconds": 0},
    },
    {
        "id": "telegram",
        "name": "Telegram",
        "type": "telegram",
        "enabled": False,
        "severities": ["P1", "P2"],
        "zones": [],
        "mode": "bot",
        "status": "needs_setup",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"bot_token": "", "chat_id": ""},
    },
    {
        "id": "email",
        "name": "Email",
        "type": "email",
        "enabled": False,
        "severities": ["P1", "P2"],
        "zones": [],
        "mode": "smtp",
        "status": "needs_setup",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {
            "provider": "smtp",
            "smtp_host": "",
            "smtp_port": 587,
            "smtp_user": "",
            "smtp_pass": "",
            "sendgrid_api_key": "",
            "sendgrid_template_id": "",
            "from_address": "",
            "from_name": "Rakshak Lens",
            "to_addresses": [],
        },
    },
    {
        "id": "webhook",
        "name": "Webhook",
        "type": "webhook",
        "enabled": False,
        "severities": ["P1", "P2"],
        "zones": [],
        "mode": "http",
        "status": "needs_setup",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"url": "", "headers": {}, "include_snapshot": False},
    },
    {
        "id": "pushover",
        "name": "Pushover iPhone",
        "type": "pushover",
        "enabled": False,
        "severities": ["P1", "P2"],
        "zones": [],
        "mode": "mobile_push",
        "status": "needs_setup",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {
            "app_token": "",
            "user_key": "",
            "device": "",
            "sound": "siren",
            "priority": 1,
            "emergency_retry": 60,
            "emergency_expire": 3600,
        },
    },
    {
        "id": "audio_relay",
        "name": "AudioRelay / Local Audio",
        "type": "ip_speaker",
        "enabled": False,
        "severities": ["P1"],
        "zones": [],
        "mode": "audio_relay",
        "status": "simulated",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"message": "Safety alert. Check the Rakshak Lens dashboard."},
    },
    {
        "id": "ip_speaker",
        "name": "IP Speaker",
        "type": "ip_speaker",
        "enabled": False,
        "severities": ["P1"],
        "zones": [],
        "mode": "http",
        "status": "needs_setup",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {
            "url": "",
            "method": "POST",
            "headers": {},
            "message": "Critical {severity} alert. {violation_type} detected in {zone} on camera {camera}. Check Rakshak Lens dashboard immediately.",
        },
    },
    {
        "id": "relay_buzzer",
        "name": "Relay / Buzzer",
        "type": "relay",
        "enabled": False,
        "severities": ["P1", "P2"],
        "zones": [],
        "mode": "dry_run",
        "status": "simulated",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"pulseSeconds": 5, "url": "", "mqtt_topic": ""},
    },
    {
        "id": "plc",
        "name": "PLC / Modbus",
        "type": "plc",
        "enabled": False,
        "severities": ["P1"],
        "zones": [],
        "mode": "modbus",
        "status": "not_implemented",
        "lastTestAt": None,
        "lastFiredAt": None,
        "lastError": "",
        "settings": {"host": "", "port": 502, "register": "", "coil": ""},
    },
]


DEFAULT_CONFIG = {
    "database": {
        "url": "postgresql://localhost:5432/rakshak_lens",
    },
    "model_server": _default_model_server_config(),
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
        "severities": ["P1", "P2"],
    },
    "email": {
        "enabled": False,
        "smtp_host": "",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_pass": "",
        "from_address": "",
        "to_addresses": [],
        "severities": ["P1", "P2"],
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "headers": {},
        "severities": ["P1", "P2"],
        "include_snapshot": False,
    },
    "alert_outputs": DEFAULT_ALERT_OUTPUTS,
    "alert_routing": {
        "channel_matrix": {
            "P1": {"inApp": True, "telegram": True, "email": True, "webhook": True, "whatsapp": False, "sms": False, "plc": False},
            "P2": {"inApp": True, "telegram": True, "email": True, "webhook": True, "whatsapp": False, "sms": False, "plc": False},
            "P3": {"inApp": True, "telegram": True, "email": False, "webhook": False, "whatsapp": False, "sms": False, "plc": False},
            "P4": {"inApp": True, "telegram": False, "email": False, "webhook": False, "whatsapp": False, "sms": False, "plc": False},
        },
        "timeouts": {
            "Fire/Smoke": {"dedupWindow": 0, "maxAlertsPerHr": 999, "autoResolve": 300, "toastDuration": 0},
            "Person Fall": {"dedupWindow": 0, "maxAlertsPerHr": 999, "autoResolve": 300, "toastDuration": 0},
            "Zone Intrusion": {"dedupWindow": 30, "maxAlertsPerHr": 60, "autoResolve": 600, "toastDuration": 10},
            "Missing-Helmet": {"dedupWindow": 60, "maxAlertsPerHr": 30, "autoResolve": 900, "toastDuration": 10},
            "Missing-Vest": {"dedupWindow": 60, "maxAlertsPerHr": 30, "autoResolve": 900, "toastDuration": 10},
            "Mobile Phone": {"dedupWindow": 60, "maxAlertsPerHr": 20, "autoResolve": 600, "toastDuration": 10},
        },
        "escalation_steps": [
            {"id": 1, "afterMinutes": 3, "role": "Floor Manager", "channel": "telegram"},
            {"id": 2, "afterMinutes": 10, "role": "Plant Manager", "channel": "email"},
        ],
        "templates": {
            "email_subject": "[Rakshak Lens {severity}] {violation_type} detected at {zone}",
            "email_body": "A safety violation has been detected:\n\nViolation: {violation_type}\nSeverity: {severity}\nCamera: {camera}\nZone: {zone}\nTime: {timestamp}\nConfidence: {confidence}\n\nThis is an automated alert from Rakshak Lens.",
            "telegram_template": "*{severity} Alert*\n{violation_type} at {zone}\nCamera: {camera}\nTime: {timestamp}",
        },
    },
    "scheduled_reports": {
        "enabled": False,
        "schedule": "weekly",
        "day_of_week": 1,
        "hour": 6,
        "recipients": [],
    },
    "global": {
        "target_fps": 6,
        "inference_fps": 2.0,
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


_SECRET_OUTPUT_KEYS = {"bot_token", "smtp_pass", "sendgrid_api_key", "app_token", "user_key"}


def _output_defaults_by_id() -> dict[str, dict]:
    return {output["id"]: deepcopy(output) for output in DEFAULT_ALERT_OUTPUTS}


def normalize_alert_outputs(config: dict) -> tuple[dict, bool]:
    """Merge alert output defaults and migrate legacy notification settings."""
    changed = False
    defaults = _output_defaults_by_id()
    existing = config.get("alert_outputs")
    existing_by_id = {item.get("id"): item for item in existing if item.get("id")} if isinstance(existing, list) else {}
    merged_outputs = []

    for output_id, default in defaults.items():
        current = existing_by_id.get(output_id, {})
        merged = _merge_defaults(current, default)
        if not current:
            changed = True
        merged_outputs.append(merged)

    for output_id, output in existing_by_id.items():
        if output_id not in defaults:
            merged_outputs.append(output)

    by_id = {output["id"]: output for output in merged_outputs}

    tg = config.get("telegram", {})
    if "telegram" in by_id:
        settings = by_id["telegram"].setdefault("settings", {})
        for src, dst in (("bot_token", "bot_token"), ("chat_id", "chat_id")):
            if tg.get(src) and not settings.get(dst):
                settings[dst] = tg[src]
                changed = True
        if tg.get("enabled") and not by_id["telegram"].get("enabled"):
            by_id["telegram"]["enabled"] = True
            changed = True
        if tg.get("severities") and by_id["telegram"].get("severities") != tg["severities"]:
            by_id["telegram"]["severities"] = tg["severities"]
            changed = True

    email = config.get("email", {})
    if "email" in by_id:
        settings = by_id["email"].setdefault("settings", {})
        for key in ("smtp_host", "smtp_port", "smtp_user", "smtp_pass", "from_address", "to_addresses"):
            if email.get(key) and not settings.get(key):
                settings[key] = email[key]
                changed = True
        if email.get("enabled") and not by_id["email"].get("enabled"):
            by_id["email"]["enabled"] = True
            changed = True
        if email.get("severities") and by_id["email"].get("severities") != email["severities"]:
            by_id["email"]["severities"] = email["severities"]
            changed = True

    webhook = config.get("webhook", {})
    if "webhook" in by_id:
        settings = by_id["webhook"].setdefault("settings", {})
        for key in ("url", "headers", "include_snapshot"):
            if webhook.get(key) and not settings.get(key):
                settings[key] = webhook[key]
                changed = True
        if webhook.get("enabled") and not by_id["webhook"].get("enabled"):
            by_id["webhook"]["enabled"] = True
            changed = True
        if webhook.get("severities") and by_id["webhook"].get("severities") != webhook["severities"]:
            by_id["webhook"]["severities"] = webhook["severities"]
            changed = True

    if config.get("alert_outputs") != merged_outputs:
        config["alert_outputs"] = merged_outputs
        changed = True
    return config, changed


def redact_alert_outputs(outputs: list[dict]) -> list[dict]:
    redacted = json.loads(json.dumps(outputs or []))
    for output in redacted:
        settings = output.get("settings", {})
        for key in list(settings.keys()):
            if key in _SECRET_OUTPUT_KEYS and settings.get(key):
                settings[key] = "***redacted***"
        headers = settings.get("headers")
        if isinstance(headers, dict):
            for header_key in list(headers.keys()):
                if "auth" in header_key.lower() or "token" in header_key.lower():
                    headers[header_key] = "***redacted***"
    return redacted


def load_config() -> dict:
    """Read config from disk. Creates default config if file is missing."""
    global _config
    with _lock:
        loaded = _load_raw_config_unlocked()
        _config = _merge_defaults(loaded, DEFAULT_CONFIG)
        if isinstance(loaded, dict) and isinstance(loaded.get("cameras"), dict):
            _config["cameras"] = deepcopy(loaded["cameras"])
        _config, outputs_normalized = normalize_alert_outputs(_config)
        _config, normalized = normalize_config(_config)
        if normalized or outputs_normalized or _config != loaded:
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

    email = config.get("email", {})
    if email.get("smtp_pass"):
        email["smtp_pass"] = "***redacted***"

    webhook = config.get("webhook", {})
    for header_key in list(webhook.get("headers", {}).keys()):
        if "auth" in header_key.lower() or "token" in header_key.lower():
            webhook["headers"][header_key] = "***redacted***"

    auth = config.get("auth", {})
    if auth.get("jwt_secret"):
        auth["jwt_secret"] = "***redacted***"

    database = config.get("database", {})
    db_url = database.get("url")
    if db_url:
        database["url"] = _redact_database_url(db_url)

    model_server = config.get("model_server", {})
    if model_server.get("token"):
        model_server["token"] = "***redacted***"

    config["alert_outputs"] = redact_alert_outputs(config.get("alert_outputs", []))

    cameras = config.get("cameras", {})
    for camera_id, camera in list(cameras.items()):
        cameras[camera_id] = redact_camera_secrets(camera)

    return config


def get_public_config() -> dict:
    """Return config data safe for normal API responses."""
    config = json.loads(json.dumps(get_config()))
    model_server = config.get("model_server")
    if isinstance(model_server, dict):
        token = model_server.get("token")
        model_server["token_configured"] = bool(token)
        model_server.pop("token", None)
    config["alert_outputs"] = redact_alert_outputs(config.get("alert_outputs", []))
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
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{CONFIG_PATH.name}.",
        suffix=".tmp",
        dir=CONFIG_PATH.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, CONFIG_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
