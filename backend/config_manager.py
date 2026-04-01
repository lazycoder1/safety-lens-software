"""
Config manager for SafetyLens backend.
Thread-safe config loading, saving, and updating with atomic writes.
"""

import json
import os
import threading
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

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
    "cameras": {
        "cam1": {
            "name": "Welding Bay",
            "video": "construction-workers-helmets.mp4",
            "zone": "Welding Bay",
            "demo": "yolo",
            "rules": ["Hard Hat Detection", "Safety Vest Detection"],
            "enabled": True,
            "fps": 6,
        },
        "cam2": {
            "name": "Warehouse Aisle",
            "video": "warehouse-forklift-aisle.mp4",
            "zone": "Warehouse",
            "demo": "yolo+vlm",
            "rules": ["Person Detection", "Forklift Detection", "Gangway Blockage (VLM)"],
            "enabled": True,
            "fps": 6,
        },
        "cam3": {
            "name": "Food Factory - PPE",
            "video": "factory-workers-hairnet-wide.mp4",
            "zone": "Food Production",
            "demo": "yoloe",
            "rules": ["Hairnet Detection", "Gloves Detection"],
            "yoloe_classes": ["person", "hairnet", "gloves", "face mask", "apron"],
            "enabled": True,
            "fps": 6,
        },
    },
}


def load_config() -> dict:
    """Read config from disk. Creates default config if file is missing."""
    global _config
    with _lock:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                _config = json.load(f)
        else:
            _config = json.loads(json.dumps(DEFAULT_CONFIG))
            _save_unlocked(_config)
        return _config


def save_config(config: dict) -> None:
    """Atomic write: write to .tmp then rename."""
    with _lock:
        _save_unlocked(config)


def _save_unlocked(config: dict) -> None:
    """Write config to disk (caller must hold _lock)."""
    global _config
    tmp_path = str(CONFIG_PATH) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    os.rename(tmp_path, CONFIG_PATH)
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
