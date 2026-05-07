"""SafetyLens mutable application state — shared across modules."""

import logging
import threading
from typing import Any, Optional

from fastapi import WebSocket

import model_manager

logger = logging.getLogger("safetylens")

# ── Mutable state ───────────────────────────────────────────────────────────

alert_subscribers: list[WebSocket] = []
model: Any = None
yoloe_model: Any = None
camera_frames: dict[str, Optional[bytes]] = {}
camera_clean_frames: dict[str, Optional[bytes]] = {}
camera_detections: dict[str, list] = {}
vlm_last_results: dict[str, dict] = {}
vlm_lock = threading.Lock()
camera_runtime_status: dict[str, str] = {}

# Thread management: cam_id -> (Thread, Event)
camera_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
vlm_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}


# ── Model loading ───────────────────────────────────────────────────────────

def load_model():
    global model, yoloe_model
    logger.info("Initializing model manager")
    model_manager.initialize()
    try:
        model = model_manager._MODEL_RUNTIMES["coco_primary"]["handle"]
        yoloe_model = model_manager._MODEL_RUNTIMES["yoloe_long_tail"]["handle"]
    except Exception:
        logger.debug("Legacy model handle sync unavailable", exc_info=True)
