"""
SafetyLens mutable application state — shared across modules.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import WebSocket
from ultralytics import YOLO

from config_manager import get_config
from constants import YOLO_MODEL_PATH, YOLOE_MODEL_PATH

logger = logging.getLogger("safetylens")

# ── Mutable state ───────────────────────────────────────────────────────────

alert_subscribers: list[WebSocket] = []
model: Optional[YOLO] = None
yoloe_model: Optional[YOLO] = None
yoloe_lock = threading.Lock()
camera_frames: dict[str, Optional[bytes]] = {}
camera_clean_frames: dict[str, Optional[bytes]] = {}
camera_detections: dict[str, list] = {}
vlm_last_results: dict[str, dict] = {}
vlm_lock = threading.Lock()

# Thread management: cam_id -> (Thread, Event)
camera_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
vlm_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}


# ── Model loading ───────────────────────────────────────────────────────────

def load_model():
    global model, yoloe_model
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    device = get_config()["global"]["device"]

    logger.info("Loading COCO pretrained YOLO26n", extra={"path": str(YOLO_MODEL_PATH)})
    model = YOLO(str(YOLO_MODEL_PATH))
    model.predict(dummy, verbose=False)  # warmup to fuse layers
    logger.info("YOLO26n model warmed up (80 COCO classes, NMS-free)")

    if YOLOE_MODEL_PATH.exists():
        try:
            logger.info("Loading YOLOe (YOLO-World) model", extra={"path": str(YOLOE_MODEL_PATH)})
            # Copy models to local filesystem to avoid WSL mount read issues
            import shutil, tempfile
            tmpdir = Path(tempfile.gettempdir())
            local_yoloe = tmpdir / YOLOE_MODEL_PATH.name
            if not local_yoloe.exists() or local_yoloe.stat().st_size != YOLOE_MODEL_PATH.stat().st_size:
                shutil.copy2(str(YOLOE_MODEL_PATH), str(local_yoloe))
                logger.info("Copied YOLOe model to local path", extra={"path": str(local_yoloe)})
            mobileclip_src = YOLOE_MODEL_PATH.parent / "backend" / "mobileclip_blt.ts"
            if not mobileclip_src.exists():
                mobileclip_src = Path(__file__).parent / "mobileclip_blt.ts"
            local_mobileclip = tmpdir / "mobileclip_blt.ts"
            if mobileclip_src.exists() and (not local_mobileclip.exists() or local_mobileclip.stat().st_size != mobileclip_src.stat().st_size):
                shutil.copy2(str(mobileclip_src), str(local_mobileclip))
                logger.info("Copied MobileCLIP to local path", extra={"path": str(local_mobileclip)})
            yoloe_model = YOLO(str(local_yoloe))
            # MobileCLIP text encoder doesn't support MPS float64, so set classes on CPU first
            yoloe_model.to("cpu")
            # Temporarily switch CWD so set_classes finds MobileCLIP from local fs
            orig_cwd = os.getcwd()
            os.chdir(str(tmpdir))
            try:
                yoloe_model.set_classes(["person"])
            finally:
                os.chdir(orig_cwd)
            yoloe_model.to(device)
            yoloe_model.predict(dummy, verbose=False)  # warmup
            logger.info("YOLOe model warmed up")
        except Exception:
            logger.exception("Failed to load YOLOe model, continuing without it")
            yoloe_model = None
    else:
        logger.warning("YOLOe model not found", extra={"path": str(YOLOE_MODEL_PATH)})
