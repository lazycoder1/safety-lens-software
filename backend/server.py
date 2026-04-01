"""
SafetyLens Demo Backend
- Loops videos with YOLO detection, streams annotated frames as MJPEG
- Runs VLM (qwen3-vl) periodically on cameras with demo=yolo+vlm
- Pushes alerts via WebSocket to the React frontend
- Config-driven: all settings from config_manager
"""

import asyncio
import base64
import json
import logging
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

from config_manager import get_config, load_config, save_config, update_config
from logging_config import setup_logging
import alert_store
import telegram_notifier

logger = logging.getLogger("safetylens")

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"  # COCO pretrained YOLO26 — 80 classes, NMS-free
YOLOE_MODEL_PATH = PROJECT_ROOT / "yoloe-11s-seg.pt"
VIDEO_DIR = PROJECT_ROOT / "test-videos"
OLLAMA_URL = "http://localhost:11434/api/generate"

# COCO class names (80 classes) — key classes for safety:
# 0: person, 15: cat, 16: dog, 24: backpack, 26: handbag, 28: suitcase,
# 32: sports ball, 39: bottle, 56: chair, 62: tv, 67: cell phone,
# 72: refrigerator, 73: book
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Safety-relevant COCO classes with colors
SAFETY_CLASSES = {0, 15, 16, 67, 7, 2, 3}  # person, cat, dog, cell_phone, truck, car, motorcycle
CLASS_COLORS = {
    0: (59, 130, 246),   # person - blue
    15: (234, 179, 8),   # cat - yellow
    16: (234, 179, 8),   # dog - yellow
    67: (239, 68, 68),   # cell phone - red
    7: (168, 85, 247),   # truck - purple
    2: (168, 162, 158),  # car - gray
    3: (249, 115, 22),   # motorcycle - orange
}

# Color palette for YOLOe open-vocabulary classes
YOLOE_COLORS = [
    (59, 130, 246),   # blue
    (34, 197, 94),    # green
    (234, 179, 8),    # yellow
    (239, 68, 68),    # red
    (168, 85, 247),   # purple
    (236, 72, 153),   # pink
    (20, 184, 166),   # teal
    (249, 115, 22),   # orange
    (99, 102, 241),   # indigo
    (6, 182, 212),    # cyan
]

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="SafetyLens Demo Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ────────────────────────────────────────────────────────────────────

alert_subscribers: list[WebSocket] = []
model: Optional[YOLO] = None
yoloe_model: Optional[YOLO] = None
yoloe_lock = threading.Lock()
camera_frames: dict[str, Optional[bytes]] = {}
camera_detections: dict[str, list] = {}
vlm_last_results: dict[str, dict] = {}
vlm_lock = threading.Lock()

# Thread management: cam_id -> (Thread, Event)
camera_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
vlm_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}


def load_model():
    global model, yoloe_model
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)

    logger.info("Loading COCO pretrained YOLO26n", extra={"path": str(YOLO_MODEL_PATH)})
    model = YOLO(str(YOLO_MODEL_PATH))
    model.predict(dummy, verbose=False)  # warmup to fuse layers
    logger.info("YOLO26n model warmed up (80 COCO classes, NMS-free)")

    if YOLOE_MODEL_PATH.exists():
        logger.info("Loading YOLOe (YOLO-World) model", extra={"path": str(YOLOE_MODEL_PATH)})
        yoloe_model = YOLO(str(YOLOE_MODEL_PATH))
        # MobileCLIP text encoder doesn't support MPS float64, so set classes on CPU first
        yoloe_model.to("cpu")
        yoloe_model.set_classes(["person"])
        yoloe_model.to("mps")
        yoloe_model.predict(dummy, verbose=False)  # warmup
        logger.info("YOLOe model warmed up")
    else:
        logger.warning("YOLOe model not found", extra={"path": str(YOLOE_MODEL_PATH)})


# ── Alert helpers ────────────────────────────────────────────────────────────

def create_alert(
    camera_id: str,
    rule: str,
    severity: str,
    confidence: float,
    description: str = "",
    source: str = "YOLO",
) -> dict:
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    # Capture snapshot from current frame
    snapshot_jpeg = camera_frames.get(camera_id)
    alert = alert_store.create_alert(
        camera_id=camera_id,
        camera_name=cam.get("name", camera_id),
        zone=cam.get("zone", "Unknown"),
        rule=rule,
        severity=severity,
        confidence=confidence,
        description=description,
        source=source,
        snapshot_jpeg=snapshot_jpeg,
    )
    # Send Telegram notification (fire-and-forget, never blocks)
    try:
        snap_url = alert.get("snapshotUrl")
        snap_full = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1]) if snap_url else None
        telegram_notifier.send_alert(alert, snap_full)
    except Exception:
        logger.exception("Telegram send failed")
    return alert


async def broadcast_alert(msg: dict):
    dead = []
    for ws in alert_subscribers:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        alert_subscribers.remove(ws)


# ── YOLO Processing ─────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, results, camera_id: str) -> tuple[np.ndarray, list]:
    annotated = frame.copy()
    detections = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
                color = CLASS_COLORS.get(cls_id, (200, 200, 200))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

    # Overlay camera info
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    cam_name = cam.get("name", camera_id)
    cam_demo = cam.get("demo", "yolo")
    overlay_text = f"{cam_name} | {cam_demo.upper()}"
    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    count_text = f"{len(detections)} detections"
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1)

    return annotated, detections


def draw_yoloe_detections(frame: np.ndarray, results, camera_id: str, class_names: list[str]) -> tuple[np.ndarray, list]:
    annotated = frame.copy()
    detections = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                color = YOLOE_COLORS[cls_id % len(YOLOE_COLORS)]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

    # Overlay camera info
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    cam_name = cam.get("name", camera_id)
    overlay_text = f"{cam_name} | YOLOE"
    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    count_text = f"{len(detections)} detections"
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (168, 85, 247), 1)

    return annotated, detections


def check_yoloe_violations(detections: list, camera_id: str) -> list:
    """Return candidate violation dicts (NOT yet persisted to DB)."""
    candidates = []
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    yoloe_classes = cam.get("yoloe_classes", [])

    persons = [d for d in detections if d["class"] == "person"]
    if not persons:
        return candidates

    ppe_classes = [c for c in yoloe_classes if c != "person"]
    for ppe in ppe_classes:
        found = [d for d in detections if d["class"] == ppe]
        if len(found) == 0:
            candidates.append({
                "camera_id": camera_id,
                "rule": f"Missing {ppe}",
                "severity": "P2",
                "confidence": max(d["confidence"] for d in persons),
                "description": f"{len(persons)} worker(s) detected without {ppe}",
                "source": "YOLOe",
            })

    return candidates


# Available COCO alert rules — each camera can opt-in to these
COCO_ALERT_RULES = {
    "mobile_phone": {"rule": "Mobile Phone Usage", "severity": "P3", "classes": ["cell phone"]},
    "animal_intrusion": {"rule": "Animal Intrusion", "severity": "P3", "classes": ["dog", "cat"]},
    "person_detected": {"rule": "Person Detected", "severity": "P4", "classes": ["person"]},
    "vehicle_detected": {"rule": "Vehicle Detected", "severity": "P4", "classes": ["truck", "car", "motorcycle"]},
    "zone_intrusion": {"rule": "Zone Intrusion", "severity": "P1", "classes": ["person"]},
}


# ── Zone helpers ─────────────────────────────────────────────────────────────

def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test. Coords are normalized 0-1."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center_normalized(bbox: list[int], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Get normalized center (0-1) of a bounding box [x1,y1,x2,y2]."""
    cx = (bbox[0] + bbox[2]) / 2.0 / frame_w
    cy = (bbox[1] + bbox[3]) / 2.0 / frame_h
    return cx, cy


def check_zone_intrusions(detections: list, camera_id: str, frame_w: int, frame_h: int) -> list:
    """Check if detected persons are inside any restricted zones for this camera."""
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    zones = cam.get("zones", [])
    enabled_rules = cam.get("alert_classes", [])

    if "zone_intrusion" not in enabled_rules or not zones:
        return []

    persons = [d for d in detections if d["class"] == "person"]
    if not persons:
        return []

    candidates = []
    for zone in zones:
        zone_name = zone.get("name", "Unknown Zone")
        zone_type = zone.get("type", "restricted")
        points = zone.get("points", [])
        if len(points) < 3:
            continue

        intruders = 0
        max_conf = 0.0
        for p in persons:
            cx, cy = bbox_center_normalized(p["bbox"], frame_w, frame_h)
            if point_in_polygon(cx, cy, points):
                intruders += 1
                max_conf = max(max_conf, p["confidence"])

        if intruders > 0:
            severity = "P1" if zone_type == "restricted" else "P2"
            candidates.append({
                "camera_id": camera_id,
                "rule": "Zone Intrusion",
                "severity": severity,
                "confidence": max_conf,
                "description": f"{intruders} person(s) in {zone_type} zone '{zone_name}'",
                "source": "YOLO",
            })

    return candidates

# Per-severity cooldown multipliers (base_cooldown * multiplier)
SEVERITY_COOLDOWN_MULT = {"P1": 1, "P2": 1, "P3": 2, "P4": 5}


def check_violations(detections: list, camera_id: str) -> list:
    """Return candidate violation dicts for COCO-based detections, filtered by camera alert_classes config."""
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    # alert_classes: list of rule keys from COCO_ALERT_RULES that this camera should fire
    # Default: mobile_phone + animal_intrusion (NOT person_detected — too noisy)
    enabled_rules = cam.get("alert_classes", ["mobile_phone", "animal_intrusion"])

    candidates = []
    persons = [d for d in detections if d["class"] == "person"]
    phones = [d for d in detections if d["class"] == "cell phone"]
    dogs = [d for d in detections if d["class"] == "dog"]
    cats = [d for d in detections if d["class"] == "cat"]
    trucks = [d for d in detections if d["class"] in ("truck", "car", "motorcycle")]

    if "mobile_phone" in enabled_rules and phones:
        candidates.append({
            "camera_id": camera_id,
            "rule": "Mobile Phone Usage",
            "severity": "P3",
            "confidence": max(d["confidence"] for d in phones),
            "description": f"Mobile phone detected near {len(persons)} worker(s)",
            "source": "YOLO",
        })

    if "animal_intrusion" in enabled_rules:
        animals = dogs + cats
        if animals:
            animal_types = []
            if dogs:
                animal_types.append(f"{len(dogs)} dog(s)")
            if cats:
                animal_types.append(f"{len(cats)} cat(s)")
            candidates.append({
                "camera_id": camera_id,
                "rule": "Animal Intrusion",
                "severity": "P3",
                "confidence": max(d["confidence"] for d in animals),
                "description": f"Animal detected: {', '.join(animal_types)}",
                "source": "YOLO",
            })

    if "person_detected" in enabled_rules and persons:
        candidates.append({
            "camera_id": camera_id,
            "rule": "Person Detected",
            "severity": "P4",
            "confidence": max(d["confidence"] for d in persons),
            "description": f"{len(persons)} person(s) detected",
            "source": "YOLO",
        })

    if "vehicle_detected" in enabled_rules and trucks:
        candidates.append({
            "camera_id": camera_id,
            "rule": "Vehicle Detected",
            "severity": "P4",
            "confidence": max(d["confidence"] for d in trucks),
            "description": f"Vehicle detected",
            "source": "YOLO",
        })

    return candidates


# ── VLM (qwen3-vl) ──────────────────────────────────────────────────────────

def call_vlm(frame: np.ndarray) -> str:
    try:
        cfg = get_config()
        vlm_cfg = cfg["vlm"]

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": vlm_cfg["model"],
                "prompt": vlm_cfg["prompt"],
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": vlm_cfg["temperature"],
                    "num_predict": vlm_cfg["max_tokens"],
                },
            },
            timeout=120,
        )

        if resp.status_code == 200:
            return resp.json().get("response", "No response from VLM")
        return f"VLM error: {resp.status_code}"
    except Exception as e:
        return f"VLM unavailable: {e}"


def vlm_worker(camera_id: str, stop_event: threading.Event):
    while not stop_event.is_set():
        cfg = get_config()
        vlm_cfg = cfg["vlm"]
        interval = vlm_cfg["interval"]

        # Sleep in small increments so we can respond to stop_event
        for _ in range(int(interval)):
            if stop_event.is_set():
                return
            time.sleep(1)

        if not vlm_cfg.get("enabled", True):
            continue

        frame_bytes = camera_frames.get(camera_id)
        if frame_bytes is None:
            continue

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        logger.info("VLM analysis started", extra={"camera_id": camera_id})
        start = time.time()
        result = call_vlm(frame)
        elapsed = time.time() - start
        logger.info("VLM analysis done", extra={"camera_id": camera_id, "elapsed": round(elapsed, 1)})

        with vlm_lock:
            vlm_last_results[camera_id] = {
                "text": result,
                "timestamp": datetime.now().isoformat(),
                "elapsed": round(elapsed, 1),
            }

        # Check for violations using keywords from config
        keywords = vlm_cfg.get("violation_keywords", [])
        is_violation = any(kw in result.lower() for kw in keywords)
        alert = create_alert(
            camera_id=camera_id,
            rule="VLM Scene Analysis",
            severity="P2" if is_violation else "P4",
            confidence=0.92,
            description=result[:200],
            source=f"VLM ({vlm_cfg['model']})",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broadcast_alert({"type": "alert", "data": alert}))
        loop.close()


# ── Video Processing Threads ─────────────────────────────────────────────────

def video_processor(camera_id: str, stop_event: threading.Event):
    cfg = get_config()
    cam = cfg["cameras"][camera_id]
    stream_type = cam.get("stream_type", "file")
    if stream_type == "rtsp":
        video_source = cam.get("rtsp_url", "")
    else:
        video_source = str(VIDEO_DIR / cam["video"])
    demo_mode = cam.get("demo", "yolo")
    yoloe_classes = cam.get("yoloe_classes", ["person"])
    g = cfg["global"]
    target_fps = cam.get("fps", g["target_fps"])
    frame_interval = 1.0 / target_fps
    alert_cooldown = g["alert_cooldown"]
    yolo_conf = g["yolo_conf"]
    jpeg_quality = g["jpeg_quality"]
    inference_width = g["inference_width"]
    device = g["device"]
    last_alert_by_rule: dict[str, float] = {}  # rule_name -> last fire time
    frame_counter = 0
    last_annotated = None

    # Pre-set YOLOe classes if needed
    if demo_mode == "yoloe" and yoloe_model is not None:
        with yoloe_lock:
            yoloe_model.to("cpu")
            yoloe_model.set_classes(yoloe_classes)
            yoloe_model.to("mps")
            logger.info("YOLOe classes set", extra={"camera_id": camera_id})

    while not stop_event.is_set():
        cap = cv2.VideoCapture(video_source)
        if stream_type == "rtsp":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error("Cannot open video source", extra={"camera_id": camera_id, "source": video_source})
            for _ in range(50):
                if stop_event.is_set():
                    return
                time.sleep(0.1)
            continue

        while cap.isOpened() and not stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # Run detection every 3rd frame, reuse last annotated on others
            if frame_counter % 3 == 1:
                try:
                    if demo_mode == "yoloe" and yoloe_model is not None:
                        with yoloe_lock:
                            # Re-read classes from config in case they were updated live
                            current_cfg = get_config()
                            current_cam = current_cfg["cameras"].get(camera_id, {})
                            current_classes = current_cam.get("yoloe_classes", yoloe_classes)
                            if current_classes != yoloe_classes:
                                yoloe_classes = current_classes
                                yoloe_model.to("cpu")
                                yoloe_model.set_classes(yoloe_classes)
                                yoloe_model.to("mps")
                                logger.info("YOLOe classes updated", extra={"camera_id": camera_id})
                            results = yoloe_model.predict(frame, conf=yolo_conf, verbose=False, device=device, imgsz=inference_width)
                        annotated, detections = draw_yoloe_detections(frame, results, camera_id, yoloe_classes)
                    elif model is not None:
                        results = model.predict(frame, conf=yolo_conf, verbose=False, device=device, imgsz=inference_width)
                        annotated, detections = draw_detections(frame, results, camera_id)
                    else:
                        annotated = frame
                        detections = []
                except Exception as e:
                    logger.error("Detection failed", extra={"camera_id": camera_id})
                    annotated = frame
                    detections = []
                last_annotated = annotated
                camera_detections[camera_id] = detections
            elif last_annotated is not None:
                annotated = last_annotated
            else:
                annotated = frame
                camera_detections[camera_id] = []

            # Check for violations (per-rule cooldown — only persist after cooldown)
            now = time.time()
            dets = camera_detections.get(camera_id, [])
            if len(dets) > 0:
                if demo_mode == "yoloe":
                    candidates = check_yoloe_violations(dets, camera_id)
                else:
                    candidates = check_violations(dets, camera_id)

                # Zone intrusion checks (works with both YOLO and YOLOe)
                h_frame, w_frame = frame.shape[:2]
                candidates.extend(check_zone_intrusions(dets, camera_id, w_frame, h_frame))

                for candidate in candidates:
                    rule_key = candidate["rule"]
                    last_fire = last_alert_by_rule.get(rule_key, 0)
                    sev_mult = SEVERITY_COOLDOWN_MULT.get(candidate["severity"], 1)
                    effective_cooldown = alert_cooldown * sev_mult
                    if now - last_fire > effective_cooldown:
                        last_alert_by_rule[rule_key] = now
                        alert = create_alert(
                            camera_id=candidate["camera_id"],
                            rule=candidate["rule"],
                            severity=candidate["severity"],
                            confidence=candidate["confidence"],
                            description=candidate["description"],
                            source=candidate["source"],
                        )
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(broadcast_alert({"type": "alert", "data": alert}))
                            loop.close()
                        except Exception:
                            pass

            # Downscale to 854px wide before JPEG encode
            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854.0 / w
                new_w = 854
                new_h = int(h * scale)
                annotated = cv2.resize(annotated, (new_w, new_h))

            # Encode to JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            camera_frames[camera_id] = buffer.tobytes()

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        if not stop_event.is_set():
            if stream_type == "rtsp":
                logger.warning("RTSP stream dropped, reconnecting in 5s", extra={"camera_id": camera_id})
                for _ in range(50):
                    if stop_event.is_set():
                        return
                    time.sleep(0.1)
            else:
                logger.debug("Video loop restarting", extra={"camera_id": camera_id})


# ── Thread management ────────────────────────────────────────────────────────

def start_camera(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam or not cam.get("enabled", True):
        return

    stop_evt = threading.Event()
    t = threading.Thread(target=video_processor, args=(cam_id, stop_evt), daemon=True)
    t.start()
    camera_threads[cam_id] = (t, stop_evt)
    camera_frames[cam_id] = None
    camera_detections[cam_id] = []
    logger.info("Started video processor", extra={"camera_id": cam_id})

    # Start VLM worker if demo includes vlm
    if cam.get("demo") == "yolo+vlm":
        start_vlm_for_camera(cam_id)


def start_vlm_for_camera(cam_id: str):
    stop_evt = threading.Event()
    t = threading.Thread(target=vlm_worker, args=(cam_id, stop_evt), daemon=True)
    t.start()
    vlm_threads[cam_id] = (t, stop_evt)
    logger.info("Started VLM worker", extra={"camera_id": cam_id})


def stop_camera(cam_id: str):
    if cam_id in camera_threads:
        thread, stop_evt = camera_threads[cam_id]
        stop_evt.set()
        thread.join(timeout=5)
        del camera_threads[cam_id]
    if cam_id in vlm_threads:
        thread, stop_evt = vlm_threads[cam_id]
        stop_evt.set()
        thread.join(timeout=5)
        del vlm_threads[cam_id]
    camera_frames.pop(cam_id, None)
    camera_detections.pop(cam_id, None)


def restart_camera(cam_id: str):
    stop_camera(cam_id)
    start_camera(cam_id)


def restart_all_cameras():
    cfg = get_config()
    for cam_id in list(camera_threads.keys()):
        stop_camera(cam_id)
    for cam_id in cfg["cameras"]:
        start_camera(cam_id)


# ── MJPEG Stream ─────────────────────────────────────────────────────────────

def mjpeg_generator(camera_id: str):
    while True:
        frame_bytes = camera_frames.get(camera_id)
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        cfg = get_config()
        fps = cfg["global"]["target_fps"]
        time.sleep(1.0 / fps)


# ── Pydantic models for request bodies ───────────────────────────────────────

class GlobalConfigUpdate(BaseModel):
    target_fps: Optional[int] = None
    yolo_conf: Optional[float] = None
    jpeg_quality: Optional[int] = None
    inference_width: Optional[int] = None
    device: Optional[str] = None
    alert_cooldown: Optional[int] = None


class VlmConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    interval: Optional[int] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    violation_keywords: Optional[list[str]] = None


class TelegramConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    severities: Optional[list[str]] = None


class CameraCreate(BaseModel):
    name: str
    video: str = ""
    zone: str
    demo: str = "yolo"
    rules: list[str] = []
    enabled: bool = True
    fps: int = 6
    yoloe_classes: list[str] = ["person"]
    stream_type: str = "file"  # "file" | "rtsp"
    rtsp_url: str = ""
    alert_classes: list[str] = ["mobile_phone", "animal_intrusion"]


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    video: Optional[str] = None
    zone: Optional[str] = None
    demo: Optional[str] = None
    rules: Optional[list[str]] = None
    enabled: Optional[bool] = None
    fps: Optional[int] = None
    yoloe_classes: Optional[list[str]] = None
    stream_type: Optional[str] = None
    rtsp_url: Optional[str] = None
    alert_classes: Optional[list[str]] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    setup_logging()
    logger.info("SafetyLens backend starting")
    alert_store.init_db()
    load_model()
    load_config()
    cfg = get_config()
    for cam_id in cfg["cameras"]:
        start_camera(cam_id)


@app.get("/api/stream/{camera_id}")
async def stream(camera_id: str):
    cfg = get_config()
    if camera_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras")
async def get_cameras():
    cfg = get_config()
    result = []
    for cam_id, cam in cfg["cameras"].items():
        result.append({
            "id": cam_id,
            "name": cam["name"],
            "zone": cam["zone"],
            "demo": cam["demo"],
            "rules": cam["rules"],
            "enabled": cam.get("enabled", True),
            "fps": cam.get("fps", cfg["global"]["target_fps"]),
            "video": cam.get("video", ""),
            "yoloe_classes": cam.get("yoloe_classes", ["person"]),
            "stream_type": cam.get("stream_type", "file"),
            "rtsp_url": cam.get("rtsp_url", ""),
            "alert_classes": cam.get("alert_classes", ["mobile_phone", "animal_intrusion"]),
            "status": "online" if cam_id in camera_threads else "offline",
            "detectionsCount": len(camera_detections.get(cam_id, [])),
        })
    return result


@app.get("/api/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    limit: int = Query(200, le=1000),
    offset: int = Query(0),
):
    return alert_store.get_alerts(severity=severity, status=status, camera_id=camera_id, limit=limit, offset=offset)


@app.get("/api/alerts/stats")
async def get_alert_stats():
    return alert_store.get_stats()


@app.get("/api/alerts/time-series")
async def get_alert_time_series(hours: int = Query(24)):
    return alert_store.get_time_series(hours)


@app.put("/api/alerts/{alert_id}/acknowledge")
async def api_acknowledge_alert(alert_id: str):
    result = alert_store.acknowledge_alert(alert_id)
    if not result:
        return JSONResponse({"error": "Alert not found"}, status_code=404)
    await broadcast_alert({"type": "updated", "data": result})
    return result


@app.put("/api/alerts/{alert_id}/resolve")
async def api_resolve_alert(alert_id: str):
    result = alert_store.resolve_alert(alert_id)
    if not result:
        return JSONResponse({"error": "Alert not found"}, status_code=404)
    await broadcast_alert({"type": "updated", "data": result})
    return result


@app.put("/api/alerts/{alert_id}/snooze")
async def api_snooze_alert(alert_id: str, minutes: int = Query(15)):
    result = alert_store.snooze_alert(alert_id, minutes)
    if not result:
        return JSONResponse({"error": "Alert not found"}, status_code=404)
    await broadcast_alert({"type": "updated", "data": result})
    return result


@app.put("/api/alerts/{alert_id}/false-positive")
async def api_false_positive_alert(alert_id: str):
    result = alert_store.mark_false_positive(alert_id)
    if not result:
        return JSONResponse({"error": "Alert not found"}, status_code=404)
    await broadcast_alert({"type": "updated", "data": result})
    return result


@app.get("/api/snapshots/{filename}")
async def serve_snapshot(filename: str):
    filepath = alert_store.SNAPSHOTS_DIR / filename
    if not filepath.exists():
        return JSONResponse({"error": "Snapshot not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/api/vlm/latest")
async def get_vlm_latest():
    with vlm_lock:
        return vlm_last_results


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await ws.accept()
    alert_subscribers.append(ws)
    logger.info("WebSocket connected", extra={"subscribers": len(alert_subscribers)})
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "acknowledge":
                alert_id = msg.get("alertId")
                result = alert_store.acknowledge_alert(alert_id)
                if result:
                    await broadcast_alert({"type": "updated", "data": result})
    except WebSocketDisconnect:
        alert_subscribers.remove(ws)
        logger.info("WebSocket disconnected", extra={"subscribers": len(alert_subscribers)})


@app.get("/api/health")
async def health():
    cfg = get_config()
    return {
        "status": "ok",
        "model": str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH.exists() else "yolov8n.pt (pretrained)",
        "cameras": list(cfg["cameras"].keys()),
        "vlm": f"{cfg['vlm']['model']} via Ollama",
        "alerts_count": alert_store.get_stats()["total"],
    }


# ── Config endpoints ─────────────────────────────────────────────────────────

@app.get("/api/config")
async def api_get_config():
    return get_config()


@app.put("/api/config/global")
async def api_update_global(body: GlobalConfigUpdate):
    cfg = get_config()
    updates = body.model_dump(exclude_none=True)
    cfg["global"].update(updates)
    save_config(cfg)
    restart_all_cameras()
    return cfg["global"]


@app.put("/api/config/vlm")
async def api_update_vlm(body: VlmConfigUpdate):
    cfg = get_config()
    updates = body.model_dump(exclude_none=True)
    cfg["vlm"].update(updates)
    save_config(cfg)
    return cfg["vlm"]


@app.put("/api/config/telegram")
async def api_update_telegram(body: TelegramConfigUpdate):
    cfg = get_config()
    if "telegram" not in cfg:
        cfg["telegram"] = {"enabled": False, "bot_token": "", "chat_id": "", "severities": ["P1", "P2"]}
    updates = body.model_dump(exclude_none=True)
    cfg["telegram"].update(updates)
    save_config(cfg)
    return cfg["telegram"]


@app.post("/api/config/telegram/test")
async def api_test_telegram():
    cfg = get_config()
    tg = cfg.get("telegram", {})
    result = telegram_notifier.test_connection(tg.get("bot_token", ""), tg.get("chat_id", ""))
    return result


# ── Camera CRUD endpoints ────────────────────────────────────────────────────

@app.post("/api/cameras")
async def api_add_camera(body: CameraCreate):
    cfg = get_config()
    # Auto-generate next cam ID
    existing_ids = [int(k.replace("cam", "")) for k in cfg["cameras"] if k.startswith("cam") and k[3:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    cam_id = f"cam{next_id}"

    cam_data = body.model_dump()
    cfg["cameras"][cam_id] = cam_data
    save_config(cfg)
    start_camera(cam_id)
    return {"id": cam_id, **cam_data}


@app.put("/api/cameras/{cam_id}")
async def api_update_camera(cam_id: str, body: CameraUpdate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    updates = body.model_dump(exclude_none=True)
    cfg["cameras"][cam_id].update(updates)
    save_config(cfg)
    restart_camera(cam_id)
    return {"id": cam_id, **cfg["cameras"][cam_id]}


@app.delete("/api/cameras/{cam_id}")
async def api_delete_camera(cam_id: str):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    stop_camera(cam_id)
    del cfg["cameras"][cam_id]
    save_config(cfg)
    return {"deleted": cam_id}


# ── Zone CRUD endpoints ─────────────────────────────────────────────────────

class ZoneCreate(BaseModel):
    name: str
    type: str = "restricted"  # "restricted" | "caution" | "ppe_required"
    color: str = "#dc2626"
    points: list[list[float]]  # [[x,y], ...] normalized 0-1


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    color: Optional[str] = None
    points: Optional[list[list[float]]] = None


@app.get("/api/cameras/{cam_id}/zones")
async def api_get_zones(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam:
        return JSONResponse({"error": "Camera not found"}, status_code=404)
    return cam.get("zones", [])


@app.post("/api/cameras/{cam_id}/zones")
async def api_add_zone(cam_id: str, body: ZoneCreate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    if "zones" not in cfg["cameras"][cam_id]:
        cfg["cameras"][cam_id]["zones"] = []

    zone_id = f"z{len(cfg['cameras'][cam_id]['zones']) + 1}"
    zone = {"id": zone_id, **body.model_dump()}
    cfg["cameras"][cam_id]["zones"].append(zone)

    # Auto-enable zone_intrusion alert class if not already
    alert_classes = cfg["cameras"][cam_id].get("alert_classes", [])
    if "zone_intrusion" not in alert_classes:
        alert_classes.append("zone_intrusion")
        cfg["cameras"][cam_id]["alert_classes"] = alert_classes

    save_config(cfg)
    logger.info("Zone created", extra={"camera_id": cam_id, "zone": zone_id})
    return zone


@app.put("/api/cameras/{cam_id}/zones/{zone_id}")
async def api_update_zone(cam_id: str, zone_id: str, body: ZoneUpdate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    zones = cfg["cameras"][cam_id].get("zones", [])
    for z in zones:
        if z["id"] == zone_id:
            updates = body.model_dump(exclude_none=True)
            z.update(updates)
            save_config(cfg)
            return z

    return JSONResponse({"error": "Zone not found"}, status_code=404)


@app.delete("/api/cameras/{cam_id}/zones/{zone_id}")
async def api_delete_zone(cam_id: str, zone_id: str):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        return JSONResponse({"error": "Camera not found"}, status_code=404)

    zones = cfg["cameras"][cam_id].get("zones", [])
    cfg["cameras"][cam_id]["zones"] = [z for z in zones if z["id"] != zone_id]
    save_config(cfg)
    logger.info("Zone deleted", extra={"camera_id": cam_id, "zone": zone_id})
    return {"deleted": zone_id}


# ── Detection Rules endpoints ────────────────────────────────────────────────

DEFAULT_DETECTION_RULES = [
    {"id": "r1", "name": "Hard Hat Detection", "model": "YOLOE", "promptType": "text", "prompts": ["hard hat", "safety helmet"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 3, "category": "PPE"},
    {"id": "r2", "name": "Safety Vest Detection", "model": "YOLOE", "promptType": "text", "prompts": ["safety vest", "high visibility vest"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 3, "category": "PPE"},
    {"id": "r3", "name": "Safety Goggles", "model": "YOLOE", "promptType": "text", "prompts": ["safety goggles", "protective eyewear"], "confidenceThreshold": 0.45, "severity": "P2", "enabled": True, "camerasCount": 2, "category": "PPE"},
    {"id": "r4", "name": "Safety Harness", "model": "YOLOE", "promptType": "visual", "prompts": ["harness-ref.jpg"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 1, "category": "PPE"},
    {"id": "r5", "name": "Zone Intrusion", "model": "YOLO26", "promptType": "internal", "prompts": ["person"], "confidenceThreshold": 0.6, "severity": "P1", "enabled": True, "camerasCount": 2, "category": "Zone Safety"},
    {"id": "r6", "name": "Person Fall Detection", "model": "YOLO-pose", "promptType": "internal", "prompts": ["pose-keypoints"], "confidenceThreshold": 0.55, "severity": "P1", "enabled": True, "camerasCount": 3, "category": "Emergency"},
    {"id": "r7", "name": "Mobile Phone Usage", "model": "YOLO26", "promptType": "internal", "prompts": ["cell phone"], "confidenceThreshold": 0.6, "severity": "P2", "enabled": True, "camerasCount": 2, "category": "Behavior"},
    {"id": "r8", "name": "Animal Detection", "model": "YOLOE", "promptType": "text", "prompts": ["snake", "dog", "cat"], "confidenceThreshold": 0.4, "severity": "P3", "enabled": True, "camerasCount": 1, "category": "Environment"},
    {"id": "r9", "name": "Gangway Blockage", "model": "VLM", "promptType": "text", "prompts": ["Is the gangway clear and unobstructed?"], "confidenceThreshold": 0.5, "severity": "P3", "enabled": True, "camerasCount": 1, "category": "Environment"},
    {"id": "r10", "name": "Fire / Smoke Detection", "model": "YOLOE", "promptType": "text", "prompts": ["fire", "smoke", "flames"], "confidenceThreshold": 0.35, "severity": "P1", "enabled": True, "camerasCount": 3, "category": "Emergency"},
    {"id": "r11", "name": "Head Cap vs Helmet", "model": "YOLOE", "promptType": "text", "prompts": ["hair net", "head cap", "hard hat"], "confidenceThreshold": 0.45, "severity": "P2", "enabled": False, "camerasCount": 0, "category": "PPE"},
    {"id": "r12", "name": "Forklift Operator Helmet", "model": "YOLOE", "promptType": "text", "prompts": ["forklift", "hard hat"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 1, "category": "PPE"},
]


@app.get("/api/detection-rules")
async def api_get_detection_rules():
    cfg = get_config()
    rules = cfg.get("detection_rules", DEFAULT_DETECTION_RULES)
    return rules


@app.put("/api/detection-rules/{rule_id}/toggle")
async def api_toggle_detection_rule(rule_id: str):
    cfg = get_config()
    if "detection_rules" not in cfg:
        cfg["detection_rules"] = json.loads(json.dumps(DEFAULT_DETECTION_RULES))
    for rule in cfg["detection_rules"]:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_config(cfg)
            return rule
    return JSONResponse({"error": "Rule not found"}, status_code=404)


class DetectionRuleCreate(BaseModel):
    name: str
    model: str
    promptType: str
    prompts: list[str]
    confidenceThreshold: float
    severity: str
    category: str = "Custom"


@app.post("/api/detection-rules")
async def api_create_detection_rule(body: DetectionRuleCreate):
    cfg = get_config()
    if "detection_rules" not in cfg:
        cfg["detection_rules"] = json.loads(json.dumps(DEFAULT_DETECTION_RULES))
    existing_ids = [int(r["id"].replace("r", "")) for r in cfg["detection_rules"] if r["id"].startswith("r") and r["id"][1:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    rule = {
        "id": f"r{next_id}",
        "enabled": True,
        "camerasCount": 0,
        **body.model_dump(),
    }
    cfg["detection_rules"].append(rule)
    save_config(cfg)
    return rule


# ── Video list endpoint ──────────────────────────────────────────────────────

@app.get("/api/alert-rules-available")
async def api_available_alert_rules():
    """Return the available COCO alert rule keys with descriptions."""
    return {
        key: {"rule": val["rule"], "severity": val["severity"], "classes": val["classes"]}
        for key, val in COCO_ALERT_RULES.items()
    }


@app.get("/api/videos")
async def api_list_videos():
    videos = sorted(
        str(f.relative_to(VIDEO_DIR))
        for f in VIDEO_DIR.rglob("*")
        if f.suffix.lower() in (".mp4", ".avi") and f.is_file()
    )
    return videos


# ── Serve frontend (production build) ────────────────────────────────────────

FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """SPA catch-all: serve index.html for any non-API route."""
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
