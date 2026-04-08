"""
SafetyLens video processing — alert creation, VLM, MJPEG streaming, thread management.
"""

import asyncio
import base64
import logging
import os
import time
import threading
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from config_manager import get_config, save_config
from constants import VIDEO_DIR, OLLAMA_URL, YOLOE_COLORS, VIOLATION_THRESHOLD
from detection import (
    draw_detections,
    check_violations,
    check_yoloe_violations,
    check_zone_intrusions,
    extract_violation_bboxes,
)
import state
import alert_store
import telegram_notifier

logger = logging.getLogger("safetylens")


# ── Alert helpers ───────────────────────────────────────────────────────────

def create_alert(
    camera_id: str,
    rule: str,
    severity: str,
    confidence: float,
    description: str = "",
    source: str = "YOLO",
    bboxes: list[dict] | None = None,
) -> dict | None:
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    # Capture snapshot from current frame — skip alert if no frame available yet
    snapshot_jpeg = state.camera_frames.get(camera_id)
    clean_snapshot_jpeg = state.camera_clean_frames.get(camera_id)
    if not snapshot_jpeg:
        logger.debug("Skipping alert — no frame captured yet", extra={"camera_id": camera_id, "rule": rule})
        return None
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
        bboxes=bboxes,
        clean_snapshot_jpeg=clean_snapshot_jpeg,
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
    for ws in state.alert_subscribers:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.alert_subscribers.remove(ws)


# ── VLM (qwen3-vl) ─────────────────────────────────────────────────────────

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

        frame_bytes = state.camera_frames.get(camera_id)
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

        with state.vlm_lock:
            state.vlm_last_results[camera_id] = {
                "text": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        if alert:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(broadcast_alert({"type": "alert", "data": alert}))
            loop.close()


# ── Video Processing Threads ────────────────────────────────────────────────

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
    active_violations: set[str] = set()  # currently active violation rules (fire once until cleared)
    violation_streak: dict[str, int] = {}  # rule -> consecutive frames with violation
    frame_counter = 0
    last_annotated = None

    # Pre-set YOLOe classes if needed
    if demo_mode == "yoloe" and state.yoloe_model is not None:
        with state.yoloe_lock:
            state.yoloe_model.to("cpu")
            import tempfile
            orig_cwd = os.getcwd()
            os.chdir(tempfile.gettempdir())
            try:
                state.yoloe_model.set_classes(yoloe_classes)
            finally:
                os.chdir(orig_cwd)
            state.yoloe_model.to(device)
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
                    if demo_mode == "yoloe" and state.yoloe_model is not None:
                        with state.yoloe_lock:
                            # Re-read classes from config in case they were updated live
                            current_cfg = get_config()
                            current_cam = current_cfg["cameras"].get(camera_id, {})
                            current_classes = current_cam.get("yoloe_classes", yoloe_classes)
                            if current_classes != yoloe_classes:
                                yoloe_classes = current_classes
                                state.yoloe_model.to("cpu")
                                import tempfile
                                _cwd = os.getcwd()
                                os.chdir(tempfile.gettempdir())
                                try:
                                    state.yoloe_model.set_classes(yoloe_classes)
                                finally:
                                    os.chdir(_cwd)
                                state.yoloe_model.to(device)
                                logger.info("YOLOe classes updated", extra={"camera_id": camera_id})
                            results = state.yoloe_model.predict(frame, conf=yolo_conf, verbose=False, device=device, imgsz=inference_width)
                        annotated, detections = draw_detections(frame, results, camera_id, class_names=yoloe_classes, colors=YOLOE_COLORS, demo_label="YOLOE")
                    elif state.model is not None:
                        results = state.model.predict(frame, conf=yolo_conf, verbose=False, device=device, imgsz=inference_width)
                        annotated, detections = draw_detections(frame, results, camera_id)
                    else:
                        annotated = frame
                        detections = []
                except Exception as e:
                    logger.error("Detection failed", extra={"camera_id": camera_id})
                    annotated = frame
                    detections = []
                last_annotated = annotated
                state.camera_detections[camera_id] = detections
            elif last_annotated is not None:
                annotated = last_annotated
            else:
                annotated = frame
                state.camera_detections[camera_id] = []

            # Check for violations (state-based with streak: must persist N frames before firing)
            dets = state.camera_detections.get(camera_id, [])
            if len(dets) > 0:
                if demo_mode == "yoloe":
                    candidates = check_yoloe_violations(dets, camera_id)
                    candidates.extend(check_violations(dets, camera_id))
                else:
                    candidates = check_violations(dets, camera_id)

                # Zone intrusion checks (works with both YOLO and YOLOe)
                h_frame, w_frame = frame.shape[:2]
                candidates.extend(check_zone_intrusions(dets, camera_id, w_frame, h_frame))

                # Determine which violations are present this frame
                current_violation_rules = {c["rule"] for c in candidates}

                # Reset streak for violations that cleared this frame
                for rule_key in list(violation_streak.keys()):
                    if rule_key not in current_violation_rules:
                        violation_streak.pop(rule_key, None)
                # Clear active violations that are no longer detected
                active_violations -= (active_violations - current_violation_rules)

                for candidate in candidates:
                    rule_key = candidate["rule"]
                    if rule_key in active_violations:
                        continue  # already fired, skip until cleared
                    # Increment streak
                    violation_streak[rule_key] = violation_streak.get(rule_key, 0) + 1
                    if violation_streak[rule_key] >= VIOLATION_THRESHOLD:
                        active_violations.add(rule_key)
                        violation_streak.pop(rule_key, None)
                        violation_bboxes = extract_violation_bboxes(candidate["rule"], dets, w_frame, h_frame)
                        alert = create_alert(
                            camera_id=candidate["camera_id"],
                            rule=candidate["rule"],
                            severity=candidate["severity"],
                            confidence=candidate["confidence"],
                            description=candidate["description"],
                            source=candidate["source"],
                            bboxes=violation_bboxes,
                        )
                        if alert:
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(broadcast_alert({"type": "alert", "data": alert}))
                                loop.close()
                            except Exception:
                                pass
            else:
                # No detections -- clear all state
                active_violations.clear()
                violation_streak.clear()

            # Downscale to 854px wide before JPEG encode
            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854.0 / w
                new_w = 854
                new_h = int(h * scale)
                clean_resized = cv2.resize(frame, (new_w, new_h))
                annotated = cv2.resize(annotated, (new_w, new_h))
            else:
                clean_resized = frame

            # Encode to JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            state.camera_frames[camera_id] = buffer.tobytes()

            # Store clean (unannotated) frame for violation snapshots
            _, clean_buffer = cv2.imencode(".jpg", clean_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            state.camera_clean_frames[camera_id] = clean_buffer.tobytes()

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


# ── Thread management ───────────────────────────────────────────────────────

def start_camera(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam or not cam.get("enabled", True):
        return

    stop_evt = threading.Event()
    t = threading.Thread(target=video_processor, args=(cam_id, stop_evt), daemon=True)
    t.start()
    state.camera_threads[cam_id] = (t, stop_evt)
    state.camera_frames[cam_id] = None
    state.camera_detections[cam_id] = []
    logger.info("Started video processor", extra={"camera_id": cam_id})

    # Start VLM worker if demo includes vlm
    if cam.get("demo") == "yolo+vlm":
        start_vlm_for_camera(cam_id)


def start_vlm_for_camera(cam_id: str):
    stop_evt = threading.Event()
    t = threading.Thread(target=vlm_worker, args=(cam_id, stop_evt), daemon=True)
    t.start()
    state.vlm_threads[cam_id] = (t, stop_evt)
    logger.info("Started VLM worker", extra={"camera_id": cam_id})


def stop_camera(cam_id: str):
    if cam_id in state.camera_threads:
        thread, stop_evt = state.camera_threads[cam_id]
        stop_evt.set()
        thread.join(timeout=5)
        del state.camera_threads[cam_id]
    if cam_id in state.vlm_threads:
        thread, stop_evt = state.vlm_threads[cam_id]
        stop_evt.set()
        thread.join(timeout=5)
        del state.vlm_threads[cam_id]
    state.camera_frames.pop(cam_id, None)
    state.camera_detections.pop(cam_id, None)


def restart_camera(cam_id: str):
    stop_camera(cam_id)
    start_camera(cam_id)


def restart_all_cameras():
    cfg = get_config()
    for cam_id in list(state.camera_threads.keys()):
        stop_camera(cam_id)
    for cam_id in cfg["cameras"]:
        start_camera(cam_id)


# ── MJPEG Stream ────────────────────────────────────────────────────────────

def mjpeg_generator(camera_id: str):
    while True:
        frame_bytes = state.camera_frames.get(camera_id)
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        cfg = get_config()
        fps = cfg["global"]["target_fps"]
        time.sleep(1.0 / fps)
