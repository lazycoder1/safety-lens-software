"""SafetyLens video processing — alerts, grouped inference, MJPEG streaming."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

import alert_store
import face_analyzer
import face_store
import inference_scheduler
import licensing
import model_manager
import state
import notification_dispatcher
from camera_capture import (
    CAMERA_STOP_TIMEOUT_SECONDS,
    open_video_capture,
    reconnect_delay_seconds,
    redact_video_source,
)
from camera_connection import build_rtsp_url
from camera_planner import build_execution_plan
from capability_registry import CLASS_TERM_TO_CAPABILITY
from config_manager import get_config
from constants import COCO_NAMES, OLLAMA_URL, VIDEO_DIR, VIOLATION_THRESHOLD
from detection import (
    apply_camera_overlay,
    check_fall_detections,
    check_violations,
    check_yoloe_violations,
    check_zone_intrusions,
    draw_detection_records,
    draw_detections,
    draw_pose_detections,
    extract_violation_bboxes,
)

LICENSE_PAUSE_INTERVAL = 1.0
_last_pose_results: dict[str, any] = {}  # camera_id -> last YOLO-pose results for fall checking

logger = logging.getLogger("safetylens")

_COCO_CLASS_TO_CAPABILITIES = {
    "person": ["person_presence", "zone_intrusion"],
    "cell phone": ["mobile_phone"],
    "dog": ["animal_presence"],
    "cat": ["animal_presence"],
    "deer": ["animal_presence"],
    "animal": ["animal_presence"],
    "car": ["vehicle_presence"],
    "truck": ["vehicle_presence"],
    "motorcycle": ["vehicle_presence"],
}

FACE_LOG_COOLDOWN_SECONDS = 10.0
STREAM_MAX_WIDTH = 854


def _positive_fps(value, fallback: float) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return fallback
    return fps if fps > 0 else fallback


def _configured_stream_fps(camera: dict, global_config: dict, target_fps: float) -> float:
    target = _positive_fps(target_fps, 1.0)
    default = min(target, 4.0)
    configured = camera.get("stream_fps", global_config.get("stream_fps", default))
    return min(target, _positive_fps(configured, default))


def _stream_publish_due(last_published_at: float, now: float, stream_fps: float) -> bool:
    return last_published_at <= 0.0 or now - last_published_at >= 1.0 / stream_fps


def _resize_for_stream(frame: np.ndarray, max_width: int = STREAM_MAX_WIDTH) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)))


def _scale_stream_detection_records(
    detections: list[dict],
    source_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> list[dict]:
    """Copy valid detections into stream coordinates without mutating inference state."""
    source_height, source_width = source_shape[:2]
    output_height, output_width = output_shape[:2]
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    scaled: list[dict] = []
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            min(output_width - 1, max(0, int(round(float(x1) * scale_x)))),
            min(output_height - 1, max(0, int(round(float(y1) * scale_y)))),
            min(output_width - 1, max(0, int(round(float(x2) * scale_x)))),
            min(output_height - 1, max(0, int(round(float(y2) * scale_y)))),
        ]
        scaled.append({**detection, "bbox": scaled_bbox})
    return scaled


def _stream_visible_detection_records(detections: list[dict]) -> list[dict]:
    """Preserve the existing COCO-only live overlay while retaining all analytics records."""
    return [
        detection
        for detection in detections
        if detection.get("model_family") in {None, "coco_primary"}
        and isinstance(detection.get("bbox"), (list, tuple))
        and len(detection["bbox"]) == 4
    ]


def _draw_stream_detection_records(
    frame: np.ndarray,
    detections: list[dict],
    camera_id: str,
    *,
    show_overlay: bool = True,
) -> np.ndarray:
    visible_detections = _stream_visible_detection_records(detections)
    annotated, _ = draw_detection_records(
        frame,
        visible_detections,
        camera_id,
        show_overlay=False,
    )
    if show_overlay:
        annotated = apply_camera_overlay(
            annotated,
            camera_id=camera_id,
            detection_count=len(visible_detections),
        )
    return annotated


def _render_stream_views(
    camera_id: str,
    frame: np.ndarray,
    detections: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize once, then draw stream-only annotations at the output resolution."""
    clean_view = _resize_for_stream(frame)
    visible_detections = _stream_visible_detection_records(detections)
    if clean_view is frame:
        stream_detections = visible_detections
    else:
        stream_detections = _scale_stream_detection_records(
            visible_detections,
            frame.shape,
            clean_view.shape,
        )
    annotated_view = _draw_stream_detection_records(
        clean_view,
        stream_detections,
        camera_id,
    )
    return annotated_view, clean_view


def _encode_stream_jpeg(frame: np.ndarray, jpeg_quality: int) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("Failed to encode stream frame")
    return buffer.tobytes()


def _preserved_source_annotation(
    frame: np.ndarray,
    last_annotated: np.ndarray | None,
    execution_plan: dict,
) -> np.ndarray | None:
    if last_annotated is None or last_annotated.shape[:2] != frame.shape[:2]:
        return None
    if frame.shape[1] <= STREAM_MAX_WIDTH:
        return last_annotated
    if execution_plan.get("run_face_recognition") or execution_plan.get("run_pose_specialist"):
        return last_annotated
    return None


def _publish_stream_frame(
    camera_id: str,
    frame: np.ndarray,
    detections: list[dict],
    *,
    jpeg_quality: int,
    source_annotated: np.ndarray | None = None,
) -> None:
    if source_annotated is None:
        annotated_view, clean_view = _render_stream_views(camera_id, frame, detections)
    else:
        clean_view = _resize_for_stream(frame)
        clean_height, clean_width = clean_view.shape[:2]
        if source_annotated.shape[:2] == clean_view.shape[:2]:
            annotated_view = source_annotated
        else:
            annotated_view = cv2.resize(source_annotated, (clean_width, clean_height))
    annotated_jpeg = _encode_stream_jpeg(annotated_view, jpeg_quality)
    clean_jpeg = _encode_stream_jpeg(clean_view, jpeg_quality)
    state.camera_frames[camera_id] = annotated_jpeg
    state.camera_clean_frames[camera_id] = clean_jpeg


def _clear_camera_observation(camera_id: str) -> None:
    """Discard current-frame state that is invalid once a source disconnects."""
    state.camera_frames[camera_id] = None
    state.camera_clean_frames[camera_id] = None
    state.camera_detections[camera_id] = []
    _last_pose_results.pop(camera_id, None)


def _normalize_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def _normalize_detection_batch(detections: list[dict], model_family: str) -> list[dict]:
    normalized: list[dict] = []
    for detection in detections:
        class_name = detection["class"]
        capability_keys = _COCO_CLASS_TO_CAPABILITIES.get(class_name, [])
        if not capability_keys:
            mapped = CLASS_TERM_TO_CAPABILITY.get(_normalize_text(class_name))
            if mapped:
                capability_keys = [mapped]
        normalized.append({
            **detection,
            "model_family": model_family,
            "capability_keys": capability_keys,
        })
    return normalized


def _detection_batch_from_records(
    records: list[dict],
    model_family: str,
    *,
    class_names: dict[int, str] | list[str] | None = None,
) -> list[dict]:
    """Normalize model records without rendering a full-resolution frame."""
    detections: list[dict] = []
    for record in records:
        class_id = int(record.get("class_id", record.get("cls", 0)))
        if record.get("class"):
            class_name = str(record["class"])
        elif class_names is None:
            class_name = COCO_NAMES.get(class_id, f"class_{class_id}")
        elif isinstance(class_names, list):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        else:
            class_name = class_names.get(class_id, f"class_{class_id}")
        detections.append({
            "class_id": class_id,
            "class": class_name,
            "confidence": float(record.get("confidence", record.get("conf", 0.0))),
            "bbox": list(map(int, record["bbox"])),
        })
    return _normalize_detection_batch(detections, model_family)


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
    try:
        snap_url = alert.get("snapshotUrl")
        snap_full = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1]) if snap_url else None
        notification_dispatcher.notify(alert, snap_full)
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
    except Exception as exc:
        return f"VLM unavailable: {exc}"


def vlm_worker(camera_id: str, stop_event: threading.Event):
    while not stop_event.is_set():
        cfg = get_config()
        vlm_cfg = cfg["vlm"]
        interval = vlm_cfg["interval"]

        for _ in range(int(interval)):
            if stop_event.is_set():
                return
            time.sleep(1)

        if not vlm_cfg.get("enabled", True) or not licensing.is_inference_allowed():
            continue

        frame_bytes = state.camera_frames.get(camera_id)
        if frame_bytes is None:
            continue

        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        logger.info("VLM analysis started", extra={"camera_id": camera_id})
        started = time.time()
        result = call_vlm(frame)
        elapsed = time.time() - started
        logger.info("VLM analysis done", extra={"camera_id": camera_id, "elapsed": round(elapsed, 1)})

        with state.vlm_lock:
            state.vlm_last_results[camera_id] = {
                "text": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed": round(elapsed, 1),
            }

        keywords = vlm_cfg.get("violation_keywords", [])
        is_violation = any(keyword in result.lower() for keyword in keywords)
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


def _run_grouped_inference(camera_id: str, frame: np.ndarray, execution_plan: dict, *, conf: float, device: str, imgsz: int):
    annotated = None
    detections: list[dict] = []
    visible_detection_count = 0
    ppe_prompts = execution_plan.get("ppe_prompt_terms") or []
    long_tail_prompts = execution_plan.get("yoloe_prompt_terms") or []
    batch_requests = []
    if execution_plan.get("run_coco_primary"):
        batch_requests.append({
            "request_id": "coco_primary",
            "model_key": "coco_primary",
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
        })
    if execution_plan.get("run_ppe_specialist") and ppe_prompts:
        batch_requests.append({
            "request_id": "ppe_specialist",
            "model_key": "ppe_specialist",
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
            "classes": ppe_prompts,
        })
    if execution_plan.get("run_yoloe_long_tail") and long_tail_prompts:
        batch_requests.append({
            "request_id": "yoloe_long_tail",
            "model_key": "yoloe_long_tail",
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
            "classes": long_tail_prompts,
        })
    record_batches = model_manager.predict_record_batches(frame, batch_requests)

    if execution_plan.get("run_coco_primary"):
        records = record_batches["coco_primary"]
        coco_detections = _detection_batch_from_records(
            records,
            "coco_primary",
        )
        detections.extend(coco_detections)
        visible_detection_count += len(coco_detections)

    if execution_plan.get("run_ppe_specialist") and ppe_prompts:
        records = record_batches["ppe_specialist"]
        ppe_detections = _detection_batch_from_records(
            records,
            "ppe_specialist",
            class_names=ppe_prompts,
        )
        detections.extend(ppe_detections)

    if execution_plan.get("run_yoloe_long_tail") and long_tail_prompts:
        records = record_batches["yoloe_long_tail"]
        long_tail_detections = _detection_batch_from_records(
            records,
            "yoloe_long_tail",
            class_names=long_tail_prompts,
        )
        detections.extend(long_tail_detections)

    if execution_plan.get("run_pose_specialist"):
        annotated = _draw_stream_detection_records(
            frame,
            detections,
            camera_id,
            show_overlay=False,
        )
        pose_results = model_manager.predict(
            "pose_specialist",
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
        annotated, fall_dets = draw_pose_detections(annotated, pose_results, fall_only=False)
        detections.extend(_normalize_detection_batch(fall_dets, "pose_specialist"))
        # Store pose results on frame state for fall checking in violation loop
        _last_pose_results[camera_id] = pose_results

    if annotated is not None:
        annotated = apply_camera_overlay(
            annotated,
            camera_id=camera_id,
            detection_count=visible_detection_count,
        )
    return annotated, detections


def _run_face_recognition(
    camera_id: str,
    frame: np.ndarray,
    annotated: np.ndarray,
    camera: dict,
    last_face_log_by_key: dict[str, float],
) -> tuple[np.ndarray, list[dict]]:
    try:
        events = face_analyzer.analyze_frame(frame)
    except Exception:
        logger.exception("Face recognition failed", extra={"camera_id": camera_id})
        return annotated, []

    _, snapshot_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snapshot_jpeg = snapshot_buffer.tobytes()
    now = time.time()
    detections: list[dict] = []

    for event in events:
        bbox = event.get("bbox") or {}
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        event_type = event["eventType"]
        label = event.get("personName") or (
            "Unknown person" if event_type == "face_unknown" else "Low-quality face"
        )
        color = (16, 185, 129) if event_type == "face_match" else (0, 193, 255) if event_type == "face_low_quality" else (0, 82, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        detection = {
            "class": event_type,
            "confidence": event.get("confidence"),
            "bbox": [x1, y1, x2, y2],
            "model_family": "face_recognition",
            "capability_keys": ["face_recognition"],
            "person_id": event.get("matchedFaceId"),
            "person_name": event.get("personName"),
            "person_group": event.get("personGroup"),
            "quality_reason": event.get("qualityReason"),
        }
        detections.append(detection)

        dedupe_subject = event.get("matchedFaceId") or event_type
        dedupe_key = f"{camera_id}:{event_type}:{dedupe_subject}"
        if now - last_face_log_by_key.get(dedupe_key, 0) < FACE_LOG_COOLDOWN_SECONDS:
            continue
        last_face_log_by_key[dedupe_key] = now
        face_store.log_face_event(
            camera_id=camera_id,
            camera_name=camera.get("name", camera_id),
            event_type=event_type,
            matched_face_id=event.get("matchedFaceId"),
            confidence=event.get("confidence"),
            bbox=bbox,
            quality_reason=event.get("qualityReason"),
            snapshot_jpeg=snapshot_jpeg,
        )

    return annotated, detections


def video_processor(camera_id: str, stop_event: threading.Event):
    """Run one camera worker and never publish observations after it exits."""
    try:
        _video_processor_loop(camera_id, stop_event)
    except Exception:
        logger.exception("Camera worker exited unexpectedly", extra={"camera_id": camera_id})
        raise
    finally:
        _clear_camera_observation(camera_id)


def _video_processor_loop(camera_id: str, stop_event: threading.Event):
    cfg = get_config()
    cam = cfg["cameras"][camera_id]
    stream_type = cam.get("stream_type", "file")
    video_source = build_rtsp_url(cam, include_credentials=True) if stream_type == "rtsp" else str(VIDEO_DIR / cam["video"])
    execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
    g = cfg["global"]
    target_fps = cam.get("fps", g["target_fps"])
    frame_interval = 1.0 / target_fps
    inference_fps = _positive_fps(
        cam.get("inference_fps", g.get("inference_fps", max(1.0, target_fps / 3))),
        max(1.0, target_fps / 3),
    )
    inference_interval = 1.0 / inference_fps
    next_inference_at = inference_scheduler.next_inference_slot(
        camera_id,
        cfg,
        inference_interval,
    )
    alert_cooldown = g["alert_cooldown"]
    yolo_conf = g["yolo_conf"]
    jpeg_quality = g["jpeg_quality"]
    inference_width = g["inference_width"]
    device = g["device"]
    last_alert_by_rule: dict[str, float] = {}
    last_face_log_by_key: dict[str, float] = {}
    active_violations: set[str] = set()
    violation_window: dict[str, list[bool]] = {}
    window_size = 15
    last_annotated = None
    reconnect_failures = 0
    ever_received_frame = False
    safe_video_source = redact_video_source(video_source)

    missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
    if missing_model_keys:
        state.camera_runtime_status[camera_id] = "awaiting_model_install"
        _clear_camera_observation(camera_id)
        logger.warning("Camera waiting on missing models", extra={"camera_id": camera_id, "missing_models": missing_model_keys})
        return

    while not stop_event.is_set():
        state.camera_runtime_status[camera_id] = (
            "starting" if not ever_received_frame and reconnect_failures == 0 else "reconnecting"
        )
        cap = open_video_capture(video_source, stream_type=stream_type)
        if not cap.isOpened():
            cap.release()
            _clear_camera_observation(camera_id)
            active_violations.clear()
            violation_window.clear()
            last_annotated = None
            delay = reconnect_delay_seconds(reconnect_failures, camera_id) if stream_type == "rtsp" else 5.0
            reconnect_failures += 1
            state.camera_runtime_status[camera_id] = "reconnecting"
            logger.warning(
                "Video source unavailable; retry scheduled",
                extra={
                    "camera_id": camera_id,
                    "source": safe_video_source,
                    "retry_seconds": round(delay, 2),
                    "failure_count": reconnect_failures,
                },
            )
            if stop_event.wait(delay):
                return
            continue

        last_stream_published_at = 0.0
        received_frame = False
        while cap.isOpened() and not stop_event.is_set():
            if not licensing.is_inference_allowed():
                time.sleep(LICENSE_PAUSE_INTERVAL)
                continue

            started = time.time()
            ok, frame = cap.read()
            if not ok:
                _clear_camera_observation(camera_id)
                active_violations.clear()
                violation_window.clear()
                last_annotated = None
                break
            if not received_frame:
                received_frame = True
                ever_received_frame = True
                reconnect_failures = 0

            current_cfg = get_config()
            current_g = current_cfg.get("global", g)
            current_cam = current_cfg["cameras"].get(camera_id, {})
            execution_plan = current_cam.get("execution_plan") or build_execution_plan(current_cam, current_cfg)
            state.camera_runtime_status[camera_id] = "running"
            current_inference_fps = _positive_fps(
                current_cam.get(
                    "inference_fps",
                    current_g.get("inference_fps", max(1.0, target_fps / 3)),
                ),
                inference_fps,
            )
            current_inference_interval = 1.0 / current_inference_fps
            if current_inference_interval != inference_interval:
                inference_fps = current_inference_fps
                inference_interval = current_inference_interval
                next_inference_at = inference_scheduler.next_inference_slot(
                    camera_id,
                    current_cfg,
                    inference_interval,
                )

            if time.monotonic() >= next_inference_at:
                try:
                    annotated, detections = _run_grouped_inference(
                        camera_id,
                        frame,
                        execution_plan,
                        conf=yolo_conf,
                        device=device,
                        imgsz=inference_width,
                    )
                    if execution_plan.get("run_face_recognition"):
                        if annotated is None:
                            annotated = _draw_stream_detection_records(
                                frame,
                                detections,
                                camera_id,
                            )
                        annotated, face_detections = _run_face_recognition(
                            camera_id,
                            frame,
                            annotated,
                            current_cam,
                            last_face_log_by_key,
                        )
                        detections.extend(face_detections)
                    if annotated is None and frame.shape[1] <= STREAM_MAX_WIDTH:
                        annotated = _draw_stream_detection_records(
                            frame,
                            detections,
                            camera_id,
                        )
                except Exception:
                    logger.exception("Detection failed", extra={"camera_id": camera_id})
                    annotated = frame
                    detections = []
                last_annotated = annotated
                state.camera_detections[camera_id] = detections
                next_inference_at = inference_scheduler.next_inference_slot(
                    camera_id,
                    current_cfg,
                    inference_interval,
                    now=time.monotonic() + 1e-9,
                )
            detections = state.camera_detections.get(camera_id, [])

            # Fall detection runs independently — pose model doesn't need COCO detections
            has_fall_candidates = False
            fall_candidates = []
            if execution_plan.get("run_pose_specialist") and camera_id in _last_pose_results:
                fall_candidates = check_fall_detections(_last_pose_results[camera_id], camera_id, frame)
                has_fall_candidates = len(fall_candidates) > 0

            if detections or has_fall_candidates:
                candidates = []
                if detections:
                    if execution_plan.get("run_ppe_specialist"):
                        candidates.extend(check_yoloe_violations(detections, camera_id))
                    candidates.extend(check_violations(detections, camera_id))

                    frame_h, frame_w = frame.shape[:2]
                    candidates.extend(check_zone_intrusions(detections, camera_id, frame_w, frame_h))

                candidates.extend(fall_candidates)
                current_violation_rules = {candidate["rule"]: candidate for candidate in candidates}

                all_tracked = set(violation_window) | set(current_violation_rules)
                for rule_key in all_tracked:
                    violation_window.setdefault(rule_key, []).append(rule_key in current_violation_rules)
                    if len(violation_window[rule_key]) > window_size:
                        violation_window[rule_key] = violation_window[rule_key][-window_size:]

                for rule_key in list(active_violations):
                    if sum(violation_window.get(rule_key, [])[-window_size:]) < 2:
                        active_violations.discard(rule_key)

                now = time.time()
                for rule_key, candidate in current_violation_rules.items():
                    if rule_key in active_violations:
                        continue
                    if now - last_alert_by_rule.get(rule_key, 0) < alert_cooldown:
                        continue
                    hits = sum(violation_window.get(rule_key, []))
                    rule_threshold = candidate.get("threshold") or VIOLATION_THRESHOLD
                    if hits < rule_threshold:
                        continue
                    active_violations.add(rule_key)
                    last_alert_by_rule[rule_key] = now
                    violation_window[rule_key] = []
                    violation_bboxes = extract_violation_bboxes(candidate["rule"], detections, frame_w, frame_h, camera_id)
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
                            logger.exception("Alert broadcast failed")

                for rule_key in list(violation_window):
                    if len(violation_window[rule_key]) >= window_size and sum(violation_window[rule_key]) == 0:
                        violation_window.pop(rule_key, None)
            else:
                for rule_key in list(violation_window):
                    violation_window[rule_key].append(False)
                    if len(violation_window[rule_key]) > window_size:
                        violation_window[rule_key] = violation_window[rule_key][-window_size:]
                    if sum(violation_window[rule_key]) == 0:
                        violation_window.pop(rule_key, None)
                        active_violations.discard(rule_key)

            stream_fps = _configured_stream_fps(current_cam, current_g, target_fps)
            stream_now = time.monotonic()
            if _stream_publish_due(last_stream_published_at, stream_now, stream_fps):
                try:
                    _publish_stream_frame(
                        camera_id,
                        frame,
                        detections,
                        jpeg_quality=jpeg_quality,
                        source_annotated=_preserved_source_annotation(
                            frame,
                            last_annotated,
                            execution_plan,
                        ),
                    )
                except Exception:
                    logger.exception("Stream frame publication failed", extra={"camera_id": camera_id})
                last_stream_published_at = stream_now

            elapsed = time.time() - started
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        _clear_camera_observation(camera_id)
        active_violations.clear()
        violation_window.clear()
        last_annotated = None
        cap.release()
        if stop_event.is_set():
            return
        if stream_type == "rtsp":
            delay = reconnect_delay_seconds(reconnect_failures, camera_id)
            reconnect_failures += 1
            state.camera_runtime_status[camera_id] = "reconnecting"
            logger.warning(
                "RTSP stream interrupted; retry scheduled",
                extra={
                    "camera_id": camera_id,
                    "source": safe_video_source,
                    "retry_seconds": round(delay, 2),
                    "failure_count": reconnect_failures,
                    "received_frame": received_frame,
                },
            )
            if stop_event.wait(delay):
                return
        else:
            logger.debug("Video loop restarting", extra={"camera_id": camera_id})


def start_camera(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam or not cam.get("enabled", True):
        return

    execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
    missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
    _clear_camera_observation(cam_id)
    if missing_model_keys:
        state.camera_runtime_status[cam_id] = "awaiting_model_install"
        logger.warning("Skipping camera start until models are ready", extra={"camera_id": cam_id, "missing_models": missing_model_keys})
        return

    stop_evt = threading.Event()
    thread = threading.Thread(target=video_processor, args=(cam_id, stop_evt), daemon=True)
    thread.start()
    state.camera_threads[cam_id] = (thread, stop_evt)
    state.camera_runtime_status[cam_id] = "starting"
    logger.info("Started video processor", extra={"camera_id": cam_id})

    if cam.get("demo") == "yolo+vlm":
        start_vlm_for_camera(cam_id)


def start_vlm_for_camera(cam_id: str):
    stop_evt = threading.Event()
    thread = threading.Thread(target=vlm_worker, args=(cam_id, stop_evt), daemon=True)
    thread.start()
    state.vlm_threads[cam_id] = (thread, stop_evt)
    logger.info("Started VLM worker", extra={"camera_id": cam_id})


def stop_camera(cam_id: str) -> bool:
    camera_stopped = True
    if cam_id in state.camera_threads:
        thread, stop_evt = state.camera_threads[cam_id]
        stop_evt.set()
        state.camera_runtime_status[cam_id] = "stopping"
        thread.join(timeout=CAMERA_STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            camera_stopped = False
            logger.warning(
                "Camera worker still stopping; retaining worker reference",
                extra={"camera_id": cam_id, "timeout_seconds": CAMERA_STOP_TIMEOUT_SECONDS},
            )
        else:
            del state.camera_threads[cam_id]
    if cam_id in state.vlm_threads:
        thread, stop_evt = state.vlm_threads[cam_id]
        stop_evt.set()
        thread.join(timeout=5)
        if not thread.is_alive():
            del state.vlm_threads[cam_id]
    if not camera_stopped:
        return False
    state.camera_frames.pop(cam_id, None)
    state.camera_clean_frames.pop(cam_id, None)
    state.camera_detections.pop(cam_id, None)
    _last_pose_results.pop(cam_id, None)
    state.camera_runtime_status[cam_id] = "offline"
    return True


def restart_camera(cam_id: str) -> bool:
    if not stop_camera(cam_id):
        logger.warning("Camera restart deferred until existing worker exits", extra={"camera_id": cam_id})
        return False
    start_camera(cam_id)
    return True


def restart_all_cameras():
    cfg = get_config()
    still_stopping = set()
    for cam_id in list(state.camera_threads):
        if not stop_camera(cam_id):
            still_stopping.add(cam_id)
    for cam_id in cfg["cameras"]:
        if cam_id not in still_stopping:
            start_camera(cam_id)


def mjpeg_generator(camera_id: str):
    while True:
        frame_bytes = state.camera_frames.get(camera_id)
        if frame_bytes is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        cfg = get_config()
        global_config = cfg["global"]
        camera = cfg["cameras"].get(camera_id, {})
        target_fps = camera.get("fps", global_config["target_fps"])
        fps = _configured_stream_fps(camera, global_config, target_fps)
        time.sleep(1.0 / fps)
