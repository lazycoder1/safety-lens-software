"""Rakshak Lens video processing — alerts, grouped inference, MJPEG streaming."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

import alert_store
import face_analyzer
import face_store
import plate_store
import licensing
import model_manager
import object_lifecycle_analytics
import state
import notification_dispatcher
import policy_engine
from camera_connection import build_rtsp_url
from camera_planner import build_execution_plan, required_model_keys_for_capabilities
from capability_registry import CAPABILITY_REGISTRY, CLASS_TERM_TO_CAPABILITY, RULE_ID_TO_CAPABILITY
from config_manager import get_config
from constants import OLLAMA_URL, VIDEO_DIR, VIOLATION_THRESHOLD, YOLOE_COLORS
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
CAMERA_START_RETRY_INTERVAL_SECONDS = 10.0
CAMERA_FRAME_WATCHDOG_INTERVAL_SECONDS = 5.0
CAMERA_STALE_RESTART_SECONDS = 60.0
RTSP_BUFFER_DRAIN_MAX_FRAMES = 30
RTSP_BUFFER_DRAIN_MAX_SECONDS = 0.04
RTSP_BUFFER_DRAIN_BLOCK_SECONDS = 0.012

logger = logging.getLogger("rakshak_lens")

_camera_watchdog_restart_at: dict[str, float] = {}


def _is_executor_shutdown_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "cannot schedule new futures after shutdown" in message
        or "cannot schedule new futures after interpreter shutdown" in message
    )

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
    "backpack": ["object_lifecycle"],
    "handbag": ["object_lifecycle"],
    "suitcase": ["object_lifecycle"],
    "umbrella": ["object_lifecycle"],
}

FACE_LOG_COOLDOWN_SECONDS = 10.0
PLATE_LOG_COOLDOWN_SECONDS = 10.0
PLATE_DETECTOR_CONFIDENCE = 0.20
PLATE_VOTE_WINDOW_SECONDS = 4.0
PLATE_FUZZY_CONFIRMATION_HITS = 2
FIRE_SMOKE_CLASS_NAMES = {0: "smoke", 1: "fire"}
FIRE_SMOKE_COLORS = {0: (160, 160, 160), 1: (0, 80, 255)}
CLOSED_SET_PPE_CLASS_NAMES = {
    0: "person",
    1: "apron",
    2: "safety harness",
    3: "safety lanyard",
}
CLOSED_SET_PPE_COLORS = {
    0: (80, 200, 255),
    1: (0, 190, 120),
    2: (255, 160, 30),
    3: (255, 90, 120),
}
DETECTION_HISTORY_LIMIT = 120


def _normalize_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def _valid_confidence(value) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < confidence <= 1:
        return confidence
    return None


def _camera_rule_override_value(camera: dict, rule_id: str, *field_names: str):
    overrides = camera.get("safety_rule_overrides") or {}
    if not isinstance(overrides, dict):
        return None
    override = overrides.get(rule_id)
    if not isinstance(override, dict):
        return None
    for field_name in field_names:
        if field_name in override:
            return override[field_name]
    return None


def _record_detection_history(
    camera_id: str,
    detections: list[dict],
    *,
    schedule_state: dict | None = None,
    model_invocations: dict | None = None,
) -> None:
    class_counts: dict[str, int] = {}
    for detection in detections:
        class_name = detection.get("class")
        if class_name:
            class_counts[str(class_name)] = class_counts.get(str(class_name), 0) + 1
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detectionsCount": len(detections),
        "detectionClassCounts": class_counts,
    }
    if schedule_state is not None:
        sample["scheduleState"] = schedule_state
    if model_invocations is not None:
        sample["modelInvocationCounts"] = model_invocations
    history = state.camera_detection_history.setdefault(camera_id, [])
    history.append(sample)
    if len(history) > DETECTION_HISTORY_LIMIT:
        del history[:-DETECTION_HISTORY_LIMIT]
    if schedule_state is not None or model_invocations is not None:
        state.camera_schedule_telemetry[camera_id] = {
            "timestamp": sample["timestamp"],
            "scheduleState": schedule_state or {},
            "modelInvocationCounts": model_invocations or {},
        }


def _advance_violation_window(
    violation_window: dict[str, list[bool]],
    current_violation_rules: dict[str, dict],
    *,
    window_size: int,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
) -> None:
    """Advance confirmation windows only when the source model produced a fresh observation."""
    all_tracked = set(violation_window) | set(current_violation_rules)
    for rule_key in all_tracked:
        if rule_key == "Fall Detected":
            if not fresh_fall_evaluated:
                continue
        elif not fresh_detection_evaluated:
            continue
        violation_window.setdefault(rule_key, []).append(rule_key in current_violation_rules)
        if len(violation_window[rule_key]) > window_size:
            violation_window[rule_key] = violation_window[rule_key][-window_size:]


def _record_empty_violation_observation(
    violation_window: dict[str, list[bool]],
    active_violations: set[str],
    *,
    window_size: int,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
) -> None:
    for rule_key in list(violation_window):
        if rule_key == "Fall Detected":
            if not fresh_fall_evaluated:
                continue
        elif not fresh_detection_evaluated:
            continue
        violation_window[rule_key].append(False)
        if len(violation_window[rule_key]) > window_size:
            violation_window[rule_key] = violation_window[rule_key][-window_size:]
        if sum(violation_window[rule_key]) == 0:
            violation_window.pop(rule_key, None)
            active_violations.discard(rule_key)


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


def _rewrite_detection_record_classes(records: list[dict], class_names: dict[int, str]) -> list[dict]:
    rewritten: list[dict] = []
    for record in records:
        item = dict(record)
        try:
            class_id = int(item.get("class_id", item.get("cls", 0)))
        except (TypeError, ValueError):
            class_id = 0
        if class_id in class_names:
            item["class"] = class_names[class_id]
        rewritten.append(item)
    return rewritten


def _rule_confidence_for_capability(camera_id: str, capability_key: str, default_conf: float) -> float:
    cfg = get_config()
    camera = cfg.get("cameras", {}).get(camera_id, {})
    rule_ids = camera.get("safety_rule_ids", [])
    rule_map = {rule.get("id"): rule for rule in cfg.get("safety_rules", [])}
    confidences: list[float] = []
    for rule_id in rule_ids:
        if RULE_ID_TO_CAPABILITY.get(rule_id) != capability_key:
            continue
        rule = rule_map.get(rule_id)
        if not rule or not rule.get("enabled", True):
            continue
        override = _valid_confidence(_camera_rule_override_value(camera, rule_id, "confidence", "conf"))
        if override is not None:
            confidences.append(override)
            continue
        confidence = _valid_confidence(rule.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
    return min(confidences) if confidences else default_conf


def _rule_confidence_for_model_family(camera_id: str, model_family: str, default_conf: float) -> float:
    cfg = get_config()
    camera = cfg.get("cameras", {}).get(camera_id, {})
    rule_ids = camera.get("safety_rule_ids", [])
    rule_map = {rule.get("id"): rule for rule in cfg.get("safety_rules", [])}
    confidences: list[float] = []
    for rule_id in rule_ids:
        capability_key = RULE_ID_TO_CAPABILITY.get(rule_id)
        if not capability_key:
            continue
        definition = CAPABILITY_REGISTRY.get(capability_key)
        if not definition or definition.get("model_family") != model_family:
            continue
        rule = rule_map.get(rule_id)
        if not rule or not rule.get("enabled", True):
            continue
        override = _valid_confidence(_camera_rule_override_value(camera, rule_id, "confidence", "conf"))
        if override is not None:
            confidences.append(override)
            continue
        confidence = _valid_confidence(rule.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
    return min(confidences) if confidences else default_conf


def _capabilities_for_model_key(execution_plan: dict, model_key: str) -> list[str]:
    overrides = execution_plan.get("capability_model_overrides") or {}
    if not isinstance(overrides, dict):
        return []
    return [
        capability
        for capability in execution_plan.get("capabilities") or []
        if overrides.get(capability) == model_key
    ]


def _rule_confidence_for_capabilities(camera_id: str, capabilities: list[str], default_conf: float) -> float:
    confidences = [
        _rule_confidence_for_capability(camera_id, capability, default_conf)
        for capability in capabilities
    ]
    return min(confidences) if confidences else default_conf


def _model_keys_for_capabilities(capabilities: list[str], capability_model_overrides: dict | None = None) -> list[str]:
    capability_set = {capability for capability in capabilities if capability in CAPABILITY_REGISTRY}
    overrides = capability_model_overrides if isinstance(capability_model_overrides, dict) else None
    return required_model_keys_for_capabilities(list(capability_set), overrides)  # type: ignore[arg-type]


def _capability_schedule_state(
    camera: dict,
    cfg: dict,
    execution_plan: dict,
    *,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    planned_capabilities = set(execution_plan.get("capabilities") or [])
    states: dict[str, dict] = {
        capability: {
            "active": True,
            "mode": "detection",
            "scheduleId": None,
            "suppressed": False,
        }
        for capability in planned_capabilities
    }
    for window in execution_plan.get("capability_windows") or []:
        mode = str(window.get("mode") or "detection")
        if mode not in {"detection", "detector", "detector_off"}:
            continue
        schedule_id = str(window.get("id") or "capability_window")
        configured_active = window.get("active", True)
        matches = False if configured_active is False else policy_engine._schedule_matches(window, now, cfg)
        for capability in window.get("capabilities") or []:
            if capability not in planned_capabilities:
                continue
            states[capability] = {
                "active": bool(matches),
                "mode": "detection",
                "scheduleId": schedule_id,
                "suppressed": not bool(matches),
                "timestamp": now.isoformat(),
            }

    suppressed = sorted(capability for capability, item in states.items() if item.get("suppressed"))
    return {
        "timestamp": now.isoformat(),
        "capabilities": states,
        "suppressedCapabilities": suppressed,
        "suppressedCount": len(suppressed),
    }


def _filter_prompt_terms_for_schedule(terms: list[str], suppressed_capabilities: set[str]) -> list[str]:
    filtered: list[str] = []
    for term in terms:
        mapped = CLASS_TERM_TO_CAPABILITY.get(_normalize_text(term))
        if mapped and mapped in suppressed_capabilities:
            continue
        filtered.append(term)
    return filtered


def _scheduled_execution_plan(execution_plan: dict, schedule_state: dict) -> dict:
    suppressed_capabilities = set(schedule_state.get("suppressedCapabilities") or [])
    if not suppressed_capabilities:
        scheduled = deepcopy(execution_plan)
        scheduled["schedule_state"] = schedule_state
        return scheduled

    scheduled = deepcopy(execution_plan)
    active_capabilities = [
        capability
        for capability in scheduled.get("capabilities", [])
        if capability not in suppressed_capabilities
    ]
    required_model_keys = _model_keys_for_capabilities(
        active_capabilities,
        scheduled.get("capability_model_overrides"),
    )
    scheduled["active_capabilities"] = active_capabilities
    scheduled["suppressed_capabilities"] = sorted(suppressed_capabilities)
    scheduled["capabilities"] = active_capabilities
    scheduled["required_model_keys"] = required_model_keys
    scheduled["run_coco_primary"] = "coco_primary" in required_model_keys
    scheduled["run_ppe_specialist"] = "ppe_specialist" in required_model_keys
    scheduled["run_ppe_closed_set_candidate"] = "ppe_closed_set_candidate" in required_model_keys
    scheduled["run_yoloe_long_tail"] = "yoloe_long_tail" in required_model_keys
    scheduled["run_fire_smoke_specialist"] = "fire_smoke_specialist" in required_model_keys
    scheduled["run_face_recognition"] = "face_recognition" in required_model_keys
    scheduled["run_pose_specialist"] = "pose_specialist" in required_model_keys
    scheduled["run_plate_recognition"] = "plate_recognition" in required_model_keys
    scheduled["ppe_prompt_terms"] = _filter_prompt_terms_for_schedule(
        scheduled.get("ppe_prompt_terms", []),
        suppressed_capabilities,
    ) if scheduled["run_ppe_specialist"] else []
    scheduled["yoloe_prompt_terms"] = _filter_prompt_terms_for_schedule(
        scheduled.get("yoloe_prompt_terms", []),
        suppressed_capabilities,
    ) if scheduled["run_yoloe_long_tail"] else []
    scheduled["schedule_state"] = schedule_state
    return scheduled


def _crowd_count_threshold_candidates(camera_id: str, detections: list[dict]) -> list[dict]:
    """Emit a policy candidate for count-threshold rules without enabling person_presence."""
    persons = [d for d in detections if d.get("class") == "person"]
    if not persons:
        return []
    return [
        {
            "camera_id": camera_id,
            "rule": "Person Detected",
            "severity": "P3",
            "confidence": max(float(d.get("confidence") or 0) for d in persons),
            "count": len(persons),
            "classes": ["person"],
            "description": f"{len(persons)} person detected detection(s)",
            "source": "COCO Primary",
            "threshold": 1,
            "metadata": {"capability": "crowd_count_threshold"},
        }
    ]


def create_alert(
    camera_id: str,
    rule: str,
    severity: str,
    confidence: float,
    description: str = "",
    source: str = "YOLO",
    bboxes: list[dict] | None = None,
    output_ids: list[str] | None = None,
    policy_id: str | None = None,
    priority: int | None = None,
    message: str | None = None,
    metadata: dict | None = None,
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
        policy_id=policy_id,
        priority=priority,
        message=message,
        metadata=metadata,
    )
    try:
        snap_url = alert.get("snapshotUrl")
        snap_full = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1]) if snap_url else None
        notification_dispatcher.notify(alert, snap_full, output_ids=output_ids)
    except Exception:
        logger.exception("Notification dispatch failed")
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
    annotated = frame.copy()
    detections: list[dict] = []
    visible_detection_count = 0
    pose_results = None
    model_invocations = {
        "coco_primary": 0,
        "ppe_specialist": 0,
        "ppe_closed_set_candidate": 0,
        "yoloe_long_tail": 0,
        "fire_smoke_specialist": 0,
        "pose_specialist": 0,
    }

    if execution_plan.get("run_coco_primary"):
        model_invocations["coco_primary"] += 1
        records = model_manager.predict_records(
            "coco_primary",
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
        annotated, coco_detections = draw_detection_records(
            annotated,
            records,
            camera_id,
            show_overlay=False,
        )
        detections.extend(_normalize_detection_batch(coco_detections, "coco_primary"))
        visible_detection_count += len(coco_detections)

    if execution_plan.get("run_ppe_specialist") and execution_plan.get("ppe_prompt_terms"):
        ppe_prompts = execution_plan["ppe_prompt_terms"]
        ppe_conf = _rule_confidence_for_model_family(camera_id, "ppe_specialist", conf)
        model_invocations["ppe_specialist"] += 1
        records = model_manager.predict_records(
            "ppe_specialist",
            frame,
            conf=ppe_conf,
            device=device,
            imgsz=imgsz,
            classes=ppe_prompts,
        )
        _ppe_annotated, ppe_detections = draw_detection_records(
            annotated,
            records,
            camera_id,
            class_names=ppe_prompts,
            colors=YOLOE_COLORS,
            show_overlay=False,
        )
        detections.extend(_normalize_detection_batch(ppe_detections, "ppe_specialist"))

    if execution_plan.get("run_ppe_closed_set_candidate"):
        candidate_capabilities = _capabilities_for_model_key(execution_plan, "ppe_closed_set_candidate")
        candidate_conf = _rule_confidence_for_capabilities(camera_id, candidate_capabilities, conf)
        model_invocations["ppe_closed_set_candidate"] += 1
        records = model_manager.predict_records(
            "ppe_closed_set_candidate",
            frame,
            conf=candidate_conf,
            device=device,
            imgsz=imgsz,
        )
        records = _rewrite_detection_record_classes(records, CLOSED_SET_PPE_CLASS_NAMES)
        _candidate_annotated, candidate_detections = draw_detection_records(
            annotated,
            records,
            camera_id,
            class_names=CLOSED_SET_PPE_CLASS_NAMES,
            colors=CLOSED_SET_PPE_COLORS,
            show_overlay=False,
        )
        detections.extend(_normalize_detection_batch(candidate_detections, "ppe_closed_set_candidate"))

    if execution_plan.get("run_yoloe_long_tail") and execution_plan.get("yoloe_prompt_terms"):
        long_tail_prompts = execution_plan["yoloe_prompt_terms"]
        model_invocations["yoloe_long_tail"] += 1
        records = model_manager.predict_records(
            "yoloe_long_tail",
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
            classes=long_tail_prompts,
        )
        _long_tail_annotated, long_tail_detections = draw_detection_records(
            annotated,
            records,
            camera_id,
            class_names=long_tail_prompts,
            colors=YOLOE_COLORS,
            show_overlay=False,
        )
        detections.extend(_normalize_detection_batch(long_tail_detections, "yoloe_long_tail"))

    if execution_plan.get("run_fire_smoke_specialist"):
        fire_conf = _rule_confidence_for_capability(camera_id, "fire_smoke", conf)
        model_invocations["fire_smoke_specialist"] += 1
        records = model_manager.predict_records(
            "fire_smoke_specialist",
            frame,
            conf=fire_conf,
            device=device,
            imgsz=imgsz,
        )
        _fire_annotated, fire_detections = draw_detection_records(
            annotated,
            records,
            camera_id,
            class_names=FIRE_SMOKE_CLASS_NAMES,
            colors=FIRE_SMOKE_COLORS,
            show_overlay=False,
        )
        detections.extend(_normalize_detection_batch(fire_detections, "fire_smoke_specialist"))

    if execution_plan.get("run_pose_specialist"):
        model_invocations["pose_specialist"] += 1
        pose_results = model_manager.predict(
            "pose_specialist",
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
        annotated, fall_dets = draw_pose_detections(annotated, pose_results, fall_only=False, camera_id=camera_id)
        detections.extend(_normalize_detection_batch(fall_dets, "pose_specialist"))

    annotated = apply_camera_overlay(
        annotated,
        camera_id=camera_id,
        detection_count=visible_detection_count,
    )
    return annotated, detections, pose_results, model_invocations


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


def _run_plate_recognition(
    camera_id: str,
    frame: np.ndarray,
    annotated: np.ndarray,
    camera: dict,
    last_plate_log_by_key: dict[str, float],
    plate_vote_window: list[dict],
    *,
    conf: float,
    device: str,
    imgsz: int,
) -> tuple[np.ndarray, list[dict]]:
    try:
        candidates = model_manager.predict_plate_records(
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
    except Exception:
        logger.exception("Plate recognition failed", extra={"camera_id": camera_id})
        return annotated, []

    _, snapshot_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snapshot_jpeg = snapshot_buffer.tobytes()
    now = time.time()
    detections: list[dict] = []
    frame_h, frame_w = frame.shape[:2]

    for candidate in candidates:
        bbox = candidate.get("bbox") or {}
        x1 = max(0, int(bbox.get("x1", 0)))
        y1 = max(0, int(bbox.get("y1", 0)))
        x2 = min(frame_w, int(bbox.get("x2", 0)))
        y2 = min(frame_h, int(bbox.get("y2", 0)))
        if x2 <= x1 or y2 <= y1:
            continue

        raw_normalized = plate_store.normalize_plate_text(candidate.get("normalizedPlate") or candidate.get("plateText"))
        normalized = raw_normalized
        quality_reason = candidate.get("qualityReason")
        matched = plate_store.find_matching_plate(normalized) if normalized and not quality_reason else None
        match_kind = "exact" if matched else None
        pending_similar = False
        if not matched and normalized and not quality_reason:
            matched, match_kind = _resolve_plate_vote_match(
                normalized,
                candidate,
                plate_vote_window,
                now,
            )
            pending_similar = match_kind == "pending_similar"
        event_type = _plate_event_type(normalized, matched, quality_reason)
        confidence = candidate.get("confidence")
        label = normalized or "Unread plate"
        if matched and match_kind == "similar":
            label = f"{normalized} ~ {matched['normalizedPlate']}"
        color = _plate_color(event_type)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        detection = {
            "class": event_type,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "model_family": "plate_recognition",
            "capability_keys": ["plate_recognition"],
            "plate_number": normalized,
            "matched_list": matched.get("list") if matched else None,
            "matched_plate": matched.get("normalizedPlate") if matched else None,
            "match_kind": match_kind,
            "quality_reason": quality_reason,
        }
        detections.append(detection)

        if pending_similar:
            continue

        dedupe_subject = (matched.get("id") if matched else None) or normalized or f"{x1}:{y1}:{x2}:{y2}"
        dedupe_key = f"{camera_id}:{event_type}:{dedupe_subject}"
        if now - last_plate_log_by_key.get(dedupe_key, 0) < PLATE_LOG_COOLDOWN_SECONDS:
            continue
        last_plate_log_by_key[dedupe_key] = now

        crop_jpeg = None
        crop = frame[y1:y2, x1:x2]
        ok, crop_buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            crop_jpeg = crop_buffer.tobytes()
        plate_store.log_plate_read(
            plate_text=normalized or candidate.get("plateText") or "",
            camera_id=camera_id,
            camera_name=camera.get("name", camera_id),
            event_type=event_type,
            matched_list_id=matched.get("id") if matched else None,
            matched_list=matched.get("list") if matched else None,
            confidence=confidence,
            detection_confidence=candidate.get("detectionConfidence"),
            ocr_confidence=candidate.get("ocrConfidence"),
            bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            vehicle_class=candidate.get("vehicleClass"),
            quality_reason=_plate_quality_reason(quality_reason, raw_normalized, matched, match_kind),
            snapshot_jpeg=snapshot_jpeg,
            crop_jpeg=crop_jpeg,
        )

    return annotated, detections


def _resolve_plate_vote_match(
    normalized: str,
    candidate: dict,
    plate_vote_window: list[dict],
    now: float,
) -> tuple[dict | None, str | None]:
    similar = plate_store.find_similar_plate(normalized)
    if not similar:
        _append_plate_vote(plate_vote_window, normalized, None, now, candidate)
        return None, None

    _append_plate_vote(plate_vote_window, normalized, similar["id"], now, candidate)
    hits = sum(1 for vote in plate_vote_window if vote.get("matched_id") == similar["id"])
    if hits >= PLATE_FUZZY_CONFIRMATION_HITS:
        return similar, "similar"
    return None, "pending_similar"


def _append_plate_vote(
    plate_vote_window: list[dict],
    normalized: str,
    matched_id: str | None,
    now: float,
    candidate: dict,
) -> None:
    plate_vote_window.append({
        "normalized": normalized,
        "matched_id": matched_id,
        "timestamp": now,
        "confidence": candidate.get("confidence") or 0,
    })
    cutoff = now - PLATE_VOTE_WINDOW_SECONDS
    del plate_vote_window[: max(0, len(plate_vote_window) - 20)]
    plate_vote_window[:] = [vote for vote in plate_vote_window if vote["timestamp"] >= cutoff]


def _plate_quality_reason(
    quality_reason: str | None,
    raw_normalized: str,
    matched: dict | None,
    match_kind: str | None,
) -> str | None:
    if quality_reason:
        return quality_reason
    if matched and match_kind == "similar":
        score = matched.get("similarityScore")
        suffix = f" ({round(float(score) * 100)}%)" if isinstance(score, (int, float)) else ""
        return f"Similar match to registered plate {matched['normalizedPlate']}; OCR read {raw_normalized}{suffix}"
    return None


def _plate_event_type(normalized: str, matched: dict | None, quality_reason: str | None) -> str:
    if quality_reason:
        return "plate_low_confidence"
    if not normalized:
        return "plate_low_confidence"
    if not matched:
        return "plate_unknown"
    if matched["list"] == "blocked":
        return "plate_blocked"
    if matched["list"] == "visitors":
        return "plate_visitor"
    return "plate_read"


def _plate_color(event_type: str) -> tuple[int, int, int]:
    if event_type == "plate_blocked":
        return (0, 0, 220)
    if event_type == "plate_visitor":
        return (220, 120, 0)
    if event_type == "plate_read":
        return (0, 160, 60)
    return (0, 193, 255)


def _coerce_fps(value, default: float) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, fps)


def _open_video_capture(video_source: str, stream_type: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_source)
    if stream_type == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _read_live_frame(cap: cv2.VideoCapture, stream_type: str) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if not ok or frame is None or stream_type != "rtsp":
        return ok, frame

    grabbed_any = False
    deadline = time.monotonic() + RTSP_BUFFER_DRAIN_MAX_SECONDS
    drained = 0
    while drained < RTSP_BUFFER_DRAIN_MAX_FRAMES and time.monotonic() < deadline:
        started = time.monotonic()
        grabbed = cap.grab()
        elapsed = time.monotonic() - started
        if not grabbed:
            break
        grabbed_any = True
        drained += 1
        if elapsed >= RTSP_BUFFER_DRAIN_BLOCK_SECONDS:
            break

    if not grabbed_any:
        return ok, frame

    latest_ok, latest_frame = cap.retrieve()
    if latest_ok and latest_frame is not None:
        return True, latest_frame
    return ok, frame


def _clear_live_frame(camera_id: str) -> None:
    state.camera_frames[camera_id] = None
    state.camera_clean_frames[camera_id] = None
    state.camera_frame_updated_at.pop(camera_id, None)


def _frame_age_seconds(camera_id: str):
    updated_at = state.camera_frame_updated_at.get(camera_id)
    if updated_at is None:
        return None
    return time.time() - updated_at


def _is_live_frame_fresh(camera_id: str) -> bool:
    frame_bytes = state.camera_frames.get(camera_id)
    age = _frame_age_seconds(camera_id)
    return frame_bytes is not None and age is not None and age <= state.CAMERA_FRAME_STALE_SECONDS


_STATUS_FRAME_CACHE: dict[tuple[str, str], bytes] = {}


def _status_frame(camera_id: str, status: str, *, jpeg_quality: int = 70) -> bytes:
    cache_key = (camera_id, status)
    cached = _STATUS_FRAME_CACHE.get(cache_key)
    if cached:
        return cached
    image = np.zeros((480, 854, 3), dtype=np.uint8)
    image[:] = (28, 31, 36)
    label = status.replace("_", " ").title()
    title = f"{camera_id} - {label}"
    subtitle = "Waiting for fresh camera frames"
    cv2.putText(image, title, (36, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (236, 239, 244), 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (36, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (156, 163, 175), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    encoded = buffer.tobytes()
    _STATUS_FRAME_CACHE[cache_key] = encoded
    return encoded


def _resize_for_stream(frame: np.ndarray, max_width: int = 854) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)))


def _publish_live_frame(camera_id: str, frame: np.ndarray, *, jpeg_quality: int) -> None:
    """Publish the newest camera frame without waiting for model inference."""
    detections = [
        detection
        for detection in state.camera_detections.get(camera_id, [])
        if isinstance(detection.get("bbox"), list) and len(detection["bbox"]) == 4
    ]
    try:
        annotated, _ = draw_detection_records(frame, detections, camera_id, show_overlay=False)
    except Exception:
        logger.debug("Could not draw cached detections on live frame", extra={"camera_id": camera_id}, exc_info=True)
        annotated = frame.copy()
    annotated = apply_camera_overlay(
        annotated,
        camera_id=camera_id,
        detection_count=len(detections),
    )
    clean_resized = _resize_for_stream(frame)
    annotated_resized = _resize_for_stream(annotated)
    _, buffer = cv2.imencode(".jpg", annotated_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    state.camera_frames[camera_id] = buffer.tobytes()
    _, clean_buffer = cv2.imencode(".jpg", clean_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    state.camera_clean_frames[camera_id] = clean_buffer.tobytes()
    state.camera_frame_updated_at[camera_id] = time.time()


def _run_detection_job(
    camera_id: str,
    frame: np.ndarray,
    scheduled_plan: dict,
    schedule_state: dict,
    current_cam: dict,
    current_cfg: dict,
    *,
    yolo_conf: float,
    device: str,
    inference_width: int,
    last_face_log_by_key: dict[str, float],
    last_plate_log_by_key: dict[str, float],
    plate_vote_window: list[dict],
) -> dict:
    fresh_pose_results = None
    model_invocations = {}
    annotated, detections, fresh_pose_results, model_invocations = _run_grouped_inference(
        camera_id,
        frame,
        scheduled_plan,
        conf=yolo_conf,
        device=device,
        imgsz=inference_width,
    )
    model_invocations.setdefault("face_recognition", 0)
    model_invocations.setdefault("plate_recognition", 0)
    if scheduled_plan.get("run_face_recognition"):
        model_invocations["face_recognition"] += 1
        annotated, face_detections = _run_face_recognition(
            camera_id,
            frame,
            annotated,
            current_cam,
            last_face_log_by_key,
        )
        detections.extend(face_detections)
    if scheduled_plan.get("run_plate_recognition"):
        model_invocations["plate_recognition"] += 1
        annotated, plate_detections = _run_plate_recognition(
            camera_id,
            frame,
            annotated,
            current_cam,
            last_plate_log_by_key,
            plate_vote_window,
            conf=min(yolo_conf, PLATE_DETECTOR_CONFIDENCE),
            device=device,
            imgsz=inference_width,
        )
        detections.extend(plate_detections)
    return {
        "frame": frame,
        "detections": detections,
        "fresh_pose_results": fresh_pose_results,
        "model_invocations": model_invocations,
        "scheduled_plan": scheduled_plan,
        "schedule_state": schedule_state,
        "current_cam": current_cam,
        "current_cfg": current_cfg,
    }


def _process_detection_observation(
    camera_id: str,
    frame: np.ndarray,
    detections: list[dict],
    fresh_pose_results,
    scheduled_plan: dict,
    current_cam: dict,
    current_cfg: dict,
    *,
    last_alert_by_rule: dict[str, float],
    active_violations: set[str],
    violation_window: dict[str, list[bool]],
    alert_cooldown: int,
    window_size: int,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    object_lifecycle_events = []
    if (
        "object_lifecycle" in scheduled_plan.get("capabilities", [])
        and object_lifecycle_analytics.is_object_lifecycle_enabled(current_cam)
    ):
        object_lifecycle_events = object_lifecycle_analytics.update_object_lifecycle(
            camera_id,
            current_cam,
            detections,
            frame_w,
            frame_h,
        )

    has_fall_candidates = False
    fall_candidates = []
    fresh_fall_evaluated = scheduled_plan.get("run_pose_specialist") and fresh_pose_results is not None
    if fresh_fall_evaluated:
        fall_candidates = check_fall_detections(fresh_pose_results, camera_id, frame)
        has_fall_candidates = len(fall_candidates) > 0

    if detections or has_fall_candidates or object_lifecycle_events:
        candidates = []
        if detections:
            if scheduled_plan.get("run_ppe_specialist") or scheduled_plan.get("run_ppe_closed_set_candidate"):
                candidates.extend(check_yoloe_violations(detections, camera_id, frame_w, frame_h))
            candidates.extend(check_violations(detections, camera_id))
            if "crowd_count_threshold" in scheduled_plan.get("capabilities", []):
                candidates.extend(_crowd_count_threshold_candidates(camera_id, detections))

            if "zone_intrusion" in scheduled_plan.get("capabilities", []):
                candidates.extend(check_zone_intrusions(detections, camera_id, frame_w, frame_h))

        candidates.extend(fall_candidates)
        candidates.extend(object_lifecycle_events)
        current_violation_rules = {candidate["rule"]: candidate for candidate in candidates}
        _advance_violation_window(
            violation_window,
            current_violation_rules,
            window_size=window_size,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
        )

        for rule_key in list(active_violations):
            if sum(violation_window.get(rule_key, [])[-window_size:]) < 2:
                active_violations.discard(rule_key)

        now = time.time()
        for rule_key, candidate in current_violation_rules.items():
            if rule_key in active_violations:
                continue
            hits = sum(violation_window.get(rule_key, []))
            rule_threshold = candidate.get("threshold") or VIOLATION_THRESHOLD
            if hits < rule_threshold:
                continue
            decisions = policy_engine.evaluate_candidate(
                candidate,
                current_cam,
                camera_id=candidate["camera_id"],
                detections=detections,
                cfg=current_cfg,
            )
            if not decisions:
                continue
            if any(decision.fallback for decision in decisions):
                if now - last_alert_by_rule.get(rule_key, 0) < alert_cooldown:
                    continue
                last_alert_by_rule[rule_key] = now
            active_violations.add(rule_key)
            violation_window[rule_key] = []
            violation_bboxes = extract_violation_bboxes(candidate["rule"], detections, frame_w, frame_h, camera_id)
            for decision in decisions:
                alert = create_alert(
                    camera_id=candidate["camera_id"],
                    rule=candidate["rule"],
                    severity=decision.severity,
                    confidence=candidate["confidence"],
                    description=candidate["description"],
                    source=candidate["source"],
                    bboxes=violation_bboxes,
                    output_ids=decision.output_ids,
                    policy_id=decision.rule_id or None,
                    priority=decision.priority,
                    message=decision.message,
                    metadata={
                        "policyRuleName": decision.rule_name,
                        "policyFallback": decision.fallback,
                        "cooldownSeconds": decision.cooldown_seconds,
                        "candidateCount": candidate.get("count"),
                        **(candidate.get("metadata") or {}),
                    },
                )
                if alert:
                    if decision.rule_id:
                        policy_engine.mark_rule_triggered(decision.rule_id, cfg=current_cfg)
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
        _record_empty_violation_observation(
            violation_window,
            active_violations,
            window_size=window_size,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
        )


def video_processor(camera_id: str, stop_event: threading.Event):
    cfg = get_config()
    cam = cfg["cameras"][camera_id]
    stream_type = cam.get("stream_type", "file")
    video_source = build_rtsp_url(cam, include_credentials=True) if stream_type == "rtsp" else str(VIDEO_DIR / cam["video"])
    execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
    g = cfg["global"]
    target_fps = cam.get("fps", g["target_fps"])
    frame_interval = 1.0 / target_fps
    alert_cooldown = g["alert_cooldown"]
    yolo_conf = g["yolo_conf"]
    jpeg_quality = g["jpeg_quality"]
    inference_width = g["inference_width"]
    device = g["device"]
    last_alert_by_rule: dict[str, float] = {}
    last_face_log_by_key: dict[str, float] = {}
    last_plate_log_by_key: dict[str, float] = {}
    plate_vote_window: list[dict] = []
    active_violations: set[str] = set()
    violation_window: dict[str, list[bool]] = {}
    window_size = 15
    frame_counter = 0
    last_annotated = None

    initial_schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
    initial_execution_plan = _scheduled_execution_plan(execution_plan, initial_schedule_state)
    missing_model_keys = model_manager.missing_model_keys(initial_execution_plan["required_model_keys"])
    if missing_model_keys:
        state.camera_runtime_status[camera_id] = "awaiting_model_install"
        state.camera_frames[camera_id] = None
        state.camera_detections[camera_id] = []
        state.camera_detection_history[camera_id] = []
        state.camera_schedule_telemetry[camera_id] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduleState": initial_schedule_state,
            "modelInvocationCounts": {},
        }
        logger.warning("Camera waiting on missing models", extra={"camera_id": camera_id, "missing_models": missing_model_keys})
        return

    while not stop_event.is_set():
        cap = _open_video_capture(video_source, stream_type)
        if not cap.isOpened():
            state.camera_runtime_status[camera_id] = "reconnecting" if stream_type == "rtsp" else "offline"
            _clear_live_frame(camera_id)
            logger.error("Cannot open video source", extra={"camera_id": camera_id, "source": video_source})
            for _ in range(50):
                if stop_event.is_set():
                    return
                time.sleep(0.1)
            continue

        inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{camera_id}-inference")
        pending_inference = None
        last_inference_started = 0.0
        try:
            while cap.isOpened() and not stop_event.is_set():
                if not licensing.is_inference_allowed():
                    time.sleep(LICENSE_PAUSE_INTERVAL)
                    continue

                started = time.time()
                ok, frame = _read_live_frame(cap, stream_type)
                if not ok:
                    state.camera_runtime_status[camera_id] = "reconnecting" if stream_type == "rtsp" else "offline"
                    _clear_live_frame(camera_id)
                    break

                frame_counter += 1
                current_cfg = get_config()
                current_g = current_cfg.get("global", {})
                current_cam = current_cfg["cameras"].get(camera_id, {})
                execution_plan = current_cam.get("execution_plan") or build_execution_plan(current_cam, current_cfg)
                schedule_state = _capability_schedule_state(current_cam, current_cfg, execution_plan)
                scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
                target_fps = _coerce_fps(current_cam.get("fps", current_g.get("target_fps", target_fps)), target_fps)
                frame_interval = 1.0 / target_fps
                inference_fps = _coerce_fps(
                    current_cam.get("inference_fps", current_g.get("inference_fps", max(1.0, target_fps / 3))),
                    max(1.0, target_fps / 3),
                )
                inference_interval = 1.0 / inference_fps
                yolo_conf = current_g.get("yolo_conf", yolo_conf)
                jpeg_quality = int(current_g.get("jpeg_quality", jpeg_quality))
                inference_width = int(current_g.get("inference_width", inference_width))
                device = current_g.get("device", device)
                alert_cooldown = int(current_g.get("alert_cooldown", alert_cooldown))
                state.camera_runtime_status[camera_id] = "running"

                _publish_live_frame(camera_id, frame, jpeg_quality=jpeg_quality)

                if pending_inference and pending_inference.done():
                    try:
                        result = pending_inference.result()
                    except Exception:
                        logger.exception("Detection failed", extra={"camera_id": camera_id})
                        result = {
                            "frame": frame,
                            "detections": [],
                            "fresh_pose_results": None,
                            "model_invocations": {},
                            "scheduled_plan": scheduled_plan,
                            "schedule_state": schedule_state,
                            "current_cam": current_cam,
                            "current_cfg": current_cfg,
                        }
                    pending_inference = None
                    detections = result["detections"]
                    state.camera_detections[camera_id] = detections
                    _record_detection_history(
                        camera_id,
                        detections,
                        schedule_state=result["schedule_state"],
                        model_invocations=result["model_invocations"],
                    )
                    _process_detection_observation(
                        camera_id,
                        result["frame"],
                        detections,
                        result["fresh_pose_results"],
                        result["scheduled_plan"],
                        result["current_cam"],
                        result["current_cfg"],
                        last_alert_by_rule=last_alert_by_rule,
                        active_violations=active_violations,
                        violation_window=violation_window,
                        alert_cooldown=alert_cooldown,
                        window_size=window_size,
                    )

                now = time.time()
                if pending_inference is None and now - last_inference_started >= inference_interval:
                    try:
                        pending_inference = inference_executor.submit(
                            _run_detection_job,
                            camera_id,
                            frame.copy(),
                            scheduled_plan,
                            schedule_state,
                            current_cam,
                            current_cfg,
                            yolo_conf=yolo_conf,
                            device=device,
                            inference_width=inference_width,
                            last_face_log_by_key=last_face_log_by_key,
                            last_plate_log_by_key=last_plate_log_by_key,
                            plate_vote_window=plate_vote_window,
                        )
                    except RuntimeError as exc:
                        if not _is_executor_shutdown_error(exc):
                            raise
                        state.camera_runtime_status[camera_id] = "offline"
                        _clear_live_frame(camera_id)
                        logger.debug("Stopping video processor during executor shutdown", extra={"camera_id": camera_id})
                        return
                    last_inference_started = now

                elapsed = time.time() - started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if pending_inference:
                pending_inference.cancel()
            inference_executor.shutdown(wait=False, cancel_futures=True)

        cap.release()
        if stop_event.is_set():
            return
        if stream_type == "rtsp":
            state.camera_runtime_status[camera_id] = "reconnecting"
            _clear_live_frame(camera_id)
            logger.warning("RTSP stream dropped, reconnecting in 5s", extra={"camera_id": camera_id})
            for _ in range(50):
                if stop_event.is_set():
                    return
                time.sleep(0.1)
        else:
            logger.debug("Video loop restarting", extra={"camera_id": camera_id})


def start_camera(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam or not cam.get("enabled", True):
        return
    if cam_id in state.camera_threads:
        thread, _stop_evt = state.camera_threads[cam_id]
        if thread.is_alive():
            return
        state.camera_threads.pop(cam_id, None)

    execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
    schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
    scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
    missing_model_keys = model_manager.missing_model_keys(scheduled_plan["required_model_keys"])
    state.camera_frames[cam_id] = None
    state.camera_detections[cam_id] = []
    state.camera_detection_history[cam_id] = []
    state.camera_clean_frames[cam_id] = None
    state.camera_frame_updated_at.pop(cam_id, None)
    state.camera_schedule_telemetry.pop(cam_id, None)
    if missing_model_keys:
        state.camera_runtime_status[cam_id] = "awaiting_model_install"
        logger.warning("Skipping camera start until models are ready", extra={"camera_id": cam_id, "missing_models": missing_model_keys})
        return

    stop_evt = threading.Event()
    thread = threading.Thread(target=video_processor, args=(cam_id, stop_evt), daemon=True)
    thread.start()
    state.camera_threads[cam_id] = (thread, stop_evt)
    state.camera_worker_started_at[cam_id] = time.time()
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
    state.camera_clean_frames.pop(cam_id, None)
    state.camera_frame_updated_at.pop(cam_id, None)
    state.camera_worker_started_at.pop(cam_id, None)
    state.camera_detections.pop(cam_id, None)
    state.camera_detection_history.pop(cam_id, None)
    state.camera_schedule_telemetry.pop(cam_id, None)
    state.camera_runtime_status[cam_id] = "offline"


def restart_camera(cam_id: str):
    stop_camera(cam_id)
    start_camera(cam_id)


def restart_all_cameras():
    cfg = get_config()
    for cam_id in list(state.camera_threads):
        stop_camera(cam_id)
    for cam_id in cfg["cameras"]:
        start_camera(cam_id)


async def camera_start_retry_loop(interval_seconds: float = CAMERA_START_RETRY_INTERVAL_SECONDS):
    """Retry cameras that were skipped because the model server was unavailable."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            cfg = get_config()
            for cam_id, cam in cfg.get("cameras", {}).items():
                if not cam.get("enabled", True):
                    continue
                if cam_id in state.camera_threads:
                    continue
                if state.camera_runtime_status.get(cam_id) != "awaiting_model_install":
                    continue
                execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
                schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
                scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
                missing_model_keys = model_manager.missing_model_keys(scheduled_plan["required_model_keys"])
                if missing_model_keys:
                    continue
                logger.info("Retrying camera start after model readiness recovered", extra={"camera_id": cam_id})
                start_camera(cam_id)
        except Exception:
            logger.exception("Camera start retry loop failed")


async def camera_frame_watchdog_loop(
    interval_seconds: float = CAMERA_FRAME_WATCHDOG_INTERVAL_SECONDS,
    restart_after_seconds: float = CAMERA_STALE_RESTART_SECONDS,
):
    """Restart enabled camera workers that stop publishing fresh frames."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            cfg = get_config()
            now = time.time()
            for cam_id, cam in cfg.get("cameras", {}).items():
                if not cam.get("enabled", True):
                    continue
                thread_info = state.camera_threads.get(cam_id)
                if not thread_info:
                    continue
                thread, _stop_event = thread_info
                if not thread.is_alive():
                    continue
                age = _frame_age_seconds(cam_id)
                worker_age = now - state.camera_worker_started_at.get(cam_id, now)
                status = state.camera_runtime_status.get(cam_id, "starting")
                if age is not None and age <= restart_after_seconds and status == "running":
                    continue
                if status == "starting" and worker_age < restart_after_seconds:
                    continue
                if status not in {"reconnecting", "starting", "stale"} and age is not None and age <= restart_after_seconds:
                    continue
                last_restart = _camera_watchdog_restart_at.get(cam_id, 0.0)
                if now - last_restart < restart_after_seconds:
                    continue
                _camera_watchdog_restart_at[cam_id] = now
                logger.warning(
                    "Restarting stale camera worker",
                    extra={
                        "camera_id": cam_id,
                        "runtime_status": status,
                        "last_frame_age_seconds": None if age is None else round(age, 1),
                    },
                )
                await asyncio.to_thread(restart_camera, cam_id)
        except Exception:
            logger.exception("Camera frame watchdog failed")


def mjpeg_generator(camera_id: str):
    while True:
        frame_bytes = state.camera_frames.get(camera_id)
        if frame_bytes is not None and _is_live_frame_fresh(camera_id):
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        else:
            status = state.camera_runtime_status.get(camera_id, "starting")
            placeholder = _status_frame(camera_id, status)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
        cfg = get_config()
        fps = cfg["global"]["target_fps"]
        time.sleep(1.0 / fps)
