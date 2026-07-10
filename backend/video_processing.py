"""SafetyLens video processing — alerts, grouped inference, MJPEG streaming."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import cv2
import numpy as np
import requests

import alert_store
import alert_delivery_store
import alert_delivery_worker
import face_analyzer
import face_store
import inference_scheduler
import licensing
import model_manager
import state
import notification_dispatcher
from alert_pipeline import AlertPipeline, DeliveryOutcome
from camera_capture import (
    CAMERA_STOP_TIMEOUT_SECONDS,
    open_video_capture,
    reconnect_delay_seconds,
    redact_video_source,
)
from camera_connection import build_rtsp_url
from camera_planner import build_execution_plan
from capability_registry import CLASS_TERM_TO_CAPABILITY
from config_manager import get_config, get_config_snapshot
from constants import (
    COCO_NAMES,
    OLLAMA_URL,
    VIDEO_DIR,
    VIOLATION_THRESHOLD,
    VLM_TIMEOUT_SECONDS,
)
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
from mjpeg_fanout import stream_fanout

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
    stream_fanout.publish(camera_id, annotated_jpeg)


def _clear_camera_observation(camera_id: str) -> None:
    """Discard current-frame state that is invalid once a source disconnects."""
    state.camera_frames[camera_id] = None
    state.camera_clean_frames[camera_id] = None
    state.camera_detections[camera_id] = []
    _last_pose_results.pop(camera_id, None)
    stream_fanout.clear(camera_id)

_alert_pipeline: AlertPipeline | None = None
_alert_pipeline_lock = threading.Lock()
_alert_event_loop: asyncio.AbstractEventLoop | None = None
_alert_backfill_task: asyncio.Task | None = None
_alert_reconciliation_task: asyncio.Task | None = None


def _broadcast_persisted_alert(alert: dict) -> None:
    if _alert_event_loop is None or not _alert_event_loop.is_running():
        logger.warning("Alert persisted without an active websocket event loop", extra={"alert_id": alert.get("id")})
        return
    broadcast = asyncio.run_coroutine_threadsafe(
        broadcast_alert({"type": "alert", "data": alert}),
        _alert_event_loop,
    )

    def log_broadcast_failure(completed) -> None:
        try:
            completed.result()
        except Exception:
            logger.exception(
                "Persisted alert websocket broadcast failed",
                extra={"alert_id": alert.get("id")},
            )

    broadcast.add_done_callback(log_broadcast_failure)


def _outbox_handoff(
    _alert: dict,
    _output_ids: list[str] | None = None,
) -> DeliveryOutcome:
    """Wake durable workers after the alert+outbox transaction commits."""
    alert_delivery_worker.wake()
    handled_targets = tuple(_output_ids or ("durable_outbox",))
    return DeliveryOutcome(handled_output_ids=handled_targets)


def _get_alert_pipeline() -> AlertPipeline:
    global _alert_pipeline
    if _alert_pipeline is None:
        with _alert_pipeline_lock:
            if _alert_pipeline is None:
                _alert_pipeline = AlertPipeline(
                    persist_alert=alert_store.create_alert,
                    deliver_alert=_outbox_handoff,
                    on_persisted=_broadcast_persisted_alert,
                    persist_queue_size=int(os.getenv("ALERT_PERSIST_QUEUE_SIZE", "256")),
                    delivery_queue_size=int(os.getenv("ALERT_DELIVERY_QUEUE_SIZE", "256")),
                    delivery_workers=int(os.getenv("ALERT_DELIVERY_WORKERS", "1")),
                    delivery_attempts=int(os.getenv("ALERT_DELIVERY_ATTEMPTS", "3")),
                    delivery_retry_delay=float(
                        os.getenv("ALERT_DELIVERY_RETRY_DELAY_SECONDS", "1.0")
                    ),
                    delivery_retry_queue_size=int(
                        os.getenv("ALERT_DELIVERY_RETRY_QUEUE_SIZE", "256")
                    ),
                )
    return _alert_pipeline


def start_alert_pipeline() -> None:
    global _alert_event_loop
    try:
        _alert_event_loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("Starting alert pipeline without a websocket event loop")
        _alert_event_loop = None

    pipeline = None
    try:
        alert_delivery_worker.start()
        pipeline = _get_alert_pipeline()
        pipeline.start()
    except Exception:
        if pipeline is not None:
            try:
                pipeline.shutdown(wait=False, timeout=1.0)
            except Exception:
                logger.error("Legacy alert pipeline rollback failed")
        try:
            alert_delivery_worker.stop(1.0)
        except Exception:
            logger.error("Durable alert worker rollback failed")
        _alert_event_loop = None
        raise


async def start_alert_delivery_workers() -> None:
    """Reconcile state, bound startup work, then continue migration online."""
    global _alert_backfill_task, _alert_reconciliation_task
    resolved, stale_has_more = await asyncio.to_thread(
        alert_store.auto_resolve_stale_alerts_batch
    )
    if resolved:
        logger.info(
            "Reconciled bounded stale-alert startup batch",
            extra={"alert_count": resolved, "has_more": stale_has_more},
        )
    cfg = get_config_snapshot()
    active_alert_cutoff = (
        datetime.now(timezone.utc)
        - timedelta(hours=alert_store.AUTO_RESOLVE_MAX_AGE_HOURS)
    ).isoformat()
    backfilled, cursor = await asyncio.to_thread(
        alert_delivery_store.backfill_active_escalations_batch,
        cfg,
        notification_dispatcher.resolve_delivery_targets,
        minimum_timestamp=active_alert_cutoff,
    )
    if backfilled:
        logger.info("Backfilled durable escalation work", extra={"delivery_count": backfilled})
    start_alert_pipeline()
    if stale_has_more:
        _alert_reconciliation_task = asyncio.create_task(
            _continue_stale_alert_reconciliation(),
            name="alert-stale-reconciliation",
        )
    if cursor is not None:
        _alert_backfill_task = asyncio.create_task(
            _continue_alert_delivery_backfill(cfg, cursor, active_alert_cutoff),
            name="alert-escalation-backfill",
        )


async def _continue_stale_alert_reconciliation() -> None:
    """Finish stale-alert reconciliation after workers begin serving."""
    global _alert_reconciliation_task
    try:
        resolved = await alert_store.reconcile_stale_alerts()
        if resolved:
            logger.info(
                "Completed background stale-alert reconciliation",
                extra={"alert_count": resolved},
            )
    except Exception as exc:
        logger.error(
            "Background stale-alert reconciliation stopped; periodic scan will retry",
            extra={"error_type": type(exc).__name__},
        )
    finally:
        _alert_reconciliation_task = None


async def _continue_alert_delivery_backfill(
    cfg: dict,
    cursor: str,
    minimum_timestamp: str,
) -> None:
    """Finish an upgrade scan in bounded transactions after serving starts."""
    global _alert_backfill_task
    total = 0
    try:
        while cursor is not None:
            inserted, cursor = await asyncio.to_thread(
                alert_delivery_store.backfill_active_escalations_batch,
                cfg,
                notification_dispatcher.resolve_delivery_targets,
                after_id=cursor,
                minimum_timestamp=minimum_timestamp,
            )
            total += inserted
            await asyncio.sleep(0)
        if total:
            logger.info(
                "Completed background escalation backfill",
                extra={"delivery_count": total},
            )
    except Exception as exc:
        logger.error(
            "Background escalation backfill stopped; restart will resume idempotently",
            extra={"error_type": type(exc).__name__},
        )
    finally:
        _alert_backfill_task = None


def stop_alert_pipeline(timeout: float = 10.0) -> bool:
    global _alert_event_loop
    deadline = time.monotonic() + max(0.0, timeout)
    drained = True
    outbox_stopped = False
    try:
        if _alert_pipeline is not None:
            try:
                drained = _alert_pipeline.shutdown(
                    wait=True,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except Exception:
                drained = False
                logger.error("Legacy alert pipeline shutdown failed")
    finally:
        try:
            outbox_stopped = alert_delivery_worker.stop(
                max(0.0, deadline - time.monotonic())
            )
        except Exception:
            logger.error("Durable alert worker shutdown failed")
        _alert_event_loop = None
    return drained and outbox_stopped


def get_alert_pipeline_stats() -> dict:
    if _alert_pipeline is None:
        return {
            "submitted": 0,
            "persisted": 0,
            "persistence_failures": 0,
            "consecutive_persistence_failures": 0,
            "last_persistence_failure_at": None,
            "last_persistence_success_at": None,
            "persistence_in_flight": False,
            "oldest_persistence_age_seconds": None,
            "callback_failures": 0,
            "delivered": 0,
            "outbox_handoffs": 0,
            "delivery_failures": 0,
            "partially_delivered": 0,
            "delivery_attempts": 0,
            "delivery_retries": 0,
            "delivery_terminal_failures": 0,
            "delivery_retry_exhausted": 0,
            "delivery_retry_queue_full": 0,
            "backpressure_events": 0,
            "running": False,
            "accepting": False,
            "active_submitters": 0,
            "persist_queue_depth": 0,
            "delivery_queue_depth": 0,
            "delivery_retry_queue_depth": 0,
            "delivery_retry_queue_capacity": 0,
            "persist_worker_alive": False,
            "retry_worker_alive": False,
            "delivery_workers_alive": 0,
            "outbox": {
                "pending": 0,
                "due": 0,
                "scheduled": 0,
                "leased": 0,
                "expired_leases": 0,
                "delivered": 0,
                "terminal": 0,
                "cancelled": 0,
                "ambiguous_history": 0,
                "oldest_due_age_seconds": None,
                "oldest_pending_age_seconds": None,
                "running": False,
                "workers_alive": 0,
                "claimer_alive": False,
                "renewer_alive": False,
                "active_sends": 0,
                "channel_inflight": {},
                "claim_errors": 0,
                "renewal_failures": 0,
                "fencing_failures": 0,
                "last_claim_at": None,
                "last_delivery_at": None,
            },
        }
    stats = _alert_pipeline.stats()
    stats["outbox"] = alert_delivery_worker.stats()
    return stats


def _encode_alert_snapshot_pair(
    annotated_frame: np.ndarray,
    clean_frame: np.ndarray,
    jpeg_quality: int,
    *,
    max_width: int = 854,
) -> tuple[bytes, bytes]:
    """Encode an alert's annotated and clean views from the same inference frame."""
    height, width = annotated_frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        size = (max_width, int(height * scale))
        annotated_frame = cv2.resize(annotated_frame, size)
        clean_frame = cv2.resize(clean_frame, size)

    annotated_ok, annotated_buffer = cv2.imencode(
        ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    clean_ok, clean_buffer = cv2.imencode(
        ".jpg", clean_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not annotated_ok or not clean_ok:
        raise RuntimeError("Failed to encode alert snapshot")
    return annotated_buffer.tobytes(), clean_buffer.tobytes()


def _encode_inference_snapshot_pair(
    camera_id: str,
    clean_frame: np.ndarray,
    detections: list[dict],
    jpeg_quality: int,
    *,
    annotated_frame: np.ndarray | None = None,
) -> tuple[bytes, bytes]:
    """Encode the exact inference frame even when resize-first skipped full-size drawing."""
    if annotated_frame is None:
        annotated_frame, clean_frame = _render_stream_views(
            camera_id,
            clean_frame,
            detections,
        )
    return _encode_alert_snapshot_pair(
        annotated_frame,
        clean_frame,
        jpeg_quality,
    )


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
    snapshot_jpeg: bytes | None = None,
    clean_snapshot_jpeg: bytes | None = None,
):
    cfg = get_config_snapshot()
    cam = cfg["cameras"].get(camera_id, {})
    if snapshot_jpeg is None:
        snapshot_jpeg = state.camera_frames.get(camera_id)
        clean_snapshot_jpeg = state.camera_clean_frames.get(camera_id)
    if not snapshot_jpeg:
        logger.debug("Skipping alert — no frame captured yet", extra={"camera_id": camera_id, "rule": rule})
        return None
    alert_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        delivery_targets = notification_dispatcher.resolve_delivery_targets(
            cfg,
            {
                "id": alert_id,
                "severity": severity,
                "timestamp": timestamp,
                "cameraId": camera_id,
            },
        )
        if not isinstance(delivery_targets, list) or any(
            not isinstance(target, dict) for target in delivery_targets
        ):
            raise TypeError("Delivery target resolver returned an invalid value")
    except Exception:
        # Alert persistence is the safety-critical path. A bad optional channel
        # configuration must not prevent the local alert from being recorded.
        delivery_targets = []
        logger.error(
            "Notification routing is invalid; persisting alert without external targets",
            extra={"alert_id": alert_id, "camera_id": camera_id},
        )
    return _get_alert_pipeline().submit({
        "alert_id": alert_id,
        "timestamp": timestamp,
        "delivery_targets": delivery_targets,
        "camera_id": camera_id,
        "camera_name": cam.get("name", camera_id),
        "zone": cam.get("zone", "Unknown"),
        "rule": rule,
        "severity": severity,
        "confidence": confidence,
        "description": description,
        "source": source,
        "snapshot_jpeg": snapshot_jpeg,
        "bboxes": bboxes,
        "clean_snapshot_jpeg": clean_snapshot_jpeg,
    })


def _reconcile_alert_persistence(
    camera_id: str,
    pending_by_rule: dict[str, object],
    active_violations: set[str],
    last_alert_by_rule: dict[str, float],
    violation_window: dict[str, list[bool]],
    *,
    now: float,
) -> None:
    """Apply completed persistence results on the owning camera thread."""
    for rule_key, submission in list(pending_by_rule.items()):
        if isinstance(submission, Mapping):
            persisted_alert = submission
        else:
            is_done = getattr(submission, "done", None)
            get_result = getattr(submission, "result", None)
            if not callable(is_done) or not callable(get_result):
                pending_by_rule.pop(rule_key, None)
                logger.error(
                    "Alert persistence submission returned an invalid result",
                    extra={"camera_id": camera_id, "rule": rule_key},
                )
                continue
            try:
                if not is_done():
                    continue
                persisted_alert = get_result()
            except Exception:
                pending_by_rule.pop(rule_key, None)
                logger.warning(
                    "Alert persistence failed; rule is eligible for fresh-inference retry",
                    extra={"camera_id": camera_id, "rule": rule_key},
                )
                continue

        pending_by_rule.pop(rule_key, None)
        if not isinstance(persisted_alert, Mapping):
            logger.error(
                "Alert persistence completed without a persisted alert",
                extra={"camera_id": camera_id, "rule": rule_key},
            )
            continue
        active_violations.add(rule_key)
        last_alert_by_rule[rule_key] = now
        violation_window[rule_key] = []


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
                "prompt": (
                    f"{vlm_cfg['prompt']}\n\n"
                    "End your response with exactly STATUS: SAFE or STATUS: VIOLATION."
                ),
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": vlm_cfg["temperature"],
                    "num_predict": vlm_cfg["max_tokens"],
                },
            },
            timeout=(5.0, VLM_TIMEOUT_SECONDS),
        )

        if resp.status_code == 200:
            return resp.json().get("response", "No response from VLM")
        return f"VLM error: {resp.status_code}"
    except Exception as exc:
        logger.warning(
            "VLM request failed",
            extra={"error_phase": "vlm_request", "error_type": type(exc).__name__},
        )
        return "VLM unavailable"


_VLM_EXPLICIT_VIOLATION_PHRASES = {
    "not wearing",
    "no helmet",
    "no vest",
    "too close",
}
_VLM_CONTEXT_ONLY_TERMS = {"forklift", "proximity", "clearance"}


def _vlm_result_verdict(result: str, keywords: list[str]) -> str:
    """Return ``safe``, ``violation``, or ``unknown`` for one VLM response."""
    normalized = str(result or "").lower()
    if not normalized or normalized.startswith(
        ("vlm error:", "vlm unavailable", "no response from vlm")
    ):
        return "unknown"
    verdicts = re.findall(r"\bstatus\s*:\s*(safe|violation)\b", normalized)
    if verdicts:
        return verdicts[-1]

    for raw_keyword in keywords:
        keyword = str(raw_keyword or "").strip().lower()
        if not keyword or keyword in _VLM_CONTEXT_ONLY_TERMS:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
        for match in pattern.finditer(normalized):
            if keyword in _VLM_EXPLICIT_VIOLATION_PHRASES:
                return "violation"
            prefix = normalized[max(0, match.start() - 40):match.start()]
            negated = re.search(
                r"\b(?:no|not|without|none|zero|free\s+of)\b"
                r"(?:\W+\w+){0,2}\W*$",
                prefix,
            )
            if not negated:
                return "violation"
    return "safe"


def _vlm_result_is_violation(result: str, keywords: list[str]) -> bool:
    """Prefer the structured verdict and safely fall back to keyword matching."""
    return _vlm_result_verdict(result, keywords) == "violation"


def _vlm_interval_seconds(value: object) -> float:
    try:
        interval = float(value)
    except (TypeError, ValueError):
        interval = 45.0
    if not math.isfinite(interval):
        interval = 45.0
    return min(24 * 60 * 60, max(5.0, interval))


def _vlm_worker_loop(camera_id: str, stop_event: threading.Event):
    pending_submission = None
    pending_generation = 0
    analysis_generation = 0
    last_safe_generation = 0
    violation_active = False

    while not stop_event.is_set():
        if pending_submission is not None:
            persisted_alert = None
            submission_complete = isinstance(pending_submission, Mapping)
            if submission_complete:
                persisted_alert = pending_submission
            else:
                is_done = getattr(pending_submission, "done", None)
                get_result = getattr(pending_submission, "result", None)
                if not callable(is_done) or not callable(get_result):
                    submission_complete = True
                    logger.error(
                        "VLM alert persistence returned an invalid result",
                        extra={"camera_id": camera_id},
                    )
                elif is_done():
                    submission_complete = True
                    try:
                        persisted_alert = get_result()
                    except Exception:
                        logger.warning(
                            "VLM alert persistence failed; incident remains retryable",
                            extra={"camera_id": camera_id},
                        )

            if submission_complete:
                pending_submission = None
                if isinstance(persisted_alert, Mapping):
                    # A SAFE result observed after submission starts a new
                    # incident generation. A late Future must not reactivate
                    # the old incident after that reset.
                    if pending_generation > last_safe_generation:
                        violation_active = True
                elif persisted_alert is not None:
                    logger.error(
                        "VLM alert persistence completed without a persisted alert",
                        extra={"camera_id": camera_id},
                    )

        cfg = get_config()
        vlm_cfg = cfg["vlm"]
        if stop_event.wait(_vlm_interval_seconds(vlm_cfg.get("interval", 45))):
            return

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
        if stop_event.is_set():
            logger.debug("Discarding VLM result after camera stop", extra={"camera_id": camera_id})
            return
        elapsed = time.time() - started
        logger.info("VLM analysis done", extra={"camera_id": camera_id, "elapsed": round(elapsed, 1)})

        with state.vlm_lock:
            state.vlm_last_results[camera_id] = {
                "text": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed": round(elapsed, 1),
            }

        analysis_generation += 1
        keywords = vlm_cfg.get("violation_keywords", [])
        verdict = _vlm_result_verdict(result, keywords)
        if verdict == "safe":
            last_safe_generation = analysis_generation
            violation_active = False
            continue
        if verdict != "violation" or violation_active or pending_submission is not None:
            continue
        if stop_event.is_set():
            return
        try:
            submission = create_alert(
                camera_id=camera_id,
                rule="VLM Scene Analysis",
                severity="P2",
                confidence=0.92,
                description=result[:200],
                source=f"VLM ({vlm_cfg['model']})",
                snapshot_jpeg=frame_bytes,
            )
        except Exception:
            submission = None
            logger.error(
                "VLM alert persistence submission failed; incident remains retryable",
                extra={"camera_id": camera_id},
            )
        if isinstance(submission, Mapping):
            violation_active = True
        elif submission is not None:
            pending_submission = submission
            pending_generation = analysis_generation
        if submission is not None:
            logger.debug("VLM alert submitted for persistence", extra={"camera_id": camera_id})


def vlm_worker(camera_id: str, stop_event: threading.Event):
    """Run one VLM companion and relinquish only its own registration."""
    try:
        _vlm_worker_loop(camera_id, stop_event)
    except Exception:
        logger.exception("VLM worker exited unexpectedly", extra={"camera_id": camera_id})
        raise
    finally:
        _deregister_worker_on_exit(state.vlm_threads, camera_id, stop_event)


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
        _finalize_camera_worker_exit(camera_id, stop_event)


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
    pending_alerts_by_rule: dict[str, object] = {}
    violation_window: dict[str, list[bool]] = {}
    window_size = 15
    last_annotated = None
    reconnect_failures = 0
    ever_received_frame = False
    safe_video_source = redact_video_source(video_source)
    last_detection_frame = None

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

            # Cached detections remain available for rendering, but only a newly
            # completed model result may advance temporal alert/fall state.
            fresh_inference_result = False
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
                    # Inference may outlive the bounded shutdown join. Never
                    # let that late result advance temporal state or enqueue
                    # an alert after the persistence pipeline has drained.
                    if stop_event.is_set():
                        break
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
                        if stop_event.is_set():
                            break
                    if annotated is None and frame.shape[1] <= STREAM_MAX_WIDTH:
                        annotated = _draw_stream_detection_records(
                            frame,
                            detections,
                            camera_id,
                        )
                    fresh_inference_result = True
                except Exception:
                    logger.exception("Detection failed", extra={"camera_id": camera_id})
                    annotated = frame
                    detections = []
                last_annotated = annotated
                if fresh_inference_result:
                    last_detection_frame = frame.copy()
                state.camera_detections[camera_id] = detections
                next_inference_at = inference_scheduler.next_inference_slot(
                    camera_id,
                    current_cfg,
                    inference_interval,
                    now=time.monotonic() + 1e-9,
                )
            detections = state.camera_detections.get(camera_id, [])

            if fresh_inference_result:
                _reconcile_alert_persistence(
                    camera_id,
                    pending_alerts_by_rule,
                    active_violations,
                    last_alert_by_rule,
                    violation_window,
                    now=time.time(),
                )

            # Fall detection runs independently — pose model doesn't need COCO detections
            has_fall_candidates = False
            fall_candidates = []
            if (
                fresh_inference_result
                and execution_plan.get("run_pose_specialist")
                and camera_id in _last_pose_results
            ):
                fall_candidates = check_fall_detections(_last_pose_results[camera_id], camera_id, frame)
                has_fall_candidates = len(fall_candidates) > 0

            if fresh_inference_result and (detections or has_fall_candidates):
                candidates = []
                frame_h, frame_w = frame.shape[:2]
                if detections:
                    if execution_plan.get("run_ppe_specialist"):
                        candidates.extend(check_yoloe_violations(detections, camera_id))
                    candidates.extend(check_violations(detections, camera_id))

                    candidates.extend(check_zone_intrusions(detections, camera_id, frame_w, frame_h))

                candidates.extend(fall_candidates)
                current_violation_rules = {candidate["rule"]: candidate for candidate in candidates}

                all_tracked = set(violation_window) | set(current_violation_rules)
                for rule_key in all_tracked:
                    violation_window.setdefault(rule_key, []).append(rule_key in current_violation_rules)
                    if len(violation_window[rule_key]) > window_size:
                        violation_window[rule_key] = violation_window[rule_key][-window_size:]

                for rule_key in list(active_violations):
                    recent_votes = violation_window.get(rule_key, [])[-window_size:]
                    # Persistence success resets the vote window. Require two
                    # fresh observations before declaring the incident clear;
                    # otherwise one still-positive frame immediately drops the
                    # active guard and permits periodic duplicate alerts.
                    if len(recent_votes) >= 2 and sum(recent_votes) < 2:
                        active_violations.discard(rule_key)

                now = time.time()
                for rule_key, candidate in current_violation_rules.items():
                    if rule_key in active_violations or rule_key in pending_alerts_by_rule:
                        continue
                    if now - last_alert_by_rule.get(rule_key, 0) < alert_cooldown:
                        continue
                    hits = sum(violation_window.get(rule_key, []))
                    rule_threshold = candidate.get("threshold") or VIOLATION_THRESHOLD
                    if hits < rule_threshold:
                        continue
                    violation_bboxes = extract_violation_bboxes(candidate["rule"], detections, frame_w, frame_h, camera_id)
                    snapshot_jpeg = None
                    clean_snapshot_jpeg = None
                    if last_detection_frame is not None:
                        try:
                            snapshot_jpeg, clean_snapshot_jpeg = _encode_inference_snapshot_pair(
                                camera_id,
                                last_detection_frame,
                                detections,
                                jpeg_quality,
                                annotated_frame=last_annotated,
                            )
                        except Exception:
                            logger.exception(
                                "Alert snapshot encoding failed",
                                extra={"camera_id": camera_id, "rule": candidate["rule"]},
                            )
                    try:
                        submission = create_alert(
                            camera_id=candidate["camera_id"],
                            rule=candidate["rule"],
                            severity=candidate["severity"],
                            confidence=candidate["confidence"],
                            description=candidate["description"],
                            source=candidate["source"],
                            bboxes=violation_bboxes,
                            snapshot_jpeg=snapshot_jpeg,
                            clean_snapshot_jpeg=clean_snapshot_jpeg,
                        )
                    except Exception:
                        submission = None
                        logger.error(
                            "Alert persistence submission failed; rule remains retryable",
                            extra={"camera_id": camera_id, "rule": candidate["rule"]},
                        )
                    if submission is not None:
                        pending_alerts_by_rule[rule_key] = submission
                        _reconcile_alert_persistence(
                            camera_id,
                            pending_alerts_by_rule,
                            active_violations,
                            last_alert_by_rule,
                            violation_window,
                            now=now,
                        )
                        logger.debug(
                            "Detection alert submitted for persistence",
                            extra={"camera_id": camera_id, "rule": candidate["rule"]},
                        )

                for rule_key in list(violation_window):
                    if len(violation_window[rule_key]) >= window_size and sum(violation_window[rule_key]) == 0:
                        violation_window.pop(rule_key, None)
            elif fresh_inference_result:
                for rule_key in list(violation_window):
                    violation_window[rule_key].append(False)
                    if len(violation_window[rule_key]) > window_size:
                        violation_window[rule_key] = violation_window[rule_key][-window_size:]
                    if (
                        len(violation_window[rule_key]) >= 2
                        and sum(violation_window[rule_key]) == 0
                    ):
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


_camera_lifecycle_locks_guard = threading.Lock()
_camera_lifecycle_locks: dict[str, threading.RLock] = {}
_camera_worker_registry_lock = threading.RLock()
_camera_lifecycle_fence_lock = threading.Lock()
_camera_lifecycle_shutdown = threading.Event()
CAMERA_WORKER_HEAL_INTERVAL_SECONDS = 10.0


def _camera_lifecycle_lock(cam_id: str) -> threading.RLock:
    """Return the stable lock that owns all worker transitions for a camera."""
    with _camera_lifecycle_locks_guard:
        return _camera_lifecycle_locks.setdefault(cam_id, threading.RLock())


def begin_camera_lifecycle_shutdown() -> None:
    """Fence every camera/VLM start before shutdown begins draining workers."""
    # A start holds this lock from its final fence check through registry
    # publication and Thread.start(). Once this returns, every pre-fence start
    # is therefore either visible to stop_all_camera_workers() or rejected.
    with _camera_lifecycle_fence_lock:
        _camera_lifecycle_shutdown.set()


def resume_camera_lifecycle() -> None:
    """Allow starts for a new application lifecycle."""
    with _camera_lifecycle_fence_lock:
        _camera_lifecycle_shutdown.clear()


def camera_lifecycle_shutting_down() -> bool:
    return _camera_lifecycle_shutdown.is_set()


def _remove_worker_if_owned(
    workers: dict[str, tuple[threading.Thread, threading.Event]],
    cam_id: str,
    ownership: tuple[threading.Thread, threading.Event],
) -> None:
    """Never let an old transition delete a newer worker registration."""
    with _camera_worker_registry_lock:
        if workers.get(cam_id) is ownership:
            del workers[cam_id]


def _deregister_worker_on_exit(
    workers: dict[str, tuple[threading.Thread, threading.Event]],
    cam_id: str,
    stop_event: threading.Event,
) -> bool:
    """Remove only the calling worker's exact thread/event registration.

    ``True`` means the caller still owns cleanup for this camera. An absent
    registration is treated as a direct invocation, which keeps the worker
    functions independently testable. A different registration means a newer
    owner already exists and must not be erased or have its observations
    cleared by the old worker's ``finally`` block.
    """
    current_thread = threading.current_thread()
    with _camera_worker_registry_lock:
        ownership = workers.get(cam_id)
        if ownership is None:
            return True
        registered_thread, registered_stop_event = ownership
        if (
            registered_thread is current_thread
            and registered_stop_event is stop_event
        ):
            del workers[cam_id]
            return True
        return False


def _finalize_camera_worker_exit(
    cam_id: str,
    stop_event: threading.Event,
) -> None:
    """Atomically relinquish ownership and clear only that owner's output.

    The registry lock stays held through observation/status cleanup. A new
    start therefore cannot register and publish between deregistration and an
    old worker's cleanup. ``stop_camera`` may safely join while holding the
    lifecycle lock because this path never tries to acquire that lock.
    """
    current_thread = threading.current_thread()
    with _camera_worker_registry_lock:
        ownership = state.camera_threads.get(cam_id)
        if ownership is not None:
            registered_thread, registered_stop_event = ownership
            if (
                registered_thread is not current_thread
                or registered_stop_event is not stop_event
            ):
                return
            del state.camera_threads[cam_id]
        _clear_camera_observation(cam_id)
        state.camera_runtime_status[cam_id] = (
            "offline" if stop_event.is_set() else "error"
        )


def _worker_registration_active(
    workers: dict[str, tuple[threading.Thread, threading.Event]],
    cam_id: str,
) -> bool:
    with _camera_worker_registry_lock:
        ownership = workers.get(cam_id)
    if ownership is None:
        return False
    thread, stop_event = ownership
    return thread.is_alive() and not stop_event.is_set()


def start_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if _camera_lifecycle_shutdown.is_set():
            logger.debug(
                "Camera start rejected during lifecycle shutdown",
                extra={"camera_id": cam_id},
            )
            return False
        cfg = get_config()
        cam = cfg["cameras"].get(cam_id)
        if not cam or not cam.get("enabled", True):
            return False

        with _camera_worker_registry_lock:
            existing = state.camera_threads.get(cam_id)
        if existing is not None:
            existing_thread, existing_stop_event = existing
            if existing_thread.is_alive():
                logger.warning(
                    "Camera worker already running; duplicate start suppressed",
                    extra={"camera_id": cam_id},
                )
                # A previous yolo+vlm start can have succeeded for the video
                # worker and failed while creating its VLM companion. A safe
                # retry heals that missing companion without replacing video
                # ownership. Never do this once the camera is stopping.
                if (
                    cam.get("demo") == "yolo+vlm"
                    and not existing_stop_event.is_set()
                ):
                    start_vlm_for_camera(cam_id)
                return False
            _remove_worker_if_owned(state.camera_threads, cam_id, existing)

        execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
        missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
        _clear_camera_observation(cam_id)
        if missing_model_keys:
            state.camera_runtime_status[cam_id] = "awaiting_model_install"
            logger.warning("Skipping camera start until models are ready", extra={"camera_id": cam_id, "missing_models": missing_model_keys})
            return False

        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(
                target=video_processor,
                args=(cam_id, stop_evt),
                daemon=True,
            )
            ownership = (thread, stop_evt)
            # Publish ownership before the OS thread can run. If start fails,
            # only this exact registration is rolled back so a retry cannot
            # be erased.
            with _camera_worker_registry_lock:
                state.camera_threads[cam_id] = ownership
            state.camera_runtime_status[cam_id] = "starting"
            try:
                thread.start()
            except Exception:
                stop_evt.set()
                _remove_worker_if_owned(state.camera_threads, cam_id, ownership)
                state.camera_runtime_status[cam_id] = "error"
                logger.exception(
                    "Failed to start video processor",
                    extra={"camera_id": cam_id},
                )
                raise
        logger.info("Started video processor", extra={"camera_id": cam_id})

        if cam.get("demo") == "yolo+vlm":
            start_vlm_for_camera(cam_id)
        return True


def start_vlm_for_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if _camera_lifecycle_shutdown.is_set():
            logger.debug(
                "VLM start rejected during lifecycle shutdown",
                extra={"camera_id": cam_id},
            )
            return False

        # Revalidate after acquiring lifecycle ownership. The healer may have
        # made its decision from a config generation that an API update has
        # since replaced while waiting for this lock.
        cfg = get_config_snapshot("cameras")
        cam = cfg.get("cameras", {}).get(cam_id)
        if (
            not cam
            or not cam.get("enabled", True)
            or cam.get("demo") != "yolo+vlm"
        ):
            return False

        with _camera_worker_registry_lock:
            camera_ownership = state.camera_threads.get(cam_id)
        if camera_ownership is None:
            return False
        camera_thread, camera_stop_event = camera_ownership
        if not camera_thread.is_alive() or camera_stop_event.is_set():
            return False

        with _camera_worker_registry_lock:
            existing = state.vlm_threads.get(cam_id)
        if existing is not None:
            existing_thread, _ = existing
            if existing_thread.is_alive():
                logger.warning(
                    "VLM worker already running; duplicate start suppressed",
                    extra={"camera_id": cam_id},
                )
                return False
            _remove_worker_if_owned(state.vlm_threads, cam_id, existing)

        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(
                target=vlm_worker,
                args=(cam_id, stop_evt),
                daemon=True,
            )
            ownership = (thread, stop_evt)
            with _camera_worker_registry_lock:
                state.vlm_threads[cam_id] = ownership
            try:
                thread.start()
            except Exception:
                stop_evt.set()
                _remove_worker_if_owned(state.vlm_threads, cam_id, ownership)
                logger.exception(
                    "Failed to start VLM worker",
                    extra={"camera_id": cam_id},
                )
                return False
        logger.info("Started VLM worker", extra={"camera_id": cam_id})
        return True


def stop_vlm_for_camera(cam_id: str) -> bool:
    """Stop only a camera's VLM companion, leaving its video worker running."""
    with _camera_lifecycle_lock(cam_id):
        with _camera_worker_registry_lock:
            ownership = state.vlm_threads.get(cam_id)
        if ownership is None:
            return True

        thread, stop_event = ownership
        stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=5)
        if thread.is_alive():
            logger.warning(
                "VLM worker still stopping; retaining worker reference",
                extra={"camera_id": cam_id, "timeout_seconds": 5},
            )
            return False
        _remove_worker_if_owned(state.vlm_threads, cam_id, ownership)
        return True


def heal_camera_workers_once() -> None:
    """Repair missing camera workers and yolo+vlm companions once.

    Each configured camera is isolated so a resource or configuration failure
    cannot prevent later cameras from being checked. Lifecycle locks in the
    start functions fence this pass against CRUD-driven restarts.
    """
    cfg = get_config_snapshot("cameras")
    with _camera_worker_registry_lock:
        registered_vlm_ids = set(state.vlm_threads)
    camera_ids = set(cfg.get("cameras", {})) | registered_vlm_ids
    for cam_id in camera_ids:
        if _camera_lifecycle_shutdown.is_set():
            return
        try:
            with _camera_lifecycle_lock(cam_id):
                if _camera_lifecycle_shutdown.is_set():
                    return

                # Refresh under transition ownership. A stale outer snapshot
                # is useful only for discovering candidates, never for making
                # a start/stop decision.
                current_cfg = get_config_snapshot("cameras")
                cam = current_cfg.get("cameras", {}).get(cam_id)
                enabled = bool(cam and cam.get("enabled", True))
                vlm_expected = bool(enabled and cam.get("demo") == "yolo+vlm")

                if not vlm_expected and cam_id in state.vlm_threads:
                    stop_vlm_for_camera(cam_id)
                if not enabled:
                    continue
                if not _worker_registration_active(state.camera_threads, cam_id):
                    start_camera(cam_id)
                    continue
                if (
                    vlm_expected
                    and not _worker_registration_active(state.vlm_threads, cam_id)
                ):
                    start_vlm_for_camera(cam_id)
        except Exception:
            logger.exception(
                "Camera lifecycle healing failed; later cameras will still be checked",
                extra={"camera_id": cam_id},
            )


async def camera_worker_healing_loop(
    *, interval_seconds: float = CAMERA_WORKER_HEAL_INTERVAL_SECONDS
) -> None:
    """Run bounded lifecycle repair until the owning asyncio task is cancelled."""
    try:
        interval = float(interval_seconds)
    except (TypeError, ValueError):
        interval = CAMERA_WORKER_HEAL_INTERVAL_SECONDS
    if not math.isfinite(interval):
        interval = CAMERA_WORKER_HEAL_INTERVAL_SECONDS
    interval = min(300.0, max(1.0, interval))

    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(heal_camera_workers_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Camera lifecycle healing pass failed")


def stop_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        camera_stopped = True
        vlm_stopped = True
        with _camera_worker_registry_lock:
            camera_ownership = state.camera_threads.get(cam_id)
        if camera_ownership is not None:
            thread, stop_evt = camera_ownership
            stop_evt.set()
            state.camera_runtime_status[cam_id] = "stopping"
            if thread is not threading.current_thread():
                thread.join(timeout=CAMERA_STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                camera_stopped = False
                logger.warning(
                    "Camera worker still stopping; retaining worker reference",
                    extra={"camera_id": cam_id, "timeout_seconds": CAMERA_STOP_TIMEOUT_SECONDS},
                )
            else:
                _remove_worker_if_owned(state.camera_threads, cam_id, camera_ownership)
        with _camera_worker_registry_lock:
            vlm_ownership = state.vlm_threads.get(cam_id)
        if vlm_ownership is not None:
            thread, stop_evt = vlm_ownership
            stop_evt.set()
            state.camera_runtime_status[cam_id] = "stopping"
            if thread is not threading.current_thread():
                thread.join(timeout=5)
            if thread.is_alive():
                vlm_stopped = False
                logger.warning(
                    "VLM worker still stopping; retaining worker reference",
                    extra={"camera_id": cam_id, "timeout_seconds": 5},
                )
            else:
                _remove_worker_if_owned(state.vlm_threads, cam_id, vlm_ownership)
        if not camera_stopped or not vlm_stopped:
            return False
        state.camera_frames.pop(cam_id, None)
        state.camera_clean_frames.pop(cam_id, None)
        state.camera_detections.pop(cam_id, None)
        _last_pose_results.pop(cam_id, None)
        state.camera_runtime_status[cam_id] = "offline"
        return True


def stop_all_camera_workers() -> bool:
    """Stop every registered video/VLM worker from a stable registry snapshot."""
    with _camera_worker_registry_lock:
        camera_ids = set(state.camera_threads) | set(state.vlm_threads)
    stopped = True
    for cam_id in camera_ids:
        if not stop_camera(cam_id):
            stopped = False
    return stopped


def restart_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if not stop_camera(cam_id):
            logger.warning("Camera restart deferred until existing worker exits", extra={"camera_id": cam_id})
            return False
        start_camera(cam_id)
        return True


def restart_all_cameras():
    cfg = get_config()
    configured = set(cfg["cameras"])
    with _camera_worker_registry_lock:
        existing = set(state.camera_threads) | set(state.vlm_threads)
    for cam_id in configured | existing:
        if cam_id in configured:
            restart_camera(cam_id)
        else:
            stop_camera(cam_id)
