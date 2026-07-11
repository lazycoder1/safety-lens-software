"""Rakshak Lens video processing — alerts, grouped inference, MJPEG streaming."""

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
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import cv2
import numpy as np
import requests

import alert_delivery_store
import alert_delivery_worker
import alert_store
import face_analyzer
import face_store
import plate_store
import inference_scheduler
import licensing
import model_manager
import object_lifecycle_analytics
import state
import notification_dispatcher
import policy_engine
from camera_capture import (
    CAMERA_STOP_TIMEOUT_SECONDS,
    CameraConnectionEvent,
    CameraConnectionTracker,
    RTSP_BUFFER_DRAIN_MAX_SECONDS,
    open_video_capture,
    reconnect_delay_seconds,
    redact_video_source,
)
from alert_pipeline import AlertPipeline, DeliveryOutcome
from camera_connection import build_rtsp_url
from camera_planner import build_execution_plan, required_model_keys_for_capabilities
from capability_registry import CAPABILITY_REGISTRY, CLASS_TERM_TO_CAPABILITY, RULE_ID_TO_CAPABILITY
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
    draw_pose_detections,
    extract_violation_bboxes,
)
from mjpeg_fanout import stream_fanout

LICENSE_PAUSE_INTERVAL = 1.0
CAMERA_START_RETRY_INTERVAL_SECONDS = 10.0
CAMERA_FRAME_WATCHDOG_INTERVAL_SECONDS = 5.0
CAMERA_STALE_RESTART_SECONDS = 60.0
RTSP_BUFFER_DRAIN_MAX_FRAMES = 30
RTSP_BUFFER_DRAIN_BLOCK_SECONDS = 0.012

logger = logging.getLogger("rakshak_lens")

_camera_watchdog_restart_at: dict[str, float] = {}
# Compatibility state retained for lifecycle cleanup and older integrations;
# the feature-rich pipeline passes fresh pose results explicitly.
_last_pose_results: dict[str, object] = {}


def _is_executor_shutdown_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "cannot schedule new futures after shutdown" in message
        or "cannot schedule new futures after interpreter shutdown" in message
    )


def _drain_inference_executor(executor: ThreadPoolExecutor, pending_inference) -> None:
    """Prevent reconnects from leaving an uncancellable inference job behind."""
    if pending_inference is not None:
        pending_inference.cancel()
    # Future.cancel() cannot stop a task that is already running. Waiting here
    # keeps a flapping camera from accumulating one model request per reconnect.
    executor.shutdown(wait=True, cancel_futures=True)

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
STREAM_MAX_WIDTH = 854
IDLE_STREAM_FPS = 1.0


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


def _stream_publication_due(
    camera_id: str,
    last_published_at: float,
    now: float,
    configured_stream_fps: float,
    had_subscribers: bool,
) -> tuple[bool, bool]:
    """Throttle idle JPEG work while retaining a one-FPS state heartbeat."""
    has_subscribers = stream_fanout.has_subscribers(camera_id)
    effective_fps = (
        configured_stream_fps
        if has_subscribers
        else min(configured_stream_fps, IDLE_STREAM_FPS)
    )
    due = (
        (has_subscribers and not had_subscribers)
        or _stream_publish_due(last_published_at, now, effective_fps)
    )
    return due, has_subscribers


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
    state.camera_frame_updated_at[camera_id] = time.time()
    stream_fanout.publish(camera_id, annotated_jpeg)


def _clear_camera_observation(camera_id: str) -> None:
    """Discard current-frame state that is invalid once a source disconnects."""
    state.camera_frames[camera_id] = None
    state.camera_clean_frames[camera_id] = None
    state.camera_detections[camera_id] = []
    state.camera_frame_updated_at.pop(camera_id, None)
    state.clear_camera_frame_dimensions(camera_id)
    stream_fanout.clear(camera_id)

_alert_pipeline: AlertPipeline | None = None
_alert_pipeline_lock = threading.Lock()
_alert_event_loop: asyncio.AbstractEventLoop | None = None
_alert_backfill_task: asyncio.Task | None = None
_alert_reconciliation_task: asyncio.Task | None = None


def _outbox_handoff(
    alert: dict,
    output_ids: list[str] | None = None,
) -> DeliveryOutcome:
    """Wake durable workers and retain feature-only output adapters."""
    alert_delivery_worker.wake()
    try:
        cfg = get_config_snapshot()
    except Exception:
        # The durable handoff itself is complete even when optional feature
        # output discovery is temporarily unavailable.
        logger.warning("Feature output discovery failed after durable outbox handoff")
        return DeliveryOutcome(handled_output_ids=("durable_outbox",))
    requested = None if output_ids is None else {str(value) for value in output_ids}
    direct_output_ids: list[str] = []
    handled = ["durable_outbox"]
    for output in cfg.get("alert_outputs", []):
        if not isinstance(output, dict):
            continue
        output_id = str(output.get("id") or "")
        if not output_id or (requested is not None and output_id not in requested):
            continue
        if requested is None and not output.get("enabled", False):
            continue
        output_type = str(output.get("type") or "").strip().lower()
        if output_type in {"in_app", "inapp", "browser_sound"}:
            handled.append(output_id)
        elif output_type not in {"telegram", "email", "webhook"}:
            direct_output_ids.append(output_id)

    if not direct_output_ids:
        return DeliveryOutcome(handled_output_ids=tuple(handled))

    results = notification_dispatcher.notify(
        alert,
        _alert_snapshot_path(alert),
        output_ids=direct_output_ids,
    )
    delivered: list[str] = []
    retry: list[str] = []
    terminal: list[str] = []
    classified: set[str] = set()
    for result in results:
        output_id = str(result.get("outputId") or "unknown")
        classified.add(output_id)
        status = result.get("status")
        if status in {notification_dispatcher.DELIVERED, notification_dispatcher.SIMULATED}:
            delivered.append(output_id)
        elif status == notification_dispatcher.FAILED:
            retry.append(output_id)
        else:
            terminal.append(output_id)
    terminal.extend(output_id for output_id in direct_output_ids if output_id not in classified)
    return DeliveryOutcome(
        delivered_output_ids=tuple(delivered),
        retry_output_ids=tuple(retry),
        terminal_output_ids=tuple(terminal),
        handled_output_ids=tuple(handled),
    )


def _alert_snapshot_path(alert: dict) -> str | None:
    snapshot_url = alert.get("snapshotUrl")
    if not snapshot_url:
        return None
    return str(alert_store.SNAPSHOTS_DIR / str(snapshot_url).split("/")[-1])


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
    """Encode an alert snapshot from the exact frame used for inference."""
    if annotated_frame is None:
        annotated_frame = _draw_stream_detection_records(
            clean_frame,
            detections,
            camera_id,
        )
    return _encode_alert_snapshot_pair(
        annotated_frame,
        clean_frame,
        jpeg_quality,
    )


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
    output_ids: list[str] | None = None,
    policy_id: str | None = None,
    priority: int | None = None,
    message: str | None = None,
    metadata: dict | None = None,
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
            output_ids=output_ids,
        )
        if not isinstance(delivery_targets, list) or any(
            not isinstance(target, dict) for target in delivery_targets
        ):
            raise TypeError("Delivery target resolver returned an invalid value")
    except Exception:
        delivery_targets = []
        logger.error(
            "Notification routing is invalid; persisting alert without external targets",
            extra={"alert_id": alert_id, "camera_id": camera_id},
        )
    return _get_alert_pipeline().submit(
        {
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
            "policy_id": policy_id,
            "priority": priority,
            "message": message,
            "metadata": metadata,
        },
        output_ids=output_ids,
    )


async def broadcast_alert(msg: dict):
    subscribers = tuple(state.alert_subscribers)
    if not subscribers:
        return
    results = await asyncio.gather(
        *(ws.send_json(msg) for ws in subscribers),
        return_exceptions=True,
    )
    for ws, result in zip(subscribers, results):
        if isinstance(result, BaseException) and ws in state.alert_subscribers:
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
    pose_results = None
    model_invocations = {
        "coco_primary": 0,
        "ppe_specialist": 0,
        "ppe_closed_set_candidate": 0,
        "yoloe_long_tail": 0,
        "fire_smoke_specialist": 0,
        "pose_specialist": 0,
    }
    ppe_prompts = execution_plan.get("ppe_prompt_terms") or []
    long_tail_prompts = execution_plan.get("yoloe_prompt_terms") or []
    batch_requests = []
    if execution_plan.get("run_coco_primary"):
        model_invocations["coco_primary"] += 1
        batch_requests.append({
            "request_id": "coco_primary",
            "model_key": "coco_primary",
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
        })
    if execution_plan.get("run_ppe_specialist") and ppe_prompts:
        ppe_conf = _rule_confidence_for_model_family(camera_id, "ppe_specialist", conf)
        model_invocations["ppe_specialist"] += 1
        batch_requests.append({
            "request_id": "ppe_specialist",
            "model_key": "ppe_specialist",
            "conf": ppe_conf,
            "device": device,
            "imgsz": imgsz,
            "classes": ppe_prompts,
        })
    if execution_plan.get("run_ppe_closed_set_candidate"):
        candidate_capabilities = _capabilities_for_model_key(
            execution_plan,
            "ppe_closed_set_candidate",
        )
        candidate_conf = _rule_confidence_for_capabilities(
            camera_id,
            candidate_capabilities,
            conf,
        )
        model_invocations["ppe_closed_set_candidate"] += 1
        batch_requests.append({
            "request_id": "ppe_closed_set_candidate",
            "model_key": "ppe_closed_set_candidate",
            "conf": candidate_conf,
            "device": device,
            "imgsz": imgsz,
        })
    if execution_plan.get("run_yoloe_long_tail") and long_tail_prompts:
        model_invocations["yoloe_long_tail"] += 1
        batch_requests.append({
            "request_id": "yoloe_long_tail",
            "model_key": "yoloe_long_tail",
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
            "classes": long_tail_prompts,
        })
    if execution_plan.get("run_fire_smoke_specialist"):
        fire_conf = _rule_confidence_for_capability(camera_id, "fire_smoke", conf)
        model_invocations["fire_smoke_specialist"] += 1
        batch_requests.append({
            "request_id": "fire_smoke_specialist",
            "model_key": "fire_smoke_specialist",
            "conf": fire_conf,
            "device": device,
            "imgsz": imgsz,
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

    if execution_plan.get("run_ppe_closed_set_candidate"):
        records = record_batches["ppe_closed_set_candidate"]
        candidate_detections = _detection_batch_from_records(
            records,
            "ppe_closed_set_candidate",
            class_names=CLOSED_SET_PPE_CLASS_NAMES,
        )
        detections.extend(candidate_detections)

    if execution_plan.get("run_yoloe_long_tail") and long_tail_prompts:
        records = record_batches["yoloe_long_tail"]
        long_tail_detections = _detection_batch_from_records(
            records,
            "yoloe_long_tail",
            class_names=long_tail_prompts,
        )
        detections.extend(long_tail_detections)

    if execution_plan.get("run_fire_smoke_specialist"):
        records = record_batches["fire_smoke_specialist"]
        fire_detections = _detection_batch_from_records(
            records,
            "fire_smoke_specialist",
            class_names=FIRE_SMOKE_CLASS_NAMES,
        )
        detections.extend(fire_detections)

    if execution_plan.get("run_pose_specialist"):
        model_invocations["pose_specialist"] += 1
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
        annotated, fall_dets = draw_pose_detections(annotated, pose_results, fall_only=False, camera_id=camera_id)
        detections.extend(_normalize_detection_batch(fall_dets, "pose_specialist"))

    if annotated is not None:
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


def _publish_camera_connection_health(
    camera_id: str,
    tracker: CameraConnectionTracker,
) -> None:
    state.update_camera_connection_health(
        camera_id,
        outage_active=tracker.outage_active,
        outage_started_monotonic=tracker.outage_started_monotonic,
        outage_failure_count=tracker.outage_failure_count,
        total_failure_count=tracker.total_failure_count,
        suppressed_failure_count=tracker.suppressed_failure_count,
        last_transition=tracker.last_transition,
        last_transition_monotonic=tracker.last_transition_monotonic,
        capture_backend=tracker.capture_backend,
    )


def _capture_backend_name(capture) -> str:
    backend = str(getattr(capture, "capture_backend", "ffmpeg"))
    return backend if backend in {"ffmpeg", "gstreamer_nvdec"} else "unknown"


def _connection_event_fields(event: CameraConnectionEvent) -> dict:
    return {
        "failure_count": event.failure_count,
        "total_failure_count": event.total_failure_count,
        "suppressed_failure_count": event.suppressed_failure_count,
        "outage_duration_seconds": event.outage_duration_seconds,
    }


def _record_rtsp_connection_failure(
    camera_id: str,
    tracker: CameraConnectionTracker,
    *,
    safe_source: str,
    retry_seconds: float,
    received_frame: bool,
    now: float,
) -> None:
    """Log only an outage transition or periodic aggregate, never every retry."""
    event = tracker.record_failure(now=now)
    _publish_camera_connection_health(camera_id, tracker)
    if event is None:
        return
    fields = {
        "camera_id": camera_id,
        "source": safe_source,
        "retry_seconds": round(retry_seconds, 2),
        "received_frame": received_frame,
        **_connection_event_fields(event),
    }
    if event.kind == "outage":
        logger.warning("Camera connection outage detected; retry scheduled", extra=fields)
    elif event.kind == "summary":
        logger.warning("Camera connection outage persists", extra=fields)


def _record_rtsp_connection_frame(
    camera_id: str,
    tracker: CameraConnectionTracker,
    *,
    now: float,
) -> bool:
    """Return true only when a stable window has forgiven reconnect debt."""
    event = tracker.record_frame(now=now)
    if event is None:
        return False
    _publish_camera_connection_health(camera_id, tracker)
    if event.kind != "recovered":
        return False
    logger.info(
        "Camera connection recovered after stable frame window",
        extra={
            "camera_id": camera_id,
            "stable_window_seconds": tracker.stable_window_seconds,
            **_connection_event_fields(event),
        },
    )
    return True


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
    return open_video_capture(video_source, stream_type=stream_type)


def _read_live_frame(cap: cv2.VideoCapture, stream_type: str) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if not ok or frame is None or stream_type != "rtsp":
        return ok, frame
    if getattr(cap, "delivers_latest_frame", False):
        return ok, frame
    if not callable(getattr(cap, "grab", None)):
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
    state.clear_camera_frame_dimensions(camera_id)


def _frame_age_seconds(camera_id: str):
    updated_at = state.camera_frame_updated_at.get(camera_id)
    if updated_at is None:
        return None
    return time.time() - updated_at


def _is_live_frame_fresh(camera_id: str) -> bool:
    frame_bytes = state.camera_frames.get(camera_id)
    age = _frame_age_seconds(camera_id)
    return frame_bytes is not None and age is not None and age <= state.CAMERA_FRAME_STALE_SECONDS


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
        if annotated is None:
            annotated = _draw_stream_detection_records(
                frame,
                detections,
                camera_id,
                show_overlay=False,
            )
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
        if annotated is None:
            annotated = _draw_stream_detection_records(
                frame,
                detections,
                camera_id,
                show_overlay=False,
            )
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
        "annotated_frame": annotated,
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
    annotated_frame: np.ndarray | None,
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
        snapshot_jpeg = None
        clean_snapshot_jpeg = None
        snapshot_encoding_attempted = False
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
            if not snapshot_encoding_attempted:
                snapshot_encoding_attempted = True
                try:
                    snapshot_jpeg, clean_snapshot_jpeg = _encode_inference_snapshot_pair(
                        camera_id,
                        frame,
                        detections,
                        int(current_cfg.get("global", {}).get("jpeg_quality", 70)),
                        annotated_frame=annotated_frame,
                    )
                except Exception:
                    logger.exception(
                        "Alert snapshot encoding failed",
                        extra={"camera_id": camera_id, "rule": candidate["rule"]},
                    )
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
                    snapshot_jpeg=snapshot_jpeg,
                    clean_snapshot_jpeg=clean_snapshot_jpeg,
                )
                if alert:
                    if decision.rule_id:
                        policy_engine.mark_rule_triggered(decision.rule_id, cfg=current_cfg)
                    logger.debug(
                        "Detection alert queued",
                        extra={"camera_id": camera_id, "rule": candidate["rule"]},
                    )

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
    last_plate_log_by_key: dict[str, float] = {}
    plate_vote_window: list[dict] = []
    active_violations: set[str] = set()
    violation_window: dict[str, list[bool]] = {}
    window_size = 15
    frame_counter = 0
    initial_schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
    initial_execution_plan = _scheduled_execution_plan(execution_plan, initial_schedule_state)
    missing_model_keys = model_manager.missing_model_keys(initial_execution_plan["required_model_keys"])
    reconnect_failures = 0
    safe_video_source = redact_video_source(video_source)
    state.clear_camera_connection_health(camera_id)
    connection_tracker = (
        CameraConnectionTracker(now=time.monotonic())
        if stream_type == "rtsp"
        else None
    )
    if connection_tracker is not None:
        _publish_camera_connection_health(camera_id, connection_tracker)
    if missing_model_keys:
        state.camera_runtime_status[camera_id] = "awaiting_model_install"
        _clear_camera_observation(camera_id)
        state.camera_detection_history[camera_id] = []
        state.camera_schedule_telemetry[camera_id] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduleState": initial_schedule_state,
            "modelInvocationCounts": {},
        }
        logger.warning("Camera waiting on missing models", extra={"camera_id": camera_id, "missing_models": missing_model_keys})
        return

    while not stop_event.is_set():
        state.camera_runtime_status[camera_id] = (
            "starting" if frame_counter == 0 and reconnect_failures == 0 else "reconnecting"
        )
        cap = _open_video_capture(video_source, stream_type)
        if connection_tracker is not None:
            connection_tracker.capture_backend = _capture_backend_name(cap)
            _publish_camera_connection_health(camera_id, connection_tracker)
        if not cap.isOpened():
            cap.release()
            if stop_event.is_set():
                return
            _clear_camera_observation(camera_id)
            active_violations.clear()
            violation_window.clear()
            last_annotated = None
            delay = reconnect_delay_seconds(reconnect_failures, camera_id) if stream_type == "rtsp" else 5.0
            reconnect_failures += 1
            state.camera_runtime_status[camera_id] = "reconnecting"
            _clear_live_frame(camera_id)
            if connection_tracker is not None:
                _record_rtsp_connection_failure(
                    camera_id,
                    connection_tracker,
                    safe_source=safe_video_source,
                    retry_seconds=delay,
                    received_frame=False,
                    now=time.monotonic(),
                )
            else:
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

        inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{camera_id}-inference")
        pending_inference = None
        next_inference_at = inference_scheduler.next_inference_slot(
            camera_id,
            get_config(),
            inference_interval,
        )
        last_stream_published_at = 0.0
        stream_had_subscribers = False
        last_annotated = None
        received_frame = False
        try:
            while cap.isOpened() and not stop_event.is_set():
                if not licensing.is_inference_allowed():
                    time.sleep(LICENSE_PAUSE_INTERVAL)
                    continue

                started = time.time()
                ok, frame = _read_live_frame(cap, stream_type)
                if not ok:
                    state.camera_runtime_status[camera_id] = "reconnecting" if stream_type == "rtsp" else "offline"
                    _clear_camera_observation(camera_id)
                    active_violations.clear()
                    violation_window.clear()
                    last_annotated = None
                    break
                if not received_frame:
                    received_frame = True
                frame_height, frame_width = frame.shape[:2]
                state.update_camera_frame_dimensions(camera_id, frame_width, frame_height)
                if connection_tracker is not None and _record_rtsp_connection_frame(
                    camera_id,
                    connection_tracker,
                    now=time.monotonic(),
                ):
                    reconnect_failures = 0

                frame_counter += 1
                current_cfg = get_config()
                current_g = current_cfg.get("global", {})
                current_cam = current_cfg["cameras"].get(camera_id, {})
                execution_plan = current_cam.get("execution_plan") or build_execution_plan(current_cam, current_cfg)
                schedule_state = _capability_schedule_state(current_cam, current_cfg, execution_plan)
                scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
                target_fps = _coerce_fps(current_cam.get("fps", current_g.get("target_fps", target_fps)), target_fps)
                frame_interval = 1.0 / target_fps
                current_inference_fps = _coerce_fps(
                    current_cam.get("inference_fps", current_g.get("inference_fps", max(1.0, target_fps / 3))),
                    max(1.0, target_fps / 3),
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
                yolo_conf = current_g.get("yolo_conf", yolo_conf)
                jpeg_quality = int(current_g.get("jpeg_quality", jpeg_quality))
                inference_width = int(current_g.get("inference_width", inference_width))
                device = current_g.get("device", device)
                alert_cooldown = int(current_g.get("alert_cooldown", alert_cooldown))
                state.camera_runtime_status[camera_id] = "running"

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
                            "annotated_frame": None,
                        }
                    pending_inference = None
                    detections = result["detections"]
                    last_annotated = result.get("annotated_frame")
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
                        result.get("annotated_frame"),
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

                stream_fps = _configured_stream_fps(current_cam, current_g, target_fps)
                stream_now = time.monotonic()
                publish_stream_frame, stream_had_subscribers = _stream_publication_due(
                    camera_id,
                    last_stream_published_at,
                    stream_now,
                    stream_fps,
                    stream_had_subscribers,
                )
                if publish_stream_frame:
                    try:
                        _publish_stream_frame(
                            camera_id,
                            frame,
                            state.camera_detections.get(camera_id, []),
                            jpeg_quality=jpeg_quality,
                            source_annotated=_preserved_source_annotation(
                                frame,
                                last_annotated,
                                execution_plan,
                            ),
                        )
                    except Exception:
                        logger.exception("Stream frame encoding failed", extra={"camera_id": camera_id})
                    else:
                        last_stream_published_at = stream_now
                now = time.monotonic()
                if pending_inference is None and now - next_inference_at > inference_interval / 2:
                    next_inference_at = inference_scheduler.next_inference_slot(
                        camera_id,
                        current_cfg,
                        inference_interval,
                        now=now,
                    )
                if pending_inference is None and now >= next_inference_at:
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
                    next_inference_at = inference_scheduler.next_inference_slot(
                        camera_id,
                        current_cfg,
                        inference_interval,
                        now=now + 1e-9,
                    )

                elapsed = time.time() - started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            _drain_inference_executor(inference_executor, pending_inference)

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
            _record_rtsp_connection_failure(
                camera_id,
                connection_tracker,
                safe_source=safe_video_source,
                retry_seconds=delay,
                received_frame=received_frame,
                now=time.monotonic(),
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
    with _camera_lifecycle_locks_guard:
        return _camera_lifecycle_locks.setdefault(cam_id, threading.RLock())


def begin_camera_lifecycle_shutdown() -> None:
    with _camera_lifecycle_fence_lock:
        _camera_lifecycle_shutdown.set()


def resume_camera_lifecycle() -> None:
    with _camera_lifecycle_fence_lock:
        _camera_lifecycle_shutdown.clear()


def camera_lifecycle_shutting_down() -> bool:
    return _camera_lifecycle_shutdown.is_set()


def _remove_worker_if_owned(
    workers: dict[str, tuple[threading.Thread, threading.Event]],
    cam_id: str,
    ownership: tuple[threading.Thread, threading.Event],
) -> None:
    with _camera_worker_registry_lock:
        if workers.get(cam_id) is ownership:
            del workers[cam_id]


def _deregister_worker_on_exit(
    workers: dict[str, tuple[threading.Thread, threading.Event]],
    cam_id: str,
    stop_event: threading.Event,
) -> bool:
    current_thread = threading.current_thread()
    with _camera_worker_registry_lock:
        ownership = workers.get(cam_id)
        if ownership is None:
            return True
        registered_thread, registered_stop_event = ownership
        if registered_thread is current_thread and registered_stop_event is stop_event:
            del workers[cam_id]
            return True
        return False


def _finalize_camera_worker_exit(cam_id: str, stop_event: threading.Event) -> None:
    current_thread = threading.current_thread()
    with _camera_worker_registry_lock:
        ownership = state.camera_threads.get(cam_id)
        if ownership is not None:
            registered_thread, registered_stop_event = ownership
            if registered_thread is not current_thread or registered_stop_event is not stop_event:
                return
            del state.camera_threads[cam_id]
        _clear_camera_observation(cam_id)
        state.clear_camera_connection_health(cam_id)
        state.camera_runtime_status[cam_id] = "offline" if stop_event.is_set() else "error"


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
            logger.debug("Camera start rejected during lifecycle shutdown", extra={"camera_id": cam_id})
            return False
        cfg = get_config()
        cam = cfg.get("cameras", {}).get(cam_id)
        if not cam or not cam.get("enabled", True):
            return False

        with _camera_worker_registry_lock:
            existing = state.camera_threads.get(cam_id)
        if existing is not None:
            existing_thread, existing_stop_event = existing
            if existing_thread.is_alive():
                if cam.get("demo") == "yolo+vlm" and not existing_stop_event.is_set():
                    start_vlm_for_camera(cam_id)
                return False
            _remove_worker_if_owned(state.camera_threads, cam_id, existing)

        execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
        schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
        scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
        missing_model_keys = model_manager.missing_model_keys(scheduled_plan["required_model_keys"])
        _clear_camera_observation(cam_id)
        state.camera_detection_history[cam_id] = []
        state.camera_frame_updated_at.pop(cam_id, None)
        state.camera_schedule_telemetry.pop(cam_id, None)
        if missing_model_keys:
            state.camera_runtime_status[cam_id] = "awaiting_model_install"
            logger.warning("Skipping camera start until models are ready", extra={"camera_id": cam_id, "missing_models": missing_model_keys})
            return False

        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(target=video_processor, args=(cam_id, stop_evt), daemon=True)
            ownership = (thread, stop_evt)
            with _camera_worker_registry_lock:
                state.camera_threads[cam_id] = ownership
            state.camera_worker_started_at[cam_id] = time.time()
            state.camera_runtime_status[cam_id] = "starting"
            try:
                thread.start()
            except Exception:
                stop_evt.set()
                _remove_worker_if_owned(state.camera_threads, cam_id, ownership)
                state.camera_runtime_status[cam_id] = "error"
                logger.exception("Failed to start video processor", extra={"camera_id": cam_id})
                raise
        logger.info("Started video processor", extra={"camera_id": cam_id})
        if cam.get("demo") == "yolo+vlm":
            start_vlm_for_camera(cam_id)
        return True


def start_vlm_for_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if _camera_lifecycle_shutdown.is_set():
            return False
        cfg = get_config_snapshot("cameras")
        cam = cfg.get("cameras", {}).get(cam_id)
        if not cam or not cam.get("enabled", True) or cam.get("demo") != "yolo+vlm":
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
            if existing[0].is_alive():
                return False
            _remove_worker_if_owned(state.vlm_threads, cam_id, existing)
        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(target=vlm_worker, args=(cam_id, stop_evt), daemon=True)
            ownership = (thread, stop_evt)
            with _camera_worker_registry_lock:
                state.vlm_threads[cam_id] = ownership
            try:
                thread.start()
            except Exception:
                stop_evt.set()
                _remove_worker_if_owned(state.vlm_threads, cam_id, ownership)
                logger.exception("Failed to start VLM worker", extra={"camera_id": cam_id})
                return False
        logger.info("Started VLM worker", extra={"camera_id": cam_id})
        return True


def stop_vlm_for_camera(cam_id: str) -> bool:
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
            return False
        _remove_worker_if_owned(state.vlm_threads, cam_id, ownership)
        return True


def heal_camera_workers_once() -> None:
    cfg = get_config_snapshot("cameras")
    with _camera_worker_registry_lock:
        registered_vlm_ids = set(state.vlm_threads)
    for cam_id in set(cfg.get("cameras", {})) | registered_vlm_ids:
        if _camera_lifecycle_shutdown.is_set():
            return
        try:
            with _camera_lifecycle_lock(cam_id):
                if _camera_lifecycle_shutdown.is_set():
                    return
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
                if vlm_expected and not _worker_registration_active(state.vlm_threads, cam_id):
                    start_vlm_for_camera(cam_id)
        except Exception:
            logger.exception("Camera lifecycle healing failed; later cameras will still be checked", extra={"camera_id": cam_id})


async def camera_worker_healing_loop(*, interval_seconds: float = CAMERA_WORKER_HEAL_INTERVAL_SECONDS) -> None:
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
            else:
                _remove_worker_if_owned(state.vlm_threads, cam_id, vlm_ownership)
        if not camera_stopped or not vlm_stopped:
            return False
        state.camera_frames.pop(cam_id, None)
        state.camera_clean_frames.pop(cam_id, None)
        state.camera_frame_updated_at.pop(cam_id, None)
        state.clear_camera_frame_dimensions(cam_id)
        state.camera_worker_started_at.pop(cam_id, None)
        state.camera_detections.pop(cam_id, None)
        state.camera_detection_history.pop(cam_id, None)
        state.camera_schedule_telemetry.pop(cam_id, None)
        _last_pose_results.pop(cam_id, None)
        state.clear_camera_connection_health(cam_id)
        state.camera_runtime_status[cam_id] = "offline"
        return True


def stop_all_camera_workers() -> bool:
    with _camera_worker_registry_lock:
        camera_ids = set(state.camera_threads) | set(state.vlm_threads)
    return all(stop_camera(cam_id) for cam_id in camera_ids)


def restart_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if not stop_camera(cam_id):
            return False
        return start_camera(cam_id)


def restart_all_cameras():
    cfg = get_config()
    configured = set(cfg.get("cameras", {}))
    with _camera_worker_registry_lock:
        existing = set(state.camera_threads) | set(state.vlm_threads)
    for cam_id in configured | existing:
        if cam_id in configured:
            restart_camera(cam_id)
        else:
            stop_camera(cam_id)
