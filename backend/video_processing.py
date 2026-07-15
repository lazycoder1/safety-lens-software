"""Rakshak Lens video processing — alerts, grouped inference, MJPEG streaming."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
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
import alert_timing
from adaptive_inference import (
    AdaptiveDecision,
    AdaptiveInferenceController,
    AdaptiveSignals,
)
import face_analyzer
import face_store
import helmet_colour
import plate_store
import inference_scheduler
import licensing
import model_manager
import object_lifecycle_analytics
import state
import notification_dispatcher
import policy_engine
from pipeline_telemetry import (
    TelemetryCapacityError,
    telemetry as pipeline_telemetry,
)
from rtdetr_phone_scheduler import (
    SUBSTITUTION_SCHEDULER as RTDETR_PHONE_SUBSTITUTION_SCHEDULER,
    PrimaryPersonTracker,
)
from ppe_substitution_scheduler import SCHEDULER as PPE_SUBSTITUTION_SCHEDULER
from person_crop_runtime import (
    PersonCropPlan,
    PersonCropPolicy,
    decide_crop_execution,
    plan_person_crops,
    remap_crop_detections,
)
from shared_inference_scheduler import (
    BatchProfile,
    InferenceWork,
    OfferStatus,
    SharedInferenceScheduler,
)
from vlm_enrichment import (
    VLMEnrichmentDispatcher,
    VLMEnrichmentWork,
    remote_vlm_endpoint_allowed,
)
from keyframe_tracker import (
    REDETECT_NO_KEYFRAME,
    REDETECT_NO_TRACKS,
    KeyframeTrackerResult,
    PersonKLTTracker,
)
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
from capability_registry import (
    CAPABILITY_REGISTRY,
    CLASS_TERM_TO_CAPABILITIES,
    RULE_ID_TO_CAPABILITY,
)
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
    has_ppe_specialist_context,
)
from mjpeg_fanout import stream_fanout

LICENSE_PAUSE_INTERVAL = 1.0
RTSP_BUFFER_DRAIN_MAX_FRAMES = 30
RTSP_BUFFER_DRAIN_BLOCK_SECONDS = 0.012

logger = logging.getLogger("rakshak_lens")

_shared_scheduler_lock = threading.Lock()
_shared_scheduler: SharedInferenceScheduler | None = None
_vlm_dispatcher_lock = threading.Lock()
_vlm_dispatcher: VLMEnrichmentDispatcher | None = None
_vlm_incident_lock = threading.Lock()
_vlm_incident_active: dict[str, str] = {}

# A VLM is advisory and runs outside the real-time pipeline.  Still bound the
# provider response before parsing it: ``requests.Response.json()`` eagerly
# buffers arbitrary content and a misbehaving remote endpoint must not be able
# to exhaust the Jetson's limited memory.
VLM_MAX_RESPONSE_BYTES = 256 * 1024
VLM_MAX_RESULT_CHARS = 8 * 1024


def _bounded_runtime_int(
    value: object,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded_runtime_float(
    value: object,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _shared_scheduler_enabled(cfg: dict) -> bool:
    configured = cfg.get("global", {}).get(
        "shared_inference_scheduler_mode",
        "off",
    )
    override = os.environ.get("SAFETYLENS_SHARED_INFERENCE_SCHEDULER_MODE")
    return (
        str(override if override is not None else configured).strip().lower()
        == "active"
    )


def _observe_shared_scheduler_drop(
    camera_id: str,
    reason: str,
    amount: int,
) -> None:
    counter_name = (
        "admissionStaleDropCount"
        if reason == "stale"
        else "schedulerLifecycleDropCount"
    )
    pipeline_telemetry.increment_camera_counter(
        camera_id,
        counter_name,
        amount,
    )


def _get_shared_scheduler(cfg: dict) -> SharedInferenceScheduler:
    global _shared_scheduler
    with _shared_scheduler_lock:
        if _shared_scheduler is None:
            global_cfg = cfg.get("global", {})
            _shared_scheduler = SharedInferenceScheduler(
                max_workers=_bounded_runtime_int(
                    global_cfg.get("shared_inference_max_workers", 4),
                    4,
                    minimum=4,
                    maximum=16,
                ),
                batch2_wait_seconds=0.006,
                singleton_wait_seconds=0.014,
                drop_observer=_observe_shared_scheduler_drop,
            )
            _shared_scheduler.start()
        return _shared_scheduler


def shutdown_pipeline_runtime(*, timeout: float = 10.0) -> bool:
    """Fence shared inference work after camera workers have stopped."""
    global _shared_scheduler
    with _shared_scheduler_lock:
        scheduler = _shared_scheduler
        _shared_scheduler = None
    scheduler_stopped = (
        True
        if scheduler is None
        else scheduler.stop(wait=True, timeout=max(0.0, timeout))
    )
    global _vlm_dispatcher
    with _vlm_dispatcher_lock:
        dispatcher = _vlm_dispatcher
        _vlm_dispatcher = None
    if dispatcher is not None:
        # Provider work is optional enrichment and deliberately never extends
        # the core camera/server shutdown critical path.
        dispatcher.shutdown(wait=False)
    return scheduler_stopped


def pipeline_runtime_stats() -> dict:
    with _shared_scheduler_lock:
        scheduler = _shared_scheduler
    with _vlm_dispatcher_lock:
        dispatcher = _vlm_dispatcher
    return {
        "sharedInferenceScheduler": (
            scheduler.stats()
            if scheduler is not None
            else {"running": False, "accepting": False}
        ),
        "vlmEnrichment": (
            dispatcher.stats()
            if dispatcher is not None
            else {"running": False, "accepting": False}
        ),
        "alertTiming": alert_timing.registry.stats(),
    }


_BATCH_PROFILE_PLAN_FLAGS = (
    "run_coco_primary",
    "run_rtdetr_phone",
    "run_ppe_specialist",
    "run_ppe_closed_set_candidate",
    "run_yoloe_long_tail",
    "run_fire_smoke_specialist",
    "run_pose_specialist",
    "run_face_recognition",
    "run_plate_recognition",
)


def _inference_batch_profile(
    camera_id: str,
    runtime_plan: dict,
    current_cfg: dict,
    *,
    yolo_conf: float,
    device: str,
    inference_width: int,
) -> BatchProfile:
    """Build a conservative compatibility identity without retaining secrets."""
    global_cfg = current_cfg.get("global", {})
    request_signatures: list[tuple] = []
    if runtime_plan.get("run_rtdetr_phone"):
        request_signatures.append(
            (
                "rtdetr_phone",
                round(
                    _rule_confidence_for_capability(
                        camera_id,
                        "person_presence",
                        yolo_conf,
                        cfg=current_cfg,
                    ),
                    6,
                ),
                round(
                    _rule_confidence_for_capability(
                        camera_id,
                        "mobile_phone",
                        min(yolo_conf, 0.15),
                        cfg=current_cfg,
                    ),
                    6,
                ),
                640,
            )
        )
    if runtime_plan.get("run_coco_primary"):
        request_signatures.append(
            (
                "coco_primary",
                round(
                    _rule_confidence_for_model_family(
                        camera_id,
                        "coco_primary",
                        yolo_conf,
                        cfg=current_cfg,
                    ),
                    6,
                ),
                _coco_inference_imgsz(
                    current_cfg,
                    inference_width,
                    runtime_plan,
                ),
            )
        )
    if runtime_plan.get("run_ppe_specialist"):
        request_signatures.append(
            (
                "ppe_specialist",
                round(
                    _rule_confidence_for_model_family(
                        camera_id,
                        "ppe_specialist",
                        yolo_conf,
                        cfg=current_cfg,
                    ),
                    6,
                ),
                _ppe_inference_imgsz(current_cfg, inference_width),
                tuple(str(value) for value in runtime_plan.get("ppe_prompt_terms", [])),
            )
        )
    if runtime_plan.get("run_ppe_closed_set_candidate"):
        capabilities = _capabilities_for_model_key(
            runtime_plan,
            "ppe_closed_set_candidate",
        )
        request_signatures.append(
            (
                "ppe_closed_set_candidate",
                round(
                    _rule_confidence_for_capabilities(
                        camera_id,
                        capabilities,
                        yolo_conf,
                        cfg=current_cfg,
                    ),
                    6,
                ),
                int(inference_width),
            )
        )
    if runtime_plan.get("run_yoloe_long_tail"):
        request_signatures.append(
            (
                "yoloe_long_tail",
                round(float(yolo_conf), 6),
                int(inference_width),
                tuple(
                    str(value) for value in runtime_plan.get("yoloe_prompt_terms", [])
                ),
            )
        )
    if runtime_plan.get("run_fire_smoke_specialist"):
        request_signatures.append(
            (
                "fire_smoke_specialist",
                round(
                    _rule_confidence_for_capability(
                        camera_id,
                        "fire_smoke",
                        yolo_conf,
                        cfg=current_cfg,
                    ),
                    6,
                ),
                int(inference_width),
            )
        )
    if runtime_plan.get("run_pose_specialist"):
        request_signatures.append(
            ("pose_specialist", round(float(yolo_conf), 6), int(inference_width))
        )
    key = (
        "pipeline-v2",
        tuple(
            (name, bool(runtime_plan.get(name))) for name in _BATCH_PROFILE_PLAN_FLAGS
        ),
        tuple(request_signatures),
        str(device),
    )
    configured_max = _bounded_runtime_int(
        global_cfg.get("shared_inference_max_batch_size", 4),
        4,
        minimum=1,
        maximum=4,
    )
    if runtime_plan.get("run_face_recognition") or runtime_plan.get(
        "run_plate_recognition"
    ):
        return BatchProfile(key=key, max_batch_size=1)

    crop_path_enabled = any(
        runtime_plan.get(_PERSON_CROP_TARGET_FLAGS[target])
        and _configured_person_crop_mode(
            current_cfg,
            f"{target}_person_crop_mode",
            f"SAFETYLENS_{target.upper()}_PERSON_CROP_MODE",
        )
        != "off"
        for target in _PERSON_CROP_TARGET_FLAGS
    )
    if crop_path_enabled:
        # Crop counts vary by frame. A scheduler cohort would therefore be a
        # claim about a fixed model batch that the downstream crop loop cannot
        # guarantee.
        return BatchProfile(key=key, max_batch_size=1)

    grouped_model_keys: list[str] = []
    if runtime_plan.get("run_coco_primary"):
        grouped_model_keys.append("coco_primary")
    if runtime_plan.get("run_ppe_specialist") and runtime_plan.get(
        "ppe_prompt_terms"
    ):
        grouped_model_keys.append("ppe_specialist")
    unsupported_grouped_route = any(
        runtime_plan.get(flag)
        for flag in (
            "run_ppe_closed_set_candidate",
            "run_yoloe_long_tail",
            "run_fire_smoke_specialist",
            "run_pose_specialist",
        )
    )

    def fixed_routes_ready(batch_size: int) -> bool:
        if unsupported_grouped_route:
            return False
        if runtime_plan.get("run_rtdetr_phone") and (
            batch_size != 2
            or not model_manager.remote_rtdetr_phone_batch_route_may_run(batch_size)
        ):
            return False
        if grouped_model_keys and not model_manager.remote_frame_batch_route_may_run(
            grouped_model_keys,
            batch_size,
        ):
            return False
        return bool(grouped_model_keys or runtime_plan.get("run_rtdetr_phone"))

    max_batch_size = 1
    if configured_max >= 4 and fixed_routes_ready(4):
        max_batch_size = 4
    elif configured_max >= 2 and fixed_routes_ready(2):
        max_batch_size = 2
    return BatchProfile(key=key, max_batch_size=max_batch_size)


# Compatibility state retained for lifecycle cleanup and older integrations;
# the feature-rich pipeline passes fresh pose results explicitly.
_last_pose_results: dict[str, object] = {}


def rtdetr_phone_substitution_stats() -> dict:
    return RTDETR_PHONE_SUBSTITUTION_SCHEDULER.stats()


def ppe_substitution_stats() -> dict:
    return PPE_SUBSTITUTION_SCHEDULER.stats()


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
_MOBILE_PHONE_PROBE_REASON = "mobile_phone_small_object_recall"
_MOBILE_PHONE_PROBE_CONTEXT_SUPPRESSION_REASON = "awaiting_primary_person_context"
_PPE_PHONE_PROBE_DEFERRAL_REASON = "deferred_for_mobile_phone_probe"
_RTDETR_PHONE_PROBE_REASON = "rtdetr_phone_track_recall"
_RTDETR_PHONE_CROP_REASON = "rtdetr_phone_person_crop"
_RTDETR_PHONE_FALLBACK_REASON = "rtdetr_phone_route_fallback"
_PPE_CADENCE_SUPPRESSION_REASON = "ppe_specialist_cadence"
_PPE_SUBSTITUTION_REASON = "ppe_specialist_track_substitution"

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
EMPTY_SCENE_SIGNATURE_SIZE = (64, 36)
EMPTY_SCENE_CHANGED_PIXEL_DELTA = 8
EMPTY_SCENE_CHANGED_FRACTION = 0.001
EMPTY_SCENE_MAX_INFERENCE_INTERVAL_SECONDS = 1.0
UNCHANGED_STREAM_MAX_INTERVAL_SECONDS = 1.0
STREAM_CLEAN_CACHE_MAX_AGE_SECONDS = 1.0


def _positive_fps(value, fallback: float) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return fallback
    return fps if fps > 0 else fallback


def _configured_stream_fps(
    camera: dict, global_config: dict, target_fps: float
) -> float:
    target = _positive_fps(target_fps, 1.0)
    default = min(target, 4.0)
    configured = camera.get("stream_fps", global_config.get("stream_fps", default))
    return min(target, _positive_fps(configured, default))


def _stream_publish_due(
    last_published_at: float, now: float, stream_fps: float
) -> bool:
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
    due = (has_subscribers and not had_subscribers) or _stream_publish_due(
        last_published_at, now, effective_fps
    )
    return due, has_subscribers


def _motion_adaptive_inference_decision(
    frame: np.ndarray,
    previous_signature: np.ndarray | None,
    *,
    last_submitted_at: float | None,
    now: float,
    alert_confirmation_required: bool,
) -> tuple[bool, np.ndarray | None, float]:
    """Skip unchanged frames without slowing alert confirmation or clearing."""
    if alert_confirmation_required:
        return True, previous_signature, 1.0
    signature = _frame_change_signature(frame)
    changed_fraction = _signature_changed_fraction(signature, previous_signature)
    if previous_signature is None:
        return True, signature, changed_fraction
    if (
        last_submitted_at is None
        or now - last_submitted_at >= EMPTY_SCENE_MAX_INFERENCE_INTERVAL_SECONDS
    ):
        return True, signature, changed_fraction
    return changed_fraction > EMPTY_SCENE_CHANGED_FRACTION, signature, changed_fraction


def _runtime_mode(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in allowed else default


def _configured_global_runtime_mode(
    cfg: dict | None,
    config_key: str,
    environment_key: str,
    *,
    allowed: set[str],
    default: str = "off",
) -> str:
    global_cfg = cfg.get("global", {}) if isinstance(cfg, dict) else {}
    configured = global_cfg.get(config_key, default)
    override = os.environ.get(environment_key)
    return _runtime_mode(
        override if override is not None else configured,
        allowed,
        default,
    )


def _record_adaptive_runtime_state(
    camera_id: str,
    decision,
    tracker_result,
) -> None:
    if decision.mode != "off":
        observation_counter = {
            "quiet": "adaptiveQuietObservationCount",
            "uncertain": "adaptiveUncertainObservationCount",
            "active": "adaptiveActiveObservationCount",
        }.get(decision.state)
        if observation_counter is not None:
            pipeline_telemetry.increment_camera_counter(
                camera_id,
                observation_counter,
            )
    if tracker_result.projections:
        pipeline_telemetry.increment_camera_counter(
            camera_id,
            "trackerProjectionFrameCount",
        )
        pipeline_telemetry.increment_camera_counter(
            camera_id,
            "trackerProjectedPersonCount",
            len(tracker_result.projections),
        )
    if tracker_result.force_redetect and bool(
        set(tracker_result.reasons) - {REDETECT_NO_KEYFRAME, REDETECT_NO_TRACKS}
    ):
        pipeline_telemetry.increment_camera_counter(
            camera_id,
            "trackerForceRedetectSignalCount",
        )
    current = dict(state.camera_schedule_telemetry.get(camera_id, {}))
    current["adaptiveInference"] = {
        "mode": decision.mode,
        "state": decision.state,
        "targetFps": round(decision.target_fps, 3),
        "baselineDue": decision.baseline_due,
        "adaptiveDue": decision.adaptive_due,
        "urgentReasons": list(decision.urgent_reasons),
    }
    current["keyframeTracker"] = {
        "projectionCount": len(tracker_result.projections),
        "aggregateConfidence": round(tracker_result.aggregate_confidence, 4),
        "forceRedetect": tracker_result.force_redetect,
        "reasons": list(tracker_result.reasons),
    }
    state.camera_schedule_telemetry[camera_id] = current


def _record_adaptive_inference_admission(camera_id: str, decision) -> None:
    """Count accepted detector work by the camera's adaptive state."""
    if decision.mode == "off":
        return
    admission_counter = {
        "quiet": "adaptiveQuietAdmissionCount",
        "uncertain": "adaptiveUncertainAdmissionCount",
        "active": "adaptiveActiveAdmissionCount",
    }.get(decision.state)
    if admission_counter is not None:
        pipeline_telemetry.increment_camera_counter(camera_id, admission_counter)


def _record_adaptive_inference_completion(
    camera_id: str,
    mode: object,
    adaptive_state: object,
) -> None:
    """Count successful inference by the state captured when work was offered."""
    if mode == "off":
        return
    completion_counter = {
        "quiet": "adaptiveQuietInferenceCount",
        "uncertain": "adaptiveUncertainInferenceCount",
        "active": "adaptiveActiveInferenceCount",
    }.get(adaptive_state)
    if completion_counter is not None:
        pipeline_telemetry.increment_camera_counter(camera_id, completion_counter)


def _record_person_crop_runtime_counters(
    camera_id: str,
    person_crop_telemetry: object,
) -> None:
    """Turn one completed inference job's bounded crop summary into counters."""
    if not isinstance(person_crop_telemetry, Mapping):
        return
    for target, attempt_counter in (
        ("ppe", "ppePersonCropAttemptCount"),
        ("phone", "phonePersonCropAttemptCount"),
    ):
        target_telemetry = person_crop_telemetry.get(target)
        if not isinstance(target_telemetry, Mapping):
            continue
        attempts = target_telemetry.get("cropInferenceAttempts", 0)
        if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts > 0:
            pipeline_telemetry.increment_camera_counter(
                camera_id,
                attempt_counter,
                attempts,
            )
        if target_telemetry.get("fallbackRequired") is True:
            pipeline_telemetry.increment_camera_counter(
                camera_id,
                "personCropFallbackCount",
            )
        full_frame_invocations = target_telemetry.get("fullFrameInvocations", 0)
        if (
            isinstance(full_frame_invocations, int)
            and not isinstance(full_frame_invocations, bool)
            and full_frame_invocations > 0
        ):
            pipeline_telemetry.increment_camera_counter(
                camera_id,
                "personCropFullFrameInvocationCount",
                full_frame_invocations,
            )


def _clear_adaptive_runtime_state(camera_id: str) -> None:
    current = state.camera_schedule_telemetry.get(camera_id)
    if not isinstance(current, dict) or not (
        "adaptiveInference" in current or "keyframeTracker" in current
    ):
        return
    updated = dict(current)
    updated.pop("adaptiveInference", None)
    updated.pop("keyframeTracker", None)
    if updated:
        state.camera_schedule_telemetry[camera_id] = updated
    else:
        state.camera_schedule_telemetry.pop(camera_id, None)


def _frame_change_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(
        gray,
        EMPTY_SCENE_SIGNATURE_SIZE,
        interpolation=cv2.INTER_AREA,
    )


def _signature_changed_fraction(
    signature: np.ndarray,
    previous_signature: np.ndarray | None,
) -> float:
    if previous_signature is None:
        return 1.0
    return float(
        (
            cv2.absdiff(signature, previous_signature) > EMPTY_SCENE_CHANGED_PIXEL_DELTA
        ).mean()
    )


def _stream_change_signature(frame: np.ndarray) -> np.ndarray:
    """Reuse the cheaper grayscale-first inference signature for streams."""
    return _frame_change_signature(frame)


def _active_stream_change_decision(
    frame: np.ndarray,
    previous_published_signature: np.ndarray | None,
    *,
    last_published_at: float,
    now: float,
    detections: list[dict],
    subscriber_joined: bool,
) -> tuple[bool, np.ndarray | None, float]:
    """Suppress only unchanged, empty active-view frames between heartbeats."""
    if detections:
        return True, None, 1.0
    signature = _stream_change_signature(frame)
    changed_fraction = _signature_changed_fraction(
        signature,
        previous_published_signature,
    )
    if (
        subscriber_joined
        or previous_published_signature is None
        or last_published_at <= 0.0
        or now - last_published_at >= UNCHANGED_STREAM_MAX_INTERVAL_SECONDS
    ):
        return True, signature, changed_fraction
    return changed_fraction > EMPTY_SCENE_CHANGED_FRACTION, signature, changed_fraction


def _resize_for_stream(
    frame: np.ndarray, max_width: int = STREAM_MAX_WIDTH
) -> np.ndarray:
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
    """Show COCO records plus confident helmet-colour evidence in live video."""
    return [
        detection
        for detection in detections
        if (
            detection.get("model_family") in {None, "coco_primary"}
            or detection.get("inference_scope") == "person_crop"
            or detection.get("helmet_colour") not in {None, "unknown"}
        )
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
    if execution_plan.get("run_face_recognition") or execution_plan.get(
        "run_pose_specialist"
    ):
        return last_annotated
    return None


def _publish_stream_frame(
    camera_id: str,
    frame: np.ndarray,
    detections: list[dict],
    *,
    jpeg_quality: int,
    source_annotated: np.ndarray | None = None,
    annotation_required: bool = True,
    cached_clean_jpeg: bytes | None = None,
) -> tuple[bytes, bool]:
    clean_jpeg_encoded = True
    if not annotation_required:
        clean_view = _resize_for_stream(frame)
        clean_jpeg = _encode_stream_jpeg(clean_view, jpeg_quality)
        annotated_jpeg = clean_jpeg
    elif source_annotated is None:
        annotated_view, clean_view = _render_stream_views(camera_id, frame, detections)
        annotated_jpeg = _encode_stream_jpeg(annotated_view, jpeg_quality)
        if cached_clean_jpeg is None:
            clean_jpeg = _encode_stream_jpeg(clean_view, jpeg_quality)
        else:
            clean_jpeg = cached_clean_jpeg
            clean_jpeg_encoded = False
    else:
        clean_view = _resize_for_stream(frame)
        clean_height, clean_width = clean_view.shape[:2]
        if source_annotated.shape[:2] == clean_view.shape[:2]:
            annotated_view = source_annotated
        else:
            annotated_view = cv2.resize(source_annotated, (clean_width, clean_height))
        annotated_jpeg = _encode_stream_jpeg(annotated_view, jpeg_quality)
        if cached_clean_jpeg is None:
            clean_jpeg = _encode_stream_jpeg(clean_view, jpeg_quality)
        else:
            clean_jpeg = cached_clean_jpeg
            clean_jpeg_encoded = False
    state.camera_frames[camera_id] = annotated_jpeg
    state.camera_clean_frames[camera_id] = clean_jpeg
    state.camera_frame_updated_at[camera_id] = time.time()
    stream_fanout.publish(camera_id, annotated_jpeg)
    return clean_jpeg, clean_jpeg_encoded


def _stream_clean_cache_due(
    cached_clean_jpeg: bytes | None,
    last_encoded_at: float,
    now: float,
) -> bool:
    return (
        cached_clean_jpeg is None
        or last_encoded_at <= 0.0
        or now - last_encoded_at >= STREAM_CLEAN_CACHE_MAX_AGE_SECONDS
    )


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
        if status in {
            notification_dispatcher.DELIVERED,
            notification_dispatcher.SIMULATED,
        }:
            delivered.append(output_id)
        elif status == notification_dispatcher.FAILED:
            retry.append(output_id)
        else:
            terminal.append(output_id)
    terminal.extend(
        output_id for output_id in direct_output_ids if output_id not in classified
    )
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


def _observe_persisted_alert_timing(alert: Mapping) -> None:
    alert_id = alert.get("id")
    if not alert_id:
        return
    persisted_ns = time.monotonic_ns()
    try:
        transition = alert_timing.registry.mark_persisted(
            str(alert_id),
            persisted_ns=persisted_ns,
        )
    except ValueError:
        # The histogram records invalid monotonic order explicitly. The
        # registry context cannot safely remain eligible for provider timing.
        timing = alert_timing.registry.pop(str(alert_id))
        if timing is not None:
            pipeline_telemetry.observe_alert_elapsed_ns(
                "confirmedToPersistedMs",
                timing.confirmed_ns,
                persisted_ns,
            )
            pipeline_telemetry.observe_alert_elapsed_ns(
                "firstPositiveToPersistedMs",
                timing.first_positive_ns,
                persisted_ns,
            )
        pipeline_telemetry.record_alert_persistence_censored()
        return
    if transition is None:
        pipeline_telemetry.record_alert_persistence_censored()
        return
    if not transition.newly_persisted:
        return
    timing = transition.context
    pipeline_telemetry.observe_alert_elapsed_ns(
        "firstPositiveToConfirmedMs",
        timing.first_positive_ns,
        timing.confirmed_ns,
    )
    pipeline_telemetry.observe_alert_elapsed_ns(
        "confirmedToPersistedMs",
        timing.confirmed_ns,
        persisted_ns,
    )
    pipeline_telemetry.observe_alert_elapsed_ns(
        "firstPositiveToPersistedMs",
        timing.first_positive_ns,
        persisted_ns,
    )
    pipeline_telemetry.register_alert_delivery_targets(
        len(timing.initial_target_keys)
    )
    deferred_outcomes = alert_timing.registry.activate_delivery_tracking(
        str(alert_id)
    )
    for deferred in deferred_outcomes:
        tracked = pipeline_telemetry.record_alert_delivery_outcome(
            deferred.outcome,
            tracked=True,
        )
        if tracked and deferred.outcome == "delivered":
            pipeline_telemetry.observe_alert_elapsed_ns(
                "firstPositiveToProviderSuccessMs",
                timing.first_positive_ns,
                deferred.completed_ns,
            )


def _broadcast_persisted_alert(alert: dict) -> None:
    _observe_persisted_alert_timing(alert)
    if _alert_event_loop is None or not _alert_event_loop.is_running():
        logger.warning(
            "Alert persisted without an active websocket event loop",
            extra={"alert_id": alert.get("id")},
        )
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
                    persist_queue_size=int(
                        os.getenv("ALERT_PERSIST_QUEUE_SIZE", "256")
                    ),
                    delivery_queue_size=int(
                        os.getenv("ALERT_DELIVERY_QUEUE_SIZE", "256")
                    ),
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
        logger.info(
            "Backfilled durable escalation work", extra={"delivery_count": backfilled}
        )
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
        alert_timing.registry.clear()
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
    runtime_probe_reason: str | None = None,
    runtime_probe_suppression_reason: str | None = None,
    runtime_deferred_model_keys: list[str] | None = None,
    person_crop_telemetry: dict | None = None,
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
    if person_crop_telemetry:
        sample["personCropTelemetry"] = deepcopy(person_crop_telemetry)
    if runtime_probe_reason:
        sample["runtimeProbeReason"] = runtime_probe_reason
    if runtime_probe_suppression_reason:
        sample["runtimeProbeSuppressionReason"] = runtime_probe_suppression_reason
    if runtime_deferred_model_keys:
        sample["runtimeDeferredModelKeys"] = list(runtime_deferred_model_keys)
    history = state.camera_detection_history.setdefault(camera_id, [])
    history.append(sample)
    if len(history) > DETECTION_HISTORY_LIMIT:
        del history[:-DETECTION_HISTORY_LIMIT]
    if (
        schedule_state is not None
        or model_invocations is not None
        or person_crop_telemetry
    ):
        previous_telemetry = state.camera_schedule_telemetry.get(camera_id, {})
        telemetry = {
            "timestamp": sample["timestamp"],
            "scheduleState": schedule_state or {},
            "modelInvocationCounts": model_invocations or {},
        }
        if person_crop_telemetry:
            telemetry["personCropTelemetry"] = deepcopy(person_crop_telemetry)
        if runtime_deferred_model_keys:
            telemetry["runtimeDeferredModelKeys"] = list(runtime_deferred_model_keys)
        phone_probe = dict(previous_telemetry.get("phoneProbe") or {})
        if runtime_probe_reason == _MOBILE_PHONE_PROBE_REASON:
            phone_detections = class_counts.get("cell phone", 0)
            phone_probe.update(
                probeCount=int(phone_probe.get("probeCount") or 0) + 1,
                hitProbeCount=int(phone_probe.get("hitProbeCount") or 0),
                lastProbeAt=sample["timestamp"],
                lastProbePhoneDetections=phone_detections,
            )
            if phone_detections:
                phone_probe["hitProbeCount"] = (
                    int(phone_probe.get("hitProbeCount") or 0) + 1
                )
                phone_probe["lastHitAt"] = sample["timestamp"]
        if (
            runtime_probe_suppression_reason
            == _MOBILE_PHONE_PROBE_CONTEXT_SUPPRESSION_REASON
        ):
            phone_probe.update(
                contextSuppressedCount=int(
                    phone_probe.get("contextSuppressedCount") or 0
                )
                + 1,
                lastContextSuppressedAt=sample["timestamp"],
                lastContextSuppressionReason=runtime_probe_suppression_reason,
            )
        if phone_probe:
            telemetry["phoneProbe"] = phone_probe
        state.camera_schedule_telemetry[camera_id] = telemetry


def _rule_observation_is_fresh(
    rule_key: str,
    *,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
    fresh_ppe_evaluated: bool,
    fresh_detection_rule_keys: set[str] | None,
) -> bool:
    if rule_key == "Fall Detected":
        return fresh_fall_evaluated
    if not fresh_detection_evaluated:
        return False
    if _is_ppe_violation_rule(rule_key) and not fresh_ppe_evaluated:
        return False
    return not (
        fresh_detection_rule_keys is not None
        and not _is_ppe_violation_rule(rule_key)
        and rule_key not in fresh_detection_rule_keys
    )


def _advance_violation_window(
    violation_window: dict[str, list[bool]],
    current_violation_rules: dict[str, dict],
    *,
    window_size: int,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
    fresh_ppe_evaluated: bool = True,
    fresh_detection_rule_keys: set[str] | None = None,
) -> None:
    """Advance confirmation windows only when the source model produced a fresh observation."""
    all_tracked = set(violation_window) | set(current_violation_rules)
    for rule_key in all_tracked:
        observed = rule_key in current_violation_rules
        if not _rule_observation_is_fresh(
            rule_key,
            fresh_detection_evaluated=fresh_detection_evaluated,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
        ):
            continue
        violation_window.setdefault(rule_key, []).append(observed)
        if len(violation_window[rule_key]) > window_size:
            violation_window[rule_key] = violation_window[rule_key][-window_size:]


def _record_empty_violation_observation(
    violation_window: dict[str, list[bool]],
    active_violations: set[str],
    *,
    window_size: int,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
    fresh_ppe_evaluated: bool = True,
    fresh_detection_rule_keys: set[str] | None = None,
) -> None:
    for rule_key in list(violation_window):
        if not _rule_observation_is_fresh(
            rule_key,
            fresh_detection_evaluated=fresh_detection_evaluated,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
        ):
            continue
        violation_window[rule_key].append(False)
        if len(violation_window[rule_key]) > window_size:
            violation_window[rule_key] = violation_window[rule_key][-window_size:]
        if sum(violation_window[rule_key]) == 0:
            violation_window.pop(rule_key, None)
            active_violations.discard(rule_key)


def _alert_confirmation_required(
    active_violations: set[str],
    violation_window: dict[str, list[bool]],
) -> bool:
    """Keep inference full-rate while an incident is confirming or clearing."""
    return bool(active_violations) or any(
        any(observations) for observations in violation_window.values()
    )


def _normalize_detection_batch(detections: list[dict], model_family: str) -> list[dict]:
    normalized: list[dict] = []
    for detection in detections:
        class_name = detection["class"]
        capability_keys = _COCO_CLASS_TO_CAPABILITIES.get(class_name, [])
        if not capability_keys:
            capability_keys = CLASS_TERM_TO_CAPABILITIES.get(
                _normalize_text(class_name),
                [],
            )
        normalized.append(
            {
                **detection,
                "model_family": model_family,
                "capability_keys": capability_keys,
            }
        )
    return normalized


def _rewrite_detection_record_classes(
    records: list[dict], class_names: dict[int, str]
) -> list[dict]:
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


def _rule_confidence_for_capability(
    camera_id: str,
    capability_key: str,
    default_conf: float,
    *,
    cfg: dict | None = None,
) -> float:
    cfg = cfg if cfg is not None else get_config()
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
        override = _valid_confidence(
            _camera_rule_override_value(camera, rule_id, "confidence", "conf")
        )
        if override is not None:
            confidences.append(override)
            continue
        confidence = _valid_confidence(rule.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
    return min(confidences) if confidences else default_conf


def _rule_confidence_for_model_family(
    camera_id: str,
    model_family: str,
    default_conf: float,
    *,
    cfg: dict | None = None,
) -> float:
    cfg = cfg if cfg is not None else get_config()
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
        override = _valid_confidence(
            _camera_rule_override_value(camera, rule_id, "confidence", "conf")
        )
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
        overrides = {}
    return [
        capability
        for capability in execution_plan.get("capabilities") or []
        if model_key in required_model_keys_for_capabilities([capability], overrides)
    ]


def _rule_confidence_for_capabilities(
    camera_id: str,
    capabilities: list[str],
    default_conf: float,
    *,
    cfg: dict | None = None,
) -> float:
    confidences = [
        _rule_confidence_for_capability(
            camera_id,
            capability,
            default_conf,
            cfg=cfg,
        )
        for capability in capabilities
    ]
    return min(confidences) if confidences else default_conf


def _coco_record_confidence_threshold(
    camera_id: str,
    record: dict,
    default_conf: float,
    *,
    cfg: dict | None = None,
) -> float:
    try:
        class_id = int(record.get("class_id", record.get("cls", -1)))
    except (TypeError, ValueError):
        return default_conf
    class_name = COCO_NAMES.get(class_id)
    capability_keys = _COCO_CLASS_TO_CAPABILITIES.get(class_name or "", [])
    if not capability_keys:
        return default_conf
    return min(
        _rule_confidence_for_capability(
            camera_id,
            capability_key,
            default_conf,
            cfg=cfg,
        )
        for capability_key in capability_keys
    )


def _filter_coco_records_for_rule_confidence(
    camera_id: str,
    records: list[dict],
    default_conf: float,
    *,
    cfg: dict | None = None,
) -> list[dict]:
    thresholds_by_class: dict[int, float] = {}
    filtered = []
    for record in records:
        try:
            class_id = int(record.get("class_id", record.get("cls", -1)))
            confidence = float(record.get("confidence") or 0)
        except (TypeError, ValueError):
            continue
        threshold = thresholds_by_class.get(class_id)
        if threshold is None:
            threshold = _coco_record_confidence_threshold(
                camera_id,
                record,
                default_conf,
                cfg=cfg,
            )
            thresholds_by_class[class_id] = threshold
        if confidence >= threshold:
            filtered.append(record)
    return filtered


def _configured_model_inference_imgsz(
    cfg: dict | None,
    setting: str,
    default_imgsz: int,
) -> int:
    """Resolve an optional model-specific width from the global settings."""
    if not isinstance(cfg, dict):
        return default_imgsz
    global_config = cfg.get("global")
    if not isinstance(global_config, dict):
        return default_imgsz
    configured = global_config.get(setting)
    if type(configured) is not int or not 160 <= configured <= 1920:
        return default_imgsz
    return configured


def _coco_inference_imgsz(
    cfg: dict | None,
    default_imgsz: int,
    execution_plan: dict | None = None,
) -> int:
    if isinstance(execution_plan, dict):
        override = execution_plan.get("coco_inference_width_override")
        if type(override) is int and 160 <= override <= 1920:
            return override
    return _configured_model_inference_imgsz(
        cfg,
        "coco_inference_width",
        default_imgsz,
    )


def _ppe_inference_imgsz(cfg: dict | None, default_imgsz: int) -> int:
    return _configured_model_inference_imgsz(
        cfg,
        "ppe_inference_width",
        default_imgsz,
    )


def _model_keys_for_capabilities(
    capabilities: list[str], capability_model_overrides: dict | None = None
) -> list[str]:
    capability_set = {
        capability for capability in capabilities if capability in CAPABILITY_REGISTRY
    }
    overrides = (
        capability_model_overrides
        if isinstance(capability_model_overrides, dict)
        else None
    )
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
        matches = (
            False
            if configured_active is False
            else policy_engine._schedule_matches(window, now, cfg)
        )
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

    suppressed = sorted(
        capability for capability, item in states.items() if item.get("suppressed")
    )
    return {
        "timestamp": now.isoformat(),
        "capabilities": states,
        "suppressedCapabilities": suppressed,
        "suppressedCount": len(suppressed),
    }


def _filter_prompt_terms_for_schedule(
    terms: list[str],
    suppressed_capabilities: set[str],
    configured_capabilities: set[str] | None = None,
) -> list[str]:
    filtered: list[str] = []
    for term in terms:
        mapped_capabilities = set(
            CLASS_TERM_TO_CAPABILITIES.get(_normalize_text(term), [])
        )
        relevant_capabilities = (
            mapped_capabilities & configured_capabilities
            if configured_capabilities is not None
            else mapped_capabilities
        )
        if relevant_capabilities and relevant_capabilities <= suppressed_capabilities:
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
    scheduled["run_ppe_closed_set_candidate"] = (
        "ppe_closed_set_candidate" in required_model_keys
    )
    scheduled["run_yoloe_long_tail"] = "yoloe_long_tail" in required_model_keys
    scheduled["run_fire_smoke_specialist"] = (
        "fire_smoke_specialist" in required_model_keys
    )
    scheduled["run_face_recognition"] = "face_recognition" in required_model_keys
    scheduled["run_pose_specialist"] = "pose_specialist" in required_model_keys
    scheduled["run_plate_recognition"] = "plate_recognition" in required_model_keys
    scheduled["ppe_prompt_terms"] = (
        _filter_prompt_terms_for_schedule(
            scheduled.get("ppe_prompt_terms", []),
            suppressed_capabilities,
            set(execution_plan.get("capabilities") or []),
        )
        if scheduled["run_ppe_specialist"]
        else []
    )
    scheduled["yoloe_prompt_terms"] = (
        _filter_prompt_terms_for_schedule(
            scheduled.get("yoloe_prompt_terms", []),
            suppressed_capabilities,
            set(execution_plan.get("capabilities") or []),
        )
        if scheduled["run_yoloe_long_tail"]
        else []
    )
    scheduled["schedule_state"] = schedule_state
    return scheduled


def _context_gated_execution_plan(
    execution_plan: dict,
    previous_detections: list[dict],
    *,
    camera: dict | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> dict:
    """Skip PPE work until COCO context can produce an actionable result."""
    if not (
        execution_plan.get("run_coco_primary")
        and execution_plan.get("run_ppe_specialist")
    ):
        return execution_plan
    ppe_capabilities = set(
        _capabilities_for_model_key(execution_plan, "ppe_specialist")
    )
    if not ppe_capabilities:
        return execution_plan
    has_required_context = has_ppe_specialist_context(
        previous_detections,
        ppe_capabilities,
        camera or {},
        frame_w,
        frame_h,
    )
    if has_required_context:
        return execution_plan

    gated = deepcopy(execution_plan)
    gated["run_ppe_specialist"] = False
    gated["ppe_prompt_terms"] = []
    gated["required_model_keys"] = [
        model_key
        for model_key in gated.get("required_model_keys", [])
        if model_key != "ppe_specialist"
    ]
    gated["runtime_suppressed_model_keys"] = ["ppe_specialist"]
    gated["runtime_suppression_reason"] = (
        "awaiting_rider_vehicle_context"
        if ppe_capabilities == {"rider_helmet_required"}
        else "awaiting_person_context"
    )
    return gated


def _mobile_phone_probe_execution_plan(
    execution_plan: dict,
    cfg: dict | None,
    *,
    now: float,
    last_probe_at: float | None,
    last_context_suppressed_at: float | None = None,
    previous_detections: list[dict] | None = None,
) -> tuple[dict, bool, bool]:
    """Recover small-phone recall only after the primary pass finds a person."""
    if not (
        execution_plan.get("run_coco_primary")
        and "mobile_phone" in set(execution_plan.get("capabilities") or [])
        and isinstance(cfg, dict)
    ):
        return execution_plan, False, False
    global_config = cfg.get("global")
    if not isinstance(global_config, dict):
        return execution_plan, False, False

    probe_width = global_config.get("mobile_phone_inference_width")
    if type(probe_width) is not int or not 160 <= probe_width <= 1920:
        return execution_plan, False, False
    default_width = global_config.get("inference_width", 960)
    if type(default_width) is not int or not 160 <= default_width <= 1920:
        default_width = 960
    normal_width = _coco_inference_imgsz(cfg, default_width)
    if probe_width <= normal_width:
        return execution_plan, False, False

    raw_interval = global_config.get("mobile_phone_probe_interval_seconds")
    if isinstance(raw_interval, bool):
        return execution_plan, False, False
    try:
        interval = float(raw_interval)
    except (TypeError, ValueError):
        return execution_plan, False, False
    if not math.isfinite(interval) or not 0.1 <= interval <= 3600:
        return execution_plan, False, False
    if last_probe_at is not None and now - last_probe_at < interval:
        return execution_plan, False, False

    has_primary_person_context = any(
        detection.get("class") == "person"
        and detection.get("model_family") in {None, "coco_primary"}
        for detection in (previous_detections or [])
    )
    if not has_primary_person_context:
        if (
            last_context_suppressed_at is not None
            and now - last_context_suppressed_at < interval
        ):
            return execution_plan, False, False
        suppressed = deepcopy(execution_plan)
        suppressed["runtime_probe_suppression_reason"] = (
            _MOBILE_PHONE_PROBE_CONTEXT_SUPPRESSION_REASON
        )
        return suppressed, False, True

    probed = deepcopy(execution_plan)
    probed["coco_inference_width_override"] = probe_width
    probed["runtime_probe_reason"] = _MOBILE_PHONE_PROBE_REASON
    if probed.get("run_ppe_specialist"):
        probed["run_ppe_specialist"] = False
        probed["ppe_prompt_terms"] = []
        probed["required_model_keys"] = [
            model_key
            for model_key in probed.get("required_model_keys", [])
            if model_key != "ppe_specialist"
        ]
        deferred_model_keys = list(probed.get("runtime_deferred_model_keys") or [])
        if "ppe_specialist" not in deferred_model_keys:
            deferred_model_keys.append("ppe_specialist")
        probed["runtime_deferred_model_keys"] = deferred_model_keys
        probed["runtime_specialist_deferral_reason"] = _PPE_PHONE_PROBE_DEFERRAL_REASON
    return probed, True, False


def _rtdetr_phone_tracker_settings(cfg: dict | None) -> tuple[int, float]:
    global_config = cfg.get("global") if isinstance(cfg, dict) else None
    if not isinstance(global_config, dict):
        return 2, 1.0
    raw_hits = global_config.get("rtdetr_phone_person_track_min_hits", 2)
    raw_ttl = global_config.get("rtdetr_phone_person_track_ttl_seconds", 1.0)
    if isinstance(raw_hits, bool):
        raw_hits = 2
    try:
        min_hits = int(raw_hits)
        ttl_seconds = float(raw_ttl)
    except (TypeError, ValueError):
        return 2, 1.0
    if not 1 <= min_hits <= 10:
        min_hits = 2
    if not math.isfinite(ttl_seconds) or not 0.25 <= ttl_seconds <= 5.0:
        ttl_seconds = 1.0
    return min_hits, ttl_seconds


def _rtdetr_phone_substitution_execution_plan(
    camera_id: str,
    execution_plan: dict,
    cfg: dict | None,
    *,
    now: float,
    stable_person_track: bool,
) -> tuple[dict, bool]:
    """Replace one phase-aligned primary slot only for stable phone context."""
    capabilities = set(execution_plan.get("capabilities") or [])
    unsupported_companion = any(
        execution_plan.get(key)
        for key in (
            "run_ppe_closed_set_candidate",
            "run_yoloe_long_tail",
            "run_fire_smoke_specialist",
            "run_face_recognition",
            "run_pose_specialist",
            "run_plate_recognition",
        )
    )
    eligible = bool(
        isinstance(cfg, dict)
        and execution_plan.get("run_coco_primary")
        and "mobile_phone" in capabilities
        and not unsupported_companion
        and stable_person_track
    )
    selected = RTDETR_PHONE_SUBSTITUTION_SCHEDULER.consider(
        camera_id,
        cfg if isinstance(cfg, dict) else {},
        now=now,
        stable_person=eligible,
    )
    if not selected:
        return execution_plan, False

    substituted = deepcopy(execution_plan)
    substituted["run_coco_primary"] = False
    substituted["run_rtdetr_phone"] = True
    substituted["partial_detection_capabilities"] = ["mobile_phone"]
    substituted["runtime_probe_reason"] = _RTDETR_PHONE_PROBE_REASON
    substituted["required_model_keys"] = [
        model_key
        for model_key in substituted.get("required_model_keys", [])
        if model_key not in {"coco_primary", "ppe_specialist"}
    ]
    if substituted.get("run_ppe_specialist"):
        substituted["run_ppe_specialist"] = False
        substituted["ppe_prompt_terms"] = []
        deferred_model_keys = list(substituted.get("runtime_deferred_model_keys") or [])
        if "ppe_specialist" not in deferred_model_keys:
            deferred_model_keys.append("ppe_specialist")
        substituted["runtime_deferred_model_keys"] = deferred_model_keys
        substituted["runtime_specialist_deferral_reason"] = (
            _PPE_PHONE_PROBE_DEFERRAL_REASON
        )
    phone_crop_mode = _configured_person_crop_mode(
        cfg,
        "phone_person_crop_mode",
        "SAFETYLENS_PHONE_PERSON_CROP_MODE",
    )
    if phone_crop_mode != "off":
        # A phone crop needs a fresh same-frame person box. Retain the primary
        # detector for that keyframe, then run RT-DETR only on its bounded
        # person crops. Since COCO is fresh, its other rules may also advance.
        substituted["run_coco_primary"] = True
        substituted.pop("partial_detection_capabilities", None)
        required_model_keys = list(execution_plan.get("required_model_keys") or [])
        if "coco_primary" not in required_model_keys:
            required_model_keys.append("coco_primary")
        substituted["required_model_keys"] = [
            model_key
            for model_key in required_model_keys
            if model_key != "ppe_specialist"
        ]
        substituted["runtime_probe_reason"] = _RTDETR_PHONE_CROP_REASON
    return substituted, True


def _ppe_specialist_cadence_execution_plan(
    camera_id: str,
    execution_plan: dict,
    cfg: dict | None,
    *,
    now: float,
    stable_person_track: bool,
    previous_detections: list[dict] | None = None,
    confirmation_required: bool = False,
) -> tuple[dict, bool, bool]:
    """Bound PPE duty and replace a primary slot when tracked context is safe."""
    if not execution_plan.get("run_ppe_specialist") or not isinstance(cfg, dict):
        return execution_plan, False, False

    ppe_capabilities = set(
        _capabilities_for_model_key(execution_plan, "ppe_specialist")
    )
    unsupported_companion = any(
        execution_plan.get(key)
        for key in (
            "run_ppe_closed_set_candidate",
            "run_yoloe_long_tail",
            "run_fire_smoke_specialist",
            "run_face_recognition",
            "run_pose_specialist",
            "run_plate_recognition",
            "run_rtdetr_phone",
        )
    )
    due, substitute = PPE_SUBSTITUTION_SCHEDULER.consider(
        camera_id,
        cfg,
        now=now,
        substitution_eligible=bool(
            execution_plan.get("run_coco_primary")
            and ppe_capabilities
            and stable_person_track
            and not unsupported_companion
        ),
        confirmation_required=confirmation_required,
    )
    if not due:
        suppressed = deepcopy(execution_plan)
        suppressed["run_ppe_specialist"] = False
        suppressed["ppe_prompt_terms"] = []
        suppressed["required_model_keys"] = [
            model_key
            for model_key in suppressed.get("required_model_keys", [])
            if model_key != "ppe_specialist"
        ]
        runtime_suppressed = list(suppressed.get("runtime_suppressed_model_keys") or [])
        if "ppe_specialist" not in runtime_suppressed:
            runtime_suppressed.append("ppe_specialist")
        suppressed["runtime_suppressed_model_keys"] = runtime_suppressed
        suppressed["runtime_suppression_reason"] = _PPE_CADENCE_SUPPRESSION_REASON
        return suppressed, False, False

    if not substitute:
        return execution_plan, True, False

    substituted = deepcopy(execution_plan)
    substituted["run_coco_primary"] = False
    substituted["run_ppe_substitution"] = True
    substituted["partial_detection_capabilities"] = sorted(ppe_capabilities)
    substituted["runtime_probe_reason"] = _PPE_SUBSTITUTION_REASON
    substituted["required_model_keys"] = [
        model_key
        for model_key in substituted.get("required_model_keys", [])
        if model_key != "coco_primary"
    ]
    substituted["ppe_context_detections"] = [
        deepcopy(detection)
        for detection in (previous_detections or [])
        if detection.get("model_family") == "coco_primary"
        and detection.get("class") in {"person", "motorcycle", "motorbike", "scooter"}
    ]
    return substituted, True, True


def _ppe_confirmation_required(
    active_violations: set[str],
    violation_window: dict[str, list[bool]],
) -> bool:
    """Accelerate PPE only while a missing-PPE incident confirms or clears."""
    return any(
        _is_ppe_violation_rule(rule)
        and (rule in active_violations or any(violation_window.get(rule) or []))
        for rule in set(active_violations) | set(violation_window)
    )


def _is_ppe_violation_rule(rule: str) -> bool:
    return rule.startswith("Missing ") or rule == "Helmet colour mismatch"


def _crowd_count_threshold_candidates(
    camera_id: str, detections: list[dict]
) -> list[dict]:
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
            class_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"class_{class_id}"
            )
        else:
            class_name = class_names.get(class_id, f"class_{class_id}")
        detections.append(
            {
                "class_id": class_id,
                "class": class_name,
                "confidence": float(record.get("confidence", record.get("conf", 0.0))),
                "bbox": list(map(int, record["bbox"])),
            }
        )
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
    first_positive_at: datetime | str | None = None,
    confirmed_at: datetime | str | None = None,
    _first_positive_monotonic_ns: int | None = None,
    _confirmed_monotonic_ns: int | None = None,
    _allow_backpressure: bool = True,
):
    cfg = get_config_snapshot()
    cam = cfg["cameras"].get(camera_id, {})
    if snapshot_jpeg is None:
        snapshot_jpeg = state.camera_frames.get(camera_id)
        clean_snapshot_jpeg = state.camera_clean_frames.get(camera_id)
    if not snapshot_jpeg:
        logger.debug(
            "Skipping alert — no frame captured yet",
            extra={"camera_id": camera_id, "rule": rule},
        )
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
    timing_registered = False
    if _first_positive_monotonic_ns is not None and _confirmed_monotonic_ns is not None:
        try:
            initial_target_keys = (
                str(target.get("target_key"))
                for target in delivery_targets
                if target.get("kind", "initial") == "initial"
                and target.get("target_key")
                and target.get("channel")
            )
            evicted_contexts = alert_timing.registry.remember(
                alert_id,
                first_positive_ns=_first_positive_monotonic_ns,
                confirmed_ns=_confirmed_monotonic_ns,
                initial_target_keys=initial_target_keys,
            )
            for evicted in evicted_contexts:
                if evicted.persisted_ns is not None:
                    pipeline_telemetry.censor_pending_alert_deliveries(
                        len(evicted.remaining_initial_target_keys)
                    )
                elif evicted.deferred_initial_outcomes:
                    pipeline_telemetry.record_alert_persistence_censored()
            timing_registered = True
        except Exception:
            logger.warning(
                "Invalid alert timing anchors ignored",
                extra={"alert_id": alert_id, "camera_id": camera_id},
            )
    try:
        submission = _get_alert_pipeline().submit(
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
                "first_positive_at": first_positive_at,
                "confirmed_at": confirmed_at,
            },
            output_ids=output_ids,
            allow_backpressure=_allow_backpressure,
        )
    except Exception:
        if timing_registered:
            alert_timing.registry.discard(alert_id)
        raise

    if not timing_registered:
        return submission

    add_done_callback = getattr(submission, "add_done_callback", None)
    if callable(add_done_callback):

        def discard_failed_timing(completed) -> None:
            try:
                persisted = completed.result()
            except Exception:
                alert_timing.registry.discard(alert_id)
                return
            if not isinstance(persisted, Mapping):
                alert_timing.registry.discard(alert_id)

        add_done_callback(discard_failed_timing)
    elif isinstance(submission, Mapping):
        _observe_persisted_alert_timing(submission)
    else:
        alert_timing.registry.discard(alert_id)
    return submission


def _persist_policy_trigger_after_alert(submission, rule_id: str) -> None:
    """Persist a UI rule timestamp only after its alert becomes durable."""
    if not rule_id:
        return
    add_done_callback = getattr(submission, "add_done_callback", None)
    if not callable(add_done_callback):
        if isinstance(submission, Mapping):
            policy_engine.mark_rule_triggered(rule_id)
        return

    def persist_after_success(completed) -> None:
        try:
            persisted = completed.result()
        except Exception:
            return
        if isinstance(persisted, Mapping):
            policy_engine.mark_rule_triggered(rule_id)

    add_done_callback(persist_after_success)


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
    resp = None
    try:
        cfg = get_config()
        vlm_cfg = cfg["vlm"]
        endpoint = str(vlm_cfg.get("url") or OLLAMA_URL)
        if vlm_cfg.get("remote_only", False) and not remote_vlm_endpoint_allowed(
            endpoint
        ):
            logger.warning("VLM endpoint rejected because it is not off-device")
            return "VLM unavailable"

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        resp = requests.post(
            endpoint,
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
            # Redirects could turn a validated remote URL into localhost or a
            # Jetson interface after the off-device check.
            allow_redirects=False,
            stream=True,
        )

        if resp.status_code == 200:
            headers = getattr(resp, "headers", {})
            content_length = headers.get("Content-Length") if isinstance(headers, Mapping) else None
            if content_length is not None:
                try:
                    declared_bytes = int(content_length)
                except (TypeError, ValueError):
                    declared_bytes = None
                if declared_bytes is not None and declared_bytes > VLM_MAX_RESPONSE_BYTES:
                    raise ValueError("VLM response exceeds the configured byte limit")

            encoded = bytearray()
            for chunk in resp.iter_content(chunk_size=16 * 1024):
                if not chunk:
                    continue
                if len(encoded) + len(chunk) > VLM_MAX_RESPONSE_BYTES:
                    raise ValueError("VLM response exceeds the configured byte limit")
                encoded.extend(chunk)
            payload = json.loads(bytes(encoded).decode("utf-8"))
            if not isinstance(payload, Mapping):
                return "No response from VLM"
            response_text = payload.get("response")
            if response_text is None:
                return "No response from VLM"
            return str(response_text)[:VLM_MAX_RESULT_CHARS]
        return f"VLM error: {resp.status_code}"
    except Exception as exc:
        logger.warning(
            "VLM request failed",
            extra={"error_phase": "vlm_request", "error_type": type(exc).__name__},
        )
        return "VLM unavailable"
    finally:
        close = getattr(resp, "close", None)
        if callable(close):
            close()


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
            prefix = normalized[max(0, match.start() - 40) : match.start()]
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


def _legacy_vlm_worker_loop(camera_id: str, stop_event: threading.Event):
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
            logger.debug(
                "Discarding VLM result after camera stop",
                extra={"camera_id": camera_id},
            )
            return
        elapsed = time.time() - started
        logger.info(
            "VLM analysis done",
            extra={"camera_id": camera_id, "elapsed": round(elapsed, 1)},
        )

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
            logger.debug(
                "VLM alert submitted for persistence", extra={"camera_id": camera_id}
            )


def _process_vlm_enrichment(work: VLMEnrichmentWork) -> str:
    payload = work.payload if isinstance(work.payload, Mapping) else {}
    frame_bytes = payload.get("frame_bytes")
    if not isinstance(frame_bytes, bytes):
        raise ValueError("VLM enrichment payload has no encoded frame")
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("VLM enrichment frame could not be decoded")
    result = call_vlm(frame)
    if _vlm_result_verdict(result, []) == "unknown":
        raise RuntimeError("VLM provider did not return an analyzable response")
    return result


def _clear_vlm_incident_after_failed_persistence(
    camera_id: str,
    generation: str,
    submission,
) -> None:
    try:
        persisted = submission.result()
    except Exception:
        persisted = None
    if isinstance(persisted, Mapping):
        return
    with _vlm_incident_lock:
        if _vlm_incident_active.get(camera_id) == generation:
            _vlm_incident_active.pop(camera_id, None)


def _vlm_work_is_current(work: VLMEnrichmentWork) -> bool:
    with _vlm_dispatcher_lock:
        dispatcher = _vlm_dispatcher
    return bool(
        dispatcher is not None
        and dispatcher.is_current_generation(work.camera_id, work.generation)
    )


def _run_if_vlm_work_current(
    work: VLMEnrichmentWork,
    action: Callable[[], object],
) -> bool:
    with _vlm_dispatcher_lock:
        dispatcher = _vlm_dispatcher
    return bool(
        dispatcher is not None
        and dispatcher.run_if_current(work.camera_id, work.generation, action)
    )


def _current_vlm_delivery_config(work: VLMEnrichmentWork) -> dict | None:
    if not _vlm_work_is_current(work):
        return None
    cfg = get_config()
    vlm_cfg = cfg.get("vlm", {})
    if not vlm_cfg.get("enabled", False):
        return None
    cameras = cfg.get("cameras")
    if isinstance(cameras, dict):
        camera = cameras.get(work.camera_id)
        if not isinstance(camera, dict) or not camera.get("enabled", True):
            return None
    return vlm_cfg


def _handle_vlm_enrichment_result(
    work: VLMEnrichmentWork,
    result: object,
    elapsed_seconds: float,
) -> None:
    vlm_cfg = _current_vlm_delivery_config(work)
    if vlm_cfg is None:
        return
    text_result = str(result or "")[:VLM_MAX_RESULT_CHARS]
    def record_result() -> None:
        with state.vlm_lock:
            state.vlm_last_results[work.camera_id] = {
                "text": text_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed": round(max(0.0, elapsed_seconds), 1),
                "advisory": True,
            }

    if not _run_if_vlm_work_current(work, record_result):
        return

    verdict = _vlm_result_verdict(
        text_result,
        vlm_cfg.get("violation_keywords", []),
    )
    should_submit = False

    def update_incident() -> None:
        nonlocal should_submit
        with _vlm_incident_lock:
            if verdict == "safe":
                if _vlm_incident_active.get(work.camera_id) == work.generation:
                    _vlm_incident_active.pop(work.camera_id, None)
                return
            if (
                verdict != "violation"
                or not vlm_cfg.get("alerting_enabled", False)
                or _vlm_incident_active.get(work.camera_id) == work.generation
            ):
                return
            _vlm_incident_active[work.camera_id] = work.generation
            should_submit = True

    if not _run_if_vlm_work_current(work, update_incident) or not should_submit:
        return

    vlm_cfg = _current_vlm_delivery_config(work)
    if vlm_cfg is None:
        with _vlm_incident_lock:
            if _vlm_incident_active.get(work.camera_id) == work.generation:
                _vlm_incident_active.pop(work.camera_id, None)
        return
    payload = work.payload if isinstance(work.payload, Mapping) else {}
    frame_bytes = payload.get("frame_bytes")
    submission_box: list[object] = []

    def submit_alert() -> None:
        try:
            submission_box.append(
                create_alert(
                    camera_id=work.camera_id,
                    rule="VLM Scene Analysis",
                    severity="P2",
                    confidence=0.92,
                    description=text_result[:200],
                    source=f"VLM ({vlm_cfg.get('model', 'remote')})",
                    snapshot_jpeg=(
                        frame_bytes if isinstance(frame_bytes, bytes) else None
                    ),
                    _allow_backpressure=False,
                )
            )
        except Exception:
            submission_box.append(None)
            logger.exception(
                "VLM advisory alert submission failed",
                extra={"camera_id": work.camera_id},
            )

    if not _run_if_vlm_work_current(work, submit_alert):
        return
    submission = submission_box[0]
    if isinstance(submission, Mapping):
        return
    add_done_callback = getattr(submission, "add_done_callback", None)
    if callable(add_done_callback):
        add_done_callback(
            lambda completed, camera_id=work.camera_id, generation=work.generation: (
                _clear_vlm_incident_after_failed_persistence(
                    camera_id,
                    generation,
                    completed,
                )
            )
        )
        return
    with _vlm_incident_lock:
        if _vlm_incident_active.get(work.camera_id) == work.generation:
            _vlm_incident_active.pop(work.camera_id, None)


def _get_vlm_dispatcher(cfg: dict) -> VLMEnrichmentDispatcher:
    global _vlm_dispatcher
    with _vlm_dispatcher_lock:
        if _vlm_dispatcher is None:
            vlm_cfg = cfg.get("vlm", {})
            _vlm_dispatcher = VLMEnrichmentDispatcher(
                process=_process_vlm_enrichment,
                on_result=_handle_vlm_enrichment_result,
                maximum_pending_cameras=_bounded_runtime_int(
                    vlm_cfg.get("maximum_pending_cameras", 32),
                    32,
                    minimum=1,
                    maximum=256,
                ),
                maximum_queue_age_seconds=_bounded_runtime_float(
                    vlm_cfg.get("maximum_queue_age_seconds", 15.0),
                    15.0,
                    minimum=1.0,
                    maximum=300.0,
                ),
                failure_threshold=_bounded_runtime_int(
                    vlm_cfg.get("failure_threshold", 3),
                    3,
                    minimum=1,
                    maximum=100,
                ),
                circuit_cooldown_seconds=_bounded_runtime_float(
                    vlm_cfg.get("circuit_cooldown_seconds", 60.0),
                    60.0,
                    minimum=0.0,
                    maximum=3600.0,
                ),
            )
            _vlm_dispatcher.start()
        return _vlm_dispatcher


def _vlm_worker_loop(camera_id: str, stop_event: threading.Event):
    initial_cfg = get_config()
    initial_vlm_cfg = initial_cfg.get("vlm", {})
    # Hand-written legacy test/site configs without the new isolation fields
    # retain their old behavior. Normalized runtime configs always take the
    # process-wide, fail-open dispatcher path below.
    if (
        "remote_only" not in initial_vlm_cfg
        and "alerting_enabled" not in initial_vlm_cfg
    ):
        _legacy_vlm_worker_loop(camera_id, stop_event)
        return

    generation = uuid4().hex
    dispatcher = _get_vlm_dispatcher(initial_cfg)
    dispatcher.register_camera(camera_id, generation)
    try:
        while not stop_event.is_set():
            cfg = get_config()
            vlm_cfg = cfg.get("vlm", {})
            if stop_event.wait(_vlm_interval_seconds(vlm_cfg.get("interval", 45))):
                return
            if (
                not vlm_cfg.get("enabled", False)
                or not licensing.is_inference_allowed()
            ):
                continue
            frame_bytes = state.camera_clean_frames.get(camera_id)
            if frame_bytes is None:
                frame_bytes = state.camera_frames.get(camera_id)
            if frame_bytes is None:
                continue
            dispatcher.offer(
                camera_id,
                generation,
                {"frame_bytes": bytes(frame_bytes)},
            )
    finally:
        dispatcher.discard_camera(camera_id, generation)
        with _vlm_incident_lock:
            if _vlm_incident_active.get(camera_id) == generation:
                _vlm_incident_active.pop(camera_id, None)


def vlm_worker(camera_id: str, stop_event: threading.Event):
    """Run one VLM companion and relinquish only its own registration."""
    try:
        _vlm_worker_loop(camera_id, stop_event)
    except Exception:
        logger.exception(
            "VLM worker exited unexpectedly", extra={"camera_id": camera_id}
        )
        raise
    finally:
        _deregister_worker_on_exit(state.vlm_threads, camera_id, stop_event)


def _run_full_frame_grouped_inference(
    camera_id: str,
    frame: np.ndarray,
    execution_plan: dict,
    *,
    conf: float,
    device: str,
    imgsz: int,
    cfg: dict | None = None,
    frame_batch_size_hint: int | None = None,
):
    annotated = None
    detections: list[dict] = []
    visible_detection_count = 0
    pose_results = None
    model_invocations = {
        "coco_primary": 0,
        "rtdetr_phone": 0,
        "rtdetr_phone_fallback": 0,
        "ppe_specialist": 0,
        "ppe_closed_set_candidate": 0,
        "yoloe_long_tail": 0,
        "fire_smoke_specialist": 0,
        "pose_specialist": 0,
    }
    ppe_prompts = execution_plan.get("ppe_prompt_terms") or []
    long_tail_prompts = execution_plan.get("yoloe_prompt_terms") or []
    batch_requests = []
    rtdetr_records = None
    rtdetr_fallback = False
    reuse_existing_primary_fallback = bool(
        execution_plan.get("reuse_existing_coco_primary_on_rtdetr_fallback")
    )
    if execution_plan.get("run_rtdetr_phone"):
        person_conf = _rule_confidence_for_capability(
            camera_id, "person_presence", conf, cfg=cfg
        )
        phone_conf = _rule_confidence_for_capability(
            camera_id, "mobile_phone", min(conf, 0.15), cfg=cfg
        )
        model_invocations["rtdetr_phone"] += 1
        try:
            rtdetr_records = model_manager.predict_rtdetr_phone_records(
                frame,
                person_conf=person_conf,
                phone_conf=phone_conf,
                frame_batch_size_hint=frame_batch_size_hint,
            )
        except model_manager.RemoteRTDETRPhoneUnavailableError:
            rtdetr_fallback = True
            model_invocations["rtdetr_phone_fallback"] += 1
    run_coco_primary = bool(
        execution_plan.get("run_coco_primary")
        or (rtdetr_fallback and not reuse_existing_primary_fallback)
    )
    if run_coco_primary:
        coco_conf = _rule_confidence_for_model_family(
            camera_id,
            "coco_primary",
            conf,
            cfg=cfg,
        )
        model_invocations["coco_primary"] += 1
        batch_requests.append(
            {
                "request_id": "coco_primary",
                "model_key": "coco_primary",
                "conf": coco_conf,
                "device": device,
                "imgsz": _coco_inference_imgsz(cfg, imgsz, execution_plan),
            }
        )
    if execution_plan.get("run_ppe_specialist") and ppe_prompts:
        ppe_conf = _rule_confidence_for_model_family(
            camera_id,
            "ppe_specialist",
            conf,
            cfg=cfg,
        )
        model_invocations["ppe_specialist"] += 1
        batch_requests.append(
            {
                "request_id": "ppe_specialist",
                "model_key": "ppe_specialist",
                "conf": ppe_conf,
                "device": device,
                "imgsz": _ppe_inference_imgsz(cfg, imgsz),
                "classes": ppe_prompts,
            }
        )
    if execution_plan.get("run_ppe_closed_set_candidate"):
        candidate_capabilities = _capabilities_for_model_key(
            execution_plan,
            "ppe_closed_set_candidate",
        )
        candidate_conf = _rule_confidence_for_capabilities(
            camera_id,
            candidate_capabilities,
            conf,
            cfg=cfg,
        )
        model_invocations["ppe_closed_set_candidate"] += 1
        batch_requests.append(
            {
                "request_id": "ppe_closed_set_candidate",
                "model_key": "ppe_closed_set_candidate",
                "conf": candidate_conf,
                "device": device,
                "imgsz": imgsz,
            }
        )
    if execution_plan.get("run_yoloe_long_tail") and long_tail_prompts:
        model_invocations["yoloe_long_tail"] += 1
        batch_requests.append(
            {
                "request_id": "yoloe_long_tail",
                "model_key": "yoloe_long_tail",
                "conf": conf,
                "device": device,
                "imgsz": imgsz,
                "classes": long_tail_prompts,
            }
        )
    if execution_plan.get("run_fire_smoke_specialist"):
        fire_conf = _rule_confidence_for_capability(
            camera_id,
            "fire_smoke",
            conf,
            cfg=cfg,
        )
        model_invocations["fire_smoke_specialist"] += 1
        batch_requests.append(
            {
                "request_id": "fire_smoke_specialist",
                "model_key": "fire_smoke_specialist",
                "conf": fire_conf,
                "device": device,
                "imgsz": imgsz,
            }
        )
    if execution_plan.get("run_pose_specialist"):
        model_invocations["pose_specialist"] += 1
        batch_requests.append(
            {
                "request_id": "pose_specialist",
                "model_key": "pose_specialist",
                "conf": conf,
                "device": device,
                "imgsz": imgsz,
            }
        )
    batch_options = (
        {"frame_batch_size_hint": frame_batch_size_hint}
        if frame_batch_size_hint is not None
        else {}
    )
    record_batches = (
        model_manager.predict_record_batches(frame, batch_requests, **batch_options)
        if batch_requests
        else {}
    )

    if execution_plan.get("run_rtdetr_phone") and not rtdetr_fallback:
        records = _filter_coco_records_for_rule_confidence(
            camera_id, rtdetr_records or [], conf, cfg=cfg
        )
        phone_detections = _detection_batch_from_records(records, "rtdetr_phone")
        detections.extend(phone_detections)
        visible_detection_count += len(phone_detections)

    if run_coco_primary:
        records = _filter_coco_records_for_rule_confidence(
            camera_id,
            record_batches["coco_primary"],
            conf,
            cfg=cfg,
        )
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
        if execution_plan.get("run_ppe_substitution"):
            detections.extend(
                deepcopy(execution_plan.get("ppe_context_detections") or [])
            )

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
        annotated = _draw_stream_detection_records(
            frame,
            detections,
            camera_id,
            show_overlay=False,
        )
        pose_results = record_batches["pose_specialist"]
        annotated, fall_dets = draw_pose_detections(
            annotated, pose_results, fall_only=False, camera_id=camera_id
        )
        detections.extend(_normalize_detection_batch(fall_dets, "pose_specialist"))

    if annotated is not None:
        annotated = apply_camera_overlay(
            annotated,
            camera_id=camera_id,
            detection_count=visible_detection_count,
        )
    return annotated, detections, pose_results, model_invocations


_PERSON_CROP_MODES = frozenset({"off", "shadow", "confirm", "active"})
_PERSON_CROP_TARGET_FLAGS = {
    "ppe": "run_ppe_specialist",
    "phone": "run_rtdetr_phone",
}
_PERSON_CROP_MODEL_FLAGS = (
    "run_rtdetr_phone",
    "run_ppe_specialist",
    "run_ppe_closed_set_candidate",
    "run_yoloe_long_tail",
    "run_fire_smoke_specialist",
    "run_pose_specialist",
)


def _configured_person_crop_mode(
    cfg: dict | None,
    setting: str,
    environment_setting: str,
) -> str:
    global_cfg = cfg.get("global", {}) if isinstance(cfg, dict) else {}
    configured = (
        str(os.environ.get(environment_setting, global_cfg.get(setting, "off")))
        .strip()
        .lower()
    )
    return configured if configured in _PERSON_CROP_MODES else "off"


def _person_crop_policy(cfg: dict | None) -> PersonCropPolicy:
    global_cfg = cfg.get("global", {}) if isinstance(cfg, dict) else {}
    return PersonCropPolicy(
        padding_fraction=_bounded_runtime_float(
            global_cfg.get("person_crop_padding_fraction", 0.12),
            0.12,
            minimum=0.0,
            maximum=1.0,
        ),
        min_person_width=_bounded_runtime_int(
            global_cfg.get("person_crop_min_person_width", 24),
            24,
            minimum=1,
            maximum=4096,
        ),
        min_person_height=_bounded_runtime_int(
            global_cfg.get("person_crop_min_person_height", 48),
            48,
            minimum=1,
            maximum=4096,
        ),
        boundary_margin=_bounded_runtime_int(
            global_cfg.get("person_crop_boundary_margin", 2),
            2,
            minimum=0,
            maximum=256,
        ),
        max_crops=_bounded_runtime_int(
            global_cfg.get("person_crop_max_crops", 8),
            8,
            minimum=1,
            maximum=32,
        ),
        person_dedup_iou=_bounded_runtime_float(
            global_cfg.get("person_crop_person_dedup_iou", 0.85),
            0.85,
            minimum=0.0,
            maximum=1.0,
        ),
        result_dedup_iou=_bounded_runtime_float(
            global_cfg.get("person_crop_result_dedup_iou", 0.55),
            0.55,
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _fresh_primary_person_detections(detections: list[dict]) -> list[dict]:
    """Accept only detector-keyframe people; tracker projections are never seeds."""
    return [
        detection
        for detection in detections
        if str(detection.get("class") or "").strip().lower() == "person"
        and detection.get("model_family") == "coco_primary"
        and detection.get("observation_kind") != "tracker_projection"
    ]


def _person_crop_telemetry_entry(
    mode: str,
    plan: PersonCropPlan | None,
    *,
    crop_attempts: int = 0,
    crop_succeeded: bool = False,
    crop_candidate_count: int = 0,
    full_frame_invocations: int = 0,
    authoritative_path: str = "full_frame",
    extra_reasons: list[str] | tuple[str, ...] = (),
) -> dict:
    reasons = list(plan.fallback_reasons if plan is not None else ())
    for reason in extra_reasons:
        if reason not in reasons:
            reasons.append(reason)
    return {
        "mode": mode,
        "freshPersonDetectionCount": (
            plan.person_detection_count if plan is not None else 0
        ),
        "validPersonCount": plan.valid_person_count if plan is not None else 0,
        "cropCount": len(plan.crops) if plan is not None else 0,
        "cropInferenceAttempts": crop_attempts,
        "cropInferenceSucceeded": crop_succeeded,
        "cropCandidateCount": crop_candidate_count,
        "fullFrameInvocations": full_frame_invocations,
        "authoritativePath": authoritative_path,
        "fallbackRequired": bool(reasons),
        "fallbackReasons": reasons,
    }


def _merge_model_invocations(*counts: dict) -> dict:
    merged: dict[str, int] = {}
    for group in counts:
        for model_key, value in group.items():
            merged[model_key] = merged.get(model_key, 0) + int(value)
    return merged


def _run_ppe_person_crops(
    camera_id: str,
    plan: PersonCropPlan,
    *,
    prompts: list[str],
    conf: float,
    device: str,
    imgsz: int,
    cfg: dict | None,
    frame_batch_size_hint: int | None,
    attempt_counter: list[int],
) -> tuple[list[dict], int]:
    ppe_conf = _rule_confidence_for_model_family(
        camera_id,
        "ppe_specialist",
        conf,
        cfg=cfg,
    )
    result_sets: list[list[dict]] = []
    for crop in plan.crops:
        attempt_counter[0] += 1
        request = {
            "request_id": "ppe_person_crop",
            "model_key": "ppe_specialist",
            "conf": ppe_conf,
            "device": device,
            "imgsz": _ppe_inference_imgsz(cfg, imgsz),
            "classes": prompts,
        }
        options = (
            {"frame_batch_size_hint": frame_batch_size_hint}
            if frame_batch_size_hint is not None
            else {}
        )
        records = model_manager.predict_record_batches(
            crop.image,
            [request],
            **options,
        )["ppe_person_crop"]
        result_sets.append(
            _detection_batch_from_records(
                records,
                "ppe_specialist",
                class_names=prompts,
            )
        )
    remapped = remap_crop_detections(
        plan,
        result_sets,
        dedup_iou_threshold=_person_crop_policy(cfg).result_dedup_iou,
    )
    return [
        {
            **detection,
            "inference_scope": "person_crop",
            "observation_kind": "fresh_crop_specialist",
        }
        for detection in remapped
    ], len(plan.crops)


def _run_phone_person_crops(
    camera_id: str,
    plan: PersonCropPlan,
    *,
    conf: float,
    cfg: dict | None,
    frame_batch_size_hint: int | None,
    attempt_counter: list[int],
) -> tuple[list[dict], int]:
    person_conf = _rule_confidence_for_capability(
        camera_id,
        "person_presence",
        conf,
        cfg=cfg,
    )
    phone_conf = _rule_confidence_for_capability(
        camera_id,
        "mobile_phone",
        min(conf, 0.15),
        cfg=cfg,
    )
    result_sets: list[list[dict]] = []
    for crop in plan.crops:
        attempt_counter[0] += 1
        records = model_manager.predict_rtdetr_phone_records(
            crop.image,
            person_conf=person_conf,
            phone_conf=phone_conf,
            frame_batch_size_hint=frame_batch_size_hint,
        )
        records = _filter_coco_records_for_rule_confidence(
            camera_id,
            records,
            conf,
            cfg=cfg,
        )
        crop_detections = _detection_batch_from_records(records, "rtdetr_phone")
        result_sets.append(
            [
                detection
                for detection in crop_detections
                if str(detection.get("class") or "").strip().lower()
                in {"cell phone", "mobile phone", "phone"}
            ]
        )
    remapped = remap_crop_detections(
        plan,
        result_sets,
        dedup_iou_threshold=_person_crop_policy(cfg).result_dedup_iou,
    )
    return [
        {
            **detection,
            "inference_scope": "person_crop",
            "observation_kind": "fresh_crop_specialist",
        }
        for detection in remapped
    ], len(plan.crops)


def _run_crop_enabled_grouped_inference(
    camera_id: str,
    frame: np.ndarray,
    execution_plan: dict,
    *,
    conf: float,
    device: str,
    imgsz: int,
    cfg: dict | None,
    frame_batch_size_hint: int | None,
    modes: dict[str, str],
) -> tuple[np.ndarray | None, list[dict], object, dict, dict]:
    """Run detector-keyframe crops while retaining the full-frame fail-open path."""
    enabled_targets = [
        target
        for target, flag in _PERSON_CROP_TARGET_FLAGS.items()
        if modes[target] != "off"
        and execution_plan.get(flag)
        and (target != "ppe" or bool(execution_plan.get("ppe_prompt_terms")))
    ]
    if not enabled_targets:
        annotated, detections, pose_results, invocations = (
            _run_full_frame_grouped_inference(
                camera_id,
                frame,
                execution_plan,
                conf=conf,
                device=device,
                imgsz=imgsz,
                cfg=cfg,
                frame_batch_size_hint=frame_batch_size_hint,
            )
        )
        return annotated, detections, pose_results, invocations, {}

    if not execution_plan.get("run_coco_primary"):
        annotated, detections, pose_results, invocations = (
            _run_full_frame_grouped_inference(
                camera_id,
                frame,
                execution_plan,
                conf=conf,
                device=device,
                imgsz=imgsz,
                cfg=cfg,
                frame_batch_size_hint=frame_batch_size_hint,
            )
        )
        telemetry = {
            target: _person_crop_telemetry_entry(
                modes[target],
                None,
                full_frame_invocations=1,
                extra_reasons=("no_fresh_primary_keyframe",),
            )
            for target in enabled_targets
        }
        return annotated, detections, pose_results, invocations, telemetry

    primary_plan = deepcopy(execution_plan)
    for flag in _PERSON_CROP_MODEL_FLAGS:
        primary_plan[flag] = False
    primary_plan["run_coco_primary"] = True
    (
        _primary_annotated,
        primary_detections,
        _primary_pose,
        primary_invocations,
    ) = _run_full_frame_grouped_inference(
        camera_id,
        frame,
        primary_plan,
        conf=conf,
        device=device,
        imgsz=imgsz,
        cfg=cfg,
        frame_batch_size_hint=frame_batch_size_hint,
    )
    fresh_people = _fresh_primary_person_detections(primary_detections)
    crop_plan = plan_person_crops(frame, fresh_people, _person_crop_policy(cfg))

    remaining_plan = deepcopy(execution_plan)
    remaining_plan["run_coco_primary"] = False
    remaining_plan["reuse_existing_coco_primary_on_rtdetr_fallback"] = True
    crop_detections: list[dict] = []
    crop_invocations = {
        "ppe_specialist": 0,
        "rtdetr_phone": 0,
        "ppe_person_crop": 0,
        "phone_person_crop": 0,
    }
    telemetry: dict[str, dict] = {}

    for target in enabled_targets:
        mode = modes[target]
        extra_reasons: list[str] = []
        if not crop_plan.crops:
            extra_reasons.append("no_fresh_person_crop")
            remaining_plan[_PERSON_CROP_TARGET_FLAGS[target]] = True
            telemetry[target] = _person_crop_telemetry_entry(
                mode,
                crop_plan,
                full_frame_invocations=1,
                extra_reasons=extra_reasons,
            )
            continue

        initial_decision = decide_crop_execution(mode, crop_plan)
        crop_results: list[dict] = []
        crop_attempts = 0
        attempt_counter = [0]
        crop_failed = False
        if initial_decision.run_person_crops:
            try:
                if target == "ppe":
                    crop_results, crop_attempts = _run_ppe_person_crops(
                        camera_id,
                        crop_plan,
                        prompts=list(execution_plan.get("ppe_prompt_terms") or []),
                        conf=conf,
                        device=device,
                        imgsz=imgsz,
                        cfg=cfg,
                        frame_batch_size_hint=frame_batch_size_hint,
                        attempt_counter=attempt_counter,
                    )
                    crop_invocations["ppe_person_crop"] += crop_attempts
                    crop_invocations["ppe_specialist"] += crop_attempts
                else:
                    crop_results, crop_attempts = _run_phone_person_crops(
                        camera_id,
                        crop_plan,
                        conf=conf,
                        cfg=cfg,
                        frame_batch_size_hint=frame_batch_size_hint,
                        attempt_counter=attempt_counter,
                    )
                    crop_invocations["phone_person_crop"] += crop_attempts
                    crop_invocations["rtdetr_phone"] += crop_attempts
            except Exception as exc:
                crop_attempts = attempt_counter[0]
                if target == "ppe":
                    crop_invocations["ppe_person_crop"] += crop_attempts
                    crop_invocations["ppe_specialist"] += crop_attempts
                else:
                    crop_invocations["phone_person_crop"] += crop_attempts
                    crop_invocations["rtdetr_phone"] += crop_attempts
                crop_failed = True
                crop_results = []
                extra_reasons.append("crop_inference_failed")
                logger.warning(
                    "Person-crop specialist failed; retaining full-frame fallback",
                    extra={
                        "camera_id": camera_id,
                        "specialist": target,
                        "error_type": type(exc).__name__,
                    },
                )

        crop_candidate = bool(crop_results)
        # A PPE crop with no prompt hit can itself be a missing-equipment
        # candidate, so confirm mode always delegates the verdict to full-frame.
        if target == "ppe" and initial_decision.run_person_crops:
            crop_candidate = True
        final_decision = decide_crop_execution(
            mode,
            crop_plan,
            crop_candidate=crop_candidate,
            crop_failed=crop_failed,
        )
        run_full_frame = final_decision.run_full_frame_specialist
        remaining_plan[_PERSON_CROP_TARGET_FLAGS[target]] = run_full_frame

        if final_decision.crop_evidence_authoritative and not crop_failed:
            crop_detections.extend(crop_results)
            authoritative_path = "person_crop"
        elif mode == "shadow":
            authoritative_path = "full_frame_shadow"
        elif mode == "confirm" and run_full_frame:
            authoritative_path = "full_frame_confirmation"
        else:
            authoritative_path = "full_frame"
        telemetry[target] = _person_crop_telemetry_entry(
            mode,
            crop_plan,
            crop_attempts=crop_attempts,
            crop_succeeded=initial_decision.run_person_crops and not crop_failed,
            crop_candidate_count=len(crop_results),
            full_frame_invocations=int(run_full_frame),
            authoritative_path=authoritative_path,
            extra_reasons=extra_reasons,
        )

    (
        _remaining_annotated,
        remaining_detections,
        pose_results,
        remaining_invocations,
    ) = _run_full_frame_grouped_inference(
        camera_id,
        frame,
        remaining_plan,
        conf=conf,
        device=device,
        imgsz=imgsz,
        cfg=cfg,
        frame_batch_size_hint=frame_batch_size_hint,
    )
    merged_detections = [
        *primary_detections,
        *remaining_detections,
        *crop_detections,
    ]
    # The crop path runs the primary and remaining specialists in separate
    # calls. Rebuild one final view so a pose layer cannot accidentally hide
    # primary-person or authoritative crop evidence.
    annotated = _draw_stream_detection_records(
        frame,
        merged_detections,
        camera_id,
        show_overlay=False,
    )
    if pose_results is not None:
        annotated, _ = draw_pose_detections(
            annotated,
            pose_results,
            fall_only=False,
            camera_id=camera_id,
        )
    annotated = apply_camera_overlay(
        annotated,
        camera_id=camera_id,
        detection_count=len(_stream_visible_detection_records(merged_detections)),
    )
    return (
        annotated,
        merged_detections,
        pose_results,
        _merge_model_invocations(
            primary_invocations,
            remaining_invocations,
            crop_invocations,
        ),
        telemetry,
    )


def _run_grouped_inference(
    camera_id: str,
    frame: np.ndarray,
    execution_plan: dict,
    *,
    conf: float,
    device: str,
    imgsz: int,
    cfg: dict | None = None,
    frame_batch_size_hint: int | None = None,
    person_crop_telemetry: dict | None = None,
):
    modes = {
        "ppe": _configured_person_crop_mode(
            cfg,
            "ppe_person_crop_mode",
            "SAFETYLENS_PPE_PERSON_CROP_MODE",
        ),
        "phone": _configured_person_crop_mode(
            cfg,
            "phone_person_crop_mode",
            "SAFETYLENS_PHONE_PERSON_CROP_MODE",
        ),
    }
    if any(
        modes[target] != "off"
        and execution_plan.get(flag)
        and (target != "ppe" or bool(execution_plan.get("ppe_prompt_terms")))
        for target, flag in _PERSON_CROP_TARGET_FLAGS.items()
    ):
        annotated, detections, pose_results, invocations, telemetry = (
            _run_crop_enabled_grouped_inference(
                camera_id,
                frame,
                execution_plan,
                conf=conf,
                device=device,
                imgsz=imgsz,
                cfg=cfg,
                frame_batch_size_hint=frame_batch_size_hint,
                modes=modes,
            )
        )
        if person_crop_telemetry is not None:
            person_crop_telemetry.update(telemetry)
        return annotated, detections, pose_results, invocations
    return _run_full_frame_grouped_inference(
        camera_id,
        frame,
        execution_plan,
        conf=conf,
        device=device,
        imgsz=imgsz,
        cfg=cfg,
        frame_batch_size_hint=frame_batch_size_hint,
    )


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

    now = time.time()
    detections: list[dict] = []
    snapshot_jpeg: bytes | None = None
    snapshot_encode_attempted = False

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
        color = (
            (16, 185, 129)
            if event_type == "face_match"
            else (0, 193, 255)
            if event_type == "face_low_quality"
            else (0, 82, 255)
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

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
        if not snapshot_encode_attempted:
            snapshot_encode_attempted = True
            snapshot_ok, snapshot_buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if snapshot_ok:
                snapshot_jpeg = snapshot_buffer.tobytes()
            else:
                logger.warning(
                    "Face event snapshot encoding failed; persisting without snapshot",
                    extra={"camera_id": camera_id},
                )
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
) -> bool:
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
        appsink_latest_buffer_drops_observable=(
            tracker.appsink_latest_buffer_drops_observable
        ),
        appsink_latest_buffer_drop_method=(
            tracker.appsink_latest_buffer_drop_method
        ),
        capture_drop_accounting=tracker.capture_drop_accounting,
        capture_drop_count_is_lower_bound=tracker.capture_drop_count_is_lower_bound,
        decoder_policy_drop_accounting=tracker.decoder_policy_drop_accounting,
    )


def _capture_backend_name(capture) -> str:
    backend = str(getattr(capture, "capture_backend", "ffmpeg"))
    return (
        backend
        if backend
        in {
            "ffmpeg",
            "gstreamer_unknown",
            "gstreamer_software",
            "gstreamer_nvdec",
        }
        else "unknown"
    )


def _capture_policy_telemetry(capture) -> dict[str, object]:
    getter = getattr(capture, "capture_policy_telemetry", None)
    if callable(getter):
        try:
            telemetry = getter()
        except Exception:
            telemetry = None
        if isinstance(telemetry, dict):
            return telemetry
    return {
        "appsinkLatestBufferDropsObservable": False,
        "appsinkLatestBufferDropMethod": "unavailable",
        "captureDropAccounting": "application-drain-only",
        "captureDropCountIsLowerBound": True,
        "decoderPolicyDropAccounting": "not-applicable",
    }


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
        logger.warning(
            "Camera connection outage detected; retry scheduled", extra=fields
        )
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

    now = time.time()
    detections: list[dict] = []
    snapshot_jpeg: bytes | None = None
    snapshot_encode_attempted = False
    frame_h, frame_w = frame.shape[:2]

    for candidate in candidates:
        bbox = candidate.get("bbox") or {}
        x1 = max(0, int(bbox.get("x1", 0)))
        y1 = max(0, int(bbox.get("y1", 0)))
        x2 = min(frame_w, int(bbox.get("x2", 0)))
        y2 = min(frame_h, int(bbox.get("y2", 0)))
        if x2 <= x1 or y2 <= y1:
            continue

        raw_normalized = plate_store.normalize_plate_text(
            candidate.get("normalizedPlate") or candidate.get("plateText")
        )
        normalized = raw_normalized
        quality_reason = candidate.get("qualityReason")
        matched = (
            plate_store.find_matching_plate(normalized)
            if normalized and not quality_reason
            else None
        )
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
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

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

        dedupe_subject = (
            (matched.get("id") if matched else None)
            or normalized
            or f"{x1}:{y1}:{x2}:{y2}"
        )
        dedupe_key = f"{camera_id}:{event_type}:{dedupe_subject}"
        if now - last_plate_log_by_key.get(dedupe_key, 0) < PLATE_LOG_COOLDOWN_SECONDS:
            continue
        last_plate_log_by_key[dedupe_key] = now

        if not snapshot_encode_attempted:
            snapshot_encode_attempted = True
            snapshot_ok, snapshot_buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if snapshot_ok:
                snapshot_jpeg = snapshot_buffer.tobytes()
            else:
                logger.warning(
                    "Plate read snapshot encoding failed; persisting without snapshot",
                    extra={"camera_id": camera_id},
                )

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
            quality_reason=_plate_quality_reason(
                quality_reason, raw_normalized, matched, match_kind
            ),
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
    hits = sum(
        1 for vote in plate_vote_window if vote.get("matched_id") == similar["id"]
    )
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
    plate_vote_window.append(
        {
            "normalized": normalized,
            "matched_id": matched_id,
            "timestamp": now,
            "confidence": candidate.get("confidence") or 0,
        }
    )
    cutoff = now - PLATE_VOTE_WINDOW_SECONDS
    del plate_vote_window[: max(0, len(plate_vote_window) - 20)]
    plate_vote_window[:] = [
        vote for vote in plate_vote_window if vote["timestamp"] >= cutoff
    ]


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
        suffix = (
            f" ({round(float(score) * 100)}%)"
            if isinstance(score, (int, float))
            else ""
        )
        return f"Similar match to registered plate {matched['normalizedPlate']}; OCR read {raw_normalized}{suffix}"
    return None


def _plate_event_type(
    normalized: str, matched: dict | None, quality_reason: str | None
) -> str:
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


def _effective_capture_fps(
    configured_fps: float,
    inference_fps: float,
    stream_type: str,
) -> float:
    """Cap non-inference RTSP frames without starving the AI cadence."""
    target = _positive_fps(configured_fps, 1.0)
    if stream_type != "rtsp":
        return target
    try:
        cap = float(os.environ.get("SAFETYLENS_RTSP_CAPTURE_FPS_CAP", "0"))
    except (TypeError, ValueError):
        cap = 0.0
    if not math.isfinite(cap) or cap <= 0:
        return target
    cap = min(60.0, cap)
    minimum_for_inference = min(target, _positive_fps(inference_fps, target))
    return max(minimum_for_inference, min(target, cap))


def _open_video_capture(
    video_source: str,
    stream_type: str,
    *,
    max_fps: float | None = None,
) -> cv2.VideoCapture:
    return open_video_capture(
        video_source,
        stream_type=stream_type,
        max_fps=max_fps,
    )


def _capture_policy_counts(cap: cv2.VideoCapture) -> tuple[int, int]:
    consume = getattr(cap, "consume_capture_policy_counts", None)
    if not callable(consume):
        return 0, 0
    try:
        dropped, duplicated = consume()
        return max(0, int(dropped)), max(0, int(duplicated))
    except (TypeError, ValueError):
        return 0, 0


def _read_live_frame_with_metadata(
    cap: cv2.VideoCapture,
    stream_type: str,
) -> tuple[bool, np.ndarray | None, int, int, int]:
    """Read the freshest available frame and its bounded capture telemetry.

    The timestamp is taken only after the final frame has crossed the decoder
    boundary.  It is therefore an honest decoded-ingress anchor, not a claim
    about the camera sensor or network timestamp.
    """
    ok, frame = cap.read()
    captured_ns = time.monotonic_ns()
    capture_drops, capture_duplicates = _capture_policy_counts(cap)
    if not ok or frame is None or stream_type != "rtsp":
        return ok, frame, captured_ns, capture_drops, capture_duplicates
    if getattr(cap, "delivers_latest_frame", False):
        return ok, frame, captured_ns, capture_drops, capture_duplicates
    if not callable(getattr(cap, "grab", None)):
        return ok, frame, captured_ns, capture_drops, capture_duplicates

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
        return ok, frame, captured_ns, capture_drops, capture_duplicates

    latest_ok, latest_frame = cap.retrieve()
    captured_ns = time.monotonic_ns()
    capture_drops += drained
    if latest_ok and latest_frame is not None:
        return True, latest_frame, captured_ns, capture_drops, capture_duplicates
    return ok, frame, captured_ns, capture_drops, capture_duplicates


def _read_live_frame(
    cap: cv2.VideoCapture, stream_type: str
) -> tuple[bool, np.ndarray | None]:
    """Compatibility wrapper for callers that do not need capture metadata."""
    ok, frame, _captured_ns, _dropped, _duplicated = _read_live_frame_with_metadata(
        cap,
        stream_type,
    )
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
    return (
        frame_bytes is not None
        and age is not None
        and age <= state.CAMERA_FRAME_STALE_SECONDS
    )


def _remote_frame_batch_size_hint(
    camera_id: str,
    cfg: dict,
    *,
    now_wall: float | None = None,
) -> int | None:
    """Bypass a batch rendezvous only when no other enabled frame is fresh."""
    now = time.time() if now_wall is None else now_wall
    for other_camera_id, camera in (cfg.get("cameras") or {}).items():
        other_camera_id = str(other_camera_id)
        if other_camera_id == camera_id:
            continue
        if isinstance(camera, dict) and not camera.get("enabled", True):
            continue
        updated_at = state.camera_frame_updated_at.get(other_camera_id)
        if (
            state.camera_frames.get(other_camera_id) is not None
            and updated_at is not None
            and now - updated_at <= state.CAMERA_FRAME_STALE_SECONDS
        ):
            return None
    return 1


def _freeze_inference_frame(frame: np.ndarray) -> np.ndarray:
    """Retain one immutable captured buffer across streaming and inference."""
    frame.setflags(write=False)
    return frame


def _decoded_ingress_monotonic_seconds(
    decoded_ingress_ns: object,
    *,
    completed_monotonic: float,
) -> float:
    """Recover frame capture age without trusting malformed/future metadata."""
    if isinstance(decoded_ingress_ns, bool):
        return completed_monotonic
    try:
        captured = float(decoded_ingress_ns) / 1_000_000_000.0
    except (TypeError, ValueError, OverflowError):
        return completed_monotonic
    if (
        not math.isfinite(captured)
        or captured <= 0
        or captured > completed_monotonic
    ):
        return completed_monotonic
    return captured


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
    frame_batch_size_hint: int | None = None,
    decoded_ingress_ns: int | None = None,
    submitted_ns: int | None = None,
    inference_signature: np.ndarray | None = None,
    submitted_monotonic: float | None = None,
    adaptive_mode: str | None = None,
    adaptive_state: str | None = None,
) -> dict:
    fresh_pose_results = None
    model_invocations = {}
    person_crop_telemetry: dict = {}
    annotated, detections, fresh_pose_results, model_invocations = (
        _run_grouped_inference(
            camera_id,
            frame,
            scheduled_plan,
            conf=yolo_conf,
            device=device,
            imgsz=inference_width,
            cfg=current_cfg,
            frame_batch_size_hint=frame_batch_size_hint,
            person_crop_telemetry=person_crop_telemetry,
        )
    )
    helmet_colour.annotate_helmet_colours(frame, detections, current_cam)
    effective_plan = scheduled_plan
    if model_invocations.get("rtdetr_phone_fallback"):
        effective_plan = deepcopy(scheduled_plan)
        effective_plan["run_coco_primary"] = True
        effective_plan["run_rtdetr_phone"] = False
        effective_plan.pop("partial_detection_capabilities", None)
        effective_plan["runtime_probe_reason"] = _RTDETR_PHONE_FALLBACK_REASON
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
        "person_crop_telemetry": person_crop_telemetry,
        "scheduled_plan": effective_plan,
        "schedule_state": schedule_state,
        "current_cam": current_cam,
        "current_cfg": current_cfg,
        "decoded_ingress_ns": decoded_ingress_ns,
        "submitted_ns": submitted_ns,
        "completed_ns": time.monotonic_ns(),
        "inference_signature": inference_signature,
        "submitted_monotonic": submitted_monotonic,
        "adaptive_mode": adaptive_mode,
        "adaptive_state": adaptive_state,
    }


def _fresh_rule_names_for_capabilities(
    current_cfg: dict,
    current_cam: dict,
    capabilities: set[str],
) -> set[str]:
    rule_map = {
        str(rule.get("id")): rule for rule in current_cfg.get("safety_rules", [])
    }
    assigned_rule_ids = current_cam.get("safety_rule_ids") or [
        rule_id
        for rule_id in rule_map
        if RULE_ID_TO_CAPABILITY.get(rule_id) in capabilities
    ]
    return {
        str(rule["name"])
        for rule_id in assigned_rule_ids
        if RULE_ID_TO_CAPABILITY.get(str(rule_id)) in capabilities
        and (rule := rule_map.get(str(rule_id)))
        and rule.get("enabled", True)
        and rule.get("type") == "alert"
        and rule.get("name")
    }


def _refresh_first_positive_anchors(
    first_positive_by_rule: dict[str, tuple[str, int]],
    current_violation_rules: Mapping[str, dict],
    active_violations: set[str],
    *,
    fresh_detection_evaluated: bool,
    fresh_fall_evaluated: bool,
    fresh_ppe_evaluated: bool,
    fresh_detection_rule_keys: set[str] | None,
    fresh_observation_monotonic_ns: int | None = None,
) -> None:
    fresh_anchor: tuple[str, int] | None = None
    tracked_rules = set(first_positive_by_rule) | set(current_violation_rules)
    for rule_key in tracked_rules:
        if not _rule_observation_is_fresh(
            rule_key,
            fresh_detection_evaluated=fresh_detection_evaluated,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
        ):
            continue
        if rule_key not in current_violation_rules or rule_key in active_violations:
            first_positive_by_rule.pop(rule_key, None)
            continue
        if rule_key not in first_positive_by_rule:
            if fresh_anchor is None:
                observed_wall = datetime.now(timezone.utc)
                observed_monotonic_ns = time.monotonic_ns()
                ingress_ns = fresh_observation_monotonic_ns
                if (
                    isinstance(ingress_ns, int)
                    and not isinstance(ingress_ns, bool)
                    and 0 <= ingress_ns <= observed_monotonic_ns
                ):
                    observed_wall -= timedelta(
                        seconds=(observed_monotonic_ns - ingress_ns) / 1_000_000_000
                    )
                    observed_monotonic_ns = ingress_ns
                fresh_anchor = (
                    observed_wall.isoformat(),
                    observed_monotonic_ns,
                )
            first_positive_by_rule[rule_key] = fresh_anchor


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
    first_positive_by_rule: dict[str, tuple[str, int]] | None = None,
    decoded_ingress_ns: int | None = None,
) -> bool:
    if first_positive_by_rule is None:
        first_positive_by_rule = {}
    frame_h, frame_w = frame.shape[:2]
    raw_partial_capabilities = scheduled_plan.get("partial_detection_capabilities")
    partial_capabilities = (
        set(raw_partial_capabilities)
        if isinstance(raw_partial_capabilities, list)
        else None
    )
    fresh_detection_rule_keys = (
        _fresh_rule_names_for_capabilities(
            current_cfg, current_cam, partial_capabilities
        )
        if partial_capabilities is not None
        else None
    )
    object_lifecycle_events = []
    if (
        "object_lifecycle" in scheduled_plan.get("capabilities", [])
        and (partial_capabilities is None or "object_lifecycle" in partial_capabilities)
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
    fresh_fall_evaluated = (
        scheduled_plan.get("run_pose_specialist") and fresh_pose_results is not None
    )
    fresh_ppe_evaluated = bool(
        scheduled_plan.get("run_ppe_specialist")
        or scheduled_plan.get("run_ppe_closed_set_candidate")
    )
    if fresh_fall_evaluated:
        fall_candidates = check_fall_detections(fresh_pose_results, camera_id, frame)
        has_fall_candidates = len(fall_candidates) > 0

    if detections or has_fall_candidates or object_lifecycle_events:
        candidates = []
        if detections:
            if scheduled_plan.get("run_ppe_specialist") or scheduled_plan.get(
                "run_ppe_closed_set_candidate"
            ):
                candidates.extend(
                    check_yoloe_violations(
                        detections,
                        camera_id,
                        frame_w,
                        frame_h,
                        capability_filter=partial_capabilities,
                    )
                )
            if partial_capabilities is None:
                candidates.extend(check_violations(detections, camera_id))
            else:
                candidates.extend(
                    check_violations(
                        detections,
                        camera_id,
                        capability_filter=partial_capabilities,
                    )
                )
            if "crowd_count_threshold" in scheduled_plan.get("capabilities", []) and (
                partial_capabilities is None
                or "crowd_count_threshold" in partial_capabilities
            ):
                candidates.extend(
                    _crowd_count_threshold_candidates(camera_id, detections)
                )

            if "zone_intrusion" in scheduled_plan.get("capabilities", []) and (
                partial_capabilities is None or "zone_intrusion" in partial_capabilities
            ):
                candidates.extend(
                    check_zone_intrusions(detections, camera_id, frame_w, frame_h)
                )

        candidates.extend(fall_candidates)
        candidates.extend(object_lifecycle_events)
        current_violation_rules = {
            candidate["rule"]: candidate for candidate in candidates
        }
        _advance_violation_window(
            violation_window,
            current_violation_rules,
            window_size=window_size,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
        )
        for rule_key in list(active_violations):
            if (
                fresh_detection_rule_keys is not None
                and not _is_ppe_violation_rule(rule_key)
                and rule_key not in fresh_detection_rule_keys
            ):
                continue
            if sum(violation_window.get(rule_key, [])[-window_size:]) < 2:
                active_violations.discard(rule_key)

        _refresh_first_positive_anchors(
            first_positive_by_rule,
            current_violation_rules,
            active_violations,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
            fresh_observation_monotonic_ns=decoded_ingress_ns,
        )

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
            confirmed_at = datetime.now(timezone.utc).isoformat()
            confirmed_monotonic_ns = time.monotonic_ns()
            first_positive_at, first_positive_monotonic_ns = first_positive_by_rule.get(
                rule_key,
                (confirmed_at, confirmed_monotonic_ns),
            )
            uses_fallback = any(decision.fallback for decision in decisions)
            if uses_fallback:
                if now - last_alert_by_rule.get(rule_key, 0) < alert_cooldown:
                    continue
            violation_bboxes = extract_violation_bboxes(
                candidate["rule"], detections, frame_w, frame_h, camera_id
            )
            if not snapshot_encoding_attempted:
                snapshot_encoding_attempted = True
                try:
                    snapshot_jpeg, clean_snapshot_jpeg = (
                        _encode_inference_snapshot_pair(
                            camera_id,
                            frame,
                            detections,
                            int(current_cfg.get("global", {}).get("jpeg_quality", 70)),
                            annotated_frame=annotated_frame,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Alert snapshot encoding failed",
                        extra={"camera_id": camera_id, "rule": candidate["rule"]},
                    )
            submitted_alert = False
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
                    first_positive_at=first_positive_at,
                    confirmed_at=confirmed_at,
                    _first_positive_monotonic_ns=first_positive_monotonic_ns,
                    _confirmed_monotonic_ns=confirmed_monotonic_ns,
                )
                if alert:
                    submitted_alert = True
                    if decision.rule_id:
                        policy_engine.start_rule_cooldown(
                            decision.rule_id,
                            camera_id=camera_id,
                        )
                        _persist_policy_trigger_after_alert(
                            alert,
                            decision.rule_id,
                        )
                    logger.debug(
                        "Detection alert queued",
                        extra={"camera_id": camera_id, "rule": candidate["rule"]},
                    )
            if submitted_alert:
                if uses_fallback:
                    last_alert_by_rule[rule_key] = now
                active_violations.add(rule_key)
                violation_window[rule_key] = []
                first_positive_by_rule.pop(rule_key, None)
            else:
                logger.warning(
                    "Detection alert submission failed; incident remains retryable",
                    extra={"camera_id": camera_id, "rule": candidate["rule"]},
                )

        for rule_key in list(violation_window):
            if (
                fresh_detection_rule_keys is not None
                and not _is_ppe_violation_rule(rule_key)
                and rule_key not in fresh_detection_rule_keys
            ):
                continue
            if (
                len(violation_window[rule_key]) >= window_size
                and sum(violation_window[rule_key]) == 0
            ):
                violation_window.pop(rule_key, None)
    else:
        _record_empty_violation_observation(
            violation_window,
            active_violations,
            window_size=window_size,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
        )
        _refresh_first_positive_anchors(
            first_positive_by_rule,
            {},
            active_violations,
            fresh_detection_evaluated=True,
            fresh_fall_evaluated=fresh_fall_evaluated,
            fresh_ppe_evaluated=fresh_ppe_evaluated,
            fresh_detection_rule_keys=fresh_detection_rule_keys,
            fresh_observation_monotonic_ns=decoded_ingress_ns,
        )
    return _alert_confirmation_required(active_violations, violation_window)


def video_processor(camera_id: str, stop_event: threading.Event):
    """Run one camera worker and never publish observations after it exits."""
    try:
        try:
            pipeline_telemetry.reset_camera(camera_id)
        except TelemetryCapacityError:
            # Telemetry is optional. Retaining the bounded 256-camera registry
            # must never prevent an otherwise licensed camera from running.
            logger.warning(
                "Camera pipeline telemetry capacity reached; worker will run without it",
                extra={"camera_id": camera_id},
            )
        _video_processor_loop(camera_id, stop_event)
    except Exception:
        logger.exception(
            "Camera worker exited unexpectedly", extra={"camera_id": camera_id}
        )
        raise
    finally:
        pipeline_telemetry.mark_camera_stopped(camera_id)
        _finalize_camera_worker_exit(camera_id, stop_event)


def _video_processor_loop(camera_id: str, stop_event: threading.Event):
    PPE_SUBSTITUTION_SCHEDULER.reset(camera_id)
    cfg = get_config()
    cam = cfg["cameras"][camera_id]
    stream_type = cam.get("stream_type", "file")
    video_source = (
        build_rtsp_url(cam, include_credentials=True)
        if stream_type == "rtsp"
        else str(VIDEO_DIR / cam["video"])
    )
    execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
    g = cfg["global"]
    configured_target_fps = _positive_fps(
        cam.get("fps", g["target_fps"]),
        _positive_fps(g["target_fps"], 1.0),
    )
    inference_fps = _positive_fps(
        cam.get(
            "inference_fps",
            g.get("inference_fps", max(1.0, configured_target_fps / 3)),
        ),
        max(1.0, configured_target_fps / 3),
    )
    target_fps = _effective_capture_fps(
        configured_target_fps,
        inference_fps,
        stream_type,
    )
    frame_interval = 1.0 / target_fps
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
    first_positive_by_rule: dict[str, tuple[str, int]] = {}
    window_size = 15
    frame_counter = 0
    initial_schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
    initial_execution_plan = _scheduled_execution_plan(
        execution_plan, initial_schedule_state
    )
    missing_model_keys = model_manager.missing_model_keys(
        initial_execution_plan["required_model_keys"]
    )
    reconnect_failures = 0
    safe_video_source = redact_video_source(video_source)
    state.clear_camera_connection_health(camera_id)
    state.clear_camera_inference_health(camera_id)
    connection_tracker = (
        CameraConnectionTracker(now=time.monotonic()) if stream_type == "rtsp" else None
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
        logger.warning(
            "Camera waiting on missing models",
            extra={"camera_id": camera_id, "missing_models": missing_model_keys},
        )
        return

    while not stop_event.is_set():
        state.camera_runtime_status[camera_id] = (
            "starting"
            if frame_counter == 0 and reconnect_failures == 0
            else "reconnecting"
        )
        cap = _open_video_capture(video_source, stream_type, max_fps=target_fps)
        if connection_tracker is not None:
            connection_tracker.capture_backend = _capture_backend_name(cap)
            capture_policy = _capture_policy_telemetry(cap)
            connection_tracker.appsink_latest_buffer_drops_observable = bool(
                capture_policy.get("appsinkLatestBufferDropsObservable", False)
            )
            connection_tracker.appsink_latest_buffer_drop_method = str(
                capture_policy.get(
                    "appsinkLatestBufferDropMethod", "unavailable"
                )
            )
            connection_tracker.capture_drop_accounting = str(
                capture_policy.get("captureDropAccounting", "unavailable")
            )
            connection_tracker.capture_drop_count_is_lower_bound = bool(
                capture_policy.get("captureDropCountIsLowerBound", True)
            )
            connection_tracker.decoder_policy_drop_accounting = str(
                capture_policy.get("decoderPolicyDropAccounting", "unknown")
            )
            _publish_camera_connection_health(camera_id, connection_tracker)
        if not cap.isOpened():
            cap.release()
            if stop_event.is_set():
                return
            _clear_camera_observation(camera_id)
            active_violations.clear()
            violation_window.clear()
            first_positive_by_rule.clear()
            last_annotated = None
            delay = (
                reconnect_delay_seconds(reconnect_failures, camera_id)
                if stream_type == "rtsp"
                else 5.0
            )
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

        shared_scheduler = None
        shared_scheduler_owner = None
        shared_scheduler_active = _shared_scheduler_enabled(get_config())
        inference_executor = None
        if shared_scheduler_active:
            try:
                shared_scheduler = _get_shared_scheduler(get_config())
                shared_scheduler_owner = uuid4().hex
                shared_scheduler.register(camera_id, shared_scheduler_owner)
            except Exception:
                shared_scheduler = None
                shared_scheduler_owner = None
                shared_scheduler_active = False
                logger.exception(
                    "Shared inference scheduler unavailable; using camera-local fallback",
                    extra={"camera_id": camera_id},
                )
        if not shared_scheduler_active:
            inference_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{camera_id}-inference",
                initializer=model_manager.warm_remote_inference_session,
            )
        pending_inference = None
        next_inference_at = inference_scheduler.next_inference_slot(
            camera_id,
            get_config(),
            inference_interval,
        )
        last_stream_checked_at = 0.0
        last_stream_published_at = 0.0
        last_clean_stream_encoded_at = 0.0
        stream_had_subscribers = False
        last_stream_signature = None
        cached_clean_stream_jpeg = None
        last_annotated = None
        last_inference_signature = None
        last_inference_submitted_at = None
        pending_inference_signature = None
        pending_inference_submitted_at = None
        last_mobile_phone_probe_at = None
        last_mobile_phone_probe_context_suppressed_at = None
        primary_person_tracker = PrimaryPersonTracker()
        runtime_cfg = get_config()
        adaptive_mode = _configured_global_runtime_mode(
            runtime_cfg,
            "adaptive_inference_mode",
            "SAFETYLENS_ADAPTIVE_INFERENCE_MODE",
            allowed={"off", "shadow", "active"},
        )
        keyframe_tracking_mode = _configured_global_runtime_mode(
            runtime_cfg,
            "keyframe_tracking_mode",
            "SAFETYLENS_KEYFRAME_TRACKING_MODE",
            allowed={"off", "shadow", "active"},
        )
        adaptive_controller = AdaptiveInferenceController(
            inference_fps,
            mode=adaptive_mode,
        )
        keyframe_tracker = PersonKLTTracker()
        tracker_result = KeyframeTrackerResult()
        adaptive_signature = None
        last_full_primary_detections: list[dict] = []
        last_full_primary_at: float | None = None
        alert_confirmation_required = False
        received_frame = False
        try:
            while cap.isOpened() and not stop_event.is_set():
                if not licensing.is_inference_allowed():
                    time.sleep(LICENSE_PAUSE_INTERVAL)
                    continue

                started = time.time()
                (
                    ok,
                    frame,
                    decoded_ingress_ns,
                    capture_drop_count,
                    capture_duplicate_count,
                ) = _read_live_frame_with_metadata(cap, stream_type)
                if not ok:
                    state.camera_runtime_status[camera_id] = (
                        "reconnecting" if stream_type == "rtsp" else "offline"
                    )
                    _clear_camera_observation(camera_id)
                    active_violations.clear()
                    violation_window.clear()
                    first_positive_by_rule.clear()
                    primary_person_tracker.clear()
                    keyframe_tracker.clear()
                    adaptive_controller.reset()
                    last_full_primary_detections = []
                    last_full_primary_at = None
                    last_annotated = None
                    break
                pipeline_telemetry.increment_camera_counter(
                    camera_id,
                    "decodedFrameCount",
                )
                if capture_drop_count:
                    pipeline_telemetry.increment_camera_counter(
                        camera_id,
                        "captureDropCount",
                        capture_drop_count,
                    )
                if capture_duplicate_count:
                    pipeline_telemetry.increment_camera_counter(
                        camera_id,
                        "captureDuplicateCount",
                        capture_duplicate_count,
                    )
                if not received_frame:
                    received_frame = True
                frame_height, frame_width = frame.shape[:2]
                state.update_camera_frame_dimensions(
                    camera_id, frame_width, frame_height
                )
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
                execution_plan = current_cam.get(
                    "execution_plan"
                ) or build_execution_plan(current_cam, current_cfg)
                schedule_state = _capability_schedule_state(
                    current_cam, current_cfg, execution_plan
                )
                scheduled_plan = _scheduled_execution_plan(
                    execution_plan, schedule_state
                )
                configured_target_fps = _coerce_fps(
                    current_cam.get(
                        "fps",
                        current_g.get("target_fps", configured_target_fps),
                    ),
                    configured_target_fps,
                )
                current_inference_fps = _coerce_fps(
                    current_cam.get(
                        "inference_fps",
                        current_g.get(
                            "inference_fps",
                            max(1.0, configured_target_fps / 3),
                        ),
                    ),
                    max(1.0, configured_target_fps / 3),
                )
                target_fps = _effective_capture_fps(
                    configured_target_fps,
                    current_inference_fps,
                    stream_type,
                )
                frame_interval = 1.0 / target_fps
                update_capture_fps = getattr(cap, "set_max_fps", None)
                if callable(update_capture_fps):
                    update_capture_fps(target_fps)
                current_inference_interval = 1.0 / current_inference_fps
                if current_inference_interval != inference_interval:
                    inference_fps = current_inference_fps
                    inference_interval = current_inference_interval
                    next_inference_at = inference_scheduler.next_inference_slot(
                        camera_id,
                        current_cfg,
                        inference_interval,
                    )
                desired_adaptive_mode = _configured_global_runtime_mode(
                    current_cfg,
                    "adaptive_inference_mode",
                    "SAFETYLENS_ADAPTIVE_INFERENCE_MODE",
                    allowed={"off", "shadow", "active"},
                )
                if current_inference_fps != adaptive_controller.configured_fps:
                    adaptive_controller = AdaptiveInferenceController(
                        current_inference_fps,
                        mode=desired_adaptive_mode,
                    )
                elif desired_adaptive_mode != adaptive_controller.mode:
                    adaptive_controller.set_mode(desired_adaptive_mode)
                adaptive_mode = desired_adaptive_mode
                keyframe_tracking_mode = _configured_global_runtime_mode(
                    current_cfg,
                    "keyframe_tracking_mode",
                    "SAFETYLENS_KEYFRAME_TRACKING_MODE",
                    allowed={"off", "shadow", "active"},
                )
                yolo_conf = current_g.get("yolo_conf", yolo_conf)
                jpeg_quality = int(current_g.get("jpeg_quality", jpeg_quality))
                inference_width = int(current_g.get("inference_width", inference_width))
                device = current_g.get("device", device)
                alert_cooldown = int(current_g.get("alert_cooldown", alert_cooldown))
                state.camera_runtime_status[camera_id] = "running"

                completed_inference = False
                completed_signature = None
                completed_submitted_at = None
                completed_scheduler_dispatched_at = None
                result = None
                completion_error: BaseException | None = None
                if (
                    shared_scheduler_active
                    and shared_scheduler is not None
                    and shared_scheduler_owner is not None
                ):
                    outcome = shared_scheduler.take_result(
                        camera_id,
                        shared_scheduler_owner,
                    )
                    if outcome is not None:
                        completed_inference = True
                        completion_error = outcome.error
                        result = outcome.value
                        completed_scheduler_dispatched_at = outcome.dispatched_at
                        if isinstance(result, Mapping):
                            completed_signature = result.get("inference_signature")
                            completed_submitted_at = result.get("submitted_monotonic")
                elif pending_inference and pending_inference.done():
                    completed_inference = True
                    completed_signature = pending_inference_signature
                    completed_submitted_at = pending_inference_submitted_at
                    try:
                        result = pending_inference.result()
                    except BaseException as exc:
                        completion_error = exc
                    pending_inference = None
                    pending_inference_signature = None
                    pending_inference_submitted_at = None

                if completed_inference:
                    if isinstance(
                        completion_error,
                        model_manager.RemoteInferenceOverloadedError,
                    ):
                        state.record_camera_inference_outcome(camera_id, "overloaded")
                        pipeline_telemetry.increment_camera_counter(
                            camera_id,
                            "admissionOverloadDropCount",
                        )
                        result = None
                    elif completion_error is not None or not isinstance(
                        result, Mapping
                    ):
                        state.record_camera_inference_outcome(camera_id, "failed")
                        pipeline_telemetry.increment_camera_counter(
                            camera_id,
                            "inferenceFailureCount",
                        )
                        logger.error(
                            "Detection failed",
                            extra={
                                "camera_id": camera_id,
                                "error_type": (
                                    type(completion_error).__name__
                                    if completion_error is not None
                                    else "InvalidInferenceResult"
                                ),
                            },
                        )
                        result = None
                    if result is not None:
                        state.record_camera_inference_outcome(camera_id, "success")
                        pipeline_telemetry.increment_camera_counter(
                            camera_id,
                            "inferenceCompletedCount",
                        )
                        _record_adaptive_inference_completion(
                            camera_id,
                            result.get("adaptive_mode"),
                            result.get("adaptive_state"),
                        )
                        completed_ns = result.get("completed_ns")
                        decoded_ns = result.get("decoded_ingress_ns")
                        submitted_ns = result.get("submitted_ns")
                        if completed_ns is not None and decoded_ns is not None:
                            pipeline_telemetry.observe_camera_elapsed_ns(
                                camera_id,
                                "decodedIngressToResultMs",
                                decoded_ns,
                                completed_ns,
                            )
                        if completed_ns is not None and submitted_ns is not None:
                            pipeline_telemetry.observe_camera_elapsed_ns(
                                camera_id,
                                "submitToResultMs",
                                submitted_ns,
                                completed_ns,
                            )
                        last_inference_signature = completed_signature
                        last_inference_submitted_at = completed_submitted_at
                        detections = result["detections"]
                        completed_at = time.monotonic()
                        result_plan = result["scheduled_plan"]
                        fresh_primary_keyframe = bool(
                            result_plan.get("run_coco_primary")
                            and not result_plan.get("partial_detection_capabilities")
                        )
                        keyframe_captured_at = _decoded_ingress_monotonic_seconds(
                            decoded_ns,
                            completed_monotonic=completed_at,
                        )
                        if (
                            adaptive_mode == "active"
                            and completed_scheduler_dispatched_at is not None
                        ):
                            adaptive_controller.record_dispatch(
                                completed_scheduler_dispatched_at
                            )
                        adaptive_controller.record_inference(
                            completed_at,
                            keyframe=fresh_primary_keyframe,
                            keyframe_at=(
                                keyframe_captured_at
                                if fresh_primary_keyframe
                                else None
                            ),
                        )
                        if fresh_primary_keyframe:
                            _, tracker_ttl = _rtdetr_phone_tracker_settings(
                                result["current_cfg"]
                            )
                            primary_person_tracker.update(
                                detections,
                                now=keyframe_captured_at,
                                ttl_seconds=tracker_ttl,
                            )
                            last_full_primary_detections = list(detections)
                            last_full_primary_at = keyframe_captured_at
                            if keyframe_tracking_mode != "off":
                                keyframe_tracker.seed(
                                    result["frame"],
                                    detections,
                                    keyframe_captured_at,
                                    zones=result["current_cam"].get("zones", []),
                                )
                        last_annotated = result.get("annotated_frame")
                        state.camera_detections[camera_id] = detections
                        _record_person_crop_runtime_counters(
                            camera_id,
                            result.get("person_crop_telemetry"),
                        )
                        _record_detection_history(
                            camera_id,
                            detections,
                            schedule_state=result["schedule_state"],
                            model_invocations=result["model_invocations"],
                            runtime_probe_reason=result["scheduled_plan"].get(
                                "runtime_probe_reason"
                            ),
                            runtime_probe_suppression_reason=result[
                                "scheduled_plan"
                            ].get("runtime_probe_suppression_reason"),
                            runtime_deferred_model_keys=result["scheduled_plan"].get(
                                "runtime_deferred_model_keys"
                            ),
                            person_crop_telemetry=result.get("person_crop_telemetry"),
                        )
                        alert_confirmation_required = _process_detection_observation(
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
                            first_positive_by_rule=first_positive_by_rule,
                            decoded_ingress_ns=decoded_ns,
                        )
                        if decoded_ns is not None:
                            pipeline_telemetry.observe_camera_elapsed_ns(
                                camera_id,
                                "decodedIngressToObservationMs",
                                decoded_ns,
                                time.monotonic_ns(),
                            )

                stream_fps = _configured_stream_fps(current_cam, current_g, target_fps)
                stream_now = time.monotonic()
                previous_stream_had_subscribers = stream_had_subscribers
                publish_stream_frame, stream_had_subscribers = _stream_publication_due(
                    camera_id,
                    last_stream_checked_at,
                    stream_now,
                    stream_fps,
                    stream_had_subscribers,
                )
                if publish_stream_frame:
                    last_stream_checked_at = stream_now
                    detections = state.camera_detections.get(camera_id, [])
                    stream_signature = None
                    if stream_had_subscribers:
                        (
                            publish_stream_frame,
                            stream_signature,
                            _stream_changed_fraction,
                        ) = _active_stream_change_decision(
                            frame,
                            last_stream_signature,
                            last_published_at=last_stream_published_at,
                            now=stream_now,
                            detections=detections,
                            subscriber_joined=(not previous_stream_had_subscribers),
                        )
                    else:
                        last_stream_signature = None
                if publish_stream_frame:
                    source_annotated = _preserved_source_annotation(
                        frame,
                        last_annotated,
                        execution_plan,
                    )
                    refresh_clean_stream_cache = _stream_clean_cache_due(
                        cached_clean_stream_jpeg,
                        last_clean_stream_encoded_at,
                        stream_now,
                    )
                    try:
                        (
                            cached_clean_stream_jpeg,
                            clean_stream_jpeg_encoded,
                        ) = _publish_stream_frame(
                            camera_id,
                            frame,
                            detections,
                            jpeg_quality=jpeg_quality,
                            source_annotated=source_annotated,
                            annotation_required=(
                                stream_had_subscribers
                                or bool(detections)
                                or source_annotated is not None
                            ),
                            cached_clean_jpeg=(
                                None
                                if refresh_clean_stream_cache
                                else cached_clean_stream_jpeg
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "Stream frame encoding failed",
                            extra={"camera_id": camera_id},
                        )
                    else:
                        last_stream_published_at = stream_now
                        if clean_stream_jpeg_encoded:
                            last_clean_stream_encoded_at = stream_now
                        if stream_signature is not None:
                            last_stream_signature = stream_signature
                now = time.monotonic()
                if keyframe_tracking_mode != "off":
                    tracker_result = keyframe_tracker.project(
                        frame,
                        now,
                        zones=current_cam.get("zones", []),
                    )
                    if (
                        keyframe_tracking_mode == "active"
                        and tracker_result.projections
                    ):
                        retained = [
                            detection
                            for detection in state.camera_detections.get(
                                camera_id,
                                [],
                            )
                            if str(detection.get("class", "")).lower() != "person"
                        ]
                        state.camera_detections[camera_id] = [
                            *retained,
                            *tracker_result.detections,
                        ]
                        last_annotated = None
                else:
                    tracker_result = KeyframeTrackerResult()

                if adaptive_mode == "off":
                    baseline_due = now >= next_inference_at
                    adaptive_decision = AdaptiveDecision(
                        mode="off",
                        state="active",
                        target_fps=current_inference_fps,
                        submit_now=baseline_due,
                        baseline_due=baseline_due,
                        adaptive_due=baseline_due,
                        urgent_reasons=(),
                    )
                else:
                    current_adaptive_signature = _frame_change_signature(frame)
                    adaptive_motion_score = _signature_changed_fraction(
                        current_adaptive_signature,
                        adaptive_signature,
                    )
                    adaptive_signature = current_adaptive_signature
                    tracked_person_present = bool(tracker_result.projections)
                    recently_detected_person = any(
                        str(detection.get("class", "")).lower() == "person"
                        for detection in last_full_primary_detections
                    ) and (
                        last_full_primary_at is not None
                        and now - last_full_primary_at <= 1.0
                    )
                    adaptive_decision = adaptive_controller.decide(
                        now,
                        AdaptiveSignals(
                            motion_score=adaptive_motion_score,
                            person_present=(
                                tracked_person_present or recently_detected_person
                            ),
                            tracker_confidence=(
                                tracker_result.aggregate_confidence
                                if tracker_result.projections
                                else None
                            ),
                            new_entry=tracker_result.new_foreground,
                            zone_entry=tracker_result.zone_entry,
                            possible_violation=(
                                bool(active_violations)
                                or _ppe_confirmation_required(
                                    active_violations,
                                    violation_window,
                                )
                            ),
                            alert_confirmation=alert_confirmation_required,
                            force_redetect=(
                                tracker_result.force_redetect
                                and bool(
                                    set(tracker_result.reasons)
                                    - {REDETECT_NO_KEYFRAME, REDETECT_NO_TRACKS}
                                )
                            ),
                        ),
                    )
                    if (
                        adaptive_mode == "shadow"
                        and adaptive_decision.adaptive_due
                    ):
                        # Shadow mode advances only its counterfactual clock;
                        # production admission still follows baseline cadence.
                        adaptive_controller.record_dispatch(now)
                if adaptive_mode != "off" or keyframe_tracking_mode != "off":
                    _record_adaptive_runtime_state(
                        camera_id,
                        adaptive_decision,
                        tracker_result,
                    )
                else:
                    _clear_adaptive_runtime_state(camera_id)
                adaptive_active = adaptive_mode == "active"
                schedule_due = (
                    adaptive_decision.submit_now
                    if adaptive_active
                    else now >= next_inference_at
                )
                inference_admission_available = (
                    shared_scheduler_active or pending_inference is None
                )
                if (
                    not adaptive_active
                    and inference_admission_available
                    and now - next_inference_at > inference_interval / 2
                ):
                    next_inference_at = inference_scheduler.next_inference_slot(
                        camera_id,
                        current_cfg,
                        inference_interval,
                        now=now,
                    )
                if not inference_admission_available and schedule_due:
                    pipeline_telemetry.increment_camera_counter(
                        camera_id,
                        "admissionBusyDropCount",
                    )
                elif not schedule_due:
                    pipeline_telemetry.increment_camera_counter(
                        camera_id,
                        "cadenceSkipCount",
                    )
                elif inference_admission_available:
                    if adaptive_active:
                        should_submit = True
                        inference_signature = current_adaptive_signature
                    else:
                        should_submit, inference_signature, _changed_fraction = (
                            _motion_adaptive_inference_decision(
                                frame,
                                last_inference_signature,
                                last_submitted_at=last_inference_submitted_at,
                                now=now,
                                alert_confirmation_required=alert_confirmation_required,
                            )
                        )
                    if not should_submit:
                        pipeline_telemetry.increment_camera_counter(
                            camera_id,
                            "motionSkipCount",
                        )
                        next_inference_at = inference_scheduler.next_inference_slot(
                            camera_id,
                            current_cfg,
                            inference_interval,
                            now=now + inference_interval / 2,
                        )
                    else:
                        min_track_hits, tracker_ttl = _rtdetr_phone_tracker_settings(
                            current_cfg
                        )
                        primary_context_detections = (
                            last_full_primary_detections
                            if last_full_primary_at is not None
                            and now - last_full_primary_at <= tracker_ttl
                            else []
                        )
                        runtime_plan = _context_gated_execution_plan(
                            scheduled_plan,
                            primary_context_detections,
                            camera=current_cam,
                            frame_w=frame.shape[1],
                            frame_h=frame.shape[0],
                        )
                        (
                            runtime_plan,
                            mobile_phone_probe_due,
                            mobile_phone_probe_context_suppressed,
                        ) = _mobile_phone_probe_execution_plan(
                            runtime_plan,
                            current_cfg,
                            now=now,
                            last_probe_at=last_mobile_phone_probe_at,
                            last_context_suppressed_at=(
                                last_mobile_phone_probe_context_suppressed_at
                            ),
                            previous_detections=primary_context_detections,
                        )
                        runtime_plan, _rtdetr_phone_selected = (
                            _rtdetr_phone_substitution_execution_plan(
                                camera_id,
                                runtime_plan,
                                current_cfg,
                                now=now,
                                stable_person_track=primary_person_tracker.has_stable_person(
                                    now=now,
                                    min_hits=min_track_hits,
                                    ttl_seconds=tracker_ttl,
                                ),
                            )
                        )
                        runtime_plan, _ppe_due, _ppe_substituted = (
                            _ppe_specialist_cadence_execution_plan(
                                camera_id,
                                runtime_plan,
                                current_cfg,
                                now=now,
                                stable_person_track=primary_person_tracker.has_stable_person(
                                    now=now,
                                    min_hits=min_track_hits,
                                    ttl_seconds=tracker_ttl,
                                ),
                                previous_detections=primary_context_detections,
                                confirmation_required=_ppe_confirmation_required(
                                    active_violations,
                                    violation_window,
                                ),
                            )
                        )
                        try:
                            submitted_ns = time.monotonic_ns()
                            frozen_frame = _freeze_inference_frame(frame)
                            if (
                                shared_scheduler_active
                                and shared_scheduler is not None
                                and shared_scheduler_owner is not None
                            ):
                                profile = _inference_batch_profile(
                                    camera_id,
                                    runtime_plan,
                                    current_cfg,
                                    yolo_conf=yolo_conf,
                                    device=device,
                                    inference_width=inference_width,
                                )
                                max_frame_age = _bounded_runtime_float(
                                    current_g.get(
                                        "shared_inference_max_frame_age_seconds",
                                        0.75,
                                    ),
                                    0.75,
                                    minimum=0.05,
                                    maximum=5.0,
                                )
                                captured_monotonic = decoded_ingress_ns / 1_000_000_000
                                urgent_reasons = tuple(adaptive_decision.urgent_reasons)

                                def run_scheduled_inference(
                                    batch_size: int,
                                    *,
                                    captured_frame=frozen_frame,
                                    captured_runtime_plan=runtime_plan,
                                    captured_schedule_state=schedule_state,
                                    captured_camera=current_cam,
                                    captured_config=current_cfg,
                                    captured_yolo_conf=yolo_conf,
                                    captured_device=device,
                                    captured_inference_width=inference_width,
                                    captured_decoded_ns=decoded_ingress_ns,
                                    captured_submitted_ns=submitted_ns,
                                    captured_signature=inference_signature,
                                    captured_submitted_at=now,
                                    captured_adaptive_mode=adaptive_decision.mode,
                                    captured_adaptive_state=adaptive_decision.state,
                                ):
                                    return _run_detection_job(
                                        camera_id,
                                        captured_frame,
                                        captured_runtime_plan,
                                        captured_schedule_state,
                                        captured_camera,
                                        captured_config,
                                        yolo_conf=captured_yolo_conf,
                                        device=captured_device,
                                        inference_width=captured_inference_width,
                                        last_face_log_by_key=last_face_log_by_key,
                                        last_plate_log_by_key=last_plate_log_by_key,
                                        plate_vote_window=plate_vote_window,
                                        frame_batch_size_hint=batch_size,
                                        decoded_ingress_ns=captured_decoded_ns,
                                        submitted_ns=captured_submitted_ns,
                                        inference_signature=captured_signature,
                                        submitted_monotonic=captured_submitted_at,
                                        adaptive_mode=captured_adaptive_mode,
                                        adaptive_state=captured_adaptive_state,
                                    )

                                offer_result = shared_scheduler.offer(
                                    camera_id,
                                    shared_scheduler_owner,
                                    InferenceWork(
                                        sequence=frame_counter,
                                        profile=profile,
                                        run=run_scheduled_inference,
                                        captured_at=captured_monotonic,
                                        expires_at=captured_monotonic + max_frame_age,
                                        urgent=bool(urgent_reasons),
                                        urgent_reasons=urgent_reasons,
                                    ),
                                )
                                if offer_result.status == OfferStatus.REPLACED:
                                    pipeline_telemetry.increment_camera_counter(
                                        camera_id,
                                        "latestSlotDropCount",
                                    )
                                elif offer_result.status == OfferStatus.STALE:
                                    pipeline_telemetry.increment_camera_counter(
                                        camera_id,
                                        "admissionStaleDropCount",
                                    )
                                elif not offer_result.accepted:
                                    pipeline_telemetry.increment_camera_counter(
                                        camera_id,
                                        "admissionBusyDropCount",
                                    )
                                if offer_result.accepted:
                                    _record_adaptive_inference_admission(
                                        camera_id,
                                        adaptive_decision,
                                    )
                                    if mobile_phone_probe_due:
                                        last_mobile_phone_probe_at = now
                                    if mobile_phone_probe_context_suppressed:
                                        last_mobile_phone_probe_context_suppressed_at = now
                            else:
                                if inference_executor is None:
                                    raise RuntimeError(
                                        "Camera-local inference executor is unavailable"
                                    )
                                pending_inference = inference_executor.submit(
                                    _run_detection_job,
                                    camera_id,
                                    frozen_frame,
                                    runtime_plan,
                                    schedule_state,
                                    current_cam,
                                    current_cfg,
                                    yolo_conf=yolo_conf,
                                    device=device,
                                    inference_width=inference_width,
                                    last_face_log_by_key=last_face_log_by_key,
                                    last_plate_log_by_key=last_plate_log_by_key,
                                    plate_vote_window=plate_vote_window,
                                    frame_batch_size_hint=_remote_frame_batch_size_hint(
                                        camera_id,
                                        current_cfg,
                                    ),
                                    decoded_ingress_ns=decoded_ingress_ns,
                                    submitted_ns=submitted_ns,
                                    inference_signature=inference_signature,
                                    submitted_monotonic=now,
                                    adaptive_mode=adaptive_decision.mode,
                                    adaptive_state=adaptive_decision.state,
                                )
                                pending_inference_signature = inference_signature
                                pending_inference_submitted_at = now
                                _record_adaptive_inference_admission(
                                    camera_id,
                                    adaptive_decision,
                                )
                                if adaptive_active:
                                    adaptive_controller.record_dispatch(now)
                                if mobile_phone_probe_due:
                                    last_mobile_phone_probe_at = now
                                if mobile_phone_probe_context_suppressed:
                                    last_mobile_phone_probe_context_suppressed_at = now
                        except RuntimeError as exc:
                            if not _is_executor_shutdown_error(exc):
                                raise
                            state.camera_runtime_status[camera_id] = "offline"
                            _clear_live_frame(camera_id)
                            logger.debug(
                                "Stopping video processor during executor shutdown",
                                extra={"camera_id": camera_id},
                            )
                            return
                        if not adaptive_active:
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
            if shared_scheduler is not None and shared_scheduler_owner is not None:
                try:
                    shared_scheduler.unregister(camera_id, shared_scheduler_owner)
                except Exception:
                    logger.exception(
                        "Could not unregister camera from shared inference scheduler",
                        extra={"camera_id": camera_id},
                    )
            elif inference_executor is not None:
                _drain_inference_executor(inference_executor, pending_inference)

        _clear_camera_observation(camera_id)
        active_violations.clear()
        violation_window.clear()
        first_positive_by_rule.clear()
        primary_person_tracker.clear()
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
            if (
                registered_thread is not current_thread
                or registered_stop_event is not stop_event
            ):
                return
            del state.camera_threads[cam_id]
        _clear_camera_observation(cam_id)
        state.clear_camera_connection_health(cam_id)
        state.clear_camera_inference_health(cam_id)
        state.camera_runtime_status[cam_id] = (
            "offline" if stop_event.is_set() else "error"
        )
    PPE_SUBSTITUTION_SCHEDULER.reset(cam_id)


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


def _vlm_companion_expected(cfg: dict, camera: dict | None) -> bool:
    if not camera or camera.get("demo") != "yolo+vlm":
        return False
    vlm_cfg = cfg.get("vlm")
    # Preserve direct legacy configs used by older integrations; normalized
    # production configs always contain an explicit enabled flag.
    return vlm_cfg is None or bool(vlm_cfg.get("enabled", False))


def start_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if _camera_lifecycle_shutdown.is_set():
            logger.debug(
                "Camera start rejected during lifecycle shutdown",
                extra={"camera_id": cam_id},
            )
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
                if (
                    _vlm_companion_expected(cfg, cam)
                    and not existing_stop_event.is_set()
                ):
                    start_vlm_for_camera(cam_id)
                return False
            _remove_worker_if_owned(state.camera_threads, cam_id, existing)

        execution_plan = cam.get("execution_plan") or build_execution_plan(cam, cfg)
        schedule_state = _capability_schedule_state(cam, cfg, execution_plan)
        scheduled_plan = _scheduled_execution_plan(execution_plan, schedule_state)
        missing_model_keys = model_manager.missing_model_keys(
            scheduled_plan["required_model_keys"]
        )
        _clear_camera_observation(cam_id)
        state.camera_detection_history[cam_id] = []
        state.camera_frame_updated_at.pop(cam_id, None)
        state.camera_schedule_telemetry.pop(cam_id, None)
        if missing_model_keys:
            state.camera_runtime_status[cam_id] = "awaiting_model_install"
            logger.warning(
                "Skipping camera start until models are ready",
                extra={"camera_id": cam_id, "missing_models": missing_model_keys},
            )
            return False

        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(
                target=video_processor, args=(cam_id, stop_evt), daemon=True
            )
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
                logger.exception(
                    "Failed to start video processor", extra={"camera_id": cam_id}
                )
                raise
        logger.info("Started video processor", extra={"camera_id": cam_id})
        if _vlm_companion_expected(cfg, cam):
            start_vlm_for_camera(cam_id)
        return True


def start_vlm_for_camera(cam_id: str) -> bool:
    with _camera_lifecycle_lock(cam_id):
        if _camera_lifecycle_shutdown.is_set():
            return False
        cfg = get_config_snapshot()
        cam = cfg.get("cameras", {}).get(cam_id)
        if (
            not cam
            or not cam.get("enabled", True)
            or not _vlm_companion_expected(cfg, cam)
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
            if existing[0].is_alive():
                return False
            _remove_worker_if_owned(state.vlm_threads, cam_id, existing)
        with _camera_lifecycle_fence_lock:
            if _camera_lifecycle_shutdown.is_set():
                return False
            stop_evt = threading.Event()
            thread = threading.Thread(
                target=vlm_worker, args=(cam_id, stop_evt), daemon=True
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
                    "Failed to start VLM worker", extra={"camera_id": cam_id}
                )
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
    cfg = get_config_snapshot()
    with _camera_worker_registry_lock:
        registered_vlm_ids = set(state.vlm_threads)
    for cam_id in set(cfg.get("cameras", {})) | registered_vlm_ids:
        if _camera_lifecycle_shutdown.is_set():
            return
        try:
            with _camera_lifecycle_lock(cam_id):
                if _camera_lifecycle_shutdown.is_set():
                    return
                current_cfg = get_config_snapshot()
                cam = current_cfg.get("cameras", {}).get(cam_id)
                enabled = bool(cam and cam.get("enabled", True))
                vlm_expected = bool(
                    enabled and _vlm_companion_expected(current_cfg, cam)
                )
                if not vlm_expected and cam_id in state.vlm_threads:
                    stop_vlm_for_camera(cam_id)
                if not enabled:
                    continue
                if not _worker_registration_active(state.camera_threads, cam_id):
                    start_camera(cam_id)
                    continue
                if vlm_expected and not _worker_registration_active(
                    state.vlm_threads, cam_id
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
        state.clear_camera_inference_health(cam_id)
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
