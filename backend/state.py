"""Rakshak Lens mutable application state — shared across modules."""

import logging
import math
import threading
import time
from typing import Any, Optional

from fastapi import WebSocket

import model_manager

logger = logging.getLogger("rakshak_lens")

# ── Mutable state ───────────────────────────────────────────────────────────

alert_subscribers: list[WebSocket] = []
model: Any = None
yoloe_model: Any = None
camera_frames: dict[str, Optional[bytes]] = {}
camera_clean_frames: dict[str, Optional[bytes]] = {}
camera_frame_updated_at: dict[str, float] = {}
camera_frame_dimensions: dict[str, tuple[int, int]] = {}
camera_worker_started_at: dict[str, float] = {}
camera_detections: dict[str, list] = {}
camera_detection_history: dict[str, list[dict[str, Any]]] = {}
camera_schedule_telemetry: dict[str, dict[str, Any]] = {}
vlm_last_results: dict[str, dict] = {}
vlm_lock = threading.Lock()
camera_runtime_status: dict[str, str] = {}

CAMERA_FRAME_STALE_SECONDS = 5.0


def update_camera_frame_dimensions(camera_id: str, frame_width: int, frame_height: int) -> None:
    """Cache source-frame dimensions used by detection coordinates."""
    width = int(frame_width)
    height = int(frame_height)
    if width > 0 and height > 0:
        camera_frame_dimensions[camera_id] = (width, height)


def get_camera_frame_dimensions(camera_id: str) -> tuple[int, int] | None:
    return camera_frame_dimensions.get(camera_id)


def clear_camera_frame_dimensions(camera_id: str) -> None:
    camera_frame_dimensions.pop(camera_id, None)

# Connection telemetry intentionally contains only bounded counters, monotonic
# timestamps, and a small transition enum.  Stream URLs and backend error text
# must never enter this state because it is returned by the health endpoint and
# included in diagnostic bundles.
CAMERA_CONNECTION_COUNTER_MAX = 2_147_483_647
CAMERA_CONNECTION_AGE_MAX_SECONDS = 31_536_000.0
_CAMERA_CONNECTION_TRANSITIONS = {
    "initializing",
    "connected",
    "outage",
    "recovered",
    "unknown",
}
_CAMERA_CAPTURE_BACKENDS = {
    "unknown",
    "ffmpeg",
    "gstreamer_unknown",
    "gstreamer_software",
    "gstreamer_nvdec",
}
_CAPTURE_DROP_ACCOUNTING = {
    "unavailable",
    "application-drain-only",
    "videorate-only",
    "videorate-plus-appsink",
}
_DECODER_POLICY_DROP_ACCOUNTING = {
    "unknown",
    "not-applicable",
    "not-configured",
    "requested-not-applied",
    "configured-not-observable",
}
_APPSINK_DROP_METHODS = {
    "unavailable",
    "native-counter",
    "sink-pad-probe-lower-bound",
}
camera_connection_health: dict[str, dict[str, Any]] = {}
camera_connection_health_lock = threading.Lock()
camera_inference_health: dict[str, dict[str, Any]] = {}
camera_inference_health_lock = threading.Lock()

# Thread management: cam_id -> (Thread, Event)
camera_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
vlm_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}


def _bounded_connection_counter(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(CAMERA_CONNECTION_COUNTER_MAX, max(0, parsed))


def _safe_monotonic(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def update_camera_connection_health(
    camera_id: str,
    *,
    outage_active: bool,
    outage_started_monotonic: float | None,
    outage_failure_count: int,
    total_failure_count: int,
    suppressed_failure_count: int,
    last_transition: str,
    last_transition_monotonic: float | None,
    capture_backend: str = "unknown",
    appsink_latest_buffer_drops_observable: bool = False,
    appsink_latest_buffer_drop_method: str = "unavailable",
    capture_drop_accounting: str = "unavailable",
    capture_drop_count_is_lower_bound: bool = True,
    decoder_policy_drop_accounting: str = "unknown",
) -> None:
    """Publish a credential-safe, bounded connection telemetry snapshot."""
    transition = str(last_transition)
    if transition not in _CAMERA_CONNECTION_TRANSITIONS:
        transition = "unknown"
    backend = str(capture_backend)
    if backend not in _CAMERA_CAPTURE_BACKENDS:
        backend = "unknown"
    drop_accounting = str(capture_drop_accounting)
    if drop_accounting not in _CAPTURE_DROP_ACCOUNTING:
        drop_accounting = "unavailable"
    decoder_drop_accounting = str(decoder_policy_drop_accounting)
    if decoder_drop_accounting not in _DECODER_POLICY_DROP_ACCOUNTING:
        decoder_drop_accounting = "unknown"
    appsink_drop_method = str(appsink_latest_buffer_drop_method)
    if appsink_drop_method not in _APPSINK_DROP_METHODS:
        appsink_drop_method = "unavailable"
    snapshot = {
        "outage_active": bool(outage_active),
        "outage_started_monotonic": _safe_monotonic(outage_started_monotonic),
        "outage_failure_count": _bounded_connection_counter(outage_failure_count),
        "total_failure_count": _bounded_connection_counter(total_failure_count),
        "suppressed_failure_count": _bounded_connection_counter(
            suppressed_failure_count
        ),
        "last_transition": transition,
        "last_transition_monotonic": _safe_monotonic(last_transition_monotonic),
        "capture_backend": backend,
        "appsink_latest_buffer_drops_observable": bool(
            appsink_latest_buffer_drops_observable
        ),
        "appsink_latest_buffer_drop_method": appsink_drop_method,
        "capture_drop_accounting": drop_accounting,
        "capture_drop_count_is_lower_bound": bool(capture_drop_count_is_lower_bound),
        "decoder_policy_drop_accounting": decoder_drop_accounting,
    }
    with camera_connection_health_lock:
        camera_connection_health[camera_id] = snapshot


def clear_camera_connection_health(camera_id: str) -> None:
    """Forget telemetry when a camera stops or changes away from RTSP."""
    with camera_connection_health_lock:
        camera_connection_health.pop(camera_id, None)


def record_camera_inference_outcome(
    camera_id: str,
    outcome: str,
    *,
    now_monotonic: float | None = None,
) -> None:
    """Record bounded, credential-free inference capacity telemetry."""
    if outcome not in {"success", "overloaded", "failed"}:
        raise ValueError(f"Unknown inference outcome: {outcome}")
    now = _safe_monotonic(now_monotonic)
    if now is None:
        now = time.monotonic()
    counter_key = f"{outcome}_count"
    with camera_inference_health_lock:
        current = camera_inference_health.get(camera_id, {})
        snapshot = dict(current)
        snapshot[counter_key] = _bounded_connection_counter(
            snapshot.get(counter_key, 0)
        )
        snapshot[counter_key] = _bounded_connection_counter(
            snapshot[counter_key] + 1
        )
        snapshot[f"last_{outcome}_monotonic"] = now
        camera_inference_health[camera_id] = snapshot


def clear_camera_inference_health(camera_id: str) -> None:
    with camera_inference_health_lock:
        camera_inference_health.pop(camera_id, None)


def get_camera_inference_health(
    camera_id: str,
    *,
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    with camera_inference_health_lock:
        stored = camera_inference_health.get(camera_id)
        snapshot = stored.copy() if stored is not None else {}
    now = _safe_monotonic(now_monotonic)
    if now is None:
        now = time.monotonic()

    def age(outcome: str) -> float | None:
        anchor = _safe_monotonic(snapshot.get(f"last_{outcome}_monotonic"))
        if anchor is None:
            return None
        return round(
            min(CAMERA_CONNECTION_AGE_MAX_SECONDS, max(0.0, now - anchor)),
            3,
        )

    return {
        "successCount": _bounded_connection_counter(snapshot.get("success_count", 0)),
        "overloadDropCount": _bounded_connection_counter(snapshot.get("overloaded_count", 0)),
        "failureCount": _bounded_connection_counter(snapshot.get("failed_count", 0)),
        "lastSuccessAgeSeconds": age("success"),
        "lastOverloadDropAgeSeconds": age("overloaded"),
        "lastFailureAgeSeconds": age("failed"),
    }


def get_camera_connection_health(
    camera_id: str,
    *,
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    """Return public connection telemetry without internal monotonic anchors."""
    with camera_connection_health_lock:
        stored = camera_connection_health.get(camera_id)
        snapshot = stored.copy() if stored is not None else None
    if snapshot is None:
        return {
            "outageActive": False,
            "failureCount": 0,
            "totalFailureCount": 0,
            "suppressedFailureCount": 0,
            "outageAgeSeconds": None,
            "lastTransition": "unknown",
            "lastTransitionAgeSeconds": None,
            "captureBackend": "unknown",
            "appsinkLatestBufferDropsObservable": False,
            "appsinkLatestBufferDropMethod": "unavailable",
            "captureDropAccounting": "unavailable",
            "captureDropCountIsLowerBound": True,
            "decoderPolicyDropAccounting": "unknown",
        }

    now = _safe_monotonic(now_monotonic)
    if now is None:
        now = time.monotonic()

    def bounded_age(anchor: float | None) -> float | None:
        if anchor is None:
            return None
        return round(
            min(CAMERA_CONNECTION_AGE_MAX_SECONDS, max(0.0, now - anchor)),
            3,
        )

    outage_active = bool(snapshot["outage_active"])
    return {
        "outageActive": outage_active,
        "failureCount": snapshot["outage_failure_count"],
        "totalFailureCount": snapshot["total_failure_count"],
        "suppressedFailureCount": snapshot["suppressed_failure_count"],
        "outageAgeSeconds": (
            bounded_age(snapshot["outage_started_monotonic"])
            if outage_active
            else None
        ),
        "lastTransition": snapshot["last_transition"],
        "lastTransitionAgeSeconds": bounded_age(
            snapshot["last_transition_monotonic"]
        ),
        "captureBackend": snapshot.get("capture_backend", "unknown"),
        "appsinkLatestBufferDropsObservable": bool(
            snapshot.get("appsink_latest_buffer_drops_observable", False)
        ),
        "appsinkLatestBufferDropMethod": snapshot.get(
            "appsink_latest_buffer_drop_method", "unavailable"
        ),
        "captureDropAccounting": snapshot.get(
            "capture_drop_accounting", "unavailable"
        ),
        "captureDropCountIsLowerBound": bool(
            snapshot.get("capture_drop_count_is_lower_bound", True)
        ),
        "decoderPolicyDropAccounting": snapshot.get(
            "decoder_policy_drop_accounting", "unknown"
        ),
    }


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
