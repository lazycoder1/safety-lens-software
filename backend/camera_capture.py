"""Bounded, credential-safe video capture helpers."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import cv2


logger = logging.getLogger("rakshak_lens")


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(maximum, max(minimum, value))


RTSP_OPEN_TIMEOUT_MS = _env_int(
    "SAFETYLENS_RTSP_OPEN_TIMEOUT_MS",
    5_000,
    minimum=1_000,
    maximum=60_000,
)
RTSP_READ_TIMEOUT_MS = _env_int(
    "SAFETYLENS_RTSP_READ_TIMEOUT_MS",
    5_000,
    minimum=1_000,
    maximum=60_000,
)
RTSP_DECODE_THREADS = _env_int(
    "SAFETYLENS_RTSP_DECODE_THREADS",
    2,
    minimum=1,
    maximum=16,
)
RTSP_BUFFER_DRAIN_MAX_SECONDS = _env_float(
    "SAFETYLENS_RTSP_BUFFER_DRAIN_MAX_SECONDS",
    0.012,
    minimum=0.004,
    maximum=0.040,
)
RTSP_RECONNECT_BASE_SECONDS = _env_float(
    "SAFETYLENS_RTSP_RECONNECT_BASE_SECONDS",
    1.0,
    minimum=0.1,
    maximum=30.0,
)
RTSP_RECONNECT_MAX_SECONDS = _env_float(
    "SAFETYLENS_RTSP_RECONNECT_MAX_SECONDS",
    60.0,
    minimum=1.0,
    maximum=300.0,
)
RTSP_RECOVERY_STABLE_SECONDS = _env_float(
    "SAFETYLENS_RTSP_RECOVERY_STABLE_SECONDS",
    30.0,
    minimum=1.0,
    maximum=3_600.0,
)
RTSP_OUTAGE_SUMMARY_SECONDS = _env_float(
    "SAFETYLENS_RTSP_OUTAGE_SUMMARY_SECONDS",
    300.0,
    minimum=10.0,
    maximum=86_400.0,
)
CAMERA_STOP_TIMEOUT_SECONDS = max(
    5.0,
    (RTSP_OPEN_TIMEOUT_MS + RTSP_READ_TIMEOUT_MS) / 1_000.0 + 2.0,
)


def _rtsp_capture_backend() -> str:
    backend = os.environ.get("SAFETYLENS_RTSP_CAPTURE_BACKEND", "ffmpeg")
    normalized = str(backend).strip().lower()
    return normalized if normalized in {"ffmpeg", "nvdec", "auto"} else "ffmpeg"


def _rtsp_max_dimension() -> int:
    try:
        value = int(os.environ.get("SAFETYLENS_RTSP_MAX_DIMENSION", "0"))
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    return min(4_096, max(320, value))


def _open_gstreamer_capture(source: str):
    """Return a Jetson NVDEC capture, or None when the runtime is unavailable."""
    try:
        from gstreamer_capture import GStreamerCapture, nvdec_runtime_available
    except (ImportError, OSError):
        return None

    if not nvdec_runtime_available():
        return None
    try:
        capture = GStreamerCapture(
            source,
            open_timeout_ms=RTSP_OPEN_TIMEOUT_MS,
            read_timeout_ms=RTSP_READ_TIMEOUT_MS,
            max_dimension=_rtsp_max_dimension(),
        )
        if capture.isOpened():
            return capture
        capture.release()
        logger.warning("Jetson NVDEC capture did not open; using FFmpeg")
        return None
    except Exception:
        # Never include the source or exception text here: both GStreamer and
        # RTSP libraries may embed credentials in their error messages.
        logger.warning("Jetson NVDEC capture initialization failed; using FFmpeg")
        return None


def redact_video_source(source: str) -> str:
    """Return a diagnostic-safe source without credentials or query values."""
    try:
        parsed = urlsplit(source)
    except ValueError:
        return "<redacted-stream-source>" if "://" in source else source
    if not parsed.scheme or not parsed.netloc:
        return source
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port_value = parsed.port
    except ValueError:
        port_value = None
    port = f":{port_value}" if port_value else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def _capture_timeout_ms(value: int | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(60_000, max(250, parsed))


def open_video_capture(
    source: str,
    *,
    stream_type: str,
    prefer_hardware: bool = True,
    open_timeout_ms: int | None = None,
    read_timeout_ms: int | None = None,
):
    """Open RTSP through FFmpeg with bounded blocking; preserve file behavior."""
    if stream_type != "rtsp":
        return cv2.VideoCapture(source)

    if prefer_hardware and _rtsp_capture_backend() in {"nvdec", "auto"}:
        capture = _open_gstreamer_capture(source)
        if capture is not None:
            return capture

    bounded_open_timeout_ms = _capture_timeout_ms(
        open_timeout_ms,
        RTSP_OPEN_TIMEOUT_MS,
    )
    bounded_read_timeout_ms = _capture_timeout_ms(
        read_timeout_ms,
        RTSP_READ_TIMEOUT_MS,
    )

    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, bounded_open_timeout_ms])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, bounded_read_timeout_ms])
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        parameters.extend([cv2.CAP_PROP_N_THREADS, RTSP_DECODE_THREADS])

    capture = cv2.VideoCapture()
    try:
        capture.open(source, cv2.CAP_FFMPEG, parameters)
    except cv2.error:
        # The caller handles this like any other bounded open failure and logs
        # only the redacted source. Falling back to CAP_ANY can expose secrets.
        return capture
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def reconnect_delay_seconds(failure_count: int, camera_id: str) -> float:
    """Capped exponential backoff with stable per-camera staggering."""
    exponent = min(20, max(0, int(failure_count)))
    digest = hashlib.sha256(camera_id.encode("utf-8")).digest()
    jitter_value = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    # Keep saturated cameras spread across 85-100% of the configured ceiling.
    # Applying jitter after capping avoids the previous behavior where every
    # camera converged on exactly the same retry boundary.
    jitter = 0.85 + jitter_value * 0.15
    capped_delay = min(
        RTSP_RECONNECT_MAX_SECONDS,
        RTSP_RECONNECT_BASE_SECONDS * (2**exponent),
    )
    return min(RTSP_RECONNECT_MAX_SECONDS, max(0.0, capped_delay * jitter))


@dataclass(frozen=True)
class CameraConnectionEvent:
    """A safe transition or periodic summary emitted by an outage tracker."""

    kind: str
    failure_count: int = 0
    total_failure_count: int = 0
    suppressed_failure_count: int = 0
    outage_duration_seconds: float = 0.0


class CameraConnectionTracker:
    """Track one RTSP connection using monotonic time only.

    A decoded frame starts a stability window; it does not immediately forgive
    reconnect debt.  This prevents open -> one-frame -> drop loops from
    retrying forever at the base delay and from generating one warning per
    attempt.
    """

    def __init__(
        self,
        *,
        stable_window_seconds: float = RTSP_RECOVERY_STABLE_SECONDS,
        summary_interval_seconds: float = RTSP_OUTAGE_SUMMARY_SECONDS,
        now: float | None = None,
    ) -> None:
        self.stable_window_seconds = min(
            3_600.0,
            max(1.0, float(stable_window_seconds)),
        )
        self.summary_interval_seconds = min(
            86_400.0,
            max(10.0, float(summary_interval_seconds)),
        )
        started_at = time.monotonic() if now is None else float(now)
        self.outage_started_monotonic: float | None = None
        self.stable_started_monotonic: float | None = None
        self.last_summary_monotonic: float | None = None
        self.last_transition = "initializing"
        self.last_transition_monotonic = started_at
        self.capture_backend = "unknown"
        self.outage_failure_count = 0
        self.total_failure_count = 0
        self.suppressed_failure_count = 0

    @property
    def outage_active(self) -> bool:
        return self.outage_started_monotonic is not None

    @staticmethod
    def _increment(value: int) -> int:
        # Match the bounded public counter without importing shared state into
        # this low-level capture module.
        return min(2_147_483_647, value + 1)

    def record_failure(self, *, now: float | None = None) -> CameraConnectionEvent | None:
        current = time.monotonic() if now is None else float(now)
        self.stable_started_monotonic = None
        self.total_failure_count = self._increment(self.total_failure_count)

        if not self.outage_active:
            self.outage_started_monotonic = current
            self.last_summary_monotonic = current
            self.last_transition = "outage"
            self.last_transition_monotonic = current
            self.outage_failure_count = 1
            self.suppressed_failure_count = 0
            return self._event("outage", current)

        self.outage_failure_count = self._increment(self.outage_failure_count)
        last_summary = self.last_summary_monotonic
        if (
            last_summary is not None
            and current - last_summary >= self.summary_interval_seconds
        ):
            self.last_summary_monotonic = current
            return self._event("summary", current)
        self.suppressed_failure_count = self._increment(
            self.suppressed_failure_count
        )
        return None

    def record_frame(self, *, now: float | None = None) -> CameraConnectionEvent | None:
        current = time.monotonic() if now is None else float(now)
        if self.stable_started_monotonic is None:
            self.stable_started_monotonic = current
            if not self.outage_active and self.last_transition == "initializing":
                self.last_transition = "connected"
                self.last_transition_monotonic = current
                return self._event("connected", current)

        if not self.outage_active:
            return None
        if current - self.stable_started_monotonic < self.stable_window_seconds:
            return None

        recovered = self._event("recovered", current)
        self.outage_started_monotonic = None
        self.last_summary_monotonic = None
        self.last_transition = "recovered"
        self.last_transition_monotonic = current
        self.outage_failure_count = 0
        self.suppressed_failure_count = 0
        return recovered

    def _event(self, kind: str, now: float) -> CameraConnectionEvent:
        duration = (
            0.0
            if self.outage_started_monotonic is None
            else max(0.0, now - self.outage_started_monotonic)
        )
        return CameraConnectionEvent(
            kind=kind,
            failure_count=self.outage_failure_count,
            total_failure_count=self.total_failure_count,
            suppressed_failure_count=self.suppressed_failure_count,
            outage_duration_seconds=round(duration, 3),
        )
