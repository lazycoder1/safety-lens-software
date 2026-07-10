"""Bounded, credential-safe video capture helpers."""

from __future__ import annotations

import hashlib
import os
from urllib.parse import urlsplit, urlunsplit

import cv2


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
RTSP_RECONNECT_BASE_SECONDS = _env_float(
    "SAFETYLENS_RTSP_RECONNECT_BASE_SECONDS",
    1.0,
    minimum=0.1,
    maximum=30.0,
)
RTSP_RECONNECT_MAX_SECONDS = _env_float(
    "SAFETYLENS_RTSP_RECONNECT_MAX_SECONDS",
    15.0,
    minimum=1.0,
    maximum=300.0,
)
CAMERA_STOP_TIMEOUT_SECONDS = max(
    5.0,
    (RTSP_OPEN_TIMEOUT_MS + RTSP_READ_TIMEOUT_MS) / 1_000.0 + 2.0,
)


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


def open_video_capture(source: str, *, stream_type: str):
    """Open RTSP through FFmpeg with bounded blocking; preserve file behavior."""
    if stream_type != "rtsp":
        return cv2.VideoCapture(source)

    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS])

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
    digest = hashlib.sha256(camera_id.encode("utf-8")).digest()[0]
    jitter = 0.9 + (digest / 255.0) * 0.2
    delay = RTSP_RECONNECT_BASE_SECONDS * (2**exponent) * jitter
    return min(RTSP_RECONNECT_MAX_SECONDS, delay)
