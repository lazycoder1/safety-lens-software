#!/usr/bin/env python3
"""Collect a bounded, credential-free live pipeline benchmark.

The collector polls only the public health endpoint and allowlists numeric
runtime telemetry.  It never stores health payloads, camera frames, endpoint
URLs, request headers, or authentication values in its report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID


SCHEMA_VERSION = 1
MAX_HEALTH_BYTES = 4 * 1024 * 1024
MAX_DURATION_SECONDS = 24 * 60 * 60
MAX_HEALTH_SAMPLES = 100_000
MIN_POLL_INTERVAL_SECONDS = 0.1
MIN_HEALTH_DURATION_COVERAGE = 0.9
MIN_HEALTH_SAMPLE_COVERAGE = 0.8
MIN_SERVER_SNAPSHOT_DURATION_COVERAGE = 0.8
MAX_HEALTH_SNAPSHOT_AGE_SECONDS = 10.0
MIN_RESOURCE_DURATION_COVERAGE = 0.9
MIN_RESOURCE_SAMPLE_COVERAGE = 0.8
# Match the bounded identifiers accepted by pipeline telemetry, including
# legacy human-readable IDs such as "Gate 1" while rejecting control text.
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.: -]{0,127}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
COUNTER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
CAPTURE_BACKENDS = {
    "unknown",
    "ffmpeg",
    "gstreamer_unknown",
    "gstreamer_software",
    "gstreamer_nvdec",
}
CAPTURE_DROP_ACCOUNTING = {
    "unavailable",
    "application-drain-only",
    "videorate-only",
    "videorate-plus-appsink",
}
DECODER_POLICY_DROP_ACCOUNTING = {
    "unknown",
    "not-applicable",
    "not-configured",
    "requested-not-applied",
    "configured-not-observable",
}
APPSINK_DROP_METHODS = {
    "unavailable",
    "native-counter",
    "sink-pad-probe-lower-bound",
}

CAMERA_COUNTER_NAMES = (
    "decodedFrameCount",
    "captureDropCount",
    "captureDuplicateCount",
    "latestSlotDropCount",
    "admissionBusyDropCount",
    "admissionStaleDropCount",
    "admissionOverloadDropCount",
    "schedulerLifecycleDropCount",
    "motionSkipCount",
    "cadenceSkipCount",
    "inferenceCompletedCount",
    "inferenceFailureCount",
    "adaptiveQuietObservationCount",
    "adaptiveUncertainObservationCount",
    "adaptiveActiveObservationCount",
    "adaptiveQuietAdmissionCount",
    "adaptiveUncertainAdmissionCount",
    "adaptiveActiveAdmissionCount",
    "adaptiveQuietInferenceCount",
    "adaptiveUncertainInferenceCount",
    "adaptiveActiveInferenceCount",
    "trackerProjectionFrameCount",
    "trackerProjectedPersonCount",
    "trackerForceRedetectSignalCount",
    "ppePersonCropAttemptCount",
    "phonePersonCropAttemptCount",
    "personCropFallbackCount",
    "personCropFullFrameInvocationCount",
)
ADAPTIVE_STATES = ("quiet", "uncertain", "active")
DROP_COUNTER_NAMES = (
    "captureDropCount",
    "latestSlotDropCount",
    "admissionBusyDropCount",
    "admissionStaleDropCount",
    "admissionOverloadDropCount",
    "schedulerLifecycleDropCount",
)
POLICY_SKIP_COUNTER_NAMES = ("motionSkipCount", "cadenceSkipCount")
CAMERA_LATENCY_NAMES = (
    "decodedIngressToResultMs",
    "decodedIngressToObservationMs",
    "submitToResultMs",
)
ALERT_LATENCY_NAMES = (
    "firstPositiveToConfirmedMs",
    "confirmedToPersistedMs",
    "firstPositiveToPersistedMs",
    "firstPositiveToProviderSuccessMs",
)
ALERT_LATENCY_FALLBACK_NAMES = {
    "firstPositiveToProviderSuccessMs": "firstPositiveToDeliveryHandoffMs",
}
ALERT_DELIVERY_COUNTER_NAMES = (
    "eligibleCount",
    "deliveredCount",
    "terminalCount",
    "cancelledCount",
    "failedAttemptCount",
    "untrackedFailureAttemptCount",
    "persistenceCensoredCount",
    "outcomeCensoredCount",
    "evictedPendingCount",
)
VLM_COUNTER_NAMES = (
    "offered",
    "replaced",
    "capacityDropped",
    "staleDropped",
    "circuitDropped",
    "processed",
    "failed",
    "lateResultDiscarded",
    "resultCallbackFailed",
)
VLM_GAUGE_NAMES = (
    "running",
    "accepting",
    "pendingCameras",
    "pendingCapacity",
    "active",
    "circuitOpen",
    "circuitRetryAfterSeconds",
)
SCHEDULER_GAUGE_NAMES = (
    "running",
    "accepting",
    "registeredCameras",
    "queued",
    "inflight",
    "completedUnconsumed",
    "maxWorkers",
    "batch2WaitMs",
    "singletonWaitMs",
)
SCHEDULER_LATENCY_NAMES = ("queue", "service", "frameAge")
SCHEDULER_COUNTER_NAMES = (
    "starts",
    "registrations",
    "unregister_queue_drops",
    "unregister_result_drops",
    "unregistrations",
    "offers_not_running",
    "offers_not_registered",
    "offers_owner_mismatch",
    "offers",
    "stale_offer_drops",
    "out_of_order_offers",
    "replaced_queued",
    "accepted_offers",
    "results_taken",
    "shutdown_queue_drops",
    "stops",
    "dispatched",
    "urgent_dispatched",
    "batches_1",
    "batches_2",
    "batches_4",
    "fenced_completions",
    "completed",
    "failed",
    "stale_drops",
    "stale_dispatch_drops",
    "stale_before_run_drops",
    "stale_completion_drops",
)
INFERENCE_TRANSPORT_COUNTERS = {
    "primaryFrameBatch": ("batch2_succeeded", "batch4_succeeded"),
    "ppeFrameBatch": ("batch2_succeeded", "batch4_succeeded"),
    "specialistFrameBatch": ("batch2_succeeded", "batch4_succeeded"),
    "rtdetrPhoneFrameBatch": ("batch1_executed", "batch2_executed"),
}

_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB")
_CPU_RE = re.compile(r"\bCPU\s+\[([^\]]*)\]")
_GPU_RE = re.compile(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%")
_VIC_RE = re.compile(r"\bVIC_FREQ\s+(\d+(?:\.\d+)?)%")
_TEMP_RE = re.compile(r"\b([A-Za-z0-9_]{1,32})@(-?\d+(?:\.\d+)?)C")
_POWER_MW_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]{1,31})\s+(\d+(?:\.\d+)?)mW"
    r"(?:/(\d+(?:\.\d+)?)mW)?"
)
_LEGACY_POWER_RE = re.compile(
    r"\b(POM_[A-Z0-9_]{1,27})\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\b"
)
_THERMAL_MARKER_RE = re.compile(
    r"\b(?:throttl\w*|overheat\w*|thermal[_ -]?(?:trip|throttle|warning)\w*)\b",
    re.IGNORECASE,
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0 or parsed != value:
        return None
    return parsed


def _safe_camera_id(value: object) -> str | None:
    if not isinstance(value, str) or CAMERA_ID_RE.fullmatch(value) is None:
        return None
    return value


def _safe_frame_dimension(value: object) -> int | None:
    parsed = _nonnegative_int(value)
    if parsed is None or parsed < 1 or parsed > 65_535:
        return None
    return parsed


def _validated_uuid(value: object) -> str | None:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _parse_timestamp_seconds(value: object) -> float | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    timestamp = parsed.timestamp()
    return timestamp if math.isfinite(timestamp) else None


def _safe_histogram(value: object, expected_bucket_count: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_counts = value.get("bucketCounts")
    if not isinstance(raw_counts, list) or len(raw_counts) != expected_bucket_count:
        return None
    counts = [_nonnegative_int(item) for item in raw_counts]
    if any(item is None for item in counts):
        return None
    result: dict[str, Any] = {
        "bucketCounts": [int(item) for item in counts if item is not None]
    }
    invalid_count = _nonnegative_int(value.get("invalidCount"))
    if invalid_count is not None:
        result["invalidCount"] = invalid_count
    return result


def _safe_bounds(value: object) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    bounds = [_finite_number(item) for item in value]
    if any(item is None or item <= 0 for item in bounds):
        return None
    parsed = [float(item) for item in bounds if item is not None]
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        return None
    return parsed


def _safe_counter_map(
    value: object,
    *,
    allowed: Iterable[str] | None = None,
) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    allowed_set = set(allowed) if allowed is not None else None
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if COUNTER_NAME_RE.fullmatch(key) is None:
            continue
        if allowed_set is not None and key not in allowed_set:
            continue
        parsed = _nonnegative_int(raw_value)
        if parsed is not None:
            result[key] = parsed
    return result


def _safe_latency_window(value: object) -> dict[str, float | int | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float | int | None] = {}
    for key in ("sampleCount", "medianMs", "p95Ms", "p99Ms", "maxMs"):
        parsed = _finite_number(value.get(key))
        if parsed is not None and parsed >= 0:
            result[key] = int(parsed) if key == "sampleCount" else round(parsed, 6)
        elif value.get(key) is None and key != "sampleCount":
            result[key] = None
    return result


def _sanitize_health_payload(
    payload: object,
    *,
    received_wall_seconds: float,
) -> dict[str, Any]:
    """Retain only bounded operational values from an untrusted health response."""

    if not isinstance(payload, dict):
        raise ValueError("health response must be a JSON object")

    pipeline = payload.get("pipelineTelemetry")
    pipeline = pipeline if isinstance(pipeline, dict) else {}
    frame_bounds = _safe_bounds(pipeline.get("frameHistogramBoundsMs"))
    alert_bounds = _safe_bounds(pipeline.get("alertHistogramBoundsMs"))
    epoch = _validated_uuid(pipeline.get("telemetryEpoch"))

    pipeline_cameras: dict[str, Any] = {}
    raw_cameras = pipeline.get("cameras")
    if isinstance(raw_cameras, dict) and frame_bounds is not None:
        for raw_camera_id, raw_camera in raw_cameras.items():
            camera_id = _safe_camera_id(raw_camera_id)
            if camera_id is None or not isinstance(raw_camera, dict):
                continue
            generation = _nonnegative_int(raw_camera.get("generation"))
            if generation is None or generation < 1:
                continue
            latency: dict[str, Any] = {}
            raw_latency = raw_camera.get("latency")
            if isinstance(raw_latency, dict):
                for latency_name in CAMERA_LATENCY_NAMES:
                    histogram = _safe_histogram(
                        raw_latency.get(latency_name),
                        len(frame_bounds) + 1,
                    )
                    if histogram is not None:
                        latency[latency_name] = histogram
            pipeline_cameras[camera_id] = {
                "generation": generation,
                "active": bool(raw_camera.get("active")),
                "counters": _safe_counter_map(
                    raw_camera.get("counters"), allowed=CAMERA_COUNTER_NAMES
                ),
                "latency": latency,
            }

    alert_latency: dict[str, Any] = {}
    raw_alerts = pipeline.get("alerts")
    raw_alert_latency = (
        raw_alerts.get("latency") if isinstance(raw_alerts, dict) else None
    )
    if isinstance(raw_alert_latency, dict) and alert_bounds is not None:
        for latency_name in ALERT_LATENCY_NAMES:
            raw_histogram = raw_alert_latency.get(latency_name)
            if raw_histogram is None:
                fallback_name = ALERT_LATENCY_FALLBACK_NAMES.get(latency_name)
                if fallback_name is not None:
                    raw_histogram = raw_alert_latency.get(fallback_name)
            histogram = _safe_histogram(
                raw_histogram,
                len(alert_bounds) + 1,
            )
            if histogram is not None:
                alert_latency[latency_name] = histogram

    alert_delivery_coverage: dict[str, Any] = {}
    raw_delivery_coverage = (
        raw_alerts.get("deliveryCoverage")
        if isinstance(raw_alerts, dict)
        else None
    )
    if isinstance(raw_delivery_coverage, dict):
        pending = _nonnegative_int(raw_delivery_coverage.get("pending"))
        counters = _safe_counter_map(
            raw_delivery_coverage.get("counters"),
            allowed=ALERT_DELIVERY_COUNTER_NAMES,
        )
        if pending is not None and counters:
            alert_delivery_coverage = {
                "unit": "initial-external-delivery-target",
                "pending": pending,
                "counters": counters,
            }

    diagnostics_cameras: dict[str, Any] = {}
    raw_diagnostics_cameras = payload.get("cameras")
    if isinstance(raw_diagnostics_cameras, list):
        for raw_camera in raw_diagnostics_cameras:
            if not isinstance(raw_camera, dict):
                continue
            camera_id = _safe_camera_id(raw_camera.get("id"))
            if camera_id is None:
                continue
            age = _finite_number(raw_camera.get("lastFrameAgeSeconds"))
            stream_type = raw_camera.get("streamType")
            if stream_type not in {"rtsp", "file"}:
                stream_type = "unknown"
            raw_connection = raw_camera.get("connection")
            raw_connection = (
                raw_connection if isinstance(raw_connection, dict) else {}
            )
            capture_backend = raw_connection.get("captureBackend")
            if capture_backend not in CAPTURE_BACKENDS:
                capture_backend = "unknown"
            drop_accounting = raw_connection.get("captureDropAccounting")
            if drop_accounting not in CAPTURE_DROP_ACCOUNTING:
                drop_accounting = "unavailable"
            decoder_drop_accounting = raw_connection.get(
                "decoderPolicyDropAccounting"
            )
            if decoder_drop_accounting not in DECODER_POLICY_DROP_ACCOUNTING:
                decoder_drop_accounting = "unknown"
            appsink_drop_method = raw_connection.get(
                "appsinkLatestBufferDropMethod"
            )
            if appsink_drop_method not in APPSINK_DROP_METHODS:
                appsink_drop_method = "unavailable"
            diagnostics_cameras[camera_id] = {
                "frameFresh": bool(raw_camera.get("frameFresh")),
                "workerRunning": bool(raw_camera.get("workerRunning")),
                "lastFrameAgeSeconds": (max(0.0, age) if age is not None else None),
                "streamType": stream_type,
                "frameWidth": _safe_frame_dimension(raw_camera.get("frameWidth")),
                "frameHeight": _safe_frame_dimension(raw_camera.get("frameHeight")),
                "connection": {
                    "captureBackend": capture_backend,
                    "hardwareAccelerationActive": (
                        raw_connection.get("hardwareAccelerationActive") is True
                    ),
                    "hardwareFallback": (
                        raw_connection.get("hardwareFallback") is True
                    ),
                    "appsinkLatestBufferDropsObservable": (
                        raw_connection.get("appsinkLatestBufferDropsObservable")
                        is True
                    ),
                    "appsinkLatestBufferDropMethod": appsink_drop_method,
                    "captureDropAccounting": drop_accounting,
                    "captureDropCountIsLowerBound": (
                        raw_connection.get("captureDropCountIsLowerBound") is not False
                    ),
                    "decoderPolicyDropAccounting": decoder_drop_accounting,
                },
            }

    scheduler_raw = payload.get("sharedInferenceScheduler")
    scheduler_raw = scheduler_raw if isinstance(scheduler_raw, dict) else {}
    scheduler: dict[str, Any] = {
        "counters": _safe_counter_map(
            scheduler_raw.get("counters"), allowed=SCHEDULER_COUNTER_NAMES
        ),
        "gauges": {},
        "latency": {},
    }
    for key in SCHEDULER_GAUGE_NAMES:
        value = scheduler_raw.get(key)
        if isinstance(value, bool):
            scheduler["gauges"][key] = value
        else:
            number = _finite_number(value)
            if number is not None and number >= 0:
                scheduler["gauges"][key] = number
    scheduler_latency_raw = scheduler_raw.get("latency")
    if isinstance(scheduler_latency_raw, dict):
        for name in SCHEDULER_LATENCY_NAMES:
            window = _safe_latency_window(scheduler_latency_raw.get(name))
            if window:
                scheduler["latency"][name] = window

    vlm_raw = payload.get("vlmEnrichment")
    vlm_raw = vlm_raw if isinstance(vlm_raw, dict) else {}
    vlm = {
        "counters": _safe_counter_map(vlm_raw, allowed=VLM_COUNTER_NAMES),
        "gauges": {},
    }
    for key in VLM_GAUGE_NAMES:
        value = vlm_raw.get(key)
        if isinstance(value, bool):
            vlm["gauges"][key] = value
        else:
            number = _finite_number(value)
            if number is not None and number >= 0:
                vlm["gauges"][key] = number

    transport_raw = payload.get("inferenceTransport")
    transport_raw = transport_raw if isinstance(transport_raw, dict) else {}
    inference_transport = {
        route: _safe_counter_map(
            transport_raw.get(route),
            allowed=allowed_counters,
        )
        for route, allowed_counters in INFERENCE_TRANSPORT_COUNTERS.items()
    }

    server_timestamp = _parse_timestamp_seconds(payload.get("timestamp"))
    snapshot_age = None
    clock_skew = False
    if server_timestamp is not None:
        raw_age = received_wall_seconds - server_timestamp
        clock_skew = raw_age < -1.0
        snapshot_age = max(0.0, raw_age)

    status = payload.get("status")
    if status not in {"ok", "degraded", "error"}:
        status = "unknown"
    return {
        "status": status,
        "snapshotTimestampSeconds": server_timestamp,
        "snapshotAgeSeconds": snapshot_age,
        "snapshotClockSkew": clock_skew,
        "diagnosticsCameras": diagnostics_cameras,
        "pipelineTelemetry": {
            "telemetryEpoch": epoch,
            "frameHistogramBoundsMs": frame_bounds,
            "alertHistogramBoundsMs": alert_bounds,
            "cameras": pipeline_cameras,
            "alerts": {
                "latency": alert_latency,
                "deliveryCoverage": alert_delivery_coverage,
            },
        },
        "sharedInferenceScheduler": scheduler,
        "vlmEnrichment": vlm,
        "inferenceTransport": inference_transport,
    }


def _validate_health_url(url: str) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise ValueError("health URL is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("health URL must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "health URL must not contain credentials, a query, or a fragment; use a token environment variable"
        )


def _auth_headers_from_env(
    *,
    bearer_token_env: str | None = None,
    token_env: str | None = None,
) -> dict[str, str]:
    if bearer_token_env and token_env:
        raise ValueError("bearer and token authentication are mutually exclusive")
    env_name = bearer_token_env or token_env
    if not env_name:
        return {}
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env_name) is None:
        raise ValueError("authentication environment variable name is invalid")
    secret = os.environ.get(env_name)
    if not secret:
        raise RuntimeError("configured authentication environment variable is empty")
    if "\r" in secret or "\n" in secret:
        raise RuntimeError("authentication value contains invalid header characters")
    scheme = "Bearer" if bearer_token_env else "Token"
    return {"Authorization": f"{scheme} {secret}"}


def _fetch_health(
    url: str,
    *,
    timeout_seconds: float,
    headers: Mapping[str, str],
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError("health endpoint returned a non-success status")
        raw = response.read(MAX_HEALTH_BYTES + 1)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if len(raw) > MAX_HEALTH_BYTES:
        raise RuntimeError("health response exceeded the size limit")
    payload = json.loads(raw.decode("utf-8"))
    received_wall = time.time()
    return (
        _sanitize_health_payload(payload, received_wall_seconds=received_wall),
        elapsed_ms,
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    parsed = [float(value) for value in values if math.isfinite(float(value))]
    if not parsed:
        return {
            "samples": 0,
            "mean": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "samples": len(parsed),
        "mean": round(statistics.fmean(parsed), 3),
        "p95": round(float(_percentile(parsed, 0.95)), 3),
        "p99": round(float(_percentile(parsed, 0.99)), 3),
        "maximum": round(max(parsed), 3),
    }


def _histogram_quantiles(
    bounds_ms: Sequence[float],
    bucket_counts: Sequence[int],
) -> dict[str, Any]:
    """Return nearest-rank p95/p99 as exposed bucket upper bounds."""

    if not bounds_ms or len(bucket_counts) != len(bounds_ms) + 1:
        raise ValueError("histogram bucket count must be one longer than bounds")
    if any(
        not math.isfinite(float(bound)) or float(bound) <= 0 for bound in bounds_ms
    ) or any(left >= right for left, right in zip(bounds_ms, bounds_ms[1:])):
        raise ValueError("histogram bounds must be finite, positive, and increasing")
    if any(
        isinstance(count, bool) or int(count) != count or count < 0
        for count in bucket_counts
    ):
        raise ValueError("histogram counts must be non-negative integers")
    total = sum(int(count) for count in bucket_counts)

    def estimate(quantile: float) -> tuple[float | None, bool]:
        if total == 0:
            return None, False
        rank = max(1, math.ceil(total * quantile))
        cumulative = 0
        for index, count in enumerate(bucket_counts):
            cumulative += int(count)
            if cumulative < rank:
                continue
            if index == len(bounds_ms):
                return None, True
            return float(bounds_ms[index]), False
        raise AssertionError("validated histogram did not contain requested rank")

    p95, p95_overflow = estimate(0.95)
    p99, p99_overflow = estimate(0.99)
    return {
        "samples": total,
        "overflowSamples": int(bucket_counts[-1]),
        "p95UpperBoundMs": p95,
        "p95Overflow": p95_overflow,
        "p99UpperBoundMs": p99,
        "p99Overflow": p99_overflow,
        "quantileMethod": "nearest-rank-upper-bucket-bound",
    }


def _jain_index(values: Iterable[float]) -> float | None:
    parsed = [max(0.0, float(value)) for value in values]
    if not parsed or not any(parsed):
        return None
    denominator = len(parsed) * sum(value * value for value in parsed)
    return round(sum(parsed) ** 2 / denominator, 6) if denominator else None


def _demand_normalized_jain(
    completed_fps: Mapping[str, float],
    demand_fps: Mapping[str, float],
) -> float | None:
    satisfaction = []
    for camera_id in sorted(set(completed_fps) & set(demand_fps)):
        demand = float(demand_fps[camera_id])
        if not math.isfinite(demand) or demand <= 0:
            continue
        completed = max(0.0, float(completed_fps[camera_id]))
        satisfaction.append(min(1.0, completed / demand))
    return _jain_index(satisfaction)


def _parse_tegrastats_line(line: str) -> dict[str, Any]:
    """Parse JetPack 5/6 tegrastats without retaining the source line."""

    parsed: dict[str, Any] = {}
    ram = _RAM_RE.search(line)
    if ram:
        parsed["ramUsedMiB"] = int(ram.group(1))
        parsed["ramTotalMiB"] = int(ram.group(2))
    swap = _SWAP_RE.search(line)
    if swap:
        parsed["swapUsedMiB"] = int(swap.group(1))
        parsed["swapTotalMiB"] = int(swap.group(2))

    cpu = _CPU_RE.search(line)
    if cpu:
        all_core_loads: list[float] = []
        online_core_loads: list[float] = []
        offline = 0
        for item in cpu.group(1).split(","):
            token = item.strip()
            if token.lower() == "off":
                offline += 1
                all_core_loads.append(0.0)
                continue
            match = re.match(r"(\d+(?:\.\d+)?)%@", token)
            if match is None:
                continue
            load = float(match.group(1))
            all_core_loads.append(load)
            online_core_loads.append(load)
        if all_core_loads:
            parsed["cpuPercent"] = round(statistics.fmean(all_core_loads), 3)
            parsed["cpuOnlinePercent"] = round(
                statistics.fmean(online_core_loads) if online_core_loads else 0.0,
                3,
            )
            parsed["cpuCoreCount"] = len(all_core_loads)
            parsed["cpuOfflineCoreCount"] = offline

    for key, pattern in (("gpuPercent", _GPU_RE), ("vicPercent", _VIC_RE)):
        match = pattern.search(line)
        if match:
            parsed[key] = float(match.group(1))

    temperatures = {
        name.lower(): float(value) for name, value in _TEMP_RE.findall(line)
    }
    if temperatures:
        parsed["temperaturesC"] = temperatures

    power_rails: dict[str, dict[str, float | None]] = {}
    for name, current, average in _POWER_MW_RE.findall(line):
        power_rails[name.lower()] = {
            "currentW": round(float(current) / 1000.0, 6),
            "averageW": (round(float(average) / 1000.0, 6) if average else None),
        }
    for name, current, average in _LEGACY_POWER_RE.findall(line):
        power_rails.setdefault(
            name.lower(),
            {
                "currentW": round(float(current) / 1000.0, 6),
                "averageW": round(float(average) / 1000.0, 6),
            },
        )
    if power_rails:
        parsed["powerRails"] = power_rails
        input_rail = power_rails.get("vdd_in") or power_rails.get("pom_5v_in")
        if input_rail is not None:
            parsed["inputPowerW"] = input_rail["currentW"]

    parsed["thermalMarker"] = bool(_THERMAL_MARKER_RE.search(line))
    return parsed


class _TegrastatsCollector:
    """Read either a growing log or one fixed tegrastats subprocess."""

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        spawn: bool = False,
        interval_ms: int = 1000,
    ) -> None:
        self._log_path = log_path
        self._spawn = spawn
        self._interval_ms = interval_ms
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._errors: Counter[str] = Counter()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._log_identity: tuple[int, int] | None = None
        self._log_position = 0

    @property
    def source(self) -> str:
        if self._spawn:
            return "spawned-tegrastats"
        if self._log_path is not None:
            return "supplied-tegrastats-log"
        return "none"

    def start(self) -> None:
        if self._spawn:
            try:
                self._process = subprocess.Popen(
                    ["tegrastats", "--interval", str(self._interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self._errors[type(exc).__name__] += 1
                return
            self._thread = threading.Thread(
                target=self._read_process,
                name="benchmark-tegrastats",
                daemon=True,
            )
        elif self._log_path is not None:
            # Establish the run boundary synchronously.  Starting at the
            # captured EOF prevents historical resource data from being
            # stamped with a current monotonic time by the tail thread.
            try:
                with self._log_path.open("rb") as handle:
                    opened = os.fstat(handle.fileno())
                    self._log_identity = (opened.st_dev, opened.st_ino)
                    self._log_position = opened.st_size
            except FileNotFoundError:
                # A file created after collection starts is entirely in-run.
                self._log_identity = None
                self._log_position = 0
            except Exception as exc:
                self._errors[type(exc).__name__] += 1
                self._log_identity = None
                self._log_position = 0
            self._thread = threading.Thread(
                target=self._tail_log,
                name="benchmark-tegrastats-log",
                daemon=True,
            )
        if self._thread is not None:
            self._thread.start()

    def _append_line(self, line: str) -> None:
        parsed = _parse_tegrastats_line(line)
        if len(parsed) == 1 and parsed.get("thermalMarker") is False:
            return
        with self._lock:
            self._samples.append({"monotonic": time.monotonic(), **parsed})

    def _read_process(self) -> None:
        try:
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                if self._stop.is_set():
                    break
                self._append_line(line)
        except Exception as exc:  # pragma: no cover - defensive I/O boundary.
            self._errors[type(exc).__name__] += 1

    def _tail_log(self) -> None:
        position = self._log_position
        identity = self._log_identity

        def drain_available() -> tuple[tuple[int, int] | None, int]:
            nonlocal identity, position
            assert self._log_path is not None
            with self._log_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                opened = os.fstat(handle.fileno())
                opened_identity = (opened.st_dev, opened.st_ino)
                if opened_identity != identity or opened.st_size < position:
                    # A new inode is a rotated log and a smaller same-inode
                    # file is a copy-truncate.  Both begin at byte zero.
                    position = 0
                identity = opened_identity
                handle.seek(position)
                while line := handle.readline():
                    self._append_line(line)
                position = handle.tell()
            return identity, position

        while not self._stop.is_set():
            try:
                assert self._log_path is not None
                if not self._log_path.exists():
                    self._stop.wait(0.1)
                    continue
                identity, position = drain_available()
            except Exception as exc:  # pragma: no cover - filesystem race guard.
                self._errors[type(exc).__name__] += 1
            self._stop.wait(0.1)
        try:
            assert self._log_path is not None
            if self._log_path.exists():
                drain_available()
        except Exception as exc:  # pragma: no cover - final-drain guard.
            self._errors[type(exc).__name__] += 1

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def result(
        self, *, started: float, ended: float
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        with self._lock:
            samples = [
                dict(sample)
                for sample in self._samples
                if started <= float(sample["monotonic"]) <= ended
            ]
        return samples, dict(sorted(self._errors.items()))


def _counter_deltas(
    previous: Mapping[str, int],
    current: Mapping[str, int],
) -> tuple[dict[str, int], list[str]]:
    deltas: dict[str, int] = {}
    resets: list[str] = []
    for key in sorted(set(previous) | set(current)):
        if key not in current:
            resets.append(key)
            continue
        current_value = current[key]
        previous_value = int(previous.get(key, 0))
        if current_value < previous_value:
            resets.append(key)
        else:
            deltas[key] = current_value - previous_value
    return deltas, resets


def _accumulate_histogram_transition(
    accumulator: dict[str, Any],
    *,
    previous_bounds: object,
    current_bounds: object,
    previous_histogram: object,
    current_histogram: object,
) -> None:
    if (
        not isinstance(previous_bounds, list)
        or previous_bounds != current_bounds
        or not isinstance(previous_histogram, dict)
        or not isinstance(current_histogram, dict)
    ):
        accumulator["invalidTransitions"] += 1
        return
    previous_counts = previous_histogram.get("bucketCounts")
    current_counts = current_histogram.get("bucketCounts")
    if (
        not isinstance(previous_counts, list)
        or not isinstance(current_counts, list)
        or len(previous_counts) != len(current_counts)
        or len(current_counts) != len(previous_bounds) + 1
    ):
        accumulator["invalidTransitions"] += 1
        return
    deltas = [
        int(current) - int(previous)
        for previous, current in zip(previous_counts, current_counts)
    ]
    if any(delta < 0 for delta in deltas):
        accumulator["counterResets"] += 1
        return
    if accumulator["boundsMs"] is None:
        accumulator["boundsMs"] = list(previous_bounds)
        accumulator["bucketCounts"] = [0] * len(deltas)
    if accumulator["boundsMs"] != previous_bounds:
        accumulator["invalidTransitions"] += 1
        return
    accumulator["bucketCounts"] = [
        current + delta for current, delta in zip(accumulator["bucketCounts"], deltas)
    ]
    previous_invalid = previous_histogram.get("invalidCount")
    current_invalid = current_histogram.get("invalidCount")
    if isinstance(previous_invalid, int) and isinstance(current_invalid, int):
        if current_invalid < previous_invalid:
            accumulator["invalidCountResets"] += 1
        else:
            accumulator["invalidSamples"] += current_invalid - previous_invalid
            accumulator["invalidCountObservable"] = True
    elif previous_invalid is not None or current_invalid is not None:
        accumulator["invalidTransitions"] += 1


def _new_histogram_accumulator() -> dict[str, Any]:
    return {
        "boundsMs": None,
        "bucketCounts": [],
        "invalidTransitions": 0,
        "counterResets": 0,
        "invalidSamples": 0,
        "invalidCountObservable": False,
        "invalidCountResets": 0,
    }


def _render_histogram_accumulator(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    bounds = accumulator.get("boundsMs")
    counts = accumulator.get("bucketCounts")
    if not isinstance(bounds, list) or not isinstance(counts, list):
        summary = _histogram_quantiles([1.0], [0, 0])
    else:
        summary = _histogram_quantiles(bounds, counts)
    summary["invalidTransitions"] = int(accumulator.get("invalidTransitions", 0))
    summary["counterResets"] = int(accumulator.get("counterResets", 0))
    summary["invalidSamples"] = (
        int(accumulator.get("invalidSamples", 0))
        if accumulator.get("invalidCountObservable") is True
        else None
    )
    summary["invalidCountObservable"] = (
        accumulator.get("invalidCountObservable") is True
    )
    summary["invalidCountResets"] = int(
        accumulator.get("invalidCountResets", 0)
    )
    return summary


def _summarize_tegrastats(
    samples: Sequence[Mapping[str, Any]],
    *,
    source: str,
    collection_errors: Mapping[str, int],
    thermal_warning_c: float,
    configured_duration_seconds: float | None = None,
    configured_interval_seconds: float | None = None,
) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in samples
            if _finite_number(sample.get(key)) is not None
        ]

    ram_util = [
        100.0 * float(sample["ramUsedMiB"]) / float(sample["ramTotalMiB"])
        for sample in samples
        if _finite_number(sample.get("ramUsedMiB")) is not None
        and _finite_number(sample.get("ramTotalMiB")) not in {None, 0.0}
    ]
    swap_util = [
        100.0 * float(sample["swapUsedMiB"]) / float(sample["swapTotalMiB"])
        for sample in samples
        if _finite_number(sample.get("swapUsedMiB")) is not None
        and _finite_number(sample.get("swapTotalMiB")) not in {None, 0.0}
    ]
    temperatures: dict[str, list[float]] = defaultdict(list)
    power_current: dict[str, list[float]] = defaultdict(list)
    power_average: dict[str, list[float]] = defaultdict(list)
    thermal_hot_samples = 0
    thermal_marker_samples = 0
    for sample in samples:
        sample_temperatures = sample.get("temperaturesC")
        if isinstance(sample_temperatures, dict):
            finite_temperatures = []
            for name, value in sample_temperatures.items():
                parsed = _finite_number(value)
                if parsed is not None and COUNTER_NAME_RE.fullmatch(str(name)):
                    temperatures[str(name)].append(parsed)
                    finite_temperatures.append(parsed)
            if finite_temperatures and max(finite_temperatures) >= thermal_warning_c:
                thermal_hot_samples += 1
        if sample.get("thermalMarker") is True:
            thermal_marker_samples += 1
        rails = sample.get("powerRails")
        if isinstance(rails, dict):
            for name, rail in rails.items():
                if COUNTER_NAME_RE.fullmatch(str(name)) is None or not isinstance(
                    rail, dict
                ):
                    continue
                current = _finite_number(rail.get("currentW"))
                average = _finite_number(rail.get("averageW"))
                if current is not None:
                    power_current[str(name)].append(current)
                if average is not None:
                    power_average[str(name)].append(average)

    rail_names = sorted(set(power_current) | set(power_average))
    monotonic_values = [
        float(sample["monotonic"])
        for sample in samples
        if _finite_number(sample.get("monotonic")) is not None
    ]
    sample_span_seconds = (
        max(monotonic_values) - min(monotonic_values)
        if len(monotonic_values) >= 2
        else 0.0
    )
    resource_measurement_required = source != "none"
    if (
        configured_duration_seconds is not None
        and configured_interval_seconds is not None
        and configured_duration_seconds > 0
        and configured_interval_seconds > 0
    ):
        expected_resource_samples = (
            math.floor(
                configured_duration_seconds / configured_interval_seconds
            )
            + 1
        )
        resource_duration_coverage = min(
            1.0,
            sample_span_seconds / configured_duration_seconds,
        )
        resource_sample_coverage = min(
            1.0,
            len(samples) / expected_resource_samples,
        )
        sufficient_resource_duration = (
            resource_duration_coverage >= MIN_RESOURCE_DURATION_COVERAGE
        )
        sufficient_resource_samples = (
            resource_sample_coverage >= MIN_RESOURCE_SAMPLE_COVERAGE
        )
        resource_valid = (
            not resource_measurement_required
            or (
                sufficient_resource_duration
                and sufficient_resource_samples
            )
        )
    else:
        expected_resource_samples = None
        resource_duration_coverage = None
        resource_sample_coverage = None
        sufficient_resource_duration = None
        sufficient_resource_samples = None
        resource_valid = not resource_measurement_required or bool(samples)
    return {
        "source": source,
        "samples": len(samples),
        "collectionErrorCounts": dict(collection_errors),
        "measurementValidity": {
            "required": resource_measurement_required,
            "valid": resource_valid,
            "sampleSpanSeconds": round(sample_span_seconds, 6),
            "expectedSamples": expected_resource_samples,
            "sampleCoverageRatio": (
                round(resource_sample_coverage, 6)
                if resource_sample_coverage is not None
                else None
            ),
            "minimumSampleCoverageRatio": MIN_RESOURCE_SAMPLE_COVERAGE,
            "sufficientSampleCoverage": sufficient_resource_samples,
            "durationCoverageRatio": (
                round(resource_duration_coverage, 6)
                if resource_duration_coverage is not None
                else None
            ),
            "minimumDurationCoverageRatio": MIN_RESOURCE_DURATION_COVERAGE,
            "sufficientDurationCoverage": sufficient_resource_duration,
        },
        "ram": {
            "usedMiB": _distribution(values("ramUsedMiB")),
            "totalMiB": _distribution(values("ramTotalMiB")),
            "utilizationPercent": _distribution(ram_util),
        },
        "swap": {
            "usedMiB": _distribution(values("swapUsedMiB")),
            "totalMiB": _distribution(values("swapTotalMiB")),
            "utilizationPercent": _distribution(swap_util),
        },
        "cpuPercent": _distribution(values("cpuPercent")),
        "cpuOnlinePercent": _distribution(values("cpuOnlinePercent")),
        "gpuPercent": _distribution(values("gpuPercent")),
        "vicPercent": _distribution(values("vicPercent")),
        "inputPowerW": _distribution(values("inputPowerW")),
        "powerRails": {
            name: {
                "currentW": _distribution(power_current.get(name, [])),
                "averageW": _distribution(power_average.get(name, [])),
            }
            for name in rail_names
        },
        "temperaturesC": {
            name: _distribution(values_for_sensor)
            for name, values_for_sensor in sorted(temperatures.items())
        },
        "thermalIndicators": {
            "warningThresholdC": thermal_warning_c,
            "samplesAtOrAboveThreshold": thermal_hot_samples,
            "explicitThrottleOrThermalMarkerSamples": thermal_marker_samples,
            "concernObserved": bool(thermal_hot_samples or thermal_marker_samples),
        },
    }


def _summarize(
    samples: Sequence[Mapping[str, Any]],
    tegrastats_samples: Sequence[Mapping[str, Any]],
    *,
    configured_duration_seconds: float,
    configured_poll_interval_seconds: float,
    demand_fps: Mapping[str, float],
    require_hardware_decode: bool,
    require_max_frame_dimension: int | None,
    camera_basis: str,
    equivalent_camera_count: int | None,
    tegrastats_source: str,
    tegrastats_interval_seconds: float,
    tegrastats_errors: Mapping[str, int],
    thermal_warning_c: float,
) -> dict[str, Any]:
    successful = [
        sample for sample in samples if isinstance(sample.get("health"), dict)
    ]
    health_errors = Counter(
        str(sample["healthError"])
        for sample in samples
        if isinstance(sample.get("healthError"), str)
    )
    if not successful:
        raise RuntimeError("no successful health samples were collected")

    first_monotonic = float(successful[0]["monotonic"])
    last_monotonic = float(successful[-1]["monotonic"])
    observed_seconds = max(0.0, last_monotonic - first_monotonic)
    duration_coverage_ratio = min(
        1.0,
        observed_seconds / configured_duration_seconds,
    )
    expected_health_samples = (
        math.floor(
            configured_duration_seconds / configured_poll_interval_seconds
        )
        + 1
    )
    health_sample_coverage_ratio = min(
        1.0,
        len(successful) / expected_health_samples,
    )
    sufficient_duration_coverage = (
        duration_coverage_ratio >= MIN_HEALTH_DURATION_COVERAGE
    )
    sufficient_health_sample_coverage = (
        health_sample_coverage_ratio >= MIN_HEALTH_SAMPLE_COVERAGE
    )
    server_snapshot_timestamps = [
        float(timestamp)
        for sample in successful
        if (
            timestamp := _finite_number(
                sample["health"].get("snapshotTimestampSeconds")
            )
        )
        is not None
    ]
    all_server_snapshot_timestamps_present = (
        len(server_snapshot_timestamps) == len(successful)
    )
    server_snapshot_timestamps_monotonic = (
        all_server_snapshot_timestamps_present
        and all(
            current >= previous
            for previous, current in zip(
                server_snapshot_timestamps,
                server_snapshot_timestamps[1:],
            )
        )
    )
    distinct_server_snapshot_count = len(set(server_snapshot_timestamps))
    at_least_two_distinct_server_snapshots = distinct_server_snapshot_count >= 2
    server_snapshot_span_seconds = (
        max(server_snapshot_timestamps) - min(server_snapshot_timestamps)
        if server_snapshot_timestamps
        else 0.0
    )
    server_snapshot_duration_coverage_ratio = min(
        1.0,
        server_snapshot_span_seconds / configured_duration_seconds,
    )
    sufficient_server_snapshot_duration_coverage = (
        server_snapshot_duration_coverage_ratio
        >= MIN_SERVER_SNAPSHOT_DURATION_COVERAGE
    )
    server_snapshot_ages = [
        float(age)
        for sample in successful
        if (
            age := _finite_number(sample["health"].get("snapshotAgeSeconds"))
        )
        is not None
    ]
    maximum_server_snapshot_age_seconds = (
        max(server_snapshot_ages) if server_snapshot_ages else None
    )
    server_snapshot_clock_skew_samples = sum(
        bool(sample["health"].get("snapshotClockSkew"))
        for sample in successful
    )
    server_snapshots_fresh = (
        all_server_snapshot_timestamps_present
        and len(server_snapshot_ages) == len(successful)
        and maximum_server_snapshot_age_seconds is not None
        and maximum_server_snapshot_age_seconds
        <= MAX_HEALTH_SNAPSHOT_AGE_SECONDS
        and server_snapshot_clock_skew_samples == 0
    )

    camera_ids: set[str] = set(demand_fps)
    telemetry_camera_ids: set[str] = set()
    camera_accumulators: dict[str, dict[str, Any]] = {}
    scheduler_deltas: dict[str, int] = defaultdict(int)
    scheduler_resets: Counter[str] = Counter()
    vlm_deltas: dict[str, int] = defaultdict(int)
    vlm_resets: Counter[str] = Counter()
    transport_deltas: dict[str, dict[str, int]] = {
        route: defaultdict(int) for route in INFERENCE_TRANSPORT_COUNTERS
    }
    transport_resets: dict[str, Counter[str]] = {
        route: Counter() for route in INFERENCE_TRANSPORT_COUNTERS
    }
    alert_histograms = {
        name: _new_histogram_accumulator() for name in ALERT_LATENCY_NAMES
    }
    alert_delivery_deltas: dict[str, int] = defaultdict(int)
    alert_delivery_resets: Counter[str] = Counter()
    alert_delivery_coverage_samples = 0
    epoch_transitions = 0
    advancing_server_snapshot_transitions = 0
    skipped_nonadvancing_server_snapshot_transitions = 0

    def camera_accumulator(camera_id: str) -> dict[str, Any]:
        if camera_id not in camera_accumulators:
            camera_accumulators[camera_id] = {
                "counterDeltas": defaultdict(int),
                "counterValidSeconds": defaultdict(float),
                "counterResets": Counter(),
                "generationTransitions": 0,
                "identitySegments": set(),
                "latency": {
                    name: _new_histogram_accumulator() for name in CAMERA_LATENCY_NAMES
                },
            }
        return camera_accumulators[camera_id]

    for sample in successful:
        health = sample["health"]
        pipeline = health["pipelineTelemetry"]
        epoch = pipeline.get("telemetryEpoch")
        for camera_id, camera in pipeline.get("cameras", {}).items():
            camera_ids.add(camera_id)
            telemetry_camera_ids.add(camera_id)
            camera_accumulator(camera_id)["identitySegments"].add(
                (epoch, camera.get("generation"))
            )
        camera_ids.update(health.get("diagnosticsCameras", {}))

    for previous_sample, current_sample in zip(successful, successful[1:]):
        previous_health = previous_sample["health"]
        current_health = current_sample["health"]
        previous_snapshot_timestamp = _finite_number(
            previous_health.get("snapshotTimestampSeconds")
        )
        current_snapshot_timestamp = _finite_number(
            current_health.get("snapshotTimestampSeconds")
        )
        if (
            not server_snapshot_timestamps_monotonic
            or previous_snapshot_timestamp is None
            or current_snapshot_timestamp is None
            or current_snapshot_timestamp <= previous_snapshot_timestamp
        ):
            skipped_nonadvancing_server_snapshot_transitions += 1
            continue
        advancing_server_snapshot_transitions += 1
        previous_pipeline = previous_health["pipelineTelemetry"]
        current_pipeline = current_health["pipelineTelemetry"]
        previous_epoch = previous_pipeline.get("telemetryEpoch")
        current_epoch = current_pipeline.get("telemetryEpoch")
        same_epoch = previous_epoch is not None and previous_epoch == current_epoch
        elapsed = current_snapshot_timestamp - previous_snapshot_timestamp
        if not same_epoch:
            epoch_transitions += 1
            continue

        previous_cameras = previous_pipeline.get("cameras", {})
        current_cameras = current_pipeline.get("cameras", {})
        for camera_id in sorted(set(previous_cameras) | set(current_cameras)):
            previous_camera = previous_cameras.get(camera_id)
            current_camera = current_cameras.get(camera_id)
            if not isinstance(previous_camera, dict) or not isinstance(
                current_camera, dict
            ):
                continue
            accumulator = camera_accumulator(camera_id)
            if previous_camera.get("generation") != current_camera.get("generation"):
                accumulator["generationTransitions"] += 1
                continue
            deltas, resets = _counter_deltas(
                previous_camera.get("counters", {}),
                current_camera.get("counters", {}),
            )
            for name, delta in deltas.items():
                accumulator["counterDeltas"][name] += delta
                accumulator["counterValidSeconds"][name] += elapsed
            accumulator["counterResets"].update(resets)
            for latency_name in CAMERA_LATENCY_NAMES:
                _accumulate_histogram_transition(
                    accumulator["latency"][latency_name],
                    previous_bounds=previous_pipeline.get("frameHistogramBoundsMs"),
                    current_bounds=current_pipeline.get("frameHistogramBoundsMs"),
                    previous_histogram=(previous_camera.get("latency") or {}).get(
                        latency_name
                    ),
                    current_histogram=(current_camera.get("latency") or {}).get(
                        latency_name
                    ),
                )

        scheduler_delta, scheduler_reset = _counter_deltas(
            previous_health["sharedInferenceScheduler"].get("counters", {}),
            current_health["sharedInferenceScheduler"].get("counters", {}),
        )
        for name, delta in scheduler_delta.items():
            scheduler_deltas[name] += delta
        scheduler_resets.update(scheduler_reset)

        vlm_delta, vlm_reset = _counter_deltas(
            previous_health["vlmEnrichment"].get("counters", {}),
            current_health["vlmEnrichment"].get("counters", {}),
        )
        for name, delta in vlm_delta.items():
            vlm_deltas[name] += delta
        vlm_resets.update(vlm_reset)

        previous_transport = previous_health.get("inferenceTransport", {})
        current_transport = current_health.get("inferenceTransport", {})
        for route in INFERENCE_TRANSPORT_COUNTERS:
            route_delta, route_resets = _counter_deltas(
                previous_transport.get(route, {}),
                current_transport.get(route, {}),
            )
            for name, delta in route_delta.items():
                transport_deltas[route][name] += delta
            transport_resets[route].update(route_resets)

        previous_alerts = (previous_pipeline.get("alerts") or {}).get("latency") or {}
        current_alerts = (current_pipeline.get("alerts") or {}).get("latency") or {}
        for latency_name in ALERT_LATENCY_NAMES:
            _accumulate_histogram_transition(
                alert_histograms[latency_name],
                previous_bounds=previous_pipeline.get("alertHistogramBoundsMs"),
                current_bounds=current_pipeline.get("alertHistogramBoundsMs"),
                previous_histogram=previous_alerts.get(latency_name),
                current_histogram=current_alerts.get(latency_name),
            )
        previous_delivery = (previous_pipeline.get("alerts") or {}).get(
            "deliveryCoverage"
        ) or {}
        current_delivery = (current_pipeline.get("alerts") or {}).get(
            "deliveryCoverage"
        ) or {}
        if previous_delivery and current_delivery:
            alert_delivery_coverage_samples += 1
            delivery_delta, delivery_resets = _counter_deltas(
                previous_delivery.get("counters", {}),
                current_delivery.get("counters", {}),
            )
            for name, delta in delivery_delta.items():
                alert_delivery_deltas[name] += delta
            alert_delivery_resets.update(delivery_resets)

    diagnostics_by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in successful:
        for camera_id, camera in sample["health"].get("diagnosticsCameras", {}).items():
            diagnostics_by_camera[camera_id].append(camera)

    camera_reports: dict[str, Any] = {}
    camera_completeness: dict[str, dict[str, Any]] = {}
    completed_fps: dict[str, float] = {}
    effective_demand_fps: dict[str, float] = {}
    for camera_id in sorted(camera_ids):
        accumulator = camera_accumulator(camera_id)
        deltas = {
            name: int(accumulator["counterDeltas"].get(name, 0))
            for name in CAMERA_COUNTER_NAMES
        }
        valid_seconds = accumulator["counterValidSeconds"]
        decoded_seconds = float(valid_seconds.get("decodedFrameCount", 0.0))
        completed_seconds = float(valid_seconds.get("inferenceCompletedCount", 0.0))
        decoded_fps = (
            deltas["decodedFrameCount"] / decoded_seconds
            if decoded_seconds > 0
            else 0.0
        )
        inference_fps = (
            deltas["inferenceCompletedCount"] / completed_seconds
            if completed_seconds > 0
            else 0.0
        )
        completed_fps[camera_id] = inference_fps
        if camera_id in demand_fps:
            camera_demand = float(demand_fps[camera_id])
            demand_source = "explicit-target-fps"
        elif demand_fps:
            camera_demand = 0.0
            demand_source = "excluded-from-explicit-expected-set"
        else:
            camera_demand = decoded_fps
            demand_source = "observed-decoded-ingress-fps-proxy"
        if camera_demand > 0:
            effective_demand_fps[camera_id] = camera_demand
            demand_satisfaction = min(1.0, inference_fps / camera_demand)
        else:
            demand_satisfaction = None

        diagnostic_samples = diagnostics_by_camera.get(camera_id, [])
        latest_diagnostic = successful[-1]["health"].get(
            "diagnosticsCameras", {}
        ).get(camera_id, {})
        latest_connection = latest_diagnostic.get("connection")
        latest_connection = (
            latest_connection if isinstance(latest_connection, dict) else {}
        )
        frame_width = _safe_frame_dimension(latest_diagnostic.get("frameWidth"))
        frame_height = _safe_frame_dimension(latest_diagnostic.get("frameHeight"))
        frame_max_dimension = (
            max(frame_width, frame_height)
            if frame_width is not None and frame_height is not None
            else None
        )
        telemetry_observed = camera_id in telemetry_camera_ids
        valid_decoded_interval = decoded_seconds > 0
        decoded_counter_coverage_ratio = min(
            1.0,
            decoded_seconds / configured_duration_seconds,
        )
        sufficient_decoded_counter_coverage = (
            decoded_counter_coverage_ratio >= MIN_HEALTH_DURATION_COVERAGE
        )
        decoded_frames_observed = deltas["decodedFrameCount"] > 0
        expected_camera = camera_id in demand_fps
        diagnostic_coverage_ratio = min(
            1.0,
            len(diagnostic_samples) / len(successful),
        )
        complete_diagnostic_coverage = len(diagnostic_samples) == len(successful)
        hardware_violation_samples = sum(
            not (
                item.get("streamType") == "rtsp"
                and isinstance(item.get("connection"), dict)
                and item["connection"].get("captureBackend")
                == "gstreamer_nvdec"
                and item["connection"].get("hardwareAccelerationActive") is True
                and item["connection"].get("hardwareFallback") is not True
            )
            for item in diagnostic_samples
        )
        hardware_decode_satisfied = (
            complete_diagnostic_coverage and hardware_violation_samples == 0
        )
        max_dimension_violation_samples = 0
        observed_frame_max_dimensions: list[int] = []
        for item in diagnostic_samples:
            sample_width = _safe_frame_dimension(item.get("frameWidth"))
            sample_height = _safe_frame_dimension(item.get("frameHeight"))
            if sample_width is None or sample_height is None:
                max_dimension_violation_samples += 1
                continue
            sample_max_dimension = max(sample_width, sample_height)
            observed_frame_max_dimensions.append(sample_max_dimension)
            if (
                require_max_frame_dimension is not None
                and sample_max_dimension > require_max_frame_dimension
            ):
                max_dimension_violation_samples += 1
        max_dimension_satisfied = (
            complete_diagnostic_coverage
            and bool(observed_frame_max_dimensions)
            and max_dimension_violation_samples == 0
        )
        complete = (
            telemetry_observed
            and valid_decoded_interval
            and decoded_frames_observed
            and (
                not expected_camera
                or (
                    sufficient_decoded_counter_coverage
                    and complete_diagnostic_coverage
                )
            )
            and (
                not expected_camera
                or not require_hardware_decode
                or hardware_decode_satisfied
            )
            and (
                not expected_camera
                or require_max_frame_dimension is None
                or max_dimension_satisfied
            )
        )
        camera_completeness[camera_id] = {
            "expected": expected_camera,
            "telemetryObserved": telemetry_observed,
            "decodedCounterValidSeconds": round(decoded_seconds, 6),
            "validDecodedCounterInterval": valid_decoded_interval,
            "decodedCounterCoverageRatio": round(
                decoded_counter_coverage_ratio, 6
            ),
            "sufficientDecodedCounterCoverage": (
                sufficient_decoded_counter_coverage
            ),
            "decodedFramesObserved": decoded_frames_observed,
            "diagnosticSamplesExpected": len(successful),
            "diagnosticSamplesPresent": len(diagnostic_samples),
            "diagnosticSampleCoverageRatio": round(
                diagnostic_coverage_ratio, 6
            ),
            "completeDiagnosticSampleCoverage": complete_diagnostic_coverage,
            "hardwareDecodeRequired": bool(require_hardware_decode),
            "hardwareDecodeSatisfied": (
                hardware_decode_satisfied if require_hardware_decode else None
            ),
            "hardwareDecodeViolationSamples": (
                hardware_violation_samples
                + (len(successful) - len(diagnostic_samples))
                if require_hardware_decode
                else None
            ),
            "requiredMaxFrameDimension": require_max_frame_dimension,
            "maxFrameDimensionSatisfied": (
                max_dimension_satisfied
                if require_max_frame_dimension is not None
                else None
            ),
            "maxFrameDimensionViolationSamples": (
                max_dimension_violation_samples
                + (len(successful) - len(diagnostic_samples))
                if require_max_frame_dimension is not None
                else None
            ),
            "maximumObservedFrameDimension": (
                max(observed_frame_max_dimensions)
                if observed_frame_max_dimensions
                else None
            ),
            "complete": complete,
        }
        age_values = [
            float(camera["lastFrameAgeSeconds"])
            for camera in diagnostic_samples
            if _finite_number(camera.get("lastFrameAgeSeconds")) is not None
        ]
        drop_deltas = {name: deltas[name] for name in DROP_COUNTER_NAMES}
        skip_deltas = {name: deltas[name] for name in POLICY_SKIP_COUNTER_NAMES}

        def counter_rate(counter_name: str) -> float | None:
            seconds = float(valid_seconds.get(counter_name, 0.0))
            return (
                round(deltas[counter_name] / seconds, 6)
                if seconds > 0
                else None
            )

        adaptive_observations = {
            state_name: deltas[f"adaptive{state_name.title()}ObservationCount"]
            for state_name in ADAPTIVE_STATES
        }
        adaptive_admissions = {
            state_name: deltas[f"adaptive{state_name.title()}AdmissionCount"]
            for state_name in ADAPTIVE_STATES
        }
        adaptive_inferences = {
            state_name: deltas[f"adaptive{state_name.title()}InferenceCount"]
            for state_name in ADAPTIVE_STATES
        }
        total_adaptive_observations = sum(adaptive_observations.values())
        total_adaptive_admissions = sum(adaptive_admissions.values())
        total_adaptive_inferences = sum(adaptive_inferences.values())
        camera_reports[camera_id] = {
            "telemetryIdentitySegments": len(accumulator["identitySegments"]),
            "generationTransitions": accumulator["generationTransitions"],
            "counterResetCounts": dict(sorted(accumulator["counterResets"].items())),
            "decodedFrames": deltas["decodedFrameCount"],
            "inferenceCompleted": deltas["inferenceCompletedCount"],
            "inferenceFailures": deltas["inferenceFailureCount"],
            "captureDuplicates": deltas["captureDuplicateCount"],
            "dropCounts": {**drop_deltas, "total": sum(drop_deltas.values())},
            "policySkipCounts": {**skip_deltas, "total": sum(skip_deltas.values())},
            "decodedFps": round(decoded_fps, 6),
            "effectiveInferenceFps": round(inference_fps, 6),
            "demandFps": round(camera_demand, 6),
            "demandSource": demand_source,
            "demandSatisfaction": (
                round(demand_satisfaction, 6)
                if demand_satisfaction is not None
                else None
            ),
            "adaptiveInference": {
                "acceptedAdmissionDefinition": (
                    "accepted camera queue offers, including offers that replace an "
                    "older latest-slot item; successful inference rates are the "
                    "executed-work metric"
                ),
                "observationCountsByState": adaptive_observations,
                "observationSharesByState": {
                    state_name: (
                        round(count / total_adaptive_observations, 6)
                        if total_adaptive_observations
                        else None
                    )
                    for state_name, count in adaptive_observations.items()
                },
                "acceptedAdmissionCountsByState": adaptive_admissions,
                "acceptedAdmissionRatesPerSecondByState": {
                    state_name: counter_rate(
                        f"adaptive{state_name.title()}AdmissionCount"
                    )
                    for state_name in ADAPTIVE_STATES
                },
                "successfulInferenceCountsByState": adaptive_inferences,
                "successfulInferenceRatesPerSecondByState": {
                    state_name: counter_rate(
                        f"adaptive{state_name.title()}InferenceCount"
                    )
                    for state_name in ADAPTIVE_STATES
                },
                "totalObservations": total_adaptive_observations,
                "totalAcceptedAdmissions": total_adaptive_admissions,
                "totalSuccessfulInferences": total_adaptive_inferences,
            },
            "keyframeTracker": {
                "projectionFrames": deltas["trackerProjectionFrameCount"],
                "projectedPeople": deltas["trackerProjectedPersonCount"],
                "forceRedetectSignalObservations": deltas[
                    "trackerForceRedetectSignalCount"
                ],
                "projectionFrameRatePerSecond": counter_rate(
                    "trackerProjectionFrameCount"
                ),
                "forceRedetectSignalRatePerSecond": counter_rate(
                    "trackerForceRedetectSignalCount"
                ),
            },
            "personCropSpecialists": {
                "ppeCropAttempts": deltas["ppePersonCropAttemptCount"],
                "phoneCropAttempts": deltas["phonePersonCropAttemptCount"],
                "fallbacks": deltas["personCropFallbackCount"],
                "fullFrameInvocations": deltas[
                    "personCropFullFrameInvocationCount"
                ],
                "ppeCropAttemptRatePerSecond": counter_rate(
                    "ppePersonCropAttemptCount"
                ),
                "phoneCropAttemptRatePerSecond": counter_rate(
                    "phonePersonCropAttemptCount"
                ),
            },
            "latency": {
                name: _render_histogram_accumulator(accumulator["latency"][name])
                for name in CAMERA_LATENCY_NAMES
            },
            "healthFreshness": {
                "presentSamples": len(diagnostic_samples),
                "freshSamples": sum(
                    bool(item.get("frameFresh")) for item in diagnostic_samples
                ),
                "workerRunningSamples": sum(
                    bool(item.get("workerRunning")) for item in diagnostic_samples
                ),
                "publishedFrameAgeSeconds": _distribution(age_values),
                "latestStreamType": latest_diagnostic.get("streamType", "unknown"),
                "latestFrameWidth": frame_width,
                "latestFrameHeight": frame_height,
                "latestFrameMaxDimension": frame_max_dimension,
                "latestCaptureBackend": latest_connection.get(
                    "captureBackend", "unknown"
                ),
                "latestHardwareAccelerationActive": latest_connection.get(
                    "hardwareAccelerationActive", False
                ),
                "latestHardwareFallback": latest_connection.get(
                    "hardwareFallback", False
                ),
                "latestCaptureDropAccounting": latest_connection.get(
                    "captureDropAccounting", "unavailable"
                ),
                "latestCaptureDropCountIsLowerBound": latest_connection.get(
                    "captureDropCountIsLowerBound", True
                ),
                "latestAppsinkDropsObservable": latest_connection.get(
                    "appsinkLatestBufferDropsObservable", False
                ),
                "latestAppsinkDropMethod": latest_connection.get(
                    "appsinkLatestBufferDropMethod", "unavailable"
                ),
                "latestDecoderPolicyDropAccounting": latest_connection.get(
                    "decoderPolicyDropAccounting", "unknown"
                ),
            },
            "measurementCompleteness": camera_completeness[camera_id],
        }

    expected_camera_ids = sorted(demand_fps)
    active_telemetry_camera_ids = sorted(
        camera_id
        for camera_id in telemetry_camera_ids
        if camera_completeness[camera_id]["validDecodedCounterInterval"]
        and camera_completeness[camera_id]["decodedFramesObserved"]
    )
    complete_expected_camera_ids = [
        camera_id
        for camera_id in expected_camera_ids
        if camera_completeness[camera_id]["complete"]
    ]
    incomplete_expected_camera_ids = sorted(
        set(expected_camera_ids) - set(complete_expected_camera_ids)
    )
    all_expected_cameras_complete = bool(expected_camera_ids) and not (
        incomplete_expected_camera_ids
    )
    all_expected_hardware_decode_satisfied = bool(expected_camera_ids) and all(
        camera_completeness[camera_id]["hardwareDecodeSatisfied"] is True
        for camera_id in expected_camera_ids
    )
    all_expected_max_dimension_satisfied = bool(expected_camera_ids) and all(
        camera_completeness[camera_id]["maxFrameDimensionSatisfied"] is True
        for camera_id in expected_camera_ids
    )
    expected_completeness = {
        "source": (
            "explicit-camera-demand-fps"
            if expected_camera_ids
            else "observed-active-camera-fallback"
        ),
        "expectedCameraIds": expected_camera_ids,
        "expectedCameraCount": len(expected_camera_ids),
        "completeExpectedCameraIds": complete_expected_camera_ids,
        "incompleteExpectedCameraIds": incomplete_expected_camera_ids,
        "allExpectedCamerasComplete": (
            all_expected_cameras_complete if expected_camera_ids else None
        ),
        "activeTelemetryCameraIds": active_telemetry_camera_ids,
        "excludedObservedCameraIds": sorted(
            camera_ids - set(expected_camera_ids)
        )
        if expected_camera_ids
        else [],
        "requirements": {
            "hardwareDecode": bool(require_hardware_decode),
            "maxFrameDimension": require_max_frame_dimension,
        },
    }

    batch_deltas = {
        str(size): int(scheduler_deltas.get(f"batches_{size}", 0)) for size in (1, 2, 4)
    }
    cohort_count = sum(batch_deltas.values())
    dispatched_from_batches = sum(
        int(size) * count for size, count in batch_deltas.items()
    )
    transport_reports: dict[str, Any] = {}
    total_transport_executions = 0
    total_transport_frames = 0
    for route, allowed_counters in INFERENCE_TRANSPORT_COUNTERS.items():
        execution_deltas: dict[str, int] = {}
        for counter_name in allowed_counters:
            match = re.fullmatch(r"batch([124])_(?:succeeded|executed)", counter_name)
            if match is None:
                continue
            size = match.group(1)
            execution_deltas[size] = int(
                transport_deltas[route].get(counter_name, 0)
            )
        successful_executions = sum(execution_deltas.values())
        successful_frames = sum(
            int(size) * executions
            for size, executions in execution_deltas.items()
        )
        total_transport_executions += successful_executions
        total_transport_frames += successful_frames
        transport_reports[route] = {
            "successfulBatchExecutionDeltas": execution_deltas,
            "successfulBatchExecutions": successful_executions,
            "successfulFramesTransported": successful_frames,
            "meanSuccessfulTransportBatchSize": (
                round(successful_frames / successful_executions, 6)
                if successful_executions
                else None
            ),
            "counterDeltas": dict(sorted(transport_deltas[route].items())),
            "counterResetCounts": dict(sorted(transport_resets[route].items())),
        }
    latest_health = successful[-1]["health"]

    observed_telemetry_count = len(telemetry_camera_ids)
    observed_active_count = len(active_telemetry_camera_ids)
    if camera_basis == "equivalent":
        declared_count = equivalent_camera_count
        interpretation = "synthetic-or-replayed-camera-equivalent-load"
    else:
        declared_count = (
            len(expected_camera_ids)
            if expected_camera_ids
            else observed_active_count
        )
        interpretation = "distinct-physical-camera-streams"

    telemetry_epoch_observed = any(
        sample["health"]["pipelineTelemetry"].get("telemetryEpoch") is not None
        for sample in successful
    )
    first_alert_delivery = (
        (successful[0]["health"]["pipelineTelemetry"].get("alerts") or {}).get(
            "deliveryCoverage"
        )
        or {}
    )
    last_alert_delivery = (
        (successful[-1]["health"]["pipelineTelemetry"].get("alerts") or {}).get(
            "deliveryCoverage"
        )
        or {}
    )
    alert_delivery_coverage_observed = (
        bool(first_alert_delivery)
        and bool(last_alert_delivery)
        and advancing_server_snapshot_transitions > 0
        and alert_delivery_coverage_samples
        == advancing_server_snapshot_transitions
    )
    pending_at_start = int(first_alert_delivery.get("pending") or 0)
    pending_at_end = int(last_alert_delivery.get("pending") or 0)
    eligible_during_window = int(alert_delivery_deltas.get("eligibleCount", 0))
    delivered_during_window = int(alert_delivery_deltas.get("deliveredCount", 0))
    terminal_during_window = int(alert_delivery_deltas.get("terminalCount", 0))
    cancelled_during_window = int(alert_delivery_deltas.get("cancelledCount", 0))
    failed_attempts_during_window = int(
        alert_delivery_deltas.get("failedAttemptCount", 0)
    )
    untracked_failures_during_window = int(
        alert_delivery_deltas.get("untrackedFailureAttemptCount", 0)
    )
    censored_during_window = sum(
        int(alert_delivery_deltas.get(name, 0))
        for name in (
            "persistenceCensoredCount",
            "outcomeCensoredCount",
            "evictedPendingCount",
        )
    )
    delivery_denominator = pending_at_start + eligible_during_window
    resolved_during_window = (
        delivered_during_window
        + terminal_during_window
        + cancelled_during_window
    )
    delivery_accounting_balance = (
        delivery_denominator - resolved_during_window - pending_at_end
    )
    provider_latency = _render_histogram_accumulator(
        alert_histograms["firstPositiveToProviderSuccessMs"]
    )
    provider_latency_valid_samples = int(provider_latency.get("samples") or 0)
    provider_latency_invalid_samples = provider_latency.get("invalidSamples")
    provider_latency_sample_accounting = (
        provider_latency_invalid_samples is not None
        and provider_latency_valid_samples + int(provider_latency_invalid_samples)
        == delivered_during_window
    )
    alert_delivery_flow_accounted = (
        alert_delivery_coverage_observed
        and epoch_transitions == 0
        and not alert_delivery_resets
        and delivery_accounting_balance == 0
        and censored_during_window == 0
        and untracked_failures_during_window == 0
    )
    valid_for_provider_success_latency = (
        alert_delivery_flow_accounted
        and delivery_denominator > 0
        and delivered_during_window > 0
        and provider_latency_sample_accounting
        and int(provider_latency_invalid_samples or 0) == 0
        and provider_latency.get("invalidTransitions") == 0
        and provider_latency.get("counterResets") == 0
        and provider_latency.get("invalidCountResets") == 0
    )
    camera_set_complete = (
        all_expected_cameras_complete
        if expected_camera_ids
        else bool(active_telemetry_camera_ids)
    )
    valid_for_pipeline_deltas = (
        len(successful) >= 2
        and at_least_two_distinct_server_snapshots
        and server_snapshot_timestamps_monotonic
        and server_snapshots_fresh
        and sufficient_server_snapshot_duration_coverage
        and observed_seconds > 0
        and sufficient_duration_coverage
        and sufficient_health_sample_coverage
        and bool(telemetry_camera_ids)
        and telemetry_epoch_observed
        and camera_set_complete
    )

    return {
        "configuredDurationSeconds": configured_duration_seconds,
        "configuredPollIntervalSeconds": configured_poll_interval_seconds,
        "observedHealthIntervalSeconds": round(observed_seconds, 6),
        "measurementValidity": {
            "atLeastTwoHealthSamples": len(successful) >= 2,
            "allHealthSamplesHaveServerSnapshotTimestamp": (
                all_server_snapshot_timestamps_present
            ),
            "serverSnapshotTimestampsMonotonic": (
                server_snapshot_timestamps_monotonic
            ),
            "distinctServerSnapshotCount": distinct_server_snapshot_count,
            "atLeastTwoDistinctServerSnapshots": (
                at_least_two_distinct_server_snapshots
            ),
            "maximumServerSnapshotAgeSeconds": (
                round(maximum_server_snapshot_age_seconds, 6)
                if maximum_server_snapshot_age_seconds is not None
                else None
            ),
            "maximumAllowedServerSnapshotAgeSeconds": (
                MAX_HEALTH_SNAPSHOT_AGE_SECONDS
            ),
            "serverSnapshotsFresh": server_snapshots_fresh,
            "serverSnapshotSpanSeconds": round(
                server_snapshot_span_seconds, 6
            ),
            "serverSnapshotDurationCoverageRatio": round(
                server_snapshot_duration_coverage_ratio, 6
            ),
            "minimumServerSnapshotDurationCoverageRatio": (
                MIN_SERVER_SNAPSHOT_DURATION_COVERAGE
            ),
            "sufficientServerSnapshotDurationCoverage": (
                sufficient_server_snapshot_duration_coverage
            ),
            "positiveObservedInterval": observed_seconds > 0,
            "configuredDurationCoverageRatio": round(
                duration_coverage_ratio, 6
            ),
            "minimumDurationCoverageRatio": MIN_HEALTH_DURATION_COVERAGE,
            "sufficientDurationCoverage": sufficient_duration_coverage,
            "expectedHealthSamples": expected_health_samples,
            "successfulHealthSampleCoverageRatio": round(
                health_sample_coverage_ratio, 6
            ),
            "minimumHealthSampleCoverageRatio": MIN_HEALTH_SAMPLE_COVERAGE,
            "sufficientHealthSampleCoverage": sufficient_health_sample_coverage,
            "telemetryEpochObserved": telemetry_epoch_observed,
            "cameraTelemetryObserved": bool(telemetry_camera_ids),
            "decodedCameraTelemetryObserved": bool(active_telemetry_camera_ids),
            "explicitExpectedCameraSet": bool(expected_camera_ids),
            "allExpectedCamerasComplete": (
                all_expected_cameras_complete if expected_camera_ids else None
            ),
            "hardwareDecodeRequirementSatisfied": (
                all_expected_hardware_decode_satisfied
                if require_hardware_decode
                else None
            ),
            "maxFrameDimensionRequirementSatisfied": (
                all_expected_max_dimension_satisfied
                if require_max_frame_dimension is not None
                else None
            ),
            "alertDeliveryCoverageObserved": alert_delivery_coverage_observed,
            "alertDeliveryFlowAccounted": alert_delivery_flow_accounted,
            "validForProviderSuccessLatency": valid_for_provider_success_latency,
            "validForPipelineDeltas": valid_for_pipeline_deltas,
        },
        "expectedCameraCompleteness": expected_completeness,
        "capacityProvenance": {
            "basis": camera_basis,
            "interpretation": interpretation,
            "observedTelemetryCameraCount": observed_telemetry_count,
            "observedActiveCameraCount": observed_active_count,
            "declaredCameraCount": declared_count,
            "cameraEquivalentResultsAreNotPhysicalCameraCertification": camera_basis
            == "equivalent",
        },
        "latencyProvenance": {
            "decodedIngressToResultMs": "local decoded-frame ingress to inference result; not sensor capture age",
            "decodedIngressToObservationMs": "local decoded-frame ingress through result application and alert observation/submission; not sensor capture age",
            "submitToResultMs": "scheduler submission to inference result",
            "publishedFrameAgeSeconds": "health/JPEG publication freshness; not sensor capture age",
            "captureDropCount": "videorate drops plus application-drain drops and appsink latest-buffer replacements when the runtime exposes them; consult each camera's lower-bound marker",
            "firstPositiveToProviderSuccessMs": "first positive through provider success, entirely on the app process-monotonic clock; PostgreSQL timestamps are durability metadata only",
        },
        "health": {
            "attemptedSamples": len(samples),
            "successfulSamples": len(successful),
            "serverSnapshotTimestampSamples": len(
                server_snapshot_timestamps
            ),
            "distinctServerSnapshotCount": distinct_server_snapshot_count,
            "serverSnapshotSpanSeconds": round(
                server_snapshot_span_seconds, 6
            ),
            "advancingServerSnapshotTransitions": (
                advancing_server_snapshot_transitions
            ),
            "skippedNonadvancingServerSnapshotTransitions": (
                skipped_nonadvancing_server_snapshot_transitions
            ),
            "errorCounts": dict(sorted(health_errors.items())),
            "statusCounts": dict(
                sorted(
                    Counter(
                        str(sample["health"]["status"]) for sample in successful
                    ).items()
                )
            ),
            "endpointSnapshotAgeSeconds": _distribution(
                float(sample["health"]["snapshotAgeSeconds"])
                for sample in successful
                if _finite_number(sample["health"].get("snapshotAgeSeconds"))
                is not None
            ),
            "httpRoundTripMs": _distribution(
                float(sample["httpRoundTripMs"])
                for sample in successful
                if _finite_number(sample.get("httpRoundTripMs")) is not None
            ),
            "clockSkewSamples": sum(
                bool(sample["health"].get("snapshotClockSkew"))
                for sample in successful
            ),
            "telemetryEpochTransitions": epoch_transitions,
        },
        "cameras": camera_reports,
        "fairness": {
            "metric": "demand-normalized-jain-index",
            "jainIndex": _demand_normalized_jain(completed_fps, effective_demand_fps),
            "cameraCount": len(effective_demand_fps),
            "cameraIds": sorted(effective_demand_fps),
            "explicitExpectedCameraSet": bool(expected_camera_ids),
            "allExpectedCamerasComplete": (
                all_expected_cameras_complete if expected_camera_ids else None
            ),
            "normalization": "effective inference FPS divided by explicit target FPS, or by observed decoded-ingress FPS when no explicit target is supplied; satisfaction capped at 1.0",
        },
        "alerts": {
            "realTimeOnly": True,
            "latency": {
                name: _render_histogram_accumulator(alert_histograms[name])
                for name in ALERT_LATENCY_NAMES
            },
            "deliveryCoverage": {
                "unit": "initial-external-delivery-target",
                "pendingAtStart": pending_at_start,
                "eligibleDuringWindow": eligible_during_window,
                "denominator": delivery_denominator,
                "deliveredDuringWindow": delivered_during_window,
                "terminalDuringWindow": terminal_during_window,
                "cancelledDuringWindow": cancelled_during_window,
                "pendingAtEnd": pending_at_end,
                "failedAttemptsDuringWindow": failed_attempts_during_window,
                "untrackedFailureAttemptsDuringWindow": (
                    untracked_failures_during_window
                ),
                "censoredTimingOrOutcomeEvents": censored_during_window,
                "accountingBalance": delivery_accounting_balance,
                "counterDeltas": dict(sorted(alert_delivery_deltas.items())),
                "counterResetCounts": dict(sorted(alert_delivery_resets.items())),
                "deliveredCoverageRatio": (
                    round(delivered_during_window / delivery_denominator, 6)
                    if delivery_denominator
                    else None
                ),
                "providerSuccessLatencyValidSamples": (
                    provider_latency_valid_samples
                ),
                "providerSuccessLatencyInvalidSamples": (
                    provider_latency_invalid_samples
                ),
                "providerSuccessLatencySampleCoverageRatio": (
                    round(
                        provider_latency_valid_samples
                        / delivered_during_window,
                        6,
                    )
                    if delivered_during_window
                    else None
                ),
                "flowAccounted": alert_delivery_flow_accounted,
                "validForProviderSuccessLatency": (
                    valid_for_provider_success_latency
                ),
            },
        },
        "sharedInferenceScheduler": {
            "provenance": "scheduler admission cohorts; not proof of one model-server or GPU batch execution",
            "latestGauges": latest_health["sharedInferenceScheduler"].get("gauges", {}),
            "counterDeltas": dict(sorted(scheduler_deltas.items())),
            "counterResetCounts": dict(sorted(scheduler_resets.items())),
            "schedulerCohortDeltas": batch_deltas,
            "dispatchedItemsFromCohortCounters": dispatched_from_batches,
            "meanSchedulerCohortSize": (
                round(dispatched_from_batches / cohort_count, 6)
                if cohort_count
                else None
            ),
            "latestRollingWindowLatency": latest_health["sharedInferenceScheduler"].get(
                "latency", {}
            ),
        },
        "inferenceTransport": {
            "provenance": "successful fixed-route model transport executions; frames are transport frame-slots and are not de-duplicated across routes",
            "routes": transport_reports,
            "aggregateSuccessfulBatchExecutions": total_transport_executions,
            "aggregateSuccessfulFramesTransported": total_transport_frames,
            "aggregateMeanSuccessfulTransportBatchSize": (
                round(total_transport_frames / total_transport_executions, 6)
                if total_transport_executions
                else None
            ),
        },
        "vlmEnrichment": {
            "latestGauges": latest_health["vlmEnrichment"].get("gauges", {}),
            "counterDeltas": dict(sorted(vlm_deltas.items())),
            "counterResetCounts": dict(sorted(vlm_resets.items())),
            "outsidePrimaryLoop": True,
        },
        "resources": _summarize_tegrastats(
            tegrastats_samples,
            source=tegrastats_source,
            collection_errors=tegrastats_errors,
            thermal_warning_c=thermal_warning_c,
            configured_duration_seconds=configured_duration_seconds,
            configured_interval_seconds=tegrastats_interval_seconds,
        ),
    }


def _insufficient_health_summary(
    samples: Sequence[Mapping[str, Any]],
    tegrastats_samples: Sequence[Mapping[str, Any]],
    *,
    configured_duration_seconds: float,
    configured_poll_interval_seconds: float,
    demand_fps: Mapping[str, float],
    require_hardware_decode: bool,
    require_max_frame_dimension: int | None,
    camera_basis: str,
    equivalent_camera_count: int | None,
    tegrastats_source: str,
    tegrastats_interval_seconds: float,
    tegrastats_errors: Mapping[str, int],
    thermal_warning_c: float,
) -> dict[str, Any]:
    health_errors = Counter(
        str(sample["healthError"])
        for sample in samples
        if isinstance(sample.get("healthError"), str)
    )
    expected_camera_ids = sorted(demand_fps)
    return {
        "configuredDurationSeconds": configured_duration_seconds,
        "configuredPollIntervalSeconds": configured_poll_interval_seconds,
        "observedHealthIntervalSeconds": 0.0,
        "measurementValidity": {
            "atLeastTwoHealthSamples": False,
            "allHealthSamplesHaveServerSnapshotTimestamp": False,
            "serverSnapshotTimestampsMonotonic": False,
            "distinctServerSnapshotCount": 0,
            "atLeastTwoDistinctServerSnapshots": False,
            "maximumServerSnapshotAgeSeconds": None,
            "maximumAllowedServerSnapshotAgeSeconds": (
                MAX_HEALTH_SNAPSHOT_AGE_SECONDS
            ),
            "serverSnapshotsFresh": False,
            "serverSnapshotSpanSeconds": 0.0,
            "serverSnapshotDurationCoverageRatio": 0.0,
            "minimumServerSnapshotDurationCoverageRatio": (
                MIN_SERVER_SNAPSHOT_DURATION_COVERAGE
            ),
            "sufficientServerSnapshotDurationCoverage": False,
            "positiveObservedInterval": False,
            "configuredDurationCoverageRatio": 0.0,
            "minimumDurationCoverageRatio": MIN_HEALTH_DURATION_COVERAGE,
            "sufficientDurationCoverage": False,
            "expectedHealthSamples": (
                math.floor(
                    configured_duration_seconds
                    / configured_poll_interval_seconds
                )
                + 1
            ),
            "successfulHealthSampleCoverageRatio": 0.0,
            "minimumHealthSampleCoverageRatio": MIN_HEALTH_SAMPLE_COVERAGE,
            "sufficientHealthSampleCoverage": False,
            "telemetryEpochObserved": False,
            "cameraTelemetryObserved": False,
            "decodedCameraTelemetryObserved": False,
            "explicitExpectedCameraSet": bool(expected_camera_ids),
            "allExpectedCamerasComplete": (
                False if expected_camera_ids else None
            ),
            "hardwareDecodeRequirementSatisfied": (
                False if require_hardware_decode else None
            ),
            "maxFrameDimensionRequirementSatisfied": (
                False if require_max_frame_dimension is not None else None
            ),
            "alertDeliveryCoverageObserved": False,
            "alertDeliveryFlowAccounted": False,
            "validForProviderSuccessLatency": False,
            "validForPipelineDeltas": False,
        },
        "expectedCameraCompleteness": {
            "source": (
                "explicit-camera-demand-fps"
                if expected_camera_ids
                else "observed-active-camera-fallback"
            ),
            "expectedCameraIds": expected_camera_ids,
            "expectedCameraCount": len(expected_camera_ids),
            "completeExpectedCameraIds": [],
            "incompleteExpectedCameraIds": expected_camera_ids,
            "allExpectedCamerasComplete": (
                False if expected_camera_ids else None
            ),
            "activeTelemetryCameraIds": [],
            "excludedObservedCameraIds": [],
            "requirements": {
                "hardwareDecode": bool(require_hardware_decode),
                "maxFrameDimension": require_max_frame_dimension,
            },
        },
        "capacityProvenance": {
            "basis": camera_basis,
            "interpretation": (
                "synthetic-or-replayed-camera-equivalent-load"
                if camera_basis == "equivalent"
                else "distinct-physical-camera-streams"
            ),
            "observedTelemetryCameraCount": 0,
            "observedActiveCameraCount": 0,
            "declaredCameraCount": (
                equivalent_camera_count
                if camera_basis == "equivalent"
                else len(expected_camera_ids)
            ),
            "cameraEquivalentResultsAreNotPhysicalCameraCertification": camera_basis
            == "equivalent",
        },
        "latencyProvenance": {
            "decodedIngressToResultMs": "local decoded-frame ingress to inference result; not sensor capture age",
            "decodedIngressToObservationMs": "local decoded-frame ingress through result application and alert observation/submission; not sensor capture age",
            "submitToResultMs": "scheduler submission to inference result",
            "publishedFrameAgeSeconds": "health/JPEG publication freshness; not sensor capture age",
            "captureDropCount": "videorate drops plus application-drain drops and appsink latest-buffer replacements when the runtime exposes them; consult each camera's lower-bound marker",
            "firstPositiveToProviderSuccessMs": "first positive through provider success, entirely on the app process-monotonic clock; PostgreSQL timestamps are durability metadata only",
        },
        "health": {
            "attemptedSamples": len(samples),
            "successfulSamples": 0,
            "serverSnapshotTimestampSamples": 0,
            "distinctServerSnapshotCount": 0,
            "serverSnapshotSpanSeconds": 0.0,
            "advancingServerSnapshotTransitions": 0,
            "skippedNonadvancingServerSnapshotTransitions": 0,
            "errorCounts": dict(sorted(health_errors.items())),
            "statusCounts": {},
            "endpointSnapshotAgeSeconds": _distribution([]),
            "httpRoundTripMs": _distribution([]),
            "clockSkewSamples": 0,
            "telemetryEpochTransitions": 0,
        },
        "cameras": {},
        "fairness": {
            "metric": "demand-normalized-jain-index",
            "jainIndex": None,
            "cameraCount": len(expected_camera_ids),
            "cameraIds": expected_camera_ids,
            "explicitExpectedCameraSet": bool(expected_camera_ids),
            "allExpectedCamerasComplete": (
                False if expected_camera_ids else None
            ),
            "normalization": "unavailable without valid health telemetry",
        },
        "alerts": {
            "realTimeOnly": True,
            "latency": {},
            "deliveryCoverage": {
                "unit": "initial-external-delivery-target",
                "pendingAtStart": 0,
                "eligibleDuringWindow": 0,
                "denominator": 0,
                "deliveredDuringWindow": 0,
                "terminalDuringWindow": 0,
                "cancelledDuringWindow": 0,
                "pendingAtEnd": 0,
                "failedAttemptsDuringWindow": 0,
                "untrackedFailureAttemptsDuringWindow": 0,
                "censoredTimingOrOutcomeEvents": 0,
                "accountingBalance": 0,
                "counterDeltas": {},
                "counterResetCounts": {},
                "deliveredCoverageRatio": None,
                "providerSuccessLatencyValidSamples": 0,
                "providerSuccessLatencyInvalidSamples": None,
                "providerSuccessLatencySampleCoverageRatio": None,
                "flowAccounted": False,
                "validForProviderSuccessLatency": False,
            },
        },
        "sharedInferenceScheduler": {
            "provenance": "scheduler admission cohorts; not proof of one model-server or GPU batch execution",
            "latestGauges": {},
            "counterDeltas": {},
            "counterResetCounts": {},
            "schedulerCohortDeltas": {"1": 0, "2": 0, "4": 0},
            "dispatchedItemsFromCohortCounters": 0,
            "meanSchedulerCohortSize": None,
            "latestRollingWindowLatency": {},
        },
        "inferenceTransport": {
            "provenance": "successful fixed-route model transport executions; frames are transport frame-slots and are not de-duplicated across routes",
            "routes": {},
            "aggregateSuccessfulBatchExecutions": 0,
            "aggregateSuccessfulFramesTransported": 0,
            "aggregateMeanSuccessfulTransportBatchSize": None,
        },
        "vlmEnrichment": {
            "latestGauges": {},
            "counterDeltas": {},
            "counterResetCounts": {},
            "outsidePrimaryLoop": True,
        },
        "resources": _summarize_tegrastats(
            tegrastats_samples,
            source=tegrastats_source,
            collection_errors=tegrastats_errors,
            thermal_warning_c=thermal_warning_c,
            configured_duration_seconds=configured_duration_seconds,
            configured_interval_seconds=tegrastats_interval_seconds,
        ),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_demand_fps(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        camera_id, separator, raw_fps = value.partition("=")
        if not separator or _safe_camera_id(camera_id) is None:
            raise ValueError("camera demand must use CAMERA_ID=FPS")
        fps = _finite_number(raw_fps)
        if fps is None or fps <= 0:
            raise ValueError("camera demand FPS must be positive and finite")
        if camera_id in result:
            raise ValueError("camera demand IDs must be unique")
        result[camera_id] = fps
    return result


def _collect_health_samples(
    *,
    url: str,
    duration_seconds: float,
    poll_interval_seconds: float,
    request_timeout_seconds: float,
    headers: Mapping[str, str],
) -> tuple[list[dict[str, Any]], float, float]:
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + duration_seconds
    poll_count = math.floor(duration_seconds / poll_interval_seconds) + 1
    for sequence in range(poll_count):
        scheduled = started + sequence * poll_interval_seconds
        now = time.monotonic()
        if now < scheduled:
            time.sleep(scheduled - now)
        now = time.monotonic()
        if now > deadline:
            break
        if sequence and now - scheduled >= poll_interval_seconds:
            # A slow or unavailable endpoint cannot turn a bounded benchmark
            # into an unbounded backlog of immediately retried polls.
            continue
        sample: dict[str, Any] = {"monotonic": now}
        try:
            health, elapsed_ms = _fetch_health(
                url,
                timeout_seconds=request_timeout_seconds,
                headers=headers,
            )
            sample["health"] = health
            sample["httpRoundTripMs"] = elapsed_ms
        except Exception as exc:
            # Exception messages can contain endpoint URLs or proxy credentials.
            sample["healthError"] = type(exc).__name__
        samples.append(sample)
    return samples, started, time.monotonic()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-url", default="http://127.0.0.1:8000/api/health")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    parser.add_argument("--label", default="live-pipeline-runtime")
    parser.add_argument("--out", type=Path, required=True)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--bearer-token-env")
    authentication.add_argument("--token-env")
    parser.add_argument(
        "--camera-demand-fps",
        action="append",
        default=[],
        metavar="CAMERA_ID=FPS",
    )
    parser.add_argument(
        "--require-hardware-decode",
        action="store_true",
        help="invalidate unless every explicitly demanded RTSP camera is using NVDEC",
    )
    parser.add_argument(
        "--require-max-frame-dimension",
        type=int,
        metavar="PIXELS",
        help="invalidate unless every explicitly demanded camera is decoded at or below this dimension",
    )
    parser.add_argument(
        "--camera-basis",
        choices=("physical", "equivalent"),
        default="physical",
    )
    parser.add_argument("--equivalent-camera-count", type=int)
    tegra = parser.add_mutually_exclusive_group()
    tegra.add_argument("--tegrastats-log", type=Path)
    tegra.add_argument("--spawn-tegrastats", action="store_true")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=1000)
    parser.add_argument("--thermal-warning-c", type=float, default=85.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not LABEL_RE.fullmatch(args.label):
        parser.error("label must be a bounded identifier")
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0
        or args.duration > MAX_DURATION_SECONDS
    ):
        parser.error("duration must be positive and no more than 24 hours")
    if (
        not math.isfinite(args.poll_interval)
        or args.poll_interval < MIN_POLL_INTERVAL_SECONDS
        or args.poll_interval > args.duration
    ):
        parser.error(
            "poll interval must be at least 0.1 seconds and no longer than duration"
        )
    if math.floor(args.duration / args.poll_interval) + 1 > MAX_HEALTH_SAMPLES:
        parser.error("duration and poll interval request too many health samples")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("request timeout must be positive and finite")
    if args.tegrastats_interval_ms < 100:
        parser.error("tegrastats interval must be at least 100 ms")
    if not math.isfinite(args.thermal_warning_c):
        parser.error("thermal warning threshold must be finite")
    if args.require_max_frame_dimension is not None and not (
        1 <= args.require_max_frame_dimension <= 65_535
    ):
        parser.error("required max frame dimension must be between 1 and 65535")
    if args.camera_basis == "equivalent":
        if args.equivalent_camera_count is None or args.equivalent_camera_count < 1:
            parser.error(
                "equivalent camera basis requires a positive --equivalent-camera-count"
            )
    elif args.equivalent_camera_count is not None:
        parser.error(
            "--equivalent-camera-count is valid only for equivalent camera basis"
        )
    try:
        _validate_health_url(args.health_url)
        headers = _auth_headers_from_env(
            bearer_token_env=args.bearer_token_env,
            token_env=args.token_env,
        )
        demand_fps = _parse_demand_fps(args.camera_demand_fps)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if (
        args.require_hardware_decode
        or args.require_max_frame_dimension is not None
    ) and not demand_fps:
        parser.error(
            "hardware decode and frame dimension requirements need at least one --camera-demand-fps"
        )

    tegrastats = _TegrastatsCollector(
        log_path=args.tegrastats_log,
        spawn=args.spawn_tegrastats,
        interval_ms=args.tegrastats_interval_ms,
    )
    resource_started = time.monotonic()
    tegrastats.start()
    try:
        samples, started, ended = _collect_health_samples(
            url=args.health_url,
            duration_seconds=args.duration,
            poll_interval_seconds=args.poll_interval,
            request_timeout_seconds=args.request_timeout,
            headers=headers,
        )
    finally:
        tegrastats.stop()
    resource_ended = time.monotonic()
    resource_samples, resource_errors = tegrastats.result(
        started=resource_started,
        ended=resource_ended,
    )
    exit_code = 0
    outcome = "valid"
    try:
        summary = _summarize(
            samples,
            resource_samples,
            configured_duration_seconds=args.duration,
            configured_poll_interval_seconds=args.poll_interval,
            demand_fps=demand_fps,
            require_hardware_decode=args.require_hardware_decode,
            require_max_frame_dimension=args.require_max_frame_dimension,
            camera_basis=args.camera_basis,
            equivalent_camera_count=args.equivalent_camera_count,
            tegrastats_source=tegrastats.source,
            tegrastats_interval_seconds=args.tegrastats_interval_ms / 1000.0,
            tegrastats_errors=resource_errors,
            thermal_warning_c=args.thermal_warning_c,
        )
    except RuntimeError:
        summary = _insufficient_health_summary(
            samples,
            resource_samples,
            configured_duration_seconds=args.duration,
            configured_poll_interval_seconds=args.poll_interval,
            demand_fps=demand_fps,
            require_hardware_decode=args.require_hardware_decode,
            require_max_frame_dimension=args.require_max_frame_dimension,
            camera_basis=args.camera_basis,
            equivalent_camera_count=args.equivalent_camera_count,
            tegrastats_source=tegrastats.source,
            tegrastats_interval_seconds=args.tegrastats_interval_ms / 1000.0,
            tegrastats_errors=resource_errors,
            thermal_warning_c=args.thermal_warning_c,
        )
        outcome = "insufficient-health-telemetry"
        exit_code = 2
    else:
        if not summary["measurementValidity"]["validForPipelineDeltas"]:
            outcome = "insufficient-health-telemetry"
            exit_code = 2
        if (
            exit_code == 0
            and tegrastats.source != "none"
            and summary["resources"]["measurementValidity"]["valid"] is not True
        ):
            outcome = "resource-telemetry-unavailable"
            exit_code = 3

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "label": args.label,
        "outcome": outcome,
        "reportId": hashlib.sha256(
            f"{args.label}:{started:.9f}:{ended:.9f}".encode("utf-8")
        ).hexdigest()[:16],
        "summary": summary,
    }
    _atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
