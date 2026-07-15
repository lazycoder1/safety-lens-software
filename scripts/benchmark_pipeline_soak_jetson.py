#!/usr/bin/env python3
"""Collect a credential-safe Jetson camera-pipeline soak report.

The collector reads the public health endpoint, Docker's one-shot container
stats, and ``tegrastats``.  It never reads camera configuration or serializes
container environments, so benchmark evidence cannot contain stream URLs,
passwords, or model-server tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB")
_GPU_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
_VIC_RE = re.compile(r"\bVIC_FREQ\s+(\d+)%")
_POWER_RE = re.compile(r"\bVDD_IN\s+(\d+)mW")
_TEMP_RE = re.compile(r"\b([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
_MEMORY_RE = re.compile(r"^\s*([0-9.]+)\s*([KMGT]i?B)\s*$", re.I)


class CounterResetError(RuntimeError):
    """Raised when a cumulative runtime counter decreases during a soak."""


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "samples": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p95": round(float(_percentile(values, 0.95)), 3),
        "p99": round(float(_percentile(values, 0.99)), 3),
        "maximum": round(max(values), 3),
    }


def _jain_index(values: list[float]) -> float | None:
    bounded = [max(0.0, float(value)) for value in values]
    if not bounded or not any(bounded):
        return None
    denominator = len(bounded) * sum(value * value for value in bounded)
    return round(sum(bounded) ** 2 / denominator, 6) if denominator else None


def _parse_tegrastats_line(line: str) -> dict[str, Any]:
    """Parse metrics present on JetPack 5 while tolerating missing fields."""
    parsed: dict[str, Any] = {}
    for key, pattern in (
        ("ram", _RAM_RE),
        ("swap", _SWAP_RE),
        ("gpu", _GPU_RE),
        ("vic", _VIC_RE),
        ("power", _POWER_RE),
    ):
        match = pattern.search(line)
        if match is None:
            continue
        if key == "ram":
            parsed["ram_used_mb"] = int(match.group(1))
            parsed["ram_total_mb"] = int(match.group(2))
        elif key == "swap":
            parsed["swap_used_mb"] = int(match.group(1))
            parsed["swap_total_mb"] = int(match.group(2))
        elif key == "gpu":
            parsed["gpu_percent"] = int(match.group(1))
        elif key == "vic":
            parsed["vic_percent"] = int(match.group(1))
        else:
            parsed["input_power_w"] = round(int(match.group(1)) / 1000.0, 3)
    temperatures = {
        name.lower(): float(value) for name, value in _TEMP_RE.findall(line)
    }
    if temperatures:
        parsed["temperatures_c"] = temperatures
    return parsed


def _memory_to_mib(value: str) -> float | None:
    match = _MEMORY_RE.match(value)
    if match is None:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "kb": 1 / 1024,
        "kib": 1 / 1024,
        "mb": 1,
        "mib": 1,
        "gb": 1024,
        "gib": 1024,
        "tb": 1024 * 1024,
        "tib": 1024 * 1024,
    }
    return amount * factors[unit]


def _safe_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Allowlist only operational fields known not to contain credentials."""
    cameras = []
    for camera in payload.get("cameras") or []:
        cameras.append(
            {
                "id": str(camera.get("id")),
                "frameFresh": bool(camera.get("frameFresh")),
                "workerRunning": bool(camera.get("workerRunning")),
                "lastFrameAgeSeconds": camera.get("lastFrameAgeSeconds"),
                "runtimeStatus": camera.get("runtimeStatus"),
                "connection": {
                    key: (camera.get("connection") or {}).get(key)
                    for key in (
                        "captureBackend",
                        "hardwareAccelerationActive",
                        "hardwareFallback",
                        "outageActive",
                        "totalFailureCount",
                    )
                },
                "inference": {
                    key: (camera.get("inference") or {}).get(key)
                    for key in (
                        "successCount",
                        "overloadDropCount",
                        "failureCount",
                        "lastSuccessAgeSeconds",
                    )
                },
            }
        )
    alert_pipeline = payload.get("alertPipeline") or {}
    return {
        "status": payload.get("status"),
        "reasons": list(payload.get("reasons") or []),
        "alertsCount": payload.get("alerts_count"),
        "cameras": cameras,
        "inferenceTransport": _safe_numeric_tree(
            payload.get("inferenceTransport") or {}
        ),
        "alertPipeline": {
            key: alert_pipeline.get(key)
            for key in (
                "submitted",
                "persisted",
                "persistence_failures",
                "delivery_failures",
                "backpressure_events",
                "persist_queue_depth",
                "delivery_queue_depth",
            )
        },
    }


def _safe_numeric_tree(value: Any) -> Any:
    """Recursively retain only dictionaries and scalar operational metrics."""
    if isinstance(value, dict):
        return {
            str(key): safe
            for key, item in value.items()
            if (safe := _safe_numeric_tree(item)) is not None
        }
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def _get_health(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        return _safe_health(json.load(response))


def _docker_stats(containers: list[str]) -> dict[str, Any]:
    if not containers:
        return {}
    command = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        *containers,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    result = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = str(item.get("Name") or item.get("Container") or "unknown")
        memory_used = str(item.get("MemUsage") or "").split("/", 1)[0].strip()
        result[name] = {
            "cpu_percent": float(str(item.get("CPUPerc") or "0").rstrip("%") or 0),
            "memory_used_mib": _memory_to_mib(memory_used),
            "pids": int(item.get("PIDs") or 0),
        }
    return result


class _TegrastatsReader:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_ms = max(100, round(interval_seconds * 1000))
        self.samples: list[dict[str, Any]] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            parsed = _parse_tegrastats_line(line)
            if parsed:
                self.samples.append({"monotonic": time.monotonic(), **parsed})

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)


def _camera_map(health: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {camera["id"]: camera for camera in health.get("cameras") or []}


def _counter_delta(first: dict[str, Any], last: dict[str, Any], key: str) -> int:
    try:
        first_value = int(first.get(key) or 0)
        last_value = int(last.get(key) or 0)
    except (TypeError, ValueError):
        return 0
    if last_value < first_value:
        raise CounterResetError(f"counter reset during measurement: {key}")
    return last_value - first_value


def _validate_counter_series(
    counter_maps: list[dict[str, Any]],
    key: str,
    *,
    context: str,
) -> None:
    """Reject a cumulative counter decrease anywhere in the sampled window."""
    previous: int | None = None
    for counters in counter_maps:
        try:
            current = int(counters.get(key) or 0)
        except (TypeError, ValueError):
            current = 0
        if previous is not None and current < previous:
            raise CounterResetError(
                f"counter reset during measurement: {context}.{key} "
                f"({previous} -> {current})"
            )
        previous = current


def _stale_gate_failed(
    camera_reports: dict[str, dict[str, Any]],
    expected_cameras: list[str],
) -> bool:
    """Treat a stale camera or an absent expected camera as a failed gate."""
    expected = set(expected_cameras)
    return any(
        bool(camera.get("stale_health_samples"))
        or (
            camera_id in expected
            and (
                not camera.get("present")
                or bool(camera.get("missing_health_samples"))
            )
        )
        for camera_id, camera in camera_reports.items()
    )


def _summarize(
    samples: list[dict[str, Any]],
    tegra_samples: list[dict[str, Any]],
    *,
    duration_seconds: float,
    expected_cameras: list[str],
) -> dict[str, Any]:
    health_samples = [sample for sample in samples if sample.get("health")]
    if len(health_samples) < 2:
        raise RuntimeError("at least two successful health samples are required")
    first_health = health_samples[0]["health"]
    last_health = health_samples[-1]["health"]
    observed_health_interval = float(health_samples[-1]["monotonic"]) - float(
        health_samples[0]["monotonic"]
    )
    if observed_health_interval <= 0:
        raise RuntimeError("successful health samples must span a positive interval")
    first_cameras = _camera_map(first_health)
    last_cameras = _camera_map(last_health)
    camera_ids = sorted(set(first_cameras) | set(last_cameras) | set(expected_cameras))
    camera_reports = {}
    rates = []
    for camera_id in camera_ids:
        observed_cameras = [
            camera
            for sample in health_samples
            for camera in sample["health"].get("cameras") or []
            if camera.get("id") == camera_id
        ]
        observed_inference = [camera.get("inference") or {} for camera in observed_cameras]
        observed_connection = [
            camera.get("connection") or {} for camera in observed_cameras
        ]
        for key in ("successCount", "failureCount", "overloadDropCount"):
            _validate_counter_series(
                observed_inference,
                key,
                context=f"camera[{camera_id}].inference",
            )
        _validate_counter_series(
            observed_connection,
            "totalFailureCount",
            context=f"camera[{camera_id}].connection",
        )
        first = first_cameras.get(camera_id, {})
        last = last_cameras.get(camera_id, {})
        inference_first = first.get("inference") or {}
        inference_last = last.get("inference") or {}
        success_delta = _counter_delta(inference_first, inference_last, "successCount")
        completion_rate = success_delta / observed_health_interval
        rates.append(completion_rate)
        age_values = [
            float(camera["lastFrameAgeSeconds"])
            for sample in health_samples
            for camera in sample["health"].get("cameras") or []
            if camera.get("id") == camera_id
            and camera.get("lastFrameAgeSeconds") is not None
        ]
        backends = sorted(
            {
                str((camera.get("connection") or {}).get("captureBackend"))
                for sample in health_samples
                for camera in sample["health"].get("cameras") or []
                if camera.get("id") == camera_id
            }
        )
        stale_samples = sum(
            1
            for sample in health_samples
            for camera in sample["health"].get("cameras") or []
            if camera.get("id") == camera_id and not camera.get("frameFresh")
        )
        present_samples = sum(
            1
            for sample in health_samples
            if any(
                camera.get("id") == camera_id
                for camera in sample["health"].get("cameras") or []
            )
        )
        camera_reports[camera_id] = {
            "present": bool(first or last),
            "present_health_samples": present_samples,
            "missing_health_samples": len(health_samples) - present_samples,
            "inference_success_delta": success_delta,
            "inference_fps": round(completion_rate, 3),
            "inference_failure_delta": _counter_delta(
                inference_first, inference_last, "failureCount"
            ),
            "overload_drop_delta": _counter_delta(
                inference_first, inference_last, "overloadDropCount"
            ),
            "connection_failure_delta": _counter_delta(
                first.get("connection") or {},
                last.get("connection") or {},
                "totalFailureCount",
            ),
            "frame_age_seconds": _distribution(age_values),
            "stale_health_samples": stale_samples,
            "capture_backends": backends,
        }

    def metric_values(key: str) -> list[float]:
        return [float(sample[key]) for sample in tegra_samples if sample.get(key) is not None]

    temperatures: dict[str, list[float]] = {}
    for sample in tegra_samples:
        for name, value in (sample.get("temperatures_c") or {}).items():
            temperatures.setdefault(name, []).append(float(value))
    resource_summary = {
        "ram_used_mb": _distribution(metric_values("ram_used_mb")),
        "swap_used_mb": _distribution(metric_values("swap_used_mb")),
        "gpu_percent": _distribution(metric_values("gpu_percent")),
        "vic_percent": _distribution(metric_values("vic_percent")),
        "input_power_w": _distribution(metric_values("input_power_w")),
        "temperatures_c": {
            name: _distribution(values) for name, values in sorted(temperatures.items())
        },
    }
    container_reports: dict[str, Any] = {}
    container_names = sorted(
        {
            name
            for sample in samples
            for name in (sample.get("containers") or {})
        }
    )
    for name in container_names:
        cpu = [
            float(sample["containers"][name]["cpu_percent"])
            for sample in samples
            if name in (sample.get("containers") or {})
        ]
        memory = [
            float(sample["containers"][name]["memory_used_mib"])
            for sample in samples
            if name in (sample.get("containers") or {})
            and sample["containers"][name].get("memory_used_mib") is not None
        ]
        container_reports[name] = {
            "cpu_percent": _distribution(cpu),
            "memory_used_mib": _distribution(memory),
        }
    _validate_counter_series(
        health_samples_as_counters := [sample["health"] for sample in health_samples],
        "alertsCount",
        context="health",
    )
    return {
        "configured_duration_seconds": duration_seconds,
        "observed_health_interval_seconds": round(observed_health_interval, 3),
        "health_samples": len(health_samples),
        "health_errors": sum(bool(sample.get("healthError")) for sample in samples),
        "container_stats_errors": sum(
            bool(sample.get("containerStatsError")) for sample in samples
        ),
        "tegrastats_samples": len(tegra_samples),
        "status_counts": dict(
            Counter(str(sample["health"].get("status")) for sample in health_samples)
        ),
        "cameras": camera_reports,
        "inference_fps_jain": _jain_index(rates),
        "alerts_delta": _counter_delta(
            health_samples_as_counters[0],
            health_samples_as_counters[-1],
            "alertsCount",
        ),
        "resources": resource_summary,
        "containers": container_reports,
        "limitations": [
            "Health frame age is publication freshness, not RTSP sensor PTS age.",
            "The current runtime does not expose capture-to-result or alert-latency p99.",
            "Camera FPS fairness is scene-demand dependent; interpret with activity state.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--health-url", default="http://127.0.0.1:8000/api/health")
    parser.add_argument("--container", action="append", default=[])
    parser.add_argument("--expect-camera", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--fail-on-stale", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.sample_interval <= 0:
        parser.error("duration and sample interval must be positive; warmup cannot be negative")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.write_text("", encoding="utf-8")
    tegrastats = _TegrastatsReader(args.sample_interval)
    samples: list[dict[str, Any]] = []
    start = time.monotonic()
    measured_start = start + args.warmup
    deadline = measured_start + args.duration
    tegrastats.start()
    try:
        sequence = 0
        while time.monotonic() < deadline:
            scheduled = start + sequence * args.sample_interval
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            now = time.monotonic()
            sample: dict[str, Any] = {
                "label": args.label,
                "sequence": sequence,
                "phase": "warmup" if now < measured_start else "measure",
                "monotonic": now,
            }
            try:
                sample["health"] = _get_health(args.health_url, 5.0)
            except Exception as exc:
                sample["healthError"] = type(exc).__name__
            try:
                sample["containers"] = _docker_stats(args.container)
            except Exception as exc:
                sample["containerStatsError"] = type(exc).__name__
            samples.append(sample)
            with args.raw_out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
            sequence += 1
    finally:
        tegrastats.stop()

    measured_samples = [sample for sample in samples if sample["phase"] == "measure"]
    measured_tegra = [
        sample
        for sample in tegrastats.samples
        if measured_start <= sample["monotonic"] <= deadline
    ]
    report = {
        "schema_version": 2,
        "label": args.label,
        "summary": _summarize(
            measured_samples,
            measured_tegra,
            duration_seconds=args.duration,
            expected_cameras=args.expect_camera,
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not measured_tegra:
        return 3
    if args.fail_on_stale:
        cameras = report["summary"]["cameras"]
        if _stale_gate_failed(cameras, args.expect_camera):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
