#!/usr/bin/env python3
"""Replay primary-plus-conditional-specialist camera load against a model server."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


PPE_CLASSES = ["motorcycle helmet", "rider helmet", "helmet"]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _resize(frame: np.ndarray, maximum_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, maximum_dimension / max(height, width))
    if scale == 1.0:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )


def _phase_offset(camera_index: int, cameras: int, period: float, mode: str) -> float:
    if mode == "aligned":
        return 0.0
    if mode == "paired":
        groups = math.ceil(cameras / 2)
        return (camera_index // 2) * period / groups
    return camera_index * period / cameras


def _specialist_due(sequence: int, duty: float) -> bool:
    """Spread specialist work deterministically without random benchmark noise."""
    if duty <= 0:
        return False
    if duty >= 1:
        return True
    return math.floor((sequence + 1) * duty) > math.floor(sequence * duty)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--cameras", type=int, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--specialist-duty", type=float, default=0.111)
    parser.add_argument("--phone-probe-interval", type=float, default=1.0)
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--admission-timeout", type=float, default=0.05)
    parser.add_argument(
        "--phase-mode",
        choices=("aligned", "paired", "staggered"),
        default="staggered",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cameras < 1 or args.fps <= 0 or args.duration <= 0:
        parser.error("cameras, fps, and duration must be positive")
    if not 0 <= args.specialist_duty <= 1:
        parser.error("specialist-duty must be between 0 and 1")
    if args.max_inflight < 1 or args.admission_timeout < 0:
        parser.error("max-inflight must be positive and admission-timeout non-negative")

    frame_sets: list[dict[int, np.ndarray]] = []
    for path in args.frames:
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            parser.error(f"could not decode frame: {path}")
        frame_sets.append({
            640: _resize(source, 640),
            960: _resize(source, 960),
        })

    token = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
    admission = threading.BoundedSemaphore(args.max_inflight)
    barrier = threading.Barrier(args.cameras + 1)
    period = 1.0 / args.fps
    probe_every = max(1, round(args.phone_probe_interval * args.fps))
    reports: list[dict] = [{} for _ in range(args.cameras)]

    def post(camera_index: int, sequence: int, phone_probe: bool, specialist: bool) -> None:
        maximum_dimension = 960 if phone_probe else 640
        frame = frame_sets[camera_index % len(frame_sets)][maximum_dimension]
        batch = [
            {
                "request_id": f"coco-{camera_index}-{sequence}",
                "model_key": "coco_primary",
                "conf": 0.15 if phone_probe else 0.3,
                "device": "cuda",
                "imgsz": maximum_dimension,
                "classes": [],
            }
        ]
        if specialist:
            batch.append(
                {
                    "request_id": f"ppe-{camera_index}-{sequence}",
                    "model_key": "ppe_specialist",
                    "conf": 0.3,
                    "device": "cuda",
                    "imgsz": 640,
                    "classes": PPE_CLASSES,
                }
            )
        height, width, channels = frame.shape
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Inference-Batch": json.dumps(batch, separators=(",", ":")),
            "X-Rakshak-Frame-Width": str(width),
            "X-Rakshak-Frame-Height": str(height),
            "X-Rakshak-Frame-Channels": str(channels),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"{args.url.rstrip('/')}/api/infer/raw/batch",
            data=frame.tobytes(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10.0) as response:
            if response.status != 200:
                raise RuntimeError(f"model server returned HTTP {response.status}")
            payload = json.load(response)
        expected = {item["request_id"] for item in batch}
        if set(payload.get("results") or {}) != expected:
            raise RuntimeError("model server returned an incomplete grouped result")

    def run_camera(camera_index: int) -> None:
        barrier.wait()
        started = benchmark_start + _phase_offset(
            camera_index,
            args.cameras,
            period,
            args.phase_mode,
        )
        deadline = benchmark_start + args.duration
        sequence = 0
        latencies: list[float] = []
        overloads = 0
        failures = 0
        specialist_requests = 0
        while True:
            scheduled = started + sequence * period
            if scheduled >= deadline:
                break
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            specialist = _specialist_due(sequence, args.specialist_duty)
            if not admission.acquire(timeout=args.admission_timeout):
                overloads += 1
                sequence += 1
                continue
            request_started = time.perf_counter()
            try:
                post(
                    camera_index,
                    sequence,
                    sequence % probe_every == 0,
                    specialist,
                )
            except Exception:
                failures += 1
            else:
                latencies.append((time.perf_counter() - request_started) * 1000.0)
                specialist_requests += int(specialist)
            finally:
                admission.release()
            sequence += 1
        reports[camera_index] = {
            "camera": camera_index,
            "scheduled": sequence,
            "successes": len(latencies),
            "specialist_requests": specialist_requests,
            "overloads": overloads,
            "failures": failures,
            "achieved_fps": round(len(latencies) / args.duration, 3),
            "latencies_ms": latencies,
        }

    benchmark_start = time.monotonic() + 0.25
    threads = [
        threading.Thread(target=run_camera, args=(index,), daemon=True)
        for index in range(args.cameras)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=args.duration + 15.0)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("camera load worker did not stop")

    all_latencies = [
        latency
        for report in reports
        for latency in report.get("latencies_ms", [])
    ]
    for report in reports:
        report.pop("latencies_ms", None)
    result = {
        "cameras": args.cameras,
        "target_fps": args.fps,
        "duration_seconds": args.duration,
        "specialist_duty_target": args.specialist_duty,
        "specialist_requests": sum(report["specialist_requests"] for report in reports),
        "phase_mode": args.phase_mode,
        "requests": len(all_latencies),
        "overloads": sum(report["overloads"] for report in reports),
        "failures": sum(report["failures"] for report in reports),
        "minimum_camera_fps": min(report["achieved_fps"] for report in reports),
        "latency_ms": {
            "median": round(statistics.median(all_latencies), 3) if all_latencies else None,
            "p95": round(_percentile(all_latencies, 0.95), 3) if all_latencies else None,
            "maximum": round(max(all_latencies), 3) if all_latencies else None,
        },
        "per_camera": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
