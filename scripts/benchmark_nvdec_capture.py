#!/usr/bin/env python3
"""Benchmark the live NVDEC capture path without printing stream secrets."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _summarize_readings(
    readings: list[tuple[int, int, int, float]],
) -> dict[str, Any]:
    frame_counts = [reading[0] for reading in readings]
    read_seconds = [reading[3] for reading in readings]
    delivered_rates = [
        frames / max(duration, sys.float_info.epsilon)
        for frames, _width, _height, duration in readings
    ]
    dimensions = sorted({(reading[1], reading[2]) for reading in readings})
    return {
        "frame_counts": frame_counts,
        "read_seconds": read_seconds,
        "delivered_rates": delivered_rates,
        "dimensions": dimensions,
    }


def _camera(camera_id: str) -> dict:
    from config_manager import get_config

    cameras = get_config().get("cameras", {})
    if not isinstance(cameras, dict) or camera_id not in cameras:
        raise SystemExit(f"Unknown camera id: {camera_id}")
    camera = cameras[camera_id]
    if camera.get("stream_type") != "rtsp":
        raise SystemExit(f"Camera is not RTSP: {camera_id}")
    return camera


def main() -> int:
    from camera_connection import build_rtsp_url
    from gstreamer_capture import GStreamerCapture

    parser = argparse.ArgumentParser()
    parser.add_argument("camera_id")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--drop-interval", type=int, default=0)
    parser.add_argument("--max-dimension", type=int, default=960)
    parser.add_argument("--copies", type=int, default=1)
    args = parser.parse_args()
    if args.copies < 1 or args.copies > 32:
        parser.error("copies must be between 1 and 32")

    source = os.environ.get("SAFETYLENS_BENCHMARK_RTSP_URL", "").strip()
    if not source:
        camera = _camera(args.camera_id)
        source = build_rtsp_url(camera, include_credentials=True)
    open_started = time.monotonic()

    def open_capture():
        try:
            capture = GStreamerCapture(
                source,
                open_timeout_ms=5_000,
                read_timeout_ms=5_000,
                max_dimension=args.max_dimension,
                max_fps=args.max_fps,
                decoder_drop_interval=args.drop_interval,
            )
        except Exception:
            return None
        return capture if capture.isOpened() else None

    with ThreadPoolExecutor(max_workers=args.copies) as pool:
        captures = list(pool.map(lambda _index: open_capture(), range(args.copies)))
    open_seconds = time.monotonic() - open_started
    opened_captures = [capture for capture in captures if capture is not None]
    if len(opened_captures) != args.copies:
        for capture in opened_captures:
            capture.release()
        print(
            json.dumps(
                {
                    "camera_id": args.camera_id,
                    "opened": False,
                    "copies_requested": args.copies,
                    "copies_opened": len(opened_captures),
                    "open_seconds": round(open_seconds, 3),
                },
                sort_keys=True,
            )
        )
        return 2

    started_cpu = time.process_time()
    started = time.monotonic()
    deadline = time.monotonic() + max(1.0, args.duration)

    def read_capture(capture):
        read_started = time.monotonic()
        frames = 0
        width = 0
        height = 0
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames += 1
            height, width = frame.shape[:2]
        return frames, width, height, time.monotonic() - read_started

    try:
        with ThreadPoolExecutor(max_workers=args.copies) as pool:
            readings = list(pool.map(read_capture, opened_captures))
    finally:
        for capture in opened_captures:
            capture.release()
    elapsed = time.monotonic() - started
    cpu_seconds = time.process_time() - started_cpu
    summary = _summarize_readings(readings)
    frame_counts = summary["frame_counts"]
    read_seconds = summary["read_seconds"]
    delivered_rates = summary["delivered_rates"]
    dimensions = summary["dimensions"]
    result = {
        "camera_id": args.camera_id,
        "opened": True,
        "copies_requested": args.copies,
        "copies_opened": len(opened_captures),
        "drop_interval": args.drop_interval,
        "drop_applied_count": sum(
            bool(capture._decoder_drop_applied) for capture in opened_captures
        ),
        "frames": sum(frame_counts),
        "open_seconds": round(open_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "delivered_fps": round(statistics.median(delivered_rates), 3),
        "minimum_delivered_fps": round(min(delivered_rates), 3),
        "maximum_delivered_fps": round(max(delivered_rates), 3),
        "aggregate_delivered_fps": round(sum(delivered_rates), 3),
        "minimum_frames": min(frame_counts),
        "maximum_frames": max(frame_counts),
        "read_seconds": {
            "minimum": round(min(read_seconds), 3),
            "median": round(statistics.median(read_seconds), 3),
            "maximum": round(max(read_seconds), 3),
        },
        "process_cpu_seconds": round(cpu_seconds, 3),
        "process_cpu_percent": round(100.0 * cpu_seconds / elapsed, 2),
        "dimensions": [
            {"width": width, "height": height} for width, height in dimensions
        ],
    }
    result["drop_applied"] = result["drop_applied_count"] == args.copies
    if len(dimensions) == 1:
        result["width"], result["height"] = dimensions[0]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
