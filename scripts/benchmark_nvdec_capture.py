#!/usr/bin/env python3
"""Benchmark the live NVDEC capture path without printing stream secrets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from camera_connection import build_rtsp_url  # noqa: E402
from config_manager import get_config  # noqa: E402
from gstreamer_capture import GStreamerCapture  # noqa: E402


def _camera(camera_id: str) -> dict:
    cameras = get_config().get("cameras", {})
    if not isinstance(cameras, dict) or camera_id not in cameras:
        raise SystemExit(f"Unknown camera id: {camera_id}")
    camera = cameras[camera_id]
    if camera.get("stream_type") != "rtsp":
        raise SystemExit(f"Camera is not RTSP: {camera_id}")
    return camera


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_id")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--drop-interval", type=int, default=0)
    parser.add_argument("--max-dimension", type=int, default=960)
    args = parser.parse_args()

    camera = _camera(args.camera_id)
    source = build_rtsp_url(camera, include_credentials=True)
    open_started = time.monotonic()
    capture = GStreamerCapture(
        source,
        open_timeout_ms=5_000,
        read_timeout_ms=5_000,
        max_dimension=args.max_dimension,
        max_fps=args.max_fps,
        decoder_drop_interval=args.drop_interval,
    )
    if not capture.isOpened():
        print(json.dumps({"camera_id": args.camera_id, "opened": False}))
        return 2

    open_seconds = time.monotonic() - open_started
    started_cpu = time.process_time()
    started = time.monotonic()
    frames = 0
    width = 0
    height = 0
    deadline = time.monotonic() + max(1.0, args.duration)
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames += 1
            height, width = frame.shape[:2]
    finally:
        capture.release()
    elapsed = time.monotonic() - started
    cpu_seconds = time.process_time() - started_cpu
    result = {
        "camera_id": args.camera_id,
        "opened": True,
        "drop_interval": args.drop_interval,
        "drop_applied": bool(capture._decoder_drop_applied),
        "frames": frames,
        "open_seconds": round(open_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "delivered_fps": round(frames / elapsed, 3),
        "process_cpu_seconds": round(cpu_seconds, 3),
        "process_cpu_percent": round(100.0 * cpu_seconds / elapsed, 2),
        "width": width,
        "height": height,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
