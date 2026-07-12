#!/usr/bin/env python3
"""Collect representative RTSP frames for TensorRT INT8 calibration."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from camera_connection import build_rtsp_url  # noqa: E402
from config_manager import get_config  # noqa: E402
from gstreamer_capture import GStreamerCapture  # noqa: E402


def _collect_camera(
    camera_id: str,
    camera: dict,
    output_dir: Path,
    *,
    frames: int,
    fps: float,
    maximum_dimension: int,
    report: dict,
) -> None:
    source = build_rtsp_url(camera, include_credentials=True)
    capture = GStreamerCapture(
        source,
        open_timeout_ms=5_000,
        read_timeout_ms=5_000,
        max_dimension=maximum_dimension,
        max_fps=fps,
    )
    written = 0
    failures = 0
    started = time.monotonic()
    try:
        if not capture.isOpened():
            report[camera_id] = {"opened": False, "frames": 0, "failures": 1}
            return
        while written < frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 3:
                    break
                continue
            target = output_dir / f"{camera_id}-{written:04d}.jpg"
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                failures += 1
                continue
            written += 1
    finally:
        capture.release()
    report[camera_id] = {
        "opened": True,
        "frames": written,
        "failures": failures,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", dest="camera_ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-camera", type=int, default=128)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-dimension", type=int, default=960)
    args = parser.parse_args()
    if args.frames_per_camera < 1 or args.fps <= 0:
        parser.error("frames-per-camera and fps must be positive")

    cameras = get_config().get("cameras") or {}
    selected = {}
    for camera_id in args.camera_ids:
        camera = cameras.get(camera_id)
        if not isinstance(camera, dict) or camera.get("stream_type") != "rtsp":
            parser.error(f"camera is unavailable or not RTSP: {camera_id}")
        selected[camera_id] = camera

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    threads = [
        threading.Thread(
            target=_collect_camera,
            args=(camera_id, camera, args.output),
            kwargs={
                "frames": args.frames_per_camera,
                "fps": args.fps,
                "maximum_dimension": args.max_dimension,
                "report": report,
            },
            daemon=True,
        )
        for camera_id, camera in selected.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=args.frames_per_camera / args.fps + 30.0)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("calibration capture did not stop")

    print(json.dumps({"cameras": report, "total_frames": sum(item["frames"] for item in report.values())}, sort_keys=True))
    return 0 if all(item.get("opened") and item.get("frames") == args.frames_per_camera for item in report.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
