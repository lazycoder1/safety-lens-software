#!/usr/bin/env python3
"""Run a real model-server ANPR smoke test and save evidence artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "test-videos" / "anpr-e2e"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "logs" / "anpr-e2e"
MODEL_PATH = PROJECT_ROOT / "models" / "plate_recognition" / "plate-detector.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rakshak Lens ANPR model-server E2E smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100", help="Model-server base URL")
    parser.add_argument("--token", default="", help="Optional model-server bearer token")
    parser.add_argument("--conf", type=float, default=0.20, help="Detector confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="cpu", help="Inference device")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing ANPR detector: {MODEL_PATH}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = _ensure_fixtures()

    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    results: dict[str, Any] = {}
    for label, path in fixtures.items():
        payload = {
            "frame_jpeg_b64": _jpeg_b64(path),
            "conf": args.conf,
            "device": args.device,
            "imgsz": args.imgsz,
        }
        response = requests.post(f"{args.base_url.rstrip('/')}/api/anpr", json=payload, headers=headers, timeout=120)
        result = {
            "status_code": response.status_code,
            "source": str(path),
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
        }
        results[label] = result
        if response.ok:
            _save_crops(label, path, result["response"].get("plates", []))

    report_path = OUTPUT_DIR / "model_server_smoke.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(report_path)
    print(json.dumps(results, indent=2))
    return 0 if _passed(results) else 1


def _ensure_fixtures() -> dict[str, Path]:
    clear = FIXTURE_DIR / "synthetic_gate_clear.jpg"
    far = FIXTURE_DIR / "synthetic_gate_far.jpg"
    blank = FIXTURE_DIR / "blank_frame.jpg"
    video = PROJECT_ROOT / "test-videos" / "demo-pack" / "02-anpr-synthetic-gate-ka05mn4523.mp4"

    if not clear.exists():
        _draw_scene(clear, scale=1.0)
    if not far.exists():
        _draw_scene(far, scale=0.46, offset_y=-45, blur=3)
    if not blank.exists():
        cv2.imwrite(str(blank), np.full((720, 1280, 3), (30, 30, 30), dtype=np.uint8))
    if not video.exists():
        _write_video(video)
    return {"clear": clear, "far": far, "blank": blank}


def _draw_scene(path: Path, *, scale: float, offset_y: int = 0, blur: int = 0) -> np.ndarray:
    img = np.full((720, 1280, 3), (58, 66, 72), dtype=np.uint8)
    cv2.rectangle(img, (0, 420), (1280, 720), (74, 80, 82), -1)
    cv2.rectangle(img, (0, 0), (1280, 180), (185, 195, 200), -1)
    cv2.rectangle(img, (180, 210), (1100, 650), (32, 36, 40), -1)
    cv2.rectangle(img, (220, 250), (1060, 610), (80, 86, 90), -1)
    car_w, car_h = int(560 * scale), int(250 * scale)
    cx, cy = 640, 475 + offset_y
    x1, y1 = int(cx - car_w / 2), int(cy - car_h / 2)
    x2, y2 = int(cx + car_w / 2), int(cy + car_h / 2)
    cv2.rectangle(img, (x1, y1 + 55), (x2, y2), (25, 25, 28), -1)
    cv2.rectangle(img, (x1 + 80, y1), (x2 - 80, y1 + 110), (35, 45, 58), -1)
    cv2.circle(img, (x1 + 105, y2), max(8, int(42 * scale)), (10, 10, 10), -1)
    cv2.circle(img, (x2 - 105, y2), max(8, int(42 * scale)), (10, 10, 10), -1)

    pw, ph = int(250 * scale), int(58 * scale)
    px1, py1 = int(cx - pw / 2), int(y2 - ph - 35 * scale)
    px2, py2 = px1 + pw, py1 + ph
    cv2.rectangle(img, (px1, py1), (px2, py2), (245, 245, 235), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (20, 20, 20), max(1, int(2 * scale)))
    font_scale = max(0.42, 1.28 * scale)
    thickness = max(1, int(3 * scale))
    text = "KA05MN4523"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(
        img,
        text,
        (int(cx - tw / 2), int(py1 + ph / 2 + th / 2 - 4 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (8, 8, 8),
        thickness,
        cv2.LINE_AA,
    )
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    cv2.imwrite(str(path), img)
    return img


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (1280, 720))
    for i in range(48):
        scale = 0.55 + min(i, 24) / 24 * 0.45
        frame = _draw_scene(FIXTURE_DIR / "_tmp.jpg", scale=scale, offset_y=int((24 - min(i, 24)) * -5))
        writer.write(frame)
    writer.release()
    (FIXTURE_DIR / "_tmp.jpg").unlink(missing_ok=True)


def _jpeg_b64(path: Path) -> str:
    frame = cv2.imread(str(path))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _save_crops(label: str, frame_path: Path, plates: list[dict[str, Any]]) -> None:
    frame = cv2.imread(str(frame_path))
    for index, plate in enumerate(plates):
        bbox = plate.get("bbox") or {}
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.imwrite(str(OUTPUT_DIR / f"{label}_crop_{index}.jpg"), frame[y1:y2, x1:x2])


def _passed(results: dict[str, Any]) -> bool:
    clear = results.get("clear", {})
    blank = results.get("blank", {})
    return (
        clear.get("status_code") == 200
        and bool(clear.get("response", {}).get("plates"))
        and blank.get("status_code") == 200
        and not blank.get("response", {}).get("plates")
    )


if __name__ == "__main__":
    raise SystemExit(main())
