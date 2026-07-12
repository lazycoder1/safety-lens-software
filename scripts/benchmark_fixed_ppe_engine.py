#!/usr/bin/env python3
"""Compare fixed-prompt PPE engines on a directory of representative images."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised on the Jetson runtime path.
    cv2 = None
    torch = None
    YOLO = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None:
        return []
    names = result.names
    detections = []
    for box in boxes:
        class_index = int(box.cls.item())
        detections.append(
            {
                "class": str(names[class_index]),
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            }
        )
    return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if cv2 is None or torch is None or YOLO is None:
        raise RuntimeError("This benchmark requires cv2, torch, and ultralytics")
    paths = image_paths(args.images_dir)
    if not paths:
        raise RuntimeError(f"No benchmark images found in {args.images_dir}")

    device = 0 if torch.cuda.is_available() else "cpu"
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "images": [path.name for path in paths],
        "models": [],
    }
    for model_path in args.models:
        model = YOLO(str(model_path))
        sample = cv2.imread(str(paths[0]))
        if sample is None:
            raise RuntimeError(f"Could not read benchmark image: {paths[0]}")
        for _ in range(args.warmup):
            model.predict(
                sample,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )

        latencies: list[float] = []
        by_image = []
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not read benchmark image: {path}")
            last_result = None
            image_latencies = []
            for _ in range(args.reps):
                started = time.perf_counter()
                last_result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=device,
                    verbose=False,
                )[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies.append(elapsed_ms)
                image_latencies.append(elapsed_ms)
            by_image.append(
                {
                    "image": path.name,
                    "mean_ms": round(statistics.mean(image_latencies), 3),
                    "detections": detections_from_result(last_result),
                }
            )

        report["models"].append(
            {
                "model": str(model_path),
                "mean_ms": round(statistics.mean(latencies), 3),
                "median_ms": round(statistics.median(latencies), 3),
                "p95_ms": round(percentile(latencies, 0.95), 3),
                "samples": len(latencies),
                "by_image": by_image,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
