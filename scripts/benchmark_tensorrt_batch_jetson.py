#!/usr/bin/env python3
"""Benchmark a fixed-batch TensorRT detector on grouped camera frames."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _predict_group(model, frames: list[Any], *, engine_batch: int, conf: float, imgsz: int):
    if engine_batch == 1:
        return [
            model.predict(
                frame,
                conf=conf,
                imgsz=imgsz,
                device=0,
                verbose=False,
            )[0]
            for frame in frames
        ]
    results = []
    for offset in range(0, len(frames), engine_batch):
        batch = frames[offset : offset + engine_batch]
        if len(batch) != engine_batch:
            raise RuntimeError("Frame group must be divisible by the engine batch")
        results.extend(
            model.predict(
                batch,
                conf=conf,
                imgsz=imgsz,
                device=0,
                verbose=False,
            )
        )
    if len(results) != len(frames):
        raise RuntimeError("TensorRT batch returned the wrong result count")
    return results


def _result_summary(result) -> dict[str, Any]:
    boxes = result.boxes
    if boxes is None:
        return {"detections": []}
    return {
        "detections": [
            {
                "class_id": int(class_id),
                "confidence": round(float(confidence), 4),
                "bbox": [round(float(value), 1) for value in xyxy],
            }
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
            )
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--engine-batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Frames timed per sample; defaults to engine batch, or two for batch-1",
    )
    parser.add_argument(
        "--task",
        choices=("detect", "segment"),
        default="detect",
        help="Ultralytics task embedded in the fixed TensorRT engine",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.frames) < 2:
        parser.error("at least two frames are required")
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    group_size = args.group_size or (args.engine_batch if args.engine_batch > 1 else 2)
    if group_size < args.engine_batch or group_size % args.engine_batch:
        parser.error("group-size must be a positive multiple of engine-batch")

    frames = []
    for path in args.frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {path}")
        frames.append(frame)

    from ultralytics import YOLO

    model = YOLO(str(args.model), task=args.task)
    for index in range(args.warmups):
        group = [
            frames[(index * group_size + offset) % len(frames)]
            for offset in range(group_size)
        ]
        _predict_group(
            model,
            group,
            engine_batch=args.engine_batch,
            conf=args.conf,
            imgsz=args.imgsz,
        )

    latencies_ms = []
    started = time.perf_counter()
    for index in range(args.repeats):
        group = [
            frames[(index * group_size + offset) % len(frames)]
            for offset in range(group_size)
        ]
        group_started = time.perf_counter()
        _predict_group(
            model,
            group,
            engine_batch=args.engine_batch,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        latencies_ms.append((time.perf_counter() - group_started) * 1000.0)
    duration = time.perf_counter() - started

    evaluation = []
    for index in range(0, len(frames), group_size):
        group = [
            frames[(index + offset) % len(frames)]
            for offset in range(group_size)
        ]
        results = _predict_group(
            model,
            group,
            engine_batch=args.engine_batch,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        evaluation.extend(
            {
                "frame": args.frames[(index + offset) % len(args.frames)].name,
                **_result_summary(result),
            }
            for offset, result in enumerate(results)
            if index + offset < len(frames)
        )

    report = {
        "model": args.model.name,
        "engine_batch": args.engine_batch,
        "group_size": group_size,
        "task": args.task,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "frames_processed": args.repeats * group_size,
        "duration_seconds": round(duration, 4),
        "throughput_fps": round(args.repeats * group_size / duration, 3),
        "group_latency_ms": {
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(_percentile(latencies_ms, 0.95), 3),
            "maximum": round(max(latencies_ms), 3),
        },
        "evaluation": evaluation,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
