#!/usr/bin/env python3
"""Compare default YOLOE segmentation postprocessing with boxes-only output."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _record(result) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    rows = boxes.data.detach().cpu().tolist()
    return [
        {
            "bbox": [round(float(value), 6) for value in row[:4]],
            "confidence": round(float(row[4]), 8),
            "class_id": int(row[5]),
        }
        for row in rows
    ]


def _run(
    *,
    model_path: Path,
    frames: list[Any],
    batch_size: int,
    warmups: int,
    iterations: int,
    conf: float,
    imgsz: int,
    device: str,
    boxes_only: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops

    class BoxesOnlySegmentationPredictor(SegmentationPredictor):
        """Retain segmentation-engine boxes without materializing unused masks."""

        def construct_result(self, pred, img, orig_img, img_path, proto):
            del proto
            if pred.shape[0]:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:],
                    pred[:, :4],
                    orig_img.shape,
                )
            return Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                boxes=pred[:, :6],
                masks=None,
            )

    model = YOLO(str(model_path), task="segment")
    predictor = BoxesOnlySegmentationPredictor if boxes_only else None
    group_cursor = 0

    def predict_group():
        nonlocal group_cursor
        group = [
            frames[(group_cursor + offset) % len(frames)]
            for offset in range(batch_size)
        ]
        group_cursor = (group_cursor + batch_size) % len(frames)
        return model.predict(
            group,
            conf=conf,
            verbose=False,
            device=device,
            imgsz=imgsz,
            predictor=predictor,
        )

    for _ in range(warmups):
        predict_group()

    durations_ms = []
    reported_speeds: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
    }
    mask_results = 0
    detection_count = 0
    for _ in range(iterations):
        started = time.perf_counter()
        results = predict_group()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        for result in results:
            mask_results += int(result.masks is not None)
            detection_count += len(result.boxes) if result.boxes is not None else 0
            for key in reported_speeds:
                value = result.speed.get(key)
                if value is not None:
                    reported_speeds[key].append(float(value))

    evaluation = []
    for index in range(0, len(frames), batch_size):
        group = [frames[(index + offset) % len(frames)] for offset in range(batch_size)]
        results = model.predict(
            group,
            conf=conf,
            verbose=False,
            device=device,
            imgsz=imgsz,
            predictor=predictor,
        )
        for offset, result in enumerate(results):
            frame_index = (index + offset) % len(frames)
            evaluation.append(
                {
                    "frame_index": frame_index,
                    "detections": _record(result),
                    "mask_count": len(result.masks) if result.masks is not None else 0,
                }
            )

    total_frames = iterations * batch_size
    total_seconds = sum(durations_ms) / 1000.0
    payload = {
        "mode": "boxes_only" if boxes_only else "default_segmentation",
        "batch_size": batch_size,
        "warmups": warmups,
        "iterations": iterations,
        "frames": total_frames,
        "throughput_fps": total_frames / total_seconds,
        "group_latency_ms": {
            "mean": statistics.fmean(durations_ms),
            "median": statistics.median(durations_ms),
            "p95": _percentile(durations_ms, 0.95),
            "max": max(durations_ms),
        },
        "reported_speed_ms_per_frame": {
            key: statistics.fmean(values) if values else None
            for key, values in reported_speeds.items()
        },
        "timed_detection_count": detection_count,
        "timed_results_with_masks": mask_results,
        "evaluation": sorted(evaluation, key=lambda item: item["frame_index"]),
    }
    model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return payload


def _parity(default: dict[str, Any], boxes_only: dict[str, Any]) -> dict[str, Any]:
    default_by_frame = {
        item["frame_index"]: item["detections"] for item in default["evaluation"]
    }
    candidate_by_frame = {
        item["frame_index"]: item["detections"] for item in boxes_only["evaluation"]
    }
    exact = default_by_frame == candidate_by_frame
    mismatches = [
        frame_index
        for frame_index in sorted(set(default_by_frame) | set(candidate_by_frame))
        if default_by_frame.get(frame_index) != candidate_by_frame.get(frame_index)
    ]
    return {
        "exact": exact,
        "mismatch_frame_indexes": mismatches,
        "default_detection_count": sum(
            len(value) for value in default_by_frame.values()
        ),
        "boxes_only_detection_count": sum(
            len(value) for value in candidate_by_frame.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--order",
        choices=("default-first", "boxes-first"),
        default="default-first",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1:
        parser.error("--warmups must be non-negative and --iterations must be positive")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")

    frames = []
    for path in args.frames:
        frame = cv2.imread(str(path))
        if frame is None:
            parser.error(f"frame is unreadable: {path}")
        frames.append(frame)

    run_arguments = {
        "model_path": args.model,
        "frames": frames,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    if args.order == "boxes-first":
        boxes_only = _run(**run_arguments, boxes_only=True)
        default = _run(**run_arguments, boxes_only=False)
    else:
        default = _run(**run_arguments, boxes_only=False)
        boxes_only = _run(**run_arguments, boxes_only=True)
    payload = {
        "model": str(args.model),
        "input_frame_count": len(frames),
        "order": args.order,
        "default": default,
        "boxes_only": boxes_only,
        "parity": _parity(default, boxes_only),
        "throughput_gain_percent": (
            boxes_only["throughput_fps"] / default["throughput_fps"] - 1.0
        )
        * 100.0,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if payload["parity"]["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
