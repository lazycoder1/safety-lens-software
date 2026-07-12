#!/usr/bin/env python3
"""Compare sequential and concurrent primary/PPE fixed-batch execution."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import cv2

import model_manager


PPE_CLASSES = ["motorcycle helmet", "rider helmet", "helmet"]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def _canonical(records: Any) -> str:
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def _run_primary(frames: list[Any], conf: float) -> list[list[dict[str, Any]]]:
    records = model_manager.predict_coco_record_batch(
        frames,
        conf=conf,
        device="cuda",
        imgsz=640,
    )
    if records is None:
        raise RuntimeError("Fixed-batch primary runtime is unavailable")
    return records


def _run_ppe(frames: list[Any], conf: float) -> list[list[dict[str, Any]]]:
    records = model_manager.predict_ppe_record_batch(
        frames,
        conf=conf,
        device="cuda",
        imgsz=640,
        classes=PPE_CLASSES,
    )
    if records is None:
        raise RuntimeError("Fixed-batch PPE runtime is unavailable")
    return records


def _sequential(
    frames: list[Any],
    primary_conf: float,
    ppe_conf: float,
) -> tuple[Any, Any]:
    return _run_primary(frames, primary_conf), _run_ppe(frames, ppe_conf)


def _concurrent(
    executor: ThreadPoolExecutor,
    frames: list[Any],
    primary_conf: float,
    ppe_conf: float,
) -> tuple[Any, Any]:
    primary = executor.submit(_run_primary, frames, primary_conf)
    ppe = executor.submit(_run_ppe, frames, ppe_conf)
    return primary.result(), ppe.result()


def _time_call(call: Callable[[], tuple[Any, Any]]) -> tuple[float, tuple[Any, Any]]:
    started = time.perf_counter()
    result = call()
    return (time.perf_counter() - started) * 1000.0, result


def _load_frames(frames_dir: Path) -> list[Any]:
    paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("At least one readable validation frame is required")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--primary-conf", type=float, default=0.35)
    parser.add_argument("--ppe-conf", type=float, default=0.20)
    args = parser.parse_args()

    frames = _load_frames(args.frames_dir)
    groups = [
        [frames[(index + offset) % len(frames)] for offset in range(args.batch_size)]
        for index in range(0, len(frames), args.batch_size)
    ]
    model_manager.initialize()

    sequential_latencies: list[float] = []
    concurrent_latencies: list[float] = []
    mismatches = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(args.warmups):
            group = groups[index % len(groups)]
            _sequential(group, args.primary_conf, args.ppe_conf)
            _concurrent(executor, group, args.primary_conf, args.ppe_conf)

        for index in range(args.repetitions):
            group = groups[index % len(groups)]
            # Alternate order to keep thermal and cache effects balanced.
            if index % 2:
                concurrent_ms, concurrent_result = _time_call(
                    lambda: _concurrent(
                        executor,
                        group,
                        args.primary_conf,
                        args.ppe_conf,
                    )
                )
                sequential_ms, sequential_result = _time_call(
                    lambda: _sequential(group, args.primary_conf, args.ppe_conf)
                )
            else:
                sequential_ms, sequential_result = _time_call(
                    lambda: _sequential(group, args.primary_conf, args.ppe_conf)
                )
                concurrent_ms, concurrent_result = _time_call(
                    lambda: _concurrent(
                        executor,
                        group,
                        args.primary_conf,
                        args.ppe_conf,
                    )
                )
            sequential_latencies.append(sequential_ms)
            concurrent_latencies.append(concurrent_ms)
            if _canonical(sequential_result) != _canonical(concurrent_result):
                mismatches += 1

    sequential_summary = _summary(sequential_latencies)
    concurrent_summary = _summary(concurrent_latencies)
    report = {
        "frames": len(frames),
        "batch_size": args.batch_size,
        "frame_groups": len(groups),
        "warmups_per_mode": args.warmups,
        "repetitions_per_mode": args.repetitions,
        "primary_conf": args.primary_conf,
        "ppe_conf": args.ppe_conf,
        "ppe_classes": PPE_CLASSES,
        "sequential": sequential_summary,
        "concurrent": concurrent_summary,
        "median_speedup_percent": round(
            (sequential_summary["median_ms"] / concurrent_summary["median_ms"] - 1.0)
            * 100.0,
            3,
        ),
        "result_mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
