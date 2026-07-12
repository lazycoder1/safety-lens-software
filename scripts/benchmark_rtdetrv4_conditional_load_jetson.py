#!/usr/bin/env python3
"""Replay a paced RT-DETRv4 specialist workload on a Jetson TensorRT engine."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from benchmark_rtdetrv4_tensorrt_jetson import (
    TensorRTModel,
    _percentile,
    _prepare_group,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--target-fps",
        type=float,
        required=True,
        help="Aggregate specialist frames per second across all cameras",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--start-at-monotonic",
        type=float,
        default=0.0,
        help="Optional shared host monotonic timestamp for synchronized workloads",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=0.0,
        help=("Start-lateness budget in seconds; defaults to one batch-group interval"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (
        args.target_fps <= 0
        or args.duration <= 0
        or args.warmups < 1
        or args.stale_after < 0
        or args.start_at_monotonic < 0
    ):
        parser.error(
            "target-fps, duration, and warmups must be positive; "
            "stale-after and start-at-monotonic must be non-negative"
        )

    frames = []
    for path in args.frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {path}")
        frames.append(frame)

    model = TensorRTModel(args.engine)
    groups = [
        _prepare_group(model, frames, index * model.batch_size)
        for index in range(max(args.warmups, len(frames)))
    ]
    for index in range(args.warmups):
        images, sizes = groups[index % len(groups)]
        model.run(images, sizes, copy_outputs=False)

    group_interval = model.batch_size / args.target_fps
    stale_after = args.stale_after or group_interval
    ready_at = time.monotonic()
    benchmark_start = args.start_at_monotonic or ready_at + 0.25
    if benchmark_start < ready_at + 0.05:
        raise RuntimeError("shared benchmark start is not far enough in the future")
    deadline = benchmark_start + args.duration
    group_index = 0
    latencies_ms: list[float] = []
    start_lateness_ms: list[float] = []
    stale_groups = 0
    while True:
        scheduled = benchmark_start + group_index * group_interval
        if scheduled >= deadline:
            break
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        actual_start = time.monotonic()
        lateness_ms = max(0.0, (actual_start - scheduled) * 1000.0)
        start_lateness_ms.append(lateness_ms)
        stale_groups += int(lateness_ms > stale_after * 1000.0)
        images, sizes = _prepare_group(
            model,
            frames,
            group_index * model.batch_size,
        )
        started = time.perf_counter()
        model.run(images, sizes, copy_outputs=True)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        group_index += 1

    frames_completed = len(latencies_ms) * model.batch_size
    result = {
        "engine": args.engine.name,
        "batch_size": model.batch_size,
        "duration_seconds": args.duration,
        "target_fps": args.target_fps,
        "stale_after_seconds": stale_after,
        "groups_completed": len(latencies_ms),
        "frames_completed": frames_completed,
        "achieved_fps": round(frames_completed / args.duration, 3),
        "stale_groups": stale_groups,
        "latency_ms": {
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(_percentile(latencies_ms, 0.95), 3),
            "maximum": round(max(latencies_ms), 3),
        },
        "start_lateness_ms": {
            "median": round(statistics.median(start_lateness_ms), 3),
            "p95": round(_percentile(start_lateness_ms, 0.95), 3),
            "maximum": round(max(start_lateness_ms), 3),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
