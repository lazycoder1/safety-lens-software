#!/usr/bin/env python3
"""Measure edge-side adaptive batch-2/batch-4 routing against a model server."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

import model_manager


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _requests(mode: str) -> list[dict[str, Any]]:
    primary = {
        "model_key": "coco_primary",
        "conf": 0.35,
        "device": "cuda",
        "imgsz": 640,
        "classes": [],
    }
    if mode == "primary":
        return [primary]
    return [
        primary,
        {
            "model_key": "ppe_specialist",
            "conf": 0.2,
            "device": "cuda",
            "imgsz": 640,
            "classes": ["motorcycle helmet", "rider helmet", "helmet"],
        },
    ]


def _run_scenario(
    *,
    name: str,
    mode: str,
    group_size: int,
    wait_seconds: float,
    early_flush_seconds: float,
    frame_batch_size_hint: int | None,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    primary_batcher = model_manager._RemotePrimaryFrameBatcher(
        wait_seconds,
        batch_size=4,
        batch2_early_flush_seconds=early_flush_seconds,
    )
    specialist_batcher = model_manager._RemoteSpecialistFrameBatcher(
        wait_seconds,
        batch_size=4,
        batch2_early_flush_seconds=early_flush_seconds,
    )
    model_manager._REMOTE_PRIMARY_FRAME_BATCHER = primary_batcher
    model_manager._REMOTE_SPECIALIST_FRAME_BATCHER = specialist_batcher
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    requests = _requests(mode)

    def run_once(pool: ThreadPoolExecutor) -> float:
        barrier = threading.Barrier(group_size + 1)

        def infer() -> dict[str, list[dict[str, Any]]]:
            barrier.wait()
            options = (
                {"frame_batch_size_hint": frame_batch_size_hint}
                if frame_batch_size_hint is not None
                else {}
            )
            return model_manager.predict_record_batches(frame, requests, **options)

        futures = [pool.submit(infer) for _ in range(group_size)]
        barrier.wait()
        started = time.perf_counter()
        for future in futures:
            future.result()
        return time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=group_size) as pool:
        for _ in range(warmups):
            run_once(pool)
        durations = [run_once(pool) for _ in range(iterations)]

    milliseconds = [duration * 1000 for duration in durations]
    stats = (
        specialist_batcher.stats() if mode == "specialist" else primary_batcher.stats()
    )
    return {
        "name": name,
        "mode": mode,
        "group_size": group_size,
        "frame_batch_size_hint": frame_batch_size_hint,
        "warmups": warmups,
        "iterations": iterations,
        "median_ms": round(statistics.median(milliseconds), 3),
        "p95_ms": round(_percentile(milliseconds, 0.95), 3),
        "max_ms": round(max(milliseconds), 3),
        "aggregate_fps_at_saturation": round(
            group_size / statistics.mean(durations), 3
        ),
        "batch_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--wait-seconds", type=float, default=0.010)
    parser.add_argument("--early-flush-seconds", type=float, default=0.006)
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only the named scenario; repeat to select more than one.",
    )
    args = parser.parse_args()

    if not model_manager.is_remote_inference_enabled():
        raise SystemExit("Remote inference must be enabled")
    scenarios = (
        ("one_primary_static_batch4", "primary", 1, 0.0, None),
        (
            "one_primary_singleton_bypass",
            "primary",
            1,
            args.early_flush_seconds,
            1,
        ),
        ("one_specialist_static_batch4", "specialist", 1, 0.0, None),
        (
            "one_specialist_singleton_bypass",
            "specialist",
            1,
            args.early_flush_seconds,
            1,
        ),
        ("two_primary_static_batch4", "primary", 2, 0.0, None),
        ("four_primary_static_batch4", "primary", 4, 0.0, None),
        (
            "two_primary_adaptive_batch2",
            "primary",
            2,
            args.early_flush_seconds,
            None,
        ),
        (
            "four_primary_adaptive_batch4",
            "primary",
            4,
            args.early_flush_seconds,
            None,
        ),
        (
            "two_specialist_adaptive_batch2",
            "specialist",
            2,
            args.early_flush_seconds,
            None,
        ),
    )
    selected = set(args.scenario or ())
    known = {
        name for name, _mode, _group_size, _early_flush, _batch_size_hint in scenarios
    }
    unknown = selected - known
    if unknown:
        parser.error(f"unknown scenario: {', '.join(sorted(unknown))}")
    results = [
        _run_scenario(
            name=name,
            mode=mode,
            group_size=group_size,
            wait_seconds=args.wait_seconds,
            early_flush_seconds=early_flush_seconds,
            frame_batch_size_hint=frame_batch_size_hint,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        for name, mode, group_size, early_flush_seconds, frame_batch_size_hint in scenarios
        if not selected or name in selected
    ]
    print(json.dumps({"scenarios": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
