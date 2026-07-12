#!/usr/bin/env python3
"""Measure the model-server RT-DETR phone route on labelled Jetson frames."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import requests


PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bbox_coordinates(
    detection: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _phone_matches_person(phone: dict[str, Any], person: dict[str, Any]) -> bool:
    phone_bbox = _bbox_coordinates(phone)
    person_bbox = _bbox_coordinates(person)
    if phone_bbox is None or person_bbox is None:
        return False
    phone_x1, phone_y1, phone_x2, phone_y2 = phone_bbox
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1
    phone_center_x = (phone_x1 + phone_x2) / 2.0
    phone_center_y = (phone_y1 + phone_y2) / 2.0
    relative_y = (phone_center_y - person_y1) / person_height
    if not (
        person_x1 - person_width * 0.25
        <= phone_center_x
        <= person_x2 + person_width * 0.25
        and -0.10 <= relative_y <= 0.85
    ):
        return False
    phone_area = (phone_x2 - phone_x1) * (phone_y2 - phone_y1)
    person_area = person_width * person_height
    return phone_area / person_area <= 0.08


def _is_actionable(records: list[dict[str, Any]]) -> bool:
    people = [item for item in records if item.get("class_id") == PERSON_CLASS_ID]
    phones = [item for item in records if item.get("class_id") == PHONE_CLASS_ID]
    return any(
        _phone_matches_person(phone, person) for phone in phones for person in people
    )


def _load_frame(path: Path) -> tuple[Path, Any]:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"could not decode frame: {path}")
    return path, frame


def _request_group(
    session: requests.Session,
    endpoint: str,
    group: list[tuple[Path, Any]],
    *,
    person_conf: float,
    phone_conf: float,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    metadata = []
    chunks = []
    for index, (path, frame) in enumerate(group):
        request_id = f"{index}-{path.name}"
        metadata.append(
            {
                "request_id": request_id,
                "person_conf": person_conf,
                "phone_conf": phone_conf,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
                "frame_channels": int(frame.shape[2]),
                "byte_length": int(frame.nbytes),
            }
        )
        chunks.append(frame.tobytes())
    started = time.perf_counter()
    response = session.post(
        endpoint,
        data=b"".join(chunks),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-RTDETR-Phone-Batch": json.dumps(metadata),
        },
        timeout=10,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return response.json()["results"], latency_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Optional paced aggregate frame rate; derives repeats from duration.",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--start-at-monotonic", type=float, default=0.0)
    parser.add_argument(
        "--stale-after",
        type=float,
        default=0.0,
        help="Paced start-lateness budget; defaults to one group interval.",
    )
    parser.add_argument("--person-conf", type=float, default=0.30)
    parser.add_argument("--phone-conf", type=float, default=0.15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    if args.target_fps < 0 or args.duration < 0 or args.start_at_monotonic < 0:
        parser.error("pacing values must be non-negative")
    if bool(args.target_fps) != bool(args.duration):
        parser.error("target-fps and duration must be provided together")
    if args.target_fps and args.stale_after < 0:
        parser.error("stale-after must be non-negative")

    frames = [_load_frame(path) for path in args.frames]
    groups = [
        [frames[(start + offset) % len(frames)] for offset in range(args.batch_size)]
        for start in range(0, len(frames), args.batch_size)
    ]
    session = requests.Session()
    for index in range(args.warmups):
        _request_group(
            session,
            args.endpoint,
            groups[index % len(groups)],
            person_conf=args.person_conf,
            phone_conf=args.phone_conf,
        )

    labelled: dict[str, bool] = {}
    for group in groups:
        results, _latency_ms = _request_group(
            session,
            args.endpoint,
            group,
            person_conf=args.person_conf,
            phone_conf=args.phone_conf,
        )
        for index, (path, _frame) in enumerate(group):
            labelled[path.name] = _is_actionable(results[f"{index}-{path.name}"])

    repeats = args.repeats
    group_interval = 0.0
    stale_after = 0.0
    benchmark_start = 0.0
    if args.target_fps:
        group_interval = args.batch_size / args.target_fps
        stale_after = args.stale_after or group_interval
        repeats = math.floor(args.duration / group_interval)
        ready_at = time.monotonic()
        benchmark_start = args.start_at_monotonic or ready_at + 0.25
        if benchmark_start < ready_at + 0.05:
            raise RuntimeError("shared benchmark start is not far enough in the future")

    latencies_ms = []
    start_lateness_ms = []
    stale_groups = 0
    started = time.perf_counter()
    for index in range(repeats):
        if args.target_fps:
            scheduled = benchmark_start + index * group_interval
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            lateness_ms = max(0.0, (time.monotonic() - scheduled) * 1000.0)
            start_lateness_ms.append(lateness_ms)
            stale_groups += int(lateness_ms > stale_after * 1000.0)
        _results, latency_ms = _request_group(
            session,
            args.endpoint,
            groups[index % len(groups)],
            person_conf=args.person_conf,
            phone_conf=args.phone_conf,
        )
        latencies_ms.append(latency_ms)
    duration_seconds = time.perf_counter() - started
    rate_duration_seconds = args.duration if args.target_fps else duration_seconds
    positive = {
        name: result for name, result in labelled.items() if name.startswith("pos-")
    }
    negative = {
        name: result for name, result in labelled.items() if name.startswith("neg-")
    }
    report = {
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "repeats": repeats,
        "frames_completed": repeats * args.batch_size,
        "target_fps": args.target_fps or None,
        "achieved_fps": (repeats * args.batch_size / args.duration)
        if args.target_fps
        else (repeats * args.batch_size / duration_seconds),
        "duration_seconds": args.duration or duration_seconds,
        "duration_target_seconds": args.duration or None,
        "stale_after_seconds": stale_after or None,
        "stale_groups": stale_groups if args.target_fps else None,
        "person_conf": args.person_conf,
        "phone_conf": args.phone_conf,
        "labelled": labelled,
        "actionable_positive": sum(positive.values()),
        "positive_total": len(positive),
        "actionable_negative": sum(negative.values()),
        "negative_total": len(negative),
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": _percentile(latencies_ms, 0.95),
            "maximum": max(latencies_ms),
        },
        "start_lateness_ms": {
            "median": statistics.median(start_lateness_ms),
            "p95": _percentile(start_lateness_ms, 0.95),
            "maximum": max(start_lateness_ms),
        }
        if start_lateness_ms
        else None,
        "request_loop_wall_seconds": duration_seconds,
        "group_fps": repeats / rate_duration_seconds,
        "frame_fps": repeats * args.batch_size / rate_duration_seconds,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
