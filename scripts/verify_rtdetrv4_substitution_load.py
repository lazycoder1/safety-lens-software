#!/usr/bin/env python3
"""Verify that RT-DETR work exactly replaces scheduled primary frame slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(
    edge: dict[str, Any],
    specialist: dict[str, Any],
    *,
    maximum_primary_latency_ms: float,
) -> dict[str, Any]:
    errors: list[str] = []
    cameras = int(edge.get("cameras") or 0)
    target_fps = float(edge.get("target_fps") or 0.0)
    duration = float(edge.get("duration_seconds") or 0.0)
    requests = int(edge.get("requests") or 0)
    substitutions = int(edge.get("substituted_requests") or 0)
    effective_requests = int(edge.get("effective_requests") or 0)
    expected_effective = round(cameras * target_fps * duration)
    frames_completed = int(
        specialist.get("frames_completed")
        or int(specialist.get("repeats") or 0) * int(specialist.get("batch_size") or 0)
    )
    specialist_duration = float(
        specialist.get("duration_seconds")
        or specialist.get("duration_target_seconds")
        or 0.0
    )
    specialist_target = float(specialist.get("target_fps") or 0.0)
    specialist_achieved = float(
        specialist.get("achieved_fps") or specialist.get("frame_fps") or 0.0
    )
    per_camera = edge.get("per_camera") or []

    if cameras < 1 or target_fps <= 0 or duration <= 0:
        errors.append("edge workload dimensions are invalid")
    if effective_requests != expected_effective:
        errors.append(
            f"effective requests {effective_requests} != scheduled {expected_effective}"
        )
    if requests + substitutions != effective_requests:
        errors.append(
            "primary requests plus substitutions do not equal effective requests"
        )
    if substitutions < 1:
        errors.append("workload contains no substituted primary frames")
    if specialist_duration != duration:
        errors.append("edge and RT-DETR durations do not match")
    if specialist_target <= 0:
        errors.append("RT-DETR target FPS is invalid")
    elif frames_completed != round(specialist_target * specialist_duration):
        errors.append("RT-DETR frame count does not match its target and duration")
    if frames_completed != substitutions:
        errors.append(
            f"RT-DETR frames {frames_completed} != substituted frames {substitutions}"
        )
    if int(edge.get("overloads") or 0) or int(edge.get("failures") or 0):
        errors.append("edge workload contains overloads or failures")
    if float(edge.get("minimum_effective_camera_fps") or 0.0) < target_fps:
        errors.append("at least one camera missed its effective FPS target")
    if len(per_camera) != cameras:
        errors.append("per-camera report count does not match camera count")
    else:
        if sum(int(item.get("successes") or 0) for item in per_camera) != requests:
            errors.append("per-camera successes do not equal primary request count")
        if (
            sum(int(item.get("substituted_requests") or 0) for item in per_camera)
            != substitutions
        ):
            errors.append("per-camera substitutions do not equal substitution count")
        if any(
            float(item.get("effective_fps") or 0.0) < target_fps for item in per_camera
        ):
            errors.append("per-camera report contains an effective FPS miss")
    reported_ppe = int(edge.get("specialist_requests") or 0)
    if reported_ppe != sum(
        int(item.get("specialist_requests") or 0) for item in per_camera
    ):
        errors.append("per-camera PPE requests do not equal the reported total")
    if specialist_achieved < specialist_target:
        errors.append("RT-DETR missed its target FPS")
    if int(specialist.get("stale_groups") or 0):
        errors.append("RT-DETR contains stale groups")
    latency = edge.get("latency_ms") or {}
    maximum_latency = float(latency.get("maximum") or 0.0)
    if maximum_latency > maximum_primary_latency_ms:
        errors.append(
            f"primary maximum latency {maximum_latency} ms exceeds "
            f"{maximum_primary_latency_ms} ms"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "cameras": cameras,
        "target_camera_fps": target_fps,
        "effective_requests": effective_requests,
        "substituted_requests": substitutions,
        "ppe_requests": reported_ppe,
        "rtdetr_achieved_fps": specialist_achieved,
        "primary_latency_ms": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--maximum-primary-latency-ms", type=float, default=250.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.maximum_primary_latency_ms <= 0:
        parser.error("maximum-primary-latency-ms must be positive")
    edge = json.loads(args.edge.read_text(encoding="utf-8"))
    specialist = json.loads(args.specialist.read_text(encoding="utf-8"))
    result = verify(
        edge,
        specialist,
        maximum_primary_latency_ms=args.maximum_primary_latency_ms,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
