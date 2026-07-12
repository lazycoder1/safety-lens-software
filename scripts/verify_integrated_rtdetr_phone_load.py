#!/usr/bin/env python3
"""Verify an integrated edge-scheduled RT-DETR substitution load report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(
    report: dict[str, Any],
    *,
    maximum_primary_latency_ms: float,
    maximum_rtdetr_latency_ms: float,
) -> dict[str, Any]:
    errors: list[str] = []
    cameras = int(report.get("cameras") or 0)
    target_fps = float(report.get("target_fps") or 0.0)
    duration = float(report.get("duration_seconds") or 0.0)
    requests = int(report.get("requests") or 0)
    substitutions = int(report.get("substituted_requests") or 0)
    attempts = int(report.get("substitution_attempts") or 0)
    effective_requests = int(report.get("effective_requests") or 0)
    expected_effective = round(cameras * target_fps * duration)
    per_camera = report.get("per_camera") or []
    rtdetr_batch = report.get("edge_rtdetr_phone_batch") or {}

    if report.get("substitution_source") != "rtdetr":
        errors.append("substitution source is not the integrated RT-DETR route")
    if cameras < 1 or target_fps <= 0 or duration <= 0:
        errors.append("workload dimensions are invalid")
    if effective_requests != expected_effective:
        errors.append(
            f"effective requests {effective_requests} != scheduled {expected_effective}"
        )
    if requests + substitutions != effective_requests:
        errors.append("primary requests plus substitutions do not equal effective requests")
    if attempts < 1 or attempts != substitutions:
        errors.append("not every RT-DETR substitution attempt completed")
    if int(report.get("overloads") or 0) or int(report.get("failures") or 0):
        errors.append("workload contains overloads or failures")
    if float(report.get("minimum_effective_camera_fps") or 0.0) < target_fps:
        errors.append("at least one camera missed its effective FPS target")
    if len(per_camera) != cameras:
        errors.append("per-camera report count does not match camera count")
    if int(rtdetr_batch.get("submitted_frames") or 0) != substitutions:
        errors.append("RT-DETR transport submissions do not match substitutions")
    if int(rtdetr_batch.get("batch2_frames") or 0) != substitutions:
        errors.append("not every RT-DETR frame used the batch-2 route")
    if int(rtdetr_batch.get("batch1_frames") or 0):
        errors.append("integrated load unexpectedly used RT-DETR batch-1")
    if int(rtdetr_batch.get("route_failures") or 0):
        errors.append("RT-DETR transport reported route failures")
    if int(rtdetr_batch.get("admission_overloads") or 0):
        errors.append("RT-DETR transport reported admission overloads")
    primary_maximum = float((report.get("latency_ms") or {}).get("maximum") or 0.0)
    if primary_maximum > maximum_primary_latency_ms:
        errors.append(
            f"primary maximum latency {primary_maximum} ms exceeds "
            f"{maximum_primary_latency_ms} ms"
        )
    rtdetr_maximum = float(
        (report.get("rtdetr_latency_ms") or {}).get("maximum") or 0.0
    )
    if rtdetr_maximum <= 0 or rtdetr_maximum > maximum_rtdetr_latency_ms:
        errors.append(
            f"RT-DETR maximum latency {rtdetr_maximum} ms exceeds valid bound "
            f"{maximum_rtdetr_latency_ms} ms"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "cameras": cameras,
        "target_camera_fps": target_fps,
        "effective_requests": effective_requests,
        "substituted_requests": substitutions,
        "ppe_requests": int(report.get("specialist_requests") or 0),
        "primary_latency_ms": report.get("latency_ms"),
        "rtdetr_latency_ms": report.get("rtdetr_latency_ms"),
        "rtdetr_batch": rtdetr_batch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-primary-latency-ms", type=float, default=250.0)
    parser.add_argument("--maximum-rtdetr-latency-ms", type=float, default=250.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.maximum_primary_latency_ms <= 0 or args.maximum_rtdetr_latency_ms <= 0:
        parser.error("latency bounds must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify(
        report,
        maximum_primary_latency_ms=args.maximum_primary_latency_ms,
        maximum_rtdetr_latency_ms=args.maximum_rtdetr_latency_ms,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
