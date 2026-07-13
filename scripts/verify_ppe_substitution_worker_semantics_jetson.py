#!/usr/bin/env python3
"""Compare additive and cached-context PPE paths through the real Jetson worker."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

import detection
import model_manager
import video_processing
from routers.safety_rules import DEFAULT_SAFETY_RULES


PPE_CLASSES = ["motorcycle helmet", "rider helmet", "helmet"]
PPE_CAPABILITY = "rider_helmet_required"
CONTEXT_CLASSES = {"person", "motorcycle", "motorbike", "scooter"}


def _normalize(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in detections:
        normalized.append(
            {
                "class": str(item.get("class") or ""),
                "confidence": round(float(item.get("confidence") or 0.0), 6),
                "bbox": [int(value) for value in item.get("bbox") or []],
                "model_family": str(item.get("model_family") or ""),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["model_family"],
            item["class"],
            item["bbox"],
            item["confidence"],
        ),
    )


def _relevant(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in detections
        if item.get("model_family") == "ppe_specialist"
        or (
            item.get("model_family") == "coco_primary"
            and str(item.get("class") or "").lower() in CONTEXT_CLASSES
        )
    ]


def _violation_signature(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "rule": item.get("rule"),
                "severity": item.get("severity"),
                "threshold": item.get("threshold"),
                "classes": sorted(item.get("classes") or []),
            }
            for item in violations
        ],
        key=lambda item: (str(item["rule"]), str(item["severity"])),
    )


def _plan(*, primary: bool, ppe: bool) -> dict[str, Any]:
    model_keys = []
    if primary:
        model_keys.append("coco_primary")
    if ppe:
        model_keys.append("ppe_specialist")
    return {
        "capabilities": [PPE_CAPABILITY],
        "required_model_keys": model_keys,
        "run_coco_primary": primary,
        "run_ppe_specialist": ppe,
        "run_ppe_closed_set_candidate": False,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_pose_specialist": False,
        "run_rtdetr_phone": False,
        "ppe_prompt_terms": list(PPE_CLASSES),
    }


def _run_group(
    camera_id: str,
    frames: list[Any],
    plans: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[tuple[list[dict[str, Any]], dict[str, int]]]:
    hint = len(frames)

    def run_one(index: int):
        _annotated, detections, _pose, invocations = (
            video_processing._run_grouped_inference(
                camera_id,
                frames[index],
                plans[index],
                conf=0.3,
                device="cuda",
                imgsz=640,
                cfg=cfg,
                frame_batch_size_hint=hint,
            )
        )
        return detections, invocations

    with ThreadPoolExecutor(max_workers=hint) as pool:
        return list(pool.map(run_one, range(hint)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--edge-url-override",
        help="Use an isolated model-server URL without mutating camera config.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.edge_url_override:
        override_url = args.edge_url_override.strip().rstrip("/")
        if not override_url.startswith(("http://", "https://")):
            parser.error("edge-url-override must be an HTTP(S) URL")
        current_settings = model_manager._remote_settings()
        model_manager._remote_settings = lambda: {
            "enabled": True,
            "url": override_url,
            "token": current_settings.get("token", ""),
            "timeout_seconds": current_settings.get("timeout_seconds", 30.0),
        }

    loaded = []
    for path in args.frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {path}")
        loaded.append((path, frame))

    cfg = {
        "global": {
            "coco_inference_width": 640,
            "ppe_inference_width": 640,
        },
        "cameras": {
            "ppe-parity": {
                "safety_rule_ids": ["ppe_rider_helmet"],
            }
        },
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    detection.get_config = lambda: cfg

    per_frame = []
    errors = []
    for start in range(0, len(loaded), 4):
        group = loaded[start : start + 4]
        paths = [item[0] for item in group]
        frames = [item[1] for item in group]

        additive = _run_group(
            "ppe-parity",
            frames,
            [_plan(primary=True, ppe=True) for _frame in frames],
            cfg,
        )
        primary = _run_group(
            "ppe-parity",
            frames,
            [_plan(primary=True, ppe=False) for _frame in frames],
            cfg,
        )
        substitution_plans = []
        for detections, _invocations in primary:
            plan = _plan(primary=False, ppe=True)
            plan.update(
                {
                    "run_ppe_substitution": True,
                    "partial_detection_capabilities": [PPE_CAPABILITY],
                    "ppe_context_detections": [
                        item
                        for item in detections
                        if str(item.get("class") or "").lower() in CONTEXT_CLASSES
                    ],
                }
            )
            substitution_plans.append(plan)
        substituted = _run_group(
            "ppe-parity",
            frames,
            substitution_plans,
            cfg,
        )

        for path, frame, additive_item, substituted_item in zip(
            paths,
            frames,
            additive,
            substituted,
        ):
            additive_detections, additive_invocations = additive_item
            substituted_detections, substituted_invocations = substituted_item
            additive_relevant = _normalize(_relevant(additive_detections))
            substituted_relevant = _normalize(_relevant(substituted_detections))
            additive_violations = _violation_signature(
                detection.check_yoloe_violations(
                    additive_detections,
                    "ppe-parity",
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                    capability_filter={PPE_CAPABILITY},
                )
            )
            substituted_violations = _violation_signature(
                detection.check_yoloe_violations(
                    substituted_detections,
                    "ppe-parity",
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                    capability_filter={PPE_CAPABILITY},
                )
            )
            detection_parity = additive_relevant == substituted_relevant
            violation_parity = additive_violations == substituted_violations
            if not detection_parity:
                errors.append(f"{path.name}: relevant detections changed")
            if not violation_parity:
                errors.append(f"{path.name}: violation outcome changed")
            if additive_invocations.get("coco_primary") != 1:
                errors.append(f"{path.name}: additive primary invocation changed")
            if additive_invocations.get("ppe_specialist") != 1:
                errors.append(f"{path.name}: additive PPE invocation changed")
            if substituted_invocations.get("coco_primary") != 0:
                errors.append(f"{path.name}: substituted path invoked primary")
            if substituted_invocations.get("ppe_specialist") != 1:
                errors.append(f"{path.name}: substituted PPE invocation changed")
            per_frame.append(
                {
                    "frame": path.name,
                    "detection_parity": detection_parity,
                    "violation_parity": violation_parity,
                    "relevant_detection_count": len(additive_relevant),
                    "violation_rules": [
                        item["rule"] for item in additive_violations
                    ],
                }
            )

    ppe_batch = model_manager.remote_ppe_batch_stats()
    expected_batch4_frames = len(loaded) - (len(loaded) % 4)
    expected_batch2_frames = len(loaded) % 4
    if ppe_batch.get("batch4_requests") != expected_batch4_frames:
        errors.append("PPE batch-4 route did not process every full group")
    if ppe_batch.get("batch2_requests") != expected_batch2_frames:
        errors.append("PPE batch-2 route did not process the final partial group")
    for key in (
        "route_fallbacks",
        "admission_overloads",
        "timeout_fallbacks",
        "singleton_bypasses",
    ):
        if int(ppe_batch.get(key) or 0):
            errors.append(f"PPE batch transport reported {key}")

    report = {
        "ok": not errors,
        "errors": errors,
        "frames": len(loaded),
        "detection_parity_frames": sum(
            item["detection_parity"] for item in per_frame
        ),
        "violation_parity_frames": sum(
            item["violation_parity"] for item in per_frame
        ),
        "ppe_batch": ppe_batch,
        "per_frame": per_frame,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
