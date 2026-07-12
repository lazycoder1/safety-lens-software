#!/usr/bin/env python3
"""Verify RT-DETR records through the real worker and mobile-alert candidate logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import detection
import video_processing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cfg = {
        "global": {"yolo_conf": 0.3},
        "cameras": {
            "semantic-check": {
                "safety_rule_ids": ["alert_mobile_phone", "alert_animal"],
            }
        },
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "name": "Mobile Phone Usage",
                "type": "alert",
                "classes": ["cell phone"],
                "severity": "P3",
                "enabled": True,
                "threshold": 2,
                "confidence": 0.15,
            },
            {
                "id": "alert_animal",
                "name": "Animal Intrusion",
                "type": "alert",
                "classes": ["dog", "cat"],
                "severity": "P2",
                "enabled": True,
                "threshold": 3,
                "confidence": 0.3,
            },
        ],
    }
    detection.get_config = lambda: cfg
    plan = {
        "capabilities": ["mobile_phone", "animal_presence"],
        "run_coco_primary": False,
        "run_rtdetr_phone": True,
        "run_ppe_specialist": False,
        "run_ppe_closed_set_candidate": False,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_pose_specialist": False,
        "partial_detection_capabilities": ["mobile_phone"],
    }
    labelled = {}
    model_families = set()
    for path in args.frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {path}")
        _annotated, detections, _pose, invocations = (
            video_processing._run_grouped_inference(
                "semantic-check",
                frame,
                plan,
                conf=0.3,
                device="cuda",
                imgsz=640,
                cfg=cfg,
                frame_batch_size_hint=1,
            )
        )
        candidates = detection.check_violations(
            detections,
            "semantic-check",
            capability_filter={"mobile_phone"},
        )
        labelled[path.name] = bool(candidates)
        model_families.update(
            item.get("model_family") for item in detections if item.get("model_family")
        )
        if invocations.get("rtdetr_phone") != 1:
            raise RuntimeError("worker did not invoke RT-DETR exactly once")

    positive = {name: value for name, value in labelled.items() if name.startswith("pos-")}
    negative = {name: value for name, value in labelled.items() if name.startswith("neg-")}
    violation_window = {
        "Animal Intrusion": [True, True, True],
        "Mobile Phone Usage": [True],
    }
    active_violations = {"Animal Intrusion"}
    video_processing._record_empty_violation_observation(
        violation_window,
        active_violations,
        window_size=15,
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
        fresh_ppe_evaluated=False,
        fresh_detection_rule_keys={"Mobile Phone Usage"},
    )
    report = {
        "labelled": labelled,
        "actionable_positive": sum(positive.values()),
        "positive_total": len(positive),
        "actionable_negative": sum(negative.values()),
        "negative_total": len(negative),
        "model_families": sorted(model_families),
        "unrelated_window_preserved": violation_window["Animal Intrusion"]
        == [True, True, True],
        "unrelated_active_preserved": "Animal Intrusion" in active_violations,
        "phone_negative_observation_recorded": violation_window[
            "Mobile Phone Usage"
        ]
        == [True, False],
    }
    errors = []
    if report["actionable_positive"] != 5 or report["positive_total"] != 6:
        errors.append("labelled positive result changed")
    if report["actionable_negative"] != 0 or report["negative_total"] != 4:
        errors.append("labelled negative result changed")
    if report["model_families"] != ["rtdetr_phone"]:
        errors.append("worker records did not retain RT-DETR attribution")
    for key in (
        "unrelated_window_preserved",
        "unrelated_active_preserved",
        "phone_negative_observation_recorded",
    ):
        if not report[key]:
            errors.append(f"partial-observation semantic failed: {key}")
    report["errors"] = errors
    report["ok"] = not errors
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
