#!/usr/bin/env python3
"""Exercise helmet detection plus colour classification through a model server."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from helmet_colour import annotate_helmet_colours  # noqa: E402


DEFAULT_HELMET_PROMPTS = ["motorcycle helmet", "rider helmet", "helmet"]


def _infer(
    url: str,
    frame_path: Path,
    *,
    prompts: list[str],
    confidence: float,
    token: str,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "model_key": "ppe_specialist",
            "conf": confidence,
            "device": "cuda",
            "imgsz": 640,
            "classes": prompts,
        },
        doseq=True,
    )
    headers = {"Content-Type": "image/jpeg"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/infer/jpeg?{query}",
        data=frame_path.read_bytes(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"model server returned HTTP {exc.code}: {detail}") from exc

    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise RuntimeError("model server response did not contain detections")
    normalized = []
    for record in detections:
        item = dict(record)
        if not item.get("class"):
            class_id = int(item.get("class_id", item.get("cls", -1)))
            if 0 <= class_id < len(prompts):
                item["class"] = prompts[class_id]
        normalized.append(item)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-server-url", required=True)
    parser.add_argument("--frames", nargs="+", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_HELMET_PROMPTS)
    parser.add_argument("--detector-confidence", type=float, default=0.30)
    parser.add_argument("--colour-confidence", type=float, default=0.45)
    parser.add_argument("--allowed-colours", nargs="*", default=[])
    parser.add_argument("--require-colour")
    parser.add_argument("--minimum-required-colour-count", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    token = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
    camera = {
        "capabilities": ["helmet_color_compliance"],
        "helmet_colour_policy": {
            "allowed_colours": args.allowed_colours,
            "min_confidence": args.colour_confidence,
        },
    }
    per_frame = []
    colour_counts: Counter[str] = Counter()
    classifier_milliseconds = []
    detector_milliseconds = []

    for frame_path in args.frames:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f"could not decode frame: {frame_path}")

        started = time.perf_counter()
        detections = _infer(
            args.model_server_url,
            frame_path,
            prompts=args.prompts,
            confidence=args.detector_confidence,
            token=token,
        )
        detector_ms = (time.perf_counter() - started) * 1000.0
        detector_milliseconds.append(detector_ms)

        started = time.perf_counter()
        annotate_helmet_colours(frame, detections, camera)
        classifier_ms = (time.perf_counter() - started) * 1000.0
        classifier_milliseconds.append(classifier_ms)

        helmets = []
        for detection in detections:
            colour = detection.get("helmet_colour")
            if colour not in {None, "unknown"}:
                colour_counts[str(colour)] += 1
            if "helmet_colour" in detection:
                helmets.append({
                    "class": detection.get("class"),
                    "detectorConfidence": round(
                        float(detection.get("confidence") or 0.0), 4
                    ),
                    "bbox": detection.get("bbox"),
                    "colour": colour,
                    "colourConfidence": round(
                        float(detection.get("helmet_colour_confidence") or 0.0), 4
                    ),
                    "compliant": detection.get("helmet_colour_compliant"),
                })
        per_frame.append({
            "frame": frame_path.name,
            "detectorMilliseconds": round(detector_ms, 3),
            "classifierMilliseconds": round(classifier_ms, 3),
            "helmets": helmets,
        })

    result = {
        "ok": True,
        "frames": len(per_frame),
        "helmetDetections": sum(len(item["helmets"]) for item in per_frame),
        "colourCounts": dict(sorted(colour_counts.items())),
        "unknownColourCount": sum(
            1
            for item in per_frame
            for helmet in item["helmets"]
            if helmet["colour"] == "unknown"
        ),
        "detectorMedianMilliseconds": round(
            statistics.median(detector_milliseconds), 3
        ),
        "classifierMedianMilliseconds": round(
            statistics.median(classifier_milliseconds), 3
        ),
        "classifierP95Milliseconds": round(
            sorted(classifier_milliseconds)[
                max(0, int(len(classifier_milliseconds) * 0.95) - 1)
            ],
            3,
        ),
        "perFrame": per_frame,
    }

    if args.require_colour:
        observed = colour_counts.get(args.require_colour, 0)
        if observed < args.minimum_required_colour_count:
            result["ok"] = False
            result["error"] = (
                f"required at least {args.minimum_required_colour_count} "
                f"{args.require_colour} helmet detections, observed {observed}"
            )

    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
