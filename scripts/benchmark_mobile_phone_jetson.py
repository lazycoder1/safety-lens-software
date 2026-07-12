#!/usr/bin/env python3
"""Compare full-frame and person-crop phone recall through a model server."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import requests


PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67
RELEVANT_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    67: "cell_phone",
}


def _infer(
    session: requests.Session,
    url: str,
    token: str,
    frame,
    *,
    conf: float,
    imgsz: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode benchmark frame")
    headers = {"Content-Type": "image/jpeg"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {
        "model_key": "coco_primary",
        "conf": conf,
        "device": "cuda",
        "imgsz": imgsz,
    }
    latencies_ms = []
    detections: list[dict[str, Any]] = []
    for _ in range(repeats):
        started = time.perf_counter()
        response = session.post(
            f"{url.rstrip('/')}/api/infer/jpeg",
            params=params,
            headers=headers,
            data=encoded.tobytes(),
            timeout=30,
        )
        response.raise_for_status()
        latencies_ms.append((time.perf_counter() - started) * 1000)
        detections = response.json().get("detections") or []
    return detections, latencies_ms


def _infer_local(
    model,
    frame,
    *,
    conf: float,
    imgsz: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    latencies_ms = []
    detections: list[dict[str, Any]] = []
    for _ in range(repeats):
        started = time.perf_counter()
        results = model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            device=0,
            verbose=False,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        boxes = results[0].boxes if results else None
        detections = [] if boxes is None else [
            {
                "bbox": [round(float(value), 2) for value in xyxy],
                "confidence": float(confidence),
                "class_id": int(class_id),
            }
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
            )
        ]
    return detections, latencies_ms


def _class_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    persons = [item for item in detections if item.get("class_id") == PERSON_CLASS_ID]
    phones = [item for item in detections if item.get("class_id") == CELL_PHONE_CLASS_ID]
    relevant: dict[str, list[float]] = {}
    for item in detections:
        class_name = RELEVANT_COCO_CLASSES.get(item.get("class_id"))
        if class_name is None:
            continue
        relevant.setdefault(class_name, []).append(
            round(float(item.get("confidence") or 0), 4)
        )
    return {
        "persons": len(persons),
        "phones": len(phones),
        "phone_confidences": [round(float(item.get("confidence") or 0), 4) for item in phones],
        "person_boxes": [
            {
                "bbox": item.get("bbox"),
                "confidence": round(float(item.get("confidence") or 0), 4),
            }
            for item in persons
        ],
        "phone_boxes": [
            {
                "bbox": item.get("bbox"),
                "confidence": round(float(item.get("confidence") or 0), 4),
            }
            for item in phones
        ],
        "relevant_class_confidences": relevant,
    }


def _expanded_crop(frame, bbox: list[int], expansion: float = 0.15):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_padding = int(box_width * expansion)
    y_padding = int(box_height * expansion)
    x1 = max(0, x1 - x_padding)
    y1 = max(0, y1 - y_padding)
    x2 = min(width, x2 + x_padding)
    y2 = min(height, y2 + y_padding)
    return frame[y1:y2, x1:x2], [x1, y1, x2, y2]


def _source_variants(frame, *, native_only: bool) -> list[tuple[str, Any]]:
    if native_only:
        return [("original", frame)]
    return [
        ("original", frame),
        ("camera_854x480", cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA)),
        ("camera_352x288", cv2.resize(frame, (352, 288), interpolation=cv2.INTER_AREA)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--model", type=Path)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--skip-person-crops", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 160 <= args.imgsz <= 1920:
        parser.error("imgsz must be between 160 and 1920")

    token = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
    session = requests.Session() if args.url else None
    local_model = None
    if args.model:
        from ultralytics import YOLO

        local_model = YOLO(str(args.model), task="detect")

    def infer_frame(frame):
        if local_model is not None:
            return _infer_local(
                local_model,
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                repeats=args.repeats,
            )
        assert session is not None and args.url is not None
        return _infer(
            session,
            args.url,
            token,
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            repeats=args.repeats,
        )

    report: dict[str, Any] = {
        "model": "yolo26s",
        "model_path": args.model.name if args.model else None,
        "transport": "local" if args.model else "http_jpeg",
        "conf": args.conf,
        "imgsz": args.imgsz,
        "images": [],
    }

    for image_path in args.images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not decode {image_path}")
        image_report: dict[str, Any] = {"image": image_path.name, "variants": []}
        for variant_name, variant_frame in _source_variants(
            frame,
            native_only=args.native_only,
        ):
            detections, latencies = infer_frame(variant_frame)
            variant_report: dict[str, Any] = {
                "variant": variant_name,
                "source_shape": list(variant_frame.shape[:2]),
                "full_frame": _class_summary(detections),
                "full_frame_latency_median_ms": round(statistics.median(latencies), 2),
                "person_crops": [],
            }
            person_detections = [] if args.skip_person_crops else [
                item for item in detections if item.get("class_id") == PERSON_CLASS_ID
            ]
            for person in person_detections[:3]:
                crop, source_bbox = _expanded_crop(variant_frame, person["bbox"])
                if crop.size == 0:
                    continue
                crop_detections, crop_latencies = infer_frame(crop)
                variant_report["person_crops"].append(
                    {
                        **_class_summary(crop_detections),
                        "source_shape": list(crop.shape[:2]),
                        "source_bbox": source_bbox,
                        "latency_median_ms": round(statistics.median(crop_latencies), 2),
                    }
                )
            image_report["variants"].append(variant_report)
        report["images"].append(image_report)

    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
