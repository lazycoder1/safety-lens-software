#!/usr/bin/env python3
"""Run focused false-positive checks against the configured YOLO26 COCO model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from constants import COCO_NAMES  # noqa: E402
import model_manager  # noqa: E402


DEFAULT_CASES = [
    {
        "id": "blank_frame_no_safety_classes",
        "path": "test-videos/anpr-e2e/blank_frame.jpg",
        "forbidden_classes": ["person", "cell phone", "car", "truck", "bus", "motorcycle", "cat", "dog"],
        "note": "Synthetic blank frame should not emit any safety-relevant COCO class.",
    },
    {
        "id": "idle_scene_no_safety_classes",
        "path": "test-videos/demo-pack/00-idle-no-detection.mp4",
        "forbidden_classes": ["person", "cell phone", "car", "truck", "bus", "motorcycle", "cat", "dog"],
        "note": "Idle demo-pack negative control should stay quiet for core safety classes.",
    },
    {
        "id": "open_office_no_vehicle_animal_phone",
        "path": "test-videos/education-office/open-office-space-914-Open office space.mp4",
        "forbidden_classes": ["cell phone", "car", "truck", "bus", "motorcycle", "cat", "dog"],
        "note": "Office footage may contain people, but should not trigger vehicle, animal, or phone classes.",
    },
    {
        "id": "construction_workers_no_phone_animal",
        "path": "test-videos/construction-workers-helmets.mp4",
        "forbidden_classes": ["cell phone", "cat", "dog"],
        "note": "Construction positive-person fixture should not invent phones or animals.",
    },
    {
        "id": "warehouse_shelves_no_phone_animal",
        "path": "test-videos/warehouse-shelves-boxes.mp4",
        "forbidden_classes": ["cell phone", "cat", "dog"],
        "note": "Warehouse shelves negative control should not invent phones or animals.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def choose_device(requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(item) for item in DEFAULT_CASES]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError("case file must be a JSON list or an object with a cases list")
    return [item for item in data if isinstance(item, dict)]


def sample_frames(path: Path, max_frames: int) -> list[tuple[int, Any]]:
    import cv2

    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        return [(0, frame)]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            count = max(1, min(max_frames, total))
            if count == 1:
                indexes = [0]
            else:
                indexes = sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})
            frames: list[tuple[int, Any]] = []
            for frame_index in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append((int(frame_index), frame))
            if frames:
                return frames

        frames = []
        frame_index = 0
        stride = 15
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % stride == 0:
                frames.append((frame_index, frame))
            frame_index += 1
        if not frames:
            raise RuntimeError(f"Could not sample frames from video: {path}")
        return frames
    finally:
        cap.release()


def result_records(result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return records
    for box in boxes:
        class_id = int(box.cls[0])
        records.append(
            {
                "class_id": class_id,
                "class_name": COCO_NAMES.get(class_id, str(class_id)),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [int(value) for value in box.xyxy[0].tolist()],
            }
        )
    return records


def evaluate_case(model: Any, case: dict[str, Any], *, conf: float, imgsz: int, device: str, max_frames: int) -> dict[str, Any]:
    rel_path = str(case.get("path") or "")
    path = ROOT / rel_path
    forbidden = {str(value) for value in case.get("forbidden_classes") or []}
    started = time.perf_counter()
    frames = sample_frames(path, max_frames)

    forbidden_counts: dict[str, int] = {class_name: 0 for class_name in sorted(forbidden)}
    observed_counts: dict[str, int] = {}
    frame_hits: list[dict[str, Any]] = []
    max_confidence_by_class: dict[str, float] = {}

    for frame_index, frame in frames:
        predictions = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        records = result_records(predictions[0] if predictions else None)
        for record in records:
            class_name = record["class_name"]
            observed_counts[class_name] = observed_counts.get(class_name, 0) + 1
            max_confidence_by_class[class_name] = max(max_confidence_by_class.get(class_name, 0.0), record["confidence"])
            if class_name in forbidden:
                forbidden_counts[class_name] = forbidden_counts.get(class_name, 0) + 1
                frame_hits.append({"frame_index": frame_index, **record})

    forbidden_detection_count = sum(forbidden_counts.values())
    return {
        "id": case.get("id"),
        "path": rel_path,
        "note": case.get("note", ""),
        "ok": forbidden_detection_count == 0,
        "sampled_frames": len(frames),
        "forbidden_classes": sorted(forbidden),
        "forbidden_counts": {key: value for key, value in forbidden_counts.items() if value},
        "forbidden_detection_count": forbidden_detection_count,
        "observed_counts": dict(sorted(observed_counts.items())),
        "max_confidence_by_class": dict(sorted(max_confidence_by_class.items())),
        "forbidden_frame_hits": frame_hits[:20],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    model_definition = model_manager.MODEL_DEFINITIONS["coco_primary"]
    model_path = Path(model_definition["local_path"])
    if not model_path.exists():
        raise RuntimeError(f"Configured coco_primary model is missing: {model_path}")
    device = choose_device(args.device)
    model = YOLO(str(model_path))
    cases = load_cases(Path(args.cases) if args.cases else None)
    case_results = [
        evaluate_case(
            model,
            case,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            max_frames=args.max_frames,
        )
        for case in cases
    ]
    false_positive_count = sum(item["forbidden_detection_count"] for item in case_results)
    return {
        "generated_at": utc_now(),
        "status": "pass" if false_positive_count == 0 else "fail",
        "false_positive_count": false_positive_count,
        "model": {
            "model_key": "coco_primary",
            "filename": model_definition["filename"],
            "path": str(model_path.relative_to(ROOT)),
            "exists": model_path.exists(),
            "bytes": model_path.stat().st_size,
        },
        "settings": {
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": device,
            "max_frames_per_case": args.max_frames,
        },
        "cases": case_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check YOLO26 COCO false positives on negative-control fixtures.")
    parser.add_argument("--cases", default="", help="Optional JSON case file")
    parser.add_argument("--out", default="qa/video_eval/results/yolo26_false_positive_doctor.json")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
