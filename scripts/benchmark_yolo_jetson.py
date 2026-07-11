#!/usr/bin/env python3
"""Benchmark nano/small YOLO models on Jetson camera frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised on Jetson/runtime path.
    cv2 = None
    torch = None
    YOLO = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="/host_tmp")
    parser.add_argument("--out", default="/out/yolo_nano_small_benchmark.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--candidate-report", default="", help="Candidate doctor JSON to stamp identity into raw benchmark output")
    parser.add_argument("--model-artifact-sha256", default="", help="Expected selected export SHA256 for the benchmarked model")
    parser.add_argument("--candidate-report-sha256", default="", help="Expected candidate report SHA256 for the benchmarked model")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n.pt", "yolo26s.pt"],
    )
    return parser.parse_args()


def _require_runtime_deps() -> None:
    if cv2 is None or torch is None or YOLO is None:
        raise RuntimeError("benchmark_yolo_jetson.py requires cv2, torch, and ultralytics")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_identity(
    candidate_report_path: Path | None,
    *,
    model_artifact_sha256: str = "",
    candidate_report_sha256: str = "",
) -> dict[str, Any]:
    if candidate_report_path is None:
        return {
            "candidate_report": {"present": False},
            "model_artifact_sha256": model_artifact_sha256 or None,
            "candidate_report_sha256": candidate_report_sha256 or None,
        }

    payload = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    actual_candidate_sha = sha256_file(candidate_report_path)
    manifest = payload.get("promotion_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("candidate report missing promotion_manifest")
    handoff = manifest.get("runtime_handoff")
    selected_export = handoff.get("selected_export") if isinstance(handoff, dict) else None
    if not isinstance(selected_export, dict) or not selected_export.get("sha256"):
        raise RuntimeError("candidate report missing promotion_manifest.runtime_handoff.selected_export.sha256")

    selected_export_sha = str(selected_export["sha256"])
    if model_artifact_sha256 and model_artifact_sha256 != selected_export_sha:
        raise RuntimeError("--model-artifact-sha256 does not match candidate report selected_export.sha256")
    if candidate_report_sha256 and candidate_report_sha256 != actual_candidate_sha:
        raise RuntimeError("--candidate-report-sha256 does not match candidate report file sha256")

    return {
        "candidate_report": {
            "present": True,
            "path": str(candidate_report_path),
            "sha256": actual_candidate_sha,
            "ok": payload.get("ok"),
            "candidate_status": manifest.get("candidate_status"),
            "selected_export_sha256": selected_export_sha,
            "selected_export_path": selected_export.get("path"),
        },
        "model_artifact_sha256": selected_export_sha,
        "candidate_report_sha256": actual_candidate_sha,
    }


def build_identity_fields(args: argparse.Namespace) -> dict[str, Any]:
    has_identity = bool(args.candidate_report or args.model_artifact_sha256 or args.candidate_report_sha256)
    if has_identity and len(args.models) != 1:
        raise RuntimeError("candidate identity can only be stamped when benchmarking exactly one model")
    return load_candidate_identity(
        Path(args.candidate_report) if args.candidate_report else None,
        model_artifact_sha256=args.model_artifact_sha256,
        candidate_report_sha256=args.candidate_report_sha256,
    )


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_frames(frames_dir: Path) -> list[tuple[str, object]]:
    _require_runtime_deps()
    frames: list[tuple[str, object]] = []
    for index in (1, 2, 3):
        path = frames_dir / f"cam{index}-bench.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read benchmark frame: {path}")
        frames.append((path.name, frame))
    return frames


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    _require_runtime_deps()
    identity = build_identity_fields(args)
    frames = load_frames(Path(args.frames_dir))
    device = 0 if torch.cuda.is_available() else "cpu"

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "model_artifact_sha256": identity.get("model_artifact_sha256"),
        "candidate_report_sha256": identity.get("candidate_report_sha256"),
        "candidate_report": identity.get("candidate_report"),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "frames": [{"name": name, "shape": list(frame.shape)} for name, frame in frames],
        "models": [],
    }

    for model_name in args.models:
        model = YOLO(model_name)
        for _ in range(args.warmup):
            for _, frame in frames:
                model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        sync_cuda()

        latencies: list[float] = []
        last_counts: list[dict[str, object]] = []
        for _ in range(args.reps):
            for frame_name, frame in frames:
                start = time.perf_counter()
                predictions = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
                sync_cuda()
                latencies.append((time.perf_counter() - start) * 1000)
                boxes = predictions[0].boxes
                last_counts.append({"frame": frame_name, "count": 0 if boxes is None else len(boxes)})

        mean_ms = statistics.mean(latencies)
        results["models"].append(
            {
                "model": model_name,
                "model_artifact_sha256": identity.get("model_artifact_sha256"),
                "candidate_report_sha256": identity.get("candidate_report_sha256"),
                "mean_ms": round(mean_ms, 2),
                "median_ms": round(statistics.median(latencies), 2),
                "p95_ms": round(percentile(latencies, 0.95), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "fps_single_stream_estimate": round(1000.0 / mean_ms, 2),
                "samples": len(latencies),
                "detections_by_frame_last_run": last_counts[-len(frames):],
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
