#!/usr/bin/env python3
"""Offline, atomic Ultralytics TensorRT export with a verified sidecar manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tensorrt_engine import build_manifest, manifest_path, write_manifest  # noqa: E402


def _require_fixed_prompt_dependencies() -> None:
    try:
        import clip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Fixed-prompt export requires the pinned Ultralytics CLIP dependency; "
            "rebuild the model-server image before exporting"
        ) from exc


def export_engine(
    *,
    source_path: Path,
    output_path: Path,
    imgsz: int,
    workspace: float,
    device: int,
    force: bool,
    classes: list[str] | None = None,
    class_groups: list[str] | None = None,
    text_encoder_path: Path | None = None,
) -> dict:
    if source_path.suffix.lower() != ".pt" or not source_path.is_file():
        raise ValueError("Source must be an existing Ultralytics .pt model")
    if output_path.suffix.lower() != ".engine":
        raise ValueError("Output must use the .engine suffix")
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing engine: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(classes or [])
    class_groups = list(class_groups or [])
    if any(not value.strip() for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("Fixed prompt classes must be non-empty and unique")
    if classes and (len(class_groups) != len(classes) or any(not value.strip() for value in class_groups)):
        raise ValueError("Each fixed prompt class requires one non-empty semantic class group")
    if not classes and class_groups:
        raise ValueError("Semantic class groups require fixed prompt classes")
    if classes:
        _require_fixed_prompt_dependencies()

    from ultralytics import YOLO
    import tensorrt

    with tempfile.TemporaryDirectory(prefix="tensorrt-export-", dir=output_path.parent) as temporary_dir:
        temporary_source = Path(temporary_dir) / source_path.name
        shutil.copy2(source_path, temporary_source)
        if classes:
            candidates = [
                text_encoder_path,
                source_path.parent / "mobileclip2_b.ts",
                ROOT / "backend" / "mobileclip2_b.ts",
                ROOT / "mobileclip2_b.ts",
            ]
            encoder = next((path for path in candidates if path is not None and path.is_file()), None)
            if encoder is None:
                raise FileNotFoundError("Fixed-prompt export requires mobileclip2_b.ts")
            shutil.copy2(encoder, Path(temporary_dir) / encoder.name)

        original_cwd = Path.cwd()
        try:
            os.chdir(temporary_dir)
            model = YOLO(str(temporary_source))
            task = str(model.task)
            if classes:
                if not hasattr(model, "set_classes"):
                    raise ValueError("Fixed prompt classes require a YOLOE model")
                model.set_classes(classes)
            exported = Path(model.export(
                format="engine",
                imgsz=imgsz,
                half=True,
                batch=1,
                dynamic=False,
                workspace=workspace,
                device=device,
                verbose=False,
            ))
        finally:
            os.chdir(original_cwd)
        if not exported.is_file():
            raise RuntimeError("Ultralytics did not produce a TensorRT engine")
        os.replace(exported, output_path)

    payload = build_manifest(
        source_path=source_path,
        engine_path=output_path,
        imgsz=imgsz,
        precision="fp16",
        task=task,
        classes=classes,
        class_groups=class_groups,
        metadata={
            "batch": 1,
            "dynamic": False,
            "tensorrtVersion": getattr(tensorrt, "__version__", "unknown"),
        },
    )
    write_manifest(manifest_path(output_path), payload)
    return {"engine": str(output_path), "manifest": str(manifest_path(output_path)), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workspace", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument("--class-group", dest="class_groups", action="append", default=[])
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or args.source.with_suffix(".engine")
    report = export_engine(
        source_path=args.source.resolve(),
        output_path=output.resolve(),
        imgsz=args.imgsz,
        workspace=args.workspace,
        device=args.device,
        force=args.force,
        classes=args.classes,
        class_groups=args.class_groups,
        text_encoder_path=args.text_encoder.resolve() if args.text_encoder else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
