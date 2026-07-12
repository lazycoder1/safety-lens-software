#!/usr/bin/env python3
"""Offline, atomic Ultralytics TensorRT export with a verified sidecar manifest."""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from tensorrt_engine import build_manifest, manifest_path, write_manifest  # noqa: E402


DEFAULT_TRTEXEC_PATH = Path("/usr/src/tensorrt/bin/trtexec")


def _require_fixed_prompt_dependencies() -> None:
    try:
        import clip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Fixed-prompt export requires the pinned Ultralytics CLIP dependency; "
            "rebuild the model-server image before exporting"
        ) from exc


def _read_onnx_metadata(onnx_path: Path) -> dict:
    """Recover the Ultralytics metadata required by its TensorRT loader."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    metadata = {}
    for item in model.metadata_props:
        try:
            metadata[item.key] = ast.literal_eval(item.value)
        except (SyntaxError, ValueError):
            metadata[item.key] = item.value
    required = {"batch", "imgsz", "names", "stride", "task"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(
            f"ONNX export is missing Ultralytics metadata: {', '.join(missing)}"
        )
    return metadata


def _wrap_trtexec_engine(
    *, bare_engine: Path, wrapped_engine: Path, metadata: dict
) -> None:
    """Add the metadata prefix expected by Ultralytics AutoBackend."""
    args = metadata.get("args")
    if isinstance(args, dict):
        args["half"] = True
    encoded = json.dumps(metadata).encode("utf-8")
    with wrapped_engine.open("wb") as output, bare_engine.open("rb") as source:
        output.write(struct.pack("<I", len(encoded)))
        output.write(encoded)
        shutil.copyfileobj(source, output, length=1024 * 1024)


def _onnx_export_worker(
    *,
    source_path: Path,
    output_path: Path,
    imgsz: int,
    batch: int,
    device: int,
    classes: list[str],
) -> None:
    """Export ONNX in a child that releases CUDA memory when it exits."""
    from ultralytics import YOLO

    original_cwd = Path.cwd()
    try:
        os.chdir(output_path.parent)
        model = YOLO(str(source_path))
        if classes:
            if not hasattr(model, "set_classes"):
                raise ValueError("Fixed prompt classes require a YOLOE model")
            model.set_classes(classes)
        exported = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                half=False,
                batch=batch,
                dynamic=False,
                simplify=False,
                opset=17,
                device=device,
                verbose=False,
            )
        )
        if not exported.is_absolute():
            exported = output_path.parent / exported
        if not exported.is_file():
            raise RuntimeError("Ultralytics did not produce an ONNX model")
        if exported != output_path:
            os.replace(exported, output_path)
    finally:
        os.chdir(original_cwd)


def _run_onnx_export_process(
    *,
    source_path: Path,
    output_path: Path,
    imgsz: int,
    batch: int,
    device: int,
    classes: list[str],
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_onnx_export_worker,
        kwargs={
            "source_path": source_path,
            "output_path": output_path,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "classes": classes,
        },
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"Isolated ONNX export failed with exit code {process.exitcode}"
        )
    if not output_path.is_file():
        raise RuntimeError("Isolated ONNX export did not produce a model")


def _export_low_memory_engine(
    *,
    source_path: Path,
    temporary_dir: Path,
    output_name: str,
    imgsz: int,
    batch: int,
    device: int,
    classes: list[str],
    workspace_mib: int,
    trtexec_path: Path,
) -> tuple[Path, str]:
    """Export ONNX first, then build a constrained TensorRT engine."""
    if workspace_mib < 1:
        raise ValueError("Low-memory TensorRT workspace must be positive")
    if not trtexec_path.is_file():
        raise FileNotFoundError(f"TensorRT builder does not exist: {trtexec_path}")

    exported_onnx = temporary_dir / f"{source_path.stem}-{imgsz}.onnx"
    _run_onnx_export_process(
        source_path=source_path,
        output_path=exported_onnx,
        imgsz=imgsz,
        batch=batch,
        device=device,
        classes=classes,
    )
    metadata = _read_onnx_metadata(exported_onnx)

    bare_engine = temporary_dir / f"{output_name}.bare"
    subprocess.run(
        [
            str(trtexec_path),
            f"--onnx={exported_onnx}",
            f"--saveEngine={bare_engine}",
            "--fp16",
            f"--workspace={workspace_mib}",
            "--heuristic",
            "--minTiming=1",
            "--avgTiming=1",
            "--buildOnly",
        ],
        check=True,
    )
    if not bare_engine.is_file():
        raise RuntimeError("trtexec did not produce a TensorRT engine")

    wrapped_engine = temporary_dir / output_name
    _wrap_trtexec_engine(
        bare_engine=bare_engine,
        wrapped_engine=wrapped_engine,
        metadata=metadata,
    )
    return wrapped_engine, str(metadata["task"])


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
    low_memory: bool = False,
    low_memory_workspace_mib: int = 256,
    trtexec_path: Path = DEFAULT_TRTEXEC_PATH,
    batch: int = 1,
) -> dict:
    if source_path.suffix.lower() != ".pt" or not source_path.is_file():
        raise ValueError("Source must be an existing Ultralytics .pt model")
    if output_path.suffix.lower() != ".engine":
        raise ValueError("Output must use the .engine suffix")
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing engine: {output_path}")
    if low_memory and low_memory_workspace_mib < 1:
        raise ValueError("Low-memory TensorRT workspace must be positive")
    if not 1 <= batch <= 8:
        raise ValueError("TensorRT batch size must be between 1 and 8")
    if low_memory and not trtexec_path.is_file():
        raise FileNotFoundError(f"TensorRT builder does not exist: {trtexec_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(classes or [])
    class_groups = list(class_groups or [])
    if any(not value.strip() for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("Fixed prompt classes must be non-empty and unique")
    if classes and (
        len(class_groups) != len(classes)
        or any(not value.strip() for value in class_groups)
    ):
        raise ValueError(
            "Each fixed prompt class requires one non-empty semantic class group"
        )
    if not classes and class_groups:
        raise ValueError("Semantic class groups require fixed prompt classes")
    if classes:
        _require_fixed_prompt_dependencies()

    import tensorrt

    with tempfile.TemporaryDirectory(
        prefix="tensorrt-export-", dir=output_path.parent
    ) as temporary_dir:
        temporary_source = Path(temporary_dir) / source_path.name
        shutil.copy2(source_path, temporary_source)
        if classes:
            candidates = [
                text_encoder_path,
                source_path.parent / "mobileclip2_b.ts",
                ROOT / "backend" / "mobileclip2_b.ts",
                ROOT / "mobileclip2_b.ts",
            ]
            encoder = next(
                (path for path in candidates if path is not None and path.is_file()),
                None,
            )
            if encoder is None:
                raise FileNotFoundError("Fixed-prompt export requires mobileclip2_b.ts")
            shutil.copy2(encoder, Path(temporary_dir) / encoder.name)

        original_cwd = Path.cwd()
        try:
            os.chdir(temporary_dir)
            if low_memory:
                exported, task = _export_low_memory_engine(
                    source_path=temporary_source,
                    temporary_dir=Path(temporary_dir),
                    output_name=output_path.name,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    classes=classes,
                    workspace_mib=low_memory_workspace_mib,
                    trtexec_path=trtexec_path,
                )
            else:
                from ultralytics import YOLO

                model = YOLO(str(temporary_source))
                task = str(model.task)
                if classes:
                    if not hasattr(model, "set_classes"):
                        raise ValueError("Fixed prompt classes require a YOLOE model")
                    model.set_classes(classes)
                exported = Path(
                    model.export(
                        format="engine",
                        imgsz=imgsz,
                        half=True,
                        batch=batch,
                        dynamic=False,
                        workspace=workspace,
                        device=device,
                        verbose=False,
                    )
                )
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
        batch=batch,
        classes=classes,
        class_groups=class_groups,
        metadata={
            "batch": batch,
            "builder": "trtexec_heuristic" if low_memory else "ultralytics",
            "dynamic": False,
            **(
                {
                    "avgTiming": 1,
                    "minTiming": 1,
                    "onnxSimplified": False,
                    "workspaceMiB": low_memory_workspace_mib,
                }
                if low_memory
                else {"workspaceGiB": workspace}
            ),
            "tensorrtVersion": getattr(tensorrt, "__version__", "unknown"),
        },
    )
    write_manifest(manifest_path(output_path), payload)
    return {
        "engine": str(output_path),
        "manifest": str(manifest_path(output_path)),
        **payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument(
        "--class-group", dest="class_groups", action="append", default=[]
    )
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Build ONNX with Ultralytics, then use constrained trtexec tactics",
    )
    parser.add_argument("--low-memory-workspace-mib", type=int, default=256)
    parser.add_argument("--trtexec", type=Path, default=DEFAULT_TRTEXEC_PATH)
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
        low_memory=args.low_memory,
        low_memory_workspace_mib=args.low_memory_workspace_mib,
        trtexec_path=args.trtexec.resolve(),
        batch=args.batch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
