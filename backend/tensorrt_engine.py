"""TensorRT engine manifest creation and runtime validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".json")


def build_manifest(
    *,
    source_path: Path,
    engine_path: Path,
    imgsz: int,
    precision: str,
    task: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if imgsz < 1:
        raise ValueError("TensorRT image size must be positive")
    precision = precision.lower()
    if precision not in {"fp16", "fp32", "int8"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    if not source_path.is_file() or not engine_path.is_file():
        raise FileNotFoundError("TensorRT source and engine files must both exist")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceFile": source_path.name,
        "sourceSha256": file_sha256(source_path),
        "engineFile": engine_path.name,
        "engineSha256": file_sha256(engine_path),
        "imgsz": imgsz,
        "precision": precision,
        "task": task,
        "metadata": metadata or {},
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_engine(
    *,
    source_path: Path,
    engine_path: Path,
    expected_task: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if engine_path.suffix.lower() != ".engine":
        return None, "Configured TensorRT artifact must use the .engine suffix"
    if not engine_path.is_file():
        return None, f"Configured TensorRT engine does not exist: {engine_path}"
    sidecar = manifest_path(engine_path)
    if not sidecar.is_file():
        return None, f"TensorRT manifest does not exist: {sidecar}"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"TensorRT manifest is unreadable: {exc}"
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        return None, f"Unsupported TensorRT manifest schema: {payload.get('schemaVersion')}"
    if payload.get("task") != expected_task:
        return None, f"TensorRT task mismatch: expected {expected_task}, found {payload.get('task')}"
    if payload.get("sourceFile") != source_path.name:
        return None, "TensorRT manifest source filename does not match the active model"
    if payload.get("engineFile") != engine_path.name:
        return None, "TensorRT manifest engine filename does not match the configured engine"
    if payload.get("precision") not in {"fp16", "fp32", "int8"}:
        return None, "TensorRT manifest precision is invalid"
    try:
        imgsz = int(payload.get("imgsz"))
    except (TypeError, ValueError):
        return None, "TensorRT manifest image size is invalid"
    if imgsz < 1:
        return None, "TensorRT manifest image size is invalid"
    if payload.get("sourceSha256") != file_sha256(source_path):
        return None, "TensorRT source-model hash does not match the active model"
    if payload.get("engineSha256") != file_sha256(engine_path):
        return None, "TensorRT engine hash does not match its manifest"
    return payload, None
