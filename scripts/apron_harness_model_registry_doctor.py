#!/usr/bin/env python3
"""Verify and optionally copy a closed-set apron/harness candidate into the model registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLANNED_MODEL_KEY = "ppe_closed_set_candidate"
PLANNED_MODEL_FILENAME = "apron-harness-ppe.onnx"
PLANNED_REGISTRY_PATH = f"models/{PLANNED_MODEL_KEY}/{PLANNED_MODEL_FILENAME}"
REGISTRY_METADATA_KIND = "apron_harness_closed_set_model_registry_metadata"
DEFAULT_APRON_PROMOTION_REPORT = ROOT / "qa/video_eval/results/apron_closed_set_promotion_report.json"
DEFAULT_HARNESS_PROMOTION_REPORT = ROOT / "qa/video_eval/results/harness_closed_set_promotion_report.json"
REQUIRED_SEED_IMPORT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _runtime_handoff(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    handoff = manifest.get("runtime_handoff")
    return handoff if isinstance(handoff, dict) else {}


def _selected_export(handoff: dict[str, Any]) -> dict[str, Any]:
    selected = handoff.get("selected_export")
    return selected if isinstance(selected, dict) else {}


def _registry_entry(handoff: dict[str, Any]) -> dict[str, Any]:
    entry = handoff.get("registry_entry")
    return entry if isinstance(entry, dict) else {}


def _validate_promotion_report(
    path: Path,
    *,
    capability: str,
    expected_candidate_report_sha256: str,
    expected_selected_export_sha256: str,
    errors: list[str],
) -> dict[str, Any]:
    payload = _load_json_if_exists(path)
    prefix = f"{capability} promotion report"
    if not payload:
        errors.append(f"{prefix} is required before registry copy: {path}")
        return {"path": str(path), "valid": False, "present": False}
    if payload.get("ok") is not True:
        errors.append(f"{prefix} ok must be true")
    if payload.get("promotion_status") != "ready_for_runtime_registration":
        errors.append(f"{prefix} promotion_status must be ready_for_runtime_registration")
    if payload.get("capability") != capability:
        errors.append(f"{prefix} capability mismatch")
    if payload.get("candidate_report_sha256") != expected_candidate_report_sha256:
        errors.append(f"{prefix} candidate_report_sha256 must match candidate report")
    selected_export = payload.get("candidate_selected_export")
    if not isinstance(selected_export, dict):
        errors.append(f"{prefix} candidate_selected_export is required")
    elif selected_export.get("sha256") != expected_selected_export_sha256:
        errors.append(f"{prefix} candidate_selected_export.sha256 must match candidate report")
    return {
        "path": str(path),
        "valid": not any(error.startswith(prefix) for error in errors),
        "present": True,
        "candidate_report_sha256": payload.get("candidate_report_sha256"),
        "candidate_selected_export_sha256": selected_export.get("sha256")
        if isinstance(selected_export, dict)
        else None,
    }


def _seed_import_required(candidate_report: dict[str, Any]) -> bool:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return False
    source_lineage = manifest.get("training_source_lineage")
    if not isinstance(source_lineage, dict):
        source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, dict):
        return False
    seed_import = source_lineage.get("seed_import_manifest")
    if not isinstance(seed_import, dict):
        return False
    gate = seed_import.get("gate")
    return isinstance(gate, dict) and gate.get("required") is True


def _candidate_seed_export_import_manifest(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    seed_export = manifest.get("seed_export_import_manifest")
    if isinstance(seed_export, dict):
        return seed_export
    capture_preflight = manifest.get("training_capture_preflight")
    if isinstance(capture_preflight, dict) and isinstance(
        capture_preflight.get("seed_export_import_manifest"), dict
    ):
        return capture_preflight["seed_export_import_manifest"]
    source_lineage = manifest.get("training_source_lineage")
    if not isinstance(source_lineage, dict):
        source_lineage = manifest.get("source_lineage")
    if isinstance(source_lineage, dict) and isinstance(
        source_lineage.get("seed_export_import_manifest"), dict
    ):
        return source_lineage["seed_export_import_manifest"]
    return {}


def _capture_manifest_sha(candidate_report: dict[str, Any]) -> str | None:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return None
    source_lineage = manifest.get("training_source_lineage")
    if not isinstance(source_lineage, dict):
        source_lineage = manifest.get("source_lineage")
    if isinstance(source_lineage, dict):
        capture_manifest = source_lineage.get("capture_manifest")
        if isinstance(capture_manifest, dict):
            file_info = capture_manifest.get("file")
            if isinstance(file_info, dict) and file_info.get("sha256"):
                return str(file_info["sha256"])
            if capture_manifest.get("manifest_sha256"):
                return str(capture_manifest["manifest_sha256"])
    capture_matrix_manifest = manifest.get("capture_matrix_manifest")
    if isinstance(capture_matrix_manifest, dict) and capture_matrix_manifest.get("source_manifest_sha256"):
        return str(capture_matrix_manifest["source_manifest_sha256"])
    return None


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _validate_source_recheck_block(source_recheck: Any, *, prefix: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(source_recheck, dict) or not source_recheck:
        errors.append(f"{prefix}.source_recheck is required")
        return {}
    if source_recheck.get("exists") is not True:
        errors.append(f"{prefix}.source_recheck.exists must be true")
    if not source_recheck.get("path"):
        errors.append(f"{prefix}.source_recheck.path is required")
    if not _valid_sha(source_recheck.get("sha256")):
        errors.append(f"{prefix}.source_recheck.sha256 must be a 64-character digest")
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        errors.append(f"{prefix}.source_recheck.evidence_boundary must preserve non-approval boundary")
    return source_recheck


def _positive_count(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _positive_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _validate_seed_export_preflight_import(item: dict[str, Any], *, index: int, errors: list[str]) -> None:
    prefix = f"seed_export_import_manifest.imports[{index}].yolo_export_preflight"
    preflight = item.get("yolo_export_preflight")
    if not isinstance(preflight, dict):
        errors.append(f"{prefix} is required")
        return
    if preflight.get("checked") is not True:
        errors.append(f"{prefix}.checked must be true")
    if preflight.get("errors") not in ([], None):
        errors.append(f"{prefix}.errors must be empty")
    raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
    preflight_sha256 = str(preflight.get("sha256") or "").strip()
    if not preflight_sha256:
        errors.append(f"{prefix}.sha256 is required")
    elif raw_export_sha256 and preflight_sha256 != raw_export_sha256:
        errors.append(f"{prefix}.sha256 must match raw_export_sha256")
    orphan_counts = (
        preflight.get("orphan_label_count_by_split")
        if isinstance(preflight.get("orphan_label_count_by_split"), dict)
        else {}
    )
    for split, count in orphan_counts.items():
        if _positive_int(count) > 0:
            errors.append(f"{prefix}.orphan_label_count_by_split.{split} must be 0")
    capability = str(item.get("capability") or "")
    required_classes = REQUIRED_SEED_IMPORT_CLASSES.get(capability, set())
    if not required_classes:
        errors.append(
            f"seed_export_import_manifest.imports[{index}].capability must be one of "
            f"{sorted(REQUIRED_SEED_IMPORT_CLASSES)}"
        )
        return
    counts = preflight.get("label_file_count_by_local_class")
    if not isinstance(counts, dict):
        errors.append(f"{prefix}.label_file_count_by_local_class is required")
        return
    for class_name in sorted(required_classes):
        if _positive_int(counts.get(class_name)) <= 0:
            errors.append(f"{prefix}.label_file_count_by_local_class.{class_name} must be greater than 0")


def _validate_seed_export_import_manifest(candidate_report: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    required = _seed_import_required(candidate_report)
    manifest = _candidate_seed_export_import_manifest(candidate_report)
    status = {
        "required": required,
        "present": bool(manifest),
        "valid": False,
        "sha256": manifest.get("sha256") if isinstance(manifest, dict) else None,
        "updated_manifest_sha256": manifest.get("updated_manifest_sha256") if isinstance(manifest, dict) else None,
        "seed_source_review_sha256": manifest.get("seed_source_review_sha256") if isinstance(manifest, dict) else None,
        "seed_import_manifest_sha256": manifest.get("seed_import_manifest_sha256") if isinstance(manifest, dict) else None,
        "source_recheck": manifest.get("source_recheck") if isinstance(manifest, dict) else None,
    }
    if not required:
        status["valid"] = True
        return status
    if not manifest:
        errors.append("seed_export_import_manifest is required before model registry copy")
        return status
    if manifest.get("valid") is not True:
        errors.append("seed_export_import_manifest.valid must be true before model registry copy")
    if manifest.get("partial_materialization") is True:
        errors.append("seed_export_import_manifest.partial_materialization must be false")
    for field in ("sha256", "seed_source_review_sha256", "seed_import_manifest_sha256", "updated_manifest_sha256"):
        if not _valid_sha(manifest.get(field)):
            errors.append(f"seed_export_import_manifest.{field} must be a 64-character digest")
    source_recheck = _validate_source_recheck_block(
        manifest.get("source_recheck"),
        prefix="seed_export_import_manifest",
        errors=errors,
    )
    status["source_recheck"] = source_recheck or None
    expected_capture_sha = _capture_manifest_sha(candidate_report)
    if expected_capture_sha and manifest.get("updated_manifest_sha256") != expected_capture_sha:
        errors.append("seed_export_import_manifest.updated_manifest_sha256 must match candidate capture manifest sha256")
    validation = manifest.get("updated_manifest_validation")
    if not isinstance(validation, dict):
        errors.append("seed_export_import_manifest.updated_manifest_validation is required")
    else:
        if validation.get("checked") is not True:
            errors.append("seed_export_import_manifest.updated_manifest_validation.checked must be true")
        if validation.get("ok") is not True:
            errors.append("seed_export_import_manifest.updated_manifest_validation.ok must be true")
        if validation.get("schema_only") is not False:
            errors.append("seed_export_import_manifest.updated_manifest_validation.schema_only must be false")
        if validation.get("mode") != "production":
            errors.append("seed_export_import_manifest.updated_manifest_validation.mode must be production")
        if validation.get("manifest_sha256") != manifest.get("updated_manifest_sha256"):
            errors.append("seed_export_import_manifest.updated_manifest_validation.manifest_sha256 must match updated_manifest_sha256")
    for field in ("imported_label_count", "imported_clip_count", "copied_image_count"):
        if not _positive_count(manifest.get(field)):
            errors.append(f"seed_export_import_manifest.{field} must be greater than 0")
    imports = manifest.get("imports")
    if not isinstance(imports, list) or not imports:
        errors.append("seed_export_import_manifest.imports must include materialized seed imports")
    else:
        for index, item in enumerate(imports):
            if not isinstance(item, dict):
                errors.append(f"seed_export_import_manifest.imports[{index}] must be an object")
                continue
            if not _positive_count(item.get("imported_label_count")):
                errors.append(f"seed_export_import_manifest.imports[{index}].imported_label_count must be greater than 0")
            if not _positive_count(item.get("copied_image_count")):
                errors.append(f"seed_export_import_manifest.imports[{index}].copied_image_count must be greater than 0")
            if item.get("errors"):
                errors.append(f"seed_export_import_manifest.imports[{index}].errors must be empty")
            if not _valid_sha(item.get("raw_export_sha256")):
                errors.append(f"seed_export_import_manifest.imports[{index}].raw_export_sha256 must be a 64-character digest")
            _validate_seed_export_preflight_import(item, index=index, errors=errors)
    status["valid"] = not any(error.startswith("seed_export_import_manifest") for error in errors)
    return status


def _resolve_export_path(raw_path: str, candidate_report_path: Path, artifact_root: Path | None) -> Path:
    export_path = Path(raw_path)
    if export_path.is_absolute():
        return export_path
    if artifact_root is not None:
        return artifact_root / export_path
    return candidate_report_path.parent / export_path


def _resolve_destination(registry_root: Path, planned_registry_path: str, errors: list[str]) -> Path:
    raw_path = Path(planned_registry_path)
    if raw_path.is_absolute():
        errors.append("planned_registry_path must be repository-relative")
        raw_path = Path(PLANNED_REGISTRY_PATH)
    destination = (registry_root / raw_path).resolve()
    root = registry_root.resolve()
    if destination != root and root not in destination.parents:
        errors.append("planned_registry_path must stay inside registry_root")
    return destination


def _model_manager_definition_status(expected_registry_path: str) -> dict[str, Any]:
    errors: list[str] = []
    status: dict[str, Any] = {
        "checked": True,
        "model_key": PLANNED_MODEL_KEY,
        "expected_registry_path": expected_registry_path,
        "registered": False,
        "valid": False,
        "errors": errors,
    }
    backend_path = str(ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    try:
        import model_manager  # type: ignore
    except Exception as exc:
        errors.append(f"model_manager import failed: {exc}")
        return status

    definition = model_manager.MODEL_DEFINITIONS.get(PLANNED_MODEL_KEY)
    if not isinstance(definition, dict):
        errors.append(f"{PLANNED_MODEL_KEY} missing from backend/model_manager.py")
        return status

    status["registered"] = True
    status["filename"] = definition.get("filename")
    status["download_url"] = definition.get("download_url")
    status["shared_asset_key"] = definition.get("shared_asset_key")
    local_path = definition.get("local_path")
    if isinstance(local_path, Path):
        try:
            status["registry_path"] = str(local_path.relative_to(ROOT))
        except ValueError:
            status["registry_path"] = str(local_path)
    else:
        errors.append("model_manager definition local_path must be a pathlib.Path")

    if definition.get("model_key") != PLANNED_MODEL_KEY:
        errors.append("model_manager definition model_key mismatch")
    if definition.get("filename") != PLANNED_MODEL_FILENAME:
        errors.append("model_manager definition filename mismatch")
    if status.get("registry_path") != expected_registry_path:
        errors.append("model_manager definition path does not match candidate handoff")
    if definition.get("download_url"):
        errors.append("closed-set candidate must use manual artifact install with empty download_url")
    if definition.get("shared_asset_key") != "apron-harness-ppe":
        errors.append("model_manager definition shared_asset_key mismatch")
    status["valid"] = not errors
    return status


def _copy_verified(source: Path, destination: Path, expected_sha256: str, *, force: bool) -> tuple[bool, str | None]:
    if destination.exists():
        existing_sha = _sha256_file(destination)
        if existing_sha == expected_sha256:
            return False, None
        if not force:
            return False, (
                "destination already exists with a different sha256; "
                "rerun with --force to replace it"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.name}.part")
    shutil.copy2(source, tmp_path)
    copied_sha = _sha256_file(tmp_path)
    if copied_sha != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        return False, "copied artifact sha256 does not match expected source sha256"
    os.replace(tmp_path, destination)
    return True, None


def _registry_metadata_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".registry.json")


def _build_registry_metadata(
    *,
    destination: Path,
    planned_registry_path: str,
    candidate_report_path: Path,
    candidate_report_sha256: str,
    source_path: Path | None,
    source_export_sha256: str | None,
    artifact_sha256: str | None,
    seed_export_import_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": REGISTRY_METADATA_KIND,
        "version": 1,
        "registered_at": utc_now(),
        "model_key": PLANNED_MODEL_KEY,
        "registry_path": planned_registry_path,
        "artifact_path": str(destination),
        "artifact_sha256": artifact_sha256,
        "candidate_report": str(candidate_report_path),
        "candidate_report_sha256": candidate_report_sha256,
        "source_export_path": str(source_path) if source_path is not None else None,
        "source_export_sha256": source_export_sha256,
        "seed_export_import_manifest_required": seed_export_import_manifest.get("required") is True,
        "seed_export_import_manifest_sha256": seed_export_import_manifest.get("sha256"),
        "seed_source_review_sha256": seed_export_import_manifest.get("seed_source_review_sha256"),
        "seed_import_manifest_sha256": seed_export_import_manifest.get("seed_import_manifest_sha256"),
        "seed_updated_manifest_sha256": seed_export_import_manifest.get("updated_manifest_sha256"),
        "seed_source_recheck": seed_export_import_manifest.get("source_recheck"),
    }


def _write_registry_metadata(destination: Path, metadata: dict[str, Any]) -> None:
    path = _registry_metadata_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.part")
    tmp_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _registry_metadata_status(
    destination: Path,
    *,
    planned_registry_path: str,
    expected_artifact_sha256: str | None,
    expected_candidate_report_sha256: str,
    expected_source_export_sha256: str | None,
    expected_seed_export_import_manifest: dict[str, Any],
) -> dict[str, Any]:
    path = _registry_metadata_path(destination)
    errors: list[str] = []
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "valid": False,
            "errors": ["registry metadata sidecar is missing"],
        }
    sha256 = _sha256_file(path)
    try:
        payload = _load_json(path)
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "sha256": sha256,
            "valid": False,
            "errors": [f"registry metadata sidecar is unreadable: {exc}"],
        }

    if payload.get("kind") != REGISTRY_METADATA_KIND:
        errors.append(f"registry metadata kind must be {REGISTRY_METADATA_KIND}")
    if payload.get("version") != 1:
        errors.append("registry metadata version must be 1")
    if payload.get("model_key") != PLANNED_MODEL_KEY:
        errors.append(f"registry metadata model_key must be {PLANNED_MODEL_KEY}")
    if payload.get("registry_path") != planned_registry_path:
        errors.append("registry metadata registry_path must match planned_registry_path")
    if payload.get("artifact_sha256") != expected_artifact_sha256:
        errors.append("registry metadata artifact_sha256 must match selected_export.sha256")
    if payload.get("candidate_report_sha256") != expected_candidate_report_sha256:
        errors.append("registry metadata candidate_report_sha256 must match candidate report")
    if payload.get("source_export_sha256") != expected_source_export_sha256:
        errors.append("registry metadata source_export_sha256 must match source export")
    seed_required = expected_seed_export_import_manifest.get("required") is True
    if payload.get("seed_export_import_manifest_required") is not seed_required:
        errors.append("registry metadata seed_export_import_manifest_required must match candidate lineage")
    if payload.get("seed_export_import_manifest_sha256") != expected_seed_export_import_manifest.get("sha256"):
        errors.append("registry metadata seed_export_import_manifest_sha256 must match candidate")
    if payload.get("seed_source_review_sha256") != expected_seed_export_import_manifest.get("seed_source_review_sha256"):
        errors.append("registry metadata seed_source_review_sha256 must match candidate")
    if payload.get("seed_import_manifest_sha256") != expected_seed_export_import_manifest.get("seed_import_manifest_sha256"):
        errors.append("registry metadata seed_import_manifest_sha256 must match candidate")
    if payload.get("seed_updated_manifest_sha256") != expected_seed_export_import_manifest.get("updated_manifest_sha256"):
        errors.append("registry metadata seed_updated_manifest_sha256 must match candidate")
    if payload.get("seed_source_recheck") != expected_seed_export_import_manifest.get("source_recheck"):
        errors.append("registry metadata seed_source_recheck must match candidate")

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256,
        "valid": not errors,
        "errors": errors,
    }


def audit_candidate_registry(
    candidate_report_path: Path,
    *,
    artifact_root: Path | None = None,
    registry_root: Path = ROOT,
    apron_promotion_report: Path | None = None,
    harness_promotion_report: Path | None = None,
    copy: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    candidate_report_sha256 = _sha256_file(candidate_report_path)
    candidate_report = _load_json(candidate_report_path)
    handoff = _runtime_handoff(candidate_report)
    selected = _selected_export(handoff)
    registry_entry = _registry_entry(handoff)
    seed_export_import_manifest = _validate_seed_export_import_manifest(candidate_report, errors)

    if candidate_report.get("ok") is not True:
        errors.append("candidate report must have ok=true")
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        errors.append("candidate report must include promotion_manifest")
    elif manifest.get("candidate_status") != "ready_for_side_by_side_runtime_test":
        errors.append("candidate_status must be ready_for_side_by_side_runtime_test")
    if not handoff:
        errors.append("candidate report must include promotion_manifest.runtime_handoff")

    planned_model_key = str(handoff.get("planned_model_key") or "")
    planned_registry_path = str(handoff.get("planned_registry_path") or "")
    if planned_model_key != PLANNED_MODEL_KEY:
        errors.append(f"planned_model_key must be {PLANNED_MODEL_KEY}")
    if planned_registry_path != PLANNED_REGISTRY_PATH:
        errors.append(f"planned_registry_path must be {PLANNED_REGISTRY_PATH}")

    expected_sha = str(selected.get("sha256") or "")
    selected_path = str(selected.get("path") or "")
    selected_suffix = str(selected.get("suffix") or Path(selected_path).suffix).lower()
    source_path = _resolve_export_path(selected_path, candidate_report_path, artifact_root) if selected_path else None
    destination = _resolve_destination(registry_root, planned_registry_path or PLANNED_REGISTRY_PATH, errors)

    if not selected:
        errors.append("runtime_handoff.selected_export is required")
    if not selected_path:
        errors.append("runtime_handoff.selected_export.path is required")
    if not expected_sha or len(expected_sha) != 64:
        errors.append("runtime_handoff.selected_export.sha256 must be a 64-character digest")
    if selected_suffix != destination.suffix.lower():
        errors.append(
            "selected export suffix must match the planned registry artifact suffix "
            f"({selected_suffix or 'unknown'} != {destination.suffix.lower()})"
        )
    if source_path is None or not source_path.exists():
        errors.append(f"selected export artifact does not exist: {selected_path or '<missing>'}")

    actual_source_sha = None
    if source_path is not None and source_path.exists():
        actual_source_sha = _sha256_file(source_path)
        if expected_sha and actual_source_sha != expected_sha:
            errors.append("selected export sha256 does not match the source artifact on disk")

    promotion_reports = {
        "apron_required": {
            "path": str(apron_promotion_report or DEFAULT_APRON_PROMOTION_REPORT),
            "valid": False,
            "present": False,
        },
        "harness_required": {
            "path": str(harness_promotion_report or DEFAULT_HARNESS_PROMOTION_REPORT),
            "valid": False,
            "present": False,
        },
    }
    if copy and candidate_report_sha256 and expected_sha and len(expected_sha) == 64:
        promotion_reports["apron_required"] = _validate_promotion_report(
            apron_promotion_report or DEFAULT_APRON_PROMOTION_REPORT,
            capability="apron_required",
            expected_candidate_report_sha256=candidate_report_sha256,
            expected_selected_export_sha256=expected_sha,
            errors=errors,
        )
        promotion_reports["harness_required"] = _validate_promotion_report(
            harness_promotion_report or DEFAULT_HARNESS_PROMOTION_REPORT,
            capability="harness_required",
            expected_candidate_report_sha256=candidate_report_sha256,
            expected_selected_export_sha256=expected_sha,
            errors=errors,
        )
    elif copy:
        errors.append("registry copy requires candidate report SHA and selected_export.sha256 before promotion validation")

    if registry_entry.get("model_key") != PLANNED_MODEL_KEY:
        errors.append(f"registry_entry.model_key must be {PLANNED_MODEL_KEY}")
    if registry_entry.get("registry_path") != planned_registry_path:
        errors.append("registry_entry.registry_path must match planned_registry_path")
    if expected_sha and registry_entry.get("source_export_sha256") != expected_sha:
        errors.append("registry_entry.source_export_sha256 must match selected_export.sha256")

    model_definition = _model_manager_definition_status(planned_registry_path or PLANNED_REGISTRY_PATH)
    if model_definition.get("valid") is not True:
        errors.append("model_manager definition does not match candidate handoff")

    copied = False
    copy_error = None
    if copy and not errors and source_path is not None:
        copied, copy_error = _copy_verified(source_path, destination, expected_sha, force=force)
        if copy_error:
            errors.append(copy_error)
        else:
            _write_registry_metadata(
                destination,
                _build_registry_metadata(
                    destination=destination,
                    planned_registry_path=planned_registry_path or PLANNED_REGISTRY_PATH,
                    candidate_report_path=candidate_report_path,
                    candidate_report_sha256=candidate_report_sha256,
                    source_path=source_path,
                    source_export_sha256=actual_source_sha,
                    artifact_sha256=expected_sha,
                    seed_export_import_manifest=seed_export_import_manifest,
                ),
            )
    elif not copy:
        warnings.append("dry run only; rerun with --copy to install the candidate artifact")

    destination_exists = destination.exists()
    destination_sha = _sha256_file(destination) if destination_exists else None
    metadata_status = _registry_metadata_status(
        destination,
        planned_registry_path=planned_registry_path or PLANNED_REGISTRY_PATH,
        expected_artifact_sha256=expected_sha or None,
        expected_candidate_report_sha256=candidate_report_sha256,
        expected_source_export_sha256=actual_source_sha,
        expected_seed_export_import_manifest=seed_export_import_manifest,
    )
    artifact_matches = bool(destination_sha and destination_sha == expected_sha)
    destination_matches = artifact_matches and metadata_status.get("valid") is True
    if artifact_matches and metadata_status.get("valid") is not True and not copy:
        warnings.append("destination artifact sha256 matches, but registry metadata is missing or stale; rerun with --copy")
    if copy and not destination_matches and not errors:
        errors.append("destination artifact and registry metadata were not registered with the expected sha256")

    registry_status = "not_ready"
    if not errors:
        registry_status = "registered" if destination_matches else "ready_to_copy"

    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "candidate_report": str(candidate_report_path),
        "candidate_report_sha256": candidate_report_sha256,
        "registry_status": registry_status,
        "copy_requested": copy,
        "copied": copied,
        "force": force,
        "model_manager_definition": model_definition,
        "source_export": {
            "path": str(source_path) if source_path is not None else None,
            "exists": bool(source_path and source_path.exists()),
            "sha256": actual_source_sha,
            "expected_sha256": expected_sha or None,
            "suffix": selected_suffix or None,
        },
        "promotion_reports": promotion_reports,
        "seed_export_import_manifest": seed_export_import_manifest,
        "destination": {
            "path": str(destination),
            "registry_path": planned_registry_path or PLANNED_REGISTRY_PATH,
            "exists": destination_exists,
            "sha256": destination_sha,
            "matches_expected_sha256": destination_matches,
            "artifact_matches_expected_sha256": artifact_matches,
            "registry_metadata": metadata_status,
        },
        "registry_entry": registry_entry,
        "next_required_gates": [
            "run_apron_side_by_side_promotion_report",
            "run_harness_side_by_side_promotion_report",
            "run_factory_ppe_jetson_full_gate",
            "add_artifact_to_factory_ppe_3cam_registry_models_after_promotion",
            "activate_ppe_closed_set_candidate_only_after_runtime_and_jetson_gates",
        ],
        "errors": errors,
        "warnings": warnings,
    }


def audit_planned_registry(
    *,
    registry_root: Path = ROOT,
) -> dict[str, Any]:
    """Audit the planned closed-set registry slot before a trained artifact exists."""

    errors: list[str] = []
    warnings = [
        "no candidate report supplied; this is a planned-registry audit only and cannot pass production registration",
        "run scripts/apron_harness_candidate_doctor.py after training to produce the candidate report",
    ]
    planned_registry_path = PLANNED_REGISTRY_PATH
    destination = _resolve_destination(registry_root, planned_registry_path, errors)
    model_definition = _model_manager_definition_status(planned_registry_path)
    if model_definition.get("valid") is not True:
        errors.append("model_manager definition does not match planned registry path")

    destination_exists = destination.exists()
    destination_sha = _sha256_file(destination) if destination_exists else None
    metadata_status = _registry_metadata_status(
        destination,
        planned_registry_path=planned_registry_path,
        expected_artifact_sha256=None,
        expected_candidate_report_sha256="",
        expected_source_export_sha256=None,
        expected_seed_export_import_manifest={
            "required": False,
            "sha256": None,
            "seed_source_review_sha256": None,
            "seed_import_manifest_sha256": None,
            "updated_manifest_sha256": None,
            "source_recheck": None,
        },
    )

    registry_status = "planned_no_candidate"
    if destination_exists:
        registry_status = "planned_artifact_present_without_candidate"
        warnings.append("registry destination exists but no candidate report was supplied to verify it")

    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "candidate_report": None,
        "candidate_report_sha256": None,
        "registry_status": registry_status,
        "copy_requested": False,
        "copied": False,
        "force": False,
        "model_manager_definition": model_definition,
        "source_export": {
            "path": None,
            "exists": False,
            "sha256": None,
            "expected_sha256": None,
            "suffix": None,
        },
        "seed_export_import_manifest": {
            "required": False,
            "present": False,
            "valid": True,
            "sha256": None,
            "updated_manifest_sha256": None,
            "seed_source_review_sha256": None,
            "seed_import_manifest_sha256": None,
            "source_recheck": None,
        },
        "destination": {
            "path": str(destination),
            "registry_path": planned_registry_path,
            "exists": destination_exists,
            "sha256": destination_sha,
            "matches_expected_sha256": False,
            "artifact_matches_expected_sha256": False,
            "registry_metadata": metadata_status,
        },
        "registry_entry": {
            "model_key": PLANNED_MODEL_KEY,
            "registry_path": planned_registry_path,
            "source_export_sha256": None,
            "activation_status": "blocked_until_candidate_report_promotion_reports_and_jetson_gate_pass",
        },
        "next_required_gates": [
            "train_closed_set_apron_harness_candidate",
            "run_apron_harness_candidate_doctor",
            "run_apron_side_by_side_promotion_report",
            "run_harness_side_by_side_promotion_report",
            "rerun_model_registry_doctor_with_candidate_report_and_copy",
            "run_factory_ppe_jetson_full_gate",
            "activate_ppe_closed_set_candidate_only_after_runtime_and_jetson_gates",
        ],
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify/copy closed-set apron/harness model artifact into the registry.")
    parser.add_argument("--candidate-report", default="", help="JSON from scripts/apron_harness_candidate_doctor.py")
    parser.add_argument(
        "--planned-audit",
        action="store_true",
        help="Write a non-promoting audit of the planned registry slot when no candidate exists yet",
    )
    parser.add_argument("--artifact-root", default="", help="Optional root for relative selected_export.path values")
    parser.add_argument("--registry-root", default=str(ROOT), help="Project root that contains the models/ registry")
    parser.add_argument(
        "--apron-promotion-report",
        default=str(DEFAULT_APRON_PROMOTION_REPORT),
        help="Closed-set apron side-by-side promotion report required before --copy",
    )
    parser.add_argument(
        "--harness-promotion-report",
        default=str(DEFAULT_HARNESS_PROMOTION_REPORT),
        help="Closed-set harness side-by-side promotion report required before --copy",
    )
    parser.add_argument("--copy", action="store_true", help="Copy the selected export into the planned registry path")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination with a different sha256")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.candidate_report:
        report = audit_candidate_registry(
            Path(args.candidate_report),
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            registry_root=Path(args.registry_root),
            apron_promotion_report=Path(args.apron_promotion_report),
            harness_promotion_report=Path(args.harness_promotion_report),
            copy=bool(args.copy),
            force=bool(args.force),
        )
    elif args.planned_audit:
        if args.copy:
            raise SystemExit("--copy requires --candidate-report")
        report = audit_planned_registry(registry_root=Path(args.registry_root))
    else:
        raise SystemExit("--candidate-report is required unless --planned-audit is set")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"{status}: registry_status={report['registry_status']}")
        if args.out:
            print(f"wrote: {args.out}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
