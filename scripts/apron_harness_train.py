#!/usr/bin/env python3
"""Plan or run closed-set apron/harness PPE training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apron_harness_dataset_doctor import (  # noqa: E402
    ALLOWED_PERMISSIONS,
    DEFAULT_SEED_IMPORT_MANIFEST,
    DEFAULT_SEED_SOURCE_REVIEW,
    MIN_COUNTS,
    REQUIRED_CLASSES,
    TAXONOMY_VERSION,
    YOLO_LABEL_FORMAT,
    validate_capture_matrix_progress,
    validate_manifest,
)


ALLOWED_MODELS = {
    "yolo26n.pt",
    "yolo26s.pt",
}
DEFAULT_EXPORT_FORMATS = ["onnx"]
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"
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


def _sidecar_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _file_lineage(path: Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None:
        return {
            "required": required,
            "path": None,
            "exists": False,
            "sha256": None,
        }
    exists = path.exists()
    return {
        "required": required,
        "path": str(path),
        "exists": exists,
        "sha256": _sha256_file(path) if exists and path.is_file() else None,
    }


def _source_recheck_lineage(seed_source_review_report: Path | None) -> dict[str, Any]:
    if seed_source_review_report is None or not seed_source_review_report.exists():
        return {}
    try:
        payload = json.loads(seed_source_review_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    source_recheck = payload.get("source_recheck")
    return source_recheck if isinstance(source_recheck, dict) else {}


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _resolve_repo_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else ROOT / path


def _validate_source_recheck_lineage(source_recheck: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not source_recheck:
        return ["source_recheck is required when seed source review gate is required"]
    if source_recheck.get("exists") is not True:
        errors.append("source_recheck.exists must be true")
    if not _valid_sha256(source_recheck.get("sha256")):
        errors.append("source_recheck.sha256 must be a 64-character digest")
    artifact_path = _resolve_repo_path(source_recheck.get("path"))
    if artifact_path is None:
        errors.append("source_recheck.path is required")
    elif not artifact_path.exists():
        errors.append(f"source_recheck.path does not exist: {source_recheck.get('path')}")
    elif source_recheck.get("sha256") and _sha256_file(artifact_path) != source_recheck.get("sha256"):
        errors.append("source_recheck.sha256 does not match source_recheck.path")
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        errors.append("source_recheck.evidence_boundary must preserve non-approval boundary")
    return errors


def _source_lineage(
    *,
    data_path: Path,
    capture_manifest_path: Path | None,
    seed_source_review_report: Path | None,
    seed_import_manifest: Path | None,
    seed_export_import_manifest: dict[str, Any] | None,
    capture_manifest_review: dict[str, Any],
    require_capture_preflight: bool,
) -> dict[str, Any]:
    seed_source_gate = (
        capture_manifest_review.get("seed_source_review")
        if isinstance(capture_manifest_review.get("seed_source_review"), dict)
        else {}
    )
    seed_import_gate = (
        capture_manifest_review.get("seed_import_manifest")
        if isinstance(capture_manifest_review.get("seed_import_manifest"), dict)
        else {}
    )
    seed_source_required = bool(seed_source_gate.get("required"))
    seed_import_required = bool(seed_import_gate.get("required"))
    return {
        "dataset_yaml": {
            "file": _file_lineage(data_path, required=True),
        },
        "capture_manifest": {
            "file": _file_lineage(
                capture_manifest_path,
                required=bool(require_capture_preflight or capture_manifest_path),
            ),
            "manifest_sha256": capture_manifest_review.get("manifest_sha256"),
            "ok": capture_manifest_review.get("ok"),
            "mode": capture_manifest_review.get("mode"),
        },
        "seed_source_review": {
            "file": _file_lineage(seed_source_review_report, required=seed_source_required),
            "gate": seed_source_gate,
            "source_recheck": _source_recheck_lineage(seed_source_review_report),
        },
        "seed_import_manifest": {
            "file": _file_lineage(seed_import_manifest, required=seed_import_required),
            "gate": seed_import_gate,
        },
        "seed_export_import_manifest": seed_export_import_manifest
        or {
            "required": False,
            "checked": False,
            "valid": None,
        },
    }


def _matrix_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _label_review_import_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".label_review_import.json")


def _seed_export_import_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".seed_export_import.json")


def _dataset_metadata(dataset_doc: dict[str, Any]) -> dict[str, Any]:
    metadata = dataset_doc.get("rakshak_lens")
    return metadata if isinstance(metadata, dict) else {}


def _resolve_dataset_source_manifest(dataset_doc: dict[str, Any], data_path: Path) -> Path | None:
    source_manifest = _dataset_metadata(dataset_doc).get("source_manifest")
    if not source_manifest:
        return None
    source_manifest_path = Path(str(source_manifest))
    if not source_manifest_path.is_absolute():
        source_manifest_path = data_path.parent / source_manifest_path
    return source_manifest_path.resolve(strict=False)


def validate_dataset_provenance(
    *,
    dataset_doc: dict[str, Any],
    data_path: Path,
    capture_manifest_path: Path | None = None,
    require_capture_preflight: bool = False,
) -> dict[str, Any]:
    metadata = _dataset_metadata(dataset_doc)
    source_manifest_path = _resolve_dataset_source_manifest(dataset_doc, data_path)
    declared_source_manifest_sha256 = str(metadata.get("source_manifest_sha256") or "")
    permission = str(metadata.get("permission") or "")
    missing_policy = str(metadata.get("missing_ppe_label_policy") or "")
    result: dict[str, Any] = {
        "required": require_capture_preflight,
        "checked": bool(metadata),
        "source_manifest": str(source_manifest_path) if source_manifest_path else None,
        "declared_source_manifest_sha256": declared_source_manifest_sha256 or None,
        "source_manifest_sha256": None,
        "permission": permission or None,
        "permission_allowed": (permission in ALLOWED_PERMISSIONS) if permission else None,
        "missing_ppe_label_policy": missing_policy or None,
        "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "errors": [],
    }
    errors: list[str] = []

    if source_manifest_path and source_manifest_path.exists():
        result["source_manifest_sha256"] = _sha256_file(source_manifest_path)

    if not require_capture_preflight:
        result["errors"] = errors
        return result

    if not metadata:
        errors.append("rakshak_lens metadata block is required when capture preflight is required")
    if source_manifest_path is None:
        errors.append("rakshak_lens.source_manifest is required when capture preflight is required")
    elif not source_manifest_path.exists():
        errors.append(f"rakshak_lens.source_manifest does not exist: {source_manifest_path}")
    if not declared_source_manifest_sha256:
        errors.append("rakshak_lens.source_manifest_sha256 is required when capture preflight is required")
    elif result["source_manifest_sha256"] and declared_source_manifest_sha256 != result["source_manifest_sha256"]:
        errors.append("rakshak_lens.source_manifest_sha256 does not match source_manifest file")
    if capture_manifest_path and source_manifest_path:
        expected_manifest_path = capture_manifest_path.resolve(strict=False)
        if source_manifest_path != expected_manifest_path:
            errors.append("rakshak_lens.source_manifest must match --capture-manifest")
        expected_manifest_sha256 = _sha256_file(expected_manifest_path) if expected_manifest_path.exists() else None
        if (
            declared_source_manifest_sha256
            and expected_manifest_sha256
            and declared_source_manifest_sha256 != expected_manifest_sha256
        ):
            errors.append("rakshak_lens.source_manifest_sha256 must match --capture-manifest")
    if permission not in ALLOWED_PERMISSIONS:
        errors.append(f"rakshak_lens.permission is not cleared for commercial training: {permission or 'missing'}")
    if missing_policy != EXPECTED_MISSING_PPE_LABEL_POLICY:
        errors.append(
            "rakshak_lens.missing_ppe_label_policy must be "
            f"{EXPECTED_MISSING_PPE_LABEL_POLICY}"
        )

    result["errors"] = errors
    return result


def validate_capture_matrix_sidecar(
    *,
    matrix_csv_path: Path,
    capture_manifest_path: Path,
    mode: str,
    progress: dict[str, Any],
) -> dict[str, Any]:
    sidecar_path = _matrix_sidecar_path(matrix_csv_path)
    if not sidecar_path.exists():
        raise ValueError(f"capture matrix sidecar missing: {sidecar_path}")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"capture matrix sidecar unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("capture matrix sidecar must be a JSON object")

    errors: list[str] = []
    matrix_sha256 = _sha256_file(matrix_csv_path)
    manifest_sha256 = _sha256_file(capture_manifest_path)
    if payload.get("kind") != "apron_harness_capture_matrix_manifest":
        errors.append("kind must be apron_harness_capture_matrix_manifest")
    if payload.get("mode") != mode:
        errors.append(f"mode {payload.get('mode')} does not match {mode}")
    if payload.get("matrix_csv_sha256") != matrix_sha256:
        errors.append("matrix_csv_sha256 does not match current capture matrix")
    if payload.get("source_manifest_sha256") != manifest_sha256:
        errors.append("source_manifest_sha256 does not match current capture manifest")
    try:
        sidecar_rows = int(payload.get("row_count"))
    except (TypeError, ValueError):
        sidecar_rows = -1
    if sidecar_rows != int(progress.get("row_count") or 0):
        errors.append(f"row_count {sidecar_rows} does not match progress {progress.get('row_count', 0)}")

    progress_payload = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    if not progress_payload:
        errors.append("progress must be present in capture matrix sidecar")
    else:
        if "gate_passed" not in progress_payload:
            errors.append("progress.gate_passed is required")
        elif bool(progress_payload.get("gate_passed")) != (progress.get("gate_passed") is True):
            errors.append(
                f"progress.gate_passed {progress_payload.get('gate_passed')} does not match current progress "
                f"{progress.get('gate_passed') is True}"
            )
        for key in [
            "row_count",
            "ready_rows",
            "target_labeled_examples",
            "captured_examples",
            "labeled_examples",
            "missing_labeled_examples",
            "unapproved_rows",
            "unsafe_storage_rows",
        ]:
            if _sidecar_int(progress_payload.get(key)) != _sidecar_int(progress.get(key)):
                errors.append(
                    f"progress.{key} {progress_payload.get(key)} does not match current progress "
                    f"{progress.get(key, 0)}"
                )

    batches = payload.get("next_capture_batches") if isinstance(payload.get("next_capture_batches"), list) else []
    if not batches:
        if _sidecar_int(progress.get("row_count")) > 0:
            errors.append("next_capture_batches must be present in capture matrix sidecar")
    else:
        batch_row_count = sum(_sidecar_int(batch.get("row_count")) for batch in batches if isinstance(batch, dict))
        batch_target = sum(
            _sidecar_int(batch.get("target_labeled_examples")) for batch in batches if isinstance(batch, dict)
        )
        batch_labeled = sum(_sidecar_int(batch.get("labeled_examples")) for batch in batches if isinstance(batch, dict))
        batch_missing = sum(
            _sidecar_int(batch.get("missing_labeled_examples")) for batch in batches if isinstance(batch, dict)
        )
        if batch_row_count != _sidecar_int(progress.get("row_count")):
            errors.append(
                f"next_capture_batches.row_count {batch_row_count} does not match current progress "
                f"{progress.get('row_count', 0)}"
            )
        if batch_target != _sidecar_int(progress.get("target_labeled_examples")):
            errors.append(
                f"next_capture_batches.target_labeled_examples {batch_target} does not match current progress "
                f"{progress.get('target_labeled_examples', 0)}"
            )
        if batch_labeled != _sidecar_int(progress.get("labeled_examples")):
            errors.append(
                f"next_capture_batches.labeled_examples {batch_labeled} does not match current progress "
                f"{progress.get('labeled_examples', 0)}"
            )
        if batch_missing != _sidecar_int(progress.get("missing_labeled_examples")):
            errors.append(
                f"next_capture_batches.missing_labeled_examples {batch_missing} does not match current progress "
                f"{progress.get('missing_labeled_examples', 0)}"
            )

    training_gate = payload.get("training_gate") if isinstance(payload.get("training_gate"), dict) else {}
    for gate_key in [
        "requires_all_rows_ready",
        "requires_manifest_reconciliation",
        "requires_non_repo_raw_storage_refs",
        "requires_permission_approved",
    ]:
        if training_gate.get(gate_key) is not True:
            errors.append(f"training_gate.{gate_key} must be true")

    if errors:
        raise ValueError(f"capture matrix sidecar invalid: {'; '.join(errors)}")

    return {
        "path": str(sidecar_path),
        "exists": True,
        "checked": True,
        "valid": True,
        "sha256": _sha256_file(sidecar_path),
        "mode": mode,
        "row_count": sidecar_rows,
        "matrix_csv_sha256": matrix_sha256,
        "source_manifest_sha256": manifest_sha256,
    }


def validate_label_review_import_sidecar(
    *,
    capture_manifest_path: Path,
    label_review_csv: Path | None = None,
) -> dict[str, Any]:
    sidecar_path = _label_review_import_sidecar_path(capture_manifest_path)
    if not sidecar_path.exists():
        raise ValueError(f"label review import sidecar missing: {sidecar_path}")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"label review import sidecar unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("label review import sidecar must be a JSON object")

    errors: list[str] = []
    manifest_doc = yaml.safe_load(capture_manifest_path.read_text(encoding="utf-8")) or {}
    manifest_sha256 = _sha256_file(capture_manifest_path)
    if payload.get("kind") != "apron_harness_label_review_import_manifest":
        errors.append("kind must be apron_harness_label_review_import_manifest")
    if payload.get("valid") is not True:
        errors.append("valid must be true")
    if payload.get("updated_manifest_sha256") != manifest_sha256:
        errors.append("updated_manifest_sha256 does not match current capture manifest")
    if not payload.get("source_manifest_sha256"):
        errors.append("source_manifest_sha256 is required")
    if not payload.get("label_review_csv_sha256"):
        errors.append("label_review_csv_sha256 is required")
    if label_review_csv is not None:
        if payload.get("label_review_csv") != str(label_review_csv):
            errors.append("label_review_csv does not match training input")
        if label_review_csv.exists() and payload.get("label_review_csv_sha256") != _sha256_file(label_review_csv):
            errors.append("label_review_csv_sha256 does not match training input")
    training_gate = payload.get("training_gate") if isinstance(payload.get("training_gate"), dict) else {}
    if training_gate.get("requires_source_manifest_sha256_match") is True:
        source_manifest = str(payload.get("source_manifest") or "").strip()
        source_sha = str(payload.get("source_manifest_sha256") or "").strip()
        if source_manifest:
            source_path = Path(source_manifest)
            if not source_path.is_absolute():
                source_path = capture_manifest_path.parent / source_path
            if source_path.exists() and _sha256_file(source_path) != source_sha:
                errors.append("source_manifest_sha256 does not match source_manifest")
    taxonomy = payload.get("taxonomy") if isinstance(payload.get("taxonomy"), dict) else {}
    if training_gate.get("requires_taxonomy_version_match") is True:
        if taxonomy.get("version") != TAXONOMY_VERSION:
            errors.append(f"taxonomy.version must be {TAXONOMY_VERSION}")
        if taxonomy.get("label_format") != YOLO_LABEL_FORMAT:
            errors.append(f"taxonomy.label_format must be {YOLO_LABEL_FORMAT}")
    try:
        imported_label_count = int(payload.get("imported_label_count") or 0)
    except (TypeError, ValueError):
        imported_label_count = 0
    try:
        invalid_approved_label_count = int(payload.get("invalid_approved_label_count") or 0)
    except (TypeError, ValueError):
        invalid_approved_label_count = -1
    try:
        imported_clip_count = int(payload.get("imported_clip_count"))
    except (TypeError, ValueError):
        imported_clip_count = -1
    try:
        invalid_clip_metadata_count = int(payload.get("invalid_clip_metadata_count"))
    except (TypeError, ValueError):
        invalid_clip_metadata_count = -1
    if imported_label_count <= 0:
        errors.append("imported_label_count must be greater than 0")
    if invalid_approved_label_count != 0:
        errors.append("invalid_approved_label_count must be 0")
    if "imported_clip_count" not in payload:
        errors.append("imported_clip_count is required")
    elif imported_clip_count < 0:
        errors.append("imported_clip_count must be greater than or equal to 0")
    if "invalid_clip_metadata_count" not in payload:
        errors.append("invalid_clip_metadata_count is required")
    elif invalid_clip_metadata_count != 0:
        errors.append("invalid_clip_metadata_count must be 0")

    counts_doc = (manifest_doc.get("counts") or {}).get("labeled_images_per_class") or {}
    sidecar_counts = (
        payload.get("updated_labeled_images_per_class")
        if isinstance(payload.get("updated_labeled_images_per_class"), dict)
        else {}
    )
    for class_name in REQUIRED_CLASSES.values():
        if int(sidecar_counts.get(class_name) or 0) != int(counts_doc.get(class_name) or 0):
            errors.append(f"updated_labeled_images_per_class.{class_name} does not match manifest count")

    yolo_labels = manifest_doc.get("yolo_labels") if isinstance(manifest_doc.get("yolo_labels"), list) else []
    try:
        merged_label_count = int(payload.get("merged_label_count") or 0)
    except (TypeError, ValueError):
        merged_label_count = -1
    if merged_label_count != len(yolo_labels):
        errors.append(f"merged_label_count {merged_label_count} does not match yolo_labels {len(yolo_labels)}")

    updated_manifest_validation = (
        payload.get("updated_manifest_validation")
        if isinstance(payload.get("updated_manifest_validation"), dict)
        else {}
    )
    if not updated_manifest_validation:
        errors.append("updated_manifest_validation is required")
    else:
        if updated_manifest_validation.get("checked") is not True:
            errors.append("updated_manifest_validation.checked must be true")
        if updated_manifest_validation.get("ok") is not True:
            errors.append("updated_manifest_validation.ok must be true")
        if updated_manifest_validation.get("schema_only") is True:
            errors.append("updated_manifest_validation.schema_only must be false")
        if updated_manifest_validation.get("mode") != "production":
            errors.append("updated_manifest_validation.mode must be production")
        if updated_manifest_validation.get("manifest_sha256") != manifest_sha256:
            errors.append("updated_manifest_validation.manifest_sha256 does not match current capture manifest")

    training_gate = payload.get("training_gate") if isinstance(payload.get("training_gate"), dict) else {}
    for gate_key in [
        "requires_approved_label_review_rows",
        "requires_review_metadata",
        "requires_cleared_permission",
        "requires_non_repo_raw_storage_refs",
        "requires_recomputed_label_counts",
        "requires_updated_manifest_validation",
        "requires_reviewed_clip_metadata",
    ]:
        if training_gate.get(gate_key) is not True:
            errors.append(f"training_gate.{gate_key} must be true")

    if errors:
        raise ValueError(f"label review import sidecar invalid: {'; '.join(errors)}")

    return {
        "path": str(sidecar_path),
        "exists": True,
        "checked": True,
        "valid": True,
        "sha256": _sha256_file(sidecar_path),
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "label_review_csv_sha256": payload.get("label_review_csv_sha256"),
        "updated_manifest_sha256": manifest_sha256,
        "imported_label_count": imported_label_count,
        "imported_clip_count": imported_clip_count,
        "invalid_clip_metadata_count": invalid_clip_metadata_count,
        "merged_label_count": merged_label_count,
        "updated_labeled_images_per_class": sidecar_counts,
        "updated_manifest_validation": updated_manifest_validation,
        "training_gate": training_gate,
    }


def _positive_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _validate_seed_export_preflight_item(item: dict[str, Any], *, index: int, errors: list[str]) -> None:
    prefix = f"imports[{index}].yolo_export_preflight"
    preflight = item.get("yolo_export_preflight")
    if not isinstance(preflight, dict):
        errors.append(f"{prefix} is required")
        return
    if preflight.get("checked") is not True:
        errors.append(f"{prefix}.checked must be true")
    if preflight.get("errors") not in ([], None):
        errors.append(f"{prefix}.errors must be empty")
    preflight_sha256 = str(preflight.get("sha256") or "").strip()
    raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
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
        errors.append(f"imports[{index}].capability must be one of {sorted(REQUIRED_SEED_IMPORT_CLASSES)}")
        return
    counts = preflight.get("label_file_count_by_local_class")
    if not isinstance(counts, dict):
        errors.append(f"{prefix}.label_file_count_by_local_class is required")
        return
    for class_name in sorted(required_classes):
        if _positive_int(counts.get(class_name)) <= 0:
            errors.append(f"{prefix}.label_file_count_by_local_class.{class_name} must be greater than 0")


def validate_seed_export_import_sidecar(
    *,
    capture_manifest_path: Path,
    seed_import_manifest: Path | None = None,
    seed_source_review_report: Path | None = None,
) -> dict[str, Any]:
    sidecar_path = _seed_export_import_sidecar_path(capture_manifest_path)
    if not sidecar_path.exists():
        raise ValueError(f"seed export import sidecar missing: {sidecar_path}")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"seed export import sidecar unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("seed export import sidecar must be a JSON object")

    errors: list[str] = []
    manifest_doc = yaml.safe_load(capture_manifest_path.read_text(encoding="utf-8")) or {}
    manifest_sha256 = _sha256_file(capture_manifest_path)
    if payload.get("kind") != "apron_harness_seed_export_import_manifest":
        errors.append("kind must be apron_harness_seed_export_import_manifest")
    if payload.get("valid") is not True:
        errors.append("valid must be true")
    if payload.get("partial_materialization") is True:
        errors.append("partial_materialization must be false")
    if payload.get("updated_manifest_sha256") != manifest_sha256:
        errors.append("updated_manifest_sha256 does not match current capture manifest")
    if seed_import_manifest is not None:
        if payload.get("seed_import_manifest") != str(seed_import_manifest):
            errors.append("seed_import_manifest does not match training input")
        if seed_import_manifest.exists() and payload.get("seed_import_manifest_sha256") != _sha256_file(seed_import_manifest):
            errors.append("seed_import_manifest_sha256 does not match training input")
    if seed_source_review_report is not None and payload.get("seed_source_review_report") != str(seed_source_review_report):
        errors.append("seed_source_review_report does not match training input")
    if not payload.get("seed_source_review_sha256"):
        errors.append("seed_source_review_sha256 is required")
    source_recheck = (
        payload.get("source_recheck")
        if isinstance(payload.get("source_recheck"), dict)
        else {}
    )
    source_recheck_errors = _validate_source_recheck_lineage(source_recheck)
    errors.extend(source_recheck_errors)
    if seed_source_review_report is not None:
        expected_source_recheck = _source_recheck_lineage(seed_source_review_report)
        if expected_source_recheck:
            if source_recheck.get("path") != expected_source_recheck.get("path"):
                errors.append("source_recheck.path does not match seed source review")
            if source_recheck.get("sha256") != expected_source_recheck.get("sha256"):
                errors.append("source_recheck.sha256 does not match seed source review")
    if not payload.get("seed_import_manifest_sha256"):
        errors.append("seed_import_manifest_sha256 is required")

    try:
        imported_label_count = int(payload.get("imported_label_count") or 0)
    except (TypeError, ValueError):
        imported_label_count = 0
    try:
        imported_clip_count = int(payload.get("imported_clip_count") or 0)
    except (TypeError, ValueError):
        imported_clip_count = 0
    try:
        copied_image_count = int(payload.get("copied_image_count") or 0)
    except (TypeError, ValueError):
        copied_image_count = 0
    if imported_label_count <= 0:
        errors.append("imported_label_count must be greater than 0")
    if imported_clip_count <= 0:
        errors.append("imported_clip_count must be greater than 0")
    if copied_image_count <= 0:
        errors.append("copied_image_count must be greater than 0")

    counts_doc = (manifest_doc.get("counts") or {}).get("labeled_images_per_class") or {}
    sidecar_counts = (
        payload.get("updated_labeled_images_per_class")
        if isinstance(payload.get("updated_labeled_images_per_class"), dict)
        else {}
    )
    for class_name in REQUIRED_CLASSES.values():
        if int(sidecar_counts.get(class_name) or 0) != int(counts_doc.get(class_name) or 0):
            errors.append(f"updated_labeled_images_per_class.{class_name} does not match manifest count")

    imports = payload.get("imports") if isinstance(payload.get("imports"), list) else []
    if not imports:
        errors.append("imports must include at least one materialized seed import")
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            errors.append(f"imports[{index}] must be an object")
            continue
        if int(item.get("imported_label_count") or 0) <= 0:
            errors.append(f"imports[{index}].imported_label_count must be greater than 0")
        if int(item.get("copied_image_count") or 0) <= 0:
            errors.append(f"imports[{index}].copied_image_count must be greater than 0")
        if item.get("errors") not in ([], None):
            errors.append(f"imports[{index}].errors must be empty")
        if not item.get("raw_export_sha256"):
            errors.append(f"imports[{index}].raw_export_sha256 is required")
        _validate_seed_export_preflight_item(item, index=index, errors=errors)

    updated_manifest_validation = (
        payload.get("updated_manifest_validation")
        if isinstance(payload.get("updated_manifest_validation"), dict)
        else {}
    )
    if not updated_manifest_validation:
        errors.append("updated_manifest_validation is required")
    else:
        if updated_manifest_validation.get("checked") is not True:
            errors.append("updated_manifest_validation.checked must be true")
        if updated_manifest_validation.get("ok") is not True:
            errors.append("updated_manifest_validation.ok must be true")
        if updated_manifest_validation.get("schema_only") is True:
            errors.append("updated_manifest_validation.schema_only must be false")
        if updated_manifest_validation.get("mode") != "production":
            errors.append("updated_manifest_validation.mode must be production")
        if updated_manifest_validation.get("manifest_sha256") != manifest_sha256:
            errors.append("updated_manifest_validation.manifest_sha256 does not match current capture manifest")

    training_gate = payload.get("training_gate") if isinstance(payload.get("training_gate"), dict) else {}
    for gate_key in [
        "requires_seed_source_review_gate",
        "requires_seed_import_manifest_gate",
        "requires_raw_export_local_sha256_match",
        "requires_yolo_export_preflight",
        "requires_class_mapping_to_local_taxonomy",
        "requires_review_metadata",
        "requires_updated_manifest_validation",
    ]:
        if training_gate.get(gate_key) is not True:
            errors.append(f"training_gate.{gate_key} must be true")

    if errors:
        raise ValueError(f"seed export import sidecar invalid: {'; '.join(errors)}")

    return {
        "path": str(sidecar_path),
        "exists": True,
        "checked": True,
        "valid": True,
        "sha256": _sha256_file(sidecar_path),
        "seed_source_review_report": payload.get("seed_source_review_report"),
        "seed_source_review_sha256": payload.get("seed_source_review_sha256"),
        "source_recheck": source_recheck,
        "seed_import_manifest": payload.get("seed_import_manifest"),
        "seed_import_manifest_sha256": payload.get("seed_import_manifest_sha256"),
        "updated_manifest_sha256": manifest_sha256,
        "imported_label_count": imported_label_count,
        "imported_clip_count": imported_clip_count,
        "copied_image_count": copied_image_count,
        "partial_materialization": payload.get("partial_materialization") is True,
        "updated_labeled_images_per_class": sidecar_counts,
        "updated_manifest_validation": updated_manifest_validation,
        "imports": imports,
    }


def normalize_model_name(model: str) -> str:
    model = str(model).strip()
    if not model:
        raise ValueError("model is required")
    if "/" in model or "\\" in model:
        raise ValueError("model must be a known nano/small checkpoint name, not a path")
    if not model.endswith(".pt"):
        model = f"{model}.pt"
    if model not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {sorted(ALLOWED_MODELS)}")
    return model


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "installed": False,
            "error": str(exc),
            "mps_built": False,
            "mps_available": False,
            "mps_probe_ok": False,
            "mps_runtime_error": str(exc),
            "cuda_available": False,
        }

    try:
        mps_built = bool(torch.backends.mps.is_built())
    except Exception:
        mps_built = False
    try:
        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        mps_available = False
    mps_probe_ok = False
    mps_runtime_error = None
    if mps_built:
        try:
            torch.ones(1, device="mps")
            mps_probe_ok = True
        except Exception as exc:
            mps_runtime_error = str(exc)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    return {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_probe_ok": mps_probe_ok,
        "mps_runtime_error": mps_runtime_error,
        "cuda_available": cuda_available,
    }


def _metric_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _metric_from_row(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _metric_float(row.get(key))
        if value is not None:
            return value
    return None


def _empty_per_class_metrics(classes: dict[int, str]) -> dict[str, dict[str, float | None]]:
    return {
        class_name: {
            "mAP50": None,
            "mAP50_95": None,
            "precision": None,
            "recall": None,
        }
        for class_name in classes.values()
    }


def extract_per_class_metrics(metrics: Any, classes: dict[int, str] = REQUIRED_CLASSES) -> dict[str, dict[str, float | None]]:
    """Extract per-class detection metrics from Ultralytics validation results."""
    per_class = _empty_per_class_metrics(classes)
    try:
        rows = metrics.summary(normalize=True, decimals=6)
    except Exception:
        rows = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            class_name = str(row.get("Class") or row.get("class") or row.get("name") or "")
            if class_name not in per_class:
                continue
            per_class[class_name] = {
                "mAP50": _metric_from_row(row, ["mAP50", "map50"]),
                "mAP50_95": _metric_from_row(row, ["mAP50-95", "mAP50_95", "map", "map50_95"]),
                "precision": _metric_from_row(row, ["Box-P", "precision", "P"]),
                "recall": _metric_from_row(row, ["Box-R", "recall", "R"]),
            }

    missing = [class_name for class_name, item in per_class.items() if item["mAP50"] is None or item["recall"] is None]
    if not missing:
        return per_class

    names = getattr(metrics, "names", {}) or {}
    ap_class_index = list(getattr(metrics, "ap_class_index", []) or [])
    box = getattr(metrics, "box", None)
    for position, class_index in enumerate(ap_class_index):
        class_name = str(names.get(int(class_index), "")) if isinstance(names, dict) else ""
        if class_name not in per_class or class_name not in missing:
            continue
        try:
            precision, recall, map50, map50_95 = box.class_result(position)
        except Exception:
            continue
        per_class[class_name] = {
            "mAP50": _metric_float(map50),
            "mAP50_95": _metric_float(map50_95),
            "precision": _metric_float(precision),
            "recall": _metric_float(recall),
        }
    return per_class


def select_device(requested: str, torch_info: dict[str, Any] | None = None) -> str:
    requested = str(requested or "auto").lower()
    if requested != "auto":
        return requested
    torch_info = torch_info if torch_info is not None else torch_status()
    mps_ready = bool(torch_info.get("mps_available")) and torch_info.get("mps_probe_ok", True) is not False
    if mps_ready:
        return "mps"
    if torch_info.get("cuda_available"):
        return "cuda"
    return "cpu"


def load_dataset_yaml(data_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        raise ValueError(f"dataset YAML not found: {data_path}")
    doc = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = doc.get("names")
    normalized: dict[int, str] = {}
    if isinstance(names, dict):
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, list):
        normalized = {index: str(value) for index, value in enumerate(names)}
    if normalized != REQUIRED_CLASSES:
        raise ValueError(f"dataset names must exactly match {REQUIRED_CLASSES}")
    for split in ["train", "val"]:
        if not doc.get(split):
            raise ValueError(f"dataset YAML must define {split}")
    return doc


def build_training_plan(
    *,
    data_path: Path,
    model: str,
    device: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: Path,
    name: str,
    export_formats: list[str],
    capture_manifest_path: Path | None = None,
    capture_matrix_csv_path: Path | None = None,
    seed_source_review_report: Path | None = DEFAULT_SEED_SOURCE_REVIEW,
    seed_import_manifest: Path | None = DEFAULT_SEED_IMPORT_MANIFEST,
    capture_preflight_mode: str = "production",
    require_capture_preflight: bool = False,
) -> dict[str, Any]:
    model_name = normalize_model_name(model)
    dataset_doc = load_dataset_yaml(data_path)
    capture_preflight_mode = str(capture_preflight_mode or "production")
    if capture_preflight_mode not in MIN_COUNTS:
        raise ValueError(f"capture_preflight_mode must be one of {sorted(MIN_COUNTS)}")
    capture_preflight: dict[str, Any] = {
        "mode": capture_preflight_mode,
        "required": require_capture_preflight,
        "checked": False,
        "gate_passed": None,
    }
    seed_export_import_manifest: dict[str, Any] | None = None
    capture_manifest_review: dict[str, Any] = {
        "required": bool(capture_manifest_path),
        "checked": False,
        "ok": None,
        "seed_source_review": {},
        "errors": [],
        "warnings": [],
    }
    if capture_manifest_path or capture_matrix_csv_path:
        if capture_manifest_path is None or capture_matrix_csv_path is None:
            raise ValueError("capture preflight requires both --capture-manifest and --capture-matrix-csv")
        capture_manifest_review = validate_manifest(
            capture_manifest_path,
            mode=capture_preflight_mode,
            schema_only=True,
            seed_source_review_report=seed_source_review_report,
            seed_import_manifest=seed_import_manifest,
        )
        capture_manifest_review["required"] = True
        capture_manifest_review["checked"] = True
        if capture_manifest_review.get("ok") is not True:
            raise ValueError(
                "capture manifest review failed: "
                + "; ".join(str(error) for error in capture_manifest_review.get("errors") or [])
            )
        seed_source_gate = (
            capture_manifest_review.get("seed_source_review")
            if isinstance(capture_manifest_review.get("seed_source_review"), dict)
            else {}
        )
        if seed_source_gate.get("required") is True:
            source_recheck_errors = _validate_source_recheck_lineage(
                _source_recheck_lineage(seed_source_review_report)
            )
            if source_recheck_errors:
                raise ValueError(
                    "source recheck lineage invalid: "
                    + "; ".join(source_recheck_errors)
                )
        capture_preflight = validate_capture_matrix_progress(
            capture_matrix_csv_path,
            manifest_path=capture_manifest_path,
            mode=capture_preflight_mode,
        )
        capture_preflight["mode"] = capture_preflight_mode
        capture_preflight["required"] = require_capture_preflight
        capture_preflight["checked"] = True
        if capture_preflight.get("gate_passed") is not True:
            raise ValueError("capture preflight gate failed; complete and approve the capture matrix before training")
        if require_capture_preflight and capture_preflight_mode == "production":
            capture_preflight["capture_matrix_manifest"] = validate_capture_matrix_sidecar(
                matrix_csv_path=capture_matrix_csv_path,
                capture_manifest_path=capture_manifest_path,
                mode=capture_preflight_mode,
                progress=capture_preflight,
            )
            capture_preflight["label_review_import_manifest"] = validate_label_review_import_sidecar(
                capture_manifest_path=capture_manifest_path,
            )
            seed_import_gate = (
                capture_manifest_review.get("seed_import_manifest")
                if isinstance(capture_manifest_review.get("seed_import_manifest"), dict)
                else {}
            )
            if seed_import_gate.get("required") is True:
                seed_export_import_manifest = validate_seed_export_import_sidecar(
                    capture_manifest_path=capture_manifest_path,
                    seed_import_manifest=seed_import_manifest,
                    seed_source_review_report=seed_source_review_report,
                )
                capture_preflight["seed_export_import_manifest"] = seed_export_import_manifest
    elif require_capture_preflight:
        raise ValueError("capture preflight is required; pass --capture-manifest and --capture-matrix-csv")

    dataset_provenance = validate_dataset_provenance(
        dataset_doc=dataset_doc,
        data_path=data_path,
        capture_manifest_path=capture_manifest_path,
        require_capture_preflight=require_capture_preflight,
    )
    if dataset_provenance["errors"]:
        raise ValueError(f"dataset provenance invalid: {'; '.join(dataset_provenance['errors'])}")

    torch_info = torch_status()
    selected_device = select_device(device, torch_info)
    return {
        "generated_at": utc_now(),
        "status": "ready_to_train",
        "model": model_name,
        "dataset_yaml": str(data_path),
        "dataset_root": dataset_doc.get("path"),
        "dataset_provenance": dataset_provenance,
        "source_lineage": _source_lineage(
            data_path=data_path,
            capture_manifest_path=capture_manifest_path,
            seed_source_review_report=seed_source_review_report,
            seed_import_manifest=seed_import_manifest,
            seed_export_import_manifest=seed_export_import_manifest,
            capture_manifest_review=capture_manifest_review,
            require_capture_preflight=require_capture_preflight,
        ),
        "classes": REQUIRED_CLASSES,
        "capture_manifest_review": capture_manifest_review,
        "capture_preflight": capture_preflight,
        "torch": torch_info,
        "selected_device": selected_device,
        "train_args": {
            "data": str(data_path),
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": selected_device,
            "project": str(project),
            "name": name,
            "plots": True,
        },
        "export_formats": export_formats,
        "production_notes": [
            "YOLOE is not accepted for this closed-set production training path.",
            "MPS or CPU local training proves model-building only; Jetson throughput still requires staging benchmarks.",
            "Missing-PPE labels are derived at runtime from person-to-visible-PPE association, not trained as classes.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_training(plan: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    train_args = dict(plan["train_args"])
    model = YOLO(plan["model"])
    started = time.time()
    results = model.train(**{key: value for key, value in train_args.items() if key != "model"})
    elapsed = time.time() - started
    metrics = model.val(data=train_args["data"], device=train_args["device"])
    per_class_metrics = extract_per_class_metrics(metrics, plan["classes"])
    exports = []
    for export_format in plan["export_formats"]:
        exports.append(str(model.export(format=export_format)))
    return {
        **plan,
        "status": "trained",
        "training_time_seconds": round(elapsed, 1),
        "result_save_dir": str(getattr(results, "save_dir", "")),
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4) if hasattr(metrics, "box") else None,
            "mAP50_95": round(float(metrics.box.map), 4) if hasattr(metrics, "box") else None,
            "precision": round(float(metrics.box.mp), 4) if hasattr(metrics, "box") else None,
            "recall": round(float(metrics.box.mr), 4) if hasattr(metrics, "box") else None,
        },
        "per_class_metrics": per_class_metrics,
        "exports": exports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run apron/harness closed-set YOLO training.")
    parser.add_argument("--data", required=True, help="Validated YOLO dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Allowed nano/small checkpoint")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="apron_harness_closed_set")
    parser.add_argument("--export-format", action="append", default=[], help="Export format such as onnx or engine")
    parser.add_argument("--capture-manifest", default="", help="Cleared capture manifest for training preflight")
    parser.add_argument("--capture-matrix-csv", default="", help="Filled capture matrix CSV for training preflight")
    parser.add_argument(
        "--seed-source-review-report",
        default=str(DEFAULT_SEED_SOURCE_REVIEW),
        help="Saved apron_harness_seed_source_review.json required for public/commercial seed clips",
    )
    parser.add_argument(
        "--seed-import-manifest",
        default=str(DEFAULT_SEED_IMPORT_MANIFEST),
        help="Filled apron_harness_seed_import_manifest YAML required for public/commercial seed clips",
    )
    parser.add_argument(
        "--capture-preflight-mode",
        choices=sorted(MIN_COUNTS),
        default="production",
        help="Capture gate depth to enforce before training. Production promotion requires production.",
    )
    parser.add_argument(
        "--require-capture-preflight",
        action="store_true",
        help="Fail unless capture manifest and matrix pass before training plan creation",
    )
    parser.add_argument("--out-plan", default="", help="Optional JSON path for dry-run plan or training result")
    parser.add_argument("--execute", action="store_true", help="Actually run training. Default is plan-only.")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_training_plan(
            data_path=Path(args.data),
            model=args.model,
            device=args.device,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=Path(args.project),
            name=args.name,
            export_formats=args.export_format or DEFAULT_EXPORT_FORMATS,
            capture_manifest_path=Path(args.capture_manifest) if args.capture_manifest else None,
            capture_matrix_csv_path=Path(args.capture_matrix_csv) if args.capture_matrix_csv else None,
            seed_source_review_report=Path(args.seed_source_review_report)
            if args.seed_source_review_report
            else None,
            seed_import_manifest=Path(args.seed_import_manifest)
            if args.seed_import_manifest
            else None,
            capture_preflight_mode=args.capture_preflight_mode,
            require_capture_preflight=bool(args.require_capture_preflight or args.execute),
        )
        output = execute_training(plan) if args.execute else plan
    except Exception as exc:
        output = {"status": "failed", "error": str(exc)}
        if args.out_plan:
            write_json(Path(args.out_plan), output)
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"failed: {exc}")
            if args.out_plan:
                print(f"wrote: {args.out_plan}")
        return 1

    if args.out_plan:
        write_json(Path(args.out_plan), output)
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        mode = "training complete" if args.execute else "dry-run plan ready"
        print(f"{mode}: {output['model']} on {output['dataset_yaml']}")
        print(f"device: {output['selected_device']}")
        if args.out_plan:
            print(f"wrote: {args.out_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
