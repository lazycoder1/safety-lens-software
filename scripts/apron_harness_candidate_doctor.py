#!/usr/bin/env python3
"""Gate trained apron/harness PPE model candidates before runtime promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apron_harness_dataset_doctor import REQUIRED_CLASSES  # noqa: E402
from apron_harness_train import ALLOWED_MODELS, EXPECTED_MISSING_PPE_LABEL_POLICY  # noqa: E402


MIN_PER_CLASS_MAP50 = 0.75
MIN_PER_CLASS_RECALL = 0.85
ACCEPTED_EXPORT_SUFFIXES = {".onnx", ".engine"}
PLANNED_REGISTRY_SUFFIX = ".onnx"
PLANNED_MODEL_KEY = "ppe_closed_set_candidate"
PLANNED_REGISTRY_PATH = "models/ppe_closed_set_candidate/apron-harness-ppe.onnx"
PLANNED_REGISTRY_DIR = "models/ppe_closed_set_candidate"
PLANNED_MODEL_FILENAME = "apron-harness-ppe.onnx"
REQUIRED_SEED_IMPORT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_classes(raw_classes: Any) -> dict[int, str]:
    if not isinstance(raw_classes, dict):
        return {}
    normalized: dict[int, str] = {}
    for key, value in raw_classes.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    try:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_artifact(path: str, result_path: Path, artifact_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if artifact_root is not None:
        return artifact_root / candidate
    return result_path.parent / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_entry(selected_export: dict[str, Any] | None, result_path: Path, model: str) -> dict[str, Any]:
    export_path = str((selected_export or {}).get("path") or "")
    export_sha256 = (selected_export or {}).get("sha256")
    return {
        "model_key": PLANNED_MODEL_KEY,
        "file": PLANNED_MODEL_FILENAME,
        "registry_path": PLANNED_REGISTRY_PATH,
        "expected_input_size": 640,
        "local_device": "mps_or_cpu_fallback",
        "jetson_device": "tensorrt_fp16_candidate",
        "source_training_model": model,
        "source_training_result": str(result_path),
        "source_export_path": export_path,
        "source_export_sha256": export_sha256,
        "activation_status": "do_not_activate_until_side_by_side_runtime_and_jetson_gates_pass",
    }


def _validate_dataset_provenance(
    result: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    provenance = result.get("dataset_provenance")
    if not isinstance(provenance, dict):
        errors.append("dataset_provenance is required from scripts/apron_harness_train.py")
        return {}

    if provenance.get("required") is not True:
        errors.append("dataset_provenance.required must be true before candidate promotion")
    if provenance.get("checked") is not True:
        errors.append("dataset_provenance.checked must be true before candidate promotion")
    if provenance.get("errors") not in ([], None):
        errors.append("dataset_provenance.errors must be empty before candidate promotion")
    if not provenance.get("source_manifest"):
        errors.append("dataset_provenance.source_manifest is required")
    if not provenance.get("declared_source_manifest_sha256"):
        errors.append("dataset_provenance.declared_source_manifest_sha256 is required")
    if not provenance.get("source_manifest_sha256"):
        errors.append("dataset_provenance.source_manifest_sha256 is required")
    if (
        provenance.get("declared_source_manifest_sha256")
        and provenance.get("source_manifest_sha256")
        and provenance.get("declared_source_manifest_sha256") != provenance.get("source_manifest_sha256")
    ):
        errors.append("dataset_provenance.declared_source_manifest_sha256 must match source_manifest_sha256")
    if provenance.get("permission_allowed") is not True:
        errors.append("dataset_provenance.permission_allowed must be true before candidate promotion")
    if provenance.get("missing_ppe_label_policy") != EXPECTED_MISSING_PPE_LABEL_POLICY:
        errors.append(
            "dataset_provenance.missing_ppe_label_policy must be "
            f"{EXPECTED_MISSING_PPE_LABEL_POLICY}"
        )
    if capture_matrix_manifest:
        expected_sha = capture_matrix_manifest.get("source_manifest_sha256")
        if expected_sha and provenance.get("source_manifest_sha256") != expected_sha:
            errors.append("dataset_provenance.source_manifest_sha256 must match capture matrix source_manifest_sha256")
    return provenance


def _lineage_section(
    lineage: dict[str, Any],
    section_name: str,
    *,
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    section = lineage.get(section_name)
    if not isinstance(section, dict):
        errors.append(f"source_lineage.{section_name} is required")
        return {}
    file_info = section.get("file")
    if not isinstance(file_info, dict):
        if required:
            errors.append(f"source_lineage.{section_name}.file is required")
        return section
    if required:
        if file_info.get("required") is not True:
            errors.append(f"source_lineage.{section_name}.file.required must be true")
        if not file_info.get("path"):
            errors.append(f"source_lineage.{section_name}.file.path is required")
        if file_info.get("exists") is not True:
            errors.append(f"source_lineage.{section_name}.file.exists must be true")
        if not file_info.get("sha256"):
            errors.append(f"source_lineage.{section_name}.file.sha256 is required")
    return section


def _validate_seed_source_lineage(section: dict[str, Any], errors: list[str]) -> None:
    gate = section.get("gate") if isinstance(section.get("gate"), dict) else {}
    if gate.get("required") is not True:
        return
    if gate.get("ok") is not True:
        errors.append("source_lineage.seed_source_review.gate.ok must be true")
    if gate.get("gate_passed") is not True:
        errors.append("source_lineage.seed_source_review.gate.gate_passed must be true")
    clip_count = int(gate.get("clip_count") or 0)
    approved_clip_count = int(gate.get("approved_clip_count") or 0)
    if clip_count <= 0:
        errors.append("source_lineage.seed_source_review.gate.clip_count must be greater than 0")
    if approved_clip_count != clip_count:
        errors.append("source_lineage.seed_source_review.gate.approved_clip_count must equal clip_count")
    _validate_source_recheck_block(
        section.get("source_recheck"),
        prefix="source_lineage.seed_source_review",
        errors=errors,
    )


def _positive_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _validate_source_recheck_block(
    source_recheck: Any,
    *,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(source_recheck, dict) or not source_recheck:
        errors.append(f"{prefix}.source_recheck is required")
        return
    if source_recheck.get("exists") is not True:
        errors.append(f"{prefix}.source_recheck.exists must be true")
    if not _valid_sha(source_recheck.get("sha256")):
        errors.append(f"{prefix}.source_recheck.sha256 must be a 64-character digest")
    if not source_recheck.get("path"):
        errors.append(f"{prefix}.source_recheck.path is required")
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        errors.append(f"{prefix}.source_recheck.evidence_boundary must preserve non-approval boundary")


def _validate_label_review_clip_metadata(
    manifest: dict[str, Any],
    *,
    prefix: str,
    errors: list[str],
) -> None:
    try:
        imported_clip_count = int(manifest.get("imported_clip_count"))
    except (TypeError, ValueError):
        imported_clip_count = -1
    try:
        invalid_clip_metadata_count = int(manifest.get("invalid_clip_metadata_count"))
    except (TypeError, ValueError):
        invalid_clip_metadata_count = -1
    if "imported_clip_count" not in manifest:
        errors.append(f"{prefix}.imported_clip_count is required")
    elif imported_clip_count < 0:
        errors.append(f"{prefix}.imported_clip_count must be greater than or equal to 0")
    if "invalid_clip_metadata_count" not in manifest:
        errors.append(f"{prefix}.invalid_clip_metadata_count is required")
    elif invalid_clip_metadata_count != 0:
        errors.append(f"{prefix}.invalid_clip_metadata_count must be 0")
    training_gate = manifest.get("training_gate") if isinstance(manifest.get("training_gate"), dict) else {}
    if training_gate.get("requires_reviewed_clip_metadata") is not True:
        errors.append(f"{prefix}.training_gate.requires_reviewed_clip_metadata must be true")


def _validate_seed_import_export_preflight(
    gate: dict[str, Any],
    errors: list[str],
    prefix: str,
) -> None:
    imports = gate.get("imports")
    if not isinstance(imports, list) or not imports:
        errors.append(f"{prefix}.imports must include approved seed import preflight records")
        return

    included: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            errors.append(f"{prefix}.imports[{index}] must be an object")
            continue
        if item.get("include_in_training") is True:
            included.append((index, item))

    if not included:
        errors.append(f"{prefix}.imports must include at least one include_in_training=true record")
        return

    for index, item in included:
        item_prefix = f"{prefix}.imports[{index}]"
        if item.get("approved_for_training") is not True:
            errors.append(f"{item_prefix}.approved_for_training must be true")
        if item.get("errors") not in ([], None):
            errors.append(f"{item_prefix}.errors must be empty")
        if item.get("blockers") not in ([], None):
            errors.append(f"{item_prefix}.blockers must be empty")

        raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
        if not raw_export_sha256:
            errors.append(f"{item_prefix}.raw_export_sha256 is required")
        if not str(item.get("raw_export_local_path") or "").strip():
            errors.append(f"{item_prefix}.raw_export_local_path is required")

        preflight = item.get("yolo_export_preflight")
        if not isinstance(preflight, dict):
            errors.append(f"{item_prefix}.yolo_export_preflight is required")
            continue
        if preflight.get("checked") is not True:
            errors.append(f"{item_prefix}.yolo_export_preflight.checked must be true")
        preflight_sha256 = str(preflight.get("sha256") or "").strip()
        if not preflight_sha256:
            errors.append(f"{item_prefix}.yolo_export_preflight.sha256 is required")
        elif raw_export_sha256 and preflight_sha256 != raw_export_sha256:
            errors.append(f"{item_prefix}.yolo_export_preflight.sha256 must match raw_export_sha256")

        capability = str(item.get("capability") or "")
        required_classes = REQUIRED_SEED_IMPORT_CLASSES.get(capability, set())
        if not required_classes:
            errors.append(f"{item_prefix}.capability must be one of {sorted(REQUIRED_SEED_IMPORT_CLASSES)}")
            continue
        counts = preflight.get("label_file_count_by_local_class")
        if not isinstance(counts, dict):
            errors.append(f"{item_prefix}.yolo_export_preflight.label_file_count_by_local_class is required")
            continue
        for class_name in sorted(required_classes):
            if _positive_int(counts.get(class_name)) <= 0:
                errors.append(
                    f"{item_prefix}.yolo_export_preflight.label_file_count_by_local_class."
                    f"{class_name} must be greater than 0"
                )


def _validate_seed_export_import_preflight(
    item: dict[str, Any],
    *,
    prefix: str,
    errors: list[str],
) -> None:
    preflight = item.get("yolo_export_preflight")
    if not isinstance(preflight, dict):
        errors.append(f"{prefix}.yolo_export_preflight is required")
        return
    if preflight.get("checked") is not True:
        errors.append(f"{prefix}.yolo_export_preflight.checked must be true")
    if preflight.get("errors") not in ([], None):
        errors.append(f"{prefix}.yolo_export_preflight.errors must be empty")
    raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
    preflight_sha256 = str(preflight.get("sha256") or "").strip()
    if not preflight_sha256:
        errors.append(f"{prefix}.yolo_export_preflight.sha256 is required")
    elif raw_export_sha256 and preflight_sha256 != raw_export_sha256:
        errors.append(f"{prefix}.yolo_export_preflight.sha256 must match raw_export_sha256")
    orphan_counts = (
        preflight.get("orphan_label_count_by_split")
        if isinstance(preflight.get("orphan_label_count_by_split"), dict)
        else {}
    )
    for split, count in orphan_counts.items():
        if _positive_int(count) > 0:
            errors.append(f"{prefix}.yolo_export_preflight.orphan_label_count_by_split.{split} must be 0")
    capability = str(item.get("capability") or "")
    required_classes = REQUIRED_SEED_IMPORT_CLASSES.get(capability, set())
    if not required_classes:
        errors.append(f"{prefix}.capability must be one of {sorted(REQUIRED_SEED_IMPORT_CLASSES)}")
        return
    counts = preflight.get("label_file_count_by_local_class")
    if not isinstance(counts, dict):
        errors.append(f"{prefix}.yolo_export_preflight.label_file_count_by_local_class is required")
        return
    for class_name in sorted(required_classes):
        if _positive_int(counts.get(class_name)) <= 0:
            errors.append(
                f"{prefix}.yolo_export_preflight.label_file_count_by_local_class."
                f"{class_name} must be greater than 0"
            )


def _validate_seed_import_lineage(section: dict[str, Any], errors: list[str]) -> None:
    gate = section.get("gate") if isinstance(section.get("gate"), dict) else {}
    if gate.get("required") is not True:
        return
    if gate.get("ok") is not True:
        errors.append("source_lineage.seed_import_manifest.gate.ok must be true")
    if gate.get("source_review_sha256_matches") is not True:
        errors.append("source_lineage.seed_import_manifest.gate.source_review_sha256_matches must be true")
    clip_count = int(gate.get("clip_count") or 0)
    approved_clip_count = int(gate.get("approved_clip_count") or 0)
    if clip_count <= 0:
        errors.append("source_lineage.seed_import_manifest.gate.clip_count must be greater than 0")
    if approved_clip_count != clip_count:
        errors.append("source_lineage.seed_import_manifest.gate.approved_clip_count must equal clip_count")
    _validate_seed_import_export_preflight(
        gate,
        errors,
        "source_lineage.seed_import_manifest.gate",
    )


def _validate_source_lineage(
    result: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    dataset_provenance: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    lineage = result.get("source_lineage")
    if not isinstance(lineage, dict):
        errors.append("source_lineage is required from scripts/apron_harness_train.py")
        return {}

    _lineage_section(lineage, "dataset_yaml", required=True, errors=errors)
    capture_required = bool(dataset_provenance.get("required"))
    capture_section = _lineage_section(
        lineage,
        "capture_manifest",
        required=capture_required,
        errors=errors,
    )
    expected_manifest_sha = (
        dataset_provenance.get("source_manifest_sha256")
        or capture_matrix_manifest.get("source_manifest_sha256")
    )
    capture_file = capture_section.get("file") if isinstance(capture_section.get("file"), dict) else {}
    capture_file_sha = capture_file.get("sha256")
    capture_manifest_sha = capture_section.get("manifest_sha256")
    if expected_manifest_sha and capture_file_sha and capture_file_sha != expected_manifest_sha:
        errors.append("source_lineage.capture_manifest.file.sha256 must match dataset source_manifest_sha256")
    if expected_manifest_sha and capture_manifest_sha and capture_manifest_sha != expected_manifest_sha:
        errors.append("source_lineage.capture_manifest.manifest_sha256 must match dataset source_manifest_sha256")
    if capture_required and capture_section.get("ok") is not True:
        errors.append("source_lineage.capture_manifest.ok must be true")
    if capture_required and capture_section.get("mode") != "production":
        errors.append("source_lineage.capture_manifest.mode must be production")

    seed_source_section = _lineage_section(
        lineage,
        "seed_source_review",
        required=bool(
            isinstance(lineage.get("seed_source_review"), dict)
            and isinstance(lineage["seed_source_review"].get("gate"), dict)
            and lineage["seed_source_review"]["gate"].get("required") is True
        ),
        errors=errors,
    )
    _validate_seed_source_lineage(seed_source_section, errors)

    seed_import_section = _lineage_section(
        lineage,
        "seed_import_manifest",
        required=bool(
            isinstance(lineage.get("seed_import_manifest"), dict)
            and isinstance(lineage["seed_import_manifest"].get("gate"), dict)
            and lineage["seed_import_manifest"]["gate"].get("required") is True
        ),
        errors=errors,
    )
    _validate_seed_import_lineage(seed_import_section, errors)
    return lineage


def _validate_label_review_import_manifest(
    *,
    capture_preflight: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    manifest = (
        capture_preflight.get("label_review_import_manifest")
        if isinstance(capture_preflight.get("label_review_import_manifest"), dict)
        else {}
    )
    if not manifest:
        errors.append("capture_preflight.label_review_import_manifest is required before candidate promotion")
        return {}
    if manifest.get("valid") is not True:
        errors.append("capture_preflight.label_review_import_manifest.valid must be true before candidate promotion")
    if not manifest.get("source_manifest_sha256"):
        errors.append("capture_preflight.label_review_import_manifest.source_manifest_sha256 is required")
    if not manifest.get("label_review_csv_sha256"):
        errors.append("capture_preflight.label_review_import_manifest.label_review_csv_sha256 is required")
    if not manifest.get("updated_manifest_sha256"):
        errors.append("capture_preflight.label_review_import_manifest.updated_manifest_sha256 is required")
    expected_sha = capture_matrix_manifest.get("source_manifest_sha256") if capture_matrix_manifest else None
    if expected_sha and manifest.get("updated_manifest_sha256") != expected_sha:
        errors.append(
            "capture_preflight.label_review_import_manifest.updated_manifest_sha256 "
            "must match capture matrix source_manifest_sha256"
        )
    validation = (
        manifest.get("updated_manifest_validation")
        if isinstance(manifest.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        errors.append("capture_preflight.label_review_import_manifest.updated_manifest_validation is required")
    else:
        if validation.get("checked") is not True:
            errors.append("capture_preflight.label_review_import_manifest.updated_manifest_validation.checked must be true")
        if validation.get("ok") is not True:
            errors.append("capture_preflight.label_review_import_manifest.updated_manifest_validation.ok must be true")
        if validation.get("schema_only") is True:
            errors.append("capture_preflight.label_review_import_manifest.updated_manifest_validation.schema_only must be false")
        if validation.get("mode") != "production":
            errors.append("capture_preflight.label_review_import_manifest.updated_manifest_validation.mode must be production")
        if expected_sha and validation.get("manifest_sha256") != expected_sha:
            errors.append(
                "capture_preflight.label_review_import_manifest.updated_manifest_validation.manifest_sha256 "
                "must match capture matrix source_manifest_sha256"
            )
    try:
        imported_label_count = int(manifest.get("imported_label_count") or 0)
    except (TypeError, ValueError):
        imported_label_count = 0
    if imported_label_count <= 0:
        errors.append("capture_preflight.label_review_import_manifest.imported_label_count must be greater than 0")
    _validate_label_review_clip_metadata(
        manifest,
        prefix="capture_preflight.label_review_import_manifest",
        errors=errors,
    )
    try:
        merged_label_count = int(manifest.get("merged_label_count") or 0)
    except (TypeError, ValueError):
        merged_label_count = 0
    if merged_label_count <= 0:
        errors.append("capture_preflight.label_review_import_manifest.merged_label_count must be greater than 0")
    counts = manifest.get("updated_labeled_images_per_class")
    if not isinstance(counts, dict):
        errors.append("capture_preflight.label_review_import_manifest.updated_labeled_images_per_class is required")
    else:
        missing_counts = [
            class_name
            for class_name in REQUIRED_CLASSES.values()
            if int(counts.get(class_name) or 0) <= 0
        ]
        if missing_counts:
            errors.append(
                "capture_preflight.label_review_import_manifest.updated_labeled_images_per_class "
                "must include positive counts for: "
                + ", ".join(missing_counts)
            )
    return manifest


def _validate_seed_export_import_manifest(
    *,
    capture_preflight: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    manifest = (
        capture_preflight.get("seed_export_import_manifest")
        if isinstance(capture_preflight.get("seed_export_import_manifest"), dict)
        else {}
    )
    if not required:
        return manifest
    if not manifest:
        errors.append("capture_preflight.seed_export_import_manifest is required before candidate promotion")
        return {}
    if manifest.get("valid") is not True:
        errors.append("capture_preflight.seed_export_import_manifest.valid must be true before candidate promotion")
    if manifest.get("partial_materialization") is True:
        errors.append("capture_preflight.seed_export_import_manifest.partial_materialization must be false")
    if not manifest.get("seed_source_review_sha256"):
        errors.append("capture_preflight.seed_export_import_manifest.seed_source_review_sha256 is required")
    _validate_source_recheck_block(
        manifest.get("source_recheck"),
        prefix="capture_preflight.seed_export_import_manifest",
        errors=errors,
    )
    if not manifest.get("seed_import_manifest_sha256"):
        errors.append("capture_preflight.seed_export_import_manifest.seed_import_manifest_sha256 is required")
    if not manifest.get("updated_manifest_sha256"):
        errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_sha256 is required")
    expected_sha = capture_matrix_manifest.get("source_manifest_sha256") if capture_matrix_manifest else None
    if expected_sha and manifest.get("updated_manifest_sha256") != expected_sha:
        errors.append(
            "capture_preflight.seed_export_import_manifest.updated_manifest_sha256 "
            "must match capture matrix source_manifest_sha256"
        )
    validation = (
        manifest.get("updated_manifest_validation")
        if isinstance(manifest.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_validation is required")
    else:
        if validation.get("checked") is not True:
            errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_validation.checked must be true")
        if validation.get("ok") is not True:
            errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_validation.ok must be true")
        if validation.get("schema_only") is True:
            errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_validation.schema_only must be false")
        if validation.get("mode") != "production":
            errors.append("capture_preflight.seed_export_import_manifest.updated_manifest_validation.mode must be production")
        if expected_sha and validation.get("manifest_sha256") != expected_sha:
            errors.append(
                "capture_preflight.seed_export_import_manifest.updated_manifest_validation.manifest_sha256 "
                "must match capture matrix source_manifest_sha256"
            )
    for field in ["imported_label_count", "imported_clip_count", "copied_image_count"]:
        try:
            value = int(manifest.get(field) or 0)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            errors.append(f"capture_preflight.seed_export_import_manifest.{field} must be greater than 0")

    imports = manifest.get("imports") if isinstance(manifest.get("imports"), list) else []
    if not imports:
        errors.append("capture_preflight.seed_export_import_manifest.imports must include materialized seed imports")
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            errors.append(f"capture_preflight.seed_export_import_manifest.imports[{index}] must be an object")
            continue
        if int(item.get("imported_label_count") or 0) <= 0:
            errors.append(
                f"capture_preflight.seed_export_import_manifest.imports[{index}]."
                "imported_label_count must be greater than 0"
            )
        if int(item.get("copied_image_count") or 0) <= 0:
            errors.append(
                f"capture_preflight.seed_export_import_manifest.imports[{index}]."
                "copied_image_count must be greater than 0"
            )
        if item.get("errors") not in ([], None):
            errors.append(f"capture_preflight.seed_export_import_manifest.imports[{index}].errors must be empty")
        if not item.get("raw_export_sha256"):
            errors.append(f"capture_preflight.seed_export_import_manifest.imports[{index}].raw_export_sha256 is required")
        _validate_seed_export_import_preflight(
            item,
            prefix=f"capture_preflight.seed_export_import_manifest.imports[{index}]",
            errors=errors,
        )

    counts = manifest.get("updated_labeled_images_per_class")
    if not isinstance(counts, dict):
        errors.append("capture_preflight.seed_export_import_manifest.updated_labeled_images_per_class is required")
    else:
        missing_counts = [
            class_name
            for class_name in REQUIRED_CLASSES.values()
            if int(counts.get(class_name) or 0) <= 0
        ]
        if missing_counts:
            errors.append(
                "capture_preflight.seed_export_import_manifest.updated_labeled_images_per_class "
                "must include positive counts for: "
                + ", ".join(missing_counts)
            )
    return manifest


def _model_manager_definition_hint() -> dict[str, Any]:
    return {
        "model_key": PLANNED_MODEL_KEY,
        "display_name": "Closed-Set Apron/Harness PPE",
        "filename": PLANNED_MODEL_FILENAME,
        "local_path": f"{PLANNED_REGISTRY_DIR}/{PLANNED_MODEL_FILENAME}",
        "download_url": "",
        "warmup_behavior": "Closed-set apron/harness PPE detect warmup",
        "shared_asset_key": "apron-harness-ppe",
    }


def validate_candidate(
    result_path: Path,
    artifact_root: Path | None = None,
    min_map50: float = MIN_PER_CLASS_MAP50,
    min_recall: float = MIN_PER_CLASS_RECALL,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    if result.get("status") != "trained":
        errors.append("training result status must be trained")

    model = str(result.get("model") or "")
    if model not in ALLOWED_MODELS:
        errors.append(f"model must be one of {sorted(ALLOWED_MODELS)}")
    if "yoloe" in model.lower():
        errors.append("YOLOE is not accepted for closed-set production promotion")

    capture_preflight = result.get("capture_preflight")
    if not isinstance(capture_preflight, dict):
        capture_preflight = {}
        errors.append("capture_preflight is required from scripts/apron_harness_train.py")
    elif capture_preflight.get("checked") is not True:
        errors.append("capture_preflight.checked must be true before candidate promotion")
    elif capture_preflight.get("gate_passed") is not True:
        errors.append("capture_preflight.gate_passed must be true before candidate promotion")
    if isinstance(capture_preflight, dict) and capture_preflight.get("mode") != "production":
        errors.append("capture_preflight.mode must be production before candidate promotion")
    capture_matrix_manifest = (
        capture_preflight.get("capture_matrix_manifest")
        if isinstance(capture_preflight.get("capture_matrix_manifest"), dict)
        else {}
    )
    if not capture_matrix_manifest:
        errors.append("capture_preflight.capture_matrix_manifest is required before candidate promotion")
    else:
        if capture_matrix_manifest.get("valid") is not True:
            errors.append("capture_preflight.capture_matrix_manifest.valid must be true before candidate promotion")
        if capture_matrix_manifest.get("mode") != "production":
            errors.append("capture_preflight.capture_matrix_manifest.mode must be production")
        if not capture_matrix_manifest.get("matrix_csv_sha256"):
            errors.append("capture_preflight.capture_matrix_manifest.matrix_csv_sha256 is required")
        if not capture_matrix_manifest.get("source_manifest_sha256"):
            errors.append("capture_preflight.capture_matrix_manifest.source_manifest_sha256 is required")

    label_review_import_manifest = _validate_label_review_import_manifest(
        capture_preflight=capture_preflight,
        capture_matrix_manifest=capture_matrix_manifest,
        errors=errors,
    )
    dataset_provenance = _validate_dataset_provenance(result, capture_matrix_manifest, errors)
    source_lineage = _validate_source_lineage(result, capture_matrix_manifest, dataset_provenance, errors)
    seed_import_required = bool(
        isinstance(source_lineage.get("seed_import_manifest"), dict)
        and isinstance(source_lineage["seed_import_manifest"].get("gate"), dict)
        and source_lineage["seed_import_manifest"]["gate"].get("required") is True
    )
    seed_export_import_manifest = _validate_seed_export_import_manifest(
        capture_preflight=capture_preflight,
        capture_matrix_manifest=capture_matrix_manifest,
        required=seed_import_required,
        errors=errors,
    )
    source_seed_export = (
        source_lineage.get("seed_export_import_manifest")
        if isinstance(source_lineage.get("seed_export_import_manifest"), dict)
        else {}
    )
    if seed_import_required:
        if not source_seed_export:
            errors.append("source_lineage.seed_export_import_manifest is required")
        elif source_seed_export.get("valid") is not True:
            errors.append("source_lineage.seed_export_import_manifest.valid must be true")
        if (
            seed_export_import_manifest.get("sha256")
            and source_seed_export.get("sha256")
            and seed_export_import_manifest.get("sha256") != source_seed_export.get("sha256")
        ):
            errors.append("source_lineage.seed_export_import_manifest.sha256 must match capture preflight")

    classes = _normalize_classes(result.get("classes"))
    if classes != REQUIRED_CLASSES:
        errors.append(f"classes must exactly match {REQUIRED_CLASSES}")

    per_class_metrics = result.get("per_class_metrics")
    if not isinstance(per_class_metrics, dict):
        errors.append("per_class_metrics is required for apron/harness promotion")
        per_class_metrics = {}

    class_metric_summary: dict[str, dict[str, float | None]] = {}
    for class_name in REQUIRED_CLASSES.values():
        metrics = per_class_metrics.get(class_name) if isinstance(per_class_metrics, dict) else None
        if not isinstance(metrics, dict):
            errors.append(f"per_class_metrics.{class_name} is required")
            class_metric_summary[class_name] = {"mAP50": None, "recall": None}
            continue
        map50 = _metric_value(metrics, "mAP50")
        recall = _metric_value(metrics, "recall")
        class_metric_summary[class_name] = {"mAP50": map50, "recall": recall}
        if map50 is None or map50 < min_map50:
            errors.append(f"{class_name} mAP50 {map50} is below {min_map50}")
        if recall is None or recall < min_recall:
            errors.append(f"{class_name} recall {recall} is below {min_recall}")

    exports = result.get("exports")
    if not isinstance(exports, list) or not exports:
        errors.append("at least one ONNX or TensorRT export artifact is required")
        exports = []

    export_status = []
    accepted_export_count = 0
    onnx_export_count = 0
    selected_export: dict[str, Any] | None = None
    for export in exports:
        export_path = _resolve_artifact(str(export), result_path, artifact_root)
        suffix = export_path.suffix.lower()
        exists = export_path.exists()
        accepted_suffix = suffix in ACCEPTED_EXPORT_SUFFIXES
        export_item = {
            "path": str(export_path),
            "exists": exists,
            "accepted_suffix": accepted_suffix,
            "suffix": suffix,
        }
        if exists and accepted_suffix:
            accepted_export_count += 1
            export_item["sha256"] = _sha256_file(export_path)
            if suffix == PLANNED_REGISTRY_SUFFIX:
                onnx_export_count += 1
            if suffix == PLANNED_REGISTRY_SUFFIX and selected_export is None:
                selected_export = export_item
        export_status.append(export_item)
        if not exists:
            errors.append(f"export artifact does not exist: {export_path}")
        if not accepted_suffix:
            errors.append(f"export artifact must be ONNX or TensorRT engine: {export_path}")

    if accepted_export_count == 0:
        errors.append("no acceptable export artifact is ready for staging")
    if onnx_export_count == 0:
        errors.append("ONNX export artifact is required for planned registry path")

    promotion_manifest = {
        "candidate_status": "ready_for_side_by_side_runtime_test" if not errors else "not_ready",
        "model": model,
        "source_training_result": str(result_path),
        "classes": REQUIRED_CLASSES,
        "training_capture_preflight": capture_preflight,
        "training_dataset_provenance": dataset_provenance,
        "training_source_lineage": source_lineage,
        "label_review_import_manifest": label_review_import_manifest,
        "seed_export_import_manifest": seed_export_import_manifest,
        "metric_thresholds": {
            "min_per_class_mAP50": min_map50,
            "min_per_class_recall": min_recall,
        },
        "class_metrics": class_metric_summary,
        "exports": export_status,
        "runtime_handoff": {
            "planned_model_key": PLANNED_MODEL_KEY,
            "planned_registry_path": PLANNED_REGISTRY_PATH,
            "current_runtime_model_family": "ppe_specialist",
            "activation_policy": "do_not_activate_until_side_by_side_runtime_and_jetson_gates_pass",
            "selected_export": selected_export,
            "registry_entry": _registry_entry(selected_export, result_path, model),
            "model_manager_definition_hint": _model_manager_definition_hint(),
            "registration_preconditions": [
                "copy_selected_export_to_planned_registry_path",
                "verify_selected_export_sha256_after_copy",
                "verify_model_manager_definition_matches_candidate_handoff",
                "run_apron_side_by_side_promotion_report",
                "run_harness_side_by_side_promotion_report",
                "run_factory_ppe_jetson_full_gate",
                "keep_ppe_specialist_as_pilot_fallback_until_promotion_passes",
            ],
        },
        "capture_matrix_manifest": capture_matrix_manifest,
        "label_review_import_manifest": label_review_import_manifest,
        "seed_export_import_manifest": seed_export_import_manifest,
        "next_required_gates": [
            "verify_training_dataset_provenance",
            "verify_training_source_lineage",
            "verify_label_review_import_manifest",
            "verify_seed_export_import_manifest",
            "verify_candidate_export_sha256",
            "copy_selected_export_to_model_registry",
            "register_closed_set_ppe_model_key",
            "run_apron_active_window_positive",
            "run_apron_false_positive_guard",
            "run_apron_detector_window_suppression",
            "run_harness_active_window_positive",
            "run_harness_false_positive_guard",
            "run_harness_detector_window_suppression",
            "run_yoloe_side_by_side_regression",
            "run_jetson_one_camera_benchmark",
            "run_jetson_three_camera_soak",
        ],
    }
    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "training_result": str(result_path),
        "errors": errors,
        "warnings": warnings,
        "promotion_manifest": promotion_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate trained apron/harness model candidates.")
    parser.add_argument("--training-result", required=True, help="JSON result from scripts/apron_harness_train.py --execute")
    parser.add_argument("--artifact-root", default="", help="Optional root for relative export artifact paths")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    report = validate_candidate(Path(args.training_result), artifact_root=artifact_root)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"{status}: {report['training_result']}")
        if args.out:
            print(f"wrote: {args.out}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
