#!/usr/bin/env python3
"""Gate closed-set apron/harness runtime promotion against the YOLOE pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


READY_STATUS = "ready_to_sell"
DEFAULT_BASELINE_MODEL_KEY = "ppe_specialist"
DEFAULT_CANDIDATE_MODEL_KEY = "ppe_closed_set_candidate"
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"
REQUIRED_CLASSES = {
    0: "person",
    1: "apron",
    2: "safety_harness",
    3: "safety_lanyard",
}
REQUIRED_SEED_IMPORT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}

VISIBLE_CLASS_ALIASES = {
    "apron_required": {
        "apron",
        "denim apron",
        "work apron",
        "kitchen apron",
        "protective apron",
    },
    "harness_required": {
        "safety_harness",
        "safety harness",
        "safety_lanyard",
        "safety lanyard",
        "fall arrest harness",
        "fall protection lanyard",
        "body harness",
        "harness",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("evidence")
    return value if isinstance(value, dict) else {}


def _analytics_summary(result: dict[str, Any]) -> dict[str, Any]:
    value = _evidence(result).get("analytics_summary")
    return value if isinstance(value, dict) else {}


def _class_counts(result: dict[str, Any]) -> dict[str, int]:
    summary_counts = _analytics_summary(result).get("class_counts")
    if isinstance(summary_counts, dict):
        return {str(key): int(value or 0) for key, value in summary_counts.items()}
    raw_counts = _evidence(result).get("max_detection_class_counts")
    if isinstance(raw_counts, dict):
        return {str(key): int(value or 0) for key, value in raw_counts.items()}
    return {}


def _schedule(result: dict[str, Any]) -> dict[str, Any]:
    value = _analytics_summary(result).get("schedule")
    return value if isinstance(value, dict) else {}


def _model_invocations(result: dict[str, Any]) -> dict[str, int]:
    value = _schedule(result).get("model_invocations")
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count or 0) for key, count in value.items()}


def _model_preflight(result: dict[str, Any]) -> dict[str, Any]:
    value = _evidence(result).get("model_preflight")
    return value if isinstance(value, dict) else {}


def _matching_alert_count(result: dict[str, Any]) -> int:
    value = _evidence(result).get("matching_alerts")
    return len(value) if isinstance(value, list) else 0


def _unexpected_alert_count(result: dict[str, Any]) -> int:
    value = _evidence(result).get("unexpected_alerts")
    return len(value) if isinstance(value, list) else 0


def _max_detections(result: dict[str, Any]) -> int:
    try:
        return int(_evidence(result).get("max_detections_count") or 0)
    except (TypeError, ValueError):
        return 0


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    try:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _screenshot_ok(result: dict[str, Any]) -> bool:
    ui = _evidence(result).get("ui_evidence")
    if not isinstance(ui, dict):
        return False
    return bool(ui.get("screenshot_exists") and ui.get("screenshot_fresh"))


def _visible_class_total(result: dict[str, Any], capability: str) -> int:
    aliases = VISIBLE_CLASS_ALIASES.get(capability, set())
    counts = _class_counts(result)
    total = 0
    for class_name, count in counts.items():
        normalized = class_name.lower().replace("-", " ").strip()
        if normalized in aliases:
            total += count
    return total


def _result_summary(result: dict[str, Any], capability: str) -> dict[str, Any]:
    return {
        "scenario_id": result.get("scenario_id"),
        "status": result.get("status"),
        "max_detections": _max_detections(result),
        "matching_alerts": _matching_alert_count(result),
        "unexpected_alerts": _unexpected_alert_count(result),
        "visible_class_total": _visible_class_total(result, capability),
        "suppressed_capabilities": _schedule(result).get("suppressed_capabilities") or [],
        "model_invocations": _model_invocations(result),
        "screenshot_ok": _screenshot_ok(result),
    }


def _candidate_capture_matrix_manifest(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    capture_preflight = manifest.get("training_capture_preflight")
    capture_matrix_manifest = manifest.get("capture_matrix_manifest")
    if not isinstance(capture_matrix_manifest, dict) and isinstance(capture_preflight, dict):
        capture_matrix_manifest = capture_preflight.get("capture_matrix_manifest")
    return capture_matrix_manifest if isinstance(capture_matrix_manifest, dict) else {}


def _candidate_label_review_import_manifest(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    label_review_import_manifest = manifest.get("label_review_import_manifest")
    capture_preflight = manifest.get("training_capture_preflight")
    if not isinstance(label_review_import_manifest, dict) and isinstance(capture_preflight, dict):
        label_review_import_manifest = capture_preflight.get("label_review_import_manifest")
    return label_review_import_manifest if isinstance(label_review_import_manifest, dict) else {}


def _candidate_seed_export_import_manifest(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    seed_export_import_manifest = manifest.get("seed_export_import_manifest")
    capture_preflight = manifest.get("training_capture_preflight")
    if not isinstance(seed_export_import_manifest, dict) and isinstance(capture_preflight, dict):
        seed_export_import_manifest = capture_preflight.get("seed_export_import_manifest")
    return seed_export_import_manifest if isinstance(seed_export_import_manifest, dict) else {}


def _candidate_dataset_provenance(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    provenance = manifest.get("training_dataset_provenance")
    return provenance if isinstance(provenance, dict) else {}


def _candidate_source_lineage(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    lineage = manifest.get("training_source_lineage")
    return lineage if isinstance(lineage, dict) else {}


def _candidate_runtime_handoff(candidate_report: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        return {}
    handoff = manifest.get("runtime_handoff")
    return handoff if isinstance(handoff, dict) else {}


def _candidate_selected_export(candidate_report: dict[str, Any]) -> dict[str, Any]:
    selected_export = _candidate_runtime_handoff(candidate_report).get("selected_export")
    return selected_export if isinstance(selected_export, dict) else {}


def _candidate_registry_entry(candidate_report: dict[str, Any]) -> dict[str, Any]:
    registry_entry = _candidate_runtime_handoff(candidate_report).get("registry_entry")
    return registry_entry if isinstance(registry_entry, dict) else {}


def _validate_candidate_dataset_provenance(
    candidate_report: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    provenance = _candidate_dataset_provenance(candidate_report)
    if not provenance:
        errors.append("candidate report must include training_dataset_provenance")
        return {}
    if provenance.get("required") is not True:
        errors.append("candidate training_dataset_provenance.required must be true")
    if provenance.get("checked") is not True:
        errors.append("candidate training_dataset_provenance.checked must be true")
    if provenance.get("errors") not in ([], None):
        errors.append("candidate training_dataset_provenance.errors must be empty")
    if not provenance.get("source_manifest"):
        errors.append("candidate training_dataset_provenance.source_manifest is required")
    if not provenance.get("declared_source_manifest_sha256"):
        errors.append("candidate training_dataset_provenance.declared_source_manifest_sha256 is required")
    if not provenance.get("source_manifest_sha256"):
        errors.append("candidate training_dataset_provenance.source_manifest_sha256 is required")
    if (
        provenance.get("declared_source_manifest_sha256")
        and provenance.get("source_manifest_sha256")
        and provenance.get("declared_source_manifest_sha256") != provenance.get("source_manifest_sha256")
    ):
        errors.append("candidate training_dataset_provenance.declared_source_manifest_sha256 must match source_manifest_sha256")
    if provenance.get("permission_allowed") is not True:
        errors.append("candidate training_dataset_provenance.permission_allowed must be true")
    if provenance.get("missing_ppe_label_policy") != EXPECTED_MISSING_PPE_LABEL_POLICY:
        errors.append(
            "candidate training_dataset_provenance.missing_ppe_label_policy must be "
            f"{EXPECTED_MISSING_PPE_LABEL_POLICY}"
        )
    expected_sha = capture_matrix_manifest.get("source_manifest_sha256") if capture_matrix_manifest else None
    if expected_sha and provenance.get("source_manifest_sha256") != expected_sha:
        errors.append("candidate training_dataset_provenance.source_manifest_sha256 must match capture matrix source_manifest_sha256")
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
        errors.append(f"candidate training_source_lineage.{section_name} is required")
        return {}
    file_info = section.get("file")
    if not isinstance(file_info, dict):
        if required:
            errors.append(f"candidate training_source_lineage.{section_name}.file is required")
        return section
    if required:
        if file_info.get("required") is not True:
            errors.append(f"candidate training_source_lineage.{section_name}.file.required must be true")
        if not file_info.get("path"):
            errors.append(f"candidate training_source_lineage.{section_name}.file.path is required")
        if file_info.get("exists") is not True:
            errors.append(f"candidate training_source_lineage.{section_name}.file.exists must be true")
        if not file_info.get("sha256"):
            errors.append(f"candidate training_source_lineage.{section_name}.file.sha256 is required")
    return section


def _validate_seed_source_lineage(section: dict[str, Any], errors: list[str]) -> None:
    gate = section.get("gate") if isinstance(section.get("gate"), dict) else {}
    if gate.get("required") is not True:
        return
    if gate.get("ok") is not True:
        errors.append("candidate training_source_lineage.seed_source_review.gate.ok must be true")
    if gate.get("gate_passed") is not True:
        errors.append("candidate training_source_lineage.seed_source_review.gate.gate_passed must be true")
    clip_count = int(gate.get("clip_count") or 0)
    approved_clip_count = int(gate.get("approved_clip_count") or 0)
    if clip_count <= 0:
        errors.append("candidate training_source_lineage.seed_source_review.gate.clip_count must be greater than 0")
    if approved_clip_count != clip_count:
        errors.append("candidate training_source_lineage.seed_source_review.gate.approved_clip_count must equal clip_count")
    _validate_source_recheck_block(
        section.get("source_recheck"),
        prefix="candidate training_source_lineage.seed_source_review",
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
        errors.append("candidate training_source_lineage.seed_import_manifest.gate.ok must be true")
    if gate.get("source_review_sha256_matches") is not True:
        errors.append("candidate training_source_lineage.seed_import_manifest.gate.source_review_sha256_matches must be true")
    clip_count = int(gate.get("clip_count") or 0)
    approved_clip_count = int(gate.get("approved_clip_count") or 0)
    if clip_count <= 0:
        errors.append("candidate training_source_lineage.seed_import_manifest.gate.clip_count must be greater than 0")
    if approved_clip_count != clip_count:
        errors.append("candidate training_source_lineage.seed_import_manifest.gate.approved_clip_count must equal clip_count")
    _validate_seed_import_export_preflight(
        gate,
        errors,
        "candidate training_source_lineage.seed_import_manifest.gate",
    )


def _validate_candidate_source_lineage(
    candidate_report: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    dataset_provenance: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    lineage = _candidate_source_lineage(candidate_report)
    if not lineage:
        errors.append("candidate report must include training_source_lineage")
        return {}

    _lineage_section(lineage, "dataset_yaml", required=True, errors=errors)
    capture_section = _lineage_section(
        lineage,
        "capture_manifest",
        required=dataset_provenance.get("required") is True,
        errors=errors,
    )
    expected_sha = dataset_provenance.get("source_manifest_sha256") or capture_matrix_manifest.get("source_manifest_sha256")
    capture_file = capture_section.get("file") if isinstance(capture_section.get("file"), dict) else {}
    capture_file_sha = capture_file.get("sha256")
    capture_manifest_sha = capture_section.get("manifest_sha256")
    if expected_sha and capture_file_sha and capture_file_sha != expected_sha:
        errors.append("candidate training_source_lineage.capture_manifest.file.sha256 must match source_manifest_sha256")
    if expected_sha and capture_manifest_sha and capture_manifest_sha != expected_sha:
        errors.append("candidate training_source_lineage.capture_manifest.manifest_sha256 must match source_manifest_sha256")
    if dataset_provenance.get("required") is True and capture_section.get("ok") is not True:
        errors.append("candidate training_source_lineage.capture_manifest.ok must be true")
    if dataset_provenance.get("required") is True and capture_section.get("mode") != "production":
        errors.append("candidate training_source_lineage.capture_manifest.mode must be production")

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


def _validate_candidate_label_review_import_manifest(
    candidate_report: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    manifest = _candidate_label_review_import_manifest(candidate_report)
    if not manifest:
        errors.append("candidate report must include label_review_import_manifest")
        return {}
    if manifest.get("valid") is not True:
        errors.append("candidate label_review_import_manifest.valid must be true")
    if not manifest.get("source_manifest_sha256"):
        errors.append("candidate label_review_import_manifest.source_manifest_sha256 is required")
    if not manifest.get("label_review_csv_sha256"):
        errors.append("candidate label_review_import_manifest.label_review_csv_sha256 is required")
    if not manifest.get("updated_manifest_sha256"):
        errors.append("candidate label_review_import_manifest.updated_manifest_sha256 is required")
    expected_sha = capture_matrix_manifest.get("source_manifest_sha256") if capture_matrix_manifest else None
    if expected_sha and manifest.get("updated_manifest_sha256") != expected_sha:
        errors.append(
            "candidate label_review_import_manifest.updated_manifest_sha256 must match "
            "capture matrix source_manifest_sha256"
        )
    validation = (
        manifest.get("updated_manifest_validation")
        if isinstance(manifest.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        errors.append("candidate label_review_import_manifest.updated_manifest_validation is required")
    else:
        if validation.get("checked") is not True:
            errors.append("candidate label_review_import_manifest.updated_manifest_validation.checked must be true")
        if validation.get("ok") is not True:
            errors.append("candidate label_review_import_manifest.updated_manifest_validation.ok must be true")
        if validation.get("schema_only") is True:
            errors.append("candidate label_review_import_manifest.updated_manifest_validation.schema_only must be false")
        if validation.get("mode") != "production":
            errors.append("candidate label_review_import_manifest.updated_manifest_validation.mode must be production")
        if expected_sha and validation.get("manifest_sha256") != expected_sha:
            errors.append(
                "candidate label_review_import_manifest.updated_manifest_validation.manifest_sha256 "
                "must match capture matrix source_manifest_sha256"
            )
    try:
        imported_label_count = int(manifest.get("imported_label_count") or 0)
    except (TypeError, ValueError):
        imported_label_count = 0
    if imported_label_count <= 0:
        errors.append("candidate label_review_import_manifest.imported_label_count must be greater than 0")
    _validate_label_review_clip_metadata(
        manifest,
        prefix="candidate label_review_import_manifest",
        errors=errors,
    )
    return manifest


def _validate_candidate_seed_export_import_manifest(
    candidate_report: dict[str, Any],
    capture_matrix_manifest: dict[str, Any],
    *,
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    manifest = _candidate_seed_export_import_manifest(candidate_report)
    if not required:
        return manifest
    if not manifest:
        errors.append("candidate report must include seed_export_import_manifest")
        return {}
    if manifest.get("valid") is not True:
        errors.append("candidate seed_export_import_manifest.valid must be true")
    if manifest.get("partial_materialization") is True:
        errors.append("candidate seed_export_import_manifest.partial_materialization must be false")
    if not manifest.get("seed_source_review_sha256"):
        errors.append("candidate seed_export_import_manifest.seed_source_review_sha256 is required")
    _validate_source_recheck_block(
        manifest.get("source_recheck"),
        prefix="candidate seed_export_import_manifest",
        errors=errors,
    )
    if not manifest.get("seed_import_manifest_sha256"):
        errors.append("candidate seed_export_import_manifest.seed_import_manifest_sha256 is required")
    if not manifest.get("updated_manifest_sha256"):
        errors.append("candidate seed_export_import_manifest.updated_manifest_sha256 is required")
    expected_sha = capture_matrix_manifest.get("source_manifest_sha256") if capture_matrix_manifest else None
    if expected_sha and manifest.get("updated_manifest_sha256") != expected_sha:
        errors.append(
            "candidate seed_export_import_manifest.updated_manifest_sha256 must match "
            "capture matrix source_manifest_sha256"
        )
    validation = (
        manifest.get("updated_manifest_validation")
        if isinstance(manifest.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        errors.append("candidate seed_export_import_manifest.updated_manifest_validation is required")
    else:
        if validation.get("checked") is not True:
            errors.append("candidate seed_export_import_manifest.updated_manifest_validation.checked must be true")
        if validation.get("ok") is not True:
            errors.append("candidate seed_export_import_manifest.updated_manifest_validation.ok must be true")
        if validation.get("schema_only") is True:
            errors.append("candidate seed_export_import_manifest.updated_manifest_validation.schema_only must be false")
        if validation.get("mode") != "production":
            errors.append("candidate seed_export_import_manifest.updated_manifest_validation.mode must be production")
        if expected_sha and validation.get("manifest_sha256") != expected_sha:
            errors.append(
                "candidate seed_export_import_manifest.updated_manifest_validation.manifest_sha256 "
                "must match capture matrix source_manifest_sha256"
            )
    for field in ["imported_label_count", "imported_clip_count", "copied_image_count"]:
        try:
            value = int(manifest.get(field) or 0)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            errors.append(f"candidate seed_export_import_manifest.{field} must be greater than 0")

    imports = manifest.get("imports") if isinstance(manifest.get("imports"), list) else []
    if not imports:
        errors.append("candidate seed_export_import_manifest.imports must include materialized seed imports")
    for index, item in enumerate(imports):
        if not isinstance(item, dict):
            errors.append(f"candidate seed_export_import_manifest.imports[{index}] must be an object")
            continue
        if int(item.get("imported_label_count") or 0) <= 0:
            errors.append(f"candidate seed_export_import_manifest.imports[{index}].imported_label_count must be greater than 0")
        if int(item.get("copied_image_count") or 0) <= 0:
            errors.append(f"candidate seed_export_import_manifest.imports[{index}].copied_image_count must be greater than 0")
        if item.get("errors") not in ([], None):
            errors.append(f"candidate seed_export_import_manifest.imports[{index}].errors must be empty")
        if not item.get("raw_export_sha256"):
            errors.append(f"candidate seed_export_import_manifest.imports[{index}].raw_export_sha256 is required")
        _validate_seed_export_import_preflight(
            item,
            prefix=f"candidate seed_export_import_manifest.imports[{index}]",
            errors=errors,
        )
    return manifest


def _validate_candidate_report(candidate_report: dict[str, Any], expected_model_key: str, errors: list[str]) -> None:
    if candidate_report.get("ok") is not True:
        errors.append("candidate report must have ok=true")
    manifest = candidate_report.get("promotion_manifest")
    if not isinstance(manifest, dict):
        errors.append("candidate report must include promotion_manifest")
        return
    if manifest.get("candidate_status") != "ready_for_side_by_side_runtime_test":
        errors.append("candidate_status must be ready_for_side_by_side_runtime_test")
    metric_thresholds = manifest.get("metric_thresholds") if isinstance(manifest.get("metric_thresholds"), dict) else {}
    class_metrics = manifest.get("class_metrics") if isinstance(manifest.get("class_metrics"), dict) else {}
    min_map50 = _metric_value(metric_thresholds, "min_per_class_mAP50")
    min_recall = _metric_value(metric_thresholds, "min_per_class_recall")
    if min_map50 is None:
        errors.append("candidate metric_thresholds.min_per_class_mAP50 is required")
    if min_recall is None:
        errors.append("candidate metric_thresholds.min_per_class_recall is required")
    for class_name in REQUIRED_CLASSES.values():
        metrics = class_metrics.get(class_name)
        if not isinstance(metrics, dict):
            errors.append(f"candidate class_metrics.{class_name} is required")
            continue
        map50 = _metric_value(metrics, "mAP50")
        recall = _metric_value(metrics, "recall")
        if map50 is None:
            errors.append(f"candidate class_metrics.{class_name}.mAP50 is required")
        elif min_map50 is not None and map50 < min_map50:
            errors.append(f"candidate class_metrics.{class_name}.mAP50 is below {min_map50}")
        if recall is None:
            errors.append(f"candidate class_metrics.{class_name}.recall is required")
        elif min_recall is not None and recall < min_recall:
            errors.append(f"candidate class_metrics.{class_name}.recall is below {min_recall}")
    capture_preflight = manifest.get("training_capture_preflight")
    if not isinstance(capture_preflight, dict):
        errors.append("candidate report must include training_capture_preflight")
    elif capture_preflight.get("checked") is not True or capture_preflight.get("gate_passed") is not True:
        errors.append("candidate training_capture_preflight must be checked and passed")
    if isinstance(capture_preflight, dict) and capture_preflight.get("mode") != "production":
        errors.append("candidate training_capture_preflight.mode must be production")
    capture_matrix_manifest = _candidate_capture_matrix_manifest(candidate_report)
    if not capture_matrix_manifest:
        errors.append("candidate report must include capture_matrix_manifest")
    else:
        if capture_matrix_manifest.get("valid") is not True:
            errors.append("candidate capture_matrix_manifest.valid must be true")
        if capture_matrix_manifest.get("mode") != "production":
            errors.append("candidate capture_matrix_manifest.mode must be production")
        if not capture_matrix_manifest.get("matrix_csv_sha256"):
            errors.append("candidate capture_matrix_manifest.matrix_csv_sha256 is required")
        if not capture_matrix_manifest.get("source_manifest_sha256"):
            errors.append("candidate capture_matrix_manifest.source_manifest_sha256 is required")
    _validate_candidate_label_review_import_manifest(candidate_report, capture_matrix_manifest, errors)
    dataset_provenance = _validate_candidate_dataset_provenance(candidate_report, capture_matrix_manifest, errors)
    source_lineage = _validate_candidate_source_lineage(
        candidate_report,
        capture_matrix_manifest,
        dataset_provenance,
        errors,
    )
    seed_import_required = bool(
        isinstance(source_lineage.get("seed_import_manifest"), dict)
        and isinstance(source_lineage["seed_import_manifest"].get("gate"), dict)
        and source_lineage["seed_import_manifest"]["gate"].get("required") is True
    )
    _validate_candidate_seed_export_import_manifest(
        candidate_report,
        capture_matrix_manifest,
        required=seed_import_required,
        errors=errors,
    )
    handoff = manifest.get("runtime_handoff")
    if not isinstance(handoff, dict):
        errors.append("candidate report must include runtime_handoff")
        return
    if handoff.get("planned_model_key") != expected_model_key:
        errors.append(f"candidate planned_model_key must be {expected_model_key}")
    if not handoff.get("planned_registry_path"):
        errors.append("candidate planned_registry_path is required")
    selected_export = handoff.get("selected_export")
    if not isinstance(selected_export, dict):
        errors.append("candidate runtime_handoff.selected_export is required")
    else:
        if not selected_export.get("sha256"):
            errors.append("candidate selected_export.sha256 is required")
        suffix = str(selected_export.get("suffix") or "").lower()
        if selected_export.get("accepted_suffix") is not True and suffix not in {".onnx", ".engine"}:
            errors.append("candidate selected_export must be an accepted .onnx or .engine export")
    registry_entry = handoff.get("registry_entry")
    if not isinstance(registry_entry, dict):
        errors.append("candidate runtime_handoff.registry_entry is required")
    else:
        if registry_entry.get("model_key") != expected_model_key:
            errors.append(f"candidate registry_entry.model_key must be {expected_model_key}")
        if registry_entry.get("registry_path") != handoff.get("planned_registry_path"):
            errors.append("candidate registry_entry.registry_path must match planned_registry_path")
        selected_sha = selected_export.get("sha256") if isinstance(selected_export, dict) else None
        if selected_sha and registry_entry.get("source_export_sha256") != selected_sha:
            errors.append("candidate registry_entry.source_export_sha256 must match selected_export.sha256")


def _validate_active(label: str, result: dict[str, Any], errors: list[str]) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label} status must be {READY_STATUS}")
    if _max_detections(result) <= 0:
        errors.append(f"{label} must have detections")
    if _matching_alert_count(result) <= 0:
        errors.append(f"{label} must have at least one matching alert")
    if _unexpected_alert_count(result) != 0:
        errors.append(f"{label} must have zero unexpected alerts")
    if not _screenshot_ok(result):
        errors.append(f"{label} must include a fresh screenshot")


def _validate_false_positive_guard(label: str, result: dict[str, Any], capability: str, errors: list[str]) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label} status must be {READY_STATUS}")
    if _max_detections(result) <= 0:
        errors.append(f"{label} must have detections")
    if _matching_alert_count(result) != 0:
        errors.append(f"{label} must have zero matching alerts")
    if _unexpected_alert_count(result) != 0:
        errors.append(f"{label} must have zero unexpected alerts")
    if _visible_class_total(result, capability) <= 0:
        errors.append(f"{label} must include visible {capability} class telemetry")
    if not _screenshot_ok(result):
        errors.append(f"{label} must include a fresh screenshot")


def _validate_model_invoked(label: str, result: dict[str, Any], model_key: str, errors: list[str]) -> None:
    invocations = _model_invocations(result)
    if model_key not in invocations:
        errors.append(f"{label} must report model invocation telemetry for {model_key}")
    elif invocations.get(model_key, 0) <= 0:
        errors.append(f"{label} must report at least one {model_key} invocation")


def _validate_required_model_preflight(
    label: str,
    result: dict[str, Any],
    model_key: str,
    errors: list[str],
) -> None:
    preflight = _model_preflight(result)
    if preflight.get("checked") is not True:
        errors.append(f"{label} must include required-model preflight")
        return
    if preflight.get("ok") is not True:
        errors.append(f"{label} required-model preflight must pass")
    required_model_keys = {str(key) for key in (preflight.get("required_model_keys") or [])}
    if model_key not in required_model_keys:
        errors.append(f"{label} required-model preflight must require {model_key}")
    missing_model_keys = {str(key) for key in (preflight.get("missing_model_keys") or [])}
    if missing_model_keys:
        errors.append(f"{label} required-model preflight must have no missing model keys")


def _validate_model_not_invoked(label: str, result: dict[str, Any], model_key: str, errors: list[str]) -> None:
    if not model_key:
        return
    invocations = _model_invocations(result)
    if invocations.get(model_key, 0) > 0:
        errors.append(f"{label} must report zero {model_key} invocations")


def _validate_suppression(
    label: str,
    result: dict[str, Any],
    capability: str,
    model_key: str,
    errors: list[str],
) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label} status must be {READY_STATUS}")
    if _max_detections(result) != 0:
        errors.append(f"{label} must emit zero detections")
    if _matching_alert_count(result) != 0 or _unexpected_alert_count(result) != 0:
        errors.append(f"{label} must emit zero alerts")
    suppressed = set(str(value) for value in (_schedule(result).get("suppressed_capabilities") or []))
    if capability not in suppressed:
        errors.append(f"{label} must suppress {capability}")
    invocations = _model_invocations(result)
    if model_key not in invocations:
        errors.append(f"{label} must report model invocation telemetry for {model_key}")
    elif invocations.get(model_key, 0) != 0:
        errors.append(f"{label} must report zero {model_key} invocations")
    if not _screenshot_ok(result):
        errors.append(f"{label} must include a fresh screenshot")


def _model_gate_summary(
    result: dict[str, Any],
    *,
    required_model_key: str,
    required_invocation_policy: str,
    forbidden_model_key: str,
) -> dict[str, Any]:
    invocations = _model_invocations(result)
    return {
        "required_model_key": required_model_key,
        "required_invocation_policy": required_invocation_policy,
        "required_model_invocations": invocations.get(required_model_key, 0),
        "forbidden_model_key": forbidden_model_key,
        "forbidden_model_invocations": invocations.get(forbidden_model_key, 0),
    }


def validate_promotion(
    *,
    capability: str,
    candidate_report_path: Path,
    baseline_active_path: Path,
    baseline_guard_path: Path,
    baseline_suppression_path: Path,
    candidate_active_path: Path,
    candidate_guard_path: Path,
    candidate_suppression_path: Path,
    baseline_model_key: str = DEFAULT_BASELINE_MODEL_KEY,
    candidate_model_key: str = DEFAULT_CANDIDATE_MODEL_KEY,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if capability not in VISIBLE_CLASS_ALIASES:
        errors.append(f"unsupported capability: {capability}")
    if baseline_model_key == candidate_model_key:
        errors.append("baseline_model_key and candidate_model_key must be distinct for side-by-side promotion")

    candidate_report_sha256 = _sha256_file(candidate_report_path)
    candidate_report = _load_json(candidate_report_path)
    baseline_active = _load_json(baseline_active_path)
    baseline_guard = _load_json(baseline_guard_path)
    baseline_suppression = _load_json(baseline_suppression_path)
    candidate_active = _load_json(candidate_active_path)
    candidate_guard = _load_json(candidate_guard_path)
    candidate_suppression = _load_json(candidate_suppression_path)
    capture_matrix_manifest = _candidate_capture_matrix_manifest(candidate_report)
    label_review_import_manifest = _candidate_label_review_import_manifest(candidate_report)
    seed_export_import_manifest = _candidate_seed_export_import_manifest(candidate_report)
    dataset_provenance = _candidate_dataset_provenance(candidate_report)
    source_lineage = _candidate_source_lineage(candidate_report)
    runtime_handoff = _candidate_runtime_handoff(candidate_report)
    selected_export = _candidate_selected_export(candidate_report)
    registry_entry = _candidate_registry_entry(candidate_report)

    _validate_candidate_report(candidate_report, candidate_model_key, errors)

    _validate_active("baseline_active", baseline_active, errors)
    _validate_model_invoked("baseline_active", baseline_active, baseline_model_key, errors)
    _validate_model_not_invoked("baseline_active", baseline_active, candidate_model_key, errors)
    _validate_false_positive_guard("baseline_false_positive_guard", baseline_guard, capability, errors)
    _validate_model_invoked("baseline_false_positive_guard", baseline_guard, baseline_model_key, errors)
    _validate_model_not_invoked("baseline_false_positive_guard", baseline_guard, candidate_model_key, errors)
    _validate_suppression("baseline_suppression", baseline_suppression, capability, baseline_model_key, errors)
    _validate_model_not_invoked("baseline_suppression", baseline_suppression, candidate_model_key, errors)

    _validate_active("candidate_active", candidate_active, errors)
    _validate_required_model_preflight("candidate_active", candidate_active, candidate_model_key, errors)
    _validate_model_invoked("candidate_active", candidate_active, candidate_model_key, errors)
    _validate_model_not_invoked("candidate_active", candidate_active, baseline_model_key, errors)
    _validate_false_positive_guard("candidate_false_positive_guard", candidate_guard, capability, errors)
    _validate_required_model_preflight(
        "candidate_false_positive_guard",
        candidate_guard,
        candidate_model_key,
        errors,
    )
    _validate_model_invoked("candidate_false_positive_guard", candidate_guard, candidate_model_key, errors)
    _validate_model_not_invoked("candidate_false_positive_guard", candidate_guard, baseline_model_key, errors)
    _validate_suppression("candidate_suppression", candidate_suppression, capability, candidate_model_key, errors)
    _validate_required_model_preflight("candidate_suppression", candidate_suppression, candidate_model_key, errors)
    _validate_model_not_invoked("candidate_suppression", candidate_suppression, baseline_model_key, errors)

    if not errors and _visible_class_total(candidate_guard, capability) < 1:
        warnings.append("candidate visible class count is minimal; broaden false-positive guard coverage before production")

    promotion_status = "ready_for_runtime_registration" if not errors else "not_ready"
    return {
        "ok": not errors,
        "generated_at": utc_now(),
        "capability": capability,
        "candidate_model_key": candidate_model_key,
        "baseline_model_key": baseline_model_key,
        "candidate_report_sha256": candidate_report_sha256,
        "promotion_status": promotion_status,
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "candidate_report": str(candidate_report_path),
            "baseline_active": str(baseline_active_path),
            "baseline_guard": str(baseline_guard_path),
            "baseline_suppression": str(baseline_suppression_path),
            "candidate_active": str(candidate_active_path),
            "candidate_guard": str(candidate_guard_path),
            "candidate_suppression": str(candidate_suppression_path),
        },
        "baseline": {
            "active": _result_summary(baseline_active, capability),
            "false_positive_guard": _result_summary(baseline_guard, capability),
            "suppression": _result_summary(baseline_suppression, capability),
        },
        "candidate": {
            "active": _result_summary(candidate_active, capability),
            "false_positive_guard": _result_summary(candidate_guard, capability),
            "suppression": _result_summary(candidate_suppression, capability),
        },
        "runtime_model_gates": {
            "baseline": {
                "active": _model_gate_summary(
                    baseline_active,
                    required_model_key=baseline_model_key,
                    required_invocation_policy="gt_zero",
                    forbidden_model_key=candidate_model_key,
                ),
                "false_positive_guard": _model_gate_summary(
                    baseline_guard,
                    required_model_key=baseline_model_key,
                    required_invocation_policy="gt_zero",
                    forbidden_model_key=candidate_model_key,
                ),
                "suppression": _model_gate_summary(
                    baseline_suppression,
                    required_model_key=baseline_model_key,
                    required_invocation_policy="zero",
                    forbidden_model_key=candidate_model_key,
                ),
            },
            "candidate": {
                "active": _model_gate_summary(
                    candidate_active,
                    required_model_key=candidate_model_key,
                    required_invocation_policy="gt_zero",
                    forbidden_model_key=baseline_model_key,
                ),
                "false_positive_guard": _model_gate_summary(
                    candidate_guard,
                    required_model_key=candidate_model_key,
                    required_invocation_policy="gt_zero",
                    forbidden_model_key=baseline_model_key,
                ),
                "suppression": _model_gate_summary(
                    candidate_suppression,
                    required_model_key=candidate_model_key,
                    required_invocation_policy="zero",
                    forbidden_model_key=baseline_model_key,
                ),
            },
        },
        "candidate_capture_matrix_manifest": capture_matrix_manifest if isinstance(capture_matrix_manifest, dict) else {},
        "candidate_label_review_import_manifest": label_review_import_manifest
        if isinstance(label_review_import_manifest, dict)
        else {},
        "candidate_seed_export_import_manifest": seed_export_import_manifest
        if isinstance(seed_export_import_manifest, dict)
        else {},
        "candidate_training_dataset_provenance": dataset_provenance if isinstance(dataset_provenance, dict) else {},
        "candidate_training_source_lineage": source_lineage if isinstance(source_lineage, dict) else {},
        "candidate_runtime_handoff": runtime_handoff if isinstance(runtime_handoff, dict) else {},
        "candidate_selected_export": selected_export if isinstance(selected_export, dict) else {},
        "candidate_registry_entry": registry_entry if isinstance(registry_entry, dict) else {},
        "next_required_gates": [
            "register_closed_set_ppe_model_key",
            "run_jetson_one_camera_benchmark",
            "run_jetson_three_camera_soak",
            "update_sales_readiness_scope",
        ] if not errors else [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate closed-set apron/harness runtime promotion evidence.")
    parser.add_argument("--capability", required=True, choices=sorted(VISIBLE_CLASS_ALIASES))
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-active", required=True)
    parser.add_argument("--baseline-guard", required=True)
    parser.add_argument("--baseline-suppression", required=True)
    parser.add_argument("--candidate-active", required=True)
    parser.add_argument("--candidate-guard", required=True)
    parser.add_argument("--candidate-suppression", required=True)
    parser.add_argument("--baseline-model-key", default=DEFAULT_BASELINE_MODEL_KEY)
    parser.add_argument("--candidate-model-key", default=DEFAULT_CANDIDATE_MODEL_KEY)
    parser.add_argument("--out", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_promotion(
        capability=args.capability,
        candidate_report_path=Path(args.candidate_report),
        baseline_active_path=Path(args.baseline_active),
        baseline_guard_path=Path(args.baseline_guard),
        baseline_suppression_path=Path(args.baseline_suppression),
        candidate_active_path=Path(args.candidate_active),
        candidate_guard_path=Path(args.candidate_guard),
        candidate_suppression_path=Path(args.candidate_suppression),
        baseline_model_key=args.baseline_model_key,
        candidate_model_key=args.candidate_model_key,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"{status}: {report['capability']} {report['promotion_status']}")
        if args.out:
            print(f"wrote: {args.out}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
