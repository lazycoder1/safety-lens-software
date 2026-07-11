#!/usr/bin/env python3
"""Audit apron/harness pilot evidence and keep production claims blocked."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

import site_config  # noqa: E402

from apron_harness_dataset_doctor import (  # noqa: E402
    LABEL_REVIEW_GUIDANCE_FIELDS,
    LABEL_REVIEW_IMPORT_REQUIRED_FIELDS,
    MIN_COUNTS,
    REQUIRED_CLASSES,
    TAXONOMY_VERSION,
    YOLO_LABEL_FORMAT,
    render_capture_matrix_csv,
    render_capture_work_order,
    render_label_review_csv,
    render_starter_label_review_csv,
    validate_capture_matrix_progress,
    validate_manifest,
)
from apron_harness_seed_source_doctor import validate_import_manifest, validate_review_bundle_manifest  # noqa: E402
from apron_harness_train import (  # noqa: E402
    EXPECTED_MISSING_PPE_LABEL_POLICY,
    build_training_plan,
    validate_capture_matrix_sidecar,
    validate_dataset_provenance,
    validate_label_review_import_sidecar,
)

DEFAULT_MODEL_PACKS = ROOT / "qa" / "video_eval" / "model_packs.yaml"
DEFAULT_RESULT_DIR = ROOT / "qa" / "video_eval" / "results"
DEFAULT_MODEL_PACK_EVIDENCE_REPORT = ROOT / "qa" / "video_eval" / "results" / "model_pack_evidence_doctor.json"
DEFAULT_CAPTURE_MANIFEST = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml"
DEFAULT_TRAINING_DATASET_YAML = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_dataset.example.yaml"
DEFAULT_CAPTURE_KICKOFF = ROOT / "qa" / "video_eval" / "results" / "apron_harness_capture_kickoff.md"
DEFAULT_CAPTURE_WORK_ORDER = ROOT / "qa" / "video_eval" / "results" / "apron_harness_capture_work_order.md"
DEFAULT_CAPTURE_MATRIX_CSV = ROOT / "qa" / "video_eval" / "results" / "apron_harness_capture_matrix.csv"
DEFAULT_PRODUCTION_CAPTURE_MATRIX_CSV = ROOT / "qa" / "video_eval" / "results" / "apron_harness_production_capture_matrix.csv"
DEFAULT_LABEL_REVIEW_CSV = ROOT / "qa" / "video_eval" / "results" / "apron_harness_label_review_template.csv"
DEFAULT_STARTER_LABEL_REVIEW_CSV = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_starter_label_review_template.csv"
)
DEFAULT_PRODUCTION_LABEL_REVIEW_CSV = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_production_label_review_template.csv"
)
DEFAULT_PRODUCTION_STARTER_LABEL_REVIEW_CSV = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_production_starter_label_review_template.csv"
)

LABEL_REVIEW_SCHEMA_SUMMARY = {
    "taxonomy_version": TAXONOMY_VERSION,
    "label_format": YOLO_LABEL_FORMAT,
    "required_import_fields": LABEL_REVIEW_IMPORT_REQUIRED_FIELDS,
    "generated_guidance_fields": LABEL_REVIEW_GUIDANCE_FIELDS,
    "required_import_sidecar": ".label_review_import.json",
    "approval_gate": {
        "requires_review_status": "approved",
        "requires_reviewer": True,
        "requires_reviewed_at": True,
        "requires_cleared_permission": True,
        "requires_non_repo_raw_storage_ref": True,
        "requires_source_manifest_sha256_match": True,
        "requires_taxonomy_version_match": True,
        "requires_required_class_ids_present": True,
    },
}
DEFAULT_SEED_SOURCE_REVIEW = ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review.json"
DEFAULT_SEED_SOURCE_REVIEW_BUNDLE = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_review_bundle.json"
)
DEFAULT_SEED_IMPORT_MANIFEST = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_seed_import_manifest.template.yaml"
DEFAULT_MINIMUM_SEED_IMPORT_MANIFEST = (
    ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_minimum_seed_import_manifest.template.yaml"
)
DEFAULT_PROMOTION_RUNBOOK = ROOT / "qa" / "video_eval" / "results" / "apron_harness_promotion_runbook.md"
DEFAULT_CANDIDATE_RUNTIME_RUNBOOK = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_candidate_runtime_runbook.md"
)
DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_candidate_runtime_preflight_refresh.json"
)
DEFAULT_CANDIDATE_RUNTIME_RUNNER = ROOT / "scripts" / "apron_harness_candidate_runtime_runner.py"
DEFAULT_SOURCE_REVIEW_RUNNER = ROOT / "scripts" / "apron_harness_source_review_runner.py"
DEFAULT_CONTROLLED_CAPTURE_RUNNER = ROOT / "scripts" / "apron_harness_controlled_capture_runner.py"
DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_review_runner_plan.json"
)
DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_controlled_capture_starter_plan.json"
)
DEFAULT_CANDIDATE_TRAINING_RUNNER = ROOT / "scripts" / "apron_harness_candidate_training_runner.py"
DEFAULT_JETSON_GATE_RUNNER = ROOT / "scripts" / "apron_harness_jetson_gate_runner.py"
DEFAULT_MODEL_REGISTRY_REPORT = ROOT / "qa" / "video_eval" / "results" / "apron_harness_model_registry_report.json"
DEFAULT_JETSON_GATE_REPORT = ROOT / "qa" / "video_eval" / "results" / "factory_ppe_jetson_gate.json"
DEFAULT_SEED_IMPORT_VALIDATION_SUMMARY = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_import_manifest_validation_summary.json"
)
DEFAULT_PRODUCTION_CAPTURE_MATRIX_VALIDATION_SUMMARY = (
    ROOT
    / "qa"
    / "video_eval"
    / "results"
    / "apron_harness_production_capture_matrix_validation_summary.json"
)
DEFAULT_PRODUCTION_GATE_PACKET = (
    ROOT / "qa" / "video_eval" / "results" / "apron_harness_production_gate_packet.json"
)
DEFAULT_FACTORY_PPE_RAW_TEMPLATE = (
    ROOT / "qa" / "video_eval" / "results" / "factory_ppe_raw_benchmark.template.json"
)
DEFAULT_FACTORY_PPE_SOAK_TEMPLATE = (
    ROOT / "qa" / "video_eval" / "results" / "factory_ppe_3cam_soak.template.json"
)
DEFAULT_TRAINING_DATASET_YAML_OUT = ROOT / "qa" / "video_eval" / "results" / "apron_harness_training_dataset.yaml"
DEFAULT_TRAINING_DATASET_ROOT = "/opt/rakshak-lens/datasets/apron_harness_ppe"

READY_STATUS = "ready_to_sell"
EXPECTED_PACK_STATUS = "pilot_ready_not_production_compliance"
PRODUCTION_PACK_STATUS = "ready_to_sell_production_compliance"
EXPECTED_SOURCING_RESULT = "public_seed_sources_found_unapproved_insufficient_for_production"
PRODUCTION_SOURCING_RESULT = "cleared_closed_set_dataset_approved"
EXPECTED_HANDOFF_STATUS = "not_registered_until_candidate_gates_pass"
PRODUCTION_HANDOFF_STATUS = "registered_closed_set_candidate"
PLANNED_MODEL_KEY = "ppe_closed_set_candidate"
PILOT_MODEL_KEY = "ppe_specialist"
DEFAULT_TRAINING_MODEL = "yolo26n.pt"
EXPECTED_SKIPPED_MODEL_PACKS = {"pose_fall_3cam"}
READINESS_SCOPE = {
    "active_vertical": "factory_ppe",
    "skipped_verticals": ["hospital", "hospitals", "rl_m"],
    "capabilities": ["apron_required", "harness_required"],
    "production_model_policy": (
        "train and promote a reviewed closed-set YOLO26 nano/small candidate; "
        "YOLOE and older YOLO families remain demo/pilot/legacy unless explicitly reapproved"
    ),
}

CAPABILITY_SCENARIOS = {
    "apron_required": {
        "active": "factory_missing_apron_active",
        "guard": "factory_apron_false_positive_guard",
        "suppression": "factory_apron_detector_window_suppression",
        "visible_aliases": {
            "apron",
            "denim apron",
            "work apron",
            "kitchen apron",
            "protective apron",
        },
    },
    "harness_required": {
        "active": "factory_missing_harness_active",
        "guard": "factory_harness_false_positive_guard",
        "suppression": "factory_harness_detector_window_suppression",
        "visible_aliases": {
            "safety_harness",
            "safety harness",
            "safety_lanyard",
            "safety lanyard",
            "fall arrest harness",
            "fall protection lanyard",
            "body harness",
            "harness",
        },
    },
}
REQUIRED_SEED_IMPORT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}
CLOSED_SET_CANDIDATE_RESULT_DIR = ROOT / "qa" / "video_eval" / "results" / "closed_set_candidate"
CLOSED_SET_CANDIDATE_YAML_TEMPLATES = [
    {
        "scenario_id": "factory_missing_apron_active_closed_set",
        "capability": "apron_required",
        "role": "active",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_missing_apron_active_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_missing_apron_active_closed_set.json",
    },
    {
        "scenario_id": "factory_apron_false_positive_guard_closed_set",
        "capability": "apron_required",
        "role": "false_positive_guard",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_apron_false_positive_guard_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_apron_false_positive_guard_closed_set.json",
    },
    {
        "scenario_id": "factory_apron_detector_window_suppression_closed_set",
        "capability": "apron_required",
        "role": "suppression",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_apron_detector_window_suppression_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_apron_detector_window_suppression_closed_set.json",
    },
    {
        "scenario_id": "factory_missing_harness_active_closed_set",
        "capability": "harness_required",
        "role": "active",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_missing_harness_active_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_missing_harness_active_closed_set.json",
    },
    {
        "scenario_id": "factory_harness_false_positive_guard_closed_set",
        "capability": "harness_required",
        "role": "false_positive_guard",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_harness_false_positive_guard_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_harness_false_positive_guard_closed_set.json",
    },
    {
        "scenario_id": "factory_harness_detector_window_suppression_closed_set",
        "capability": "harness_required",
        "role": "suppression",
        "config_path": ROOT / "qa" / "video_eval" / "focused" / "factory_harness_detector_window_suppression_closed_set.yaml",
        "expected_result_path": CLOSED_SET_CANDIDATE_RESULT_DIR / "factory_harness_detector_window_suppression_closed_set.json",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _artifact_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else ROOT / path


def _render_training_dataset_yaml(
    *,
    capture_manifest: Path,
    dataset_root: str,
) -> str:
    source_manifest = capture_manifest.resolve(strict=False)
    payload = {
        "path": dataset_root,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": REQUIRED_CLASSES,
        "rakshak_lens": {
            "dataset_id": "apron_harness_ppe_closed_set",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256_file(capture_manifest) if capture_manifest.exists() else None,
            "permission": "controlled_capture_cleared",
            "missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
            "generated_by": "scripts/apron_harness_readiness_doctor.py",
            "generated_at": utc_now(),
            "status": "blocked_until_capture_matrix_and_label_review_pass",
            "training_gate_note": (
                "Use this dataset YAML only with --capture-preflight-mode production "
                "and --require-capture-preflight after the production capture matrix "
                "and label-review import sidecar pass."
            ),
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _write_training_dataset_yaml(
    *,
    path: Path | None,
    capture_manifest: Path,
    dataset_root: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _rel(path) if path else None,
        "generated": False,
        "exists": False,
        "dataset_root": dataset_root,
        "source_manifest": _rel(capture_manifest),
        "source_manifest_sha256": _sha256_file(capture_manifest) if capture_manifest.exists() else None,
    }
    if path is None:
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_training_dataset_yaml(
            capture_manifest=capture_manifest,
            dataset_root=dataset_root,
        ),
        encoding="utf-8",
    )
    result.update(
        {
            "generated": True,
            "exists": path.exists(),
            "sha256": _sha256_file(path),
        }
    )
    return result


def _matrix_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _capture_matrix_batch_sidecar(batch: dict[str, Any]) -> dict[str, Any]:
    matrix = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
    target_labeled_examples = 0
    labeled_examples = 0
    required_label_classes: set[str] = set()
    capture_types: dict[str, int] = {}
    for row in matrix:
        if not isinstance(row, dict):
            continue
        try:
            recommended = int(str(row.get("recommended_examples") or "0"))
        except ValueError:
            recommended = 0
        try:
            labeled = int(str(row.get("labeled_examples") or "0"))
        except ValueError:
            labeled = 0
        target_labeled_examples += max(0, recommended)
        labeled_examples += max(0, labeled)
        for class_name in row.get("required_label_classes") or []:
            required_label_classes.add(str(class_name))
        capture_type = str(row.get("capture_type") or "unknown")
        capture_types[capture_type] = capture_types.get(capture_type, 0) + 1
    return {
        "batch_id": batch.get("batch_id"),
        "target_capability": batch.get("target_capability"),
        "minimum_labeled_images": batch.get("minimum_labeled_images"),
        "recommended_label_review_rows": batch.get("recommended_label_review_rows", target_labeled_examples),
        "target_labeled_examples": target_labeled_examples,
        "labeled_examples": labeled_examples,
        "missing_labeled_examples": max(0, target_labeled_examples - labeled_examples),
        "missing_label_annotations": max(0, target_labeled_examples - labeled_examples),
        "required_label_classes": sorted(required_label_classes),
        "capture_types": capture_types,
        "row_count": len(matrix),
    }


def _write_capture_matrix_sidecar(
    *,
    matrix_csv_path: Path,
    mode: str,
    source_manifest: Path,
    source_manifest_sha256: Any,
    row_count: int,
    capture_deficit: dict[str, Any],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar_path = _matrix_sidecar_path(matrix_csv_path)
    payload = {
        "generated_at": utc_now(),
        "kind": "apron_harness_capture_matrix_manifest",
        "mode": mode,
        "matrix_csv": _rel(matrix_csv_path),
        "matrix_csv_sha256": _sha256_file(matrix_csv_path),
        "source_manifest": _rel(source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "row_count": row_count,
        "required_labeled_images_per_class": {
            "pilot": MIN_COUNTS["pilot"],
            "production": MIN_COUNTS["production"],
        },
        "required_label_classes": sorted(REQUIRED_CLASSES.values()),
        "target_labeled_examples": capture_deficit.get(
            "target_labeled_examples",
            capture_deficit.get("total_missing_label_annotations", 0),
        ),
        "missing_label_annotations": capture_deficit.get("total_missing_label_annotations", 0),
        "next_capture_batches": [
            _capture_matrix_batch_sidecar(batch)
            for batch in capture_deficit.get("next_capture_batches") or []
            if isinstance(batch, dict)
        ],
        "training_gate": {
            "requires_all_rows_ready": True,
            "requires_manifest_reconciliation": True,
            "requires_non_repo_raw_storage_refs": True,
            "requires_permission_approved": True,
        },
    }
    if progress:
        payload["progress"] = {
            "gate_passed": bool(progress.get("gate_passed")),
            "ready_rows": progress.get("ready_rows", 0),
            "row_count": progress.get("row_count", row_count),
            "target_labeled_examples": progress.get("target_labeled_examples", 0),
            "captured_examples": progress.get("captured_examples", 0),
            "labeled_examples": progress.get("labeled_examples", 0),
            "missing_labeled_examples": progress.get("missing_labeled_examples", 0),
            "unapproved_rows": progress.get("unapproved_rows", 0),
            "unsafe_storage_rows": progress.get("unsafe_storage_rows", 0),
        }
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "path": _rel(sidecar_path),
        "generated": True,
        "exists": sidecar_path.exists(),
        "sha256": _sha256_file(sidecar_path),
        "matrix_csv_sha256": payload["matrix_csv_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "mode": mode,
        "row_count": row_count,
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        separator = ";" if ";" in value else ","
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


def _capture_batch_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "capture_type": row.get("capture_type") or "",
                "variant_or_tag": row.get("variant_or_tag") or row.get("variant") or row.get("tag") or "",
                "recommended_examples": row.get("recommended_examples") or 0,
                "required_label_classes": _string_list(row.get("required_label_classes")),
                "camera_angles": _string_list(row.get("camera_angles")),
                "distance_bands": _string_list(row.get("distance_bands")),
                "lighting": _string_list(row.get("lighting")),
                "motion_blur": _string_list(row.get("motion_blur")),
            }
        )
    return compact


def _join_list(value: list[str]) -> str:
    return ", ".join(value) if value else "none"


def _render_capture_batch_summary(batch: dict[str, Any]) -> list[str]:
    rows = _capture_batch_rows(batch)
    required_classes = sorted({class_name for row in rows for class_name in row["required_label_classes"]})
    positive_rows = [row for row in rows if row["capture_type"] == "positive_variant"]
    negative_rows = [row for row in rows if row["capture_type"] == "hard_negative"]
    return [
        f"### `{batch.get('batch_id')}`",
        "",
        f"- Capability: `{batch.get('target_capability')}`",
        f"- Minimum labeled images: `{batch.get('minimum_labeled_images', 0)}`",
        f"- Recommended label-review rows: `{batch.get('recommended_label_review_rows', 0)}`",
        f"- Required label classes: `{_join_list(required_classes)}`",
        f"- Positive rows: `{len(positive_rows)}`",
        f"- Hard-negative rows: `{len(negative_rows)}`",
        "",
        "| Type | Variant / Tag | Examples | Required Labels |",
        "|---|---|---:|---|",
        *[
            "| "
            f"`{row['capture_type']}` | "
            f"`{row['variant_or_tag']}` | "
            f"{row['recommended_examples']} | "
            f"{_join_list(row['required_label_classes'])} |"
            for row in rows
        ],
        "",
    ]


def _capture_starter_rows(
    batches: list[dict[str, Any]],
    *,
    max_rows_per_capability: int = 2,
) -> list[dict[str, Any]]:
    starter_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for batch in batches:
        rows = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
        batch_candidates: list[dict[str, Any]] = []
        for preferred_type in ("positive_variant", "hard_negative"):
            batch_candidates.extend(
                row
                for row in rows
                if isinstance(row, dict) and row.get("capture_type") == preferred_type
            )
        for row in batch_candidates:
            if not isinstance(row, dict):
                continue
            capability = str(row.get("target_capability") or batch.get("target_capability") or "unknown")
            key = (capability, str(row.get("capture_type") or "unknown"))
            if key in seen:
                continue
            if sum(1 for seen_capability, _ in seen if seen_capability == capability) >= max_rows_per_capability:
                continue
            starter_rows.append(
                {
                    "row_id": row.get("row_id") or "unknown",
                    "target_capability": capability,
                    "capture_type": row.get("capture_type") or "unknown",
                    "variant_or_tag": row.get("variant_or_tag") or row.get("variant") or row.get("tag") or "unknown",
                    "recommended_examples": row.get("recommended_examples") or 0,
                    "required_label_classes": _string_list(row.get("required_label_classes")),
                    "camera_angles": _string_list(row.get("camera_angles")),
                    "distance_bands": _string_list(row.get("distance_bands")),
                    "lighting": _string_list(row.get("lighting")),
                    "motion_blur": _string_list(row.get("motion_blur")),
                }
            )
            seen.add(key)
    return starter_rows


def _render_capture_starter_rows(
    title: str,
    batches: list[dict[str, Any]],
    *,
    matrix_path: Path | None,
    rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    rows = rows if rows is not None else _capture_starter_rows(batches)
    if not rows:
        return []
    lines = [
        f"### {title}",
        "",
        f"- Matrix: `{_rel(matrix_path) if matrix_path else 'not generated'}`",
        "- Fill these rows first, then continue down the same matrix until every row is approved and reconciled to the capture manifest.",
        "",
        "| Capability | Row ID | Type | Examples | Required Labels | Coverage Checklist |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        coverage = (
            f"angles={_join_list(_string_list(row.get('camera_angles')))}; "
            f"distance={_join_list(_string_list(row.get('distance_bands')))}; "
            f"lighting={_join_list(_string_list(row.get('lighting')))}; "
            f"motion={_join_list(_string_list(row.get('motion_blur')))}"
        )
        lines.append(
            "| "
            f"`{row['target_capability']}` | "
            f"`{row['row_id']}` | "
            f"`{row['capture_type']}`/`{row['variant_or_tag']}` | "
            f"{row['recommended_examples']} | "
            f"{_join_list(row['required_label_classes'])} | "
            f"{coverage} |"
        )
    lines.extend(
        [
            "",
            "Required per approved row: `source_clip_id`, `image_path`, `label_path`, `review_status=approved`, `reviewer`, `reviewed_at`, `permission=approved`, and external `raw_storage_ref`.",
            "",
        ]
    )
    return lines


def _capture_matrix_starter_rows_or_none(matrix_path: Path | None) -> list[dict[str, Any]] | None:
    if matrix_path is None:
        return None
    rows = _capture_matrix_starter_rows(matrix_path)
    return rows or None


def render_capture_kickoff(
    report: dict[str, Any],
    *,
    capture_manifest: Path,
    capture_work_order_out: Path | None,
    capture_matrix_csv_out: Path | None,
    production_capture_matrix_csv_out: Path | None,
    label_review_csv_out: Path | None,
    starter_label_review_csv_out: Path | None,
    production_label_review_csv_out: Path | None,
    production_starter_label_review_csv_out: Path | None,
) -> str:
    capture_deficit = report.get("capture_deficit") if isinstance(report.get("capture_deficit"), dict) else {}
    production_deficit = (
        report.get("production_capture_deficit")
        if isinstance(report.get("production_capture_deficit"), dict)
        else {}
    )
    pilot_batches = [
        batch for batch in capture_deficit.get("next_capture_batches") or []
        if isinstance(batch, dict)
    ]
    production_batches = [
        batch for batch in production_deficit.get("next_capture_batches") or []
        if isinstance(batch, dict)
    ]

    lines = [
        "# Apron/Harness Controlled Capture Kickoff",
        "",
        "This is a controlled-capture handoff for the closed-set apron and harness production path. It does not approve public datasets, does not allow training from unreviewed footage, and does not unlock production sales claims.",
        "",
        "## Open First",
        "",
        f"- Full work order: `{_rel(capture_work_order_out) if capture_work_order_out else 'not generated'}`",
        f"- Pilot capture matrix: `{_rel(capture_matrix_csv_out) if capture_matrix_csv_out else 'not generated'}`",
        f"- Production capture matrix: `{_rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else 'not generated'}`",
        f"- Pilot starter label-review CSV: `{_rel(starter_label_review_csv_out) if starter_label_review_csv_out else 'not generated'}`",
        f"- Pilot label-review CSV: `{_rel(label_review_csv_out) if label_review_csv_out else 'not generated'}`",
        f"- Production starter label-review CSV: `{_rel(production_starter_label_review_csv_out) if production_starter_label_review_csv_out else 'not generated'}`",
        f"- Production label-review CSV: `{_rel(production_label_review_csv_out) if production_label_review_csv_out else 'not generated'}`",
        f"- Capture manifest: `{_rel(capture_manifest)}`",
        f"- Manifest SHA256: `{report.get('manifest_sha256', 'unknown')}`",
        "",
        "## Targets",
        "",
        "| Mode | Required Labels Per Class | Missing Label Annotations | Label-Review Rows |",
        "|---|---:|---:|---:|",
        "| pilot | "
        f"{MIN_COUNTS['pilot']} | "
        f"{capture_deficit.get('total_missing_label_annotations', 0)} | "
        f"{capture_deficit.get('recommended_label_review_rows', 0)} |",
        "| production | "
        f"{MIN_COUNTS['production']} | "
        f"{production_deficit.get('total_missing_label_annotations', 0)} | "
        f"{production_deficit.get('recommended_label_review_rows', 0)} |",
        "",
        "## Immediate Starter Rows",
        "",
    ]
    pilot_starter = _render_capture_starter_rows(
        "Pilot Starter Rows",
        pilot_batches,
        matrix_path=capture_matrix_csv_out,
        rows=_capture_matrix_starter_rows_or_none(capture_matrix_csv_out),
    )
    production_starter = _render_capture_starter_rows(
        "Production Starter Rows",
        production_batches,
        matrix_path=production_capture_matrix_csv_out,
        rows=_capture_matrix_starter_rows_or_none(production_capture_matrix_csv_out),
    )
    if pilot_starter or production_starter:
        lines.extend(pilot_starter)
        lines.extend(production_starter)
    else:
        lines.extend(["- No starter rows were generated from the capture batches.", ""])
    lines.extend([
        "## Starter Validation Loop",
        "",
        "- Use this loop only after the four production starter rows have reviewed labels, cleared permission, and non-repo raw storage references.",
        "- The validation command must print `LABEL_REVIEW_VALIDATION: gate=pass` before import.",
        "- The import must write `qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json` next to the reviewed manifest, and the controlled-capture runner must validate that sidecar before treating the import as successful.",
        "- The starter-reviewed manifest is an intermediate progress checkpoint only; it is not enough for production training or promotion.",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        "  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \\",
        "  --mode production \\",
        "  --schema-only \\",
        "  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \\",
        "  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \\",
        "  --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
        "",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        "  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \\",
        "  --mode production \\",
        "  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \\",
        "  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \\",
        "  --import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv \\",
        "  --emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml",
        "",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        "  --manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml \\",
        "  --mode production \\",
        "  --schema-only \\",
        "  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \\",
        "  --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml \\",
        "  --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
        "```",
        "",
        "## Capture Rules",
        "",
        "- Capture or import footage only with explicit commercial permission.",
        "- Store raw identifiable videos and full frame exports outside the git repo.",
        "- Use a remote storage URI or an absolute external mount for `raw_storage_ref`; relative paths and paths under this repo are treated as repo-local and blocked.",
        "- Commit only manifests, CSVs, sidecars, redacted checks, and result summaries.",
        "- Every usable image row needs `source_clip_id`, `image_path`, `label_path`, `review_status=approved`, `reviewer`, `reviewed_at`, `permission`, and `raw_storage_ref`.",
        "- Label object boxes only: `person`, `apron`, `safety_harness`, and `safety_lanyard`. Missing PPE is derived by runtime association, not by missing-PPE label classes.",
        "- Keep active-window and inactive-window clips as separate YAML scenarios; production claims require detector-window telemetry with zero candidates and zero model invocations outside active windows.",
        "- Train only a reviewed YOLO26 nano/small candidate for this production path. YOLO11/YOLOv8 sources remain legacy evidence or rejected shortcuts.",
        "",
        "## Pilot Batches",
        "",
    ])
    for batch in pilot_batches:
        lines.extend(_render_capture_batch_summary(batch))

    lines.extend(
        [
            "## Production Batches",
            "",
        ]
    )
    for batch in production_batches:
        lines.extend(_render_capture_batch_summary(batch))

    lines.extend(
        [
            "## Validation Commands",
            "",
            "```bash",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
            "  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \\",
            "  --schema-only \\",
            "  --validate-capture-matrix-csv qa/video_eval/results/apron_harness_capture_matrix.csv",
            "",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
            "  --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml \\",
            "  --mode production \\",
            "  --schema-only \\",
            "  --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            "",
            ".venv/bin/python scripts/apron_harness_readiness_doctor.py \\",
            "  --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json \\",
            "  --seed-source-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json \\",
            "  --seed-import-manifest qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml \\",
            "  --capture-kickoff-out qa/video_eval/results/apron_harness_capture_kickoff.md \\",
            "  --out qa/video_eval/results/apron_harness_readiness_doctor.json",
            "```",
            "",
            "## Stop Conditions",
            "",
            "- Stop before training if permission is unknown, raw storage is inside the repo, labels are not reviewed, or matrix totals are not reconciled to the manifest.",
            "- Stop and ask before using paid, gated, academic-only, customer-private, or identifiable third-party footage.",
            "- Keep production blocked until reviewed capture/import data, YOLO26n/s training, side-by-side promotion, model registry, active-window telemetry, false-positive guards, and Jetson three-camera gates all pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _seed_source_next_review_queue(
    seed_source_review: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    queue = seed_source_review.get("review_queue") if isinstance(seed_source_review, dict) else []
    if not isinstance(queue, list):
        return []

    compact: list[dict[str, Any]] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        artifacts = item.get("review_artifacts") if isinstance(item.get("review_artifacts"), dict) else {}
        legal_references = item.get("legal_references") if isinstance(item.get("legal_references"), dict) else {}
        fill_plan = (
            item.get("seed_import_fill_plan")
            if isinstance(item.get("seed_import_fill_plan"), dict)
            else {}
        )
        compact.append(
            {
                "review_priority": item.get("review_priority"),
                "source_ref": item.get("source_ref"),
                "source_url": item.get("source_url") or item.get("url") or legal_references.get("source_url"),
                "url": item.get("source_url") or item.get("url") or legal_references.get("source_url"),
                "checked": item.get("checked"),
                "license_note": item.get("license_note") or legal_references.get("license_note"),
                "capability": item.get("capability"),
                "approval_status": item.get("approval_status"),
                "training_usable": item.get("training_usable") is True,
                "next_action": item.get("next_action"),
                "blocker": item.get("blocker"),
                "required_review": item.get("required_review") if isinstance(item.get("required_review"), list) else [],
                "review_packet_path": artifacts.get("review_packet_path"),
                "review_packet_sha256": artifacts.get("review_packet_sha256"),
                "review_evidence_template_path": artifacts.get("review_evidence_template_path"),
                "review_evidence_template_sha256": artifacts.get("review_evidence_template_sha256"),
                "review_prefill_path": artifacts.get("review_prefill_path"),
                "review_prefill_sha256": artifacts.get("review_prefill_sha256"),
                "review_checklist_csv_path": artifacts.get("review_checklist_csv_path"),
                "review_checklist_csv_sha256": artifacts.get("review_checklist_csv_sha256"),
                "seed_import_manifest_template_path": artifacts.get("seed_import_manifest_template_path"),
                "seed_import_manifest_template_sha256": artifacts.get("seed_import_manifest_template_sha256"),
                "seed_import_fill_plan": {
                    "non_approving": fill_plan.get("non_approving") is True,
                    "required_local_classes": list(fill_plan.get("required_local_classes") or []),
                    "reviewed_class_mapping_starter": dict(
                        fill_plan.get("reviewed_class_mapping_starter") or {}
                    ),
                    "missing_required_classes_from_suggestion": list(
                        fill_plan.get("missing_required_classes_from_suggestion") or []
                    ),
                    "expected_count_classes_that_must_be_nonzero": list(
                        fill_plan.get("expected_count_classes_that_must_be_nonzero") or []
                    ),
                    "required_fields_before_include_in_training": list(
                        fill_plan.get("required_fields_before_include_in_training") or []
                    ),
                },
            }
        )
        if len(compact) >= limit:
            break
    return compact


def _seed_source_coverage_summary(seed_source_review: dict[str, Any]) -> dict[str, Any]:
    plan = seed_source_review.get("source_coverage_plan") if isinstance(seed_source_review, dict) else {}
    if not isinstance(plan, dict) or not plan:
        return {
            "available": False,
            "coverage_gap_count": None,
            "training_usable_count": 0,
            "capabilities": {},
            "next_action": "rerun apron_harness_seed_source_doctor with --source-coverage-plan-out",
        }
    artifact_path = plan.get("path")
    artifact_sha256 = plan.get("sha256")
    if "capabilities" not in plan and artifact_path:
        coverage_path = Path(str(artifact_path))
        if not coverage_path.is_absolute():
            coverage_path = ROOT / coverage_path
        try:
            loaded_plan = _load_json(coverage_path)
            if isinstance(loaded_plan, dict):
                plan = loaded_plan
        except Exception:
            pass

    capabilities: dict[str, Any] = {}
    for capability in ("apron_required", "harness_required"):
        raw = (plan.get("capabilities") or {}).get(capability)
        if not isinstance(raw, dict):
            continue
        person_box = raw.get("person_box_reconciliation") if isinstance(raw.get("person_box_reconciliation"), dict) else {}
        priority_plan = raw.get("priority_coverage_plan") if isinstance(raw.get("priority_coverage_plan"), dict) else {}
        selected_sources: list[dict[str, Any]] = []
        for item in priority_plan.get("selected_sources") or []:
            if not isinstance(item, dict):
                continue
            selected_sources.append(
                {
                    "source_ref": item.get("source_ref"),
                    "review_priority": item.get("review_priority"),
                    "mapped_local_classes": item.get("mapped_local_classes") or [],
                    "missing_local_classes": item.get("missing_local_classes") or [],
                    "training_usable": item.get("training_usable") is True,
                    "approval_status": item.get("approval_status"),
                }
            )
        capabilities[capability] = {
            "required_local_classes": raw.get("required_local_classes") or [],
            "reviewable_source_count": int(raw.get("reviewable_source_count") or 0),
            "missing_local_classes_across_reviewable_sources": (
                raw.get("missing_local_classes_across_reviewable_sources") or []
            ),
            "complete_single_source_refs": raw.get("complete_single_source_refs") or [],
            "person_box_reconciliation_status": person_box.get("status"),
            "person_box_candidate_source_refs": person_box.get("candidate_person_source_refs") or [],
            "priority_coverage_status": priority_plan.get("status"),
            "priority_selected_sources": selected_sources,
        }

    return {
        "available": True,
        "kind": plan.get("kind"),
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "source_review_report": plan.get("source_review_report"),
        "source_review_sha256": plan.get("source_review_sha256"),
        "source_review_gate_passed": plan.get("source_review_gate_passed"),
        "candidate_count": int(plan.get("candidate_count") or 0),
        "training_usable_count": int(plan.get("training_usable_count") or 0),
        "coverage_gap_count": int(plan.get("coverage_gap_count") or 0),
        "missing_target_classes_across_reviewable_sources": (
            plan.get("missing_target_classes_across_reviewable_sources") or []
        ),
        "next_action": plan.get("next_action"),
        "capabilities": capabilities,
    }


def _seed_source_minimum_approval_path(
    seed_source_coverage: dict[str, Any],
    seed_source_next_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    review_by_source = {
        str(item.get("source_ref")): item
        for item in seed_source_next_reviews
        if isinstance(item, dict) and item.get("source_ref")
    }
    capabilities = seed_source_coverage.get("capabilities") if isinstance(seed_source_coverage, dict) else {}
    if not isinstance(capabilities, dict) or not capabilities:
        return {
            "checked": False,
            "available": False,
            "coverage_gap_count": seed_source_coverage.get("coverage_gap_count")
            if isinstance(seed_source_coverage, dict)
            else None,
            "training_usable_count": seed_source_coverage.get("training_usable_count", 0)
            if isinstance(seed_source_coverage, dict)
            else 0,
            "capabilities": {},
            "minimum_review_source_refs": [],
            "approval_ready": False,
            "blockers": ["seed_source_coverage_summary_missing"],
            "evidence_boundary": (
                "agent-selected minimum path is not approval; rerun seed-source coverage and complete "
                "human/legal review before training."
            ),
        }

    minimum_refs: set[str] = set()
    path_capabilities: dict[str, Any] = {}
    blockers: set[str] = {"human_legal_review_required"}
    for capability in ("apron_required", "harness_required"):
        item = capabilities.get(capability)
        if not isinstance(item, dict):
            blockers.add(f"{capability}_coverage_missing")
            continue
        selected_sources = []
        for source in item.get("priority_selected_sources") or []:
            if not isinstance(source, dict):
                continue
            source_ref = source.get("source_ref")
            if source_ref:
                minimum_refs.add(str(source_ref))
            review = review_by_source.get(str(source_ref)) if source_ref else {}
            selected_sources.append(
                {
                    "source_ref": source_ref,
                    "review_priority": source.get("review_priority"),
                    "mapped_local_classes": source.get("mapped_local_classes") or [],
                    "missing_local_classes": source.get("missing_local_classes") or [],
                    "training_usable": source.get("training_usable") is True,
                    "approval_status": source.get("approval_status"),
                    "source_url": review.get("source_url") or review.get("url") if review else None,
                    "license_note": review.get("license_note") if review else None,
                    "review_packet_path": review.get("review_packet_path") if review else None,
                    "review_evidence_template_path": review.get("review_evidence_template_path") if review else None,
                    "next_action": review.get("next_action") if review else None,
                }
            )
        if not selected_sources:
            blockers.add(f"{capability}_minimum_source_missing")
        if any(source.get("missing_local_classes") for source in selected_sources):
            blockers.add(f"{capability}_selected_source_class_gap")
        if any(source.get("training_usable") is not True for source in selected_sources):
            blockers.add("selected_sources_not_training_usable")
        if any(source.get("approval_status") != "approved_for_training" for source in selected_sources):
            blockers.add("selected_sources_not_approved_for_training")
        path_capabilities[capability] = {
            "required_local_classes": item.get("required_local_classes") or [],
            "complete_single_source_refs": item.get("complete_single_source_refs") or [],
            "selected_source_count": len(selected_sources),
            "selected_sources": selected_sources,
            "all_selected_sources_training_usable": bool(selected_sources)
            and all(source.get("training_usable") is True for source in selected_sources),
            "approval_ready": bool(selected_sources)
            and all(source.get("approval_status") == "approved_for_training" for source in selected_sources),
        }

    return {
        "checked": True,
        "available": True,
        "coverage_gap_count": seed_source_coverage.get("coverage_gap_count"),
        "training_usable_count": seed_source_coverage.get("training_usable_count", 0),
        "capabilities": path_capabilities,
        "minimum_review_source_refs": sorted(minimum_refs),
        "approval_ready": False,
        "blockers": sorted(blockers),
        "evidence_boundary": (
            "agent-selected minimum path is not approval; every selected source still requires filled "
            "human/legal review evidence, checklist SHA, reviewed model_packs, and seed import "
            "validation before training."
        ),
    }


def _seed_import_export_preflight_summary(seed_import_review: dict[str, Any]) -> dict[str, Any]:
    imports = seed_import_review.get("imports") if isinstance(seed_import_review, dict) else []
    if not isinstance(imports, list):
        imports = []
    included = [
        item for item in imports
        if isinstance(item, dict) and item.get("include_in_training") is True
    ]
    preflight_checked = [
        item for item in included
        if isinstance(item.get("yolo_export_preflight"), dict)
        and item["yolo_export_preflight"].get("checked") is True
    ]
    preflight_approved = [
        item for item in preflight_checked
        if item.get("approved_for_training") is True
    ]
    artifact_checked = [
        item for item in included
        if isinstance(item.get("review_artifact_preflight"), dict)
        and item["review_artifact_preflight"].get("checked") is True
    ]
    artifact_approved = [
        item for item in artifact_checked
        if not item["review_artifact_preflight"].get("errors")
    ]
    artifact_error_count = sum(
        len(item["review_artifact_preflight"].get("errors") or [])
        for item in artifact_checked
    )
    missing_local_path = [
        item for item in included
        if not str(item.get("raw_export_local_path") or "").strip()
    ]
    blockers = [
        str(blocker)
        for blocker in seed_import_review.get("blockers") or []
        if str(blocker).strip()
    ]
    aggregate_blockers = [blocker for blocker in blockers if not blocker.startswith("imports[")]
    row_blockers = [blocker for blocker in blockers if blocker.startswith("imports[")]
    top_blockers = [*aggregate_blockers, *row_blockers][:5]
    if seed_import_review.get("gate_passed") is True:
        blocked_reason = "passed"
    elif not included:
        blocked_reason = "no_seed_imports_included_for_training"
    elif not preflight_checked:
        blocked_reason = "no_included_imports_have_reviewed_export_preflight"
    elif not preflight_approved:
        blocked_reason = "no_included_imports_have_approved_export_preflight"
    elif artifact_error_count:
        blocked_reason = "review_artifact_preflight_errors"
    else:
        blocked_reason = "seed_import_gate_failed"
    return {
        "required_for_training": True,
        "required_manifest_field": "raw_export_local_path",
        "required_export_format": "yolo",
        "checks": [
            "local_reviewed_export_zip_exists",
            "local_export_sha256_matches_raw_export_sha256",
            "data_yaml_class_names_include_mapped_source_labels",
            "train_valid_test_images_and_labels_present",
            "required_local_class_label_file_counts_meet_expected_counts",
            "review_packet_path_and_sha256_match_seed_source_review",
            "review_evidence_template_path_and_sha256_match_seed_source_review",
            "generated_review_artifact_files_exist_and_hash_match",
        ],
        "import_count": len(imports),
        "included_count": len(included),
        "preflight_checked_count": len(preflight_checked),
        "preflight_approved_count": len(preflight_approved),
        "review_artifact_checked_count": len(artifact_checked),
        "review_artifact_approved_count": len(artifact_approved),
        "review_artifact_error_count": artifact_error_count,
        "missing_raw_export_local_path_count": len(missing_local_path),
        "gate_passed": seed_import_review.get("gate_passed") is True,
        "blocked_reason": blocked_reason,
        "blocker_count": len(blockers),
        "top_blockers": top_blockers,
    }


def _seed_import_fill_contract_summary(seed_import_manifest: Path | None) -> dict[str, Any]:
    if seed_import_manifest is None:
        return {"available": False, "path": None}
    result: dict[str, Any] = {
        "available": False,
        "path": _rel(seed_import_manifest),
        "approval_boundary": "",
        "required_before_include_in_training_count": 0,
        "forbidden_until_approved": [],
        "validation_commands": [],
    }
    if not seed_import_manifest.exists():
        return result
    try:
        manifest = _load_yaml(seed_import_manifest)
    except Exception as exc:
        result["error"] = str(exc)
        return result
    contract = manifest.get("fill_contract") if isinstance(manifest.get("fill_contract"), dict) else {}
    required = contract.get("required_before_include_in_training")
    forbidden = contract.get("forbidden_until_approved")
    commands = contract.get("validation_commands")
    result.update(
        {
            "available": bool(contract),
            "approval_boundary": str(contract.get("approval_boundary") or ""),
            "required_before_include_in_training_count": len(required) if isinstance(required, list) else 0,
            "forbidden_until_approved": [str(item) for item in forbidden] if isinstance(forbidden, list) else [],
            "validation_commands": [str(item) for item in commands] if isinstance(commands, list) else [],
        }
    )
    return result


def _seed_import_template_summary(seed_import_manifest: Path | None) -> dict[str, Any]:
    if seed_import_manifest is None:
        return {"available": False, "path": None}
    result: dict[str, Any] = {
        "available": False,
        "path": _rel(seed_import_manifest),
        "template_scope": "",
        "selected_source_refs": [],
        "import_count": 0,
        "enabled_import_count": 0,
        "approval_boundary": "",
    }
    if not seed_import_manifest.exists():
        return result
    try:
        manifest = _load_yaml(seed_import_manifest)
    except Exception as exc:
        result["error"] = str(exc)
        return result
    imports = manifest.get("imports") if isinstance(manifest.get("imports"), list) else []
    selected_refs = manifest.get("selected_source_refs")
    if not isinstance(selected_refs, list):
        selected_refs = [
            item.get("source_ref")
            for item in imports
            if isinstance(item, dict) and item.get("source_ref")
        ]
    contract = manifest.get("fill_contract") if isinstance(manifest.get("fill_contract"), dict) else {}
    result.update(
        {
            "available": True,
            "template_scope": str(manifest.get("template_scope") or "all_reviewable_sources"),
            "selected_source_refs": [str(value) for value in selected_refs if str(value)],
            "import_count": len([item for item in imports if isinstance(item, dict)]),
            "enabled_import_count": len(
                [
                    item
                    for item in imports
                    if isinstance(item, dict) and item.get("include_in_training") is True
                ]
            ),
            "approval_boundary": str(contract.get("approval_boundary") or ""),
        }
    )
    return result


def _minimum_seed_import_template_consistency(
    minimum_path: dict[str, Any],
    template_summary: dict[str, Any],
) -> dict[str, Any]:
    expected_refs = sorted(
        str(value)
        for value in minimum_path.get("minimum_review_source_refs") or []
        if str(value)
    )
    template_refs = sorted(
        str(value)
        for value in template_summary.get("selected_source_refs") or []
        if str(value)
    )
    blockers: list[str] = []
    if template_summary.get("available") is not True:
        blockers.append("minimum_seed_import_template_missing")
    if template_summary.get("template_scope") != "minimum_priority_coverage_sources":
        blockers.append("minimum_seed_import_template_scope_mismatch")
    if expected_refs != template_refs:
        blockers.append("minimum_seed_import_template_source_refs_mismatch")
    if int(template_summary.get("import_count") or 0) != len(expected_refs):
        blockers.append("minimum_seed_import_template_import_count_mismatch")
    if int(template_summary.get("enabled_import_count") or 0) != 0:
        blockers.append("minimum_seed_import_template_has_enabled_training_rows")
    return {
        "checked": True,
        "valid": not blockers,
        "expected_source_refs": expected_refs,
        "template_source_refs": template_refs,
        "source_refs_match": expected_refs == template_refs,
        "template_scope": template_summary.get("template_scope"),
        "import_count": int(template_summary.get("import_count") or 0),
        "enabled_import_count": int(template_summary.get("enabled_import_count") or 0),
        "blockers": blockers,
        "evidence_boundary": (
            "Consistency only proves the minimum import template matches the selected review path "
            "and keeps training disabled; it does not approve any source."
        ),
    }


def _evidence(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("evidence")
    return value if isinstance(value, dict) else {}


def _analytics_summary(result: dict[str, Any]) -> dict[str, Any]:
    value = _evidence(result).get("analytics_summary")
    return value if isinstance(value, dict) else {}


def _schedule(result: dict[str, Any]) -> dict[str, Any]:
    value = _analytics_summary(result).get("schedule")
    return value if isinstance(value, dict) else {}


def _class_counts(result: dict[str, Any]) -> dict[str, int]:
    counts = _analytics_summary(result).get("class_counts")
    if not isinstance(counts, dict):
        counts = _evidence(result).get("max_detection_class_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        try:
            normalized[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


def _visible_class_total(result: dict[str, Any], aliases: set[str]) -> int:
    total = 0
    for class_name, count in _class_counts(result).items():
        normalized = class_name.lower().replace("-", " ").strip()
        if normalized in aliases:
            total += count
    return total


def _model_invocations(result: dict[str, Any]) -> dict[str, int]:
    raw = _schedule(result).get("model_invocations")
    if not isinstance(raw, dict):
        return {}
    invocations: dict[str, int] = {}
    for key, value in raw.items():
        try:
            invocations[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            invocations[str(key)] = 0
    return invocations


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


def _screenshot_ok(result: dict[str, Any]) -> bool:
    ui = _evidence(result).get("ui_evidence")
    if not isinstance(ui, dict):
        return False
    return bool(ui.get("screenshot_exists") and ui.get("screenshot_fresh"))


def _load_result(result_dir: Path, scenario_id: str, errors: list[str]) -> dict[str, Any]:
    path = result_dir / f"{scenario_id}.json"
    if not path.exists():
        errors.append(f"{scenario_id}: missing result JSON at {_rel(path)}")
        return {}
    try:
        return _load_json(path)
    except Exception as exc:
        errors.append(f"{scenario_id}: result JSON is unreadable: {exc}")
        return {}


def _scenario_summary(result: dict[str, Any], capability: str, aliases: set[str]) -> dict[str, Any]:
    return {
        "scenario_id": result.get("scenario_id"),
        "status": result.get("status"),
        "max_detections": _max_detections(result),
        "matching_alerts": _matching_alert_count(result),
        "unexpected_alerts": _unexpected_alert_count(result),
        "visible_class_total": _visible_class_total(result, aliases),
        "suppressed_capabilities": _schedule(result).get("suppressed_capabilities") or [],
        "model_invocations": _model_invocations(result),
        "screenshot_ok": _screenshot_ok(result),
        "capability": capability,
    }


def _candidate_required_evidence(role: str, capability: str) -> list[str]:
    common = [
        "YAML validate/plan/apply command logs are recorded in the result JSON",
        "YAML validate/plan/apply commands all returned exit code 0",
        "required-model preflight confirms ppe_closed_set_candidate is ready before polling",
        "fresh live-view screenshot is captured",
        "runtime schedule telemetry includes ppe_closed_set_candidate model_invocations",
    ]
    if role == "active":
        return [
            *common,
            "max_detections_count > 0",
            "matching alert count > 0",
            "delivery_summary proves in_app/browser_sound delivery succeeded or was explicitly simulated",
            "unexpected alert count == 0",
            f"{capability} active capability window is open",
        ]
    if role == "false_positive_guard":
        return [
            *common,
            "max_detections_count > 0 on visible non-violation footage",
            "visible target-class telemetry is nonzero",
            "matching alert count == 0",
            "unexpected alert count == 0",
            f"{capability} active capability window is open",
        ]
    if role == "suppression":
        return [
            *common,
            "max_detections_count == 0",
            "matching alert count == 0",
            "unexpected alert count == 0",
            f"{capability} appears in suppressed_capabilities",
            "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window",
        ]
    return common


def _candidate_observed_evidence(result: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    ui = evidence.get("ui_evidence") if isinstance(evidence.get("ui_evidence"), dict) else {}
    delivery = evidence.get("delivery_summary") if isinstance(evidence.get("delivery_summary"), dict) else {}
    stream = evidence.get("stream_probe") if isinstance(evidence.get("stream_probe"), dict) else {}
    yaml_commands = result.get("yaml_commands") if isinstance(result.get("yaml_commands"), list) else []
    yaml_command_names = [
        _yaml_command_action(command)
        for command in yaml_commands
        if isinstance(command, dict)
    ]
    yaml_command_returncodes = [
        command.get("returncode")
        for command in yaml_commands
        if isinstance(command, dict)
    ]
    expected_yaml_commands = {"validate", "plan", "apply"}
    yaml_successful_command_count = sum(1 for returncode in yaml_command_returncodes if returncode == 0)
    return {
        "yaml_command_log_count": len(yaml_commands),
        "yaml_successful_command_count": yaml_successful_command_count,
        "yaml_commands_ok": (
            len(yaml_commands) >= 3
            and yaml_successful_command_count == len(yaml_commands)
            and expected_yaml_commands.issubset(set(yaml_command_names))
        ),
        "yaml_command_names": yaml_command_names,
        "model_preflight_ok": (evidence.get("model_preflight") or {}).get("ok")
        if isinstance(evidence.get("model_preflight"), dict)
        else None,
        "stream_ok": stream.get("ok"),
        "screenshot_path": ui.get("screenshot_path"),
        "screenshot_fresh": ui.get("screenshot_fresh"),
        "delivery_outputs": sorted(str(key) for key in delivery.keys()),
        "max_detections": summary.get("max_detections"),
        "matching_alerts": summary.get("matching_alerts"),
        "unexpected_alerts": summary.get("unexpected_alerts"),
        "suppressed_capabilities": summary.get("suppressed_capabilities") or [],
        "model_invocations": summary.get("model_invocations") or {},
    }


def _normalize_artifact_string(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(_artifact_path(value).resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(value)


def _yaml_command_args(command: Any) -> list[str]:
    if not isinstance(command, dict):
        return []
    raw_args = command.get("args")
    if not isinstance(raw_args, list):
        return []
    return [str(arg) for arg in raw_args]


def _yaml_command_action(command: Any) -> str:
    args = _yaml_command_args(command)
    actions = [arg for arg in args if arg in {"validate", "plan", "apply"}]
    return actions[-1] if actions else ""


def _yaml_command_config_path(command: Any) -> str:
    args = _yaml_command_args(command)
    if "--config" not in args:
        return ""
    index = args.index("--config")
    if index + 1 >= len(args):
        return ""
    return _normalize_artifact_string(args[index + 1])


def _candidate_runtime_identity_error(
    *,
    result: dict[str, Any],
    template: dict[str, Any],
    observed: dict[str, Any],
) -> str | None:
    scenario_id = str(template.get("scenario_id") or "")
    if result.get("scenario_id") != scenario_id:
        return "candidate_runtime_scenario_id_mismatch"
    manifest_path = _artifact_path(result.get("manifest_path") or "qa/video_eval/manifest.yaml")
    if not manifest_path.exists():
        return "candidate_runtime_manifest_missing_for_sha_check"
    if result.get("manifest_sha256") != _sha256_file(manifest_path):
        return "candidate_runtime_manifest_sha256_mismatch"
    try:
        manifest = _load_yaml(manifest_path)
    except Exception:
        return "candidate_runtime_manifest_unreadable_for_video_check"
    scenarios = manifest.get("scenarios") if isinstance(manifest.get("scenarios"), list) else []
    scenario = next(
        (
            item for item in scenarios
            if isinstance(item, dict) and str(item.get("id") or "") == scenario_id
        ),
        None,
    )
    if not isinstance(scenario, dict):
        return "candidate_runtime_manifest_scenario_missing_for_video_check"
    expected_camera_id = str(scenario.get("camera_id") or "")
    if expected_camera_id and result.get("camera_id") != expected_camera_id:
        return "candidate_runtime_camera_id_mismatch"
    expected_source = scenario.get("source") if isinstance(scenario.get("source"), dict) else {}
    result_source = result.get("source") if isinstance(result.get("source"), dict) else {}
    if expected_source != result_source:
        return "candidate_runtime_source_metadata_mismatch"
    expected_video = scenario.get("local_video")
    if expected_video and _normalize_artifact_string(result.get("video")) != _normalize_artifact_string(expected_video):
        return "candidate_runtime_video_path_mismatch"
    video_path = _artifact_path(result.get("video"))
    if not video_path.exists():
        return "candidate_runtime_video_missing_for_sha_check"
    if result.get("video_sha256") != _sha256_file(video_path):
        return "candidate_runtime_video_sha256_mismatch"
    expected_config_path = _normalize_artifact_string(template.get("config_path"))
    if expected_config_path:
        result_config_path = _normalize_artifact_string(result.get("config_path"))
        if result_config_path != expected_config_path:
            return "candidate_runtime_config_path_mismatch"
        config_path = _artifact_path(template.get("config_path"))
        if not config_path.exists():
            return "candidate_runtime_config_missing_for_sha_check"
        if result.get("config_sha256") != _sha256_file(config_path):
            return "candidate_runtime_config_sha256_mismatch"
    if observed.get("yaml_commands_ok") is not True:
        return "candidate_runtime_yaml_validate_plan_apply_missing_or_failed"
    ordered_commands = [
        command
        for command in (result.get("yaml_commands") or [])
        if isinstance(command, dict) and _yaml_command_action(command)
    ]
    ordered_actions = [_yaml_command_action(command) for command in ordered_commands]
    if ordered_actions[:3] != ["validate", "plan", "apply"]:
        return "candidate_runtime_yaml_validate_plan_apply_order_mismatch"
    ordered_by_action = dict(zip(ordered_actions[:3], ordered_commands[:3], strict=True))
    for action in ("validate", "plan", "apply"):
        command = ordered_by_action.get(action)
        if command is None:
            return f"candidate_runtime_yaml_{action}_missing"
        if expected_config_path and _yaml_command_config_path(command) != expected_config_path:
            return f"candidate_runtime_yaml_{action}_config_path_mismatch"
        if command.get("returncode") != 0:
            return f"candidate_runtime_yaml_{action}_failed"
    apply_args = _yaml_command_args(ordered_by_action.get("apply"))
    if "--yes" not in apply_args:
        return "candidate_runtime_yaml_apply_missing_yes"
    return None


def _candidate_model_preflight_error(
    scenario_id: str,
    result: dict[str, Any],
    model_preflight: dict[str, Any],
) -> str | None:
    if model_preflight.get("checked") is not True:
        return "missing_required_model_preflight"
    if model_preflight.get("ok") is not True:
        return "required_model_preflight_not_ok"
    required_model_keys = {
        str(model_key) for model_key in (model_preflight.get("required_model_keys") or [])
    }
    if PLANNED_MODEL_KEY not in required_model_keys:
        return f"required_model_preflight_missing_required_key:{PLANNED_MODEL_KEY}"
    missing_model_keys = {
        str(model_key) for model_key in (model_preflight.get("missing_model_keys") or [])
    }
    if missing_model_keys:
        return "required_model_preflight_has_missing_models"
    return None


def _candidate_invocation_isolation_error(summary: dict[str, Any]) -> str | None:
    invocations = _summary_model_invocations(summary)
    if invocations.get(PILOT_MODEL_KEY, 0) > 0:
        return f"pilot_model_invoked_in_candidate_runtime:{PILOT_MODEL_KEY}"
    return None


def _validate_active(label: str, result: dict[str, Any], errors: list[str]) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label}: status must be {READY_STATUS}")
    if _max_detections(result) <= 0:
        errors.append(f"{label}: must have detections")
    if _matching_alert_count(result) <= 0:
        errors.append(f"{label}: must have at least one matching alert")
    if _unexpected_alert_count(result) != 0:
        errors.append(f"{label}: must have zero unexpected alerts")
    if not _screenshot_ok(result):
        errors.append(f"{label}: must include a fresh screenshot")


def _validate_guard(label: str, result: dict[str, Any], aliases: set[str], errors: list[str]) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label}: status must be {READY_STATUS}")
    if _max_detections(result) <= 0:
        errors.append(f"{label}: must have detections")
    if _matching_alert_count(result) != 0:
        errors.append(f"{label}: must have zero matching alerts")
    if _unexpected_alert_count(result) != 0:
        errors.append(f"{label}: must have zero unexpected alerts")
    if _visible_class_total(result, aliases) <= 0:
        errors.append(f"{label}: must include visible PPE class telemetry")
    if _model_invocations(result).get(PILOT_MODEL_KEY, 0) <= 0:
        errors.append(f"{label}: must report {PILOT_MODEL_KEY} invocation telemetry")
    if not _screenshot_ok(result):
        errors.append(f"{label}: must include a fresh screenshot")


def _validate_suppression(label: str, result: dict[str, Any], capability: str, errors: list[str]) -> None:
    if result.get("status") != READY_STATUS:
        errors.append(f"{label}: status must be {READY_STATUS}")
    if _max_detections(result) != 0:
        errors.append(f"{label}: must emit zero detections")
    if _matching_alert_count(result) != 0 or _unexpected_alert_count(result) != 0:
        errors.append(f"{label}: must emit zero alerts")
    suppressed = set(str(value) for value in (_schedule(result).get("suppressed_capabilities") or []))
    if capability not in suppressed:
        errors.append(f"{label}: must suppress {capability}")
    invocations = _model_invocations(result)
    if PILOT_MODEL_KEY not in invocations:
        errors.append(f"{label}: must report model invocation telemetry for {PILOT_MODEL_KEY}")
    elif invocations.get(PILOT_MODEL_KEY) != 0:
        errors.append(f"{label}: must report zero {PILOT_MODEL_KEY} invocations")
    if not _screenshot_ok(result):
        errors.append(f"{label}: must include a fresh screenshot")


def _basename(value: Any) -> str:
    return Path(str(value or "")).name


def _md_cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|").replace("`", "'").strip()


def _compact_string_list(value: Any, *, limit: int = 8) -> str:
    if not isinstance(value, list):
        return ""
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if len(normalized) > limit:
        normalized = [*normalized[:limit], f"+{len(normalized) - limit} more"]
    return ", ".join(normalized)


def _compact_mapping(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for key, raw_labels in sorted(value.items()):
        labels = _compact_string_list(raw_labels) if isinstance(raw_labels, list) else str(raw_labels or "")
        parts.append(f"{key}={labels}")
    return "; ".join(parts)


def _seed_import_fill_plan_summary(fill_plan: Any) -> str:
    if not isinstance(fill_plan, dict) or not fill_plan:
        return "open fill plan in seed-import template"
    required_classes = _compact_string_list(fill_plan.get("required_local_classes") or [])
    mapping = _compact_mapping(fill_plan.get("reviewed_class_mapping_starter"))
    missing = _compact_string_list(fill_plan.get("missing_required_classes_from_suggestion") or []) or "none"
    nonzero = _compact_string_list(
        fill_plan.get("expected_count_classes_that_must_be_nonzero") or []
    )
    return (
        f"classes: {required_classes}; mapping starter: {mapping}; "
        f"missing from suggestion: {missing}; nonzero counts: {nonzero}"
    )


def _target_class_coverage_summary(value: Any) -> str:
    coverage = value if isinstance(value, dict) else {}
    if not coverage:
        return "unknown"
    parts = []
    for class_name in ("person", "apron", "safety_harness", "safety_lanyard"):
        status = coverage.get(class_name)
        if status is True:
            rendered = "yes"
        elif status is False:
            rendered = "no"
        else:
            rendered = str(status or "unknown")
        parts.append(f"{class_name}:{rendered}")
    return ", ".join(parts)


def _production_blocker_action(blocker: str) -> dict[str, Any]:
    seed_review_command = (
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
        "--out qa/video_eval/results/apron_harness_seed_source_review.json "
        "--work-order-out qa/video_eval/results/apron_harness_seed_source_review.md "
        "--import-template-out qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml "
        "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml "
        "--review-checklist-csv-out qa/video_eval/results/apron_harness_seed_source_review_checklist.csv "
        "--review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence "
        "--review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets "
        "--next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json "
        "--review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md "
        "--source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json "
        "--review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json"
    )
    seed_next_review_batch_validate_command = (
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
        "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json"
    )
    seed_review_bundle_validate_command = (
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
        "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json"
    )
    seed_review_handoff_command = (
        seed_review_command
        + " && "
        + seed_next_review_batch_validate_command
        + " && "
        + seed_review_bundle_validate_command
    )
    seed_import_validate_command = (
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
        "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json "
        "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json "
        "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml"
    )
    pilot_capture_progress_command = (
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py "
        "--manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml "
        "--mode pilot --schema-only "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
        "--validate-capture-matrix-csv qa/video_eval/results/apron_harness_capture_matrix.csv"
    )
    production_capture_progress_command = (
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py "
        "--manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml "
        "--mode production --schema-only "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
        "--validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv"
    )
    label_review_validate_command = (
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py "
        "--manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml "
        "--mode production --schema-only "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
        "--validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv"
    )
    label_review_import_command = (
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py "
        "--manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml "
        "--mode production "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
        "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv "
        "--emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml"
    )
    training_command = (
        ".venv/bin/python scripts/apron_harness_train.py "
        "--data /path/to/cleared/dataset.yaml "
        "--capture-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml "
        "--capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml "
        "--capture-preflight-mode production --require-capture-preflight "
        "--model yolo26n.pt --device mps --epochs 100 --batch 8 --export-format onnx "
        "--out-plan /path/to/cleared/apron_harness_yolo26n_result.json --execute"
    )
    candidate_command = (
        ".venv/bin/python scripts/apron_harness_candidate_doctor.py "
        "--training-result /path/to/cleared/apron_harness_yolo26n_result.json "
        "--out qa/video_eval/results/apron_harness_candidate_report.json"
    )
    registry_copy_command = (
        ".venv/bin/python scripts/apron_harness_model_registry_doctor.py "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--copy --out qa/video_eval/results/apron_harness_model_registry_report.json"
    )
    jetson_command = (
        ".venv/bin/python scripts/jetson_benchmark_doctor.py "
        "--model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam "
        "--model apron-harness-ppe.onnx "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json "
        "--soak-report /path/to/cleared/factory_ppe_3cam_soak.json "
        "--require-full-gate --out qa/video_eval/results/factory_ppe_jetson_gate.json"
    )
    jetson_raw_template_command = (
        ".venv/bin/python scripts/jetson_benchmark_doctor.py "
        "--model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam "
        "--model apron-harness-ppe.onnx "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--write-raw-template qa/video_eval/results/factory_ppe_raw_benchmark.template.json"
    )
    jetson_soak_template_command = (
        ".venv/bin/python scripts/jetson_benchmark_doctor.py "
        "--model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam "
        "--model apron-harness-ppe.onnx "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--write-soak-template qa/video_eval/results/factory_ppe_3cam_soak.template.json"
    )
    jetson_raw_capture_command = (
        ".venv/bin/python scripts/benchmark_yolo_jetson.py "
        "--models models/ppe_closed_set_candidate/apron-harness-ppe.onnx "
        "--frames-dir /host_tmp "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--out qa/video_eval/results/factory_ppe_raw_benchmark.json"
    )
    jetson_soak_build_command = (
        ".venv/bin/python scripts/jetson_benchmark_doctor.py "
        "--model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam "
        "--model apron-harness-ppe.onnx "
        "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
        "--soak-metrics qa/video_eval/results/factory_ppe_3cam_soak_metrics.yaml "
        "--active-result apron_required=qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json "
        "--active-result harness_required=qa/video_eval/results/closed_set_candidate/factory_missing_harness_active_closed_set.json "
        "--guard-result apron_required=qa/video_eval/results/closed_set_candidate/factory_apron_false_positive_guard_closed_set.json "
        "--guard-result harness_required=qa/video_eval/results/closed_set_candidate/factory_harness_false_positive_guard_closed_set.json "
        "--suppression-result apron_required=qa/video_eval/results/closed_set_candidate/factory_apron_detector_window_suppression_closed_set.json "
        "--suppression-result harness_required=qa/video_eval/results/closed_set_candidate/factory_harness_detector_window_suppression_closed_set.json "
        "--build-soak-report qa/video_eval/results/factory_ppe_3cam_soak.json"
    )
    mps_probe_command = ".venv/bin/python scripts/model_pack_doctor.py --out qa/video_eval/results/model_pack_device_probe.json"
    seed_review_contract = {
        "agent_may_prefill_hints_only": True,
        "human_approval_required": True,
        "required_approvals": [
            "license_review",
            "commercial_use_review",
            "class_mapping_review",
            "training_use_review",
        ],
        "must_not_do": [
            "treat public-page license text as training approval without filled evidence",
            "import Roboflow/public seed data before approved source evidence exists",
            "set approved=true on agent-prefilled evidence templates",
        ],
    }
    seed_review_summary = [
        "human/legal approval evidence is required before public seed data can enter training",
        "agent-prefilled review hints are not approval and must remain approved=false",
        "filled seed-import manifest must point to reviewed local YOLO export ZIPs with SHA-256 values",
    ]
    seed_import_contract = {
        "source_review": [
            "source_review_sha256 matches the current approved seed-source review",
            "source/capability entry is training_usable and approved_for_training in the reviewed model-packs file",
            "reviewed_by, reviewed_at, manifest_import_path, review_evidence_path, and review_evidence_sha256 are present",
        ],
        "review_artifacts": [
            "review_packet_path and review_packet_sha256 match the generated seed-source review packet",
            "review_evidence_template_path and review_evidence_template_sha256 match the generated evidence template",
            "generated review packet and evidence-template files still exist and hash to the recorded SHA-256 values",
        ],
        "export_archive": [
            "raw_export_ref is a remote immutable export reference, not a local workstation path",
            "raw_export_sha256 matches raw_export_local_path",
            "raw_export_local_path is a reviewed local YOLO export ZIP",
            "data.yaml class names plus paired image/label files prove person plus the required target-PPE class counts",
        ],
    }
    seed_import_summary = [
        "filled seed-import manifest must match the current seed-source review fingerprint",
        "generated review packet/template paths and SHA-256 values must be preserved and re-hashed",
        "local reviewed YOLO export ZIP must pass SHA-256, class mapping, split, image/label pairing, and label-count preflight",
    ]
    candidate_artifact_contract = {
        "training_result": [
            "production capture preflight passed",
            "dataset provenance matches reviewed capture manifest SHA",
            "source lineage includes approved seed/import evidence or controlled-capture-only evidence",
            "per-class metrics are present for person, apron, safety_harness, and safety_lanyard",
        ],
        "candidate_report": [
            "ok=true",
            "promotion_manifest.candidate_status=ready_for_side_by_side_runtime_test",
            "runtime_handoff.selected_export.sha256 is present",
            "metric thresholds and per-class mAP50/recall meet promotion minimums",
        ],
        "registry_copy": [
            "selected export SHA matches the copied ONNX artifact",
            "artifact is installed at models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "model registry report uses the same candidate_report_sha256",
        ],
    }
    candidate_artifact_summary = [
        "train/export only from reviewed production data with passed capture preflight",
        "candidate doctor must provide selected export SHA and passing per-class apron/harness metrics",
        "model registry copy must preserve the same candidate_report_sha256 and selected-export SHA",
    ]
    jetson_evidence_contract = {
        "identity": [
            "raw benchmark, soak report, promotion reports, and model-registry report use the same candidate_report_sha256",
            "raw benchmark, soak report, promotion reports, and model-registry report use the same selected-export/model_artifact_sha256",
            "pass --candidate-report so the Jetson doctor derives expected identity from the reviewed candidate report",
            "candidate seed_export_import_manifest preserves source_recheck path, SHA, and non-approval boundary from the reviewed source lineage",
        ],
        "raw_benchmark": [
            "cuda=true on target Jetson-class hardware unless --allow-cpu is intentionally used outside the production gate",
            "frames cover at least the factory_ppe_3cam max camera count",
            "model entry matches apron-harness-ppe.onnx",
            "mean_ms, p95_ms, samples, and estimated fps/camera satisfy factory_ppe_3cam.jetson_resource_limits",
        ],
        "three_camera_soak": [
            "camera_count covers 3 active cameras",
            "soak_minutes meets factory_ppe_3cam.jetson_resource_limits.soak_minutes",
            "fps_per_camera, mean_latency_ms, p95_latency_ms, model_server_mean_latency_ms_per_request, ram_mb, gpu_utilization_percent, false_positive_count, and stream_restarts satisfy limits",
        ],
        "positive_alerts": [
            "per_class_alert_count.apron_required > 0",
            "per_class_alert_count.harness_required > 0",
        ],
        "false_positive_guard": [
            "false_positive_guard.apron_required.visible_class_total > 0 and matching_alerts == 0",
            "false_positive_guard.harness_required.visible_class_total > 0 and matching_alerts == 0",
        ],
        "detector_window_suppression": [
            "detector_window_suppression.apron_required.max_detections == 0",
            "detector_window_suppression.apron_required.matching_alerts == 0",
            "detector_window_suppression.apron_required.model_invocations all == 0",
            "detector_window_suppression.harness_required.max_detections == 0",
            "detector_window_suppression.harness_required.matching_alerts == 0",
            "detector_window_suppression.harness_required.model_invocations all == 0",
        ],
    }
    jetson_evidence_summary = [
        "same candidate_report_sha256 and selected-export SHA across raw benchmark, soak, promotion, and registry reports",
        "candidate seed_export_import_manifest carries source_recheck path/SHA/non-approval boundary into the Jetson gate",
        "raw CUDA/TensorRT benchmark meets factory_ppe_3cam latency and fps/camera limits",
        "3-camera soak proves positive apron/harness alerts, false-positive guards, and detector-window suppression",
    ]

    actions: dict[str, dict[str, Any]] = {
        "local_apron_harness_pilot_scenarios_not_ready": {
            "next_action": "rerun the active, false-positive, and detector-window pilot YAML scenarios before promotion work",
            "command": ".venv/bin/python scripts/video_eval.py --manifest qa/video_eval/manifest.yaml report",
            "artifacts": ["qa/video_eval/results", "qa/video_eval/SALES_READINESS_REPORT.md"],
        },
        "missing_or_failed_apron_closed_set_promotion_report": {
            "next_action": "after a candidate model exists, run the apron side-by-side promotion report from YAML-only candidate scenarios",
            "command": "see the apron command in this runbook's side-by-side promotion section",
            "artifacts": ["qa/video_eval/results/apron_closed_set_promotion_report.json"],
        },
        "missing_or_failed_harness_closed_set_promotion_report": {
            "next_action": "after a candidate model exists, run the harness side-by-side promotion report from YAML-only candidate scenarios",
            "command": "see the harness command in this runbook's side-by-side promotion section",
            "artifacts": ["qa/video_eval/results/harness_closed_set_promotion_report.json"],
        },
        "missing_or_failed_apron_harness_model_registry_report": {
            "next_action": "run the planned model-registry audit now, then rerun with candidate report and --copy after training",
            "command": ".venv/bin/python scripts/apron_harness_model_registry_doctor.py --planned-audit --out qa/video_eval/results/apron_harness_model_registry_report.json",
            "artifacts": ["qa/video_eval/results/apron_harness_model_registry_report.json"],
        },
        "apron_harness_model_registry_not_registered": {
            "next_action": "copy the verified candidate export into the dormant registry path only after candidate doctor passes",
            "command": registry_copy_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_model_registry_report.json",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
            ],
        },
        "missing_or_failed_factory_ppe_jetson_full_gate": {
            "next_action": "run raw CUDA/TensorRT benchmark and three-camera soak for the exact copied candidate artifact",
            "command": jetson_command,
            "artifacts": [
                "qa/video_eval/results/factory_ppe_jetson_gate.json",
                "/path/to/cleared/factory_ppe_raw_benchmark.json",
                "/path/to/cleared/factory_ppe_3cam_soak.json",
            ],
            "template_commands": [
                jetson_raw_template_command,
                jetson_soak_template_command,
            ],
            "template_artifacts": [
                "qa/video_eval/results/factory_ppe_raw_benchmark.template.json",
                "qa/video_eval/results/factory_ppe_3cam_soak.template.json",
            ],
            "raw_benchmark_capture_command": jetson_raw_capture_command,
            "soak_report_build_command": jetson_soak_build_command,
            "evidence_contract_summary": jetson_evidence_summary,
            "evidence_contract": jetson_evidence_contract,
        },
        "missing_or_failed_model_pack_evidence_doctor": {
            "next_action": "rerun model-pack evidence after every YAML/result/artifact update",
            "command": ".venv/bin/python scripts/model_pack_evidence_doctor.py --out qa/video_eval/results/model_pack_evidence_doctor.json",
            "artifacts": ["qa/video_eval/results/model_pack_evidence_doctor.json"],
        },
        "closed_set_candidate_yaml_templates_invalid": {
            "next_action": "fix closed-set candidate focused YAML templates before attempting candidate runtime evidence",
            "command": ".venv/bin/python scripts/safetylens_site.py --config qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml validate",
            "artifacts": ["qa/video_eval/focused/*closed_set*.yaml"],
        },
        "closed_set_candidate_runtime_evidence_missing_or_invalid": {
            "next_action": "run all closed-set candidate YAML scenarios after ppe_closed_set_candidate is registered",
            "command": (
                ".venv/bin/python scripts/video_eval.py run --scenario factory_missing_apron_active_closed_set && "
                ".venv/bin/python scripts/video_eval.py run --scenario factory_apron_false_positive_guard_closed_set && "
                ".venv/bin/python scripts/video_eval.py run --scenario factory_apron_detector_window_suppression_closed_set && "
                ".venv/bin/python scripts/video_eval.py run --scenario factory_missing_harness_active_closed_set && "
                ".venv/bin/python scripts/video_eval.py run --scenario factory_harness_false_positive_guard_closed_set && "
                ".venv/bin/python scripts/video_eval.py run --scenario factory_harness_detector_window_suppression_closed_set"
            ),
            "artifacts": [
                "qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json",
                "qa/video_eval/results/closed_set_candidate/factory_apron_false_positive_guard_closed_set.json",
                "qa/video_eval/results/closed_set_candidate/factory_apron_detector_window_suppression_closed_set.json",
                "qa/video_eval/results/closed_set_candidate/factory_missing_harness_active_closed_set.json",
                "qa/video_eval/results/closed_set_candidate/factory_harness_false_positive_guard_closed_set.json",
                "qa/video_eval/results/closed_set_candidate/factory_harness_detector_window_suppression_closed_set.json",
            ],
            "evidence_contract_summary": [
                "each candidate scenario must be applied from YAML one at a time",
                "active scenarios must emit detections, matching alerts, delivery evidence, screenshots, and ppe_closed_set_candidate invocations",
                "false-positive guards must emit visible PPE telemetry but zero missing-PPE alerts",
                "detector-window suppression scenarios must emit zero detections, zero alerts, suppressed capability telemetry, and zero ppe_closed_set_candidate invocations",
            ],
        },
        "missing_or_failed_apron_harness_seed_source_review": {
            "next_action": "regenerate and validate the seed-source review packet, next review batch, and handoff bundle",
            "command": seed_review_handoff_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_seed_source_review.json",
                "qa/video_eval/results/apron_harness_next_source_review_batch.json",
                "qa/video_eval/results/apron_harness_source_review_kickoff.md",
                "qa/video_eval/results/apron_harness_source_review_bundle.json",
                "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
            ],
            "evidence_contract_summary": [
                "generated source-review batch and bundle must pass hash validation before reviewer signoff",
                "validation is non-approving and must not mark public sources training-usable",
            ],
        },
        "closed_set_public_seed_sources_not_curated_or_approved": {
            "next_action": "regenerate and validate the source-review handoff, then complete human/legal approval evidence before importing any public seed data",
            "command": seed_review_handoff_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_next_source_review_batch.json",
                "qa/video_eval/results/apron_harness_source_review_kickoff.md",
                "qa/video_eval/results/apron_harness_source_review_bundle.json",
                "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
                "qa/video_eval/results/apron_harness_seed_source_review_checklist.csv",
                "qa/video_eval/results/apron_harness_seed_source_review_packets",
                "qa/video_eval/results/apron_harness_seed_source_review_evidence",
            ],
            "evidence_contract_summary": seed_review_summary,
            "evidence_contract": seed_review_contract,
        },
        "apron_harness_seed_source_review_not_training_usable": {
            "next_action": "fill the review checklist and evidence templates, then apply the approved checklist to a reviewed model-packs file",
            "command": ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --apply-review-checklist-csv /path/to/filled/apron_harness_seed_source_review_checklist.csv --updated-model-packs-out /path/to/reviewed/model_packs.yaml",
            "artifacts": [
                "qa/video_eval/results/apron_harness_seed_source_review_checklist.csv",
                "qa/video_eval/results/apron_harness_seed_source_review_evidence",
            ],
            "evidence_contract_summary": seed_review_summary,
            "evidence_contract": seed_review_contract,
        },
        "missing_or_failed_apron_harness_source_review_bundle": {
            "next_action": "regenerate and validate the non-approving source-review handoff bundle before reviewer signoff",
            "command": seed_review_handoff_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_source_review_bundle.json",
                "qa/video_eval/results/apron_harness_seed_source_review_packets",
                "qa/video_eval/results/apron_harness_seed_source_review_evidence",
            ],
            "evidence_contract_summary": [
                "review bundle hash validation is required before using generated packets/templates for source approval",
                "bundle validation is non-approving and must not set any training_usable or include_in_training fields",
            ],
        },
        "minimum_seed_import_template_not_consistent": {
            "next_action": "regenerate the seed-source review and minimum seed-import template, then verify selected refs and non-training defaults before approval",
            "command": seed_review_command,
            "artifacts": [
                "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
                "qa/video_eval/results/apron_harness_readiness_doctor.json",
                "qa/video_eval/results/apron_harness_source_coverage_plan.json",
                "qa/video_eval/results/apron_harness_source_review_bundle.json",
            ],
            "evidence_contract_summary": [
                "minimum template selected_source_refs must match seed_source_minimum_approval_path.minimum_review_source_refs",
                "minimum template scope must remain minimum_priority_coverage_sources",
                "enabled_import_count must stay 0 until human/legal approval and reviewed local export evidence exist",
            ],
            "evidence_contract": {
                "source_refs": [
                    "selected_source_refs exactly match the readiness minimum approval path",
                    "import source_refs cover the selected_source_refs with no extra training-enabled rows",
                ],
                "non_approving": [
                    "template generation is a planning artifact only",
                    "all generated imports keep include_in_training=false before approval",
                ],
                "scope": [
                    "template_scope=minimum_priority_coverage_sources",
                    "approval_boundary states source selection is not training approval",
                ],
            },
        },
        "missing_or_failed_apron_harness_seed_import_manifest": {
            "next_action": "create a filled seed-import manifest only for reviewed sources approved for training",
            "command": seed_import_validate_command,
            "artifacts": [
                "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
                "qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml",
            ],
            "evidence_contract_summary": seed_import_summary,
            "evidence_contract": seed_import_contract,
        },
        "apron_harness_seed_import_manifest_not_training_usable": {
            "next_action": "validate source-review lineage, generated review-artifact hashes, local reviewed YOLO export ZIP paths, raw export SHA-256 values, image/label pairing, and class mappings in the seed-import manifest",
            "command": seed_import_validate_command,
            "artifacts": [
                "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
                "qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml",
            ],
            "evidence_contract_summary": seed_import_summary,
            "evidence_contract": seed_import_contract,
        },
        "closed_set_pilot_label_minimums_not_met": {
            "next_action": "collect or import enough reviewed pilot labels for person, apron, safety_harness, and safety_lanyard",
            "command": pilot_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_capture_kickoff.md",
                "qa/video_eval/results/apron_harness_capture_matrix.csv",
                "qa/video_eval/results/apron_harness_capture_work_order.md",
            ],
        },
        "closed_set_production_label_minimums_not_met": {
            "next_action": "collect or import at least 1000 reviewed production labels per required class before production compliance promotion",
            "command": production_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_capture_kickoff.md",
                "qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            ],
        },
        "closed_set_capture_matrix_not_complete_or_approved": {
            "next_action": "complete and approve the pilot capture matrix rows, then reconcile them with the capture manifest",
            "command": pilot_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_capture_kickoff.md",
                "qa/video_eval/results/apron_harness_capture_matrix.csv",
            ],
        },
        "closed_set_capture_matrix_manifest_counts_do_not_match": {
            "next_action": "update the capture manifest label counts so they match the approved pilot capture matrix sidecar",
            "command": pilot_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_capture_matrix.csv.manifest.json",
                "qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml",
            ],
        },
        "closed_set_capture_matrix_sidecar_invalid": {
            "next_action": "regenerate the pilot capture matrix sidecar from the current pilot matrix and manifest",
            "command": pilot_capture_progress_command,
            "artifacts": ["qa/video_eval/results/apron_harness_capture_matrix.csv.manifest.json"],
        },
        "closed_set_production_capture_matrix_not_complete_or_approved": {
            "next_action": "complete and approve the production capture matrix rows before training with production preflight",
            "command": production_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_capture_kickoff.md",
                "qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            ],
        },
        "closed_set_production_capture_matrix_manifest_counts_do_not_match": {
            "next_action": "update the production capture manifest label counts so they match the production matrix sidecar",
            "command": production_capture_progress_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_production_capture_matrix.csv.manifest.json",
                "qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml",
            ],
        },
        "closed_set_production_capture_matrix_sidecar_invalid": {
            "next_action": "regenerate the production capture matrix sidecar from the current production matrix and manifest",
            "command": production_capture_progress_command,
            "artifacts": ["qa/video_eval/results/apron_harness_production_capture_matrix.csv.manifest.json"],
        },
        "closed_set_label_review_import_sidecar_invalid": {
            "next_action": "validate the filled production label-review CSV, then import approved rows into a reviewed capture manifest and keep the .label_review_import.json sidecar",
            "command": label_review_validate_command + " && " + label_review_import_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_production_label_review_template.csv",
                "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
            ],
            "evidence_contract_summary": [
                "label-review validation must pass with schema-only/no-write mode before import mutates the reviewed manifest",
                "label-review import sidecar must be valid and tied to the reviewed capture manifest SHA",
                "updated_manifest_validation must be checked, production mode, non-schema-only, and ok=true",
                "approved rows must prove reviewed clip metadata, cleared permission/storage, and recomputed class counts",
            ],
        },
        "closed_set_training_capture_preflight_not_checked": {
            "next_action": "rerun readiness with a reviewed production capture manifest and require production training preflight",
            "command": training_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_training_dataset.yaml",
                "/path/to/cleared/apron_harness_yolo26n_result.json",
            ],
        },
        "closed_set_training_capture_preflight_not_passed": {
            "next_action": "fix capture manifest, matrix sidecars, label-review import, dataset provenance, and source lineage until production training preflight passes",
            "command": training_command,
            "artifacts": [
                "qa/video_eval/results/apron_harness_training_dataset.yaml",
                "/path/to/cleared/apron_harness_yolo26n_result.json",
            ],
        },
        "closed_set_production_training_plan_preflight_failed": {
            "next_action": "rerun the closed-set training dry-run against the reviewed production manifest and resolve reported preflight errors",
            "command": training_command,
            "artifacts": ["/path/to/cleared/apron_harness_yolo26n_result.json"],
        },
        "local_closed_set_training_dry_run_not_on_mps": {
            "next_action": "fix local Apple Silicon/PyTorch MPS availability, then rerun model-pack and training dry-run probes",
            "command": mps_probe_command,
            "artifacts": ["qa/video_eval/results/model_pack_device_probe.json"],
            "evidence_contract_summary": [
                "training dry-run selected_device must be mps",
                "training_torch_status must show mps_available=true and mps_probe_ok=true",
            ],
        },
        "factory_ppe_pack_status_not_promoted": {
            "next_action": "promote factory_ppe_3cam status only after candidate, registry, side-by-side, and Jetson gates pass",
            "command": "edit qa/video_eval/model_packs.yaml only after all production gates pass",
            "artifacts": ["qa/video_eval/model_packs.yaml"],
        },
        "closed_set_runtime_handoff_not_registered": {
            "next_action": "change runtime_handoff.status only after the copied candidate artifact and promotion gates pass",
            "command": "edit qa/video_eval/model_packs.yaml only after registry and side-by-side gates pass",
            "artifacts": ["qa/video_eval/model_packs.yaml"],
        },
        "closed_set_model_key_not_registered": {
            "next_action": "add ppe_closed_set_candidate to the active factory pack only after candidate runtime and Jetson gates pass",
            "command": "edit qa/video_eval/model_packs.yaml only after candidate/Jetson promotion gates pass",
            "artifacts": ["qa/video_eval/model_packs.yaml"],
        },
        "closed_set_registry_model_missing": {
            "next_action": "add the copied candidate artifact to factory_ppe_3cam.registry_models only after model-registry doctor verifies it",
            "command": registry_copy_command,
            "artifacts": [
                "qa/video_eval/model_packs.yaml",
                "qa/video_eval/results/apron_harness_model_registry_report.json",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
            ],
        },
        "closed_set_model_manager_definition_invalid": {
            "next_action": "repair backend/model_manager.py so ppe_closed_set_candidate points at the planned ONNX registry path",
            "command": ".venv/bin/python scripts/apron_harness_model_registry_doctor.py --planned-audit --out qa/video_eval/results/apron_harness_model_registry_report.json",
            "artifacts": ["backend/model_manager.py", "qa/video_eval/results/apron_harness_model_registry_report.json"],
        },
        "closed_set_model_artifact_missing": {
            "next_action": "train/export the candidate, run candidate doctor, then copy the verified ONNX artifact into the planned registry path",
            "command": training_command + " && " + candidate_command + " && " + registry_copy_command,
            "artifacts": [
                "/path/to/cleared/apron_harness_yolo26n_result.json",
                "qa/video_eval/results/apron_harness_candidate_report.json",
                "qa/video_eval/results/apron_harness_model_registry_report.json",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
                "qa/video_eval/results/apron_closed_set_promotion_report.json",
                "qa/video_eval/results/harness_closed_set_promotion_report.json",
            ],
            "evidence_contract_summary": candidate_artifact_summary,
            "evidence_contract": candidate_artifact_contract,
        },
    }

    if blocker.startswith("closed_set_training_dataset_provenance_"):
        return {
            "blocker": blocker,
            "next_action": "fix dataset YAML rakshak_lens provenance fields and rerun production training preflight",
            "command": training_command,
            "artifacts": ["qa/video_eval/results/apron_harness_training_dataset.yaml"],
        }

    action = actions.get(blocker)
    if action is None:
        action = {
            "next_action": "inspect the readiness JSON and the gate artifact named by optional_gate_status, then rerun the relevant doctor",
            "command": ".venv/bin/python scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json",
            "artifacts": ["qa/video_eval/results/apron_harness_readiness_doctor.json"],
        }
    response = {
        "blocker": blocker,
        "next_action": action["next_action"],
        "command": action.get("command", ""),
        "artifacts": list(action.get("artifacts") or []),
    }
    for key, value in action.items():
        if key in response or value in (None, "", [], {}):
            continue
        response[key] = value
    return response


def _production_blocker_actions(blockers: list[str]) -> list[dict[str, Any]]:
    return [_production_blocker_action(str(blocker)) for blocker in blockers]


def _prioritized_next_actions(
    blockers: list[str],
    blocker_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    action_by_blocker = {
        str(action.get("blocker")): action
        for action in blocker_actions
        if action.get("blocker")
    }
    priority_groups = [
        {
            "id": "approve_or_capture_training_data",
            "title": "Approve public seed sources or collect controlled apron/harness capture data",
            "representative_blockers": [
                "missing_or_failed_apron_harness_seed_source_review",
                "closed_set_public_seed_sources_not_curated_or_approved",
                "missing_or_failed_apron_harness_source_review_bundle",
                "minimum_seed_import_template_not_consistent",
                "missing_or_failed_apron_harness_seed_import_manifest",
                "apron_harness_seed_source_review_not_training_usable",
                "apron_harness_seed_import_manifest_not_training_usable",
            ],
            "blockers": [
                "missing_or_failed_apron_harness_seed_source_review",
                "closed_set_public_seed_sources_not_curated_or_approved",
                "missing_or_failed_apron_harness_source_review_bundle",
                "minimum_seed_import_template_not_consistent",
                "missing_or_failed_apron_harness_seed_import_manifest",
                "apron_harness_seed_source_review_not_training_usable",
                "apron_harness_seed_import_manifest_not_training_usable",
                "closed_set_pilot_label_minimums_not_met",
                "closed_set_production_label_minimums_not_met",
                "closed_set_capture_matrix_not_complete_or_approved",
                "closed_set_production_capture_matrix_not_complete_or_approved",
            ],
        },
        {
            "id": "produce_reviewed_training_manifest",
            "title": "Import reviewed labels and produce a production-ready capture manifest",
            "representative_blockers": [
                "closed_set_label_review_import_sidecar_invalid",
                "closed_set_production_training_plan_preflight_failed",
                "closed_set_training_capture_preflight_not_passed",
                "closed_set_training_capture_preflight_not_checked",
            ],
            "blockers": [
                "closed_set_capture_matrix_manifest_counts_do_not_match",
                "closed_set_production_capture_matrix_manifest_counts_do_not_match",
                "closed_set_label_review_import_sidecar_invalid",
                "closed_set_training_capture_preflight_not_checked",
                "closed_set_training_capture_preflight_not_passed",
                "closed_set_production_training_plan_preflight_failed",
            ],
        },
        {
            "id": "train_and_register_candidate",
            "title": "Train YOLO26 closed-set apron/harness candidate and register the verified ONNX export",
            "representative_blockers": [
                "closed_set_model_artifact_missing",
                "apron_harness_model_registry_not_registered",
                "missing_or_failed_apron_closed_set_promotion_report",
            ],
            "blockers": [
                "missing_or_failed_apron_closed_set_promotion_report",
                "missing_or_failed_harness_closed_set_promotion_report",
                "apron_harness_model_registry_not_registered",
                "closed_set_candidate_runtime_evidence_missing_or_invalid",
                "closed_set_runtime_handoff_not_registered",
                "closed_set_model_key_not_registered",
                "closed_set_registry_model_missing",
                "closed_set_model_artifact_missing",
                "factory_ppe_pack_status_not_promoted",
            ],
        },
        {
            "id": "prove_edge_gate",
            "title": "Run factory PPE Jetson raw benchmark and three-camera soak on the exact candidate artifact",
            "representative_blockers": ["missing_or_failed_factory_ppe_jetson_full_gate"],
            "blockers": ["missing_or_failed_factory_ppe_jetson_full_gate"],
        },
    ]
    used: set[str] = set()
    next_actions: list[dict[str, Any]] = []
    blocker_set = {str(blocker) for blocker in blockers}
    for index, group in enumerate(priority_groups, start=1):
        present = [blocker for blocker in group["blockers"] if blocker in blocker_set]
        if not present:
            continue
        preferred = [
            blocker
            for blocker in group.get("representative_blockers", [])
            if blocker in present
        ]
        representative_blocker = preferred[0] if preferred else present[0]
        representative = action_by_blocker.get(representative_blocker, {})
        used.update(present)
        next_actions.append(
            {
                "priority": index,
                "id": group["id"],
                "title": group["title"],
                "status": "blocked_until_evidence_uploaded",
                "blockers": present,
                "next_action": representative.get("next_action", ""),
                "command": representative.get("command", ""),
                "artifacts": list(representative.get("artifacts") or []),
                "evidence_contract_summary": list(
                    representative.get("evidence_contract_summary") or []
                ),
            }
        )
    remaining = [
        blocker
        for blocker in blockers
        if str(blocker) not in used
    ]
    for offset, blocker in enumerate(remaining, start=len(next_actions) + 1):
        action = action_by_blocker.get(str(blocker), {})
        next_actions.append(
            {
                "priority": offset,
                "id": str(blocker),
                "title": action.get("next_action", "Resolve production blocker"),
                "status": "blocked_until_evidence_uploaded",
                "blockers": [str(blocker)],
                "next_action": action.get("next_action", ""),
                "command": action.get("command", ""),
                "artifacts": list(action.get("artifacts") or []),
                "evidence_contract_summary": list(
                    action.get("evidence_contract_summary") or []
                ),
            }
        )
    return next_actions


def _result_path(result_dir: Path, scenario_id: str) -> Path:
    return result_dir / f"{scenario_id}.json"


def _render_promotion_runbook(
    *,
    report: dict[str, Any],
    result_dir: Path,
    model_packs_path: Path,
    capture_manifest: Path,
    production_capture_matrix_csv_out: Path | None,
) -> str:
    closed_set = report.get("closed_set_handoff") or {}
    candidate_template_status = report.get("closed_set_candidate_yaml_templates") or {}
    candidate_templates = candidate_template_status.get("templates") or []
    candidate_runtime_evidence = report.get("closed_set_candidate_runtime_evidence") or {}
    pretrained_shortcut_review = report.get("pretrained_shortcut_review") or {}
    pretrained_shortcut_candidates = pretrained_shortcut_review.get("candidates")
    if not isinstance(pretrained_shortcut_candidates, list):
        pretrained_shortcut_candidates = []
    optional = report.get("optional_gate_status") or {}
    model_registry_handoff = report.get("model_registry_handoff") or {}
    jetson_full_gate = report.get("jetson_full_gate") or {}
    jetson_template_handoff = report.get("jetson_template_handoff") or {}
    blockers = report.get("production_blockers") or []
    blocker_actions = report.get("production_blocker_actions") or []
    next_evidence = report.get("next_required_evidence") or []
    seed_source_next_reviews = report.get("seed_source_next_review_queue") or []
    seed_source_coverage = report.get("seed_source_coverage_summary") or {}
    seed_source_minimum_approval_path = report.get("seed_source_minimum_approval_path") or {}
    seed_import_export_preflight = report.get("seed_import_export_preflight_summary") or {}
    minimum_seed_import_template = report.get("minimum_seed_import_manifest_template_summary") or {}
    minimum_seed_import_consistency = report.get("minimum_seed_import_manifest_template_consistency") or {}
    training_torch_status = closed_set.get("training_torch_status") if isinstance(closed_set, dict) else {}
    if not isinstance(training_torch_status, dict):
        training_torch_status = {}
    candidate_report = "qa/video_eval/results/apron_harness_candidate_report.json"
    seed_source_review = "qa/video_eval/results/apron_harness_seed_source_review.json"
    seed_source_work_order = "qa/video_eval/results/apron_harness_seed_source_review.md"
    seed_source_checklist = "qa/video_eval/results/apron_harness_seed_source_review_checklist.csv"
    seed_source_evidence_templates = "qa/video_eval/results/apron_harness_seed_source_review_evidence"
    seed_source_review_packets = "qa/video_eval/results/apron_harness_seed_source_review_packets"
    seed_source_next_review_batch = "qa/video_eval/results/apron_harness_next_source_review_batch.json"
    seed_source_review_kickoff = "qa/video_eval/results/apron_harness_source_review_kickoff.md"
    seed_source_coverage_plan = "qa/video_eval/results/apron_harness_source_coverage_plan.json"
    seed_source_review_bundle = "qa/video_eval/results/apron_harness_source_review_bundle.json"
    seed_import_template = "qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml"
    minimum_seed_import_template_path = "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml"
    filled_seed_import_manifest = "/path/to/filled/apron_harness_seed_import_manifest.yaml"
    label_review_csv = "qa/video_eval/results/apron_harness_label_review_template.csv"
    production_label_review_csv = "qa/video_eval/results/apron_harness_production_label_review_template.csv"
    filled_production_label_review_csv = "/path/to/filled/apron_harness_production_label_review.csv"
    reviewed_capture_manifest = "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml"
    training_dataset_yaml = (
        (closed_set.get("training_dataset_yaml_handoff") or {}).get("path")
        or closed_set.get("production_training_dataset_yaml")
        or closed_set.get("training_dataset_yaml")
        or "qa/video_eval/results/apron_harness_training_dataset.yaml"
    )
    jetson_gate = "qa/video_eval/results/factory_ppe_jetson_gate.json"
    model_registry_report = "qa/video_eval/results/apron_harness_model_registry_report.json"
    raw_benchmark = "qa/video_eval/results/factory_ppe_raw_benchmark.json"
    soak_report = "qa/video_eval/results/factory_ppe_3cam_soak.json"
    soak_metrics = "qa/video_eval/results/factory_ppe_3cam_soak_metrics.yaml"
    raw_benchmark_template = "qa/video_eval/results/factory_ppe_raw_benchmark.template.json"
    soak_report_template = "qa/video_eval/results/factory_ppe_3cam_soak.template.json"
    seed_import_validation_summary = (
        "qa/video_eval/results/apron_harness_seed_import_manifest_validation_summary.json"
    )
    production_capture_matrix_validation_summary = (
        "qa/video_eval/results/apron_harness_production_capture_matrix_validation_summary.json"
    )
    production_gate_packet = "qa/video_eval/results/apron_harness_production_gate_packet.json"
    apron_promotion = "qa/video_eval/results/apron_closed_set_promotion_report.json"
    harness_promotion = "qa/video_eval/results/harness_closed_set_promotion_report.json"
    candidate_result_dir = "qa/video_eval/results/closed_set_candidate"

    lines = [
        "# Apron/Harness Closed-Set Promotion Runbook",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Current Gate Status",
        "",
        f"- Pilot gate passed: `{report.get('pilot_gate_passed')}`",
        f"- Production gate passed: `{report.get('production_gate_passed')}`",
        f"- Sales status: `{report.get('sales_status')}`",
        f"- Production blocker count: `{report.get('production_blocker_count', len(blockers))}`",
        f"- Model packs: `{_rel(model_packs_path)}`",
        f"- Result dir: `{_rel(result_dir)}`",
        f"- Capture manifest: `{_rel(capture_manifest)}`",
        f"- Reviewed capture manifest target: `{reviewed_capture_manifest}`",
        f"- Production capture matrix: `{_rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else 'not generated'}`",
        f"- Pilot label review template: `{label_review_csv}`",
        f"- Production label review template: `{production_label_review_csv}`",
        f"- Training dataset YAML: `{training_dataset_yaml}`",
        "",
        "Optional gate status:",
        "",
    ]
    for key in (
        "seed_source_review",
        "seed_source_review_bundle",
        "seed_import_manifest",
        "apron_promotion",
        "harness_promotion",
        "model_registry",
        "jetson_gate",
        "model_pack_evidence",
    ):
        lines.append(f"- `{key}`: `{optional.get(key)}`")

    lines.extend([
        "",
        "Production blockers:",
        "",
    ])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")

    lines.extend([
        "",
        "Production blocker action map:",
        "",
    ])
    if blocker_actions:
        for item in blocker_actions:
            if not isinstance(item, dict):
                continue
            artifacts = ", ".join(f"`{_md_cell(value)}`" for value in item.get("artifacts") or [])
            lines.append(f"- `{_md_cell(item.get('blocker'))}`: {_md_cell(item.get('next_action'))}")
            if item.get("command"):
                lines.append(f"  Command: `{_md_cell(item.get('command'))}`")
            if artifacts:
                lines.append(f"  Artifacts: {artifacts}")
            template_commands = item.get("template_commands")
            if isinstance(template_commands, list):
                for command in template_commands:
                    lines.append(f"  Template command: `{_md_cell(command)}`")
            template_artifacts = item.get("template_artifacts")
            if isinstance(template_artifacts, list) and template_artifacts:
                rendered_templates = ", ".join(f"`{_md_cell(value)}`" for value in template_artifacts)
                lines.append(f"  Template artifacts: {rendered_templates}")
            if item.get("raw_benchmark_capture_command"):
                lines.append(f"  Raw benchmark capture: `{_md_cell(item.get('raw_benchmark_capture_command'))}`")
            if item.get("soak_report_build_command"):
                lines.append(f"  Soak report build: `{_md_cell(item.get('soak_report_build_command'))}`")
            evidence_summary = item.get("evidence_contract_summary")
            if isinstance(evidence_summary, list):
                for requirement in evidence_summary:
                    lines.append(f"  Requires: {_md_cell(requirement)}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "Next required evidence:",
        "",
    ])
    if next_evidence:
        lines.extend(f"- `{item}`" for item in next_evidence)
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Required Preconditions",
        "",
        "1. Run the public seed-source review gate before importing any Roboflow/public seed source into the capture manifest.",
        "2. If public seed exports are approved, materialize them through the dataset doctor so the capture manifest, converted labels, copied images, and `.seed_export_import.json` sidecar are generated together with preserved YOLO export preflight evidence.",
        "3. Fill and approve the production capture matrix with commercial-safe apron, harness, lanyard, and person labels.",
        "4. Fill the generated label-review CSV with one reviewed YOLO label file per row, then convert approved rows into the capture manifest `yolo_labels` block.",
        "5. Reconcile the approved matrix totals with the capture manifest label counts.",
        "6. Confirm `closed_set_handoff.production_training_plan_preflight.ok=true` in the readiness JSON; this is the trainer dry-run using the production capture manifest, production matrix sidecar, label-review import sidecar, dataset provenance, and source lineage.",
        "7. Train a nano/small closed-set model with `--capture-preflight-mode production` and `--require-capture-preflight`.",
        "8. Run the candidate doctor and keep the output at the candidate-report path below. The training result must carry `per_class_metrics` for person, apron, safety_harness, and safety_lanyard plus `dataset_provenance` and `source_lineage` that match the current production capture manifest SHA, cleared permission policy, and seed review/import evidence. The candidate report exposes these as `training_dataset_provenance` and `training_source_lineage`, and its selected export must be ONNX for the dormant registry slot.",
        "9. Run the model registry doctor in dry-run mode, then with `--copy`, to verify the selected export SHA and install it at the dormant `ppe_closed_set_candidate` registry path.",
        "10. Produce candidate active, false-positive guard, and detector-window suppression result JSON through YAML-only scenario configs.",
        "11. Activate or promote `ppe_closed_set_candidate` only after side-by-side YAML/runtime tests and the Jetson full gate pass.",
        "",
        "## Pretrained Shortcut Review",
        "",
        f"- Status: `{pretrained_shortcut_review.get('status') or 'not_recorded'}`",
        f"- Checked: `{pretrained_shortcut_review.get('checked') or ''}`",
        f"- Requirement: {_md_cell(pretrained_shortcut_review.get('requirement'))}",
        f"- Resulting path: {_md_cell(pretrained_shortcut_review.get('resulting_path'))}",
        "",
        "| Source | Decision | Target Coverage | Artifact | Blocker |",
        "| --- | --- | --- | --- | --- |",
    ])
    if pretrained_shortcut_candidates:
        for candidate in pretrained_shortcut_candidates:
            if not isinstance(candidate, dict):
                continue
            lines.append(
                "| "
                f"`{_md_cell(candidate.get('source_ref'))}` | "
                f"`{_md_cell(candidate.get('decision'))}` | "
                f"{_md_cell(_target_class_coverage_summary(candidate.get('target_class_coverage')))} | "
                f"`{_md_cell(candidate.get('local_artifact_status'))}` | "
                f"{_md_cell(candidate.get('blocker'))} |"
            )
    else:
        lines.append("| none | `not_recorded` | unknown | unknown | Shortcut review missing. |")
    lines.extend([
        "",
        "## Public Seed-Source Review",
        "",
        "Canonical refresh order for generated source-review artifacts, readiness JSON, and sales reports:",
        "",
        "1. Generate the seed-source review handoff, review packets, minimum import template, and review bundle.",
        "2. Validate the next-review batch.",
        "3. Validate the source-review bundle before running readiness; stale source-review hashes are failed evidence.",
        "4. Validate the filled seed-import manifest after approval. The generated template is expected to stay blocked.",
        "5. Run `scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json`; it also writes concise seed-import and production capture-matrix validation summaries plus the machine-readable production gate packet.",
        "6. Run `scripts/video_eval.py --manifest qa/video_eval/manifest.yaml report` to refresh sales-readiness docs.",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        f"  --out {seed_source_review} \\",
        f"  --work-order-out {seed_source_work_order} \\",
        f"  --import-template-out {seed_import_template} \\",
        f"  --minimum-import-template-out {minimum_seed_import_template_path} \\",
        f"  --review-checklist-csv-out {seed_source_checklist} \\",
        f"  --review-evidence-template-dir {seed_source_evidence_templates} \\",
        f"  --review-packet-dir {seed_source_review_packets} \\",
        f"  --next-review-batch-out {seed_source_next_review_batch} \\",
        f"  --review-kickoff-out {seed_source_review_kickoff} \\",
        f"  --source-coverage-plan-out {seed_source_coverage_plan} \\",
        f"  --review-bundle-out {seed_source_review_bundle}",
        "```",
        "",
        "Validate the generated next-review batch and handoff bundle before using packets/templates for review:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        f"  --validate-next-review-batch {seed_source_next_review_batch}",
        "",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        f"  --validate-review-bundle {seed_source_review_bundle}",
        "```",
        "",
        "Inspect the readiness handoff summaries and machine-readable production gate packet:",
        "",
        "```bash",
        f"jq '.' {seed_import_validation_summary}",
        f"jq '.' {production_capture_matrix_validation_summary}",
        f"jq '.' {production_gate_packet}",
        f"jq '.candidate_runtime_execution_plan' {production_gate_packet}",
        f"jq '.candidate_runtime_execution_plan.steps[].artifact_status' {production_gate_packet}",
        f"jq '.candidate_training_execution_plan' {production_gate_packet}",
        f"jq '.candidate_training_execution_plan.runner' {production_gate_packet}",
        f"jq '.jetson_gate_execution_plan' {production_gate_packet}",
        f"jq '.jetson_gate_execution_plan.runner' {production_gate_packet}",
        f"jq '.model_registry_handoff.artifact_status' {production_gate_packet}",
        f"jq '.first_unblock.source_review_execution_plan' {production_gate_packet}",
        f"jq '.first_unblock.source_review_runner' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.runner' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.starter_capture_rows' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.label_review_templates' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.starter_success_criteria' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.starter_execution_plan' {production_gate_packet}",
        f"jq '.first_unblock.controlled_capture_path.post_capture_evidence_checklist' {production_gate_packet}",
        "```",
        "",
        "The production gate packet must show the candidate runtime execution plan, candidate training execution plan, Jetson gate execution plan, the public source-review execution plan, the guarded source-review runner, and the controlled-capture section with its guarded runner. The candidate runtime plan must preserve the one-detection-at-a-time YAML validate/plan/apply/run sequence for the six apron/harness closed-set active, false-positive guard, and detector-window suppression scenarios. The candidate training plan must preserve the reviewed-data preflight, explicit --run-training boundary, candidate doctor, side-by-side promotion, and registry copy sequence. The Jetson gate runner must refuse execution until candidate report, raw benchmark, and three-camera soak report paths are supplied as existing non-placeholder evidence, then run the full gate with --require-full-gate. The source-review runner must stay validation-first, refuse seed export imports until a filled seed-import manifest passes, and validate the emitted `.seed_export_import.json` sidecar against the seed-source review after materialization. The controlled-capture runner must refuse label-review imports until filled CSV and seed-import manifests are supplied, `LABEL_REVIEW_VALIDATION: gate=pass` succeeds, and the emitted `.label_review_import.json` sidecar validates against the filled label-review CSV. The controlled-capture packet section must show the production capture matrix SHA, missing labeled-example count, required operator fields, starter capture rows, production label-review templates, starter validation/import commands, starter success criteria, starter execution plan, post-capture evidence checklist, and full label-review validation/import commands before a capture team or another agent uses it.",
        "",
        "Validate that the production gate packet is fresh and still matches the saved readiness report before handing it to another agent or automation:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_readiness_doctor.py \\",
        f"  --validate-production-gate-packet {production_gate_packet} \\",
        "  --readiness-report qa/video_eval/results/apron_harness_readiness_doctor.json",
        "```",
        "",
        "After approval, validate the filled import manifest before adding public seed clips to the capture manifest:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py \\",
        f"  --validate-review-bundle {seed_source_review_bundle} \\",
        f"  --validate-import-manifest {filled_seed_import_manifest}",
        "```",
        "",
        "The filled import-manifest command must exit `0` and print `IMPORT_MANIFEST: gate=pass` before any public seed export is materialized. A template or partially filled manifest is expected to exit nonzero with `IMPORT_MANIFEST: gate=blocked`.",
        "",
        "This gate is expected to remain blocked until public seed sources have explicit approval metadata and a manifest import plan. Do not train from Roboflow/public seed data just because the public page lists CC BY 4.0.",
        "",
        "### Source Coverage Snapshot",
        "",
        f"- Available: `{bool(seed_source_coverage.get('available'))}`",
        f"- Coverage gaps: `{seed_source_coverage.get('coverage_gap_count')}`",
        f"- Training-usable sources: `{seed_source_coverage.get('training_usable_count', 0)}`",
        f"- Next action: {_md_cell(seed_source_coverage.get('next_action'))}",
        "",
        "| Capability | Missing Classes | Person Boxes | Priority Coverage | Selected Sources |",
        "| --- | --- | --- | --- | --- |",
    ])
    coverage_capabilities = seed_source_coverage.get("capabilities") if isinstance(seed_source_coverage, dict) else {}
    if isinstance(coverage_capabilities, dict) and coverage_capabilities:
        for capability in ("apron_required", "harness_required"):
            item = coverage_capabilities.get(capability)
            if not isinstance(item, dict):
                continue
            selected_sources = ", ".join(
                str(source.get("source_ref"))
                for source in item.get("priority_selected_sources") or []
                if isinstance(source, dict) and source.get("source_ref")
            )
            lines.append(
                "| "
                f"`{capability}` | "
                f"{_md_cell(', '.join(str(value) for value in item.get('missing_local_classes_across_reviewable_sources') or []) or 'none')} | "
                f"`{_md_cell(item.get('person_box_reconciliation_status'))}` | "
                f"`{_md_cell(item.get('priority_coverage_status'))}` | "
                f"{_md_cell(selected_sources or 'none')} |"
            )
    else:
        lines.append("| none | unknown | unknown | unknown | none |")

    minimum_capabilities = (
        seed_source_minimum_approval_path.get("capabilities")
        if isinstance(seed_source_minimum_approval_path, dict)
        else {}
    )
    lines.extend([
        "",
        "### Minimum Approval Path",
        "",
        f"- Coverage gaps: `{seed_source_minimum_approval_path.get('coverage_gap_count')}`",
        f"- Training-usable sources: `{seed_source_minimum_approval_path.get('training_usable_count', 0)}`",
        f"- Boundary: {_md_cell(seed_source_minimum_approval_path.get('evidence_boundary'))}",
        "",
        "| Capability | Selected Source | Priority | Mapped Classes | Missing Classes | Packet | Evidence Template | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    if isinstance(minimum_capabilities, dict) and minimum_capabilities:
        for capability in ("apron_required", "harness_required"):
            item = minimum_capabilities.get(capability)
            if not isinstance(item, dict):
                continue
            for source in item.get("selected_sources") or []:
                if not isinstance(source, dict):
                    continue
                status = "training_usable" if source.get("training_usable") else "pending_human_legal_review"
                lines.append(
                    "| "
                    f"`{capability}` | "
                    f"`{_md_cell(source.get('source_ref'))}` | "
                    f"`{_md_cell(source.get('review_priority'))}` | "
                    f"{_md_cell(', '.join(str(value) for value in source.get('mapped_local_classes') or []) or 'none')} | "
                    f"{_md_cell(', '.join(str(value) for value in source.get('missing_local_classes') or []) or 'none')} | "
                    f"`{_md_cell(source.get('review_packet_path'))}` | "
                    f"`{_md_cell(source.get('review_evidence_template_path'))}` | "
                    f"`{status}` |"
                )
    else:
        lines.append("| none | none | unknown | unknown | unknown | missing | missing | blocked |")

    lines.extend([
        "",
        "Minimum seed-import template:",
        "",
        f"- Path: `{minimum_seed_import_template.get('path') or minimum_seed_import_template_path}`",
        f"- Scope: `{minimum_seed_import_template.get('template_scope') or 'minimum_priority_coverage_sources'}`",
        f"- Sources: {_md_cell(', '.join(str(value) for value in minimum_seed_import_template.get('selected_source_refs') or []) or 'none')}",
        f"- Imports enabled for training: `{minimum_seed_import_template.get('enabled_import_count', 0)}`",
        f"- Consistency: valid=`{minimum_seed_import_consistency.get('valid')}`, refs_match=`{minimum_seed_import_consistency.get('source_refs_match')}`",
        f"- Consistency boundary: {_md_cell(minimum_seed_import_consistency.get('evidence_boundary'))}",
    ])

    lines.extend([
        "",
        "### Seed Import Export Preflight",
        "",
        "- Every `include_in_training=true` seed import must include `raw_export_local_path` pointing to the reviewed local YOLO export ZIP.",
        "- Validation checks ZIP SHA-256, `data.yaml` class names, mapped source labels, train/valid/test image and label presence, matching image files for every label file, and required local-class label-file counts before the seed import gate can pass.",
        "- If the seed-source review generated review packets/templates, every included seed import must preserve the matching `review_artifacts` paths and SHA-256 values, and validation re-hashes those generated files before the gate can pass.",
        f"- Current included imports: `{seed_import_export_preflight.get('included_count', 0)}`",
        f"- Current reviewed-export preflight approvals: `{seed_import_export_preflight.get('preflight_approved_count', 0)}`",
        f"- Current review-artifact preflight approvals: `{seed_import_export_preflight.get('review_artifact_approved_count', 0)}`",
        f"- Current review-artifact preflight errors: `{seed_import_export_preflight.get('review_artifact_error_count', 0)}`",
        f"- Missing reviewed local export ZIP paths: `{seed_import_export_preflight.get('missing_raw_export_local_path_count', 0)}`",
        "",
        "After both seed gates pass, materialize approved YOLO seed exports into a capture manifest and write the required `.seed_export_import.json` sidecar. That sidecar must preserve the fresh `source_recheck` artifact path/SHA/non-approval boundary from the seed-source review, YOLO export preflight evidence for the exact ZIP SHA, zero orphan labels, and required local-class label counts. The source-review runner validates the emitted sidecar after materialization; command exit `0` alone is not enough. The camera-angle, distance, lighting, motion-blur, positive-variant, and hard-negative values must be the operator's reviewed metadata for that seed source, not placeholders:",
        "",
        "If seed-export materialization fails, the dataset doctor still writes the `.seed_export_import.json` sidecar with `valid=false`, import errors, warnings, and `partial_materialization` so the next SSH session can resume from the exact gate failure.",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        f"  --manifest {_rel(capture_manifest)} \\",
        "  --mode production \\",
        f"  --seed-source-review-report {seed_source_review} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        "  --import-approved-seed-exports \\",
        "  --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml \\",
        "  --seed-import-camera-angle front \\",
        "  --seed-import-distance-band medium \\",
        "  --seed-import-lighting indoor_bright \\",
        "  --seed-import-motion-blur low \\",
        "  --seed-import-positive-variant-tags full_body_safety_harness\\;visible_lanyard_or_tether",
        "```",
        "",
        "### Next Seed-Source Reviews",
        "",
    ])
    if seed_source_next_reviews:
        lines.extend([
            "| Priority | Capability | Source | Source URL | Next Action | Import Fill Plan | Packet | Evidence Template | Required Review |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for item in seed_source_next_reviews:
            if not isinstance(item, dict):
                continue
            required_review = ", ".join(str(value) for value in item.get("required_review") or [])
            import_fill_plan = _seed_import_fill_plan_summary(item.get("seed_import_fill_plan"))
            lines.append(
                "| "
                f"`{_md_cell(item.get('review_priority'))}` | "
                f"`{_md_cell(item.get('capability'))}` | "
                f"`{_md_cell(item.get('source_ref'))}` | "
                f"`{_md_cell(item.get('source_url'))}` | "
                f"`{_md_cell(item.get('next_action'))}` | "
                f"{_md_cell(import_fill_plan)} | "
                f"`{_md_cell(item.get('review_packet_path'))}` | "
                f"`{_md_cell(item.get('review_evidence_template_path'))}` | "
                f"{_md_cell(required_review)} |"
            )
    else:
        lines.append("- No seed-source review queue was available in the readiness report.")
    lines.extend([
        "",
        "## Label Review Template",
        "",
        f"- Pilot template: `{label_review_csv}`",
        f"- Production template: `{production_label_review_csv}`",
        "",
        "Each row represents one planned YOLO label file from the capture matrix. Fill `source_clip_id`, `image_path`, `label_path`, `review_status=approved`, `reviewer`, `reviewed_at`, permission, and raw storage after manual review, then convert approved rows into `yolo_labels` entries in the cleared capture manifest. Generated rows also carry `source_manifest_sha256`, `taxonomy_version`, `required_label_class_ids`, and `label_format`; approved rows that include these fields must match the source manifest and class taxonomy. The CSV itself is a work aid; it does not replace strict manifest validation.",
        "",
        f"- Taxonomy: `{TAXONOMY_VERSION}`",
        f"- Label format: `{YOLO_LABEL_FORMAT}`",
        "- Required import sidecar: `.label_review_import.json` with strict reviewed-manifest validation",
        "- Status naming: label-review CSV rows use `review_status=approved`; public seed-import manifest rows use `review_status=approved_for_training` before `include_in_training=true`.",
        "",
        "Validate the filled production label-review CSV in no-write mode before changing the manifest:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        f"  --manifest {_rel(capture_manifest)} \\",
        "  --mode production \\",
        "  --schema-only \\",
        f"  --seed-source-review-report {seed_source_review} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        f"  --validate-label-review-csv {filled_production_label_review_csv}",
        "```",
        "",
        "The validation command must report `LABEL_REVIEW_VALIDATION: gate=pass` before import. Then import approved rows into a reviewed capture manifest and write the required `.label_review_import.json` sidecar:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        f"  --manifest {_rel(capture_manifest)} \\",
        "  --mode production \\",
        f"  --seed-source-review-report {seed_source_review} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        f"  --import-label-review-csv {filled_production_label_review_csv} \\",
        f"  --emit-updated-manifest {reviewed_capture_manifest}",
        "```",
        "",
        "After this step, use the reviewed manifest path for readiness, dataset YAML generation, training, promotion, model-registry copy, and Jetson gates. The template manifest is only the planning input. If that manifest includes public/commercial seed clips, the production trainer, candidate doctor, promotion doctor, model registry doctor, and readiness doctor also require the matching `.seed_export_import.json` sidecar produced by the approved seed-export materialization command, with preserved source-recheck lineage and YOLO export preflight evidence.",
        "",
        "## Training Dataset YAML",
        "",
        f"- Generated handoff: `{training_dataset_yaml}`",
        "",
        "This YAML carries the capture-manifest path, source-manifest SHA, permission policy, and YOLO class map expected by the production trainer. Keep the dataset root pointed at the cleared image/label directory on the deployed training machine, and do not train until capture preflight passes.",
        "",
        "Regenerate it from the reviewed capture manifest after label import:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_dataset_doctor.py \\",
        f"  --manifest {reviewed_capture_manifest} \\",
        "  --mode production \\",
        f"  --seed-source-review-report {seed_source_review} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        f"  --emit-yolo-dataset-yaml {training_dataset_yaml}",
        "```",
        "",
        "Dry-run the production training preflight before executing training:",
        "",
        f"- Current dry-run device: `{closed_set.get('selected_device')}`",
        f"- Torch version: `{training_torch_status.get('version') or 'unknown'}`",
        f"- MPS built: `{training_torch_status.get('mps_built')}`",
        f"- MPS available: `{training_torch_status.get('mps_available')}`",
        f"- MPS probe OK: `{training_torch_status.get('mps_probe_ok')}`",
        f"- MPS runtime error: `{_md_cell(training_torch_status.get('mps_runtime_error') or '')}`",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_train.py \\",
        f"  --data {training_dataset_yaml} \\",
        "  --model yolo26n.pt \\",
        "  --device mps \\",
        f"  --capture-manifest {reviewed_capture_manifest} \\",
        f"  --capture-matrix-csv {_rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else 'qa/video_eval/results/apron_harness_production_capture_matrix.csv'} \\",
        f"  --seed-source-review-report {seed_source_review} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        "  --capture-preflight-mode production \\",
        "  --require-capture-preflight \\",
        "  --out-plan qa/video_eval/results/apron_harness_training_plan.json",
        "```",
        "",
        "If this dry-run fails, `--out-plan` is still written with `status=failed` and the preflight error. Keep that JSON with the reviewed manifest/matrix artifacts so the next SSH session can continue from the exact gate failure.",
        "",
        "Only after that dry-run passes, add `--execute` to train and export the candidate. The output JSON is the training result consumed by the candidate doctor. Keep an ONNX export in the result; TensorRT engine exports may be benchmarked later, but the planned registry slot selects the ONNX artifact.",
        "",
        "## Candidate Doctor",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_candidate_doctor.py \\",
        "  --training-result /path/to/cleared/apron_harness_yolo_result.json \\",
        f"  --out {candidate_report}",
        "```",
        "",
        "The training script emits `per_class_metrics` from Ultralytics validation output. Do not hand-edit those metrics into the result JSON; rerun training/validation if a class metric is missing.",
        "",
        "## Model Registry Copy",
        "",
        "Current model-registry handoff:",
        "",
        f"- Report: `{model_registry_handoff.get('path') or model_registry_report}`",
        f"- Status: `{model_registry_handoff.get('status')}`",
        f"- Registry status: `{model_registry_handoff.get('registry_status')}`",
        f"- Model definition valid: `{model_registry_handoff.get('model_definition_valid')}`",
        f"- Destination exists: `{model_registry_handoff.get('destination_exists')}`",
        f"- Metadata sidecar valid: `{model_registry_handoff.get('metadata_valid')}`",
        f"- Boundary: {model_registry_handoff.get('evidence_boundary', 'planned audit is not registration')}",
        "",
        "Registry artifact status:",
        "",
        "| Artifact | Path | Exists | OK | Blockers |",
        "| --- | --- | --- | --- | --- |",
    ])
    for artifact in model_registry_handoff.get("artifact_status") or []:
        if not isinstance(artifact, dict):
            continue
        lines.append(
            "| "
            f"`{_md_cell(artifact.get('name'))}` | "
            f"`{_md_cell(artifact.get('path'))}` | "
            f"`{_md_cell(artifact.get('exists'))}` | "
            f"`{_md_cell(artifact.get('ok'))}` | "
            f"`{_md_cell(','.join(str(item) for item in artifact.get('blockers') or []))}` |"
        )
    lines.extend([
        "",
        "Dry-run first:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_model_registry_doctor.py \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --out {model_registry_report}",
        "```",
        "",
        "Copy only after the dry-run passes:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_model_registry_doctor.py \\",
        f"  --candidate-report {candidate_report} \\",
        "  --copy \\",
        f"  --out {model_registry_report}",
        "```",
        "",
        "This copies the selected export to `models/ppe_closed_set_candidate/apron-harness-ppe.onnx` only after verifying the candidate report, selected-export SHA, registry entry, and dormant backend model definition.",
        "",
        "## YAML-Only Candidate Evidence",
        "",
        "Baseline pilot result JSON already exists under the current `result_dir`. Candidate result JSON must be produced from separate YAML configs after the closed-set model is available. Do not edit `backend/config.json`, DB rows, or result JSON by hand.",
        "The eval runner now performs a required-model preflight from `/api/health` and the final camera execution plan; candidate runs should block immediately with `ppe_closed_set_candidate` missing until the copied ONNX artifact is actually ready in the model registry.",
        "",
        f"- Candidate runtime evidence valid: `{candidate_runtime_evidence.get('valid')}`",
        f"- Candidate runtime result files present: `{candidate_runtime_evidence.get('present_result_count', 0)}/{candidate_runtime_evidence.get('result_count', 0)}`",
        f"- Candidate runtime valid promotion results: `{candidate_runtime_evidence.get('valid_result_count', 0)}/{candidate_runtime_evidence.get('result_count', 0)}`",
        f"- Missing candidate runtime results: `{candidate_runtime_evidence.get('missing_result_count', 0)}`",
        f"- Missing-model preflight blocks: `{candidate_runtime_evidence.get('preflight_blocked_missing_model_count', 0)}/{candidate_runtime_evidence.get('result_count', 0)}`",
        f"- Candidate YAML preflight valid: `{candidate_template_status.get('valid')}`",
        f"- Candidate YAML templates valid: `{candidate_template_status.get('valid_template_count', 0)}/{candidate_template_status.get('template_count', 0)}`",
        f"- Candidate YAML CLI validate/plan valid: `{sum(1 for template in candidate_templates if isinstance(template, dict) and template.get('cli_preflight_ok') is True)}/{len(candidate_templates)}`",
        "",
        "Candidate runtime result status:",
        "",
        "| Scenario | Capability | Role | File | Status | Required Evidence | Observed Evidence | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    candidate_runtime_rows = (
        candidate_runtime_evidence.get("results")
        if isinstance(candidate_runtime_evidence.get("results"), list)
        else []
    )
    if candidate_runtime_rows:
        for row in candidate_runtime_rows:
            if not isinstance(row, dict):
                continue
            summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
            errors = ", ".join(str(error) for error in row.get("errors") or [])
            required = "; ".join(str(item) for item in row.get("required_evidence") or [])
            observed = row.get("observed_evidence") if isinstance(row.get("observed_evidence"), dict) else {}
            observed_summary = (
                f"yaml_logs={observed.get('yaml_command_log_count')}; "
                f"yaml_ok={observed.get('yaml_commands_ok')}; "
                f"yaml_success={observed.get('yaml_successful_command_count')}/"
                f"{observed.get('yaml_command_log_count')}; "
                f"preflight_ok={observed.get('model_preflight_ok')}; "
                f"stream_ok={observed.get('stream_ok')}; "
                f"screenshot_fresh={observed.get('screenshot_fresh')}; "
                f"delivery={_compact_string_list(observed.get('delivery_outputs') or []) or 'none'}; "
                f"detections={observed.get('max_detections')}; "
                f"alerts={observed.get('matching_alerts')}; "
                f"unexpected={observed.get('unexpected_alerts')}; "
                f"suppressed={_compact_string_list(observed.get('suppressed_capabilities') or []) or 'none'}; "
                f"model_invocations={_compact_mapping(observed.get('model_invocations') or {})}"
            )
            lines.append(
                "| "
                f"`{_md_cell(row.get('scenario_id'))}` | "
                f"`{_md_cell(row.get('capability'))}` | "
                f"`{_md_cell(row.get('role'))}` | "
                f"`{_md_cell(row.get('path'))}` | "
                f"`{_md_cell(summary.get('status') or ('missing' if not row.get('exists') else 'unknown'))}` | "
                f"{_md_cell(required or 'none')} | "
                f"{_md_cell(observed_summary)} | "
                f"{_md_cell(errors or 'none')} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | none | none | none |")
    lines.extend([
        "",
        "Candidate YAML templates validated by this readiness run:",
        "",
    ])
    if candidate_templates:
        for template in candidate_templates:
            lines.append(
                f"- `{template.get('scenario_id')}`: config=`{template.get('config_path')}`, "
                f"expected_result=`{template.get('expected_result_path')}`, "
                f"yaml_valid=`{template.get('valid')}`, "
                f"cli_preflight_ok=`{template.get('cli_preflight_ok')}`, "
                f"required_model_plan_ok=`{template.get('required_model_plan_ok')}`, "
                f"one_at_a_time_ok=`{template.get('one_at_a_time_ok')}`"
            )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "Expected candidate result paths:",
        "",
        f"- `{candidate_result_dir}/factory_missing_apron_active_closed_set.json`",
        f"- `{candidate_result_dir}/factory_apron_false_positive_guard_closed_set.json`",
        f"- `{candidate_result_dir}/factory_apron_detector_window_suppression_closed_set.json`",
        f"- `{candidate_result_dir}/factory_missing_harness_active_closed_set.json`",
        f"- `{candidate_result_dir}/factory_harness_false_positive_guard_closed_set.json`",
        f"- `{candidate_result_dir}/factory_harness_detector_window_suppression_closed_set.json`",
        "",
        "For each candidate YAML config, the manifest already carries a matching scenario ID. Run one block at a time: these candidate YAML files set `site.merge_existing=false`, so each apply replaces the active camera config with exactly one candidate scenario. Run validate/plan/apply with the SSH-facing site tool, then execute the scenario:",
        "",
        "```bash",
    ])
    if candidate_templates:
        for template in candidate_templates:
            scenario_id = str(template.get("scenario_id") or "")
            config_path = str(template.get("config_path") or "")
            if not scenario_id or not config_path:
                continue
            commands = template.get("commands") if isinstance(template.get("commands"), dict) else {}
            lines.extend([
                f"# {scenario_id}",
                str(commands.get("backup") or f".venv/bin/python scripts/safetylens_site.py export --output qa/video_eval/results/site_config_backups/before_{scenario_id}.yaml"),
                str(commands.get("validate") or f".venv/bin/python scripts/safetylens_site.py --config {config_path} validate"),
                str(commands.get("plan") or f".venv/bin/python scripts/safetylens_site.py --config {config_path} plan"),
                str(commands.get("apply") or f".venv/bin/python scripts/safetylens_site.py --config {config_path} apply --yes"),
                str(commands.get("run") or f".venv/bin/python scripts/video_eval.py run --scenario {scenario_id}"),
                str(commands.get("restore") or f".venv/bin/python scripts/safetylens_site.py --config qa/video_eval/results/site_config_backups/before_{scenario_id}.yaml apply --yes"),
                "",
            ])
    else:
        lines.extend([
            ".venv/bin/python scripts/safetylens_site.py --config qa/video_eval/focused/<candidate>.yaml validate",
            ".venv/bin/python scripts/safetylens_site.py --config qa/video_eval/focused/<candidate>.yaml plan",
            ".venv/bin/python scripts/safetylens_site.py --config qa/video_eval/focused/<candidate>.yaml apply --yes",
            ".venv/bin/python scripts/video_eval.py run --scenario <candidate_scenario_id>",
        ])
    lines.extend([
        "```",
        "",
        "## Side-By-Side Promotion Reports",
        "",
        "The candidate promotion reports must use the same source manifest SHA as the current readiness handoff, carry `candidate_training_source_lineage`, and use the same `candidate_report_sha256` plus `candidate_selected_export.sha256` across apron, harness, and the model registry report. Stale reports from an older capture manifest, missing lineage, a different candidate report, or a different trained artifact must stay blocked.",
        "",
        "Apron:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_promotion_doctor.py \\",
        "  --capability apron_required \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --baseline-active {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['apron_required']['active']))} \\",
        f"  --baseline-guard {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['apron_required']['guard']))} \\",
        f"  --baseline-suppression {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['apron_required']['suppression']))} \\",
        f"  --candidate-active {candidate_result_dir}/factory_missing_apron_active_closed_set.json \\",
        f"  --candidate-guard {candidate_result_dir}/factory_apron_false_positive_guard_closed_set.json \\",
        f"  --candidate-suppression {candidate_result_dir}/factory_apron_detector_window_suppression_closed_set.json \\",
        f"  --out {apron_promotion}",
        "```",
        "",
        "Harness:",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_promotion_doctor.py \\",
        "  --capability harness_required \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --baseline-active {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['harness_required']['active']))} \\",
        f"  --baseline-guard {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['harness_required']['guard']))} \\",
        f"  --baseline-suppression {_rel(_result_path(result_dir, CAPABILITY_SCENARIOS['harness_required']['suppression']))} \\",
        f"  --candidate-active {candidate_result_dir}/factory_missing_harness_active_closed_set.json \\",
        f"  --candidate-guard {candidate_result_dir}/factory_harness_false_positive_guard_closed_set.json \\",
        f"  --candidate-suppression {candidate_result_dir}/factory_harness_detector_window_suppression_closed_set.json \\",
        f"  --out {harness_promotion}",
        "```",
        "",
        "## Jetson Full Gate",
        "",
        f"- Current report: `{jetson_full_gate.get('path') or jetson_gate}`",
        f"- Current status: `{jetson_full_gate.get('status') or optional.get('jetson_gate')}`",
        f"- Gate status: `{jetson_full_gate.get('gate_status')}`",
        f"- Production gate: `{jetson_full_gate.get('production_gate')}`",
        f"- Candidate report present: `{(jetson_full_gate.get('candidate_report') or {}).get('present')}`",
        f"- Raw benchmark present: `{(jetson_full_gate.get('raw_benchmark') or {}).get('present')}`",
        f"- Three-camera soak present: `{(jetson_full_gate.get('soak_report') or {}).get('present')}`",
        "",
        "Jetson template handoff:",
        "",
        f"- Template contract status: `{jetson_template_handoff.get('status')}`",
        f"- Valid templates: `{jetson_template_handoff.get('valid_template_contract_count', 0)}/{jetson_template_handoff.get('template_count', 0)}`",
        f"- Identity-stamped templates: `{jetson_template_handoff.get('identity_stamped_count', 0)}/{jetson_template_handoff.get('template_count', 0)}`",
        f"- Boundary: {jetson_template_handoff.get('evidence_boundary', 'templates are not evidence')}",
        "",
        "The Jetson raw benchmark and soak report must reference the same candidate report SHA and candidate export SHA as the apron/harness promotion plus model-registry reports. Pass `--candidate-report` so the Jetson doctor derives both expected identity values from the reviewed candidate report and verifies the candidate `seed_export_import_manifest` preserves the `source_recheck` path, SHA, and non-approval boundary from source review. Only use explicit hash flags as an extra cross-check. The soak report must also include positive `per_class_alert_count` values for `apron_required` and `harness_required`, visible-PPE `false_positive_guard` telemetry for both capabilities with zero matching alerts, and detector-window suppression telemetry for both capabilities with zero detections, zero alerts, and zero model invocations.",
        "",
        "Current Jetson gate errors:",
        "",
        *(
            [f"- `{error}`" for error in (jetson_full_gate.get("errors") or [])]
            if jetson_full_gate.get("errors")
            else ["- none"]
        ),
        "",
        "Current next Jetson gates:",
        "",
        *(
            [f"- `{item}`" for item in (jetson_full_gate.get("next_required_gates") or [])]
            if jetson_full_gate.get("next_required_gates")
            else ["- none"]
        ),
        "",
        "Write fillable raw benchmark and soak templates on the target box before collecting evidence:",
        "",
        "```bash",
        ".venv/bin/python scripts/jetson_benchmark_doctor.py \\",
        "  --model-pack qa/video_eval/model_packs.yaml \\",
        "  --pack factory_ppe_3cam \\",
        "  --model apron-harness-ppe.onnx \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --write-raw-template {raw_benchmark_template}",
        "",
        ".venv/bin/python scripts/jetson_benchmark_doctor.py \\",
        "  --model-pack qa/video_eval/model_packs.yaml \\",
        "  --pack factory_ppe_3cam \\",
        "  --model apron-harness-ppe.onnx \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --write-soak-template {soak_report_template}",
        "```",
        "",
        "The template files are not evidence. They intentionally contain zero/placeholder metrics and must fail the production gate until replaced with real target-device measurements and runtime telemetry.",
        "",
        "Collect the raw candidate benchmark with candidate identity stamped into the output:",
        "",
        "```bash",
        ".venv/bin/python scripts/benchmark_yolo_jetson.py \\",
        "  --models models/ppe_closed_set_candidate/apron-harness-ppe.onnx \\",
        "  --frames-dir /host_tmp \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --out {raw_benchmark}",
        "```",
        "",
        "When `--candidate-report` is supplied, the raw benchmark runner requires exactly one model and writes the candidate-report SHA plus selected-export SHA into the top-level JSON and model row. The Jetson doctor rejects the gate if these values do not match the soak, promotion, and registry reports.",
        "",
        "Build the soak report from measured target-device metrics and YAML-only candidate scenario outputs:",
        "",
        "```bash",
        ".venv/bin/python scripts/jetson_benchmark_doctor.py \\",
        "  --model-pack qa/video_eval/model_packs.yaml \\",
        "  --pack factory_ppe_3cam \\",
        "  --model apron-harness-ppe.onnx \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --soak-metrics {soak_metrics} \\",
        f"  --active-result apron_required={candidate_result_dir}/factory_missing_apron_active_closed_set.json \\",
        f"  --active-result harness_required={candidate_result_dir}/factory_missing_harness_active_closed_set.json \\",
        f"  --guard-result apron_required={candidate_result_dir}/factory_apron_false_positive_guard_closed_set.json \\",
        f"  --guard-result harness_required={candidate_result_dir}/factory_harness_false_positive_guard_closed_set.json \\",
        f"  --suppression-result apron_required={candidate_result_dir}/factory_apron_detector_window_suppression_closed_set.json \\",
        f"  --suppression-result harness_required={candidate_result_dir}/factory_harness_detector_window_suppression_closed_set.json \\",
        f"  --build-soak-report {soak_report}",
        "```",
        "",
        "The soak metrics file must come from the target-device run and include camera_count, soak_minutes, fps_per_camera, mean_latency_ms, p95_latency_ms, model_server_mean_latency_ms_per_request, ram_mb, gpu_utilization_percent, and stream_restarts. The builder derives alert counts, visible-PPE guard counts, and detector-window suppression telemetry from the result JSONs; it does not edit counts to pass.",
        "",
        "Canonical command:",
        "",
        "```bash",
        ".venv/bin/python scripts/jetson_benchmark_doctor.py \\",
        "  --pack factory_ppe_3cam \\",
        "  --model apron-harness-ppe.onnx \\",
        f"  --candidate-report {candidate_report} \\",
        f"  --raw-benchmark {raw_benchmark} \\",
        f"  --soak-report {soak_report} \\",
        "  --require-full-gate \\",
        f"  --out {jetson_gate}",
        "```",
        "",
        "## Final Readiness Recheck",
        "",
        "```bash",
        ".venv/bin/python scripts/model_pack_evidence_doctor.py \\",
        "  --out qa/video_eval/results/model_pack_evidence_doctor.json",
        "",
        ".venv/bin/python scripts/apron_harness_readiness_doctor.py \\",
        "  --out qa/video_eval/results/apron_harness_readiness_doctor.json \\",
        f"  --capture-manifest {reviewed_capture_manifest} \\",
        f"  --training-dataset-yaml {training_dataset_yaml} \\",
        f"  --seed-import-manifest {filled_seed_import_manifest} \\",
        f"  --apron-promotion-report {apron_promotion} \\",
        f"  --harness-promotion-report {harness_promotion} \\",
        f"  --model-registry-report {model_registry_report} \\",
        f"  --jetson-gate-report {jetson_gate} \\",
        f"  --seed-import-validation-summary-out {seed_import_validation_summary} \\",
        f"  --production-capture-matrix-validation-summary-out {production_capture_matrix_validation_summary}",
        "",
        f"jq '.' {seed_import_validation_summary}",
        f"jq '.' {production_capture_matrix_validation_summary}",
        "",
        ".venv/bin/python scripts/video_eval.py report",
        "```",
        "",
        "## Current Capture Progress",
        "",
        f"- Pilot missing label minimums: `{closed_set.get('missing_label_minimums')}`",
        f"- Production missing label minimums: `{closed_set.get('production_missing_label_minimums')}`",
    ])
    return "\n".join(lines) + "\n"


def _render_candidate_runtime_runbook(report: dict[str, Any]) -> str:
    candidate_templates = report.get("closed_set_candidate_yaml_templates") or {}
    if not isinstance(candidate_templates, dict):
        candidate_templates = {}
    candidate_runtime = report.get("closed_set_candidate_runtime_evidence") or {}
    if not isinstance(candidate_runtime, dict):
        candidate_runtime = {}
    runtime_results = {
        str(result.get("scenario_id")): result
        for result in candidate_runtime.get("results") or []
        if isinstance(result, dict) and result.get("scenario_id")
    }
    template_rows = [
        row
        for row in candidate_templates.get("templates") or []
        if isinstance(row, dict)
    ]
    lines = [
        "# Apron/Harness Candidate Runtime Runbook",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Scope",
        "",
        f"- Required model key: `{PLANNED_MODEL_KEY}`",
        f"- Current runtime evidence valid: `{candidate_runtime.get('valid')}`",
        f"- Current result files present: `{candidate_runtime.get('present_result_count')}/{candidate_runtime.get('result_count')}`",
        f"- Current missing-model preflight blocks: `{candidate_runtime.get('preflight_blocked_missing_model_count')}/{candidate_runtime.get('result_count')}`",
        "- Run exactly one scenario at a time. Each YAML file uses `site.merge_existing=false` and one camera.",
        "- Do not promote production until active, false-positive guard, detector-window suppression, side-by-side promotion, registry, and Jetson gates pass.",
        "",
        "## Preconditions",
        "",
        "1. `ppe_closed_set_candidate` has been copied into the model registry from a passing candidate report.",
        "2. `models/ppe_closed_set_candidate/apron-harness-ppe.onnx` and its `.registry.json` sidecar match the candidate report SHA.",
        "3. The backend can load the registered model before polling runtime evidence.",
        "4. The current production gate packet validates cleanly against the readiness report.",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_readiness_doctor.py \\",
        "  --validate-production-gate-packet qa/video_eval/results/apron_harness_production_gate_packet.json \\",
        "  --readiness-report qa/video_eval/results/apron_harness_readiness_doctor.json",
        "```",
        "",
        "## Runner",
        "",
        "Use plan mode first; execute mode refuses to run until the closed-set candidate is registered with a model artifact and registry sidecar.",
        "`--refresh-blocked-preflight` is only for refreshing blocked JSON after manifest/config changes while the model is still missing; it must fail if any scenario produces real detection evidence or does not block on `ppe_closed_set_candidate`.",
        "",
        "```bash",
        ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --json",
        ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --refresh-blocked-preflight --json \\",
        "  --out qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json",
        ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --execute --json",
        "```",
        "",
        "## One-Detection Runtime Sequence",
        "",
    ]
    if not template_rows:
        lines.append("- No candidate YAML templates were found.")
        return "\n".join(lines) + "\n"

    for index, row in enumerate(template_rows, start=1):
        scenario_id = str(row.get("scenario_id") or "")
        result = runtime_results.get(scenario_id, {})
        commands = row.get("commands") if isinstance(row.get("commands"), dict) else {}
        lines.extend([
            f"### {index}. `{scenario_id}`",
            "",
            f"- Capability: `{_md_cell(row.get('capability'))}`",
            f"- Role: `{_md_cell(row.get('role'))}`",
            f"- Config: `{_md_cell(row.get('config_path'))}`",
            f"- Expected result: `{_md_cell(row.get('expected_result_path'))}`",
            f"- Required model plan OK: `{row.get('required_model_plan_ok')}`",
            f"- One-at-a-time OK: `{row.get('one_at_a_time_ok')}`",
            f"- Capability window OK: `{row.get('window_ok')}`",
            f"- Current result valid: `{result.get('valid')}`",
            f"- Current preflight blocked missing model: `{result.get('preflight_blocked_missing_required_model')}`",
            "",
            "```bash",
        ])
        for name in ("backup", "validate", "plan", "apply", "run", "restore"):
            command = commands.get(name)
            if command:
                lines.append(str(command))
        lines.extend([
            "```",
            "",
            "Required evidence:",
            "",
        ])
        required_evidence = result.get("required_evidence") if isinstance(result, dict) else []
        if required_evidence:
            for item in required_evidence:
                lines.append(f"- {_md_cell(item)}")
        else:
            lines.append("- Runtime result has not published required evidence yet.")
        errors = result.get("errors") if isinstance(result, dict) else []
        if errors:
            lines.extend([
                "",
                "Current blocking errors:",
                "",
            ])
            for error in errors:
                lines.append(f"- `{_md_cell(error)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _summary_int(summary: dict[str, Any], key: str) -> int:
    try:
        return int(summary.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_runtime_blocked_missing_model_scenario_ids(
    candidate_runtime: dict[str, Any],
) -> list[str]:
    scenario_ids: list[str] = []
    for result in candidate_runtime.get("results", []):
        if not isinstance(result, dict):
            continue
        errors = result.get("errors") if isinstance(result.get("errors"), list) else []
        blocked_missing_model = (
            result.get("preflight_blocked_missing_required_model") is True
            or "blocked_missing_required_model_preflight" in {str(error) for error in errors}
        )
        if not blocked_missing_model:
            continue
        scenario_id = result.get("scenario_id")
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        if not scenario_id:
            scenario_id = summary.get("scenario_id")
        if not scenario_id and result.get("path"):
            scenario_id = Path(str(result["path"])).stem
        if scenario_id:
            scenario_ids.append(str(scenario_id))
    return scenario_ids


def _capture_progress_summary(
    progress: dict[str, Any],
    csv_info: dict[str, Any],
    manifest_info: dict[str, Any],
) -> dict[str, Any]:
    reconciliation_raw = progress.get("manifest_reconciliation")
    reconciliation = reconciliation_raw if isinstance(reconciliation_raw, dict) else {}
    return {
        "path": progress.get("path") or csv_info.get("path"),
        "gate_passed": progress.get("gate_passed") is True,
        "row_count": _summary_int(progress, "row_count"),
        "ready_rows": _summary_int(progress, "ready_rows"),
        "target_labeled_examples": _summary_int(progress, "target_labeled_examples"),
        "captured_examples": _summary_int(progress, "captured_examples"),
        "labeled_examples": _summary_int(progress, "labeled_examples"),
        "missing_labeled_examples": _summary_int(progress, "missing_labeled_examples"),
        "unapproved_rows": _summary_int(progress, "unapproved_rows"),
        "unsafe_storage_rows": _summary_int(progress, "unsafe_storage_rows"),
        "manifest_reconciliation_checked": reconciliation.get("checked") is True,
        "manifest_reconciliation_gate_passed": reconciliation.get("gate_passed") is True,
        "required_labeled_images_per_class": reconciliation.get("required_labeled_images_per_class") or {},
        "missing_manifest_counts": reconciliation.get("missing_manifest_counts") or {},
        "csv_generated": csv_info.get("generated") is True,
        "csv_exists": csv_info.get("exists") is True,
        "csv_sha256": csv_info.get("sha256"),
        "manifest_generated": manifest_info.get("generated") is True,
        "manifest_exists": manifest_info.get("exists") is True,
        "manifest_sha256": manifest_info.get("sha256"),
        "blockers": [str(blocker) for blocker in (progress.get("blockers") or [])],
        "errors": [str(error) for error in (progress.get("errors") or [])],
        "warnings": [str(warning) for warning in (progress.get("warnings") or [])],
    }


def _seed_import_validation_summary(
    seed_import_review: dict[str, Any],
    seed_source_review: dict[str, Any],
) -> dict[str, Any]:
    blockers = [str(blocker) for blocker in (seed_import_review.get("blockers") or [])]
    return {
        "ok": seed_import_review.get("ok") is True and seed_import_review.get("gate_passed") is True,
        "gate_passed": seed_import_review.get("gate_passed") is True,
        "status": (
            "ready_for_training_import"
            if seed_import_review.get("gate_passed") is True
            else "blocked_pending_import_manifest"
        ),
        "candidate_count": _summary_int(seed_source_review, "candidate_count"),
        "training_usable_count": _summary_int(seed_source_review, "training_usable_count"),
        "capability_coverage": list(seed_source_review.get("capability_coverage") or []),
        "training_usable_capability_coverage": list(
            seed_source_review.get("training_usable_capability_coverage") or []
        ),
        "missing_training_capabilities": list(seed_source_review.get("missing_training_capabilities") or []),
        "import_manifest_review": {
            "ok": seed_import_review.get("ok") is True,
            "gate_passed": seed_import_review.get("gate_passed") is True,
            "path": seed_import_review.get("path"),
            "source_review_sha256_matches": seed_import_review.get("source_review_sha256_matches") is True,
            "included_count": _summary_int(seed_import_review, "included_count"),
            "approved_count": _summary_int(seed_import_review, "approved_count"),
            "blocker_count": len(blockers),
            "blockers": blockers[:8],
        },
        "next_action": (
            "fill human/legal source review evidence, set include_in_training only for approved exports, "
            "stamp raw_export_sha256/local reviewed YOLO export ZIP SHA, then rerun validate-import-manifest"
        ),
    }


def _production_capture_matrix_validation_summary(
    *,
    progress: dict[str, Any],
    counts: dict[str, Any],
) -> dict[str, Any]:
    blockers = [str(blocker) for blocker in (progress.get("blockers") or [])]
    capabilities: dict[str, Any] = {}
    for capability, value in (progress.get("capabilities") or {}).items():
        if isinstance(value, dict):
            capabilities[str(capability)] = {
                "row_count": _summary_int(value, "row_count"),
                "ready_rows": _summary_int(value, "ready_rows"),
                "target_labeled_examples": _summary_int(value, "target_labeled_examples"),
                "captured_examples": _summary_int(value, "captured_examples"),
                "labeled_examples": _summary_int(value, "labeled_examples"),
                "missing_labeled_examples": _summary_int(value, "missing_labeled_examples"),
            }
    return {
        "ok": progress.get("gate_passed") is True,
        "mode": "production",
        "schema_only": True,
        "counts": counts,
        "capture_matrix_progress": {
            "ok": progress.get("ok") is True,
            "gate_passed": progress.get("gate_passed") is True,
            "path": progress.get("path"),
            "row_count": _summary_int(progress, "row_count"),
            "ready_rows": _summary_int(progress, "ready_rows"),
            "target_labeled_examples": _summary_int(progress, "target_labeled_examples"),
            "captured_examples": _summary_int(progress, "captured_examples"),
            "labeled_examples": _summary_int(progress, "labeled_examples"),
            "missing_labeled_examples": _summary_int(progress, "missing_labeled_examples"),
            "unapproved_rows": _summary_int(progress, "unapproved_rows"),
            "unsafe_storage_rows": _summary_int(progress, "unsafe_storage_rows"),
            "capabilities": capabilities,
            "blocker_count": len(blockers),
            "blockers": blockers[:8],
        },
        "next_action": (
            "complete capture matrix rows with approved review_status, cleared permission, external "
            "raw_storage_ref, captured/labeled example counts, then import reviewed labels into a reviewed manifest"
        ),
    }


def _capture_matrix_starter_rows(matrix_csv_path: Path) -> list[dict[str, Any]]:
    if not matrix_csv_path.exists():
        return []
    try:
        rows = list(csv.DictReader(matrix_csv_path.read_text(encoding="utf-8").splitlines()))
    except Exception:
        return []
    wanted = [
        ("apron_required_closed_set_capture", "positive_variant"),
        ("apron_required_closed_set_capture", "hard_negative"),
        ("harness_required_closed_set_capture", "positive_variant"),
        ("harness_required_closed_set_capture", "hard_negative"),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        batch_id = str(row.get("batch_id") or "")
        capture_type = str(row.get("capture_type") or "")
        key = (batch_id, capture_type)
        if key not in wanted or key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "batch_id": batch_id,
                "row_id": row.get("row_id"),
                "target_capability": row.get("target_capability"),
                "capture_type": capture_type,
                "variant_or_tag": row.get("variant_or_tag"),
                "recommended_examples": _summary_int(row, "recommended_examples"),
                "required_label_classes": [
                    value.strip()
                    for value in str(row.get("required_label_classes") or "").split(";")
                    if value.strip()
                ],
                "camera_angles": row.get("camera_angles"),
                "distance_bands": row.get("distance_bands"),
                "lighting": row.get("lighting"),
                "motion_blur": row.get("motion_blur"),
                "notes": row.get("notes"),
                "operator_fill_fields": [
                    "captured_examples",
                    "labeled_examples",
                    "review_status=approved",
                    "permission=controlled_capture_cleared",
                    "raw_storage_ref=non_repo_external_storage",
                    "owner",
                    "due_date",
                ],
            }
        )
        if len(seen) == len(wanted):
            break
    return selected


def _post_capture_evidence_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "filled_production_capture_matrix",
            "required_artifact": "qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            "validator": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            "pass_signal": "capture_matrix_progress.gate_passed=true",
            "must_prove": [
                "21 ready rows",
                "0 missing labeled examples",
                "0 unapproved rows",
                "0 unsafe storage rows",
            ],
        },
        {
            "id": "filled_production_label_review_csv",
            "required_artifact": "/path/to/filled/apron_harness_production_label_review.csv",
            "validator": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv",
            "pass_signal": "LABEL_REVIEW_VALIDATION: gate=pass",
            "must_prove": [
                "review_status=approved",
                "permission=controlled_capture_cleared",
                "raw_storage_ref=non_repo_external_storage",
                "reviewed clip metadata",
            ],
        },
        {
            "id": "reviewed_capture_manifest_sidecar",
            "required_artifact": "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
            "validator": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --mode production",
            "pass_signal": "updated_manifest_validation.ok=true",
            "must_prove": [
                "updated_manifest_validation.schema_only=false",
                "updated_manifest_validation.mode=production",
                "updated_manifest_sha256 matches reviewed manifest",
                "imported_label_count satisfies production counts",
            ],
        },
        {
            "id": "production_training_preflight",
            "required_artifact": "qa/video_eval/results/apron_harness_training_dataset.yaml",
            "validator": ".venv/bin/python scripts/apron_harness_train.py --data /path/to/cleared/dataset.yaml --capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv --capture-preflight-mode production --require-capture-preflight --model yolo26n.pt --device mps --out-plan /path/to/cleared/apron_harness_training_plan.json",
            "pass_signal": "ready_to_train",
            "must_prove": [
                "dataset provenance SHA matches reviewed manifest",
                "capture matrix sidecar SHA matches matrix",
                "label-review import sidecar is valid",
                "missing-PPE policy is explicit",
            ],
        },
    ]


def _candidate_training_required_input_status(
    closed_set: dict[str, Any],
    *,
    source_recheck_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reviewed_capture_manifest = "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml"
    production_capture_matrix = "qa/video_eval/results/apron_harness_production_capture_matrix.csv"
    training_capture_preflight = closed_set.get("training_capture_preflight")
    if not isinstance(training_capture_preflight, dict):
        training_capture_preflight = {}
    label_review_import = training_capture_preflight.get("label_review_import_manifest")
    if not isinstance(label_review_import, dict):
        label_review_import = {}
    return {
        "dataset_yaml": {
            "path": "/path/to/cleared/dataset.yaml",
            "exists": False,
            "status": "required_user_supplied_reviewed_dataset_yaml",
        },
        "reviewed_capture_manifest": {
            "path": reviewed_capture_manifest,
            "exists": _artifact_path(reviewed_capture_manifest).exists(),
            "status": "present"
            if _artifact_path(reviewed_capture_manifest).exists()
            else "missing_reviewed_capture_manifest",
        },
        "seed_import_manifest": {
            "path": "/path/to/filled/apron_harness_seed_import_manifest.yaml",
            "exists": False,
            "status": "required_filled_approved_seed_import_manifest",
        },
        "source_recheck_artifact": source_recheck_artifact or {},
        "production_capture_matrix": {
            "path": production_capture_matrix,
            "exists": _artifact_path(production_capture_matrix).exists(),
            "gate_passed": training_capture_preflight.get("gate_passed") is True,
            "missing_labeled_examples": _summary_int(
                training_capture_preflight,
                "missing_labeled_examples",
            ),
            "unapproved_rows": _summary_int(training_capture_preflight, "unapproved_rows"),
            "unsafe_storage_rows": _summary_int(training_capture_preflight, "unsafe_storage_rows"),
        },
        "label_review_import_sidecar": {
            "path": label_review_import.get("path")
            or "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
            "valid": label_review_import.get("valid") is True,
            "error": label_review_import.get("error"),
        },
    }


def _production_gate_packet(report: dict[str, Any]) -> dict[str, Any]:
    closed_set = report.get("closed_set_handoff") or {}
    if not isinstance(closed_set, dict):
        closed_set = {}
    candidate_runtime = report.get("closed_set_candidate_runtime_evidence") or {}
    candidate_templates = report.get("closed_set_candidate_yaml_templates") or {}
    capture_progress = report.get("closed_set_capture_progress_summary")
    if not isinstance(capture_progress, dict):
        capture_progress = {}
    production_capture_progress = capture_progress.get("production")
    if not isinstance(production_capture_progress, dict):
        production_capture_progress = {}
    production_capture_summary = report.get("production_capture_matrix_validation_summary")
    if not isinstance(production_capture_summary, dict):
        production_capture_summary = {}
    production_capture_matrix_path = production_capture_progress.get("path") or (
        production_capture_summary.get("capture_matrix_progress") or {}
    ).get("path")
    production_capture_matrix_artifact = (
        _artifact_path(production_capture_matrix_path)
        if production_capture_matrix_path
        else DEFAULT_PRODUCTION_CAPTURE_MATRIX_CSV
    )
    production_capture_matrix_sidecar = production_capture_matrix_artifact.with_suffix(
        production_capture_matrix_artifact.suffix + ".manifest.json"
    )
    production_capture_matrix_sidecar_doc: dict[str, Any] = {}
    if production_capture_matrix_sidecar.exists():
        try:
            loaded_sidecar = _load_json(production_capture_matrix_sidecar)
            if isinstance(loaded_sidecar, dict):
                production_capture_matrix_sidecar_doc = loaded_sidecar
        except Exception:
            production_capture_matrix_sidecar_doc = {}
    minimum_path = report.get("seed_source_minimum_approval_path")
    if not isinstance(minimum_path, dict):
        minimum_path = {}
    next_source_reviews = report.get("seed_source_next_review_queue")
    if not isinstance(next_source_reviews, list):
        next_source_reviews = []
    minimum_source_refs = minimum_path.get("minimum_review_source_refs")
    if not isinstance(minimum_source_refs, list):
        minimum_source_refs = []
    review_by_ref = {
        str(review.get("source_ref")): review
        for review in next_source_reviews
        if isinstance(review, dict) and review.get("source_ref")
    }
    minimum_review_sources = [
        review_by_ref[str(source_ref)]
        for source_ref in minimum_source_refs
        if str(source_ref) in review_by_ref
    ]
    seed_source_review = report.get("seed_source_review")
    if not isinstance(seed_source_review, dict):
        seed_source_review = {}
    source_recheck_artifact = seed_source_review.get("source_recheck")
    if not isinstance(source_recheck_artifact, dict):
        source_recheck_artifact = {}
    post_capture_evidence_checklist = _post_capture_evidence_checklist()
    return {
        "kind": "apron_harness_production_gate_packet",
        "generated_at": report.get("generated_at"),
        "scope": READINESS_SCOPE,
        "status": {
            "sales_status": report.get("sales_status"),
            "pilot_gate_passed": report.get("pilot_gate_passed") is True,
            "production_gate_passed": report.get("production_gate_passed") is True,
            "production_blocker_count": _summary_int(report, "production_blocker_count"),
            "factory_ppe_pack_status": report.get("factory_ppe_pack_status"),
            "runtime_handoff_status": report.get("runtime_handoff_status"),
            "sourcing_status": report.get("sourcing_status"),
        },
        "inputs": report.get("inputs") if isinstance(report.get("inputs"), dict) else {},
        "optional_gate_status": report.get("optional_gate_status")
        if isinstance(report.get("optional_gate_status"), dict)
        else {},
        "production_blockers": list(report.get("production_blockers") or []),
        "next_actions": list(report.get("next_actions") or []),
        "next_required_evidence": list(report.get("next_required_evidence") or []),
        "post_capture_evidence_checklist": post_capture_evidence_checklist,
        "seed_import_manifest_validation_summary": report.get("seed_import_manifest_validation_summary") or {},
        "production_capture_matrix_validation_summary": (
            report.get("production_capture_matrix_validation_summary") or {}
        ),
        "capture_progress_summary": report.get("closed_set_capture_progress_summary") or {},
        "training_readiness": closed_set.get("training_readiness") or {},
        "training_capture_preflight": closed_set.get("training_capture_preflight") or {},
        "production_training_plan_preflight": (
            closed_set.get("production_training_plan_preflight") or {}
        ),
        "model_registry_handoff": report.get("model_registry_handoff") or {},
        "jetson_template_handoff": report.get("jetson_template_handoff") or {},
        "jetson_full_gate": report.get("jetson_full_gate") or {},
        "candidate_runtime_gate": {
            "yaml_templates_valid": candidate_templates.get("valid") is True,
            "runtime_evidence_valid": candidate_runtime.get("valid") is True,
            "present_result_count": _summary_int(candidate_runtime, "present_result_count"),
            "valid_promotion_result_count": _summary_int(
                candidate_runtime,
                "valid_promotion_result_count",
            ),
            "preflight_blocked_missing_model_count": _summary_int(
                candidate_runtime,
                "preflight_blocked_missing_model_count",
            ),
            "blocked_missing_model_scenario_ids": (
                _candidate_runtime_blocked_missing_model_scenario_ids(candidate_runtime)
            ),
            "required_contract": [
                "apply one YAML scenario at a time with safetylens_site.py validate/plan/apply",
                "active windows require detections, matching alerts, delivery evidence, screenshots, and ppe_closed_set_candidate invocations",
                "false-positive guards require visible PPE telemetry with zero missing-PPE alerts",
                "inactive detector windows require zero detections, zero alerts, suppressed capability telemetry, and zero ppe_closed_set_candidate invocations",
            ],
        },
        "candidate_runtime_execution_plan": {
            "status": (
                "blocked_until_ppe_closed_set_candidate_registered"
                if candidate_runtime.get("valid") is not True
                else "complete"
            ),
            "required_model_key": PLANNED_MODEL_KEY,
            "one_detection_at_a_time": True,
            "runbook": report.get("candidate_runtime_runbook") or {},
            "runner": {
                "path": _rel(DEFAULT_CANDIDATE_RUNTIME_RUNNER),
                "sha256": _sha256_file(DEFAULT_CANDIDATE_RUNTIME_RUNNER)
                if DEFAULT_CANDIDATE_RUNTIME_RUNNER.exists()
                else None,
                "plan_command": ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --json",
                "refresh_blocked_preflight_command": ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --refresh-blocked-preflight --json --out qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json",
                "refresh_blocked_preflight_artifact": {
                    "path": _rel(DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT),
                    "exists": DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT.exists(),
                    "sha256": _sha256_file(DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT)
                    if DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT.exists()
                    else None,
                    "evidence_boundary": "durable proof that the current scenario JSON was refreshed to blocked_missing_required_model_preflight; not detection evidence and not a production pass",
                },
                "execute_command": ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py --execute --json",
                "guardrail": "execute mode refuses to run until ppe_closed_set_candidate is registered with model artifact and registry sidecar present; refresh-blocked-preflight mode is only allowed to refresh blocked missing-model preflight JSON",
            },
            "scenario_order": [
                row.get("scenario_id")
                for row in candidate_templates.get("templates", [])
                if isinstance(row, dict)
            ],
            "steps": [
                {
                    "step": index + 1,
                    "scenario_id": row.get("scenario_id"),
                    "capability": row.get("capability"),
                    "role": row.get("role"),
                    "config_path": row.get("config_path"),
                    "expected_result_path": row.get("expected_result_path"),
                    "commands": row.get("commands") or {},
                    "required_model_plan_ok": row.get("required_model_plan_ok") is True,
                    "one_at_a_time_ok": row.get("one_at_a_time_ok") is True,
                    "window_ok": row.get("window_ok") is True,
                    "expected_evidence": next(
                        (
                            result.get("required_evidence") or []
                            for result in candidate_runtime.get("results", [])
                            if isinstance(result, dict)
                            and result.get("scenario_id") == row.get("scenario_id")
                        ),
                        [],
                    ),
                    "current_result": next(
                        (
                            {
                                "path": result.get("path"),
                                "exists": result.get("exists") is True,
                                "valid": result.get("valid") is True,
                                "preflight_blocked_missing_required_model": result.get(
                                    "preflight_blocked_missing_required_model"
                                )
                                is True,
                                "errors": result.get("errors") or [],
                            }
                            for result in candidate_runtime.get("results", [])
                            if isinstance(result, dict)
                            and result.get("scenario_id") == row.get("scenario_id")
                        ),
                        {},
                    ),
                }
                for index, row in enumerate(candidate_templates.get("templates", []))
                if isinstance(row, dict)
            ],
            "success_criteria": [
                "run only one closed-set scenario at a time through scripts/video_eval.py",
                "each scenario records safetylens_site.py validate, plan, and apply --yes command evidence",
                "active-window scenarios require detections, matching alerts, delivery evidence, screenshots, and ppe_closed_set_candidate invocations",
                "false-positive guard scenarios require visible PPE telemetry, zero matching missing-PPE alerts, zero unexpected alerts, and ppe_closed_set_candidate invocations",
                "inactive detector-window scenarios require suppressed capability telemetry, zero emitted detections, zero alerts, and zero ppe_closed_set_candidate invocations",
                "production promotion remains blocked until side-by-side promotion, registry, and Jetson full gate pass",
            ],
        },
        "candidate_training_execution_plan": {
            "status": "blocked_until_reviewed_production_manifest_and_training_dataset",
            "required_model_key": PLANNED_MODEL_KEY,
            "training_model": DEFAULT_TRAINING_MODEL,
            "required_input_status": _candidate_training_required_input_status(
                closed_set,
                source_recheck_artifact=source_recheck_artifact,
            ),
            "runner": {
                "path": _rel(DEFAULT_CANDIDATE_TRAINING_RUNNER),
                "sha256": (
                    _sha256_file(DEFAULT_CANDIDATE_TRAINING_RUNNER)
                    if DEFAULT_CANDIDATE_TRAINING_RUNNER.exists()
                    else None
                ),
                "plan_command": ".venv/bin/python scripts/apron_harness_candidate_training_runner.py --json",
                "execute_command": ".venv/bin/python scripts/apron_harness_candidate_training_runner.py --execute --json --dataset-yaml /path/to/cleared/dataset.yaml --capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --training-result /path/to/cleared/apron_harness_yolo26n_result.json",
                "training_command": ".venv/bin/python scripts/apron_harness_candidate_training_runner.py --execute --run-training --json --dataset-yaml /path/to/cleared/dataset.yaml --capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --training-result /path/to/cleared/apron_harness_yolo26n_result.json",
                "guardrail": "execute mode refuses until reviewed production dataset YAML, capture manifest, seed-import manifest, source-recheck artifact, and capture matrix exist with matching hashes; actual training also requires --run-training, and registry copy requires both side-by-side promotion reports to be ready_for_runtime_registration with matching candidate-report SHA and selected-export SHA",
            },
            "steps": [
                {
                    "step": 1,
                    "id": "training_preflight_plan",
                    "command": ".venv/bin/python scripts/apron_harness_train.py --data /path/to/cleared/dataset.yaml --capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --capture-preflight-mode production --require-capture-preflight --model yolo26n.pt --device mps --epochs 100 --batch 8 --export-format onnx --out-plan /path/to/cleared/apron_harness_yolo26n_result.plan.json",
                    "pass_signal": "status=ready_to_train",
                    "writes": ["/path/to/cleared/apron_harness_yolo26n_result.plan.json"],
                },
                {
                    "step": 2,
                    "id": "train_export_candidate",
                    "command": ".venv/bin/python scripts/apron_harness_train.py --data /path/to/cleared/dataset.yaml --capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml --capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --capture-preflight-mode production --require-capture-preflight --model yolo26n.pt --device mps --epochs 100 --batch 8 --export-format onnx --out-plan /path/to/cleared/apron_harness_yolo26n_result.json --execute",
                    "writes": ["/path/to/cleared/apron_harness_yolo26n_result.json"],
                    "guardrail": "runner requires --run-training before this step is executed",
                },
                {
                    "step": 3,
                    "id": "candidate_doctor_report",
                    "command": ".venv/bin/python scripts/apron_harness_candidate_doctor.py --training-result /path/to/cleared/apron_harness_yolo26n_result.json --out qa/video_eval/results/apron_harness_candidate_report.json",
                    "pass_signal": "candidate ok=true",
                    "writes": ["qa/video_eval/results/apron_harness_candidate_report.json"],
                },
                {
                    "step": 4,
                    "id": "side_by_side_promotion_reports",
                    "commands": [
                        "see qa/video_eval/results/apron_harness_promotion_runbook.md apron side-by-side promotion command",
                        "see qa/video_eval/results/apron_harness_promotion_runbook.md harness side-by-side promotion command",
                    ],
                    "writes": [
                        "qa/video_eval/results/apron_closed_set_promotion_report.json",
                        "qa/video_eval/results/harness_closed_set_promotion_report.json",
                    ],
                    "pass_signal": "promotion_status=ready_for_runtime_registration for apron and harness",
                },
                {
                    "step": 5,
                    "id": "registry_copy_after_promotions",
                    "command": ".venv/bin/python scripts/apron_harness_model_registry_doctor.py --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --apron-promotion-report qa/video_eval/results/apron_closed_set_promotion_report.json --harness-promotion-report qa/video_eval/results/harness_closed_set_promotion_report.json --copy --out qa/video_eval/results/apron_harness_model_registry_report.json",
                    "writes": [
                        "qa/video_eval/results/apron_harness_model_registry_report.json",
                        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
                    ],
                    "guardrail": "do not copy until candidate doctor and both side-by-side promotion reports pass with matching candidate-report SHA and selected-export SHA; the registry doctor enforces this in --copy mode",
                },
            ],
            "success_criteria": [
                "training preflight status is ready_to_train on device mps",
                "dataset provenance SHA matches reviewed production capture manifest",
                "capture matrix and label-review sidecars are valid production evidence",
                "candidate doctor selects an accepted ONNX export and passes apron/harness metrics",
                "apron and harness side-by-side promotion reports use the same candidate report and selected-export SHA",
                "registry copy preserves the same candidate_report_sha256 and selected-export SHA",
            ],
        },
        "jetson_gate_execution_plan": {
            "status": "blocked_until_candidate_report_raw_benchmark_and_soak_report",
            "required_model_key": PLANNED_MODEL_KEY,
            "model": "apron-harness-ppe.onnx",
            "runner": {
                "path": _rel(DEFAULT_JETSON_GATE_RUNNER),
                "sha256": (
                    _sha256_file(DEFAULT_JETSON_GATE_RUNNER)
                    if DEFAULT_JETSON_GATE_RUNNER.exists()
                    else None
                ),
                "plan_command": ".venv/bin/python scripts/apron_harness_jetson_gate_runner.py --json",
                "execute_command": ".venv/bin/python scripts/apron_harness_jetson_gate_runner.py --execute --json --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json --soak-report /path/to/cleared/factory_ppe_3cam_soak.json",
                "guardrail": "execute mode refuses until candidate report, raw benchmark, and soak report are supplied as existing non-placeholder paths; the runner always invokes scripts/jetson_benchmark_doctor.py with --require-full-gate",
            },
            "templates": report.get("jetson_template_handoff") or {},
            "full_gate": report.get("jetson_full_gate") or {},
            "full_gate_command": ".venv/bin/python scripts/jetson_benchmark_doctor.py --model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam --model apron-harness-ppe.onnx --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json --soak-report /path/to/cleared/factory_ppe_3cam_soak.json --require-full-gate --out qa/video_eval/results/factory_ppe_jetson_gate.json",
            "steps": [
                {
                    "step": 1,
                    "id": "stamp_candidate_identity",
                    "command": ".venv/bin/python scripts/jetson_benchmark_doctor.py --model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam --model apron-harness-ppe.onnx --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --write-raw-template qa/video_eval/results/factory_ppe_raw_benchmark.template.json --write-soak-template qa/video_eval/results/factory_ppe_3cam_soak.template.json",
                    "pass_signal": "raw and soak templates carry candidate_report_sha256 and model_artifact_sha256 from the reviewed candidate",
                },
                {
                    "step": 2,
                    "id": "run_raw_benchmark",
                    "command": ".venv/bin/python scripts/benchmark_yolo_jetson.py --model models/ppe_closed_set_candidate/apron-harness-ppe.onnx --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --out qa/video_eval/results/factory_ppe_raw_benchmark.json",
                    "pass_signal": "raw benchmark has positive sample count, p95 latency, FPS, candidate_report_sha256, and model_artifact_sha256",
                },
                {
                    "step": 3,
                    "id": "run_three_camera_soak",
                    "command": ".venv/bin/python scripts/jetson_benchmark_doctor.py --model-pack qa/video_eval/model_packs.yaml --pack factory_ppe_3cam --model apron-harness-ppe.onnx --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --build-soak-report qa/video_eval/results/factory_ppe_3cam_soak.json --soak-metrics qa/video_eval/results/factory_ppe_3cam_soak_metrics.yaml --active-result apron_required=qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json --active-result harness_required=qa/video_eval/results/closed_set_candidate/factory_missing_harness_active_closed_set.json --guard-result apron_required=qa/video_eval/results/closed_set_candidate/factory_apron_false_positive_guard_closed_set.json --guard-result harness_required=qa/video_eval/results/closed_set_candidate/factory_harness_false_positive_guard_closed_set.json --suppression-result apron_required=qa/video_eval/results/closed_set_candidate/factory_apron_detector_window_suppression_closed_set.json --suppression-result harness_required=qa/video_eval/results/closed_set_candidate/factory_harness_detector_window_suppression_closed_set.json",
                    "pass_signal": "soak report proves active apron/harness alerts, false-positive guards, detector-window suppression, and candidate identity",
                },
                {
                    "step": 4,
                    "id": "validate_full_gate",
                    "command": ".venv/bin/python scripts/apron_harness_jetson_gate_runner.py --execute --json --candidate-report qa/video_eval/results/apron_harness_candidate_report.json --raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json --soak-report /path/to/cleared/factory_ppe_3cam_soak.json --out qa/video_eval/results/factory_ppe_jetson_gate.json",
                    "pass_signal": "factory_ppe_3cam Jetson full gate ok=true",
                },
            ],
            "success_criteria": [
                "candidate report SHA and selected-export SHA match across candidate, raw benchmark, soak report, side-by-side promotions, registry, and Jetson gate",
                "candidate seed_export_import_manifest preserves source_recheck path/SHA/non-approval boundary into the Jetson gate",
                "raw benchmark proves usable Jetson latency and FPS for apron-harness-ppe.onnx",
                "three-camera soak proves positive apron_required and harness_required alerts",
                "false-positive guard scenarios show visible PPE evidence and zero matching missing-PPE alerts",
                "inactive detector-window scenarios show zero detections, zero alerts, and zero ppe_closed_set_candidate invocations",
                "production promotion remains blocked until this full gate passes with --require-full-gate",
            ],
        },
        "promotion_guardrails": [
            "do not mark factory_ppe_3cam ready_to_sell_production_compliance while production_blockers is non-empty",
            "do not register ppe_closed_set_candidate until reviewed data, candidate doctor, side-by-side promotion, registry, and Jetson gates pass",
            "do not use public or commercial seed exports for training until human/legal source review and seed-import manifest gates pass",
            "do not treat fillable Jetson templates as benchmark evidence until candidate_report_sha256 and model_artifact_sha256 are stamped from a verified candidate",
        ],
        "first_unblock": {
            "id": "complete_human_legal_source_review_or_controlled_capture",
            "status": "blocked_until_human_legal_review_or_cleared_capture",
            "minimum_approval_path": minimum_path,
            "source_recheck_artifact": source_recheck_artifact,
            "minimum_review_sources": minimum_review_sources,
            "source_review_execution_plan": [
                {
                    "step": 1,
                    "id": "validate_source_review_bundle",
                    "command": ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
                    "pass_signal": "REVIEW_BUNDLE: ok=True",
                    "writes": [],
                },
                {
                    "step": 2,
                    "id": "fill_minimum_review_evidence",
                    "required_sources": minimum_source_refs,
                    "required_artifacts": [
                        {
                            "source_ref": review.get("source_ref"),
                            "review_packet_path": review.get("review_packet_path"),
                            "review_evidence_template_path": review.get(
                                "review_evidence_template_path"
                            ),
                            "review_prefill_path": review.get("review_prefill_path"),
                        }
                        for review in minimum_review_sources
                    ],
                    "stop_if": [
                        "license/export terms unclear",
                        "approval_status is not approved_for_training",
                        "raw_export_sha256 missing",
                        "customer-private or identifiable footage without explicit approval",
                    ],
                },
                {
                    "step": 3,
                    "id": "validate_seed_import_manifest",
                    "command": ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
                    "pass_signal": "IMPORT_MANIFEST: gate=pass",
                    "writes": [],
                },
                {
                    "step": 4,
                    "id": "materialize_approved_seed_exports",
                    "command": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --import-approved-seed-exports --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml --seed-import-camera-angle front --seed-import-distance-band medium --seed-import-lighting indoor_bright --seed-import-motion-blur low",
                    "writes": [
                        "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml",
                        "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml.seed_export_import.json",
                    ],
                    "pass_signal": "seed_export_import.valid=true",
                    "required_evidence": [
                        "source_recheck.path and source_recheck.sha256 match the seed-source review",
                        "source_recheck.evidence_boundary preserves the non-approval boundary that source research does not approve training",
                        "seed_source_review_sha256 and seed_import_manifest_sha256 are present",
                        "YOLO export preflight is preserved for every materialized import",
                        "updated_manifest_validation.ok=true",
                    ],
                },
                {
                    "step": 5,
                    "id": "rerun_readiness_packet",
                    "command": ".venv/bin/python scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json",
                    "expected_boundary": "Production remains blocked unless production label counts, side-by-side promotion, registry, and Jetson gates pass.",
                },
            ],
            "source_review_runner": {
                "path": _rel(DEFAULT_SOURCE_REVIEW_RUNNER),
                "sha256": (
                    _sha256_file(DEFAULT_SOURCE_REVIEW_RUNNER)
                    if DEFAULT_SOURCE_REVIEW_RUNNER.exists()
                    else None
                ),
                "plan_command": ".venv/bin/python scripts/apron_harness_source_review_runner.py --json --out qa/video_eval/results/apron_harness_source_review_runner_plan.json",
                "plan_artifact": {
                    "path": _rel(DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT),
                    "exists": DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT.exists(),
                    "sha256": _sha256_file(DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT)
                    if DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT.exists()
                    else None,
                    "evidence_boundary": "durable guarded source-review runner plan only; not approval evidence and not seed-export materialization evidence",
                },
                "execute_command": ".venv/bin/python scripts/apron_harness_source_review_runner.py --execute --json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --capture-manifest /path/to/cleared/apron_harness_capture_manifest.yaml --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml",
                "guardrail": "execute mode validates the review bundle but refuses to import seed exports until a filled non-placeholder seed import manifest passes IMPORT_MANIFEST: gate=pass, cleared capture/emit paths are supplied, and the emitted .seed_export_import.json sidecar validates against the seed-source review after materialization",
            },
            "controlled_capture_path": {
                "status": "blocked_until_capture_matrix_and_label_review_pass",
                "production_capture_matrix_path": _rel(production_capture_matrix_artifact),
                "production_capture_matrix_sha256": (
                    _sha256_file(production_capture_matrix_artifact)
                    if production_capture_matrix_artifact.exists()
                    else None
                ),
                "production_capture_matrix_sidecar_path": _rel(production_capture_matrix_sidecar),
                "production_capture_matrix_sidecar_sha256": (
                    _sha256_file(production_capture_matrix_sidecar)
                    if production_capture_matrix_sidecar.exists()
                    else None
                ),
                "row_count": _summary_int(production_capture_progress, "row_count"),
                "ready_rows": _summary_int(production_capture_progress, "ready_rows"),
                "target_labeled_examples": _summary_int(
                    production_capture_progress,
                    "target_labeled_examples",
                ),
                "captured_examples": _summary_int(production_capture_progress, "captured_examples"),
                "labeled_examples": _summary_int(production_capture_progress, "labeled_examples"),
                "missing_labeled_examples": _summary_int(
                    production_capture_progress,
                    "missing_labeled_examples",
                ),
                "unapproved_rows": _summary_int(production_capture_progress, "unapproved_rows"),
                "unsafe_storage_rows": _summary_int(production_capture_progress, "unsafe_storage_rows"),
                "required_labeled_images_per_class": (
                    production_capture_progress.get("required_labeled_images_per_class") or {}
                ),
                "missing_manifest_counts": (
                    production_capture_progress.get("missing_manifest_counts") or {}
                ),
                "capabilities": (
                    production_capture_summary.get("capture_matrix_progress", {}).get("capabilities", {})
                    if isinstance(production_capture_summary.get("capture_matrix_progress"), dict)
                    else {}
                ),
                "next_capture_batches": (
                    production_capture_matrix_sidecar_doc.get("next_capture_batches") or []
                ),
                "starter_capture_rows": _capture_matrix_starter_rows(production_capture_matrix_artifact),
                "starter_execution_plan": [
                    {
                        "step": 1,
                        "id": "review_starter_rows",
                        "action": "Fill the four production starter label-review rows with approved labels, cleared permission, and non-repo raw storage references.",
                        "required_rows": [
                            "apron_required_closed_set_capture.positive.denim_apron",
                            "apron_required_closed_set_capture.hard_negative.jacket",
                            "harness_required_closed_set_capture.positive.fall_arrest_harness",
                            "harness_required_closed_set_capture.hard_negative.backpack_straps",
                        ],
                        "stop_if": [
                            "permission is not controlled_capture_cleared",
                            "raw_storage_ref is missing or inside the repo",
                            "review_status is not approved",
                        ],
                    },
                    {
                        "step": 2,
                        "id": "validate_starter_label_review_csv",
                        "command": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
                        "pass_signal": "LABEL_REVIEW_VALIDATION: gate=pass",
                        "writes": [],
                    },
                    {
                        "step": 3,
                        "id": "import_starter_label_review_csv",
                        "command": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv --emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml",
                        "writes": [
                            "qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml",
                            "qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json",
                        ],
                    },
                    {
                        "step": 4,
                        "id": "recheck_starter_reviewed_manifest",
                        "command": ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
                        "expected_boundary": "May still fail full production totals, but must preserve matrix SHA and show manifest-count progress.",
                    },
                    {
                        "step": 5,
                        "id": "rerun_readiness_packet",
                        "command": ".venv/bin/python scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json",
                        "expected_boundary": "Production must remain blocked until the full matrix, production label counts, side-by-side promotion, registry, and Jetson gates pass.",
                    },
                ],
                "operator_handoff": {
                    "capture_kickoff": closed_set.get("capture_kickoff") or {},
                    "capture_work_order": closed_set.get("capture_work_order") or {},
                    "required_kickoff_phrases": [
                        "Starter Validation Loop",
                        "LABEL_REVIEW_VALIDATION: gate=pass",
                        "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json",
                        "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
                        "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
                        "not enough for production training or promotion",
                    ],
                    "required_work_order_phrases": [
                        "Required Follow-Up Commands",
                        "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
                        "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
                        "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv",
                        ".seed_export_import.json",
                        "--capture-preflight-mode production --require-capture-preflight",
                    ],
                },
                "label_review_templates": {
                    "full_production": closed_set.get("production_label_review_csv") or {},
                    "starter_production": closed_set.get("production_starter_label_review_csv") or {},
                },
                "required_operator_fields": [
                    "captured_examples",
                    "labeled_examples",
                    "review_status=approved",
                    "permission=controlled_capture_cleared",
                    "raw_storage_ref=non_repo_external_storage",
                    "owner",
                    "due_date",
                ],
                "commands": [
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv",
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv --emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml",
                ],
                "starter_commands": [
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv --emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml",
                    ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
                ],
                "runner": {
                    "path": _rel(DEFAULT_CONTROLLED_CAPTURE_RUNNER),
                    "sha256": (
                        _sha256_file(DEFAULT_CONTROLLED_CAPTURE_RUNNER)
                        if DEFAULT_CONTROLLED_CAPTURE_RUNNER.exists()
                        else None
                    ),
                    "plan_command": ".venv/bin/python scripts/apron_harness_controlled_capture_runner.py --mode starter --json --out qa/video_eval/results/apron_harness_controlled_capture_starter_plan.json",
                    "plan_artifact": {
                        "path": _rel(DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT),
                        "exists": DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT.exists(),
                        "sha256": _sha256_file(DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT)
                        if DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT.exists()
                        else None,
                        "evidence_boundary": "durable guarded starter controlled-capture runner plan only; not label approval evidence and not reviewed manifest import evidence",
                    },
                    "execute_command": ".venv/bin/python scripts/apron_harness_controlled_capture_runner.py --mode starter --execute --json --label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
                    "guardrail": "execute mode refuses to import label reviews until a filled non-placeholder label-review CSV and seed-import manifest are supplied, LABEL_REVIEW_VALIDATION: gate=pass succeeds, and the emitted .label_review_import.json sidecar validates against the filled label-review CSV after import",
                },
                "starter_success_criteria": {
                    "filled_starter_csv": (
                        "starter label-review validation command exits 0 and reports "
                        "LABEL_REVIEW_VALIDATION: gate=pass for approved rows only"
                    ),
                    "import_sidecar": (
                        "import writes qa/video_eval/results/"
                        "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json "
                        "with source manifest SHA, label review CSV SHA, reviewed manifest SHA, "
                        "imported row count, and permission/storage/count gates"
                    ),
                    "starter_reviewed_manifest": (
                        "qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml "
                        "exists and validates in production schema-only mode"
                    ),
                    "post_import_recheck": (
                        "production capture-matrix recheck may still fail total production counts, "
                        "but must preserve the matrix SHA and show progress in manifest counts"
                    ),
                    "promotion_boundary": (
                        "starter reviewed manifest is not sufficient for production training/promotion "
                        "until full production matrix, production label counts, side-by-side, registry, "
                        "and Jetson gates pass"
                    ),
                },
                "post_capture_evidence_checklist": post_capture_evidence_checklist,
            },
            "next_source_reviews": next_source_reviews[:5],
            "commands": [
                ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
                ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
                ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            ],
            "evidence_boundary": (
                "This section selects the first unblock path only; it is not approval. "
                "Public seed data still needs filled human/legal review evidence, checklist review, "
                "reviewed model_packs, reviewed seed-import manifest, and local export ZIP SHA checks."
            ),
        },
    }


def validate_production_gate_packet(
    packet_path: Path,
    *,
    readiness_report_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        packet = _load_json(packet_path)
    except Exception as exc:
        packet = {}
        errors.append(f"production gate packet unreadable: {exc}")
    try:
        readiness = _load_json(readiness_report_path)
    except Exception as exc:
        readiness = {}
        errors.append(f"readiness report unreadable: {exc}")

    scope = packet.get("scope") if isinstance(packet.get("scope"), dict) else {}
    status = packet.get("status") if isinstance(packet.get("status"), dict) else {}
    candidate_gate = (
        packet.get("candidate_runtime_gate")
        if isinstance(packet.get("candidate_runtime_gate"), dict)
        else {}
    )
    first_unblock = (
        packet.get("first_unblock")
        if isinstance(packet.get("first_unblock"), dict)
        else {}
    )
    controlled_capture_path = (
        first_unblock.get("controlled_capture_path")
        if isinstance(first_unblock.get("controlled_capture_path"), dict)
        else {}
    )
    candidate_execution_plan = (
        packet.get("candidate_runtime_execution_plan")
        if isinstance(packet.get("candidate_runtime_execution_plan"), dict)
        else {}
    )
    candidate_training_plan = (
        packet.get("candidate_training_execution_plan")
        if isinstance(packet.get("candidate_training_execution_plan"), dict)
        else {}
    )
    jetson_gate_plan = (
        packet.get("jetson_gate_execution_plan")
        if isinstance(packet.get("jetson_gate_execution_plan"), dict)
        else {}
    )
    readiness_candidate = (
        readiness.get("closed_set_candidate_runtime_evidence")
        if isinstance(readiness.get("closed_set_candidate_runtime_evidence"), dict)
        else {}
    )
    readiness_candidate_templates = (
        readiness.get("closed_set_candidate_yaml_templates")
        if isinstance(readiness.get("closed_set_candidate_yaml_templates"), dict)
        else {}
    )

    if packet.get("kind") != "apron_harness_production_gate_packet":
        errors.append("kind must be apron_harness_production_gate_packet")
    if scope.get("active_vertical") != "factory_ppe":
        errors.append("scope.active_vertical must be factory_ppe")
    if scope.get("skipped_verticals") != ["hospital", "hospitals", "rl_m"]:
        errors.append("scope.skipped_verticals must be ['hospital', 'hospitals', 'rl_m']")
    if scope.get("capabilities") != ["apron_required", "harness_required"]:
        errors.append("scope.capabilities must be apron_required,harness_required")

    status_expectations = {
        "sales_status": readiness.get("sales_status"),
        "pilot_gate_passed": readiness.get("pilot_gate_passed") is True,
        "production_gate_passed": readiness.get("production_gate_passed") is True,
        "production_blocker_count": _summary_int(readiness, "production_blocker_count"),
        "factory_ppe_pack_status": readiness.get("factory_ppe_pack_status"),
        "runtime_handoff_status": readiness.get("runtime_handoff_status"),
        "sourcing_status": readiness.get("sourcing_status"),
    }
    for key, expected in status_expectations.items():
        if status.get(key) != expected:
            errors.append(f"status.{key} mismatch: packet={status.get(key)!r} readiness={expected!r}")

    if packet.get("production_blockers") != readiness.get("production_blockers"):
        errors.append("production_blockers mismatch")
    packet_next_actions = packet.get("next_actions") if isinstance(packet.get("next_actions"), list) else []
    readiness_next_actions = (
        readiness.get("next_actions") if isinstance(readiness.get("next_actions"), list) else []
    )
    if packet_next_actions != readiness_next_actions:
        errors.append("next_actions mismatch")
    if packet.get("seed_import_manifest_validation_summary") != readiness.get(
        "seed_import_manifest_validation_summary"
    ):
        errors.append("seed_import_manifest_validation_summary mismatch")
    if packet.get("production_capture_matrix_validation_summary") != readiness.get(
        "production_capture_matrix_validation_summary"
    ):
        errors.append("production_capture_matrix_validation_summary mismatch")
    if first_unblock.get("minimum_approval_path") != readiness.get(
        "seed_source_minimum_approval_path"
    ):
        errors.append("first_unblock.minimum_approval_path mismatch")
    expected_next_source_reviews = readiness.get("seed_source_next_review_queue")
    if not isinstance(expected_next_source_reviews, list):
        expected_next_source_reviews = []
    if first_unblock.get("next_source_reviews") != expected_next_source_reviews[:5]:
        errors.append("first_unblock.next_source_reviews mismatch")
    minimum_path = readiness.get("seed_source_minimum_approval_path")
    if not isinstance(minimum_path, dict):
        minimum_path = {}
    minimum_source_refs = minimum_path.get("minimum_review_source_refs")
    if not isinstance(minimum_source_refs, list):
        minimum_source_refs = []
    review_by_ref = {
        str(review.get("source_ref")): review
        for review in expected_next_source_reviews
        if isinstance(review, dict) and review.get("source_ref")
    }
    expected_minimum_reviews = [
        review_by_ref[str(source_ref)]
        for source_ref in minimum_source_refs
        if str(source_ref) in review_by_ref
    ]
    if first_unblock.get("minimum_review_sources") != expected_minimum_reviews:
        errors.append("first_unblock.minimum_review_sources mismatch")
    readiness_seed_source_review = (
        readiness.get("seed_source_review")
        if isinstance(readiness.get("seed_source_review"), dict)
        else {}
    )
    expected_source_recheck = (
        readiness_seed_source_review.get("source_recheck")
        if isinstance(readiness_seed_source_review.get("source_recheck"), dict)
        else {}
    )
    source_recheck_artifact = first_unblock.get("source_recheck_artifact")
    if source_recheck_artifact != expected_source_recheck:
        errors.append("first_unblock.source_recheck_artifact mismatch")
    if isinstance(source_recheck_artifact, dict):
        source_recheck_path = source_recheck_artifact.get("path")
        source_recheck_sha = source_recheck_artifact.get("sha256")
        if not source_recheck_path or not source_recheck_sha:
            errors.append("first_unblock.source_recheck_artifact missing path or sha")
        else:
            source_recheck_file = _artifact_path(str(source_recheck_path))
            if not source_recheck_file.exists():
                errors.append(
                    "first_unblock.source_recheck_artifact missing on disk: "
                    f"{source_recheck_path}"
                )
            elif _sha256_file(source_recheck_file) != source_recheck_sha:
                errors.append("first_unblock.source_recheck_artifact sha mismatch")
        if "does not approve" not in str(source_recheck_artifact.get("evidence_boundary") or ""):
            errors.append("first_unblock.source_recheck_artifact missing approval boundary")
    else:
        errors.append("first_unblock.source_recheck_artifact missing")
    readiness_capture_progress = readiness.get("closed_set_capture_progress_summary")
    if not isinstance(readiness_capture_progress, dict):
        readiness_capture_progress = {}
    readiness_production_capture = readiness_capture_progress.get("production")
    if not isinstance(readiness_production_capture, dict):
        readiness_production_capture = {}
    readiness_capture_summary = readiness.get("production_capture_matrix_validation_summary")
    if not isinstance(readiness_capture_summary, dict):
        readiness_capture_summary = {}
    readiness_capture_matrix_progress = readiness_capture_summary.get("capture_matrix_progress")
    if not isinstance(readiness_capture_matrix_progress, dict):
        readiness_capture_matrix_progress = {}
    readiness_closed_set = readiness.get("closed_set_handoff")
    if not isinstance(readiness_closed_set, dict):
        readiness_closed_set = {}
    controlled_capture_expectations = {
        "status": "blocked_until_capture_matrix_and_label_review_pass",
        "row_count": _summary_int(readiness_production_capture, "row_count"),
        "ready_rows": _summary_int(readiness_production_capture, "ready_rows"),
        "target_labeled_examples": _summary_int(
            readiness_production_capture,
            "target_labeled_examples",
        ),
        "captured_examples": _summary_int(readiness_production_capture, "captured_examples"),
        "labeled_examples": _summary_int(readiness_production_capture, "labeled_examples"),
        "missing_labeled_examples": _summary_int(
            readiness_production_capture,
            "missing_labeled_examples",
        ),
        "unapproved_rows": _summary_int(readiness_production_capture, "unapproved_rows"),
        "unsafe_storage_rows": _summary_int(readiness_production_capture, "unsafe_storage_rows"),
        "required_labeled_images_per_class": (
            readiness_production_capture.get("required_labeled_images_per_class") or {}
        ),
        "missing_manifest_counts": readiness_production_capture.get("missing_manifest_counts") or {},
        "capabilities": readiness_capture_matrix_progress.get("capabilities") or {},
    }
    for key, expected in controlled_capture_expectations.items():
        if controlled_capture_path.get(key) != expected:
            errors.append(
                f"first_unblock.controlled_capture_path.{key} mismatch: "
                f"packet={controlled_capture_path.get(key)!r} readiness={expected!r}"
            )
    controlled_capture_matrix_path = controlled_capture_path.get("production_capture_matrix_path")
    if not controlled_capture_matrix_path:
        errors.append("first_unblock.controlled_capture_path missing production capture matrix path")
        expected_starter_capture_rows: list[dict[str, Any]] = []
    else:
        controlled_capture_matrix = _artifact_path(controlled_capture_matrix_path)
        if not controlled_capture_matrix.exists():
            errors.append(
                "first_unblock.controlled_capture_path production capture matrix missing on disk: "
                f"{controlled_capture_matrix_path}"
            )
            expected_starter_capture_rows = []
        elif _sha256_file(controlled_capture_matrix) != controlled_capture_path.get(
            "production_capture_matrix_sha256"
        ):
            errors.append("first_unblock.controlled_capture_path production capture matrix sha mismatch")
            expected_starter_capture_rows = []
        else:
            expected_starter_capture_rows = _capture_matrix_starter_rows(controlled_capture_matrix)
    if controlled_capture_path.get("starter_capture_rows") != expected_starter_capture_rows:
        errors.append("first_unblock.controlled_capture_path.starter_capture_rows mismatch")
    starter_commands = controlled_capture_path.get("starter_commands")
    starter_execution_plan = controlled_capture_path.get("starter_execution_plan")
    if not isinstance(starter_execution_plan, list):
        errors.append("first_unblock.controlled_capture_path.starter_execution_plan missing")
    else:
        expected_step_ids = [
            "review_starter_rows",
            "validate_starter_label_review_csv",
            "import_starter_label_review_csv",
            "recheck_starter_reviewed_manifest",
            "rerun_readiness_packet",
        ]
        if [step.get("id") for step in starter_execution_plan if isinstance(step, dict)] != expected_step_ids:
            errors.append("first_unblock.controlled_capture_path.starter_execution_plan step order mismatch")
        if len(starter_execution_plan) >= 4 and isinstance(starter_execution_plan[0], dict):
            expected_row_ids = [
                row.get("row_id")
                for row in expected_starter_capture_rows
                if isinstance(row, dict)
            ]
            if starter_execution_plan[0].get("required_rows") != expected_row_ids:
                errors.append(
                    "first_unblock.controlled_capture_path.starter_execution_plan "
                    "required_rows mismatch"
                )
        if isinstance(starter_commands, list) and len(starter_commands) >= 3 and len(starter_execution_plan) >= 4:
            for step_index, command_index in ((1, 0), (2, 1), (3, 2)):
                step = starter_execution_plan[step_index]
                if not isinstance(step, dict) or step.get("command") != starter_commands[command_index]:
                    errors.append(
                        "first_unblock.controlled_capture_path.starter_execution_plan "
                        f"command mismatch at step {step_index + 1}"
                    )
        starter_execution_text = "\n".join(
            json.dumps(step, sort_keys=True)
            for step in starter_execution_plan
            if isinstance(step, dict)
        )
        for phrase in [
            "LABEL_REVIEW_VALIDATION: gate=pass",
            "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json",
            "--validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv",
            "Production must remain blocked",
        ]:
            if phrase not in starter_execution_text:
                errors.append(
                    "first_unblock.controlled_capture_path.starter_execution_plan "
                    f"missing phrase: {phrase}"
                )
    expected_operator_handoff = {
        "capture_kickoff": readiness_closed_set.get("capture_kickoff") or {},
        "capture_work_order": readiness_closed_set.get("capture_work_order") or {},
        "required_kickoff_phrases": [
            "Starter Validation Loop",
            "LABEL_REVIEW_VALIDATION: gate=pass",
            "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json",
            "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            "not enough for production training or promotion",
        ],
        "required_work_order_phrases": [
            "Required Follow-Up Commands",
            "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
            "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv",
            ".seed_export_import.json",
            "--capture-preflight-mode production --require-capture-preflight",
        ],
    }
    operator_handoff = controlled_capture_path.get("operator_handoff")
    if operator_handoff != expected_operator_handoff:
        errors.append("first_unblock.controlled_capture_path.operator_handoff mismatch")
    if isinstance(operator_handoff, dict):
        for artifact_key in ("capture_kickoff", "capture_work_order"):
            artifact = operator_handoff.get(artifact_key)
            if not isinstance(artifact, dict):
                errors.append(
                    f"first_unblock.controlled_capture_path.operator_handoff.{artifact_key} missing"
                )
                continue
            artifact_path = artifact.get("path")
            if not artifact_path:
                errors.append(
                    f"first_unblock.controlled_capture_path.operator_handoff.{artifact_key} missing path"
                )
                continue
            artifact_file = _artifact_path(artifact_path)
            if not artifact_file.exists():
                errors.append(
                    "first_unblock.controlled_capture_path.operator_handoff "
                    f"{artifact_key} missing on disk: {artifact_path}"
                )
                continue
            if artifact.get("sha256") and _sha256_file(artifact_file) != artifact.get("sha256"):
                errors.append(
                    "first_unblock.controlled_capture_path.operator_handoff "
                    f"{artifact_key} sha mismatch"
                )
            if artifact_key == "capture_kickoff":
                kickoff_text = artifact_file.read_text(encoding="utf-8")
                for phrase in operator_handoff.get("required_kickoff_phrases") or []:
                    if str(phrase) not in kickoff_text:
                        errors.append(
                            "first_unblock.controlled_capture_path.operator_handoff "
                            f"capture_kickoff missing phrase: {phrase}"
                        )
            if artifact_key == "capture_work_order":
                work_order_text = artifact_file.read_text(encoding="utf-8")
                for phrase in operator_handoff.get("required_work_order_phrases") or []:
                    if str(phrase) not in work_order_text:
                        errors.append(
                            "first_unblock.controlled_capture_path.operator_handoff "
                            f"capture_work_order missing phrase: {phrase}"
                        )
    sidecar_path = controlled_capture_path.get("production_capture_matrix_sidecar_path")
    expected_next_capture_batches: list[Any] = []
    if not sidecar_path:
        errors.append("first_unblock.controlled_capture_path missing production capture matrix sidecar path")
    else:
        sidecar_artifact = _artifact_path(sidecar_path)
        if not sidecar_artifact.exists():
            errors.append(
                "first_unblock.controlled_capture_path production capture matrix sidecar missing on disk: "
                f"{sidecar_path}"
            )
        elif _sha256_file(sidecar_artifact) != controlled_capture_path.get(
            "production_capture_matrix_sidecar_sha256"
        ):
            errors.append("first_unblock.controlled_capture_path production capture matrix sidecar sha mismatch")
        else:
            try:
                sidecar_doc = _load_json(sidecar_artifact)
                if isinstance(sidecar_doc, dict):
                    expected_next_capture_batches = list(sidecar_doc.get("next_capture_batches") or [])
            except Exception as exc:
                errors.append(
                    "first_unblock.controlled_capture_path production capture matrix sidecar unreadable: "
                    f"{exc}"
                )
    if controlled_capture_path.get("next_capture_batches") != expected_next_capture_batches:
        errors.append("first_unblock.controlled_capture_path.next_capture_batches mismatch")
    expected_label_review_templates = {
        "full_production": readiness_closed_set.get("production_label_review_csv") or {},
        "starter_production": readiness_closed_set.get("production_starter_label_review_csv") or {},
    }
    label_review_templates = controlled_capture_path.get("label_review_templates")
    if label_review_templates != expected_label_review_templates:
        errors.append("first_unblock.controlled_capture_path.label_review_templates mismatch")
    if isinstance(label_review_templates, dict):
        for label, template in label_review_templates.items():
            if not isinstance(template, dict):
                errors.append(
                    f"first_unblock.controlled_capture_path.label_review_templates.{label} must be an object"
                )
                continue
            template_path = template.get("path")
            if not template_path:
                errors.append(
                    f"first_unblock.controlled_capture_path.label_review_templates.{label} missing path"
                )
                continue
            template_artifact = _artifact_path(template_path)
            if not template_artifact.exists():
                errors.append(
                    "first_unblock.controlled_capture_path.label_review_templates "
                    f"{label} missing on disk: {template_path}"
                )
            elif _sha256_file(template_artifact) != template.get("sha256"):
                errors.append(
                    "first_unblock.controlled_capture_path.label_review_templates "
                    f"{label} sha mismatch"
                )
    controlled_capture_commands = controlled_capture_path.get("commands")
    if not isinstance(controlled_capture_commands, list) or not any(
        "--import-label-review-csv" in str(command)
        for command in controlled_capture_commands
    ):
        errors.append("first_unblock.controlled_capture_path.commands missing label-review import command")
    if not isinstance(starter_commands, list):
        errors.append("first_unblock.controlled_capture_path.starter_commands missing")
    else:
        if not any("--validate-label-review-csv" in str(command) for command in starter_commands):
            errors.append("first_unblock.controlled_capture_path.starter_commands missing starter validation command")
        if not any("--import-label-review-csv" in str(command) for command in starter_commands):
            errors.append("first_unblock.controlled_capture_path.starter_commands missing starter import command")
        if not any("starter_reviewed.yaml" in str(command) for command in starter_commands):
            errors.append("first_unblock.controlled_capture_path.starter_commands missing starter reviewed manifest command")
    controlled_capture_runner = controlled_capture_path.get("runner")
    controlled_capture_runner_artifact_count = 0
    if not isinstance(controlled_capture_runner, dict):
        errors.append("first_unblock.controlled_capture_path.runner missing")
    else:
        runner_path = controlled_capture_runner.get("path")
        if runner_path != _rel(DEFAULT_CONTROLLED_CAPTURE_RUNNER):
            errors.append("first_unblock.controlled_capture_path.runner path mismatch")
        runner_artifact = _artifact_path(str(runner_path or ""))
        if not runner_path or not runner_artifact.exists():
            errors.append("first_unblock.controlled_capture_path.runner missing on disk")
        elif _sha256_file(runner_artifact) != controlled_capture_runner.get("sha256"):
            errors.append("first_unblock.controlled_capture_path.runner sha mismatch")
        else:
            controlled_capture_runner_artifact_count = 1
        runner_text = json.dumps(controlled_capture_runner, sort_keys=True)
        for phrase in [
            "apron_harness_controlled_capture_runner.py --mode starter --json",
            "qa/video_eval/results/apron_harness_controlled_capture_starter_plan.json",
            "apron_harness_controlled_capture_runner.py --mode starter --execute --json",
            "LABEL_REVIEW_VALIDATION: gate=pass",
            "refuses to import label reviews",
            ".label_review_import.json sidecar validates against the filled label-review CSV after import",
        ]:
            if phrase not in runner_text:
                errors.append(
                    "first_unblock.controlled_capture_path.runner "
                    f"missing phrase: {phrase}"
                )
        plan_artifact = controlled_capture_runner.get("plan_artifact")
        if not isinstance(plan_artifact, dict):
            errors.append("first_unblock.controlled_capture_path.runner plan artifact missing")
        else:
            plan_path = plan_artifact.get("path")
            if plan_path != _rel(DEFAULT_CONTROLLED_CAPTURE_RUNNER_STARTER_PLAN_REPORT):
                errors.append("first_unblock.controlled_capture_path.runner plan artifact path mismatch")
            plan_file = _artifact_path(str(plan_path or ""))
            if plan_artifact.get("exists") is True:
                if not plan_file.exists():
                    errors.append("first_unblock.controlled_capture_path.runner plan artifact marked exists but missing")
                elif _sha256_file(plan_file) != plan_artifact.get("sha256"):
                    errors.append("first_unblock.controlled_capture_path.runner plan artifact sha mismatch")
    starter_success_criteria = controlled_capture_path.get("starter_success_criteria")
    expected_starter_success_keys = {
        "filled_starter_csv",
        "import_sidecar",
        "starter_reviewed_manifest",
        "post_import_recheck",
        "promotion_boundary",
    }
    if not isinstance(starter_success_criteria, dict):
        errors.append("first_unblock.controlled_capture_path.starter_success_criteria missing")
    else:
        missing_starter_success_keys = expected_starter_success_keys - set(
            starter_success_criteria.keys()
        )
        if missing_starter_success_keys:
            errors.append(
                "first_unblock.controlled_capture_path.starter_success_criteria missing keys: "
                + ", ".join(sorted(missing_starter_success_keys))
            )
        starter_success_text = "\n".join(str(value) for value in starter_success_criteria.values())
        for required_phrase in [
            "LABEL_REVIEW_VALIDATION: gate=pass",
            ".label_review_import.json",
            "starter_reviewed.yaml",
            "not sufficient for production training/promotion",
        ]:
            if required_phrase not in starter_success_text:
                errors.append(
                    "first_unblock.controlled_capture_path.starter_success_criteria "
                    f"missing phrase: {required_phrase}"
                )
    post_capture_evidence_checklist = controlled_capture_path.get("post_capture_evidence_checklist")
    if packet.get("post_capture_evidence_checklist") != post_capture_evidence_checklist:
        errors.append("post_capture_evidence_checklist must match controlled_capture_path checklist")
    expected_post_capture_ids = [
        "filled_production_capture_matrix",
        "filled_production_label_review_csv",
        "reviewed_capture_manifest_sidecar",
        "production_training_preflight",
    ]
    if not isinstance(post_capture_evidence_checklist, list):
        errors.append("first_unblock.controlled_capture_path.post_capture_evidence_checklist missing")
    else:
        if [
            item.get("id")
            for item in post_capture_evidence_checklist
            if isinstance(item, dict)
        ] != expected_post_capture_ids:
            errors.append(
                "first_unblock.controlled_capture_path.post_capture_evidence_checklist id order mismatch"
            )
        checklist_text = "\n".join(
            json.dumps(item, sort_keys=True)
            for item in post_capture_evidence_checklist
            if isinstance(item, dict)
        )
        for phrase in [
            "capture_matrix_progress.gate_passed=true",
            "LABEL_REVIEW_VALIDATION: gate=pass",
            "updated_manifest_validation.ok=true",
            "ready_to_train",
            "raw_storage_ref=non_repo_external_storage",
            "--capture-preflight-mode production --require-capture-preflight",
        ]:
            if phrase not in checklist_text:
                errors.append(
                    "first_unblock.controlled_capture_path.post_capture_evidence_checklist "
                    f"missing phrase: {phrase}"
                )
    controlled_capture_fields = controlled_capture_path.get("required_operator_fields")
    if not isinstance(controlled_capture_fields, list) or "raw_storage_ref=non_repo_external_storage" not in (
        controlled_capture_fields or []
    ):
        errors.append("first_unblock.controlled_capture_path.required_operator_fields missing storage guardrail")
    packet_minimum_reviews = first_unblock.get("minimum_review_sources")
    minimum_review_source_refs: list[str] = []
    minimum_review_artifact_count = 0
    minimum_review_artifact_sha_match_count = 0
    if not isinstance(packet_minimum_reviews, list) or not packet_minimum_reviews:
        errors.append("first_unblock.minimum_review_sources missing")
    else:
        for review in packet_minimum_reviews:
            if not isinstance(review, dict):
                errors.append("first_unblock.minimum_review_sources contains non-object")
                continue
            source_ref = str(review.get("source_ref") or "")
            if source_ref:
                minimum_review_source_refs.append(source_ref)
            if not review.get("review_packet_path") or not review.get("review_packet_sha256"):
                errors.append(f"first_unblock.minimum_review_sources missing review packet for {review.get('source_ref')}")
            else:
                minimum_review_artifact_count += 1
                packet_artifact = _artifact_path(review.get("review_packet_path"))
                if not packet_artifact.exists():
                    errors.append(
                        "first_unblock.minimum_review_sources review packet missing on disk "
                        f"for {review.get('source_ref')}: {review.get('review_packet_path')}"
                    )
                elif _sha256_file(packet_artifact) != review.get("review_packet_sha256"):
                    errors.append(
                        "first_unblock.minimum_review_sources review packet sha mismatch "
                        f"for {review.get('source_ref')}"
                    )
                else:
                    minimum_review_artifact_sha_match_count += 1
            if not review.get("review_evidence_template_path") or not review.get("review_evidence_template_sha256"):
                errors.append(
                    "first_unblock.minimum_review_sources missing review evidence template "
                    f"for {review.get('source_ref')}"
                )
            else:
                minimum_review_artifact_count += 1
                evidence_artifact = _artifact_path(review.get("review_evidence_template_path"))
                if not evidence_artifact.exists():
                    errors.append(
                        "first_unblock.minimum_review_sources review evidence template missing on disk "
                        f"for {review.get('source_ref')}: {review.get('review_evidence_template_path')}"
                    )
                elif _sha256_file(evidence_artifact) != review.get("review_evidence_template_sha256"):
                    errors.append(
                        "first_unblock.minimum_review_sources review evidence template sha mismatch "
                        f"for {review.get('source_ref')}"
                    )
                else:
                    minimum_review_artifact_sha_match_count += 1
            if not review.get("review_prefill_path") or not review.get("review_prefill_sha256"):
                errors.append(
                    "first_unblock.minimum_review_sources missing review prefill memo "
                    f"for {review.get('source_ref')}"
                )
            else:
                minimum_review_artifact_count += 1
                prefill_artifact = _artifact_path(review.get("review_prefill_path"))
                if not prefill_artifact.exists():
                    errors.append(
                        "first_unblock.minimum_review_sources review prefill memo missing on disk "
                        f"for {review.get('source_ref')}: {review.get('review_prefill_path')}"
                    )
                elif _sha256_file(prefill_artifact) != review.get("review_prefill_sha256"):
                    errors.append(
                        "first_unblock.minimum_review_sources review prefill memo sha mismatch "
                        f"for {review.get('source_ref')}"
                    )
                else:
                    minimum_review_artifact_sha_match_count += 1
            fill_plan = review.get("seed_import_fill_plan")
            if not isinstance(fill_plan, dict) or "review_status=approved_for_training" not in (
                fill_plan.get("required_fields_before_include_in_training") or []
            ):
                errors.append(
                    "first_unblock.minimum_review_sources missing approved_for_training fill plan "
                    f"for {review.get('source_ref')}"
                )
    source_review_execution_plan = first_unblock.get("source_review_execution_plan")
    expected_source_review_step_ids = [
        "validate_source_review_bundle",
        "fill_minimum_review_evidence",
        "validate_seed_import_manifest",
        "materialize_approved_seed_exports",
        "rerun_readiness_packet",
    ]
    source_review_execution_required_sources: list[str] = []
    if not isinstance(source_review_execution_plan, list):
        errors.append("first_unblock.source_review_execution_plan missing")
    else:
        if [
            step.get("id")
            for step in source_review_execution_plan
            if isinstance(step, dict)
        ] != expected_source_review_step_ids:
            errors.append("first_unblock.source_review_execution_plan step order mismatch")
        if len(source_review_execution_plan) >= 2 and isinstance(
            source_review_execution_plan[1],
            dict,
        ):
            raw_required_sources = source_review_execution_plan[1].get("required_sources")
            if isinstance(raw_required_sources, list):
                source_review_execution_required_sources = [
                    str(source_ref) for source_ref in raw_required_sources
                ]
            if source_review_execution_required_sources != minimum_review_source_refs:
                errors.append(
                    "first_unblock.source_review_execution_plan required_sources mismatch"
                )
        source_review_execution_text = "\n".join(
            json.dumps(step, sort_keys=True)
            for step in source_review_execution_plan
            if isinstance(step, dict)
        )
        for phrase in [
            "REVIEW_BUNDLE: ok=True",
            "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml",
            "--import-approved-seed-exports",
            ".seed_export_import.json",
            "approval_status is not approved_for_training",
            "customer-private or identifiable footage without explicit approval",
            "Production remains blocked",
        ]:
            if phrase not in source_review_execution_text:
                errors.append(
                    "first_unblock.source_review_execution_plan "
                    f"missing phrase: {phrase}"
                )
    source_review_runner = first_unblock.get("source_review_runner")
    source_review_runner_artifact_count = 0
    if not isinstance(source_review_runner, dict):
        errors.append("first_unblock.source_review_runner missing")
    else:
        runner_path = source_review_runner.get("path")
        if runner_path != _rel(DEFAULT_SOURCE_REVIEW_RUNNER):
            errors.append("first_unblock.source_review_runner path mismatch")
        runner_artifact = _artifact_path(str(runner_path or ""))
        if not runner_path or not runner_artifact.exists():
            errors.append("first_unblock.source_review_runner missing on disk")
        elif _sha256_file(runner_artifact) != source_review_runner.get("sha256"):
            errors.append("first_unblock.source_review_runner sha mismatch")
        else:
            source_review_runner_artifact_count = 1
        runner_text = json.dumps(source_review_runner, sort_keys=True)
        for phrase in [
            "apron_harness_source_review_runner.py --json",
            "qa/video_eval/results/apron_harness_source_review_runner_plan.json",
            "apron_harness_source_review_runner.py --execute --json",
            "IMPORT_MANIFEST: gate=pass",
            "refuses to import seed exports",
            ".seed_export_import.json sidecar validates against the seed-source review after materialization",
        ]:
            if phrase not in runner_text:
                errors.append(
                    "first_unblock.source_review_runner "
                    f"missing phrase: {phrase}"
                )
        plan_artifact = source_review_runner.get("plan_artifact")
        if not isinstance(plan_artifact, dict):
            errors.append("first_unblock.source_review_runner plan artifact missing")
        else:
            plan_path = plan_artifact.get("path")
            if plan_path != _rel(DEFAULT_SOURCE_REVIEW_RUNNER_PLAN_REPORT):
                errors.append("first_unblock.source_review_runner plan artifact path mismatch")
            plan_file = _artifact_path(str(plan_path or ""))
            if plan_artifact.get("exists") is True:
                if not plan_file.exists():
                    errors.append("first_unblock.source_review_runner plan artifact marked exists but missing")
                elif _sha256_file(plan_file) != plan_artifact.get("sha256"):
                    errors.append("first_unblock.source_review_runner plan artifact sha mismatch")
    first_unblock_commands = first_unblock.get("commands")
    if not isinstance(first_unblock_commands, list) or not any(
        "--validate-import-manifest" in str(command)
        for command in first_unblock_commands
    ):
        errors.append("first_unblock.commands missing seed-import validation command")
    if first_unblock.get("status") != "blocked_until_human_legal_review_or_cleared_capture":
        errors.append("first_unblock.status must remain blocked until human/legal review or cleared capture")

    candidate_expectations = {
        "runtime_evidence_valid": readiness_candidate.get("valid") is True,
        "present_result_count": _summary_int(readiness_candidate, "present_result_count"),
        "valid_promotion_result_count": _summary_int(
            readiness_candidate,
            "valid_promotion_result_count",
        ),
        "preflight_blocked_missing_model_count": _summary_int(
            readiness_candidate,
            "preflight_blocked_missing_model_count",
        ),
    }
    for key, expected in candidate_expectations.items():
        if candidate_gate.get(key) != expected:
            errors.append(
                f"candidate_runtime_gate.{key} mismatch: "
                f"packet={candidate_gate.get(key)!r} readiness={expected!r}"
            )
    expected_blocked_scenarios = _candidate_runtime_blocked_missing_model_scenario_ids(
        readiness_candidate
    )
    if candidate_gate.get("blocked_missing_model_scenario_ids") != expected_blocked_scenarios:
        errors.append("candidate_runtime_gate.blocked_missing_model_scenario_ids mismatch")
    required_contract = candidate_gate.get("required_contract")
    if not isinstance(required_contract, list) or not any(
        "inactive detector windows require zero detections" in str(item)
        for item in required_contract
    ):
        errors.append("candidate_runtime_gate.required_contract missing detector-window suppression contract")
    expected_candidate_template_rows = [
        row
        for row in readiness_candidate_templates.get("templates", [])
        if isinstance(row, dict)
    ]
    expected_candidate_scenario_order = [
        row.get("scenario_id") for row in expected_candidate_template_rows
    ]
    candidate_execution_steps = candidate_execution_plan.get("steps")
    candidate_execution_step_rows = [
        step
        for step in (candidate_execution_steps or [])
        if isinstance(step, dict)
    ] if isinstance(candidate_execution_steps, list) else []
    if candidate_execution_plan.get("required_model_key") != PLANNED_MODEL_KEY:
        errors.append("candidate_runtime_execution_plan.required_model_key mismatch")
    if candidate_execution_plan.get("one_detection_at_a_time") is not True:
        errors.append("candidate_runtime_execution_plan.one_detection_at_a_time must be true")
    candidate_runtime_runbook = candidate_execution_plan.get("runbook")
    candidate_runtime_runbook_artifact_count = 0
    if not isinstance(candidate_runtime_runbook, dict):
        errors.append("candidate_runtime_execution_plan.runbook missing")
    else:
        runbook_path = candidate_runtime_runbook.get("path")
        if not runbook_path:
            errors.append("candidate_runtime_execution_plan.runbook missing path")
        else:
            runbook_artifact = _artifact_path(str(runbook_path))
            if not runbook_artifact.exists():
                errors.append(
                    f"candidate_runtime_execution_plan.runbook missing on disk: {runbook_path}"
                )
            elif _sha256_file(runbook_artifact) != candidate_runtime_runbook.get("sha256"):
                errors.append("candidate_runtime_execution_plan.runbook sha mismatch")
            else:
                candidate_runtime_runbook_artifact_count = 1
                runbook_text = runbook_artifact.read_text(encoding="utf-8")
                for phrase in [
                    "Apron/Harness Candidate Runtime Runbook",
                    "Run exactly one scenario at a time",
                    "ppe_closed_set_candidate",
                    "factory_missing_apron_active_closed_set",
                    "factory_harness_detector_window_suppression_closed_set",
                    ".venv/bin/python scripts/safetylens_site.py",
                    ".venv/bin/python scripts/video_eval.py run --scenario",
                    "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window",
                ]:
                    if phrase not in runbook_text:
                        errors.append(
                            "candidate_runtime_execution_plan.runbook "
                            f"missing phrase: {phrase}"
                        )
    candidate_runtime_runner = candidate_execution_plan.get("runner")
    candidate_runtime_runner_artifact_count = 0
    if not isinstance(candidate_runtime_runner, dict):
        errors.append("candidate_runtime_execution_plan.runner missing")
    else:
        runner_path = candidate_runtime_runner.get("path")
        if runner_path != _rel(DEFAULT_CANDIDATE_RUNTIME_RUNNER):
            errors.append("candidate_runtime_execution_plan.runner path mismatch")
        runner_artifact = _artifact_path(str(runner_path or ""))
        if not runner_path or not runner_artifact.exists():
            errors.append("candidate_runtime_execution_plan.runner missing on disk")
        elif _sha256_file(runner_artifact) != candidate_runtime_runner.get("sha256"):
            errors.append("candidate_runtime_execution_plan.runner sha mismatch")
        else:
            candidate_runtime_runner_artifact_count = 1
        runner_text = json.dumps(candidate_runtime_runner, sort_keys=True)
        for phrase in [
            "apron_harness_candidate_runtime_runner.py --json",
            "apron_harness_candidate_runtime_runner.py --refresh-blocked-preflight --json",
            "qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json",
            "apron_harness_candidate_runtime_runner.py --execute --json",
            "refuses to run until ppe_closed_set_candidate is registered",
        ]:
            if phrase not in runner_text:
                errors.append(
                    "candidate_runtime_execution_plan.runner "
                    f"missing phrase: {phrase}"
                )
        refresh_artifact = candidate_runtime_runner.get("refresh_blocked_preflight_artifact")
        if not isinstance(refresh_artifact, dict):
            errors.append("candidate_runtime_execution_plan.runner refresh artifact missing")
        else:
            refresh_path = refresh_artifact.get("path")
            if refresh_path != _rel(DEFAULT_CANDIDATE_RUNTIME_PREFLIGHT_REFRESH_REPORT):
                errors.append("candidate_runtime_execution_plan.runner refresh artifact path mismatch")
            refresh_file = _artifact_path(str(refresh_path or ""))
            if refresh_artifact.get("exists") is True:
                if not refresh_file.exists():
                    errors.append("candidate_runtime_execution_plan.runner refresh artifact marked exists but missing")
                elif _sha256_file(refresh_file) != refresh_artifact.get("sha256"):
                    errors.append("candidate_runtime_execution_plan.runner refresh artifact sha mismatch")
    if candidate_execution_plan.get("scenario_order") != expected_candidate_scenario_order:
        errors.append("candidate_runtime_execution_plan.scenario_order mismatch")
    if not isinstance(candidate_execution_steps, list):
        errors.append("candidate_runtime_execution_plan.steps missing")
    elif len(candidate_execution_step_rows) != len(expected_candidate_template_rows):
        errors.append("candidate_runtime_execution_plan.steps count mismatch")
    else:
        runtime_results_by_id = {
            str(result.get("scenario_id")): result
            for result in readiness_candidate.get("results", [])
            if isinstance(result, dict) and result.get("scenario_id")
        }
        for index, (step, expected_row) in enumerate(
            zip(candidate_execution_step_rows, expected_candidate_template_rows),
        ):
            scenario_id = expected_row.get("scenario_id")
            expected_result = runtime_results_by_id.get(str(scenario_id), {})
            for key in ("scenario_id", "capability", "role", "config_path", "expected_result_path"):
                if step.get(key) != expected_row.get(key):
                    errors.append(
                        f"candidate_runtime_execution_plan.steps[{index}].{key} mismatch"
                    )
            if step.get("step") != index + 1:
                errors.append(f"candidate_runtime_execution_plan.steps[{index}].step mismatch")
            if step.get("commands") != (expected_row.get("commands") or {}):
                errors.append(
                    f"candidate_runtime_execution_plan.steps[{index}].commands mismatch"
                )
            if step.get("expected_evidence") != (expected_result.get("required_evidence") or []):
                errors.append(
                    f"candidate_runtime_execution_plan.steps[{index}].expected_evidence mismatch"
                )
    candidate_execution_text = json.dumps(candidate_execution_plan, sort_keys=True)
    for phrase in [
        "factory_missing_apron_active_closed_set",
        "factory_apron_false_positive_guard_closed_set",
        "factory_apron_detector_window_suppression_closed_set",
        "factory_missing_harness_active_closed_set",
        "factory_harness_false_positive_guard_closed_set",
        "factory_harness_detector_window_suppression_closed_set",
        ".venv/bin/python scripts/safetylens_site.py",
        ".venv/bin/python scripts/video_eval.py run --scenario",
        "ppe_closed_set_candidate invocations",
        "suppressed capability telemetry",
        "production promotion remains blocked",
    ]:
        if phrase not in candidate_execution_text:
            errors.append(f"candidate_runtime_execution_plan missing phrase: {phrase}")
    candidate_training_steps = candidate_training_plan.get("steps")
    candidate_training_step_rows = [
        step
        for step in (candidate_training_steps or [])
        if isinstance(step, dict)
    ] if isinstance(candidate_training_steps, list) else []
    expected_candidate_training_step_ids = [
        "training_preflight_plan",
        "train_export_candidate",
        "candidate_doctor_report",
        "side_by_side_promotion_reports",
        "registry_copy_after_promotions",
    ]
    candidate_training_runner_artifact_count = 0
    if candidate_training_plan.get("required_model_key") != PLANNED_MODEL_KEY:
        errors.append("candidate_training_execution_plan.required_model_key mismatch")
    if candidate_training_plan.get("training_model") != DEFAULT_TRAINING_MODEL:
        errors.append("candidate_training_execution_plan.training_model mismatch")
    if candidate_training_plan.get("required_input_status") != _candidate_training_required_input_status(
        readiness_closed_set,
        source_recheck_artifact=source_recheck_artifact
        if isinstance(source_recheck_artifact, dict)
        else None,
    ):
        errors.append("candidate_training_execution_plan.required_input_status mismatch")
    candidate_training_runner = candidate_training_plan.get("runner")
    if not isinstance(candidate_training_runner, dict):
        errors.append("candidate_training_execution_plan.runner missing")
    else:
        runner_path = candidate_training_runner.get("path")
        if runner_path != _rel(DEFAULT_CANDIDATE_TRAINING_RUNNER):
            errors.append("candidate_training_execution_plan.runner path mismatch")
        runner_artifact = _artifact_path(str(runner_path or ""))
        if not runner_path or not runner_artifact.exists():
            errors.append("candidate_training_execution_plan.runner missing on disk")
        elif _sha256_file(runner_artifact) != candidate_training_runner.get("sha256"):
            errors.append("candidate_training_execution_plan.runner sha mismatch")
        else:
            candidate_training_runner_artifact_count = 1
        runner_text = json.dumps(candidate_training_runner, sort_keys=True)
        for phrase in [
            "apron_harness_candidate_training_runner.py --json",
            "apron_harness_candidate_training_runner.py --execute --json",
            "--run-training",
            "actual training also requires --run-training",
        ]:
            if phrase not in runner_text:
                errors.append(
                    "candidate_training_execution_plan.runner "
                    f"missing phrase: {phrase}"
                )
    if not isinstance(candidate_training_steps, list):
        errors.append("candidate_training_execution_plan.steps missing")
    elif [step.get("id") for step in candidate_training_step_rows] != expected_candidate_training_step_ids:
        errors.append("candidate_training_execution_plan.steps id order mismatch")
    candidate_training_text = json.dumps(candidate_training_plan, sort_keys=True)
    for phrase in [
        "--capture-preflight-mode production",
        "--require-capture-preflight",
        "--device mps",
        "--model yolo26n.pt",
        "--export-format onnx",
        "scripts/apron_harness_candidate_doctor.py",
        "scripts/apron_harness_model_registry_doctor.py",
        "qa/video_eval/results/apron_closed_set_promotion_report.json",
        "qa/video_eval/results/harness_closed_set_promotion_report.json",
        "do not copy until candidate doctor and both side-by-side promotion reports pass",
    ]:
        if phrase not in candidate_training_text:
            errors.append(f"candidate_training_execution_plan missing phrase: {phrase}")
    jetson_gate_steps = jetson_gate_plan.get("steps")
    jetson_gate_step_rows = [
        step
        for step in (jetson_gate_steps or [])
        if isinstance(step, dict)
    ] if isinstance(jetson_gate_steps, list) else []
    expected_jetson_gate_step_ids = [
        "stamp_candidate_identity",
        "run_raw_benchmark",
        "run_three_camera_soak",
        "validate_full_gate",
    ]
    jetson_gate_runner_artifact_count = 0
    if jetson_gate_plan.get("required_model_key") != PLANNED_MODEL_KEY:
        errors.append("jetson_gate_execution_plan.required_model_key mismatch")
    if jetson_gate_plan.get("model") != "apron-harness-ppe.onnx":
        errors.append("jetson_gate_execution_plan.model mismatch")
    if jetson_gate_plan.get("templates") != packet.get("jetson_template_handoff"):
        errors.append("jetson_gate_execution_plan.templates mismatch")
    if jetson_gate_plan.get("full_gate") != packet.get("jetson_full_gate"):
        errors.append("jetson_gate_execution_plan.full_gate mismatch")
    jetson_gate_runner = jetson_gate_plan.get("runner")
    if not isinstance(jetson_gate_runner, dict):
        errors.append("jetson_gate_execution_plan.runner missing")
    else:
        runner_path = jetson_gate_runner.get("path")
        if runner_path != _rel(DEFAULT_JETSON_GATE_RUNNER):
            errors.append("jetson_gate_execution_plan.runner path mismatch")
        runner_artifact = _artifact_path(str(runner_path or ""))
        if not runner_path or not runner_artifact.exists():
            errors.append("jetson_gate_execution_plan.runner missing on disk")
        elif _sha256_file(runner_artifact) != jetson_gate_runner.get("sha256"):
            errors.append("jetson_gate_execution_plan.runner sha mismatch")
        else:
            jetson_gate_runner_artifact_count = 1
        runner_text = json.dumps(jetson_gate_runner, sort_keys=True)
        for phrase in [
            "apron_harness_jetson_gate_runner.py --json",
            "apron_harness_jetson_gate_runner.py --execute --json",
            "--require-full-gate",
            "refuses until candidate report, raw benchmark, and soak report",
        ]:
            if phrase not in runner_text:
                errors.append(
                    "jetson_gate_execution_plan.runner "
                    f"missing phrase: {phrase}"
                )
    if not isinstance(jetson_gate_steps, list):
        errors.append("jetson_gate_execution_plan.steps missing")
    elif [step.get("id") for step in jetson_gate_step_rows] != expected_jetson_gate_step_ids:
        errors.append("jetson_gate_execution_plan.steps id order mismatch")
    jetson_gate_text = json.dumps(jetson_gate_plan, sort_keys=True)
    for phrase in [
        "--write-raw-template",
        "--write-soak-template",
        "scripts/benchmark_yolo_jetson.py",
        "--build-soak-report",
        "--active-result apron_required=",
        "--guard-result harness_required=",
        "--suppression-result apron_required=",
        "candidate seed_export_import_manifest preserves source_recheck path/SHA/non-approval boundary",
        "scripts/apron_harness_jetson_gate_runner.py --execute --json",
        "--require-full-gate",
        "zero ppe_closed_set_candidate invocations",
        "production promotion remains blocked",
    ]:
        if phrase not in jetson_gate_text:
            errors.append(f"jetson_gate_execution_plan missing phrase: {phrase}")
    guardrails = packet.get("promotion_guardrails")
    if not isinstance(guardrails, list) or not any(
        "do not register ppe_closed_set_candidate" in str(item)
        for item in guardrails
    ):
        errors.append("promotion_guardrails missing ppe_closed_set_candidate registration guardrail")
    if not isinstance(guardrails, list) or not any(
        "do not mark factory_ppe_3cam ready_to_sell_production_compliance" in str(item)
        for item in guardrails
    ):
        errors.append("promotion_guardrails missing production-compliance guardrail")

    return {
        "ok": not errors,
        "path": _rel(packet_path),
        "readiness_report": _rel(readiness_report_path),
        "kind": packet.get("kind"),
        "sales_status": status.get("sales_status"),
        "production_blocker_count": status.get("production_blocker_count"),
        "next_action_count": len(packet_next_actions),
        "first_unblock_source_count": len(first_unblock.get("next_source_reviews") or []),
        "minimum_review_source_count": len(minimum_review_source_refs),
        "minimum_review_source_refs": minimum_review_source_refs,
        "minimum_review_artifact_count": minimum_review_artifact_count,
        "minimum_review_artifact_sha_match_count": minimum_review_artifact_sha_match_count,
        "source_review_execution_step_count": len(source_review_execution_plan or [])
        if isinstance(source_review_execution_plan, list)
        else 0,
        "source_review_execution_required_source_count": len(
            source_review_execution_required_sources
        ),
        "source_review_runner_artifact_count": source_review_runner_artifact_count,
        "controlled_capture_status": controlled_capture_path.get("status"),
        "controlled_capture_matrix_path": controlled_capture_path.get("production_capture_matrix_path"),
        "controlled_capture_missing_labeled_examples": controlled_capture_path.get(
            "missing_labeled_examples"
        ),
        "controlled_capture_next_batch_count": len(
            controlled_capture_path.get("next_capture_batches") or []
        ),
        "controlled_capture_starter_row_count": len(
            controlled_capture_path.get("starter_capture_rows") or []
        ),
        "controlled_capture_label_template_count": len(
            controlled_capture_path.get("label_review_templates") or {}
        ),
        "controlled_capture_starter_command_count": len(
            controlled_capture_path.get("starter_commands") or []
        ),
        "controlled_capture_runner_artifact_count": controlled_capture_runner_artifact_count,
        "controlled_capture_starter_success_criterion_count": len(
            controlled_capture_path.get("starter_success_criteria") or {}
        ),
        "controlled_capture_operator_handoff_artifact_count": len(
            [
                artifact
                for artifact in (
                    (controlled_capture_path.get("operator_handoff") or {}).get("capture_kickoff"),
                    (controlled_capture_path.get("operator_handoff") or {}).get("capture_work_order"),
                )
                if isinstance(artifact, dict) and artifact.get("path")
            ]
        ),
        "controlled_capture_starter_execution_step_count": len(
            controlled_capture_path.get("starter_execution_plan") or []
        ),
        "controlled_capture_starter_execution_command_match_count": sum(
            1
            for step_index, command_index in ((1, 0), (2, 1), (3, 2))
            if isinstance(controlled_capture_path.get("starter_execution_plan"), list)
            and len(controlled_capture_path.get("starter_execution_plan") or []) > step_index
            and isinstance((controlled_capture_path.get("starter_execution_plan") or [])[step_index], dict)
            and isinstance(controlled_capture_path.get("starter_commands"), list)
            and len(controlled_capture_path.get("starter_commands") or []) > command_index
            and (controlled_capture_path.get("starter_execution_plan") or [])[step_index].get("command")
            == (controlled_capture_path.get("starter_commands") or [])[command_index]
        ),
        "controlled_capture_post_capture_check_count": len(
            controlled_capture_path.get("post_capture_evidence_checklist") or []
        ),
        "candidate_runtime_execution_step_count": len(candidate_execution_step_rows),
        "candidate_runtime_execution_scenario_count": len(
            candidate_execution_plan.get("scenario_order") or []
        )
        if isinstance(candidate_execution_plan.get("scenario_order"), list)
        else 0,
        "candidate_runtime_runbook_artifact_count": candidate_runtime_runbook_artifact_count,
        "candidate_runtime_runner_artifact_count": candidate_runtime_runner_artifact_count,
        "candidate_training_execution_step_count": len(candidate_training_step_rows),
        "candidate_training_runner_artifact_count": candidate_training_runner_artifact_count,
        "jetson_gate_execution_step_count": len(jetson_gate_step_rows),
        "jetson_gate_runner_artifact_count": jetson_gate_runner_artifact_count,
        "errors": errors,
    }


def _summary_model_invocations(summary: dict[str, Any]) -> dict[str, int]:
    raw = summary.get("model_invocations")
    if not isinstance(raw, dict):
        return {}
    invocations: dict[str, int] = {}
    for key, value in raw.items():
        try:
            invocations[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            invocations[str(key)] = 0
    return invocations


def _summary_suppressed_capabilities(summary: dict[str, Any]) -> set[str]:
    raw = summary.get("suppressed_capabilities")
    if not isinstance(raw, list):
        return set()
    return {str(value) for value in raw}


def _promotion_active_summary_error(label: str, summary: Any, model_key: str) -> str | None:
    if not isinstance(summary, dict):
        return f"missing_{label}"
    if not summary.get("scenario_id"):
        return f"{label}_missing_scenario_id"
    if summary.get("status") != READY_STATUS:
        return f"{label}_status:{summary.get('status')}"
    if _summary_int(summary, "max_detections") <= 0:
        return f"{label}_missing_detections"
    if _summary_int(summary, "matching_alerts") <= 0:
        return f"{label}_missing_alerts"
    if _summary_int(summary, "unexpected_alerts") != 0:
        return f"{label}_unexpected_alerts"
    invocations = _summary_model_invocations(summary)
    if model_key not in invocations:
        return f"{label}_missing_model_invocation:{model_key}"
    if invocations.get(model_key, 0) <= 0:
        return f"{label}_model_not_invoked:{model_key}"
    if summary.get("screenshot_ok") is not True:
        return f"{label}_missing_screenshot"
    return None


def _promotion_guard_summary_error(label: str, summary: Any, model_key: str) -> str | None:
    if not isinstance(summary, dict):
        return f"missing_{label}"
    if not summary.get("scenario_id"):
        return f"{label}_missing_scenario_id"
    if summary.get("status") != READY_STATUS:
        return f"{label}_status:{summary.get('status')}"
    if _summary_int(summary, "max_detections") <= 0:
        return f"{label}_missing_detections"
    if _summary_int(summary, "matching_alerts") != 0:
        return f"{label}_unexpected_matching_alerts"
    if _summary_int(summary, "unexpected_alerts") != 0:
        return f"{label}_unexpected_alerts"
    if _summary_int(summary, "visible_class_total") <= 0:
        return f"{label}_missing_visible_class"
    invocations = _summary_model_invocations(summary)
    if model_key not in invocations:
        return f"{label}_missing_model_invocation:{model_key}"
    if invocations.get(model_key, 0) <= 0:
        return f"{label}_model_not_invoked:{model_key}"
    if summary.get("screenshot_ok") is not True:
        return f"{label}_missing_screenshot"
    return None


def _promotion_suppression_summary_error(
    label: str,
    summary: Any,
    capability: str,
    model_key: str,
) -> str | None:
    if not isinstance(summary, dict):
        return f"missing_{label}"
    if not summary.get("scenario_id"):
        return f"{label}_missing_scenario_id"
    if summary.get("status") != READY_STATUS:
        return f"{label}_status:{summary.get('status')}"
    if _summary_int(summary, "max_detections") != 0:
        return f"{label}_emitted_detections"
    if _summary_int(summary, "matching_alerts") != 0 or _summary_int(summary, "unexpected_alerts") != 0:
        return f"{label}_emitted_alerts"
    if capability not in _summary_suppressed_capabilities(summary):
        return f"{label}_missing_suppressed_capability"
    invocations = _summary_model_invocations(summary)
    if model_key not in invocations:
        return f"{label}_missing_model_invocation:{model_key}"
    if invocations.get(model_key, 0) != 0:
        return f"{label}_model_invoked:{model_key}"
    if summary.get("screenshot_ok") is not True:
        return f"{label}_missing_screenshot"
    return None


def _promotion_runtime_group_error(
    group_name: str,
    group: dict[str, Any],
    capability: str,
    model_key: str,
) -> str | None:
    active_error = _promotion_active_summary_error(f"{group_name}_active", group.get("active"), model_key)
    if active_error:
        return active_error
    guard_error = _promotion_guard_summary_error(
        f"{group_name}_false_positive_guard",
        group.get("false_positive_guard"),
        model_key,
    )
    if guard_error:
        return guard_error
    return _promotion_suppression_summary_error(
        f"{group_name}_suppression",
        group.get("suppression"),
        capability,
        model_key,
    )


def _promotion_dataset_provenance_error(
    provenance: Any,
    sidecar: dict[str, Any],
) -> str | None:
    if not isinstance(provenance, dict):
        return "missing_candidate_training_dataset_provenance"
    if provenance.get("required") is not True:
        return "candidate_training_dataset_provenance_not_required"
    if provenance.get("checked") is not True:
        return "candidate_training_dataset_provenance_not_checked"
    if provenance.get("errors") not in ([], None):
        return "candidate_training_dataset_provenance_has_errors"
    if not provenance.get("source_manifest"):
        return "candidate_training_dataset_provenance_missing_source_manifest"
    if not provenance.get("declared_source_manifest_sha256"):
        return "candidate_training_dataset_provenance_missing_declared_source_manifest_sha256"
    if not provenance.get("source_manifest_sha256"):
        return "candidate_training_dataset_provenance_missing_source_manifest_sha256"
    if (
        provenance.get("declared_source_manifest_sha256")
        and provenance.get("source_manifest_sha256")
        and provenance.get("declared_source_manifest_sha256") != provenance.get("source_manifest_sha256")
    ):
        return "candidate_training_dataset_provenance_declared_source_manifest_sha_mismatch"
    if provenance.get("permission_allowed") is not True:
        return "candidate_training_dataset_provenance_permission_not_allowed"
    if provenance.get("missing_ppe_label_policy") != EXPECTED_MISSING_PPE_LABEL_POLICY:
        return "candidate_training_dataset_provenance_wrong_missing_ppe_policy"
    expected_sha = sidecar.get("source_manifest_sha256")
    if expected_sha and provenance.get("source_manifest_sha256") != expected_sha:
        return "candidate_training_dataset_provenance_source_manifest_sha_mismatch"
    return None


def _promotion_lineage_file_error(
    lineage: dict[str, Any],
    section_name: str,
    *,
    required: bool,
) -> str | None:
    section = lineage.get(section_name)
    if not isinstance(section, dict):
        return f"missing_candidate_training_source_lineage_{section_name}"
    file_info = section.get("file")
    if not isinstance(file_info, dict):
        return f"missing_candidate_training_source_lineage_{section_name}_file" if required else None
    if required:
        if file_info.get("required") is not True:
            return f"candidate_training_source_lineage_{section_name}_file_not_required"
        if not file_info.get("path"):
            return f"candidate_training_source_lineage_{section_name}_file_missing_path"
        if file_info.get("exists") is not True:
            return f"candidate_training_source_lineage_{section_name}_file_missing"
        if not file_info.get("sha256"):
            return f"candidate_training_source_lineage_{section_name}_file_missing_sha256"
    return None


def _positive_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _promotion_seed_import_export_preflight_error(gate: dict[str, Any]) -> str | None:
    imports = gate.get("imports")
    if not isinstance(imports, list) or not imports:
        return "candidate_training_source_lineage_seed_import_manifest_missing_imports"

    included: list[dict[str, Any]] = []
    for item in imports:
        if isinstance(item, dict) and item.get("include_in_training") is True:
            included.append(item)
    if not included:
        return "candidate_training_source_lineage_seed_import_manifest_missing_included_imports"

    for item in included:
        if item.get("approved_for_training") is not True:
            return "candidate_training_source_lineage_seed_import_manifest_unapproved_import"
        if item.get("errors") not in ([], None) or item.get("blockers") not in ([], None):
            return "candidate_training_source_lineage_seed_import_manifest_import_has_errors"
        raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
        if not raw_export_sha256 or not str(item.get("raw_export_local_path") or "").strip():
            return "candidate_training_source_lineage_seed_import_manifest_missing_reviewed_export"

        preflight = item.get("yolo_export_preflight")
        if not isinstance(preflight, dict) or preflight.get("checked") is not True:
            return "candidate_training_source_lineage_seed_import_manifest_missing_reviewed_export_preflight"
        preflight_sha256 = str(preflight.get("sha256") or "").strip()
        if not preflight_sha256:
            return "candidate_training_source_lineage_seed_import_manifest_export_preflight_missing_sha"
        if preflight_sha256 != raw_export_sha256:
            return "candidate_training_source_lineage_seed_import_manifest_export_preflight_sha_mismatch"

        required_classes = REQUIRED_SEED_IMPORT_CLASSES.get(str(item.get("capability") or ""), set())
        if not required_classes:
            return "candidate_training_source_lineage_seed_import_manifest_unknown_capability"
        counts = preflight.get("label_file_count_by_local_class")
        if not isinstance(counts, dict):
            return "candidate_training_source_lineage_seed_import_manifest_export_preflight_missing_label_counts"
        for class_name in sorted(required_classes):
            if _positive_int(counts.get(class_name)) <= 0:
                return "candidate_training_source_lineage_seed_import_manifest_export_preflight_missing_label_counts"
    return None


def _promotion_label_review_clip_metadata_error(manifest: dict[str, Any]) -> str | None:
    if "imported_clip_count" not in manifest:
        return "candidate_label_review_import_manifest_imported_clip_count_missing"
    if _positive_int(manifest.get("imported_clip_count")) < 0:
        return "candidate_label_review_import_manifest_imported_clip_count_negative"
    if "invalid_clip_metadata_count" not in manifest:
        return "candidate_label_review_import_manifest_invalid_clip_metadata_count_missing"
    if _positive_int(manifest.get("invalid_clip_metadata_count")) != 0:
        return "candidate_label_review_import_manifest_invalid_clip_metadata_count_nonzero"
    training_gate = manifest.get("training_gate") if isinstance(manifest.get("training_gate"), dict) else {}
    if training_gate.get("requires_reviewed_clip_metadata") is not True:
        return "candidate_label_review_import_manifest_requires_reviewed_clip_metadata_missing"
    return None


def _promotion_seed_export_preflight_error(item: dict[str, Any]) -> str | None:
    preflight = item.get("yolo_export_preflight")
    if not isinstance(preflight, dict):
        return "candidate_seed_export_import_manifest_import_missing_yolo_preflight"
    if preflight.get("checked") is not True:
        return "candidate_seed_export_import_manifest_import_yolo_preflight_not_checked"
    if preflight.get("errors") not in ([], None):
        return "candidate_seed_export_import_manifest_import_yolo_preflight_has_errors"
    raw_export_sha256 = str(item.get("raw_export_sha256") or "").strip()
    preflight_sha256 = str(preflight.get("sha256") or "").strip()
    if not preflight_sha256:
        return "candidate_seed_export_import_manifest_import_yolo_preflight_missing_sha"
    if raw_export_sha256 and preflight_sha256 != raw_export_sha256:
        return "candidate_seed_export_import_manifest_import_yolo_preflight_sha_mismatch"
    orphan_counts = (
        preflight.get("orphan_label_count_by_split")
        if isinstance(preflight.get("orphan_label_count_by_split"), dict)
        else {}
    )
    for count in orphan_counts.values():
        if _positive_int(count) > 0:
            return "candidate_seed_export_import_manifest_import_yolo_preflight_orphan_labels"
    required_classes = REQUIRED_SEED_IMPORT_CLASSES.get(str(item.get("capability") or ""), set())
    if not required_classes:
        return "candidate_seed_export_import_manifest_import_unknown_capability"
    counts = preflight.get("label_file_count_by_local_class")
    if not isinstance(counts, dict):
        return "candidate_seed_export_import_manifest_import_yolo_preflight_missing_label_counts"
    for class_name in sorted(required_classes):
        if _positive_int(counts.get(class_name)) <= 0:
            return "candidate_seed_export_import_manifest_import_yolo_preflight_missing_label_counts"
    return None


def _promotion_seed_export_import_manifest_error(manifest: Any, sidecar: dict[str, Any]) -> str | None:
    if not isinstance(manifest, dict):
        return "missing_candidate_seed_export_import_manifest"
    if manifest.get("valid") is not True:
        return "invalid_candidate_seed_export_import_manifest"
    if not manifest.get("seed_source_review_sha256") or not manifest.get("seed_import_manifest_sha256"):
        return "candidate_seed_export_import_manifest_missing_hashes"
    source_recheck = (
        manifest.get("source_recheck")
        if isinstance(manifest.get("source_recheck"), dict)
        else {}
    )
    if not source_recheck:
        return "candidate_seed_export_import_manifest_missing_source_recheck"
    if source_recheck.get("exists") is not True:
        return "candidate_seed_export_import_manifest_source_recheck_missing_artifact"
    if not _valid_sha(source_recheck.get("sha256")):
        return "candidate_seed_export_import_manifest_source_recheck_missing_hash"
    if not source_recheck.get("path"):
        return "candidate_seed_export_import_manifest_source_recheck_missing_path"
    if "does not approve" not in str(source_recheck.get("evidence_boundary") or ""):
        return "candidate_seed_export_import_manifest_source_recheck_missing_approval_boundary"
    expected_sha = sidecar.get("source_manifest_sha256")
    if expected_sha and manifest.get("updated_manifest_sha256") != expected_sha:
        return "candidate_seed_export_import_manifest_source_manifest_sha_mismatch"
    validation = (
        manifest.get("updated_manifest_validation")
        if isinstance(manifest.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        return "candidate_seed_export_import_manifest_validation_missing"
    if validation.get("checked") is not True or validation.get("ok") is not True:
        return "candidate_seed_export_import_manifest_validation_failed"
    if validation.get("schema_only") is True:
        return "candidate_seed_export_import_manifest_validation_schema_only"
    if validation.get("mode") != "production":
        return "candidate_seed_export_import_manifest_validation_not_production"
    if expected_sha and validation.get("manifest_sha256") != expected_sha:
        return "candidate_seed_export_import_manifest_validation_sha_mismatch"
    for field in ["imported_label_count", "imported_clip_count", "copied_image_count"]:
        if _positive_int(manifest.get(field)) <= 0:
            return f"candidate_seed_export_import_manifest_{field}_not_positive"
    imports = manifest.get("imports")
    if not isinstance(imports, list) or not imports:
        return "candidate_seed_export_import_manifest_missing_imports"
    for item in imports:
        if not isinstance(item, dict):
            return "candidate_seed_export_import_manifest_invalid_import"
        if _positive_int(item.get("imported_label_count")) <= 0:
            return "candidate_seed_export_import_manifest_imported_label_count_not_positive"
        if _positive_int(item.get("copied_image_count")) <= 0:
            return "candidate_seed_export_import_manifest_copied_image_count_not_positive"
        if item.get("errors") not in ([], None):
            return "candidate_seed_export_import_manifest_import_has_errors"
        if not item.get("raw_export_sha256"):
            return "candidate_seed_export_import_manifest_import_missing_export_sha"
        preflight_error = _promotion_seed_export_preflight_error(item)
        if preflight_error:
            return preflight_error
    return None


def _promotion_source_lineage_error(
    lineage: Any,
    sidecar: dict[str, Any],
    provenance: dict[str, Any],
) -> str | None:
    if not isinstance(lineage, dict):
        return "missing_candidate_training_source_lineage"
    dataset_error = _promotion_lineage_file_error(lineage, "dataset_yaml", required=True)
    if dataset_error:
        return dataset_error
    capture_error = _promotion_lineage_file_error(
        lineage,
        "capture_manifest",
        required=provenance.get("required") is True,
    )
    if capture_error:
        return capture_error
    expected_sha = provenance.get("source_manifest_sha256") or sidecar.get("source_manifest_sha256")
    capture = lineage.get("capture_manifest") if isinstance(lineage.get("capture_manifest"), dict) else {}
    capture_file = capture.get("file") if isinstance(capture.get("file"), dict) else {}
    if expected_sha and capture_file.get("sha256") and capture_file.get("sha256") != expected_sha:
        return "candidate_training_source_lineage_capture_manifest_file_sha_mismatch"
    if expected_sha and capture.get("manifest_sha256") and capture.get("manifest_sha256") != expected_sha:
        return "candidate_training_source_lineage_capture_manifest_sha_mismatch"
    if provenance.get("required") is True and capture.get("ok") is not True:
        return "candidate_training_source_lineage_capture_manifest_not_ok"
    if provenance.get("required") is True and capture.get("mode") != "production":
        return "candidate_training_source_lineage_capture_manifest_not_production"

    seed_source = lineage.get("seed_source_review") if isinstance(lineage.get("seed_source_review"), dict) else {}
    seed_source_gate = seed_source.get("gate") if isinstance(seed_source.get("gate"), dict) else {}
    if seed_source_gate.get("required") is True:
        seed_source_error = _promotion_lineage_file_error(lineage, "seed_source_review", required=True)
        if seed_source_error:
            return seed_source_error
        if seed_source_gate.get("ok") is not True:
            return "candidate_training_source_lineage_seed_source_review_not_ok"
        if seed_source_gate.get("gate_passed") is not True:
            return "candidate_training_source_lineage_seed_source_review_not_passed"
        if int(seed_source_gate.get("clip_count") or 0) <= 0:
            return "candidate_training_source_lineage_seed_source_review_missing_clips"
        if int(seed_source_gate.get("approved_clip_count") or 0) != int(seed_source_gate.get("clip_count") or 0):
            return "candidate_training_source_lineage_seed_source_review_unapproved_clips"

    seed_import = lineage.get("seed_import_manifest") if isinstance(lineage.get("seed_import_manifest"), dict) else {}
    seed_import_gate = seed_import.get("gate") if isinstance(seed_import.get("gate"), dict) else {}
    if seed_import_gate.get("required") is True:
        seed_import_error = _promotion_lineage_file_error(lineage, "seed_import_manifest", required=True)
        if seed_import_error:
            return seed_import_error
        if seed_import_gate.get("ok") is not True:
            return "candidate_training_source_lineage_seed_import_manifest_not_ok"
        if seed_import_gate.get("source_review_sha256_matches") is not True:
            return "candidate_training_source_lineage_seed_import_manifest_source_review_sha_mismatch"
        if int(seed_import_gate.get("clip_count") or 0) <= 0:
            return "candidate_training_source_lineage_seed_import_manifest_missing_clips"
        if int(seed_import_gate.get("approved_clip_count") or 0) != int(seed_import_gate.get("clip_count") or 0):
            return "candidate_training_source_lineage_seed_import_manifest_unapproved_clips"
        seed_import_preflight_error = _promotion_seed_import_export_preflight_error(seed_import_gate)
        if seed_import_preflight_error:
            return seed_import_preflight_error
    return None


def _promotion_artifact_identity_error(payload: dict[str, Any]) -> str | None:
    selected_export = payload.get("candidate_selected_export")
    if not isinstance(selected_export, dict):
        return "missing_candidate_selected_export"
    selected_sha = selected_export.get("sha256")
    if not selected_sha:
        return "candidate_selected_export_missing_sha256"
    suffix = str(selected_export.get("suffix") or "").lower()
    if selected_export.get("accepted_suffix") is not True and suffix not in {".onnx", ".engine"}:
        return "candidate_selected_export_unaccepted_suffix"
    registry_entry = payload.get("candidate_registry_entry")
    if not isinstance(registry_entry, dict):
        return "missing_candidate_registry_entry"
    if registry_entry.get("model_key") != PLANNED_MODEL_KEY:
        return f"candidate_registry_entry_wrong_model:{registry_entry.get('model_key')}"
    if registry_entry.get("source_export_sha256") != selected_sha:
        return "candidate_registry_entry_source_export_sha_mismatch"
    return None


def _promotion_selected_export_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    selected_export = payload.get("candidate_selected_export")
    if not isinstance(selected_export, dict):
        return None
    sha = selected_export.get("sha256")
    return str(sha) if sha else None


def _promotion_candidate_report_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    sha = payload.get("candidate_report_sha256")
    return str(sha) if sha else None


def _model_registry_candidate_report_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    sha = payload.get("candidate_report_sha256")
    return str(sha) if sha else None


def _jetson_gate_artifact_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    if payload.get("model_artifact_sha256"):
        return str(payload["model_artifact_sha256"])
    for key in ("raw_benchmark", "soak_report"):
        summary = payload.get(key)
        if isinstance(summary, dict) and summary.get("model_artifact_sha256"):
            return str(summary["model_artifact_sha256"])
    return None


def _jetson_gate_candidate_report_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    if payload.get("candidate_report_sha256"):
        return str(payload["candidate_report_sha256"])
    for key in ("raw_benchmark", "soak_report"):
        summary = payload.get(key)
        if isinstance(summary, dict) and summary.get("candidate_report_sha256"):
            return str(summary["candidate_report_sha256"])
    return None


def _jetson_gate_report_summary(path: Path | None, status: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": _rel(path) if path else None,
        "exists": False,
        "status": status,
        "ok": False,
        "gate_status": None,
        "production_gate": False,
        "model": None,
        "model_artifact_sha256": None,
        "candidate_report_sha256": None,
        "candidate_report": {"present": False},
        "raw_benchmark": {"present": False},
        "soak_report": {"present": False},
        "errors": [],
        "warnings": [],
        "next_required_gates": [],
        "inputs": {},
    }
    if path is None:
        summary["errors"] = ["jetson gate report path is not configured"]
        return summary
    if not path.exists():
        summary["errors"] = [f"jetson gate report missing: {_rel(path)}"]
        return summary
    summary["exists"] = True
    try:
        payload = _load_json(path)
    except Exception as exc:
        summary["errors"] = [f"jetson gate report unreadable: {exc}"]
        return summary

    summary.update(
        {
            "ok": payload.get("ok") is True,
            "gate_status": payload.get("gate_status"),
            "production_gate": payload.get("production_gate") is True,
            "model": payload.get("model"),
            "model_artifact_sha256": payload.get("model_artifact_sha256"),
            "candidate_report_sha256": payload.get("candidate_report_sha256"),
            "errors": list(payload.get("errors") or []),
            "warnings": list(payload.get("warnings") or []),
            "next_required_gates": list(payload.get("next_required_gates") or []),
            "inputs": payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {},
        }
    )
    for key in ("candidate_report", "raw_benchmark", "soak_report"):
        value = payload.get(key)
        summary[key] = value if isinstance(value, dict) else {"present": False}
    return summary


def _jetson_template_handoff_summary(
    *,
    raw_template_path: Path = DEFAULT_FACTORY_PPE_RAW_TEMPLATE,
    soak_template_path: Path = DEFAULT_FACTORY_PPE_SOAK_TEMPLATE,
    expected_model: str = "apron-harness-ppe.onnx",
) -> dict[str, Any]:
    template_specs = {
        "raw_benchmark_template": (raw_template_path, "jetson_raw_benchmark"),
        "three_camera_soak_template": (soak_template_path, "jetson_three_camera_soak"),
    }
    templates: dict[str, Any] = {}
    blockers: list[str] = []
    for name, (path, expected_kind) in template_specs.items():
        item: dict[str, Any] = {
            "path": _rel(path),
            "exists": path.exists(),
            "template": False,
            "evidence_kind": None,
            "model": None,
            "candidate_report_sha256": None,
            "model_artifact_sha256": None,
            "valid_template_contract": False,
            "identity_stamped": False,
            "errors": [],
        }
        if not path.exists():
            item["errors"].append(f"template missing: {_rel(path)}")
        else:
            try:
                payload = _load_json(path)
                item.update(
                    {
                        "template": payload.get("template") is True,
                        "evidence_kind": payload.get("evidence_kind"),
                        "model": payload.get("model"),
                        "candidate_report_sha256": payload.get("candidate_report_sha256"),
                        "model_artifact_sha256": payload.get("model_artifact_sha256"),
                    }
                )
                if payload.get("template") is not True:
                    item["errors"].append("template=true is required")
                if payload.get("evidence_kind") != expected_kind:
                    item["errors"].append(f"evidence_kind must be {expected_kind}")
                if payload.get("model") != expected_model:
                    item["errors"].append(f"model must be {expected_model}")
            except Exception as exc:
                item["errors"].append(f"template unreadable: {exc}")
        item["identity_stamped"] = bool(item.get("candidate_report_sha256") and item.get("model_artifact_sha256"))
        item["valid_template_contract"] = not item["errors"]
        if not item["valid_template_contract"]:
            blockers.append(f"{name}:invalid_template_contract")
        templates[name] = item
    valid_contract_count = sum(1 for item in templates.values() if item.get("valid_template_contract"))
    identity_stamped_count = sum(1 for item in templates.values() if item.get("identity_stamped"))
    return {
        "checked": True,
        "valid": not blockers,
        "template_count": len(templates),
        "valid_template_contract_count": valid_contract_count,
        "identity_stamped_count": identity_stamped_count,
        "templates": templates,
        "status": "ready_for_candidate_identity" if not blockers else "invalid_template_contract",
        "evidence_boundary": (
            "Templates are fillable contracts only; they are not Jetson benchmark evidence "
            "until candidate_report_sha256 and model_artifact_sha256 are stamped from a reviewed candidate."
        ),
        "blockers": blockers,
    }


def _model_registry_handoff_summary(path: Path | None, status: str) -> dict[str, Any]:
    def _artifact_status(name: str, artifact_path: Any) -> dict[str, Any]:
        if not artifact_path:
            return {
                "name": name,
                "path": "",
                "exists": False,
                "sha256": None,
                "ok": False,
                "blockers": ["missing_path"],
            }
        candidate = _artifact_path(artifact_path)
        exists = candidate.exists() and candidate.is_file()
        return {
            "name": name,
            "path": str(artifact_path),
            "exists": exists,
            "sha256": _sha256_file(candidate) if exists else None,
            "ok": exists,
            "blockers": [] if exists else ["missing"],
        }

    summary: dict[str, Any] = {
        "checked": True,
        "path": _rel(path) if path else None,
        "exists": False,
        "status": status,
        "ok": False,
        "registry_status": None,
        "candidate_report_present": False,
        "candidate_report_sha256": None,
        "model_definition_valid": False,
        "model_definition_registered": False,
        "destination_path": None,
        "destination_exists": False,
        "destination_matches_expected_sha256": False,
        "metadata_exists": False,
        "metadata_valid": False,
        "artifact_status": [],
        "copy_requested": False,
        "copied": False,
        "next_required_gates": [],
        "errors": [],
        "warnings": [],
        "evidence_boundary": (
            "A planned registry audit is not model registration; copy only after a candidate report "
            "and selected-export SHA pass promotion review."
        ),
    }
    if path is None:
        summary["errors"] = ["model registry report path is not configured"]
        return summary
    if not path.exists():
        summary["errors"] = [f"model registry report missing: {_rel(path)}"]
        return summary
    summary["exists"] = True
    summary["artifact_status"] = [_artifact_status("model_registry_report", _rel(path))]
    try:
        payload = _load_json(path)
    except Exception as exc:
        summary["errors"] = [f"model registry report unreadable: {exc}"]
        return summary

    model_definition = payload.get("model_manager_definition")
    if not isinstance(model_definition, dict):
        model_definition = {}
    destination = payload.get("destination")
    if not isinstance(destination, dict):
        destination = {}
    metadata = destination.get("registry_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    candidate_report = payload.get("candidate_report")
    summary.update(
        {
            "ok": payload.get("ok") is True,
            "registry_status": payload.get("registry_status"),
            "candidate_report_present": bool(candidate_report),
            "candidate_report_sha256": payload.get("candidate_report_sha256"),
            "model_definition_valid": model_definition.get("valid") is True,
            "model_definition_registered": model_definition.get("registered") is True,
            "destination_path": destination.get("registry_path") or destination.get("path"),
            "destination_exists": destination.get("exists") is True,
            "destination_matches_expected_sha256": destination.get("matches_expected_sha256") is True,
            "metadata_exists": metadata.get("exists") is True,
            "metadata_valid": metadata.get("valid") is True,
            "copy_requested": payload.get("copy_requested") is True,
            "copied": payload.get("copied") is True,
            "next_required_gates": list(payload.get("next_required_gates") or []),
            "errors": list(payload.get("errors") or []),
            "warnings": list(payload.get("warnings") or []),
        }
    )
    metadata_path = metadata.get("path") or (
        f"{summary['destination_path']}.registry.json"
        if summary.get("destination_path")
        else None
    )
    summary["artifact_status"] = [
        _artifact_status("model_registry_report", _rel(path)),
        _artifact_status("destination_model", summary.get("destination_path")),
        _artifact_status("registry_metadata", metadata_path),
    ]
    return summary


def _model_registry_artifact_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    destination = payload.get("destination")
    if isinstance(destination, dict) and destination.get("sha256"):
        return str(destination["sha256"])
    source_export = payload.get("source_export")
    if isinstance(source_export, dict):
        if source_export.get("expected_sha256"):
            return str(source_export["expected_sha256"])
        if source_export.get("sha256"):
            return str(source_export["sha256"])
    return None


def _training_dataset_provenance_status(
    provenance: Any,
    *,
    expected_source_manifest_sha256: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    status: dict[str, Any] = {
        "checked": False,
        "valid": False,
        "expected_source_manifest_sha256": expected_source_manifest_sha256,
        "blockers": blockers,
    }
    if not isinstance(provenance, dict):
        blockers.append("closed_set_training_dataset_provenance_missing")
        return status

    status["checked"] = provenance.get("checked") is True
    if provenance.get("required") is not True:
        blockers.append("closed_set_training_dataset_provenance_not_required")
    if provenance.get("checked") is not True:
        blockers.append("closed_set_training_dataset_provenance_not_checked")
    if provenance.get("errors") not in ([], None):
        blockers.append("closed_set_training_dataset_provenance_has_errors")
    if not provenance.get("source_manifest"):
        blockers.append("closed_set_training_dataset_provenance_missing_source_manifest")
    if not provenance.get("declared_source_manifest_sha256"):
        blockers.append("closed_set_training_dataset_provenance_missing_declared_source_manifest_sha256")
    if not provenance.get("source_manifest_sha256"):
        blockers.append("closed_set_training_dataset_provenance_missing_source_manifest_sha256")
    if (
        provenance.get("declared_source_manifest_sha256")
        and provenance.get("source_manifest_sha256")
        and provenance.get("declared_source_manifest_sha256") != provenance.get("source_manifest_sha256")
    ):
        blockers.append("closed_set_training_dataset_provenance_declared_source_manifest_sha_mismatch")
    if provenance.get("permission_allowed") is not True:
        blockers.append("closed_set_training_dataset_provenance_permission_not_allowed")
    if provenance.get("missing_ppe_label_policy") != EXPECTED_MISSING_PPE_LABEL_POLICY:
        blockers.append("closed_set_training_dataset_provenance_wrong_missing_ppe_policy")
    if (
        expected_source_manifest_sha256
        and provenance.get("source_manifest_sha256")
        and provenance.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        blockers.append("closed_set_training_dataset_provenance_source_manifest_sha_mismatch")
    status["valid"] = not blockers
    return status


def _training_readiness_status(
    *,
    training_plan: dict[str, Any],
    production_training_plan_preflight: dict[str, Any],
    training_capture_preflight: dict[str, Any],
    production_missing_counts: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    dry_run_status = str(training_plan.get("status") or "unknown")
    preflight_checked = production_training_plan_preflight.get("checked") is True
    preflight_ok = production_training_plan_preflight.get("ok") is True
    preflight_status = str(production_training_plan_preflight.get("status") or "unknown")

    if dry_run_status != "ready_to_train":
        blockers.append("training_dry_run_not_ready")
    if not preflight_checked:
        blockers.append("production_training_plan_preflight_not_checked")
    elif not preflight_ok:
        blockers.append("production_training_plan_preflight_failed")
    elif preflight_status != "ready_to_train":
        blockers.append("production_training_plan_not_ready")
    if training_capture_preflight.get("required") is True and training_capture_preflight.get("gate_passed") is not True:
        blockers.append("production_capture_preflight_gate_failed")
    if production_missing_counts:
        blockers.append("production_label_minimums_not_met")

    return {
        "status": "ready_to_train" if not blockers else "blocked",
        "dry_run_status": dry_run_status,
        "production_preflight_checked": preflight_checked,
        "production_preflight_ok": preflight_ok,
        "production_preflight_status": preflight_status if preflight_checked else None,
        "blockers": blockers,
    }


def _iter_camera_docs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    cameras = doc.get("cameras") if isinstance(doc, dict) else {}
    if isinstance(cameras, dict):
        return [camera for camera in cameras.values() if isinstance(camera, dict)]
    if isinstance(cameras, list):
        return [camera for camera in cameras if isinstance(camera, dict)]
    return []


def _plan_payload(plan_result: Any) -> dict[str, Any]:
    if isinstance(plan_result, dict):
        payload = plan_result
    elif hasattr(plan_result, "to_dict"):
        payload = plan_result.to_dict()
    elif hasattr(plan_result, "__dict__"):
        payload = dict(plan_result.__dict__)
    else:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _plan_config_doc(plan_result: Any) -> dict[str, Any]:
    payload = _plan_payload(plan_result)
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    desired = config.get("desired_config") if isinstance(config.get("desired_config"), dict) else {}
    return desired


def _candidate_template_runtime_checks(
    doc: dict[str, Any],
    *,
    plan_result: Any,
    capability: str,
    role: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    matching_cameras = [
        camera
        for camera in _iter_camera_docs(doc)
        if capability in {str(value) for value in (camera.get("capabilities") or [])}
    ]
    if not matching_cameras:
        errors.append(f"no camera declares capability {capability}")
        return False, errors

    override_ok = False
    window_ok = False
    required_model_plan_ok = False
    for camera in matching_cameras:
        overrides = camera.get("capability_model_overrides")
        if overrides is None:
            overrides = camera.get("model_overrides")
        if isinstance(overrides, dict):
            raw_override = overrides.get(capability)
            model_key = raw_override.get("model_key") if isinstance(raw_override, dict) else raw_override
            override_ok = override_ok or model_key == PLANNED_MODEL_KEY

        raw_windows = camera.get("capability_windows")
        if raw_windows is None:
            raw_windows = camera.get("capability_active_windows")
        if isinstance(raw_windows, dict):
            schedule = raw_windows.get(capability)
            if isinstance(schedule, dict):
                windows = schedule.get("windows")
                has_windows = isinstance(windows, list) and bool(windows)
                if role == "suppression":
                    window_ok = window_ok or (has_windows and schedule.get("active") is False)
                else:
                    window_ok = window_ok or has_windows

    plan_doc = _plan_config_doc(plan_result)
    for camera in _iter_camera_docs(plan_doc):
        if capability not in {str(value) for value in (camera.get("capabilities") or [])}:
            continue
        execution_plan = camera.get("execution_plan")
        if not isinstance(execution_plan, dict):
            continue
        required_keys = {str(value) for value in (execution_plan.get("required_model_keys") or [])}
        required_model_plan_ok = required_model_plan_ok or PLANNED_MODEL_KEY in required_keys

    if not override_ok:
        errors.append(f"{capability} must override to {PLANNED_MODEL_KEY}")
    if not window_ok:
        if role == "suppression":
            errors.append(f"{capability} suppression template must declare active=false capability window")
        else:
            errors.append(f"{capability} template must declare an active capability window")
    if not required_model_plan_ok:
        errors.append(
            f"{capability} compiled execution plan must require {PLANNED_MODEL_KEY}"
        )
    return not errors, errors


def _candidate_template_isolation_checks(doc: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    site = doc.get("site") if isinstance(doc.get("site"), dict) else {}
    camera_count = len(_iter_camera_docs(doc))
    merge_existing = site.get("merge_existing")
    status = {
        "one_at_a_time": True,
        "merge_existing": merge_existing,
        "apply_replaces_existing_config": merge_existing is False,
        "camera_count": camera_count,
    }
    if merge_existing is not False:
        errors.append("candidate YAML site.merge_existing must be false for one-at-a-time execution")
    if camera_count != 1:
        errors.append("candidate YAML must declare exactly one camera for one-at-a-time execution")
    return not errors, status, errors


def _site_cli_preflight(config_path: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for command_name in ("validate", "plan"):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "safetylens_site.py"),
            "--config",
            _rel(config_path),
            command_name,
        ]
        command_text = ".venv/bin/python scripts/safetylens_site.py --config " f"{_rel(config_path)} {command_name}"
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            checks[command_name] = {
                "command": command_text,
                "exit_code": completed.returncode,
                "ok": completed.returncode == 0,
                "stdout_sha256": hashlib.sha256((completed.stdout or "").encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256((completed.stderr or "").encode("utf-8")).hexdigest(),
                "output_bytes": len(output.encode("utf-8")),
            }
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + (
                (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            )
            checks[command_name] = {
                "command": command_text,
                "exit_code": None,
                "ok": False,
                "timed_out": True,
                "stdout_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "output_bytes": len(output.encode("utf-8")),
            }
    return {
        "checked": True,
        "valid": all(item.get("ok") is True for item in checks.values()),
        "checks": checks,
    }


def _candidate_yaml_template_status(
    templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    template_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    templates = templates or CLOSED_SET_CANDIDATE_YAML_TEMPLATES

    for template in templates:
        scenario_id = str(template.get("scenario_id") or "")
        capability = str(template.get("capability") or "")
        role = str(template.get("role") or "")
        config_path = Path(template.get("config_path") or "")
        result_path = Path(template.get("expected_result_path") or "")
        backup_path = f"qa/video_eval/results/site_config_backups/before_{scenario_id}.yaml"
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "capability": capability,
            "role": role,
            "config_path": _rel(config_path),
            "expected_result_path": _rel(result_path),
            "commands": {
                "backup": f".venv/bin/python scripts/safetylens_site.py export --output {backup_path}",
                "validate": f".venv/bin/python scripts/safetylens_site.py --config {_rel(config_path)} validate",
                "plan": f".venv/bin/python scripts/safetylens_site.py --config {_rel(config_path)} plan",
                "apply": f".venv/bin/python scripts/safetylens_site.py --config {_rel(config_path)} apply --yes",
                "run": f".venv/bin/python scripts/video_eval.py run --scenario {scenario_id}",
                "restore": f".venv/bin/python scripts/safetylens_site.py --config {backup_path} apply --yes",
            },
            "execution_policy": {
                "one_at_a_time": True,
                "merge_existing": None,
                "apply_replaces_existing_config": False,
                "camera_count": 0,
            },
            "exists": config_path.exists(),
            "validate_ok": False,
            "plan_ok": False,
            "model_override_ok": False,
            "required_model_plan_ok": False,
            "one_at_a_time_ok": False,
            "window_ok": False,
            "cli_preflight_ok": False,
            "cli_preflight": {"checked": False, "valid": False, "checks": {}},
            "valid": False,
            "errors": [],
            "warnings": [],
        }
        if not config_path.exists():
            row["errors"].append(f"candidate YAML template missing: {_rel(config_path)}")
            blockers.append(f"{scenario_id}:missing_candidate_yaml_template")
            template_rows.append(row)
            continue

        try:
            doc = _load_yaml(config_path)
        except Exception as exc:
            row["errors"].append(f"candidate YAML template unreadable: {exc}")
            blockers.append(f"{scenario_id}:unreadable_candidate_yaml_template")
            template_rows.append(row)
            continue

        validate_result = site_config.load_site_config(config_path)
        row["validate_ok"] = validate_result.ok
        row["warnings"].extend(str(warning) for warning in validate_result.warnings)
        row["errors"].extend(str(error) for error in validate_result.errors)

        plan_result = site_config.build_plan(config_path)
        row["plan_ok"] = plan_result.ok
        row["warnings"].extend(str(warning) for warning in plan_result.warnings)
        row["errors"].extend(str(error) for error in plan_result.errors if str(error) not in row["errors"])
        cli_preflight = _site_cli_preflight(config_path)
        row["cli_preflight"] = cli_preflight
        row["cli_preflight_ok"] = cli_preflight.get("valid") is True
        for name, check in (cli_preflight.get("checks") or {}).items():
            if check.get("ok") is not True:
                row["errors"].append(f"safetylens_site.py {name} command failed")

        runtime_ok, runtime_errors = _candidate_template_runtime_checks(
            doc,
            plan_result=plan_result,
            capability=capability,
            role=role,
        )
        isolation_ok, isolation_status, isolation_errors = _candidate_template_isolation_checks(doc)
        row["execution_policy"] = isolation_status
        row["one_at_a_time_ok"] = isolation_ok
        row["model_override_ok"] = not any(
            "must override" in error or "no camera declares" in error
            for error in runtime_errors
        )
        row["required_model_plan_ok"] = not any(
            "compiled execution plan must require" in error or "no camera declares" in error
            for error in runtime_errors
        )
        row["window_ok"] = not any(
            "window" in error or "no camera declares" in error
            for error in runtime_errors
        )
        row["errors"].extend(runtime_errors)
        row["errors"].extend(isolation_errors)
        row["valid"] = (
            row["exists"]
            and row["validate_ok"]
            and row["plan_ok"]
            and row["cli_preflight_ok"]
            and runtime_ok
            and isolation_ok
        )
        if not row["valid"]:
            blockers.append(f"{scenario_id}:invalid_candidate_yaml_template")
        template_rows.append(row)

    return {
        "checked": True,
        "valid": not blockers,
        "required_model_key": PLANNED_MODEL_KEY,
        "template_count": len(template_rows),
        "valid_template_count": sum(1 for row in template_rows if row.get("valid") is True),
        "templates": template_rows,
        "blockers": blockers,
    }


def _candidate_runtime_evidence_status(
    templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    templates = templates or CLOSED_SET_CANDIDATE_YAML_TEMPLATES

    for template in templates:
        scenario_id = str(template.get("scenario_id") or "")
        capability = str(template.get("capability") or "")
        role = str(template.get("role") or "")
        result_path = Path(template.get("expected_result_path") or "")
        aliases = set(CAPABILITY_SCENARIOS.get(capability, {}).get("visible_aliases") or [])
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "capability": capability,
            "role": role,
            "path": _rel(result_path),
            "exists": result_path.exists(),
            "valid": False,
            "preflight_blocked_missing_required_model": False,
            "summary": {},
            "errors": [],
        }
        if not result_path.exists():
            row["errors"].append(f"candidate result JSON missing: {_rel(result_path)}")
            blockers.append(f"{scenario_id}:missing_candidate_runtime_result")
            rows.append(row)
            continue
        try:
            result = _load_json(result_path)
        except Exception as exc:
            row["errors"].append(f"candidate result JSON unreadable: {exc}")
            blockers.append(f"{scenario_id}:unreadable_candidate_runtime_result")
            rows.append(row)
            continue

        summary = _scenario_summary(result, capability, aliases)
        row["summary"] = summary
        row["required_evidence"] = _candidate_required_evidence(role, capability)
        row["observed_evidence"] = _candidate_observed_evidence(result, summary)
        identity_error = _candidate_runtime_identity_error(
            result=result,
            template=template,
            observed=row["observed_evidence"],
        )
        evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        model_preflight = (
            evidence.get("model_preflight") if isinstance(evidence.get("model_preflight"), dict) else {}
        )
        missing_model_keys = {
            str(model_key) for model_key in (model_preflight.get("missing_model_keys") or [])
        }
        if (
            result.get("status") == "blocked"
            and model_preflight.get("checked") is True
            and model_preflight.get("ok") is False
            and PLANNED_MODEL_KEY in missing_model_keys
        ):
            error = "blocked_missing_required_model_preflight"
            row["preflight_blocked_missing_required_model"] = True
            if identity_error:
                row["errors"].append(identity_error)
                blockers.append(f"{scenario_id}:{identity_error}")
        elif identity_error:
            error = identity_error
        else:
            error = _candidate_model_preflight_error(scenario_id, result, model_preflight)
            if not error:
                error = _candidate_invocation_isolation_error(summary)
            if not error and role == "active":
                error = _promotion_active_summary_error(role, summary, PLANNED_MODEL_KEY)
            elif not error and role == "false_positive_guard":
                error = _promotion_guard_summary_error(role, summary, PLANNED_MODEL_KEY)
            elif not error and role == "suppression":
                error = _promotion_suppression_summary_error(role, summary, capability, PLANNED_MODEL_KEY)
            elif not error:
                error = f"unknown_candidate_runtime_role:{role}"
        if error:
            row["errors"].append(error)
            blockers.append(f"{scenario_id}:{error}")
        row["valid"] = not row["errors"]
        rows.append(row)

    return {
        "checked": True,
        "valid": not blockers,
        "required_model_key": PLANNED_MODEL_KEY,
        "result_count": len(rows),
        "present_result_count": sum(1 for row in rows if row.get("exists") is True),
        "valid_result_count": sum(1 for row in rows if row.get("valid") is True),
        "missing_result_count": sum(1 for row in rows if row.get("exists") is not True),
        "preflight_blocked_missing_model_count": sum(
            1 for row in rows if row.get("preflight_blocked_missing_required_model") is True
        ),
        "results": rows,
        "blockers": blockers,
    }


def _model_manager_definition_status(planned_registry_path: str) -> dict[str, Any]:
    errors: list[str] = []
    status: dict[str, Any] = {
        "checked": True,
        "model_key": PLANNED_MODEL_KEY,
        "expected_registry_path": planned_registry_path,
        "registered": False,
        "valid": False,
        "artifact_exists": False,
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
    status["display_name"] = definition.get("display_name")
    status["filename"] = definition.get("filename")
    status["download_url"] = definition.get("download_url")
    status["warmup_behavior"] = definition.get("warmup_behavior")
    status["shared_asset_key"] = definition.get("shared_asset_key")

    local_path = definition.get("local_path")
    if isinstance(local_path, Path):
        status["local_path"] = str(local_path)
        try:
            status["registry_path"] = str(local_path.relative_to(ROOT))
        except ValueError:
            status["registry_path"] = str(local_path)
        status["artifact_exists"] = local_path.exists()
    else:
        errors.append("local_path must be a pathlib.Path")

    expected_filename = Path(planned_registry_path).name if planned_registry_path else "apron-harness-ppe.onnx"
    if definition.get("model_key") != PLANNED_MODEL_KEY:
        errors.append("definition.model_key mismatch")
    if definition.get("filename") != expected_filename:
        errors.append("definition.filename mismatch")
    if planned_registry_path and status.get("registry_path") != planned_registry_path:
        errors.append("definition.local_path does not match planned_registry_path")
    if definition.get("download_url"):
        errors.append("closed-set candidate must be a manual artifact install with empty download_url")
    if definition.get("shared_asset_key") != "apron-harness-ppe":
        errors.append("definition.shared_asset_key mismatch")

    status["valid"] = not errors
    return status


def _promotion_gate_ok(path: Path | None, capability: str) -> tuple[bool, str]:
    if path is None:
        return False, "missing"
    if not path.exists():
        return False, f"missing:{_rel(path)}"
    try:
        payload = _load_json(path)
    except Exception:
        return False, f"unreadable:{_rel(path)}"
    if payload.get("ok") is not True:
        return False, f"failed:{_rel(path)}"
    if payload.get("promotion_status") != "ready_for_runtime_registration":
        return False, f"not_ready:{_rel(path)}"
    if payload.get("capability") != capability:
        return False, f"wrong_capability:{payload.get('capability')}"
    if payload.get("candidate_model_key") != PLANNED_MODEL_KEY:
        return False, f"wrong_candidate_model:{payload.get('candidate_model_key')}"
    if payload.get("baseline_model_key") != PILOT_MODEL_KEY:
        return False, f"wrong_baseline_model:{payload.get('baseline_model_key')}"
    candidate_report_sha = str(payload.get("candidate_report_sha256") or "")
    if len(candidate_report_sha) != 64:
        return False, f"missing_candidate_report_sha256:{_rel(path)}"
    artifact_error = _promotion_artifact_identity_error(payload)
    if artifact_error:
        return False, f"{artifact_error}:{_rel(path)}"
    sidecar = payload.get("candidate_capture_matrix_manifest")
    if not isinstance(sidecar, dict):
        return False, f"missing_candidate_capture_matrix_manifest:{_rel(path)}"
    if sidecar.get("valid") is not True:
        return False, f"invalid_candidate_capture_matrix_manifest:{_rel(path)}"
    if sidecar.get("mode") != "production":
        return False, f"candidate_capture_matrix_manifest_not_production:{_rel(path)}"
    if not sidecar.get("matrix_csv_sha256") or not sidecar.get("source_manifest_sha256"):
        return False, f"candidate_capture_matrix_manifest_missing_hashes:{_rel(path)}"
    label_sidecar = payload.get("candidate_label_review_import_manifest")
    if not isinstance(label_sidecar, dict):
        return False, f"missing_candidate_label_review_import_manifest:{_rel(path)}"
    if label_sidecar.get("valid") is not True:
        return False, f"invalid_candidate_label_review_import_manifest:{_rel(path)}"
    if not label_sidecar.get("label_review_csv_sha256") or not label_sidecar.get("updated_manifest_sha256"):
        return False, f"candidate_label_review_import_manifest_missing_hashes:{_rel(path)}"
    if label_sidecar.get("updated_manifest_sha256") != sidecar.get("source_manifest_sha256"):
        return False, f"candidate_label_review_import_manifest_source_manifest_sha_mismatch:{_rel(path)}"
    validation = (
        label_sidecar.get("updated_manifest_validation")
        if isinstance(label_sidecar.get("updated_manifest_validation"), dict)
        else {}
    )
    if not validation:
        return False, f"candidate_label_review_import_manifest_validation_missing:{_rel(path)}"
    if validation.get("checked") is not True or validation.get("ok") is not True:
        return False, f"candidate_label_review_import_manifest_validation_failed:{_rel(path)}"
    if validation.get("schema_only") is True:
        return False, f"candidate_label_review_import_manifest_validation_schema_only:{_rel(path)}"
    if validation.get("mode") != "production":
        return False, f"candidate_label_review_import_manifest_validation_not_production:{_rel(path)}"
    if validation.get("manifest_sha256") != sidecar.get("source_manifest_sha256"):
        return False, f"candidate_label_review_import_manifest_validation_sha_mismatch:{_rel(path)}"
    clip_metadata_error = _promotion_label_review_clip_metadata_error(label_sidecar)
    if clip_metadata_error:
        return False, f"{clip_metadata_error}:{_rel(path)}"
    provenance_error = _promotion_dataset_provenance_error(
        payload.get("candidate_training_dataset_provenance"),
        sidecar,
    )
    if provenance_error:
        return False, f"{provenance_error}:{_rel(path)}"
    source_lineage_error = _promotion_source_lineage_error(
        payload.get("candidate_training_source_lineage"),
        sidecar,
        payload.get("candidate_training_dataset_provenance") or {},
    )
    if source_lineage_error:
        return False, f"{source_lineage_error}:{_rel(path)}"
    source_lineage = (
        payload.get("candidate_training_source_lineage")
        if isinstance(payload.get("candidate_training_source_lineage"), dict)
        else {}
    )
    seed_import = (
        source_lineage.get("seed_import_manifest")
        if isinstance(source_lineage.get("seed_import_manifest"), dict)
        else {}
    )
    seed_import_gate = seed_import.get("gate") if isinstance(seed_import.get("gate"), dict) else {}
    if seed_import_gate.get("required") is True:
        seed_export_error = _promotion_seed_export_import_manifest_error(
            payload.get("candidate_seed_export_import_manifest"),
            sidecar,
        )
        if seed_export_error:
            return False, f"{seed_export_error}:{_rel(path)}"
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        return False, f"missing_baseline_runtime_evidence:{_rel(path)}"
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        return False, f"missing_candidate_runtime_evidence:{_rel(path)}"
    baseline_error = _promotion_runtime_group_error("baseline", baseline, capability, PILOT_MODEL_KEY)
    if baseline_error:
        return False, f"invalid_{baseline_error}:{_rel(path)}"
    candidate_error = _promotion_runtime_group_error("candidate", candidate, capability, PLANNED_MODEL_KEY)
    if candidate_error:
        return False, f"invalid_{candidate_error}:{_rel(path)}"
    return True, f"ok:{_rel(path)}"


def _promotion_matches_handoff_source(
    path: Path | None,
    *,
    expected_source_manifest_sha256: str | None,
) -> tuple[bool, str]:
    if path is None or not expected_source_manifest_sha256:
        return True, "not_checked"
    try:
        payload = _load_json(path)
    except Exception:
        return True, "not_checked"
    sidecar = payload.get("candidate_capture_matrix_manifest")
    label_sidecar = payload.get("candidate_label_review_import_manifest")
    seed_export_sidecar = payload.get("candidate_seed_export_import_manifest")
    provenance = payload.get("candidate_training_dataset_provenance")
    source_lineage = payload.get("candidate_training_source_lineage")
    sidecar_sha = sidecar.get("source_manifest_sha256") if isinstance(sidecar, dict) else None
    label_sidecar_sha = (
        label_sidecar.get("updated_manifest_sha256") if isinstance(label_sidecar, dict) else None
    )
    label_sidecar_validation_sha = None
    if isinstance(label_sidecar, dict):
        validation = label_sidecar.get("updated_manifest_validation")
        if isinstance(validation, dict):
            label_sidecar_validation_sha = validation.get("manifest_sha256")
    seed_export_sidecar_sha = (
        seed_export_sidecar.get("updated_manifest_sha256")
        if isinstance(seed_export_sidecar, dict)
        else None
    )
    seed_export_sidecar_validation_sha = None
    if isinstance(seed_export_sidecar, dict):
        validation = seed_export_sidecar.get("updated_manifest_validation")
        if isinstance(validation, dict):
            seed_export_sidecar_validation_sha = validation.get("manifest_sha256")
    provenance_sha = provenance.get("source_manifest_sha256") if isinstance(provenance, dict) else None
    source_lineage_capture_sha = None
    source_lineage_capture_manifest_sha = None
    if isinstance(source_lineage, dict):
        capture_lineage = source_lineage.get("capture_manifest")
        if isinstance(capture_lineage, dict):
            source_lineage_capture_manifest_sha = capture_lineage.get("manifest_sha256")
            capture_file = capture_lineage.get("file")
            if isinstance(capture_file, dict):
                source_lineage_capture_sha = capture_file.get("sha256")
    if sidecar_sha and sidecar_sha != expected_source_manifest_sha256:
        return False, f"candidate_capture_matrix_manifest_source_manifest_sha_mismatch:{_rel(path)}"
    if label_sidecar_sha and label_sidecar_sha != expected_source_manifest_sha256:
        return False, f"candidate_label_review_import_manifest_source_manifest_sha_mismatch:{_rel(path)}"
    if label_sidecar_validation_sha and label_sidecar_validation_sha != expected_source_manifest_sha256:
        return False, f"candidate_label_review_import_manifest_validation_source_manifest_sha_mismatch:{_rel(path)}"
    if seed_export_sidecar_sha and seed_export_sidecar_sha != expected_source_manifest_sha256:
        return False, f"candidate_seed_export_import_manifest_source_manifest_sha_mismatch:{_rel(path)}"
    if seed_export_sidecar_validation_sha and seed_export_sidecar_validation_sha != expected_source_manifest_sha256:
        return False, f"candidate_seed_export_import_manifest_validation_source_manifest_sha_mismatch:{_rel(path)}"
    if provenance_sha and provenance_sha != expected_source_manifest_sha256:
        return False, f"candidate_training_dataset_provenance_source_manifest_sha_mismatch:{_rel(path)}"
    if source_lineage_capture_sha and source_lineage_capture_sha != expected_source_manifest_sha256:
        return False, f"candidate_training_source_lineage_capture_manifest_file_sha_mismatch:{_rel(path)}"
    if source_lineage_capture_manifest_sha and source_lineage_capture_manifest_sha != expected_source_manifest_sha256:
        return False, f"candidate_training_source_lineage_capture_manifest_sha_mismatch:{_rel(path)}"
    return True, f"ok:{_rel(path)}"


def _jetson_gate_ok(path: Path | None, expected_model_name: str) -> tuple[bool, str]:
    if path is None:
        return False, "missing"
    if not path.exists():
        return False, f"missing:{_rel(path)}"
    try:
        payload = _load_json(path)
    except Exception:
        return False, f"unreadable:{_rel(path)}"
    if payload.get("ok") is not True:
        return False, f"failed:{_rel(path)}"
    if payload.get("pack_id") != "factory_ppe_3cam":
        return False, f"wrong_pack:{payload.get('pack_id')}"
    if payload.get("gate_status") != "jetson_gate_passed" or payload.get("production_gate") is not True:
        return False, f"gate_false:{_rel(path)}"
    model = payload.get("model")
    if model and _basename(model) != _basename(expected_model_name):
        return False, f"wrong_model:{model}"
    raw = payload.get("raw_benchmark")
    soak = payload.get("soak_report")
    if not isinstance(raw, dict) or raw.get("present") is not True:
        return False, f"missing_raw_benchmark:{_rel(path)}"
    if not isinstance(soak, dict) or soak.get("present") is not True:
        return False, f"missing_soak_report:{_rel(path)}"
    return True, f"ok:{_rel(path)}"


def _model_registry_gate_ok(path: Path | None, expected_registry_path: str) -> tuple[bool, str]:
    if path is None:
        return False, "missing"
    if not path.exists():
        return False, f"missing:{_rel(path)}"
    try:
        payload = _load_json(path)
    except Exception:
        return False, f"unreadable:{_rel(path)}"
    if payload.get("ok") is not True:
        return False, f"failed:{_rel(path)}"
    if payload.get("registry_status") != "registered":
        return False, f"not_registered:{payload.get('registry_status')}"
    candidate_report_sha256 = str(payload.get("candidate_report_sha256") or "")
    if len(candidate_report_sha256) != 64:
        return False, f"missing_candidate_report_sha256:{_rel(path)}"
    model_definition = payload.get("model_manager_definition")
    if not isinstance(model_definition, dict) or model_definition.get("valid") is not True:
        return False, f"invalid_model_manager_definition:{_rel(path)}"
    if model_definition.get("model_key") != PLANNED_MODEL_KEY:
        return False, f"wrong_model_definition_key:{model_definition.get('model_key')}"
    if model_definition.get("expected_registry_path") != expected_registry_path:
        return False, f"wrong_model_definition_path:{model_definition.get('expected_registry_path')}"
    registry_entry = payload.get("registry_entry")
    if not isinstance(registry_entry, dict):
        return False, f"missing_registry_entry:{_rel(path)}"
    if registry_entry.get("model_key") != PLANNED_MODEL_KEY:
        return False, f"wrong_registry_entry_model:{registry_entry.get('model_key')}"
    if registry_entry.get("registry_path") != expected_registry_path:
        return False, f"wrong_registry_entry_path:{registry_entry.get('registry_path')}"
    source_export = payload.get("source_export")
    destination = payload.get("destination")
    if not isinstance(source_export, dict):
        return False, f"missing_source_export:{_rel(path)}"
    if not isinstance(destination, dict):
        return False, f"missing_destination:{_rel(path)}"
    expected_sha = source_export.get("expected_sha256")
    destination_sha = destination.get("sha256")
    if not expected_sha or len(str(expected_sha)) != 64:
        return False, f"missing_source_export_sha256:{_rel(path)}"
    if destination.get("registry_path") != expected_registry_path:
        return False, f"wrong_destination_path:{destination.get('registry_path')}"
    if destination.get("exists") is not True:
        return False, f"destination_missing:{_rel(path)}"
    if destination.get("matches_expected_sha256") is not True:
        return False, f"destination_sha_mismatch:{_rel(path)}"
    if destination_sha != expected_sha:
        return False, f"destination_source_sha_mismatch:{_rel(path)}"
    metadata = destination.get("registry_metadata")
    if not isinstance(metadata, dict):
        return False, f"missing_registry_metadata:{_rel(path)}"
    if metadata.get("exists") is not True:
        return False, f"missing_registry_metadata:{_rel(path)}"
    metadata_sha = str(metadata.get("sha256") or "")
    if len(metadata_sha) != 64:
        return False, f"missing_registry_metadata_sha256:{_rel(path)}"
    if metadata.get("valid") is not True:
        return False, f"invalid_registry_metadata:{_rel(path)}"
    return True, f"ok:{_rel(path)}"


def _model_registry_blocker(model_registry_status: str) -> str:
    if model_registry_status.startswith("not_registered:"):
        return "apron_harness_model_registry_not_registered"
    return "missing_or_failed_apron_harness_model_registry_report"


def _model_pack_evidence_gate_ok(
    path: Path | None,
    *,
    model_packs_path: Path,
    result_dir: Path,
    expected_pack_status: str,
) -> tuple[bool, str]:
    if path is None:
        return False, "missing"
    if not path.exists():
        return False, f"missing:{_rel(path)}"
    try:
        payload = _load_json(path)
    except Exception:
        return False, f"unreadable:{_rel(path)}"
    if payload.get("ok") is not True:
        return False, f"failed:{_rel(path)}"
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    if inputs.get("model_packs") != _rel(model_packs_path):
        return False, f"stale_model_packs:{inputs.get('model_packs')}"
    if inputs.get("result_dir") != _rel(result_dir):
        return False, f"stale_result_dir:{inputs.get('result_dir')}"
    packs = payload.get("packs") if isinstance(payload.get("packs"), dict) else {}
    skipped_model_packs = {
        str(pack_id)
        for pack_id in payload.get("skipped_model_packs") or []
        if str(pack_id).strip()
    }
    missing_skipped = sorted(EXPECTED_SKIPPED_MODEL_PACKS - skipped_model_packs)
    if missing_skipped:
        return False, f"missing_skipped_model_packs:{','.join(missing_skipped)}"
    active_skipped = sorted(pack_id for pack_id in EXPECTED_SKIPPED_MODEL_PACKS if pack_id in packs)
    if active_skipped:
        return False, f"out_of_scope_model_pack_present:{','.join(active_skipped)}"
    factory_ppe = packs.get("factory_ppe_3cam") if isinstance(packs.get("factory_ppe_3cam"), dict) else {}
    if factory_ppe.get("ok") is not True:
        return False, f"factory_ppe_failed:{_rel(path)}"
    if str(factory_ppe.get("status") or "") != expected_pack_status:
        return False, f"stale_factory_ppe_status:{factory_ppe.get('status')}"
    return True, f"ok:{_rel(path)}"


def _seed_source_review_gate(
    path: Path | None,
    *,
    model_packs_path: Path,
    expected_candidate_count: int,
) -> tuple[bool, bool, str, dict[str, Any]]:
    if path is None:
        return False, False, "missing", {}
    if not path.exists():
        return False, False, f"missing:{_rel(path)}", {}
    try:
        payload = _load_json(path)
    except Exception:
        return False, False, f"unreadable:{_rel(path)}", {}
    if payload.get("ok") is not True:
        return False, bool(payload.get("gate_passed")), f"failed:{_rel(path)}", payload
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    if inputs.get("model_packs") != _rel(model_packs_path):
        return False, bool(payload.get("gate_passed")), f"stale_model_packs:{inputs.get('model_packs')}", payload
    try:
        candidate_count = int(payload.get("candidate_count") or 0)
    except (TypeError, ValueError):
        candidate_count = -1
    if candidate_count != expected_candidate_count:
        return False, bool(payload.get("gate_passed")), f"stale_candidate_count:{candidate_count}", payload
    status = "pass" if payload.get("gate_passed") else "blocked"
    return True, bool(payload.get("gate_passed")), f"{status}:{_rel(path)}", payload


def _seed_import_manifest_gate(
    path: Path | None,
    *,
    seed_source_review: dict[str, Any],
    required: bool,
) -> tuple[bool, bool, str, dict[str, Any]]:
    if not required:
        return True, True, "not_required", {}
    if path is None:
        return False, False, "missing", {}
    if not path.exists():
        return False, False, f"missing:{_rel(path)}", {}
    try:
        payload = validate_import_manifest(path, seed_source_review)
    except Exception as exc:
        return False, False, f"unreadable:{_rel(path)}:{exc}", {}
    if payload.get("ok") is not True:
        return False, bool(payload.get("gate_passed")), f"failed:{_rel(path)}", payload
    status = "pass" if payload.get("gate_passed") else "blocked"
    return True, bool(payload.get("gate_passed")), f"{status}:{_rel(path)}", payload


def _seed_source_review_bundle_gate(
    path: Path | None,
    *,
    seed_source_review: dict[str, Any],
    required: bool,
) -> tuple[bool, str, dict[str, Any]]:
    if not required:
        return True, "not_required", {}
    if path is None:
        return False, "missing", {}
    if not path.exists():
        return False, f"missing:{_rel(path)}", {}
    try:
        payload = validate_review_bundle_manifest(path, seed_source_review)
    except Exception as exc:
        return False, f"unreadable:{_rel(path)}:{exc}", {}
    status = "pass" if payload.get("ok") is True else "failed"
    return bool(payload.get("ok") is True), f"{status}:{_rel(path)}", payload


def _closed_set_handoff_status(
    capture_manifest: Path,
    training_dataset_yaml: Path,
    training_model: str,
    seed_source_review_report: Path | None = None,
    seed_import_manifest: Path | None = None,
    capture_kickoff_out: Path | None = None,
    capture_work_order_out: Path | None = None,
    capture_matrix_csv_out: Path | None = None,
    production_capture_matrix_csv_out: Path | None = None,
    label_review_csv_out: Path | None = None,
    starter_label_review_csv_out: Path | None = None,
    production_label_review_csv_out: Path | None = None,
    production_starter_label_review_csv_out: Path | None = None,
    training_dataset_yaml_out: Path | None = None,
    training_dataset_root: str = DEFAULT_TRAINING_DATASET_ROOT,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_report: dict[str, Any] = {}
    training_plan: dict[str, Any] = {}
    training_dataset_yaml_handoff = _write_training_dataset_yaml(
        path=training_dataset_yaml_out,
        capture_manifest=capture_manifest,
        dataset_root=training_dataset_root,
    )
    production_training_dataset_yaml = (
        training_dataset_yaml_out
        if training_dataset_yaml_handoff.get("generated") is True and training_dataset_yaml_out is not None
        else training_dataset_yaml
    )

    try:
        dataset_report = validate_manifest(
            capture_manifest,
            mode="pilot",
            schema_only=True,
            seed_source_review_report=seed_source_review_report,
        )
    except Exception as exc:
        errors.append(f"capture manifest schema check failed: {exc}")
        dataset_report = {
            "ok": False,
            "manifest": str(capture_manifest),
            "errors": [str(exc)],
            "warnings": [],
            "counts": {},
        }

    try:
        production_dataset_report = validate_manifest(
            capture_manifest,
            mode="production",
            schema_only=True,
            seed_source_review_report=seed_source_review_report,
        )
    except Exception as exc:
        warnings.append(f"production capture manifest schema check failed: {exc}")
        production_dataset_report = {
            "ok": False,
            "manifest": str(capture_manifest),
            "errors": [str(exc)],
            "warnings": [],
            "counts": {},
            "capture_deficit": {},
        }

    warnings.extend(str(warning) for warning in dataset_report.get("warnings") or [])
    if dataset_report.get("ok") is not True:
        errors.extend(str(error) for error in dataset_report.get("errors") or [])

    try:
        training_plan = build_training_plan(
            data_path=production_training_dataset_yaml,
            model=training_model,
            device="auto",
            epochs=100,
            imgsz=640,
            batch=8,
            project=ROOT / "runs",
            name="apron_harness_closed_set",
            export_formats=["onnx"],
        )
    except Exception as exc:
        errors.append(f"training dry-run failed: {exc}")
        training_plan = {"status": "failed", "error": str(exc)}

    selected_device = str(training_plan.get("selected_device") or "")
    if selected_device and selected_device != "mps":
        warnings.append(f"training dry-run selected {selected_device}; local MPS performance gate is not satisfied")

    counts = dataset_report.get("counts") if isinstance(dataset_report.get("counts"), dict) else {}
    capture_deficit = (
        dataset_report.get("capture_deficit")
        if isinstance(dataset_report.get("capture_deficit"), dict)
        else {}
    )
    production_capture_deficit = (
        production_dataset_report.get("capture_deficit")
        if isinstance(production_dataset_report.get("capture_deficit"), dict)
        else {}
    )
    capture_kickoff: dict[str, Any] = {
        "path": _rel(capture_kickoff_out) if capture_kickoff_out else None,
        "generated": False,
        "exists": False,
    }
    capture_work_order: dict[str, Any] = {
        "path": _rel(capture_work_order_out) if capture_work_order_out else None,
        "generated": False,
        "exists": False,
    }
    capture_matrix_csv: dict[str, Any] = {
        "path": _rel(capture_matrix_csv_out) if capture_matrix_csv_out else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
    }
    capture_matrix_manifest: dict[str, Any] = {
        "path": _rel(_matrix_sidecar_path(capture_matrix_csv_out)) if capture_matrix_csv_out else None,
        "generated": False,
        "exists": False,
    }
    production_capture_matrix_csv: dict[str, Any] = {
        "path": _rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
    }
    production_capture_matrix_manifest: dict[str, Any] = {
        "path": _rel(_matrix_sidecar_path(production_capture_matrix_csv_out))
        if production_capture_matrix_csv_out
        else None,
        "generated": False,
        "exists": False,
    }
    label_review_csv: dict[str, Any] = {
        "path": _rel(label_review_csv_out) if label_review_csv_out else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
        "schema": LABEL_REVIEW_SCHEMA_SUMMARY,
    }
    starter_label_review_csv: dict[str, Any] = {
        "path": _rel(starter_label_review_csv_out) if starter_label_review_csv_out else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
        "schema": LABEL_REVIEW_SCHEMA_SUMMARY,
        "scope": "immediate_starter_rows",
    }
    production_label_review_csv: dict[str, Any] = {
        "path": _rel(production_label_review_csv_out) if production_label_review_csv_out else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
        "schema": LABEL_REVIEW_SCHEMA_SUMMARY,
    }
    production_starter_label_review_csv: dict[str, Any] = {
        "path": _rel(production_starter_label_review_csv_out)
        if production_starter_label_review_csv_out
        else None,
        "generated": False,
        "exists": False,
        "row_count": 0,
        "schema": LABEL_REVIEW_SCHEMA_SUMMARY,
        "scope": "immediate_starter_rows",
    }
    work_order_report = dict(dataset_report)
    work_order_report["manifest"] = _rel(capture_manifest)
    work_order_report["production_capture_deficit"] = production_capture_deficit
    if capture_kickoff_out is not None:
        try:
            capture_kickoff_out.parent.mkdir(parents=True, exist_ok=True)
            capture_kickoff_out.write_text(
                render_capture_kickoff(
                    work_order_report,
                    capture_manifest=capture_manifest,
                    capture_work_order_out=capture_work_order_out,
                    capture_matrix_csv_out=capture_matrix_csv_out,
                    production_capture_matrix_csv_out=production_capture_matrix_csv_out,
                    label_review_csv_out=label_review_csv_out,
                    starter_label_review_csv_out=starter_label_review_csv_out,
                    production_label_review_csv_out=production_label_review_csv_out,
                    production_starter_label_review_csv_out=production_starter_label_review_csv_out,
                ),
                encoding="utf-8",
            )
            capture_kickoff["generated"] = True
            capture_kickoff["exists"] = capture_kickoff_out.exists()
            capture_kickoff["sha256"] = _sha256_file(capture_kickoff_out)
            capture_kickoff["source_manifest_sha256"] = dataset_report.get("manifest_sha256")
        except Exception as exc:
            errors.append(f"capture kickoff generation failed: {exc}")
            capture_kickoff["error"] = str(exc)
    if capture_work_order_out is not None:
        try:
            capture_work_order_out.parent.mkdir(parents=True, exist_ok=True)
            capture_work_order_out.write_text(render_capture_work_order(work_order_report), encoding="utf-8")
            capture_work_order["generated"] = True
            capture_work_order["exists"] = capture_work_order_out.exists()
            capture_work_order["sha256"] = _sha256_file(capture_work_order_out)
            capture_work_order["source_manifest_sha256"] = dataset_report.get("manifest_sha256")
        except Exception as exc:
            errors.append(f"capture work order generation failed: {exc}")
            capture_work_order["error"] = str(exc)
    if capture_matrix_csv_out is not None:
        try:
            capture_matrix_csv_out.parent.mkdir(parents=True, exist_ok=True)
            capture_matrix_csv_out.write_text(render_capture_matrix_csv(work_order_report), encoding="utf-8")
            capture_matrix_csv["generated"] = True
            capture_matrix_csv["exists"] = capture_matrix_csv_out.exists()
            capture_matrix_csv["sha256"] = _sha256_file(capture_matrix_csv_out)
            capture_matrix_csv["source_manifest_sha256"] = dataset_report.get("manifest_sha256")
            capture_matrix_csv["row_count"] = sum(
                len(batch.get("capture_matrix") or [])
                for batch in (capture_deficit.get("next_capture_batches") or [])
                if isinstance(batch, dict)
            )
            capture_matrix_manifest = _write_capture_matrix_sidecar(
                matrix_csv_path=capture_matrix_csv_out,
                mode="pilot",
                source_manifest=capture_manifest,
                source_manifest_sha256=dataset_report.get("manifest_sha256"),
                row_count=int(capture_matrix_csv["row_count"]),
                capture_deficit=capture_deficit,
            )
        except Exception as exc:
            errors.append(f"capture matrix CSV generation failed: {exc}")
            capture_matrix_csv["error"] = str(exc)
            capture_matrix_manifest["error"] = str(exc)
    if label_review_csv_out is not None:
        try:
            label_review_csv_out.parent.mkdir(parents=True, exist_ok=True)
            rendered_label_review_csv = render_label_review_csv(work_order_report, mode="pilot")
            label_review_csv_out.write_text(rendered_label_review_csv, encoding="utf-8")
            label_review_csv["generated"] = True
            label_review_csv["exists"] = label_review_csv_out.exists()
            label_review_csv["sha256"] = _sha256_file(label_review_csv_out)
            label_review_csv["source_manifest_sha256"] = dataset_report.get("manifest_sha256")
            label_review_csv["row_count"] = max(0, len(rendered_label_review_csv.splitlines()) - 1)
        except Exception as exc:
            errors.append(f"label review CSV generation failed: {exc}")
            label_review_csv["error"] = str(exc)
    if starter_label_review_csv_out is not None:
        try:
            starter_label_review_csv_out.parent.mkdir(parents=True, exist_ok=True)
            rendered_starter_label_review_csv = render_starter_label_review_csv(
                work_order_report,
                mode="pilot",
            )
            starter_label_review_csv_out.write_text(rendered_starter_label_review_csv, encoding="utf-8")
            starter_label_review_csv["generated"] = True
            starter_label_review_csv["exists"] = starter_label_review_csv_out.exists()
            starter_label_review_csv["sha256"] = _sha256_file(starter_label_review_csv_out)
            starter_label_review_csv["source_manifest_sha256"] = dataset_report.get("manifest_sha256")
            starter_label_review_csv["row_count"] = max(
                0,
                len(rendered_starter_label_review_csv.splitlines()) - 1,
            )
        except Exception as exc:
            errors.append(f"starter label review CSV generation failed: {exc}")
            starter_label_review_csv["error"] = str(exc)
    production_work_order_report = dict(production_dataset_report)
    production_work_order_report["manifest"] = _rel(capture_manifest)
    if production_capture_matrix_csv_out is not None:
        try:
            production_capture_matrix_csv_out.parent.mkdir(parents=True, exist_ok=True)
            production_capture_matrix_csv_out.write_text(render_capture_matrix_csv(production_work_order_report), encoding="utf-8")
            production_capture_matrix_csv["generated"] = True
            production_capture_matrix_csv["exists"] = production_capture_matrix_csv_out.exists()
            production_capture_matrix_csv["sha256"] = _sha256_file(production_capture_matrix_csv_out)
            production_capture_matrix_csv["source_manifest_sha256"] = production_dataset_report.get("manifest_sha256")
            production_capture_matrix_csv["row_count"] = sum(
                len(batch.get("capture_matrix") or [])
                for batch in (production_capture_deficit.get("next_capture_batches") or [])
                if isinstance(batch, dict)
            )
            production_capture_matrix_manifest = _write_capture_matrix_sidecar(
                matrix_csv_path=production_capture_matrix_csv_out,
                mode="production",
                source_manifest=capture_manifest,
                source_manifest_sha256=production_dataset_report.get("manifest_sha256"),
                row_count=int(production_capture_matrix_csv["row_count"]),
                capture_deficit=production_capture_deficit,
            )
        except Exception as exc:
            errors.append(f"production capture matrix CSV generation failed: {exc}")
            production_capture_matrix_csv["error"] = str(exc)
            production_capture_matrix_manifest["error"] = str(exc)
    if production_label_review_csv_out is not None:
        try:
            production_label_review_csv_out.parent.mkdir(parents=True, exist_ok=True)
            rendered_production_label_review_csv = render_label_review_csv(
                production_work_order_report,
                mode="production",
            )
            production_label_review_csv_out.write_text(rendered_production_label_review_csv, encoding="utf-8")
            production_label_review_csv["generated"] = True
            production_label_review_csv["exists"] = production_label_review_csv_out.exists()
            production_label_review_csv["sha256"] = _sha256_file(production_label_review_csv_out)
            production_label_review_csv["source_manifest_sha256"] = production_dataset_report.get("manifest_sha256")
            production_label_review_csv["row_count"] = max(
                0,
                len(rendered_production_label_review_csv.splitlines()) - 1,
            )
        except Exception as exc:
            errors.append(f"production label review CSV generation failed: {exc}")
            production_label_review_csv["error"] = str(exc)
    if production_starter_label_review_csv_out is not None:
        try:
            production_starter_label_review_csv_out.parent.mkdir(parents=True, exist_ok=True)
            rendered_production_starter_label_review_csv = render_starter_label_review_csv(
                production_work_order_report,
                mode="production",
            )
            production_starter_label_review_csv_out.write_text(
                rendered_production_starter_label_review_csv,
                encoding="utf-8",
            )
            production_starter_label_review_csv["generated"] = True
            production_starter_label_review_csv["exists"] = production_starter_label_review_csv_out.exists()
            production_starter_label_review_csv["sha256"] = _sha256_file(
                production_starter_label_review_csv_out
            )
            production_starter_label_review_csv["source_manifest_sha256"] = production_dataset_report.get(
                "manifest_sha256"
            )
            production_starter_label_review_csv["row_count"] = max(
                0,
                len(rendered_production_starter_label_review_csv.splitlines()) - 1,
            )
        except Exception as exc:
            errors.append(f"production starter label review CSV generation failed: {exc}")
            production_starter_label_review_csv["error"] = str(exc)
    capture_matrix_progress: dict[str, Any] = {
        "path": _rel(capture_matrix_csv_out) if capture_matrix_csv_out else None,
        "gate_passed": False,
        "row_count": 0,
        "ready_rows": 0,
        "target_labeled_examples": 0,
        "captured_examples": 0,
        "labeled_examples": 0,
        "missing_labeled_examples": 0,
        "unapproved_rows": 0,
        "unsafe_storage_rows": 0,
        "blockers": ["capture matrix CSV was not generated"],
    }
    if capture_matrix_csv_out is not None and capture_matrix_csv_out.exists():
        try:
            capture_matrix_progress = validate_capture_matrix_progress(
                capture_matrix_csv_out,
                manifest_path=capture_manifest,
                mode="pilot",
            )
            capture_matrix_progress["path"] = _rel(capture_matrix_csv_out)
            if capture_matrix_progress.get("ok") is not True:
                errors.extend(str(error) for error in capture_matrix_progress.get("errors") or [])
            warnings.extend(str(warning) for warning in capture_matrix_progress.get("warnings") or [])
            if capture_matrix_manifest.get("generated") and capture_matrix_csv_out is not None:
                capture_matrix_manifest = _write_capture_matrix_sidecar(
                    matrix_csv_path=capture_matrix_csv_out,
                    mode="pilot",
                    source_manifest=capture_manifest,
                    source_manifest_sha256=dataset_report.get("manifest_sha256"),
                    row_count=int(capture_matrix_csv.get("row_count") or capture_matrix_progress.get("row_count") or 0),
                    capture_deficit=capture_deficit,
                    progress=capture_matrix_progress,
                )
        except Exception as exc:
            errors.append(f"capture matrix progress validation failed: {exc}")
            capture_matrix_progress["error"] = str(exc)
    capture_matrix_sidecar_validation: dict[str, Any] = {
        "path": _rel(_matrix_sidecar_path(capture_matrix_csv_out)) if capture_matrix_csv_out else None,
        "checked": False,
        "valid": False,
    }
    if capture_matrix_csv_out is not None and capture_matrix_csv_out.exists():
        try:
            capture_matrix_sidecar_validation = validate_capture_matrix_sidecar(
                matrix_csv_path=capture_matrix_csv_out,
                capture_manifest_path=capture_manifest,
                mode="pilot",
                progress=capture_matrix_progress,
            )
            capture_matrix_sidecar_validation["checked"] = True
            capture_matrix_sidecar_validation["valid"] = True
        except Exception as exc:
            errors.append(f"capture matrix sidecar validation failed: {exc}")
            capture_matrix_sidecar_validation["checked"] = True
            capture_matrix_sidecar_validation["valid"] = False
            capture_matrix_sidecar_validation["error"] = str(exc)
    production_capture_matrix_progress: dict[str, Any] = {
        "path": _rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else None,
        "gate_passed": False,
        "row_count": 0,
        "ready_rows": 0,
        "target_labeled_examples": 0,
        "captured_examples": 0,
        "labeled_examples": 0,
        "missing_labeled_examples": 0,
        "unapproved_rows": 0,
        "unsafe_storage_rows": 0,
        "blockers": ["production capture matrix CSV was not generated"],
    }
    production_capture_matrix_sidecar_validation: dict[str, Any] = {
        "path": _rel(_matrix_sidecar_path(production_capture_matrix_csv_out))
        if production_capture_matrix_csv_out
        else None,
        "checked": False,
        "valid": False,
    }
    label_review_import_sidecar_validation: dict[str, Any] = {
        "path": _rel(capture_manifest.with_suffix(capture_manifest.suffix + ".label_review_import.json")),
        "checked": False,
        "valid": False,
    }
    production_training_dataset_provenance: dict[str, Any] = {
        "required": True,
        "checked": False,
        "source_manifest": None,
        "declared_source_manifest_sha256": None,
        "source_manifest_sha256": None,
        "permission": None,
        "permission_allowed": None,
        "missing_ppe_label_policy": None,
        "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "errors": [],
    }
    production_training_dataset_provenance_source = "production_dataset_provenance_preflight"
    production_training_plan_preflight: dict[str, Any] = {
        "checked": False,
        "ok": False,
        "status": None,
        "error": None,
        "inputs": {
            "data": _rel(production_training_dataset_yaml),
            "capture_manifest": _rel(capture_manifest),
            "capture_matrix_csv": _rel(production_capture_matrix_csv_out)
            if production_capture_matrix_csv_out
            else None,
        },
    }
    if production_capture_matrix_csv_out is not None and production_capture_matrix_csv_out.exists():
        try:
            production_capture_matrix_progress = validate_capture_matrix_progress(
                production_capture_matrix_csv_out,
                manifest_path=capture_manifest,
                mode="production",
            )
            production_capture_matrix_progress["path"] = _rel(production_capture_matrix_csv_out)
            if production_capture_matrix_progress.get("ok") is not True:
                errors.extend(str(error) for error in production_capture_matrix_progress.get("errors") or [])
            warnings.extend(str(warning) for warning in production_capture_matrix_progress.get("warnings") or [])
            if production_capture_matrix_manifest.get("generated") and production_capture_matrix_csv_out is not None:
                production_capture_matrix_manifest = _write_capture_matrix_sidecar(
                    matrix_csv_path=production_capture_matrix_csv_out,
                    mode="production",
                    source_manifest=capture_manifest,
                    source_manifest_sha256=production_dataset_report.get("manifest_sha256"),
                    row_count=int(
                        production_capture_matrix_csv.get("row_count")
                        or production_capture_matrix_progress.get("row_count")
                        or 0
                    ),
                    capture_deficit=production_capture_deficit,
                    progress=production_capture_matrix_progress,
                )
            production_capture_matrix_sidecar_validation = validate_capture_matrix_sidecar(
                matrix_csv_path=production_capture_matrix_csv_out,
                capture_manifest_path=capture_manifest,
                mode="production",
                progress=production_capture_matrix_progress,
            )
            production_capture_matrix_sidecar_validation["checked"] = True
            production_capture_matrix_sidecar_validation["valid"] = True
        except Exception as exc:
            errors.append(f"production capture matrix progress/sidecar validation failed: {exc}")
            production_capture_matrix_progress["error"] = str(exc)
            production_capture_matrix_sidecar_validation["checked"] = True
            production_capture_matrix_sidecar_validation["valid"] = False
            production_capture_matrix_sidecar_validation["error"] = str(exc)
    try:
        label_review_import_sidecar_validation = validate_label_review_import_sidecar(
            capture_manifest_path=capture_manifest,
        )
        label_review_import_sidecar_validation["path"] = _rel(
            Path(str(label_review_import_sidecar_validation.get("path") or ""))
        )
        label_review_import_sidecar_validation["checked"] = True
        label_review_import_sidecar_validation["valid"] = True
    except Exception as exc:
        label_review_import_sidecar_validation["checked"] = True
        label_review_import_sidecar_validation["valid"] = False
        label_review_import_sidecar_validation["error"] = str(exc)
    try:
        production_training_dataset_provenance = validate_dataset_provenance(
            dataset_doc=_load_yaml(production_training_dataset_yaml),
            data_path=production_training_dataset_yaml,
            capture_manifest_path=capture_manifest,
            require_capture_preflight=True,
        )
    except Exception as exc:
        production_training_dataset_provenance["errors"] = [str(exc)]
    if production_capture_matrix_csv_out is not None and production_capture_matrix_csv_out.exists():
        production_training_plan_preflight["checked"] = True
        try:
            preflight_plan = build_training_plan(
                data_path=production_training_dataset_yaml,
                model=training_model,
                device="auto",
                epochs=100,
                imgsz=640,
                batch=8,
                project=ROOT / "runs",
                name="apron_harness_closed_set",
                export_formats=["onnx"],
                capture_manifest_path=capture_manifest,
                capture_matrix_csv_path=production_capture_matrix_csv_out,
                seed_source_review_report=seed_source_review_report,
                seed_import_manifest=seed_import_manifest,
                capture_preflight_mode="production",
                require_capture_preflight=True,
            )
            production_training_plan_preflight.update(
                {
                    "ok": True,
                    "status": preflight_plan.get("status"),
                    "model": preflight_plan.get("model"),
                    "selected_device": preflight_plan.get("selected_device"),
                    "capture_preflight": preflight_plan.get("capture_preflight") or {},
                    "dataset_provenance": preflight_plan.get("dataset_provenance") or {},
                    "source_lineage": preflight_plan.get("source_lineage") or {},
                    "export_formats": preflight_plan.get("export_formats") or [],
                    "train_args": preflight_plan.get("train_args") or {},
                }
            )
            preflight_provenance = preflight_plan.get("dataset_provenance")
            if isinstance(preflight_provenance, dict) and preflight_provenance:
                production_training_dataset_provenance = dict(preflight_provenance)
                production_training_dataset_provenance_source = "production_training_plan_preflight"
        except Exception as exc:
            production_training_plan_preflight["ok"] = False
            production_training_plan_preflight["error"] = str(exc)
    missing_counts = {
        class_name: {
            "current": int(counts.get(class_name) or 0),
            "pilot_required": MIN_COUNTS["pilot"],
            "production_required": MIN_COUNTS["production"],
        }
        for class_name in REQUIRED_CLASSES.values()
        if int(counts.get(class_name) or 0) < MIN_COUNTS["pilot"]
    }
    production_missing_counts = {
        class_name: {
            "current": int(counts.get(class_name) or 0),
            "pilot_required": MIN_COUNTS["pilot"],
            "production_required": MIN_COUNTS["production"],
        }
        for class_name in REQUIRED_CLASSES.values()
        if int(counts.get(class_name) or 0) < MIN_COUNTS["production"]
    }

    training_capture_preflight = dict(training_plan.get("capture_preflight") or {})
    training_capture_preflight["mode"] = "production"
    training_capture_preflight["required"] = True
    if production_capture_matrix_csv_out is not None:
        training_capture_preflight = dict(production_capture_matrix_progress)
        training_capture_preflight["mode"] = "production"
        training_capture_preflight["required"] = True
        training_capture_preflight["checked"] = bool(production_capture_matrix_csv_out.exists())
        training_capture_preflight["capture_matrix_manifest"] = production_capture_matrix_sidecar_validation
        training_capture_preflight["label_review_import_manifest"] = label_review_import_sidecar_validation

    raw_training_dataset_provenance = training_plan.get("dataset_provenance")
    training_dataset_provenance = (
        dict(raw_training_dataset_provenance)
        if isinstance(raw_training_dataset_provenance, dict)
        else {}
    )
    raw_training_source_lineage = training_plan.get("source_lineage")
    training_source_lineage = (
        dict(raw_training_source_lineage)
        if isinstance(raw_training_source_lineage, dict)
        else {}
    )
    expected_training_source_manifest_sha256 = (
        production_capture_matrix_sidecar_validation.get("source_manifest_sha256")
        or production_dataset_report.get("manifest_sha256")
    )
    training_dataset_provenance_status = _training_dataset_provenance_status(
        production_training_dataset_provenance,
        expected_source_manifest_sha256=expected_training_source_manifest_sha256,
    )
    training_readiness = _training_readiness_status(
        training_plan=training_plan,
        production_training_plan_preflight=production_training_plan_preflight,
        training_capture_preflight=training_capture_preflight,
        production_missing_counts=production_missing_counts,
    )

    return {
        "ok": not errors,
        "capture_manifest": _rel(capture_manifest),
        "capture_manifest_sha256": dataset_report.get("manifest_sha256"),
        "training_dataset_yaml": _rel(training_dataset_yaml),
        "production_training_dataset_yaml": _rel(production_training_dataset_yaml),
        "training_dataset_yaml_handoff": training_dataset_yaml_handoff,
        "dataset_schema_ok": bool(dataset_report.get("ok")),
        "dataset_counts": counts,
        "missing_label_minimums": missing_counts,
        "production_missing_label_minimums": production_missing_counts,
        "required_labeled_images_per_class": {
            "pilot": MIN_COUNTS["pilot"],
            "production": MIN_COUNTS["production"],
        },
        "training_readiness": training_readiness,
        "training_dry_run_status": training_plan.get("status"),
        "training_model": training_plan.get("model"),
        "training_torch_status": training_plan.get("torch") or {},
        "training_capture_preflight": training_capture_preflight,
        "training_dataset_provenance": training_dataset_provenance,
        "production_training_dataset_provenance": production_training_dataset_provenance,
        "production_training_dataset_provenance_source": production_training_dataset_provenance_source,
        "training_dataset_provenance_status": training_dataset_provenance_status,
        "training_source_lineage": training_source_lineage,
        "production_training_plan_preflight": production_training_plan_preflight,
        "selected_device": selected_device or None,
        "export_formats": training_plan.get("export_formats") or [],
        "train_args": training_plan.get("train_args") or {},
        "capture_deficit": capture_deficit,
        "production_capture_deficit": production_capture_deficit,
        "capture_kickoff": capture_kickoff,
        "capture_work_order": capture_work_order,
        "capture_matrix_csv": capture_matrix_csv,
        "capture_matrix_manifest": capture_matrix_manifest,
        "capture_matrix_sidecar_validation": capture_matrix_sidecar_validation,
        "label_review_csv": label_review_csv,
        "starter_label_review_csv": starter_label_review_csv,
        "production_label_review_csv": production_label_review_csv,
        "production_starter_label_review_csv": production_starter_label_review_csv,
        "production_capture_matrix_csv": production_capture_matrix_csv,
        "production_capture_matrix_manifest": production_capture_matrix_manifest,
        "capture_matrix_progress": capture_matrix_progress,
        "production_capture_matrix_progress": production_capture_matrix_progress,
        "production_capture_matrix_sidecar_validation": production_capture_matrix_sidecar_validation,
        "label_review_import_sidecar_validation": label_review_import_sidecar_validation,
        "errors": errors,
        "warnings": warnings,
    }


def audit_apron_harness_readiness(
    *,
    model_packs_path: Path = DEFAULT_MODEL_PACKS,
    result_dir: Path = DEFAULT_RESULT_DIR,
    model_pack_evidence_report: Path | None = DEFAULT_MODEL_PACK_EVIDENCE_REPORT,
    capture_manifest: Path = DEFAULT_CAPTURE_MANIFEST,
    training_dataset_yaml: Path = DEFAULT_TRAINING_DATASET_YAML,
    training_model: str = DEFAULT_TRAINING_MODEL,
    seed_source_review_report: Path | None = DEFAULT_SEED_SOURCE_REVIEW,
    seed_source_review_bundle: Path | None = DEFAULT_SEED_SOURCE_REVIEW_BUNDLE,
    seed_import_manifest: Path | None = DEFAULT_SEED_IMPORT_MANIFEST,
    apron_promotion_report: Path | None = None,
    harness_promotion_report: Path | None = None,
    model_registry_report: Path | None = DEFAULT_MODEL_REGISTRY_REPORT,
    jetson_gate_report: Path | None = DEFAULT_JETSON_GATE_REPORT,
    capture_kickoff_out: Path | None = None,
    capture_work_order_out: Path | None = None,
    capture_matrix_csv_out: Path | None = None,
    production_capture_matrix_csv_out: Path | None = None,
    label_review_csv_out: Path | None = None,
    starter_label_review_csv_out: Path | None = None,
    production_label_review_csv_out: Path | None = None,
    production_starter_label_review_csv_out: Path | None = None,
    training_dataset_yaml_out: Path | None = None,
    training_dataset_root: str = DEFAULT_TRAINING_DATASET_ROOT,
    promotion_runbook_out: Path | None = None,
    candidate_runtime_runbook_out: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    packs_doc = _load_yaml(model_packs_path)
    factory_pack = (packs_doc.get("packs") or {}).get("factory_ppe_3cam") or {}
    production_training_plan = factory_pack.get("production_training_plan") or {}
    pretrained_shortcut_review = production_training_plan.get("pretrained_shortcut_review") or {}

    capability_reports: dict[str, Any] = {}
    for capability, scenario_info in CAPABILITY_SCENARIOS.items():
        capability_errors: list[str] = []
        aliases = set(scenario_info["visible_aliases"])
        active = _load_result(result_dir, scenario_info["active"], capability_errors)
        guard = _load_result(result_dir, scenario_info["guard"], capability_errors)
        suppression = _load_result(result_dir, scenario_info["suppression"], capability_errors)

        if active:
            _validate_active(f"{capability}.active", active, capability_errors)
        if guard:
            _validate_guard(f"{capability}.false_positive_guard", guard, aliases, capability_errors)
        if suppression:
            _validate_suppression(f"{capability}.suppression", suppression, capability, capability_errors)

        capability_reports[capability] = {
            "ok": not capability_errors,
            "errors": capability_errors,
            "scenarios": {
                "active": _scenario_summary(active, capability, aliases),
                "false_positive_guard": _scenario_summary(guard, capability, aliases),
                "suppression": _scenario_summary(suppression, capability, aliases),
            },
        }
        errors.extend(capability_errors)

    pack_status = str(factory_pack.get("status") or "")
    allowed_pack_statuses = {EXPECTED_PACK_STATUS, PRODUCTION_PACK_STATUS}
    if pack_status not in allowed_pack_statuses:
        errors.append(f"factory_ppe_3cam status must be one of {sorted(allowed_pack_statuses)}")

    sourcing = factory_pack.get("sourcing_status") or {}
    sourcing_result = str(sourcing.get("apron_harness_result") or "")
    allowed_sourcing_results = {EXPECTED_SOURCING_RESULT, PRODUCTION_SOURCING_RESULT}
    if sourcing_result not in allowed_sourcing_results:
        errors.append(
            "factory_ppe_3cam sourcing_status.apron_harness_result must be one of "
            f"{sorted(allowed_sourcing_results)}"
        )
    sourcing_candidate_count = len(sourcing.get("candidate_sources") or [])
    seed_source_review_ok, seed_source_review_gate_passed, seed_source_review_status, seed_source_review = (
        _seed_source_review_gate(
            seed_source_review_report,
            model_packs_path=model_packs_path,
            expected_candidate_count=sourcing_candidate_count,
        )
    )
    seed_import_required = sourcing_result != PRODUCTION_SOURCING_RESULT
    seed_import_ok, seed_import_gate_passed, seed_import_status, seed_import_review = (
        _seed_import_manifest_gate(
            seed_import_manifest,
            seed_source_review=seed_source_review,
            required=seed_import_required,
        )
    )
    seed_source_bundle_ok, seed_source_bundle_status, seed_source_bundle_review = (
        _seed_source_review_bundle_gate(
            seed_source_review_bundle,
            seed_source_review=seed_source_review,
            required=seed_import_required,
        )
    )
    seed_source_next_review_queue = _seed_source_next_review_queue(seed_source_review)
    seed_source_coverage_summary = _seed_source_coverage_summary(seed_source_review)
    seed_source_minimum_approval_path = _seed_source_minimum_approval_path(
        seed_source_coverage_summary,
        seed_source_next_review_queue,
    )
    minimum_seed_import_template_summary = _seed_import_template_summary(
        DEFAULT_MINIMUM_SEED_IMPORT_MANIFEST
    )
    minimum_seed_import_template_consistency = _minimum_seed_import_template_consistency(
        seed_source_minimum_approval_path,
        minimum_seed_import_template_summary,
    )

    handoff = factory_pack.get("runtime_handoff") or {}
    handoff_status = str(handoff.get("status") or "")
    allowed_handoff_statuses = {EXPECTED_HANDOFF_STATUS, PRODUCTION_HANDOFF_STATUS}
    if handoff_status not in allowed_handoff_statuses:
        errors.append(f"factory_ppe_3cam runtime_handoff.status must be one of {sorted(allowed_handoff_statuses)}")
    if handoff.get("planned_model_key") != PLANNED_MODEL_KEY:
        errors.append(f"runtime_handoff.planned_model_key must be {PLANNED_MODEL_KEY}")
    active_model_keys = set(factory_pack.get("model_keys") or [])
    registry_models = factory_pack.get("registry_models") or {}
    registered_model_keys = set(registry_models.keys()) if isinstance(registry_models, dict) else set()
    planned_model_active = PLANNED_MODEL_KEY in active_model_keys
    planned_model_registered = PLANNED_MODEL_KEY in registered_model_keys
    model_manager_definition = _model_manager_definition_status(
        str(handoff.get("planned_registry_path") or "")
    )

    apron_promotion_ok, apron_promotion_status = _promotion_gate_ok(
        apron_promotion_report,
        "apron_required",
    )
    harness_promotion_ok, harness_promotion_status = _promotion_gate_ok(
        harness_promotion_report,
        "harness_required",
    )
    model_registry_ok, model_registry_status = _model_registry_gate_ok(
        model_registry_report,
        str(handoff.get("planned_registry_path") or ""),
    )
    jetson_gate_ok, jetson_gate_status = _jetson_gate_ok(
        jetson_gate_report,
        "apron-harness-ppe.onnx",
    )
    jetson_full_gate = _jetson_gate_report_summary(jetson_gate_report, jetson_gate_status)
    jetson_template_handoff = _jetson_template_handoff_summary()
    model_pack_evidence_ok, model_pack_evidence_status = _model_pack_evidence_gate_ok(
        model_pack_evidence_report,
        model_packs_path=model_packs_path,
        result_dir=result_dir,
        expected_pack_status=pack_status,
    )
    candidate_yaml_templates = _candidate_yaml_template_status()
    candidate_runtime_evidence = _candidate_runtime_evidence_status()
    closed_set_handoff = _closed_set_handoff_status(
        capture_manifest,
        training_dataset_yaml,
        training_model,
        seed_source_review_report=seed_source_review_report,
        seed_import_manifest=seed_import_manifest,
        capture_kickoff_out=capture_kickoff_out,
        capture_work_order_out=capture_work_order_out,
        capture_matrix_csv_out=capture_matrix_csv_out,
        production_capture_matrix_csv_out=production_capture_matrix_csv_out,
        label_review_csv_out=label_review_csv_out,
        starter_label_review_csv_out=starter_label_review_csv_out,
        production_label_review_csv_out=production_label_review_csv_out,
        production_starter_label_review_csv_out=production_starter_label_review_csv_out,
        training_dataset_yaml_out=training_dataset_yaml_out,
        training_dataset_root=training_dataset_root,
    )
    errors.extend(str(error) for error in closed_set_handoff.get("errors") or [])
    warnings.extend(str(warning) for warning in closed_set_handoff.get("warnings") or [])

    expected_promotion_source_sha = (
        (closed_set_handoff.get("production_capture_matrix_sidecar_validation") or {}).get("source_manifest_sha256")
        or (closed_set_handoff.get("training_dataset_provenance") or {}).get("source_manifest_sha256")
    )
    if apron_promotion_ok:
        apron_handoff_ok, apron_handoff_status = _promotion_matches_handoff_source(
            apron_promotion_report,
            expected_source_manifest_sha256=expected_promotion_source_sha,
        )
        if not apron_handoff_ok:
            apron_promotion_ok = False
            apron_promotion_status = apron_handoff_status
    if harness_promotion_ok:
        harness_handoff_ok, harness_handoff_status = _promotion_matches_handoff_source(
            harness_promotion_report,
            expected_source_manifest_sha256=expected_promotion_source_sha,
        )
        if not harness_handoff_ok:
            harness_promotion_ok = False
            harness_promotion_status = harness_handoff_status
    if apron_promotion_ok and harness_promotion_ok:
        apron_export_sha = _promotion_selected_export_sha(apron_promotion_report)
        harness_export_sha = _promotion_selected_export_sha(harness_promotion_report)
        apron_candidate_report_sha = _promotion_candidate_report_sha(apron_promotion_report)
        harness_candidate_report_sha = _promotion_candidate_report_sha(harness_promotion_report)
        if apron_export_sha and harness_export_sha and apron_export_sha != harness_export_sha:
            mismatch_status = "candidate_selected_export_sha_mismatch_between_promotions"
            apron_promotion_ok = False
            harness_promotion_ok = False
            apron_promotion_status = mismatch_status
            harness_promotion_status = mismatch_status
        elif (
            apron_candidate_report_sha
            and harness_candidate_report_sha
            and apron_candidate_report_sha != harness_candidate_report_sha
        ):
            mismatch_status = "candidate_report_sha_mismatch_between_promotions"
            apron_promotion_ok = False
            harness_promotion_ok = False
            apron_promotion_status = mismatch_status
            harness_promotion_status = mismatch_status
        else:
            expected_export_sha = apron_export_sha or harness_export_sha
            expected_candidate_report_sha = apron_candidate_report_sha or harness_candidate_report_sha
            if model_registry_ok:
                model_registry_artifact_sha = _model_registry_artifact_sha(model_registry_report)
                if expected_export_sha and not model_registry_artifact_sha:
                    model_registry_ok = False
                    model_registry_status = (
                        "missing_model_artifact_sha256:"
                        f"{_rel(model_registry_report) if model_registry_report else 'missing'}"
                    )
                elif expected_export_sha and model_registry_artifact_sha != expected_export_sha:
                    model_registry_ok = False
                    model_registry_status = (
                        "model_artifact_sha_mismatch:"
                        f"{_rel(model_registry_report) if model_registry_report else 'missing'}"
                    )
                model_registry_candidate_report_sha = _model_registry_candidate_report_sha(model_registry_report)
                if expected_candidate_report_sha and not model_registry_candidate_report_sha:
                    model_registry_ok = False
                    model_registry_status = (
                        "missing_candidate_report_sha256:"
                        f"{_rel(model_registry_report) if model_registry_report else 'missing'}"
                    )
                elif (
                    expected_candidate_report_sha
                    and model_registry_candidate_report_sha != expected_candidate_report_sha
                ):
                    model_registry_ok = False
                    model_registry_status = (
                        "candidate_report_sha_mismatch:"
                        f"{_rel(model_registry_report) if model_registry_report else 'missing'}"
                    )
            if jetson_gate_ok:
                jetson_artifact_sha = _jetson_gate_artifact_sha(jetson_gate_report)
                if expected_export_sha and not jetson_artifact_sha:
                    jetson_gate_ok = False
                    jetson_gate_status = (
                        "missing_model_artifact_sha256:"
                        f"{_rel(jetson_gate_report) if jetson_gate_report else 'missing'}"
                    )
                elif expected_export_sha and jetson_artifact_sha != expected_export_sha:
                    jetson_gate_ok = False
                    jetson_gate_status = (
                        "model_artifact_sha_mismatch:"
                        f"{_rel(jetson_gate_report) if jetson_gate_report else 'missing'}"
                    )
                jetson_candidate_report_sha = _jetson_gate_candidate_report_sha(jetson_gate_report)
                if expected_candidate_report_sha and not jetson_candidate_report_sha:
                    jetson_gate_ok = False
                    jetson_gate_status = (
                        "missing_candidate_report_sha256:"
                        f"{_rel(jetson_gate_report) if jetson_gate_report else 'missing'}"
                    )
                elif (
                    expected_candidate_report_sha
                    and jetson_candidate_report_sha != expected_candidate_report_sha
                ):
                    jetson_gate_ok = False
                    jetson_gate_status = (
                        "candidate_report_sha_mismatch:"
                        f"{_rel(jetson_gate_report) if jetson_gate_report else 'missing'}"
                    )

    model_registry_handoff = _model_registry_handoff_summary(model_registry_report, model_registry_status)
    optional_gates_passed = (
        apron_promotion_ok
        and harness_promotion_ok
        and model_registry_ok
        and jetson_gate_ok
        and model_pack_evidence_ok
    )
    pilot_gate_passed = not any(report["errors"] for report in capability_reports.values())
    production_blockers: list[str] = []
    promotion_readiness_blockers: list[str] = []
    if not pilot_gate_passed:
        promotion_readiness_blockers.append("local_apron_harness_pilot_scenarios_not_ready")
    if not apron_promotion_ok:
        promotion_readiness_blockers.append("missing_or_failed_apron_closed_set_promotion_report")
    if not harness_promotion_ok:
        promotion_readiness_blockers.append("missing_or_failed_harness_closed_set_promotion_report")
    if not model_registry_ok:
        promotion_readiness_blockers.append(_model_registry_blocker(model_registry_status))
    if not jetson_gate_ok:
        promotion_readiness_blockers.append("missing_or_failed_factory_ppe_jetson_full_gate")
    if not model_pack_evidence_ok:
        promotion_readiness_blockers.append("missing_or_failed_model_pack_evidence_doctor")
    if candidate_yaml_templates.get("valid") is not True:
        promotion_readiness_blockers.append("closed_set_candidate_yaml_templates_invalid")
    if candidate_runtime_evidence.get("valid") is not True:
        promotion_readiness_blockers.append("closed_set_candidate_runtime_evidence_missing_or_invalid")
    if sourcing_result != PRODUCTION_SOURCING_RESULT and not seed_source_review_ok:
        promotion_readiness_blockers.append("missing_or_failed_apron_harness_seed_source_review")
    if sourcing_result != PRODUCTION_SOURCING_RESULT:
        promotion_readiness_blockers.append("closed_set_public_seed_sources_not_curated_or_approved")
    if sourcing_result != PRODUCTION_SOURCING_RESULT and not seed_source_review_gate_passed:
        promotion_readiness_blockers.append("apron_harness_seed_source_review_not_training_usable")
    if seed_import_required and not seed_source_bundle_ok:
        promotion_readiness_blockers.append("missing_or_failed_apron_harness_source_review_bundle")
    if seed_import_required and not seed_import_ok:
        promotion_readiness_blockers.append("missing_or_failed_apron_harness_seed_import_manifest")
    if seed_import_required and not seed_import_gate_passed:
        promotion_readiness_blockers.append("apron_harness_seed_import_manifest_not_training_usable")
    if (
        seed_import_required
        and minimum_seed_import_template_consistency.get("valid") is not True
    ):
        promotion_readiness_blockers.append("minimum_seed_import_template_not_consistent")
    if closed_set_handoff.get("missing_label_minimums"):
        promotion_readiness_blockers.append("closed_set_pilot_label_minimums_not_met")
    if closed_set_handoff.get("production_missing_label_minimums"):
        promotion_readiness_blockers.append("closed_set_production_label_minimums_not_met")
    capture_progress = closed_set_handoff.get("capture_matrix_progress") or {}
    if capture_progress.get("gate_passed") is not True:
        promotion_readiness_blockers.append("closed_set_capture_matrix_not_complete_or_approved")
    manifest_reconciliation = capture_progress.get("manifest_reconciliation") or {}
    if manifest_reconciliation.get("gate_passed") is False:
        promotion_readiness_blockers.append("closed_set_capture_matrix_manifest_counts_do_not_match")
    sidecar_validation = closed_set_handoff.get("capture_matrix_sidecar_validation") or {}
    if sidecar_validation.get("checked") is True and sidecar_validation.get("valid") is not True:
        promotion_readiness_blockers.append("closed_set_capture_matrix_sidecar_invalid")
    production_capture_progress = closed_set_handoff.get("production_capture_matrix_progress") or {}
    if production_capture_progress.get("gate_passed") is not True:
        promotion_readiness_blockers.append("closed_set_production_capture_matrix_not_complete_or_approved")
    production_manifest_reconciliation = production_capture_progress.get("manifest_reconciliation") or {}
    if production_manifest_reconciliation.get("gate_passed") is False:
        promotion_readiness_blockers.append("closed_set_production_capture_matrix_manifest_counts_do_not_match")
    production_sidecar_validation = closed_set_handoff.get("production_capture_matrix_sidecar_validation") or {}
    if production_sidecar_validation.get("checked") is True and production_sidecar_validation.get("valid") is not True:
        promotion_readiness_blockers.append("closed_set_production_capture_matrix_sidecar_invalid")
    label_review_import_sidecar_validation = closed_set_handoff.get("label_review_import_sidecar_validation") or {}
    if (
        label_review_import_sidecar_validation.get("checked") is True
        and label_review_import_sidecar_validation.get("valid") is not True
    ):
        promotion_readiness_blockers.append("closed_set_label_review_import_sidecar_invalid")
    training_preflight = closed_set_handoff.get("training_capture_preflight") or {}
    if training_preflight.get("checked") is not True:
        promotion_readiness_blockers.append("closed_set_training_capture_preflight_not_checked")
    elif training_preflight.get("gate_passed") is not True:
        promotion_readiness_blockers.append("closed_set_training_capture_preflight_not_passed")
    production_training_plan_preflight = closed_set_handoff.get("production_training_plan_preflight") or {}
    if (
        production_training_plan_preflight.get("checked") is True
        and production_training_plan_preflight.get("ok") is not True
    ):
        promotion_readiness_blockers.append("closed_set_production_training_plan_preflight_failed")
    training_dataset_provenance_status = closed_set_handoff.get("training_dataset_provenance_status") or {}
    if training_dataset_provenance_status.get("valid") is not True:
        promotion_readiness_blockers.extend(
            str(blocker)
            for blocker in training_dataset_provenance_status.get("blockers") or []
        )
    if closed_set_handoff.get("selected_device") not in {None, "mps"}:
        promotion_readiness_blockers.append("local_closed_set_training_dry_run_not_on_mps")

    production_blockers.extend(promotion_readiness_blockers)

    if promotion_readiness_blockers and pack_status == PRODUCTION_PACK_STATUS:
        errors.append("factory_ppe_3cam status is production before closed-set data/runtime/Jetson gates pass")
    if promotion_readiness_blockers and handoff_status == PRODUCTION_HANDOFF_STATUS:
        errors.append("factory_ppe_3cam runtime_handoff.status is registered before closed-set data/runtime/Jetson gates pass")
    if promotion_readiness_blockers and planned_model_active:
        errors.append(f"{PLANNED_MODEL_KEY} is active before closed-set data/runtime/Jetson gates pass")
    if promotion_readiness_blockers and planned_model_registered:
        errors.append(f"{PLANNED_MODEL_KEY} is registered before closed-set data/runtime/Jetson gates pass")

    if pack_status != PRODUCTION_PACK_STATUS:
        production_blockers.append("factory_ppe_pack_status_not_promoted")
    if handoff_status != PRODUCTION_HANDOFF_STATUS:
        production_blockers.append("closed_set_runtime_handoff_not_registered")
    if not planned_model_active:
        production_blockers.append("closed_set_model_key_not_registered")
    if not planned_model_registered:
        production_blockers.append("closed_set_registry_model_missing")
    if model_manager_definition.get("valid") is not True:
        production_blockers.append("closed_set_model_manager_definition_invalid")
    if model_manager_definition.get("artifact_exists") is not True:
        production_blockers.append("closed_set_model_artifact_missing")

    production_blocker_actions = _production_blocker_actions(production_blockers)
    next_actions = _prioritized_next_actions(production_blockers, production_blocker_actions)
    production_gate_passed = optional_gates_passed and not production_blockers and not errors
    sales_status = "ready_to_sell_production_compliance" if production_gate_passed else EXPECTED_PACK_STATUS
    report = {
        "ok": not errors,
        "generated_at": utc_now(),
        "inputs": {
            "model_packs": _rel(model_packs_path),
            "result_dir": _rel(result_dir),
            "model_pack_evidence_report": _rel(model_pack_evidence_report) if model_pack_evidence_report else None,
            "capture_manifest": _rel(capture_manifest),
            "training_dataset_yaml": _rel(training_dataset_yaml),
            "training_dataset_yaml_out": _rel(training_dataset_yaml_out) if training_dataset_yaml_out else None,
            "training_dataset_root": training_dataset_root,
            "training_model": training_model,
            "capture_kickoff_out": _rel(capture_kickoff_out) if capture_kickoff_out else None,
            "capture_work_order_out": _rel(capture_work_order_out) if capture_work_order_out else None,
            "capture_matrix_csv_out": _rel(capture_matrix_csv_out) if capture_matrix_csv_out else None,
            "production_capture_matrix_csv_out": _rel(production_capture_matrix_csv_out) if production_capture_matrix_csv_out else None,
            "label_review_csv_out": _rel(label_review_csv_out) if label_review_csv_out else None,
            "starter_label_review_csv_out": _rel(starter_label_review_csv_out)
            if starter_label_review_csv_out
            else None,
            "production_label_review_csv_out": _rel(production_label_review_csv_out)
            if production_label_review_csv_out
            else None,
            "production_starter_label_review_csv_out": _rel(production_starter_label_review_csv_out)
            if production_starter_label_review_csv_out
            else None,
            "seed_source_review_report": _rel(seed_source_review_report) if seed_source_review_report else None,
            "seed_source_review_bundle": _rel(seed_source_review_bundle) if seed_source_review_bundle else None,
            "seed_import_manifest": _rel(seed_import_manifest) if seed_import_manifest else None,
            "promotion_runbook_out": _rel(promotion_runbook_out) if promotion_runbook_out else None,
            "candidate_runtime_runbook_out": _rel(candidate_runtime_runbook_out)
            if candidate_runtime_runbook_out
            else None,
            "apron_promotion_report": _rel(apron_promotion_report) if apron_promotion_report else None,
            "harness_promotion_report": _rel(harness_promotion_report) if harness_promotion_report else None,
            "model_registry_report": _rel(model_registry_report) if model_registry_report else None,
            "jetson_gate_report": _rel(jetson_gate_report) if jetson_gate_report else None,
        },
        "pilot_gate_passed": pilot_gate_passed,
        "production_gate_passed": production_gate_passed,
        "sales_status": sales_status,
        "scope": READINESS_SCOPE,
        "factory_ppe_pack_status": pack_status,
        "runtime_handoff_status": handoff_status,
        "sourcing_status": sourcing_result,
        "sourcing_candidate_count": sourcing_candidate_count,
        "sourcing_candidate_sources": sourcing.get("candidate_sources") or [],
        "seed_source_review": seed_source_review,
        "seed_source_review_queue_summary": seed_source_review.get("review_queue_summary")
        if isinstance(seed_source_review.get("review_queue_summary"), dict)
        else {},
        "seed_source_next_review_queue": seed_source_next_review_queue,
        "seed_source_coverage_summary": seed_source_coverage_summary,
        "seed_source_minimum_approval_path": seed_source_minimum_approval_path,
        "seed_source_review_bundle": seed_source_bundle_review,
        "seed_source_review_bundle_ok": seed_source_bundle_ok,
        "seed_source_review_bundle_artifact_count": int(
            seed_source_bundle_review.get("checked_artifact_count")
            or seed_source_bundle_review.get("artifact_count")
            or 0
        )
        if seed_source_bundle_review
        else 0,
        "seed_source_review_gate_passed": seed_source_review_gate_passed,
        "seed_source_review_training_usable_count": int(seed_source_review.get("training_usable_count") or 0)
        if seed_source_review
        else 0,
        "seed_import_manifest_review": seed_import_review,
        "seed_import_export_preflight_summary": _seed_import_export_preflight_summary(seed_import_review),
        "seed_import_fill_contract_summary": _seed_import_fill_contract_summary(seed_import_manifest),
        "minimum_seed_import_manifest_template_summary": minimum_seed_import_template_summary,
        "minimum_seed_import_manifest_template_consistency": minimum_seed_import_template_consistency,
        "seed_import_manifest_gate_passed": seed_import_gate_passed,
        "seed_import_manifest_included_count": int(seed_import_review.get("included_count") or 0)
        if seed_import_review
        else 0,
        "seed_import_manifest_approved_count": int(seed_import_review.get("approved_count") or 0)
        if seed_import_review
        else 0,
        "pretrained_shortcut_review": pretrained_shortcut_review,
        "closed_set_model_manager_definition": model_manager_definition,
        "closed_set_model_key_active": planned_model_active,
        "closed_set_registry_model_registered": planned_model_registered,
        "model_registry_handoff": model_registry_handoff,
        "jetson_full_gate": jetson_full_gate,
        "jetson_template_handoff": jetson_template_handoff,
        "closed_set_candidate_yaml_templates": candidate_yaml_templates,
        "closed_set_candidate_runtime_evidence": candidate_runtime_evidence,
        "optional_gate_status": {
            "apron_promotion": apron_promotion_status,
            "harness_promotion": harness_promotion_status,
            "model_registry": model_registry_status,
            "jetson_gate": jetson_gate_status,
            "model_pack_evidence": model_pack_evidence_status,
            "seed_source_review": seed_source_review_status,
            "seed_source_review_bundle": seed_source_bundle_status,
            "seed_import_manifest": seed_import_status,
        },
        "closed_set_handoff": closed_set_handoff,
        "closed_set_capture_progress_summary": {
            "pilot": _capture_progress_summary(
                capture_progress,
                closed_set_handoff.get("capture_matrix_csv") or {},
                closed_set_handoff.get("capture_matrix_manifest") or {},
            ),
            "production": _capture_progress_summary(
                production_capture_progress,
                closed_set_handoff.get("production_capture_matrix_csv") or {},
                closed_set_handoff.get("production_capture_matrix_manifest") or {},
            ),
        },
        "seed_import_manifest_validation_summary": _seed_import_validation_summary(
            seed_import_review,
            seed_source_review,
        ),
        "production_capture_matrix_validation_summary": _production_capture_matrix_validation_summary(
            progress=production_capture_progress,
            counts=closed_set_handoff.get("dataset_counts") or {},
        ),
        "production_blockers": production_blockers,
        "production_blocker_count": len(production_blockers),
        "production_blocker_actions": production_blocker_actions,
        "next_actions": next_actions,
        "next_required_evidence": [
            "approved_apron_harness_seed_source_review_or_controlled_capture_only",
            "approved_apron_harness_seed_import_manifest_for_public_sources",
            "cleared_apron_harness_dataset_manifest",
            "trained_closed_set_nano_or_small_candidate",
            "candidate_doctor_report_ready_for_side_by_side_runtime_test",
            "apron_side_by_side_promotion_report",
            "harness_side_by_side_promotion_report",
            "factory_ppe_jetson_raw_benchmark",
            "factory_ppe_jetson_three_camera_soak",
        ],
        "capabilities": capability_reports,
        "errors": errors,
        "warnings": warnings,
    }
    candidate_runtime_runbook: dict[str, Any] = {
        "path": _rel(candidate_runtime_runbook_out) if candidate_runtime_runbook_out else None,
        "generated": False,
        "exists": False,
    }
    if candidate_runtime_runbook_out is not None:
        try:
            candidate_runtime_runbook_out.parent.mkdir(parents=True, exist_ok=True)
            candidate_runtime_runbook_out.write_text(
                _render_candidate_runtime_runbook(report),
                encoding="utf-8",
            )
            candidate_runtime_runbook["generated"] = True
            candidate_runtime_runbook["exists"] = candidate_runtime_runbook_out.exists()
            candidate_runtime_runbook["sha256"] = _sha256_file(candidate_runtime_runbook_out)
        except Exception as exc:
            errors.append(f"candidate runtime runbook generation failed: {exc}")
            candidate_runtime_runbook["error"] = str(exc)
            report["ok"] = False
    report["candidate_runtime_runbook"] = candidate_runtime_runbook
    report["production_gate_packet"] = _production_gate_packet(report)
    promotion_runbook: dict[str, Any] = {
        "path": _rel(promotion_runbook_out) if promotion_runbook_out else None,
        "generated": False,
        "exists": False,
    }
    if promotion_runbook_out is not None:
        try:
            promotion_runbook_out.parent.mkdir(parents=True, exist_ok=True)
            promotion_runbook_out.write_text(
                _render_promotion_runbook(
                    report=report,
                    result_dir=result_dir,
                    model_packs_path=model_packs_path,
                    capture_manifest=capture_manifest,
                    production_capture_matrix_csv_out=production_capture_matrix_csv_out,
                ),
                encoding="utf-8",
            )
            promotion_runbook["generated"] = True
            promotion_runbook["exists"] = promotion_runbook_out.exists()
            promotion_runbook["sha256"] = _sha256_file(promotion_runbook_out)
        except Exception as exc:
            errors.append(f"promotion runbook generation failed: {exc}")
            promotion_runbook["error"] = str(exc)
            report["ok"] = False
    report["promotion_runbook"] = promotion_runbook
    report["errors"] = errors
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit apron/harness pilot and production readiness gates.")
    parser.add_argument("--model-packs", default=str(DEFAULT_MODEL_PACKS), help="Path to model_packs.yaml")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing video_eval result JSON")
    parser.add_argument(
        "--model-pack-evidence-report",
        default=str(DEFAULT_MODEL_PACK_EVIDENCE_REPORT),
        help="Saved model_pack_evidence_doctor.json generated from the same model_packs/result-dir inputs",
    )
    parser.add_argument("--capture-manifest", default=str(DEFAULT_CAPTURE_MANIFEST), help="Apron/harness capture manifest to schema-check")
    parser.add_argument("--training-dataset-yaml", default=str(DEFAULT_TRAINING_DATASET_YAML), help="YOLO dataset YAML for dry-run training plan")
    parser.add_argument("--training-model", default=DEFAULT_TRAINING_MODEL, help="Allowed nano/small model for dry-run training plan")
    parser.add_argument(
        "--seed-source-review-report",
        default=str(DEFAULT_SEED_SOURCE_REVIEW),
        help="Saved apron_harness_seed_source_review.json generated from the same model_packs input",
    )
    parser.add_argument(
        "--seed-source-review-bundle",
        default=str(DEFAULT_SEED_SOURCE_REVIEW_BUNDLE),
        help="Non-approving source-review handoff bundle JSON whose artifact hashes must match the seed-source review",
    )
    parser.add_argument(
        "--seed-import-manifest",
        default=str(DEFAULT_SEED_IMPORT_MANIFEST),
        help="Filled apron/harness public seed import manifest; template is blocked until approved",
    )
    parser.add_argument("--apron-promotion-report", default="", help="Optional closed-set apron promotion report")
    parser.add_argument("--harness-promotion-report", default="", help="Optional closed-set harness promotion report")
    parser.add_argument(
        "--model-registry-report",
        default=str(DEFAULT_MODEL_REGISTRY_REPORT),
        help="Closed-set model registry report from scripts/apron_harness_model_registry_doctor.py",
    )
    parser.add_argument(
        "--jetson-gate-report",
        default=str(DEFAULT_JETSON_GATE_REPORT),
        help="factory_ppe_3cam Jetson gate report from scripts/jetson_benchmark_doctor.py",
    )
    parser.add_argument(
        "--capture-kickoff-out",
        default=str(DEFAULT_CAPTURE_KICKOFF),
        help="Markdown output path for the concise controlled-capture kickoff handoff",
    )
    parser.add_argument(
        "--capture-work-order-out",
        default=str(DEFAULT_CAPTURE_WORK_ORDER),
        help="Markdown output path for the generated apron/harness capture work order",
    )
    parser.add_argument(
        "--capture-matrix-csv-out",
        default=str(DEFAULT_CAPTURE_MATRIX_CSV),
        help="CSV output path for the generated apron/harness capture matrix checklist",
    )
    parser.add_argument(
        "--production-capture-matrix-csv-out",
        default=str(DEFAULT_PRODUCTION_CAPTURE_MATRIX_CSV),
        help="CSV output path for the generated production apron/harness capture matrix checklist",
    )
    parser.add_argument(
        "--label-review-csv-out",
        default=str(DEFAULT_LABEL_REVIEW_CSV),
        help="CSV output path for the generated apron/harness label review template",
    )
    parser.add_argument(
        "--starter-label-review-csv-out",
        default=str(DEFAULT_STARTER_LABEL_REVIEW_CSV),
        help="CSV output path for the immediate starter label-review template",
    )
    parser.add_argument(
        "--production-label-review-csv-out",
        default=str(DEFAULT_PRODUCTION_LABEL_REVIEW_CSV),
        help="CSV output path for the generated production apron/harness label review template",
    )
    parser.add_argument(
        "--production-starter-label-review-csv-out",
        default=str(DEFAULT_PRODUCTION_STARTER_LABEL_REVIEW_CSV),
        help="CSV output path for the immediate production starter label-review template",
    )
    parser.add_argument(
        "--training-dataset-yaml-out",
        default=str(DEFAULT_TRAINING_DATASET_YAML_OUT),
        help="Generated YOLO dataset YAML with capture-manifest provenance for production training preflight",
    )
    parser.add_argument(
        "--training-dataset-root",
        default=DEFAULT_TRAINING_DATASET_ROOT,
        help="Dataset root written into the generated YOLO dataset YAML",
    )
    parser.add_argument(
        "--promotion-runbook-out",
        default=str(DEFAULT_PROMOTION_RUNBOOK),
        help="Markdown output path for the generated closed-set promotion runbook",
    )
    parser.add_argument(
        "--candidate-runtime-runbook-out",
        default=str(DEFAULT_CANDIDATE_RUNTIME_RUNBOOK),
        help="Markdown output path for the closed-set candidate runtime scenario runbook",
    )
    parser.add_argument(
        "--seed-import-validation-summary-out",
        default=str(DEFAULT_SEED_IMPORT_VALIDATION_SUMMARY),
        help="Concise JSON output path for the blocked/ready seed-import validation handoff",
    )
    parser.add_argument(
        "--production-capture-matrix-validation-summary-out",
        default=str(DEFAULT_PRODUCTION_CAPTURE_MATRIX_VALIDATION_SUMMARY),
        help="Concise JSON output path for the production capture-matrix validation handoff",
    )
    parser.add_argument(
        "--production-gate-packet-out",
        default=str(DEFAULT_PRODUCTION_GATE_PACKET),
        help="Machine-readable JSON handoff for the apron/harness production gate and next actions",
    )
    parser.add_argument(
        "--validate-production-gate-packet",
        default="",
        help="Read-only validation of a production gate packet against a saved readiness report",
    )
    parser.add_argument(
        "--readiness-report",
        default="",
        help="Saved readiness report used with --validate-production-gate-packet",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_production_gate_packet:
        readiness_report = (
            Path(args.readiness_report)
            if args.readiness_report
            else Path(args.out)
            if args.out
            else DEFAULT_RESULT_DIR / "apron_harness_readiness_doctor.json"
        )
        validation = validate_production_gate_packet(
            Path(args.validate_production_gate_packet),
            readiness_report_path=readiness_report,
        )
        if args.json:
            print(json.dumps(validation, indent=2))
        else:
            print(
                "PRODUCTION_GATE_PACKET: "
                f"ok={validation['ok']} "
                f"blockers={validation.get('production_blocker_count')} "
                f"next_actions={validation.get('next_action_count')} "
                f"minimum_sources={validation.get('minimum_review_source_count')} "
                f"artifacts={validation.get('minimum_review_artifact_sha_match_count')}/"
                f"{validation.get('minimum_review_artifact_count')} "
                f"source_steps={validation.get('source_review_execution_step_count')} "
                f"source_required={validation.get('source_review_execution_required_source_count')} "
                f"source_review_runners={validation.get('source_review_runner_artifact_count')} "
                f"capture_missing={validation.get('controlled_capture_missing_labeled_examples')} "
                f"capture_batches={validation.get('controlled_capture_next_batch_count')} "
                f"starter_rows={validation.get('controlled_capture_starter_row_count')} "
                f"label_templates={validation.get('controlled_capture_label_template_count')} "
                f"starter_commands={validation.get('controlled_capture_starter_command_count')} "
                f"capture_runners={validation.get('controlled_capture_runner_artifact_count')} "
                f"starter_criteria={validation.get('controlled_capture_starter_success_criterion_count')} "
                f"operator_handoffs={validation.get('controlled_capture_operator_handoff_artifact_count')} "
                f"starter_steps={validation.get('controlled_capture_starter_execution_step_count')} "
                f"starter_command_matches={validation.get('controlled_capture_starter_execution_command_match_count')} "
                f"post_capture_checks={validation.get('controlled_capture_post_capture_check_count')} "
                f"candidate_runtime_steps={validation.get('candidate_runtime_execution_step_count')} "
                f"candidate_runtime_scenarios={validation.get('candidate_runtime_execution_scenario_count')} "
                f"candidate_runtime_runbooks={validation.get('candidate_runtime_runbook_artifact_count')} "
                f"candidate_runtime_runners={validation.get('candidate_runtime_runner_artifact_count')} "
                f"candidate_training_steps={validation.get('candidate_training_execution_step_count')} "
                f"candidate_training_runners={validation.get('candidate_training_runner_artifact_count')} "
                f"jetson_gate_steps={validation.get('jetson_gate_execution_step_count')} "
                f"jetson_gate_runners={validation.get('jetson_gate_runner_artifact_count')}"
            )
            for error in validation["errors"]:
                print(f"ERROR: {error}")
        return 0 if validation["ok"] else 1

    report = audit_apron_harness_readiness(
        model_packs_path=Path(args.model_packs),
        result_dir=Path(args.result_dir),
        model_pack_evidence_report=(
            Path(args.model_pack_evidence_report) if args.model_pack_evidence_report else None
        ),
        capture_manifest=Path(args.capture_manifest),
        training_dataset_yaml=Path(args.training_dataset_yaml),
        training_model=args.training_model,
        seed_source_review_report=Path(args.seed_source_review_report) if args.seed_source_review_report else None,
        seed_source_review_bundle=Path(args.seed_source_review_bundle) if args.seed_source_review_bundle else None,
        seed_import_manifest=Path(args.seed_import_manifest) if args.seed_import_manifest else None,
        apron_promotion_report=Path(args.apron_promotion_report) if args.apron_promotion_report else None,
        harness_promotion_report=Path(args.harness_promotion_report) if args.harness_promotion_report else None,
        model_registry_report=Path(args.model_registry_report) if args.model_registry_report else None,
        jetson_gate_report=Path(args.jetson_gate_report) if args.jetson_gate_report else None,
        capture_kickoff_out=Path(args.capture_kickoff_out) if args.capture_kickoff_out else None,
        capture_work_order_out=Path(args.capture_work_order_out) if args.capture_work_order_out else None,
        capture_matrix_csv_out=Path(args.capture_matrix_csv_out) if args.capture_matrix_csv_out else None,
        production_capture_matrix_csv_out=(
            Path(args.production_capture_matrix_csv_out) if args.production_capture_matrix_csv_out else None
        ),
        label_review_csv_out=Path(args.label_review_csv_out) if args.label_review_csv_out else None,
        starter_label_review_csv_out=(
            Path(args.starter_label_review_csv_out) if args.starter_label_review_csv_out else None
        ),
        production_label_review_csv_out=(
            Path(args.production_label_review_csv_out) if args.production_label_review_csv_out else None
        ),
        production_starter_label_review_csv_out=(
            Path(args.production_starter_label_review_csv_out)
            if args.production_starter_label_review_csv_out
            else None
        ),
        training_dataset_yaml_out=(
            Path(args.training_dataset_yaml_out) if args.training_dataset_yaml_out else None
        ),
        training_dataset_root=args.training_dataset_root,
        promotion_runbook_out=Path(args.promotion_runbook_out) if args.promotion_runbook_out else None,
        candidate_runtime_runbook_out=Path(args.candidate_runtime_runbook_out)
        if args.candidate_runtime_runbook_out
        else None,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    seed_import_summary_path: Path | None = (
        Path(args.seed_import_validation_summary_out)
        if args.seed_import_validation_summary_out
        else None
    )
    if seed_import_summary_path is not None:
        seed_import_summary_path.parent.mkdir(parents=True, exist_ok=True)
        seed_import_summary_path.write_text(
            json.dumps(report["seed_import_manifest_validation_summary"], indent=2) + "\n",
            encoding="utf-8",
        )
    production_matrix_summary_path: Path | None = (
        Path(args.production_capture_matrix_validation_summary_out)
        if args.production_capture_matrix_validation_summary_out
        else None
    )
    if production_matrix_summary_path is not None:
        production_matrix_summary_path.parent.mkdir(parents=True, exist_ok=True)
        production_matrix_summary_path.write_text(
            json.dumps(report["production_capture_matrix_validation_summary"], indent=2) + "\n",
            encoding="utf-8",
        )
    production_gate_packet_path: Path | None = (
        Path(args.production_gate_packet_out)
        if args.production_gate_packet_out
        else None
    )
    if production_gate_packet_path is not None:
        production_gate_packet_path.parent.mkdir(parents=True, exist_ok=True)
        production_gate_packet_path.write_text(
            json.dumps(report["production_gate_packet"], indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(
            f"{status}: pilot_gate={report['pilot_gate_passed']} "
            f"production_gate={report['production_gate_passed']} "
            f"sales_status={report['sales_status']}"
        )
        if args.out:
            print(f"wrote: {args.out}")
        if seed_import_summary_path is not None:
            print(f"wrote: {seed_import_summary_path}")
        if production_matrix_summary_path is not None:
            print(f"wrote: {production_matrix_summary_path}")
        if production_gate_packet_path is not None:
            print(f"wrote: {production_gate_packet_path}")
        kickoff = (report.get("closed_set_handoff") or {}).get("capture_kickoff") or {}
        if kickoff.get("generated"):
            print(f"wrote: {kickoff.get('path')}")
        work_order = (report.get("closed_set_handoff") or {}).get("capture_work_order") or {}
        if work_order.get("generated"):
            print(f"wrote: {work_order.get('path')}")
        matrix_csv = (report.get("closed_set_handoff") or {}).get("capture_matrix_csv") or {}
        if matrix_csv.get("generated"):
            print(f"wrote: {matrix_csv.get('path')}")
        matrix_manifest = (report.get("closed_set_handoff") or {}).get("capture_matrix_manifest") or {}
        if matrix_manifest.get("generated"):
            print(f"wrote: {matrix_manifest.get('path')}")
        label_review_csv = (report.get("closed_set_handoff") or {}).get("label_review_csv") or {}
        if label_review_csv.get("generated"):
            print(f"wrote: {label_review_csv.get('path')}")
        starter_label_review_csv = (
            (report.get("closed_set_handoff") or {}).get("starter_label_review_csv") or {}
        )
        if starter_label_review_csv.get("generated"):
            print(f"wrote: {starter_label_review_csv.get('path')}")
        production_matrix_csv = (report.get("closed_set_handoff") or {}).get("production_capture_matrix_csv") or {}
        if production_matrix_csv.get("generated"):
            print(f"wrote: {production_matrix_csv.get('path')}")
        production_matrix_manifest = (
            (report.get("closed_set_handoff") or {}).get("production_capture_matrix_manifest") or {}
        )
        if production_matrix_manifest.get("generated"):
            print(f"wrote: {production_matrix_manifest.get('path')}")
        production_label_review_csv = (
            (report.get("closed_set_handoff") or {}).get("production_label_review_csv") or {}
        )
        if production_label_review_csv.get("generated"):
            print(f"wrote: {production_label_review_csv.get('path')}")
        production_starter_label_review_csv = (
            (report.get("closed_set_handoff") or {}).get("production_starter_label_review_csv") or {}
        )
        if production_starter_label_review_csv.get("generated"):
            print(f"wrote: {production_starter_label_review_csv.get('path')}")
        training_dataset = (report.get("closed_set_handoff") or {}).get("training_dataset_yaml_handoff") or {}
        if training_dataset.get("generated"):
            print(f"wrote: {training_dataset.get('path')}")
        promotion_runbook = report.get("promotion_runbook") or {}
        if promotion_runbook.get("generated"):
            print(f"wrote: {promotion_runbook.get('path')}")
        candidate_runtime_runbook = report.get("candidate_runtime_runbook") or {}
        if candidate_runtime_runbook.get("generated"):
            print(f"wrote: {candidate_runtime_runbook.get('path')}")
        matrix_progress = (report.get("closed_set_handoff") or {}).get("capture_matrix_progress") or {}
        if matrix_progress.get("path"):
            print(
                "capture_progress: "
                f"gate={'pass' if matrix_progress.get('gate_passed') else 'blocked'} "
                f"ready_rows={matrix_progress.get('ready_rows', 0)}/"
                f"{matrix_progress.get('row_count', 0)} "
                f"missing_labeled_examples={matrix_progress.get('missing_labeled_examples', 0)}"
            )
        production_matrix_progress = (
            (report.get("closed_set_handoff") or {}).get("production_capture_matrix_progress") or {}
        )
        if production_matrix_progress.get("path"):
            print(
                "production_capture_progress: "
                f"gate={'pass' if production_matrix_progress.get('gate_passed') else 'blocked'} "
                f"ready_rows={production_matrix_progress.get('ready_rows', 0)}/"
                f"{production_matrix_progress.get('row_count', 0)} "
                f"missing_labeled_examples={production_matrix_progress.get('missing_labeled_examples', 0)}"
            )
        for blocker in report["production_blockers"]:
            print(f"BLOCKED: {blocker}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
