#!/usr/bin/env python3
"""Validate apron/harness PPE capture packs before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CLASSES = {
    0: "person",
    1: "apron",
    2: "safety_harness",
    3: "safety_lanyard",
}
ALLOWED_PERMISSIONS = {
    "internal_cleared",
    "customer_cleared",
    "controlled_capture_cleared",
    "commercial_dataset_approved",
    "written_commercial_permission",
}
BLOCKED_PERMISSIONS = {
    "unknown",
    "customer_private_unapproved",
    "academic_only",
    "research_only",
    "non_commercial",
}
REQUIRED_CAMERA_ANGLES = {"front", "side", "elevated_cctv"}
REQUIRED_DISTANCE_BANDS = {"close", "medium", "wide_surveillance"}
REQUIRED_LIGHTING = {"indoor_bright", "dim_indoor", "backlit", "glare"}
REQUIRED_MOTION_BLUR = {"low", "medium_or_high"}
REQUIRED_APRON_POSITIVES = {
    "denim_apron",
    "work_apron",
    "kitchen_or_food_service_apron",
    "protective_industrial_apron",
    "partial_side_apron",
}
REQUIRED_APRON_HARD_NEGATIVES = {
    "safety_vest",
    "jacket",
    "lab_coat",
    "shirt_color_block",
    "tool_belt",
    "loose_cloth_or_scarf",
}
REQUIRED_HARNESS_POSITIVES = {
    "full_body_safety_harness",
    "fall_arrest_harness",
    "visible_lanyard_or_tether",
    "harness_over_safety_vest",
    "partially_hidden_harness",
}
REQUIRED_HARNESS_HARD_NEGATIVES = {
    "backpack_straps",
    "tool_belts",
    "seat_belts",
    "ropes_cables_slings_or_hoses",
    "reflective_vest_stripes",
}
COVERAGE_REQUIREMENTS = {
    "camera_angles": REQUIRED_CAMERA_ANGLES,
    "distance_bands": REQUIRED_DISTANCE_BANDS,
    "lighting": REQUIRED_LIGHTING,
    "motion_blur": REQUIRED_MOTION_BLUR,
    "apron_positive_variants": REQUIRED_APRON_POSITIVES,
    "apron_hard_negative_tags": REQUIRED_APRON_HARD_NEGATIVES,
    "harness_positive_variants": REQUIRED_HARNESS_POSITIVES,
    "harness_hard_negative_tags": REQUIRED_HARNESS_HARD_NEGATIVES,
}
MIN_COUNTS = {
    "pilot": 300,
    "production": 1000,
}
APPROVED_LABEL_REVIEW_STATUS = "approved"
TAXONOMY_VERSION = "apron_harness_v1"
YOLO_LABEL_FORMAT = "ultralytics_yolo_txt_normalized_xywh"
LABEL_SPLITS = {"train", "val", "test"}
REQUIRED_STRICT_LABEL_SPLITS = {"train", "val"}
REQUIRED_PRODUCTION_LABEL_SPLITS = {"train", "val", "test"}
PRODUCTION_HOLDOUT_SPLIT_MIN_FRACTION = 0.10
HEX_DIGITS = set("0123456789abcdefABCDEF")
DEFAULT_DATASET_SPLITS = {
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apron_harness_seed_source_doctor import validate_import_manifest  # noqa: E402

DEFAULT_SEED_SOURCE_REVIEW = ROOT / "qa" / "video_eval" / "results" / "apron_harness_seed_source_review.json"
DEFAULT_SEED_IMPORT_MANIFEST = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_seed_import_manifest.template.yaml"
REVIEW_EVIDENCE_KIND = "apron_harness_seed_source_review_evidence"
PUBLIC_SEED_SOURCES = {
    "public_seed_source",
    "public_dataset",
    "roboflow_universe",
    "open_dataset",
}
REQUIRED_SEED_IMPORT_REVIEW_FIELDS = {
    "class_mapping",
    "dataset_card_provenance",
    "export_terms",
    "hard_negative_coverage",
    "license_terms",
    "manifest_import_plan",
    "person_box_coverage",
    "privacy_and_identity_risk",
    "train_val_test_split",
}
REQUIRED_SEED_SOURCE_REVIEW_FIELDS = REQUIRED_SEED_IMPORT_REVIEW_FIELDS
ALLOWED_SEED_IMPORT_EXPORT_FORMATS = {"yolo"}
ALLOWED_RAW_EXPORT_REF_SCHEMES = {"az", "gs", "hf", "https", "oci", "roboflow", "s3"}
REQUIRED_SEED_IMPORT_COUNT_CLASSES = {
    "apron_required": {"person", "apron"},
    "harness_required": {"person", "safety_harness", "safety_lanyard"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def _normalize_classes(raw_classes: Any) -> dict[int, str]:
    if not isinstance(raw_classes, dict):
        return {}
    normalized: dict[int, str] = {}
    for key, value in raw_classes.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        normalized[class_id] = str(value)
    return normalized


def _count_for_class(counts: dict[str, Any], class_name: str) -> int:
    try:
        return int(counts.get(class_name) or 0)
    except (TypeError, ValueError):
        return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_review_fingerprint(report: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "generated_at",
            "import_manifest_review",
            "review_checklist_apply",
            "review_evidence_templates",
            "review_packets",
        }
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _source_recheck_lineage(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    source_recheck = report.get("source_recheck")
    return source_recheck if isinstance(source_recheck, dict) else {}


def _coverage_values(doc: dict[str, Any], field: str) -> set[str]:
    coverage = doc.get("coverage") or {}
    if not isinstance(coverage, dict):
        return set()
    return _as_set(coverage.get(field))


def _clip_backed_coverage_values(clips: list[dict[str, Any]]) -> dict[str, set[str]]:
    backed = {field: set() for field in COVERAGE_REQUIREMENTS}
    for clip in clips:
        if clip.get("camera_angle"):
            backed["camera_angles"].add(str(clip["camera_angle"]))
        if clip.get("distance_band"):
            backed["distance_bands"].add(str(clip["distance_band"]))
        if clip.get("lighting"):
            backed["lighting"].add(str(clip["lighting"]))
        if clip.get("motion_blur"):
            backed["motion_blur"].add(str(clip["motion_blur"]))

        apron_positive_values = (
            _as_set(clip.get("positive_variant_tags"))
            | _as_set(clip.get("ppe_variant_tags"))
            | _as_set(clip.get("apron_positive_variants"))
        )
        harness_positive_values = (
            _as_set(clip.get("positive_variant_tags"))
            | _as_set(clip.get("ppe_variant_tags"))
            | _as_set(clip.get("harness_positive_variants"))
        )
        hard_negative_values = _as_set(clip.get("hard_negative_tags"))
        backed["apron_positive_variants"].update(apron_positive_values)
        backed["harness_positive_variants"].update(harness_positive_values)
        backed["apron_hard_negative_tags"].update(
            hard_negative_values | _as_set(clip.get("apron_hard_negative_tags"))
        )
        backed["harness_hard_negative_tags"].update(
            hard_negative_values | _as_set(clip.get("harness_hard_negative_tags"))
        )
    return backed


def _coverage_targets(missing_coverage: dict[str, list[str]], field: str) -> list[str]:
    missing = missing_coverage.get(field) or []
    if missing:
        return missing
    return sorted(COVERAGE_REQUIREMENTS[field])


def _minimum_split_class_counts(mode: str) -> dict[str, int]:
    if mode != "production":
        return {}
    minimum = math.ceil(MIN_COUNTS["production"] * PRODUCTION_HOLDOUT_SPLIT_MIN_FRACTION)
    return {
        "val": minimum,
        "test": minimum,
    }


def _examples_per_item(total: int, item_count: int, *, default_minimum: int = 10, fraction: float = 1.0) -> int:
    if item_count <= 0:
        return 0
    if total <= 0:
        return default_minimum
    return max(default_minimum, math.ceil((total * fraction) / item_count))


def _capture_matrix(
    *,
    batch_id: str,
    target_capability: str,
    minimum_labeled_images: int,
    required_label_classes: list[str],
    camera_angles: list[str],
    distance_bands: list[str],
    lighting: list[str],
    motion_blur: list[str],
    positive_variants: list[str],
    hard_negative_tags: list[str],
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    positive_examples = _examples_per_item(minimum_labeled_images, len(positive_variants))
    hard_negative_examples = _examples_per_item(
        minimum_labeled_images,
        len(hard_negative_tags),
        default_minimum=10,
        fraction=0.2,
    )
    for variant in positive_variants:
        matrix.append(
            {
                "row_id": f"{batch_id}.positive.{variant}",
                "target_capability": target_capability,
                "capture_type": "positive_variant",
                "variant_or_tag": variant,
                "recommended_examples": positive_examples,
                "required_label_classes": required_label_classes,
                "camera_angles": camera_angles,
                "distance_bands": distance_bands,
                "lighting": lighting,
                "motion_blur": motion_blur,
                "notes": "Visible target PPE must be boxed together with every person in frame.",
            }
        )
    for tag in hard_negative_tags:
        matrix.append(
            {
                "row_id": f"{batch_id}.hard_negative.{tag}",
                "target_capability": target_capability,
                "capture_type": "hard_negative",
                "variant_or_tag": tag,
                "recommended_examples": hard_negative_examples,
                "required_label_classes": ["person"],
                "camera_angles": camera_angles,
                "distance_bands": distance_bands,
                "lighting": lighting,
                "motion_blur": motion_blur,
                "notes": "Confusing item should be present, but no target PPE box should be added unless it is truly visible.",
            }
        )
    return matrix


def _recommended_label_review_rows(batches: list[dict[str, Any]]) -> int:
    total = 0
    for batch in batches:
        matrix = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
        for row in matrix:
            try:
                total += int(str(row.get("recommended_examples") or "0"))
            except ValueError:
                continue
    return total


def build_capture_deficit(doc: dict[str, Any], mode: str = "pilot") -> dict[str, Any]:
    """Return the concrete capture work still needed before training."""
    minimum = MIN_COUNTS[mode]
    counts_doc = (doc.get("counts") or {}).get("labeled_images_per_class") or {}
    count_summary = {
        class_name: _count_for_class(counts_doc, class_name)
        for class_name in REQUIRED_CLASSES.values()
    }
    missing_labels = {
        class_name: {
            "current": count,
            "required": minimum,
            "missing": max(0, minimum - count),
        }
        for class_name, count in count_summary.items()
        if count < minimum
    }
    missing_coverage = {
        field: sorted(required - _coverage_values(doc, field))
        for field, required in COVERAGE_REQUIREMENTS.items()
    }
    coverage_deficit_count = sum(len(values) for values in missing_coverage.values())

    batches: list[dict[str, Any]] = []
    apron_minimum = max(
        missing_labels.get("person", {}).get("missing", 0),
        missing_labels.get("apron", {}).get("missing", 0),
    )
    apron_needs_capture = apron_minimum > 0 or any(
        missing_coverage.get(field)
        for field in [
            "camera_angles",
            "distance_bands",
            "lighting",
            "motion_blur",
            "apron_positive_variants",
            "apron_hard_negative_tags",
        ]
    )
    if apron_needs_capture:
        batch_id = "apron_required_closed_set_capture"
        target_capability = "apron_required"
        required_label_classes = ["person", "apron"]
        camera_angles = _coverage_targets(missing_coverage, "camera_angles")
        distance_bands = _coverage_targets(missing_coverage, "distance_bands")
        lighting = _coverage_targets(missing_coverage, "lighting")
        motion_blur = _coverage_targets(missing_coverage, "motion_blur")
        positive_variants = _coverage_targets(missing_coverage, "apron_positive_variants")
        hard_negative_tags = _coverage_targets(missing_coverage, "apron_hard_negative_tags")
        batches.append(
            {
                "batch_id": batch_id,
                "target_capability": target_capability,
                "purpose": "Capture visible apron positives and hard negatives for the closed-set PPE model.",
                "minimum_labeled_images": apron_minimum,
                "required_label_classes": required_label_classes,
                "camera_angles": camera_angles,
                "distance_bands": distance_bands,
                "lighting": lighting,
                "motion_blur": motion_blur,
                "positive_variants": positive_variants,
                "hard_negative_tags": hard_negative_tags,
                "capture_matrix": _capture_matrix(
                    batch_id=batch_id,
                    target_capability=target_capability,
                    minimum_labeled_images=apron_minimum,
                    required_label_classes=required_label_classes,
                    camera_angles=camera_angles,
                    distance_bands=distance_bands,
                    lighting=lighting,
                    motion_blur=motion_blur,
                    positive_variants=positive_variants,
                    hard_negative_tags=hard_negative_tags,
                ),
                "acceptance_note": "Every clip must have cleared commercial permission and YOLO labels for person plus visible apron where present.",
            }
        )

    harness_minimum = max(
        missing_labels.get("person", {}).get("missing", 0),
        missing_labels.get("safety_harness", {}).get("missing", 0),
        missing_labels.get("safety_lanyard", {}).get("missing", 0),
    )
    harness_needs_capture = harness_minimum > 0 or any(
        missing_coverage.get(field)
        for field in [
            "camera_angles",
            "distance_bands",
            "lighting",
            "motion_blur",
            "harness_positive_variants",
            "harness_hard_negative_tags",
        ]
    )
    if harness_needs_capture:
        batch_id = "harness_required_closed_set_capture"
        target_capability = "harness_required"
        required_label_classes = ["person", "safety_harness", "safety_lanyard"]
        camera_angles = _coverage_targets(missing_coverage, "camera_angles")
        distance_bands = _coverage_targets(missing_coverage, "distance_bands")
        lighting = _coverage_targets(missing_coverage, "lighting")
        motion_blur = _coverage_targets(missing_coverage, "motion_blur")
        positive_variants = _coverage_targets(missing_coverage, "harness_positive_variants")
        hard_negative_tags = _coverage_targets(missing_coverage, "harness_hard_negative_tags")
        batches.append(
            {
                "batch_id": batch_id,
                "target_capability": target_capability,
                "purpose": "Capture visible safety harness/lanyard positives and hard negatives for the closed-set PPE model.",
                "minimum_labeled_images": harness_minimum,
                "required_label_classes": required_label_classes,
                "camera_angles": camera_angles,
                "distance_bands": distance_bands,
                "lighting": lighting,
                "motion_blur": motion_blur,
                "positive_variants": positive_variants,
                "hard_negative_tags": hard_negative_tags,
                "capture_matrix": _capture_matrix(
                    batch_id=batch_id,
                    target_capability=target_capability,
                    minimum_labeled_images=harness_minimum,
                    required_label_classes=required_label_classes,
                    camera_angles=camera_angles,
                    distance_bands=distance_bands,
                    lighting=lighting,
                    motion_blur=motion_blur,
                    positive_variants=positive_variants,
                    hard_negative_tags=hard_negative_tags,
                ),
                "acceptance_note": "Every clip must have cleared commercial permission and YOLO labels for person, harness, and lanyard where visible.",
            }
        )

    for batch in batches:
        batch["recommended_label_review_rows"] = _recommended_label_review_rows([batch])

    return {
        "mode": mode,
        "required_per_class": minimum,
        "labeled_images_per_class": count_summary,
        "missing_label_minimums": missing_labels,
        "total_missing_label_annotations": sum(item["missing"] for item in missing_labels.values()),
        "recommended_label_review_rows": _recommended_label_review_rows(batches),
        "missing_coverage": missing_coverage,
        "coverage_deficit_count": coverage_deficit_count,
        "next_capture_batches": batches,
    }


def _validate_yolo_label(path: Path, errors: list[str]) -> set[int]:
    present_classes: set[int] = set()
    if not path.exists():
        errors.append(f"label file does not exist: {path}")
        return present_classes
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number} must have 5 YOLO fields")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(part) for part in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number} contains non-numeric YOLO fields")
            continue
        if class_id not in REQUIRED_CLASSES:
            errors.append(f"{path}:{line_number} class id {class_id} is not in apron/harness taxonomy")
        else:
            present_classes.add(class_id)
        if not all(0.0 <= value <= 1.0 for value in [x_center, y_center, width, height]):
            errors.append(f"{path}:{line_number} YOLO coordinates must be normalized between 0 and 1")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path}:{line_number} YOLO width and height must be positive")
    return present_classes


def _validate_label_review_metadata(
    *,
    label_item: dict[str, Any],
    label_path: Path,
    clip_ids: set[str],
    errors: list[str],
) -> bool:
    approved = True
    review_status = str(label_item.get("review_status") or "")
    if review_status != APPROVED_LABEL_REVIEW_STATUS:
        errors.append(
            f"yolo_labels path {label_path} must have review_status={APPROVED_LABEL_REVIEW_STATUS}"
        )
        approved = False
    if not label_item.get("reviewer"):
        errors.append(f"yolo_labels path {label_path} must include reviewer")
        approved = False
    if not label_item.get("reviewed_at"):
        errors.append(f"yolo_labels path {label_path} must include reviewed_at")
        approved = False
    source_clip_id = str(label_item.get("source_clip_id") or "")
    if not source_clip_id:
        errors.append(f"yolo_labels path {label_path} must include source_clip_id")
        approved = False
    elif clip_ids and source_clip_id not in clip_ids:
        errors.append(
            f"yolo_labels path {label_path} source_clip_id is not listed in clips: {source_clip_id}"
        )
        approved = False
    split = str(label_item.get("split") or "")
    if not split:
        errors.append(f"yolo_labels path {label_path} must include split")
        approved = False
    elif split not in LABEL_SPLITS:
        errors.append(
            f"yolo_labels path {label_path} has invalid split {split}; expected one of {sorted(LABEL_SPLITS)}"
        )
        approved = False
    return approved


def _dataset_paths(doc: dict[str, Any]) -> dict[str, str]:
    dataset = doc.get("dataset") or {}
    if not isinstance(dataset, dict):
        dataset = {}
    paths = dict(DEFAULT_DATASET_SPLITS)
    for key in ["root", "train", "val", "test"]:
        if dataset.get(key):
            paths[key] = str(dataset[key])
    return paths


def _clip_requires_seed_source_review(clip: dict[str, Any]) -> bool:
    source = str(clip.get("source") or "").strip()
    permission = str(clip.get("permission") or "").strip()
    return (
        source in PUBLIC_SEED_SOURCES
        or source.startswith("public_")
        or source.startswith("roboflow")
        or permission == "commercial_dataset_approved"
    )


def _load_seed_source_review(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "seed source review report is required for public seed clips"
    if not path.exists():
        return None, f"seed source review report does not exist: {path}"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"seed source review report is unreadable: {path}: {exc}"
    if not isinstance(report, dict):
        return None, f"seed source review report must be a JSON object: {path}"
    return report, None


def _seed_source_index(report: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(report, dict):
        return {}
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        return {}
    indexed: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        source_ref = str(item.get("source_ref") or "")
        if source_ref:
            indexed.setdefault(source_ref, []).append(item)
    return indexed


def _source_review_priority(value: Any) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 999999
    return priority if priority > 0 else 999999


def _seed_source_suggestions(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        return []
    suggestions: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        capability = str(item.get("capability") or "")
        if capability not in {"apron_required", "harness_required"}:
            continue
        suggestions.append(
            {
                "review_priority": item.get("review_priority"),
                "source_ref": item.get("source_ref"),
                "capability": capability,
                "approval_status": item.get("approval_status"),
                "training_usable": item.get("training_usable") is True,
                "review_focus": item.get("review_focus"),
                "url": item.get("url"),
                "blocker": item.get("blocker"),
            }
        )
    return sorted(
        suggestions,
        key=lambda item: (
            _source_review_priority(item.get("review_priority")),
            str(item.get("source_ref") or ""),
        ),
    )


def _load_seed_import_manifest(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "seed import manifest is required for public seed clips"
    if not path.exists():
        return None, f"seed import manifest does not exist: {path}"
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, f"seed import manifest is unreadable: {path}: {exc}"
    if not isinstance(doc, dict):
        return None, f"seed import manifest must be a YAML object: {path}"
    if doc.get("version") != 1:
        return None, "seed import manifest version must be 1"
    if doc.get("kind") != "apron_harness_seed_import_manifest":
        return None, "seed import manifest kind must be apron_harness_seed_import_manifest"
    imports = doc.get("imports")
    if not isinstance(imports, list) or not imports:
        return None, "seed import manifest imports must contain at least one source import entry"
    return doc, None


def _seed_import_index(doc: dict[str, Any] | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not isinstance(doc, dict):
        return {}
    imports = doc.get("imports")
    if not isinstance(imports, list):
        return {}
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in imports:
        if not isinstance(item, dict):
            continue
        source_ref = str(item.get("source_ref") or "")
        capability = str(item.get("capability") or "")
        if source_ref and capability:
            indexed.setdefault((source_ref, capability), []).append(item)
    return indexed


def _approved_seed_source_candidate(
    *,
    seed_source_candidates: dict[str, list[dict[str, Any]]],
    source_ref: str,
    capability: str,
) -> dict[str, Any] | None:
    for candidate in seed_source_candidates.get(source_ref) or []:
        if (
            str(candidate.get("capability") or "") == capability
            and not _seed_source_candidate_approval_errors(candidate)
        ):
            return candidate
    return None


def _review_evidence_errors(
    path_value: Any,
    sha256_value: Any,
    *,
    source_ref: str = "",
    capability: str = "",
    reviewed_by: str = "",
    reviewed_at: str = "",
) -> list[str]:
    errors: list[str] = []
    raw_path = str(path_value or "").strip()
    raw_sha256 = str(sha256_value or "").strip()
    evidence_path: Path | None = None
    if not raw_path:
        errors.append("review_evidence_path is required")
    elif "://" in raw_path:
        errors.append("review_evidence_path must be a local evidence file path, not a URL")
    else:
        evidence_path = Path(raw_path).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = ROOT / evidence_path
        if not evidence_path.is_file():
            errors.append("review_evidence_path must point to an existing local file")

    if not raw_sha256:
        errors.append("review_evidence_sha256 is required")
    elif len(raw_sha256) != 64 or any(char not in HEX_DIGITS for char in raw_sha256):
        errors.append("review_evidence_sha256 must be a 64-character SHA-256 hex digest")

    if evidence_path is not None and evidence_path.is_file() and len(raw_sha256) == 64:
        actual_sha256 = _sha256_file(evidence_path)
        if actual_sha256 != raw_sha256:
            errors.append("review_evidence_sha256 does not match review_evidence_path")
        try:
            evidence_doc = yaml.safe_load(evidence_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            errors.append(f"review_evidence_path is not parseable YAML/JSON: {exc}")
            evidence_doc = {}
        if not isinstance(evidence_doc, dict):
            errors.append("review_evidence_path must contain a mapping document")
            evidence_doc = {}
        if evidence_doc:
            if evidence_doc.get("kind") != REVIEW_EVIDENCE_KIND:
                errors.append(f"review evidence kind must be {REVIEW_EVIDENCE_KIND}")
            if evidence_doc.get("version") != 1:
                errors.append("review evidence version must be 1")
            if source_ref and evidence_doc.get("source_ref") != source_ref:
                errors.append("review evidence source_ref must match the approved source")
            if capability and evidence_doc.get("capability") != capability:
                errors.append("review evidence capability must match the approved source")
            if reviewed_by and evidence_doc.get("reviewed_by") != reviewed_by:
                errors.append("review evidence reviewed_by must match the approved source")
            if reviewed_at and evidence_doc.get("reviewed_at") != reviewed_at:
                errors.append("review evidence reviewed_at must match the approved source")
            review_items = (
                evidence_doc.get("review_items")
                if isinstance(evidence_doc.get("review_items"), dict)
                else {}
            )
            if not review_items:
                errors.append("review evidence review_items are required")
            for item in sorted(REQUIRED_SEED_SOURCE_REVIEW_FIELDS):
                review_item = review_items.get(item) if isinstance(review_items.get(item), dict) else {}
                if review_item.get("approved") is not True:
                    errors.append(f"review evidence {item}.approved must be true")
                if not str(review_item.get("evidence_ref") or "").strip():
                    errors.append(f"review evidence {item}.evidence_ref is required")
    return errors


def _seed_source_candidate_approval_errors(candidate: dict[str, Any] | None) -> list[str]:
    if not candidate:
        return ["approved seed source review candidate is required"]
    errors: list[str] = []
    if candidate.get("training_usable") is not True:
        errors.append("training_usable must be true")
    if str(candidate.get("approval_status") or "") != "approved_for_training":
        errors.append("approval_status must be approved_for_training")
    if not candidate.get("manifest_import_path"):
        errors.append("manifest_import_path is required")
    if not candidate.get("reviewed_by"):
        errors.append("reviewed_by is required")
    if not candidate.get("reviewed_at"):
        errors.append("reviewed_at is required")
    errors.extend(
        _review_evidence_errors(
            candidate.get("review_evidence_path"),
            candidate.get("review_evidence_sha256"),
            source_ref=str(candidate.get("source_ref") or ""),
            capability=str(candidate.get("capability") or ""),
            reviewed_by=str(candidate.get("reviewed_by") or ""),
            reviewed_at=str(candidate.get("reviewed_at") or ""),
        )
    )
    completed_review = (
        candidate.get("completed_review") if isinstance(candidate.get("completed_review"), dict) else {}
    )
    missing_reviews = sorted(
        field
        for field in REQUIRED_SEED_SOURCE_REVIEW_FIELDS
        if completed_review.get(field) is not True
    )
    if missing_reviews:
        errors.append("completed_review missing approvals: " + ", ".join(missing_reviews))
    return errors


def _seed_import_entry_errors(
    entry: dict[str, Any],
    *,
    capability: str,
    expected_manifest_import_path: str | None,
    approved_source_candidate: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    source_candidate_errors = _seed_source_candidate_approval_errors(approved_source_candidate)
    if source_candidate_errors:
        errors.append(
            "approved seed source review candidate is required before seed import: "
            + "; ".join(source_candidate_errors)
        )
    if entry.get("include_in_training") is not True:
        errors.append("include_in_training must be true")
    if str(entry.get("review_status") or "") != "approved_for_training":
        errors.append("review_status must be approved_for_training")
    if not entry.get("reviewed_by"):
        errors.append("reviewed_by is required")
    if not entry.get("reviewed_at"):
        errors.append("reviewed_at is required")
    manifest_import_path = str(entry.get("manifest_import_path") or "")
    if not manifest_import_path:
        errors.append("manifest_import_path is required")
    elif expected_manifest_import_path and manifest_import_path != expected_manifest_import_path:
        errors.append("manifest_import_path must match the seed source review")
    raw_export_ref = str(entry.get("raw_export_ref") or "").strip()
    if not raw_export_ref:
        errors.append("raw_export_ref is required")
    elif "://" not in raw_export_ref:
        errors.append("raw_export_ref must be a remote immutable export reference, not a local path")
    else:
        raw_export_scheme = raw_export_ref.split("://", 1)[0].lower()
        if raw_export_scheme not in ALLOWED_RAW_EXPORT_REF_SCHEMES:
            errors.append(
                "raw_export_ref scheme must be one of "
                + ", ".join(sorted(ALLOWED_RAW_EXPORT_REF_SCHEMES))
            )
    raw_export_sha256 = str(entry.get("raw_export_sha256") or "").strip()
    if not raw_export_sha256:
        errors.append("raw_export_sha256 is required")
    elif len(raw_export_sha256) != 64 or any(char not in HEX_DIGITS for char in raw_export_sha256):
        errors.append("raw_export_sha256 must be a 64-character SHA-256 hex digest")
    if str(entry.get("export_format") or "") not in ALLOWED_SEED_IMPORT_EXPORT_FORMATS:
        errors.append(
            "export_format must be one of " + ", ".join(sorted(ALLOWED_SEED_IMPORT_EXPORT_FORMATS))
        )

    completed_review = (
        entry.get("completed_review") if isinstance(entry.get("completed_review"), dict) else {}
    )
    missing_reviews = sorted(
        field for field in REQUIRED_SEED_IMPORT_REVIEW_FIELDS if completed_review.get(field) is not True
    )
    if missing_reviews:
        errors.append("completed_review missing approvals: " + ", ".join(missing_reviews))
    class_mapping = entry.get("class_mapping") if isinstance(entry.get("class_mapping"), dict) else {}
    if not class_mapping:
        errors.append("class_mapping is required")
    else:
        mapped_classes = {
            str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
            for value in class_mapping.values()
            if str(value or "").strip()
        }
        missing_mapped_classes = sorted(
            REQUIRED_SEED_IMPORT_COUNT_CLASSES.get(capability, set()) - mapped_classes
        )
        if missing_mapped_classes:
            errors.append(
                "class_mapping must map source classes to required local classes: "
                + ", ".join(missing_mapped_classes)
            )
    if not entry.get("person_box_policy"):
        errors.append("person_box_policy is required")
    if not entry.get("hard_negative_policy"):
        errors.append("hard_negative_policy is required")
    split_plan = entry.get("split_plan") if isinstance(entry.get("split_plan"), dict) else {}
    missing_splits = [split for split in ["train", "val", "test"] if not split_plan.get(split)]
    if missing_splits:
        errors.append("split_plan missing required splits: " + ", ".join(missing_splits))
    expected_counts = (
        entry.get("expected_labeled_images_per_class")
        if isinstance(entry.get("expected_labeled_images_per_class"), dict)
        else {}
    )
    if not expected_counts:
        errors.append("expected_labeled_images_per_class is required")
    for class_name in sorted(REQUIRED_SEED_IMPORT_COUNT_CLASSES.get(capability, set())):
        if _count_for_class(expected_counts, class_name) <= 0:
            errors.append(f"expected_labeled_images_per_class.{class_name} must be greater than 0")
    return errors


def _validate_seed_import_clip(
    *,
    clip: dict[str, Any],
    prefix: str,
    seed_import_manifest: Path | None,
    seed_import_doc: dict[str, Any] | None,
    seed_import_entries: dict[tuple[str, str], list[dict[str, Any]]],
    seed_import_load_error: str | None,
    seed_source_candidates: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    source_ref = str(clip.get("source_ref") or "").strip()
    target_capabilities = sorted(
        _as_set(clip.get("target_capabilities")) & {"apron_required", "harness_required"}
    )
    status: dict[str, Any] = {
        "clip_id": clip.get("clip_id"),
        "source_ref": source_ref,
        "target_capabilities": target_capabilities,
        "import_approved": False,
    }
    if not source_ref:
        return status
    if seed_import_load_error:
        errors.append(f"{prefix}.source_ref {source_ref} cannot be imported: {seed_import_load_error}")
        return status
    if not seed_import_doc:
        errors.append(f"{prefix}.source_ref {source_ref} requires a seed import manifest")
        return status

    approved_capabilities: list[str] = []
    capability_details: list[dict[str, Any]] = []
    for capability in target_capabilities:
        entries = seed_import_entries.get((source_ref, capability)) or []
        source_review_candidate = next(
            (
                item
                for item in seed_source_candidates.get(source_ref) or []
                if str(item.get("capability") or "") == capability
            ),
            None,
        )
        expected_manifest_import_path = (
            str(source_review_candidate.get("manifest_import_path"))
            if source_review_candidate and source_review_candidate.get("manifest_import_path")
            else None
        )
        entry_details = []
        capability_approved = False
        for entry in entries:
            entry_errors = _seed_import_entry_errors(
                entry,
                capability=capability,
                expected_manifest_import_path=expected_manifest_import_path,
                approved_source_candidate=source_review_candidate,
            )
            entry_approved = not entry_errors
            capability_approved = capability_approved or entry_approved
            entry_details.append(
                {
                    "include_in_training": entry.get("include_in_training") is True,
                    "review_status": entry.get("review_status"),
                    "manifest_import_path": entry.get("manifest_import_path"),
                    "review_priority": entry.get("review_priority")
                    or (source_review_candidate or {}).get("review_priority"),
                    "review_focus": entry.get("review_focus")
                    or (source_review_candidate or {}).get("review_focus"),
                    "raw_export_ref": entry.get("raw_export_ref"),
                    "raw_export_sha256": entry.get("raw_export_sha256"),
                    "reviewed_by": entry.get("reviewed_by"),
                    "reviewed_at": entry.get("reviewed_at"),
                    "approved_for_training": entry_approved,
                    "errors": entry_errors,
                }
            )
        capability_details.append(
            {
                "capability": capability,
                "entry_count": len(entries),
                "approved": capability_approved,
                "entries": entry_details,
            }
        )
        if capability_approved:
            approved_capabilities.append(capability)
        else:
            errors.append(
                f"{prefix}.source_ref {source_ref} requires approved seed import manifest entry "
                f"for {capability}"
            )

    status["seed_import_manifest"] = str(seed_import_manifest) if seed_import_manifest else None
    status["approved_capabilities"] = approved_capabilities
    status["capabilities"] = capability_details
    status["import_approved"] = sorted(approved_capabilities) == target_capabilities
    return status


def _validate_seed_source_clip(
    *,
    clip: dict[str, Any],
    prefix: str,
    seed_source_review_report: Path | None,
    seed_source_review: dict[str, Any] | None,
    seed_source_candidates: dict[str, list[dict[str, Any]]],
    seed_source_load_error: str | None,
    errors: list[str],
) -> dict[str, Any]:
    source_ref = str(clip.get("source_ref") or "").strip()
    target_capabilities = sorted(
        _as_set(clip.get("target_capabilities")) & {"apron_required", "harness_required"}
    )
    status = {
        "clip_id": clip.get("clip_id"),
        "source_ref": source_ref,
        "target_capabilities": target_capabilities,
        "training_approved": False,
    }
    if not source_ref:
        errors.append(f"{prefix}.source_ref is required for public/commercial dataset seed clips")
        return status
    if seed_source_load_error:
        errors.append(f"{prefix}.source_ref {source_ref} cannot be approved: {seed_source_load_error}")
        return status
    if not seed_source_review or seed_source_review.get("ok") is not True:
        errors.append(f"{prefix}.source_ref {source_ref} requires a successful seed source review report")
        return status
    candidates = seed_source_candidates.get(source_ref) or []
    if not candidates:
        errors.append(f"{prefix}.source_ref is not listed in seed source review candidates: {source_ref}")
        return status

    approved_candidates = [
        item
        for item in candidates
        if not _seed_source_candidate_approval_errors(item)
    ]
    approved_capabilities = {str(item.get("capability") or "") for item in approved_candidates}
    missing_capabilities = sorted(set(target_capabilities) - approved_capabilities)
    if missing_capabilities:
        errors.append(
            f"{prefix}.source_ref {source_ref} is not approved for training capabilities: "
            + ", ".join(missing_capabilities)
        )
        return status

    status["training_approved"] = True
    status["seed_source_review_report"] = str(seed_source_review_report)
    status["capabilities"] = [
        {
            "capability": str(item.get("capability") or ""),
            "manifest_import_path": item.get("manifest_import_path"),
            "review_priority": item.get("review_priority"),
            "review_focus": item.get("review_focus"),
            "reviewed_by": item.get("reviewed_by"),
            "reviewed_at": item.get("reviewed_at"),
            "review_evidence_path": item.get("review_evidence_path"),
            "review_evidence_sha256": item.get("review_evidence_sha256"),
        }
        for item in approved_candidates
        if str(item.get("capability") or "") in target_capabilities
    ]
    return status


def build_yolo_dataset_config(manifest_path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    paths = _dataset_paths(doc)
    dataset_root = paths.get("root") or str(manifest_path.parent)
    config: dict[str, Any] = {
        "path": dataset_root,
        "train": paths["train"],
        "val": paths["val"],
        "names": REQUIRED_CLASSES,
        "rakshak_lens": {
            "dataset_id": doc.get("dataset_id"),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "permission": (doc.get("license") or {}).get("permission"),
            "missing_ppe_label_policy": "derive_missing_ppe_from_person_to_visible_ppe_association",
        },
    }
    if paths.get("test"):
        config["test"] = paths["test"]
    return config


def _format_values(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values)


def _format_md_cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|").strip()


def render_capture_work_order(report: dict[str, Any]) -> str:
    deficit = report.get("capture_deficit") if isinstance(report.get("capture_deficit"), dict) else {}
    production_deficit = (
        report.get("production_capture_deficit")
        if isinstance(report.get("production_capture_deficit"), dict)
        else {}
    )
    missing_labels = deficit.get("missing_label_minimums") or {}
    production_missing_labels = production_deficit.get("missing_label_minimums") or {}
    missing_coverage = deficit.get("missing_coverage") or {}
    batches = deficit.get("next_capture_batches") or []
    manifest = str(report.get("manifest") or "")
    mode = str(report.get("mode") or "pilot")

    lines = [
        "# Apron/Harness Capture Work Order",
        "",
        f"- Manifest: `{manifest}`",
        f"- Mode: `{mode}`",
        f"- Required labels per class: `{deficit.get('required_per_class', 0)}`",
        f"- Missing label annotations: `{deficit.get('total_missing_label_annotations', 0)}`",
        f"- Recommended label-review rows: `{deficit.get('recommended_label_review_rows', 0)}`",
        f"- Coverage deficits: `{deficit.get('coverage_deficit_count', 0)}`",
    ]
    if report.get("manifest_sha256"):
        lines.append(f"- Manifest SHA256: `{report.get('manifest_sha256')}`")
    lines.extend([
        "",
        "## Label Deficit",
        "",
        "| Class | Current | Required | Missing |",
        "|---|---:|---:|---:|",
    ])
    for class_name in REQUIRED_CLASSES.values():
        item = missing_labels.get(class_name) or {}
        current = item.get("current", (deficit.get("labeled_images_per_class") or {}).get(class_name, 0))
        required = item.get("required", deficit.get("required_per_class", 0))
        missing = item.get("missing", 0)
        lines.append(f"| `{class_name}` | {current} | {required} | {missing} |")

    lines.extend([
        "",
        "Label annotations are per-class coverage counts; label-review rows are planned image/label-file rows. One reviewed image can count for multiple required classes when it contains a person plus visible target PPE.",
    ])

    if production_deficit:
        lines.extend([
            "",
            "## Production Capture Target",
            "",
            f"- Required labels per class: `{production_deficit.get('required_per_class', 0)}`",
            f"- Missing label annotations: `{production_deficit.get('total_missing_label_annotations', 0)}`",
            f"- Recommended label-review rows: `{production_deficit.get('recommended_label_review_rows', 0)}`",
            f"- Coverage deficits: `{production_deficit.get('coverage_deficit_count', 0)}`",
            "",
            "| Class | Current | Production Required | Production Missing |",
            "|---|---:|---:|---:|",
        ])
        for class_name in REQUIRED_CLASSES.values():
            item = production_missing_labels.get(class_name) or {}
            current = item.get("current", (production_deficit.get("labeled_images_per_class") or {}).get(class_name, 0))
            required = item.get("required", production_deficit.get("required_per_class", 0))
            missing = item.get("missing", 0)
            lines.append(f"| `{class_name}` | {current} | {required} | {missing} |")
        lines.extend([
            "",
            "Production compliance is not unlocked by clearing the pilot checklist alone. Keep the pack in pilot status until the production target, side-by-side promotion reports, and Jetson gate all pass.",
        ])

    lines.extend([
        "",
        "## Coverage Deficit",
        "",
        "| Coverage Field | Missing Values |",
        "|---|---|",
    ])
    for field in COVERAGE_REQUIREMENTS:
        lines.append(f"| `{field}` | {_format_values(missing_coverage.get(field) or [])} |")

    seed_source_suggestions = (
        report.get("seed_source_suggestions")
        if isinstance(report.get("seed_source_suggestions"), list)
        else []
    )
    if seed_source_suggestions:
        lines.extend([
            "",
            "## Public Seed Source Review Queue",
            "",
            "These public candidates are not training-approved yet. Review them in priority order before using any source in a capture manifest or training dataset.",
            "",
            "| Priority | Source | Capability | Approval | Training | Review Focus | URL | Blocker |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for item in seed_source_suggestions:
            lines.append(
                "| "
                f"`{item.get('review_priority')}` | "
                f"`{item.get('source_ref')}` | "
                f"`{item.get('capability')}` | "
                f"`{item.get('approval_status')}` | "
                f"`{bool(item.get('training_usable'))}` | "
                f"{_format_md_cell(item.get('review_focus'))} | "
                f"{_format_md_cell(item.get('url'))} | "
                f"{_format_md_cell(item.get('blocker'))} |"
            )

    lines.extend(["", "## Next Capture Batches", ""])
    if not batches:
        lines.append("No capture batches are currently required for this mode.")
    for batch in batches:
        lines.extend(
            [
                f"### `{batch.get('batch_id', 'unnamed_batch')}`",
                "",
                f"- Target capability: `{batch.get('target_capability', 'unknown')}`",
                f"- Minimum labeled images: `{batch.get('minimum_labeled_images', 0)}`",
                f"- Recommended label-review rows: `{batch.get('recommended_label_review_rows', 0)}`",
                f"- Required label classes: {_format_values(batch.get('required_label_classes'))}",
                f"- Camera angles: {_format_values(batch.get('camera_angles'))}",
                f"- Distance bands: {_format_values(batch.get('distance_bands'))}",
                f"- Lighting: {_format_values(batch.get('lighting'))}",
                f"- Motion blur: {_format_values(batch.get('motion_blur'))}",
                f"- Positive variants: {_format_values(batch.get('positive_variants'))}",
                f"- Hard negatives: {_format_values(batch.get('hard_negative_tags'))}",
                f"- Acceptance: {batch.get('acceptance_note', 'Use cleared footage and reviewed YOLO labels.')}",
                "",
            ]
        )
        matrix = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
        if matrix:
            lines.extend(
                [
                    "| Capture Type | Variant / Tag | Recommended Examples | Required Labels |",
                    "|---|---|---:|---|",
                ]
            )
            for row in matrix:
                lines.append(
                    f"| `{row.get('capture_type', 'unknown')}` "
                    f"| `{row.get('variant_or_tag', 'unknown')}` "
                    f"| {row.get('recommended_examples', 0)} "
                    f"| {_format_values(row.get('required_label_classes'))} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Required Follow-Up Commands",
            "",
            "```bash",
            f".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest {manifest} --mode {mode}",
            f".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest {manifest} --mode production",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --schema-only --emit-starter-label-review-csv qa/video_eval/results/apron_harness_starter_label_review_template.csv",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --emit-starter-label-review-csv qa/video_eval/results/apron_harness_production_starter_label_review_template.csv",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml --mode production --schema-only --emit-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv --emit-starter-label-review-csv qa/video_eval/results/apron_harness_production_starter_label_review_template.csv --emit-label-review-csv qa/video_eval/results/apron_harness_production_label_review_template.csv",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.starter_reviewed.yaml",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml",
            ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json",
            ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest /path/to/cleared/apron_harness_capture_manifest.yaml --mode production --schema-only --seed-source-review-report /path/to/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --import-approved-seed-exports --emit-updated-manifest /path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml --seed-import-camera-angle front --seed-import-distance-band medium --seed-import-lighting indoor_bright --seed-import-motion-blur low",
            ".venv/bin/python scripts/apron_harness_train.py --data /path/to/cleared/dataset.yaml --capture-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml --capture-matrix-csv /path/to/filled/apron_harness_production_capture_matrix.csv --capture-preflight-mode production --require-capture-preflight --model yolo26n.pt --device auto --out-plan /path/to/cleared/apron_harness_training_plan.json",
            "```",
            "",
            "Production training now requires the generated sidecar manifest next to the capture matrix CSV, for example `/path/to/filled/apron_harness_production_capture_matrix.csv.manifest.json`, plus the `.label_review_import.json` sidecar next to the reviewed capture manifest. The training preflight verifies matrix SHA, source manifest SHA, label-review import SHA, row count, permission gate, storage gate, manifest reconciliation, and dataset.yaml `rakshak_lens.source_manifest`, `rakshak_lens.source_manifest_sha256`, permission, and missing-PPE policy before returning `ready_to_train`.",
            "",
            "Use the starter label-review CSVs for the first controlled-capture loop only. A starter import can produce an intermediate reviewed manifest and sidecar, but it does not satisfy the full pilot or production gate until the full matrix, full label-review CSV, strict manifest validation, and production capture preflight pass.",
            "",
            "`--validate-label-review-csv` is a no-write preflight for filled CSVs. Run it with `--schema-only` against the pre-import manifest; it must report `LABEL_REVIEW_VALIDATION: gate=pass` before the corresponding import command is run.",
            "",
            "Approved public seed exports can be materialized only after the seed-source review report, source-review bundle hash validation, and filled seed-import manifest all pass. The importer validates the raw YOLO ZIP SHA-256 and preflight, remaps source labels into the local apron/harness taxonomy, writes local `images/` and `labels/`, creates manifest clips with explicit operator-supplied coverage metadata, recomputes counts from the converted YOLO label files, and writes a `.seed_export_import.json` sidecar.",
            "",
            "If approved seed-export materialization fails, the same `.seed_export_import.json` sidecar is still written with `valid=false`, import errors, warnings, and `partial_materialization` so the next SSH session can resume from the exact gate failure.",
            "",
            "The label-review importer converts only approved CSV rows into capture-manifest `yolo_labels`, creates `clips` only from filled reviewed row metadata for new `source_clip_id` values, rejects uncleared permissions and repo-local raw storage references, recomputes `counts.labeled_images_per_class` from the referenced YOLO label files, and leaves strict validation to the manifest doctor. It does not approve rows, invent absent clip metadata, or bypass missing label files.",
            "",
            "The importer also writes a `.label_review_import.json` sidecar next to the reviewed manifest. Keep that sidecar with the manifest; it records the source-manifest SHA-256, filled label-review CSV SHA-256, reviewed-manifest SHA-256, imported/skipped row counts, imported clip counts, recomputed class counts, and the review/permission/storage/count gates used for the import.",
            "",
            "That label-review sidecar must include `updated_manifest_validation` from a strict, non-schema-only validation of the reviewed manifest. Production training and promotion reject sidecars where the reviewed manifest validation is missing, failed, schema-only, non-production, or tied to a different manifest SHA.",
            "",
            "Strict capture manifest validation also requires every `yolo_labels` entry to include `review_status=approved`, `reviewer`, `reviewed_at`, `split`, and a `source_clip_id` that exists in `clips`; pilot mode requires train and validation labels for every required class, production mode also requires held-out test labels plus at least 10% of the production per-class target in both `val` and `test`, one source clip cannot appear in multiple splits, and unreviewed auto-label output cannot count toward training readiness.",
            "",
            "Top-level `coverage` values are not trusted unless they are backed by clip metadata: camera angle, distance band, lighting, motion blur, `positive_variant_tags`, and `hard_negative_tags`.",
            "",
            "Do not train or promote a production model from footage that is paid, gated, academic-only, customer-private, or identifiable without explicit approval and cleared storage.",
            "",
        ]
    )
    return "\n".join(lines)


CAPTURE_MATRIX_CSV_FIELDS = [
    "batch_id",
    "row_id",
    "target_capability",
    "capture_type",
    "variant_or_tag",
    "recommended_examples",
    "required_label_classes",
    "camera_angles",
    "distance_bands",
    "lighting",
    "motion_blur",
    "notes",
    "captured_examples",
    "labeled_examples",
    "review_status",
    "permission",
    "raw_storage_ref",
    "owner",
    "due_date",
    "status_notes",
]
CAPTURE_PROGRESS_STATUSES = {
    "not_started",
    "captured",
    "labeled",
    "reviewed",
    "approved",
    "blocked",
}
CAPTURE_READY_STATUS = "approved"
REPO_STORAGE_REFS = {"", "repo", "git", "committed_source"}


def _raw_storage_ref_is_repo_local(raw_storage_ref: str) -> bool:
    text = str(raw_storage_ref or "").strip()
    if text in REPO_STORAGE_REFS:
        return True
    if "://" in text and not text.lower().startswith("file://"):
        return False
    if text.lower().startswith("file://"):
        text = text[7:]
    path = Path(text)
    if not path.is_absolute():
        return True
    try:
        path.resolve(strict=False).relative_to(ROOT.resolve(strict=False))
        return True
    except ValueError:
        return False


CAPTURE_PROGRESS_DEFAULTS = {
    "captured_examples": "0",
    "labeled_examples": "0",
    "review_status": "not_started",
    "permission": "unknown",
    "raw_storage_ref": "",
    "owner": "",
    "due_date": "",
    "status_notes": "",
}
LABEL_REVIEW_IMPORT_REQUIRED_FIELDS = [
    "planned_label_id",
    "batch_id",
    "row_id",
    "target_capability",
    "capture_type",
    "variant_or_tag",
    "required_label_classes",
    "suggested_split",
    "source_clip_id",
    "image_path",
    "label_path",
    "review_status",
    "reviewer",
    "reviewed_at",
    "permission",
    "raw_storage_ref",
    "notes",
]
LABEL_REVIEW_CLIP_METADATA_FIELDS = [
    "clip_source",
    "clip_camera_angle",
    "clip_distance_band",
    "clip_lighting",
    "clip_motion_blur",
    "clip_expected_visible_classes",
    "clip_positive_variant_tags",
    "clip_hard_negative_tags",
    "clip_notes",
]
LABEL_REVIEW_GUIDANCE_FIELDS = [
    "source_manifest",
    "source_manifest_sha256",
    "taxonomy_version",
    "taxonomy_class_ids",
    "required_label_class_ids",
    "label_format",
    "required_review_status",
    "requires_reviewer",
    "requires_reviewed_at",
    "allowed_permissions",
    "requires_non_repo_raw_storage_ref",
    "label_file_must_exist",
    "import_sidecar_required",
    *LABEL_REVIEW_CLIP_METADATA_FIELDS,
]
LABEL_REVIEW_CSV_FIELDS = LABEL_REVIEW_IMPORT_REQUIRED_FIELDS + LABEL_REVIEW_GUIDANCE_FIELDS
STARTER_LABEL_REVIEW_ROW_IDS = {
    "apron_required_closed_set_capture.positive.denim_apron",
    "apron_required_closed_set_capture.positive.kitchen_or_food_service_apron",
    "harness_required_closed_set_capture.positive.fall_arrest_harness",
    "harness_required_closed_set_capture.positive.full_body_safety_harness",
}


def _taxonomy_class_ids_text() -> str:
    return ";".join(f"{class_id}:{class_name}" for class_id, class_name in REQUIRED_CLASSES.items())


def _required_label_class_ids_text(required_label_classes: Any) -> str:
    class_ids_by_name = {class_name: class_id for class_id, class_name in REQUIRED_CLASSES.items()}
    class_ids: list[str] = []
    for raw_name in str(required_label_classes or "").split(";"):
        class_name = raw_name.strip()
        if not class_name or class_name not in class_ids_by_name:
            continue
        class_ids.append(str(class_ids_by_name[class_name]))
    return ";".join(class_ids)


def _parse_required_label_class_ids(value: Any, *, row_number: int, errors: list[str]) -> set[int]:
    class_ids: set[int] = set()
    for raw_value in str(value or "").split(";"):
        token = raw_value.strip()
        if not token:
            continue
        if ":" in token:
            token = token.split(":", 1)[0].strip()
        if "=" in token:
            token = token.rsplit("=", 1)[-1].strip()
        try:
            class_id = int(token)
        except ValueError:
            errors.append(f"line {row_number}: required_label_class_ids contains non-numeric id {raw_value}")
            continue
        if class_id not in REQUIRED_CLASSES:
            errors.append(f"line {row_number}: required_label_class_ids contains unknown class id {class_id}")
            continue
        class_ids.add(class_id)
    return class_ids


def _suggested_label_split(example_index: int, total_examples: int, mode: str) -> str:
    if mode == "production" and total_examples >= 3:
        test_count = max(1, math.ceil(total_examples * PRODUCTION_HOLDOUT_SPLIT_MIN_FRACTION))
        val_count = max(1, math.ceil(total_examples * PRODUCTION_HOLDOUT_SPLIT_MIN_FRACTION))
        if example_index <= test_count:
            return "test"
        if example_index <= test_count + val_count:
            return "val"
        return "train"
    if total_examples >= 2:
        val_count = max(1, math.ceil(total_examples * 0.2))
        if example_index <= val_count:
            return "val"
    return "train"


def _capture_matrix_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    deficit = report.get("capture_deficit") if isinstance(report.get("capture_deficit"), dict) else {}
    rows: list[dict[str, Any]] = []
    for batch in deficit.get("next_capture_batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_id = str(batch.get("batch_id") or "")
        matrix = batch.get("capture_matrix") if isinstance(batch.get("capture_matrix"), list) else []
        for row in matrix:
            if not isinstance(row, dict):
                continue
            rendered: dict[str, Any] = {"batch_id": batch_id}
            for field in CAPTURE_MATRIX_CSV_FIELDS:
                if field == "batch_id":
                    continue
                value = row.get(field)
                if value is None and field in CAPTURE_PROGRESS_DEFAULTS:
                    value = CAPTURE_PROGRESS_DEFAULTS[field]
                if isinstance(value, list):
                    value = ";".join(str(item) for item in value)
                rendered[field] = value if value is not None else ""
            rows.append(rendered)
    return rows


def render_capture_matrix_csv(report: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CAPTURE_MATRIX_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in _capture_matrix_rows(report):
        writer.writerow(row)
    return buffer.getvalue()


def render_label_review_csv(
    report: dict[str, Any],
    mode: str = "pilot",
    *,
    include_row_ids: set[str] | None = None,
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LABEL_REVIEW_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    source_manifest = str(report.get("manifest") or "")
    source_manifest_sha256 = str(report.get("manifest_sha256") or "")
    allowed_permissions = ";".join(sorted(ALLOWED_PERMISSIONS))
    for row in _capture_matrix_rows(report):
        row_id = str(row.get("row_id") or "capture_row")
        if include_row_ids is not None and row_id not in include_row_ids:
            continue
        try:
            recommended_examples = int(str(row.get("recommended_examples") or "0"))
        except ValueError:
            recommended_examples = 0
        required_label_classes = row.get("required_label_classes") or ""
        required_label_class_ids = _required_label_class_ids_text(required_label_classes)
        capture_type = str(row.get("capture_type") or "")
        variant_or_tag = str(row.get("variant_or_tag") or "")
        clip_positive_variant_tags = variant_or_tag if capture_type == "positive_variant" else ""
        clip_hard_negative_tags = variant_or_tag if capture_type == "hard_negative" else ""
        for example_index in range(1, max(0, recommended_examples) + 1):
            split = _suggested_label_split(example_index, recommended_examples, mode)
            planned_label_id = f"{row_id}.label_{example_index:04d}"
            writer.writerow(
                {
                    "planned_label_id": planned_label_id,
                    "batch_id": row.get("batch_id") or "",
                    "row_id": row_id,
                    "target_capability": row.get("target_capability") or "",
                    "capture_type": row.get("capture_type") or "",
                    "variant_or_tag": row.get("variant_or_tag") or "",
                    "required_label_classes": required_label_classes,
                    "suggested_split": split,
                    "source_clip_id": "",
                    "image_path": f"images/{split}/{planned_label_id}.jpg",
                    "label_path": f"labels/{split}/{planned_label_id}.txt",
                    "review_status": "not_started",
                    "reviewer": "",
                    "reviewed_at": "",
                    "permission": "unknown",
                    "raw_storage_ref": "",
                    "notes": (
                        "Convert approved rows into capture manifest yolo_labels entries; "
                        "review_status must be approved before training."
                    ),
                    "source_manifest": source_manifest,
                    "source_manifest_sha256": source_manifest_sha256,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "taxonomy_class_ids": _taxonomy_class_ids_text(),
                    "required_label_class_ids": required_label_class_ids,
                    "label_format": YOLO_LABEL_FORMAT,
                    "required_review_status": APPROVED_LABEL_REVIEW_STATUS,
                    "requires_reviewer": "true",
                    "requires_reviewed_at": "true",
                    "allowed_permissions": allowed_permissions,
                    "requires_non_repo_raw_storage_ref": "true",
                    "label_file_must_exist": "true",
                    "import_sidecar_required": "true",
                    "clip_source": "controlled_capture",
                    "clip_camera_angle": "",
                    "clip_distance_band": "",
                    "clip_lighting": "",
                    "clip_motion_blur": "",
                    "clip_expected_visible_classes": required_label_classes,
                    "clip_positive_variant_tags": clip_positive_variant_tags,
                    "clip_hard_negative_tags": clip_hard_negative_tags,
                    "clip_notes": (
                        "Fill clip metadata when source_clip_id is not already listed in "
                        "the capture manifest clips."
                    ),
                }
            )
    return buffer.getvalue()


def render_starter_label_review_csv(report: dict[str, Any], mode: str = "pilot") -> str:
    """Render the minimum first-fill label-review CSV from the starter rows."""
    return render_label_review_csv(
        report,
        mode=mode,
        include_row_ids=STARTER_LABEL_REVIEW_ROW_IDS,
    )


def _clip_items_by_id_from_doc(doc: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    clips = doc.get("clips")
    if not isinstance(clips, list):
        return [], {}
    clip_order: list[str] = []
    clips_by_id: dict[str, dict[str, Any]] = {}
    for clip in clips:
        if not isinstance(clip, dict) or not clip.get("clip_id"):
            continue
        clip_id = str(clip["clip_id"])
        if clip_id not in clips_by_id:
            clip_order.append(clip_id)
        clips_by_id[clip_id] = dict(clip)
    return clip_order, clips_by_id


def _clip_ids_from_doc(doc: dict[str, Any]) -> set[str]:
    _, clips_by_id = _clip_items_by_id_from_doc(doc)
    return set(clips_by_id)


def _append_unique_values(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    for value in additions:
        if value not in merged:
            merged.append(value)
    return merged


def _list_or_csv_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return _split_csv_values(value)


def _label_review_clip_from_row(
    *,
    row: dict[str, Any],
    row_number: int,
    permission: str,
    raw_storage_ref: str,
    errors: list[str],
) -> dict[str, Any] | None:
    clip_id = str(row.get("source_clip_id") or "").strip()
    if not clip_id:
        return None

    clip_source = str(row.get("clip_source") or "controlled_capture").strip()
    camera_angle = str(row.get("clip_camera_angle") or "").strip()
    distance_band = str(row.get("clip_distance_band") or "").strip()
    lighting = str(row.get("clip_lighting") or "").strip()
    motion_blur = str(row.get("clip_motion_blur") or "").strip()
    required_fields = {
        "clip_source": clip_source,
        "clip_camera_angle": camera_angle,
        "clip_distance_band": distance_band,
        "clip_lighting": lighting,
        "clip_motion_blur": motion_blur,
    }
    for field, value in required_fields.items():
        if not value:
            errors.append(f"line {row_number}: {field} is required for new source_clip_id {clip_id}")

    allowed_values_by_field = {
        "clip_camera_angle": REQUIRED_CAMERA_ANGLES,
        "clip_distance_band": REQUIRED_DISTANCE_BANDS,
        "clip_lighting": REQUIRED_LIGHTING,
        "clip_motion_blur": REQUIRED_MOTION_BLUR,
    }
    observed_values_by_field = {
        "clip_camera_angle": camera_angle,
        "clip_distance_band": distance_band,
        "clip_lighting": lighting,
        "clip_motion_blur": motion_blur,
    }
    for field, allowed_values in allowed_values_by_field.items():
        value = observed_values_by_field[field]
        if value and value not in allowed_values:
            errors.append(
                f"line {row_number}: {field} must be one of {sorted(allowed_values)}"
            )

    target_capabilities = [
        value
        for value in _split_csv_values(row.get("target_capability"))
        if value in {"apron_required", "harness_required"}
    ]
    if not target_capabilities:
        errors.append(
            f"line {row_number}: target_capability must include apron_required or harness_required"
        )

    expected_visible_classes = _split_csv_values(
        row.get("clip_expected_visible_classes") or row.get("required_label_classes")
    )
    required_class_names = set(REQUIRED_CLASSES.values())
    unknown_classes = sorted(
        class_name for class_name in expected_visible_classes if class_name not in required_class_names
    )
    if not expected_visible_classes:
        errors.append(f"line {row_number}: clip_expected_visible_classes is required")
    elif unknown_classes:
        errors.append(
            f"line {row_number}: clip_expected_visible_classes contains unknown classes {unknown_classes}"
        )

    capture_type = str(row.get("capture_type") or "").strip()
    variant_or_tag = str(row.get("variant_or_tag") or "").strip()
    positive_variant_tags = _split_csv_values(row.get("clip_positive_variant_tags"))
    hard_negative_tags = _split_csv_values(row.get("clip_hard_negative_tags"))
    if capture_type == "positive_variant" and variant_or_tag:
        positive_variant_tags = _append_unique_values(positive_variant_tags, [variant_or_tag])
    if capture_type == "hard_negative" and variant_or_tag:
        hard_negative_tags = _append_unique_values(hard_negative_tags, [variant_or_tag])

    if errors:
        return None
    clip = {
        "clip_id": clip_id,
        "source": clip_source,
        "permission": permission,
        "camera_angle": camera_angle,
        "distance_band": distance_band,
        "lighting": lighting,
        "motion_blur": motion_blur,
        "target_capabilities": target_capabilities,
        "expected_visible_classes": expected_visible_classes,
        "positive_variant_tags": positive_variant_tags,
        "hard_negative_tags": hard_negative_tags,
        "raw_storage_ref": raw_storage_ref,
        "notes": str(row.get("clip_notes") or row.get("notes") or "").strip(),
    }
    return {key: value for key, value in clip.items() if value not in ("", [], None)}


def _merge_label_review_clip_metadata(
    *,
    existing_clip: dict[str, Any],
    row_clip: dict[str, Any],
    row_number: int,
    errors: list[str],
) -> None:
    for field in [
        "source",
        "permission",
        "camera_angle",
        "distance_band",
        "lighting",
        "motion_blur",
        "raw_storage_ref",
    ]:
        existing_value = existing_clip.get(field)
        row_value = row_clip.get(field)
        if existing_value and row_value and existing_value != row_value:
            errors.append(
                f"line {row_number}: source_clip_id {existing_clip.get('clip_id')} "
                f"has conflicting {field}: {existing_value} vs {row_value}"
            )
    for field in [
        "target_capabilities",
        "expected_visible_classes",
        "positive_variant_tags",
        "hard_negative_tags",
    ]:
        existing_clip[field] = _append_unique_values(
            _list_or_csv_values(existing_clip.get(field)),
            _list_or_csv_values(row_clip.get(field)),
        )


def _approved_label_counts_from_entries(
    *,
    manifest_path: Path,
    label_items: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in REQUIRED_CLASSES.values()}
    for item in label_items:
        if str(item.get("review_status") or "") != APPROVED_LABEL_REVIEW_STATUS:
            continue
        if not item.get("path"):
            continue
        label_path = manifest_path.parent / str(item["path"])
        present_classes = _validate_yolo_label(label_path, errors)
        for class_id in present_classes:
            counts[REQUIRED_CLASSES[class_id]] += 1
    return counts


def _label_review_import_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".label_review_import.json")


def build_label_review_import_sidecar(
    *,
    import_report: dict[str, Any],
    updated_manifest_path: Path,
    updated_manifest_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = updated_manifest_validation or {
        "checked": False,
        "ok": False,
        "errors": ["updated manifest validation was not run"],
        "warnings": [],
    }
    return {
        "generated_at": utc_now(),
        "kind": "apron_harness_label_review_import_manifest",
        "valid": import_report.get("ok") is True and validation.get("ok") is True,
        "source_manifest": import_report.get("manifest"),
        "source_manifest_sha256": import_report.get("manifest_sha256"),
        "label_review_csv": import_report.get("label_review_csv"),
        "label_review_csv_sha256": import_report.get("label_review_csv_sha256"),
        "updated_manifest": str(updated_manifest_path),
        "updated_manifest_sha256": _sha256_file(updated_manifest_path),
        "existing_label_count": import_report.get("existing_label_count", 0),
        "imported_label_count": import_report.get("imported_label_count", 0),
        "skipped_label_count": import_report.get("skipped_label_count", 0),
        "invalid_approved_label_count": import_report.get("invalid_approved_label_count", 0),
        "imported_clip_count": import_report.get("imported_clip_count", 0),
        "invalid_clip_metadata_count": import_report.get("invalid_clip_metadata_count", 0),
        "merged_label_count": import_report.get("merged_label_count", 0),
        "updated_labeled_images_per_class": import_report.get("updated_labeled_images_per_class", {}),
        "updated_manifest_validation": validation,
        "label_review_csv_schema": {
            "required_import_fields": LABEL_REVIEW_IMPORT_REQUIRED_FIELDS,
            "generated_guidance_fields": LABEL_REVIEW_GUIDANCE_FIELDS,
        },
        "taxonomy": {
            "version": TAXONOMY_VERSION,
            "classes": REQUIRED_CLASSES,
            "label_format": YOLO_LABEL_FORMAT,
        },
        "training_gate": {
            "requires_approved_label_review_rows": True,
            "requires_review_metadata": True,
            "requires_reviewed_clip_metadata": True,
            "requires_cleared_permission": True,
            "requires_non_repo_raw_storage_refs": True,
            "requires_recomputed_label_counts": True,
            "requires_updated_manifest_validation": True,
            "requires_source_manifest_sha256_match": True,
            "requires_taxonomy_version_match": True,
        },
    }


def label_review_import_validation_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return a compact strict-validation summary suitable for the import sidecar."""
    return {
        "checked": True,
        "mode": report.get("mode"),
        "schema_only": report.get("schema_only") is True,
        "ok": report.get("ok") is True,
        "manifest": report.get("manifest"),
        "manifest_sha256": report.get("manifest_sha256"),
        "approved_label_file_count": report.get("approved_label_file_count", 0),
        "label_files_per_split": report.get("label_files_per_split", {}),
        "approved_label_images_per_split_per_class": report.get(
            "approved_label_images_per_split_per_class",
            {},
        ),
        "minimum_split_class_counts": report.get("minimum_split_class_counts", {}),
        "counts": report.get("counts", {}),
        "errors": report.get("errors", []),
        "warnings": report.get("warnings", []),
    }


def import_label_review_csv_into_manifest(
    *,
    manifest_path: Path,
    label_review_csv_path: Path,
) -> dict[str, Any]:
    """Merge approved label-review CSV rows into a capture manifest YAML document."""
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.exists():
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "label_review_csv": str(label_review_csv_path),
            "imported_label_count": 0,
            "skipped_label_count": 0,
            "errors": [f"manifest does not exist: {manifest_path}"],
            "warnings": warnings,
            "updated_manifest": None,
        }
    if not label_review_csv_path.exists():
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "label_review_csv": str(label_review_csv_path),
            "imported_label_count": 0,
            "skipped_label_count": 0,
            "errors": [f"label review CSV does not exist: {label_review_csv_path}"],
            "warnings": warnings,
            "updated_manifest": None,
        }

    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "label_review_csv": str(label_review_csv_path),
            "imported_label_count": 0,
            "skipped_label_count": 0,
            "errors": [f"manifest must be a YAML object: {manifest_path}"],
            "warnings": warnings,
            "updated_manifest": None,
        }

    try:
        csv_text = label_review_csv_path.read_text(encoding="utf-8")
        reader = csv.DictReader(csv_text.splitlines())
        rows = list(reader)
    except Exception as exc:
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "label_review_csv": str(label_review_csv_path),
            "imported_label_count": 0,
            "skipped_label_count": 0,
            "errors": [f"label review CSV is unreadable: {label_review_csv_path}: {exc}"],
            "warnings": warnings,
            "updated_manifest": None,
        }

    fieldnames = reader.fieldnames or []
    missing_fields = [field for field in LABEL_REVIEW_IMPORT_REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        errors.append(f"label review CSV missing required columns: {missing_fields}")
    missing_guidance_fields = [field for field in LABEL_REVIEW_GUIDANCE_FIELDS if field not in fieldnames]
    if missing_guidance_fields:
        warnings.append(
            "label review CSV missing generated guidance columns: "
            + ", ".join(missing_guidance_fields)
        )
    if not rows:
        warnings.append("label review CSV has no rows")

    existing_label_items_raw = doc.get("yolo_labels") or []
    if not isinstance(existing_label_items_raw, list):
        errors.append("existing yolo_labels must be a list before CSV import")
        existing_label_items_raw = []
    existing_label_items = [
        dict(item) for item in existing_label_items_raw if isinstance(item, dict) and item.get("path")
    ]
    if len(existing_label_items) != len(existing_label_items_raw):
        errors.append("existing yolo_labels entries must be objects with path before CSV import")

    clip_order, clips_by_id = _clip_items_by_id_from_doc(doc)
    clip_ids = set(clips_by_id)
    imported_clip_ids: set[str] = set()
    merged_by_path: dict[str, dict[str, Any]] = {
        str(item["path"]): dict(item) for item in existing_label_items
    }
    imported_paths: set[str] = set()
    imported_label_count = 0
    skipped_label_count = 0
    invalid_approved_label_count = 0
    invalid_clip_metadata_count = 0

    for row_number, row in enumerate(rows, start=2):
        review_status = str(row.get("review_status") or "").strip()
        if review_status != APPROVED_LABEL_REVIEW_STATUS:
            skipped_label_count += 1
            continue

        row_errors: list[str] = []
        label_path_value = str(row.get("label_path") or "").strip()
        if not label_path_value:
            row_errors.append(f"line {row_number}: label_path is required for approved rows")
        elif Path(label_path_value).is_absolute():
            row_errors.append(f"line {row_number}: label_path must be relative to the manifest")

        split = str(row.get("suggested_split") or "").strip()
        label_item = {
            "path": label_path_value,
            "review_status": APPROVED_LABEL_REVIEW_STATUS,
            "reviewer": str(row.get("reviewer") or "").strip(),
            "reviewed_at": str(row.get("reviewed_at") or "").strip(),
            "source_clip_id": str(row.get("source_clip_id") or "").strip(),
            "split": split,
        }
        for optional_field in [
            "planned_label_id",
            "image_path",
            "permission",
            "raw_storage_ref",
            "target_capability",
            "row_id",
        ]:
            value = str(row.get(optional_field) or "").strip()
            if value:
                label_item[optional_field] = value

        source_manifest_sha256 = str(row.get("source_manifest_sha256") or "").strip()
        if source_manifest_sha256 and source_manifest_sha256 != _sha256_file(manifest_path):
            row_errors.append(
                f"line {row_number}: source_manifest_sha256 does not match {manifest_path}"
            )
        taxonomy_version = str(row.get("taxonomy_version") or "").strip()
        if taxonomy_version and taxonomy_version != TAXONOMY_VERSION:
            row_errors.append(
                f"line {row_number}: taxonomy_version must be {TAXONOMY_VERSION}"
            )
        label_format = str(row.get("label_format") or "").strip()
        if label_format and label_format != YOLO_LABEL_FORMAT:
            row_errors.append(f"line {row_number}: label_format must be {YOLO_LABEL_FORMAT}")
        required_label_class_ids = str(row.get("required_label_class_ids") or "").strip()
        if not required_label_class_ids:
            required_label_class_ids = _required_label_class_ids_text(row.get("required_label_classes"))
        required_class_ids = _parse_required_label_class_ids(
            required_label_class_ids,
            row_number=row_number,
            errors=row_errors,
        )
        if label_path_value and required_class_ids:
            present_class_ids = _validate_yolo_label(manifest_path.parent / label_path_value, row_errors)
            missing_required_class_ids = sorted(required_class_ids - present_class_ids)
            for class_id in missing_required_class_ids:
                row_errors.append(
                    "line "
                    f"{row_number}: label file {label_path_value} missing required class "
                    f"{class_id}:{REQUIRED_CLASSES[class_id]}"
                )

        permission = str(row.get("permission") or "unknown").strip()
        if permission not in ALLOWED_PERMISSIONS:
            row_errors.append(
                f"line {row_number}: permission is not cleared for commercial training: {permission}"
            )
        raw_storage_ref = str(row.get("raw_storage_ref") or "").strip()
        if _raw_storage_ref_is_repo_local(raw_storage_ref):
            row_errors.append(
                f"line {row_number}: raw_storage_ref must point outside the repo for approved rows"
            )

        source_clip_id = str(label_item.get("source_clip_id") or "").strip()
        pending_new_clip: dict[str, Any] | None = None
        pending_updated_clip: dict[str, Any] | None = None
        validation_clip_ids = set(clip_ids)
        if source_clip_id and source_clip_id not in clip_ids:
            clip_errors: list[str] = []
            row_clip = _label_review_clip_from_row(
                row=row,
                row_number=row_number,
                permission=permission,
                raw_storage_ref=raw_storage_ref,
                errors=clip_errors,
            )
            if row_clip is None:
                invalid_clip_metadata_count += 1
                row_errors.extend(clip_errors)
            else:
                pending_new_clip = row_clip
                validation_clip_ids.add(source_clip_id)
        elif source_clip_id in imported_clip_ids and any(
            str(row.get(field) or "").strip() for field in LABEL_REVIEW_CLIP_METADATA_FIELDS
        ):
            clip_errors = []
            row_clip = _label_review_clip_from_row(
                row=row,
                row_number=row_number,
                permission=permission,
                raw_storage_ref=raw_storage_ref,
                errors=clip_errors,
            )
            if row_clip is None:
                invalid_clip_metadata_count += 1
                row_errors.extend(clip_errors)
            else:
                merged_clip = dict(clips_by_id[source_clip_id])
                _merge_label_review_clip_metadata(
                    existing_clip=merged_clip,
                    row_clip=row_clip,
                    row_number=row_number,
                    errors=row_errors,
                )
                pending_updated_clip = merged_clip

        if label_path_value:
            _validate_label_review_metadata(
                label_item=label_item,
                label_path=manifest_path.parent / label_path_value,
                clip_ids=validation_clip_ids,
                errors=row_errors,
            )
        if label_path_value in imported_paths:
            row_errors.append(f"line {row_number}: duplicate approved label_path {label_path_value}")

        if row_errors:
            invalid_approved_label_count += 1
            errors.extend(row_errors)
            continue

        if pending_new_clip is not None and source_clip_id:
            clips_by_id[source_clip_id] = pending_new_clip
            clip_order.append(source_clip_id)
            clip_ids.add(source_clip_id)
            imported_clip_ids.add(source_clip_id)
        elif pending_updated_clip is not None and source_clip_id:
            clips_by_id[source_clip_id] = pending_updated_clip

        imported_paths.add(label_path_value)
        merged_by_path[label_path_value] = label_item
        imported_label_count += 1

    merged_label_items = [merged_by_path[path] for path in sorted(merged_by_path)]
    approved_counts = _approved_label_counts_from_entries(
        manifest_path=manifest_path,
        label_items=merged_label_items,
        errors=errors,
    )
    updated_doc = dict(doc)
    updated_doc["clips"] = [clips_by_id[clip_id] for clip_id in clip_order if clip_id in clips_by_id]
    updated_doc["yolo_labels"] = merged_label_items
    counts_doc = updated_doc.get("counts") if isinstance(updated_doc.get("counts"), dict) else {}
    counts_doc = dict(counts_doc)
    counts_doc["labeled_images_per_class"] = approved_counts
    updated_doc["counts"] = counts_doc

    if imported_label_count == 0:
        warnings.append("no approved label-review rows were imported")

    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "label_review_csv": str(label_review_csv_path),
        "label_review_csv_sha256": _sha256_file(label_review_csv_path),
        "existing_label_count": len(existing_label_items),
        "imported_label_count": imported_label_count,
        "skipped_label_count": skipped_label_count,
        "invalid_approved_label_count": invalid_approved_label_count,
        "imported_clip_count": len(imported_clip_ids),
        "invalid_clip_metadata_count": invalid_clip_metadata_count,
        "merged_label_count": len(merged_label_items),
        "updated_labeled_images_per_class": approved_counts,
        "errors": errors,
        "warnings": warnings,
        "updated_manifest": updated_doc,
    }


def validate_label_review_csv(
    *,
    manifest_path: Path,
    label_review_csv_path: Path,
) -> dict[str, Any]:
    """Validate a filled label-review CSV without writing an updated manifest."""
    report = import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_csv_path,
    )
    validation = {
        key: value
        for key, value in report.items()
        if key != "updated_manifest"
    }
    validation["checked"] = True
    validation["gate_passed"] = (
        report.get("ok") is True
        and int(report.get("imported_label_count") or 0) > 0
        and int(report.get("invalid_approved_label_count") or 0) == 0
        and int(report.get("invalid_clip_metadata_count") or 0) == 0
    )
    if report.get("ok") is True and int(report.get("imported_label_count") or 0) <= 0:
        validation.setdefault("warnings", [])
        validation["warnings"] = list(validation.get("warnings") or [])
        validation["warnings"].append("label review CSV has no approved rows to validate")
    return validation


def _parse_nonnegative_int(value: Any, *, field: str, row_id: str, errors: list[str]) -> int:
    try:
        parsed = int(str(value or "0").strip())
    except ValueError:
        errors.append(f"{row_id}.{field} must be a non-negative integer")
        return 0
    if parsed < 0:
        errors.append(f"{row_id}.{field} must be a non-negative integer")
        return 0
    return parsed


def _split_csv_values(value: Any) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _sanitize_path_part(value: Any) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value or "").strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "seed"


def _safe_relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _load_yolo_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return names
    return {}


def _local_class_id_by_source_name(class_mapping: dict[str, Any]) -> dict[str, int]:
    local_ids_by_name = {class_name: class_id for class_id, class_name in REQUIRED_CLASSES.items()}
    mapped: dict[str, int] = {}
    for source_name, local_name in class_mapping.items():
        normalized_local = str(local_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_local in local_ids_by_name:
            mapped[str(source_name)] = local_ids_by_name[normalized_local]
    return mapped


def _seed_export_member_split(member: str) -> str | None:
    normalized = member.replace("\\", "/").lower()
    for split in ("train", "valid", "val", "test"):
        marker = f"/{split}/"
        if normalized.startswith(f"{split}/") or marker in normalized:
            return "val" if split in {"valid", "val"} else split
    return None


def _member_after_directory(member: str, directory: str) -> str | None:
    parts = member.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part.lower() == directory:
            tail = "/".join(parts[index + 1 :]).strip("/")
            return tail or None
    return None


def _seed_export_image_index(members: list[str]) -> dict[tuple[str, str], str]:
    indexed: dict[tuple[str, str], str] = {}
    for member in members:
        split = _seed_export_member_split(member)
        suffix = Path(member).suffix.lower()
        if split not in LABEL_SPLITS or suffix not in IMAGE_EXTENSIONS:
            continue
        rel = _member_after_directory(member, "images")
        if not rel:
            continue
        key = Path(rel).with_suffix("").as_posix()
        indexed.setdefault((split, key), member)
    return indexed


def _seed_export_label_key(member: str) -> str | None:
    rel = _member_after_directory(member, "labels")
    if not rel:
        return None
    return Path(rel).with_suffix("").as_posix()


def _resolve_seed_export_path(path_value: Any) -> Path:
    raw_path = str(path_value or "").strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _convert_seed_yolo_label(
    *,
    label_text: str,
    label_member: str,
    source_names: dict[int, str],
    local_class_ids_by_source_name: dict[str, int],
    errors: list[str],
) -> tuple[str, set[int], set[str]]:
    converted_lines: list[str] = []
    present_classes: set[int] = set()
    ignored_source_classes: set[str] = set()
    for line_number, raw_line in enumerate(label_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            errors.append(f"{label_member}:{line_number} must have 5 YOLO fields")
            continue
        try:
            source_class_id = int(parts[0])
            coordinates = [float(part) for part in parts[1:]]
        except ValueError:
            errors.append(f"{label_member}:{line_number} contains non-numeric YOLO fields")
            continue
        source_name = source_names.get(source_class_id)
        if source_name is None:
            errors.append(f"{label_member}:{line_number} references unknown source class id {source_class_id}")
            continue
        local_class_id = local_class_ids_by_source_name.get(source_name)
        if local_class_id is None:
            ignored_source_classes.add(source_name)
            continue
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"{label_member}:{line_number} YOLO coordinates must be normalized between 0 and 1")
            continue
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            errors.append(f"{label_member}:{line_number} YOLO width and height must be positive")
            continue
        converted_lines.append(" ".join([str(local_class_id), *parts[1:]]))
        present_classes.add(local_class_id)
    return "\n".join(converted_lines) + ("\n" if converted_lines else ""), present_classes, ignored_source_classes


def _seed_export_import_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".seed_export_import.json")


def build_seed_export_import_sidecar(
    *,
    import_report: dict[str, Any],
    updated_manifest_path: Path,
    updated_manifest_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = updated_manifest_validation or {
        "checked": False,
        "ok": False,
        "errors": ["updated manifest validation was not run"],
        "warnings": [],
    }
    return {
        "generated_at": utc_now(),
        "kind": "apron_harness_seed_export_import_manifest",
        "valid": import_report.get("ok") is True and validation.get("ok") is True,
        "source_manifest": import_report.get("manifest"),
        "source_manifest_sha256": import_report.get("manifest_sha256"),
        "seed_source_review_report": import_report.get("seed_source_review_report"),
        "seed_source_review_sha256": import_report.get("seed_source_review_sha256"),
        "source_recheck": import_report.get("source_recheck") or {},
        "seed_import_manifest": import_report.get("seed_import_manifest"),
        "seed_import_manifest_sha256": import_report.get("seed_import_manifest_sha256"),
        "updated_manifest": str(updated_manifest_path),
        "updated_manifest_sha256": _sha256_file(updated_manifest_path)
        if updated_manifest_path.exists()
        else "",
        "output_root": import_report.get("output_root"),
        "imported_clip_count": import_report.get("imported_clip_count", 0),
        "imported_label_count": import_report.get("imported_label_count", 0),
        "copied_image_count": import_report.get("copied_image_count", 0),
        "skipped_label_count": import_report.get("skipped_label_count", 0),
        "partial_materialization": import_report.get("ok") is not True
        and (
            int(import_report.get("imported_clip_count") or 0) > 0
            or int(import_report.get("imported_label_count") or 0) > 0
            or int(import_report.get("copied_image_count") or 0) > 0
        ),
        "updated_labeled_images_per_class": import_report.get("updated_labeled_images_per_class", {}),
        "imports": import_report.get("imports", []),
        "errors": import_report.get("errors", []),
        "warnings": import_report.get("warnings", []),
        "updated_manifest_validation": validation,
        "training_gate": {
            "requires_seed_source_review_gate": True,
            "requires_seed_import_manifest_gate": True,
            "requires_raw_export_local_sha256_match": True,
            "requires_yolo_export_preflight": True,
            "requires_class_mapping_to_local_taxonomy": True,
            "requires_review_metadata": True,
            "requires_updated_manifest_validation": True,
        },
    }


def _seed_import_metadata_errors(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = {
        "camera_angle": REQUIRED_CAMERA_ANGLES,
        "distance_band": REQUIRED_DISTANCE_BANDS,
        "lighting": REQUIRED_LIGHTING,
        "motion_blur": REQUIRED_MOTION_BLUR,
    }
    for field, allowed_values in required_fields.items():
        value = str(metadata.get(field) or "").strip()
        if not value:
            errors.append(f"seed import clip metadata {field} is required")
        elif value not in allowed_values:
            errors.append(
                f"seed import clip metadata {field} must be one of {sorted(allowed_values)}"
            )
    return errors


def import_approved_seed_exports_into_manifest(
    *,
    manifest_path: Path,
    seed_source_review_report: Path,
    seed_import_manifest: Path,
    output_manifest_path: Path | None = None,
    output_root: Path | None = None,
    seed_clip_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize approved public seed YOLO exports into a capture manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    metadata = dict(seed_clip_metadata or {})
    errors.extend(_seed_import_metadata_errors(metadata))

    if not manifest_path.exists():
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "seed_source_review_report": str(seed_source_review_report),
            "seed_import_manifest": str(seed_import_manifest),
            "imported_label_count": 0,
            "errors": [f"manifest does not exist: {manifest_path}"],
            "warnings": warnings,
            "updated_manifest": None,
        }
    try:
        doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "seed_source_review_report": str(seed_source_review_report),
            "seed_import_manifest": str(seed_import_manifest),
            "imported_label_count": 0,
            "errors": [f"manifest is unreadable: {manifest_path}: {exc}"],
            "warnings": warnings,
            "updated_manifest": None,
        }
    if not isinstance(doc, dict):
        errors.append(f"manifest must be a YAML object: {manifest_path}")
        doc = {}

    seed_source_review, seed_source_error = _load_seed_source_review(seed_source_review_report)
    if seed_source_error:
        errors.append(seed_source_error)
    seed_import_doc, seed_import_error = _load_seed_import_manifest(seed_import_manifest)
    if seed_import_error:
        errors.append(seed_import_error)

    seed_import_review: dict[str, Any] = {}
    if seed_source_review and seed_import_doc and not seed_import_error:
        seed_import_review = validate_import_manifest(seed_import_manifest, seed_source_review)
        if seed_import_review.get("gate_passed") is not True:
            errors.extend(f"seed_import_manifest.{error}" for error in seed_import_review.get("errors") or [])
            errors.extend(f"seed_import_manifest.{blocker}" for blocker in seed_import_review.get("blockers") or [])

    if errors:
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path) if manifest_path.exists() else "",
            "seed_source_review_report": str(seed_source_review_report),
            "seed_source_review_sha256": source_review_fingerprint(seed_source_review)
            if seed_source_review
            else "",
            "source_recheck": _source_recheck_lineage(seed_source_review),
            "seed_import_manifest": str(seed_import_manifest),
            "seed_import_manifest_sha256": _sha256_file(seed_import_manifest)
            if seed_import_manifest.exists()
            else "",
            "seed_import_review": seed_import_review,
            "imported_clip_count": 0,
            "imported_label_count": 0,
            "copied_image_count": 0,
            "skipped_label_count": 0,
            "updated_labeled_images_per_class": {},
            "imports": seed_import_review.get("imports") if seed_import_review else [],
            "errors": errors,
            "warnings": warnings,
            "updated_manifest": None,
        }

    output_base = (output_root or (output_manifest_path.parent if output_manifest_path else manifest_path.parent))
    output_base.mkdir(parents=True, exist_ok=True)
    existing_clips_raw = doc.get("clips") or []
    existing_labels_raw = doc.get("yolo_labels") or []
    if not isinstance(existing_clips_raw, list):
        errors.append("existing clips must be a list before seed export import")
        existing_clips_raw = []
    if not isinstance(existing_labels_raw, list):
        errors.append("existing yolo_labels must be a list before seed export import")
        existing_labels_raw = []
    existing_clips = [dict(item) for item in existing_clips_raw if isinstance(item, dict) and item.get("clip_id")]
    existing_labels = [dict(item) for item in existing_labels_raw if isinstance(item, dict) and item.get("path")]
    if len(existing_clips) != len(existing_clips_raw):
        errors.append("existing clips entries must be objects with clip_id before seed export import")
    if len(existing_labels) != len(existing_labels_raw):
        errors.append("existing yolo_labels entries must be objects with path before seed export import")
    if errors:
        return {
            "ok": False,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "seed_source_review_report": str(seed_source_review_report),
            "seed_source_review_sha256": source_review_fingerprint(seed_source_review),
            "source_recheck": _source_recheck_lineage(seed_source_review),
            "seed_import_manifest": str(seed_import_manifest),
            "seed_import_manifest_sha256": _sha256_file(seed_import_manifest),
            "seed_import_review": seed_import_review,
            "imported_clip_count": 0,
            "imported_label_count": 0,
            "copied_image_count": 0,
            "skipped_label_count": 0,
            "updated_labeled_images_per_class": {},
            "imports": seed_import_review.get("imports") if seed_import_review else [],
            "errors": errors,
            "warnings": warnings,
            "updated_manifest": None,
        }

    raw_imports = [item for item in seed_import_doc.get("imports") or [] if isinstance(item, dict)]
    import_review_by_key = {
        (str(item.get("source_ref") or ""), str(item.get("capability") or "")): item
        for item in seed_import_review.get("imports") or []
        if isinstance(item, dict)
    }
    clips_by_id: dict[str, dict[str, Any]] = {}
    clip_order: list[str] = []
    for item in existing_clips:
        clip_id = str(item["clip_id"])
        clips_by_id[clip_id] = dict(item)
        if clip_id not in clip_order:
            clip_order.append(clip_id)
    labels_by_path = {str(item["path"]): dict(item) for item in existing_labels}
    imported_clip_count = 0
    imported_label_count = 0
    copied_image_count = 0
    skipped_label_count = 0
    import_summaries: list[dict[str, Any]] = []

    for entry in raw_imports:
        source_ref = str(entry.get("source_ref") or "")
        capability = str(entry.get("capability") or "")
        review_item = import_review_by_key.get((source_ref, capability)) or {}
        yolo_export_preflight = (
            review_item.get("yolo_export_preflight")
            if isinstance(review_item.get("yolo_export_preflight"), dict)
            else {}
        )
        review_artifact_preflight = (
            review_item.get("review_artifact_preflight")
            if isinstance(review_item.get("review_artifact_preflight"), dict)
            else {}
        )
        if review_item.get("approved_for_training") is not True:
            continue
        export_path = _resolve_seed_export_path(entry.get("raw_export_local_path"))
        class_mapping = entry.get("class_mapping") if isinstance(entry.get("class_mapping"), dict) else {}
        local_class_ids_by_source = _local_class_id_by_source_name(class_mapping)
        entry_errors: list[str] = []
        entry_warnings: list[str] = []
        imported_for_entry = 0
        copied_images_for_entry = 0
        if not local_class_ids_by_source:
            entry_errors.append("class_mapping does not map any source labels to local classes")
        if not export_path.exists():
            entry_errors.append(f"raw_export_local_path does not exist: {export_path}")
        elif _sha256_file(export_path) != str(entry.get("raw_export_sha256") or ""):
            entry_errors.append("raw_export_local_path SHA-256 does not match raw_export_sha256")
        if entry_errors:
            errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
            import_summaries.append(
                {
                    "source_ref": source_ref,
                    "capability": capability,
                    "raw_export_ref": entry.get("raw_export_ref"),
                    "raw_export_sha256": entry.get("raw_export_sha256"),
                    "raw_export_local_path": entry.get("raw_export_local_path"),
                    "imported_label_count": 0,
                    "copied_image_count": 0,
                    "yolo_export_preflight": yolo_export_preflight,
                    "review_artifact_preflight": review_artifact_preflight,
                    "errors": entry_errors,
                    "warnings": entry_warnings,
                }
            )
            continue

        with zipfile.ZipFile(export_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            unsafe_members = [
                name
                for name in members
                if Path(name).is_absolute() or ".." in Path(name).parts
            ]
            if unsafe_members:
                entry_errors.append("raw_export_local_path contains unsafe archive member paths")
                errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
                continue
            data_yaml_candidates = [
                name
                for name in members
                if Path(name).name in {"data.yaml", "dataset.yaml"}
            ]
            if not data_yaml_candidates:
                entry_errors.append("YOLO export archive must include data.yaml or dataset.yaml")
                errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
                continue
            data_yaml_path = sorted(data_yaml_candidates)[0]
            try:
                data_yaml = yaml.safe_load(archive.read(data_yaml_path).decode("utf-8")) or {}
            except Exception as exc:
                entry_errors.append(f"YOLO export data.yaml is not parseable: {exc}")
                errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
                continue
            source_names = _load_yolo_names(data_yaml.get("names") if isinstance(data_yaml, dict) else None)
            if not source_names:
                entry_errors.append("YOLO export data.yaml must define class names")
                errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
                continue
            image_index = _seed_export_image_index(members)
            label_members = [
                name
                for name in members
                if _seed_export_member_split(name) in LABEL_SPLITS
                and "/labels/" in name.replace("\\", "/").lower()
                and Path(name).suffix.lower() == ".txt"
            ]
            for label_member in sorted(label_members):
                split = _seed_export_member_split(label_member)
                label_key = _seed_export_label_key(label_member)
                if split is None or label_key is None:
                    continue
                image_member = image_index.get((split, label_key))
                if not image_member:
                    entry_errors.append(f"missing matching image for YOLO label: {label_member}")
                    continue
                conversion_errors: list[str] = []
                label_text = archive.read(label_member).decode("utf-8")
                converted_label, present_classes, ignored_source_classes = _convert_seed_yolo_label(
                    label_text=label_text,
                    label_member=label_member,
                    source_names=source_names,
                    local_class_ids_by_source_name=local_class_ids_by_source,
                    errors=conversion_errors,
                )
                if ignored_source_classes:
                    entry_warnings.append(
                        f"{label_member}: ignored unmapped source classes "
                        + ", ".join(sorted(ignored_source_classes))
                    )
                if conversion_errors:
                    entry_errors.extend(conversion_errors)
                    continue
                if not converted_label.strip():
                    skipped_label_count += 1
                    entry_warnings.append(f"{label_member}: skipped because no mapped local classes were present")
                    continue
                source_key = _sanitize_path_part(source_ref)
                capability_key = _sanitize_path_part(capability)
                member_hash = hashlib.sha1(label_member.encode("utf-8")).hexdigest()[:12]
                base_name = _sanitize_path_part(f"{source_ref}_{capability}_{split}_{Path(label_key).stem}_{member_hash}")
                label_rel = Path("labels") / split / "seed" / source_key / capability_key / f"{base_name}.txt"
                image_suffix = Path(image_member).suffix.lower() or ".jpg"
                image_rel = Path("images") / split / "seed" / source_key / capability_key / f"{base_name}{image_suffix}"
                label_out = output_base / label_rel
                image_out = output_base / image_rel
                label_out.parent.mkdir(parents=True, exist_ok=True)
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.write_text(converted_label, encoding="utf-8")
                image_out.write_bytes(archive.read(image_member))
                copied_images_for_entry += 1
                copied_image_count += 1

                clip_id = f"seed_{base_name}"
                if clip_id not in clips_by_id:
                    clip = {
                        "clip_id": clip_id,
                        "source": "public_seed_source",
                        "permission": "commercial_dataset_approved",
                        "source_ref": source_ref,
                        "camera_angle": metadata["camera_angle"],
                        "distance_band": metadata["distance_band"],
                        "lighting": metadata["lighting"],
                        "motion_blur": metadata["motion_blur"],
                        "target_capabilities": [capability],
                        "expected_visible_classes": [
                            REQUIRED_CLASSES[class_id] for class_id in sorted(present_classes)
                        ],
                        "positive_variant_tags": _split_csv_values(metadata.get("positive_variant_tags")),
                        "hard_negative_tags": _split_csv_values(metadata.get("hard_negative_tags")),
                        "raw_export_ref": entry.get("raw_export_ref"),
                        "raw_export_sha256": entry.get("raw_export_sha256"),
                        "notes": "Materialized from reviewed public seed YOLO export.",
                    }
                    clips_by_id[clip_id] = clip
                    clip_order.append(clip_id)
                    imported_clip_count += 1
                labels_by_path[str(label_rel)] = {
                    "path": str(label_rel),
                    "image_path": str(image_rel),
                    "review_status": APPROVED_LABEL_REVIEW_STATUS,
                    "reviewer": entry.get("reviewed_by"),
                    "reviewed_at": entry.get("reviewed_at"),
                    "source_clip_id": clip_id,
                    "split": split,
                    "permission": "commercial_dataset_approved",
                    "raw_storage_ref": entry.get("raw_export_ref"),
                    "target_capability": capability,
                    "source_ref": source_ref,
                    "raw_export_sha256": entry.get("raw_export_sha256"),
                    "seed_import_manifest": str(seed_import_manifest),
                    "seed_import_manifest_sha256": _sha256_file(seed_import_manifest),
                    "source_export_label_member": label_member,
                    "source_export_image_member": image_member,
                }
                imported_label_count += 1
                imported_for_entry += 1
        errors.extend(f"{source_ref}.{capability}.{error}" for error in entry_errors)
        warnings.extend(f"{source_ref}.{capability}.{warning}" for warning in entry_warnings)
        import_summaries.append(
            {
                "source_ref": source_ref,
                "capability": capability,
                "raw_export_ref": entry.get("raw_export_ref"),
                "raw_export_sha256": entry.get("raw_export_sha256"),
                "raw_export_local_path": entry.get("raw_export_local_path"),
                "imported_label_count": imported_for_entry,
                "copied_image_count": copied_images_for_entry,
                "yolo_export_preflight": yolo_export_preflight,
                "review_artifact_preflight": review_artifact_preflight,
                "errors": entry_errors,
                "warnings": entry_warnings,
            }
        )

    updated_doc = dict(doc)
    updated_doc["clips"] = [clips_by_id[key] for key in clip_order if key in clips_by_id]
    updated_doc["yolo_labels"] = [labels_by_path[key] for key in sorted(labels_by_path)]
    count_manifest_path = output_manifest_path or manifest_path
    approved_counts = _approved_label_counts_from_entries(
        manifest_path=count_manifest_path,
        label_items=updated_doc["yolo_labels"],
        errors=errors,
    )
    counts_doc = updated_doc.get("counts") if isinstance(updated_doc.get("counts"), dict) else {}
    counts_doc = dict(counts_doc)
    counts_doc["labeled_images_per_class"] = approved_counts
    updated_doc["counts"] = counts_doc
    if imported_label_count == 0:
        errors.append("no approved seed export labels were imported")

    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "seed_source_review_report": str(seed_source_review_report),
        "seed_source_review_sha256": source_review_fingerprint(seed_source_review),
        "source_recheck": _source_recheck_lineage(seed_source_review),
        "seed_import_manifest": str(seed_import_manifest),
        "seed_import_manifest_sha256": _sha256_file(seed_import_manifest),
        "seed_import_review": seed_import_review,
        "output_root": str(output_base),
        "imported_clip_count": imported_clip_count,
        "imported_label_count": imported_label_count,
        "copied_image_count": copied_image_count,
        "skipped_label_count": skipped_label_count,
        "updated_labeled_images_per_class": approved_counts,
        "imports": import_summaries,
        "errors": errors,
        "warnings": warnings,
        "updated_manifest": updated_doc if not errors else None,
    }


def _manifest_count_summary(manifest_path: Path) -> tuple[dict[str, int], str]:
    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    counts_doc = (doc.get("counts") or {}).get("labeled_images_per_class") or {}
    return (
        {
            class_name: _count_for_class(counts_doc, class_name)
            for class_name in REQUIRED_CLASSES.values()
        },
        _sha256_file(manifest_path),
    )


def _build_manifest_reconciliation(
    *,
    manifest_path: Path | None,
    required_counts: dict[str, int],
    mode: str,
    has_matrix_rows: bool,
) -> dict[str, Any]:
    if manifest_path is None:
        return {"checked": False, "gate_passed": None}

    manifest_counts, manifest_sha256 = _manifest_count_summary(manifest_path)
    minimum = MIN_COUNTS[mode]
    required = dict(required_counts)
    if not has_matrix_rows:
        required = {class_name: minimum for class_name in REQUIRED_CLASSES.values()}
    missing = {
        class_name: {
            "manifest_count": manifest_counts.get(class_name, 0),
            "required_count": required_count,
            "missing": max(0, required_count - manifest_counts.get(class_name, 0)),
        }
        for class_name, required_count in required.items()
        if manifest_counts.get(class_name, 0) < required_count
    }
    return {
        "checked": True,
        "gate_passed": not missing,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "mode": mode,
        "required_labeled_images_per_class": required,
        "manifest_labeled_images_per_class": manifest_counts,
        "missing_manifest_counts": missing,
    }


def validate_capture_matrix_progress(
    matrix_csv_path: Path,
    *,
    manifest_path: Path | None = None,
    mode: str = "pilot",
) -> dict[str, Any]:
    """Validate a filled capture matrix and report whether it clears production data handoff."""
    errors: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    row_summaries: list[dict[str, Any]] = []
    capability_summary: dict[str, dict[str, int]] = {}
    required_counts_by_class = {class_name: 0 for class_name in REQUIRED_CLASSES.values()}

    if not matrix_csv_path.exists():
        return {
            "ok": False,
            "gate_passed": False,
            "path": str(matrix_csv_path),
            "exists": False,
            "row_count": 0,
            "ready_rows": 0,
            "target_labeled_examples": 0,
            "captured_examples": 0,
            "labeled_examples": 0,
            "missing_labeled_examples": 0,
            "unapproved_rows": 0,
            "unsafe_storage_rows": 0,
            "capabilities": {},
            "manifest_reconciliation": _build_manifest_reconciliation(
                manifest_path=manifest_path,
                required_counts=required_counts_by_class,
                mode=mode,
                has_matrix_rows=False,
            ),
            "rows": [],
            "blockers": [f"capture matrix CSV does not exist: {matrix_csv_path}"],
            "errors": [f"capture matrix CSV does not exist: {matrix_csv_path}"],
            "warnings": warnings,
        }

    try:
        rows = list(csv.DictReader(matrix_csv_path.read_text(encoding="utf-8").splitlines()))
    except Exception as exc:
        return {
            "ok": False,
            "gate_passed": False,
            "path": str(matrix_csv_path),
            "exists": True,
            "row_count": 0,
            "ready_rows": 0,
            "target_labeled_examples": 0,
            "captured_examples": 0,
            "labeled_examples": 0,
            "missing_labeled_examples": 0,
            "unapproved_rows": 0,
            "unsafe_storage_rows": 0,
            "capabilities": {},
            "manifest_reconciliation": _build_manifest_reconciliation(
                manifest_path=manifest_path,
                required_counts=required_counts_by_class,
                mode=mode,
                has_matrix_rows=False,
            ),
            "rows": [],
            "blockers": [f"capture matrix CSV is unreadable: {exc}"],
            "errors": [f"capture matrix CSV is unreadable: {exc}"],
            "warnings": warnings,
        }

    missing_fields = [field for field in CAPTURE_MATRIX_CSV_FIELDS if rows and field not in rows[0]]
    if missing_fields:
        errors.append(f"capture matrix CSV missing required columns: {missing_fields}")
    if not rows:
        errors.append("capture matrix CSV has no rows")

    seen_row_ids: set[str] = set()
    ready_rows = 0
    target_labeled_examples = 0
    captured_examples = 0
    labeled_examples = 0
    missing_labeled_examples = 0
    unapproved_rows = 0
    unsafe_storage_rows = 0

    for index, row in enumerate(rows, start=2):
        row_id = str(row.get("row_id") or f"line_{index}").strip()
        target_capability = str(row.get("target_capability") or "unknown").strip()
        if row_id in seen_row_ids:
            errors.append(f"{row_id} is duplicated in capture matrix CSV")
        seen_row_ids.add(row_id)

        recommended = _parse_nonnegative_int(
            row.get("recommended_examples"),
            field="recommended_examples",
            row_id=row_id,
            errors=errors,
        )
        captured = _parse_nonnegative_int(
            row.get("captured_examples"),
            field="captured_examples",
            row_id=row_id,
            errors=errors,
        )
        labeled = _parse_nonnegative_int(
            row.get("labeled_examples"),
            field="labeled_examples",
            row_id=row_id,
            errors=errors,
        )
        if captured < labeled:
            errors.append(f"{row_id}.captured_examples must be >= labeled_examples")

        required_label_classes = _split_csv_values(row.get("required_label_classes"))
        required_for_reconciliation = max(recommended, labeled)
        for class_name in required_label_classes:
            if class_name in required_counts_by_class:
                required_counts_by_class[class_name] += required_for_reconciliation

        review_status = str(row.get("review_status") or "not_started").strip()
        permission = str(row.get("permission") or "unknown").strip()
        raw_storage_ref = str(row.get("raw_storage_ref") or "").strip()
        if review_status not in CAPTURE_PROGRESS_STATUSES:
            errors.append(f"{row_id}.review_status is not recognized: {review_status}")

        permission_ok = permission in ALLOWED_PERMISSIONS
        storage_ok = not _raw_storage_ref_is_repo_local(raw_storage_ref)
        row_missing = max(0, recommended - labeled)
        row_blockers: list[str] = []
        if captured < labeled:
            row_blockers.append("captured_examples_less_than_labeled_examples")
        if row_missing:
            row_blockers.append(f"missing_labeled_examples={row_missing}")
        if review_status != CAPTURE_READY_STATUS:
            row_blockers.append(f"review_status={review_status}")
        if not permission_ok:
            row_blockers.append(f"permission={permission}")
            unapproved_rows += 1
        if not storage_ok:
            row_blockers.append("raw_storage_ref_missing_or_repo")
            unsafe_storage_rows += 1

        row_ready = not row_blockers and captured >= labeled
        if row_ready:
            ready_rows += 1
        else:
            blockers.append(f"{row_id}: {', '.join(row_blockers)}")

        target_labeled_examples += recommended
        captured_examples += captured
        labeled_examples += labeled
        missing_labeled_examples += row_missing

        capability = capability_summary.setdefault(
            target_capability,
            {
                "row_count": 0,
                "ready_rows": 0,
                "target_labeled_examples": 0,
                "captured_examples": 0,
                "labeled_examples": 0,
                "missing_labeled_examples": 0,
            },
        )
        capability["row_count"] += 1
        capability["ready_rows"] += 1 if row_ready else 0
        capability["target_labeled_examples"] += recommended
        capability["captured_examples"] += captured
        capability["labeled_examples"] += labeled
        capability["missing_labeled_examples"] += row_missing

        row_summaries.append(
            {
                "row_id": row_id,
                "target_capability": target_capability,
                "recommended_examples": recommended,
                "captured_examples": captured,
                "labeled_examples": labeled,
                "missing_labeled_examples": row_missing,
                "review_status": review_status,
                "permission": permission,
                "raw_storage_ref": raw_storage_ref,
                "ready": row_ready,
                "blockers": row_blockers,
            }
        )

    manifest_reconciliation = _build_manifest_reconciliation(
        manifest_path=manifest_path,
        required_counts=required_counts_by_class,
        mode=mode,
        has_matrix_rows=bool(rows),
    )
    if manifest_reconciliation.get("checked") and not manifest_reconciliation.get("gate_passed"):
        for class_name, item in (manifest_reconciliation.get("missing_manifest_counts") or {}).items():
            blockers.append(
                f"manifest.{class_name}: missing_manifest_count={item.get('missing', 0)} "
                f"(manifest={item.get('manifest_count', 0)}, required={item.get('required_count', 0)})"
            )

    no_rows_allowed = (
        not rows
        and bool(manifest_reconciliation.get("checked"))
        and bool(manifest_reconciliation.get("gate_passed"))
    )
    if not rows and no_rows_allowed:
        errors = [error for error in errors if error != "capture matrix CSV has no rows"]
    gate_passed = (
        not errors
        and (no_rows_allowed or ready_rows == len(rows))
        and manifest_reconciliation.get("gate_passed") is not False
    )
    return {
        "ok": not errors,
        "gate_passed": gate_passed,
        "path": str(matrix_csv_path),
        "exists": True,
        "sha256": _sha256_file(matrix_csv_path),
        "row_count": len(rows),
        "ready_rows": ready_rows,
        "target_labeled_examples": target_labeled_examples,
        "captured_examples": captured_examples,
        "labeled_examples": labeled_examples,
        "missing_labeled_examples": missing_labeled_examples,
        "unapproved_rows": unapproved_rows,
        "unsafe_storage_rows": unsafe_storage_rows,
        "capabilities": capability_summary,
        "manifest_reconciliation": manifest_reconciliation,
        "rows": row_summaries,
        "blockers": blockers,
        "errors": errors,
        "warnings": warnings,
    }


def validate_manifest(
    manifest_path: Path,
    mode: str = "pilot",
    schema_only: bool = False,
    seed_source_review_report: Path | None = None,
    seed_import_manifest: Path | None = None,
) -> dict[str, Any]:
    doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    warnings: list[str] = []
    seed_source_review: dict[str, Any] | None = None
    seed_source_load_error: str | None = None
    seed_source_candidates: dict[str, list[dict[str, Any]]] = {}
    seed_source_clip_reviews: list[dict[str, Any]] = []
    seed_import_doc: dict[str, Any] | None = None
    seed_import_load_error: str | None = None
    seed_import_entries: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seed_import_clip_reviews: list[dict[str, Any]] = []
    seed_import_review: dict[str, Any] = {}
    seed_source_suggestions: list[dict[str, Any]] = []

    if doc.get("version") != 1:
        errors.append("version must be 1")
    if not doc.get("dataset_id"):
        errors.append("dataset_id is required")

    classes = _normalize_classes(doc.get("classes"))
    if classes != REQUIRED_CLASSES:
        errors.append(f"classes must exactly match {REQUIRED_CLASSES}")

    license_doc = doc.get("license") or {}
    permission = str(license_doc.get("permission") or "unknown")
    if permission in BLOCKED_PERMISSIONS or permission not in ALLOWED_PERMISSIONS:
        errors.append(f"license.permission is not cleared for commercial training: {permission}")
    if license_doc.get("contains_identifiable_people") is True:
        raw_storage = str(license_doc.get("raw_storage") or "")
        if raw_storage in {"", "repo", "committed_source", "git"}:
            errors.append("identifiable raw footage must be stored outside the repo")

    for field, required in COVERAGE_REQUIREMENTS.items():
        actual = _coverage_values(doc, field)
        missing = sorted(required - actual)
        if missing:
            errors.append(f"coverage.{field} missing required values: {missing}")

    clips = doc.get("clips")
    clip_ids: set[str] = set()
    clip_items: list[dict[str, Any]] = []
    seed_source_review_needed = False
    if not isinstance(clips, list) or not clips:
        errors.append("clips must contain at least one metadata entry")
    else:
        for index, clip in enumerate(clips):
            prefix = f"clips[{index}]"
            if not isinstance(clip, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in [
                "clip_id",
                "source",
                "permission",
                "camera_angle",
                "distance_band",
                "lighting",
                "motion_blur",
            ]:
                if not clip.get(field):
                    errors.append(f"{prefix}.{field} is required")
            clip_permission = str(clip.get("permission") or "unknown")
            if clip_permission in BLOCKED_PERMISSIONS or clip_permission not in ALLOWED_PERMISSIONS:
                errors.append(f"{prefix}.permission is not cleared: {clip_permission}")
            if not _as_set(clip.get("target_capabilities")) & {"apron_required", "harness_required"}:
                errors.append(f"{prefix}.target_capabilities must include apron_required or harness_required")
            if clip.get("clip_id"):
                clip_ids.add(str(clip["clip_id"]))
            clip_items.append(clip)
            seed_source_review_needed = seed_source_review_needed or _clip_requires_seed_source_review(clip)

    if seed_source_review_report is not None:
        loaded_seed_source_review, loaded_seed_source_error = _load_seed_source_review(
            seed_source_review_report
        )
        if loaded_seed_source_review is not None:
            seed_source_review = loaded_seed_source_review
            seed_source_candidates = _seed_source_index(seed_source_review)
            seed_source_suggestions = _seed_source_suggestions(seed_source_review)
        elif seed_source_review_needed:
            seed_source_load_error = loaded_seed_source_error
    elif seed_source_review_needed:
        seed_source_review, seed_source_load_error = _load_seed_source_review(None)

    if seed_source_review_needed:
        seed_import_doc, seed_import_load_error = _load_seed_import_manifest(seed_import_manifest)
        seed_import_entries = _seed_import_index(seed_import_doc)
        if seed_source_review and seed_import_doc and not seed_import_load_error:
            source_review_sha256 = str(seed_import_doc.get("source_review_sha256") or "").strip()
            if not source_review_sha256:
                errors.append("seed import manifest source_review_sha256 is required")
            elif source_review_sha256 != source_review_fingerprint(seed_source_review):
                errors.append("seed import manifest source_review_sha256 does not match seed source review")
            seed_import_review = validate_import_manifest(seed_import_manifest, seed_source_review)
            for error in seed_import_review.get("errors") or []:
                errors.append(f"seed_import_manifest.{error}")
            for blocker in seed_import_review.get("blockers") or []:
                errors.append(f"seed_import_manifest.{blocker}")
        for index, clip in enumerate(clip_items):
            if not _clip_requires_seed_source_review(clip):
                continue
            seed_source_clip_reviews.append(
                _validate_seed_source_clip(
                    clip=clip,
                    prefix=f"clips[{index}]",
                    seed_source_review_report=seed_source_review_report,
                    seed_source_review=seed_source_review,
                    seed_source_candidates=seed_source_candidates,
                    seed_source_load_error=seed_source_load_error,
                    errors=errors,
                )
            )
            seed_import_clip_reviews.append(
                _validate_seed_import_clip(
                    clip=clip,
                    prefix=f"clips[{index}]",
                    seed_import_manifest=seed_import_manifest,
                    seed_import_doc=seed_import_doc,
                    seed_import_entries=seed_import_entries,
                    seed_import_load_error=seed_import_load_error,
                    seed_source_candidates=seed_source_candidates,
                    errors=errors,
                )
            )

    if not schema_only and clip_items:
        clip_backed_coverage = _clip_backed_coverage_values(clip_items)
        for field in COVERAGE_REQUIREMENTS:
            declared_values = _coverage_values(doc, field)
            missing_backing = sorted(declared_values - clip_backed_coverage.get(field, set()))
            if missing_backing:
                errors.append(f"coverage.{field} declares values not backed by clips: {missing_backing}")

    counts_doc = (doc.get("counts") or {}).get("labeled_images_per_class") or {}
    minimum = MIN_COUNTS[mode]
    count_summary = {
        class_name: _count_for_class(counts_doc, class_name)
        for class_name in REQUIRED_CLASSES.values()
    }
    if schema_only:
        for class_name, count in count_summary.items():
            if count < minimum:
                warnings.append(f"{class_name} has {count} labels; {minimum} needed for {mode}")
    else:
        for class_name, count in count_summary.items():
            if count < minimum:
                errors.append(f"{class_name} has {count} labels; {minimum} needed for {mode}")

    raw_label_items = doc.get("yolo_labels") or []
    if raw_label_items and not isinstance(raw_label_items, list):
        errors.append("yolo_labels must be a list")
        raw_label_items = []
    label_items = [item for item in raw_label_items if isinstance(item, dict)]
    if len(label_items) != len(raw_label_items):
        errors.append("yolo_labels entries must be objects")
    label_paths = []
    label_files_per_split = {split: 0 for split in sorted(LABEL_SPLITS)}
    approved_label_images_per_split_per_class = {
        split: {class_name: 0 for class_name in REQUIRED_CLASSES.values()}
        for split in sorted(LABEL_SPLITS)
    }
    source_clip_splits: dict[str, set[str]] = {}
    label_entries: list[tuple[Path, str, bool]] = []
    approved_label_file_count = 0
    for item in label_items:
        if not item.get("path"):
            errors.append("yolo_labels entries must include path")
            continue
        label_path = manifest_path.parent / str(item.get("path"))
        label_paths.append(label_path)
        metadata_ok = schema_only or _validate_label_review_metadata(
            label_item=item,
            label_path=label_path,
            clip_ids=clip_ids,
            errors=errors,
        )
        split = str(item.get("split") or "")
        source_clip_id = str(item.get("source_clip_id") or "")
        label_entries.append((label_path, split, metadata_ok))
        if metadata_ok and split in label_files_per_split:
            label_files_per_split[split] += 1
            if source_clip_id:
                source_clip_splits.setdefault(source_clip_id, set()).add(split)
        if not schema_only and metadata_ok:
            approved_label_file_count += 1
    duplicate_label_paths = sorted(
        str(path) for path in {path for path in label_paths if label_paths.count(path) > 1}
    )
    for label_path in duplicate_label_paths:
        errors.append(f"duplicate yolo_labels path: {label_path}")

    observed_label_images_per_class = {class_name: 0 for class_name in REQUIRED_CLASSES.values()}
    for label_path, split, metadata_ok in label_entries:
        present_classes = _validate_yolo_label(label_path, errors)
        for class_id in present_classes:
            class_name = REQUIRED_CLASSES[class_id]
            observed_label_images_per_class[class_name] += 1
            if metadata_ok and split in approved_label_images_per_split_per_class:
                approved_label_images_per_split_per_class[split][class_name] += 1
    if not label_paths:
        if schema_only:
            warnings.append("no yolo_labels listed; label-file syntax was not checked")
        else:
            errors.append("yolo_labels must list reviewed YOLO label files for strict validation")
    elif not schema_only:
        required_coverage_splits = (
            REQUIRED_PRODUCTION_LABEL_SPLITS if mode == "production" else REQUIRED_STRICT_LABEL_SPLITS
        )
        missing_splits = sorted(
            split for split in required_coverage_splits if label_files_per_split.get(split, 0) == 0
        )
        if missing_splits:
            errors.append(
                f"yolo_labels must include reviewed label files for splits: {', '.join(missing_splits)}"
            )
        missing_split_classes = [
            f"{split}.{class_name}"
            for split in sorted(required_coverage_splits)
            for class_name in REQUIRED_CLASSES.values()
            if approved_label_images_per_split_per_class.get(split, {}).get(class_name, 0) == 0
        ]
        if missing_split_classes:
            errors.append(
                "yolo_labels must include reviewed required-split label coverage for classes: "
                + ", ".join(missing_split_classes)
            )
        minimum_split_class_counts = _minimum_split_class_counts(mode)
        for split, minimum_split_count in sorted(minimum_split_class_counts.items()):
            split_counts = approved_label_images_per_split_per_class.get(split, {})
            for class_name in REQUIRED_CLASSES.values():
                observed_split_count = int(split_counts.get(class_name, 0))
                if observed_split_count < minimum_split_count:
                    errors.append(
                        f"yolo_labels split {split}.{class_name} has {observed_split_count} "
                        f"reviewed labels; {minimum_split_count} needed for {mode}"
                    )
        for source_clip_id, splits in sorted(source_clip_splits.items()):
            if len(splits) > 1:
                errors.append(
                    f"source_clip_id {source_clip_id} appears in multiple dataset splits: "
                    f"{', '.join(sorted(splits))}"
                )
        for class_name, declared_count in count_summary.items():
            observed_count = observed_label_images_per_class.get(class_name, 0)
            if declared_count > observed_count:
                errors.append(
                    f"counts.labeled_images_per_class.{class_name} declares {declared_count} "
                    f"but only {observed_count} label files contain {class_name}"
                )
            elif observed_count > declared_count:
                warnings.append(
                    f"{class_name} appears in {observed_count} label files; manifest declares {declared_count}"
                )

    paths = _dataset_paths(doc)
    capture_deficit = build_capture_deficit(doc, mode=mode)
    return {
        "ok": not errors,
        "mode": mode,
        "schema_only": schema_only,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_paths": paths,
        "required_classes": REQUIRED_CLASSES,
        "counts": count_summary,
        "observed_label_images_per_class": observed_label_images_per_class,
        "approved_label_file_count": approved_label_file_count,
        "unapproved_label_file_count": max(0, len(label_paths) - approved_label_file_count),
        "label_files_per_split": label_files_per_split,
        "approved_label_images_per_split_per_class": approved_label_images_per_split_per_class,
        "minimum_split_class_counts": _minimum_split_class_counts(mode),
        "capture_deficit": capture_deficit,
        "seed_source_suggestions": seed_source_suggestions,
        "seed_source_review": {
            "required": seed_source_review_needed,
            "report": str(seed_source_review_report) if seed_source_review_report else None,
            "ok": None if seed_source_review is None else bool(seed_source_review.get("ok")),
            "gate_passed": None if seed_source_review is None else bool(seed_source_review.get("gate_passed")),
            "clip_count": len(seed_source_clip_reviews),
            "approved_clip_count": sum(
                1 for item in seed_source_clip_reviews if item.get("training_approved") is True
            ),
            "clips": seed_source_clip_reviews,
        },
        "seed_import_manifest": {
            "required": seed_source_review_needed,
            "report": str(seed_import_manifest) if seed_import_manifest else None,
            "ok": None if seed_import_doc is None else seed_import_load_error is None,
            "gate_passed": None if not seed_import_review else seed_import_review.get("gate_passed") is True,
            "source_review_sha256": None
            if seed_import_doc is None
            else seed_import_doc.get("source_review_sha256"),
            "source_review_sha256_matches": None
            if seed_source_review is None or seed_import_doc is None
            else seed_import_doc.get("source_review_sha256") == source_review_fingerprint(seed_source_review),
            "included_count": int(seed_import_review.get("included_count") or 0)
            if seed_import_review
            else 0,
            "approved_count": int(seed_import_review.get("approved_count") or 0)
            if seed_import_review
            else 0,
            "clip_count": len(seed_import_clip_reviews),
            "approved_clip_count": sum(
                1 for item in seed_import_clip_reviews if item.get("import_approved") is True
            ),
            "imports": seed_import_review.get("imports") if seed_import_review else [],
            "errors": seed_import_review.get("errors") if seed_import_review else [],
            "blockers": seed_import_review.get("blockers") if seed_import_review else [],
            "clips": seed_import_clip_reviews,
        },
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate apron/harness PPE dataset capture manifests.")
    parser.add_argument("--manifest", default="", help="Path to apron/harness capture manifest YAML")
    parser.add_argument("--mode", choices=sorted(MIN_COUNTS), default="pilot")
    parser.add_argument("--schema-only", action="store_true", help="Warn instead of failing on count minimums")
    parser.add_argument(
        "--seed-source-review-report",
        default=str(DEFAULT_SEED_SOURCE_REVIEW),
        help=(
            "JSON report from apron_harness_seed_source_doctor.py; required when clips use "
            "public/commercial dataset seed sources"
        ),
    )
    parser.add_argument(
        "--seed-import-manifest",
        default=str(DEFAULT_SEED_IMPORT_MANIFEST),
        help=(
            "Filled apron_harness_seed_import_manifest YAML; required when public/commercial "
            "seed clips are imported into training"
        ),
    )
    parser.add_argument(
        "--emit-yolo-dataset-yaml",
        default="",
        help="Write an Ultralytics-compatible dataset.yaml after strict validation passes",
    )
    parser.add_argument(
        "--emit-capture-work-order",
        default="",
        help="Write a Markdown capture work order from the manifest deficit report",
    )
    parser.add_argument(
        "--emit-capture-matrix-csv",
        default="",
        help="Write a CSV checklist of capture matrix rows from the manifest deficit report",
    )
    parser.add_argument(
        "--emit-label-review-csv",
        default="",
        help="Write a fillable CSV template for reviewed YOLO label-file metadata",
    )
    parser.add_argument(
        "--emit-starter-label-review-csv",
        default="",
        help=(
            "Write a smaller fillable label-review CSV for the immediate apron/harness "
            "starter rows only"
        ),
    )
    parser.add_argument(
        "--import-label-review-csv",
        default="",
        help=(
            "Read a filled label-review CSV and merge approved rows into capture manifest "
            "yolo_labels metadata"
        ),
    )
    parser.add_argument(
        "--validate-label-review-csv",
        default="",
        help=(
            "Validate a filled label-review CSV without writing an updated capture manifest; "
            "fails until at least one approved row passes all gates"
        ),
    )
    parser.add_argument(
        "--import-approved-seed-exports",
        action="store_true",
        help=(
            "Materialize approved entries from --seed-import-manifest into local images/labels "
            "and merge them into the capture manifest"
        ),
    )
    parser.add_argument(
        "--seed-import-output-root",
        default="",
        help=(
            "Directory where materialized seed images/labels are written; defaults to the "
            "--emit-updated-manifest directory"
        ),
    )
    parser.add_argument("--seed-import-camera-angle", default="", help="Camera angle metadata for imported seed clips")
    parser.add_argument("--seed-import-distance-band", default="", help="Distance-band metadata for imported seed clips")
    parser.add_argument("--seed-import-lighting", default="", help="Lighting metadata for imported seed clips")
    parser.add_argument("--seed-import-motion-blur", default="", help="Motion-blur metadata for imported seed clips")
    parser.add_argument(
        "--seed-import-positive-variant-tags",
        default="",
        help="Semicolon-separated positive variant tags for imported seed clips",
    )
    parser.add_argument(
        "--seed-import-hard-negative-tags",
        default="",
        help="Semicolon-separated hard-negative tags for imported seed clips",
    )
    parser.add_argument(
        "--emit-updated-manifest",
        default="",
        help="Write a capture manifest YAML updated from an import action",
    )
    parser.add_argument(
        "--validate-capture-matrix-csv",
        default="",
        help="Validate a filled capture matrix CSV and fail until every row is labeled, approved, and commercially cleared",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a short text summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.manifest and not args.validate_capture_matrix_csv:
        parser.error("--manifest is required unless --validate-capture-matrix-csv is used")

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is not None:
        report = validate_manifest(
            manifest_path,
            mode=args.mode,
            schema_only=args.schema_only,
            seed_source_review_report=Path(args.seed_source_review_report)
            if args.seed_source_review_report
            else None,
            seed_import_manifest=Path(args.seed_import_manifest)
            if args.seed_import_manifest
            else None,
        )
    else:
        report = {
            "ok": True,
            "mode": args.mode,
            "schema_only": args.schema_only,
            "manifest": None,
            "errors": [],
            "warnings": [],
        }
    if args.emit_yolo_dataset_yaml:
        if manifest_path is None:
            report["errors"].append("cannot emit dataset.yaml without --manifest")
            report["ok"] = False
        if args.schema_only:
            report["errors"].append("cannot emit dataset.yaml from --schema-only validation")
            report["ok"] = False
        if report["ok"]:
            out_path = Path(args.emit_yolo_dataset_yaml)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            config = build_yolo_dataset_config(manifest_path)
            out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            report["emitted_yolo_dataset_yaml"] = str(out_path)
    if args.emit_capture_work_order:
        if manifest_path is None:
            report["errors"].append("cannot emit capture work order without --manifest")
            report["ok"] = False
        else:
            out_path = Path(args.emit_capture_work_order)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(render_capture_work_order(report), encoding="utf-8")
            report["emitted_capture_work_order"] = str(out_path)
    if args.emit_capture_matrix_csv:
        if manifest_path is None:
            report["errors"].append("cannot emit capture matrix CSV without --manifest")
            report["ok"] = False
        else:
            out_path = Path(args.emit_capture_matrix_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(render_capture_matrix_csv(report), encoding="utf-8")
            report["emitted_capture_matrix_csv"] = str(out_path)
    if args.emit_label_review_csv:
        if manifest_path is None:
            report["errors"].append("cannot emit label review CSV without --manifest")
            report["ok"] = False
        else:
            out_path = Path(args.emit_label_review_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(render_label_review_csv(report, mode=args.mode), encoding="utf-8")
            report["emitted_label_review_csv"] = str(out_path)
    if args.emit_starter_label_review_csv:
        if manifest_path is None:
            report["errors"].append("cannot emit starter label review CSV without --manifest")
            report["ok"] = False
        else:
            out_path = Path(args.emit_starter_label_review_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(render_starter_label_review_csv(report, mode=args.mode), encoding="utf-8")
            report["emitted_starter_label_review_csv"] = str(out_path)
    label_review_import: dict[str, Any] | None = None
    label_review_validation: dict[str, Any] | None = None
    seed_export_import: dict[str, Any] | None = None
    if args.import_label_review_csv and args.import_approved_seed_exports:
        report["errors"].append(
            "run --import-label-review-csv and --import-approved-seed-exports separately"
        )
        report["ok"] = False
    if args.import_label_review_csv:
        if manifest_path is None:
            report["errors"].append("cannot import label review CSV without --manifest")
            report["ok"] = False
        else:
            label_review_import = import_label_review_csv_into_manifest(
                manifest_path=manifest_path,
                label_review_csv_path=Path(args.import_label_review_csv),
            )
            report["label_review_import"] = {
                key: value
                for key, value in label_review_import.items()
                if key != "updated_manifest"
            }
            if not label_review_import["ok"]:
                report["errors"].extend(label_review_import["errors"])
                report["ok"] = False
    if args.validate_label_review_csv:
        if manifest_path is None:
            report["errors"].append("cannot validate label review CSV without --manifest")
            report["ok"] = False
        else:
            label_review_validation = validate_label_review_csv(
                manifest_path=manifest_path,
                label_review_csv_path=Path(args.validate_label_review_csv),
            )
            report["label_review_validation"] = label_review_validation
            if label_review_validation.get("ok") is not True:
                report["errors"].extend(str(error) for error in label_review_validation.get("errors") or [])
                report["ok"] = False
            if label_review_validation.get("gate_passed") is not True:
                report["ok"] = False
    if args.import_approved_seed_exports:
        if manifest_path is None:
            report["errors"].append("cannot import approved seed exports without --manifest")
            report["ok"] = False
        elif not args.emit_updated_manifest:
            report["errors"].append(
                "cannot import approved seed exports without --emit-updated-manifest"
            )
            report["ok"] = False
        else:
            seed_export_import = import_approved_seed_exports_into_manifest(
                manifest_path=manifest_path,
                seed_source_review_report=Path(args.seed_source_review_report),
                seed_import_manifest=Path(args.seed_import_manifest),
                output_manifest_path=Path(args.emit_updated_manifest),
                output_root=Path(args.seed_import_output_root)
                if args.seed_import_output_root
                else None,
                seed_clip_metadata={
                    "camera_angle": args.seed_import_camera_angle,
                    "distance_band": args.seed_import_distance_band,
                    "lighting": args.seed_import_lighting,
                    "motion_blur": args.seed_import_motion_blur,
                    "positive_variant_tags": args.seed_import_positive_variant_tags,
                    "hard_negative_tags": args.seed_import_hard_negative_tags,
                },
            )
            report["seed_export_import"] = {
                key: value
                for key, value in seed_export_import.items()
                if key != "updated_manifest"
            }
            if not seed_export_import["ok"]:
                report["errors"].extend(seed_export_import["errors"])
                report["warnings"].extend(seed_export_import["warnings"])
                report["ok"] = False
                failure_sidecar_path = _seed_export_import_sidecar_path(
                    Path(args.emit_updated_manifest)
                )
                failure_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                failure_sidecar_payload = build_seed_export_import_sidecar(
                    import_report=seed_export_import,
                    updated_manifest_path=Path(args.emit_updated_manifest),
                )
                failure_sidecar_path.write_text(
                    json.dumps(failure_sidecar_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                report["emitted_seed_export_import_sidecar"] = str(failure_sidecar_path)
    if args.emit_updated_manifest:
        updated_manifest: dict[str, Any] | None = None
        import_kind = ""
        if label_review_import and label_review_import.get("ok"):
            updated_manifest = label_review_import["updated_manifest"]
            import_kind = "label_review"
        elif seed_export_import and seed_export_import.get("ok"):
            updated_manifest = seed_export_import["updated_manifest"]
            import_kind = "seed_export"

        if not args.import_label_review_csv and not args.import_approved_seed_exports:
            report["errors"].append(
                "cannot emit updated manifest without --import-label-review-csv or --import-approved-seed-exports"
            )
            report["ok"] = False
        elif updated_manifest is None:
            report["ok"] = False
        else:
            out_path = Path(args.emit_updated_manifest)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                yaml.safe_dump(updated_manifest, sort_keys=False),
                encoding="utf-8",
            )
            report["emitted_updated_manifest"] = str(out_path)
            updated_manifest_validation_report = validate_manifest(
                out_path,
                mode=args.mode,
                schema_only=False,
                seed_source_review_report=Path(args.seed_source_review_report)
                if args.seed_source_review_report
                else None,
                seed_import_manifest=Path(args.seed_import_manifest)
                if args.seed_import_manifest
                else None,
            )
            updated_manifest_validation = label_review_import_validation_summary(
                updated_manifest_validation_report
            )
            if import_kind == "label_review":
                report["label_review_import"]["updated_manifest_validation"] = updated_manifest_validation
                sidecar_path = _label_review_import_sidecar_path(out_path)
                sidecar_payload = build_label_review_import_sidecar(
                    import_report=label_review_import,
                    updated_manifest_path=out_path,
                    updated_manifest_validation=updated_manifest_validation,
                )
                report_key = "emitted_label_review_import_sidecar"
                validation_error = "updated manifest strict validation failed after label review import"
            else:
                report["seed_export_import"]["updated_manifest_validation"] = updated_manifest_validation
                sidecar_path = _seed_export_import_sidecar_path(out_path)
                sidecar_payload = build_seed_export_import_sidecar(
                    import_report=seed_export_import,
                    updated_manifest_path=out_path,
                    updated_manifest_validation=updated_manifest_validation,
                )
                report_key = "emitted_seed_export_import_sidecar"
                validation_error = "updated manifest strict validation failed after seed export import"
            sidecar_path.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")
            report[report_key] = str(sidecar_path)
            if updated_manifest_validation.get("ok") is not True:
                report["errors"].append(validation_error)
                report["errors"].extend(str(error) for error in updated_manifest_validation.get("errors") or [])
                report["ok"] = False
    if args.validate_capture_matrix_csv:
        progress_report = validate_capture_matrix_progress(
            Path(args.validate_capture_matrix_csv),
            manifest_path=manifest_path,
            mode=args.mode,
        )
        report["capture_matrix_progress"] = progress_report
        if not progress_report["ok"]:
            report["errors"].extend(progress_report["errors"])
            report["ok"] = False
        if not progress_report["gate_passed"]:
            report["ok"] = False
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        subject = report["manifest"] or args.validate_capture_matrix_csv
        print(f"{status}: {subject} ({report['mode']})")
        if report.get("emitted_yolo_dataset_yaml"):
            print(f"Wrote {report['emitted_yolo_dataset_yaml']}")
        if report.get("emitted_capture_work_order"):
            print(f"Wrote {report['emitted_capture_work_order']}")
        if report.get("emitted_capture_matrix_csv"):
            print(f"Wrote {report['emitted_capture_matrix_csv']}")
        if report.get("emitted_label_review_csv"):
            print(f"Wrote {report['emitted_label_review_csv']}")
        if report.get("emitted_starter_label_review_csv"):
            print(f"Wrote {report['emitted_starter_label_review_csv']}")
        if report.get("emitted_updated_manifest"):
            print(f"Wrote {report['emitted_updated_manifest']}")
        if report.get("emitted_label_review_import_sidecar"):
            print(f"Wrote {report['emitted_label_review_import_sidecar']}")
        if report.get("emitted_seed_export_import_sidecar"):
            print(f"Wrote {report['emitted_seed_export_import_sidecar']}")
        imported = report.get("label_review_import") or {}
        if imported:
            print(
                "LABEL_REVIEW_IMPORT: "
                f"imported={imported.get('imported_label_count', 0)} "
                f"skipped={imported.get('skipped_label_count', 0)} "
                f"merged={imported.get('merged_label_count', 0)}"
            )
        validated = report.get("label_review_validation") or {}
        if validated:
            print(
                "LABEL_REVIEW_VALIDATION: "
                f"gate={'pass' if validated.get('gate_passed') else 'blocked'} "
                f"approved={validated.get('imported_label_count', 0)} "
                f"skipped={validated.get('skipped_label_count', 0)} "
                f"invalid={validated.get('invalid_approved_label_count', 0)}"
            )
        seed_imported = report.get("seed_export_import") or {}
        if seed_imported:
            print(
                "SEED_EXPORT_IMPORT: "
                f"imported_labels={seed_imported.get('imported_label_count', 0)} "
                f"imported_clips={seed_imported.get('imported_clip_count', 0)} "
                f"copied_images={seed_imported.get('copied_image_count', 0)} "
                f"skipped={seed_imported.get('skipped_label_count', 0)}"
            )
        progress = report.get("capture_matrix_progress") or {}
        if progress:
            print(
                "CAPTURE_PROGRESS: "
                f"gate={'pass' if progress.get('gate_passed') else 'blocked'} "
                f"rows={progress.get('row_count', 0)} "
                f"ready_rows={progress.get('ready_rows', 0)} "
                f"labeled_examples={progress.get('labeled_examples', 0)} "
                f"missing_labeled_examples={progress.get('missing_labeled_examples', 0)}"
            )
            for blocker in (progress.get("blockers") or [])[:10]:
                print(f"BLOCKED: {blocker}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
        deficit = report.get("capture_deficit") or {}
        if deficit.get("next_capture_batches"):
            print(
                "CAPTURE: "
                f"missing_label_annotations={deficit.get('total_missing_label_annotations', 0)} "
                f"coverage_deficits={deficit.get('coverage_deficit_count', 0)} "
                f"next_batches={len(deficit.get('next_capture_batches') or [])}"
            )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
