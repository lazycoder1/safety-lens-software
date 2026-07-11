"""Tests for closed-set apron/harness training handoff."""

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "scripts" / "apron_harness_train.py"
DATASET_DOCTOR_PATH = ROOT / "scripts" / "apron_harness_dataset_doctor.py"
TEMPLATE_PATH = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml"
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"


def _load_train():
    spec = importlib.util.spec_from_file_location("apron_harness_train", TRAIN_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dataset_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_dataset_doctor", DATASET_DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_yolo_export_zip(path: Path) -> str:
    label_text = "0 0.500 0.500 0.750 0.900\n1 0.520 0.560 0.220 0.450\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", yaml.safe_dump({"names": ["person", "apron"]}, sort_keys=False))
        for split in ("train", "valid", "test"):
            archive.writestr(f"{split}/images/frame_00001.jpg", b"fake image")
            archive.writestr(f"{split}/labels/frame_00001.txt", label_text)
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_capture_matrix_sidecar(matrix_path: Path, manifest_path: Path, mode: str = "production") -> Path:
    rows = list(csv.DictReader(matrix_path.read_text(encoding="utf-8").splitlines()))
    target_labeled_examples = sum(int(row.get("recommended_examples") or 0) for row in rows)
    captured_examples = sum(int(row.get("captured_examples") or 0) for row in rows)
    labeled_examples = sum(int(row.get("labeled_examples") or 0) for row in rows)
    missing_labeled_examples = sum(
        max(0, int(row.get("recommended_examples") or 0) - int(row.get("labeled_examples") or 0))
        for row in rows
    )
    sidecar_path = matrix_path.with_suffix(matrix_path.suffix + ".manifest.json")
    payload = {
        "generated_at": "2026-06-21T00:00:00+00:00",
        "kind": "apron_harness_capture_matrix_manifest",
        "mode": mode,
        "matrix_csv": str(matrix_path),
        "matrix_csv_sha256": _sha256_file(matrix_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "row_count": len(rows),
        "required_label_classes": ["person", "apron", "safety_harness", "safety_lanyard"],
        "next_capture_batches": [
            {
                "batch_id": "unit_test_capture",
                "target_capability": "apron_required",
                "target_labeled_examples": target_labeled_examples,
                "captured_examples": captured_examples,
                "labeled_examples": labeled_examples,
                "missing_labeled_examples": missing_labeled_examples,
                "missing_label_annotations": missing_labeled_examples,
                "row_count": len(rows),
            }
        ],
        "training_gate": {
            "requires_all_rows_ready": True,
            "requires_manifest_reconciliation": True,
            "requires_non_repo_raw_storage_refs": True,
            "requires_permission_approved": True,
        },
        "progress": {
            "gate_passed": True,
            "ready_rows": len(rows),
            "row_count": len(rows),
            "target_labeled_examples": target_labeled_examples,
            "captured_examples": captured_examples,
            "labeled_examples": labeled_examples,
            "missing_labeled_examples": missing_labeled_examples,
            "unapproved_rows": 0,
            "unsafe_storage_rows": 0,
        },
    }
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_label_review_import_sidecar(manifest_path)
    return sidecar_path


def _write_label_review_import_sidecar(manifest_path: Path) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    yolo_labels = manifest.get("yolo_labels") if isinstance(manifest.get("yolo_labels"), list) else []
    counts = (manifest.get("counts") or {}).get("labeled_images_per_class") or {}
    source_manifest_path = manifest_path.with_suffix(".source.yaml")
    if not source_manifest_path.exists():
        source_manifest_path.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    imported_clip_count = len(manifest.get("clips") if isinstance(manifest.get("clips"), list) else [])
    payload = {
        "generated_at": "2026-06-22T00:00:00+00:00",
        "kind": "apron_harness_label_review_import_manifest",
        "valid": True,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "label_review_csv": str(manifest_path.with_suffix(".label_review.csv")),
        "label_review_csv_sha256": "c" * 64,
        "updated_manifest": str(manifest_path),
        "updated_manifest_sha256": _sha256_file(manifest_path),
        "existing_label_count": 0,
        "imported_label_count": len(yolo_labels),
        "imported_clip_count": imported_clip_count,
        "skipped_label_count": 0,
        "invalid_approved_label_count": 0,
        "invalid_clip_metadata_count": 0,
        "merged_label_count": len(yolo_labels),
        "updated_labeled_images_per_class": {
            class_name: int(counts.get(class_name) or 0)
            for class_name in ["person", "apron", "safety_harness", "safety_lanyard"]
        },
        "updated_manifest_validation": {
            "checked": True,
            "mode": "production",
            "schema_only": False,
            "ok": True,
            "manifest_sha256": _sha256_file(manifest_path),
            "approved_label_file_count": len(yolo_labels),
            "label_files_per_split": {
                "test": 100,
                "val": 100,
                "train": max(0, len(yolo_labels) - 200),
            },
            "minimum_split_class_counts": {
                "test": 100,
                "val": 100,
            },
            "counts": {
                class_name: int(counts.get(class_name) or 0)
                for class_name in ["person", "apron", "safety_harness", "safety_lanyard"]
            },
            "errors": [],
            "warnings": [],
        },
        "taxonomy": {
            "version": "apron_harness_v1",
            "classes": {
                0: "person",
                1: "apron",
                2: "safety_harness",
                3: "safety_lanyard",
            },
            "label_format": "ultralytics_yolo_txt_normalized_xywh",
        },
        "training_gate": {
            "requires_approved_label_review_rows": True,
            "requires_review_metadata": True,
            "requires_cleared_permission": True,
            "requires_non_repo_raw_storage_refs": True,
            "requires_recomputed_label_counts": True,
            "requires_updated_manifest_validation": True,
            "requires_source_manifest_sha256_match": True,
            "requires_taxonomy_version_match": True,
            "requires_reviewed_clip_metadata": True,
        },
    }
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def _write_seed_export_import_sidecar(
    *,
    manifest_path: Path,
    seed_review_path: Path,
    seed_import_path: Path,
    seed_review_payload: dict,
    stale_manifest_sha256: bool = False,
    partial_materialization: bool = False,
    include_yolo_preflight: bool = True,
) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    counts = (manifest.get("counts") or {}).get("labeled_images_per_class") or {}
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8")) or {}
    import_entry = seed_import_doc["imports"][0]
    manifest_sha256 = "0" * 64 if stale_manifest_sha256 else _sha256_file(manifest_path)
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".seed_export_import.json")
    required_counts = {
        "apron_required": {"person": 3, "apron": 3},
        "harness_required": {"person": 3, "safety_harness": 3, "safety_lanyard": 3},
    }.get(import_entry["capability"], {})
    payload = {
        "generated_at": "2026-06-22T00:00:00+00:00",
        "kind": "apron_harness_seed_export_import_manifest",
        "valid": True,
        "source_manifest": str(manifest_path.with_suffix(".seed_source.yaml")),
        "source_manifest_sha256": "d" * 64,
        "seed_source_review_report": str(seed_review_path),
        "seed_source_review_sha256": _source_review_fingerprint(seed_review_payload),
        "source_recheck": seed_review_payload.get("source_recheck") or {},
        "seed_import_manifest": str(seed_import_path),
        "seed_import_manifest_sha256": _sha256_file(seed_import_path),
        "updated_manifest": str(manifest_path),
        "updated_manifest_sha256": manifest_sha256,
        "output_root": str(manifest_path.parent),
        "imported_clip_count": 3,
        "imported_label_count": 3,
        "copied_image_count": 3,
        "skipped_label_count": 0,
        "partial_materialization": partial_materialization,
        "updated_labeled_images_per_class": {
            class_name: int(counts.get(class_name) or 0)
            for class_name in ["person", "apron", "safety_harness", "safety_lanyard"]
        },
        "imports": [
            {
                "source_ref": import_entry["source_ref"],
                "capability": import_entry["capability"],
                "raw_export_ref": import_entry["raw_export_ref"],
                "raw_export_sha256": import_entry["raw_export_sha256"],
                "raw_export_local_path": import_entry["raw_export_local_path"],
                "imported_label_count": 3,
                "copied_image_count": 3,
                "errors": [],
                "warnings": [],
                "yolo_export_preflight": {
                    "checked": True,
                    "sha256": import_entry["raw_export_sha256"],
                    "source_classes": sorted(import_entry.get("class_mapping", {}).keys()),
                    "mapped_local_classes": sorted(import_entry.get("class_mapping", {}).values()),
                    "image_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "label_file_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "orphan_label_count_by_split": {},
                    "label_file_count_by_local_class": required_counts,
                    "errors": [],
                    "warnings": [],
                },
            }
        ],
        "updated_manifest_validation": {
            "checked": True,
            "mode": "production",
            "schema_only": False,
            "ok": True,
            "manifest_sha256": manifest_sha256,
            "approved_label_file_count": len(manifest.get("yolo_labels") or []),
            "label_files_per_split": {
                "test": 100,
                "val": 100,
                "train": max(0, len(manifest.get("yolo_labels") or []) - 200),
            },
            "minimum_split_class_counts": {
                "test": 100,
                "val": 100,
            },
            "counts": {
                class_name: int(counts.get(class_name) or 0)
                for class_name in ["person", "apron", "safety_harness", "safety_lanyard"]
            },
            "errors": [],
            "warnings": [],
        },
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
    if not include_yolo_preflight:
        payload["imports"][0].pop("yolo_export_preflight")
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def _write_dataset_yaml(
    path: Path,
    names: dict | None = None,
    source_manifest: Path | str | None = None,
    source_manifest_sha256: str | None = "auto",
    permission: str = "controlled_capture_cleared",
    missing_policy: str = EXPECTED_MISSING_PPE_LABEL_POLICY,
) -> None:
    payload = {
        "path": str(path.parent / "dataset"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names
        if names is not None
        else {
            0: "person",
            1: "apron",
            2: "safety_harness",
            3: "safety_lanyard",
        },
    }
    if source_manifest is not None:
        metadata = {
            "dataset_id": "apron_harness_training_gate_test",
            "source_manifest": str(source_manifest),
            "permission": permission,
            "missing_ppe_label_policy": missing_policy,
        }
        if source_manifest_sha256 == "auto":
            source_manifest_path = Path(source_manifest)
            if source_manifest_path.exists():
                metadata["source_manifest_sha256"] = _sha256_file(source_manifest_path)
        elif source_manifest_sha256 is not None:
            metadata["source_manifest_sha256"] = source_manifest_sha256
        payload["rakshak_lens"] = metadata
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_capture_manifest(path: Path, counts: dict[str, int]) -> None:
    max_count = max((int(value or 0) for value in counts.values()), default=0)
    label_items: list[dict[str, str]] = []
    if max_count > 0:
        for index in range(max_count):
            if index < 100:
                split = "test"
            elif index < 200:
                split = "val"
            else:
                split = "train"
            label_path = path.parent / "labels" / split / f"frame_{index:05d}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            if index < int(counts.get("person") or 0):
                lines.append("0 0.500 0.500 0.750 0.900")
            if index < int(counts.get("apron") or 0):
                lines.append("1 0.520 0.560 0.220 0.450")
            if index < int(counts.get("safety_harness") or 0):
                lines.append("2 0.500 0.500 0.420 0.700")
            if index < int(counts.get("safety_lanyard") or 0):
                lines.append("3 0.700 0.500 0.100 0.500")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            label_items.append(
                {
                    "path": str(label_path.relative_to(path.parent)),
                    "review_status": "approved",
                    "reviewer": "qa_reviewer",
                    "reviewed_at": "2026-06-22T00:00:00+00:00",
                    "source_clip_id": f"factory_apron_{split}_001",
                    "split": split,
                }
            )
    payload = {
        "version": 1,
        "dataset_id": "apron_harness_training_gate_test",
        "license": {
            "permission": "controlled_capture_cleared",
            "contains_identifiable_people": True,
            "raw_storage": "external_private_store",
        },
        "classes": {
            0: "person",
            1: "apron",
            2: "safety_harness",
            3: "safety_lanyard",
        },
        "counts": {"labeled_images_per_class": counts},
        "coverage": {
            "camera_angles": ["front", "side", "elevated_cctv"],
            "distance_bands": ["close", "medium", "wide_surveillance"],
            "lighting": ["indoor_bright", "dim_indoor", "backlit", "glare"],
            "motion_blur": ["low", "medium_or_high"],
            "apron_positive_variants": [
                "denim_apron",
                "work_apron",
                "kitchen_or_food_service_apron",
                "protective_industrial_apron",
                "partial_side_apron",
            ],
            "apron_hard_negative_tags": [
                "safety_vest",
                "jacket",
                "lab_coat",
                "shirt_color_block",
                "tool_belt",
                "loose_cloth_or_scarf",
            ],
            "harness_positive_variants": [
                "full_body_safety_harness",
                "fall_arrest_harness",
                "visible_lanyard_or_tether",
                "harness_over_safety_vest",
                "partially_hidden_harness",
            ],
            "harness_hard_negative_tags": [
                "backpack_straps",
                "tool_belts",
                "seat_belts",
                "ropes_cables_slings_or_hoses",
                "reflective_vest_stripes",
            ],
        },
        "clips": [
            {
                "clip_id": "factory_apron_train_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "front",
                "distance_band": "medium",
                "lighting": "indoor_bright",
                "motion_blur": "low",
                "target_capabilities": ["apron_required"],
            },
            {
                "clip_id": "factory_apron_val_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "side",
                "distance_band": "close",
                "lighting": "dim_indoor",
                "motion_blur": "low",
                "target_capabilities": ["apron_required", "harness_required"],
            },
            {
                "clip_id": "factory_apron_test_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "elevated_cctv",
                "distance_band": "wide_surveillance",
                "lighting": "glare",
                "motion_blur": "medium_or_high",
                "target_capabilities": ["apron_required", "harness_required"],
            },
        ],
        "yolo_labels": label_items,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _seed_source_review_payload(training_usable: bool = True) -> dict:
    return {
        "ok": True,
        "gate_passed": training_usable,
        "candidates": [
            {
                "source_ref": "roboflow_apron_detection",
                "capability": "apron_required",
                "approval_status": "approved_for_training"
                if training_usable
                else "unreviewed",
                "training_usable": training_usable,
                "manifest_import_path": "qa/video_eval/datasets/imported/apron_seed.yaml"
                if training_usable
                else None,
            }
        ],
    }


def _source_review_fingerprint(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_seed_source_review(path: Path, training_usable: bool = True) -> None:
    payload = _seed_source_review_payload(training_usable=training_usable)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_approved_seed_source_review(path: Path, include_source_recheck: bool = True) -> dict:
    reviewed_by = "qa_reviewer"
    reviewed_at = "2026-06-22T00:00:00+00:00"
    review_fields = {
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
    evidence_path = path.with_suffix(".review_evidence.yaml")
    evidence_doc = {
        "version": 1,
        "kind": "apron_harness_seed_source_review_evidence",
        "source_ref": "roboflow_apron_detection",
        "capability": "apron_required",
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "review_items": {
            field: {
                "approved": True,
                "evidence_ref": f"unit-test-reviewed-{field}",
            }
            for field in sorted(review_fields)
        },
    }
    evidence_path.write_text(yaml.safe_dump(evidence_doc, sort_keys=False), encoding="utf-8")
    payload = _seed_source_review_payload(training_usable=True)
    candidate = payload["candidates"][0]
    candidate.update(
        {
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
            "review_evidence_path": str(evidence_path),
            "review_evidence_sha256": _sha256_file(evidence_path),
            "completed_review": {field: True for field in review_fields},
        }
    )
    if include_source_recheck:
        source_recheck_path = path.with_suffix(".source_recheck.md")
        source_recheck_path.write_text(
            "# Unit Test Source Recheck\n\n"
            "Fresh source research evidence only; this does not approve any source for training.\n",
            encoding="utf-8",
        )
        payload["source_recheck"] = {
            "path": str(source_recheck_path),
            "exists": True,
            "sha256": _sha256_file(source_recheck_path),
            "evidence_boundary": (
                "Fresh source research evidence only; this does not approve any source for training "
                "or authorize public export import."
            ),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_seed_import_manifest(path: Path, include_in_training: bool = True) -> None:
    raw_export_sha256 = ""
    raw_export_local_path = ""
    if include_in_training:
        export_zip_path = path.with_suffix(".yolo_export.zip")
        raw_export_sha256 = _write_yolo_export_zip(export_zip_path)
        raw_export_local_path = str(export_zip_path)
    review_fields = {
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
    payload = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": _source_review_fingerprint(_seed_source_review_payload()),
        "imports": [
            {
                "source_ref": "roboflow_apron_detection",
                "capability": "apron_required",
                "include_in_training": include_in_training,
                "review_status": "approved_for_training" if include_in_training else "needs_review",
                "reviewed_by": "qa_reviewer" if include_in_training else "",
                "reviewed_at": "2026-06-22T00:00:00+00:00" if include_in_training else "",
                "manifest_import_path": "qa/video_eval/datasets/imported/apron_seed.yaml"
                if include_in_training
                else "",
                "raw_export_ref": "s3://cleared-seed-exports/apron_seed_yolo_export.zip"
                if include_in_training
                else "",
                "raw_export_sha256": raw_export_sha256 if include_in_training else "",
                "raw_export_local_path": raw_export_local_path if include_in_training else "",
                "export_format": "yolo",
                "completed_review": {field: include_in_training for field in review_fields},
                "class_mapping": {"person": "person", "apron": "apron"} if include_in_training else {},
                "person_box_policy": "Source person boxes were reviewed and reconciled."
                if include_in_training
                else "",
                "hard_negative_policy": "Hard negatives are tracked in the capture manifest."
                if include_in_training
                else "",
                "split_plan": {
                    "train": "reviewed train split",
                    "val": "reviewed validation split",
                    "test": "reviewed held-out test split",
                }
                if include_in_training
                else {"train": "", "val": "", "test": ""},
                "expected_labeled_images_per_class": {
                    "person": 1,
                    "apron": 1,
                    "safety_harness": 0,
                    "safety_lanyard": 0,
                }
                if include_in_training
                else {
                    "person": 0,
                    "apron": 0,
                    "safety_harness": 0,
                    "safety_lanyard": 0,
                },
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_approved_capture_matrix(path: Path, mode: str = "pilot") -> None:
    doctor = _load_dataset_doctor()
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--mode",
            mode,
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(path),
        ]
    )
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["captured_examples"] = row["recommended_examples"]
        row["labeled_examples"] = row["recommended_examples"]
        row["review_status"] = "approved"
        row["permission"] = "controlled_capture_cleared"
        row["raw_storage_ref"] = f"s3://cleared-training-test/{row['row_id']}"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.CAPTURE_MATRIX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_training_plan_accepts_yolo26_candidate(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path)

    plan = trainer.build_training_plan(
        data_path=data_path,
        model="yolo26n",
        device="cpu",
        epochs=10,
        imgsz=640,
        batch=4,
        project=tmp_path / "runs",
        name="unit_test",
        export_formats=["onnx"],
    )

    assert plan["status"] == "ready_to_train"
    assert plan["model"] == "yolo26n.pt"
    assert plan["selected_device"] == "cpu"
    assert plan["train_args"]["data"] == str(data_path)
    assert plan["export_formats"] == ["onnx"]
    assert plan["capture_preflight"]["checked"] is False
    assert plan["source_lineage"]["dataset_yaml"]["file"]["required"] is True
    assert plan["source_lineage"]["dataset_yaml"]["file"]["exists"] is True
    assert plan["source_lineage"]["dataset_yaml"]["file"]["sha256"] == _sha256_file(data_path)
    assert plan["source_lineage"]["capture_manifest"]["file"]["required"] is False


def test_training_plan_rejects_yoloe_for_production_path(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path)

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yoloe-11s-seg.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
        )
    except ValueError as exc:
        assert "model must be one of" in str(exc)
    else:
        raise AssertionError("YOLOE should be rejected for closed-set production training")


def test_training_plan_rejects_legacy_models_for_production_path(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path)

    for model_name in ["yolo11n.pt", "yolov8n.pt"]:
        try:
            trainer.build_training_plan(
                data_path=data_path,
                model=model_name,
                device="cpu",
                epochs=10,
                imgsz=640,
                batch=4,
                project=tmp_path / "runs",
                name="unit_test",
                export_formats=["onnx"],
            )
        except ValueError as exc:
            message = str(exc)
            assert "model must be one of" in message
            assert "yolo26n.pt" in message
            assert "yolo26s.pt" in message
        else:
            raise AssertionError(f"{model_name} should be rejected for new production training")


def test_training_plan_rejects_wrong_class_schema(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path, names={0: "person", 1: "helmet"})

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
        )
    except ValueError as exc:
        assert "dataset names must exactly match" in str(exc)
    else:
        raise AssertionError("wrong class schema should be rejected")


def test_training_dry_run_writes_plan_json(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    out_path = tmp_path / "plan.json"
    _write_dataset_yaml(data_path)

    exit_code = trainer.main([
        "--data",
        str(data_path),
        "--model",
        "yolo26s",
        "--device",
        "cpu",
        "--epochs",
        "5",
        "--out-plan",
        str(out_path),
    ])

    assert exit_code == 0
    assert out_path.exists()
    assert '"model": "yolo26s.pt"' in out_path.read_text(encoding="utf-8")


def test_training_extracts_per_class_metrics_from_ultralytics_summary():
    trainer = _load_train()

    class FakeMetrics:
        def summary(self, normalize: bool = True, decimals: int = 6):
            assert normalize is True
            assert decimals == 6
            return [
                {"Class": "person", "Box-P": 0.91, "Box-R": 0.92, "mAP50": 0.93, "mAP50-95": 0.81},
                {"Class": "apron", "Box-P": 0.82, "Box-R": 0.86, "mAP50": 0.88, "mAP50-95": 0.7},
                {"Class": "safety_harness", "Box-P": 0.84, "Box-R": 0.87, "mAP50": 0.89, "mAP50-95": 0.72},
                {"Class": "safety_lanyard", "Box-P": 0.8, "Box-R": 0.85, "mAP50": 0.86, "mAP50-95": 0.68},
            ]

    metrics = trainer.extract_per_class_metrics(FakeMetrics())

    assert metrics["person"] == {
        "mAP50": 0.93,
        "mAP50_95": 0.81,
        "precision": 0.91,
        "recall": 0.92,
    }
    assert metrics["apron"]["mAP50"] == 0.88
    assert metrics["safety_harness"]["recall"] == 0.87
    assert metrics["safety_lanyard"]["mAP50_95"] == 0.68


def test_training_extracts_per_class_metrics_from_ultralytics_class_result_fallback():
    trainer = _load_train()

    class FakeBox:
        def class_result(self, index: int):
            return {
                0: (0.91, 0.92, 0.93, 0.81),
                1: (0.82, 0.86, 0.88, 0.7),
                2: (0.84, 0.87, 0.89, 0.72),
                3: (0.8, 0.85, 0.86, 0.68),
            }[index]

    class FakeMetrics:
        names = {0: "person", 1: "apron", 2: "safety_harness", 3: "safety_lanyard"}
        ap_class_index = [0, 1, 2, 3]
        box = FakeBox()

        def summary(self, normalize: bool = True, decimals: int = 6):
            raise RuntimeError("summary unavailable")

    metrics = trainer.extract_per_class_metrics(FakeMetrics())

    assert metrics["person"]["precision"] == 0.91
    assert metrics["apron"]["recall"] == 0.86
    assert metrics["safety_harness"]["mAP50"] == 0.89
    assert metrics["safety_lanyard"]["mAP50_95"] == 0.68


def test_training_plan_rejects_required_capture_preflight_without_files(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path)

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "capture preflight is required" in str(exc)
    else:
        raise AssertionError("required capture preflight should fail without files")


def test_training_plan_accepts_matching_capture_preflight(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    plan = trainer.build_training_plan(
        data_path=data_path,
        model="yolo26n.pt",
        device="cpu",
        epochs=10,
        imgsz=640,
        batch=4,
        project=tmp_path / "runs",
        name="unit_test",
        export_formats=["onnx"],
        capture_manifest_path=manifest_path,
        capture_matrix_csv_path=matrix_path,
        capture_preflight_mode="production",
        require_capture_preflight=True,
    )

    assert plan["status"] == "ready_to_train"
    assert plan["capture_preflight"]["mode"] == "production"
    assert plan["capture_preflight"]["checked"] is True
    assert plan["capture_preflight"]["gate_passed"] is True
    assert plan["capture_preflight"]["manifest_reconciliation"]["gate_passed"] is True
    assert plan["capture_preflight"]["capture_matrix_manifest"]["mode"] == "production"
    assert plan["capture_preflight"]["capture_matrix_manifest"]["valid"] is True
    assert plan["capture_preflight"]["capture_matrix_manifest"]["matrix_csv_sha256"] == _sha256_file(matrix_path)
    assert plan["capture_preflight"]["label_review_import_manifest"]["valid"] is True
    assert plan["capture_preflight"]["label_review_import_manifest"]["updated_manifest_sha256"] == _sha256_file(manifest_path)
    assert plan["capture_preflight"]["label_review_import_manifest"]["imported_label_count"] == 2404
    assert plan["capture_preflight"]["label_review_import_manifest"]["imported_clip_count"] == 3
    assert plan["capture_preflight"]["label_review_import_manifest"]["invalid_clip_metadata_count"] == 0
    assert (
        plan["capture_preflight"]["label_review_import_manifest"]["training_gate"]["requires_reviewed_clip_metadata"]
        is True
    )
    assert plan["dataset_provenance"]["declared_source_manifest_sha256"] == _sha256_file(manifest_path)
    assert plan["dataset_provenance"]["source_manifest_sha256"] == _sha256_file(manifest_path)
    assert plan["dataset_provenance"]["permission"] == "controlled_capture_cleared"
    assert plan["dataset_provenance"]["errors"] == []
    assert plan["source_lineage"]["capture_manifest"]["file"]["required"] is True
    assert plan["source_lineage"]["capture_manifest"]["file"]["exists"] is True
    assert plan["source_lineage"]["capture_manifest"]["file"]["sha256"] == _sha256_file(manifest_path)
    assert plan["source_lineage"]["capture_manifest"]["manifest_sha256"] == _sha256_file(manifest_path)
    assert plan["source_lineage"]["capture_manifest"]["mode"] == "production"


def test_training_plan_rejects_public_seed_manifest_without_training_approval(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    _write_seed_source_review(seed_review_path, training_usable=False)

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "capture manifest review failed" in str(exc)
        assert "roboflow_apron_detection is not approved for training capabilities" in str(exc)
    else:
        raise AssertionError("public seed manifest should fail without training-approved source review")


def test_training_plan_rejects_public_seed_manifest_without_import_approval(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    _write_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path, include_in_training=False)

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "capture manifest review failed" in str(exc)
        assert "requires approved seed import manifest entry for apron_required" in str(exc)
    else:
        raise AssertionError("public seed manifest should fail without approved import manifest")


def test_training_plan_rejects_public_seed_manifest_without_source_recheck(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path, include_source_recheck=False)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
    )

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "source recheck lineage invalid" in str(exc)
        assert "source_recheck is required" in str(exc)
    else:
        raise AssertionError("public seed manifest should fail without source recheck lineage")


def test_training_plan_carries_approved_seed_import_export_preflight_lineage(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
    )

    plan = trainer.build_training_plan(
        data_path=data_path,
        model="yolo26n.pt",
        device="cpu",
        epochs=10,
        imgsz=640,
        batch=4,
        project=tmp_path / "runs",
        name="unit_test",
        export_formats=["onnx"],
        capture_manifest_path=manifest_path,
        capture_matrix_csv_path=matrix_path,
        seed_source_review_report=seed_review_path,
        seed_import_manifest=seed_import_path,
        capture_preflight_mode="production",
        require_capture_preflight=True,
    )

    seed_import_gate = plan["source_lineage"]["seed_import_manifest"]["gate"]
    assert seed_import_gate["required"] is True
    assert seed_import_gate["gate_passed"] is True
    assert seed_import_gate["source_review_sha256_matches"] is True
    assert seed_import_gate["approved_count"] == 1
    approved_import = seed_import_gate["imports"][0]
    export_zip_path = seed_import_path.with_suffix(".yolo_export.zip")
    assert approved_import["approved_for_training"] is True
    assert approved_import["raw_export_local_path"] == str(export_zip_path)
    assert approved_import["raw_export_sha256"] == _sha256_file(export_zip_path)
    assert approved_import["yolo_export_preflight"]["checked"] is True
    assert approved_import["yolo_export_preflight"]["sha256"] == _sha256_file(export_zip_path)
    assert approved_import["yolo_export_preflight"]["label_file_count_by_local_class"] == {
        "apron": 3,
        "person": 3,
    }
    seed_export_sidecar = plan["capture_preflight"]["seed_export_import_manifest"]
    assert seed_export_sidecar["valid"] is True
    assert seed_export_sidecar["source_recheck"]["sha256"] == seed_review_payload["source_recheck"]["sha256"]
    assert seed_export_sidecar["updated_manifest_sha256"] == _sha256_file(manifest_path)
    assert seed_export_sidecar["seed_import_manifest_sha256"] == _sha256_file(seed_import_path)
    assert seed_export_sidecar["imported_label_count"] == 3
    assert seed_export_sidecar["imports"][0]["yolo_export_preflight"]["sha256"] == _sha256_file(export_zip_path)
    assert seed_export_sidecar["imports"][0]["yolo_export_preflight"]["orphan_label_count_by_split"] == {}
    assert plan["source_lineage"]["seed_export_import_manifest"]["valid"] is True


def test_training_plan_rejects_seed_export_sidecar_without_source_recheck(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    sidecar_path = _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("source_recheck")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "seed export import sidecar invalid" in str(exc)
        assert "source_recheck is required" in str(exc)
    else:
        raise AssertionError("seed export sidecar should fail without source recheck lineage")


def test_training_plan_rejects_approved_seed_import_without_seed_export_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "seed export import sidecar missing" in str(exc)
    else:
        raise AssertionError("approved public seed imports should require seed export materialization sidecar")


def test_training_plan_rejects_stale_seed_export_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
        stale_manifest_sha256=True,
    )

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "seed export import sidecar invalid" in str(exc)
        assert "updated_manifest_sha256 does not match current capture manifest" in str(exc)
    else:
        raise AssertionError("stale seed export materialization sidecar should block training")


def test_training_plan_rejects_seed_export_sidecar_without_yolo_preflight(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
        include_yolo_preflight=False,
    )

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "seed export import sidecar invalid" in str(exc)
        assert "imports[0].yolo_export_preflight is required" in str(exc)
    else:
        raise AssertionError("seed export materialization sidecar should keep YOLO preflight evidence")


def test_training_plan_rejects_partial_seed_export_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    seed_review_path = tmp_path / "seed_review.json"
    seed_import_path = tmp_path / "seed_import.yaml"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_apron_detection"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    seed_review_payload = _write_approved_seed_source_review(seed_review_path)
    _write_seed_import_manifest(seed_import_path)
    seed_import_doc = yaml.safe_load(seed_import_path.read_text(encoding="utf-8"))
    seed_import_doc["source_review_sha256"] = _source_review_fingerprint(seed_review_payload)
    seed_import_path.write_text(yaml.safe_dump(seed_import_doc, sort_keys=False), encoding="utf-8")
    _write_seed_export_import_sidecar(
        manifest_path=manifest_path,
        seed_review_path=seed_review_path,
        seed_import_path=seed_import_path,
        seed_review_payload=seed_review_payload,
        partial_materialization=True,
    )

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            seed_source_review_report=seed_review_path,
            seed_import_manifest=seed_import_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "seed export import sidecar invalid" in str(exc)
        assert "partial_materialization must be false" in str(exc)
    else:
        raise AssertionError("partial seed export materialization sidecar should block training")


def test_training_plan_rejects_required_preflight_without_dataset_source_manifest(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "rakshak_lens.source_manifest is required" in str(exc)
    else:
        raise AssertionError("required preflight should reject dataset YAML without source_manifest")


def test_training_plan_rejects_dataset_source_manifest_mismatch(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    other_manifest_path = tmp_path / "other_capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    counts = {
        "person": 2404,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    _write_capture_manifest(manifest_path, counts)
    _write_capture_manifest(other_manifest_path, counts)
    _write_dataset_yaml(data_path, source_manifest=other_manifest_path)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "rakshak_lens.source_manifest must match --capture-manifest" in str(exc)
    else:
        raise AssertionError("required preflight should reject mismatched dataset source manifest")


def test_training_plan_rejects_dataset_source_manifest_sha_mismatch(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_dataset_yaml(data_path, source_manifest=manifest_path, source_manifest_sha256="0" * 64)
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "rakshak_lens.source_manifest_sha256 does not match source_manifest file" in str(exc)
    else:
        raise AssertionError("required preflight should reject mismatched dataset source manifest SHA")


def test_training_plan_rejects_dataset_permission_not_cleared(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_dataset_yaml(data_path, source_manifest=manifest_path, permission="research_only")
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "rakshak_lens.permission is not cleared for commercial training: research_only" in str(exc)
    else:
        raise AssertionError("required preflight should reject non-commercial dataset permission")


def test_training_plan_rejects_wrong_missing_ppe_label_policy(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_dataset_yaml(data_path, source_manifest=manifest_path, missing_policy="train_missing_ppe_as_classes")
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "rakshak_lens.missing_ppe_label_policy must be" in str(exc)
    else:
        raise AssertionError("required preflight should reject wrong missing-PPE label policy")


def test_training_plan_rejects_missing_production_capture_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "capture matrix sidecar missing" in str(exc)
    else:
        raise AssertionError("production capture preflight should require a sidecar manifest")


def test_training_plan_rejects_stale_production_capture_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    matrix_path.write_text(matrix_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "matrix_csv_sha256 does not match current capture matrix" in str(exc)
    else:
        raise AssertionError("stale production capture sidecar should block training")


def test_training_plan_rejects_capture_sidecar_progress_count_mismatch(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    sidecar_path = _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["progress"]["target_labeled_examples"] = 1
    sidecar["next_capture_batches"][0]["missing_labeled_examples"] = 1
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        message = str(exc)
        assert "progress.target_labeled_examples" in message
        assert "next_capture_batches.missing_labeled_examples" in message
    else:
        raise AssertionError("capture sidecar count mismatch should block training")


def test_training_plan_rejects_capture_sidecar_gate_passed_mismatch(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    sidecar_path = _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["progress"]["gate_passed"] = False
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "progress.gate_passed False does not match current progress True" in str(exc)
    else:
        raise AssertionError("capture sidecar gate_passed mismatch should block training")


def test_training_plan_rejects_missing_production_label_review_import_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json").unlink()

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "label review import sidecar missing" in str(exc)
    else:
        raise AssertionError("production capture preflight should require a label review import sidecar")


def test_training_plan_rejects_stale_production_label_review_import_sidecar(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    label_sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    label_sidecar = json.loads(label_sidecar_path.read_text(encoding="utf-8"))
    label_sidecar["updated_manifest_sha256"] = "0" * 64
    label_sidecar_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "updated_manifest_sha256 does not match current capture manifest" in str(exc)
    else:
        raise AssertionError("stale label review import sidecar should block training")


def test_training_plan_rejects_failed_label_review_manifest_validation(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    label_sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    label_sidecar = json.loads(label_sidecar_path.read_text(encoding="utf-8"))
    label_sidecar["updated_manifest_validation"]["ok"] = False
    label_sidecar_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "updated_manifest_validation.ok must be true" in str(exc)
    else:
        raise AssertionError("failed reviewed-manifest validation should block training")


def test_training_plan_rejects_label_review_sidecar_without_reviewed_clip_metadata_gate(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    label_sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    label_sidecar = json.loads(label_sidecar_path.read_text(encoding="utf-8"))
    label_sidecar["training_gate"].pop("requires_reviewed_clip_metadata")
    label_sidecar_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "training_gate.requires_reviewed_clip_metadata must be true" in str(exc)
    else:
        raise AssertionError("label-review sidecar without reviewed clip metadata gate should block training")


def test_training_plan_rejects_invalid_label_review_clip_metadata(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    label_sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    label_sidecar = json.loads(label_sidecar_path.read_text(encoding="utf-8"))
    label_sidecar["invalid_clip_metadata_count"] = 1
    label_sidecar_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "invalid_clip_metadata_count must be 0" in str(exc)
    else:
        raise AssertionError("invalid label-review clip metadata should block training")


def test_training_plan_rejects_stale_label_review_source_manifest_sha(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    source_manifest_path = manifest_path.with_suffix(".source.yaml")
    source_manifest_path.write_text("stale: true\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "source_manifest_sha256 does not match source_manifest" in str(exc)
    else:
        raise AssertionError("stale label-review source manifest should block training")


def test_training_plan_rejects_label_review_taxonomy_mismatch(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path, source_manifest=manifest_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
    )
    _write_approved_capture_matrix(matrix_path, mode="production")
    _write_capture_matrix_sidecar(matrix_path, manifest_path, mode="production")
    label_sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".label_review_import.json")
    label_sidecar = json.loads(label_sidecar_path.read_text(encoding="utf-8"))
    label_sidecar["taxonomy"]["version"] = "old_taxonomy"
    label_sidecar["taxonomy"]["label_format"] = "old_format"
    label_sidecar_path.write_text(json.dumps(label_sidecar, indent=2) + "\n", encoding="utf-8")

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        message = str(exc)
        assert "taxonomy.version must be apron_harness_v1" in message
        assert "taxonomy.label_format must be ultralytics_yolo_txt_normalized_xywh" in message
    else:
        raise AssertionError("label-review taxonomy mismatch should block training")


def test_training_plan_rejects_incomplete_capture_preflight(tmp_path: Path):
    trainer = _load_train()
    doctor = _load_dataset_doctor()
    data_path = tmp_path / "dataset.yaml"
    manifest_path = tmp_path / "capture_manifest.yaml"
    matrix_path = tmp_path / "capture_matrix.csv"
    _write_dataset_yaml(data_path)
    _write_capture_manifest(
        manifest_path,
        {
            "person": 0,
            "apron": 0,
            "safety_harness": 0,
            "safety_lanyard": 0,
        },
    )
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--mode",
            "production",
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(matrix_path),
        ]
    )

    try:
        trainer.build_training_plan(
            data_path=data_path,
            model="yolo26n.pt",
            device="cpu",
            epochs=10,
            imgsz=640,
            batch=4,
            project=tmp_path / "runs",
            name="unit_test",
            export_formats=["onnx"],
            capture_manifest_path=manifest_path,
            capture_matrix_csv_path=matrix_path,
            capture_preflight_mode="production",
            require_capture_preflight=True,
        )
    except ValueError as exc:
        assert "capture preflight gate failed" in str(exc)
    else:
        raise AssertionError("incomplete capture preflight should block training")


def test_training_execute_requires_capture_preflight(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    _write_dataset_yaml(data_path)

    exit_code = trainer.main([
        "--data",
        str(data_path),
        "--model",
        "yolo26n",
        "--device",
        "cpu",
        "--execute",
    ])

    assert exit_code == 1


def test_training_execute_failure_writes_out_plan(tmp_path: Path):
    trainer = _load_train()
    data_path = tmp_path / "dataset.yaml"
    out_path = tmp_path / "failed_training_plan.json"
    _write_dataset_yaml(data_path)

    exit_code = trainer.main([
        "--data",
        str(data_path),
        "--model",
        "yolo26n",
        "--device",
        "cpu",
        "--execute",
        "--out-plan",
        str(out_path),
        "--json",
    ])

    assert exit_code == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "capture preflight is required" in payload["error"]


def test_auto_device_prefers_mps_then_cuda_then_cpu():
    trainer = _load_train()

    assert trainer.select_device("auto", {"mps_available": True, "cuda_available": True}) == "mps"
    assert trainer.select_device("auto", {"mps_available": False, "cuda_available": True}) == "cuda"
    assert trainer.select_device("auto", {"mps_available": False, "cuda_available": False}) == "cpu"
