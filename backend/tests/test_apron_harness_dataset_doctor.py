"""Tests for apron/harness PPE capture-pack validation."""

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "apron_harness_dataset_doctor.py"
REVIEW_EVIDENCE_FILE = ROOT / "backend" / "tests" / "fixtures" / "apron_harness_seed_source_review_evidence.yaml"
REVIEW_EVIDENCE_PATH = str(REVIEW_EVIDENCE_FILE.relative_to(ROOT))
TEMPLATE_PATH = ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REVIEW_EVIDENCE_SHA256 = _sha256_file(REVIEW_EVIDENCE_FILE)


def _write_yolo_export_zip(path: Path, capability: str, *, orphan_labels: bool = False) -> str:
    if capability == "harness_required":
        names = ["top", "safety-harness", "lanyard"]
        label_text = (
            "0 0.500 0.500 0.750 0.900\n"
            "1 0.520 0.560 0.220 0.450\n"
            "2 0.700 0.500 0.100 0.500\n"
        )
    else:
        names = ["person", "apron"]
        label_text = "0 0.500 0.500 0.750 0.900\n1 0.520 0.560 0.220 0.450\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", yaml.safe_dump({"names": names}, sort_keys=False))
        for split in ("train", "valid", "test"):
            archive.writestr(f"{split}/images/frame_00001.jpg", b"fake image")
            label_name = "frame_orphan.txt" if orphan_labels else "frame_00001.txt"
            archive.writestr(f"{split}/labels/{label_name}", label_text)
    return _sha256_file(path)


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_dataset_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_manifest(label_path: str = "labels/frame_001.txt") -> dict:
    return {
        "version": 1,
        "dataset_id": "apron_harness_unit_test",
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
        "counts": {
            "labeled_images_per_class": {
                "person": 350,
                "apron": 320,
                "safety_harness": 330,
                "safety_lanyard": 310,
            }
        },
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
                "clip_id": "factory_harness_train_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "side",
                "distance_band": "medium",
                "lighting": "indoor_bright",
                "motion_blur": "low",
                "target_capabilities": ["harness_required"],
                "expected_visible_classes": ["person", "safety_harness", "safety_lanyard"],
                "positive_variant_tags": [
                    "full_body_safety_harness",
                    "visible_lanyard_or_tether",
                    "harness_over_safety_vest",
                ],
                "hard_negative_tags": ["backpack_straps", "tool_belts"],
            },
            {
                "clip_id": "factory_harness_val_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "front",
                "distance_band": "close",
                "lighting": "dim_indoor",
                "motion_blur": "low",
                "target_capabilities": ["apron_required", "harness_required"],
                "expected_visible_classes": ["person", "apron", "safety_harness", "safety_lanyard"],
                "positive_variant_tags": [
                    "denim_apron",
                    "work_apron",
                    "kitchen_or_food_service_apron",
                ],
                "hard_negative_tags": ["safety_vest", "jacket", "lab_coat"],
            },
            {
                "clip_id": "factory_harness_test_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "elevated_cctv",
                "distance_band": "wide_surveillance",
                "lighting": "glare",
                "motion_blur": "medium_or_high",
                "target_capabilities": ["apron_required", "harness_required"],
                "expected_visible_classes": ["person", "apron", "safety_harness", "safety_lanyard"],
                "positive_variant_tags": [
                    "protective_industrial_apron",
                    "partial_side_apron",
                    "fall_arrest_harness",
                    "partially_hidden_harness",
                ],
                "hard_negative_tags": ["seat_belts", "ropes_cables_slings_or_hoses"],
            },
            {
                "clip_id": "factory_harness_backlit_001",
                "source": "controlled_capture",
                "permission": "controlled_capture_cleared",
                "camera_angle": "front",
                "distance_band": "medium",
                "lighting": "backlit",
                "motion_blur": "medium_or_high",
                "target_capabilities": ["apron_required", "harness_required"],
                "expected_visible_classes": ["person", "apron", "safety_harness", "safety_lanyard"],
                "positive_variant_tags": [],
                "hard_negative_tags": [
                    "shirt_color_block",
                    "tool_belt",
                    "loose_cloth_or_scarf",
                    "reflective_vest_stripes",
                ],
            },
        ],
        "yolo_labels": [
            {
                "path": label_path,
                "review_status": "approved",
                "reviewer": "qa_reviewer",
                "reviewed_at": "2026-06-21T00:00:00+00:00",
                "source_clip_id": "factory_harness_train_001",
                "split": "train",
            }
        ],
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _seed_source_review_payload(
    *,
    source_ref: str = "roboflow_safety_harness_dataset",
    capability: str = "harness_required",
    training_usable: bool = True,
) -> dict:
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
    source_recheck_path = ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_recheck_2026_06_24.md"
    return {
        "ok": True,
        "gate_passed": training_usable,
        "source_recheck": {
            "path": str(source_recheck_path),
            "exists": source_recheck_path.exists(),
            "sha256": _sha256_file(source_recheck_path) if source_recheck_path.exists() else "7" * 64,
            "evidence_boundary": (
                "Fresh source research evidence only; this does not approve any source for training "
                "or authorize public export import."
            ),
        },
        "candidates": [
            {
                "source_ref": source_ref,
                "capability": capability,
                "review_priority": 10,
                "review_focus": "Review harness source mapping before import.",
                "approval_status": "approved_for_training"
                if training_usable
                else "unreviewed",
                "training_usable": training_usable,
                "manifest_import_path": "qa/video_eval/datasets/imported/harness_seed.yaml"
                if training_usable
                else None,
                "reviewed_by": "qa_reviewer" if training_usable else "",
                "reviewed_at": "2026-06-22T00:00:00+00:00" if training_usable else "",
                "review_evidence_path": REVIEW_EVIDENCE_PATH if training_usable else "",
                "review_evidence_sha256": REVIEW_EVIDENCE_SHA256 if training_usable else "",
                "completed_review": {field: training_usable for field in review_fields},
            }
        ],
    }


def _source_review_fingerprint(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_seed_source_review(
    path: Path,
    *,
    source_ref: str = "roboflow_safety_harness_dataset",
    capability: str = "harness_required",
    training_usable: bool = True,
) -> None:
    payload = _seed_source_review_payload(
        source_ref=source_ref,
        capability=capability,
        training_usable=training_usable,
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_seed_import_manifest(
    path: Path,
    *,
    source_ref: str = "roboflow_safety_harness_dataset",
    capability: str = "harness_required",
    include_in_training: bool = True,
    raw_export_ref: str = "s3://cleared-seed-exports/seed_yolo_export.zip",
    raw_export_sha256: str = "a" * 64,
    source_review_sha256: str | None = None,
    orphan_labels: bool = False,
) -> None:
    raw_export_local_path = ""
    if include_in_training:
        export_zip_path = path.with_suffix(".yolo_export.zip")
        actual_export_sha256 = _write_yolo_export_zip(
            export_zip_path,
            capability,
            orphan_labels=orphan_labels,
        )
        raw_export_local_path = str(export_zip_path)
        if raw_export_sha256 == "a" * 64:
            raw_export_sha256 = actual_export_sha256
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
    manifest_import_path = (
        "qa/video_eval/datasets/imported/harness_seed.yaml"
        if capability == "harness_required"
        else "qa/video_eval/datasets/imported/apron_seed.yaml"
    )
    payload = {
        "version": 1,
        "kind": "apron_harness_seed_import_manifest",
        "source_review_sha256": source_review_sha256
        or _source_review_fingerprint(
            _seed_source_review_payload(source_ref=source_ref, capability=capability)
        ),
        "imports": [
            {
                "review_priority": 10,
                "review_focus": "Review harness source mapping before import.",
                "source_ref": source_ref,
                "capability": capability,
                "include_in_training": include_in_training,
                "review_status": "approved_for_training" if include_in_training else "needs_review",
                "reviewed_by": "qa_reviewer" if include_in_training else "",
                "reviewed_at": "2026-06-22T00:00:00+00:00" if include_in_training else "",
                "manifest_import_path": manifest_import_path if include_in_training else "",
                "raw_export_ref": raw_export_ref if include_in_training else "",
                "raw_export_sha256": raw_export_sha256 if include_in_training else "",
                "raw_export_local_path": raw_export_local_path if include_in_training else "",
                "export_format": "yolo",
                "completed_review": {field: include_in_training for field in review_fields},
                "class_mapping": (
                    {"top": "person", "safety-harness": "safety_harness", "lanyard": "safety_lanyard"}
                    if capability == "harness_required"
                    else {"person": "person", "apron": "apron"}
                )
                if include_in_training
                else {},
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
                    "apron": 1 if capability == "apron_required" else 0,
                    "safety_harness": 1 if capability == "harness_required" else 0,
                    "safety_lanyard": 1 if capability == "harness_required" else 0,
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


def _write_reviewed_label_files(labels_dir: Path, count: int = 350) -> list[dict[str, str]]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for index in range(count):
        path = labels_dir / f"frame_{index:04d}.txt"
        path.write_text(
            "0 0.500 0.500 0.750 0.900\n"
            "1 0.520 0.560 0.220 0.450\n"
            "2 0.500 0.500 0.420 0.700\n"
            "3 0.700 0.500 0.100 0.500\n",
            encoding="utf-8",
        )
        if index % 7 == 0:
            split = "test"
            source_clip_id = "factory_harness_test_001"
        elif index % 5 == 0:
            split = "val"
            source_clip_id = "factory_harness_val_001"
        else:
            split = "train"
            source_clip_id = "factory_harness_train_001"
        entries.append(
            {
                "path": str(path.relative_to(labels_dir.parent)),
                "review_status": "approved",
                "reviewer": "qa_reviewer",
                "reviewed_at": "2026-06-21T00:00:00+00:00",
                "source_clip_id": source_clip_id,
                "split": split,
            }
        )
    return entries


def _write_label_review_csv(
    path: Path,
    entries: list[dict[str, str]],
    *,
    permission: str = "controlled_capture_cleared",
    raw_storage_ref: str = "s3://cleared-apron-harness/labels",
    clip_metadata_by_source_clip_id: dict[str, dict[str, str]] | None = None,
) -> None:
    doctor = _load_doctor()
    clip_metadata_by_source_clip_id = clip_metadata_by_source_clip_id or {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.LABEL_REVIEW_CSV_FIELDS)
        writer.writeheader()
        for index, entry in enumerate(entries, start=1):
            label_path = entry["path"]
            clip_metadata = clip_metadata_by_source_clip_id.get(entry["source_clip_id"], {})
            writer.writerow(
                {
                    "planned_label_id": f"reviewed_label_{index:04d}",
                    "batch_id": "apron_harness_reviewed_capture",
                    "row_id": "apron_harness_reviewed_capture.reviewed",
                    "target_capability": "apron_required;harness_required",
                    "capture_type": "reviewed_label",
                    "variant_or_tag": "mixed",
                    "required_label_classes": "person;apron;safety_harness;safety_lanyard",
                    "suggested_split": entry["split"],
                    "source_clip_id": entry["source_clip_id"],
                    "image_path": label_path.replace("labels/", "images/").replace(".txt", ".jpg"),
                    "label_path": label_path,
                    "review_status": entry["review_status"],
                    "reviewer": entry["reviewer"],
                    "reviewed_at": entry["reviewed_at"],
                    "permission": permission,
                    "raw_storage_ref": raw_storage_ref,
                    "notes": "",
                    **clip_metadata,
                }
            )


def test_template_manifest_is_schema_valid_with_count_warnings():
    doctor = _load_doctor()

    report = doctor.validate_manifest(TEMPLATE_PATH, schema_only=True)

    assert report["ok"] is True
    assert len(report["manifest_sha256"]) == 64
    assert any("person has 0 labels" in warning for warning in report["warnings"])
    deficit = report["capture_deficit"]
    assert deficit["total_missing_label_annotations"] == 1200
    assert deficit["recommended_label_review_rows"] == 720
    assert deficit["coverage_deficit_count"] == 0
    assert [batch["batch_id"] for batch in deficit["next_capture_batches"]] == [
        "apron_required_closed_set_capture",
        "harness_required_closed_set_capture",
    ]
    assert deficit["next_capture_batches"][0]["minimum_labeled_images"] == 300
    assert deficit["next_capture_batches"][0]["recommended_label_review_rows"] == 360
    assert "safety_vest" in deficit["next_capture_batches"][0]["hard_negative_tags"]
    assert len(deficit["next_capture_batches"][0]["capture_matrix"]) == 11
    assert deficit["next_capture_batches"][0]["capture_matrix"][0]["recommended_examples"] == 60
    assert deficit["next_capture_batches"][0]["capture_matrix"][-1]["capture_type"] == "hard_negative"
    assert deficit["next_capture_batches"][1]["required_label_classes"] == [
        "person",
        "safety_harness",
        "safety_lanyard",
    ]
    assert len(deficit["next_capture_batches"][1]["capture_matrix"]) == 10
    assert deficit["next_capture_batches"][1]["recommended_label_review_rows"] == 360


def test_valid_pilot_capture_manifest_passes(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["observed_label_images_per_class"] == {
        "person": 350,
        "apron": 350,
        "safety_harness": 350,
        "safety_lanyard": 350,
    }
    assert report["approved_label_file_count"] == 350
    assert report["unapproved_label_file_count"] == 0
    assert report["label_files_per_split"]["train"] > 0
    assert report["label_files_per_split"]["val"] > 0
    assert report["label_files_per_split"]["test"] > 0
    assert report["approved_label_images_per_split_per_class"]["train"]["apron"] > 0
    assert report["approved_label_images_per_split_per_class"]["val"]["safety_harness"] > 0
    assert report["approved_label_images_per_split_per_class"]["val"]["safety_lanyard"] > 0
    assert report["capture_deficit"]["total_missing_label_annotations"] == 0
    assert report["capture_deficit"]["next_capture_batches"] == []


def test_capture_deficit_reports_label_and_coverage_gaps(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 310,
        "apron": 125,
        "safety_harness": 300,
        "safety_lanyard": 90,
    }
    manifest["coverage"]["camera_angles"] = ["front"]
    manifest["coverage"]["apron_positive_variants"] = ["denim_apron"]
    manifest["coverage"]["harness_hard_negative_tags"] = ["backpack_straps"]
    manifest["yolo_labels"] = []
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, schema_only=True)

    assert report["ok"] is False
    deficit = report["capture_deficit"]
    assert deficit["missing_label_minimums"]["apron"]["missing"] == 175
    assert deficit["missing_label_minimums"]["safety_lanyard"]["missing"] == 210
    assert "side" in deficit["missing_coverage"]["camera_angles"]
    assert "work_apron" in deficit["missing_coverage"]["apron_positive_variants"]
    assert "tool_belts" in deficit["missing_coverage"]["harness_hard_negative_tags"]
    batches = {batch["batch_id"]: batch for batch in deficit["next_capture_batches"]}
    assert batches["apron_required_closed_set_capture"]["minimum_labeled_images"] == 175
    assert batches["harness_required_closed_set_capture"]["minimum_labeled_images"] == 210
    assert batches["apron_required_closed_set_capture"]["capture_matrix"][0]["recommended_examples"] == 44
    harness_matrix = batches["harness_required_closed_set_capture"]["capture_matrix"]
    assert any(row["variant_or_tag"] == "tool_belts" for row in harness_matrix)


def test_emit_capture_work_order_from_template_schema_only(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_work_order.md"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-work-order",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rendered = output_path.read_text(encoding="utf-8")
    assert "# Apron/Harness Capture Work Order" in rendered
    assert "Missing label annotations: `1200`" in rendered
    assert "Recommended label-review rows: `720`" in rendered
    assert "Recommended label-review rows: `360`" in rendered
    assert "Label annotations are per-class coverage counts" in rendered
    assert "Coverage deficits: `0`" in rendered
    assert "Manifest SHA256:" in rendered
    assert "| Capture Type | Variant / Tag | Recommended Examples | Required Labels |" in rendered
    assert "| `positive_variant` | `denim_apron` | 60 | person, apron |" in rendered
    assert "| `hard_negative` | `tool_belts` | 12 | person |" in rendered
    assert "`apron_required_closed_set_capture`" in rendered
    assert "`harness_required_closed_set_capture`" in rendered
    assert "review_status=approved" in rendered
    assert "--mode production --import-label-review-csv" in rendered
    assert "--schema-only --import-label-review-csv" not in rendered
    assert "strict, non-schema-only validation" in rendered
    assert "Do not train or promote a production model" in rendered


def test_apron_harness_dataset_plan_uses_strict_label_review_import():
    plan = (ROOT / "docs/plan/apron-harness-closed-set-ppe-dataset-plan.md").read_text(
        encoding="utf-8"
    )

    assert "--import-label-review-csv /path/to/filled/apron_harness_label_review.csv" in plan
    assert "--schema-only \\\n  --import-label-review-csv" not in plan


def test_capture_work_order_includes_seed_source_review_queue(tmp_path: Path):
    doctor = _load_doctor()
    review_path = tmp_path / "seed_source_review.json"
    review_payload = {
        "ok": True,
        "gate_passed": False,
        "candidates": [
            {
                "source_ref": "roboflow_kitchen_hygiene",
                "capability": "apron_required",
                "review_priority": 20,
                "approval_status": "unreviewed",
                "training_usable": False,
                "review_focus": "Review apron/no-apron taxonomy first.",
                "url": "https://example.test/kitchen",
                "blocker": "needs_export_terms_review",
            },
            {
                "source_ref": "roboflow_work_at_height_safety",
                "capability": "harness_required",
                "review_priority": 10,
                "approval_status": "unreviewed",
                "training_usable": False,
                "review_focus": "Review harness/person/lanyard mapping first.",
                "url": "https://example.test/work-at-height",
                "blocker": "needs_provenance_review",
            },
        ],
    }
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")

    report = doctor.validate_manifest(
        TEMPLATE_PATH,
        schema_only=True,
        seed_source_review_report=review_path,
    )
    rendered = doctor.render_capture_work_order(report)

    assert report["seed_source_suggestions"][0]["source_ref"] == "roboflow_work_at_height_safety"
    assert "## Public Seed Source Review Queue" in rendered
    assert "These public candidates are not training-approved yet." in rendered
    assert rendered.index("roboflow_work_at_height_safety") < rendered.index("roboflow_kitchen_hygiene")
    assert "Review harness/person/lanyard mapping first." in rendered
    assert "needs_export_terms_review" in rendered
    assert "--emit-starter-label-review-csv qa/video_eval/results/apron_harness_starter_label_review_template.csv" in rendered
    assert "--emit-starter-label-review-csv qa/video_eval/results/apron_harness_production_starter_label_review_template.csv" in rendered
    assert "--schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered
    assert "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered
    assert "--schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in rendered
    assert "apron_harness_capture_manifest.starter_reviewed.yaml" in rendered
    assert "A starter import can produce an intermediate reviewed manifest and sidecar" in rendered
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in rendered
    assert ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in rendered
    assert '--review-bundle-out ""' not in rendered
    assert rendered.index("--validate-review-bundle") < rendered.index("--import-approved-seed-exports")
    assert "source-review bundle hash validation" in rendered


def test_emit_capture_matrix_csv_from_template_schema_only(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 21
    assert rows[0]["batch_id"] == "apron_required_closed_set_capture"
    assert rows[0]["capture_type"] == "positive_variant"
    assert rows[0]["variant_or_tag"] == "denim_apron"
    assert rows[0]["recommended_examples"] == "60"
    assert rows[0]["required_label_classes"] == "person;apron"
    assert rows[0]["captured_examples"] == "0"
    assert rows[0]["labeled_examples"] == "0"
    assert rows[0]["review_status"] == "not_started"
    assert rows[0]["permission"] == "unknown"
    assert rows[0]["raw_storage_ref"] == ""
    assert rows[-1]["batch_id"] == "harness_required_closed_set_capture"
    assert rows[-1]["capture_type"] == "hard_negative"
    assert rows[-1]["variant_or_tag"] == "tool_belts"
    assert rows[-1]["recommended_examples"] == "12"

    progress = doctor.validate_capture_matrix_progress(output_path)
    assert progress["ok"] is True
    assert progress["gate_passed"] is False
    assert progress["row_count"] == 21
    assert progress["ready_rows"] == 0
    assert progress["missing_labeled_examples"] == 720
    assert progress["unapproved_rows"] == 21
    assert progress["unsafe_storage_rows"] == 21
    assert progress["manifest_reconciliation"]["checked"] is False


def test_emit_label_review_csv_from_template_schema_only(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_label_review.csv"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-label-review-csv",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 720
    assert rows[0]["planned_label_id"] == "apron_required_closed_set_capture.positive.denim_apron.label_0001"
    assert rows[0]["batch_id"] == "apron_required_closed_set_capture"
    assert rows[0]["required_label_classes"] == "person;apron"
    assert rows[0]["required_label_class_ids"] == "0;1"
    assert rows[0]["suggested_split"] == "val"
    assert rows[0]["label_path"].startswith("labels/val/")
    assert rows[0]["review_status"] == "not_started"
    assert rows[0]["source_clip_id"] == ""
    assert rows[0]["source_manifest_sha256"] == doctor._sha256_file(TEMPLATE_PATH)
    assert rows[0]["taxonomy_version"] == doctor.TAXONOMY_VERSION
    assert rows[0]["taxonomy_class_ids"] == "0:person;1:apron;2:safety_harness;3:safety_lanyard"
    assert rows[0]["label_format"] == doctor.YOLO_LABEL_FORMAT
    assert rows[0]["required_review_status"] == "approved"
    assert rows[0]["requires_non_repo_raw_storage_ref"] == "true"
    assert rows[0]["import_sidecar_required"] == "true"
    assert rows[0]["clip_source"] == "controlled_capture"
    assert rows[0]["clip_camera_angle"] == ""
    assert rows[0]["clip_expected_visible_classes"] == "person;apron"
    assert rows[0]["clip_positive_variant_tags"] == "denim_apron"
    assert rows[0]["clip_hard_negative_tags"] == ""
    assert "source_clip_id" in rows[0]["clip_notes"]
    assert rows[-1]["batch_id"] == "harness_required_closed_set_capture"
    assert rows[-1]["suggested_split"] == "train"
    split_counts = {split: 0 for split in doctor.LABEL_SPLITS}
    for row in rows:
        split_counts[row["suggested_split"]] += 1
    assert split_counts["val"] > 0
    assert split_counts["train"] > split_counts["val"]
    assert split_counts["test"] == 0


def test_emit_starter_label_review_csv_from_template_schema_only(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_starter_label_review.csv"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-starter-label-review-csv",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 240
    row_ids = {row["row_id"] for row in rows}
    assert row_ids == doctor.STARTER_LABEL_REVIEW_ROW_IDS
    assert "apron_required_closed_set_capture.positive.denim_apron" in row_ids
    assert "harness_required_closed_set_capture.positive.fall_arrest_harness" in row_ids
    assert {row["suggested_split"] for row in rows} == {"train", "val"}
    assert {row["review_status"] for row in rows} == {"not_started"}


def test_emit_production_label_review_csv_includes_holdout_splits(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_production_label_review.csv"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--mode",
            "production",
            "--schema-only",
            "--emit-label-review-csv",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2404
    split_counts = {split: 0 for split in doctor.LABEL_SPLITS}
    for row in rows:
        split_counts[row["suggested_split"]] += 1
    assert split_counts["test"] > 0
    assert split_counts["val"] > 0
    assert split_counts["train"] > split_counts["val"]
    assert rows[0]["suggested_split"] == "test"


def test_emit_production_starter_label_review_csv_includes_holdout_splits(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_production_starter_label_review.csv"

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--mode",
            "production",
            "--schema-only",
            "--emit-starter-label-review-csv",
            str(output_path),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 800
    assert {row["row_id"] for row in rows} == doctor.STARTER_LABEL_REVIEW_ROW_IDS
    split_counts = {split: 0 for split in doctor.LABEL_SPLITS}
    for row in rows:
        split_counts[row["suggested_split"]] += 1
    assert split_counts["test"] > 0
    assert split_counts["val"] > 0
    assert split_counts["train"] > split_counts["val"]


def test_import_label_review_csv_emits_strict_valid_manifest(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels")
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--schema-only",
            "--import-label-review-csv",
            str(label_review_path),
            "--emit-updated-manifest",
            str(updated_manifest_path),
        ]
    )

    assert exit_code == 0
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".label_review_import.json"
    )
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["kind"] == "apron_harness_label_review_import_manifest"
    assert sidecar["source_manifest"] == str(manifest_path)
    assert sidecar["source_manifest_sha256"] == _sha256_file(manifest_path)
    assert sidecar["label_review_csv"] == str(label_review_path)
    assert sidecar["label_review_csv_sha256"] == _sha256_file(label_review_path)
    assert sidecar["updated_manifest"] == str(updated_manifest_path)
    assert sidecar["updated_manifest_sha256"] == _sha256_file(updated_manifest_path)
    assert sidecar["valid"] is True
    assert sidecar["imported_label_count"] == 350
    assert sidecar["taxonomy"]["version"] == doctor.TAXONOMY_VERSION
    assert sidecar["taxonomy"]["classes"] == {
        "0": "person",
        "1": "apron",
        "2": "safety_harness",
        "3": "safety_lanyard",
    }
    assert sidecar["taxonomy"]["label_format"] == doctor.YOLO_LABEL_FORMAT
    assert sidecar["label_review_csv_schema"]["required_import_fields"] == doctor.LABEL_REVIEW_IMPORT_REQUIRED_FIELDS
    assert "source_manifest_sha256" in sidecar["label_review_csv_schema"]["generated_guidance_fields"]
    assert sidecar["training_gate"]["requires_recomputed_label_counts"] is True
    assert sidecar["training_gate"]["requires_updated_manifest_validation"] is True
    assert sidecar["training_gate"]["requires_source_manifest_sha256_match"] is True
    assert sidecar["training_gate"]["requires_taxonomy_version_match"] is True
    assert sidecar["updated_manifest_validation"]["checked"] is True
    assert sidecar["updated_manifest_validation"]["mode"] == "pilot"
    assert sidecar["updated_manifest_validation"]["schema_only"] is False
    assert sidecar["updated_manifest_validation"]["ok"] is True
    assert sidecar["updated_manifest_validation"]["manifest_sha256"] == _sha256_file(updated_manifest_path)
    updated_manifest = yaml.safe_load(updated_manifest_path.read_text(encoding="utf-8"))
    assert len(updated_manifest["yolo_labels"]) == 350
    assert updated_manifest["counts"]["labeled_images_per_class"] == {
        "person": 350,
        "apron": 350,
        "safety_harness": 350,
        "safety_lanyard": 350,
    }
    first_label = updated_manifest["yolo_labels"][0]
    assert first_label["review_status"] == "approved"
    assert first_label["permission"] == "controlled_capture_cleared"
    assert first_label["raw_storage_ref"] == "s3://cleared-apron-harness/labels"

    strict_report = doctor.validate_manifest(updated_manifest_path)
    assert strict_report["ok"] is True
    assert strict_report["approved_label_file_count"] == 350


def test_validate_label_review_csv_cli_passes_for_filled_review(tmp_path: Path, capsys):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    manifest_path = tmp_path / "source_manifest.yaml"
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    _write_manifest(manifest_path, manifest)
    _write_label_review_csv(label_review_path, _write_reviewed_label_files(tmp_path / "labels"))

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--schema-only",
            "--validate-label-review-csv",
            str(label_review_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in output
    report = doctor.validate_label_review_csv(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )
    assert report["gate_passed"] is True
    assert report["imported_label_count"] == 350
    assert "updated_manifest" not in report


def test_validate_label_review_csv_cli_blocks_blank_starter_template(tmp_path: Path, capsys):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_starter_label_review.csv"
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-starter-label-review-csv",
            str(output_path),
        ]
    )

    exit_code = doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--validate-label-review-csv",
            str(output_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "LABEL_REVIEW_VALIDATION: gate=blocked" in output
    report = doctor.validate_label_review_csv(
        manifest_path=TEMPLATE_PATH,
        label_review_csv_path=output_path,
    )
    assert report["gate_passed"] is False
    assert report["imported_label_count"] == 0
    assert "no approved label-review rows" in "\n".join(report["warnings"])


def test_import_label_review_csv_creates_reviewed_clips_from_row_metadata(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels")
    source_rewrites = {
        "factory_harness_train_001": "reviewed_capture_train_001",
        "factory_harness_val_001": "reviewed_capture_val_001",
        "factory_harness_test_001": "reviewed_capture_test_001",
    }
    for entry in label_entries:
        entry["source_clip_id"] = source_rewrites[entry["source_clip_id"]]
    clip_metadata = {
        "reviewed_capture_train_001": {
            "clip_source": "controlled_capture",
            "clip_camera_angle": "side",
            "clip_distance_band": "medium",
            "clip_lighting": "indoor_bright",
            "clip_motion_blur": "low",
            "clip_expected_visible_classes": "person;apron;safety_harness;safety_lanyard",
            "clip_positive_variant_tags": "work_apron;full_body_safety_harness",
            "clip_hard_negative_tags": "safety_vest;backpack_straps",
            "clip_notes": "Reviewed train clip metadata.",
        },
        "reviewed_capture_val_001": {
            "clip_source": "controlled_capture",
            "clip_camera_angle": "front",
            "clip_distance_band": "close",
            "clip_lighting": "dim_indoor",
            "clip_motion_blur": "low",
            "clip_expected_visible_classes": "person;apron;safety_harness;safety_lanyard",
            "clip_positive_variant_tags": "denim_apron;visible_lanyard_or_tether",
            "clip_hard_negative_tags": "jacket;tool_belts",
            "clip_notes": "Reviewed validation clip metadata.",
        },
        "reviewed_capture_test_001": {
            "clip_source": "controlled_capture",
            "clip_camera_angle": "elevated_cctv",
            "clip_distance_band": "wide_surveillance",
            "clip_lighting": "glare",
            "clip_motion_blur": "medium_or_high",
            "clip_expected_visible_classes": "person;apron;safety_harness;safety_lanyard",
            "clip_positive_variant_tags": "protective_industrial_apron;fall_arrest_harness",
            "clip_hard_negative_tags": "shirt_color_block;seat_belts",
            "clip_notes": "Reviewed held-out clip metadata.",
        },
    }
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(
        label_review_path,
        label_entries,
        clip_metadata_by_source_clip_id=clip_metadata,
    )
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--schema-only",
            "--import-label-review-csv",
            str(label_review_path),
            "--emit-updated-manifest",
            str(updated_manifest_path),
        ]
    )

    assert exit_code == 0
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".label_review_import.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["valid"] is True
    assert sidecar["imported_clip_count"] == 3
    assert sidecar["invalid_clip_metadata_count"] == 0
    assert sidecar["training_gate"]["requires_reviewed_clip_metadata"] is True
    updated_manifest = yaml.safe_load(updated_manifest_path.read_text(encoding="utf-8"))
    clips_by_id = {clip["clip_id"]: clip for clip in updated_manifest["clips"]}
    assert clips_by_id["reviewed_capture_train_001"]["camera_angle"] == "side"
    assert clips_by_id["reviewed_capture_val_001"]["expected_visible_classes"] == [
        "person",
        "apron",
        "safety_harness",
        "safety_lanyard",
    ]
    assert "fall_arrest_harness" in clips_by_id["reviewed_capture_test_001"]["positive_variant_tags"]
    strict_report = doctor.validate_manifest(updated_manifest_path)
    assert strict_report["ok"] is True


def test_import_label_review_csv_rejects_new_clip_without_metadata(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    label_entries[0]["source_clip_id"] = "reviewed_capture_missing_metadata_001"
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    _write_manifest(manifest_path, manifest)

    report = doctor.import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["invalid_clip_metadata_count"] == 1
    assert any("clip_camera_angle is required" in error for error in report["errors"])
    assert any("source_clip_id is not listed in clips" in error for error in report["errors"])


def test_import_label_review_csv_records_failed_strict_validation(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--schema-only",
            "--import-label-review-csv",
            str(label_review_path),
            "--emit-updated-manifest",
            str(updated_manifest_path),
        ]
    )

    assert exit_code == 1
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".label_review_import.json"
    )
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["valid"] is False
    assert sidecar["updated_manifest_validation"]["checked"] is True
    assert sidecar["updated_manifest_validation"]["ok"] is False
    assert any("300 needed for pilot" in error for error in sidecar["updated_manifest_validation"]["errors"])


def test_import_label_review_csv_production_mode_records_production_validation(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1000)
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--mode",
            "production",
            "--schema-only",
            "--import-label-review-csv",
            str(label_review_path),
            "--emit-updated-manifest",
            str(updated_manifest_path),
        ]
    )

    assert exit_code == 0
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".label_review_import.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["valid"] is True
    assert sidecar["updated_manifest_validation"]["mode"] == "production"
    assert sidecar["updated_manifest_validation"]["schema_only"] is False
    assert sidecar["updated_manifest_validation"]["ok"] is True
    assert sidecar["updated_manifest_validation"]["minimum_split_class_counts"] == {
        "test": 100,
        "val": 100,
    }


def test_import_approved_seed_exports_materializes_yolo_export(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest_path = tmp_path / "source_manifest.yaml"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.reviewed.yaml"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path)
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--seed-source-review-report",
            str(review_path),
            "--seed-import-manifest",
            str(import_path),
            "--import-approved-seed-exports",
            "--emit-updated-manifest",
            str(updated_manifest_path),
            "--seed-import-camera-angle",
            "front",
            "--seed-import-distance-band",
            "medium",
            "--seed-import-lighting",
            "indoor_bright",
            "--seed-import-motion-blur",
            "low",
            "--seed-import-positive-variant-tags",
            "full_body_safety_harness;visible_lanyard_or_tether",
        ]
    )

    assert exit_code == 0
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".seed_export_import.json"
    )
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["kind"] == "apron_harness_seed_export_import_manifest"
    assert sidecar["valid"] is True
    assert sidecar["source_manifest"] == str(manifest_path)
    assert sidecar["seed_import_manifest"] == str(import_path)
    assert sidecar["source_recheck"]["path"].endswith(
        "apron_harness_source_recheck_2026_06_24.md"
    )
    assert len(sidecar["source_recheck"]["sha256"]) == 64
    assert "does not approve" in sidecar["source_recheck"]["evidence_boundary"]
    assert sidecar["imported_label_count"] == 3
    assert sidecar["copied_image_count"] == 3
    assert sidecar["training_gate"]["requires_yolo_export_preflight"] is True
    assert sidecar["updated_manifest_validation"]["ok"] is True
    sidecar_preflight = sidecar["imports"][0]["yolo_export_preflight"]
    assert sidecar_preflight["checked"] is True
    assert sidecar_preflight["orphan_label_count_by_split"] == {}
    assert sidecar_preflight["label_file_count_by_local_class"] == {
        "person": 3,
        "safety_harness": 3,
        "safety_lanyard": 3,
    }

    updated_manifest = yaml.safe_load(updated_manifest_path.read_text(encoding="utf-8"))
    seed_labels = [
        item
        for item in updated_manifest["yolo_labels"]
        if item.get("source_ref") == "roboflow_safety_harness_dataset"
    ]
    assert len(seed_labels) == 3
    assert {item["split"] for item in seed_labels} == {"train", "val", "test"}
    assert updated_manifest["counts"]["labeled_images_per_class"] == {
        "person": 353,
        "apron": 350,
        "safety_harness": 353,
        "safety_lanyard": 353,
    }
    for item in seed_labels:
        label_path = updated_manifest_path.parent / item["path"]
        image_path = updated_manifest_path.parent / item["image_path"]
        assert label_path.exists()
        assert image_path.exists()
        class_ids = {line.split()[0] for line in label_path.read_text(encoding="utf-8").splitlines()}
        assert class_ids == {"0", "2", "3"}
        assert item["review_status"] == "approved"
        assert item["raw_storage_ref"] == "s3://cleared-seed-exports/seed_yolo_export.zip"

    seed_clips = [
        item
        for item in updated_manifest["clips"]
        if item.get("source_ref") == "roboflow_safety_harness_dataset"
    ]
    assert len(seed_clips) == 3
    assert all(item["source"] == "public_seed_source" for item in seed_clips)
    assert all(item["permission"] == "commercial_dataset_approved" for item in seed_clips)
    assert all(item["camera_angle"] == "front" for item in seed_clips)

    strict_report = doctor.validate_manifest(
        updated_manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )
    assert strict_report["ok"] is True
    assert strict_report["seed_import_manifest"]["approved_clip_count"] == 3


def test_import_approved_seed_exports_rejects_orphan_yolo_labels(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest_path = tmp_path / "source_manifest.yaml"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.reviewed.yaml"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, orphan_labels=True)
    _write_manifest(manifest_path, manifest)

    report = doctor.import_approved_seed_exports_into_manifest(
        manifest_path=manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
        output_manifest_path=updated_manifest_path,
        seed_clip_metadata={
            "camera_angle": "front",
            "distance_band": "medium",
            "lighting": "indoor_bright",
            "motion_blur": "low",
        },
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["copied_image_count"] == 0
    assert report["updated_manifest"] is None
    errors = "\n".join(report["errors"])
    assert "seed_import_manifest.imports[0].YOLO export train labels must have matching images" in errors
    assert "seed_import_manifest.imports[0].YOLO export valid labels must have matching images" in errors
    assert "seed_import_manifest.imports[0].YOLO export test labels must have matching images" in errors
    import_preflight = report["imports"][0]["yolo_export_preflight"]
    assert import_preflight["orphan_label_count_by_split"] == {
        "test": 1,
        "train": 1,
        "valid": 1,
    }
    assert not (tmp_path / "labels" / "train" / "seed").exists()
    assert not (tmp_path / "images" / "train" / "seed").exists()


def test_import_approved_seed_exports_rejects_unapproved_seed_import(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["yolo_labels"] = []
    manifest_path = tmp_path / "source_manifest.yaml"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.reviewed.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, include_in_training=False)
    _write_manifest(manifest_path, manifest)

    report = doctor.import_approved_seed_exports_into_manifest(
        manifest_path=manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
        output_manifest_path=updated_manifest_path,
        seed_clip_metadata={
            "camera_angle": "front",
            "distance_band": "medium",
            "lighting": "indoor_bright",
            "motion_blur": "low",
        },
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["updated_manifest"] is None
    assert any("include_in_training=false" in error for error in report["errors"])
    assert not (tmp_path / "labels" / "train" / "seed").exists()


def test_import_approved_seed_exports_cli_writes_failure_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest_path = tmp_path / "source_manifest.yaml"
    updated_manifest_path = tmp_path / "apron_harness_capture_manifest.reviewed.yaml"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, include_in_training=False)
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main(
        [
            "--manifest",
            str(manifest_path),
            "--seed-source-review-report",
            str(review_path),
            "--seed-import-manifest",
            str(import_path),
            "--import-approved-seed-exports",
            "--emit-updated-manifest",
            str(updated_manifest_path),
            "--seed-import-camera-angle",
            "front",
            "--seed-import-distance-band",
            "medium",
            "--seed-import-lighting",
            "indoor_bright",
            "--seed-import-motion-blur",
            "low",
        ]
    )

    assert exit_code == 1
    assert not updated_manifest_path.exists()
    sidecar_path = updated_manifest_path.with_suffix(
        updated_manifest_path.suffix + ".seed_export_import.json"
    )
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["kind"] == "apron_harness_seed_export_import_manifest"
    assert sidecar["valid"] is False
    assert sidecar["updated_manifest"] == str(updated_manifest_path)
    assert sidecar["updated_manifest_sha256"] == ""
    assert sidecar["updated_manifest_validation"]["checked"] is False
    assert sidecar["partial_materialization"] is False
    assert sidecar["source_recheck"]["path"].endswith(
        "apron_harness_source_recheck_2026_06_24.md"
    )
    assert "does not approve" in sidecar["source_recheck"]["evidence_boundary"]
    assert sidecar["training_gate"]["requires_seed_import_manifest_gate"] is True
    assert any("include_in_training=false" in error for error in sidecar["errors"])


def test_import_label_review_csv_rejects_uncleared_approved_row(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries, permission="unknown")
    _write_manifest(manifest_path, manifest)

    report = doctor.import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["invalid_approved_label_count"] == 1
    assert any("permission is not cleared" in error for error in report["errors"])


def test_import_label_review_csv_rejects_repo_local_raw_storage_ref(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(
        label_review_path,
        label_entries,
        raw_storage_ref="qa/video_eval/raw/apron_harness_clip.mp4",
    )
    _write_manifest(manifest_path, manifest)

    report = doctor.import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["invalid_approved_label_count"] == 1
    assert any("raw_storage_ref must point outside the repo" in error for error in report["errors"])


def test_import_label_review_csv_rejects_stale_source_manifest_guidance(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    rows = list(csv.DictReader(label_review_path.read_text(encoding="utf-8").splitlines()))
    rows[0]["source_manifest_sha256"] = "a" * 64
    with label_review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.LABEL_REVIEW_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_manifest(manifest_path, manifest)

    report = doctor.import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["invalid_approved_label_count"] == 1
    assert any("source_manifest_sha256 does not match" in error for error in report["errors"])


def test_import_label_review_csv_rejects_label_file_missing_required_classes(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 0,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest["yolo_labels"] = []
    label_entries = _write_reviewed_label_files(tmp_path / "labels", count=1)
    (tmp_path / label_entries[0]["path"]).write_text(
        "0 0.500 0.500 0.750 0.900\n",
        encoding="utf-8",
    )
    label_review_path = tmp_path / "apron_harness_label_review.csv"
    manifest_path = tmp_path / "source_manifest.yaml"
    _write_label_review_csv(label_review_path, label_entries)
    _write_manifest(manifest_path, manifest)

    report = doctor.import_label_review_csv_into_manifest(
        manifest_path=manifest_path,
        label_review_csv_path=label_review_path,
    )

    assert report["ok"] is False
    assert report["imported_label_count"] == 0
    assert report["invalid_approved_label_count"] == 1
    assert any("missing required class 1:apron" in error for error in report["errors"])


def test_capture_matrix_progress_gate_passes_when_rows_are_filled_and_approved(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["captured_examples"] = row["recommended_examples"]
        row["labeled_examples"] = row["recommended_examples"]
        row["review_status"] = "approved"
        row["permission"] = "controlled_capture_cleared"
        row["raw_storage_ref"] = f"s3://cleared-capture/{row['row_id']}"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.CAPTURE_MATRIX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    progress = doctor.validate_capture_matrix_progress(output_path)

    assert progress["ok"] is True
    assert progress["gate_passed"] is True
    assert progress["ready_rows"] == 21
    assert progress["target_labeled_examples"] == 720
    assert progress["labeled_examples"] == 720
    assert progress["missing_labeled_examples"] == 0
    assert progress["unapproved_rows"] == 0
    assert progress["unsafe_storage_rows"] == 0
    assert progress["manifest_reconciliation"]["checked"] is False
    assert doctor.main(["--validate-capture-matrix-csv", str(output_path)]) == 0


def test_capture_matrix_progress_blocks_repo_local_raw_storage_refs(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["captured_examples"] = row["recommended_examples"]
        row["labeled_examples"] = row["recommended_examples"]
        row["review_status"] = "approved"
        row["permission"] = "controlled_capture_cleared"
        row["raw_storage_ref"] = f"{ROOT}/qa/video_eval/raw/{row['row_id']}.mp4"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.CAPTURE_MATRIX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    progress = doctor.validate_capture_matrix_progress(output_path)

    assert progress["ok"] is True
    assert progress["gate_passed"] is False
    assert progress["ready_rows"] == 0
    assert progress["unsafe_storage_rows"] == len(rows)
    assert all("raw_storage_ref_missing_or_repo" in blocker for blocker in progress["blockers"])


def test_capture_matrix_progress_reconciles_approved_rows_with_manifest_counts(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 720,
        "apron": 300,
        "safety_harness": 300,
        "safety_lanyard": 300,
    }
    _write_manifest(manifest_path, manifest)
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["captured_examples"] = row["recommended_examples"]
        row["labeled_examples"] = row["recommended_examples"]
        row["review_status"] = "approved"
        row["permission"] = "controlled_capture_cleared"
        row["raw_storage_ref"] = f"s3://cleared-capture/{row['row_id']}"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.CAPTURE_MATRIX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    progress = doctor.validate_capture_matrix_progress(output_path, manifest_path=manifest_path)

    reconciliation = progress["manifest_reconciliation"]
    assert progress["gate_passed"] is True
    assert reconciliation["checked"] is True
    assert reconciliation["gate_passed"] is True
    assert reconciliation["required_labeled_images_per_class"] == {
        "person": 720,
        "apron": 300,
        "safety_harness": 300,
        "safety_lanyard": 300,
    }
    assert reconciliation["missing_manifest_counts"] == {}


def test_capture_matrix_progress_blocks_when_manifest_counts_do_not_match_filled_rows(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 300,
        "apron": 300,
        "safety_harness": 300,
        "safety_lanyard": 300,
    }
    _write_manifest(manifest_path, manifest)
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["captured_examples"] = row["recommended_examples"]
        row["labeled_examples"] = row["recommended_examples"]
        row["review_status"] = "approved"
        row["permission"] = "controlled_capture_cleared"
        row["raw_storage_ref"] = f"s3://cleared-capture/{row['row_id']}"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=doctor.CAPTURE_MATRIX_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    progress = doctor.validate_capture_matrix_progress(output_path, manifest_path=manifest_path)

    reconciliation = progress["manifest_reconciliation"]
    assert progress["gate_passed"] is False
    assert progress["ready_rows"] == 21
    assert reconciliation["gate_passed"] is False
    assert reconciliation["missing_manifest_counts"]["person"] == {
        "manifest_count": 300,
        "required_count": 720,
        "missing": 420,
    }
    assert any("manifest.person" in blocker for blocker in progress["blockers"])


def test_capture_matrix_progress_cli_blocks_incomplete_or_unapproved_rows(tmp_path: Path):
    doctor = _load_doctor()
    output_path = tmp_path / "apron_harness_capture_matrix.csv"
    doctor.main(
        [
            "--manifest",
            str(TEMPLATE_PATH),
            "--schema-only",
            "--emit-capture-matrix-csv",
            str(output_path),
        ]
    )

    exit_code = doctor.main(["--validate-capture-matrix-csv", str(output_path)])

    assert exit_code == 1


def test_capture_work_order_includes_coverage_gaps(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 310,
        "apron": 125,
        "safety_harness": 300,
        "safety_lanyard": 90,
    }
    manifest["coverage"]["camera_angles"] = ["front"]
    manifest["coverage"]["apron_positive_variants"] = ["denim_apron"]
    manifest["coverage"]["harness_hard_negative_tags"] = ["backpack_straps"]
    manifest["yolo_labels"] = []
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, schema_only=True)
    rendered = doctor.render_capture_work_order(report)

    assert "| `camera_angles` | elevated_cctv, side |" in rendered
    assert "work_apron" in rendered
    assert "tool_belts" in rendered
    assert "| `apron` | 125 | 300 | 175 |" in rendered
    assert "| `safety_lanyard` | 90 | 300 | 210 |" in rendered


def test_capture_manifest_rejects_unapproved_customer_private_footage(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest(label_path="")
    manifest["license"]["permission"] = "customer_private_unapproved"
    manifest["license"]["raw_storage"] = "repo"
    manifest["clips"][0]["permission"] = "customer_private_unapproved"
    manifest["yolo_labels"] = []
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert any("not cleared" in error for error in report["errors"])
    assert any("outside the repo" in error for error in report["errors"])


def test_public_seed_clip_requires_source_ref(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0].pop("source_ref", None)
    review_path = tmp_path / "seed_review.json"
    _write_seed_source_review(review_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, seed_source_review_report=review_path)

    assert report["ok"] is False
    assert report["seed_source_review"]["required"] is True
    assert report["seed_source_review"]["clip_count"] == 1
    assert any("source_ref is required" in error for error in report["errors"])


def test_public_seed_clip_requires_training_approved_seed_review(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    _write_seed_source_review(review_path, training_usable=False)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, seed_source_review_report=review_path)

    assert report["ok"] is False
    assert report["seed_source_review"]["required"] is True
    assert report["seed_source_review"]["approved_clip_count"] == 0
    assert any(
        "source_ref roboflow_safety_harness_dataset is not approved for training capabilities: harness_required"
        in error
        for error in report["errors"]
    )


def test_public_seed_clip_requires_approved_seed_import_manifest(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, include_in_training=False)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_source_review"]["approved_clip_count"] == 1
    assert report["seed_import_manifest"]["required"] is True
    assert report["seed_import_manifest"]["approved_clip_count"] == 0
    assert any(
        "requires approved seed import manifest entry for harness_required" in error
        for error in report["errors"]
    )


def test_public_seed_clip_passes_with_source_specific_training_approval(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is True
    assert report["seed_source_review"] == {
        "required": True,
        "report": str(review_path),
        "ok": True,
        "gate_passed": True,
        "clip_count": 1,
        "approved_clip_count": 1,
        "clips": [
            {
                "clip_id": "factory_harness_train_001",
                "source_ref": "roboflow_safety_harness_dataset",
                "target_capabilities": ["harness_required"],
                "training_approved": True,
                "seed_source_review_report": str(review_path),
                "capabilities": [
                    {
                        "capability": "harness_required",
                        "manifest_import_path": "qa/video_eval/datasets/imported/harness_seed.yaml",
                        "review_priority": 10,
                        "review_focus": "Review harness source mapping before import.",
                        "reviewed_by": "qa_reviewer",
                        "reviewed_at": "2026-06-22T00:00:00+00:00",
                        "review_evidence_path": REVIEW_EVIDENCE_PATH,
                        "review_evidence_sha256": REVIEW_EVIDENCE_SHA256,
                    }
                ],
            }
        ],
    }
    assert report["seed_import_manifest"]["required"] is True
    assert report["seed_import_manifest"]["report"] == str(import_path)
    assert report["seed_import_manifest"]["approved_clip_count"] == 1
    assert report["seed_import_manifest"]["clips"][0]["import_approved"] is True
    entry = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]
    assert entry["review_priority"] == 10
    assert entry["review_focus"] == "Review harness source mapping before import."
    assert entry["raw_export_ref"] == "s3://cleared-seed-exports/seed_yolo_export.zip"
    assert entry["raw_export_sha256"] == _sha256_file(import_path.with_suffix(".yolo_export.zip"))
    assert report["seed_import_manifest"]["imports"][0]["raw_export_local_path"] == str(
        import_path.with_suffix(".yolo_export.zip")
    )
    assert report["seed_import_manifest"]["imports"][0]["yolo_export_preflight"]["checked"] is True
    assert entry["reviewed_by"] == "qa_reviewer"


def test_public_seed_import_fingerprint_ignores_generated_review_artifacts(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_payload = _seed_source_review_payload()
    source_review_sha256 = _source_review_fingerprint(review_payload)
    review_payload["review_packets"] = {"roboflow_safety_harness_dataset": "generated packet"}
    review_payload["review_evidence_templates"] = ["generated template"]
    review_payload["review_checklist_apply"] = {"updated": False}
    review_path = tmp_path / "seed_review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_import_manifest(import_path, source_review_sha256=source_review_sha256)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is True
    assert report["seed_import_manifest"]["source_review_sha256_matches"] is True


def test_public_seed_clip_rejects_training_usable_review_without_human_evidence(
    tmp_path: Path,
):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_payload = _seed_source_review_payload()
    candidate = review_payload["candidates"][0]
    candidate.pop("completed_review")
    candidate.pop("reviewed_by")
    candidate.pop("reviewed_at")
    review_path = tmp_path / "seed_review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_import_manifest(
        import_path,
        source_review_sha256=_source_review_fingerprint(review_payload),
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_source_review"]["approved_clip_count"] == 0
    assert report["seed_import_manifest"]["approved_clip_count"] == 0
    assert any(
        "is not approved for training capabilities: harness_required" in error
        for error in report["errors"]
    )
    entry_errors = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]["errors"]
    assert any(
        "approved seed source review candidate is required before seed import" in error
        for error in entry_errors
    )
    assert any("reviewed_by is required" in error for error in entry_errors)
    assert any("reviewed_at is required" in error for error in entry_errors)
    assert any("completed_review missing approvals" in error for error in entry_errors)


def test_public_seed_clip_rejects_training_usable_review_without_evidence_bundle(
    tmp_path: Path,
):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_payload = _seed_source_review_payload()
    candidate = review_payload["candidates"][0]
    candidate.pop("review_evidence_path")
    candidate.pop("review_evidence_sha256")
    review_path = tmp_path / "seed_review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_import_manifest(
        import_path,
        source_review_sha256=_source_review_fingerprint(review_payload),
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_source_review"]["approved_clip_count"] == 0
    assert report["seed_import_manifest"]["approved_clip_count"] == 0
    entry_errors = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]["errors"]
    assert any("review_evidence_path is required" in error for error in entry_errors)
    assert any("review_evidence_sha256 is required" in error for error in entry_errors)


def test_public_seed_clip_rejects_training_usable_review_with_malformed_evidence_bundle(
    tmp_path: Path,
):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_payload = _seed_source_review_payload()
    evidence_path = tmp_path / "bad_seed_source_review.yaml"
    evidence_path.write_text(
        yaml.safe_dump(
            {
                "kind": "apron_harness_seed_source_review_evidence",
                "version": 1,
                "source_ref": "different_source",
                "capability": "harness_required",
                "reviewed_by": "qa_reviewer",
                "reviewed_at": "2026-06-22T00:00:00+00:00",
                "review_items": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    candidate = review_payload["candidates"][0]
    candidate["review_evidence_path"] = str(evidence_path)
    candidate["review_evidence_sha256"] = _sha256_file(evidence_path)
    review_path = tmp_path / "seed_review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_import_manifest(
        import_path,
        source_review_sha256=_source_review_fingerprint(review_payload),
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_source_review"]["approved_clip_count"] == 0
    entry_errors = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]["errors"]
    assert any("review evidence source_ref must match the approved source" in error for error in entry_errors)
    assert any("review evidence review_items are required" in error for error in entry_errors)


def test_public_seed_clip_rejects_seed_import_bound_to_different_source_review(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, source_review_sha256="0" * 64)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_import_manifest"]["source_review_sha256_matches"] is False
    assert any(
        "seed import manifest source_review_sha256 does not match seed source review" in error
        for error in report["errors"]
    )


def test_public_seed_clip_rejects_local_raw_export_ref(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(
        import_path,
        raw_export_ref="qa/video_eval/datasets/imported/seed_yolo_export.zip",
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    assert report["seed_import_manifest"]["approved_clip_count"] == 0
    entry_errors = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]["errors"]
    assert any("raw_export_ref must be a remote immutable export reference" in error for error in entry_errors)


def test_public_seed_clip_rejects_invalid_raw_export_sha256(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["clips"][0]["source"] = "public_seed_source"
    manifest["clips"][0]["permission"] = "commercial_dataset_approved"
    manifest["clips"][0]["source_ref"] = "roboflow_safety_harness_dataset"
    review_path = tmp_path / "seed_review.json"
    import_path = tmp_path / "seed_import.yaml"
    _write_seed_source_review(review_path)
    _write_seed_import_manifest(import_path, raw_export_sha256="not-a-sha256")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(
        manifest_path,
        seed_source_review_report=review_path,
        seed_import_manifest=import_path,
    )

    assert report["ok"] is False
    entry_errors = report["seed_import_manifest"]["clips"][0]["capabilities"][0]["entries"][0]["errors"]
    assert any("raw_export_sha256 must be a 64-character SHA-256 hex digest" in error for error in entry_errors)


def test_capture_manifest_rejects_bad_yolo_labels(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text(
        "4 0.500 0.500 0.750 0.900\n"
        "1 1.200 0.560 0.220 0.450\n"
        "2 0.500 0.500 0.000 0.700\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, _valid_manifest())

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert any("class id 4" in error for error in report["errors"])
    assert any("normalized between 0 and 1" in error for error in report["errors"])
    assert any("width and height must be positive" in error for error in report["errors"])


def test_capture_manifest_rejects_unreviewed_yolo_labels(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text(
        "0 0.500 0.500 0.750 0.900\n"
        "1 0.520 0.560 0.220 0.450\n"
        "2 0.500 0.500 0.420 0.700\n"
        "3 0.700 0.500 0.100 0.500\n",
        encoding="utf-8",
    )
    manifest = _valid_manifest()
    manifest["yolo_labels"] = [{"path": "labels/frame_001.txt"}]
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1,
        "apron": 1,
        "safety_harness": 1,
        "safety_lanyard": 1,
    }
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert report["approved_label_file_count"] == 0
    assert report["unapproved_label_file_count"] == 1
    errors = "\n".join(report["errors"])
    assert "review_status=approved" in errors
    assert "must include reviewer" in errors
    assert "must include reviewed_at" in errors
    assert "must include source_clip_id" in errors
    assert "must include split" in errors


def test_capture_manifest_rejects_unknown_label_source_clip(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text(
        "0 0.500 0.500 0.750 0.900\n",
        encoding="utf-8",
    )
    manifest = _valid_manifest()
    manifest["yolo_labels"] = [
        {
            "path": "labels/frame_001.txt",
            "review_status": "approved",
            "reviewer": "qa_reviewer",
            "reviewed_at": "2026-06-21T00:00:00+00:00",
            "source_clip_id": "missing_clip",
            "split": "train",
        }
    ]
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1,
        "apron": 0,
        "safety_harness": 0,
        "safety_lanyard": 0,
    }
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert any("source_clip_id is not listed in clips" in error for error in report["errors"])


def test_capture_manifest_rejects_inflated_declared_counts(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text(
        "0 0.500 0.500 0.750 0.900\n"
        "1 0.520 0.560 0.220 0.450\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, _valid_manifest())

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "counts.labeled_images_per_class.person declares 350 but only 1 label files contain person" in errors
    assert "counts.labeled_images_per_class.apron declares 320 but only 1 label files contain apron" in errors
    assert "counts.labeled_images_per_class.safety_harness declares 330 but only 0 label files contain safety_harness" in errors


def test_capture_manifest_rejects_duplicate_label_paths(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text(
        "0 0.500 0.500 0.750 0.900\n"
        "1 0.520 0.560 0.220 0.450\n",
        encoding="utf-8",
    )
    manifest = _valid_manifest()
    manifest["yolo_labels"] = [
        {
            "path": "labels/frame_001.txt",
            "review_status": "approved",
            "reviewer": "qa_reviewer",
            "reviewed_at": "2026-06-21T00:00:00+00:00",
            "source_clip_id": "factory_harness_train_001",
            "split": "train",
        },
        {
            "path": "labels/frame_001.txt",
            "review_status": "approved",
            "reviewer": "qa_reviewer",
            "reviewed_at": "2026-06-21T00:00:00+00:00",
            "source_clip_id": "factory_harness_train_001",
            "split": "train",
        },
    ]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert any("duplicate yolo_labels path" in error for error in report["errors"])


def test_capture_manifest_rejects_missing_validation_split(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    for item in manifest["yolo_labels"]:
        item["source_clip_id"] = "factory_harness_train_001"
        item["split"] = "train"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert report["label_files_per_split"]["train"] == 350
    assert report["label_files_per_split"]["val"] == 0
    assert any("splits: val" in error for error in report["errors"])


def test_capture_manifest_rejects_source_clip_leakage_between_splits(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    for index, item in enumerate(manifest["yolo_labels"]):
        item["source_clip_id"] = "factory_harness_train_001"
        item["split"] = "train" if index < 200 else "val"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert report["label_files_per_split"]["train"] == 200
    assert report["label_files_per_split"]["val"] == 150
    assert any("appears in multiple dataset splits" in error for error in report["errors"])


def test_capture_manifest_rejects_coverage_not_backed_by_clips(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    for clip in manifest["clips"]:
        clip["positive_variant_tags"] = [
            tag for tag in clip.get("positive_variant_tags", []) if tag != "denim_apron"
        ]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    assert any(
        "coverage.apron_positive_variants declares values not backed by clips: ['denim_apron']" in error
        for error in report["errors"]
    )


def test_capture_manifest_rejects_missing_required_class_coverage_in_validation_split(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(labels_dir, count=450)
    for item in manifest["yolo_labels"]:
        if item["split"] == "val":
            (tmp_path / item["path"]).write_text(
                "0 0.500 0.500 0.750 0.900\n",
                encoding="utf-8",
            )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path)

    assert report["ok"] is False
    split_counts = report["approved_label_images_per_split_per_class"]
    assert split_counts["val"]["person"] > 0
    assert split_counts["val"]["apron"] == 0
    assert split_counts["val"]["safety_harness"] == 0
    assert split_counts["val"]["safety_lanyard"] == 0
    errors = "\n".join(report["errors"])
    assert "val.apron" in errors
    assert "val.safety_harness" in errors
    assert "val.safety_lanyard" in errors


def test_production_manifest_requires_test_split_class_coverage(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(labels_dir, count=450)
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 450,
        "apron": 450,
        "safety_harness": 450,
        "safety_lanyard": 450,
    }
    for item in manifest["yolo_labels"]:
        if item["split"] == "test":
            (tmp_path / item["path"]).write_text(
                "0 0.500 0.500 0.750 0.900\n",
                encoding="utf-8",
            )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, mode="production")

    assert report["ok"] is False
    split_counts = report["approved_label_images_per_split_per_class"]
    assert split_counts["test"]["person"] > 0
    assert split_counts["test"]["apron"] == 0
    assert split_counts["test"]["safety_harness"] == 0
    assert split_counts["test"]["safety_lanyard"] == 0
    errors = "\n".join(report["errors"])
    assert "test.apron" in errors
    assert "test.safety_harness" in errors
    assert "test.safety_lanyard" in errors


def test_production_manifest_requires_minimum_holdout_split_counts(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(labels_dir, count=1000)
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1000,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    for index, item in enumerate(manifest["yolo_labels"]):
        if index < 920:
            item["split"] = "train"
            item["source_clip_id"] = "factory_harness_train_001"
        elif index < 960:
            item["split"] = "val"
            item["source_clip_id"] = "factory_harness_val_001"
        else:
            item["split"] = "test"
            item["source_clip_id"] = "factory_harness_test_001"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)

    report = doctor.validate_manifest(manifest_path, mode="production")

    assert report["ok"] is False
    assert report["minimum_split_class_counts"] == {"test": 100, "val": 100}
    split_counts = report["approved_label_images_per_split_per_class"]
    assert split_counts["val"]["apron"] == 40
    assert split_counts["test"]["safety_harness"] == 40
    errors = "\n".join(report["errors"])
    assert "val.apron has 40 reviewed labels; 100 needed for production" in errors
    assert "test.safety_lanyard has 40 reviewed labels; 100 needed for production" in errors


def test_production_mode_requires_larger_class_counts(tmp_path: Path):
    doctor = _load_doctor()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "frame_001.txt").write_text("0 0.500 0.500 0.750 0.900\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, _valid_manifest())

    report = doctor.validate_manifest(manifest_path, mode="production")

    assert report["ok"] is False
    assert any("1000 needed for production" in error for error in report["errors"])


def test_emit_yolo_dataset_yaml_after_strict_validation(tmp_path: Path):
    doctor = _load_doctor()
    manifest = _valid_manifest()
    manifest["yolo_labels"] = _write_reviewed_label_files(tmp_path / "labels")
    manifest["dataset"] = {
        "root": str(tmp_path / "dataset"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "dataset.yaml"
    _write_manifest(manifest_path, manifest)

    exit_code = doctor.main([
        "--manifest",
        str(manifest_path),
        "--emit-yolo-dataset-yaml",
        str(output_path),
    ])

    assert exit_code == 0
    rendered = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert rendered["path"] == str(tmp_path / "dataset")
    assert rendered["train"] == "images/train"
    assert rendered["val"] == "images/val"
    assert rendered["test"] == "images/test"
    assert rendered["names"] == {
        0: "person",
        1: "apron",
        2: "safety_harness",
        3: "safety_lanyard",
    }
    assert rendered["rakshak_lens"]["missing_ppe_label_policy"] == (
        "derive_missing_ppe_from_person_to_visible_ppe_association"
    )
    assert rendered["rakshak_lens"]["source_manifest_sha256"] == doctor._sha256_file(manifest_path)


def test_emit_yolo_dataset_yaml_rejects_schema_only(tmp_path: Path):
    doctor = _load_doctor()
    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "dataset.yaml"
    _write_manifest(manifest_path, _valid_manifest(label_path=""))

    exit_code = doctor.main([
        "--manifest",
        str(manifest_path),
        "--schema-only",
        "--emit-yolo-dataset-yaml",
        str(output_path),
    ])

    assert exit_code == 1
    assert not output_path.exists()
