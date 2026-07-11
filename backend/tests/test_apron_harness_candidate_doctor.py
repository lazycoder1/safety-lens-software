"""Tests for apron/harness trained-candidate promotion gate."""

from pathlib import Path
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DOCTOR_PATH = ROOT / "scripts" / "apron_harness_candidate_doctor.py"
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_candidate_doctor", CANDIDATE_DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_training_result(export_path: str) -> dict:
    return {
        "status": "trained",
        "model": "yolo26n.pt",
        "classes": {
            0: "person",
            1: "apron",
            2: "safety_harness",
            3: "safety_lanyard",
        },
        "metrics": {
            "mAP50": 0.81,
            "recall": 0.9,
        },
        "capture_preflight": {
            "mode": "production",
            "checked": True,
            "gate_passed": True,
            "manifest_reconciliation": {
                "checked": True,
                "gate_passed": True,
            },
            "capture_matrix_manifest": {
                "path": "apron_harness_production_capture_matrix.csv.manifest.json",
                "checked": True,
                "valid": True,
                "mode": "production",
                "row_count": 21,
                "matrix_csv_sha256": "a" * 64,
                "source_manifest_sha256": "b" * 64,
            },
            "label_review_import_manifest": {
                "path": "apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
                "checked": True,
                "valid": True,
                "source_manifest_sha256": "c" * 64,
                "label_review_csv_sha256": "e" * 64,
                "updated_manifest_sha256": "b" * 64,
                "imported_label_count": 2404,
                "imported_clip_count": 3,
                "invalid_clip_metadata_count": 0,
                "merged_label_count": 2404,
                "updated_labeled_images_per_class": {
                    "person": 2404,
                    "apron": 1000,
                    "safety_harness": 1000,
                    "safety_lanyard": 1000,
                },
                "updated_manifest_validation": {
                    "checked": True,
                    "mode": "production",
                    "schema_only": False,
                    "ok": True,
                    "manifest_sha256": "b" * 64,
                    "errors": [],
                    "warnings": [],
                },
                "training_gate": {
                    "requires_reviewed_clip_metadata": True,
                },
            },
        },
        "dataset_provenance": {
            "required": True,
            "checked": True,
            "source_manifest": "/cleared/apron_harness_capture_manifest.yaml",
            "declared_source_manifest_sha256": "b" * 64,
            "source_manifest_sha256": "b" * 64,
            "permission": "controlled_capture_cleared",
            "permission_allowed": True,
            "missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
            "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
            "errors": [],
        },
        "source_lineage": {
            "dataset_yaml": {
                "file": {
                    "required": True,
                    "path": "/cleared/dataset.yaml",
                    "exists": True,
                    "sha256": "1" * 64,
                },
            },
            "capture_manifest": {
                "file": {
                    "required": True,
                    "path": "/cleared/apron_harness_capture_manifest.yaml",
                    "exists": True,
                    "sha256": "b" * 64,
                },
                "manifest_sha256": "b" * 64,
                "ok": True,
                "mode": "production",
            },
            "seed_source_review": {
                "file": {
                    "required": False,
                    "path": None,
                    "exists": False,
                    "sha256": None,
                },
                "gate": {"required": False},
            },
            "seed_import_manifest": {
                "file": {
                    "required": False,
                    "path": None,
                    "exists": False,
                    "sha256": None,
                },
                "gate": {"required": False},
            },
        },
        "per_class_metrics": {
            "person": {"mAP50": 0.90, "recall": 0.95},
            "apron": {"mAP50": 0.80, "recall": 0.88},
            "safety_harness": {"mAP50": 0.79, "recall": 0.89},
            "safety_lanyard": {"mAP50": 0.76, "recall": 0.86},
        },
        "exports": [export_path],
    }


def _approved_seed_source_lineage() -> dict:
    return {
        "file": {
            "required": True,
            "path": "/cleared/seed_source_review.json",
            "exists": True,
            "sha256": "2" * 64,
        },
        "source_recheck": {
            "path": "qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md",
            "exists": True,
            "sha256": "7" * 64,
            "evidence_boundary": "Fresh source research evidence only; this does not approve training.",
        },
        "gate": {
            "required": True,
            "ok": True,
            "gate_passed": True,
            "clip_count": 1,
            "approved_clip_count": 1,
        },
    }


def _approved_seed_import_lineage(capability: str = "harness_required") -> dict:
    required_counts = (
        {"person": 3, "safety_harness": 3, "safety_lanyard": 3}
        if capability == "harness_required"
        else {"person": 3, "apron": 3}
    )
    return {
        "file": {
            "required": True,
            "path": "/cleared/seed_import.yaml",
            "exists": True,
            "sha256": "3" * 64,
        },
        "gate": {
            "required": True,
            "ok": True,
            "source_review_sha256_matches": True,
            "clip_count": 1,
            "approved_clip_count": 1,
            "imports": [
                {
                    "source_ref": "roboflow_seed",
                    "capability": capability,
                    "include_in_training": True,
                    "approved_for_training": True,
                    "raw_export_sha256": "4" * 64,
                    "raw_export_local_path": "/cleared/seed_yolo_export.zip",
                    "errors": [],
                    "blockers": [],
                    "yolo_export_preflight": {
                        "checked": True,
                        "sha256": "4" * 64,
                        "label_file_count_by_local_class": required_counts,
                    },
                }
            ],
        },
    }


def _approved_seed_export_import_manifest() -> dict:
    return {
        "path": "/cleared/apron_harness_capture_manifest.yaml.seed_export_import.json",
        "exists": True,
        "checked": True,
        "valid": True,
        "sha256": "5" * 64,
        "seed_source_review_report": "/cleared/seed_source_review.json",
        "seed_source_review_sha256": "6" * 64,
        "source_recheck": {
            "path": "qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md",
            "exists": True,
            "sha256": "7" * 64,
            "evidence_boundary": "Fresh source research evidence only; this does not approve training.",
        },
        "seed_import_manifest": "/cleared/seed_import.yaml",
        "seed_import_manifest_sha256": "3" * 64,
        "updated_manifest_sha256": "b" * 64,
        "imported_label_count": 3,
        "imported_clip_count": 3,
        "copied_image_count": 3,
        "partial_materialization": False,
        "updated_labeled_images_per_class": {
            "person": 2404,
            "apron": 1000,
            "safety_harness": 1000,
            "safety_lanyard": 1000,
        },
        "updated_manifest_validation": {
            "checked": True,
            "mode": "production",
            "schema_only": False,
            "ok": True,
            "manifest_sha256": "b" * 64,
            "errors": [],
            "warnings": [],
        },
        "imports": [
            {
                "source_ref": "roboflow_seed",
                "capability": "harness_required",
                "raw_export_ref": "s3://cleared-seed-exports/harness_seed_yolo_export.zip",
                "raw_export_sha256": "4" * 64,
                "raw_export_local_path": "/cleared/seed_yolo_export.zip",
                "imported_label_count": 3,
                "copied_image_count": 3,
                "errors": [],
                "warnings": [],
                "yolo_export_preflight": {
                    "checked": True,
                    "sha256": "4" * 64,
                    "source_classes": ["person", "safety-harness", "lanyard"],
                    "mapped_local_classes": ["person", "safety_harness", "safety_lanyard"],
                    "image_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "label_file_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "orphan_label_count_by_split": {},
                    "label_file_count_by_local_class": {
                        "person": 3,
                        "safety_harness": 3,
                        "safety_lanyard": 3,
                    },
                    "errors": [],
                    "warnings": [],
                },
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_candidate_doctor_accepts_trained_exported_candidate(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    _write_json(result_path, _valid_training_result(str(export_path)))

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is True
    manifest = report["promotion_manifest"]
    assert manifest["candidate_status"] == "ready_for_side_by_side_runtime_test"
    handoff = manifest["runtime_handoff"]
    assert handoff["planned_model_key"] == "ppe_closed_set_candidate"
    assert handoff["planned_registry_path"] == "models/ppe_closed_set_candidate/apron-harness-ppe.onnx"
    assert handoff["current_runtime_model_family"] == "ppe_specialist"
    assert handoff["activation_policy"] == "do_not_activate_until_side_by_side_runtime_and_jetson_gates_pass"
    assert handoff["selected_export"]["path"] == str(export_path)
    assert handoff["selected_export"]["suffix"] == ".onnx"
    assert len(handoff["selected_export"]["sha256"]) == 64
    assert handoff["registry_entry"]["model_key"] == "ppe_closed_set_candidate"
    assert handoff["registry_entry"]["file"] == "apron-harness-ppe.onnx"
    assert handoff["registry_entry"]["source_export_sha256"] == handoff["selected_export"]["sha256"]
    assert handoff["model_manager_definition_hint"]["model_key"] == "ppe_closed_set_candidate"
    assert "run_factory_ppe_jetson_full_gate" in handoff["registration_preconditions"]
    assert "verify_candidate_export_sha256" in manifest["next_required_gates"]
    assert "copy_selected_export_to_model_registry" in manifest["next_required_gates"]
    assert "run_jetson_three_camera_soak" in manifest["next_required_gates"]
    assert manifest["training_capture_preflight"]["gate_passed"] is True
    assert manifest["training_dataset_provenance"]["declared_source_manifest_sha256"] == "b" * 64
    assert manifest["training_dataset_provenance"]["source_manifest_sha256"] == "b" * 64
    assert manifest["training_dataset_provenance"]["permission_allowed"] is True
    assert manifest["training_source_lineage"]["capture_manifest"]["file"]["sha256"] == "b" * 64
    assert manifest["capture_matrix_manifest"]["valid"] is True
    assert manifest["capture_matrix_manifest"]["matrix_csv_sha256"] == "a" * 64
    assert manifest["label_review_import_manifest"]["valid"] is True
    assert manifest["label_review_import_manifest"]["updated_manifest_sha256"] == "b" * 64
    assert "verify_training_source_lineage" in manifest["next_required_gates"]
    assert "verify_label_review_import_manifest" in manifest["next_required_gates"]


def test_candidate_doctor_rejects_dry_run_plan(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "training_plan.json"
    payload = _valid_training_result("best.onnx")
    payload["status"] = "ready_to_train"
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path, artifact_root=tmp_path)

    assert report["ok"] is False
    assert any("status must be trained" in error for error in report["errors"])


def test_candidate_doctor_rejects_missing_capture_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload.pop("capture_preflight")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("capture_preflight is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_failed_capture_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["gate_passed"] = False
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("capture_preflight.gate_passed" in error for error in report["errors"])


def test_candidate_doctor_rejects_missing_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"].pop("capture_matrix_manifest")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("capture_preflight.capture_matrix_manifest is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_invalid_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["capture_matrix_manifest"]["valid"] = False
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("capture_preflight.capture_matrix_manifest.valid" in error for error in report["errors"])


def test_candidate_doctor_rejects_missing_label_review_import_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"].pop("label_review_import_manifest")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "capture_preflight.label_review_import_manifest is required" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_label_review_import_manifest_source_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["label_review_import_manifest"]["updated_manifest_sha256"] = "f" * 64
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "label_review_import_manifest.updated_manifest_sha256 must match" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_label_review_import_manifest_validation_failure(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["label_review_import_manifest"]["updated_manifest_validation"]["ok"] = False
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "label_review_import_manifest.updated_manifest_validation.ok must be true" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_label_review_import_manifest_without_clip_metadata_gate(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["label_review_import_manifest"]["training_gate"].pop(
        "requires_reviewed_clip_metadata"
    )
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "label_review_import_manifest.training_gate.requires_reviewed_clip_metadata must be true" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_invalid_label_review_clip_metadata(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["label_review_import_manifest"]["invalid_clip_metadata_count"] = 1
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "label_review_import_manifest.invalid_clip_metadata_count must be 0" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_missing_dataset_provenance(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload.pop("dataset_provenance")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("dataset_provenance is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_dataset_provenance_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["dataset_provenance"]["source_manifest_sha256"] = "c" * 64
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("dataset_provenance.source_manifest_sha256 must match" in error for error in report["errors"])


def test_candidate_doctor_rejects_missing_source_lineage(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload.pop("source_lineage")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("source_lineage is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_source_lineage_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["capture_manifest"]["file"]["sha256"] = "c" * 64
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("source_lineage.capture_manifest.file.sha256 must match" in error for error in report["errors"])


def test_candidate_doctor_rejects_required_seed_lineage_without_approved_import(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = {
        "file": {
            "required": True,
            "path": "/cleared/seed_source_review.json",
            "exists": True,
            "sha256": "2" * 64,
        },
        "gate": {
            "required": True,
            "ok": True,
            "gate_passed": True,
            "clip_count": 1,
            "approved_clip_count": 1,
        },
    }
    payload["source_lineage"]["seed_import_manifest"] = {
        "file": {
            "required": True,
            "path": "/cleared/seed_import.yaml",
            "exists": True,
            "sha256": "3" * 64,
        },
        "gate": {
            "required": True,
            "ok": True,
            "source_review_sha256_matches": False,
            "clip_count": 1,
            "approved_clip_count": 0,
        },
    }
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("source_review_sha256_matches must be true" in error for error in report["errors"])
    assert any("approved_clip_count must equal clip_count" in error for error in report["errors"])


def test_candidate_doctor_rejects_required_seed_source_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_source_review"].pop("source_recheck")
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    payload["capture_preflight"]["seed_export_import_manifest"] = _approved_seed_export_import_manifest()
    payload["source_lineage"]["seed_export_import_manifest"] = _approved_seed_export_import_manifest()
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "source_lineage.seed_source_review.source_recheck is required" in error
        for error in report["errors"]
    )


def test_candidate_doctor_accepts_required_seed_lineage_with_export_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    payload["capture_preflight"]["seed_export_import_manifest"] = _approved_seed_export_import_manifest()
    payload["source_lineage"]["seed_export_import_manifest"] = _approved_seed_export_import_manifest()
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is True
    assert report["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"]["gate"]["imports"][0][
        "yolo_export_preflight"
    ]["checked"] is True
    assert report["promotion_manifest"]["seed_export_import_manifest"]["valid"] is True
    assert "verify_seed_export_import_manifest" in report["promotion_manifest"]["next_required_gates"]


def test_candidate_doctor_rejects_required_seed_lineage_without_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("seed_export_import_manifest is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_stale_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["updated_manifest_sha256"] = "f" * 64
    payload["capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("seed_export_import_manifest.updated_manifest_sha256 must match" in error for error in report["errors"])


def test_candidate_doctor_rejects_seed_export_sidecar_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar.pop("source_recheck")
    payload["capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "capture_preflight.seed_export_import_manifest.source_recheck is required" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_seed_export_sidecar_without_yolo_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["imports"][0].pop("yolo_export_preflight")
    payload["capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "seed_export_import_manifest.imports[0].yolo_export_preflight is required" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_partial_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["partial_materialization"] = True
    payload["capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("seed_export_import_manifest.partial_materialization must be false" in error for error in report["errors"])


def test_candidate_doctor_rejects_required_seed_import_without_export_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    payload["source_lineage"]["seed_import_manifest"]["gate"]["imports"][0].pop("yolo_export_preflight")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("yolo_export_preflight is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_dataset_provenance_declared_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["dataset_provenance"]["declared_source_manifest_sha256"] = "c" * 64
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any(
        "dataset_provenance.declared_source_manifest_sha256 must match source_manifest_sha256" in error
        for error in report["errors"]
    )


def test_candidate_doctor_rejects_dataset_provenance_permission_not_allowed(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["dataset_provenance"]["permission"] = "research_only"
    payload["dataset_provenance"]["permission_allowed"] = False
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("dataset_provenance.permission_allowed" in error for error in report["errors"])


def test_candidate_doctor_rejects_pilot_capture_preflight(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["capture_preflight"]["mode"] = "pilot"
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("capture_preflight.mode must be production" in error for error in report["errors"])


def test_candidate_doctor_rejects_yoloe_candidate(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload["model"] = "yoloe-11s-seg.pt"
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("model must be one of" in error for error in report["errors"])


def test_candidate_doctor_requires_per_class_metrics(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(export_path))
    payload.pop("per_class_metrics")
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("per_class_metrics is required" in error for error in report["errors"])


def test_candidate_doctor_rejects_low_class_recall_and_missing_export(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result("missing.pt")
    payload["per_class_metrics"]["safety_lanyard"]["recall"] = 0.50
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path, artifact_root=tmp_path)

    assert report["ok"] is False
    assert any("safety_lanyard recall" in error for error in report["errors"])
    assert any("export artifact does not exist" in error for error in report["errors"])
    assert any("must be ONNX or TensorRT" in error for error in report["errors"])


def test_candidate_doctor_rejects_engine_only_candidate_for_onnx_registry(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.engine"
    export_path.write_bytes(b"fake tensorrt")
    result_path = tmp_path / "training_result.json"
    _write_json(result_path, _valid_training_result(str(export_path)))

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is False
    assert any("ONNX export artifact is required" in error for error in report["errors"])
    assert report["promotion_manifest"]["runtime_handoff"]["selected_export"] is None


def test_candidate_doctor_selects_onnx_when_engine_is_also_present(tmp_path: Path):
    doctor = _load_doctor()
    engine_path = tmp_path / "best.engine"
    engine_path.write_bytes(b"fake tensorrt")
    onnx_path = tmp_path / "best.onnx"
    onnx_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    payload = _valid_training_result(str(engine_path))
    payload["exports"].append(str(onnx_path))
    _write_json(result_path, payload)

    report = doctor.validate_candidate(result_path)

    assert report["ok"] is True
    selected = report["promotion_manifest"]["runtime_handoff"]["selected_export"]
    assert selected["path"] == str(onnx_path)
    assert selected["suffix"] == ".onnx"


def test_candidate_doctor_cli_writes_report(tmp_path: Path):
    doctor = _load_doctor()
    export_path = tmp_path / "best.onnx"
    export_path.write_bytes(b"fake onnx")
    result_path = tmp_path / "training_result.json"
    out_path = tmp_path / "candidate_report.json"
    _write_json(result_path, _valid_training_result(str(export_path)))

    exit_code = doctor.main([
        "--training-result",
        str(result_path),
        "--out",
        str(out_path),
    ])

    assert exit_code == 0
    assert out_path.exists()
    assert "ready_for_side_by_side_runtime_test" in out_path.read_text(encoding="utf-8")
