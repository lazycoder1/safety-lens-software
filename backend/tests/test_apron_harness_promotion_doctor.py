"""Tests for closed-set apron/harness runtime promotion gate."""

from pathlib import Path
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[2]
PROMOTION_DOCTOR_PATH = ROOT / "scripts" / "apron_harness_promotion_doctor.py"
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_promotion_doctor", PROMOTION_DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _candidate_report() -> dict:
    return {
        "ok": True,
        "promotion_manifest": {
            "candidate_status": "ready_for_side_by_side_runtime_test",
            "metric_thresholds": {
                "min_per_class_mAP50": 0.75,
                "min_per_class_recall": 0.85,
            },
            "class_metrics": {
                "person": {"mAP50": 0.90, "recall": 0.95},
                "apron": {"mAP50": 0.80, "recall": 0.88},
                "safety_harness": {"mAP50": 0.79, "recall": 0.89},
                "safety_lanyard": {"mAP50": 0.76, "recall": 0.86},
            },
            "training_capture_preflight": {
                "mode": "production",
                "checked": True,
                "gate_passed": True,
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
            "training_dataset_provenance": {
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
            "training_source_lineage": {
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
            "runtime_handoff": {
                "planned_model_key": "ppe_closed_set_candidate",
                "planned_registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                "selected_export": {
                    "path": "/cleared/runs/apron-harness-ppe.onnx",
                    "exists": True,
                    "accepted_suffix": True,
                    "suffix": ".onnx",
                    "sha256": "d" * 64,
                },
                "registry_entry": {
                    "model_key": "ppe_closed_set_candidate",
                    "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                    "source_export_sha256": "d" * 64,
                },
            },
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


def _result(
    scenario_id: str,
    *,
    detections: int,
    matching_alerts: int,
    unexpected_alerts: int,
    class_counts: dict[str, int] | None = None,
    suppressed: list[str] | None = None,
    invocations: dict[str, int] | None = None,
    screenshot: bool = True,
    required_model_key: str | None = None,
) -> dict:
    evidence = {
        "max_detections_count": detections,
        "ui_evidence": {
            "screenshot_exists": screenshot,
            "screenshot_fresh": screenshot,
        },
        "matching_alerts": [{"id": f"{scenario_id}_alert"} for _ in range(matching_alerts)],
        "unexpected_alerts": [{"id": f"{scenario_id}_unexpected"} for _ in range(unexpected_alerts)],
        "analytics_summary": {
            "class_counts": class_counts or {},
            "schedule": {
                "suppressed_capabilities": suppressed or [],
                "model_invocations": invocations or {},
            },
        },
    }
    if required_model_key:
        evidence["model_preflight"] = {
            "checked": True,
            "ok": True,
            "required_model_keys": [required_model_key],
            "missing_model_keys": [],
        }
    return {
        "scenario_id": scenario_id,
        "status": "ready_to_sell",
        "evidence": evidence,
    }


def _write_promotion_inputs(tmp_path: Path, *, bad_candidate_guard: bool = False) -> dict[str, Path]:
    paths = {
        "candidate_report": tmp_path / "candidate_report.json",
        "baseline_active": tmp_path / "baseline_active.json",
        "baseline_guard": tmp_path / "baseline_guard.json",
        "baseline_suppression": tmp_path / "baseline_suppression.json",
        "candidate_active": tmp_path / "candidate_active.json",
        "candidate_guard": tmp_path / "candidate_guard.json",
        "candidate_suppression": tmp_path / "candidate_suppression.json",
    }
    _write_json(paths["candidate_report"], _candidate_report())
    _write_json(
        paths["baseline_active"],
        _result(
            "factory_missing_apron_active",
            detections=7,
            matching_alerts=1,
            unexpected_alerts=0,
            invocations={"ppe_specialist": 1},
        ),
    )
    _write_json(
        paths["baseline_guard"],
        _result(
            "factory_apron_false_positive_guard",
            detections=9,
            matching_alerts=0,
            unexpected_alerts=0,
            class_counts={"denim apron": 5, "person": 2},
            invocations={"ppe_specialist": 1},
        ),
    )
    _write_json(
        paths["baseline_suppression"],
        _result(
            "factory_apron_detector_window_suppression",
            detections=0,
            matching_alerts=0,
            unexpected_alerts=0,
            suppressed=["apron_required"],
            invocations={"ppe_specialist": 0},
        ),
    )
    _write_json(
        paths["candidate_active"],
        _result(
            "factory_missing_apron_active_closed_set",
            detections=6,
            matching_alerts=1,
            unexpected_alerts=0,
            invocations={"ppe_closed_set_candidate": 1},
            required_model_key="ppe_closed_set_candidate",
        ),
    )
    _write_json(
        paths["candidate_guard"],
        _result(
            "factory_apron_false_positive_guard_closed_set",
            detections=4,
            matching_alerts=0,
            unexpected_alerts=0,
            class_counts={} if bad_candidate_guard else {"apron": 4, "person": 2},
            invocations={"ppe_closed_set_candidate": 1},
            required_model_key="ppe_closed_set_candidate",
        ),
    )
    _write_json(
        paths["candidate_suppression"],
        _result(
            "factory_apron_detector_window_suppression_closed_set",
            detections=0,
            matching_alerts=0,
            unexpected_alerts=0,
            suppressed=["apron_required"],
            invocations={"ppe_closed_set_candidate": 0},
            required_model_key="ppe_closed_set_candidate",
        ),
    )
    return paths


def test_promotion_doctor_accepts_side_by_side_runtime_evidence(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is True
    assert report["promotion_status"] == "ready_for_runtime_registration"
    assert report["candidate_report_sha256"] == doctor._sha256_file(paths["candidate_report"])
    assert report["candidate"]["false_positive_guard"]["visible_class_total"] == 4
    assert report["candidate_capture_matrix_manifest"]["valid"] is True
    assert report["candidate_capture_matrix_manifest"]["matrix_csv_sha256"] == "a" * 64
    assert report["candidate_label_review_import_manifest"]["valid"] is True
    assert report["candidate_label_review_import_manifest"]["updated_manifest_sha256"] == "b" * 64
    assert report["candidate_training_dataset_provenance"]["source_manifest_sha256"] == "b" * 64
    assert report["candidate_training_source_lineage"]["capture_manifest"]["file"]["sha256"] == "b" * 64
    assert report["candidate_selected_export"]["sha256"] == "d" * 64
    assert report["candidate_registry_entry"]["source_export_sha256"] == "d" * 64
    assert report["runtime_model_gates"]["baseline"]["active"]["required_model_key"] == "ppe_specialist"
    assert report["runtime_model_gates"]["baseline"]["active"]["required_model_invocations"] == 1
    assert report["runtime_model_gates"]["candidate"]["active"]["required_model_key"] == "ppe_closed_set_candidate"
    assert report["runtime_model_gates"]["candidate"]["active"]["forbidden_model_key"] == "ppe_specialist"
    assert report["runtime_model_gates"]["candidate"]["suppression"]["required_invocation_policy"] == "zero"
    assert "run_jetson_three_camera_soak" in report["next_required_gates"]


def test_promotion_doctor_rejects_candidate_without_safety_lanyard_metrics(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["class_metrics"].pop("safety_lanyard")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="harness_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate class_metrics.safety_lanyard is required" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_without_visible_ppe_telemetry(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path, bad_candidate_guard=True)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate_false_positive_guard must include visible apron_required" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_active_without_candidate_invocation(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_active"].read_text(encoding="utf-8"))
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"] = {"ppe_specialist": 1}
    _write_json(paths["candidate_active"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate_active must report model invocation telemetry for ppe_closed_set_candidate" in error
        for error in report["errors"]
    )
    assert any("candidate_active must report zero ppe_specialist invocations" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_active_without_required_model_preflight(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_active"].read_text(encoding="utf-8"))
    payload["evidence"].pop("model_preflight")
    _write_json(paths["candidate_active"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate_active must include required-model preflight" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_active_that_also_invokes_baseline_model(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_active"].read_text(encoding="utf-8"))
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"] = {
        "ppe_closed_set_candidate": 1,
        "ppe_specialist": 1,
    }
    _write_json(paths["candidate_active"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate_active must report zero ppe_specialist invocations" in error for error in report["errors"])


def test_promotion_doctor_rejects_baseline_active_that_invokes_candidate_model(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["baseline_active"].read_text(encoding="utf-8"))
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"] = {
        "ppe_specialist": 1,
        "ppe_closed_set_candidate": 1,
    }
    _write_json(paths["baseline_active"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "baseline_active must report zero ppe_closed_set_candidate invocations" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_candidate_suppression_with_candidate_invocation(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_suppression"].read_text(encoding="utf-8"))
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"] = {
        "ppe_closed_set_candidate": 1,
    }
    _write_json(paths["candidate_suppression"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate_suppression must report zero ppe_closed_set_candidate invocations" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_candidate_without_capture_preflight(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"].pop("training_capture_preflight")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_capture_preflight" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_with_pilot_capture_preflight(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_capture_preflight"]["mode"] = "pilot"
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_capture_preflight.mode must be production" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_without_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"].pop("capture_matrix_manifest")
    payload["promotion_manifest"]["training_capture_preflight"].pop("capture_matrix_manifest")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate report must include capture_matrix_manifest" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_without_label_review_import_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"].pop("label_review_import_manifest")
    payload["promotion_manifest"]["training_capture_preflight"].pop("label_review_import_manifest")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("candidate report must include label_review_import_manifest" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_with_failed_label_review_import_validation(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["label_review_import_manifest"]["updated_manifest_validation"]["ok"] = False
    payload["promotion_manifest"]["training_capture_preflight"]["label_review_import_manifest"][
        "updated_manifest_validation"
    ]["ok"] = False
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate label_review_import_manifest.updated_manifest_validation.ok must be true" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_candidate_without_label_review_clip_metadata_gate(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["label_review_import_manifest"]["training_gate"].pop(
        "requires_reviewed_clip_metadata"
    )
    payload["promotion_manifest"]["training_capture_preflight"]["label_review_import_manifest"]["training_gate"].pop(
        "requires_reviewed_clip_metadata"
    )
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate label_review_import_manifest.training_gate.requires_reviewed_clip_metadata must be true" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_candidate_with_invalid_label_review_clip_metadata(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["label_review_import_manifest"]["invalid_clip_metadata_count"] = 1
    payload["promotion_manifest"]["training_capture_preflight"]["label_review_import_manifest"][
        "invalid_clip_metadata_count"
    ] = 1
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate label_review_import_manifest.invalid_clip_metadata_count must be 0" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_candidate_without_dataset_provenance(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"].pop("training_dataset_provenance")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_dataset_provenance" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_with_dataset_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_dataset_provenance"]["source_manifest_sha256"] = "c" * 64
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_dataset_provenance.source_manifest_sha256 must match" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_without_source_lineage(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"].pop("training_source_lineage")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_source_lineage" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_source_lineage_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["capture_manifest"]["file"]["sha256"] = "c" * 64
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("training_source_lineage.capture_manifest.file.sha256 must match" in error for error in report["errors"])


def test_promotion_doctor_rejects_required_seed_import_without_export_preflight(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage()
    )
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"]["gate"]["imports"][0].pop(
        "yolo_export_preflight"
    )
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("yolo_export_preflight is required" in error for error in report["errors"])


def test_promotion_doctor_rejects_required_seed_source_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"].pop("source_recheck")
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    payload["promotion_manifest"]["training_source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["training_capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate training_source_lineage.seed_source_review.source_recheck is required" in error
        for error in report["errors"]
    )


def test_promotion_doctor_accepts_required_seed_import_with_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    payload["promotion_manifest"]["training_source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["training_capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is True
    assert report["candidate_seed_export_import_manifest"]["valid"] is True


def test_promotion_doctor_rejects_required_seed_import_without_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("seed_export_import_manifest" in error for error in report["errors"])


def test_promotion_doctor_rejects_seed_export_sidecar_without_yolo_preflight(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["imports"][0].pop("yolo_export_preflight")
    payload["promotion_manifest"]["training_source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["training_capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate seed_export_import_manifest.imports[0].yolo_export_preflight is required" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_seed_export_sidecar_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar.pop("source_recheck")
    payload["promotion_manifest"]["training_source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["training_capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any(
        "candidate seed_export_import_manifest.source_recheck is required" in error
        for error in report["errors"]
    )


def test_promotion_doctor_rejects_partial_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["promotion_manifest"]["training_source_lineage"]["seed_import_manifest"] = _approved_seed_import_lineage()
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["partial_materialization"] = True
    payload["promotion_manifest"]["training_source_lineage"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["training_capture_preflight"]["seed_export_import_manifest"] = seed_sidecar
    payload["promotion_manifest"]["seed_export_import_manifest"] = seed_sidecar
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("seed_export_import_manifest.partial_materialization must be false" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_without_selected_export(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["runtime_handoff"].pop("selected_export")
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("runtime_handoff.selected_export is required" in error for error in report["errors"])


def test_promotion_doctor_rejects_candidate_registry_sha_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    payload = json.loads(paths["candidate_report"].read_text(encoding="utf-8"))
    payload["promotion_manifest"]["runtime_handoff"]["registry_entry"]["source_export_sha256"] = "e" * 64
    _write_json(paths["candidate_report"], payload)

    report = doctor.validate_promotion(
        capability="apron_required",
        candidate_report_path=paths["candidate_report"],
        baseline_active_path=paths["baseline_active"],
        baseline_guard_path=paths["baseline_guard"],
        baseline_suppression_path=paths["baseline_suppression"],
        candidate_active_path=paths["candidate_active"],
        candidate_guard_path=paths["candidate_guard"],
        candidate_suppression_path=paths["candidate_suppression"],
    )

    assert report["ok"] is False
    assert any("source_export_sha256 must match selected_export.sha256" in error for error in report["errors"])


def test_promotion_doctor_cli_writes_report(tmp_path: Path):
    doctor = _load_doctor()
    paths = _write_promotion_inputs(tmp_path)
    out_path = tmp_path / "promotion_report.json"

    exit_code = doctor.main([
        "--capability",
        "apron_required",
        "--candidate-report",
        str(paths["candidate_report"]),
        "--baseline-active",
        str(paths["baseline_active"]),
        "--baseline-guard",
        str(paths["baseline_guard"]),
        "--baseline-suppression",
        str(paths["baseline_suppression"]),
        "--candidate-active",
        str(paths["candidate_active"]),
        "--candidate-guard",
        str(paths["candidate_guard"]),
        "--candidate-suppression",
        str(paths["candidate_suppression"]),
        "--out",
        str(out_path),
    ])

    assert exit_code == 0
    assert out_path.exists()
    assert "ready_for_runtime_registration" in out_path.read_text(encoding="utf-8")
