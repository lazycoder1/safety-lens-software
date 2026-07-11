"""Tests for model-pack saved evidence audit."""

from pathlib import Path
import importlib.util
import json
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "model_pack_evidence_doctor.py"
MODEL_PACKS_PATH = ROOT / "qa" / "video_eval" / "model_packs.yaml"
MANIFEST_PATH = ROOT / "qa" / "video_eval" / "manifest.yaml"
RESULT_DIR = ROOT / "qa" / "video_eval" / "results"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("model_pack_evidence_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_pack_evidence_doctor_accepts_current_yolo26_saved_evidence():
    doctor = _load_doctor()

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is True
    assert report["stats"]["pack_count"] >= 5
    assert report["stats"]["shared_source_count"] >= 10
    assert report["stats"]["source_research_max_age_days"] == 14
    assert report["stats"]["unique_scenario_count"] >= 72
    assert report["stats"]["ready_result_count"] == report["stats"]["unique_scenario_count"]
    assert report["stats"]["yaml_lifecycle_check_count"] >= 66
    assert report["stats"]["model_evidence_check_count"] >= report["stats"]["unique_scenario_count"]
    assert report["stats"]["scenario_isolation_check_count"] == report["stats"]["unique_scenario_count"]
    assert report["stats"]["log_evidence_check_count"] == report["stats"]["unique_scenario_count"]
    assert report["stats"]["active_window_check_count"] >= 42
    assert report["stats"]["delivery_check_count"] >= 52
    assert report["stats"]["detector_suppression_check_count"] >= 24
    assert report["stats"]["capability_window_config_check_count"] >= 60
    assert report["stats"]["detector_window_gap_count"] == 0
    assert report["errors"] == []
    base_coverage = report["packs"]["base_3cam"]["detector_window_coverage"]
    assert set(base_coverage["covered_capabilities"]) >= {
        "person_presence",
        "vehicle_presence",
        "mobile_phone",
        "zone_intrusion",
        "queue_monitoring",
        "crowd_count_threshold",
        "animal_presence",
    }
    assert set(base_coverage["documented_gaps"]) == set()


def test_model_pack_evidence_doctor_rejects_capability_window_without_daily_weekly_shape(tmp_path: Path):
    doctor = _load_doctor()
    config_path = tmp_path / "focused.yaml"
    config_path.write_text(
        """
cameras:
  eval_factory_apron_detector_window_suppression:
    capability_windows:
      apron_required:
        active: false
        windows:
          - from: "09:00"
            to: "17:00"
""",
        encoding="utf-8",
    )
    scenario = {
        "config_path": str(config_path),
        "camera_id": "eval_factory_apron_detector_window_suppression",
        "expected": {
            "analytics": [
                {
                    "type": "detector_suppression",
                    "capability": "apron_required",
                }
            ]
        },
    }

    errors, checked = doctor._check_capability_window_config(scenario)

    assert checked == 1
    assert errors == [
        "apron_required: capability window needs days plus from/to time for daily/weekly proof"
    ]


def test_model_pack_evidence_doctor_can_skip_archived_hospital_pack():
    doctor = _load_doctor()

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
        skipped_model_packs={"pose_fall_3cam"},
    )

    assert report["ok"] is True
    assert report["skipped_model_packs"] == ["pose_fall_3cam"]
    assert "pose_fall_3cam" not in report["packs"]
    rendered = json.dumps(report)
    assert "hospital_public_area_person_down_public" not in rendered
    assert "hospital_exercise_false_positive_guard" not in rendered


def test_model_pack_evidence_doctor_rejects_detector_window_regression(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "person_presence_detector_window_suppression.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evidence"]["max_detections_count"] = 1
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"]["coco_primary"] = 1
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "person_presence_detector_window_suppression" in errors
    assert "detector suppression emitted more than 0 detections" in errors
    assert "model coco_primary invocations 1 exceed 0" in errors


def test_model_pack_evidence_doctor_rejects_active_window_regression(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "factory_missing_apron_active.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    schedule = payload["evidence"]["analytics_summary"]["schedule"]
    schedule["suppressed_capabilities"] = ["apron_required"]
    schedule["suppressed_count"] = 1
    schedule["model_invocations"]["ppe_specialist"] = 0
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_missing_apron_active" in errors
    assert "active window suppressed planned capability apron_required" in errors
    assert "active window model ppe_specialist invocations 0 are not positive" in errors


def test_model_pack_evidence_doctor_rejects_legacy_factory_ppe_training_policy(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    plan = payload["packs"]["factory_ppe_3cam"]["production_training_plan"]
    plan["scale_up_train_command"] = plan["scale_up_train_command"].replace("yolo26s.pt", "yolo11n.pt")
    plan["fallback_train_command"] = plan["scale_up_train_command"]
    plan["conservative_baselines"] = ["yolo11n", "yolov8n"]
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "fallback_train_command is deprecated" in errors
    assert "conservative_baselines is deprecated" in errors
    assert "scale_up_train_command model yolo11n.pt must be one of ['yolo26n.pt', 'yolo26s.pt']" in errors


def test_model_pack_evidence_doctor_rejects_factory_ppe_registry_copy_without_promotion_reports(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["packs"]["factory_ppe_3cam"]["production_training_plan"][
        "registry_copy_command"
    ] = (
        ".venv/bin/python scripts/apron_harness_model_registry_doctor.py "
        "--candidate-report /path/to/cleared/apron_harness_candidate_report.json "
        "--copy --out /path/to/cleared/apron_harness_model_registry_report.json"
    )
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_ppe_3cam registry_copy_command must pass --apron-promotion-report" in errors
    assert "factory_ppe_3cam registry_copy_command must pass --harness-promotion-report" in errors


def test_model_pack_evidence_doctor_rejects_runtime_model_drift(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "person_presence_active.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    models = payload["evidence"]["health"]["models"]
    coco = next(model for model in models if model["model_key"] == "coco_primary")
    coco["filename"] = "yolo11n.pt"
    coco["download_url"] = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "person_presence_active" in errors
    assert "runtime model coco_primary filename yolo11n.pt does not match yolo26n.pt" in errors


def test_model_pack_evidence_doctor_rejects_missing_yaml_plan_evidence(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "person_presence_active.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["yaml_commands"] = [
        command
        for command in payload["yaml_commands"]
        if "plan" not in [str(arg) for arg in command.get("args") or []]
    ]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "person_presence_active" in errors
    assert "missing YAML plan command evidence" in errors


def test_model_pack_evidence_doctor_rejects_unisolated_focused_scenario(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "factory_apron_false_positive_guard.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    final_camera = payload["evidence"]["final_camera"]
    final_camera["capabilities"] = ["person_presence", "apron_required"]
    final_camera["execution_plan"]["capabilities"] = ["person_presence", "apron_required"]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_apron_false_positive_guard" in errors
    assert "scenario is not one-detection isolated" in errors
    assert "apron_required" in errors
    assert "person_presence" in errors


def test_model_pack_evidence_doctor_rejects_ppe_production_claim_without_closed_set_candidate(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["packs"]["factory_ppe_3cam"]["status"] = "ready_to_sell_production_compliance"
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_ppe_3cam cannot claim production readiness" in errors
    assert "ppe_closed_set_candidate is registered" in errors
    assert "active PPE model key is ppe_specialist" in errors


def test_model_pack_evidence_doctor_rejects_missing_source_research_metadata(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["shared_sources"]["paddleocr_ppocrv6"].pop("checked")
    payload["shared_sources"]["paddleocr_ppocrv6"].pop("decision")
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "shared_sources.paddleocr_ppocrv6 must include checked date" in errors
    assert "shared_sources.paddleocr_ppocrv6 must record accepted/rejected/candidate decision" in errors


def test_model_pack_evidence_doctor_rejects_incomplete_runtime_source_research(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["shared_sources"]["ultralytics_yolo26"].pop("version_note", None)
    payload["shared_sources"]["ultralytics_yolo26"].pop("export_note", None)
    payload["shared_sources"]["ultralytics_yolo26"].pop("runtime_note", None)
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "shared_sources.ultralytics_yolo26 must include release_note or version_note for registry models" in errors
    assert "shared_sources.ultralytics_yolo26 must include export_note for registry models" in errors
    assert "shared_sources.ultralytics_yolo26 must include runtime_note or edge_note for registry models" in errors


def test_model_pack_evidence_doctor_rejects_incomplete_registry_model_handoff(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    plate_detector = payload["packs"]["anpr_gate_1cam"]["registry_models"]["plate_recognition"]
    plate_detector.pop("provenance_note", None)
    plate_detector.pop("jetson_device", None)
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "anpr_gate_1cam.plate_recognition must include jetson_device" in errors
    assert "anpr_gate_1cam.plate_recognition manual/internal source must include provenance_note" in errors


def test_model_pack_evidence_doctor_rejects_stale_source_research(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["shared_sources"]["paddleocr_ppocrv6"]["checked"] = "2026-05-01"
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "shared_sources.paddleocr_ppocrv6 checked date 2026-05-01 is older than 14 days" in errors


def test_model_pack_evidence_doctor_rejects_invalid_source_research_max_age_policy(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["policy"]["source_research_max_age_days"] = 0
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    assert report["stats"]["source_research_max_age_days"] == 14
    errors = "\n".join(report["errors"])
    assert "policy.source_research_max_age_days must be a positive integer" in errors


def test_model_pack_evidence_doctor_rejects_missing_registry_source_ref(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["packs"]["anpr_gate_1cam"]["registry_models"]["ppocrv6_tiny"].pop("source_ref")
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "anpr_gate_1cam.ppocrv6_tiny must include source_ref for researched HTTP source" in errors


def test_model_pack_evidence_doctor_rejects_undocumented_detector_window_gaps(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    base_pack = payload["packs"]["base_3cam"]
    base_pack.pop("detector_window_gaps", None)
    base_pack["evidence_scenarios"] = [
        scenario_id
        for scenario_id in base_pack["evidence_scenarios"]
        if scenario_id != "animal_presence_detector_window_suppression"
    ]
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=model_packs_path,
        manifest_path=MANIFEST_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "base_3cam: animal_presence lacks detector-window suppression evidence or documented gap" in errors
    assert "crowd_count_threshold lacks detector-window suppression evidence or documented gap" not in errors


def test_model_pack_evidence_doctor_rejects_missing_delivery_proof(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "factory_missing_apron_active.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evidence"]["delivery_summary"].pop("browser_sound")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_missing_apron_active" in errors
    assert "expected output browser_sound missing from delivery_summary" in errors


def test_model_pack_evidence_doctor_rejects_missing_log_evidence(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "factory_missing_apron_active.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evidence"]["health"]["storage"].pop("logs")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "factory_missing_apron_active" in errors
    assert "runtime health log storage missing dir" in errors
    assert "runtime health log storage has no files" in errors
    assert "runtime health log storage has no bytes" in errors


def test_model_pack_evidence_doctor_rejects_missing_receiver_capture(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "construction_person_email_delivery.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evidence"]["smtp_capture"]["message_count"] = 0
    payload["evidence"]["smtp_capture"]["messages"] = []
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_model_pack_evidence(
        model_packs_path=MODEL_PACKS_PATH,
        manifest_path=MANIFEST_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "construction_person_email_delivery" in errors
    assert "SMTP delivery captured 0 messages, expected at least 1" in errors


def test_model_pack_evidence_doctor_cli_writes_report(tmp_path: Path):
    doctor = _load_doctor()
    out_path = tmp_path / "model_pack_evidence_doctor.json"

    exit_code = doctor.main(
        [
            "--model-packs",
            str(MODEL_PACKS_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--result-dir",
            str(RESULT_DIR),
            "--skip-model-pack",
            "pose_fall_3cam",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["skipped_model_packs"] == ["pose_fall_3cam"]
    assert "pose_fall_3cam" not in report["packs"]
    assert report["stats"]["ready_result_count"] == report["stats"]["unique_scenario_count"]
