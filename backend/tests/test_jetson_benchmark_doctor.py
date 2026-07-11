"""Tests for Jetson benchmark evidence gate."""

from pathlib import Path
import hashlib
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "jetson_benchmark_doctor.py"
MODEL_PACKS_PATH = ROOT / "qa" / "video_eval" / "model_packs.yaml"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("jetson_benchmark_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_recheck() -> dict:
    return {
        "path": "qa/video_eval/results/apron_harness_source_recheck_2026_06_24.md",
        "sha256": "a" * 64,
        "exists": True,
        "evidence_boundary": "Fresh source recheck does not approve public seed use by itself.",
    }


def _seed_export_import_manifest() -> dict:
    return {
        "required": True,
        "valid": True,
        "partial_materialization": False,
        "sha256": "b" * 64,
        "seed_source_review_sha256": "a" * 64,
        "seed_import_manifest_sha256": "e" * 64,
        "updated_manifest_sha256": "f" * 64,
        "source_recheck": _source_recheck(),
    }


def _candidate_report_payload(
    export_sha256: str = "d" * 64,
    *,
    ok: bool = True,
    candidate_status: str = "ready_for_side_by_side_runtime_test",
) -> dict:
    return {
        "ok": ok,
        "promotion_manifest": {
            "candidate_status": candidate_status,
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
            "runtime_handoff": {
                "selected_export": {
                    "path": "/path/to/cleared/apron-harness-ppe.onnx",
                    "suffix": ".onnx",
                    "sha256": export_sha256,
                },
                "registry_entry": {
                    "model_key": "ppe_closed_set_candidate",
                    "file": "apron-harness-ppe.onnx",
                    "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                    "source_export_sha256": export_sha256,
                },
            },
            "seed_export_import_manifest": _seed_export_import_manifest(),
        },
    }


def _raw_benchmark(
    model: str = "apron-harness-ppe.onnx",
    artifact_sha256: str = "d" * 64,
    candidate_report_sha256: str = "c" * 64,
) -> dict:
    return {
        "device": "NVIDIA Jetson",
        "torch": "2.0.0",
        "cuda": True,
        "candidate_report_sha256": candidate_report_sha256,
        "imgsz": 640,
        "conf": 0.35,
        "frames": [
            {"name": "cam1-bench.jpg", "shape": [720, 1280, 3]},
            {"name": "cam2-bench.jpg", "shape": [720, 1280, 3]},
            {"name": "cam3-bench.jpg", "shape": [720, 1280, 3]},
        ],
        "models": [
            {
                "model": model,
                "model_artifact_sha256": artifact_sha256,
                "mean_ms": 120.0,
                "median_ms": 112.0,
                "p95_ms": 190.0,
                "fps_single_stream_estimate": 8.33,
                "samples": 36,
                "detections_by_frame_last_run": [],
            }
        ],
    }


def _passing_soak(
    model: str = "apron-harness-ppe.onnx",
    artifact_sha256: str = "d" * 64,
    candidate_report_sha256: str = "c" * 64,
) -> dict:
    return {
        "pack_id": "factory_ppe_3cam",
        "model": model,
        "model_artifact_sha256": artifact_sha256,
        "candidate_report_sha256": candidate_report_sha256,
        "camera_count": 3,
        "soak_minutes": 30,
        "fps_per_camera": {"cam1": 2.4, "cam2": 2.1, "cam3": 1.9},
        "mean_latency_ms": 210.0,
        "p95_latency_ms": 340.0,
        "model_server_mean_latency_ms_per_request": 260.0,
        "ram_mb": 3000,
        "gpu_utilization_percent": 72,
        "per_class_alert_count": {"apron_required": 3, "harness_required": 2},
        "detector_window_suppression": {
            "apron_required": {
                "suppressed_capabilities": ["apron_required"],
                "max_detections": 0,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "model_invocations": {"ppe_closed_set_candidate": 0},
            },
            "harness_required": {
                "suppressed_capabilities": ["harness_required"],
                "max_detections": 0,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "model_invocations": {"ppe_closed_set_candidate": 0},
            },
        },
        "false_positive_guard": {
            "apron_required": {
                "visible_class_total": 3,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "false_positive_count": 0,
            },
            "harness_required": {
                "visible_class_total": 3,
                "matching_alerts": 0,
                "unexpected_alerts": 0,
                "false_positive_count": 0,
            },
        },
        "false_positive_count": 0,
        "stream_restarts": 0,
    }


def _soak_metrics_payload() -> dict:
    return {
        "camera_count": 3,
        "soak_minutes": 30,
        "fps_per_camera": {"cam1": 2.4, "cam2": 2.1, "cam3": 1.9},
        "mean_latency_ms": 210.0,
        "p95_latency_ms": 340.0,
        "model_server_mean_latency_ms_per_request": 260.0,
        "ram_mb": 3000,
        "gpu_utilization_percent": 72,
        "stream_restarts": 0,
    }


def _scenario_result(
    scenario_id: str,
    *,
    matching_alerts: int = 0,
    unexpected_alerts: int = 0,
    class_counts: dict | None = None,
    max_detections_count: int = 0,
    status: str = "ready_to_sell",
    suppressed_capability: str | None = None,
    model_invocations: dict | None = None,
) -> dict:
    schedule_state = {
        "suppressedCapabilities": [suppressed_capability] if suppressed_capability else [],
        "suppressedCount": 1 if suppressed_capability else 0,
        "capabilities": {},
    }
    if suppressed_capability:
        schedule_state["capabilities"][suppressed_capability] = {
            "active": False,
            "mode": "detection",
            "suppressed": True,
        }
    return {
        "scenario_id": scenario_id,
        "status": status,
        "evidence": {
            "matching_alerts": [{"rule": "expected"} for _ in range(matching_alerts)],
            "unexpected_alerts": [{"rule": "unexpected"} for _ in range(unexpected_alerts)],
            "max_detections_count": max_detections_count,
            "max_detection_class_counts": class_counts or {},
            "analytics_summary": {
                "class_counts": class_counts or {},
                "schedule": {
                    "ok": True,
                    "suppressed_capabilities": [suppressed_capability] if suppressed_capability else [],
                    "suppressed_count": 1 if suppressed_capability else 0,
                    "model_invocations": model_invocations or {"ppe_closed_set_candidate": 1},
                },
            },
            "final_camera": {
                "recentDetectionClassCountsMax": class_counts or {},
                "scheduleTelemetry": {
                    "scheduleState": schedule_state,
                    "modelInvocationCounts": model_invocations or {"ppe_closed_set_candidate": 1},
                },
            },
        },
    }


def test_raw_benchmark_passes_but_does_not_unlock_production_gate(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    _write_json(raw_path, _raw_benchmark())

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        model="apron-harness-ppe.onnx",
    )

    assert report["ok"] is True
    assert report["gate_status"] == "raw_benchmark_passed_soak_missing"
    assert report["production_gate"] is False
    assert report["model_artifact_sha256"] == "d" * 64
    assert report["raw_benchmark"]["estimated_fps_per_camera_at_max_cameras"] >= 1
    assert "run_three_camera_soak_report" in report["next_required_gates"]


def test_full_jetson_gate_passes_with_raw_and_three_camera_soak(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        candidate_report_sha256=candidate_report_sha256,
        require_full_gate=True,
    )

    assert report["ok"] is True
    assert report["gate_status"] == "jetson_gate_passed"
    assert report["production_gate"] is True
    assert report["model_artifact_sha256"] == "d" * 64
    assert report["candidate_report_sha256"] == candidate_report_sha256
    assert report["soak_report"]["required_positive_alert_capabilities"] == {
        "apron_required": 3.0,
        "harness_required": 2.0,
    }
    assert report["soak_report"]["detector_window_suppression"]["apron_required"]["max_detections"] == 0.0
    assert report["soak_report"]["detector_window_suppression"]["harness_required"]["model_invocations"] == {
        "ppe_closed_set_candidate": 0.0
    }
    assert report["soak_report"]["false_positive_guard"]["apron_required"]["visible_class_total"] == 3.0
    assert report["soak_report"]["false_positive_guard"]["harness_required"]["matching_alerts"] == 0.0
    assert report["errors"] == []


def test_factory_ppe_templates_capture_candidate_identity_and_required_soak_contract(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)

    raw_template = doctor.build_raw_benchmark_template(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
    )
    soak_template = doctor.build_soak_report_template(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
    )

    assert raw_template["template"] is True
    assert raw_template["evidence_kind"] == "jetson_raw_benchmark"
    assert raw_template["candidate_report_sha256"] == candidate_report_sha256
    assert raw_template["model_artifact_sha256"] == "d" * 64
    assert len(raw_template["frames"]) == 3
    assert raw_template["models"][0]["candidate_report_sha256"] == candidate_report_sha256
    assert raw_template["models"][0]["model_artifact_sha256"] == "d" * 64
    assert "Use the same candidate_report_sha256" in raw_template["instructions"][2]

    assert soak_template["template"] is True
    assert soak_template["evidence_kind"] == "jetson_three_camera_soak"
    assert soak_template["camera_count"] == 3
    assert soak_template["candidate_report_sha256"] == candidate_report_sha256
    assert soak_template["per_class_alert_count"] == {
        "apron_required": 0,
        "harness_required": 0,
    }
    assert soak_template["detector_window_suppression"]["apron_required"]["model_invocations"] == {
        "ppe_closed_set_candidate": 0
    }
    assert soak_template["false_positive_guard"]["harness_required"]["visible_class_total"] == 0


def test_template_json_fails_validation_without_traceback(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw_template.json"
    soak_path = tmp_path / "soak_template.json"
    raw_template = doctor.build_raw_benchmark_template(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        candidate_report_sha256="c" * 64,
    )
    raw_template["models"][0]["mean_ms"] = "REPLACE_ME"
    _write_json(raw_path, raw_template)
    _write_json(
        soak_path,
        doctor.build_soak_report_template(
            pack_id="factory_ppe_3cam",
            pack_path=MODEL_PACKS_PATH,
            model="apron-harness-ppe.onnx",
            model_artifact_sha256="d" * 64,
            candidate_report_sha256="c" * 64,
        ),
    )

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        candidate_report_sha256="c" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "raw benchmark mean_ms must be numeric" in errors
    assert "soak report per_class_alert_count must include positive counts" in errors


def test_build_soak_report_from_scenario_results_passes_gate(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    metrics_path = tmp_path / "soak_metrics.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(metrics_path, _soak_metrics_payload())
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))

    active_apron = tmp_path / "active_apron.json"
    active_harness = tmp_path / "active_harness.json"
    guard_apron = tmp_path / "guard_apron.json"
    guard_harness = tmp_path / "guard_harness.json"
    suppression_apron = tmp_path / "suppression_apron.json"
    suppression_harness = tmp_path / "suppression_harness.json"
    _write_json(active_apron, _scenario_result("factory_missing_apron_active", matching_alerts=2, max_detections_count=4))
    _write_json(active_harness, _scenario_result("factory_missing_harness_active", matching_alerts=1, max_detections_count=4))
    _write_json(
        guard_apron,
        _scenario_result(
            "factory_apron_false_positive_guard",
            class_counts={"person": 1, "denim apron": 3, "work apron": 1},
            max_detections_count=5,
        ),
    )
    _write_json(
        guard_harness,
        _scenario_result(
            "factory_harness_false_positive_guard",
            class_counts={"person": 1, "safety harness": 2, "safety lanyard": 1},
            max_detections_count=4,
        ),
    )
    _write_json(
        suppression_apron,
        _scenario_result(
            "factory_apron_detector_window_suppression",
            suppressed_capability="apron_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )
    _write_json(
        suppression_harness,
        _scenario_result(
            "factory_harness_detector_window_suppression",
            suppressed_capability="harness_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )

    soak = doctor.build_soak_report_from_results(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        soak_metrics_path=metrics_path,
        active_result_paths={"apron_required": active_apron, "harness_required": active_harness},
        guard_result_paths={"apron_required": guard_apron, "harness_required": guard_harness},
        suppression_result_paths={"apron_required": suppression_apron, "harness_required": suppression_harness},
    )
    _write_json(soak_path, soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert soak["source_result_failures"] == []
    assert soak["per_class_alert_count"] == {"apron_required": 2.0, "harness_required": 1.0}
    assert soak["false_positive_guard"]["apron_required"]["visible_class_total"] == 4.0
    assert soak["false_positive_guard"]["harness_required"]["visible_class_total"] == 3.0
    assert soak["detector_window_suppression"]["apron_required"]["model_invocations"] == {
        "ppe_closed_set_candidate": 0.0
    }
    assert report["ok"] is True
    assert report["gate_status"] == "jetson_gate_passed"


def test_build_soak_report_preserves_false_positive_guard_failure(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    metrics_path = tmp_path / "soak_metrics.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(metrics_path, _soak_metrics_payload())
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))

    active_apron = tmp_path / "active_apron.json"
    active_harness = tmp_path / "active_harness.json"
    guard_apron = tmp_path / "guard_apron.json"
    guard_harness = tmp_path / "guard_harness.json"
    suppression_apron = tmp_path / "suppression_apron.json"
    suppression_harness = tmp_path / "suppression_harness.json"
    _write_json(active_apron, _scenario_result("factory_missing_apron_active", matching_alerts=1))
    _write_json(active_harness, _scenario_result("factory_missing_harness_active", matching_alerts=1))
    _write_json(guard_apron, _scenario_result("factory_apron_false_positive_guard", class_counts={"person": 2}))
    _write_json(
        guard_harness,
        _scenario_result("factory_harness_false_positive_guard", class_counts={"safety harness": 1}),
    )
    _write_json(
        suppression_apron,
        _scenario_result(
            "factory_apron_detector_window_suppression",
            suppressed_capability="apron_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )
    _write_json(
        suppression_harness,
        _scenario_result(
            "factory_harness_detector_window_suppression",
            suppressed_capability="harness_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )

    soak = doctor.build_soak_report_from_results(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        soak_metrics_path=metrics_path,
        active_result_paths={"apron_required": active_apron, "harness_required": active_harness},
        guard_result_paths={"apron_required": guard_apron, "harness_required": guard_harness},
        suppression_result_paths={"apron_required": suppression_apron, "harness_required": suppression_harness},
    )
    _write_json(soak_path, soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert soak["false_positive_guard"]["apron_required"]["visible_class_total"] == 0.0
    assert report["ok"] is False
    assert any(
        "false_positive_guard apron_required must include visible PPE evidence" in error
        for error in report["errors"]
    )


def test_cli_builds_soak_report_from_result_files(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    metrics_path = tmp_path / "soak_metrics.json"
    out_path = tmp_path / "factory_ppe_3cam_soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    _write_json(metrics_path, _soak_metrics_payload())

    active_apron = tmp_path / "active_apron.json"
    active_harness = tmp_path / "active_harness.json"
    guard_apron = tmp_path / "guard_apron.json"
    guard_harness = tmp_path / "guard_harness.json"
    suppression_apron = tmp_path / "suppression_apron.json"
    suppression_harness = tmp_path / "suppression_harness.json"
    _write_json(active_apron, _scenario_result("factory_missing_apron_active", matching_alerts=2))
    _write_json(active_harness, _scenario_result("factory_missing_harness_active", matching_alerts=1))
    _write_json(guard_apron, _scenario_result("factory_apron_false_positive_guard", class_counts={"denim apron": 2}))
    _write_json(guard_harness, _scenario_result("factory_harness_false_positive_guard", class_counts={"safety harness": 2}))
    _write_json(
        suppression_apron,
        _scenario_result(
            "factory_apron_detector_window_suppression",
            suppressed_capability="apron_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )
    _write_json(
        suppression_harness,
        _scenario_result(
            "factory_harness_detector_window_suppression",
            suppressed_capability="harness_required",
            model_invocations={"ppe_closed_set_candidate": 0},
        ),
    )

    exit_code = doctor.main(
        [
            "--pack",
            "factory_ppe_3cam",
            "--model-pack",
            str(MODEL_PACKS_PATH),
            "--model",
            "apron-harness-ppe.onnx",
            "--candidate-report",
            str(candidate_path),
            "--soak-metrics",
            str(metrics_path),
            "--active-result",
            f"apron_required={active_apron}",
            "--active-result",
            f"harness_required={active_harness}",
            "--guard-result",
            f"apron_required={guard_apron}",
            "--guard-result",
            f"harness_required={guard_harness}",
            "--suppression-result",
            f"apron_required={suppression_apron}",
            "--suppression-result",
            f"harness_required={suppression_harness}",
            "--build-soak-report",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["candidate_report_sha256"] == _sha256_file(candidate_path)
    assert payload["source_results"]["active"]["apron_required"]["scenario_id"] == "factory_missing_apron_active"
    assert payload["detector_window_suppression"]["harness_required"]["suppressed"] is True


def test_full_jetson_gate_derives_candidate_identity_from_candidate_report(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert report["ok"] is True
    assert report["gate_status"] == "jetson_gate_passed"
    assert report["model_artifact_sha256"] == "d" * 64
    assert report["candidate_report_sha256"] == candidate_report_sha256
    assert report["inputs"]["candidate_report"] == str(candidate_path)
    assert report["candidate_report"]["selected_export_sha256"] == "d" * 64
    assert report["candidate_report"]["sha256"] == candidate_report_sha256
    assert report["candidate_report"]["class_metrics"]["safety_lanyard"]["recall"] == 0.86
    assert report["candidate_report"]["seed_source_recheck"]["sha256"] == "a" * 64


def test_full_jetson_gate_rejects_candidate_report_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    payload = _candidate_report_payload()
    payload["promotion_manifest"]["seed_export_import_manifest"].pop("source_recheck")
    _write_json(candidate_path, payload)
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert (
        "candidate seed_export_import_manifest.source_recheck is required"
        in report["errors"]
    )


def test_full_jetson_gate_rejects_candidate_report_without_safety_lanyard_metrics(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    payload = _candidate_report_payload()
    payload["promotion_manifest"]["class_metrics"].pop("safety_lanyard")
    _write_json(candidate_path, payload)
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert any(
        "candidate report missing promotion_manifest.class_metrics.safety_lanyard" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_candidate_report_export_hash_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload(export_sha256="e" * 64))
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "raw benchmark model_artifact_sha256 does not match expected value" in errors
    assert "soak report model_artifact_sha256 does not match expected value" in errors


def test_full_jetson_gate_rejects_explicit_candidate_report_hash_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        candidate_report_path=candidate_path,
        model="apron-harness-ppe.onnx",
        candidate_report_sha256="c" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert "--candidate-report-sha256 does not match candidate report file sha256" in report["errors"]


def test_full_jetson_gate_rejects_artifact_hash_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(raw_path, _raw_benchmark(artifact_sha256="d" * 64))
    _write_json(soak_path, _passing_soak(artifact_sha256="e" * 64))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    errors = "\n".join(report["errors"])
    assert "raw benchmark model_artifact_sha256 must match soak report model_artifact_sha256" in errors
    assert "soak report model_artifact_sha256 does not match expected value" in errors


def test_full_jetson_gate_rejects_candidate_report_hash_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256="c" * 64))
    _write_json(soak_path, _passing_soak(candidate_report_sha256="e" * 64))

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        candidate_report_sha256="c" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    errors = "\n".join(report["errors"])
    assert "raw benchmark candidate_report_sha256 must match soak report candidate_report_sha256" in errors
    assert "soak report candidate_report_sha256 does not match expected value" in errors


def test_full_jetson_gate_rejects_missing_required_ppe_alert_counts(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak["per_class_alert_count"] = {"apron_required": 2, "helmet_required": 4}
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    assert any(
        "positive counts for: harness_required" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_missing_detector_window_suppression(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak.pop("detector_window_suppression")
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    assert any(
        "detector_window_suppression must include required capabilities" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_detector_window_model_invocations(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak["detector_window_suppression"]["harness_required"]["model_invocations"] = {
        "ppe_closed_set_candidate": 1
    }
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    assert any(
        "detector_window_suppression harness_required invoked a model" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_missing_false_positive_guard(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak.pop("false_positive_guard")
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    assert any(
        "false_positive_guard must include required capabilities" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_false_positive_guard_alerts(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak["false_positive_guard"]["harness_required"]["matching_alerts"] = 1
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        model_artifact_sha256="d" * 64,
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    assert any(
        "false_positive_guard harness_required emitted alerts" in error
        for error in report["errors"]
    )


def test_full_jetson_gate_rejects_missing_and_weak_soak_metrics(tmp_path: Path):
    doctor = _load_doctor()
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    bad_soak = _passing_soak()
    bad_soak.pop("per_class_alert_count")
    bad_soak.update(
        {
            "fps_per_camera": {"cam1": 0.8, "cam2": 0.7, "cam3": 0.6},
            "mean_latency_ms": 500.0,
            "model_server_mean_latency_ms_per_request": 700.0,
            "ram_mb": 4096,
            "gpu_utilization_percent": 95,
            "false_positive_count": 2,
            "stream_restarts": 1,
        }
    )
    _write_json(raw_path, _raw_benchmark())
    _write_json(soak_path, bad_soak)

    report = doctor.validate_jetson_benchmark(
        pack_id="factory_ppe_3cam",
        pack_path=MODEL_PACKS_PATH,
        raw_benchmark_path=raw_path,
        soak_report_path=soak_path,
        model="apron-harness-ppe.onnx",
        require_full_gate=True,
    )

    assert report["ok"] is False
    assert report["gate_status"] == "not_ready"
    errors = "\n".join(report["errors"])
    assert "missing required metrics" in errors
    assert "min fps_per_camera" in errors
    assert "mean_latency_ms" in errors
    assert "model_server_mean_latency_ms_per_request" in errors
    assert "ram_mb" in errors
    assert "gpu_utilization_percent" in errors
    assert "false_positive_count" in errors
    assert "stream_restarts" in errors


def test_cli_writes_full_gate_report(tmp_path: Path):
    doctor = _load_doctor()
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    soak_path = tmp_path / "soak.json"
    out_path = tmp_path / "jetson_gate.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_report_sha256 = _sha256_file(candidate_path)
    _write_json(raw_path, _raw_benchmark(candidate_report_sha256=candidate_report_sha256))
    _write_json(soak_path, _passing_soak(candidate_report_sha256=candidate_report_sha256))

    exit_code = doctor.main(
        [
            "--pack",
            "factory_ppe_3cam",
            "--model-pack",
            str(MODEL_PACKS_PATH),
            "--model",
            "apron-harness-ppe.onnx",
            "--candidate-report",
            str(candidate_path),
            "--raw-benchmark",
            str(raw_path),
            "--soak-report",
            str(soak_path),
            "--require-full-gate",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "jetson_gate_passed"
    assert payload["input_file_sha256s"] == {
        "candidate_report": candidate_report_sha256,
        "raw_benchmark": _sha256_file(raw_path),
        "soak_report": _sha256_file(soak_path),
    }
