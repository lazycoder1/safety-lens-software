"""Tests for apron/harness readiness audit."""

from pathlib import Path
import csv
import importlib.util
import json
import shutil

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCTOR_PATH = ROOT / "scripts" / "apron_harness_readiness_doctor.py"
RUNNER_PATH = ROOT / "scripts" / "apron_harness_candidate_runtime_runner.py"
SOURCE_REVIEW_RUNNER_PATH = ROOT / "scripts" / "apron_harness_source_review_runner.py"
CONTROLLED_CAPTURE_RUNNER_PATH = ROOT / "scripts" / "apron_harness_controlled_capture_runner.py"
CANDIDATE_TRAINING_RUNNER_PATH = ROOT / "scripts" / "apron_harness_candidate_training_runner.py"
JETSON_GATE_RUNNER_PATH = ROOT / "scripts" / "apron_harness_jetson_gate_runner.py"
MODEL_PACKS_PATH = ROOT / "qa" / "video_eval" / "model_packs.yaml"
RESULT_DIR = ROOT / "qa" / "video_eval" / "results"
EXPECTED_MISSING_PPE_LABEL_POLICY = "derive_missing_ppe_from_person_to_visible_ppe_association"


def _load_doctor():
    spec = importlib.util.spec_from_file_location("apron_harness_readiness_doctor", DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_seed_source_doctor():
    path = ROOT / "scripts" / "apron_harness_seed_source_doctor.py"
    spec = importlib.util.spec_from_file_location("apron_harness_seed_source_doctor_for_readiness_tests", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_candidate_runtime_runner():
    spec = importlib.util.spec_from_file_location(
        "apron_harness_candidate_runtime_runner_for_readiness_tests",
        RUNNER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_source_review_runner():
    spec = importlib.util.spec_from_file_location(
        "apron_harness_source_review_runner_for_readiness_tests",
        SOURCE_REVIEW_RUNNER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_controlled_capture_runner():
    spec = importlib.util.spec_from_file_location(
        "apron_harness_controlled_capture_runner_for_readiness_tests",
        CONTROLLED_CAPTURE_RUNNER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_candidate_training_runner():
    spec = importlib.util.spec_from_file_location(
        "apron_harness_candidate_training_runner_for_readiness_tests",
        CANDIDATE_TRAINING_RUNNER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_jetson_gate_runner():
    spec = importlib.util.spec_from_file_location(
        "apron_harness_jetson_gate_runner_for_readiness_tests",
        JETSON_GATE_RUNNER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate_training_packet() -> dict:
    source_recheck = ROOT / "qa" / "video_eval" / "results" / "apron_harness_source_recheck_2026_06_24.md"
    return {
        "candidate_training_execution_plan": {
            "status": "blocked_until_reviewed_production_manifest_and_training_dataset",
            "required_model_key": "ppe_closed_set_candidate",
            "training_model": "yolo26n.pt",
            "required_input_status": {
                "reviewed_capture_manifest": {
                    "path": "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml",
                    "exists": False,
                    "status": "missing_reviewed_capture_manifest",
                },
                "source_recheck_artifact": {
                    "path": str(source_recheck),
                    "sha256": _sha256_file(source_recheck) if source_recheck.exists() else "",
                    "evidence_boundary": (
                        "Fresh source research evidence only; this does not approve "
                        "any source for training."
                    ),
                },
            },
            "success_criteria": {
                "training_preflight": "status=ready_to_train",
                "candidate_report": "candidate_status=ready_for_side_by_side_runtime_test",
            },
            "steps": [
                {
                    "step": 1,
                    "id": "training_preflight_plan",
                    "command": "preflight --out-plan /path/to/cleared/apron_harness_yolo26n_result.plan.json",
                    "writes": ["/path/to/cleared/apron_harness_yolo26n_result.plan.json"],
                    "pass_signal": "status=ready_to_train",
                },
                {
                    "step": 2,
                    "id": "train_export_candidate",
                    "command": "train --out-plan /path/to/cleared/apron_harness_yolo26n_result.json --execute",
                    "writes": ["/path/to/cleared/apron_harness_yolo26n_result.json"],
                },
                {
                    "step": 3,
                    "id": "candidate_doctor_report",
                    "command": "candidate-doctor --training-result /path/to/cleared/apron_harness_yolo26n_result.json",
                },
                {
                    "step": 5,
                    "id": "registry_copy_after_promotions",
                    "command": "registry-copy",
                },
            ],
        }
    }


def _candidate_runtime_packet(
    expected_result_path: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    result_path = expected_result_path or Path("qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json")
    config = config_path or Path("qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml")
    return {
        "candidate_runtime_execution_plan": {
            "required_model_key": "ppe_closed_set_candidate",
            "one_detection_at_a_time": True,
            "scenario_order": ["factory_missing_apron_active_closed_set"],
            "success_criteria": {
                "active": "matching alert and ppe_closed_set_candidate invocation required",
                "suppression": "zero model invocations outside the active window",
            },
            "runbook": {},
            "steps": [
                {
                    "step": 1,
                    "scenario_id": "factory_missing_apron_active_closed_set",
                    "capability": "apron_required",
                    "role": "active",
                    "config_path": config,
                    "expected_result_path": result_path,
                    "commands": {
                        "backup": "backup",
                        "validate": "validate",
                        "plan": "plan",
                        "apply": "apply",
                        "run": "run-fails",
                        "restore": "restore",
                    },
                }
            ],
        },
        "model_registry_handoff": {
            "registry_status": "registered",
            "destination_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "destination_exists": True,
            "metadata_valid": True,
        },
    }


def _jetson_gate_packet() -> dict:
    return {
        "jetson_gate_execution_plan": {
            "status": "blocked_until_candidate_report_raw_benchmark_and_soak_report",
            "required_model_key": "ppe_closed_set_candidate",
            "model": "apron-harness-ppe.onnx",
            "steps": [],
            "full_gate_command": (
                "jetson-gate --candidate-report qa/video_eval/results/apron_harness_candidate_report.json "
                "--raw-benchmark /path/to/cleared/factory_ppe_raw_benchmark.json "
                "--soak-report /path/to/cleared/factory_ppe_3cam_soak.json "
                "--require-full-gate --out qa/video_eval/results/factory_ppe_jetson_gate.json"
            ),
            "success_criteria": [
                "candidate report SHA and selected-export SHA match across evidence",
                "production promotion remains blocked until this full gate passes",
            ],
        }
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_markdown_cell_sanitizes_inline_code_breakers():
    doctor = _load_doctor()

    assert doctor._md_cell("run `sw_vers` | retry") == "run 'sw_vers' \\| retry"


def _attach_label_review_import_sidecar(capture_manifest: Path) -> None:
    manifest = yaml.safe_load(capture_manifest.read_text(encoding="utf-8")) or {}
    label_path = capture_manifest.parent / "labels" / "train" / "frame_00001.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "0 0.500 0.500 0.750 0.900\n"
        "1 0.520 0.560 0.220 0.450\n"
        "2 0.500 0.500 0.420 0.700\n"
        "3 0.700 0.500 0.100 0.500\n",
        encoding="utf-8",
    )
    clips = manifest.get("clips") if isinstance(manifest.get("clips"), list) else []
    source_clip_id = str((clips[0] or {}).get("clip_id") or "replace_with_clip_id") if clips else "replace_with_clip_id"
    manifest["yolo_labels"] = [
        {
            "path": str(label_path.relative_to(capture_manifest.parent)),
            "review_status": "approved",
            "reviewer": "qa_reviewer",
            "reviewed_at": "2026-06-22T00:00:00+00:00",
            "source_clip_id": source_clip_id,
            "split": "train",
        }
    ]
    capture_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    counts = (manifest.get("counts") or {}).get("labeled_images_per_class") or {}
    sidecar = {
        "generated_at": "2026-06-22T00:00:00+00:00",
        "kind": "apron_harness_label_review_import_manifest",
        "valid": True,
        "source_manifest": str(capture_manifest.with_suffix(".source.yaml")),
        "source_manifest_sha256": "b" * 64,
        "label_review_csv": str(capture_manifest.with_suffix(".label_review.csv")),
        "label_review_csv_sha256": "c" * 64,
        "updated_manifest": str(capture_manifest),
        "updated_manifest_sha256": _sha256_file(capture_manifest),
        "existing_label_count": 0,
        "imported_label_count": 1,
        "imported_clip_count": 0,
        "skipped_label_count": 0,
        "invalid_approved_label_count": 0,
        "invalid_clip_metadata_count": 0,
        "merged_label_count": 1,
        "updated_labeled_images_per_class": counts,
        "updated_manifest_validation": {
            "checked": True,
            "mode": "production",
            "schema_only": False,
            "ok": True,
            "manifest_sha256": _sha256_file(capture_manifest),
            "errors": [],
            "warnings": [],
        },
        "training_gate": {
            "requires_approved_label_review_rows": True,
            "requires_review_metadata": True,
            "requires_cleared_permission": True,
            "requires_non_repo_raw_storage_refs": True,
            "requires_recomputed_label_counts": True,
            "requires_updated_manifest_validation": True,
            "requires_reviewed_clip_metadata": True,
        },
    }
    capture_manifest.with_suffix(capture_manifest.suffix + ".label_review_import.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )


def _promotion_active_summary(capability: str, model_key: str) -> dict:
    return {
        "scenario_id": f"{capability}_active",
        "status": "ready_to_sell",
        "max_detections": 6,
        "matching_alerts": 1,
        "unexpected_alerts": 0,
        "visible_class_total": 0,
        "suppressed_capabilities": [],
        "model_invocations": {model_key: 1},
        "screenshot_ok": True,
    }


def _promotion_guard_summary(capability: str, model_key: str) -> dict:
    return {
        "scenario_id": f"{capability}_false_positive_guard",
        "status": "ready_to_sell",
        "max_detections": 4,
        "matching_alerts": 0,
        "unexpected_alerts": 0,
        "visible_class_total": 3,
        "suppressed_capabilities": [],
        "model_invocations": {model_key: 1},
        "screenshot_ok": True,
    }


def _promotion_suppression_summary(capability: str, model_key: str) -> dict:
    return {
        "scenario_id": f"{capability}_detector_window_suppression",
        "status": "ready_to_sell",
        "max_detections": 0,
        "matching_alerts": 0,
        "unexpected_alerts": 0,
        "visible_class_total": 0,
        "suppressed_capabilities": [capability],
        "model_invocations": {model_key: 0},
        "screenshot_ok": True,
    }


def _promotion_runtime_group(capability: str, model_key: str) -> dict:
    return {
        "active": _promotion_active_summary(capability, model_key),
        "false_positive_guard": _promotion_guard_summary(capability, model_key),
        "suppression": _promotion_suppression_summary(capability, model_key),
    }


def test_candidate_training_runner_plan_marks_missing_reviewed_inputs_unsupplied(tmp_path: Path):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    for path in (dataset_yaml, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")
    missing_capture_manifest = tmp_path / "missing_reviewed_capture.yaml"
    training_result = tmp_path / "training_result.json"

    plan = runner.build_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(missing_capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_result.with_suffix(".plan.json")),
    )

    assert plan["dataset_yaml_supplied"] is True
    assert plan["seed_import_manifest_supplied"] is True
    assert plan["capture_manifest_supplied"] is False
    assert plan["training_result_supplied"] is True
    assert plan["required_input_status"]["reviewed_capture_manifest"]["exists"] is False
    assert plan["required_input_status"]["reviewed_capture_manifest"]["status"] == (
        "missing_reviewed_capture_manifest"
    )
    assert plan["success_criteria"]["training_preflight"] == "status=ready_to_train"
    artifacts = {item["name"]: item for item in plan["artifact_status"]}
    assert artifacts["dataset_yaml"]["exists"] is True
    assert len(artifacts["dataset_yaml"]["sha256"]) == 64
    assert artifacts["seed_import_manifest"]["exists"] is True
    assert artifacts["source_recheck_artifact"]["exists"] is True
    assert artifacts["source_recheck_artifact"]["sha_matches"] is True
    assert artifacts["capture_manifest"]["exists"] is False
    assert artifacts["capture_manifest"]["blockers"] == ["missing"]
    assert artifacts["training_result"]["output"] is True
    assert artifacts["training_result"]["ok"] is True
    assert artifacts["candidate_report"]["output"] is True
    assert artifacts["candidate_report"]["ok"] is True
    assert artifacts["apron_promotion_report"]["exists"] is False
    assert artifacts["apron_promotion_report"]["blockers"] == ["missing"]
    assert artifacts["harness_promotion_report"]["exists"] is False


def test_candidate_training_runner_plan_reports_stale_source_recheck_artifact(tmp_path: Path):
    runner = _load_candidate_training_runner()
    packet = _candidate_training_packet()
    packet["candidate_training_execution_plan"]["required_input_status"][
        "source_recheck_artifact"
    ]["sha256"] = "0" * 64

    plan = runner.build_plan(packet)

    artifacts = {item["name"]: item for item in plan["artifact_status"]}
    source_recheck_status = artifacts["source_recheck_artifact"]
    assert source_recheck_status["exists"] is True
    assert source_recheck_status["sha_matches"] is False
    assert source_recheck_status["ok"] is False
    assert source_recheck_status["blockers"] == ["sha_mismatch"]


def test_candidate_training_runner_execute_blocks_stale_source_recheck_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")
    packet = _candidate_training_packet()
    packet["candidate_training_execution_plan"]["required_input_status"][
        "source_recheck_artifact"
    ]["sha256"] = "0" * 64
    calls: list[str] = []

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        packet,
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_result.with_suffix(".plan.json")),
        run_training=True,
        run_candidate_doctor=False,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "source-recheck artifact is not ready" in result["blockers"][0]
    assert "source_recheck_artifact:sha_mismatch" in result["blockers"][0]
    assert calls == []


def test_candidate_training_runner_execute_blocks_failed_packet_training_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")
    packet = _candidate_training_packet()
    packet["candidate_training_execution_plan"]["required_input_status"][
        "production_capture_matrix"
    ] = {
        "path": str(tmp_path / "capture_matrix.csv"),
        "exists": True,
        "gate_passed": False,
        "missing_labeled_examples": 12,
        "unapproved_rows": 3,
        "unsafe_storage_rows": 2,
    }
    packet["candidate_training_execution_plan"]["required_input_status"][
        "label_review_import_sidecar"
    ] = {
        "path": str(tmp_path / "capture.yaml.label_review_import.json"),
        "valid": False,
        "error": "label review import sidecar missing",
    }
    calls: list[str] = []

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        packet,
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_result.with_suffix(".plan.json")),
        run_training=True,
        run_candidate_doctor=False,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "candidate training packet gates are not ready" in result["blockers"][0]
    assert "production_capture_matrix_gate_not_passed" in result["blockers"][0]
    assert "label_review_import_sidecar_not_valid" in result["blockers"][0]
    assert calls == []


def test_candidate_runtime_runner_plan_preserves_sequence_and_success_criteria():
    runner = _load_candidate_runtime_runner()
    config_path = Path("qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml")

    plan = runner.build_plan(_candidate_runtime_packet(config_path=config_path))

    assert plan["one_detection_at_a_time"] is True
    assert plan["scenario_order"] == ["factory_missing_apron_active_closed_set"]
    assert "matching alert" in plan["success_criteria"]["active"]
    assert "zero model invocations" in plan["success_criteria"]["suppression"]
    step_artifacts = {item["name"]: item for item in plan["steps"][0]["artifact_status"]}
    assert step_artifacts["config"]["exists"] is True
    assert len(step_artifacts["config"]["sha256"]) == 64
    assert step_artifacts["expected_result"]["output"] is True


def _candidate_runtime_result(
    capability: str,
    role: str,
    *,
    config_path: Path | str = "candidate.yaml",
    scenario_id: str | None = None,
) -> dict:
    config_artifact = Path(config_path)
    if config_artifact.is_absolute() and not config_artifact.exists():
        config_artifact.parent.mkdir(parents=True, exist_ok=True)
        config_artifact.write_text("candidate: true\n", encoding="utf-8")
    visible_class = "apron" if capability == "apron_required" else "safety_harness"
    is_suppression = role == "suppression"
    is_guard = role == "false_positive_guard"
    max_detections = 0 if is_suppression else (4 if is_guard else 6)
    matching_alerts = [] if is_suppression or is_guard else [{"id": "candidate-alert"}]
    class_counts = {} if is_suppression else {"person": max_detections}
    if is_guard:
        class_counts[visible_class] = 3
    video_path = "test-videos/construction-workers-helmets.mp4"
    camera_id = "eval_factory_missing_apron_closed_set"
    source = {}
    if scenario_id:
        manifest_doc = yaml.safe_load((ROOT / "qa" / "video_eval" / "manifest.yaml").read_text(encoding="utf-8"))
        for scenario in manifest_doc.get("scenarios") or []:
            if isinstance(scenario, dict) and scenario.get("id") == scenario_id:
                video_path = scenario.get("local_video") or video_path
                camera_id = scenario.get("camera_id") or camera_id
                source = dict(scenario.get("source") or {})
                break
    return {
        "scenario_id": scenario_id or f"{capability}_{role}",
        "status": "ready_to_sell",
        "manifest_path": "qa/video_eval/manifest.yaml",
        "manifest_sha256": _sha256_file(ROOT / "qa" / "video_eval" / "manifest.yaml"),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_artifact) if config_artifact.exists() else None,
        "camera_id": camera_id,
        "video": video_path,
        "video_sha256": _sha256_file(ROOT / video_path),
        "source": source,
        "yaml_commands": [
            {"args": ["safetylens_site.py", "--config", str(config_path), "validate"], "returncode": 0},
            {"args": ["safetylens_site.py", "--config", str(config_path), "plan"], "returncode": 0},
            {"args": ["safetylens_site.py", "--config", str(config_path), "apply", "--yes"], "returncode": 0},
        ],
        "evidence": {
            "model_preflight": {
                "checked": True,
                "ok": True,
                "required_model_keys": ["ppe_closed_set_candidate"],
                "missing_model_keys": [],
            },
            "stream_probe": {"ok": True},
            "ui_evidence": {
                "screenshot_exists": True,
                "screenshot_fresh": True,
                "screenshot_path": "/tmp/candidate.png",
            },
            "delivery_summary": (
                {}
                if is_suppression or is_guard
                else {
                    "in_app": {"delivered": 1},
                    "browser_sound": {"simulated": 1},
                }
            ),
            "max_detections_count": max_detections,
            "matching_alerts": matching_alerts,
            "unexpected_alerts": [],
            "analytics_summary": {
                "class_counts": class_counts,
                "schedule": {
                    "suppressed_capabilities": [capability] if is_suppression else [],
                    "model_invocations": {
                        "ppe_closed_set_candidate": 0 if is_suppression else 1,
                    },
                },
            },
        },
    }


def _install_valid_candidate_runtime_results(doctor, tmp_path: Path) -> None:
    result_dir = tmp_path / "closed_set_candidate"
    result_dir.mkdir()
    templates = []
    for template in doctor.CLOSED_SET_CANDIDATE_YAML_TEMPLATES:
        scenario_id = template["scenario_id"]
        capability = template["capability"]
        role = template["role"]
        result_path = result_dir / f"{scenario_id}.json"
        updated = dict(template)
        updated["expected_result_path"] = result_path
        config_path = template["config_path"]
        _write_json(
            result_path,
            _candidate_runtime_result(
                capability,
                role,
                config_path=config_path,
                scenario_id=scenario_id,
            ),
        )
        templates.append(updated)
    doctor.CLOSED_SET_CANDIDATE_YAML_TEMPLATES = templates


def _promotion_report(
    capability: str,
    source_manifest_sha256: str = "b" * 64,
    export_sha256: str = "d" * 64,
    candidate_report_sha256: str = "c" * 64,
) -> dict:
    return {
        "ok": True,
        "capability": capability,
        "promotion_status": "ready_for_runtime_registration",
        "candidate_model_key": "ppe_closed_set_candidate",
        "baseline_model_key": "ppe_specialist",
        "candidate_report_sha256": candidate_report_sha256,
        "candidate_capture_matrix_manifest": {
            "path": "apron_harness_production_capture_matrix.csv.manifest.json",
            "checked": True,
            "valid": True,
            "mode": "production",
            "row_count": 21,
            "matrix_csv_sha256": "a" * 64,
            "source_manifest_sha256": source_manifest_sha256,
        },
        "candidate_label_review_import_manifest": {
            "path": "apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
            "checked": True,
            "valid": True,
            "source_manifest_sha256": "c" * 64,
            "label_review_csv_sha256": "e" * 64,
            "updated_manifest_sha256": source_manifest_sha256,
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
                "manifest_sha256": source_manifest_sha256,
                "errors": [],
                "warnings": [],
            },
            "training_gate": {
                "requires_reviewed_clip_metadata": True,
            },
        },
        "candidate_training_dataset_provenance": {
            "required": True,
            "checked": True,
            "source_manifest": "/cleared/apron_harness_capture_manifest.yaml",
            "declared_source_manifest_sha256": source_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "permission": "controlled_capture_cleared",
            "permission_allowed": True,
            "missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
            "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
            "errors": [],
        },
        "candidate_training_source_lineage": {
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
                    "sha256": source_manifest_sha256,
                },
                "manifest_sha256": source_manifest_sha256,
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
        "candidate_selected_export": {
            "path": "/cleared/runs/apron-harness-ppe.onnx",
            "exists": True,
            "accepted_suffix": True,
            "suffix": ".onnx",
            "sha256": export_sha256,
        },
        "candidate_registry_entry": {
            "model_key": "ppe_closed_set_candidate",
            "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "source_export_sha256": export_sha256,
        },
        "baseline": _promotion_runtime_group(capability, "ppe_specialist"),
        "candidate": _promotion_runtime_group(capability, "ppe_closed_set_candidate"),
    }


def _candidate_report_with_selected_export(export_sha256: str = "d" * 64) -> dict:
    return {
        "ok": True,
        "promotion_manifest": {
            "candidate_status": "ready_for_side_by_side_runtime_test",
            "seed_export_import_manifest": _approved_seed_export_import_manifest(),
            "runtime_handoff": {
                "selected_export": {
                    "path": "/cleared/runs/apron-harness-ppe.onnx",
                    "sha256": export_sha256,
                    "suffix": ".onnx",
                    "accepted_suffix": True,
                }
            }
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
        "gate": {
            "required": True,
            "ok": True,
            "gate_passed": True,
            "clip_count": 1,
            "approved_clip_count": 1,
        },
    }


def _approved_seed_export_import_manifest(source_manifest_sha256: str = "b" * 64) -> dict:
    return {
        "path": "/cleared/apron_harness_capture_manifest.yaml.seed_export_import.json",
        "exists": True,
        "checked": True,
        "valid": True,
        "partial_materialization": False,
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
        "updated_manifest_sha256": source_manifest_sha256,
        "imported_label_count": 3,
        "imported_clip_count": 3,
        "copied_image_count": 3,
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
            "manifest_sha256": source_manifest_sha256,
            "errors": [],
            "warnings": [],
        },
        "imports": [
            {
                "source_ref": "roboflow_seed",
                "capability": "apron_required",
                "raw_export_ref": "s3://cleared-seed-exports/apron_seed_yolo_export.zip",
                "raw_export_sha256": "4" * 64,
                "raw_export_local_path": "/cleared/seed_yolo_export.zip",
                "imported_label_count": 3,
                "copied_image_count": 3,
                "errors": [],
                "warnings": [],
                "yolo_export_preflight": {
                    "checked": True,
                    "sha256": "4" * 64,
                    "source_classes": ["person", "apron"],
                    "mapped_local_classes": ["person", "apron"],
                    "image_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "label_file_count_by_split": {"train": 1, "valid": 1, "test": 1},
                    "orphan_label_count_by_split": {},
                    "label_file_count_by_local_class": {
                        "person": 3,
                        "apron": 3,
                    },
                    "errors": [],
                    "warnings": [],
                },
            }
        ],
    }


def _jetson_gate_report(
    artifact_sha256: str = "d" * 64,
    candidate_report_sha256: str = "c" * 64,
) -> dict:
    return {
        "ok": True,
        "pack_id": "factory_ppe_3cam",
        "model": "apron-harness-ppe.onnx",
        "model_artifact_sha256": artifact_sha256,
        "candidate_report_sha256": candidate_report_sha256,
        "gate_status": "jetson_gate_passed",
        "production_gate": True,
        "raw_benchmark": {
            "present": True,
            "model_artifact_sha256": artifact_sha256,
            "candidate_report_sha256": candidate_report_sha256,
        },
        "soak_report": {
            "present": True,
            "camera_count": 3,
            "model_artifact_sha256": artifact_sha256,
            "candidate_report_sha256": candidate_report_sha256,
        },
    }


def _model_registry_report(
    artifact_sha256: str = "d" * 64,
    candidate_report_sha256: str = "c" * 64,
) -> dict:
    return {
        "ok": True,
        "registry_status": "registered",
        "candidate_report_sha256": candidate_report_sha256,
        "copy_requested": True,
        "copied": True,
        "model_manager_definition": {
            "checked": True,
            "model_key": "ppe_closed_set_candidate",
            "expected_registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "registered": True,
            "valid": True,
            "artifact_exists": True,
            "errors": [],
            "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
        },
        "source_export": {
            "path": "/cleared/runs/apron-harness-ppe.onnx",
            "exists": True,
            "sha256": artifact_sha256,
            "expected_sha256": artifact_sha256,
            "suffix": ".onnx",
        },
        "destination": {
            "path": "/repo/models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "exists": True,
            "sha256": artifact_sha256,
            "matches_expected_sha256": True,
            "registry_metadata": {
                "path": "/repo/models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
                "exists": True,
                "sha256": "e" * 64,
                "valid": True,
                "errors": [],
            },
        },
        "registry_entry": {
            "model_key": "ppe_closed_set_candidate",
            "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
            "source_export_sha256": artifact_sha256,
        },
        "errors": [],
        "warnings": [],
    }


def _mps_training_plan(source_manifest_sha256: str = "b" * 64) -> dict:
    return {
        "status": "ready_to_train",
        "model": "yolo26n.pt",
        "selected_device": "mps",
        "export_formats": ["onnx"],
        "train_args": {},
        "capture_preflight": {"required": True, "checked": True, "gate_passed": True},
        "dataset_provenance": {
            "required": True,
            "checked": True,
            "source_manifest": "/cleared/apron_harness_capture_manifest.yaml",
            "declared_source_manifest_sha256": source_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
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
                    "sha256": source_manifest_sha256,
                },
                "manifest_sha256": source_manifest_sha256,
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
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _model_pack_evidence_report(model_packs_path: Path, result_dir: Path, pack_status: str) -> dict:
    return {
        "ok": True,
        "skipped_model_packs": ["pose_fall_3cam"],
        "inputs": {
            "model_packs": _rel(model_packs_path),
            "manifest": "qa/video_eval/manifest.yaml",
            "result_dir": _rel(result_dir),
        },
        "packs": {
            "factory_ppe_3cam": {
                "ok": True,
                "status": pack_status,
                "errors": [],
                "warnings": [],
            }
        },
    }


def test_model_registry_gate_rejects_missing_registry_metadata(tmp_path: Path):
    doctor = _load_doctor()
    report_path = tmp_path / "model_registry.json"
    payload = _model_registry_report()
    payload["destination"].pop("registry_metadata")
    _write_json(report_path, payload)

    ok, status = doctor._model_registry_gate_ok(
        report_path,
        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
    )

    assert ok is False
    assert status.startswith("missing_registry_metadata:")


def test_model_registry_handoff_summary_separates_planned_audit_from_registration(tmp_path: Path):
    doctor = _load_doctor()
    report_path = tmp_path / "model_registry.json"
    payload = _model_registry_report()
    payload["registry_status"] = "planned_no_candidate"
    payload["candidate_report_sha256"] = None
    payload["candidate_report"] = None
    payload["copy_requested"] = False
    payload["copied"] = False
    payload["destination"]["exists"] = False
    payload["destination"]["matches_expected_sha256"] = False
    payload["destination"]["registry_metadata"]["exists"] = False
    payload["destination"]["registry_metadata"]["valid"] = False
    payload["next_required_gates"] = ["run_apron_harness_candidate_doctor"]
    _write_json(report_path, payload)

    summary = doctor._model_registry_handoff_summary(
        report_path,
        "not_registered:planned_no_candidate",
    )

    assert summary["checked"] is True
    assert summary["exists"] is True
    assert summary["status"] == "not_registered:planned_no_candidate"
    assert summary["registry_status"] == "planned_no_candidate"
    assert summary["candidate_report_present"] is False
    assert summary["model_definition_valid"] is True
    assert summary["destination_exists"] is False
    assert summary["metadata_valid"] is False
    assert summary["copy_requested"] is False
    assert summary["copied"] is False
    assert summary["next_required_gates"] == ["run_apron_harness_candidate_doctor"]
    artifacts = {item["name"]: item for item in summary["artifact_status"]}
    assert artifacts["model_registry_report"]["exists"] is True
    assert len(artifacts["model_registry_report"]["sha256"]) == 64
    assert artifacts["destination_model"]["exists"] is False
    assert artifacts["destination_model"]["blockers"] == ["missing"]
    assert artifacts["registry_metadata"]["exists"] is False
    assert "planned registry audit is not model registration" in summary["evidence_boundary"]


def test_model_registry_blocker_names_planned_registry_separately():
    doctor = _load_doctor()

    assert doctor._model_registry_blocker("not_registered:planned_no_candidate") == (
        "apron_harness_model_registry_not_registered"
    )
    assert doctor._model_registry_blocker("candidate_report_sha_mismatch:report.json") == (
        "missing_or_failed_apron_harness_model_registry_report"
    )


def test_jetson_gate_report_summary_surfaces_missing_inputs(tmp_path: Path):
    doctor = _load_doctor()
    report_path = tmp_path / "factory_ppe_jetson_gate.json"
    _write_json(
        report_path,
        {
            "ok": False,
            "gate_status": "not_ready",
            "production_gate": False,
            "errors": [
                "full Jetson gate requires --raw-benchmark",
                "full Jetson gate requires --soak-report",
                "factory_ppe_3cam full Jetson gate requires --candidate-report",
            ],
            "next_required_gates": [
                "run_raw_cuda_model_benchmark",
                "run_three_camera_soak_report",
            ],
            "candidate_report": {"present": False},
            "raw_benchmark": {"present": False},
            "soak_report": {"present": False},
        },
    )

    summary = doctor._jetson_gate_report_summary(report_path, f"failed:{report_path}")

    assert summary["exists"] is True
    assert summary["ok"] is False
    assert summary["gate_status"] == "not_ready"
    assert summary["candidate_report"]["present"] is False
    assert summary["raw_benchmark"]["present"] is False
    assert summary["soak_report"]["present"] is False
    assert "full Jetson gate requires --raw-benchmark" in summary["errors"]
    assert summary["next_required_gates"] == [
        "run_raw_cuda_model_benchmark",
        "run_three_camera_soak_report",
    ]


def test_apron_harness_readiness_accepts_current_pilot_evidence_and_blocks_production():
    doctor = _load_doctor()

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is True
    assert report["pilot_gate_passed"] is True
    assert report["production_gate_passed"] is False
    assert report["sales_status"] == "pilot_ready_not_production_compliance"
    assert report["capabilities"]["apron_required"]["ok"] is True
    assert report["capabilities"]["harness_required"]["ok"] is True
    assert "missing_or_failed_factory_ppe_jetson_full_gate" in report["production_blockers"]
    assert report["jetson_full_gate"]["status"]
    assert "closed_set_public_seed_sources_not_curated_or_approved" in report["production_blockers"]
    assert "apron_harness_seed_source_review_not_training_usable" in report["production_blockers"]
    assert "apron_harness_seed_import_manifest_not_training_usable" in report["production_blockers"]
    model_registry_handoff = report["model_registry_handoff"]
    assert model_registry_handoff["checked"] is True
    assert model_registry_handoff["status"] == "not_registered:planned_no_candidate"
    assert model_registry_handoff["registry_status"] == "planned_no_candidate"
    assert model_registry_handoff["model_definition_valid"] is True
    assert model_registry_handoff["destination_exists"] is False
    assert model_registry_handoff["metadata_valid"] is False
    assert model_registry_handoff["candidate_report_present"] is False
    registry_artifacts = {
        item["name"]: item
        for item in model_registry_handoff["artifact_status"]
    }
    assert registry_artifacts["model_registry_report"]["exists"] is True
    assert registry_artifacts["destination_model"]["exists"] is False
    assert registry_artifacts["registry_metadata"]["exists"] is False
    assert "planned registry audit is not model registration" in model_registry_handoff["evidence_boundary"]
    blocker_actions = {
        item["blocker"]: item
        for item in report["production_blocker_actions"]
    }
    assert set(blocker_actions) == set(report["production_blockers"])
    assert "qa/video_eval/results/apron_harness_next_source_review_batch.json" in blocker_actions[
        "closed_set_public_seed_sources_not_curated_or_approved"
    ]["artifacts"]
    assert "qa/video_eval/results/apron_harness_source_review_kickoff.md" in blocker_actions[
        "closed_set_public_seed_sources_not_curated_or_approved"
    ]["artifacts"]
    assert "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in blocker_actions[
        "closed_set_public_seed_sources_not_curated_or_approved"
    ]["command"]
    assert "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in blocker_actions[
        "closed_set_public_seed_sources_not_curated_or_approved"
    ]["artifacts"]
    assert "scripts/apron_harness_model_registry_doctor.py" in blocker_actions[
        "closed_set_model_artifact_missing"
    ]["command"]
    artifact_action = blocker_actions["closed_set_model_artifact_missing"]
    assert "qa/video_eval/results/apron_harness_model_registry_report.json" in artifact_action[
        "artifacts"
    ]
    assert "models/ppe_closed_set_candidate/apron-harness-ppe.onnx" in artifact_action[
        "artifacts"
    ]
    assert "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json" in artifact_action[
        "artifacts"
    ]
    assert "qa/video_eval/results/apron_closed_set_promotion_report.json" in artifact_action[
        "artifacts"
    ]
    assert "qa/video_eval/results/harness_closed_set_promotion_report.json" in artifact_action[
        "artifacts"
    ]
    assert "train/export only from reviewed production data" in artifact_action[
        "evidence_contract_summary"
    ][0]
    assert "runtime_handoff.selected_export.sha256 is present" in artifact_action[
        "evidence_contract"
    ]["candidate_report"]
    jetson_action = blocker_actions["missing_or_failed_factory_ppe_jetson_full_gate"]
    assert "--require-full-gate" in jetson_action["command"]
    assert "--write-raw-template" in jetson_action["template_commands"][0]
    assert "--write-soak-template" in jetson_action["template_commands"][1]
    assert "qa/video_eval/results/factory_ppe_raw_benchmark.template.json" in jetson_action[
        "template_artifacts"
    ]
    assert "scripts/benchmark_yolo_jetson.py" in jetson_action["raw_benchmark_capture_command"]
    assert "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json" in jetson_action[
        "raw_benchmark_capture_command"
    ]
    assert "same candidate_report_sha256" in jetson_action["evidence_contract_summary"][0]
    assert "per_class_alert_count.apron_required > 0" in jetson_action["evidence_contract"][
        "positive_alerts"
    ]
    assert (
        "detector_window_suppression.harness_required.model_invocations all == 0"
        in jetson_action["evidence_contract"]["detector_window_suppression"]
    )
    jetson_template_handoff = report["jetson_template_handoff"]
    assert jetson_template_handoff["status"] == "ready_for_candidate_identity"
    assert jetson_template_handoff["valid_template_contract_count"] == 2
    assert jetson_template_handoff["template_count"] == 2
    assert jetson_template_handoff["identity_stamped_count"] == 0
    assert jetson_template_handoff["templates"]["raw_benchmark_template"]["valid_template_contract"] is True
    assert jetson_template_handoff["templates"]["three_camera_soak_template"]["valid_template_contract"] is True
    seed_action = blocker_actions["closed_set_public_seed_sources_not_curated_or_approved"]
    assert seed_action["evidence_contract"]["agent_may_prefill_hints_only"] is True
    assert seed_action["evidence_contract"]["human_approval_required"] is True
    seed_import_action = blocker_actions["apron_harness_seed_import_manifest_not_training_usable"]
    assert "generated review-artifact hashes" in seed_import_action["next_action"]
    assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in seed_import_action[
        "command"
    ]
    assert "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in seed_import_action[
        "artifacts"
    ]
    assert "generated review packet/template paths" in seed_import_action[
        "evidence_contract_summary"
    ][1]
    assert "review_packet_path and review_packet_sha256" in seed_import_action[
        "evidence_contract"
    ]["review_artifacts"][0]
    assert "raw_export_ref is a remote immutable export reference" in seed_import_action[
        "evidence_contract"
    ]["export_archive"][0]
    assert report["sourcing_status"] == "public_seed_sources_found_unapproved_insufficient_for_production"
    assert report["sourcing_candidate_count"] >= 5
    assert report["optional_gate_status"]["seed_source_review"].startswith("blocked:")
    assert report["optional_gate_status"]["seed_source_review_bundle"].startswith(("pass:", "failed:"))
    assert report["seed_source_review_bundle_artifact_count"] >= 1
    assert report["optional_gate_status"]["seed_import_manifest"].startswith("blocked:")
    assert report["seed_source_review_gate_passed"] is False
    assert report["seed_source_review_training_usable_count"] == 0
    assert report["seed_source_review_queue_summary"]["candidate_count"] >= 5
    next_actions = report["next_actions"]
    assert next_actions[0]["id"] == "approve_or_capture_training_data"
    assert next_actions[0]["priority"] == 1
    assert "closed_set_public_seed_sources_not_curated_or_approved" in next_actions[0]["blockers"]
    assert "apron_harness_seed_source_review_not_training_usable" in next_actions[0]["blockers"]
    assert "human/legal approval evidence is required" in next_actions[0][
        "evidence_contract_summary"
    ][0]
    assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in next_actions[
        0
    ]["command"]
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in next_actions[
        0
    ]["command"]
    assert "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in next_actions[0][
        "artifacts"
    ]
    assert "qa/video_eval/results/apron_harness_source_review_bundle.json" in next_actions[0]["artifacts"]
    assert next_actions[1]["id"] == "produce_reviewed_training_manifest"
    assert "closed_set_label_review_import_sidecar_invalid" in next_actions[1]["blockers"]
    assert "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json" in next_actions[1]["command"]
    assert "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in next_actions[1]["command"]
    assert "--schema-only --seed-source-review-report" in next_actions[1]["command"]
    assert "--validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in next_actions[1]["command"]
    assert "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in next_actions[1]["command"]
    assert "--emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml" in next_actions[1]["command"]
    assert "schema-only/no-write mode" in next_actions[1]["evidence_contract_summary"][0]
    assert "label-review import sidecar must be valid" in next_actions[1]["evidence_contract_summary"][1]
    assert "production mode, non-schema-only" in next_actions[1]["evidence_contract_summary"][2]
    assert "recomputed class counts" in next_actions[1]["evidence_contract_summary"][3]
    assert next_actions[2]["id"] == "train_and_register_candidate"
    assert "scripts/apron_harness_train.py" in next_actions[2]["command"]
    assert "--capture-preflight-mode production" in next_actions[2]["command"]
    assert "--require-capture-preflight" in next_actions[2]["command"]
    assert "--device mps" in next_actions[2]["command"]
    assert "--device auto" not in next_actions[2]["command"]
    assert "scripts/apron_harness_candidate_doctor.py" in next_actions[2]["command"]
    assert "scripts/apron_harness_model_registry_doctor.py" in next_actions[2]["command"]
    assert "qa/video_eval/results/apron_harness_model_registry_report.json" in next_actions[2][
        "artifacts"
    ]
    assert "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json" in next_actions[2][
        "artifacts"
    ]
    assert "qa/video_eval/results/apron_closed_set_promotion_report.json" in next_actions[2][
        "artifacts"
    ]
    assert "qa/video_eval/results/harness_closed_set_promotion_report.json" in next_actions[2][
        "artifacts"
    ]
    assert "closed_set_model_artifact_missing" in next_actions[2]["blockers"]
    assert next_actions[3]["id"] == "prove_edge_gate"
    assert "--require-full-gate" in next_actions[3]["command"]
    coverage_summary = report["seed_source_coverage_summary"]
    assert coverage_summary["available"] is True
    assert coverage_summary["coverage_gap_count"] == 0
    assert coverage_summary["training_usable_count"] == 0
    assert coverage_summary["capabilities"]["apron_required"]["priority_coverage_status"] == (
        "candidate_coverage_complete_pending_review"
    )
    assert coverage_summary["capabilities"]["apron_required"]["person_box_reconciliation_status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    assert coverage_summary["capabilities"]["apron_required"]["missing_local_classes_across_reviewable_sources"] == []
    assert coverage_summary["capabilities"]["harness_required"]["priority_coverage_status"] == (
        "candidate_coverage_complete_pending_review"
    )
    assert coverage_summary["capabilities"]["harness_required"]["person_box_reconciliation_status"] == (
        "candidate_person_mapping_present_pending_review"
    )
    assert coverage_summary["capabilities"]["harness_required"]["missing_local_classes_across_reviewable_sources"] == []
    minimum_path = report["seed_source_minimum_approval_path"]
    assert minimum_path["checked"] is True
    assert minimum_path["coverage_gap_count"] == 0
    assert minimum_path["training_usable_count"] == 0
    assert minimum_path["approval_ready"] is False
    assert "not approval" in minimum_path["evidence_boundary"]
    assert "selected_sources_not_training_usable" in minimum_path["blockers"]
    assert "roboflow_workspace_otd88_fjwepfj1" in minimum_path["minimum_review_source_refs"]
    assert "roboflow_harness_s4xxh" in minimum_path["minimum_review_source_refs"]
    assert minimum_path["capabilities"]["apron_required"]["selected_source_count"] == 1
    assert minimum_path["capabilities"]["harness_required"]["selected_source_count"] == 2
    assert minimum_path["capabilities"]["apron_required"]["selected_sources"][0]["review_packet_path"].endswith(
        "roboflow_workspace_otd88_fjwepfj1__apron_required.review_packet.md"
    )
    assert any(
        source["source_ref"] == "roboflow_harness_s4xxh"
        and source["missing_local_classes"] == []
        for source in minimum_path["capabilities"]["harness_required"]["selected_sources"]
    )
    minimum_import_template = report["minimum_seed_import_manifest_template_summary"]
    assert minimum_import_template["available"] is True
    assert minimum_import_template["template_scope"] == "minimum_priority_coverage_sources"
    assert minimum_import_template["import_count"] == 3
    assert minimum_import_template["enabled_import_count"] == 0
    assert minimum_import_template["selected_source_refs"] == [
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    minimum_template_consistency = report["minimum_seed_import_manifest_template_consistency"]
    assert minimum_template_consistency["checked"] is True
    assert minimum_template_consistency["valid"] is True
    assert minimum_template_consistency["source_refs_match"] is True
    assert minimum_template_consistency["enabled_import_count"] == 0
    assert minimum_template_consistency["expected_source_refs"] == minimum_path["minimum_review_source_refs"]
    assert minimum_template_consistency["template_source_refs"] == minimum_import_template["selected_source_refs"]
    assert "does not approve any source" in minimum_template_consistency["evidence_boundary"]
    next_reviews = report["seed_source_next_review_queue"]
    assert len(next_reviews) == 5
    assert next_reviews[0]["source_ref"] == "roboflow_work_at_height_safety"
    assert next_reviews[0]["source_url"] == (
        "https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety"
    )
    assert next_reviews[0]["url"] == next_reviews[0]["source_url"]
    assert next_reviews[0]["license_note"] == "Roboflow Universe page lists CC BY 4.0."
    assert next_reviews[0]["checked"] == "2026-06-24"
    assert next_reviews[0]["capability"] == "harness_required"
    assert next_reviews[0]["review_priority"] == 10
    assert next_reviews[0]["training_usable"] is False
    assert next_reviews[0]["review_packet_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_packet.md"
    )
    assert next_reviews[0]["review_evidence_template_path"].endswith(
        "roboflow_work_at_height_safety__harness_required.review_evidence.yaml"
    )
    assert report["seed_import_manifest_gate_passed"] is False
    assert report["seed_import_manifest_included_count"] == 0
    assert report["seed_import_manifest_approved_count"] == 0
    fill_contract = report["seed_import_fill_contract_summary"]
    assert fill_contract["available"] is True
    assert "This template is not approval" in fill_contract["approval_boundary"]
    assert fill_contract["required_before_include_in_training_count"] >= 10
    assert "include_in_training=true" in fill_contract["forbidden_until_approved"]
    assert any("--validate-import-manifest" in command for command in fill_contract["validation_commands"])
    assert report["seed_import_export_preflight_summary"]["required_manifest_field"] == "raw_export_local_path"
    assert report["seed_import_export_preflight_summary"]["required_export_format"] == "yolo"
    assert "local_reviewed_export_zip_exists" in report["seed_import_export_preflight_summary"]["checks"]
    assert (
        "review_packet_path_and_sha256_match_seed_source_review"
        in report["seed_import_export_preflight_summary"]["checks"]
    )
    assert (
        "review_evidence_template_path_and_sha256_match_seed_source_review"
        in report["seed_import_export_preflight_summary"]["checks"]
    )
    assert report["seed_import_export_preflight_summary"]["included_count"] == 0
    assert report["seed_import_export_preflight_summary"]["preflight_approved_count"] == 0
    assert report["seed_import_export_preflight_summary"]["review_artifact_checked_count"] == 0
    assert report["seed_import_export_preflight_summary"]["review_artifact_error_count"] == 0
    assert report["seed_import_export_preflight_summary"]["blocked_reason"] == (
        "no_seed_imports_included_for_training"
    )
    assert report["seed_import_export_preflight_summary"]["blocker_count"] >= 1
    assert "no seed imports are approved for training" in report[
        "seed_import_export_preflight_summary"
    ]["top_blockers"]
    assert "closed_set_pilot_label_minimums_not_met" in report["production_blockers"]
    assert "closed_set_capture_matrix_not_complete_or_approved" in report["production_blockers"]
    assert "closed_set_production_capture_matrix_not_complete_or_approved" in report["production_blockers"]
    assert report["production_blocker_count"] == len(report["production_blockers"])
    assert report["production_blocker_count"] >= 1
    candidate_templates = report["closed_set_candidate_yaml_templates"]
    assert candidate_templates["checked"] is True
    assert candidate_templates["valid"] is True
    assert candidate_templates["template_count"] == 6
    assert candidate_templates["valid_template_count"] == 6
    assert all(row["required_model_plan_ok"] is True for row in candidate_templates["templates"])
    assert all(row["one_at_a_time_ok"] is True for row in candidate_templates["templates"])
    assert all(row["cli_preflight_ok"] is True for row in candidate_templates["templates"])
    assert all(row["cli_preflight"]["checks"]["validate"]["exit_code"] == 0 for row in candidate_templates["templates"])
    assert all(row["cli_preflight"]["checks"]["plan"]["exit_code"] == 0 for row in candidate_templates["templates"])
    assert all(
        len(row["cli_preflight"]["checks"]["validate"]["stdout_sha256"]) == 64
        for row in candidate_templates["templates"]
    )
    assert all(row["execution_policy"]["apply_replaces_existing_config"] is True for row in candidate_templates["templates"])
    assert all(row["execution_policy"]["camera_count"] == 1 for row in candidate_templates["templates"])
    first_candidate_commands = candidate_templates["templates"][0]["commands"]
    assert first_candidate_commands["backup"] == (
        ".venv/bin/python scripts/safetylens_site.py export --output "
        "qa/video_eval/results/site_config_backups/before_factory_missing_apron_active_closed_set.yaml"
    )
    assert first_candidate_commands["validate"].startswith(
        ".venv/bin/python scripts/safetylens_site.py --config "
    )
    assert first_candidate_commands["validate"].endswith(" validate")
    assert first_candidate_commands["plan"].endswith(" plan")
    assert first_candidate_commands["apply"].endswith(" apply --yes")
    assert first_candidate_commands["run"].endswith("factory_missing_apron_active_closed_set")
    assert first_candidate_commands["restore"] == (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/results/site_config_backups/before_factory_missing_apron_active_closed_set.yaml apply --yes"
    )
    assert "closed_set_candidate_yaml_templates_invalid" not in report["production_blockers"]
    candidate_runtime = report["closed_set_candidate_runtime_evidence"]
    assert candidate_runtime["checked"] is True
    assert candidate_runtime["valid"] is False
    assert candidate_runtime["result_count"] == 6
    assert candidate_runtime["present_result_count"] == 6
    assert candidate_runtime["valid_result_count"] == 0
    assert candidate_runtime["missing_result_count"] == 0
    assert candidate_runtime["preflight_blocked_missing_model_count"] == 6
    assert "closed_set_candidate_runtime_evidence_missing_or_invalid" in report["production_blockers"]
    blocker_actions = {
        item["blocker"]: item
        for item in report["production_blocker_actions"]
    }
    candidate_runtime_action = blocker_actions["closed_set_candidate_runtime_evidence_missing_or_invalid"]
    assert "factory_missing_apron_active_closed_set" in candidate_runtime_action["command"]
    assert "factory_harness_detector_window_suppression_closed_set" in candidate_runtime_action["command"]
    assert "qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json" in candidate_runtime_action[
        "artifacts"
    ]
    assert "zero ppe_closed_set_candidate invocations" in candidate_runtime_action[
        "evidence_contract_summary"
    ][3]
    handoff = report["closed_set_handoff"]
    assert handoff["dataset_schema_ok"] is True
    assert len(handoff["capture_manifest_sha256"]) == 64
    assert handoff["training_readiness"]["status"] == "blocked"
    assert "production_training_plan_preflight_not_checked" in handoff["training_readiness"]["blockers"]
    assert handoff["training_dry_run_status"] == "ready_to_train"
    assert handoff["training_model"] == "yolo26n.pt"
    assert isinstance(handoff["training_torch_status"], dict)
    assert handoff["training_capture_preflight"]["checked"] is False
    assert "closed_set_training_capture_preflight_not_checked" in report["production_blockers"]
    assert handoff["selected_device"] in {"cpu", "mps", "cuda"}
    if handoff["selected_device"] != "mps":
        assert "local_closed_set_training_dry_run_not_on_mps" in report["production_blockers"]
        assert "scripts/model_pack_doctor.py" in blocker_actions[
            "local_closed_set_training_dry_run_not_on_mps"
        ]["command"]
        assert (
            "training dry-run selected_device must be mps"
            in blocker_actions["local_closed_set_training_dry_run_not_on_mps"]["evidence_contract_summary"]
        )
    assert set(handoff["missing_label_minimums"]) == {"person", "apron", "safety_harness", "safety_lanyard"}
    assert set(handoff["production_missing_label_minimums"]) == {"person", "apron", "safety_harness", "safety_lanyard"}
    assert handoff["required_labeled_images_per_class"] == {"pilot": 300, "production": 1000}
    deficit = handoff["capture_deficit"]
    assert deficit["total_missing_label_annotations"] == 1200
    assert deficit["recommended_label_review_rows"] == 720
    assert deficit["coverage_deficit_count"] == 0
    assert [batch["target_capability"] for batch in deficit["next_capture_batches"]] == [
        "apron_required",
        "harness_required",
    ]
    production_deficit = handoff["production_capture_deficit"]
    assert production_deficit["mode"] == "production"
    assert production_deficit["required_per_class"] == 1000
    assert production_deficit["total_missing_label_annotations"] == 4000
    assert production_deficit["recommended_label_review_rows"] == 2404
    assert [batch["target_capability"] for batch in production_deficit["next_capture_batches"]] == [
        "apron_required",
        "harness_required",
    ]
    assert handoff["capture_work_order"]["generated"] is False
    assert handoff["capture_work_order"]["path"] is None


def test_minimum_seed_import_template_consistency_blocker_has_specific_action():
    doctor = _load_doctor()

    action = doctor._production_blocker_action("minimum_seed_import_template_not_consistent")

    assert action["blocker"] == "minimum_seed_import_template_not_consistent"
    assert "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in action[
        "command"
    ]
    assert "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in action[
        "artifacts"
    ]
    assert "qa/video_eval/results/apron_harness_source_coverage_plan.json" in action["artifacts"]
    assert "selected_source_refs must match" in action["evidence_contract_summary"][0]
    assert "enabled_import_count must stay 0" in action["evidence_contract_summary"][2]
    assert "include_in_training=false" in action["evidence_contract"]["non_approving"][1]
    assert "template_scope=minimum_priority_coverage_sources" in action["evidence_contract"]["scope"][0]


def test_minimum_seed_import_template_consistency_blocker_is_prioritized_with_training_data_actions():
    doctor = _load_doctor()
    blockers = ["minimum_seed_import_template_not_consistent"]
    blocker_actions = doctor._production_blocker_actions(blockers)

    next_actions = doctor._prioritized_next_actions(blockers, blocker_actions)

    assert len(next_actions) == 1
    assert next_actions[0]["priority"] == 1
    assert next_actions[0]["id"] == "approve_or_capture_training_data"
    assert next_actions[0]["blockers"] == ["minimum_seed_import_template_not_consistent"]
    assert "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml" in next_actions[
        0
    ]["command"]
    assert "selected_source_refs must match" in next_actions[0]["evidence_contract_summary"][0]


@pytest.mark.parametrize(
    "blocker",
    [
        "missing_or_failed_apron_harness_seed_source_review",
        "missing_or_failed_apron_harness_source_review_bundle",
        "missing_or_failed_apron_harness_seed_import_manifest",
    ],
)
def test_missing_seed_review_and_import_artifacts_are_prioritized_with_training_data_actions(blocker: str):
    doctor = _load_doctor()
    blocker_actions = doctor._production_blocker_actions([blocker])

    next_actions = doctor._prioritized_next_actions([blocker], blocker_actions)

    assert len(next_actions) == 1
    assert next_actions[0]["priority"] == 1
    assert next_actions[0]["id"] == "approve_or_capture_training_data"
    assert next_actions[0]["blockers"] == [blocker]
    assert next_actions[0]["command"]
    assert next_actions[0]["artifacts"]
    if blocker in {
        "missing_or_failed_apron_harness_seed_source_review",
        "missing_or_failed_apron_harness_source_review_bundle",
        "missing_or_failed_apron_harness_seed_import_manifest",
    }:
        assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in next_actions[
            0
        ]["command"]
    if blocker in {
        "missing_or_failed_apron_harness_seed_source_review",
        "missing_or_failed_apron_harness_source_review_bundle",
        "missing_or_failed_apron_harness_seed_import_manifest",
    }:
        assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in next_actions[
            0
        ]["command"]
    if blocker == "missing_or_failed_apron_harness_seed_import_manifest":
        assert "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in next_actions[
            0
        ]["command"]


def test_candidate_yaml_template_status_rejects_missing_template(tmp_path: Path):
    doctor = _load_doctor()

    status = doctor._candidate_yaml_template_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": tmp_path / "missing_candidate.yaml",
                "expected_result_path": tmp_path / "missing_result.json",
            }
        ]
    )

    assert status["checked"] is True
    assert status["valid"] is False
    assert status["template_count"] == 1
    assert status["valid_template_count"] == 0
    assert status["templates"][0]["exists"] is False
    assert status["blockers"] == ["factory_missing_apron_active_closed_set:missing_candidate_yaml_template"]


def test_candidate_template_runtime_checks_require_compiled_model_preflight_key():
    doctor = _load_doctor()
    doc = {
        "cameras": {
            "candidate": {
                "capabilities": ["apron_required"],
                "capability_model_overrides": {"apron_required": "ppe_closed_set_candidate"},
                "capability_windows": {
                    "apron_required": {
                        "windows": [{"days": ["mon"], "from": "00:00", "to": "23:59"}],
                    },
                },
            }
        }
    }
    plan_result = {
        "config": {
            "desired_config": {
                "cameras": {
                    "candidate": {
                        "capabilities": ["apron_required"],
                        "execution_plan": {"required_model_keys": ["ppe_specialist"]},
                    }
                }
            }
        }
    }

    ok, errors = doctor._candidate_template_runtime_checks(
        doc,
        plan_result=plan_result,
        capability="apron_required",
        role="active",
    )

    assert ok is False
    assert "apron_required compiled execution plan must require ppe_closed_set_candidate" in errors


def test_candidate_template_isolation_checks_reject_merge_or_multiple_cameras():
    doctor = _load_doctor()
    doc = {
        "site": {"merge_existing": True},
        "cameras": {
            "candidate_a": {"name": "A", "zone": "Z", "demo": "yolo"},
            "candidate_b": {"name": "B", "zone": "Z", "demo": "yolo"},
        },
    }

    ok, status, errors = doctor._candidate_template_isolation_checks(doc)

    assert ok is False
    assert status["apply_replaces_existing_config"] is False
    assert status["camera_count"] == 2
    assert "candidate YAML site.merge_existing must be false for one-at-a-time execution" in errors
    assert "candidate YAML must declare exactly one camera for one-at-a-time execution" in errors


def test_candidate_runtime_evidence_status_reports_missing_result(tmp_path: Path):
    doctor = _load_doctor()

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": tmp_path / "candidate.yaml",
                "expected_result_path": tmp_path / "missing_result.json",
            }
        ]
    )

    assert status["checked"] is True
    assert status["valid"] is False
    assert status["result_count"] == 1
    assert status["present_result_count"] == 0
    assert status["valid_result_count"] == 0
    assert status["missing_result_count"] == 1
    assert status["results"][0]["exists"] is False
    assert status["blockers"] == ["factory_missing_apron_active_closed_set:missing_candidate_runtime_result"]


def test_candidate_runtime_evidence_status_reports_missing_model_preflight(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    _write_json(
        result_path,
        {
            **_candidate_runtime_result(
                "apron_required",
                "active",
                config_path=tmp_path / "candidate.yaml",
                scenario_id="factory_missing_apron_active_closed_set",
            ),
            "status": "blocked",
            "evidence": {
                "model_preflight": {
                    "checked": True,
                    "ok": False,
                    "reason": "required_models_not_ready",
                    "required_model_keys": ["ppe_closed_set_candidate"],
                    "missing_model_keys": ["ppe_closed_set_candidate"],
                }
            },
        },
    )

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": tmp_path / "candidate.yaml",
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["checked"] is True
    assert status["valid"] is False
    assert status["result_count"] == 1
    assert status["present_result_count"] == 1
    assert status["valid_result_count"] == 0
    assert status["missing_result_count"] == 0
    assert status["preflight_blocked_missing_model_count"] == 1
    assert status["results"][0]["exists"] is True
    assert status["results"][0]["preflight_blocked_missing_required_model"] is True
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:blocked_missing_required_model_preflight"
    ]


def test_candidate_runtime_evidence_rejects_missing_required_model_preflight(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["evidence"].pop("model_preflight")
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:missing_required_model_preflight"
    ]


def test_candidate_runtime_evidence_rejects_pilot_model_fallback_invocation(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_apron_false_positive_guard_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "false_positive_guard",
        config_path=config_path,
        scenario_id="factory_apron_false_positive_guard_closed_set",
    )
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"]["ppe_specialist"] = 1
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_apron_false_positive_guard_closed_set",
                "capability": "apron_required",
                "role": "false_positive_guard",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_apron_false_positive_guard_closed_set:"
        "pilot_model_invoked_in_candidate_runtime:ppe_specialist"
    ]


def test_candidate_runtime_evidence_rejects_wrong_config_path(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    expected_config = tmp_path / "expected_candidate.yaml"
    wrong_config = tmp_path / "wrong_candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=wrong_config,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": expected_config,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_config_path_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_apply_without_yes(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["yaml_commands"][2]["args"] = ["safetylens_site.py", "--config", str(config_path), "apply"]
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_yaml_apply_missing_yes"
    ]


def test_candidate_runtime_evidence_rejects_stale_config_sha(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    config_path.write_text("candidate: changed-after-run\n", encoding="utf-8")
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_config_sha256_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_stale_manifest_sha(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["manifest_sha256"] = "0" * 64
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_manifest_sha256_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_stale_video_sha(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["video_sha256"] = "0" * 64
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_video_sha256_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_wrong_manifest_video_path(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    wrong_video = "test-videos/warehouse-worker-aisle.mp4"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["video"] = wrong_video
    payload["video_sha256"] = _sha256_file(ROOT / wrong_video)
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_video_path_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_wrong_camera_id(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["camera_id"] = "other_camera"
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_camera_id_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_source_metadata_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["source"] = {**payload["source"], "provider": "Different Provider"}
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:candidate_runtime_source_metadata_mismatch"
    ]


def test_candidate_runtime_evidence_rejects_yaml_command_order_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"
    config_path = tmp_path / "candidate.yaml"
    payload = _candidate_runtime_result(
        "apron_required",
        "active",
        config_path=config_path,
        scenario_id="factory_missing_apron_active_closed_set",
    )
    payload["yaml_commands"][0], payload["yaml_commands"][1] = (
        payload["yaml_commands"][1],
        payload["yaml_commands"][0],
    )
    _write_json(result_path, payload)

    status = doctor._candidate_runtime_evidence_status(
        [
            {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "capability": "apron_required",
                "role": "active",
                "config_path": config_path,
                "expected_result_path": result_path,
            }
        ]
    )

    assert status["valid"] is False
    assert status["valid_result_count"] == 0
    assert status["blockers"] == [
        "factory_missing_apron_active_closed_set:"
        "candidate_runtime_yaml_validate_plan_apply_order_mismatch"
    ]


def test_apron_harness_readiness_reports_missing_seed_source_review(tmp_path: Path):
    doctor = _load_doctor()

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        seed_source_review_report=tmp_path / "missing_seed_source_review.json",
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["seed_source_review"].startswith("missing:")
    assert "missing_or_failed_apron_harness_seed_source_review" in report["production_blockers"]


def test_apron_harness_readiness_reports_missing_source_review_bundle(tmp_path: Path):
    doctor = _load_doctor()

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        seed_source_review_bundle=tmp_path / "missing_source_review_bundle.json",
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["seed_source_review_bundle"].startswith("missing:")
    assert report["seed_source_review_bundle_ok"] is False
    assert "missing_or_failed_apron_harness_source_review_bundle" in report["production_blockers"]


def test_apron_harness_readiness_rejects_shallow_optional_gate_reports(tmp_path: Path):
    doctor = _load_doctor()
    shallow = {"ok": True}
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, shallow)
    _write_json(harness_report, shallow)
    _write_json(jetson_report, shallow)

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith("not_ready:")
    assert report["optional_gate_status"]["harness_promotion"].startswith("not_ready:")
    assert report["optional_gate_status"]["jetson_gate"].startswith("wrong_pack:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]
    assert "missing_or_failed_harness_closed_set_promotion_report" in report["production_blockers"]
    assert "missing_or_failed_factory_ppe_jetson_full_gate" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_runtime_summary(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    shallow_candidate = _promotion_report("apron_required")
    shallow_candidate["candidate"]["false_positive_guard"]["visible_class_total"] = 0
    _write_json(apron_report, shallow_candidate)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "invalid_candidate_false_positive_guard_missing_visible_class:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload.pop("candidate_capture_matrix_manifest")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_capture_matrix_manifest:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_label_review_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload.pop("candidate_label_review_import_manifest")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_label_review_import_manifest:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_failed_label_review_validation(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_label_review_import_manifest"]["updated_manifest_validation"]["ok"] = False
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_label_review_import_manifest_validation_failed:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_label_review_clip_metadata(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_label_review_import_manifest"].pop("imported_clip_count")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_label_review_import_manifest_imported_clip_count_missing:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_dataset_provenance(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload.pop("candidate_training_dataset_provenance")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_training_dataset_provenance:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_source_lineage(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload.pop("candidate_training_source_lineage")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_training_source_lineage:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_source_lineage_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["capture_manifest"]["file"]["sha256"] = "c" * 64
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_training_source_lineage_capture_manifest_file_sha_mismatch:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_seed_import_without_export_preflight(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage("apron_required")
    )
    payload["candidate_training_source_lineage"]["seed_import_manifest"]["gate"]["imports"][0].pop(
        "yolo_export_preflight"
    )
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_training_source_lineage_seed_import_manifest_missing_reviewed_export_preflight:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_accepts_promotion_seed_import_with_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["candidate_training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage("apron_required")
    )
    payload["candidate_seed_export_import_manifest"] = _approved_seed_export_import_manifest()
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["optional_gate_status"]["apron_promotion"].startswith("ok:")


def test_apron_harness_readiness_rejects_promotion_seed_import_without_seed_export_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["candidate_training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage("apron_required")
    )
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_seed_export_import_manifest:"
    )
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_seed_export_sidecar_without_yolo_preflight(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["candidate_training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage("apron_required")
    )
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar["imports"][0].pop("yolo_export_preflight")
    payload["candidate_seed_export_import_manifest"] = seed_sidecar
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_seed_export_import_manifest_import_missing_yolo_preflight:"
    )
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_seed_export_sidecar_without_source_recheck(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_source_lineage"]["seed_source_review"] = _approved_seed_source_lineage()
    payload["candidate_training_source_lineage"]["seed_import_manifest"] = (
        _approved_seed_import_lineage("apron_required")
    )
    seed_sidecar = _approved_seed_export_import_manifest()
    seed_sidecar.pop("source_recheck")
    payload["candidate_seed_export_import_manifest"] = seed_sidecar
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_seed_export_import_manifest_missing_source_recheck:"
    )
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_dataset_manifest_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload["candidate_training_dataset_provenance"]["declared_source_manifest_sha256"] = "c" * 64
    payload["candidate_training_dataset_provenance"]["source_manifest_sha256"] = "c" * 64
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_training_dataset_provenance_source_manifest_sha_mismatch:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_rejects_promotion_without_selected_export(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    payload = _promotion_report("apron_required")
    payload.pop("candidate_selected_export")
    _write_json(apron_report, payload)
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "missing_candidate_selected_export:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]


def test_valid_optional_reports_do_not_override_data_and_mps_blockers(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required"))
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["optional_gate_status"]["apron_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["jetson_gate"].startswith("ok:")
    assert report["optional_gate_status"]["model_pack_evidence"].startswith("ok:")
    assert report["production_gate_passed"] is False
    assert report["sales_status"] == "pilot_ready_not_production_compliance"
    model_definition = report["closed_set_model_manager_definition"]
    assert model_definition["registered"] is True
    assert model_definition["valid"] is True
    assert model_definition["expected_registry_path"] == "models/ppe_closed_set_candidate/apron-harness-ppe.onnx"
    assert model_definition["registry_path"] == "models/ppe_closed_set_candidate/apron-harness-ppe.onnx"
    assert model_definition["artifact_exists"] is False
    assert "closed_set_model_artifact_missing" in report["production_blockers"]
    assert "closed_set_pilot_label_minimums_not_met" in report["production_blockers"]
    assert "closed_set_production_label_minimums_not_met" in report["production_blockers"]
    assert "closed_set_capture_matrix_not_complete_or_approved" in report["production_blockers"]
    assert "factory_ppe_pack_status_not_promoted" in report["production_blockers"]
    assert "closed_set_runtime_handoff_not_registered" in report["production_blockers"]


def test_apron_harness_readiness_blocks_mismatched_promotion_export_sha(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required", export_sha256="d" * 64))
    _write_json(harness_report, _promotion_report("harness_required", export_sha256="e" * 64))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"] == "candidate_selected_export_sha_mismatch_between_promotions"
    assert report["optional_gate_status"]["harness_promotion"] == "candidate_selected_export_sha_mismatch_between_promotions"
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]
    assert "missing_or_failed_harness_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_blocks_mismatched_promotion_candidate_report_sha(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    _write_json(
        apron_report,
        _promotion_report("apron_required", candidate_report_sha256="c" * 64),
    )
    _write_json(
        harness_report,
        _promotion_report("harness_required", candidate_report_sha256="f" * 64),
    )

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"] == "candidate_report_sha_mismatch_between_promotions"
    assert report["optional_gate_status"]["harness_promotion"] == "candidate_report_sha_mismatch_between_promotions"
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]
    assert "missing_or_failed_harness_closed_set_promotion_report" in report["production_blockers"]


def test_apron_harness_readiness_blocks_registry_candidate_report_sha_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    model_registry_report = tmp_path / "model_registry.json"
    _write_json(
        apron_report,
        _promotion_report("apron_required", candidate_report_sha256="c" * 64),
    )
    _write_json(
        harness_report,
        _promotion_report("harness_required", candidate_report_sha256="c" * 64),
    )
    _write_json(
        model_registry_report,
        _model_registry_report(candidate_report_sha256="f" * 64),
    )

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        model_registry_report=model_registry_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["model_registry"].startswith("candidate_report_sha_mismatch:")
    assert "missing_or_failed_apron_harness_model_registry_report" in report["production_blockers"]


def test_apron_harness_readiness_blocks_jetson_artifact_sha_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required", export_sha256="d" * 64))
    _write_json(harness_report, _promotion_report("harness_required", export_sha256="d" * 64))
    _write_json(jetson_report, _jetson_gate_report(artifact_sha256="e" * 64))

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["jetson_gate"].startswith("model_artifact_sha_mismatch:")
    assert "missing_or_failed_factory_ppe_jetson_full_gate" in report["production_blockers"]


def test_apron_harness_readiness_blocks_jetson_candidate_report_sha_mismatch(tmp_path: Path):
    doctor = _load_doctor()
    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required", candidate_report_sha256="c" * 64))
    _write_json(harness_report, _promotion_report("harness_required", candidate_report_sha256="c" * 64))
    _write_json(jetson_report, _jetson_gate_report(candidate_report_sha256="e" * 64))

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert report["optional_gate_status"]["jetson_gate"].startswith("candidate_report_sha_mismatch:")
    assert "missing_or_failed_factory_ppe_jetson_full_gate" in report["production_blockers"]


def test_apron_harness_readiness_blocks_stale_model_pack_evidence(tmp_path: Path):
    doctor = _load_doctor()
    stale_evidence = tmp_path / "model_pack_evidence_doctor.json"
    _write_json(
        stale_evidence,
        _model_pack_evidence_report(
            tmp_path / "other_model_packs.yaml",
            RESULT_DIR,
            "pilot_ready_not_production_compliance",
        ),
    )

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=stale_evidence,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["model_pack_evidence"].startswith("stale_model_packs:")
    assert "missing_or_failed_model_pack_evidence_doctor" in report["production_blockers"]


def test_apron_harness_readiness_blocks_active_out_of_scope_hospital_pack(tmp_path: Path):
    doctor = _load_doctor()
    stale_evidence = tmp_path / "model_pack_evidence_doctor.json"
    payload = _model_pack_evidence_report(
        MODEL_PACKS_PATH,
        RESULT_DIR,
        "pilot_ready_not_production_compliance",
    )
    payload["packs"]["pose_fall_3cam"] = {
        "ok": True,
        "status": "ready_to_sell_with_scope_limits",
        "errors": [],
        "warnings": [],
    }
    _write_json(stale_evidence, payload)

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=stale_evidence,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["model_pack_evidence"].startswith(
        "out_of_scope_model_pack_present:pose_fall_3cam"
    )
    assert "missing_or_failed_model_pack_evidence_doctor" in report["production_blockers"]


def test_apron_harness_readiness_requires_hospital_pack_skip_marker(tmp_path: Path):
    doctor = _load_doctor()
    stale_evidence = tmp_path / "model_pack_evidence_doctor.json"
    payload = _model_pack_evidence_report(
        MODEL_PACKS_PATH,
        RESULT_DIR,
        "pilot_ready_not_production_compliance",
    )
    payload["skipped_model_packs"] = []
    _write_json(stale_evidence, payload)

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=stale_evidence,
    )

    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["model_pack_evidence"].startswith(
        "missing_skipped_model_packs:pose_fall_3cam"
    )
    assert "missing_or_failed_model_pack_evidence_doctor" in report["production_blockers"]


def test_apron_harness_readiness_rejects_detector_window_regression(tmp_path: Path):
    doctor = _load_doctor()
    result_dir = tmp_path / "results"
    shutil.copytree(RESULT_DIR, result_dir)
    target = result_dir / "factory_harness_detector_window_suppression.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evidence"]["max_detections_count"] = 1
    payload["evidence"]["analytics_summary"]["schedule"]["model_invocations"]["ppe_specialist"] = 1
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=result_dir,
    )

    assert report["ok"] is False
    errors = "\n".join(report["errors"])
    assert "harness_required.suppression: must emit zero detections" in errors
    assert "harness_required.suppression: must report zero ppe_specialist invocations" in errors


def test_apron_harness_readiness_rejects_premature_production_pack_status(tmp_path: Path):
    doctor = _load_doctor()
    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    payload["packs"]["factory_ppe_3cam"]["status"] = "ready_to_sell_production_compliance"
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=model_packs_path,
        result_dir=RESULT_DIR,
    )

    assert report["ok"] is False
    assert any("status is production before closed-set data/runtime/Jetson gates pass" in error for error in report["errors"])


def test_apron_harness_readiness_blocks_promoted_gate_with_only_pilot_counts(tmp_path: Path):
    doctor = _load_doctor()

    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    factory_pack = payload["packs"]["factory_ppe_3cam"]
    factory_pack["status"] = "ready_to_sell_production_compliance"
    factory_pack["sourcing_status"]["apron_harness_result"] = "cleared_closed_set_dataset_approved"
    factory_pack["runtime_handoff"]["status"] = "registered_closed_set_candidate"
    factory_pack["model_keys"].append("ppe_closed_set_candidate")
    factory_pack["registry_models"]["ppe_closed_set_candidate"] = {
        "file": "apron-harness-ppe.onnx",
        "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
        "expected_input_size": 640,
        "local_device": "mps",
        "jetson_device": "tensorrt_fp16_candidate",
    }
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    evidence_report = tmp_path / "model_pack_evidence_doctor.json"
    _write_json(
        evidence_report,
        _model_pack_evidence_report(
            model_packs_path,
            RESULT_DIR,
            "ready_to_sell_production_compliance",
        ),
    )

    capture_manifest = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest = yaml.safe_load((ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml").read_text(encoding="utf-8"))
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 300,
        "apron": 300,
        "safety_harness": 300,
        "safety_lanyard": 300,
    }
    capture_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    doctor.build_training_plan = lambda **_: _mps_training_plan(doctor._sha256_file(capture_manifest))

    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    model_registry_report = tmp_path / "model_registry.json"
    jetson_report = tmp_path / "jetson_gate.json"
    capture_manifest_sha256 = doctor._sha256_file(capture_manifest)
    _write_json(apron_report, _promotion_report("apron_required", capture_manifest_sha256))
    _write_json(harness_report, _promotion_report("harness_required", capture_manifest_sha256))
    _write_json(model_registry_report, _model_registry_report())
    _write_json(jetson_report, _jetson_gate_report())
    _install_valid_candidate_runtime_results(doctor, tmp_path)

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=model_packs_path,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=evidence_report,
        capture_manifest=capture_manifest,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        model_registry_report=model_registry_report,
        jetson_gate_report=jetson_report,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is False
    assert report["pilot_gate_passed"] is True
    assert report["production_gate_passed"] is False
    assert report["sales_status"] == "pilot_ready_not_production_compliance"
    assert report["closed_set_handoff"]["missing_label_minimums"] == {}
    assert set(report["closed_set_handoff"]["production_missing_label_minimums"]) == {
        "person",
        "apron",
        "safety_harness",
        "safety_lanyard",
    }
    assert "closed_set_production_label_minimums_not_met" in report["production_blockers"]
    assert any("status is production before closed-set data/runtime/Jetson gates pass" in error for error in report["errors"])
    assert report["closed_set_handoff"]["capture_matrix_progress"]["gate_passed"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_progress"]["gate_passed"] is False


def test_apron_harness_readiness_accepts_fully_promoted_closed_set_gate(tmp_path: Path):
    doctor = _load_doctor()
    doctor._model_manager_definition_status = lambda planned_registry_path: {
        "checked": True,
        "model_key": "ppe_closed_set_candidate",
        "expected_registry_path": planned_registry_path,
        "registered": True,
        "valid": True,
        "artifact_exists": True,
        "errors": [],
        "registry_path": planned_registry_path,
    }

    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    factory_pack = payload["packs"]["factory_ppe_3cam"]
    factory_pack["status"] = "ready_to_sell_production_compliance"
    factory_pack["sourcing_status"]["apron_harness_result"] = "cleared_closed_set_dataset_approved"
    factory_pack["runtime_handoff"]["status"] = "registered_closed_set_candidate"
    factory_pack["model_keys"].append("ppe_closed_set_candidate")
    factory_pack["registry_models"]["ppe_closed_set_candidate"] = {
        "file": "apron-harness-ppe.onnx",
        "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
        "expected_input_size": 640,
        "local_device": "mps",
        "jetson_device": "tensorrt_fp16_candidate",
    }
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    evidence_report = tmp_path / "model_pack_evidence_doctor.json"
    _write_json(
        evidence_report,
        _model_pack_evidence_report(
            model_packs_path,
            RESULT_DIR,
            "ready_to_sell_production_compliance",
        ),
    )

    capture_manifest = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest = yaml.safe_load((ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml").read_text(encoding="utf-8"))
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1000,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    capture_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _attach_label_review_import_sidecar(capture_manifest)
    doctor.build_training_plan = lambda **_: _mps_training_plan(doctor._sha256_file(capture_manifest))

    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    model_registry_report = tmp_path / "model_registry.json"
    jetson_report = tmp_path / "jetson_gate.json"
    capture_manifest_sha256 = doctor._sha256_file(capture_manifest)
    _write_json(apron_report, _promotion_report("apron_required", capture_manifest_sha256))
    _write_json(harness_report, _promotion_report("harness_required", capture_manifest_sha256))
    _write_json(model_registry_report, _model_registry_report())
    _write_json(jetson_report, _jetson_gate_report())
    _install_valid_candidate_runtime_results(doctor, tmp_path)

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=model_packs_path,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=evidence_report,
        capture_manifest=capture_manifest,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        model_registry_report=model_registry_report,
        jetson_gate_report=jetson_report,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is True
    assert report["pilot_gate_passed"] is True
    assert report["production_gate_passed"] is True
    assert report["sales_status"] == "ready_to_sell_production_compliance"
    assert report["factory_ppe_pack_status"] == "ready_to_sell_production_compliance"
    assert report["runtime_handoff_status"] == "registered_closed_set_candidate"
    assert report["optional_gate_status"]["model_registry"].startswith("ok:")
    assert report["closed_set_model_key_active"] is True
    assert report["closed_set_registry_model_registered"] is True
    assert report["closed_set_handoff"]["missing_label_minimums"] == {}
    assert report["closed_set_handoff"]["production_missing_label_minimums"] == {}
    assert report["production_blockers"] == []
    assert report["closed_set_handoff"]["capture_matrix_progress"]["gate_passed"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_progress"]["gate_passed"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_sidecar_validation"]["valid"] is True
    assert report["closed_set_handoff"]["label_review_import_sidecar_validation"]["valid"] is True
    assert report["closed_set_handoff"]["training_capture_preflight"]["capture_matrix_manifest"]["valid"] is True
    assert report["closed_set_handoff"]["training_capture_preflight"]["label_review_import_manifest"]["valid"] is True
    label_review_schema = report["closed_set_handoff"]["label_review_csv"]["schema"]
    assert label_review_schema["taxonomy_version"] == doctor.TAXONOMY_VERSION
    assert label_review_schema["label_format"] == doctor.YOLO_LABEL_FORMAT
    assert "source_manifest_sha256" in label_review_schema["generated_guidance_fields"]
    assert label_review_schema["approval_gate"]["requires_required_class_ids_present"] is True
    assert report["closed_set_handoff"]["production_training_plan_preflight"]["ok"] is True
    assert report["closed_set_handoff"]["production_training_plan_preflight"]["source_lineage"]["capture_manifest"]["file"]["sha256"] == capture_manifest_sha256
    assert report["next_actions"] == []


def test_apron_harness_readiness_blocks_stale_promotion_source_manifest(tmp_path: Path):
    doctor = _load_doctor()

    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    factory_pack = payload["packs"]["factory_ppe_3cam"]
    factory_pack["status"] = "ready_to_sell_production_compliance"
    factory_pack["sourcing_status"]["apron_harness_result"] = "cleared_closed_set_dataset_approved"
    factory_pack["runtime_handoff"]["status"] = "registered_closed_set_candidate"
    factory_pack["model_keys"].append("ppe_closed_set_candidate")
    factory_pack["registry_models"]["ppe_closed_set_candidate"] = {
        "file": "apron-harness-ppe.onnx",
        "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
        "expected_input_size": 640,
        "local_device": "mps",
        "jetson_device": "tensorrt_fp16_candidate",
    }
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    evidence_report = tmp_path / "model_pack_evidence_doctor.json"
    _write_json(
        evidence_report,
        _model_pack_evidence_report(
            model_packs_path,
            RESULT_DIR,
            "ready_to_sell_production_compliance",
        ),
    )

    capture_manifest = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest = yaml.safe_load((ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml").read_text(encoding="utf-8"))
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1000,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    capture_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    capture_manifest_sha256 = doctor._sha256_file(capture_manifest)
    doctor.build_training_plan = lambda **_: _mps_training_plan(capture_manifest_sha256)

    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required", "c" * 64))
    _write_json(harness_report, _promotion_report("harness_required", capture_manifest_sha256))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=model_packs_path,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=evidence_report,
        capture_manifest=capture_manifest,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is False
    assert report["production_gate_passed"] is False
    assert report["optional_gate_status"]["apron_promotion"].startswith(
        "candidate_capture_matrix_manifest_source_manifest_sha_mismatch:"
    )
    assert report["optional_gate_status"]["harness_promotion"].startswith("ok:")
    assert "missing_or_failed_apron_closed_set_promotion_report" in report["production_blockers"]
    assert any("status is production before closed-set data/runtime/Jetson gates pass" in error for error in report["errors"])


def test_apron_harness_readiness_blocks_production_without_training_dataset_provenance(tmp_path: Path):
    doctor = _load_doctor()

    model_packs_path = tmp_path / "model_packs.yaml"
    payload = yaml.safe_load(MODEL_PACKS_PATH.read_text(encoding="utf-8"))
    factory_pack = payload["packs"]["factory_ppe_3cam"]
    factory_pack["status"] = "ready_to_sell_production_compliance"
    factory_pack["sourcing_status"]["apron_harness_result"] = "cleared_closed_set_dataset_approved"
    factory_pack["runtime_handoff"]["status"] = "registered_closed_set_candidate"
    factory_pack["model_keys"].append("ppe_closed_set_candidate")
    factory_pack["registry_models"]["ppe_closed_set_candidate"] = {
        "file": "apron-harness-ppe.onnx",
        "registry_path": "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
        "expected_input_size": 640,
        "local_device": "mps",
        "jetson_device": "tensorrt_fp16_candidate",
    }
    model_packs_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    evidence_report = tmp_path / "model_pack_evidence_doctor.json"
    _write_json(
        evidence_report,
        _model_pack_evidence_report(
            model_packs_path,
            RESULT_DIR,
            "ready_to_sell_production_compliance",
        ),
    )

    capture_manifest = tmp_path / "apron_harness_capture_manifest.yaml"
    manifest = yaml.safe_load((ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml").read_text(encoding="utf-8"))
    manifest["counts"]["labeled_images_per_class"] = {
        "person": 1000,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    capture_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    plan_without_provenance = _mps_training_plan(doctor._sha256_file(capture_manifest))
    plan_without_provenance.pop("dataset_provenance")
    doctor.build_training_plan = lambda **_: plan_without_provenance

    apron_report = tmp_path / "apron_promotion.json"
    harness_report = tmp_path / "harness_promotion.json"
    jetson_report = tmp_path / "jetson_gate.json"
    _write_json(apron_report, _promotion_report("apron_required"))
    _write_json(harness_report, _promotion_report("harness_required"))
    _write_json(jetson_report, _jetson_gate_report())

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=model_packs_path,
        result_dir=RESULT_DIR,
        model_pack_evidence_report=evidence_report,
        capture_manifest=capture_manifest,
        apron_promotion_report=apron_report,
        harness_promotion_report=harness_report,
        jetson_gate_report=jetson_report,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is False
    assert report["production_gate_passed"] is False
    assert "closed_set_training_dataset_provenance_has_errors" in report["production_blockers"]
    assert "closed_set_training_dataset_provenance_missing_source_manifest_sha256" in report["production_blockers"]
    assert "closed_set_training_dataset_provenance_not_required" not in report["production_blockers"]
    assert report["closed_set_handoff"]["training_dataset_provenance_status"]["valid"] is False
    assert report["closed_set_handoff"]["production_training_dataset_provenance"]["required"] is True
    assert report["closed_set_handoff"]["production_training_dataset_provenance_source"] == (
        "production_dataset_provenance_preflight"
    )
    assert report["closed_set_handoff"]["training_dataset_provenance"] == {}
    assert any("status is production before closed-set data/runtime/Jetson gates pass" in error for error in report["errors"])


def test_training_dataset_provenance_status_rejects_declared_source_manifest_mismatch():
    doctor = _load_doctor()
    provenance = {
        "required": True,
        "checked": True,
        "source_manifest": "/cleared/apron_harness_capture_manifest.yaml",
        "declared_source_manifest_sha256": "c" * 64,
        "source_manifest_sha256": "b" * 64,
        "permission": "controlled_capture_cleared",
        "permission_allowed": True,
        "missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "errors": [],
    }

    status = doctor._training_dataset_provenance_status(
        provenance,
        expected_source_manifest_sha256="b" * 64,
    )

    assert status["valid"] is False
    assert (
        "closed_set_training_dataset_provenance_declared_source_manifest_sha_mismatch"
        in status["blockers"]
    )


def test_promotion_dataset_provenance_rejects_declared_source_manifest_mismatch():
    doctor = _load_doctor()
    provenance = {
        "required": True,
        "checked": True,
        "source_manifest": "/cleared/apron_harness_capture_manifest.yaml",
        "declared_source_manifest_sha256": "c" * 64,
        "source_manifest_sha256": "b" * 64,
        "permission": "controlled_capture_cleared",
        "permission_allowed": True,
        "missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "expected_missing_ppe_label_policy": EXPECTED_MISSING_PPE_LABEL_POLICY,
        "errors": [],
    }
    sidecar = {"source_manifest_sha256": "b" * 64}

    error = doctor._promotion_dataset_provenance_error(provenance, sidecar)

    assert error == "candidate_training_dataset_provenance_declared_source_manifest_sha_mismatch"


def test_apron_harness_readiness_blocks_stale_production_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    original_writer = doctor._write_capture_matrix_sidecar

    def stale_sidecar_writer(**kwargs):
        info = original_writer(**kwargs)
        if kwargs.get("mode") == "production":
            sidecar_path = Path(info["path"])
            if not sidecar_path.is_absolute() and not sidecar_path.exists():
                sidecar_path = ROOT / sidecar_path
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            payload["matrix_csv_sha256"] = "stale"
            sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            info["sha256"] = doctor._sha256_file(sidecar_path)
        return info

    doctor._write_capture_matrix_sidecar = stale_sidecar_writer

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is False
    assert report["production_gate_passed"] is False
    assert "closed_set_production_capture_matrix_sidecar_invalid" in report["production_blockers"]
    assert report["closed_set_handoff"]["production_capture_matrix_sidecar_validation"]["valid"] is False
    assert "matrix_csv_sha256 does not match current capture matrix" in "\n".join(report["errors"])


def test_apron_harness_readiness_blocks_stale_pilot_capture_sidecar(tmp_path: Path):
    doctor = _load_doctor()
    original_writer = doctor._write_capture_matrix_sidecar

    def stale_sidecar_writer(**kwargs):
        info = original_writer(**kwargs)
        if kwargs.get("mode") == "pilot":
            sidecar_path = Path(info["path"])
            if not sidecar_path.is_absolute() and not sidecar_path.exists():
                sidecar_path = ROOT / sidecar_path
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            payload["source_manifest_sha256"] = "stale"
            sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            info["sha256"] = doctor._sha256_file(sidecar_path)
        return info

    doctor._write_capture_matrix_sidecar = stale_sidecar_writer

    report = doctor.audit_apron_harness_readiness(
        model_packs_path=MODEL_PACKS_PATH,
        result_dir=RESULT_DIR,
        capture_matrix_csv_out=tmp_path / "apron_harness_capture_matrix.csv",
        production_capture_matrix_csv_out=tmp_path / "apron_harness_production_capture_matrix.csv",
    )

    assert report["ok"] is False
    assert report["production_gate_passed"] is False
    assert "closed_set_capture_matrix_sidecar_invalid" in report["production_blockers"]
    assert report["closed_set_handoff"]["capture_matrix_sidecar_validation"]["valid"] is False
    assert "source_manifest_sha256 does not match current capture manifest" in "\n".join(report["errors"])


def test_candidate_runtime_runner_restores_after_failed_run(monkeypatch: pytest.MonkeyPatch):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(runner, "_load_json", lambda _path: _candidate_runtime_packet())
    monkeypatch.setattr(runner, "_model_ready", lambda _packet: (True, []))

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {
            "command": command,
            "returncode": 1 if command == "run-fails" else 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    exit_code = runner.main(["--packet", "packet.json", "--readiness-report", "report.json", "--execute", "--json"])

    assert exit_code == 1
    assert calls == ["backup", "validate", "plan", "apply", "run-fails", "restore"]


def test_candidate_runtime_runner_execute_refuses_when_model_missing(monkeypatch: pytest.MonkeyPatch):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(runner, "_load_json", lambda _path: _candidate_runtime_packet())
    monkeypatch.setattr(
        runner,
        "_model_ready",
        lambda _packet: (False, ["ppe_closed_set_candidate missing"]),
    )
    monkeypatch.setattr(runner, "_run_shell", lambda command: calls.append(command) or {})

    exit_code = runner.main(["--packet", "packet.json", "--readiness-report", "report.json", "--execute", "--json"])

    assert exit_code == 2
    assert calls == []


def test_candidate_runtime_runner_refreshes_blocked_missing_model_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []
    out_path = tmp_path / "preflight_refresh.json"

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(runner, "_load_json", lambda _path: _candidate_runtime_packet())
    monkeypatch.setattr(
        runner,
        "_model_ready",
        lambda _packet: (False, ["ppe_closed_set_candidate missing"]),
    )

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(
        runner,
        "_validate_step_result",
        lambda step: {
            "ok": False,
            "blockers": [
                "factory_missing_apron_active_closed_set:blocked_missing_required_model_preflight"
            ],
            "result": {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "preflight_blocked_missing_required_model": True,
                "errors": ["blocked_missing_required_model_preflight"],
            },
        },
    )

    exit_code = runner.main([
        "--packet",
        "packet.json",
        "--readiness-report",
        "report.json",
        "--refresh-blocked-preflight",
        "--out",
        str(out_path),
        "--json",
    ])

    assert exit_code == 0
    assert calls == ["backup", "validate", "plan", "apply", "run-fails", "restore"]
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "refresh_blocked_preflight"
    assert saved["ok"] is True
    assert saved["out"] == str(out_path)


def test_candidate_runtime_runner_refresh_accepts_nonzero_blocked_run(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(runner, "_load_json", lambda _path: _candidate_runtime_packet())
    monkeypatch.setattr(
        runner,
        "_model_ready",
        lambda _packet: (False, ["ppe_closed_set_candidate missing"]),
    )

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {
            "command": command,
            "returncode": 1 if command == "run-fails" else 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(
        runner,
        "_validate_step_result",
        lambda step: {
            "ok": False,
            "blockers": [
                "factory_missing_apron_active_closed_set:blocked_missing_required_model_preflight"
            ],
            "result": {
                "scenario_id": "factory_missing_apron_active_closed_set",
                "preflight_blocked_missing_required_model": True,
                "errors": ["blocked_missing_required_model_preflight"],
            },
        },
    )

    exit_code = runner.main([
        "--packet",
        "packet.json",
        "--readiness-report",
        "report.json",
        "--refresh-blocked-preflight",
        "--json",
    ])

    assert exit_code == 0
    assert calls == ["backup", "validate", "plan", "apply", "run-fails", "restore"]


def test_candidate_runtime_runner_refresh_blocks_when_model_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(runner, "_load_json", lambda _path: _candidate_runtime_packet())
    monkeypatch.setattr(runner, "_model_ready", lambda _packet: (True, []))
    monkeypatch.setattr(runner, "_run_shell", lambda command: calls.append(command) or {})

    exit_code = runner.main([
        "--packet",
        "packet.json",
        "--readiness-report",
        "report.json",
        "--refresh-blocked-preflight",
        "--json",
    ])

    assert exit_code == 1
    assert calls == []


def test_candidate_runtime_runner_plan_preserves_current_result_status():
    runner = _load_candidate_runtime_runner()
    packet = _candidate_runtime_packet()
    packet["candidate_runtime_execution_plan"]["steps"][0]["current_result"] = {
        "path": "qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json",
        "exists": True,
        "valid": False,
        "preflight_blocked_missing_required_model": True,
        "errors": ["blocked_missing_required_model_preflight"],
    }

    plan = runner.build_plan(packet)

    current_result = plan["steps"][0]["current_result"]
    assert current_result["exists"] is True
    assert current_result["valid"] is False
    assert current_result["preflight_blocked_missing_required_model"] is True
    assert current_result["errors"] == ["blocked_missing_required_model_preflight"]


def test_candidate_runtime_runner_restores_after_missing_result_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []
    result_path = tmp_path / "missing_result.json"

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda _path: _candidate_runtime_packet(expected_result_path=result_path),
    )
    monkeypatch.setattr(runner, "_model_ready", lambda _packet: (True, []))

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    exit_code = runner.main(["--packet", "packet.json", "--readiness-report", "report.json", "--execute", "--json"])

    assert exit_code == 1
    assert calls == ["backup", "validate", "plan", "apply", "run-fails", "restore"]


def test_candidate_runtime_runner_accepts_valid_result_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_runtime_runner()
    calls: list[str] = []
    result_path = tmp_path / "factory_missing_apron_active_closed_set.json"

    monkeypatch.setattr(
        runner,
        "validate_production_gate_packet",
        lambda packet_path, readiness_report_path=None: {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda _path: _candidate_runtime_packet(expected_result_path=result_path),
    )
    monkeypatch.setattr(runner, "_model_ready", lambda _packet: (True, []))

    def fake_run(command: str) -> dict:
        calls.append(command)
        if command == "run-fails":
            result_path.write_text(
                json.dumps(
                    _candidate_runtime_result(
                        "apron_required",
                        "active",
                        scenario_id="factory_missing_apron_active_closed_set",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    exit_code = runner.main(["--packet", "packet.json", "--readiness-report", "report.json", "--execute", "--json"])

    assert exit_code == 0
    assert calls == ["backup", "validate", "plan", "apply", "run-fails", "restore"]


def test_source_review_runner_blocks_after_materialization_without_valid_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_source_review_runner()
    seed_import = tmp_path / "seed_import.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    emit_manifest = tmp_path / "seed_imported.yaml"
    for path in (seed_import, capture_manifest):
        path.write_text("ok: true\n", encoding="utf-8")

    packet = {
        "first_unblock": {
            "source_review_execution_plan": [
                {"id": "validate_source_review_bundle", "command": "bundle"},
                {
                    "id": "validate_seed_import_manifest",
                    "command": "validate-import /path/to/filled/apron_harness_seed_import_manifest.yaml",
                },
                {
                    "id": "materialize_approved_seed_exports",
                    "command": (
                        "materialize /path/to/filled/apron_harness_seed_import_manifest.yaml "
                        "/path/to/cleared/apron_harness_capture_manifest.yaml "
                        "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml"
                    ),
                },
            ]
        }
    }

    def fake_run(command: str) -> dict:
        if command == "bundle":
            stdout = "REVIEW_BUNDLE: ok=True\n"
        elif command.startswith("validate-import"):
            stdout = "IMPORT_MANIFEST: gate=pass\n"
        else:
            stdout = ""
        return {"command": command, "returncode": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(
        runner,
        "validate_seed_export_import_sidecar",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("sidecar missing")),
    )

    result = runner.execute_plan(
        packet,
        seed_import_manifest=str(seed_import),
        capture_manifest=str(capture_manifest),
        emit_updated_manifest=str(emit_manifest),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "seed export import sidecar invalid after materialization" in result["blockers"][0]


def test_source_review_runner_validates_seed_sidecar_against_seed_source_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_source_review_runner()
    seed_import = tmp_path / "seed_import.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    emit_manifest = tmp_path / "seed_imported.yaml"
    seed_review_report = tmp_path / "seed_source_review.json"
    for path in (seed_import, capture_manifest, seed_review_report):
        path.write_text("ok: true\n", encoding="utf-8")

    packet = {
        "first_unblock": {
            "source_review_execution_plan": [
                {"id": "validate_source_review_bundle", "command": "bundle"},
                {
                    "id": "validate_seed_import_manifest",
                    "command": "validate-import /path/to/filled/apron_harness_seed_import_manifest.yaml",
                },
                {
                    "id": "materialize_approved_seed_exports",
                    "command": (
                        "materialize /path/to/filled/apron_harness_seed_import_manifest.yaml "
                        "/path/to/cleared/apron_harness_capture_manifest.yaml "
                        "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml"
                    ),
                },
            ]
        }
    }

    def fake_run(command: str) -> dict:
        if command == "bundle":
            stdout = "REVIEW_BUNDLE: ok=True\n"
        elif command.startswith("validate-import"):
            stdout = "IMPORT_MANIFEST: gate=pass\n"
        else:
            stdout = ""
        return {"command": command, "returncode": 0, "stdout": stdout, "stderr": ""}

    observed_kwargs: dict[str, Path] = {}

    def fake_validate_seed_export_import_sidecar(**kwargs):
        observed_kwargs.update(kwargs)
        return {"valid": True}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_SEED_SOURCE_REVIEW_REPORT", seed_review_report)
    monkeypatch.setattr(
        runner,
        "validate_seed_export_import_sidecar",
        fake_validate_seed_export_import_sidecar,
    )

    result = runner.execute_plan(
        packet,
        seed_import_manifest=str(seed_import),
        capture_manifest=str(capture_manifest),
        emit_updated_manifest=str(emit_manifest),
    )

    assert result["ok"] is True
    assert observed_kwargs["seed_source_review_report"] == seed_review_report
    assert observed_kwargs["seed_import_manifest"] == seed_import
    assert observed_kwargs["capture_manifest_path"] == emit_manifest


def test_source_review_runner_plan_marks_missing_inputs_unsupplied(tmp_path: Path):
    runner = _load_source_review_runner()
    seed_import = tmp_path / "seed_import.yaml"
    seed_import.write_text("ok: true\n", encoding="utf-8")
    missing_capture_manifest = tmp_path / "missing_capture.yaml"
    emit_manifest = tmp_path / "seed_imported.yaml"
    source_recheck = tmp_path / "source_recheck.md"
    source_recheck.write_text("fresh source notes\n", encoding="utf-8")
    review_packet = tmp_path / "source_a.review_packet.md"
    review_packet.write_text("# Source A\n", encoding="utf-8")
    packet = {
        "first_unblock": {
            "minimum_approval_path": {
                "checked": True,
                "training_usable_count": 0,
            },
            "source_recheck_artifact": {
                "path": str(source_recheck),
                "sha256": _sha256_file(source_recheck),
                "evidence_boundary": "Fresh source research evidence only; this does not approve training.",
            },
            "minimum_review_sources": [
                {
                    "source_ref": "source_a",
                    "capability": "apron_required",
                    "approval_status": "unreviewed",
                    "review_packet_path": str(review_packet),
                    "review_packet_sha256": _sha256_file(review_packet),
                    "review_evidence_template_path": str(tmp_path / "missing_evidence.yaml"),
                    "review_evidence_template_sha256": "a" * 64,
                    "review_prefill_path": str(tmp_path / "missing_prefill.md"),
                    "review_prefill_sha256": "d" * 64,
                    "review_checklist_csv_path": str(tmp_path / "missing_checklist.csv"),
                    "review_checklist_csv_sha256": "b" * 64,
                    "seed_import_manifest_template_path": str(tmp_path / "missing_seed_import.yaml"),
                    "seed_import_manifest_template_sha256": "c" * 64,
                }
            ],
            "next_source_reviews": [
                {"source_ref": "source_b", "approval_status": "unreviewed"}
            ],
            "evidence_boundary": "not approval until legal review evidence is filled",
            "source_review_execution_plan": [
                {"id": "fill_minimum_review_evidence", "required_sources": ["source_a"]}
            ]
        }
    }

    plan = runner.build_plan(
        packet,
        seed_import_manifest=str(seed_import),
        capture_manifest=str(missing_capture_manifest),
        emit_updated_manifest=str(emit_manifest),
    )

    assert plan["seed_import_manifest_supplied"] is True
    assert plan["capture_manifest_supplied"] is False
    assert plan["emit_updated_manifest_supplied"] is True
    assert plan["minimum_approval_path"]["training_usable_count"] == 0
    assert plan["source_recheck_artifact_status"]["ok"] is True
    assert plan["source_recheck_artifact_status"]["sha_matches"] is True
    assert [row["source_ref"] for row in plan["minimum_review_sources"]] == ["source_a"]
    assert [row["source_ref"] for row in plan["next_source_reviews"]] == ["source_b"]
    assert "not approval" in plan["evidence_boundary"]
    artifact_status = plan["minimum_review_artifact_status"][0]
    assert artifact_status["source_ref"] == "source_a"
    assert artifact_status["ok"] is False
    assert artifact_status["artifacts"]["review_packet"]["sha_matches"] is True
    assert "review_evidence_template_missing" in artifact_status["blockers"]
    assert "review_prefill_missing" in artifact_status["blockers"]
    assert "review_checklist_csv_missing" in artifact_status["blockers"]
    assert "seed_import_manifest_template_missing" in artifact_status["blockers"]


def test_source_review_runner_plan_blocks_stale_source_recheck_artifact(tmp_path: Path):
    runner = _load_source_review_runner()
    source_recheck = tmp_path / "source_recheck.md"
    source_recheck.write_text("fresh source notes\n", encoding="utf-8")
    packet = {
        "first_unblock": {
            "source_recheck_artifact": {
                "path": str(source_recheck),
                "sha256": "0" * 64,
                "evidence_boundary": "Fresh source research evidence only; this does not approve training.",
            },
            "source_review_execution_plan": [],
        }
    }

    plan = runner.build_plan(packet)

    status = plan["source_recheck_artifact_status"]
    assert status["ok"] is False
    assert status["exists"] is True
    assert status["sha_matches"] is False
    assert status["blockers"] == ["source_recheck_sha_mismatch"]


def test_source_review_runner_execute_blocks_stale_source_recheck_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_source_review_runner()
    source_recheck = tmp_path / "source_recheck.md"
    source_recheck.write_text("fresh source notes\n", encoding="utf-8")
    seed_import = tmp_path / "seed_import.yaml"
    seed_import.write_text("ok: true\n", encoding="utf-8")
    capture_manifest = tmp_path / "capture.yaml"
    capture_manifest.write_text("ok: true\n", encoding="utf-8")
    calls: list[str] = []

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "REVIEW_BUNDLE: ok=True\n", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        {
            "first_unblock": {
                "source_recheck_artifact": {
                    "path": str(source_recheck),
                    "sha256": "0" * 64,
                },
                "source_review_execution_plan": [
                    {"id": "validate_source_review_bundle", "command": "bundle"},
                ],
            }
        },
        seed_import_manifest=str(seed_import),
        capture_manifest=str(capture_manifest),
        emit_updated_manifest=str(tmp_path / "seed_imported.yaml"),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "source-recheck artifact is not ready" in result["blockers"][0]
    assert "source_recheck_sha_mismatch" in result["blockers"][0]
    assert calls == []


def test_source_review_runner_requires_review_bundle_pass_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_source_review_runner()
    seed_import = tmp_path / "seed_import.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    for path in (seed_import, capture_manifest):
        path.write_text("ok: true\n", encoding="utf-8")
    packet = {
        "first_unblock": {
            "source_review_execution_plan": [
                {"id": "validate_source_review_bundle", "command": "bundle"},
                {
                    "id": "validate_seed_import_manifest",
                    "command": "validate-import /path/to/filled/apron_harness_seed_import_manifest.yaml",
                },
            ]
        }
    }

    calls = []

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "bundle checked\n", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        packet,
        seed_import_manifest=str(seed_import),
        capture_manifest=str(capture_manifest),
        emit_updated_manifest=str(tmp_path / "seed_imported.yaml"),
    )

    assert result["ok"] is False
    assert result["blocked"] is False
    assert result["blockers"] == ["REVIEW_BUNDLE: ok=True is required before source review execution"]
    assert calls == ["bundle"]


def test_controlled_capture_runner_plan_marks_missing_inputs_unsupplied(tmp_path: Path):
    runner = _load_controlled_capture_runner()
    label_review_csv = tmp_path / "label_review.csv"
    seed_import = tmp_path / "seed_import.yaml"
    for path in (label_review_csv, seed_import):
        path.write_text("ok\n", encoding="utf-8")
    missing_capture_manifest = tmp_path / "missing_capture.yaml"
    emit_manifest = tmp_path / "reviewed.yaml"
    capture_matrix = tmp_path / "capture_matrix.csv"
    capture_matrix.write_text("row_id,review_status\n", encoding="utf-8")

    plan = runner.build_plan(
        {
            "first_unblock": {
                "controlled_capture_path": {
                    "starter_execution_plan": [],
                    "production_capture_matrix_path": str(capture_matrix),
                    "production_capture_matrix_sha256": _sha256_file(capture_matrix),
                    "production_capture_matrix_sidecar_path": str(
                        tmp_path / "missing_matrix_sidecar.json"
                    ),
                    "production_capture_matrix_sidecar_sha256": "a" * 64,
                }
            }
        },
        mode="starter",
        label_review_csv=str(label_review_csv),
        seed_import_manifest=str(seed_import),
        capture_manifest=str(missing_capture_manifest),
        emit_updated_manifest=str(emit_manifest),
    )

    assert plan["label_review_csv_supplied"] is True
    assert plan["seed_import_manifest_supplied"] is True
    assert plan["capture_manifest_supplied"] is False
    assert plan["emit_updated_manifest_supplied"] is True
    artifacts = {item["name"]: item for item in plan["artifact_status"]}
    assert artifacts["production_capture_matrix"]["sha_matches"] is True
    assert artifacts["production_capture_matrix_sidecar"]["ok"] is False
    assert artifacts["production_capture_matrix_sidecar"]["blockers"] == ["missing"]
    assert plan["production_capture_matrix_gate"]["gate_passed"] is False


def test_controlled_capture_runner_plan_reports_source_recheck_status(tmp_path: Path):
    runner = _load_controlled_capture_runner()
    source_recheck = tmp_path / "source_recheck.md"
    source_recheck.write_text("fresh source notes\n", encoding="utf-8")

    plan = runner.build_plan(
        {
            "first_unblock": {
                "source_recheck_artifact": {
                    "path": str(source_recheck),
                    "sha256": _sha256_file(source_recheck),
                },
                "controlled_capture_path": {"starter_execution_plan": []},
            }
        },
        mode="starter",
    )

    assert plan["source_recheck_artifact_status"]["exists"] is True
    assert plan["source_recheck_artifact_status"]["sha_matches"] is True


def test_controlled_capture_runner_full_mode_blocks_failed_matrix_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_controlled_capture_runner()
    label_review_csv = tmp_path / "label_review.csv"
    seed_import = tmp_path / "seed_import.yaml"
    matrix = tmp_path / "matrix.csv"
    sidecar = tmp_path / "matrix.csv.manifest.json"
    for path in (label_review_csv, seed_import):
        path.write_text("ok\n", encoding="utf-8")
    matrix.write_text("row_id,review_status\n", encoding="utf-8")
    sidecar.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    def fake_run(command: str) -> dict:
        calls.append(command)
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        {
            "first_unblock": {
                "controlled_capture_path": {
                    "row_count": 21,
                    "ready_rows": 0,
                    "missing_labeled_examples": 2404,
                    "unapproved_rows": 21,
                    "unsafe_storage_rows": 21,
                    "production_capture_matrix_path": str(matrix),
                    "production_capture_matrix_sha256": _sha256_file(matrix),
                    "production_capture_matrix_sidecar_path": str(sidecar),
                    "production_capture_matrix_sidecar_sha256": _sha256_file(sidecar),
                    "commands": [
                        "validate --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv"
                    ],
                }
            }
        },
        mode="full",
        label_review_csv=str(label_review_csv),
        seed_import_manifest=str(seed_import),
        capture_manifest=None,
        emit_updated_manifest=str(tmp_path / "reviewed.yaml"),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "production_capture_matrix_gate_not_passed" in result["blockers"][0]
    assert calls == []


def test_controlled_capture_runner_plan_preserves_post_capture_checklist():
    runner = _load_controlled_capture_runner()
    packet = {
        "first_unblock": {
            "controlled_capture_path": {
                "starter_execution_plan": [],
                "starter_success_criteria": {
                    "filled_starter_csv": "LABEL_REVIEW_VALIDATION: gate=pass",
                    "promotion_boundary": "starter rows are not enough for production training",
                },
                "post_capture_evidence_checklist": [
                    {
                        "id": "filled_production_capture_matrix",
                        "pass_signal": "capture_matrix_progress.gate_passed=true",
                    }
                ],
            }
        }
    }

    plan = runner.build_plan(packet, mode="starter")

    assert "LABEL_REVIEW_VALIDATION: gate=pass" in plan["starter_success_criteria"][
        "filled_starter_csv"
    ]
    assert "not enough for production training" in plan["starter_success_criteria"][
        "promotion_boundary"
    ]
    assert [item["id"] for item in plan["post_capture_evidence_checklist"]] == [
        "filled_production_capture_matrix"
    ]


def test_controlled_capture_runner_blocks_after_import_without_valid_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_controlled_capture_runner()
    label_review_csv = tmp_path / "label_review.csv"
    seed_import = tmp_path / "seed_import.yaml"
    emit_manifest = tmp_path / "reviewed.yaml"
    for path in (label_review_csv, seed_import):
        path.write_text("ok\n", encoding="utf-8")

    packet = {
        "first_unblock": {
            "controlled_capture_path": {
                "starter_execution_plan": [
                    {
                        "id": "validate_starter_label_review_csv",
                        "command": (
                            "validate --validate-label-review-csv "
                            "/path/to/filled/apron_harness_production_starter_label_review.csv"
                        ),
                    },
                    {
                        "id": "import_starter_label_review_csv",
                        "command": (
                            "import --import-label-review-csv "
                            "/path/to/filled/apron_harness_production_starter_label_review.csv "
                            "--out qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml"
                        ),
                    },
                ]
            }
        }
    }

    def fake_run(command: str) -> dict:
        stdout = "LABEL_REVIEW_VALIDATION: gate=pass\n" if "--validate-label-review-csv" in command else ""
        return {"command": command, "returncode": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(
        runner,
        "validate_label_review_import_sidecar",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("sidecar missing")),
    )

    result = runner.execute_plan(
        packet,
        mode="starter",
        label_review_csv=str(label_review_csv),
        seed_import_manifest=str(seed_import),
        capture_manifest=None,
        emit_updated_manifest=str(emit_manifest),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "label review import sidecar invalid after import" in result["blockers"][0]


def test_controlled_capture_runner_validates_sidecar_against_label_review_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_controlled_capture_runner()
    label_review_csv = tmp_path / "label_review.csv"
    seed_import = tmp_path / "seed_import.yaml"
    emit_manifest = tmp_path / "reviewed.yaml"
    for path in (label_review_csv, seed_import):
        path.write_text("ok\n", encoding="utf-8")

    packet = {
        "first_unblock": {
            "controlled_capture_path": {
                "starter_execution_plan": [
                    {
                        "id": "validate_starter_label_review_csv",
                        "command": (
                            "validate --validate-label-review-csv "
                            "/path/to/filled/apron_harness_production_starter_label_review.csv"
                        ),
                    },
                    {
                        "id": "import_starter_label_review_csv",
                        "command": (
                            "import --import-label-review-csv "
                            "/path/to/filled/apron_harness_production_starter_label_review.csv "
                            "--out qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml"
                        ),
                    },
                ]
            }
        }
    }

    def fake_run(command: str) -> dict:
        stdout = "LABEL_REVIEW_VALIDATION: gate=pass\n" if "--validate-label-review-csv" in command else ""
        return {"command": command, "returncode": 0, "stdout": stdout, "stderr": ""}

    observed_kwargs: dict[str, Path] = {}

    def fake_validate_label_review_import_sidecar(**kwargs):
        observed_kwargs.update(kwargs)
        return {"valid": True}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(
        runner,
        "validate_label_review_import_sidecar",
        fake_validate_label_review_import_sidecar,
    )

    result = runner.execute_plan(
        packet,
        mode="starter",
        label_review_csv=str(label_review_csv),
        seed_import_manifest=str(seed_import),
        capture_manifest=None,
        emit_updated_manifest=str(emit_manifest),
    )

    assert result["ok"] is True
    assert observed_kwargs["capture_manifest_path"] == emit_manifest
    assert observed_kwargs["label_review_csv"] == label_review_csv


def test_candidate_training_runner_requires_ready_to_train_preflight_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_run_shell",
        lambda command: {"command": command, "returncode": 0, "stdout": "", "stderr": ""},
    )

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_result.with_suffix(".plan.json")),
        run_training=True,
        run_candidate_doctor=False,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "status=ready_to_train" in result["blockers"][0]


def test_candidate_training_runner_requires_trained_result_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "status=trained" in result["blockers"][0]


def test_candidate_training_runner_blocks_registry_copy_without_promotion_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    candidate_report = tmp_path / "candidate_report.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "trained"}) + "\n", encoding="utf-8")
        elif command.startswith("candidate-doctor"):
            candidate_report.write_text(
                json.dumps(_candidate_report_with_selected_export()) + "\n",
                encoding="utf-8",
            )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_REPORT", str(candidate_report))
    monkeypatch.setattr(runner, "DEFAULT_APRON_PROMOTION_REPORT", str(tmp_path / "apron_promotion.json"))
    monkeypatch.setattr(runner, "DEFAULT_HARNESS_PROMOTION_REPORT", str(tmp_path / "harness_promotion.json"))

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=True,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "apron_required side-by-side promotion report is missing or unreadable" in result["blockers"]


def test_candidate_training_runner_requires_candidate_report_ready_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    candidate_report = tmp_path / "candidate_report.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "trained"}) + "\n", encoding="utf-8")
        elif command.startswith("candidate-doctor"):
            candidate_report.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "promotion_manifest": {
                            "candidate_status": "needs_work",
                            "runtime_handoff": {
                                "selected_export": {"sha256": "d" * 64},
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_REPORT", str(candidate_report))

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "candidate_status must be ready_for_side_by_side_runtime_test" in result["blockers"][0]


def test_candidate_training_runner_rejects_candidate_report_without_source_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    candidate_report = tmp_path / "candidate_report.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "trained"}) + "\n", encoding="utf-8")
        elif command.startswith("candidate-doctor"):
            payload = _candidate_report_with_selected_export()
            payload["promotion_manifest"]["seed_export_import_manifest"].pop("source_recheck")
            candidate_report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_REPORT", str(candidate_report))

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=False,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert (
        "candidate report seed_export_import_manifest.source_recheck is required"
        in result["blockers"][0]
    )


def test_candidate_training_runner_allows_registry_copy_after_matching_promotion_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    candidate_report = tmp_path / "candidate_report.json"
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "trained"}) + "\n", encoding="utf-8")
        elif command.startswith("candidate-doctor"):
            candidate_report.write_text(
                json.dumps(_candidate_report_with_selected_export()) + "\n",
                encoding="utf-8",
            )
            candidate_sha = runner._sha256_file(candidate_report)
            for capability, path in (
                ("apron_required", apron_promotion),
                ("harness_required", harness_promotion),
            ):
                path.write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "capability": capability,
                            "promotion_status": "ready_for_runtime_registration",
                            "candidate_report_sha256": candidate_sha,
                            "candidate_selected_export": {"sha256": "d" * 64},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_REPORT", str(candidate_report))
    monkeypatch.setattr(runner, "DEFAULT_APRON_PROMOTION_REPORT", str(apron_promotion))
    monkeypatch.setattr(runner, "DEFAULT_HARNESS_PROMOTION_REPORT", str(harness_promotion))

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=True,
    )

    assert result["ok"] is True
    assert result["blocked"] is False
    assert [run["step_id"] for run in result["runs"]][-1] == "registry_copy_after_promotions"


def test_candidate_training_runner_blocks_registry_copy_on_promotion_selected_export_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_candidate_training_runner()
    dataset_yaml = tmp_path / "dataset.yaml"
    capture_manifest = tmp_path / "capture.yaml"
    seed_import = tmp_path / "seed_import.yaml"
    training_result = tmp_path / "training_result.json"
    training_plan = training_result.with_suffix(".plan.json")
    candidate_report = tmp_path / "candidate_report.json"
    apron_promotion = tmp_path / "apron_promotion.json"
    harness_promotion = tmp_path / "harness_promotion.json"
    for path in (dataset_yaml, capture_manifest, seed_import):
        path.write_text("ok: true\n", encoding="utf-8")

    def fake_run(command: str) -> dict:
        if command.startswith("preflight"):
            training_plan.write_text(json.dumps({"status": "ready_to_train"}) + "\n", encoding="utf-8")
        elif command.startswith("train"):
            training_result.write_text(json.dumps({"status": "trained"}) + "\n", encoding="utf-8")
        elif command.startswith("candidate-doctor"):
            candidate_report.write_text(
                json.dumps(_candidate_report_with_selected_export(export_sha256="d" * 64)) + "\n",
                encoding="utf-8",
            )
            candidate_sha = runner._sha256_file(candidate_report)
            apron_promotion.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "capability": "apron_required",
                        "promotion_status": "ready_for_runtime_registration",
                        "candidate_report_sha256": candidate_sha,
                        "candidate_selected_export": {"sha256": "d" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            harness_promotion.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "capability": "harness_required",
                        "promotion_status": "ready_for_runtime_registration",
                        "candidate_report_sha256": candidate_sha,
                        "candidate_selected_export": {"sha256": "e" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_REPORT", str(candidate_report))
    monkeypatch.setattr(runner, "DEFAULT_APRON_PROMOTION_REPORT", str(apron_promotion))
    monkeypatch.setattr(runner, "DEFAULT_HARNESS_PROMOTION_REPORT", str(harness_promotion))

    result = runner.execute_plan(
        _candidate_training_packet(),
        dataset_yaml=str(dataset_yaml),
        capture_manifest=str(capture_manifest),
        seed_import_manifest=str(seed_import),
        training_result=str(training_result),
        training_plan_path=str(training_plan),
        run_training=True,
        run_candidate_doctor=True,
        run_registry_copy=True,
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "harness_required side-by-side promotion candidate_selected_export.sha256 mismatch" in result["blockers"]


def test_jetson_gate_runner_rejects_zero_exit_without_passed_gate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_jetson_gate_runner()
    candidate_report = tmp_path / "candidate_report.json"
    raw_benchmark = tmp_path / "raw.json"
    soak_report = tmp_path / "soak.json"
    out_path = tmp_path / "gate.json"
    for path in (candidate_report, raw_benchmark, soak_report):
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_run_shell",
        lambda command: {"command": command, "returncode": 0, "stdout": "", "stderr": ""},
    )

    result = runner.execute_plan(
        _jetson_gate_packet(),
        candidate_report=str(candidate_report),
        raw_benchmark=str(raw_benchmark),
        soak_report=str(soak_report),
        out_path=str(out_path),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "gate_status=jetson_gate_passed" in result["blockers"][0]


def test_jetson_gate_runner_plan_marks_missing_inputs_unsupplied(tmp_path: Path):
    runner = _load_jetson_gate_runner()
    candidate_report = tmp_path / "candidate_report.json"
    raw_benchmark = tmp_path / "raw.json"
    candidate_report.write_text("{}", encoding="utf-8")
    raw_benchmark.write_text("{}", encoding="utf-8")
    missing_soak_report = tmp_path / "missing_soak.json"
    out_path = tmp_path / "gate.json"

    plan = runner.build_plan(
        _jetson_gate_packet(),
        candidate_report=str(candidate_report),
        raw_benchmark=str(raw_benchmark),
        soak_report=str(missing_soak_report),
        out_path=str(out_path),
    )

    assert plan["candidate_report_supplied"] is True
    assert plan["raw_benchmark_supplied"] is True
    assert plan["soak_report_supplied"] is False
    assert plan["required_model_key"] == "ppe_closed_set_candidate"
    assert plan["model"] == "apron-harness-ppe.onnx"
    assert str(out_path) in plan["full_gate_command"]
    assert "--require-full-gate" in plan["full_gate_command"]
    assert "selected-export SHA" in plan["success_criteria"][0]
    artifacts = {item["name"]: item for item in plan["artifact_status"]}
    assert artifacts["candidate_report"]["exists"] is True
    assert len(artifacts["candidate_report"]["sha256"]) == 64
    assert artifacts["raw_benchmark"]["exists"] is True
    assert artifacts["soak_report"]["exists"] is False
    assert artifacts["soak_report"]["blockers"] == ["missing"]
    assert artifacts["gate_output"]["output"] is True
    assert artifacts["gate_output"]["ok"] is True


def test_jetson_gate_runner_accepts_passed_gate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_jetson_gate_runner()
    candidate_report = tmp_path / "candidate_report.json"
    raw_benchmark = tmp_path / "raw.json"
    soak_report = tmp_path / "soak.json"
    out_path = tmp_path / "gate.json"
    for path in (candidate_report, raw_benchmark, soak_report):
        path.write_text("{}", encoding="utf-8")

    def fake_run(command: str) -> dict:
        out_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "gate_status": "jetson_gate_passed",
                    "inputs": {
                        "candidate_report": str(candidate_report),
                        "raw_benchmark": str(raw_benchmark),
                        "soak_report": str(soak_report),
                    },
                    "input_file_sha256s": {
                        "candidate_report": _sha256_file(candidate_report),
                        "raw_benchmark": _sha256_file(raw_benchmark),
                        "soak_report": _sha256_file(soak_report),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        _jetson_gate_packet(),
        candidate_report=str(candidate_report),
        raw_benchmark=str(raw_benchmark),
        soak_report=str(soak_report),
        out_path=str(out_path),
    )

    assert result["ok"] is True
    assert result["blocked"] is False
    assert result["gate_output"]["gate_status"] == "jetson_gate_passed"


def test_jetson_gate_runner_rejects_stale_passed_gate_output_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _load_jetson_gate_runner()
    candidate_report = tmp_path / "candidate_report.json"
    raw_benchmark = tmp_path / "raw.json"
    soak_report = tmp_path / "soak.json"
    out_path = tmp_path / "gate.json"
    for path in (candidate_report, raw_benchmark, soak_report):
        path.write_text("{}", encoding="utf-8")

    def fake_run(command: str) -> dict:
        out_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "gate_status": "jetson_gate_passed",
                    "inputs": {
                        "candidate_report": str(candidate_report),
                        "raw_benchmark": str(raw_benchmark),
                        "soak_report": str(soak_report),
                    },
                    "input_file_sha256s": {
                        "candidate_report": "0" * 64,
                        "raw_benchmark": _sha256_file(raw_benchmark),
                        "soak_report": _sha256_file(soak_report),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run_shell", fake_run)

    result = runner.execute_plan(
        _jetson_gate_packet(),
        candidate_report=str(candidate_report),
        raw_benchmark=str(raw_benchmark),
        soak_report=str(soak_report),
        out_path=str(out_path),
    )

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "Jetson full gate output file hash mismatch: candidate_report" in result["blockers"]


def test_apron_harness_readiness_cli_writes_report(tmp_path: Path):
    doctor = _load_doctor()
    seed_doctor = _load_seed_source_doctor()
    runtime_runner = _load_candidate_runtime_runner()
    source_review_runner = _load_source_review_runner()
    controlled_capture_runner = _load_controlled_capture_runner()
    candidate_training_runner = _load_candidate_training_runner()
    jetson_gate_runner = _load_jetson_gate_runner()
    out_path = tmp_path / "apron_harness_readiness_doctor.json"
    kickoff_path = tmp_path / "apron_harness_capture_kickoff.md"
    work_order_path = tmp_path / "apron_harness_capture_work_order.md"
    matrix_csv_path = tmp_path / "apron_harness_capture_matrix.csv"
    production_matrix_csv_path = tmp_path / "apron_harness_production_capture_matrix.csv"
    label_review_csv_path = tmp_path / "apron_harness_label_review_template.csv"
    starter_label_review_csv_path = tmp_path / "apron_harness_starter_label_review_template.csv"
    production_label_review_csv_path = tmp_path / "apron_harness_production_label_review_template.csv"
    production_starter_label_review_csv_path = (
        tmp_path / "apron_harness_production_starter_label_review_template.csv"
    )
    training_dataset_yaml_path = tmp_path / "apron_harness_training_dataset.yaml"
    promotion_runbook_path = tmp_path / "apron_harness_promotion_runbook.md"
    candidate_runtime_runbook_path = tmp_path / "apron_harness_candidate_runtime_runbook.md"
    seed_import_validation_summary_path = (
        tmp_path / "apron_harness_seed_import_manifest_validation_summary.json"
    )
    production_capture_matrix_validation_summary_path = (
        tmp_path / "apron_harness_production_capture_matrix_validation_summary.json"
    )
    production_gate_packet_path = tmp_path / "apron_harness_production_gate_packet.json"
    seed_source_review_path = tmp_path / "apron_harness_seed_source_review.json"
    seed_source_work_order_path = tmp_path / "apron_harness_seed_source_review.md"
    seed_import_template_path = tmp_path / "apron_harness_seed_import_manifest.template.yaml"
    minimum_seed_import_template_path = tmp_path / "apron_harness_minimum_seed_import_manifest.template.yaml"
    seed_checklist_path = tmp_path / "apron_harness_seed_source_review_checklist.csv"
    seed_evidence_dir = tmp_path / "apron_harness_seed_source_review_evidence"
    seed_packet_dir = tmp_path / "apron_harness_seed_source_review_packets"
    next_review_batch_path = tmp_path / "apron_harness_next_source_review_batch.json"
    source_review_kickoff_path = tmp_path / "apron_harness_source_review_kickoff.md"
    source_coverage_plan_path = tmp_path / "apron_harness_source_coverage_plan.json"
    seed_bundle_path = tmp_path / "apron_harness_source_review_bundle.json"

    seed_exit_code = seed_doctor.main(
        [
            "--model-packs",
            str(MODEL_PACKS_PATH),
            "--out",
            str(seed_source_review_path),
            "--work-order-out",
            str(seed_source_work_order_path),
            "--import-template-out",
            str(seed_import_template_path),
            "--minimum-import-template-out",
            str(minimum_seed_import_template_path),
            "--review-checklist-csv-out",
            str(seed_checklist_path),
            "--review-evidence-template-dir",
            str(seed_evidence_dir),
            "--review-packet-dir",
            str(seed_packet_dir),
            "--next-review-batch-out",
            str(next_review_batch_path),
            "--review-kickoff-out",
            str(source_review_kickoff_path),
            "--source-coverage-plan-out",
            str(source_coverage_plan_path),
            "--review-bundle-out",
            str(seed_bundle_path),
        ]
    )
    assert seed_exit_code == 0

    exit_code = doctor.main(
        [
            "--model-packs",
            str(MODEL_PACKS_PATH),
            "--result-dir",
            str(RESULT_DIR),
            "--seed-source-review-report",
            str(seed_source_review_path),
            "--seed-source-review-bundle",
            str(seed_bundle_path),
            "--out",
            str(out_path),
            "--capture-kickoff-out",
            str(kickoff_path),
            "--capture-work-order-out",
            str(work_order_path),
            "--capture-matrix-csv-out",
            str(matrix_csv_path),
            "--production-capture-matrix-csv-out",
            str(production_matrix_csv_path),
            "--label-review-csv-out",
            str(label_review_csv_path),
            "--starter-label-review-csv-out",
            str(starter_label_review_csv_path),
            "--production-label-review-csv-out",
            str(production_label_review_csv_path),
            "--production-starter-label-review-csv-out",
            str(production_starter_label_review_csv_path),
            "--training-dataset-yaml-out",
            str(training_dataset_yaml_path),
            "--training-dataset-root",
            "/mnt/cleared/apron_harness_ppe",
            "--promotion-runbook-out",
            str(promotion_runbook_path),
            "--candidate-runtime-runbook-out",
            str(candidate_runtime_runbook_path),
            "--seed-import-validation-summary-out",
            str(seed_import_validation_summary_path),
            "--production-capture-matrix-validation-summary-out",
            str(production_capture_matrix_validation_summary_path),
            "--production-gate-packet-out",
            str(production_gate_packet_path),
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    assert kickoff_path.exists()
    rendered_kickoff = kickoff_path.read_text(encoding="utf-8")
    assert "Apron/Harness Controlled Capture Kickoff" in rendered_kickoff
    assert "Train only a reviewed YOLO26 nano/small candidate" in rendered_kickoff
    assert "YOLO11/YOLOv8 sources remain legacy evidence or rejected shortcuts" in rendered_kickoff
    assert "relative paths and paths under this repo are treated as repo-local and blocked" in rendered_kickoff
    assert "--seed-source-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in rendered_kickoff
    assert "Immediate Starter Rows" in rendered_kickoff
    assert "Pilot Starter Rows" in rendered_kickoff
    assert "Production Starter Rows" in rendered_kickoff
    assert "Pilot starter label-review CSV" in rendered_kickoff
    assert "Production starter label-review CSV" in rendered_kickoff
    assert str(starter_label_review_csv_path) in rendered_kickoff
    assert str(production_starter_label_review_csv_path) in rendered_kickoff
    assert "apron_required_closed_set_capture.positive.denim_apron" in rendered_kickoff
    assert "apron_required_closed_set_capture.hard_negative.jacket" in rendered_kickoff
    assert "harness_required_closed_set_capture.positive.fall_arrest_harness" in rendered_kickoff
    assert "harness_required_closed_set_capture.hard_negative.backpack_straps" in rendered_kickoff
    assert "`positive_variant`/`denim_apron` | 60 | person, apron" in rendered_kickoff
    assert "`hard_negative`/`jacket` | 10 | person" in rendered_kickoff
    assert "`positive_variant`/`fall_arrest_harness` | 200 | person, safety_harness, safety_lanyard" in rendered_kickoff
    assert "`hard_negative`/`backpack_straps` | 40 | person" in rendered_kickoff
    assert "Required per approved row: `source_clip_id`, `image_path`, `label_path`" in rendered_kickoff
    assert "external `raw_storage_ref`" in rendered_kickoff
    assert "Starter Validation Loop" in rendered_kickoff
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in rendered_kickoff
    assert "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json" in rendered_kickoff
    assert "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered_kickoff
    assert "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered_kickoff
    assert "--emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml" in rendered_kickoff
    assert "--manifest qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml" in rendered_kickoff
    assert "not enough for production training or promotion" in rendered_kickoff
    assert "Production Batches" in rendered_kickoff
    assert "detector-window telemetry with zero candidates and zero model invocations" in rendered_kickoff
    assert "apron_required_closed_set_capture" in rendered_kickoff
    assert "harness_required_closed_set_capture" in rendered_kickoff
    assert work_order_path.exists()
    rendered_work_order = work_order_path.read_text(encoding="utf-8")
    assert "Missing label annotations: `1200`" in rendered_work_order
    assert "## Production Capture Target" in rendered_work_order
    assert "Missing label annotations: `4000`" in rendered_work_order
    assert ".venv/bin/python scripts/apron_harness_dataset_doctor.py --manifest" in rendered_work_order
    assert "--mode production" in rendered_work_order
    assert "--emit-starter-label-review-csv qa/video_eval/results/apron_harness_starter_label_review_template.csv" in rendered_work_order
    assert "--emit-starter-label-review-csv qa/video_eval/results/apron_harness_production_starter_label_review_template.csv" in rendered_work_order
    assert "--schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered_work_order
    assert "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv" in rendered_work_order
    assert "--schema-only --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in rendered_work_order
    assert "apron_harness_capture_manifest.starter_reviewed.yaml" in rendered_work_order
    assert "A starter import can produce an intermediate reviewed manifest and sidecar" in rendered_work_order
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in rendered_work_order
    assert matrix_csv_path.exists()
    matrix_rows = list(csv.DictReader(matrix_csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(matrix_rows) == 21
    matrix_manifest_path = matrix_csv_path.with_suffix(matrix_csv_path.suffix + ".manifest.json")
    assert matrix_manifest_path.exists()
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    assert matrix_manifest["mode"] == "pilot"
    assert matrix_manifest["row_count"] == 21
    assert matrix_manifest["training_gate"]["requires_manifest_reconciliation"] is True
    assert matrix_manifest["progress"]["gate_passed"] is False
    assert matrix_manifest["next_capture_batches"][0]["batch_id"] == "apron_required_closed_set_capture"
    assert matrix_manifest["next_capture_batches"][0]["target_labeled_examples"] == 360
    assert matrix_manifest["next_capture_batches"][0]["missing_labeled_examples"] == 360
    assert matrix_manifest["next_capture_batches"][0]["missing_label_annotations"] == 360
    assert matrix_manifest["next_capture_batches"][0]["recommended_label_review_rows"] == 360
    assert matrix_manifest["next_capture_batches"][0]["required_label_classes"] == ["apron", "person"]
    assert matrix_manifest["next_capture_batches"][0]["capture_types"] == {
        "hard_negative": 6,
        "positive_variant": 5,
    }
    assert production_matrix_csv_path.exists()
    production_matrix_rows = list(csv.DictReader(production_matrix_csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(production_matrix_rows) == 21
    production_matrix_manifest_path = production_matrix_csv_path.with_suffix(
        production_matrix_csv_path.suffix + ".manifest.json"
    )
    assert production_matrix_manifest_path.exists()
    production_matrix_manifest = json.loads(production_matrix_manifest_path.read_text(encoding="utf-8"))
    assert production_matrix_manifest["mode"] == "production"
    assert production_matrix_manifest["row_count"] == 21
    assert production_matrix_manifest["training_gate"]["requires_permission_approved"] is True
    assert production_matrix_manifest["progress"]["gate_passed"] is False
    assert production_matrix_manifest["next_capture_batches"][0]["batch_id"] == (
        "apron_required_closed_set_capture"
    )
    assert production_matrix_manifest["next_capture_batches"][0]["target_labeled_examples"] == 1204
    assert production_matrix_manifest["next_capture_batches"][0]["missing_labeled_examples"] == 1204
    assert production_matrix_manifest["next_capture_batches"][0]["missing_label_annotations"] == 1204
    assert production_matrix_manifest["next_capture_batches"][0]["recommended_label_review_rows"] == 1204
    assert production_matrix_manifest["next_capture_batches"][0]["required_label_classes"] == [
        "apron",
        "person",
    ]
    assert production_matrix_manifest["next_capture_batches"][1]["target_labeled_examples"] == 1200
    assert production_matrix_manifest["next_capture_batches"][1]["missing_labeled_examples"] == 1200
    assert production_matrix_manifest["next_capture_batches"][1]["required_label_classes"] == [
        "person",
        "safety_harness",
        "safety_lanyard",
    ]
    assert label_review_csv_path.exists()
    label_review_rows = list(csv.DictReader(label_review_csv_path.read_text(encoding="utf-8").splitlines()))
    assert starter_label_review_csv_path.exists()
    starter_label_review_rows = list(
        csv.DictReader(starter_label_review_csv_path.read_text(encoding="utf-8").splitlines())
    )
    assert len(label_review_rows) == 720
    assert len(starter_label_review_rows) == 240
    assert production_label_review_csv_path.exists()
    production_label_review_rows = list(
        csv.DictReader(production_label_review_csv_path.read_text(encoding="utf-8").splitlines())
    )
    assert production_starter_label_review_csv_path.exists()
    production_starter_label_review_rows = list(
        csv.DictReader(production_starter_label_review_csv_path.read_text(encoding="utf-8").splitlines())
    )
    assert len(production_label_review_rows) == 2404
    assert len(production_starter_label_review_rows) == 800
    report = json.loads(out_path.read_text(encoding="utf-8"))
    seed_import_summary = report["seed_import_manifest_validation_summary"]
    assert seed_import_summary["status"] == "blocked_pending_import_manifest"
    assert seed_import_summary["candidate_count"] >= 5
    assert seed_import_summary["training_usable_count"] == 0
    assert seed_import_summary["import_manifest_review"]["included_count"] == 0
    assert seed_import_summary["import_manifest_review"]["approved_count"] == 0
    assert seed_import_summary["import_manifest_review"]["blocker_count"] >= 1
    assert "fill human/legal source review evidence" in seed_import_summary["next_action"]
    assert seed_import_validation_summary_path.exists()
    written_seed_import_summary = json.loads(
        seed_import_validation_summary_path.read_text(encoding="utf-8")
    )
    assert written_seed_import_summary == seed_import_summary
    production_matrix_summary = report["production_capture_matrix_validation_summary"]
    assert production_matrix_summary["ok"] is False
    assert production_matrix_summary["mode"] == "production"
    assert production_matrix_summary["capture_matrix_progress"]["row_count"] == 21
    assert production_matrix_summary["capture_matrix_progress"]["ready_rows"] == 0
    assert production_matrix_summary["capture_matrix_progress"]["target_labeled_examples"] == 2404
    assert production_matrix_summary["capture_matrix_progress"]["missing_labeled_examples"] == 2404
    assert production_matrix_summary["capture_matrix_progress"]["unapproved_rows"] == 21
    assert production_matrix_summary["capture_matrix_progress"]["unsafe_storage_rows"] == 21
    assert production_matrix_summary["capture_matrix_progress"]["capabilities"]["apron_required"][
        "target_labeled_examples"
    ] == 1204
    assert production_matrix_summary["capture_matrix_progress"]["capabilities"]["harness_required"][
        "target_labeled_examples"
    ] == 1200
    assert "complete capture matrix rows" in production_matrix_summary["next_action"]
    assert production_capture_matrix_validation_summary_path.exists()
    written_production_matrix_summary = json.loads(
        production_capture_matrix_validation_summary_path.read_text(encoding="utf-8")
    )
    assert written_production_matrix_summary == production_matrix_summary
    assert production_gate_packet_path.exists()
    written_production_gate_packet = json.loads(
        production_gate_packet_path.read_text(encoding="utf-8")
    )
    assert written_production_gate_packet == report["production_gate_packet"]
    assert written_production_gate_packet["kind"] == "apron_harness_production_gate_packet"
    assert report["scope"] == written_production_gate_packet["scope"]
    assert written_production_gate_packet["scope"]["active_vertical"] == "factory_ppe"
    assert written_production_gate_packet["scope"]["skipped_verticals"] == [
        "hospital",
        "hospitals",
        "rl_m",
    ]
    assert written_production_gate_packet["scope"]["capabilities"] == [
        "apron_required",
        "harness_required",
    ]
    assert "YOLO26 nano/small" in written_production_gate_packet["scope"][
        "production_model_policy"
    ]
    assert written_production_gate_packet["status"]["sales_status"] == (
        "pilot_ready_not_production_compliance"
    )
    assert written_production_gate_packet["status"]["pilot_gate_passed"] is True
    assert written_production_gate_packet["status"]["production_gate_passed"] is False
    assert written_production_gate_packet["status"]["production_blocker_count"] == report[
        "production_blocker_count"
    ]
    assert written_production_gate_packet["next_actions"][0]["id"] == (
        "approve_or_capture_training_data"
    )
    assert written_production_gate_packet["next_actions"][1]["id"] == (
        "produce_reviewed_training_manifest"
    )
    assert written_production_gate_packet["next_actions"][2]["id"] == (
        "train_and_register_candidate"
    )
    assert written_production_gate_packet["seed_import_manifest_validation_summary"] == (
        seed_import_summary
    )
    assert written_production_gate_packet["production_capture_matrix_validation_summary"] == (
        production_matrix_summary
    )
    assert written_production_gate_packet["candidate_runtime_gate"][
        "yaml_templates_valid"
    ] is True
    assert written_production_gate_packet["candidate_runtime_gate"][
        "runtime_evidence_valid"
    ] is False
    assert written_production_gate_packet["candidate_runtime_gate"][
        "present_result_count"
    ] == 6
    assert written_production_gate_packet["candidate_runtime_gate"][
        "blocked_missing_model_scenario_ids"
    ] == [
        "factory_missing_apron_active_closed_set",
        "factory_apron_false_positive_guard_closed_set",
        "factory_apron_detector_window_suppression_closed_set",
        "factory_missing_harness_active_closed_set",
        "factory_harness_false_positive_guard_closed_set",
        "factory_harness_detector_window_suppression_closed_set",
    ]
    assert (
        "inactive detector windows require zero detections, zero alerts, suppressed capability telemetry, and zero ppe_closed_set_candidate invocations"
        in written_production_gate_packet["candidate_runtime_gate"]["required_contract"]
    )
    assert candidate_runtime_runbook_path.exists()
    rendered_candidate_runtime_runbook = candidate_runtime_runbook_path.read_text(encoding="utf-8")
    assert "Apron/Harness Candidate Runtime Runbook" in rendered_candidate_runtime_runbook
    assert "Run exactly one scenario at a time" in rendered_candidate_runtime_runbook
    assert "factory_missing_apron_active_closed_set" in rendered_candidate_runtime_runbook
    assert "factory_harness_detector_window_suppression_closed_set" in rendered_candidate_runtime_runbook
    assert (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml validate"
    ) in rendered_candidate_runtime_runbook
    assert (
        ".venv/bin/python scripts/video_eval.py run --scenario "
        "factory_missing_apron_active_closed_set"
    ) in rendered_candidate_runtime_runbook
    assert (
        ".venv/bin/python scripts/apron_harness_candidate_runtime_runner.py "
        "--refresh-blocked-preflight --json"
    ) in rendered_candidate_runtime_runbook
    assert (
        "--out qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json"
        in rendered_candidate_runtime_runbook
    )
    assert "only for refreshing blocked JSON" in rendered_candidate_runtime_runbook
    assert (
        "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window"
        in rendered_candidate_runtime_runbook
    )
    candidate_runtime_execution_plan = written_production_gate_packet[
        "candidate_runtime_execution_plan"
    ]
    assert candidate_runtime_execution_plan["status"] == (
        "blocked_until_ppe_closed_set_candidate_registered"
    )
    assert candidate_runtime_execution_plan["required_model_key"] == "ppe_closed_set_candidate"
    assert candidate_runtime_execution_plan["one_detection_at_a_time"] is True
    assert candidate_runtime_execution_plan["runbook"]["path"] == str(
        candidate_runtime_runbook_path
    )
    assert len(candidate_runtime_execution_plan["runbook"]["sha256"]) == 64
    assert candidate_runtime_execution_plan["runner"]["path"] == (
        "scripts/apron_harness_candidate_runtime_runner.py"
    )
    assert len(candidate_runtime_execution_plan["runner"]["sha256"]) == 64
    assert "apron_harness_candidate_runtime_runner.py --json" in candidate_runtime_execution_plan[
        "runner"
    ]["plan_command"]
    assert (
        "apron_harness_candidate_runtime_runner.py --refresh-blocked-preflight --json"
        in candidate_runtime_execution_plan["runner"]["refresh_blocked_preflight_command"]
    )
    assert (
        "qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json"
        in candidate_runtime_execution_plan["runner"]["refresh_blocked_preflight_command"]
    )
    refresh_artifact = candidate_runtime_execution_plan["runner"][
        "refresh_blocked_preflight_artifact"
    ]
    assert refresh_artifact["path"] == (
        "qa/video_eval/results/apron_harness_candidate_runtime_preflight_refresh.json"
    )
    assert "not detection evidence" in refresh_artifact["evidence_boundary"]
    assert "apron_harness_candidate_runtime_runner.py --execute --json" in candidate_runtime_execution_plan[
        "runner"
    ]["execute_command"]
    assert "refuses to run until ppe_closed_set_candidate is registered" in candidate_runtime_execution_plan[
        "runner"
    ]["guardrail"]
    assert "refresh-blocked-preflight mode" in candidate_runtime_execution_plan[
        "runner"
    ]["guardrail"]
    assert candidate_runtime_execution_plan["scenario_order"] == [
        "factory_missing_apron_active_closed_set",
        "factory_apron_false_positive_guard_closed_set",
        "factory_apron_detector_window_suppression_closed_set",
        "factory_missing_harness_active_closed_set",
        "factory_harness_false_positive_guard_closed_set",
        "factory_harness_detector_window_suppression_closed_set",
    ]
    assert len(candidate_runtime_execution_plan["steps"]) == 6
    first_candidate_runtime_step = candidate_runtime_execution_plan["steps"][0]
    assert first_candidate_runtime_step["step"] == 1
    assert first_candidate_runtime_step["capability"] == "apron_required"
    assert first_candidate_runtime_step["role"] == "active"
    assert first_candidate_runtime_step["required_model_plan_ok"] is True
    assert first_candidate_runtime_step["one_at_a_time_ok"] is True
    assert first_candidate_runtime_step["window_ok"] is True
    assert first_candidate_runtime_step["commands"]["validate"].endswith(
        "factory_missing_apron_active_closed_set.yaml validate"
    )
    assert first_candidate_runtime_step["commands"]["apply"].endswith(" apply --yes")
    assert first_candidate_runtime_step["commands"]["run"].endswith(
        "factory_missing_apron_active_closed_set"
    )
    assert first_candidate_runtime_step["current_result"][
        "preflight_blocked_missing_required_model"
    ] is True
    assert (
        "required-model preflight confirms ppe_closed_set_candidate is ready before polling"
        in first_candidate_runtime_step["expected_evidence"]
    )
    suppression_step = candidate_runtime_execution_plan["steps"][2]
    assert suppression_step["role"] == "suppression"
    assert (
        "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window"
        in suppression_step["expected_evidence"]
    )
    assert (
        "production promotion remains blocked until side-by-side promotion, registry, and Jetson full gate pass"
        in candidate_runtime_execution_plan["success_criteria"]
    )
    candidate_training_execution_plan = written_production_gate_packet[
        "candidate_training_execution_plan"
    ]
    assert candidate_training_execution_plan["status"] == (
        "blocked_until_reviewed_production_manifest_and_training_dataset"
    )
    assert candidate_training_execution_plan["required_model_key"] == "ppe_closed_set_candidate"
    assert candidate_training_execution_plan["training_model"] == "yolo26n.pt"
    training_input_status = candidate_training_execution_plan["required_input_status"]
    assert training_input_status["dataset_yaml"] == {
        "path": "/path/to/cleared/dataset.yaml",
        "exists": False,
        "status": "required_user_supplied_reviewed_dataset_yaml",
    }
    assert training_input_status["reviewed_capture_manifest"]["exists"] is False
    assert training_input_status["reviewed_capture_manifest"]["status"] == (
        "missing_reviewed_capture_manifest"
    )
    assert training_input_status["seed_import_manifest"]["exists"] is False
    assert training_input_status["source_recheck_artifact"] == report["seed_source_review"][
        "source_recheck"
    ]
    assert training_input_status["source_recheck_artifact"]["path"].endswith(
        "apron_harness_source_recheck_2026_06_24.md"
    )
    assert training_input_status["production_capture_matrix"]["exists"] is True
    assert training_input_status["production_capture_matrix"]["gate_passed"] is False
    assert training_input_status["production_capture_matrix"]["missing_labeled_examples"] == 2404
    assert training_input_status["label_review_import_sidecar"]["valid"] is False
    assert candidate_training_execution_plan["runner"]["path"] == (
        "scripts/apron_harness_candidate_training_runner.py"
    )
    assert len(candidate_training_execution_plan["runner"]["sha256"]) == 64
    assert "apron_harness_candidate_training_runner.py --json" in candidate_training_execution_plan[
        "runner"
    ]["plan_command"]
    assert "apron_harness_candidate_training_runner.py --execute --json" in candidate_training_execution_plan[
        "runner"
    ]["execute_command"]
    assert "--run-training" in candidate_training_execution_plan["runner"]["training_command"]
    assert "actual training also requires --run-training" in candidate_training_execution_plan[
        "runner"
    ]["guardrail"]
    assert [step["id"] for step in candidate_training_execution_plan["steps"]] == [
        "training_preflight_plan",
        "train_export_candidate",
        "candidate_doctor_report",
        "side_by_side_promotion_reports",
        "registry_copy_after_promotions",
    ]
    assert "--capture-preflight-mode production" in candidate_training_execution_plan["steps"][0][
        "command"
    ]
    assert "--require-capture-preflight" in candidate_training_execution_plan["steps"][0]["command"]
    assert "--device mps" in candidate_training_execution_plan["steps"][0]["command"]
    assert "--execute" in candidate_training_execution_plan["steps"][1]["command"]
    assert "runner requires --run-training" in candidate_training_execution_plan["steps"][1][
        "guardrail"
    ]
    assert "scripts/apron_harness_candidate_doctor.py" in candidate_training_execution_plan["steps"][2][
        "command"
    ]
    assert "qa/video_eval/results/apron_closed_set_promotion_report.json" in candidate_training_execution_plan[
        "steps"
    ][3]["writes"]
    assert "scripts/apron_harness_model_registry_doctor.py" in candidate_training_execution_plan[
        "steps"
    ][4]["command"]
    assert (
        "registry copy preserves the same candidate_report_sha256 and selected-export SHA"
        in candidate_training_execution_plan["success_criteria"]
    )
    jetson_gate_execution_plan = written_production_gate_packet[
        "jetson_gate_execution_plan"
    ]
    assert jetson_gate_execution_plan["status"] == (
        "blocked_until_candidate_report_raw_benchmark_and_soak_report"
    )
    assert jetson_gate_execution_plan["required_model_key"] == "ppe_closed_set_candidate"
    assert jetson_gate_execution_plan["model"] == "apron-harness-ppe.onnx"
    assert jetson_gate_execution_plan["runner"]["path"] == (
        "scripts/apron_harness_jetson_gate_runner.py"
    )
    assert len(jetson_gate_execution_plan["runner"]["sha256"]) == 64
    assert "apron_harness_jetson_gate_runner.py --json" in jetson_gate_execution_plan[
        "runner"
    ]["plan_command"]
    assert "apron_harness_jetson_gate_runner.py --execute --json" in jetson_gate_execution_plan[
        "runner"
    ]["execute_command"]
    assert "--require-full-gate" in jetson_gate_execution_plan["runner"]["guardrail"]
    assert (
        "refuses until candidate report, raw benchmark, and soak report"
        in jetson_gate_execution_plan["runner"]["guardrail"]
    )
    assert [step["id"] for step in jetson_gate_execution_plan["steps"]] == [
        "stamp_candidate_identity",
        "run_raw_benchmark",
        "run_three_camera_soak",
        "validate_full_gate",
    ]
    assert "--write-raw-template" in jetson_gate_execution_plan["steps"][0]["command"]
    assert "--write-soak-template" in jetson_gate_execution_plan["steps"][0]["command"]
    assert "scripts/benchmark_yolo_jetson.py" in jetson_gate_execution_plan["steps"][1][
        "command"
    ]
    assert "--build-soak-report" in jetson_gate_execution_plan["steps"][2]["command"]
    assert "--suppression-result harness_required=" in jetson_gate_execution_plan["steps"][2][
        "command"
    ]
    assert "--require-full-gate" in jetson_gate_execution_plan["full_gate_command"]
    assert "scripts/apron_harness_jetson_gate_runner.py --execute --json" in jetson_gate_execution_plan[
        "steps"
    ][3]["command"]
    assert (
        "inactive detector-window scenarios show zero detections, zero alerts, and zero ppe_closed_set_candidate invocations"
        in jetson_gate_execution_plan["success_criteria"]
    )
    assert (
        "candidate seed_export_import_manifest preserves source_recheck path/SHA/non-approval boundary into the Jetson gate"
        in jetson_gate_execution_plan["success_criteria"]
    )
    assert any(
        "do not mark factory_ppe_3cam ready_to_sell_production_compliance"
        in guardrail
        for guardrail in written_production_gate_packet["promotion_guardrails"]
    )
    assert any(
        "do not register ppe_closed_set_candidate" in guardrail
        for guardrail in written_production_gate_packet["promotion_guardrails"]
    )
    first_unblock = written_production_gate_packet["first_unblock"]
    assert first_unblock["id"] == "complete_human_legal_source_review_or_controlled_capture"
    assert first_unblock["status"] == "blocked_until_human_legal_review_or_cleared_capture"
    assert first_unblock["minimum_approval_path"] == report["seed_source_minimum_approval_path"]
    assert first_unblock["next_source_reviews"] == report["seed_source_next_review_queue"][:5]
    assert first_unblock["minimum_approval_path"]["minimum_review_source_refs"] == [
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    assert first_unblock["source_recheck_artifact"] == report["seed_source_review"][
        "source_recheck"
    ]
    assert first_unblock["source_recheck_artifact"]["path"].endswith(
        "apron_harness_source_recheck_2026_06_24.md"
    )
    assert len(first_unblock["source_recheck_artifact"]["sha256"]) == 64
    assert "does not approve" in first_unblock["source_recheck_artifact"][
        "evidence_boundary"
    ]
    assert [
        review["source_ref"]
        for review in first_unblock["minimum_review_sources"]
    ] == first_unblock["minimum_approval_path"]["minimum_review_source_refs"]
    assert all(
        review["review_prefill_path"].endswith(".md")
        and len(review["review_prefill_sha256"]) == 64
        for review in first_unblock["minimum_review_sources"]
    )
    source_review_execution_plan = first_unblock["source_review_execution_plan"]
    assert [step["id"] for step in source_review_execution_plan] == [
        "validate_source_review_bundle",
        "fill_minimum_review_evidence",
        "validate_seed_import_manifest",
        "materialize_approved_seed_exports",
        "rerun_readiness_packet",
    ]
    assert "--validate-review-bundle" in source_review_execution_plan[0]["command"]
    assert source_review_execution_plan[0]["pass_signal"] == "REVIEW_BUNDLE: ok=True"
    assert source_review_execution_plan[1]["required_sources"] == [
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    assert len(source_review_execution_plan[1]["required_artifacts"]) == 3
    assert all(
        artifact["review_prefill_path"].endswith(".md")
        for artifact in source_review_execution_plan[1]["required_artifacts"]
    )
    assert (
        "approval_status is not approved_for_training"
        in source_review_execution_plan[1]["stop_if"]
    )
    assert (
        "customer-private or identifiable footage without explicit approval"
        in source_review_execution_plan[1]["stop_if"]
    )
    assert "--validate-import-manifest" in source_review_execution_plan[2]["command"]
    assert source_review_execution_plan[2]["pass_signal"] == "IMPORT_MANIFEST: gate=pass"
    assert "--import-approved-seed-exports" in source_review_execution_plan[3]["command"]
    assert (
        "/path/to/cleared/apron_harness_capture_manifest.seed_imported.yaml.seed_export_import.json"
        in source_review_execution_plan[3]["writes"]
    )
    assert any(
        "source_recheck.path and source_recheck.sha256" in evidence
        for evidence in source_review_execution_plan[3]["required_evidence"]
    )
    assert any(
        "non-approval boundary" in evidence
        for evidence in source_review_execution_plan[3]["required_evidence"]
    )
    assert "Production remains blocked" in source_review_execution_plan[4]["expected_boundary"]
    source_review_runner_packet = first_unblock["source_review_runner"]
    assert source_review_runner_packet["path"] == "scripts/apron_harness_source_review_runner.py"
    assert len(source_review_runner_packet["sha256"]) == 64
    assert "apron_harness_source_review_runner.py --json" in source_review_runner_packet[
        "plan_command"
    ]
    assert (
        "--out qa/video_eval/results/apron_harness_source_review_runner_plan.json"
        in source_review_runner_packet["plan_command"]
    )
    assert source_review_runner_packet["plan_artifact"]["path"] == (
        "qa/video_eval/results/apron_harness_source_review_runner_plan.json"
    )
    assert "durable guarded source-review runner plan only" in source_review_runner_packet[
        "plan_artifact"
    ]["evidence_boundary"]
    assert "apron_harness_source_review_runner.py --execute --json" in source_review_runner_packet[
        "execute_command"
    ]
    assert "IMPORT_MANIFEST: gate=pass" in source_review_runner_packet["guardrail"]
    assert "refuses to import seed exports" in source_review_runner_packet["guardrail"]
    assert ".seed_export_import.json sidecar validates against the seed-source review after materialization" in source_review_runner_packet[
        "guardrail"
    ]
    controlled_capture_path = first_unblock["controlled_capture_path"]
    assert controlled_capture_path["status"] == "blocked_until_capture_matrix_and_label_review_pass"
    assert controlled_capture_path["production_capture_matrix_path"] == str(production_matrix_csv_path)
    assert len(controlled_capture_path["production_capture_matrix_sha256"]) == 64
    assert controlled_capture_path["production_capture_matrix_sidecar_path"] == str(
        production_matrix_manifest_path
    )
    assert len(controlled_capture_path["production_capture_matrix_sidecar_sha256"]) == 64
    assert controlled_capture_path["row_count"] == 21
    assert controlled_capture_path["ready_rows"] == 0
    assert controlled_capture_path["missing_labeled_examples"] == 2404
    assert controlled_capture_path["unapproved_rows"] == 21
    assert controlled_capture_path["unsafe_storage_rows"] == 21
    assert controlled_capture_path["required_labeled_images_per_class"] == {
        "person": 2404,
        "apron": 1000,
        "safety_harness": 1000,
        "safety_lanyard": 1000,
    }
    assert controlled_capture_path["capabilities"]["apron_required"]["missing_labeled_examples"] == 1204
    assert controlled_capture_path["capabilities"]["harness_required"]["missing_labeled_examples"] == 1200
    assert [
        batch["batch_id"] for batch in controlled_capture_path["next_capture_batches"]
    ] == [
        "apron_required_closed_set_capture",
        "harness_required_closed_set_capture",
    ]
    assert controlled_capture_path["next_capture_batches"][0]["recommended_label_review_rows"] == 1204
    assert controlled_capture_path["next_capture_batches"][1]["recommended_label_review_rows"] == 1200
    assert [
        row["row_id"] for row in controlled_capture_path["starter_capture_rows"]
    ] == [
        "apron_required_closed_set_capture.positive.denim_apron",
        "apron_required_closed_set_capture.hard_negative.jacket",
        "harness_required_closed_set_capture.positive.fall_arrest_harness",
        "harness_required_closed_set_capture.hard_negative.backpack_straps",
    ]
    assert controlled_capture_path["starter_capture_rows"][0]["required_label_classes"] == [
        "person",
        "apron",
    ]
    assert controlled_capture_path["starter_capture_rows"][2]["required_label_classes"] == [
        "person",
        "safety_harness",
        "safety_lanyard",
    ]
    assert "raw_storage_ref=non_repo_external_storage" in controlled_capture_path[
        "starter_capture_rows"
    ][0]["operator_fill_fields"]
    starter_execution_plan = controlled_capture_path["starter_execution_plan"]
    assert [step["id"] for step in starter_execution_plan] == [
        "review_starter_rows",
        "validate_starter_label_review_csv",
        "import_starter_label_review_csv",
        "recheck_starter_reviewed_manifest",
        "rerun_readiness_packet",
    ]
    assert (
        "LABEL_REVIEW_VALIDATION: gate=pass"
        in starter_execution_plan[1]["pass_signal"]
    )
    assert (
        "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv"
        in starter_execution_plan[1]["command"]
    )
    assert (
        "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv"
        in starter_execution_plan[2]["command"]
    )
    assert (
        "qa/video_eval/results/apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json"
        in starter_execution_plan[2]["writes"]
    )
    assert (
        "--validate-capture-matrix-csv qa/video_eval/results/apron_harness_production_capture_matrix.csv"
        in starter_execution_plan[3]["command"]
    )
    assert "Production must remain blocked" in starter_execution_plan[4]["expected_boundary"]
    assert starter_execution_plan[1]["command"] == controlled_capture_path["starter_commands"][0]
    assert starter_execution_plan[2]["command"] == controlled_capture_path["starter_commands"][1]
    assert starter_execution_plan[3]["command"] == controlled_capture_path["starter_commands"][2]
    operator_handoff = controlled_capture_path["operator_handoff"]
    assert operator_handoff["capture_kickoff"]["path"] == str(kickoff_path)
    assert len(operator_handoff["capture_kickoff"]["sha256"]) == 64
    assert operator_handoff["capture_work_order"]["path"] == str(work_order_path)
    assert len(operator_handoff["capture_work_order"]["sha256"]) == 64
    assert "Starter Validation Loop" in operator_handoff["required_kickoff_phrases"]
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in operator_handoff["required_kickoff_phrases"]
    assert (
        "apron_harness_capture_manifest.starter_reviewed.yaml.label_review_import.json"
        in operator_handoff["required_kickoff_phrases"]
    )
    assert "Required Follow-Up Commands" in operator_handoff["required_work_order_phrases"]
    assert (
        "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json"
        in operator_handoff["required_work_order_phrases"]
    )
    assert (
        "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv"
        in operator_handoff["required_work_order_phrases"]
    )
    assert ".seed_export_import.json" in operator_handoff["required_work_order_phrases"]
    assert (
        "--capture-preflight-mode production --require-capture-preflight"
        in operator_handoff["required_work_order_phrases"]
    )
    label_review_templates = controlled_capture_path["label_review_templates"]
    assert label_review_templates["full_production"]["path"] == str(production_label_review_csv_path)
    assert label_review_templates["full_production"]["row_count"] == 2404
    assert len(label_review_templates["full_production"]["sha256"]) == 64
    assert label_review_templates["starter_production"]["path"] == str(
        production_starter_label_review_csv_path
    )
    assert label_review_templates["starter_production"]["row_count"] == 800
    assert label_review_templates["starter_production"]["scope"] == "immediate_starter_rows"
    assert len(label_review_templates["starter_production"]["sha256"]) == 64
    assert (
        label_review_templates["starter_production"]["schema"]["approval_gate"][
            "requires_non_repo_raw_storage_ref"
        ]
        is True
    )
    assert "raw_storage_ref=non_repo_external_storage" in controlled_capture_path[
        "required_operator_fields"
    ]
    assert any(
        "--import-label-review-csv" in command
        for command in controlled_capture_path["commands"]
    )
    assert len(controlled_capture_path["starter_commands"]) == 3
    assert any(
        "--validate-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv"
        in command
        for command in controlled_capture_path["starter_commands"]
    )
    assert any(
        "--import-label-review-csv /path/to/filled/apron_harness_production_starter_label_review.csv"
        in command
        for command in controlled_capture_path["starter_commands"]
    )
    assert any(
        "apron_harness_capture_manifest.starter_reviewed.yaml" in command
        for command in controlled_capture_path["starter_commands"]
    )
    controlled_capture_runner_packet = controlled_capture_path["runner"]
    assert controlled_capture_runner_packet["path"] == (
        "scripts/apron_harness_controlled_capture_runner.py"
    )
    assert len(controlled_capture_runner_packet["sha256"]) == 64
    assert "apron_harness_controlled_capture_runner.py --mode starter --json" in controlled_capture_runner_packet[
        "plan_command"
    ]
    assert (
        "--out qa/video_eval/results/apron_harness_controlled_capture_starter_plan.json"
        in controlled_capture_runner_packet["plan_command"]
    )
    assert controlled_capture_runner_packet["plan_artifact"]["path"] == (
        "qa/video_eval/results/apron_harness_controlled_capture_starter_plan.json"
    )
    assert "durable guarded starter controlled-capture runner plan only" in controlled_capture_runner_packet[
        "plan_artifact"
    ]["evidence_boundary"]
    assert "apron_harness_controlled_capture_runner.py --mode starter --execute --json" in controlled_capture_runner_packet[
        "execute_command"
    ]
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in controlled_capture_runner_packet[
        "guardrail"
    ]
    assert "refuses to import label reviews" in controlled_capture_runner_packet[
        "guardrail"
    ]
    assert ".label_review_import.json sidecar validates against the filled label-review CSV after import" in controlled_capture_runner_packet[
        "guardrail"
    ]
    starter_success_criteria = controlled_capture_path["starter_success_criteria"]
    assert set(starter_success_criteria) == {
        "filled_starter_csv",
        "import_sidecar",
        "starter_reviewed_manifest",
        "post_import_recheck",
        "promotion_boundary",
    }
    assert (
        "LABEL_REVIEW_VALIDATION: gate=pass"
        in starter_success_criteria["filled_starter_csv"]
    )
    assert ".label_review_import.json" in starter_success_criteria["import_sidecar"]
    assert "starter_reviewed.yaml" in starter_success_criteria["starter_reviewed_manifest"]
    assert "preserve the matrix SHA" in starter_success_criteria["post_import_recheck"]
    assert (
        "not sufficient for production training/promotion"
        in starter_success_criteria["promotion_boundary"]
    )
    post_capture_checklist = controlled_capture_path["post_capture_evidence_checklist"]
    assert written_production_gate_packet["post_capture_evidence_checklist"] == post_capture_checklist
    assert [item["id"] for item in post_capture_checklist] == [
        "filled_production_capture_matrix",
        "filled_production_label_review_csv",
        "reviewed_capture_manifest_sidecar",
        "production_training_preflight",
    ]
    assert post_capture_checklist[0]["pass_signal"] == "capture_matrix_progress.gate_passed=true"
    assert post_capture_checklist[1]["pass_signal"] == "LABEL_REVIEW_VALIDATION: gate=pass"
    assert post_capture_checklist[2]["pass_signal"] == "updated_manifest_validation.ok=true"
    assert post_capture_checklist[3]["pass_signal"] == "ready_to_train"
    assert "raw_storage_ref=non_repo_external_storage" in post_capture_checklist[1]["must_prove"]
    assert (
        "--capture-preflight-mode production --require-capture-preflight"
        in post_capture_checklist[3]["validator"]
    )
    for minimum_review in first_unblock["minimum_review_sources"]:
        assert "apron_harness_seed_source_review_packets" in minimum_review["review_packet_path"]
        assert minimum_review["review_packet_path"].endswith(".review_packet.md")
        assert len(minimum_review["review_packet_sha256"]) == 64
        assert "apron_harness_seed_source_review_evidence" in minimum_review[
            "review_evidence_template_path"
        ]
        assert minimum_review["review_evidence_template_path"].endswith(".review_evidence.yaml")
        assert len(minimum_review["review_evidence_template_sha256"]) == 64
        required_fields = minimum_review["seed_import_fill_plan"][
            "required_fields_before_include_in_training"
        ]
        assert "review_status=approved_for_training" in required_fields
        assert "raw_export_sha256" in required_fields
        assert "raw_export_local_path" in required_fields
    assert any("--validate-review-bundle" in command for command in first_unblock["commands"])
    assert any("--validate-import-manifest" in command for command in first_unblock["commands"])
    assert any("--validate-capture-matrix-csv" in command for command in first_unblock["commands"])
    assert "not approval" in first_unblock["evidence_boundary"]
    validation_exit_code = doctor.main(
        [
            "--validate-production-gate-packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
        ]
    )
    assert validation_exit_code == 0
    runner_plan_exit_code = runtime_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--json",
        ]
    )
    assert runner_plan_exit_code == 0
    runner_execute_exit_code = runtime_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--execute",
            "--json",
        ]
    )
    assert runner_execute_exit_code == 2
    source_review_plan_out = tmp_path / "source_review_plan.json"
    source_review_plan_exit_code = source_review_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--out",
            str(source_review_plan_out),
            "--json",
        ]
    )
    assert source_review_plan_exit_code == 0
    source_review_plan_doc = json.loads(source_review_plan_out.read_text(encoding="utf-8"))
    assert source_review_plan_doc["mode"] == "plan"
    assert source_review_plan_doc["out"] == str(source_review_plan_out)
    assert source_review_plan_doc["plan"]["safe_to_execute_without_approval"] is False
    assert source_review_plan_doc["plan"]["source_recheck_artifact_status"]["ok"] is True
    assert source_review_plan_doc["plan"]["source_recheck_artifact_status"][
        "sha_matches"
    ] is True
    assert source_review_plan_doc["plan"]["source_recheck_artifact_status"][
        "path"
    ].endswith("apron_harness_source_recheck_2026_06_24.md")
    source_review_execute_exit_code = source_review_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--execute",
            "--json",
        ]
    )
    assert source_review_execute_exit_code == 2
    controlled_capture_plan_out = tmp_path / "controlled_capture_plan.json"
    controlled_capture_plan_exit_code = controlled_capture_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--mode",
            "starter",
            "--out",
            str(controlled_capture_plan_out),
            "--json",
        ]
    )
    assert controlled_capture_plan_exit_code == 0
    controlled_capture_plan_doc = json.loads(
        controlled_capture_plan_out.read_text(encoding="utf-8")
    )
    assert controlled_capture_plan_doc["mode"] == "plan"
    assert controlled_capture_plan_doc["capture_mode"] == "starter"
    assert controlled_capture_plan_doc["out"] == str(controlled_capture_plan_out)
    assert controlled_capture_plan_doc["plan"]["safe_to_execute_without_reviewed_csv"] is False
    controlled_capture_execute_exit_code = controlled_capture_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--mode",
            "starter",
            "--execute",
            "--json",
        ]
    )
    assert controlled_capture_execute_exit_code == 2
    candidate_training_plan_exit_code = candidate_training_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--json",
        ]
    )
    assert candidate_training_plan_exit_code == 0
    candidate_training_execute_exit_code = candidate_training_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--execute",
            "--json",
        ]
    )
    assert candidate_training_execute_exit_code == 2
    jetson_gate_plan_exit_code = jetson_gate_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--json",
        ]
    )
    assert jetson_gate_plan_exit_code == 0
    jetson_gate_execute_exit_code = jetson_gate_runner.main(
        [
            "--packet",
            str(production_gate_packet_path),
            "--readiness-report",
            str(out_path),
            "--execute",
            "--json",
        ]
    )
    assert jetson_gate_execute_exit_code == 2
    production_gate_validation = doctor.validate_production_gate_packet(
        production_gate_packet_path,
        readiness_report_path=out_path,
    )
    assert production_gate_validation["ok"] is True
    assert production_gate_validation["minimum_review_source_count"] == 3
    assert production_gate_validation["minimum_review_source_refs"] == [
        "roboflow_harness_s4xxh",
        "roboflow_work_at_height_safety",
        "roboflow_workspace_otd88_fjwepfj1",
    ]
    assert production_gate_validation["minimum_review_artifact_count"] == 9
    assert production_gate_validation["minimum_review_artifact_sha_match_count"] == 9
    assert production_gate_validation["source_review_execution_step_count"] == 5
    assert production_gate_validation["source_review_execution_required_source_count"] == 3
    assert production_gate_validation["source_review_runner_artifact_count"] == 1
    assert production_gate_validation["candidate_runtime_execution_step_count"] == 6
    assert production_gate_validation["candidate_runtime_execution_scenario_count"] == 6
    assert production_gate_validation["candidate_runtime_runbook_artifact_count"] == 1
    assert production_gate_validation["candidate_runtime_runner_artifact_count"] == 1
    assert production_gate_validation["candidate_training_execution_step_count"] == 5
    assert production_gate_validation["candidate_training_runner_artifact_count"] == 1
    assert production_gate_validation["jetson_gate_execution_step_count"] == 4
    assert production_gate_validation["jetson_gate_runner_artifact_count"] == 1
    assert (
        production_gate_validation["controlled_capture_status"]
        == "blocked_until_capture_matrix_and_label_review_pass"
    )
    assert production_gate_validation["controlled_capture_missing_labeled_examples"] == 2404
    assert production_gate_validation["controlled_capture_next_batch_count"] == 2
    assert production_gate_validation["controlled_capture_starter_row_count"] == 4
    assert production_gate_validation["controlled_capture_label_template_count"] == 2
    assert production_gate_validation["controlled_capture_starter_command_count"] == 3
    assert production_gate_validation["controlled_capture_runner_artifact_count"] == 1
    assert production_gate_validation["controlled_capture_starter_success_criterion_count"] == 5
    assert production_gate_validation["controlled_capture_operator_handoff_artifact_count"] == 2
    assert production_gate_validation["controlled_capture_starter_execution_step_count"] == 5
    assert production_gate_validation["controlled_capture_starter_execution_command_match_count"] == 3
    assert production_gate_validation["controlled_capture_post_capture_check_count"] == 4
    stale_gate_packet_path = tmp_path / "stale_apron_harness_production_gate_packet.json"
    stale_gate_packet = dict(written_production_gate_packet)
    stale_gate_packet["status"] = dict(stale_gate_packet["status"])
    stale_gate_packet["status"]["production_blocker_count"] = 999
    stale_gate_packet_path.write_text(
        json.dumps(stale_gate_packet, indent=2) + "\n",
        encoding="utf-8",
    )
    stale_validation_exit_code = doctor.main(
        [
            "--validate-production-gate-packet",
            str(stale_gate_packet_path),
            "--readiness-report",
            str(out_path),
        ]
    )
    assert stale_validation_exit_code == 1
    stale_sha_gate_packet_path = tmp_path / "stale_sha_apron_harness_production_gate_packet.json"
    stale_sha_gate_packet = json.loads(json.dumps(written_production_gate_packet))
    stale_sha_gate_packet["first_unblock"]["minimum_review_sources"][0][
        "review_packet_sha256"
    ] = "0" * 64
    stale_sha_gate_packet_path.write_text(
        json.dumps(stale_sha_gate_packet, indent=2) + "\n",
        encoding="utf-8",
    )
    stale_sha_validation_exit_code = doctor.main(
        [
            "--validate-production-gate-packet",
            str(stale_sha_gate_packet_path),
            "--readiness-report",
            str(out_path),
        ]
    )
    assert stale_sha_validation_exit_code == 1
    assert report["closed_set_handoff"]["capture_kickoff"]["path"] == str(kickoff_path)
    assert report["closed_set_handoff"]["capture_kickoff"]["generated"] is True
    assert len(report["closed_set_handoff"]["capture_kickoff"]["sha256"]) == 64
    assert report["closed_set_handoff"]["label_review_csv"]["row_count"] == len(label_review_rows)
    assert report["closed_set_handoff"]["starter_label_review_csv"]["row_count"] == len(
        starter_label_review_rows
    )
    assert report["closed_set_handoff"]["production_label_review_csv"]["row_count"] == len(
        production_label_review_rows
    )
    assert report["closed_set_handoff"]["production_starter_label_review_csv"]["row_count"] == len(
        production_starter_label_review_rows
    )
    assert training_dataset_yaml_path.exists()
    training_dataset_yaml = yaml.safe_load(training_dataset_yaml_path.read_text(encoding="utf-8"))
    assert training_dataset_yaml["path"] == "/mnt/cleared/apron_harness_ppe"
    assert training_dataset_yaml["names"] == {
        0: "person",
        1: "apron",
        2: "safety_harness",
        3: "safety_lanyard",
    }
    assert training_dataset_yaml["rakshak_lens"]["source_manifest_sha256"] == doctor._sha256_file(
        ROOT / "qa" / "video_eval" / "datasets" / "apron_harness_capture_manifest.template.yaml"
    )
    assert training_dataset_yaml["rakshak_lens"]["permission"] == "controlled_capture_cleared"
    assert promotion_runbook_path.exists()
    promotion_runbook = promotion_runbook_path.read_text(encoding="utf-8")
    assert "Apron/Harness Closed-Set Promotion Runbook" in promotion_runbook
    assert f"- Production blocker count: `{report['production_blocker_count']}`" in promotion_runbook
    assert promotion_runbook.count("Optional gate status:") == 1
    assert "## Training Dataset YAML" in promotion_runbook
    assert "--schema-only --seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml --validate-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in promotion_runbook
    assert "--import-label-review-csv /path/to/filled/apron_harness_production_label_review.csv" in promotion_runbook
    assert "LABEL_REVIEW_VALIDATION: gate=pass" in promotion_runbook
    assert ".label_review_import.json` sidecar validates" in promotion_runbook
    assert "Taxonomy: `apron_harness_v1`" in promotion_runbook
    assert "Label format: `ultralytics_yolo_txt_normalized_xywh`" in promotion_runbook
    assert "Required import sidecar: `.label_review_import.json`" in promotion_runbook
    assert (
        "label-review CSV rows use `review_status=approved`; public seed-import manifest rows use "
        "`review_status=approved_for_training` before `include_in_training=true`"
    ) in promotion_runbook
    assert "--emit-updated-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml" in promotion_runbook
    assert "Production blocker action map:" in promotion_runbook
    assert (
        "--mode pilot --schema-only --seed-source-review-report "
        "qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest "
        "/path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv "
        "qa/video_eval/results/apron_harness_capture_matrix.csv"
    ) in promotion_runbook
    assert (
        "--mode production --schema-only --seed-source-review-report "
        "qa/video_eval/results/apron_harness_seed_source_review.json --seed-import-manifest "
        "/path/to/filled/apron_harness_seed_import_manifest.yaml --validate-capture-matrix-csv "
        "qa/video_eval/results/apron_harness_production_capture_matrix.csv"
    ) in promotion_runbook
    assert "closed_set_public_seed_sources_not_curated_or_approved" in promotion_runbook
    assert "qa/video_eval/results/apron_harness_next_source_review_batch.json" in promotion_runbook
    assert "qa/video_eval/results/apron_harness_source_review_kickoff.md" in promotion_runbook
    assert "--review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md" in promotion_runbook
    assert f"`seed_source_review_bundle`: `pass:{seed_bundle_path}`" in promotion_runbook
    assert "--source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json" in promotion_runbook
    assert "--review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json" in promotion_runbook
    assert (
        "--minimum-import-template-out qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml"
        in promotion_runbook
    )
    assert "Canonical refresh order for generated source-review artifacts" in promotion_runbook
    assert "Validate the source-review bundle before running readiness" in promotion_runbook
    assert "scripts/apron_harness_readiness_doctor.py --out qa/video_eval/results/apron_harness_readiness_doctor.json" in promotion_runbook
    assert (
        "it also writes concise seed-import and production capture-matrix validation summaries plus the machine-readable production gate packet"
        in promotion_runbook
    )
    assert "--seed-import-validation-summary-out qa/video_eval/results/apron_harness_seed_import_manifest_validation_summary.json" in promotion_runbook
    assert "--production-capture-matrix-validation-summary-out qa/video_eval/results/apron_harness_production_capture_matrix_validation_summary.json" in promotion_runbook
    assert "jq '.' qa/video_eval/results/apron_harness_seed_import_manifest_validation_summary.json" in promotion_runbook
    assert "jq '.' qa/video_eval/results/apron_harness_production_capture_matrix_validation_summary.json" in promotion_runbook
    assert "jq '.' qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert "jq '.candidate_runtime_execution_plan.steps[].artifact_status' qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert "jq '.candidate_training_execution_plan.runner' qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert "jq '.jetson_gate_execution_plan.runner' qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert "jq '.model_registry_handoff.artifact_status' qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert (
        "jq '.candidate_runtime_execution_plan' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.source_review_execution_plan' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path.starter_capture_rows' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path.label_review_templates' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path.starter_success_criteria' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path.starter_execution_plan' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert (
        "jq '.first_unblock.controlled_capture_path.post_capture_evidence_checklist' "
        "qa/video_eval/results/apron_harness_production_gate_packet.json"
    ) in promotion_runbook
    assert "production capture matrix SHA" in promotion_runbook
    assert "missing labeled-example count" in promotion_runbook
    assert "starter capture rows" in promotion_runbook
    assert "production label-review templates" in promotion_runbook
    assert "starter validation/import commands" in promotion_runbook
    assert "starter success criteria" in promotion_runbook
    assert "starter execution plan" in promotion_runbook
    assert "post-capture evidence checklist" in promotion_runbook
    assert "candidate runtime execution plan" in promotion_runbook
    assert "one-detection-at-a-time YAML validate/plan/apply/run sequence" in promotion_runbook
    assert "public source-review execution plan" in promotion_runbook
    assert "Validate that the production gate packet is fresh" in promotion_runbook
    assert "--validate-production-gate-packet qa/video_eval/results/apron_harness_production_gate_packet.json" in promotion_runbook
    assert "--readiness-report qa/video_eval/results/apron_harness_readiness_doctor.json" in promotion_runbook
    assert "scripts/video_eval.py --manifest qa/video_eval/manifest.yaml report" in promotion_runbook
    assert "Validate the generated next-review batch and handoff bundle before using packets/templates for review:" in promotion_runbook
    assert "--review-bundle-out \"\"" not in promotion_runbook
    assert "--validate-next-review-batch qa/video_eval/results/apron_harness_next_source_review_batch.json" in promotion_runbook
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in promotion_runbook
    assert "--validate-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in promotion_runbook
    assert "IMPORT_MANIFEST: gate=pass" in promotion_runbook
    assert "IMPORT_MANIFEST: gate=blocked" in promotion_runbook
    assert ".seed_export_import.json` sidecar. That sidecar must preserve" in promotion_runbook
    assert "fresh `source_recheck` artifact path/SHA/non-approval boundary" in promotion_runbook
    assert "command exit `0` alone is not enough" in promotion_runbook
    assert "exit nonzero" in promotion_runbook
    assert (
        "Requires: human/legal approval evidence is required before public seed data can enter training"
        in promotion_runbook
    )
    assert (
        "Requires: same candidate_report_sha256 and selected-export SHA across raw benchmark, soak, promotion, and registry reports"
        in promotion_runbook
    )
    assert (
        "candidate seed_export_import_manifest carries source_recheck path/SHA/non-approval boundary into the Jetson gate"
        in promotion_runbook
    )
    assert (
        "verifies the candidate `seed_export_import_manifest` preserves the `source_recheck` path, SHA, and non-approval boundary"
        in promotion_runbook
    )
    assert "Template contract status: `ready_for_candidate_identity`" in promotion_runbook
    assert "Valid templates: `2/2`" in promotion_runbook
    assert "Identity-stamped templates: `0/2`" in promotion_runbook
    assert "Templates are fillable contracts only; they are not Jetson benchmark evidence" in promotion_runbook
    assert "--write-raw-template qa/video_eval/results/factory_ppe_raw_benchmark.template.json" in promotion_runbook
    assert "--write-soak-template qa/video_eval/results/factory_ppe_3cam_soak.template.json" in promotion_runbook
    assert "The template files are not evidence." in promotion_runbook
    assert "Collect the raw candidate benchmark with candidate identity stamped into the output:" in promotion_runbook
    assert "scripts/benchmark_yolo_jetson.py" in promotion_runbook
    assert "--candidate-report qa/video_eval/results/apron_harness_candidate_report.json" in promotion_runbook
    assert "requires exactly one model" in promotion_runbook
    assert (
        "Requires: 3-camera soak proves positive apron/harness alerts, false-positive guards, and detector-window suppression"
        in promotion_runbook
    )
    assert (
        "Requires: candidate doctor must provide selected export SHA and passing per-class apron/harness metrics"
        in promotion_runbook
    )
    assert "selected export must be ONNX for the dormant registry slot" in promotion_runbook
    assert "TensorRT engine exports may be benchmarked later" in promotion_runbook
    if "local_closed_set_training_dry_run_not_on_mps" in report["production_blockers"]:
        assert "local_closed_set_training_dry_run_not_on_mps" in promotion_runbook
        assert "scripts/model_pack_doctor.py" in promotion_runbook
        assert "Current dry-run device:" in promotion_runbook
        assert "MPS probe OK:" in promotion_runbook
        assert "training_torch_status must show mps_available=true" in promotion_runbook
    assert "--emit-yolo-dataset-yaml" in promotion_runbook
    assert f"--data {training_dataset_yaml_path}" in promotion_runbook
    assert "--device mps" in promotion_runbook
    assert "--device auto" not in promotion_runbook
    assert "--capture-manifest qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml" in promotion_runbook
    assert "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in promotion_runbook
    assert "If this dry-run fails, `--out-plan` is still written with `status=failed`" in promotion_runbook
    assert "### Seed Import Export Preflight" in promotion_runbook
    assert "### Source Coverage Snapshot" in promotion_runbook
    assert "### Minimum Approval Path" in promotion_runbook
    assert "Boundary: agent-selected minimum path is not approval" in promotion_runbook
    assert "candidate_coverage_complete_pending_review" in promotion_runbook
    assert "candidate_person_mapping_present_pending_review" in promotion_runbook
    assert "roboflow_workspace_otd88_fjwepfj1" in promotion_runbook
    assert "roboflow_harness_s4xxh" in promotion_runbook
    assert "pending_human_legal_review" in promotion_runbook
    assert "roboflow_workspace_otd88_fjwepfj1__apron_required.review_packet.md" in promotion_runbook
    assert "roboflow_harness_s4xxh__harness_required.review_packet.md" in promotion_runbook
    assert "Minimum seed-import template:" in promotion_runbook
    assert "apron_harness_minimum_seed_import_manifest.template.yaml" in promotion_runbook
    assert "Scope: `minimum_priority_coverage_sources`" in promotion_runbook
    assert "Imports enabled for training: `0`" in promotion_runbook
    assert "Consistency: valid=`True`, refs_match=`True`" in promotion_runbook
    assert "Consistency boundary:" in promotion_runbook
    assert "`raw_export_local_path`" in promotion_runbook
    assert "reviewed local YOLO export ZIP" in promotion_runbook
    assert "`review_artifacts` paths and SHA-256 values" in promotion_runbook
    assert "Current review-artifact preflight approvals:" in promotion_runbook
    assert "If seed-export materialization fails, the dataset doctor still writes" in promotion_runbook
    assert "`partial_materialization`" in promotion_runbook
    assert "## Pretrained Shortcut Review" in promotion_runbook
    assert "### Next Seed-Source Reviews" in promotion_runbook
    assert "roboflow_work_at_height_safety" in promotion_runbook
    assert "https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety" in promotion_runbook
    assert "Import Fill Plan" in promotion_runbook
    assert (
        "classes: person, safety_harness, safety_lanyard; mapping starter: "
        "person=person; safety_harness=harness; missing from suggestion: safety_lanyard; "
        "nonzero counts: person, safety_harness, safety_lanyard"
    ) in promotion_runbook
    assert "roboflow_work_at_height_safety__harness_required.review_packet.md" in promotion_runbook
    assert "roboflow_work_at_height_safety__harness_required.review_evidence.yaml" in promotion_runbook
    assert "no_acceptable_pretrained_shortcut_found" in promotion_runbook
    assert "qualcomm_ppe_detection_hf" in promotion_runbook
    assert "kaggle_yolov8_ppe_apron_notebook" in promotion_runbook
    assert "scripts/apron_harness_candidate_doctor.py" in promotion_runbook
    assert "Current model-registry handoff:" in promotion_runbook
    assert "Registry status: `planned_no_candidate`" in promotion_runbook
    assert "Model definition valid: `True`" in promotion_runbook
    assert "Destination exists: `False`" in promotion_runbook
    assert "Metadata sidecar valid: `False`" in promotion_runbook
    assert "Registry artifact status:" in promotion_runbook
    assert "`model_registry_report`" in promotion_runbook
    assert "`destination_model`" in promotion_runbook
    assert "`registry_metadata`" in promotion_runbook
    assert "planned registry audit is not model registration" in promotion_runbook
    assert "scripts/apron_harness_promotion_doctor.py" in promotion_runbook
    assert "scripts/jetson_benchmark_doctor.py" in promotion_runbook
    assert "factory_missing_apron_active_closed_set" in promotion_runbook
    assert "qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml" in promotion_runbook
    assert (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml validate"
        in promotion_runbook
    )
    assert (
        ".venv/bin/python scripts/safetylens_site.py export --output "
        "qa/video_eval/results/site_config_backups/before_factory_missing_apron_active_closed_set.yaml"
        in promotion_runbook
    )
    assert (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml plan"
        in promotion_runbook
    )
    assert (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml apply --yes"
        in promotion_runbook
    )
    assert (
        ".venv/bin/python scripts/video_eval.py run --scenario factory_missing_apron_active_closed_set"
        in promotion_runbook
    )
    assert (
        ".venv/bin/python scripts/safetylens_site.py --config "
        "qa/video_eval/results/site_config_backups/before_factory_missing_apron_active_closed_set.yaml apply --yes"
        in promotion_runbook
    )
    assert "qa/video_eval/focused/<candidate>.yaml" not in promotion_runbook
    assert "qa/video_eval/results/closed_set_candidate/factory_missing_apron_active_closed_set.json" in promotion_runbook
    assert "required_model_plan_ok=`True`" in promotion_runbook
    assert "cli_preflight_ok=`True`" in promotion_runbook
    assert "one_at_a_time_ok=`True`" in promotion_runbook
    assert "required-model preflight" in promotion_runbook
    assert "`ppe_closed_set_candidate` missing" in promotion_runbook
    assert "Candidate runtime evidence valid: `False`" in promotion_runbook
    assert "Candidate runtime result files present: `6/6`" in promotion_runbook
    assert "Candidate runtime valid promotion results: `0/6`" in promotion_runbook
    assert "Missing candidate runtime results: `0`" in promotion_runbook
    assert "Missing-model preflight blocks: `6/6`" in promotion_runbook
    assert "Candidate YAML preflight valid: `True`" in promotion_runbook
    assert "Candidate YAML templates valid: `6/6`" in promotion_runbook
    assert "Candidate YAML CLI validate/plan valid: `6/6`" in promotion_runbook
    assert "Candidate runtime result status:" in promotion_runbook
    assert "Required Evidence" in promotion_runbook
    assert "Observed Evidence" in promotion_runbook
    assert "fresh live-view screenshot is captured" in promotion_runbook
    assert "delivery_summary proves in_app/browser_sound delivery" in promotion_runbook
    assert "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window" in promotion_runbook
    assert "yaml_logs=3" in promotion_runbook
    assert "yaml_ok=True" in promotion_runbook
    assert "yaml_success=3/3" in promotion_runbook
    assert "preflight_ok=False" in promotion_runbook
    assert "factory_missing_apron_active_closed_set" in promotion_runbook
    assert "blocked_missing_required_model_preflight" in promotion_runbook
    assert "--capability apron_required" in promotion_runbook
    assert "--capability harness_required" in promotion_runbook
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pilot_gate_passed"] is True
    assert report["production_gate_passed"] is False
    assert report["pretrained_shortcut_review"]["status"] == "no_acceptable_pretrained_shortcut_found"
    assert {
        candidate["source_ref"]
        for candidate in report["pretrained_shortcut_review"]["candidates"]
    } >= {
        "qualcomm_ppe_detection_hf",
        "github_safety_detection_yolov8",
        "github_ppe_detection_yolo_vinayak",
        "kaggle_yolov8_ppe_apron_notebook",
    }
    assert report["closed_set_handoff"]["capture_work_order"]["generated"] is True
    assert report["closed_set_handoff"]["capture_work_order"]["exists"] is True
    assert len(report["closed_set_handoff"]["capture_work_order"]["sha256"]) == 64
    assert (
        report["closed_set_handoff"]["capture_work_order"]["source_manifest_sha256"]
        == report["closed_set_handoff"]["capture_manifest_sha256"]
    )
    assert report["closed_set_handoff"]["capture_matrix_csv"]["generated"] is True
    assert report["closed_set_handoff"]["capture_matrix_csv"]["exists"] is True
    assert report["closed_set_handoff"]["capture_matrix_csv"]["row_count"] == 21
    assert report["closed_set_handoff"]["capture_matrix_manifest"]["generated"] is True
    assert report["closed_set_handoff"]["capture_matrix_manifest"]["exists"] is True
    assert report["closed_set_handoff"]["capture_matrix_manifest"]["row_count"] == 21
    assert len(report["closed_set_handoff"]["capture_matrix_manifest"]["sha256"]) == 64
    assert (
        report["closed_set_handoff"]["capture_matrix_manifest"]["matrix_csv_sha256"]
        == report["closed_set_handoff"]["capture_matrix_csv"]["sha256"]
    )
    assert report["closed_set_handoff"]["capture_matrix_sidecar_validation"]["valid"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_csv"]["generated"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_csv"]["exists"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_csv"]["row_count"] == 21
    assert report["closed_set_handoff"]["production_capture_matrix_manifest"]["generated"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_manifest"]["exists"] is True
    assert report["closed_set_handoff"]["production_capture_matrix_manifest"]["row_count"] == 21
    assert len(report["closed_set_handoff"]["production_capture_matrix_manifest"]["sha256"]) == 64
    assert (
        report["closed_set_handoff"]["production_capture_matrix_manifest"]["matrix_csv_sha256"]
        == report["closed_set_handoff"]["production_capture_matrix_csv"]["sha256"]
    )
    assert report["closed_set_handoff"]["production_capture_matrix_sidecar_validation"]["valid"] is True
    assert report["closed_set_handoff"]["training_capture_preflight"]["capture_matrix_manifest"]["valid"] is True
    assert report["closed_set_handoff"]["training_dataset_yaml_handoff"]["generated"] is True
    assert report["closed_set_handoff"]["training_dataset_yaml_handoff"]["exists"] is True
    assert report["closed_set_handoff"]["training_dataset_yaml_handoff"]["dataset_root"] == (
        "/mnt/cleared/apron_harness_ppe"
    )
    assert report["closed_set_handoff"]["production_training_dataset_yaml"] == str(training_dataset_yaml_path)
    assert report["closed_set_handoff"]["production_training_dataset_provenance"]["source_manifest_sha256"] == (
        report["closed_set_handoff"]["capture_manifest_sha256"]
    )
    assert report["closed_set_handoff"]["training_dataset_provenance_status"]["valid"] is True
    assert "closed_set_training_dataset_provenance_has_errors" not in report["production_blockers"]
    assert report["closed_set_candidate_yaml_templates"]["valid"] is True
    assert report["closed_set_candidate_runtime_evidence"]["valid"] is False
    assert report["closed_set_candidate_runtime_evidence"]["present_result_count"] == 6
    assert report["closed_set_candidate_runtime_evidence"]["missing_result_count"] == 0
    assert report["closed_set_candidate_runtime_evidence"]["preflight_blocked_missing_model_count"] == 6
    candidate_runtime_rows = report["closed_set_candidate_runtime_evidence"]["results"]
    active_candidate = next(
        row
        for row in candidate_runtime_rows
        if row["scenario_id"] == "factory_missing_apron_active_closed_set"
    )
    assert "fresh live-view screenshot is captured" in active_candidate["required_evidence"]
    assert (
        "delivery_summary proves in_app/browser_sound delivery succeeded or was explicitly simulated"
        in active_candidate["required_evidence"]
    )
    assert active_candidate["observed_evidence"]["yaml_command_log_count"] == 3
    assert active_candidate["observed_evidence"]["yaml_successful_command_count"] == 3
    assert active_candidate["observed_evidence"]["yaml_commands_ok"] is True
    assert set(active_candidate["observed_evidence"]["yaml_command_names"]) == {"validate", "plan", "apply"}
    assert active_candidate["observed_evidence"]["model_preflight_ok"] is False
    suppression_candidate = next(
        row
        for row in candidate_runtime_rows
        if row["scenario_id"] == "factory_harness_detector_window_suppression_closed_set"
    )
    assert (
        "ppe_closed_set_candidate model_invocations == 0 during the inactive capability window"
        in suppression_candidate["required_evidence"]
    )
    training_plan_preflight = report["closed_set_handoff"]["production_training_plan_preflight"]
    assert training_plan_preflight["checked"] is True
    assert training_plan_preflight["ok"] is False
    assert "capture preflight gate failed" in training_plan_preflight["error"]
    assert "closed_set_production_training_plan_preflight_failed" in report["production_blockers"]
    assert report["promotion_runbook"]["generated"] is True
    assert report["promotion_runbook"]["exists"] is True
    assert report["promotion_runbook"]["path"] == str(promotion_runbook_path)
    assert len(report["promotion_runbook"]["sha256"]) == 64
    assert report["candidate_runtime_runbook"]["generated"] is True
    assert report["candidate_runtime_runbook"]["exists"] is True
    assert report["candidate_runtime_runbook"]["path"] == str(candidate_runtime_runbook_path)
    assert len(report["candidate_runtime_runbook"]["sha256"]) == 64
    assert report["closed_set_handoff"]["production_capture_deficit"]["required_per_class"] == 1000
    assert report["closed_set_handoff"]["production_capture_deficit"]["total_missing_label_annotations"] == 4000
    assert len(report["closed_set_handoff"]["capture_matrix_csv"]["sha256"]) == 64
    assert (
        report["closed_set_handoff"]["capture_matrix_csv"]["source_manifest_sha256"]
        == report["closed_set_handoff"]["capture_manifest_sha256"]
    )
    progress = report["closed_set_handoff"]["capture_matrix_progress"]
    assert progress["gate_passed"] is False
    assert progress["row_count"] == 21
    assert progress["ready_rows"] == 0
    assert progress["target_labeled_examples"] == 720
    assert progress["missing_labeled_examples"] == 720
    assert progress["unapproved_rows"] == 21
    assert progress["unsafe_storage_rows"] == 21
    reconciliation = progress["manifest_reconciliation"]
    assert reconciliation["checked"] is True
    assert reconciliation["gate_passed"] is False
    assert reconciliation["required_labeled_images_per_class"]["person"] == 720
    assert reconciliation["missing_manifest_counts"]["person"]["missing"] == 720
    assert "closed_set_capture_matrix_not_complete_or_approved" in report["production_blockers"]
    assert "closed_set_capture_matrix_manifest_counts_do_not_match" in report["production_blockers"]
    production_progress = report["closed_set_handoff"]["production_capture_matrix_progress"]
    assert production_progress["gate_passed"] is False
    assert production_progress["row_count"] == 21
    assert production_progress["target_labeled_examples"] == 2404
    assert production_progress["manifest_reconciliation"]["missing_manifest_counts"]["person"]["missing"] == 2404
    progress_summary = report["closed_set_capture_progress_summary"]
    assert progress_summary["pilot"]["gate_passed"] is False
    assert progress_summary["pilot"]["row_count"] == 21
    assert progress_summary["pilot"]["ready_rows"] == 0
    assert progress_summary["pilot"]["target_labeled_examples"] == 720
    assert progress_summary["pilot"]["missing_labeled_examples"] == 720
    assert progress_summary["pilot"]["manifest_reconciliation_checked"] is True
    assert progress_summary["pilot"]["manifest_reconciliation_gate_passed"] is False
    assert progress_summary["pilot"]["missing_manifest_counts"]["person"]["missing"] == 720
    assert progress_summary["pilot"]["csv_generated"] is True
    assert progress_summary["pilot"]["manifest_generated"] is True
    assert len(progress_summary["pilot"]["csv_sha256"]) == 64
    assert len(progress_summary["pilot"]["manifest_sha256"]) == 64
    assert progress_summary["production"]["gate_passed"] is False
    assert progress_summary["production"]["row_count"] == 21
    assert progress_summary["production"]["target_labeled_examples"] == 2404
    assert progress_summary["production"]["missing_manifest_counts"]["person"]["missing"] == 2404
    assert progress_summary["production"]["csv_generated"] is True
    assert progress_summary["production"]["manifest_generated"] is True
    assert len(progress_summary["production"]["csv_sha256"]) == 64
    assert len(progress_summary["production"]["manifest_sha256"]) == 64
    assert "closed_set_production_capture_matrix_not_complete_or_approved" in report["production_blockers"]
    assert "closed_set_production_capture_matrix_manifest_counts_do_not_match" in report["production_blockers"]
