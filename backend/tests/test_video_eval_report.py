"""Tests for sales-readiness report helpers."""

from pathlib import Path
import importlib.util

import yaml


ROOT = Path(__file__).resolve().parents[2]
VIDEO_EVAL_PATH = ROOT / "scripts" / "video_eval.py"


def _load_video_eval():
    spec = importlib.util.spec_from_file_location("video_eval", VIDEO_EVAL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_preflight_status_flags_missing_candidate_model():
    video_eval = _load_video_eval()

    camera = {
        "id": "eval_factory_missing_apron_active_closed_set",
        "execution_plan": {
            "required_model_keys": ["coco_primary", "ppe_closed_set_candidate"],
        },
    }
    health = {
        "models": [
            {"model_key": "coco_primary", "status": "ready", "is_ready": True},
            {
                "model_key": "ppe_closed_set_candidate",
                "status": "not_downloaded",
                "is_ready": False,
                "active_path": None,
            },
        ],
    }

    preflight = video_eval.model_preflight_status(camera, health)

    assert preflight["checked"] is True
    assert preflight["ok"] is False
    assert preflight["reason"] == "required_models_not_ready"
    assert preflight["required_model_keys"] == ["coco_primary", "ppe_closed_set_candidate"]
    assert preflight["missing_model_keys"] == ["ppe_closed_set_candidate"]
    assert preflight["models"]["ppe_closed_set_candidate"]["status"] == "not_downloaded"


def test_model_preflight_status_accepts_ready_models():
    video_eval = _load_video_eval()

    camera = {
        "id": "eval_factory_missing_apron_active",
        "executionPlan": {
            "requiredModelKeys": ["coco_primary", "ppe_specialist"],
        },
    }
    health = {
        "models": [
            {"model_key": "coco_primary", "status": "ready"},
            {"modelKey": "ppe_specialist", "status": "ready", "isReady": True},
        ],
    }

    preflight = video_eval.model_preflight_status(camera, health)

    assert preflight["checked"] is True
    assert preflight["ok"] is True
    assert preflight["reason"] == "ok"
    assert preflight["missing_model_keys"] == []


def test_scenario_result_dir_can_route_candidate_results_to_subdir(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"

    result_dir = video_eval.scenario_result_dir(
        artifact_root,
        {"runtime": {"result_subdir": "closed_set_candidate"}},
    )

    assert result_dir == artifact_root / "results" / "closed_set_candidate"


def test_scenario_result_dir_rejects_unsafe_subdir(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"

    try:
        video_eval.scenario_result_dir(
            artifact_root,
            {"runtime": {"result_subdir": "../outside"}},
        )
    except ValueError as exc:
        assert "result_subdir" in str(exc)
    else:
        raise AssertionError("unsafe result_subdir should raise ValueError")


def test_run_scenario_blocks_missing_video_without_unbound_video_path(tmp_path):
    video_eval = _load_video_eval()
    video_eval.ROOT = tmp_path
    artifact_root = tmp_path / "qa" / "video_eval"
    manifest_path = tmp_path / "manifest.yaml"
    config_path = tmp_path / "site.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: missing_video_scenario
    config_path: {config_path}
    local_video: test-videos/does-not-exist.mp4
    camera_id: eval_missing_video
    source: {{type: fixture}}
    expected: {{}}
""",
        encoding="utf-8",
    )

    result = video_eval.run_scenario(manifest_path, "missing_video_scenario")

    assert result["status"] == "blocked"
    assert any("Video file not found" in error for error in result["blocking_errors"])
    result_path = artifact_root / "results" / "missing_video_scenario.json"
    assert result_path.exists()


def test_model_invocation_expectation_requires_schedule_telemetry():
    video_eval = _load_video_eval()

    evidence = {
        "expected_analytics": [
            {
                "type": "model_invocation",
                "model_keys": ["ppe_closed_set_candidate"],
                "min_model_invocations": 1,
            }
        ],
        "analytics_summary": {
            "schedule": {
                "ok": True,
                "model_invocations": {"ppe_closed_set_candidate": 1},
            }
        },
    }

    assert video_eval.analytics_expectations_pass(evidence) is True


def test_model_invocation_expectation_fails_without_required_invocation():
    video_eval = _load_video_eval()

    evidence = {
        "expected_analytics": [
            {
                "type": "model_invocation",
                "model_keys": ["ppe_closed_set_candidate"],
                "min_model_invocations": 1,
            }
        ],
        "analytics_summary": {
            "schedule": {
                "ok": True,
                "model_invocations": {"ppe_closed_set_candidate": 0},
            }
        },
    }

    assert video_eval.analytics_expectations_pass(evidence) is False


def test_route_obstruction_expectation_can_assert_negative_control():
    video_eval = _load_video_eval()

    evidence = {
        "expected_analytics": [
            {
                "type": "route_obstruction",
                "max_count": 0,
                "max_zone_count": 0,
                "obstruction_active": False,
            }
        ],
        "analytics_summary": {
            "obstruction": {
                "ok": True,
                "object_count": 0,
                "max_zone_count": 0,
                "obstruction_active": False,
            }
        },
    }

    assert video_eval.analytics_expectations_pass(evidence) is True

    evidence["analytics_summary"]["obstruction"]["object_count"] = 1
    assert video_eval.analytics_expectations_pass(evidence) is False


def test_unknown_analytics_expectation_type_fails_closed():
    video_eval = _load_video_eval()

    expectations = [{"type": "candidate_model_actually_ran"}]
    evidence = {
        "expected_analytics": expectations,
        "analytics_summary": {
            "schedule": {
                "ok": True,
                "model_invocations": {"ppe_closed_set_candidate": 1},
            }
        },
    }

    assert video_eval.unsupported_analytics_expectation_types(expectations) == [
        "candidate_model_actually_ran"
    ]
    assert video_eval.analytics_expectations_pass(evidence) is False


def test_manifest_analytics_expectation_types_are_supported():
    video_eval = _load_video_eval()
    manifest = yaml.safe_load((ROOT / "qa" / "video_eval" / "manifest.yaml").read_text())
    expectations = [
        analytic
        for scenario in manifest.get("scenarios", [])
        for analytic in (scenario.get("expected") or {}).get("analytics", [])
        if isinstance(analytic, dict)
    ]

    assert video_eval.unsupported_analytics_expectation_types(expectations) == []


def test_model_pack_device_gate_formats_cpu_fallback_boundary():
    video_eval = _load_video_eval()

    lines = video_eval.format_model_pack_device_gate(
        {
            "generated_at": "2026-06-20T16:38:49+00:00",
            "host": "edge-mac",
            "platform": {
                "system": "Darwin",
                "machine": "arm64",
                "python": "3.12.9",
                "macos": {
                    "checked": True,
                    "product_version": "26.2",
                    "build_version": "25C56",
                },
            },
            "torch": {
                "version": "2.10.0",
                "mps_built": True,
                "mps_available": False,
                "cuda_available": False,
            },
            "mps_acceptance_gate": "mps_unavailable_cpu_fallback_only",
            "local_performance_acceptance_gate": "blocked_cpu_fallback_only",
            "model_artifact_layout_gate": "blocked_legacy_artifacts_present",
            "artifact_layout": {
                "gate": "blocked_legacy_artifacts_present",
                "unexpected_paths": ["yoloe-26n-seg.pt"],
            },
            "packs": {
                "base_3cam": {
                    "local_device_satisfied": True,
                    "local_functional_satisfied": True,
                    "local_performance_gate_satisfied": False,
                    "local_device_reasons": ["cpu_fallback_only"],
                }
            },
        }
    )

    rendered = "\n".join(lines)
    assert "mps_unavailable_cpu_fallback_only" in rendered
    assert "blocked_cpu_fallback_only" in rendered
    assert "macOS `26.2` build `25C56`" in rendered
    assert "base_3cam:wiring-only(cpu_fallback_only)" in rendered
    assert "Model artifact layout gate: `blocked_legacy_artifacts_present`" in rendered
    assert "unexpected legacy artifacts=`1`" in rendered
    assert "`yoloe-26n-seg.pt`" in rendered
    assert "not Apple Silicon MPS performance" in rendered


def test_model_pack_device_gate_qualifies_factory_ppe_local_performance_when_production_blocked():
    video_eval = _load_video_eval()

    lines = video_eval.format_model_pack_device_gate(
        {
            "generated_at": "2026-06-23T13:02:59+00:00",
            "host": "edge-mac",
            "platform": {"system": "Darwin", "machine": "arm64", "python": "3.12.9"},
            "torch": {"version": "2.10.0", "mps_built": True, "mps_available": True},
            "mps_acceptance_gate": "mps_available",
            "local_performance_acceptance_gate": "pass",
            "packs": {
                "factory_ppe_3cam": {
                    "local_functional_satisfied": True,
                    "local_performance_gate_satisfied": True,
                    "local_device_reasons": ["mps_available"],
                },
            },
        },
        {
            "production_gate_passed": False,
            "sales_status": "pilot_ready_not_production_compliance",
        },
    )

    rendered = "\n".join(lines)
    assert "factory_ppe_3cam:perf-ok(mps_available)" in rendered
    assert "Factory PPE device qualifier:" in rendered
    assert "factory_ppe_3cam local MPS/device status is `perf-ok`" in rendered
    assert "apron/harness production sales status is `pilot_ready_not_production_compliance`" in rendered


def test_model_pack_device_probe_is_not_treated_as_scenario_result(tmp_path, monkeypatch):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: person_presence_active
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person detection"]
""",
        encoding="utf-8",
    )
    (result_dir / "model_pack_device_probe.json").write_text(
        '{"mps_acceptance_gate": "mps_unavailable_cpu_fallback_only"}',
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "clip.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, _claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    assert "`person_presence_active`" in report
    assert "`None`" not in report
    assert "Model Pack Device Gate" in report


def test_report_includes_capability_window_schedule_shape(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    config_path = tmp_path / "focused.yaml"
    config_path.write_text(
        """
cameras:
  eval_factory_apron_detector_window_suppression:
    capability_windows:
      apron_required:
        active: false
        windows:
          - days: [mon, tue, wed, thu, fri]
            from: "09:00"
            to: "17:30"
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: factory_apron_detector_window_suppression
    config_path: {config_path}
    camera_id: eval_factory_apron_detector_window_suppression
    source: {{type: fixture, provider: test}}
    sales_claims: ["Apron detector schedule"]
""",
        encoding="utf-8",
    )
    (result_dir / "factory_apron_detector_window_suppression.json").write_text(
        '{"scenario_id": "factory_apron_detector_window_suppression", "status": "ready_to_sell", "video": "apron.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    expected = "schedule_config=capability_windows(apron_required:inactive:daily_weekly:mon/tue/wed/thu/fri:09:00-17:30)"
    assert expected in report
    assert expected in claims


def test_report_skips_out_of_scope_verticals_and_records_skipped_brochure(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
coverage_boundaries:
  skipped_final_send_brochures:
    - code: RL-M
      file: marketing/suite/final-send/RakshakLens_RL-M_Hospitals_Medicare_Datasheet.pdf
      reason: Skipped for this validation cycle.
  skipped_verticals: [hospital, hospitals, rl_m]
  skipped_scenario_ids: [hospital_person_down]
  skipped_model_packs: [pose_fall_3cam]
scenarios:
  - id: person_presence_active
    vertical: education
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person detection claim"]
  - id: hospital_person_down
    vertical: hospital
    source: {{type: fixture, provider: test}}
    sales_claims: ["Hospital person-down claim"]
  - id: rl_m_wandering
    vertical: rl_m
    source: {{type: fixture, provider: test}}
    sales_claims: ["RL-M wandering claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "person.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )
    (result_dir / "hospital_person_down.json").write_text(
        '{"scenario_id": "hospital_person_down", "status": "ready_to_sell", "video": "hospital.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )
    (result_dir / "rl_m_wandering.json").write_text(
        '{"scenario_id": "rl_m_wandering", "status": "ready_to_sell", "video": "rl-m.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )
    (result_dir / "model_pack_device_probe.json").write_text(
        """
{
  "generated_at": "2026-06-23T00:00:00+00:00",
  "host": "test-host",
  "platform": {"system": "Darwin", "machine": "arm64", "python": "3.12.9"},
  "torch": {"version": "2.10.0", "mps_built": true, "mps_available": true},
  "mps_acceptance_gate": "pass",
  "local_performance_acceptance_gate": "pass",
  "packs": {
    "base_3cam": {"local_functional_satisfied": true, "local_performance_gate_satisfied": true, "local_device_reasons": ["mps_available"]},
    "pose_fall_3cam": {"local_functional_satisfied": true, "local_performance_gate_satisfied": true, "local_device_reasons": ["mps_available"]}
  }
}
""",
        encoding="utf-8",
    )
    (result_dir / "model_pack_evidence_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-23T00:00:00+00:00",
  "stats": {"pack_count": 2, "unique_scenario_count": 2, "ready_result_count": 2},
  "packs": {
    "base_3cam": {"ok": true, "scenario_count": 1},
    "pose_fall_3cam": {"ok": true, "scenario_count": 1}
  }
}
""",
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    assert "`person_presence_active`" in report
    assert "`hospital_person_down`" not in report
    assert "`rl_m_wandering`" not in report
    assert "## Skipped Brochures" in report
    assert "`RL-M`" in report
    assert "base_3cam:perf-ok" in report
    assert "base_3cam:ok" in report
    assert "pose_fall_3cam:perf-ok" not in report
    assert "pose_fall_3cam:ok" not in report
    assert "Skipped model packs for current sales scope: `pose_fall_3cam`" in report
    assert "Hospital person-down claim" not in claims
    assert "RL-M wandering claim" not in claims
    assert "Person detection claim" in claims


def test_model_pack_evidence_gate_formats_saved_evidence_summary():
    video_eval = _load_video_eval()

    lines = video_eval.format_model_pack_evidence_gate(
        {
            "ok": True,
            "generated_at": "2026-06-21T08:00:00+00:00",
            "stats": {
                "pack_count": 5,
                "unique_scenario_count": 57,
                "unique_config_count": 55,
                "ready_result_count": 57,
                "yaml_apply_skipped_count": 2,
                "log_evidence_check_count": 57,
                "active_window_check_count": 33,
                "delivery_check_count": 91,
                "detector_suppression_check_count": 22,
            },
            "packs": {
                "base_3cam": {"ok": True, "scenario_count": 18},
                "factory_ppe_3cam": {"ok": True, "scenario_count": 30},
            },
        },
        {
            "production_gate_passed": False,
            "sales_status": "pilot_ready_not_production_compliance",
        },
    )

    rendered = "\n".join(lines)
    assert "model_pack_evidence_doctor.json" in rendered
    assert "Acceptance gate: `pass`" in rendered
    assert "yaml_apply_skipped=`2`" in rendered
    assert "log_checks=`57`" in rendered
    assert "active_window_checks=`33`" in rendered
    assert "delivery_checks=`91`" in rendered
    assert "detector_window_checks=`22`" in rendered
    assert "factory_ppe_3cam:ok(30 scenarios)" in rendered
    assert "Factory PPE production qualifier:" in rendered
    assert "factory_ppe_3cam local evidence is `ok`" in rendered
    assert "apron/harness production sales status is `pilot_ready_not_production_compliance`" in rendered


def test_model_pack_evidence_doctor_is_not_treated_as_scenario_result(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: person_presence_active
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person detection"]
""",
        encoding="utf-8",
    )
    (result_dir / "model_pack_evidence_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-21T08:00:00+00:00",
  "stats": {
    "pack_count": 5,
    "unique_scenario_count": 57,
    "unique_config_count": 55,
    "ready_result_count": 57,
    "detector_suppression_check_count": 22
  },
  "packs": {
    "base_3cam": {"ok": true, "scenario_count": 18}
  }
}
""",
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "clip.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, _claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    assert "`person_presence_active`" in report
    assert "`None`" not in report
    assert "Model Pack Evidence Gate" in report
    assert "ready_results=`57`" in report


def test_apron_harness_readiness_gate_formats_blocked_production_summary():
    video_eval = _load_video_eval()

    lines = video_eval.format_apron_harness_readiness_gate(
        {
            "generated_at": "2026-06-21T09:00:00+00:00",
            "pilot_gate_passed": True,
            "production_gate_passed": False,
            "sales_status": "pilot_ready_not_production_compliance",
            "sourcing_status": "public_seed_sources_found_unapproved_insufficient_for_production",
            "sourcing_candidate_count": 5,
            "seed_source_review_training_usable_count": 0,
            "seed_source_review_bundle_ok": True,
            "seed_source_review_bundle": {
                "ok": True,
                "artifact_count": 59,
            },
            "seed_import_manifest_included_count": 0,
            "seed_import_manifest_approved_count": 0,
            "seed_import_export_preflight_summary": {
                "required_manifest_field": "raw_export_local_path",
                "blocked_reason": "no_seed_imports_included_for_training",
                "blocker_count": 2,
                "top_blockers": [
                    "imports[0].roboflow_work_at_height_safety: include_in_training=false",
                    "no seed imports are approved for training",
                ],
                "checks": [
                    "local_reviewed_export_zip_exists",
                    "local_export_sha256_matches_raw_export_sha256",
                    "required_local_class_label_file_counts_meet_expected_counts",
                    "review_packet_path_and_sha256_match_seed_source_review",
                ],
                "preflight_checked_count": 0,
                "preflight_approved_count": 0,
                "missing_raw_export_local_path_count": 0,
                "review_artifact_checked_count": 0,
                "review_artifact_error_count": 0,
            },
            "seed_import_fill_contract_summary": {
                "available": True,
                "required_before_include_in_training_count": 12,
                "forbidden_until_approved": [
                    "include_in_training=true",
                    "approval_status=approved_for_training",
                ],
                "validation_commands": [
                    ".venv/bin/python scripts/apron_harness_seed_source_doctor.py --validate-import-manifest /path/to/filled.yaml",
                    ".venv/bin/python scripts/apron_harness_readiness_doctor.py --seed-import-manifest /path/to/filled.yaml",
                ],
            },
            "seed_source_next_review_queue": [
                {
                    "review_priority": 10,
                    "source_ref": "roboflow_work_at_height_safety",
                    "capability": "harness_required",
                    "source_url": "https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety",
                    "license_note": "Roboflow Universe page lists CC BY 4.0.",
                    "seed_import_fill_plan": {
                        "required_local_classes": ["person", "safety_harness", "safety_lanyard"],
                        "missing_required_classes_from_suggestion": ["safety_lanyard"],
                        "expected_count_classes_that_must_be_nonzero": [
                            "person",
                            "safety_harness",
                            "safety_lanyard",
                        ],
                    },
                    "review_packet_path": (
                        "qa/video_eval/results/apron_harness_seed_source_review_packets/"
                        "roboflow_work_at_height_safety__harness_required.review_packet.md"
                    ),
                },
                {
                    "review_priority": 15,
                    "source_ref": "roboflow_safety_food_system",
                    "capability": "apron_required",
                    "seed_import_fill_plan": {
                        "required_local_classes": ["apron", "person"],
                        "missing_required_classes_from_suggestion": ["person"],
                        "expected_count_classes_that_must_be_nonzero": ["apron", "person"],
                    },
                    "review_packet_path": (
                        "qa/video_eval/results/apron_harness_seed_source_review_packets/"
                        "roboflow_safety_food_system__apron_required.review_packet.md"
                    ),
                },
            ],
            "seed_source_minimum_approval_path": {
                "checked": True,
                "coverage_gap_count": 0,
                "training_usable_count": 0,
                "minimum_review_source_refs": [
                    "roboflow_safety_food_system",
                    "roboflow_work_at_height_safety",
                ],
                "capabilities": {
                    "apron_required": {
                        "selected_sources": [
                            {
                                "source_ref": "roboflow_safety_food_system",
                                "mapped_local_classes": ["apron"],
                                "missing_local_classes": ["person"],
                                "training_usable": False,
                            },
                        ],
                    },
                    "harness_required": {
                        "selected_sources": [
                            {
                                "source_ref": "roboflow_work_at_height_safety",
                                "mapped_local_classes": ["person", "safety_harness"],
                                "missing_local_classes": ["safety_lanyard"],
                                "training_usable": False,
                            },
                        ],
                    },
                },
                "evidence_boundary": "agent-selected minimum path is not approval",
            },
            "minimum_seed_import_manifest_template_summary": {
                "available": True,
                "path": "qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml",
                "template_scope": "minimum_priority_coverage_sources",
                "selected_source_refs": [
                    "roboflow_safety_food_system",
                    "roboflow_work_at_height_safety",
                ],
                "import_count": 2,
                "enabled_import_count": 0,
            },
            "minimum_seed_import_manifest_template_consistency": {
                "checked": True,
                "valid": True,
                "source_refs_match": True,
                "enabled_import_count": 0,
            },
            "optional_gate_status": {
                "model_registry": "not_registered:planned_no_candidate",
                "seed_source_review": "blocked:qa/video_eval/results/apron_harness_seed_source_review.json",
                "seed_source_review_bundle": "pass:qa/video_eval/results/apron_harness_source_review_bundle.json",
                "seed_import_manifest": "blocked:qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml",
            },
            "production_blockers": [
                "missing_or_failed_factory_ppe_jetson_full_gate",
                "closed_set_public_seed_sources_not_curated_or_approved",
            ],
            "production_blocker_count": 2,
            "jetson_template_handoff": {
                "checked": True,
                "status": "ready_for_candidate_identity",
                "template_count": 2,
                "valid_template_contract_count": 2,
                "identity_stamped_count": 0,
            },
            "model_registry_handoff": {
                "checked": True,
                "status": "not_registered:planned_no_candidate",
                "registry_status": "planned_no_candidate",
                "model_definition_valid": True,
                "destination_exists": False,
                "metadata_valid": False,
            },
            "next_actions": [
                {
                    "priority": 1,
                    "id": "approve_or_capture_training_data",
                    "title": "Approve public seed sources or collect controlled apron/harness capture data",
                    "evidence_contract_summary": [
                        "human/legal approval evidence is required before public seed data can enter training",
                    ],
                },
                {
                    "priority": 2,
                    "id": "prove_edge_gate",
                    "title": "Run factory PPE Jetson raw benchmark and three-camera soak on the exact candidate artifact",
                    "artifacts": [
                        "qa/video_eval/results/factory_ppe_jetson_gate.json",
                        "/path/to/cleared/factory_ppe_raw_benchmark.json",
                        "/path/to/cleared/factory_ppe_3cam_soak.json",
                    ],
                    "evidence_contract_summary": [
                        "same candidate_report_sha256 and selected-export SHA across raw benchmark, soak, promotion, and registry reports",
                    ],
                },
                {
                    "priority": 3,
                    "id": "train_and_register_candidate",
                    "title": "Train YOLO26 closed-set apron/harness candidate and register the verified ONNX export",
                    "artifacts": [
                        "/path/to/cleared/apron_harness_yolo26n_result.json",
                        "qa/video_eval/results/apron_harness_candidate_report.json",
                        "qa/video_eval/results/apron_harness_model_registry_report.json",
                        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx",
                        "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json",
                        "qa/video_eval/results/apron_closed_set_promotion_report.json",
                        "qa/video_eval/results/harness_closed_set_promotion_report.json",
                    ],
                    "evidence_contract_summary": [
                        "train/export only from reviewed production data with passed capture preflight",
                    ],
                },
            ],
            "closed_set_candidate_yaml_templates": {
                "valid": True,
                "template_count": 6,
                "valid_template_count": 6,
                "templates": [
                    {
                        "scenario_id": "factory_missing_apron_active_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                    {
                        "scenario_id": "factory_apron_false_positive_guard_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                    {
                        "scenario_id": "factory_apron_detector_window_suppression_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                    {
                        "scenario_id": "factory_missing_harness_active_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                    {
                        "scenario_id": "factory_harness_false_positive_guard_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                    {
                        "scenario_id": "factory_harness_detector_window_suppression_closed_set",
                        "required_model_plan_ok": True,
                        "cli_preflight_ok": True,
                        "one_at_a_time_ok": True,
                    },
                ],
            },
            "closed_set_candidate_runtime_evidence": {
                "valid": False,
                "result_count": 6,
                "present_result_count": 6,
                "valid_result_count": 0,
                "preflight_blocked_missing_model_count": 6,
                "results": [
                    {
                        "scenario_id": "factory_missing_apron_active_closed_set",
                        "errors": ["blocked_missing_required_model_preflight"],
                    },
                    {
                        "scenario_id": "factory_harness_detector_window_suppression_closed_set",
                        "errors": ["blocked_missing_required_model_preflight"],
                    },
                ],
            },
            "closed_set_handoff": {
                "dataset_schema_ok": True,
                "training_dry_run_status": "ready_to_train",
                "training_model": "yolo26n.pt",
                "selected_device": "mps",
                "training_torch_status": {
                    "installed": True,
                    "version": "2.10.0",
                    "mps_built": True,
                    "mps_available": True,
                    "mps_probe_ok": True,
                    "mps_runtime_error": None,
                    "cuda_available": False,
                },
                "training_capture_preflight": {
                    "required": False,
                    "checked": False,
                    "gate_passed": None,
                },
                "production_training_plan_preflight": {
                    "checked": True,
                    "ok": False,
                    "error": "capture preflight gate failed; complete and approve the capture matrix before training",
                    "inputs": {
                        "data": "qa/video_eval/results/apron_harness_training_dataset.yaml",
                        "capture_manifest": "qa/video_eval/datasets/apron_harness_capture_manifest.template.yaml",
                        "capture_matrix_csv": "qa/video_eval/results/apron_harness_production_capture_matrix.csv",
                    },
                },
                "missing_label_minimums": {
                    "person": {"current": 0},
                    "apron": {"current": 0},
                    "safety_harness": {"current": 0},
                    "safety_lanyard": {"current": 0},
                },
                "capture_deficit": {
                    "total_missing_label_annotations": 1200,
                    "recommended_label_review_rows": 720,
                    "coverage_deficit_count": 0,
                    "next_capture_batches": [
                        {
                            "batch_id": "apron_required_closed_set_capture",
                            "capture_matrix": [{"row_id": "a1"}, {"row_id": "a2"}],
                        },
                        {
                            "batch_id": "harness_required_closed_set_capture",
                            "capture_matrix": [{"row_id": "h1"}],
                        },
                    ],
                },
                "capture_matrix_csv": {
                    "path": "qa/video_eval/results/apron_harness_capture_matrix.csv",
                    "generated": True,
                    "row_count": 21,
                },
                "production_capture_matrix_sidecar_validation": {
                    "path": "qa/video_eval/results/apron_harness_production_capture_matrix.csv.manifest.json",
                    "checked": True,
                    "valid": True,
                },
                "label_review_import_sidecar_validation": {
                    "path": "qa/video_eval/results/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json",
                    "checked": True,
                    "valid": False,
                },
                "capture_matrix_progress": {
                    "gate_passed": False,
                    "row_count": 21,
                    "ready_rows": 0,
                    "labeled_examples": 0,
                    "missing_labeled_examples": 720,
                    "unapproved_rows": 21,
                    "rows": [
                        {
                            "row_id": "apron_required_closed_set_capture.positive.denim_apron",
                            "target_capability": "apron_required",
                            "missing_labeled_examples": 60,
                            "blockers": [
                                "missing_labeled_examples=60",
                                "review_status=not_started",
                                "permission=unknown",
                            ],
                        },
                        {
                            "row_id": (
                                "apron_required_closed_set_capture.positive."
                                "kitchen_or_food_service_apron"
                            ),
                            "target_capability": "apron_required",
                            "missing_labeled_examples": 60,
                            "blockers": [
                                "missing_labeled_examples=60",
                                "review_status=not_started",
                                "permission=unknown",
                            ],
                        },
                        {
                            "row_id": "harness_required_closed_set_capture.positive.fall_arrest_harness",
                            "target_capability": "harness_required",
                            "missing_labeled_examples": 60,
                            "blockers": [
                                "missing_labeled_examples=60",
                                "review_status=not_started",
                                "permission=unknown",
                            ],
                        },
                    ],
                    "capabilities": {
                        "apron_required": {
                            "row_count": 11,
                            "ready_rows": 0,
                            "labeled_examples": 0,
                            "missing_labeled_examples": 360,
                        },
                        "harness_required": {
                            "row_count": 10,
                            "ready_rows": 0,
                            "labeled_examples": 0,
                            "missing_labeled_examples": 360,
                        },
                    },
                    "manifest_reconciliation": {
                        "checked": True,
                        "gate_passed": False,
                    },
                },
                "production_capture_matrix_progress": {
                    "gate_passed": False,
                    "row_count": 21,
                    "ready_rows": 0,
                    "labeled_examples": 0,
                    "missing_labeled_examples": 2404,
                    "unapproved_rows": 21,
                    "rows": [
                        {
                            "row_id": "apron_required_closed_set_capture.positive.denim_apron",
                            "target_capability": "apron_required",
                            "missing_labeled_examples": 200,
                            "blockers": [
                                "missing_labeled_examples=200",
                                "review_status=not_started",
                                "permission=unknown",
                            ],
                        },
                        {
                            "row_id": (
                                "harness_required_closed_set_capture.positive."
                                "fall_arrest_harness"
                            ),
                            "target_capability": "harness_required",
                            "missing_labeled_examples": 200,
                            "blockers": [
                                "missing_labeled_examples=200",
                                "review_status=not_started",
                                "permission=unknown",
                            ],
                        },
                    ],
                    "capabilities": {
                        "apron_required": {
                            "row_count": 11,
                            "ready_rows": 0,
                            "labeled_examples": 0,
                            "missing_labeled_examples": 1204,
                        },
                        "harness_required": {
                            "row_count": 10,
                            "ready_rows": 0,
                            "labeled_examples": 0,
                            "missing_labeled_examples": 1200,
                        },
                    },
                    "manifest_reconciliation": {
                        "checked": True,
                        "gate_passed": False,
                    },
                },
            },
            "capabilities": {
                "apron_required": {
                    "ok": True,
                    "scenarios": {
                        "false_positive_guard": {"visible_class_total": 6},
                        "suppression": {"model_invocations": {"ppe_specialist": 0}},
                    },
                },
                "harness_required": {
                    "ok": True,
                    "scenarios": {
                        "false_positive_guard": {"visible_class_total": 11},
                        "suppression": {"model_invocations": {"ppe_specialist": 0}},
                    },
                },
            },
        }
    )

    rendered = "\n".join(lines)
    assert "apron_harness_readiness_doctor.json" in rendered
    assert "Pilot gate: `pass`" in rendered
    assert "Production gate: `blocked`" in rendered
    assert "Production blocker count: `2`" in rendered
    assert "pilot_ready_not_production_compliance" in rendered
    assert "Source status: `public_seed_sources_found_unapproved_insufficient_for_production`" in rendered
    assert "Seed-source review: `blocked:qa/video_eval/results/apron_harness_seed_source_review.json`" in rendered
    assert "Source-review bundle: `pass:qa/video_eval/results/apron_harness_source_review_bundle.json`" in rendered
    assert "hashes_ok=`True`" in rendered
    assert "artifacts=`59`" in rendered
    assert "Next seed-source reviews:" in rendered
    assert "harness_required/roboflow_work_at_height_safety(priority=10" in rendered
    assert "license=Roboflow Universe page lists CC BY 4.0." in rendered
    assert "url=https://universe.roboflow.com/proyecto-prevencion-predictiva/work-at-height-safety" in rendered
    assert "apron_required/roboflow_safety_food_system(priority=15" in rendered
    assert "Seed import fill plans:" in rendered
    assert (
        "roboflow_work_at_height_safety(classes=person, safety_harness, safety_lanyard, "
        "missing_suggestion=safety_lanyard, nonzero_counts=person, safety_harness, safety_lanyard)"
    ) in rendered
    assert (
        "roboflow_safety_food_system(classes=apron, person, "
        "missing_suggestion=person, nonzero_counts=apron, person)"
    ) in rendered
    assert "Minimum seed-source approval path:" in rendered
    assert "sources=`roboflow_safety_food_system, roboflow_work_at_height_safety`" in rendered
    assert (
        "capabilities=`apron_required=roboflow_safety_food_system, "
        "harness_required=roboflow_work_at_height_safety`"
    ) in rendered
    assert "coverage_gaps=`0`" in rendered
    assert "agent-selected minimum path is not approval" in rendered
    assert "Seed-import manifest: `blocked:qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml`" in rendered
    assert "Minimum seed-import template:" in rendered
    assert "path=`qa/video_eval/datasets/apron_harness_minimum_seed_import_manifest.template.yaml`" in rendered
    assert "scope=`minimum_priority_coverage_sources`" in rendered
    assert "sources=`roboflow_safety_food_system, roboflow_work_at_height_safety`" in rendered
    assert "imports=`2`, enabled=`0`, consistency_valid=`True`, refs_match=`True`" in rendered
    assert "Seed import fill contract: required_before_include=`12`" in rendered
    assert "`include_in_training=true`" in rendered
    assert "validation_commands=`2`" in rendered
    assert "Seed export preflight: field=`raw_export_local_path`" in rendered
    assert "blocked_reason=`no_seed_imports_included_for_training`" in rendered
    assert "checked=`0`" in rendered
    assert "missing_local_zip=`0`" in rendered
    assert "review_artifacts_checked=`0`" in rendered
    assert "review_artifact_errors=`0`" in rendered
    assert "Seed export preflight blockers:" in rendered
    assert "imports[0].roboflow_work_at_height_safety: include_in_training=false" in rendered
    assert "no seed imports are approved for training" in rendered
    assert "Seed export proof checks:" in rendered
    assert "`local_reviewed_export_zip_exists`" in rendered
    assert "`local_export_sha256_matches_raw_export_sha256`" in rendered
    assert "`required_local_class_label_file_counts_meet_expected_counts`" in rendered
    assert "`review_packet_path_and_sha256_match_seed_source_review`" in rendered
    assert "Seed export materialization gate: command exit `0` is not enough" in rendered
    assert "the emitted `.seed_export_import.json` sidecar must validate after materialization" in rendered
    assert "Model registry report: `not_registered:planned_no_candidate`" in rendered
    assert (
        "Model registry handoff: status=`not_registered:planned_no_candidate`, "
        "registry_status=`planned_no_candidate`, model_definition_valid=`True`, "
        "destination_exists=`False`, metadata_valid=`False`."
    ) in rendered
    assert "Jetson template handoff: status=`ready_for_candidate_identity`" in rendered
    assert "valid_templates=`2/2`" in rendered
    assert "identity_stamped=`0/2`" in rendered
    assert "Closed-set candidate YAML preflight: valid=`True`, templates=`6/6`" in rendered
    assert "cli_validate_plan=`6/6`" in rendered
    assert "required_model_plan=`6/6`" in rendered
    assert "one_at_a_time=`6/6`" in rendered
    assert "Closed-set candidate runtime evidence: valid=`False`, files_present=`6/6`, valid_results=`0/6`" in rendered
    assert "missing_model_preflight_blocks=`6`" in rendered
    assert "factory_missing_apron_active_closed_set:blocked_missing_required_model_preflight" in rendered
    assert "training_usable=`0`" in rendered
    assert "harness_required:ok(visible=11, ppe_specialist_off_invocations=0)" in rendered
    assert "Closed-set handoff: dataset_schema=pass" in rendered
    assert "training_ready=`dry_run_only`" in rendered
    assert "dry_run=`ready_to_train`" in rendered
    assert "device=`mps`" in rendered
    assert "Local training device gate:" in rendered
    assert "selected=`mps`" in rendered
    assert "torch=`2.10.0`" in rendered
    assert "mps_built=`True`" in rendered
    assert "mps_available=`True`" in rendered
    assert "mps_probe_ok=`True`" in rendered
    assert "Production training preflight:" in rendered
    assert "checked=`True`" in rendered
    assert "ok=`False`" in rendered
    assert "capture preflight gate failed; complete and approve the capture matrix before training" in rendered
    assert "capture_matrix_csv=qa/video_eval/results/apron_harness_production_capture_matrix.csv" in rendered
    assert "missing_label_classes=`4`" in rendered
    assert "training_preflight=`not_required_for_dry_run`" in rendered
    assert "Capture deficit: missing_label_annotations=`1200`" in rendered
    assert "recommended_label_review_rows=`720`" in rendered
    assert "next_batches=`2`" in rendered
    assert "matrix_rows=`3`" in rendered
    assert "apron_harness_capture_work_order.md" in rendered
    assert "apron_harness_capture_matrix.csv" in rendered
    assert "rows=`21`" in rendered
    assert "Production capture matrix sidecar gate: `pass`" in rendered
    assert "Label-review import sidecar gate: `failed`" in rendered
    assert "the emitted `.label_review_import.json` sidecar must validate after import" in rendered
    assert "Capture progress: gate=`blocked`" in rendered
    assert "ready_rows=`0/21`" in rendered
    assert "missing_labeled_examples=`720`" in rendered
    assert "manifest_counts=`failed`" in rendered
    assert "Capture progress by capability:" in rendered
    assert "apron_required(ready=0/11, labeled=0, missing=360)" in rendered
    assert "harness_required(ready=0/10, labeled=0, missing=360)" in rendered
    assert "Capture next blocked rows:" in rendered
    assert (
        "apron_required=apron_required_closed_set_capture.positive.denim_apron"
        "(missing=60, blockers=missing_labeled_examples=60, review_status=not_started, permission=unknown)"
    ) in rendered
    assert (
        "harness_required=harness_required_closed_set_capture.positive.fall_arrest_harness"
        "(missing=60, blockers=missing_labeled_examples=60, review_status=not_started, permission=unknown)"
    ) in rendered
    assert "Production capture progress by capability:" in rendered
    assert "apron_required(ready=0/11, labeled=0, missing=1204)" in rendered
    assert "harness_required(ready=0/10, labeled=0, missing=1200)" in rendered
    assert "Production capture next blocked rows:" in rendered
    assert (
        "apron_required=apron_required_closed_set_capture.positive.denim_apron"
        "(missing=200, blockers=missing_labeled_examples=200, review_status=not_started, permission=unknown)"
    ) in rendered
    assert "Next production actions:" in rendered
    assert "1:approve_or_capture_training_data" in rendered
    assert "2:prove_edge_gate" in rendered
    assert "Next production evidence:" in rendered
    assert "Next production artifacts:" in rendered
    assert "2:prove_edge_gate=qa/video_eval/results/factory_ppe_jetson_gate.json" in rendered
    assert "factory_ppe_3cam_soak.json" in rendered
    assert (
        "3:train_and_register_candidate=/path/to/cleared/apron_harness_yolo26n_result.json"
        in rendered
    )
    assert "models/ppe_closed_set_candidate/apron-harness-ppe.onnx.registry.json" in rendered
    assert "qa/video_eval/results/harness_closed_set_promotion_report.json" in rendered
    assert (
        "1:approve_or_capture_training_data=human/legal approval evidence is required before public seed data can enter training"
        in rendered
    )
    assert (
        "2:prove_edge_gate=same candidate_report_sha256 and selected-export SHA across raw benchmark, soak, promotion, and registry reports"
        in rendered
    )


def test_apron_harness_readiness_doctor_is_not_treated_as_scenario_result(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: person_presence_active
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person detection"]
""",
        encoding="utf-8",
    )
    (result_dir / "apron_harness_readiness_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-21T09:00:00+00:00",
  "pilot_gate_passed": true,
  "production_gate_passed": false,
  "sales_status": "pilot_ready_not_production_compliance",
  "production_blockers": ["missing_or_failed_factory_ppe_jetson_full_gate"],
  "closed_set_handoff": {
    "dataset_schema_ok": true,
    "training_dry_run_status": "ready_to_train",
    "training_model": "yolo26n.pt",
    "selected_device": "cpu",
    "missing_label_minimums": {"person": {"current": 0}},
    "capture_deficit": {
      "total_missing_label_annotations": 300,
      "coverage_deficit_count": 0,
      "next_capture_batches": [{"batch_id": "apron_required_closed_set_capture", "capture_matrix": [{"row_id": "a1"}]}]
    },
    "capture_matrix_csv": {
      "path": "qa/video_eval/results/apron_harness_capture_matrix.csv",
      "generated": true,
      "row_count": 1
    }
  },
  "capabilities": {
    "apron_required": {
      "ok": true,
      "scenarios": {
        "false_positive_guard": {"visible_class_total": 6},
        "suppression": {"model_invocations": {"ppe_specialist": 0}}
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "clip.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, _claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    assert "`person_presence_active`" in report
    assert "`None`" not in report
    assert "Apron/Harness Production Gate" in report
    assert "Production gate: `blocked`" in report
    assert "Production blocker count: `1`" in report
    assert "Capture deficit:" in report
    assert "apron_harness_capture_work_order.md" in report
    assert "apron_harness_capture_matrix.csv" in report


def test_apron_harness_claim_status_uses_production_gate_not_raw_runtime_result(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
coverage_boundaries:
  tested_final_send_brochures:
    - code: RL-F
      file: marketing/suite/final-send/RakshakLens_RL-F_Factory_Datasheet.pdf
      scenario_ids: [factory_missing_apron_active, person_presence_active]
scenarios:
  - id: factory_missing_apron_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Apron claim"]
  - id: person_presence_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "apron_harness_readiness_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-23T09:00:00+00:00",
  "pilot_gate_passed": true,
  "production_gate_passed": false,
  "sales_status": "pilot_ready_not_production_compliance",
  "production_blockers": ["closed_set_model_artifact_missing"],
  "closed_set_handoff": {"dataset_schema_ok": true},
  "capabilities": {}
}
""",
        encoding="utf-8",
    )
    (result_dir / "factory_missing_apron_active.json").write_text(
        '{"scenario_id": "factory_missing_apron_active", "status": "ready_to_sell", "video": "apron.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "person.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    assert "| Scenario | Sales Status | Runtime Status | Video | Evidence Summary |" in report
    assert (
        "| `factory_missing_apron_active` | "
        "`pilot_ready_not_production_compliance (runtime=ready_to_sell)` | "
        "`ready_to_sell` |"
    ) in report
    assert "factory_missing_apron_active: pilot_ready_not_production_compliance (runtime=ready_to_sell)" in report
    assert "person_presence_active: ready_to_sell" in report
    assert (
        "| Apron claim | `factory_missing_apron_active` | "
        "`pilot_ready_not_production_compliance (runtime=ready_to_sell)` |"
    ) in claims
    assert "| Person claim | `person_presence_active` | `ready_to_sell` |" in claims


def test_apron_harness_claim_status_blocks_when_readiness_gate_missing(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
coverage_boundaries:
  tested_final_send_brochures:
    - code: RL-F
      file: marketing/suite/final-send/RakshakLens_RL-F_Factory_Datasheet.pdf
      scenario_ids: [factory_missing_harness_active, person_presence_active]
scenarios:
  - id: factory_missing_harness_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Harness claim"]
  - id: person_presence_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Person claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "factory_missing_harness_active.json").write_text(
        '{"scenario_id": "factory_missing_harness_active", "status": "ready_to_sell", "video": "harness.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )
    (result_dir / "person_presence_active.json").write_text(
        '{"scenario_id": "person_presence_active", "status": "ready_to_sell", "video": "person.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    blocked_status = "blocked_missing_apron_harness_readiness_gate (runtime=ready_to_sell)"
    assert f"| `factory_missing_harness_active` | `{blocked_status}` | `ready_to_sell` |" in report
    assert f"factory_missing_harness_active: {blocked_status}" in report
    assert f"| Harness claim | `factory_missing_harness_active` | `{blocked_status}` |" in claims
    assert "person_presence_active: ready_to_sell" in report
    assert "| Person claim | `person_presence_active` | `ready_to_sell` |" in claims


def test_apron_harness_claim_status_blocks_when_readiness_gate_is_malformed(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: factory_missing_apron_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Apron claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "apron_harness_readiness_doctor.json").write_text(
        '{"generated_at": "2026-06-23T09:00:00+00:00", "production_gate_passed": true}',
        encoding="utf-8",
    )
    (result_dir / "factory_missing_apron_active.json").write_text(
        '{"scenario_id": "factory_missing_apron_active", "status": "ready_to_sell", "video": "apron.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    blocked_status = "blocked_invalid_apron_harness_readiness_gate (runtime=ready_to_sell)"
    assert f"| `factory_missing_apron_active` | `{blocked_status}` | `ready_to_sell` |" in report
    assert f"| Apron claim | `factory_missing_apron_active` | `{blocked_status}` |" in claims


def test_apron_harness_claim_status_blocks_contradictory_production_gate(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: factory_missing_harness_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Harness claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "apron_harness_readiness_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-23T09:00:00+00:00",
  "pilot_gate_passed": true,
  "production_gate_passed": true,
  "sales_status": "ready_to_sell",
  "production_blockers": ["closed_set_model_artifact_missing"]
}
""",
        encoding="utf-8",
    )
    (result_dir / "factory_missing_harness_active.json").write_text(
        '{"scenario_id": "factory_missing_harness_active", "status": "ready_to_sell", "video": "harness.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    blocked_status = "blocked_invalid_apron_harness_readiness_gate (runtime=ready_to_sell)"
    assert f"| `factory_missing_harness_active` | `{blocked_status}` | `ready_to_sell` |" in report
    assert f"| Harness claim | `factory_missing_harness_active` | `{blocked_status}` |" in claims


def test_apron_harness_claim_status_blocks_when_blocked_gate_has_no_blockers(tmp_path):
    video_eval = _load_video_eval()
    artifact_root = tmp_path / "qa" / "video_eval"
    result_dir = artifact_root / "results"
    result_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""
artifact_root: {artifact_root}
scenarios:
  - id: factory_missing_apron_active
    vertical: factory
    source: {{type: fixture, provider: test}}
    sales_claims: ["Apron claim"]
""",
        encoding="utf-8",
    )
    (result_dir / "apron_harness_readiness_doctor.json").write_text(
        """
{
  "ok": true,
  "generated_at": "2026-06-23T09:00:00+00:00",
  "pilot_gate_passed": true,
  "production_gate_passed": false,
  "sales_status": "pilot_ready_not_production_compliance",
  "production_blockers": []
}
""",
        encoding="utf-8",
    )
    (result_dir / "factory_missing_apron_active.json").write_text(
        '{"scenario_id": "factory_missing_apron_active", "status": "ready_to_sell", "video": "apron.mp4", "source": {"type": "fixture"}, "evidence": {}}',
        encoding="utf-8",
    )

    sales_path, claims_path = video_eval.report(manifest_path)

    report = sales_path.read_text(encoding="utf-8")
    claims = claims_path.read_text(encoding="utf-8")
    blocked_status = "blocked_invalid_apron_harness_readiness_gate (runtime=ready_to_sell)"
    assert f"| `factory_missing_apron_active` | `{blocked_status}` | `ready_to_sell` |" in report
    assert f"| Apron claim | `factory_missing_apron_active` | `{blocked_status}` |" in claims
