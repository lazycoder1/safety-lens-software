"""Contract tests for sales-readiness model pack definitions."""

from pathlib import Path
import importlib.util
import json
import shlex

import yaml

import capability_registry
import model_manager
import site_config


ROOT = Path(__file__).resolve().parents[2]
MODEL_PACKS_PATH = ROOT / "qa" / "video_eval" / "model_packs.yaml"
MANIFEST_PATH = ROOT / "qa" / "video_eval" / "manifest.yaml"
MODEL_PACK_DOCTOR_PATH = ROOT / "scripts" / "model_pack_doctor.py"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_pack_doctor():
    spec = importlib.util.spec_from_file_location("model_pack_doctor", MODEL_PACK_DOCTOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_model_packs_are_defined():
    packs = _load_yaml(MODEL_PACKS_PATH)

    assert set(packs["packs"]) >= {
        "base_3cam",
        "factory_ppe_3cam",
        "fire_smoke_3cam",
        "pose_fall_3cam",
        "anpr_gate_1cam",
    }
    policy = packs.get("policy") or {}
    assert policy.get("evidence_doctor_command") == (
        ".venv/bin/python scripts/model_pack_evidence_doctor.py "
        "--skip-model-pack pose_fall_3cam "
        "--out qa/video_eval/results/model_pack_evidence_doctor.json"
    )
    assert policy.get("yolo26_false_positive_doctor_command") == (
        ".venv/bin/python scripts/yolo26_false_positive_doctor.py "
        "--out qa/video_eval/results/yolo26_false_positive_doctor.json"
    )
    assert policy.get("apron_harness_readiness_doctor_command") == (
        ".venv/bin/python scripts/apron_harness_readiness_doctor.py "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml "
        "--capture-kickoff-out qa/video_eval/results/apron_harness_capture_kickoff.md "
        "--out qa/video_eval/results/apron_harness_readiness_doctor.json"
    )
    assert policy.get("apron_harness_seed_source_review_command") == (
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
    assert policy.get("apron_harness_seed_source_review_bundle_validate_command") == (
        ".venv/bin/python scripts/apron_harness_seed_source_doctor.py "
        "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json"
    )
    assert '--review-bundle-out ""' not in policy.get(
        "apron_harness_seed_source_review_bundle_validate_command",
        "",
    )
    assert "--apply-review-checklist-csv" in policy.get(
        "apron_harness_seed_source_review_apply_command",
        "",
    )
    assert "--updated-model-packs-out" in policy.get(
        "apron_harness_seed_source_review_apply_command",
        "",
    )
    assert packs.get("source_of_truth", {}).get("evidence_doctor") == (
        "qa/video_eval/results/model_pack_evidence_doctor.json"
    )
    assert packs.get("source_of_truth", {}).get("yolo26_false_positive_doctor") == (
        "qa/video_eval/results/yolo26_false_positive_doctor.json"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_readiness_doctor") == (
        "qa/video_eval/results/apron_harness_readiness_doctor.json"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_capture_kickoff") == (
        "qa/video_eval/results/apron_harness_capture_kickoff.md"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_promotion_runbook") == (
        "qa/video_eval/results/apron_harness_promotion_runbook.md"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_seed_import_template") == (
        "qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_seed_source_review_checklist") == (
        "qa/video_eval/results/apron_harness_seed_source_review_checklist.csv"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_seed_source_review_evidence_templates") == (
        "qa/video_eval/results/apron_harness_seed_source_review_evidence"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_seed_source_review_packets") == (
        "qa/video_eval/results/apron_harness_seed_source_review_packets"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_next_source_review_batch") == (
        "qa/video_eval/results/apron_harness_next_source_review_batch.json"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_source_review_kickoff") == (
        "qa/video_eval/results/apron_harness_source_review_kickoff.md"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_source_coverage_plan") == (
        "qa/video_eval/results/apron_harness_source_coverage_plan.json"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_label_review_csv") == (
        "qa/video_eval/results/apron_harness_label_review_template.csv"
    )
    assert packs.get("source_of_truth", {}).get("apron_harness_production_label_review_csv") == (
        "qa/video_eval/results/apron_harness_production_label_review_template.csv"
    )
    assert packs.get("policy", {}).get("apron_harness_promotion_runbook_command") == (
        ".venv/bin/python scripts/apron_harness_readiness_doctor.py "
        "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json "
        "--seed-import-manifest qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml "
        "--capture-kickoff-out qa/video_eval/results/apron_harness_capture_kickoff.md "
        "--out qa/video_eval/results/apron_harness_readiness_doctor.json "
        "--promotion-runbook-out qa/video_eval/results/apron_harness_promotion_runbook.md"
    )


def test_model_pack_model_keys_match_runtime_registry():
    packs = _load_yaml(MODEL_PACKS_PATH)
    runtime_model_keys = set(model_manager.MODEL_DEFINITIONS)

    for pack_id, pack in packs["packs"].items():
        assert pack.get("model_keys"), f"{pack_id} must declare model_keys"
        assert set(pack["model_keys"]) <= runtime_model_keys

        registry_models = pack.get("registry_models") or {}
        for model_key in pack["model_keys"]:
            assert model_key in registry_models, f"{pack_id} must document {model_key}"


def test_model_pack_registry_metadata_matches_runtime_registry():
    packs = _load_yaml(MODEL_PACKS_PATH)

    for pack_id, pack in packs["packs"].items():
        registry_models = pack.get("registry_models") or {}
        for model_key in pack.get("model_keys") or []:
            definition = model_manager.MODEL_DEFINITIONS[model_key]
            documented = registry_models[model_key]
            runtime_path = definition["local_path"].relative_to(ROOT)

            assert documented.get("file") == definition["filename"], f"{pack_id}/{model_key} file drifted"
            assert documented.get("registry_path") == str(runtime_path), f"{pack_id}/{model_key} path drifted"
            if definition.get("download_url"):
                runtime_source = definition["download_url"]
                documented_source = str(documented.get("source_url") or "")
                assert documented_source, f"{pack_id}/{model_key} source missing"
                assert runtime_source.startswith(documented_source) or documented_source == runtime_source, (
                    f"{pack_id}/{model_key} source drifted"
                )


def test_model_pack_doctor_flags_yolo11_artifacts_as_legacy(tmp_path, monkeypatch):
    doctor = _load_model_pack_doctor()
    legacy_path = tmp_path / "frontend" / "yolo11n.pt"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-yolo11")
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    monkeypatch.setattr(doctor, "LEGACY_MODEL_ARTIFACT_PATHS", [legacy_path])

    status = doctor.artifact_layout_status()

    assert status["ok"] is False
    assert status["gate"] == "blocked_legacy_artifacts_present"
    assert status["unexpected_paths"] == ["frontend/yolo11n.pt"]


def test_model_manager_promotes_legacy_weight_to_registry_path(tmp_path: Path, monkeypatch):
    legacy_path = tmp_path / "legacy" / "yolo26n.pt"
    local_path = tmp_path / "models" / "coco_primary" / "yolo26n.pt"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-yolo26n")
    definition = dict(model_manager.MODEL_DEFINITIONS["coco_primary"])
    definition["local_path"] = local_path
    definition["legacy_paths"] = [legacy_path]
    monkeypatch.setitem(model_manager.MODEL_DEFINITIONS, "coco_primary", definition)

    resolved = model_manager._resolve_existing_path("coco_primary")

    assert resolved == local_path
    assert local_path.read_bytes() == b"legacy-yolo26n"


def test_model_pack_capabilities_match_capability_registry():
    packs = _load_yaml(MODEL_PACKS_PATH)
    runtime_capabilities = set(capability_registry.CAPABILITY_REGISTRY)

    for pack_id, pack in packs["packs"].items():
        assert pack.get("capabilities"), f"{pack_id} must declare capabilities"
        assert set(pack["capabilities"]) <= runtime_capabilities


def test_factory_ppe_pretrained_shortcut_review_blocks_unqualified_models():
    packs = _load_yaml(MODEL_PACKS_PATH)
    factory_pack = packs["packs"]["factory_ppe_3cam"]
    review = factory_pack["production_training_plan"]["pretrained_shortcut_review"]
    candidates = review["candidates"]
    shared_sources = packs["shared_sources"]

    assert review["status"] == "no_acceptable_pretrained_shortcut_found"
    assert "person, apron, safety_harness, and safety_lanyard" in review["requirement"]
    assert {candidate["source_ref"] for candidate in candidates} == {
        "qualcomm_ppe_detection_hf",
        "hf_hexmon_vyra_yolo_ppe_detection",
        "hf_melihuzunoglu_ppe_detection",
        "hf_yihong_construction_hazard_detection",
        "github_safety_detection_yolov8",
        "github_ppe_detection_yolo_vinayak",
        "sh17_ppe",
        "kaggle_yolov8_ppe_apron_notebook",
    }
    assert all(candidate["source_ref"] in shared_sources for candidate in candidates)
    assert all(candidate["decision"] != "accepted" for candidate in candidates)
    assert all(candidate["local_artifact_status"] != "installed" for candidate in candidates)
    assert all(
        candidate["target_class_coverage"].get("apron") is not True
        or candidate["target_class_coverage"].get("safety_harness") is not True
        or candidate["target_class_coverage"].get("safety_lanyard") is not True
        for candidate in candidates
    )
    candidate_by_source = {candidate["source_ref"]: candidate for candidate in candidates}
    assert candidate_by_source["hf_hexmon_vyra_yolo_ppe_detection"]["decision"] == (
        "rejected_for_factory_apron_harness_shortcut"
    )
    assert candidate_by_source["hf_melihuzunoglu_ppe_detection"]["license_access_status"] == (
        "agpl_3_0_and_educational_research_disclaimer"
    )
    assert candidate_by_source["hf_yihong_construction_hazard_detection"]["artifact_format"] == (
        "yolo26_yolo11_pt_and_onnx"
    )
    assert "closed-set YOLO26n/s candidate" in review["resulting_path"]


def test_model_pack_evidence_scenarios_exist_in_manifest():
    packs = _load_yaml(MODEL_PACKS_PATH)
    manifest = _load_yaml(MANIFEST_PATH)
    scenario_ids = {scenario["id"] for scenario in manifest.get("scenarios", [])}

    for pack_id, pack in packs["packs"].items():
        assert pack.get("evidence_scenarios"), f"{pack_id} must list evidence scenarios"
        missing = [scenario_id for scenario_id in pack["evidence_scenarios"] if scenario_id not in scenario_ids]
        assert missing == [], f"{pack_id} references missing scenarios: {missing}"


def test_model_pack_yaml_validation_commands_reference_existing_configs():
    packs = _load_yaml(MODEL_PACKS_PATH)

    for pack_id, pack in packs["packs"].items():
        commands = pack.get("local_yaml_validation") or []
        config_paths = []
        for command in commands:
            parts = shlex.split(command)
            if "--config" in parts:
                config_paths.append(parts[parts.index("--config") + 1])

        assert config_paths, f"{pack_id} must include at least one YAML validation command"
        missing = [path for path in config_paths if not (ROOT / path).exists()]
        assert missing == [], f"{pack_id} references missing YAML configs: {missing}"


def test_model_pack_site_yaml_configs_parse_semantically():
    packs = _load_yaml(MODEL_PACKS_PATH)
    manifest = _load_yaml(MANIFEST_PATH)
    scenarios_by_id = {scenario["id"]: scenario for scenario in manifest.get("scenarios", [])}
    config_paths = {
        scenarios_by_id[scenario_id]["config_path"]
        for pack in packs["packs"].values()
        for scenario_id in pack.get("evidence_scenarios", [])
        if scenarios_by_id.get(scenario_id, {}).get("config_path")
    }

    assert config_paths, "model packs must reference focused YAML configs"
    for config_path in sorted(config_paths):
        result = site_config.load_site_config(ROOT / config_path, strict_env=False)
        assert result.ok, f"{config_path} failed semantic validation: {result.errors}"


def test_model_pack_local_commands_cover_every_manifest_scenario():
    packs = _load_yaml(MODEL_PACKS_PATH)
    manifest = _load_yaml(MANIFEST_PATH)
    scenarios_by_id = {scenario["id"]: scenario for scenario in manifest.get("scenarios", [])}

    for pack_id, pack in packs["packs"].items():
        pack_scenarios = set(pack["evidence_scenarios"])
        commands = pack.get("local_yaml_validation") or []

        validated_config_paths = set()
        runnable_scenario_ids = set()
        for command in commands:
            parts = shlex.split(command)
            if "--config" in parts and parts[-1] == "validate":
                validated_config_paths.add(parts[parts.index("--config") + 1])
            if parts[:3] == [".venv/bin/python", "scripts/video_eval.py", "run"] and "--scenario" in parts:
                runnable_scenario_ids.add(parts[parts.index("--scenario") + 1])

        expected_config_paths = {
            scenarios_by_id[scenario_id]["config_path"]
            for scenario_id in pack_scenarios
            if scenarios_by_id.get(scenario_id, {}).get("config_path")
        }

        assert expected_config_paths <= validated_config_paths, pack_id
        assert pack_scenarios <= runnable_scenario_ids, pack_id


def test_model_pack_result_evidence_is_ready_and_matches_detector_windows():
    packs = _load_yaml(MODEL_PACKS_PATH)
    manifest = _load_yaml(MANIFEST_PATH)
    scenarios_by_id = {scenario["id"]: scenario for scenario in manifest.get("scenarios", [])}
    result_dir = ROOT / "qa" / "video_eval" / "results"

    for pack_id, pack in packs["packs"].items():
        for scenario_id in pack.get("evidence_scenarios", []):
            scenario = scenarios_by_id[scenario_id]
            result_path = result_dir / f"{scenario_id}.json"
            assert result_path.exists(), f"{pack_id}/{scenario_id} missing result JSON"

            result = _load_json(result_path)
            assert result.get("status") == "ready_to_sell", f"{pack_id}/{scenario_id} is not ready"
            assert not result.get("blocking_errors"), f"{pack_id}/{scenario_id} has blocking errors"

            yaml_commands = result.get("yaml_commands") or []
            assert yaml_commands, f"{pack_id}/{scenario_id} must record YAML commands"
            assert all(int(command.get("returncode", 1)) == 0 for command in yaml_commands), scenario_id

            if scenario.get("config_path"):
                assert result.get("config_path") == scenario["config_path"]
                ui = (result.get("evidence") or {}).get("ui_evidence") or {}
                if (scenario.get("expected") or {}).get("ui_evidence", {}).get("stream_should_render"):
                    assert ui.get("screenshot_exists") is True, scenario_id
                    assert ui.get("screenshot_fresh") is True, scenario_id

            evidence = result.get("evidence") or {}
            analytics_summary = evidence.get("analytics_summary") or {}
            schedule = analytics_summary.get("schedule") or {}
            expected_analytics = (scenario.get("expected") or {}).get("analytics") or []
            for expectation in expected_analytics:
                if expectation.get("type") != "detector_suppression":
                    continue
                assert int(evidence.get("max_detections_count") or 0) <= int(expectation.get("max_detections", 0)), scenario_id
                assert not evidence.get("matching_alerts"), scenario_id
                assert not evidence.get("unexpected_alerts"), scenario_id
                capability = expectation.get("capability")
                if capability:
                    assert capability in (schedule.get("suppressed_capabilities") or []), scenario_id
                invocations = schedule.get("model_invocations") or {}
                for model_key in expectation.get("model_keys") or []:
                    assert int(invocations.get(str(model_key), 0)) <= int(expectation.get("max_model_invocations", 0)), scenario_id


def test_model_packs_keep_blocked_claim_boundaries_explicit():
    packs = _load_yaml(MODEL_PACKS_PATH)

    for pack_id, pack in packs["packs"].items():
        assert pack.get("unlocked_claims"), f"{pack_id} must state unlocked claims"
        assert pack.get("does_not_unlock"), f"{pack_id} must state claims it does not unlock"

    assert "Production ANPR accuracy" in " ".join(packs["packs"]["anpr_gate_1cam"]["does_not_unlock"])
    assert "Jetson 3-camera" in " ".join(packs["packs"]["factory_ppe_3cam"]["does_not_unlock"])


def test_factory_ppe_apron_harness_sourcing_contract_is_machine_checkable():
    packs = _load_yaml(MODEL_PACKS_PATH)
    ppe_pack = packs["packs"]["factory_ppe_3cam"]
    sourcing = ppe_pack.get("sourcing_status") or {}
    dataset_validation = ppe_pack.get("dataset_validation") or {}

    assert sourcing.get("apron_harness_result") == "public_seed_sources_found_unapproved_insufficient_for_production"
    candidate_sources = sourcing.get("candidate_sources") or []
    assert len(candidate_sources) >= 5
    assert {item.get("capability") for item in candidate_sources} >= {"apron_required", "harness_required"}
    assert all(item.get("status") == "candidate_seed_only" for item in candidate_sources)
    assert all(item.get("approval_status") in {"unreviewed", "rejected"} for item in candidate_sources)
    rejected_sources = [item for item in candidate_sources if item.get("approval_status") == "rejected"]
    assert [item.get("source_ref") for item in rejected_sources] == ["roboflow_harness_detection_v1"]
    assert rejected_sources[0].get("blocker") == "source_unavailable_after_agent_search_keep_blocked_or_replace_with_verifiable_source"
    assert all(item.get("approval_status") == "unreviewed" for item in candidate_sources if item not in rejected_sources)
    assert all(item.get("approved_for_training") is False for item in candidate_sources)
    assert all(item.get("blocker") for item in candidate_sources)
    assert all("manifest_import_plan" in (item.get("required_review") or []) for item in candidate_sources)

    capture_plan = sourcing.get("capture_plan")
    assert capture_plan, "factory_ppe_3cam must point to the apron/harness capture plan"
    assert (ROOT / capture_plan).exists(), f"capture plan does not exist: {capture_plan}"

    template_manifest = dataset_validation.get("template_manifest")
    assert template_manifest, "factory_ppe_3cam must point to the apron/harness dataset template"
    assert (ROOT / template_manifest).exists(), f"dataset template does not exist: {template_manifest}"
    example_dataset_yaml = dataset_validation.get("example_dataset_yaml")
    assert example_dataset_yaml, "factory_ppe_3cam must point to the apron/harness dataset YAML example"
    assert (ROOT / example_dataset_yaml).exists(), f"dataset YAML example does not exist: {example_dataset_yaml}"
    assert "scripts/apron_harness_dataset_doctor.py" in dataset_validation.get("schema_check_command", "")
    assert "scripts/apron_harness_seed_source_doctor.py" in dataset_validation.get("seed_source_review_command", "")
    assert "--review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence" in dataset_validation.get("seed_source_review_command", "")
    assert "--review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets" in dataset_validation.get("seed_source_review_command", "")
    assert "--next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json" in dataset_validation.get("seed_source_review_command", "")
    assert "--review-kickoff-out qa/video_eval/results/apron_harness_source_review_kickoff.md" in dataset_validation.get("seed_source_review_command", "")
    assert "--source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json" in dataset_validation.get("seed_source_review_command", "")
    assert "--review-bundle-out qa/video_eval/results/apron_harness_source_review_bundle.json" in dataset_validation.get("seed_source_review_command", "")
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in dataset_validation.get("seed_source_review_bundle_validate_command", "")
    assert "--apply-review-checklist-csv" in dataset_validation.get("seed_source_review_apply_command", "")
    assert "--updated-model-packs-out" in dataset_validation.get("seed_source_review_apply_command", "")
    assert "scripts/apron_harness_seed_source_doctor.py" in dataset_validation.get("seed_import_validation_command", "")
    assert "--review-evidence-template-dir qa/video_eval/results/apron_harness_seed_source_review_evidence" in dataset_validation.get("seed_import_validation_command", "")
    assert "--review-packet-dir qa/video_eval/results/apron_harness_seed_source_review_packets" in dataset_validation.get("seed_import_validation_command", "")
    assert "--next-review-batch-out qa/video_eval/results/apron_harness_next_source_review_batch.json" in dataset_validation.get("seed_import_validation_command", "")
    assert "--source-coverage-plan-out qa/video_eval/results/apron_harness_source_coverage_plan.json" in dataset_validation.get("seed_import_validation_command", "")
    assert "--validate-review-bundle qa/video_eval/results/apron_harness_source_review_bundle.json" in dataset_validation.get("seed_import_validation_command", "")
    assert "--schema-only" in dataset_validation.get("schema_check_command", "")
    assert "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json" in dataset_validation.get("schema_check_command", "")
    assert "--seed-import-manifest qa/video_eval/datasets/apron_harness_seed_import_manifest.template.yaml" in dataset_validation.get("schema_check_command", "")
    assert "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json" in dataset_validation.get("real_capture_command", "")
    assert "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in dataset_validation.get("real_capture_command", "")
    assert "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json" in dataset_validation.get("emit_training_yaml_command", "")
    assert "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in dataset_validation.get("emit_training_yaml_command", "")
    assert "/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml" in dataset_validation.get("production_capture_command", "")
    assert "/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml" in dataset_validation.get("production_capture_matrix_progress_command", "")
    assert "/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml" in dataset_validation.get("emit_training_yaml_command", "")
    assert "--validate-import-manifest" in dataset_validation.get("seed_import_validation_command", "")
    seed_import_template = dataset_validation.get("seed_import_template")
    assert seed_import_template, "factory_ppe_3cam must point to the seed import manifest template"
    assert (ROOT / seed_import_template).exists(), f"seed import template does not exist: {seed_import_template}"
    assert "--mode production" in dataset_validation.get("production_capture_command", "")
    assert "--emit-yolo-dataset-yaml" in dataset_validation.get("emit_training_yaml_command", "")
    assert dataset_validation.get("label_review_template") == (
        "qa/video_eval/results/apron_harness_label_review_template.csv"
    )
    assert dataset_validation.get("production_label_review_template") == (
        "qa/video_eval/results/apron_harness_production_label_review_template.csv"
    )
    label_review_import_command = dataset_validation.get("label_review_import_command", "")
    assert "--mode production" in label_review_import_command
    assert "--import-label-review-csv" in label_review_import_command
    assert "--emit-updated-manifest" in label_review_import_command
    assert "--schema-only" in label_review_import_command
    assert dataset_validation.get("label_review_import_sidecar") == (
        "/path/to/cleared/apron_harness_capture_manifest.reviewed.yaml.label_review_import.json"
    )
    label_review_gate = dataset_validation.get("label_review_gate", "")
    assert "yolo_labels" in label_review_gate
    assert "recomputed_label_counts" in label_review_gate
    assert "cleared_permission" in label_review_gate
    assert "label_review_import_sidecar_hashes" in label_review_gate
    assert "strict_updated_manifest_validation" in label_review_gate

    next_action = sourcing.get("next_required_action", "")
    assert "YOLO26" in next_action
    assert "YOLO11" in next_action
    assert "legacy" in next_action
    assert "YOLOE" not in next_action, "YOLOE must not be the production apron/harness next step"


def test_factory_ppe_apron_harness_training_plan_keeps_yoloe_pilot_only():
    packs = _load_yaml(MODEL_PACKS_PATH)
    training_plan = packs["packs"]["factory_ppe_3cam"].get("production_training_plan") or {}

    assert training_plan.get("status") == "blocked_until_cleared_data"
    assert "scripts/apron_harness_train.py" in training_plan.get("dry_run_command", "")
    assert "--execute" not in training_plan.get("dry_run_command", "")
    assert "qa/video_eval/datasets/apron_harness_dataset.example.yaml" in training_plan.get("example_dry_run_command", "")
    assert "--execute" in training_plan.get("local_train_command", "")
    assert "yolo26n.pt" in training_plan.get("local_train_command", "")
    assert "yolo26s.pt" in training_plan.get("scale_up_train_command", "")
    assert not training_plan.get("fallback_train_command")
    for command_name in ["dry_run_command", "local_train_command", "scale_up_train_command"]:
        command = training_plan.get(command_name, "")
        assert "--capture-preflight-mode production" in command, command_name
        assert "--capture-manifest /path/to/cleared/apron_harness_capture_manifest.reviewed.yaml" in command, command_name
        assert "apron_harness_production_capture_matrix.csv" in command, command_name
        assert "--seed-source-review-report qa/video_eval/results/apron_harness_seed_source_review.json" in command, command_name
        assert "--seed-import-manifest /path/to/filled/apron_harness_seed_import_manifest.yaml" in command, command_name
    assert "scripts/apron_harness_candidate_doctor.py" in training_plan.get("candidate_gate_command", "")
    registry_copy_command = training_plan.get("registry_copy_command", "")
    assert "scripts/apron_harness_model_registry_doctor.py" in registry_copy_command
    assert "--candidate-report /path/to/cleared/apron_harness_candidate_report.json" in registry_copy_command
    assert "--copy" in registry_copy_command
    assert set(training_plan.get("target_classes") or []) >= {
        "person",
        "apron",
        "safety_harness",
        "safety_lanyard",
    }
    assert set(training_plan.get("preferred_candidates") or []) >= {"yolo26n", "yolo26s"}
    assert set(training_plan.get("legacy_runtime_baselines_not_for_new_training") or []) >= {
        "yolo11n",
        "yolo11s",
        "yolov8n",
        "yolov8s",
    }
    assert any("yoloe" in model for model in training_plan.get("pilot_only_models") or [])
    assert training_plan.get("min_labeled_images_per_class_pilot", 0) >= 300
    assert training_plan.get("min_labeled_images_per_class_production", 0) >= 1000

    required_gates = set(training_plan.get("required_gates") or [])
    assert {
        "legal_clearance_for_training_and_validation_footage",
        "local_yaml_active_window_positive",
        "local_yaml_false_positive_guard",
        "local_detector_window_suppression_zero_candidates",
        "apron_harness_seed_import_review_artifacts_match_source_review_paths_and_sha256",
        "apron_harness_dataset_doctor_passes",
        "apron_harness_train_dry_run_passes",
        "training_result_per_class_metrics_emitted_for_all_required_classes",
        "training_result_label_review_import_sidecar_valid",
        "training_result_label_review_import_updated_manifest_validation_valid",
        "apron_harness_candidate_doctor_passes",
        "apron_harness_model_registry_doctor_copies_and_verifies_selected_export",
        "promotion_result_label_review_import_sidecar_valid",
        "promotion_result_label_review_import_updated_manifest_validation_valid",
        "side_by_side_against_current_yoloe_pilot",
        "onnx_or_tensorrt_export",
        "jetson_three_camera_soak",
    } <= required_gates


def test_factory_ppe_closed_set_runtime_handoff_is_not_active_without_candidate():
    packs = _load_yaml(MODEL_PACKS_PATH)
    ppe_pack = packs["packs"]["factory_ppe_3cam"]
    handoff = ppe_pack.get("runtime_handoff") or {}

    planned_model_key = handoff.get("planned_model_key")
    planned_definition = model_manager.MODEL_DEFINITIONS.get(planned_model_key)

    assert handoff.get("status") == "not_registered_until_candidate_gates_pass"
    assert planned_model_key == "ppe_closed_set_candidate"
    assert planned_definition is not None
    assert planned_definition["filename"] == "apron-harness-ppe.onnx"
    assert str(planned_definition["local_path"].relative_to(ROOT)) == handoff["planned_registry_path"]
    assert planned_model_key not in ppe_pack.get("model_keys", [])
    assert planned_model_key not in (ppe_pack.get("registry_models") or {})
    assert handoff.get("current_runtime_model_family") == "ppe_specialist"
    assert handoff.get("current_runtime_path") == "yoloe_open_vocab_pilot"

    ppe_capabilities = set(ppe_pack.get("capabilities") or [])
    routed_model_families = {
        capability_registry.CAPABILITY_REGISTRY[key]["model_family"]
        for key in ppe_capabilities
    }
    assert "ppe_specialist" in routed_model_families
    assert planned_model_key not in routed_model_families

    promotion_policy = " ".join(handoff.get("promotion_policy") or [])
    assert "scripts/apron_harness_candidate_doctor.py" in promotion_policy
    assert "side-by-side YAML/runtime tests" in promotion_policy
    assert "scripts/apron_harness_promotion_doctor.py" in handoff.get("side_by_side_gate_command", "")
    assert "scripts/apron_harness_promotion_doctor.py" in handoff.get("side_by_side_gate_command_harness", "")
    assert "--capability apron_required" in handoff.get("side_by_side_gate_command", "")
    assert "--capability harness_required" in handoff.get("side_by_side_gate_command_harness", "")
    assert "scripts/jetson_benchmark_doctor.py" in handoff.get("jetson_gate_command", "")
    assert "--require-full-gate" in handoff.get("jetson_gate_command", "")
    assert "--candidate-report /path/to/cleared/apron_harness_candidate_report.json" in handoff.get("jetson_gate_command", "")

    jetson_benchmark = ppe_pack.get("jetson_benchmark") or {}
    assert "scripts/jetson_benchmark_doctor.py" in jetson_benchmark.get("doctor_command", "")
    assert "--soak-report" in jetson_benchmark.get("doctor_command", "")
    assert "--candidate-report /path/to/cleared/apron_harness_candidate_report.json" in jetson_benchmark.get("doctor_command", "")

    required_artifacts = set(handoff.get("required_artifacts_before_registration") or [])
    assert {
        "training_result_json_status_trained",
        "candidate_doctor_report_ready_for_side_by_side_runtime_test",
            "candidate_doctor_selected_export_sha256",
            "candidate_doctor_registry_entry",
            "model_registry_doctor_report_registered",
            "onnx_or_tensorrt_export_file",
        "jetson_three_camera_soak_report",
    } <= required_artifacts
    limits = ppe_pack.get("jetson_resource_limits") or {}
    assert set(limits.get("required_positive_alert_capabilities") or []) == {
        "apron_required",
        "harness_required",
    }
    assert set(limits.get("required_detector_window_suppression_capabilities") or []) == {
        "apron_required",
        "harness_required",
    }
    assert set(limits.get("required_false_positive_guard_capabilities") or []) == {
        "apron_required",
        "harness_required",
    }

    first_runtime_tests = set(handoff.get("first_runtime_tests_after_registration") or [])
    assert {
        "factory_missing_apron_active_closed_set",
        "factory_apron_detector_window_suppression_closed_set",
        "factory_missing_harness_active_closed_set",
        "factory_harness_detector_window_suppression_closed_set",
    } <= first_runtime_tests

    forbidden = set(handoff.get("forbidden_before_gates") or [])
    assert {
        "adding_empty_or_placeholder_artifact_to_factory_ppe_3cam_registry_models",
        "activating_ppe_closed_set_candidate_in_capability_registry",
        "marking_factory_ppe_3cam_ready_to_sell_for_production_compliance",
    } <= forbidden


def test_model_packs_define_jetson_resource_limits():
    packs = _load_yaml(MODEL_PACKS_PATH)
    required_fields = {
        "target_hardware",
        "max_cameras",
        "target_fps_per_camera",
        "minimum_acceptable_fps_per_camera",
        "model_format_target",
        "transport_limit",
        "max_mean_latency_ms_per_frame",
        "max_model_server_mean_latency_ms_per_request",
        "max_ram_mb",
        "max_gpu_utilization_percent",
        "soak_minutes",
        "required_metrics",
    }

    for pack_id, pack in packs["packs"].items():
        benchmark = pack.get("jetson_benchmark") or {}
        doctor_command = benchmark.get("doctor_command", "")
        doctor_parts = shlex.split(doctor_command)
        assert doctor_parts[:2] == [".venv/bin/python", "scripts/jetson_benchmark_doctor.py"], pack_id
        assert "--pack" in doctor_parts and doctor_parts[doctor_parts.index("--pack") + 1] == pack_id
        assert "--raw-benchmark" in doctor_parts, pack_id
        assert "--soak-report" in doctor_parts, pack_id
        assert "--require-full-gate" in doctor_parts, pack_id

        limits = pack.get("jetson_resource_limits") or {}
        assert required_fields <= set(limits), f"{pack_id} must define Jetson resource limits"
        assert 1 <= limits["max_cameras"] <= 3
        assert limits["target_fps_per_camera"] >= limits["minimum_acceptable_fps_per_camera"] > 0
        assert limits["max_mean_latency_ms_per_frame"] > 0
        assert limits["max_model_server_mean_latency_ms_per_request"] > 0
        assert limits["max_ram_mb"] > 0
        assert 0 < limits["max_gpu_utilization_percent"] <= 100
        assert limits["soak_minutes"] >= 30
        assert limits["required_metrics"], f"{pack_id} must list benchmark metrics"


def test_model_pack_doctor_allows_explicit_cpu_model_server_fallback():
    doctor = _load_model_pack_doctor()

    status = doctor.local_device_status(
        "model_server_cpu_or_mps_if_supported",
        {"mps_available": False},
    )

    assert status["satisfied"] is True
    assert status["reason"] == "cpu_fallback_only"
    assert status["performance_gate_satisfied"] is False
    assert status["evidence_scope"] == "functional_wiring_only_cpu_fallback"


def test_model_pack_doctor_separates_functional_and_performance_gates(tmp_path: Path):
    doctor = _load_model_pack_doctor()
    report = doctor.build_report(MODEL_PACKS_PATH)

    assert report["model_artifact_layout_gate"] in {"pass", "blocked_legacy_artifacts_present"}
    assert "artifact_layout" in report
    macos_info = report["platform"].get("macos") or {}
    assert "checked" in macos_info
    if report["platform"]["system"] == "Darwin":
        assert macos_info["checked"] is True
        assert "product_version" in macos_info

    assert report["mps_acceptance_gate"] in {"mps_available", "mps_unavailable_cpu_fallback_only"}
    if report["mps_acceptance_gate"] == "mps_unavailable_cpu_fallback_only":
        assert report["local_performance_acceptance_gate"] == "blocked_cpu_fallback_only"
        base_pack = report["packs"]["base_3cam"]
        assert base_pack["local_functional_satisfied"] is True
        assert base_pack["local_performance_gate_satisfied"] is False
        assert "functional_wiring_only_cpu_fallback" in base_pack["local_evidence_scopes"]


def test_model_pack_doctor_flags_legacy_root_model_artifacts(tmp_path: Path):
    doctor = _load_model_pack_doctor()
    stale_artifact = tmp_path / "yoloe-26n-seg.pt"
    stale_artifact.write_bytes(b"stale duplicate")

    doctor.LEGACY_MODEL_ARTIFACT_PATHS = [stale_artifact]
    status = doctor.artifact_layout_status()

    assert status["ok"] is False
    assert status["gate"] == "blocked_legacy_artifacts_present"
    assert status["unexpected_count"] == 1
    assert str(stale_artifact) in status["unexpected_paths"]


def test_benchmark_helpers_use_current_model_defaults():
    run_all = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    autolabel = (ROOT / "scripts" / "02_autolabel_yolo_world.py").read_text(encoding="utf-8")
    train = (ROOT / "scripts" / "03_train_yolo.py").read_text(encoding="utf-8")
    compile_results = (ROOT / "scripts" / "05_compile_results.py").read_text(encoding="utf-8")

    assert "step2_label_yoloe26" in run_all
    assert "step3_train_yolo26n" in run_all
    assert "YOLOE-26" in run_all
    assert "YOLO26n" in run_all

    assert "from ultralytics import YOLOE" in autolabel
    assert 'YOLOE_MODEL = "yoloe-26s-seg.pt"' in autolabel
    assert "YOLOWorld" not in autolabel
    assert "yolov8s-worldv2.pt" not in autolabel

    assert 'MODEL_NAME = "yolo26n.pt"' in train
    assert "yolov8n.pt" not in train

    assert "YOLOE-26 promptable detection" in compile_results
    assert "YOLOv8n" not in compile_results
    assert "YOLO-World" not in compile_results
