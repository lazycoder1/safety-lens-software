#!/usr/bin/env python3
"""Audit model-pack YAML and saved runtime evidence."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import site_config  # noqa: E402


DEFAULT_MODEL_PACKS = ROOT / "qa" / "video_eval" / "model_packs.yaml"
DEFAULT_MANIFEST = ROOT / "qa" / "video_eval" / "manifest.yaml"
DEFAULT_RESULT_DIR = ROOT / "qa" / "video_eval" / "results"
ACCEPTED_DELIVERY_STATUSES = {"delivered", "sent", "success", "ok", "simulated"}
SUPPRESSION_ANALYTICS_TYPES = {"detector_suppression", "capability_schedule_suppression"}
FACTORY_PPE_PRODUCTION_MODELS = {"yolo26n.pt", "yolo26s.pt"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _command_parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _command_option(command: str, option: str) -> str | None:
    parts = _command_parts(command)
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        prefix = f"{option}="
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def _normalize_checkpoint_name(model_name: str) -> str:
    model_name = str(model_name).strip()
    if model_name and not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    return model_name


def _model_pack_commands(pack: dict[str, Any]) -> tuple[set[str], set[str]]:
    validated_config_paths: set[str] = set()
    runnable_scenario_ids: set[str] = set()
    for command in pack.get("local_yaml_validation") or []:
        parts = _command_parts(str(command))
        if "--config" in parts and parts and parts[-1] == "validate":
            validated_config_paths.add(parts[parts.index("--config") + 1])
        if parts[:3] == [".venv/bin/python", "scripts/video_eval.py", "run"] and "--scenario" in parts:
            runnable_scenario_ids.add(parts[parts.index("--scenario") + 1])
    return validated_config_paths, runnable_scenario_ids


def _expected_config_paths(pack_scenarios: set[str], scenarios_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(scenarios_by_id[scenario_id]["config_path"])
        for scenario_id in pack_scenarios
        if scenarios_by_id.get(scenario_id, {}).get("config_path")
    }


def _detector_suppression_expectations(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (scenario.get("expected") or {}).get("analytics") or []
        if isinstance(item, dict) and item.get("type") == "detector_suppression"
    ]


def _scenario_requires_active_window(scenario: dict[str, Any]) -> bool:
    if not scenario.get("config_path"):
        return False
    expected = scenario.get("expected") or {}
    analytics = [
        item
        for item in expected.get("analytics") or []
        if isinstance(item, dict) and item.get("type") not in SUPPRESSION_ANALYTICS_TYPES
    ]
    suppression = [
        item
        for item in expected.get("analytics") or []
        if isinstance(item, dict) and item.get("type") in SUPPRESSION_ANALYTICS_TYPES
    ]
    if suppression:
        return False
    return bool(analytics or expected.get("detections") or expected.get("alerts"))


def _config_path_for_scenario(scenario: dict[str, Any]) -> Path | None:
    config_path = scenario.get("config_path")
    if not config_path:
        return None
    path = Path(str(config_path))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _window_has_daily_weekly_shape(window: Any) -> bool:
    if not isinstance(window, dict):
        return False
    days = window.get("days")
    return bool(
        isinstance(days, list)
        and days
        and str(window.get("from") or "").strip()
        and str(window.get("to") or "").strip()
    )


def _camera_capability_window(
    scenario: dict[str, Any],
    capability: str,
) -> tuple[dict[str, Any] | None, str | None]:
    windows, error = _camera_capability_windows(scenario)
    if error:
        return None, error
    if not isinstance(windows, dict):
        return None, "camera capability_windows missing"
    entry = windows.get(capability)
    if isinstance(entry, list):
        return {"active": True, "windows": entry}, None
    if isinstance(entry, dict):
        return entry, None
    return None, f"capability_windows.{capability} missing"


def _camera_capability_windows(scenario: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    config_path = _config_path_for_scenario(scenario)
    if config_path is None:
        return None, "scenario has no config_path"
    if not config_path.exists():
        return None, f"config file missing: {_rel(config_path)}"
    try:
        config = _load_yaml(config_path)
    except Exception as exc:
        return None, f"config file unreadable: {exc}"
    cameras = config.get("cameras") if isinstance(config.get("cameras"), dict) else {}
    camera_id = str(scenario.get("camera_id") or "")
    camera = cameras.get(camera_id)
    if not isinstance(camera, dict):
        return None, f"camera {camera_id} missing from config"
    windows = camera.get("capability_windows")
    if windows is None:
        windows = camera.get("capability_active_windows")
    if not isinstance(windows, dict):
        return None, "camera capability_windows missing"
    return windows, None


def _check_capability_window_config(scenario: dict[str, Any]) -> tuple[list[str], int]:
    expected_active: set[str] = set()
    if _scenario_requires_active_window(scenario):
        windows, error = _camera_capability_windows(scenario)
        if error:
            return [error], 1
        expected_active = {str(capability) for capability in (windows or {})}
    expected_suppressed = {
        str(expectation.get("capability"))
        for expectation in _detector_suppression_expectations(scenario)
        if expectation.get("capability")
    }
    expected = [(capability, True) for capability in sorted(expected_active)]
    expected.extend((capability, False) for capability in sorted(expected_suppressed))
    if not expected:
        return [], 0

    errors: list[str] = []
    checked = 0
    for capability, should_be_active in expected:
        checked += 1
        window, error = _camera_capability_window(scenario, capability)
        if error:
            errors.append(f"{capability}: {error}")
            continue
        assert window is not None
        active = window.get("active", True) is not False
        if active is not should_be_active:
            expected_state = "active" if should_be_active else "inactive"
            observed_state = "active" if active else "inactive"
            errors.append(
                f"{capability}: capability window is {observed_state}, expected {expected_state}"
            )
        windows = window.get("windows")
        if not isinstance(windows, list) or not windows:
            errors.append(f"{capability}: capability window needs at least one window")
            continue
        if not any(_window_has_daily_weekly_shape(item) for item in windows):
            errors.append(
                f"{capability}: capability window needs days plus from/to time for daily/weekly proof"
            )
    return errors, checked


def _check_yaml_commands(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    yaml_commands = result.get("yaml_commands") or []
    if not yaml_commands:
        return ["missing YAML command evidence"]
    for command in yaml_commands:
        try:
            returncode = int(command.get("returncode", 1))
        except (TypeError, ValueError):
            returncode = 1
        if returncode != 0:
            args = command.get("args") or []
            errors.append(f"YAML command failed: {' '.join(str(arg) for arg in args)}")
    return errors


def _check_yaml_lifecycle(scenario: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], int, int]:
    config_path = scenario.get("config_path")
    if not config_path:
        return [], 0, 0

    expected_config = (ROOT / str(config_path)).resolve()
    actions = {"validate": False, "plan": False, "apply": False}
    skipped_apply_count = 0
    errors: list[str] = []

    for command in result.get("yaml_commands") or []:
        args = [str(arg) for arg in command.get("args") or []]
        if not any(arg.endswith("scripts/safetylens_site.py") for arg in args):
            continue
        action = next((candidate for candidate in actions if candidate in args), None)
        if not action:
            continue
        if "--config" not in args:
            errors.append(f"YAML {action} command missing --config")
            continue
        try:
            observed_config = Path(args[args.index("--config") + 1]).resolve()
        except (IndexError, TypeError):
            errors.append(f"YAML {action} command missing config path")
            continue
        if observed_config != expected_config:
            errors.append(
                f"YAML {action} command used {observed_config}, expected {expected_config}"
            )
        actions[action] = True
        if action == "apply" and command.get("skipped"):
            skipped_apply_count += 1

    missing = sorted(action for action, seen in actions.items() if not seen)
    for action in missing:
        errors.append(f"missing YAML {action} command evidence")
    return errors, 1, skipped_apply_count


def _check_ui_evidence(scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    if not scenario.get("config_path"):
        return []
    expected_ui = (scenario.get("expected") or {}).get("ui_evidence") or {}
    if not expected_ui.get("stream_should_render"):
        return []
    ui = (result.get("evidence") or {}).get("ui_evidence") or {}
    errors: list[str] = []
    if ui.get("screenshot_exists") is not True:
        errors.append("fresh UI evidence missing screenshot file")
    if ui.get("screenshot_fresh") is not True:
        errors.append("fresh UI evidence missing screenshot_fresh=true")
    return errors


def _check_log_evidence(result: dict[str, Any]) -> tuple[list[str], int]:
    evidence = result.get("evidence") or {}
    health = evidence.get("health") or {}
    storage = health.get("storage") or {}
    logs = storage.get("logs") or {}
    errors: list[str] = []
    if not isinstance(logs, dict):
        return ["runtime health evidence is missing log storage snapshot"], 0
    if not logs.get("dir"):
        errors.append("runtime health log storage missing dir")
    if _int_value(logs.get("files"), 0) <= 0:
        errors.append("runtime health log storage has no files")
    if _int_value(logs.get("bytes"), 0) <= 0:
        errors.append("runtime health log storage has no bytes")
    return errors, 1


def _check_alert_evidence(scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    evidence = result.get("evidence") or {}
    expected = scenario.get("expected") or {}
    errors: list[str] = []
    if expected.get("alerts") and not evidence.get("matching_alerts"):
        errors.append("expected alert evidence is missing")
    if evidence.get("unexpected_alerts"):
        errors.append("unexpected alerts were recorded")
    return errors


def _expected_output_ids(scenario: dict[str, Any]) -> set[str]:
    output_ids: set[str] = set()
    expected = scenario.get("expected") or {}
    for alert in expected.get("alerts") or []:
        if not isinstance(alert, dict):
            continue
        for output_id in alert.get("output_ids") or alert.get("outputIds") or []:
            output_ids.add(str(output_id))
    return output_ids


def _accepted_delivery_count(statuses: dict[str, Any]) -> int:
    total = 0
    for status in ACCEPTED_DELIVERY_STATUSES:
        try:
            total += int(statuses.get(status) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _check_webhook_capture(expectation: dict[str, Any], capture: Any) -> list[str]:
    if not isinstance(capture, dict):
        return ["expected webhook delivery capture is missing"]
    errors: list[str] = []
    if capture.get("ok") is not True:
        errors.append("expected webhook delivery capture is not ok")
    request_count = _int_value(capture.get("request_count"), 0)
    min_requests = _int_value(expectation.get("min_requests"), 1)
    if request_count < min_requests:
        errors.append(f"webhook delivery captured {request_count} requests, expected at least {min_requests}")

    alerts = [alert for alert in capture.get("alerts") or [] if isinstance(alert, dict)]
    expected_policy = expectation.get("policy_id")
    if expected_policy and not any(
        alert.get("policyId") == expected_policy or alert.get("policy_id") == expected_policy
        for alert in alerts
    ):
        errors.append(f"webhook delivery capture missing policy {expected_policy}")
    expected_rule = expectation.get("rule")
    if expected_rule and not any(
        alert.get("rule") == expected_rule or alert.get("violation_type") == expected_rule
        for alert in alerts
    ):
        errors.append(f"webhook delivery capture missing rule {expected_rule}")
    return errors


def _check_smtp_capture(expectation: dict[str, Any], capture: Any) -> list[str]:
    if not isinstance(capture, dict):
        return ["expected SMTP delivery capture is missing"]
    errors: list[str] = []
    if capture.get("ok") is not True:
        errors.append("expected SMTP delivery capture is not ok")
    message_count = _int_value(capture.get("message_count"), 0)
    min_messages = _int_value(expectation.get("min_messages"), 1)
    if message_count < min_messages:
        errors.append(f"SMTP delivery captured {message_count} messages, expected at least {min_messages}")

    messages = [message for message in capture.get("messages") or [] if isinstance(message, dict)]
    subject_contains = expectation.get("subject_contains")
    if subject_contains and not any(subject_contains in str(message.get("subject", "")) for message in messages):
        errors.append(f"SMTP delivery capture missing subject text {subject_contains}")
    body_contains = expectation.get("body_contains")
    if body_contains and not any(body_contains in str(message.get("body_preview", "")) for message in messages):
        errors.append(f"SMTP delivery capture missing body text {body_contains}")
    recipient = expectation.get("recipient")
    if recipient and not any(recipient in (message.get("recipients") or []) for message in messages):
        errors.append(f"SMTP delivery capture missing recipient {recipient}")
    return errors


def _check_delivery_evidence(scenario: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], int]:
    evidence = result.get("evidence") or {}
    expected = scenario.get("expected") or {}
    expected_output_ids = _expected_output_ids(scenario)
    expected_delivery = expected.get("delivery") or {}
    if not expected_output_ids and not expected_delivery:
        return [], 0

    errors: list[str] = []
    checked = 0
    observed_expected_output_ids = {str(output_id) for output_id in evidence.get("expected_output_ids") or []}
    if expected_output_ids and observed_expected_output_ids != expected_output_ids:
        errors.append(
            "result expected_output_ids "
            f"{sorted(observed_expected_output_ids)} do not match manifest {sorted(expected_output_ids)}"
        )

    delivery_summary = evidence.get("delivery_summary") or {}
    if expected_output_ids and not isinstance(delivery_summary, dict):
        errors.append("delivery_summary is missing or not a mapping")
        delivery_summary = {}

    for output_id in sorted(expected_output_ids):
        checked += 1
        entry = delivery_summary.get(output_id)
        if not isinstance(entry, dict):
            errors.append(f"expected output {output_id} missing from delivery_summary")
            continue
        statuses = entry.get("statuses") or {}
        if not isinstance(statuses, dict) or _accepted_delivery_count(statuses) <= 0:
            errors.append(f"expected output {output_id} has no accepted delivery status")

    observed_expected_delivery = evidence.get("expected_delivery") or {}
    if expected_delivery and observed_expected_delivery != expected_delivery:
        errors.append("result expected_delivery does not match manifest delivery expectation")

    webhook_expected = expected_delivery.get("webhook_capture") or {}
    if webhook_expected:
        checked += 1
        errors.extend(_check_webhook_capture(webhook_expected, evidence.get("webhook_capture")))

    smtp_expected = expected_delivery.get("smtp_capture") or {}
    if smtp_expected:
        checked += 1
        errors.extend(_check_smtp_capture(smtp_expected, evidence.get("smtp_capture")))

    return errors, checked


def _check_model_evidence(pack: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], int]:
    evidence = result.get("evidence") or {}
    health = evidence.get("health") or {}
    health_models = health.get("models")
    if not isinstance(health_models, list):
        return ["runtime health evidence is missing model registry snapshot"], 0

    models_by_key = {
        str(model.get("model_key")): model
        for model in health_models
        if isinstance(model, dict) and model.get("model_key")
    }
    registry_models = pack.get("registry_models") or {}
    errors: list[str] = []
    checked = 0

    for model_key in pack.get("model_keys") or []:
        model_key = str(model_key)
        checked += 1
        expected = registry_models.get(model_key) or {}
        observed = models_by_key.get(model_key)
        if not observed:
            errors.append(f"runtime health missing model {model_key}")
            continue

        expected_file = str(expected.get("file") or "")
        observed_file = str(observed.get("filename") or "")
        if expected_file and observed_file != expected_file:
            errors.append(f"runtime model {model_key} filename {observed_file} does not match {expected_file}")

        expected_source = str(expected.get("source_url") or "")
        observed_source = str(observed.get("download_url") or "")
        if expected_source.startswith("http") and observed_source and not observed_source.startswith(expected_source):
            errors.append(f"runtime model {model_key} source {observed_source} does not match {expected_source}")

    return errors, checked


def _multi_capability_exceptions(pack: dict[str, Any]) -> tuple[dict[str, set[str]], list[str]]:
    errors: list[str] = []
    exceptions: dict[str, set[str]] = {}
    raw = pack.get("multi_capability_scenario_exceptions") or []
    if not isinstance(raw, list):
        return exceptions, ["multi_capability_scenario_exceptions must be a list when present"]
    for item in raw:
        if not isinstance(item, dict):
            errors.append("multi_capability_scenario_exceptions entries must be mappings")
            continue
        scenario_id = str(item.get("scenario") or "")
        allowed = item.get("allowed_capabilities") or []
        if not scenario_id:
            errors.append("multi-capability exception missing scenario")
            continue
        if not item.get("reason"):
            errors.append(f"{scenario_id}: multi-capability exception missing reason")
        if not isinstance(allowed, list) or not allowed:
            errors.append(f"{scenario_id}: multi-capability exception needs allowed_capabilities")
            continue
        exceptions[scenario_id] = {str(capability) for capability in allowed}
    return exceptions, errors


def _final_camera_capabilities(result: dict[str, Any]) -> list[str]:
    evidence = result.get("evidence") or {}
    camera = evidence.get("final_camera") or {}
    execution_plan = camera.get("execution_plan") or {}
    raw = execution_plan.get("capabilities")
    if raw is None:
        raw = camera.get("capabilities")
    if not isinstance(raw, list):
        return []
    return [str(capability) for capability in raw if capability]


def _check_scenario_isolation(
    scenario_id: str,
    result: dict[str, Any],
    exceptions: dict[str, set[str]],
) -> tuple[list[str], int]:
    capabilities = _final_camera_capabilities(result)
    if not capabilities:
        return ["final camera evidence is missing planned capabilities"], 0

    unique_capabilities = set(capabilities)
    if scenario_id in exceptions:
        allowed = exceptions[scenario_id]
        if unique_capabilities != allowed:
            return [
                f"multi-capability scenario capabilities {sorted(unique_capabilities)} do not match allowed {sorted(allowed)}"
            ], 1
        return [], 1

    if len(unique_capabilities) > 1:
        return [
            f"scenario is not one-detection isolated; planned capabilities are {sorted(unique_capabilities)}"
        ], 1
    return [], 1


def _check_detector_suppression(scenario: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], int]:
    evidence = result.get("evidence") or {}
    analytics_summary = evidence.get("analytics_summary") or {}
    schedule = analytics_summary.get("schedule") or {}
    errors: list[str] = []
    checked = 0

    for expectation in _detector_suppression_expectations(scenario):
        checked += 1
        max_detections = int(expectation.get("max_detections", 0))
        if int(evidence.get("max_detections_count") or 0) > max_detections:
            errors.append(f"detector suppression emitted more than {max_detections} detections")
        if evidence.get("matching_alerts"):
            errors.append("detector suppression emitted matching alerts")
        if evidence.get("unexpected_alerts"):
            errors.append("detector suppression emitted unexpected alerts")
        capability = expectation.get("capability")
        if capability and capability not in (schedule.get("suppressed_capabilities") or []):
            errors.append(f"detector suppression missing suppressed capability {capability}")
        invocations = schedule.get("model_invocations") or {}
        max_model_invocations = int(expectation.get("max_model_invocations", 0))
        for model_key in expectation.get("model_keys") or []:
            observed = int(invocations.get(str(model_key), 0))
            if observed > max_model_invocations:
                errors.append(
                    f"detector suppression model {model_key} invocations {observed} exceed {max_model_invocations}"
                )
    return errors, checked


def _check_active_window(scenario: dict[str, Any], result: dict[str, Any]) -> tuple[list[str], int]:
    if not _scenario_requires_active_window(scenario):
        return [], 0

    evidence = result.get("evidence") or {}
    analytics_summary = evidence.get("analytics_summary") or {}
    schedule = analytics_summary.get("schedule")
    camera = evidence.get("final_camera") or {}
    execution_plan = camera.get("execution_plan") or {}
    errors: list[str] = []

    if not isinstance(schedule, dict):
        return ["active window schedule evidence is missing"], 1

    if schedule.get("ok") is not True:
        errors.append("active window schedule ok is not true")

    suppressed_capabilities = {str(capability) for capability in schedule.get("suppressed_capabilities") or []}
    for capability in _final_camera_capabilities(result):
        if capability in suppressed_capabilities:
            errors.append(f"active window suppressed planned capability {capability}")

    invocations = schedule.get("model_invocations")
    if not isinstance(invocations, dict):
        errors.append("active window model invocation evidence is missing")
        invocations = {}

    required_model_keys = [str(model_key) for model_key in execution_plan.get("required_model_keys") or []]
    if required_model_keys:
        for model_key in required_model_keys:
            observed = _int_value(invocations.get(model_key), 0)
            if observed <= 0:
                errors.append(f"active window model {model_key} invocations {observed} are not positive")
    elif not any(_int_value(value, 0) > 0 for value in invocations.values()):
        errors.append("active window has no positive model invocations")

    return errors, 1


def _validate_config(path: Path) -> list[str]:
    if not path.exists():
        return [f"config path does not exist: {_rel(path)}"]
    result = site_config.load_site_config(path, strict_env=False)
    return [] if result.ok else [f"semantic site config validation failed: {result.errors}"]


def _pack_claims_production_readiness(status: Any) -> bool:
    normalized = str(status or "").strip().lower()
    return normalized in {
        "ready_to_sell",
        "ready_to_sell_with_scope_limits",
        "ready_to_sell_production_compliance",
        "production_ready",
    }


def _model_stem(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized[:-3] if normalized.endswith(".pt") else normalized


def _source_has_url(source: dict[str, Any]) -> bool:
    url_keys = ("url", "docs_url", "paper", "license_url", "jetpack_archive_url")
    return any(str(source.get(key) or "").startswith("http") for key in url_keys)


def _parse_checked_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _source_research_policy(packs_doc: dict[str, Any]) -> tuple[int, list[str]]:
    errors: list[str] = []
    policy = packs_doc.get("policy") or {}
    if "source_research_max_age_days" not in policy:
        return 14, errors
    max_age_days = _positive_int(policy.get("source_research_max_age_days"))
    if max_age_days is None:
        errors.append("policy.source_research_max_age_days must be a positive integer")
        return 14, errors
    return max_age_days, errors


def _check_source_research_contract(packs_doc: dict[str, Any]) -> tuple[list[str], int]:
    errors: list[str] = []
    policy = packs_doc.get("policy") or {}
    model_selection_rule = str(policy.get("model_selection_rule") or "").lower()
    if "web" not in model_selection_rule or "research" not in model_selection_rule:
        errors.append("policy.model_selection_rule must require current web research")
    max_age_days, max_age_errors = _source_research_policy(packs_doc)
    errors.extend(max_age_errors)
    today = datetime.now(timezone.utc).date()

    shared_sources = packs_doc.get("shared_sources") or {}
    if not isinstance(shared_sources, dict) or not shared_sources:
        return errors + ["shared_sources must document researched model/data sources"], max_age_days

    registry_source_refs: dict[str, list[str]] = {}
    required_registry_model_fields = {
        "expected_input_size",
        "recommended_confidence",
        "local_device",
        "jetson_device",
    }
    registry_models_by_pack = {
        str(pack_id): pack.get("registry_models") or {}
        for pack_id, pack in (packs_doc.get("packs") or {}).items()
        if isinstance(pack, dict)
    }
    for pack_id, registry_models in registry_models_by_pack.items():
        for model_key, model in registry_models.items():
            if not isinstance(model, dict):
                continue
            source_ref = str(model.get("source_ref") or "")
            if source_ref:
                registry_source_refs.setdefault(source_ref, []).append(f"{pack_id}.{model_key}")

    for source_id, source in shared_sources.items():
        if not isinstance(source, dict):
            errors.append(f"shared_sources.{source_id} must be a mapping")
            continue
        checked_date = _parse_checked_date(source.get("checked"))
        if not source.get("checked"):
            errors.append(f"shared_sources.{source_id} must include checked date")
        elif checked_date is None:
            errors.append(f"shared_sources.{source_id} checked date must be ISO-8601")
        elif checked_date > today:
            errors.append(f"shared_sources.{source_id} checked date {checked_date.isoformat()} is in the future")
        elif max_age_days > 0 and (today - checked_date).days > max_age_days:
            errors.append(
                f"shared_sources.{source_id} checked date {checked_date.isoformat()} is older than "
                f"{max_age_days} days"
            )
        if not _source_has_url(source):
            errors.append(f"shared_sources.{source_id} must include a primary source URL")
        if not (source.get("license_note") or source.get("access_note")):
            errors.append(f"shared_sources.{source_id} must include license_note or access_note")
        if not source.get("relevance"):
            errors.append(f"shared_sources.{source_id} must explain relevance")
        if not source.get("decision"):
            errors.append(f"shared_sources.{source_id} must record accepted/rejected/candidate decision")
        if source_id in registry_source_refs:
            if not (source.get("release_note") or source.get("version_note")):
                errors.append(
                    f"shared_sources.{source_id} must include release_note or version_note for registry models"
                )
            if not source.get("export_note"):
                errors.append(f"shared_sources.{source_id} must include export_note for registry models")
            if not (source.get("runtime_note") or source.get("edge_note")):
                errors.append(
                    f"shared_sources.{source_id} must include runtime_note or edge_note for registry models"
                )

    for pack_id, registry_models in registry_models_by_pack.items():
        for model_key, model in registry_models.items():
            if not isinstance(model, dict):
                errors.append(f"{pack_id}.{model_key} registry model must be a mapping")
                continue
            for field in sorted(required_registry_model_fields):
                if model.get(field) in (None, ""):
                    errors.append(f"{pack_id}.{model_key} must include {field}")
            if not model.get("source_url"):
                errors.append(f"{pack_id}.{model_key} must include source_url or manual provenance marker")
            if not model.get("license_note"):
                errors.append(f"{pack_id}.{model_key} must include license_note")
            source_ref = model.get("source_ref")
            if str(model.get("source_url") or "").startswith("http") and not source_ref:
                errors.append(f"{pack_id}.{model_key} must include source_ref for researched HTTP source")
            if source_ref and source_ref not in shared_sources:
                errors.append(f"{pack_id}.{model_key} source_ref {source_ref} missing from shared_sources")
            if not str(model.get("source_url") or "").startswith("http") and not model.get("provenance_note"):
                errors.append(f"{pack_id}.{model_key} manual/internal source must include provenance_note")

    return errors, max_age_days


def _scenario_detector_suppression_capabilities(scenario: dict[str, Any]) -> set[str]:
    return {
        str(expectation.get("capability"))
        for expectation in _detector_suppression_expectations(scenario)
        if expectation.get("capability")
    }


def _check_detector_window_coverage(
    pack_id: str,
    pack: dict[str, Any],
    pack_scenarios: set[str],
    scenarios_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    capabilities = {str(capability) for capability in pack.get("capabilities") or []}
    covered_capabilities: set[str] = set()

    for scenario_id in pack_scenarios:
        scenario = scenarios_by_id.get(scenario_id) or {}
        covered_capabilities.update(_scenario_detector_suppression_capabilities(scenario))

    gap_items = pack.get("detector_window_gaps") or []
    if not isinstance(gap_items, list):
        gap_items = []
        errors.append(f"{pack_id}: detector_window_gaps must be a list when present")
    documented_gaps: dict[str, dict[str, Any]] = {}
    for item in gap_items:
        if not isinstance(item, dict):
            errors.append(f"{pack_id}: detector_window_gaps entries must be mappings")
            continue
        capability = str(item.get("capability") or "")
        if not capability:
            errors.append(f"{pack_id}: detector_window_gaps entry missing capability")
            continue
        documented_gaps[capability] = item
        if not item.get("status"):
            errors.append(f"{pack_id}: detector_window_gaps.{capability} missing status")
        if not item.get("reason"):
            errors.append(f"{pack_id}: detector_window_gaps.{capability} missing reason")

    missing = sorted(capabilities - covered_capabilities - set(documented_gaps))
    for capability in missing:
        errors.append(f"{pack_id}: {capability} lacks detector-window suppression evidence or documented gap")

    if _pack_claims_production_readiness(pack.get("status")) and documented_gaps:
        errors.append(f"{pack_id}: production-ready packs cannot carry detector_window_gaps")

    return errors, {
        "capabilities": sorted(capabilities),
        "covered_capabilities": sorted(covered_capabilities),
        "documented_gaps": documented_gaps,
        "missing_capabilities": missing,
    }


def _check_pack_policy(pack_id: str, pack: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if pack_id != "factory_ppe_3cam":
        return errors

    status = pack.get("status")
    model_keys = set(str(model_key) for model_key in pack.get("model_keys") or [])
    capabilities = set(str(capability) for capability in pack.get("capabilities") or [])
    production_plan = pack.get("production_training_plan") or {}
    runtime_handoff = pack.get("runtime_handoff") or {}
    planned_model_key = str(runtime_handoff.get("planned_model_key") or "ppe_closed_set_candidate")
    pilot_only_models = {_model_stem(model) for model in production_plan.get("pilot_only_models") or []}

    if {"apron_required", "harness_required"} & capabilities:
        if _pack_claims_production_readiness(status) and planned_model_key not in model_keys:
            errors.append(
                "factory_ppe_3cam cannot claim production readiness until ppe_closed_set_candidate is registered"
            )
        if _pack_claims_production_readiness(status) and "ppe_specialist" in model_keys:
            errors.append(
                "factory_ppe_3cam cannot claim production readiness while the active PPE model key is ppe_specialist"
            )
        if runtime_handoff.get("current_runtime_path") == "yoloe_open_vocab_pilot":
            registry_models = pack.get("registry_models") or {}
            ppe_model = registry_models.get("ppe_specialist") or {}
            ppe_file = str(ppe_model.get("file") or "")
            if ppe_file and _model_stem(ppe_file) not in pilot_only_models:
                errors.append("factory_ppe_3cam ppe_specialist file must be listed under pilot_only_models")

    dry_run_command = str(production_plan.get("dry_run_command") or "")
    local_train_command = str(production_plan.get("local_train_command") or "")
    fallback_train_command = str(production_plan.get("fallback_train_command") or "")
    registry_copy_command = str(production_plan.get("registry_copy_command") or "")
    scale_up_train_command = str(production_plan.get("scale_up_train_command") or "")
    if fallback_train_command:
        errors.append("factory_ppe_3cam fallback_train_command is deprecated; use scale_up_train_command with yolo26n/s")
    if production_plan.get("conservative_baselines"):
        errors.append("factory_ppe_3cam conservative_baselines is deprecated; keep older models under legacy_runtime_baselines_not_for_new_training")
    for label, command in {
        "dry_run_command": dry_run_command,
        "local_train_command": local_train_command,
        "scale_up_train_command": scale_up_train_command,
    }.items():
        if command and "--capture-preflight-mode production" not in command:
            errors.append(f"factory_ppe_3cam {label} must require production capture preflight")
        if command and "apron_harness_production_capture_matrix.csv" not in command:
            errors.append(f"factory_ppe_3cam {label} must use the production capture matrix")
        model_name = _command_option(command, "--model") if command else None
        normalized_model = _normalize_checkpoint_name(model_name or "")
        if normalized_model and normalized_model not in FACTORY_PPE_PRODUCTION_MODELS:
            errors.append(
                f"factory_ppe_3cam {label} model {normalized_model} must be one of {sorted(FACTORY_PPE_PRODUCTION_MODELS)}"
            )

    if registry_copy_command:
        if "--copy" not in _command_parts(registry_copy_command):
            errors.append("factory_ppe_3cam registry_copy_command must use --copy")
        if not _command_option(registry_copy_command, "--candidate-report"):
            errors.append("factory_ppe_3cam registry_copy_command must pass --candidate-report")
        if not _command_option(registry_copy_command, "--apron-promotion-report"):
            errors.append("factory_ppe_3cam registry_copy_command must pass --apron-promotion-report")
        if not _command_option(registry_copy_command, "--harness-promotion-report"):
            errors.append("factory_ppe_3cam registry_copy_command must pass --harness-promotion-report")
    else:
        errors.append("factory_ppe_3cam registry_copy_command is required")

    return errors


def audit_model_pack_evidence(
    *,
    model_packs_path: Path = DEFAULT_MODEL_PACKS,
    manifest_path: Path = DEFAULT_MANIFEST,
    result_dir: Path = DEFAULT_RESULT_DIR,
    skipped_model_packs: set[str] | None = None,
) -> dict[str, Any]:
    packs_doc = _load_yaml(model_packs_path)
    manifest = _load_yaml(manifest_path)
    scenarios_by_id = {str(scenario["id"]): scenario for scenario in manifest.get("scenarios", [])}
    skipped_model_packs = {str(pack_id) for pack_id in (skipped_model_packs or set())}

    global_errors: list[str] = []
    global_warnings: list[str] = []
    pack_reports: dict[str, Any] = {}
    unique_scenarios: set[str] = set()
    unique_configs: set[str] = set()
    suppression_count = 0
    model_evidence_count = 0
    isolation_check_count = 0
    yaml_lifecycle_check_count = 0
    yaml_apply_skipped_count = 0
    ready_count = 0
    detector_window_gap_count = 0
    delivery_check_count = 0
    log_evidence_check_count = 0
    active_window_check_count = 0
    capability_window_config_check_count = 0

    source_research_errors, source_research_max_age_days = _check_source_research_contract(packs_doc)
    global_errors.extend(source_research_errors)

    active_packs = {
        str(pack_id): pack
        for pack_id, pack in (packs_doc.get("packs") or {}).items()
        if str(pack_id) not in skipped_model_packs
    }

    for pack_id, pack in active_packs.items():
        pack_errors: list[str] = []
        pack_warnings: list[str] = []
        pack_errors.extend(_check_pack_policy(str(pack_id), pack))
        multi_capability_exceptions, exception_errors = _multi_capability_exceptions(pack)
        pack_errors.extend(exception_errors)
        pack_scenarios = {str(scenario_id) for scenario_id in pack.get("evidence_scenarios") or []}
        unique_scenarios.update(pack_scenarios)
        validated_configs, runnable_scenarios = _model_pack_commands(pack)
        expected_configs = _expected_config_paths(pack_scenarios, scenarios_by_id)
        unique_configs.update(expected_configs)

        missing_scenarios = sorted(pack_scenarios - set(scenarios_by_id))
        for scenario_id in missing_scenarios:
            pack_errors.append(f"{scenario_id}: missing from manifest")
        for scenario_id in sorted(set(multi_capability_exceptions) - pack_scenarios):
            pack_errors.append(f"{scenario_id}: multi-capability exception is not listed as an evidence scenario")

        for config_path in sorted(expected_configs - validated_configs):
            pack_errors.append(f"{config_path}: missing local YAML validate command")
        for scenario_id in sorted(pack_scenarios - runnable_scenarios):
            pack_errors.append(f"{scenario_id}: missing video_eval run command")

        coverage_errors, detector_window_coverage = _check_detector_window_coverage(
            str(pack_id),
            pack,
            pack_scenarios,
            scenarios_by_id,
        )
        pack_errors.extend(coverage_errors)
        detector_window_gap_count += len(detector_window_coverage["documented_gaps"])

        scenario_summaries: dict[str, Any] = {}
        for config_path in sorted(expected_configs):
            errors = _validate_config(ROOT / config_path)
            for error in errors:
                pack_errors.append(f"{config_path}: {error}")

        for scenario_id in sorted(pack_scenarios):
            scenario = scenarios_by_id.get(scenario_id)
            if scenario is None:
                continue
            result_path = result_dir / f"{scenario_id}.json"
            scenario_errors: list[str] = []
            scenario_warnings: list[str] = []
            result_status = "missing"

            if not result_path.exists():
                scenario_errors.append("missing result JSON")
            else:
                try:
                    result = _load_json(result_path)
                except Exception as exc:
                    result = {}
                    scenario_errors.append(f"result JSON is unreadable: {exc}")
                result_status = str(result.get("status") or "unknown")
                if result_status == "ready_to_sell":
                    ready_count += 1
                else:
                    scenario_errors.append(f"result status is {result_status}, expected ready_to_sell")
                if result.get("blocking_errors"):
                    scenario_errors.append(f"blocking errors recorded: {result['blocking_errors']}")
                scenario_errors.extend(_check_yaml_commands(result))
                lifecycle_errors, checked_lifecycle, skipped_apply_count = _check_yaml_lifecycle(scenario, result)
                yaml_lifecycle_check_count += checked_lifecycle
                yaml_apply_skipped_count += skipped_apply_count
                scenario_errors.extend(lifecycle_errors)
                if scenario.get("config_path") and result.get("config_path") != scenario.get("config_path"):
                    scenario_errors.append(
                        f"result config_path {result.get('config_path')} does not match manifest {scenario.get('config_path')}"
                    )
                scenario_errors.extend(_check_ui_evidence(scenario, result))
                log_errors, checked_logs = _check_log_evidence(result)
                log_evidence_check_count += checked_logs
                scenario_errors.extend(log_errors)
                scenario_errors.extend(_check_alert_evidence(scenario, result))
                delivery_errors, checked_delivery = _check_delivery_evidence(scenario, result)
                delivery_check_count += checked_delivery
                scenario_errors.extend(delivery_errors)
                model_errors, checked_models = _check_model_evidence(pack, result)
                model_evidence_count += checked_models
                scenario_errors.extend(model_errors)
                isolation_errors, checked_isolation = _check_scenario_isolation(
                    scenario_id,
                    result,
                    multi_capability_exceptions,
                )
                isolation_check_count += checked_isolation
                scenario_errors.extend(isolation_errors)
                suppression_errors, checked_suppression = _check_detector_suppression(scenario, result)
                suppression_count += checked_suppression
                scenario_errors.extend(suppression_errors)
                active_errors, checked_active = _check_active_window(scenario, result)
                active_window_check_count += checked_active
                scenario_errors.extend(active_errors)
                window_config_errors, checked_window_config = _check_capability_window_config(scenario)
                capability_window_config_check_count += checked_window_config
                scenario_errors.extend(window_config_errors)

            for error in scenario_errors:
                pack_errors.append(f"{scenario_id}: {error}")
            for warning in scenario_warnings:
                pack_warnings.append(f"{scenario_id}: {warning}")

            scenario_summaries[scenario_id] = {
                "result_path": _rel(result_path),
                "status": result_status,
                "has_config": bool(scenario.get("config_path")),
                "detector_suppression_checks": len(_detector_suppression_expectations(scenario)),
                "active_window_checks": 1 if _scenario_requires_active_window(scenario) else 0,
                "capability_window_config_checks": (
                    1
                    if _scenario_requires_active_window(scenario)
                    else len(_detector_suppression_expectations(scenario))
                ),
                "delivery_checks": len(_expected_output_ids(scenario))
                + len((scenario.get("expected") or {}).get("delivery") or {}),
                "log_evidence_checks": 1,
                "ok": not scenario_errors,
                "errors": scenario_errors,
                "warnings": scenario_warnings,
            }

        pack_reports[str(pack_id)] = {
            "ok": not pack_errors,
            "status": pack.get("status"),
            "scenario_count": len(pack_scenarios),
            "config_count": len(expected_configs),
            "errors": pack_errors,
            "warnings": pack_warnings,
            "detector_window_coverage": detector_window_coverage,
            "scenarios": scenario_summaries,
        }
        global_errors.extend(f"{pack_id}: {error}" for error in pack_errors)
        global_warnings.extend(f"{pack_id}: {warning}" for warning in pack_warnings)

    return {
        "ok": not global_errors,
        "generated_at": utc_now(),
        "inputs": {
            "model_packs": _rel(model_packs_path),
            "manifest": _rel(manifest_path),
            "result_dir": _rel(result_dir),
        },
        "stats": {
            "pack_count": len(active_packs),
            "shared_source_count": len(packs_doc.get("shared_sources") or {}),
            "source_research_max_age_days": source_research_max_age_days,
            "unique_scenario_count": len(unique_scenarios),
            "unique_config_count": len(unique_configs),
            "ready_result_count": ready_count,
            "yaml_lifecycle_check_count": yaml_lifecycle_check_count,
            "yaml_apply_skipped_count": yaml_apply_skipped_count,
            "model_evidence_check_count": model_evidence_count,
            "scenario_isolation_check_count": isolation_check_count,
            "detector_suppression_check_count": suppression_count,
            "detector_window_gap_count": detector_window_gap_count,
            "delivery_check_count": delivery_check_count,
            "log_evidence_check_count": log_evidence_check_count,
            "active_window_check_count": active_window_check_count,
            "capability_window_config_check_count": capability_window_config_check_count,
        },
        "errors": global_errors,
        "warnings": global_warnings,
        "skipped_model_packs": sorted(skipped_model_packs),
        "packs": pack_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit model-pack YAML and saved video-eval evidence.")
    parser.add_argument("--model-packs", default=str(DEFAULT_MODEL_PACKS), help="Path to model_packs.yaml")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to manifest.yaml")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing scenario result JSON")
    parser.add_argument(
        "--skip-model-pack",
        action="append",
        default=[],
        help="Model pack ID to omit from the active evidence scope. Repeat for multiple packs.",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_model_pack_evidence(
        model_packs_path=Path(args.model_packs),
        manifest_path=Path(args.manifest),
        result_dir=Path(args.result_dir),
        skipped_model_packs={str(pack_id) for pack_id in args.skip_model_pack},
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        stats = report["stats"]
        print(
            f"{status}: {stats['pack_count']} packs, {stats['unique_scenario_count']} scenarios, "
            f"{stats['ready_result_count']} ready results, "
            f"{stats['yaml_lifecycle_check_count']} YAML lifecycle checks, "
            f"{stats['model_evidence_check_count']} model evidence checks, "
            f"{stats['scenario_isolation_check_count']} isolation checks, "
            f"{stats['log_evidence_check_count']} log checks, "
            f"{stats['active_window_check_count']} active-window checks, "
            f"{stats['capability_window_config_check_count']} capability-window config checks, "
            f"{stats['delivery_check_count']} delivery checks, "
            f"{stats['detector_suppression_check_count']} detector-window checks, "
            f"{stats['detector_window_gap_count']} detector-window gaps"
        )
        if args.out:
            print(f"wrote: {args.out}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARN: {warning}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
