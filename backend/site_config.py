"""Canonical site YAML import/export for Rakshak Lens deployments."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from camera_planner import ALLOWED_CAPABILITY_MODEL_OVERRIDES
from capability_registry import CAPABILITY_REGISTRY, normalize_capability_key
from camera_config_utils import normalize_config
from config_manager import (
    DEFAULT_CONFIG,
    get_config,
    get_public_config,
    normalize_alert_outputs,
    save_config,
)

SITE_CONFIG_ENV = "SAFETYLENS_SITE_CONFIG"
DEFAULT_SITE_CONFIG_PATH = Path("/etc/safetylens/site.yaml")
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

MANAGED_TOP_LEVEL_KEYS = {
    "alert_outputs",
    "alert_routing",
    "automation_rules",
    "cameras",
    "global",
    "retention",
    "safety_rules",
    "scheduled_reports",
    "site",
    "vlm",
}

VOLATILE_ALERT_OUTPUT_FIELDS = {"lastError", "lastFiredAt", "lastTestAt", "status"}
VOLATILE_AUTOMATION_RULE_FIELDS = {"lastTriggered"}

CAMERA_REQUIRED_FIELDS = {"name", "zone"}
CAMERA_EVENT_POLICY_PRESET = "camera_event_default"
DEFAULT_CAMERA_EVENT_MESSAGE = "{severity} {violation_type} on {camera} in {zone}"
VALID_EVENT_SEVERITIES = {"P1", "P2", "P3", "P4"}
ACTION_BY_OUTPUT_ID = {
    "browser_sound": "play_sound",
    "telegram": "send_telegram",
    "email": "send_email",
    "webhook": "webhook",
    "pushover": "pushover",
    "relay": "trigger_relay",
    "relay_buzzer": "trigger_relay",
    "plc": "trigger_plc",
}


@dataclass
class SiteConfigResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] | None = None
    source_path: str | None = None


@dataclass
class SiteConfigPlan:
    source_path: str
    added_cameras: list[str]
    updated_cameras: list[str]
    removed_cameras: list[str]
    added_outputs: list[str]
    updated_outputs: list[str]
    removed_outputs: list[str]
    added_rules: list[str]
    updated_rules: list[str]
    removed_rules: list[str]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "added_cameras": self.added_cameras,
            "updated_cameras": self.updated_cameras,
            "removed_cameras": self.removed_cameras,
            "added_outputs": self.added_outputs,
            "updated_outputs": self.updated_outputs,
            "removed_outputs": self.removed_outputs,
            "added_rules": self.added_rules,
            "updated_rules": self.updated_rules,
            "removed_rules": self.removed_rules,
            "warnings": self.warnings,
        }

    def summary_lines(self) -> list[str]:
        sections = [
            ("Cameras", self.added_cameras, self.updated_cameras, self.removed_cameras),
            ("Alert outputs", self.added_outputs, self.updated_outputs, self.removed_outputs),
            ("Automation rules", self.added_rules, self.updated_rules, self.removed_rules),
        ]
        lines = [f"Site config: {self.source_path}"]
        for label, added, updated, removed in sections:
            lines.append(
                f"{label}: +{len(added)} add, ~{len(updated)} update, -{len(removed)} remove"
            )
            if added:
                lines.append(f"  add: {', '.join(added)}")
            if updated:
                lines.append(f"  update: {', '.join(updated)}")
            if removed:
                lines.append(f"  remove: {', '.join(removed)}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return lines


def default_site_config_path() -> Path:
    return Path(os.environ.get(SITE_CONFIG_ENV, str(DEFAULT_SITE_CONFIG_PATH))).expanduser()


def read_site_yaml(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else default_site_config_path()
    if not source.exists():
        raise FileNotFoundError(f"Site config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Site config must be a YAML object")
    return data


def load_site_config(
    path: str | Path | None = None,
    *,
    base_config: dict[str, Any] | None = None,
    strict_env: bool = True,
) -> SiteConfigResult:
    source = Path(path) if path else default_site_config_path()
    try:
        raw = read_site_yaml(source)
    except Exception as exc:
        return SiteConfigResult(ok=False, errors=[str(exc)], source_path=str(source))

    resolved, missing_env = resolve_env_refs(raw)
    errors, warnings = validate_site_document(resolved)
    if strict_env and missing_env:
        errors.extend(f"Missing environment variable: {name}" for name in missing_env)
    elif missing_env:
        warnings.extend(f"Unresolved environment reference: {name}" for name in missing_env)

    if errors:
        return SiteConfigResult(ok=False, errors=errors, warnings=warnings, source_path=str(source))

    try:
        config = site_document_to_runtime_config(resolved, base_config=base_config)
    except Exception as exc:
        return SiteConfigResult(ok=False, errors=[str(exc)], warnings=warnings, source_path=str(source))

    warnings.extend(validate_runtime_references(config))
    return SiteConfigResult(ok=True, warnings=warnings, config=config, source_path=str(source))


def resolve_env_refs(value: Any) -> tuple[Any, list[str]]:
    missing: list[str] = []

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: visit(val) for key, val in item.items()}
        if isinstance(item, list):
            return [visit(val) for val in item]
        if not isinstance(item, str):
            return item

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                if name not in missing:
                    missing.append(name)
                return match.group(0)
            return os.environ[name]

        return ENV_REF_RE.sub(replace, item)

    return visit(value), missing


def validate_site_document(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    unknown = sorted(set(doc) - MANAGED_TOP_LEVEL_KEYS - {"database", "model_server", "auth"})
    for key in unknown:
        warnings.append(f"Unknown top-level key will be preserved if already present: {key}")

    cameras = doc.get("cameras", {})
    if cameras is not None and not isinstance(cameras, (dict, list)):
        errors.append("cameras must be a mapping or list")
    else:
        for camera_id, camera in _iter_named_items(cameras, "cam"):
            if not isinstance(camera, dict):
                errors.append(f"cameras.{camera_id} must be an object")
                continue
            missing = [field for field in CAMERA_REQUIRED_FIELDS if not camera.get(field)]
            for field_name in missing:
                errors.append(f"cameras.{camera_id}.{field_name} is required")
            stream_type = camera.get("stream_type", camera.get("streamType", "rtsp"))
            if stream_type not in {"rtsp", "file"}:
                errors.append(f"cameras.{camera_id}.stream_type must be rtsp or file")
            if stream_type == "rtsp" and not any(
                camera.get(key) for key in ("rtsp_url", "rtspUrl", "host")
            ):
                errors.append(f"cameras.{camera_id} needs rtsp_url or host")
            if stream_type == "file" and not camera.get("video"):
                errors.append(f"cameras.{camera_id}.video is required for file streams")
            _validate_capability_windows(camera_id, camera, errors, warnings)
            _validate_capability_model_overrides(camera_id, camera, errors)
            _validate_camera_event_policy(camera_id, camera, errors)

    outputs = doc.get("alert_outputs", [])
    if outputs is not None and not isinstance(outputs, (dict, list)):
        errors.append("alert_outputs must be a mapping or list")

    rules = doc.get("automation_rules", [])
    if rules is not None and not isinstance(rules, (dict, list)):
        errors.append("automation_rules must be a mapping or list")

    return errors, warnings


def _validate_capability_windows(camera_id: str, camera: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    raw = camera.get("capability_windows")
    if raw is None:
        raw = camera.get("capability_active_windows")
    if raw is None:
        return
    if not isinstance(raw, (dict, list)):
        errors.append(f"cameras.{camera_id}.capability_windows must be a mapping or list")
        return

    entries: list[tuple[str, Any]]
    if isinstance(raw, dict):
        entries = [(str(key), value) for key, value in raw.items()]
    else:
        entries = [(str(index), value) for index, value in enumerate(raw, start=1)]

    for entry_id, entry in entries:
        if isinstance(raw, dict):
            capabilities = [entry_id]
            schedule = entry
        elif isinstance(entry, dict):
            raw_capabilities = entry.get("capabilities", entry.get("capability"))
            if isinstance(raw_capabilities, str):
                capabilities = [raw_capabilities]
            elif isinstance(raw_capabilities, list):
                capabilities = [str(value) for value in raw_capabilities if isinstance(value, str)]
            else:
                capabilities = []
            schedule = entry
        else:
            errors.append(f"cameras.{camera_id}.capability_windows[{entry_id}] must be an object")
            continue

        for capability in capabilities:
            if capability not in CAPABILITY_REGISTRY:
                warnings.append(f"cameras.{camera_id}.capability_windows.{entry_id} references unknown capability {capability}")

        if isinstance(schedule, list):
            windows = schedule
        elif isinstance(schedule, dict):
            mode = str(schedule.get("mode") or "detection")
            if mode not in {"detection", "detector", "detector_off", "alert_policy"}:
                errors.append(f"cameras.{camera_id}.capability_windows.{entry_id}.mode must be detection, detector, detector_off, or alert_policy")
            windows = schedule.get("windows", [])
        else:
            errors.append(f"cameras.{camera_id}.capability_windows.{entry_id} must be an object or list of windows")
            continue

        if not isinstance(windows, list):
            errors.append(f"cameras.{camera_id}.capability_windows.{entry_id}.windows must be a list")
            continue
        for window_index, window in enumerate(windows, start=1):
            if not isinstance(window, dict):
                errors.append(f"cameras.{camera_id}.capability_windows.{entry_id}.windows[{window_index}] must be an object")
                continue
            if "from" not in window or "to" not in window:
                errors.append(f"cameras.{camera_id}.capability_windows.{entry_id}.windows[{window_index}] needs from and to")


def _validate_capability_model_overrides(camera_id: str, camera: dict[str, Any], errors: list[str]) -> None:
    raw = camera.get("capability_model_overrides")
    if raw is None:
        raw = camera.get("model_overrides")
    if raw is None:
        return
    if not isinstance(raw, dict):
        errors.append(f"cameras.{camera_id}.capability_model_overrides must be a mapping")
        return

    for raw_capability, raw_override in raw.items():
        if not isinstance(raw_capability, str):
            errors.append(f"cameras.{camera_id}.capability_model_overrides keys must be capability names")
            continue
        capability = normalize_capability_key(raw_capability)
        if not capability:
            errors.append(f"cameras.{camera_id}.capability_model_overrides.{raw_capability} references unknown capability")
            continue

        model_key = raw_override.get("model_key") if isinstance(raw_override, dict) else raw_override
        if not isinstance(model_key, str):
            errors.append(f"cameras.{camera_id}.capability_model_overrides.{raw_capability} must be a model key or object with model_key")
            continue

        allowed = ALLOWED_CAPABILITY_MODEL_OVERRIDES.get(capability, set())
        if model_key not in allowed:
            allowed_text = ", ".join(sorted(allowed)) if allowed else "no overrides"
            errors.append(
                f"cameras.{camera_id}.capability_model_overrides.{raw_capability}={model_key} is not supported; allowed: {allowed_text}"
            )


def _validate_camera_event_policy(camera_id: str, camera: dict[str, Any], errors: list[str]) -> None:
    raw = _camera_event_policy(camera)
    if raw is None:
        return
    path = f"cameras.{camera_id}.event_policy"
    if not isinstance(raw, dict):
        errors.append(f"{path} must be an object")
        return

    enabled = _as_bool(raw.get("enabled"), default=True)
    raw_output_ids = _get_alias(raw, "output_ids", "outputIds", "channels", "channel_ids")
    output_ids, output_error = _coerce_string_list(raw_output_ids)
    if output_error:
        errors.append(f"{path}.output_ids must be a string or list")
    if enabled and not output_ids:
        errors.append(f"{path}.output_ids is required when enabled")

    severity = _get_alias(raw, "severity", "alert_severity", "alertSeverity")
    if severity is not None:
        normalized = str(severity).upper()
        if normalized != "INHERIT" and normalized not in VALID_EVENT_SEVERITIES:
            errors.append(f"{path}.severity must be P1, P2, P3, P4, or inherit")

    priority = _get_alias(raw, "priority")
    if priority is not None:
        try:
            if int(priority) < 1:
                errors.append(f"{path}.priority must be 1 or higher")
        except (TypeError, ValueError):
            errors.append(f"{path}.priority must be an integer")

    cooldown = _get_alias(raw, "cooldown_seconds", "cooldownSeconds")
    if cooldown is not None:
        try:
            if int(cooldown) < 0:
                errors.append(f"{path}.cooldown_seconds must be 0 or higher")
        except (TypeError, ValueError):
            errors.append(f"{path}.cooldown_seconds must be an integer")

    min_confidence = _get_alias(raw, "min_confidence", "minConfidence")
    if min_confidence is not None:
        try:
            confidence = float(min_confidence)
            if confidence < 0 or confidence > 1:
                errors.append(f"{path}.min_confidence must be between 0 and 1")
        except (TypeError, ValueError):
            errors.append(f"{path}.min_confidence must be a number")

    schedule = _get_alias(raw, "schedule", "active_windows", "activeWindows")
    if schedule is not None:
        _validate_schedule_windows(f"{path}.schedule", schedule, errors)


def _validate_schedule_windows(path: str, schedule: Any, errors: list[str]) -> None:
    if isinstance(schedule, dict):
        windows = schedule.get("windows", [])
    elif isinstance(schedule, list):
        windows = schedule
    else:
        errors.append(f"{path} must be an object or list of windows")
        return

    if not isinstance(windows, list):
        errors.append(f"{path}.windows must be a list")
        return

    for window_index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            errors.append(f"{path}.windows[{window_index}] must be an object")
            continue
        if "from" not in window or "to" not in window:
            errors.append(f"{path}.windows[{window_index}] needs from and to")


def validate_runtime_references(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    camera_ids = set(config.get("cameras", {}))
    output_ids = {output.get("id") for output in config.get("alert_outputs", [])}

    for rule in config.get("automation_rules", []):
        rule_id = rule.get("id") or rule.get("name", "unnamed")
        for camera_id in rule.get("cameras") or []:
            if camera_id not in camera_ids:
                warnings.append(f"automation_rules.{rule_id} references unknown camera {camera_id}")
        for output_id in rule.get("outputIds") or rule.get("output_ids") or []:
            if output_id not in output_ids:
                warnings.append(f"automation_rules.{rule_id} references unknown output {output_id}")
    return warnings


def site_document_to_runtime_config(
    doc: dict[str, Any],
    *,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config if base_config is not None else get_config())
    if not config:
        config = copy.deepcopy(DEFAULT_CONFIG)
    merge_existing = bool((doc.get("site") or {}).get("merge_existing"))

    for key in ("site", "global", "vlm", "retention", "alert_routing", "scheduled_reports"):
        if key in doc and isinstance(doc[key], dict):
            config[key] = _deep_merge(config.get(key, {}), doc[key])

    for key in ("database", "model_server", "auth"):
        if key in doc and isinstance(doc[key], dict):
            config[key] = _deep_merge(config.get(key, {}), doc[key])

    if "alert_outputs" in doc:
        incoming = _normalize_list_section(doc["alert_outputs"])
        config["alert_outputs"] = _merge_list_by_id(config.get("alert_outputs", []), incoming) if merge_existing else incoming

    if "safety_rules" in doc:
        incoming = _normalize_list_section(doc["safety_rules"])
        config["safety_rules"] = _merge_list_by_id(config.get("safety_rules", []), incoming) if merge_existing else incoming

    if "automation_rules" in doc:
        incoming = _normalize_automation_rules(doc["automation_rules"])
    else:
        incoming = []

    camera_event_rules = _camera_event_policy_rules_from_section(doc.get("cameras")) if "cameras" in doc else []
    if "automation_rules" in doc or camera_event_rules:
        incoming = [*incoming, *camera_event_rules]
        config["automation_rules"] = _merge_list_by_id(config.get("automation_rules", []), incoming) if merge_existing else incoming

    if "cameras" in doc:
        incoming = _normalize_cameras(doc["cameras"])
        config["cameras"] = _merge_dict_by_key(config.get("cameras", {}), incoming) if merge_existing else incoming

    config.setdefault("site", {})
    config["site"]["config_source"] = "yaml"
    config, _outputs_changed = normalize_alert_outputs(config)
    config, _normalized = normalize_config(config)
    return config


def build_plan(path: str | Path | None = None, *, base_config: dict[str, Any] | None = None) -> SiteConfigResult:
    current = copy.deepcopy(base_config if base_config is not None else get_config())
    result = load_site_config(path, base_config=current)
    if not result.ok or result.config is None:
        return result

    desired = result.config
    plan = SiteConfigPlan(
        source_path=result.source_path or str(path or default_site_config_path()),
        added_cameras=_added(current.get("cameras", {}), desired.get("cameras", {})),
        updated_cameras=_updated(current.get("cameras", {}), desired.get("cameras", {})),
        removed_cameras=_removed(current.get("cameras", {}), desired.get("cameras", {})),
        added_outputs=_added(_by_id(current.get("alert_outputs", [])), _by_id(desired.get("alert_outputs", []))),
        updated_outputs=_updated(
            _by_id(_without_volatile_output_fields(current.get("alert_outputs", []))),
            _by_id(_without_volatile_output_fields(desired.get("alert_outputs", []))),
        ),
        removed_outputs=_removed(_by_id(current.get("alert_outputs", [])), _by_id(desired.get("alert_outputs", []))),
        added_rules=_added(_by_id(current.get("automation_rules", [])), _by_id(desired.get("automation_rules", []))),
        updated_rules=_updated(
            _by_id(_without_volatile_rule_fields(current.get("automation_rules", []))),
            _by_id(_without_volatile_rule_fields(desired.get("automation_rules", []))),
        ),
        removed_rules=_removed(_by_id(current.get("automation_rules", [])), _by_id(desired.get("automation_rules", []))),
        warnings=result.warnings,
    )
    result.config = {"desired_config": desired, "plan": plan.as_dict(), "summary": plan.summary_lines()}
    return result


def apply_site_config(path: str | Path | None = None) -> SiteConfigResult:
    result = load_site_config(path, base_config=get_config())
    if result.ok and result.config is not None:
        save_config(result.config)
    return result


def export_site_config(path: str | Path | None = None, *, redacted: bool = True) -> Path:
    target = Path(path) if path else default_site_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    config = get_public_config() if redacted else get_config()
    doc = {key: copy.deepcopy(config[key]) for key in sorted(MANAGED_TOP_LEVEL_KEYS) if key in config}
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False, allow_unicode=False)
    return target


def result_to_json(result: SiteConfigResult) -> str:
    return json.dumps(
        {
            "ok": result.ok,
            "errors": result.errors,
            "warnings": result.warnings,
            "source_path": result.source_path,
            "config": result.config,
        },
        indent=2,
        default=str,
    )


def _deep_merge(base: Any, incoming: Any) -> Any:
    if isinstance(base, dict) and isinstance(incoming, dict):
        merged = copy.deepcopy(base)
        for key, value in incoming.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(incoming)


def _iter_named_items(section: Any, prefix: str):
    if isinstance(section, dict):
        for key, value in section.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("id", key)
                yield key, item
            else:
                yield key, value
    elif isinstance(section, list):
        for index, value in enumerate(section, start=1):
            if isinstance(value, dict):
                item = dict(value)
                item_id = item.get("id") or f"{prefix}{index}"
                item["id"] = item_id
                yield item_id, item
            else:
                yield f"{prefix}{index}", value


def _normalize_list_section(section: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item_id, item in _iter_named_items(section, "item"):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("id", item_id)
        items.append(normalized)
    return items


def _normalize_automation_rules(section: Any) -> list[dict[str, Any]]:
    rules = []
    for rule in _normalize_list_section(section):
        if "then_actions" in rule and "thenActions" not in rule:
            rule["thenActions"] = rule.pop("then_actions")
        if "else_actions" in rule and "elseActions" not in rule:
            rule["elseActions"] = rule.pop("else_actions")
        if "cooldown_seconds" in rule and "cooldownSeconds" not in rule:
            rule["cooldownSeconds"] = rule.pop("cooldown_seconds")
        if "output_ids" in rule and "outputIds" not in rule:
            rule["outputIds"] = rule.pop("output_ids")
        if "message_template" in rule and "messageTemplate" not in rule:
            rule["messageTemplate"] = rule.pop("message_template")
        rule.setdefault("conditions", [])
        rule.setdefault("thenActions", [])
        rule.setdefault("elseActions", [])
        rule.setdefault("cameras", [])
        rule.setdefault("cooldownSeconds", 60)
        rule.setdefault("priority", 5)
        rule.setdefault("enabled", True)
        rule.setdefault("lastTriggered", None)
        rule.setdefault("preset", None)
        rules.append(rule)
    return rules


def _camera_event_policy_rules_from_section(section: Any) -> list[dict[str, Any]]:
    if not isinstance(section, (dict, list)):
        return []
    rules: list[dict[str, Any]] = []
    for camera_id, camera in _iter_named_items(section, "cam"):
        if not isinstance(camera, dict):
            continue
        policy = _camera_event_policy(camera)
        if not isinstance(policy, dict):
            continue
        rules.append(_camera_event_policy_rule(str(camera_id), camera, policy))
    return _normalize_automation_rules(rules)


def _camera_event_policy(camera: dict[str, Any]) -> Any:
    if "event_policy" in camera:
        return camera["event_policy"]
    if "eventPolicy" in camera:
        return camera["eventPolicy"]
    return None


def _camera_event_policy_rule(camera_id: str, camera: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    severity = _normalized_severity(_get_alias(policy, "severity", "alert_severity", "alertSeverity"))
    output_ids, _output_error = _coerce_string_list(
        _get_alias(policy, "output_ids", "outputIds", "channels", "channel_ids")
    )
    min_confidence = _as_float(_get_alias(policy, "min_confidence", "minConfidence"), default=0.0)
    conditions = copy.deepcopy(policy.get("conditions") if isinstance(policy.get("conditions"), list) else [])
    if min_confidence > 0:
        conditions.append({"type": "confidence_above", "params": {"value": _format_number(min_confidence)}})

    then_actions = [{"type": "create_alert", "params": {"severity": severity} if severity else {}}]
    for output_id in output_ids:
        action_type = ACTION_BY_OUTPUT_ID.get(output_id)
        if action_type and not any(action.get("type") == action_type for action in then_actions):
            then_actions.append({"type": action_type, "params": {}})

    camera_name = str(camera.get("name") or camera_id)
    camera_zone = str(camera.get("zone") or "Unknown")
    schedule = _normalize_event_schedule(_get_alias(policy, "schedule", "active_windows", "activeWindows"))
    return {
        "id": str(policy.get("id") or f"camera_event_{camera_id}"),
        "name": str(policy.get("name") or f"Default Event Policy - {camera_name}"),
        "description": str(
            policy.get("description") or f"Camera-level event routing for {camera_name} in {camera_zone}."
        ),
        "enabled": _as_bool(policy.get("enabled"), default=True),
        "trigger": str(policy.get("trigger") or "detection"),
        "cameras": [camera_id],
        "conditions": conditions,
        "thenActions": then_actions,
        "elseActions": copy.deepcopy(policy.get("elseActions") or policy.get("else_actions") or []),
        "cooldownSeconds": max(0, _as_int(_get_alias(policy, "cooldown_seconds", "cooldownSeconds"), default=60)),
        "priority": max(1, _as_int(_get_alias(policy, "priority"), default=5)),
        "severity": severity,
        "outputIds": output_ids,
        "messageTemplate": str(
            _get_alias(policy, "message_template", "messageTemplate") or DEFAULT_CAMERA_EVENT_MESSAGE
        ),
        "schedule": schedule,
        "preset": CAMERA_EVENT_POLICY_PRESET,
    }


def _normalize_event_schedule(schedule: Any) -> dict[str, Any] | None:
    if isinstance(schedule, dict):
        windows = schedule.get("windows")
        if isinstance(windows, list) and windows:
            normalized = copy.deepcopy(schedule)
            normalized["windows"] = windows
            return normalized
        return None
    if isinstance(schedule, list) and schedule:
        return {"windows": copy.deepcopy(schedule)}
    return None


def _normalize_cameras(section: Any) -> dict[str, dict[str, Any]]:
    cameras: dict[str, dict[str, Any]] = {}
    for camera_id, camera in _iter_named_items(section, "cam"):
        if not isinstance(camera, dict):
            continue
        normalized = dict(camera)
        normalized.pop("id", None)
        if "streamType" in normalized and "stream_type" not in normalized:
            normalized["stream_type"] = normalized.pop("streamType")
        if "rtspUrl" in normalized and "rtsp_url" not in normalized:
            normalized["rtsp_url"] = normalized.pop("rtspUrl")
        normalized.pop("event_policy", None)
        normalized.pop("eventPolicy", None)
        connection = normalized.pop("connection", None)
        if isinstance(connection, dict):
            normalized.update({key: value for key, value in connection.items() if value not in (None, "")})
        normalized.setdefault("stream_type", "rtsp")
        normalized.setdefault("enabled", True)
        normalized.setdefault("fps", DEFAULT_CONFIG["global"]["target_fps"])
        normalized.setdefault("rules", [])
        normalized.setdefault("capabilities", [])
        normalized.setdefault("safety_rule_ids", [])
        normalized.setdefault("custom_long_tail_terms", [])
        cameras[str(camera_id)] = normalized
    return cameras


def _get_alias(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _coerce_string_list(value: Any) -> tuple[list[str], bool]:
    if value is None:
        return [], False
    if isinstance(value, str):
        return [value], False
    if not isinstance(value, list):
        return [], True
    return [str(item) for item in value if str(item).strip()], False


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_severity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).upper()
    if normalized == "INHERIT":
        return None
    if normalized in VALID_EVENT_SEVERITIES:
        return normalized
    return None


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _merge_list_by_id(existing: Any, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_id = _by_id(existing if isinstance(existing, list) else [])
    order = [str(item.get("id")) for item in existing if isinstance(item, dict) and item.get("id")] if isinstance(existing, list) else []
    for item in incoming:
        item_id = str(item.get("id"))
        if not item_id:
            continue
        current = merged_by_id.get(item_id, {})
        merged_by_id[item_id] = _deep_merge(current, item)
        if item_id not in order:
            order.append(item_id)
    return [merged_by_id[item_id] for item_id in order if item_id in merged_by_id]


def _merge_dict_by_key(existing: Any, incoming: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = copy.deepcopy(existing if isinstance(existing, dict) else {})
    for key, value in incoming.items():
        merged[key] = _deep_merge(merged.get(key, {}), value)
    return merged


def _by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in items if item.get("id")}


def _without_volatile_output_fields(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        output = copy.deepcopy(item)
        for field_name in VOLATILE_ALERT_OUTPUT_FIELDS:
            output.pop(field_name, None)
        normalized.append(output)
    return normalized


def _without_volatile_rule_fields(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rule = copy.deepcopy(item)
        for field_name in VOLATILE_AUTOMATION_RULE_FIELDS:
            rule.pop(field_name, None)
        normalized.append(rule)
    return normalized


def _added(current: dict[str, Any], desired: dict[str, Any]) -> list[str]:
    return sorted(set(desired) - set(current))


def _removed(current: dict[str, Any], desired: dict[str, Any]) -> list[str]:
    return sorted(set(current) - set(desired))


def _updated(current: dict[str, Any], desired: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(current) & set(desired) if current[key] != desired[key])
