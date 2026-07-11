"""Runtime evaluation for camera-scoped automation policies."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

from config_manager import get_config, save_config

DEFAULT_MESSAGE_TEMPLATE = "{severity} {violation_type} on {camera} in {zone}"
DAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

_last_triggered_by_key: dict[str, float] = {}


@dataclass
class PolicyDecision:
    rule_id: str
    rule_name: str
    severity: str
    priority: int
    message: str
    output_ids: list[str] | None
    cooldown_seconds: int
    fallback: bool = False


def evaluate_candidate(
    candidate: dict[str, Any],
    camera: dict[str, Any],
    *,
    camera_id: str,
    detections: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[PolicyDecision]:
    """Return policy decisions for a confirmed detection candidate.

    If no executable automation rule is scoped to the camera, the existing
    safety-rule behavior is preserved with a fallback decision.
    """
    cfg = cfg or get_config()
    event = _build_event(candidate, camera, camera_id=camera_id, detections=detections or [], now=now, cfg=cfg)
    rules = sorted(
        [rule for rule in cfg.get("automation_rules", []) if _rule_scopes_camera(rule, camera_id)],
        key=lambda rule: int(rule.get("priority") or 0),
        reverse=True,
    )
    executable_rules = [rule for rule in rules if rule.get("enabled", True)]
    if not executable_rules:
        return [_fallback_decision(candidate)]

    decisions: list[PolicyDecision] = []
    matched_policy = False
    for rule in executable_rules:
        if not _trigger_matches(rule.get("trigger", "detection"), event):
            continue
        if not all(_condition_matches(condition, event) for condition in rule.get("conditions", [])):
            continue
        matched_policy = True
        if not _schedule_matches(rule.get("schedule") or camera.get("active_windows"), event["now"], cfg):
            continue
        cooldown = int(rule.get("cooldownSeconds") or rule.get("cooldown_seconds") or 60)
        cooldown_key = f"{camera_id}:{rule.get('id')}"
        monotonic_now = time.monotonic()
        if monotonic_now - _last_triggered_by_key.get(cooldown_key, 0) < cooldown:
            continue
        severity = _severity_for_rule(rule, candidate)
        output_ids = _output_ids_for_rule(rule)
        message = render_template(rule.get("messageTemplate") or rule.get("message_template") or DEFAULT_MESSAGE_TEMPLATE, event | {"severity": severity})
        decisions.append(
            PolicyDecision(
                rule_id=str(rule.get("id") or rule.get("name")),
                rule_name=rule.get("name") or candidate.get("rule", "Alert"),
                severity=severity,
                priority=int(rule.get("priority") or 5),
                message=message,
                output_ids=output_ids,
                cooldown_seconds=cooldown,
            )
        )
        _last_triggered_by_key[cooldown_key] = monotonic_now
    if decisions or matched_policy:
        return decisions
    return [_fallback_decision(candidate)]


def mark_rule_triggered(rule_id: str, *, cfg: dict[str, Any] | None = None) -> None:
    """Persist lastTriggered for UI visibility. Best-effort and non-fatal."""
    if not rule_id:
        return
    cfg = cfg or get_config()
    changed = False
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for rule in cfg.get("automation_rules", []):
        if rule.get("id") == rule_id:
            rule["lastTriggered"] = timestamp
            changed = True
            break
    if changed:
        try:
            save_config(cfg)
        except Exception:
            # Alert creation should not fail because the UI timestamp could not
            # be persisted.
            pass


def render_template(template: str, data: dict[str, Any]) -> str:
    rendered = template or ""
    aliases = {
        "violation_type": data.get("rule"),
        "camera": data.get("cameraName"),
        "camera_id": data.get("cameraId"),
        "zone": data.get("zone"),
        "confidence": data.get("confidence"),
        "timestamp": data.get("timestamp"),
        "severity": data.get("severity"),
        "description": data.get("description"),
    }
    for key, value in aliases.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


def _build_event(
    candidate: dict[str, Any],
    camera: dict[str, Any],
    *,
    camera_id: str,
    detections: list[dict[str, Any]],
    now: datetime | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ts = now or datetime.now(_site_tz(cfg))
    rule_name = candidate.get("rule", "Alert")
    event_classes = set(_normalize_class(value) for value in candidate.get("classes", []))
    for detection in detections:
        if detection.get("class"):
            event_classes.add(_normalize_class(detection["class"]))
    event_classes.add(_normalize_class(rule_name))
    if str(rule_name).lower().startswith("missing "):
        missing_item = str(rule_name)[8:].strip()
        if missing_item:
            normalized_item = _normalize_class(missing_item)
            event_classes.add(f"missing {normalized_item}")
            event_classes.add(f"no {normalized_item}")
    trigger = "zone_enter" if rule_name == "Zone Intrusion" else "detection"
    return {
        "trigger": trigger,
        "now": ts,
        "timestamp": ts.isoformat(),
        "cameraId": camera_id,
        "cameraName": camera.get("name", camera_id),
        "zone": candidate.get("zone") or camera.get("zone", "Unknown"),
        "rule": rule_name,
        "confidence": float(candidate.get("confidence") or 0),
        "description": candidate.get("description", ""),
        "classes": event_classes,
        "count": int(candidate.get("count") or len(detections) or 1),
        "candidate": candidate,
        "camera": camera,
    }


def _fallback_decision(candidate: dict[str, Any]) -> PolicyDecision:
    severity = candidate.get("severity", "P4")
    return PolicyDecision(
        rule_id="",
        rule_name=candidate.get("rule", "Alert"),
        severity=severity,
        priority=5,
        message=render_template(DEFAULT_MESSAGE_TEMPLATE, {"severity": severity, **candidate}),
        output_ids=None,
        cooldown_seconds=0,
        fallback=True,
    )


def _rule_scopes_camera(rule: dict[str, Any], camera_id: str) -> bool:
    cameras = rule.get("cameras") or []
    return not cameras or camera_id in cameras


def _trigger_matches(rule_trigger: str, event: dict[str, Any]) -> bool:
    if rule_trigger == event["trigger"]:
        return True
    if rule_trigger == "detection" and event["trigger"] in {"detection", "zone_enter"}:
        return True
    if rule_trigger == "count_threshold":
        return True
    return False


def _condition_matches(condition: dict[str, Any], event: dict[str, Any]) -> bool:
    ctype = condition.get("type", "")
    params = condition.get("params") or {}
    if ctype == "class_is":
        expected = {_normalize_class(item) for item in _split_csv(params.get("classes", ""))}
        return bool(expected & event["classes"])
    if ctype == "zone_is":
        expected = _normalize_class(params.get("zone", ""))
        return expected in {_normalize_class(event.get("zone", "")), _normalize_class(event["camera"].get("zone", ""))}
    if ctype == "confidence_above":
        try:
            return float(event.get("confidence") or 0) >= float(params.get("value") or 0)
        except (TypeError, ValueError):
            return False
    if ctype == "count_exceeds":
        try:
            return int(event.get("count") or 0) > int(params.get("count") or 0)
        except (TypeError, ValueError):
            return False
    if ctype == "time_between":
        return _time_between(event["now"].time(), params.get("from", "00:00"), params.get("to", "23:59"))
    # Plate/face conditions are kept for UI compatibility. They do not match
    # generic safety detection events.
    if ctype in {"plate_in_list", "face_in_group"}:
        return False
    return True


def _schedule_matches(schedule: Any, now: datetime, cfg: dict[str, Any]) -> bool:
    if not schedule:
        return True
    windows = schedule.get("windows") if isinstance(schedule, dict) else schedule
    if not isinstance(windows, list) or not windows:
        return True
    local_now = now.astimezone(_site_tz(cfg))
    weekday = local_now.weekday()
    for window in windows:
        if not isinstance(window, dict):
            continue
        days = window.get("days") or window.get("weekdays") or []
        day_indexes = {_day_index(day) for day in days}
        day_indexes.discard(None)
        if day_indexes and weekday not in day_indexes:
            continue
        if _time_between(local_now.time(), window.get("from", "00:00"), window.get("to", "23:59")):
            return True
    return False


def _severity_for_rule(rule: dict[str, Any], candidate: dict[str, Any]) -> str:
    if rule.get("severity"):
        return str(rule["severity"])
    for action in rule.get("thenActions", []):
        if action.get("type") == "create_alert":
            severity = (action.get("params") or {}).get("severity")
            if severity:
                return str(severity)
    return candidate.get("severity", "P4")


def _output_ids_for_rule(rule: dict[str, Any]) -> list[str] | None:
    explicit = rule.get("outputIds") or rule.get("output_ids")
    if explicit:
        return [str(item) for item in explicit]
    output_ids: list[str] = []
    action_map = {
        "send_telegram": "telegram",
        "send_email": "email",
        "webhook": "webhook",
        "play_sound": "browser_sound",
        "trigger_plc": "plc",
        "trigger_relay": "relay_buzzer",
        "relay": "relay_buzzer",
        "pushover": "pushover",
    }
    for action in rule.get("thenActions", []):
        output_id = action_map.get(action.get("type"))
        if output_id and output_id not in output_ids:
            output_ids.append(output_id)
    return output_ids or None


def _site_tz(cfg: dict[str, Any]) -> ZoneInfo:
    raw = cfg.get("site", {}).get("timezone") or os.environ.get("SAFETYLENS_SITE_TIMEZONE") or "Asia/Kolkata"
    try:
        return ZoneInfo(str(raw))
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _normalize_class(value: str) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ").strip()


def _time_between(value: dt_time, start: str, end: str) -> bool:
    start_time = _parse_time(start, dt_time(0, 0))
    end_time = _parse_time(end, dt_time(23, 59))
    if start_time <= end_time:
        return start_time <= value <= end_time
    return value >= start_time or value <= end_time


def _parse_time(value: str, default: dt_time) -> dt_time:
    try:
        hour, minute = str(value).split(":", 1)
        return dt_time(int(hour), int(minute[:2]))
    except Exception:
        return default


def _day_index(value: Any) -> int | None:
    if isinstance(value, int) and 0 <= value <= 6:
        return value
    return DAY_ALIASES.get(str(value).lower())
