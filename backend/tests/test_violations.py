"""Tests for check_violations and check_yoloe_violations from detection.py."""

from unittest import mock

import pytest

from detection import (
    check_violations,
    check_yoloe_violations,
    SEVERITY_COOLDOWN_MULT,
)
from routers.safety_rules import DEFAULT_SAFETY_RULES


def _det(cls, conf=0.8):
    """Shorthand to build a detection dict."""
    return {"class": cls, "confidence": conf}


def _cfg_with_alert_classes(cam_id, alert_classes):
    """Build a mock config using old alert_classes (backward compat path)."""
    return {
        "cameras": {cam_id: {"alert_classes": alert_classes}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }


def _cfg_with_safety_rule_ids(cam_id, rule_ids):
    """Build a mock config using new safety_rule_ids."""
    return {
        "cameras": {cam_id: {"safety_rule_ids": rule_ids}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }


# ── check_violations — mobile phone ─────────────────────────────────────────

@mock.patch("detection.get_config")
def test_check_violations_mobile_phone(mock_cfg):
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["mobile_phone", "animal_intrusion"])
    dets = [_det("person"), _det("cell phone", 0.75)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Mobile Phone Usage"
    assert violations[0]["severity"] == "P3"
    assert violations[0]["confidence"] == 0.75


# ── check_violations — animal intrusion ──────────────────────────────────────

@mock.patch("detection.get_config")
def test_check_violations_animal_intrusion(mock_cfg):
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["mobile_phone", "animal_intrusion"])
    dets = [_det("person"), _det("dog", 0.65)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Animal Intrusion"
    assert "1 dog(s)" in violations[0]["description"]


# ── check_violations — person_detected when enabled ──────────────────────────

@mock.patch("detection.get_config")
def test_check_violations_person_detected_when_enabled(mock_cfg):
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["person_detected"])
    dets = [_det("person", 0.9)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Person Detected"
    assert violations[0]["severity"] == "P4"


# ── check_violations — person_detected NOT in defaults ───────────────────────

@mock.patch("detection.get_config")
def test_check_violations_person_detected_when_disabled(mock_cfg):
    """Default camera config does NOT include person_detected."""
    mock_cfg.return_value = {
        "cameras": {"cam1": {}},  # no alert_classes/safety_rule_ids => defaults to mobile_phone + animal_intrusion
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [_det("person", 0.9)]
    violations = check_violations(dets, "cam1")
    # person_detected is not in default alert_classes, so no violation
    assert len(violations) == 0


# ── check_violations — no detections ─────────────────────────────────────────

@mock.patch("detection.get_config")
def test_check_violations_no_detections_returns_empty(mock_cfg):
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["mobile_phone", "animal_intrusion"])
    violations = check_violations([], "cam1")
    assert violations == []


# ── check_violations — alert_classes filtering ───────────────────────────────

@mock.patch("detection.get_config")
def test_check_violations_alert_classes_filtering(mock_cfg):
    """Only rules in alert_classes should fire."""
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["animal_intrusion"])
    # Both phone and dog present, but only animal_intrusion enabled
    dets = [_det("person"), _det("cell phone", 0.8), _det("dog", 0.7)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Animal Intrusion"


@mock.patch("detection.get_config")
def test_check_violations_vehicle_detected(mock_cfg):
    mock_cfg.return_value = _cfg_with_alert_classes("cam1", ["vehicle_detected"])
    dets = [_det("truck", 0.85)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Vehicle Detected"
    assert violations[0]["severity"] == "P4"


# ── SEVERITY_COOLDOWN_MULT ───────────────────────────────────────────────────

def test_severity_cooldown_multipliers():
    assert SEVERITY_COOLDOWN_MULT["P1"] == 1
    assert SEVERITY_COOLDOWN_MULT["P2"] == 1
    assert SEVERITY_COOLDOWN_MULT["P3"] == 2
    assert SEVERITY_COOLDOWN_MULT["P4"] == 5

    # P1/P2 have base cooldown, P3 is 2x, P4 is 5x
    base = 60
    assert base * SEVERITY_COOLDOWN_MULT["P1"] == 60
    assert base * SEVERITY_COOLDOWN_MULT["P3"] == 120
    assert base * SEVERITY_COOLDOWN_MULT["P4"] == 300


# ── check_yoloe_violations ───────────────────────────────────────────────────

@mock.patch("detection.get_config")
def test_check_yoloe_violations_missing_ppe(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"yoloe_classes": ["person", "hairnet", "gloves"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    # person detected but no hairnet, no gloves
    dets = [_det("person", 0.9)]
    violations = check_yoloe_violations(dets, "cam3")
    assert len(violations) == 2
    rules = {v["rule"] for v in violations}
    assert "Missing hairnet" in rules
    assert "Missing gloves" in rules


@mock.patch("detection.get_config")
def test_check_yoloe_violations_no_persons(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"yoloe_classes": ["person", "hairnet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    # No person detected — no violations even if no hairnet
    dets = [_det("hairnet", 0.8)]
    violations = check_yoloe_violations(dets, "cam3")
    assert violations == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_all_ppe_present(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"yoloe_classes": ["person", "hairnet", "gloves"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [_det("person", 0.9), _det("hairnet", 0.85), _det("gloves", 0.8)]
    violations = check_yoloe_violations(dets, "cam3")
    assert violations == []
