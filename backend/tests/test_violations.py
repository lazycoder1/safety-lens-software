"""Tests for check_violations and check_yoloe_violations from detection.py."""

import json
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pytest

from detection import (
    _fall_analysis,
    _is_fall,
    check_fall_detections,
    check_violations,
    check_yoloe_violations,
    SEVERITY_COOLDOWN_MULT,
)
from routers.safety_rules import DEFAULT_SAFETY_RULES


def _det(cls, conf=0.8, bbox=None):
    """Shorthand to build a detection dict."""
    detection = {"class": cls, "confidence": conf}
    if bbox is not None:
        detection["bbox"] = bbox
    return detection


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


def _pose_keypoints(*, confidence: float = 0.9, horizontal: bool = False, exercise: bool = False):
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    if exercise:
        keypoints[5] = [45.0, 48.0, confidence]
        keypoints[6] = [55.0, 48.0, confidence]
        keypoints[11] = [105.0, 80.0, confidence]
        keypoints[12] = [115.0, 80.0, confidence]
        keypoints[13] = [130.0, 40.0, confidence]
        keypoints[14] = [140.0, 42.0, confidence]
        return keypoints
    if horizontal:
        keypoints[5] = [20.0, 50.0, confidence]
        keypoints[6] = [30.0, 50.0, confidence]
        keypoints[11] = [90.0, 55.0, confidence]
        keypoints[12] = [100.0, 55.0, confidence]
    else:
        keypoints[5] = [50.0, 20.0, confidence]
        keypoints[6] = [70.0, 20.0, confidence]
        keypoints[11] = [50.0, 100.0, confidence]
        keypoints[12] = [70.0, 100.0, confidence]
    keypoints[13] = [120.0, 58.0, confidence]
    return keypoints


class _ArrayTensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeBoxes:
    def __init__(self, bbox, confidence=0.9):
        self.conf = [confidence]
        self.xyxy = [bbox]

    def __len__(self):
        return len(self.conf)


def _pose_results(keypoints, bbox=None, confidence=0.9):
    return [
        SimpleNamespace(
            keypoints=SimpleNamespace(data=[_ArrayTensor(keypoints)]),
            boxes=_FakeBoxes(bbox or [0, 0, 200, 100], confidence),
        )
    ]


def test_is_fall_rejects_wide_box_without_usable_pose():
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    assert _is_fall(keypoints, [0, 0, 200, 80]) is False
    analysis = _fall_analysis(keypoints, [0, 0, 200, 80])
    assert analysis["reason"] == "insufficient_pose"
    assert analysis["usablePose"] is False


def test_is_fall_accepts_horizontal_body_with_usable_pose():
    assert _is_fall(_pose_keypoints(horizontal=True), [0, 0, 200, 80]) is True
    analysis = _fall_analysis(_pose_keypoints(horizontal=True), [0, 0, 200, 80])
    assert analysis["reason"] == "wide_body_box"
    assert analysis["usablePose"] is True


def test_is_fall_rejects_upright_body_with_diagnostics():
    analysis = _fall_analysis(_pose_keypoints(), [0, 0, 80, 200])
    assert analysis["isFall"] is False
    assert analysis["reason"] == "upright_or_not_fall"
    assert analysis["usablePose"] is True


def test_fall_analysis_marks_floor_exercise_pose():
    analysis = _fall_analysis(_pose_keypoints(exercise=True), [0, 0, 200, 100])
    assert analysis["isFall"] is True
    assert analysis["floorExercisePose"] is True
    assert analysis["raisedKneeCount"] == 2


def test_fall_analysis_marks_wide_horizontal_situp_pose_as_floor_exercise():
    keypoints = _pose_keypoints(horizontal=True)
    keypoints[5] = [330.0, 590.0, 0.99]
    keypoints[6] = [335.0, 600.0, 0.99]
    keypoints[11] = [560.0, 590.0, 0.99]
    keypoints[12] = [565.0, 605.0, 0.99]
    keypoints[13] = [775.0, 535.0, 0.98]
    keypoints[14] = [775.0, 550.0, 0.98]

    analysis = _fall_analysis(keypoints, [140, 460, 960, 685])

    assert analysis["isFall"] is True
    assert analysis["reason"] == "wide_body_box"
    assert analysis["raisedKneeCount"] == 2
    assert analysis["floorExercisePose"] is True


@mock.patch("detection.get_config")
def test_check_fall_detections_suppresses_floor_exercise_when_configured(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {
            "therapy_cam": {
                "fall_detection": {
                    "suppress_floor_exercise": True,
                },
            },
        },
    }
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    candidates = check_fall_detections(
        _pose_results(_pose_keypoints(exercise=True), [0, 0, 200, 100]),
        "therapy_cam",
        frame,
    )
    assert candidates == []


@mock.patch("detection.get_config")
def test_check_fall_detections_keeps_default_person_down_behavior(mock_cfg):
    mock_cfg.return_value = {"cameras": {"corridor_cam": {}}}
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    candidates = check_fall_detections(
        _pose_results(_pose_keypoints(exercise=True), [0, 0, 200, 100]),
        "corridor_cam",
        frame,
    )
    assert len(candidates) == 1
    assert candidates[0]["rule"] == "Fall Detected"
    assert candidates[0]["threshold"] == 8


@mock.patch("detection.get_config")
def test_check_fall_detections_accepts_remote_pose_records(mock_cfg):
    mock_cfg.return_value = {"cameras": {"corridor_cam": {}}}
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    keypoints = _pose_keypoints(exercise=True)

    candidates = check_fall_detections(
        [
            {
                "class_id": 0,
                "confidence": 0.9,
                "bbox": [0, 0, 200, 100],
                "keypoints": keypoints,
            }
        ],
        "corridor_cam",
        frame,
    )

    assert len(candidates) == 1
    assert candidates[0]["rule"] == "Fall Detected"
    assert candidates[0]["confidence"] == 0.9


@mock.patch("detection.get_config")
def test_check_fall_detections_uses_camera_confirmation_threshold(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {
            "corridor_cam": {
                "fall_detection": {
                    "confirmation_threshold": 2,
                },
            },
        },
    }
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    candidates = check_fall_detections(
        _pose_results(_pose_keypoints(exercise=True), [0, 0, 200, 100]),
        "corridor_cam",
        frame,
    )
    assert len(candidates) == 1
    assert candidates[0]["threshold"] == 2
    assert candidates[0]["metadata"]["fallConfirmationThreshold"] == 2
    json.dumps(candidates[0]["metadata"])


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
    dets = [_det("person", 0.9), _det("person", 0.82)]
    violations = check_violations(dets, "cam1")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Person Detected"
    assert violations[0]["severity"] == "P4"
    assert violations[0]["count"] == 2


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
    dets = [_det("person", 0.9, [0, 0, 100, 200])]
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
    dets = [
        _det("person", 0.9, [0, 0, 100, 200]),
        _det("hairnet", 0.85, [20, 10, 40, 40]),
        _det("gloves", 0.8, [30, 120, 70, 180]),
    ]
    violations = check_yoloe_violations(dets, "cam3")
    assert violations == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_accepts_overlapping_ppe_box_when_center_is_offset(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"safety_rule_ids": ["ppe_vest"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [100, 50, 220, 320]),
        _det("safety vest", 0.83, [180, 110, 260, 250]),
    ]
    violations = check_yoloe_violations(dets, "cam3")
    assert violations == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_rejects_far_away_ppe_box(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"safety_rule_ids": ["ppe_vest"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [100, 50, 220, 320]),
        _det("safety vest", 0.83, [260, 110, 340, 250]),
    ]
    violations = check_yoloe_violations(dets, "cam3")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing safety vest"
    assert violations[0]["metadata"]["ppeDetectionCount"] == 1
    assert violations[0]["metadata"]["coveredPersonCount"] == 0


@mock.patch("detection.get_config")
def test_check_yoloe_violations_ignores_tiny_edge_person_for_ppe(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"safety_rule_ids": ["ppe_vest"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [120, 60, 280, 360]),
        _det("safety vest", 0.83, [150, 120, 250, 260]),
        _det("person", 0.37, [835, 390, 853, 716]),
    ]
    violations = check_yoloe_violations(dets, "cam3", frame_w=854, frame_h=720)
    assert violations == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_keeps_normal_missing_person_after_edge_filter(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"cam3": {"safety_rule_ids": ["ppe_vest"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [120, 60, 280, 360]),
        _det("person", 0.82, [360, 80, 520, 370]),
        _det("safety vest", 0.83, [150, 120, 250, 260]),
        _det("person", 0.37, [835, 390, 853, 716]),
    ]
    violations = check_yoloe_violations(dets, "cam3", frame_w=854, frame_h=720)
    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing safety vest"
    assert violations[0]["metadata"]["personCount"] == 3
    assert violations[0]["metadata"]["evaluablePersonCount"] == 2
    assert violations[0]["metadata"]["ignoredPersonCount"] == 1
    assert violations[0]["metadata"]["coveredPersonCount"] == 1


@mock.patch("detection.get_config")
def test_check_yoloe_violations_ignores_people_outside_ppe_evaluation_zone(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {
            "cam3": {
                "safety_rule_ids": ["ppe_vest"],
                "zones": [
                    {
                        "id": "ppe_zone",
                        "type": "ppe_evaluation",
                        "analytics": "ppe",
                        "points": [[0.55, 0.1], [0.95, 0.1], [0.95, 0.95], [0.55, 0.95]],
                    }
                ],
            }
        },
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [620, 80, 900, 650]),
        _det("safety vest", 0.83, [680, 160, 840, 410]),
        _det("person", 0.82, [60, 330, 220, 710]),
    ]
    violations = check_yoloe_violations(dets, "cam3", frame_w=1000, frame_h=800)
    assert violations == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_reports_missing_ppe_inside_evaluation_zone(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {
            "cam3": {
                "safety_rule_ids": ["ppe_vest"],
                "zones": [
                    {
                        "id": "ppe_zone",
                        "type": "ppe_evaluation",
                        "analytics": "ppe",
                        "points": [[0.55, 0.1], [0.95, 0.1], [0.95, 0.95], [0.55, 0.95]],
                    }
                ],
            }
        },
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [620, 80, 900, 650]),
        _det("person", 0.82, [60, 330, 220, 710]),
    ]
    violations = check_yoloe_violations(dets, "cam3", frame_w=1000, frame_h=800)
    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing safety vest"
    assert violations[0]["metadata"]["personCount"] == 2
    assert violations[0]["metadata"]["evaluablePersonCount"] == 1
    assert violations[0]["metadata"]["ignoredPersonCount"] == 1
    assert violations[0]["metadata"]["ppeEvaluationZoneIds"] == ["ppe_zone"]
    diagnostics = violations[0]["metadata"]["personDiagnostics"]
    assert diagnostics[0]["inPpeEvaluationScope"] is True
    assert diagnostics[1]["inPpeEvaluationScope"] is False


@mock.patch("detection.get_config")
def test_check_yoloe_violations_uses_safety_rule_ids_when_yoloe_classes_are_stale(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {
            "cam5": {
                "safety_rule_ids": ["ppe_helmet"],
                "ppe_rule_ids": [],
                "yoloe_classes": ["person", "safety vest"],
            }
        },
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [{"class": "person", "confidence": 0.91, "bbox": [0, 0, 100, 200]}]
    violations = check_yoloe_violations(dets, "cam5")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing helmet"
    assert violations[0]["severity"] == "P2"


@mock.patch("detection.get_config")
def test_rider_helmet_rule_does_not_flag_people_without_vehicle(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"gate_cam": {"safety_rule_ids": ["ppe_rider_helmet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [_det("person", 0.91, [410, 180, 560, 620])]

    violations = check_yoloe_violations(dets, "gate_cam", frame_w=1000, frame_h=800)

    assert violations == []


@mock.patch("detection.get_config")
def test_rider_helmet_rule_flags_rider_without_helmet(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"gate_cam": {"safety_rule_ids": ["ppe_rider_helmet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [410, 180, 560, 620]),
        _det("motorcycle", 0.86, [320, 480, 650, 700]),
    ]

    violations = check_yoloe_violations(dets, "gate_cam", frame_w=1000, frame_h=800)

    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing rider helmet"
    assert violations[0]["threshold"] == 5
    assert "missing rider helmet" in violations[0]["classes"]
    assert violations[0]["metadata"]["safetyRuleId"] == "ppe_rider_helmet"
    assert violations[0]["metadata"]["riderCount"] == 1
    assert violations[0]["metadata"]["violatingRiderCount"] == 1


@mock.patch("detection.get_config")
def test_rider_helmet_rule_accepts_helmeted_rider(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"gate_cam": {"safety_rule_ids": ["ppe_rider_helmet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [410, 180, 560, 620]),
        _det("motorcycle", 0.86, [320, 480, 650, 700]),
        _det("motorcycle helmet", 0.82, [440, 190, 520, 250]),
    ]

    violations = check_yoloe_violations(dets, "gate_cam", frame_w=1000, frame_h=800)

    assert violations == []


@mock.patch("detection.get_config")
def test_rider_helmet_rule_ignores_bicycle_walk_by(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"gate_cam": {"safety_rule_ids": ["ppe_rider_helmet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    dets = [
        _det("person", 0.91, [410, 180, 560, 620]),
        _det("bicycle", 0.86, [320, 480, 650, 700]),
    ]

    violations = check_yoloe_violations(dets, "gate_cam", frame_w=1000, frame_h=800)

    assert violations == []


@mock.patch("detection.get_config")
def test_rider_helmet_rule_ignores_weak_or_tiny_rider_candidates(mock_cfg):
    mock_cfg.return_value = {
        "cameras": {"gate_cam": {"safety_rule_ids": ["ppe_rider_helmet"]}},
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }
    weak_person = [
        _det("person", 0.62, [410, 180, 560, 620]),
        _det("motorcycle", 0.86, [320, 480, 650, 700]),
    ]
    tiny_person = [
        _det("person", 0.91, [410, 180, 560, 260]),
        _det("motorcycle", 0.86, [320, 220, 650, 300]),
    ]

    assert check_yoloe_violations(weak_person, "gate_cam", frame_w=1000, frame_h=800) == []
    assert check_yoloe_violations(tiny_person, "gate_cam", frame_w=1000, frame_h=800) == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_includes_rule_threshold(mock_cfg):
    rules = json.loads(json.dumps(DEFAULT_SAFETY_RULES))
    for rule in rules:
        if rule["id"] == "ppe_helmet":
            rule["threshold"] = 10
            break
    mock_cfg.return_value = {
        "cameras": {
            "cam5": {
                "safety_rule_ids": ["ppe_helmet"],
                "ppe_rule_ids": [],
                "yoloe_classes": ["person", "safety vest"],
            }
        },
        "safety_rules": rules,
    }
    dets = [{"class": "person", "confidence": 0.91, "bbox": [0, 0, 100, 200]}]
    violations = check_yoloe_violations(dets, "cam5")
    assert len(violations) == 1
    assert violations[0]["threshold"] == 10
    assert violations[0]["metadata"]["ruleThreshold"] == 10
    assert violations[0]["metadata"]["thresholdSource"] == "safety_rule"
    assert violations[0]["metadata"]["safetyRuleId"] == "ppe_helmet"


@mock.patch("detection.get_config")
def test_check_yoloe_violations_uses_camera_rule_threshold_override(mock_cfg):
    rules = json.loads(json.dumps(DEFAULT_SAFETY_RULES))
    for rule in rules:
        if rule["id"] == "ppe_gloves":
            rule["threshold"] = 3
            break
    mock_cfg.return_value = {
        "cameras": {
            "glove_cam": {
                "safety_rule_ids": ["ppe_gloves"],
                "safety_rule_overrides": {
                    "ppe_gloves": {
                        "threshold": 12,
                    },
                },
            }
        },
        "safety_rules": rules,
    }
    dets = [{"class": "person", "confidence": 0.91, "bbox": [0, 0, 100, 200]}]
    violations = check_yoloe_violations(dets, "glove_cam")
    assert len(violations) == 1
    assert violations[0]["rule"] == "Missing gloves"
    assert violations[0]["threshold"] == 12
    assert violations[0]["metadata"]["ruleThreshold"] == 12
    assert violations[0]["metadata"]["thresholdSource"] == "camera_override"
    assert violations[0]["metadata"]["safetyRuleId"] == "ppe_gloves"


@mock.patch("detection.get_config")
def test_check_yoloe_violations_accepts_group_name_threshold_override(mock_cfg):
    rules = json.loads(json.dumps(DEFAULT_SAFETY_RULES))
    mock_cfg.return_value = {
        "cameras": {
            "glove_cam": {
                "safety_rule_ids": ["ppe_gloves"],
                "safety_rule_overrides": {
                    "gloves": {
                        "threshold": 9,
                    },
                },
            }
        },
        "safety_rules": rules,
    }
    dets = [{"class": "person", "confidence": 0.91, "bbox": [0, 0, 100, 200]}]
    violations = check_yoloe_violations(dets, "glove_cam")
    assert len(violations) == 1
    assert violations[0]["threshold"] == 9
    assert violations[0]["metadata"]["thresholdSource"] == "camera_override"
