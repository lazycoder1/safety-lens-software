"""Tests for runtime automation policy evaluation."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import policy_engine


@pytest.fixture(autouse=True)
def reset_cooldowns():
    policy_engine._last_triggered_by_key.clear()
    yield
    policy_engine._last_triggered_by_key.clear()


def test_ppe_missing_candidate_matches_no_helmet_rule_and_routes_outputs():
    cfg = {
        "site": {"timezone": "Asia/Kolkata"},
        "automation_rules": [
            {
                "id": "ppe_line_1",
                "name": "Line 1 helmet alerts",
                "enabled": True,
                "trigger": "detection",
                "cameras": ["line1"],
                "conditions": [
                    {"type": "class_is", "params": {"classes": "no_helmet"}},
                    {"type": "zone_is", "params": {"zone": "Assembly"}},
                    {"type": "confidence_above", "params": {"value": "0.7"}},
                ],
                "thenActions": [{"type": "create_alert", "params": {"severity": "P2"}}],
                "outputIds": ["floor_telegram", "manager_email"],
                "messageTemplate": "{severity} {violation_type} at {camera} in {zone}",
                "cooldownSeconds": 0,
                "priority": 9,
            }
        ],
    }
    candidate = {
        "camera_id": "line1",
        "rule": "Missing helmet",
        "severity": "P3",
        "confidence": 0.91,
        "description": "Worker without helmet",
        "source": "PPE Specialist",
    }
    camera = {"name": "Line 1", "zone": "Assembly"}

    decisions = policy_engine.evaluate_candidate(
        candidate,
        camera,
        camera_id="line1",
        detections=[{"class": "person", "confidence": 0.95}],
        cfg=cfg,
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.rule_id == "ppe_line_1"
    assert decision.severity == "P2"
    assert decision.priority == 9
    assert decision.output_ids == ["floor_telegram", "manager_email"]
    assert decision.message == "P2 Missing helmet at Line 1 in Assembly"
    assert decision.fallback is False


def test_schedule_blocks_matching_rule_outside_active_window():
    cfg = {
        "site": {"timezone": "Asia/Kolkata"},
        "automation_rules": [
            {
                "id": "after_hours",
                "name": "After-hours intrusion",
                "enabled": True,
                "trigger": "zone_enter",
                "cameras": ["gate"],
                "conditions": [{"type": "class_is", "params": {"classes": "zone intrusion"}}],
                "schedule": {"windows": [{"days": ["mon"], "from": "22:00", "to": "06:00"}]},
                "cooldownSeconds": 0,
            }
        ],
    }
    candidate = {
        "camera_id": "gate",
        "rule": "Zone Intrusion",
        "severity": "P1",
        "confidence": 0.88,
        "description": "Person in restricted zone",
        "source": "YOLO",
    }

    decisions = policy_engine.evaluate_candidate(
        candidate,
        {"name": "Gate", "zone": "Gate"},
        camera_id="gate",
        detections=[{"class": "person"}],
        cfg=cfg,
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert decisions == []


def test_fallback_preserves_candidate_when_no_camera_rules_exist():
    decisions = policy_engine.evaluate_candidate(
        {
            "camera_id": "cam1",
            "rule": "Fire",
            "severity": "P1",
            "confidence": 0.94,
            "description": "Fire detected",
            "source": "Fire / Smoke Specialist",
        },
        {"name": "Boiler", "zone": "Utility"},
        camera_id="cam1",
        cfg={"automation_rules": []},
    )

    assert len(decisions) == 1
    assert decisions[0].fallback is True
    assert decisions[0].severity == "P1"


def test_fallback_preserved_when_scoped_automation_rules_do_not_match_event():
    cfg = {
        "site": {"timezone": "Asia/Kolkata"},
        "automation_rules": [
            {
                "id": "fire_watch",
                "name": "Fire Emergency",
                "enabled": True,
                "trigger": "detection",
                "cameras": [],
                "conditions": [{"type": "class_is", "params": {"classes": "fire,smoke"}}],
                "cooldownSeconds": 30,
            }
        ],
    }

    decisions = policy_engine.evaluate_candidate(
        {
            "camera_id": "cam3",
            "rule": "Vehicle Detected",
            "severity": "P4",
            "confidence": 0.9,
            "count": 1,
            "description": "1 vehicle detected detection(s)",
            "source": "COCO Primary",
        },
        {"name": "NVR Channel 3", "zone": "Camera 3"},
        camera_id="cam3",
        detections=[{"class": "motorcycle", "confidence": 0.9}],
        cfg=cfg,
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert len(decisions) == 1
    assert decisions[0].fallback is True
    assert decisions[0].rule_name == "Vehicle Detected"
    assert decisions[0].severity == "P4"


def test_count_threshold_rule_uses_candidate_count():
    cfg = {
        "site": {"timezone": "Asia/Kolkata"},
        "automation_rules": [
            {
                "id": "crowd_watch",
                "name": "Crowd watch",
                "enabled": True,
                "trigger": "count_threshold",
                "cameras": ["office"],
                "conditions": [
                    {"type": "class_is", "params": {"classes": "person"}},
                    {"type": "count_exceeds", "params": {"count": "3"}},
                ],
                "outputIds": ["in_app"],
                "cooldownSeconds": 0,
                "priority": 6,
                "severity": "P3",
            }
        ],
    }
    candidate = {
        "camera_id": "office",
        "rule": "Person Detected",
        "severity": "P4",
        "confidence": 0.88,
        "count": 4,
        "description": "4 person detected detection(s)",
        "source": "COCO Primary",
    }

    decisions = policy_engine.evaluate_candidate(
        candidate,
        {"name": "Open Office", "zone": "Office Floor"},
        camera_id="office",
        detections=[{"class": "person"}],
        cfg=cfg,
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert len(decisions) == 1
    assert decisions[0].rule_id == "crowd_watch"
    assert decisions[0].severity == "P3"
    assert decisions[0].output_ids == ["in_app"]
