"""Tests for converting discovered cameras into site YAML."""

import copy
import os
from unittest import mock

import yaml

import config_manager
import site_config
import camera_discovery
from camera_discovery import discovery_to_site_document


def test_discover_cameras_continues_when_ws_discovery_is_blocked(monkeypatch):
    monkeypatch.setattr(
        camera_discovery,
        "ws_discover",
        lambda timeout_seconds=4.0: (_ for _ in ()).throw(PermissionError("multicast blocked")),
    )
    monkeypatch.setattr(camera_discovery, "_iter_candidate_hosts", lambda network: ([], []))

    result = camera_discovery.discover_cameras(["192.168.29.0/30"], timeout_seconds=0.01)

    assert result["devices"] == []
    assert any("ONVIF WS-Discovery failed" in warning for warning in result["warnings"])


def test_discovery_to_site_document_builds_valid_yaml_ready_cameras(tmp_path):
    discovery = {
        "cidrs": ["192.168.29.0/24"],
        "warnings": [],
        "devices": [
            {
                "fingerprint": "uuid:gate-camera",
                "source": "onvif",
                "host": "192.168.29.250",
                "name": "Gate Camera",
                "vendor": "Matrix",
                "model": "SATATYA",
                "onvif_uuid": "uuid:gate-camera",
                "onvif_xaddr": "http://192.168.29.250/onvif/device_service",
                "onvif_port": 80,
                "rtsp_port": 554,
                "stream_candidates": [
                    {"label": "main", "name": "Main", "path": "/unicaststream/1"},
                ],
                "recommended_stream": "main",
                "auth_state": "needs_credentials",
            }
        ],
    }

    doc = discovery_to_site_document(
        discovery,
        zone="Main Gate",
        capabilities=["person_presence", "vehicle_presence"],
        username_env="CAMERA_USER",
        password_env="CAMERA_PASSWORD",
        enabled=True,
    )
    camera_id, camera = next(iter(doc["cameras"].items()))

    assert camera_id == "discovered_gate_camera_192_168_29_250"
    assert doc["site"]["merge_existing"] is True
    assert camera["host"] == "192.168.29.250"
    assert camera["stream_path"] == "/unicaststream/1"
    assert camera["username"] == "${CAMERA_USER}"
    assert camera["password"] == "${CAMERA_PASSWORD}"
    assert camera["safety_rule_ids"] == ["alert_person", "alert_vehicle"]
    assert set(camera["capability_windows"]) == {"person_presence", "vehicle_presence"}
    assert camera["event_policy"]["output_ids"] == ["in_app"]
    assert camera["event_policy"]["severity"] == "inherit"
    assert camera["event_policy"]["schedule"]["windows"][0]["days"] == ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    assert camera["needs_stream_path_review"] is False

    path = tmp_path / "discovered.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with mock.patch.dict(os.environ, {"CAMERA_USER": "admin", "CAMERA_PASSWORD": "secret"}):
        result = site_config.load_site_config(path, base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG))

    assert result.ok, result.errors
    normalized = result.config["cameras"][camera_id]
    rule = next(
        rule
        for rule in result.config["automation_rules"]
        if rule["id"] == f"camera_event_{camera_id}"
    )
    assert normalized["execution_plan"]["run_coco_primary"] is True
    assert normalized["execution_plan"]["capability_windows"][0]["mode"] == "detection"
    assert "event_policy" not in normalized
    assert rule["preset"] == site_config.CAMERA_EVENT_POLICY_PRESET
    assert rule["outputIds"] == ["in_app"]
    assert rule["schedule"]["windows"][0]["from"] == "00:00"


def test_discovery_to_site_document_can_override_or_skip_event_policy():
    discovery = {
        "cidrs": ["192.168.29.0/24"],
        "warnings": [],
        "devices": [
            {
                "fingerprint": "uuid:ppe-camera",
                "host": "192.168.29.251",
                "name": "PPE Camera",
                "rtsp_port": 554,
                "stream_candidates": [],
                "recommended_stream": "main",
            }
        ],
    }

    doc = discovery_to_site_document(
        discovery,
        capabilities=["apron_required"],
        capability_model_overrides={"apron_required": "ppe_closed_set_candidate"},
        event_output_ids=["line_webhook", "floor_sound"],
        event_severity="P2",
        event_priority=3,
        event_cooldown_seconds=90,
        event_min_confidence=0.67,
    )
    camera = next(iter(doc["cameras"].values()))

    assert camera["event_policy"]["output_ids"] == ["line_webhook", "floor_sound"]
    assert camera["event_policy"]["severity"] == "P2"
    assert camera["event_policy"]["priority"] == 3
    assert camera["event_policy"]["cooldown_seconds"] == 90
    assert camera["event_policy"]["min_confidence"] == 0.67
    assert camera["capability_model_overrides"] == {
        "apron_required": "ppe_closed_set_candidate"
    }

    no_policy = discovery_to_site_document(discovery, include_event_policy=False)
    assert "event_policy" not in next(iter(no_policy["cameras"].values()))


def test_discovery_to_site_document_builds_closed_set_ppe_camera_yaml(tmp_path):
    discovery = {
        "cidrs": ["192.168.29.0/24"],
        "warnings": [],
        "devices": [
            {
                "fingerprint": "uuid:factory-ppe-camera",
                "source": "rtsp_probe",
                "host": "192.168.29.252",
                "name": "Factory PPE Camera",
                "rtsp_port": 554,
                "stream_candidates": [{"path": "/Streaming/Channels/101"}],
                "recommended_stream": "main",
            }
        ],
    }

    doc = discovery_to_site_document(
        discovery,
        zone="Factory PPE",
        profile="work_zone_ppe",
        capabilities=["apron_required", "harness_required"],
        capability_model_overrides={
            "apron_required": "ppe_closed_set_candidate",
            "harness_required": "ppe_closed_set_candidate",
        },
        event_output_ids=["in_app", "browser_sound"],
        event_severity="P2",
        event_priority=2,
        event_cooldown_seconds=45,
        event_min_confidence=0.2,
    )
    camera_id, camera = next(iter(doc["cameras"].items()))

    assert camera_id == "discovered_factory_ppe_camera_192_168_29_252"
    assert camera["profile"] == "work_zone_ppe"
    assert camera["capabilities"] == ["apron_required", "harness_required"]
    assert camera["safety_rule_ids"] == ["ppe_apron", "ppe_harness"]
    assert camera["capability_model_overrides"] == {
        "apron_required": "ppe_closed_set_candidate",
        "harness_required": "ppe_closed_set_candidate",
    }
    assert set(camera["capability_windows"]) == {"apron_required", "harness_required"}
    assert camera["event_policy"]["output_ids"] == ["in_app", "browser_sound"]

    path = tmp_path / "factory_ppe_discovered.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    normalized = result.config["cameras"][camera_id]
    plan = normalized["execution_plan"]
    assert plan["required_model_keys"] == ["ppe_closed_set_candidate"]
    assert plan["run_ppe_closed_set_candidate"] is True
    assert plan["run_ppe_specialist"] is False
    assert normalized["capability_model_overrides"] == {
        "apron_required": "ppe_closed_set_candidate",
        "harness_required": "ppe_closed_set_candidate",
    }
    rule = next(
        rule
        for rule in result.config["automation_rules"]
        if rule["id"] == f"camera_event_{camera_id}"
    )
    assert rule["outputIds"] == ["in_app", "browser_sound"]
    assert rule["priority"] == 2
    assert rule["cooldownSeconds"] == 45
