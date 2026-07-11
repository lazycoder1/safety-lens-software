"""Tests for site YAML import and planning."""

import copy
from pathlib import Path

import config_manager
import site_config


ROOT = Path(__file__).resolve().parents[2]

CLOSED_SET_CANDIDATE_YAMLS = [
    ("qa/video_eval/focused/factory_missing_apron_active_closed_set.yaml", "apron_required"),
    ("qa/video_eval/focused/factory_apron_false_positive_guard_closed_set.yaml", "apron_required"),
    ("qa/video_eval/focused/factory_apron_detector_window_suppression_closed_set.yaml", "apron_required"),
    ("qa/video_eval/focused/factory_missing_harness_active_closed_set.yaml", "harness_required"),
    ("qa/video_eval/focused/factory_harness_false_positive_guard_closed_set.yaml", "harness_required"),
    ("qa/video_eval/focused/factory_harness_detector_window_suppression_closed_set.yaml", "harness_required"),
]


def test_export_site_config_creates_nested_output_directory(tmp_path, monkeypatch):
    nested_output = tmp_path / "site_config_backups" / "before_eval.yaml"
    monkeypatch.setattr(
        site_config,
        "get_public_config",
        lambda: {
            "site": {"name": "Eval Site"},
            "global": {"timezone": "Asia/Kolkata"},
            "cameras": {},
            "alert_outputs": [],
            "automation_rules": [],
        },
    )

    written = site_config.export_site_config(nested_output)

    assert written == nested_output
    assert nested_output.exists()
    assert nested_output.parent.is_dir()
    rendered = nested_output.read_text(encoding="utf-8")
    assert "site:" in rendered
    assert "cameras: {}" in rendered


def test_load_site_config_normalizes_deployment_yaml(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
site:
  name: Plant A
  timezone: Asia/Kolkata
global:
  alert_cooldown: 30
alert_outputs:
  floor_telegram:
    name: Floor Telegram
    type: telegram
    enabled: true
    severities: [P1, P2]
    zones: []
    mode: live
    settings:
      bot_token: token
      chat_id: "-100"
automation_rules:
  ppe_line_1:
    name: Line 1 helmet alerts
    trigger: detection
    cameras: [line1]
    conditions:
      - type: class_is
        params: {classes: no_helmet}
    output_ids: [floor_telegram]
    message_template: "{severity} {violation_type} at {camera}"
    severity: P2
    cooldown_seconds: 45
cameras:
  line1:
    name: Line 1
    zone: Assembly
    stream_type: rtsp
    host: 192.168.10.11
    rtsp_port: 554
    stream_path: /stream1
    capabilities: [person_presence, helmet_required]
    safety_rule_ids: [ppe_helmet]
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    assert result.config is not None
    assert result.config["site"]["config_source"] == "yaml"
    assert result.config["global"]["alert_cooldown"] == 30
    assert result.config["cameras"]["line1"]["host"] == "192.168.10.11"
    assert result.config["cameras"]["line1"]["execution_plan"]["run_ppe_specialist"] is True

    rule = result.config["automation_rules"][0]
    assert rule["id"] == "ppe_line_1"
    assert rule["outputIds"] == ["floor_telegram"]
    assert rule["messageTemplate"] == "{severity} {violation_type} at {camera}"
    assert rule["cooldownSeconds"] == 45


def test_load_site_config_compiles_camera_event_policy(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
alert_outputs:
  line_webhook:
    name: Line Webhook
    type: webhook
    enabled: true
    severities: [P1, P2]
    zones: []
    mode: live
    settings: {url: "https://example.test/line"}
cameras:
  cam_factory:
    name: Factory Camera
    zone: Packaging
    stream_type: file
    video: factory.mp4
    capabilities: [apron_required]
    safety_rule_ids: [ppe_apron]
    event_policy:
      output_ids: [line_webhook]
      severity: P2
      priority: 4
      cooldown_seconds: 90
      min_confidence: 0.67
      message_template: "{severity} {violation_type} on {camera}"
      schedule:
        windows:
          - days: [mon, tue]
            from: "08:00"
            to: "18:00"
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["cam_factory"]
    assert "event_policy" not in camera
    assert "eventPolicy" not in camera

    rules = {rule["id"]: rule for rule in result.config["automation_rules"]}
    rule = rules["camera_event_cam_factory"]
    assert rule["preset"] == site_config.CAMERA_EVENT_POLICY_PRESET
    assert rule["cameras"] == ["cam_factory"]
    assert rule["outputIds"] == ["line_webhook"]
    assert rule["severity"] == "P2"
    assert rule["priority"] == 4
    assert rule["cooldownSeconds"] == 90
    assert rule["messageTemplate"] == "{severity} {violation_type} on {camera}"
    assert rule["schedule"] == {
        "windows": [{"days": ["mon", "tue"], "from": "08:00", "to": "18:00"}],
    }
    assert {"type": "confidence_above", "params": {"value": "0.67"}} in rule["conditions"]
    assert rule["thenActions"][0] == {"type": "create_alert", "params": {"severity": "P2"}}


def test_load_site_config_rejects_malformed_camera_event_policy(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  cam_factory:
    name: Factory Camera
    zone: Packaging
    stream_type: file
    video: factory.mp4
    event_policy:
      output_ids: []
      severity: P9
      priority: 0
      cooldown_seconds: -1
      min_confidence: 1.5
      schedule:
        windows:
          - days: [mon]
            from: "08:00"
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok is False
    assert any("event_policy.output_ids is required" in error for error in result.errors)
    assert any("event_policy.severity must be P1, P2, P3, P4, or inherit" in error for error in result.errors)
    assert any("event_policy.priority must be 1 or higher" in error for error in result.errors)
    assert any("event_policy.cooldown_seconds must be 0 or higher" in error for error in result.errors)
    assert any("event_policy.min_confidence must be between 0 and 1" in error for error in result.errors)
    assert any("event_policy.schedule.windows[1] needs from and to" in error for error in result.errors)


def test_build_plan_reports_camera_output_and_rule_changes(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
alert_outputs:
  ops:
    name: Ops
    type: webhook
    enabled: true
    severities: [P1]
    zones: []
    mode: live
    settings: {url: "https://example.test/hook"}
automation_rules:
  fire:
    name: Fire
    trigger: detection
    cameras: [cam_new]
    output_ids: [ops]
cameras:
  cam_new:
    name: New Camera
    zone: Yard
    stream_type: file
    video: sample.mp4
""",
        encoding="utf-8",
    )
    base = copy.deepcopy(config_manager.DEFAULT_CONFIG)
    base["cameras"] = {}
    base["alert_outputs"] = []
    base["automation_rules"] = []

    result = site_config.build_plan(path, base_config=base)

    assert result.ok, result.errors
    plan = result.config["plan"]
    assert plan["added_cameras"] == ["cam_new"]
    assert "ops" in plan["added_outputs"]
    assert plan["added_rules"] == ["fire"]


def test_load_site_config_validates_capability_windows(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  cam_apron:
    name: Apron Camera
    zone: Cafe
    stream_type: file
    video: public/mixkit-barista-denim-apron.mp4
    capabilities: [apron_required]
    safety_rule_ids: [ppe_apron]
    capability_windows:
      apron_required:
        mode: detection
        windows:
          - days: [mon]
            from: "09:00"
            to: "17:00"
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["cam_apron"]
    assert camera["execution_plan"]["capability_windows"] == [
        {
            "id": "capability_window_1",
            "capabilities": ["apron_required"],
            "mode": "detection",
            "windows": [{"days": ["mon"], "from": "09:00", "to": "17:00"}],
        }
    ]


def test_load_site_config_accepts_closed_set_candidate_override_for_side_by_side(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  cam_apron_candidate:
    name: Apron Candidate Camera
    zone: Cafe
    stream_type: file
    video: public/mixkit-barista-denim-apron.mp4
    capabilities: [apron_required]
    safety_rule_ids: [ppe_apron]
    capability_model_overrides:
      apron_required: ppe_closed_set_candidate
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["cam_apron_candidate"]
    assert camera["execution_plan"]["required_model_keys"] == ["ppe_closed_set_candidate"]
    assert camera["execution_plan"]["run_ppe_closed_set_candidate"] is True
    assert camera["execution_plan"]["run_ppe_specialist"] is False


def test_load_site_config_rejects_unsupported_capability_model_override(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  cam_boots:
    name: Boots Camera
    zone: Factory
    stream_type: file
    video: boots.mp4
    capabilities: [boots_required]
    safety_rule_ids: [ppe_boots]
    capability_model_overrides:
      boots_required: ppe_closed_set_candidate
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok is False
    assert any("boots_required=ppe_closed_set_candidate is not supported" in error for error in result.errors)


def test_closed_set_candidate_yaml_scenarios_compile_to_candidate_model_only():
    for path, capability in CLOSED_SET_CANDIDATE_YAMLS:
        result = site_config.load_site_config(
            ROOT / path,
            base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
        )

        assert result.ok, f"{path}: {result.errors}"
        camera = next(iter(result.config["cameras"].values()))
        plan = camera["execution_plan"]
        assert plan["capabilities"] == [capability]
        assert plan["capability_model_overrides"] == {
            capability: "ppe_closed_set_candidate",
        }
        assert plan["required_model_keys"] == ["ppe_closed_set_candidate"]
        assert plan["run_ppe_closed_set_candidate"] is True
        assert plan["run_ppe_specialist"] is False
        assert plan["run_coco_primary"] is False
        assert camera["yoloe_classes"] == []


def test_load_site_config_rejects_malformed_capability_windows(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  cam_apron:
    name: Apron Camera
    zone: Cafe
    stream_type: file
    video: public/mixkit-barista-denim-apron.mp4
    capability_windows:
      apron_required:
        mode: bad_mode
        windows:
          - days: [mon]
            from: "09:00"
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok is False
    assert any("mode must be detection, detector, detector_off, or alert_policy" in error for error in result.errors)
    assert any("needs from and to" in error for error in result.errors)


def test_merge_existing_keeps_unspecified_cameras_outputs_and_rules(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
site:
  merge_existing: true
alert_outputs:
  eval_sound:
    name: Eval Sound
    type: browser_sound
    enabled: true
    severities: [P4]
    zones: []
    mode: browser
    settings: {}
automation_rules:
  eval_rule:
    name: Eval Rule
    trigger: detection
    cameras: [eval_cam]
cameras:
  eval_cam:
    name: Eval Camera
    zone: Eval Zone
    stream_type: file
    video: eval.mp4
""",
        encoding="utf-8",
    )
    base = copy.deepcopy(config_manager.DEFAULT_CONFIG)
    base["cameras"] = {
        "existing_cam": {
            "name": "Existing Camera",
            "zone": "Existing Zone",
            "stream_type": "file",
            "video": "existing.mp4",
        }
    }
    base["alert_outputs"] = [
        {
            "id": "existing_output",
            "name": "Existing Output",
            "type": "in_app",
            "enabled": True,
            "severities": ["P1"],
            "zones": [],
            "mode": "live",
            "settings": {},
        }
    ]
    base["automation_rules"] = [
        {
            "id": "existing_rule",
            "name": "Existing Rule",
            "trigger": "detection",
            "cameras": ["existing_cam"],
            "conditions": [],
            "thenActions": [],
            "elseActions": [],
            "cooldownSeconds": 60,
            "priority": 1,
            "lastTriggered": None,
            "preset": None,
        }
    ]

    result = site_config.load_site_config(path, base_config=base)

    assert result.ok, result.errors
    assert result.config is not None
    assert set(result.config["cameras"]) >= {"existing_cam", "eval_cam"}
    assert {"existing_output", "eval_sound"}.issubset(
        {output["id"] for output in result.config["alert_outputs"]}
    )
    assert {"existing_rule", "eval_rule"}.issubset(
        {rule["id"] for rule in result.config["automation_rules"]}
    )


def test_queue_capabilities_survive_site_config_normalization(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  queue_cam:
    name: Queue Camera
    zone: Lobby
    stream_type: file
    video: queue.ogv
    capabilities: [person_presence, crowd_count_threshold, queue_monitoring]
    safety_rule_ids: [alert_person]
    zones:
      - id: queue_zone
        name: Queue Zone
        type: queue
        analytics: queue
        points:
          - [0.0, 0.2]
          - [1.0, 0.2]
          - [1.0, 1.0]
          - [0.0, 1.0]
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["queue_cam"]
    assert camera["capabilities"] == ["person_presence", "crowd_count_threshold", "queue_monitoring"]
    assert camera["zones"][0]["analytics"] == "queue"


def test_crowd_count_suppression_yaml_stays_capability_pure():
    path = ROOT / "qa" / "video_eval" / "focused" / "office_crowd_count_detector_suppression.yaml"

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["eval_office_crowd_count"]
    assert camera["capabilities"] == ["crowd_count_threshold"]
    assert camera["safety_rule_ids"] == []
    assert camera["execution_plan"]["capabilities"] == ["crowd_count_threshold"]
    assert camera["execution_plan"]["required_model_keys"] == ["coco_primary"]
    assert camera["execution_plan"]["run_coco_primary"] is True
    assert camera["execution_plan"]["capability_windows"] == [
        {
            "id": "office_crowd_count_inactive_window",
            "capabilities": ["crowd_count_threshold"],
            "mode": "detection",
            "windows": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"], "from": "00:00", "to": "23:59"}],
            "active": False,
        }
    ]


def test_crowd_count_active_yaml_stays_capability_pure():
    path = ROOT / "qa" / "video_eval" / "focused" / "office_crowd_count_active.yaml"

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["eval_office_crowd_count"]
    assert camera["capabilities"] == ["crowd_count_threshold"]
    assert camera["safety_rule_ids"] == []
    assert camera["execution_plan"]["capabilities"] == ["crowd_count_threshold"]
    assert camera["execution_plan"]["required_model_keys"] == ["coco_primary"]
    rule = result.config["automation_rules"][0]
    assert rule["trigger"] == "count_threshold"
    assert {"type": "count_exceeds", "params": {"count": "3"}} in rule["conditions"]


def test_route_obstruction_capabilities_survive_site_config_normalization(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  obstruction_cam:
    name: Route Obstruction Camera
    zone: Gate Route
    stream_type: file
    video: blockade.ogv
    capabilities: [vehicle_presence, route_obstruction]
    safety_rule_ids: [alert_vehicle]
    obstruction_threshold: 2
    zones:
      - id: gate_route
        name: Gate Route Keep Clear
        type: route
        analytics: obstruction
        classes: [car, person, bicycle]
        threshold: 2
        area_square_meters: 10.0
        severity_thresholds:
          medium_density_objects_per_square_meter: 0.2
          high_density_objects_per_square_meter: 0.3
        points:
          - [0.0, 0.2]
          - [1.0, 0.2]
          - [1.0, 1.0]
          - [0.0, 1.0]
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["obstruction_cam"]
    assert set(camera["capabilities"]) == {"vehicle_presence", "route_obstruction"}
    assert camera["obstruction_threshold"] == 2
    assert camera["zones"][0]["analytics"] == "obstruction"
    assert camera["zones"][0]["classes"] == ["car", "person", "bicycle"]
    assert camera["zones"][0]["area_square_meters"] == 10.0
    assert camera["zones"][0]["severity_thresholds"]["medium_density_objects_per_square_meter"] == 0.2


def test_object_lifecycle_capabilities_survive_site_config_normalization(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        """
cameras:
  object_cam:
    name: Object Lifecycle Camera
    zone: Retail Floor
    stream_type: file
    video: object-removal.mp4
    capabilities: [object_lifecycle]
    object_classes: [handbag, suitcase, umbrella]
    object_removal_after_seconds: 1
    object_event_linger_seconds: 10
    zones:
      - id: bag_watch
        name: Bag Watch Zone
        type: object_watch
        analytics: object_lifecycle
        classes: [handbag, suitcase, umbrella]
        removal_after_seconds: 1
        points:
          - [0.0, 0.4]
          - [1.0, 0.4]
          - [1.0, 1.0]
          - [0.0, 1.0]
""",
        encoding="utf-8",
    )

    result = site_config.load_site_config(
        path,
        base_config=copy.deepcopy(config_manager.DEFAULT_CONFIG),
    )

    assert result.ok, result.errors
    camera = result.config["cameras"]["object_cam"]
    assert camera["capabilities"] == ["object_lifecycle"]
    assert camera["object_classes"] == ["handbag", "suitcase", "umbrella"]
    assert camera["object_removal_after_seconds"] == 1
    assert camera["zones"][0]["analytics"] == "object_lifecycle"
    assert camera["zones"][0]["classes"] == ["handbag", "suitcase", "umbrella"]
