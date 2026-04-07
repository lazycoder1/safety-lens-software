"""Tests for FastAPI API endpoints — uses TestClient with Postgres test DB."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Set DATABASE_URL to test DB before importing alert_store
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/safetylens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = tempfile.mkdtemp()
_test_snapshots = Path(_tmpdir) / "snapshots"
_test_config = Path(_tmpdir) / "test_config.json"

import alert_store
alert_store.SNAPSHOTS_DIR = _test_snapshots

import config_manager
config_manager.CONFIG_PATH = _test_config

# Mock YOLO so server doesn't try to load real models
with mock.patch("state.load_model"):
    import server
    import state

# Use TestClient (no real model loading)
state.model = None
state.yoloe_model = None


@pytest.fixture(autouse=True)
def fresh_state():
    """Reset DB, config, and server state before each test."""
    alert_store.SNAPSHOTS_DIR = _test_snapshots
    alert_store.init_db()

    # Truncate alerts table
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alerts")
        conn.commit()

    # Re-create snapshots dir
    _test_snapshots.mkdir(parents=True, exist_ok=True)

    # Reset config_manager
    config_manager._config = None
    config_manager.CONFIG_PATH = _test_config
    if _test_config.exists():
        _test_config.unlink()
    config_manager.load_config()

    # Clear server state
    state.camera_threads.clear()
    state.vlm_threads.clear()
    state.camera_frames.clear()
    state.camera_detections.clear()
    state.alert_subscribers.clear()

    yield

    # Clean up snapshot files
    if _test_snapshots.exists():
        for f in _test_snapshots.iterdir():
            f.unlink(missing_ok=True)


client = TestClient(server.app, raise_server_exceptions=False)


def _create_test_alert(**kwargs):
    defaults = dict(
        camera_id="cam1", camera_name="Test", zone="ZoneA",
        rule="Helmet", severity="P2", confidence=0.9,
        description="Test alert", source="YOLO",
    )
    defaults.update(kwargs)
    return alert_store.create_alert(**defaults)


# ── GET /api/health ──────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "cameras" in data


# ── GET /api/cameras ─────────────────────────────────────────────────────────

def test_get_cameras():
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert isinstance(cameras, list)
    assert len(cameras) == 3
    cam_ids = {c["id"] for c in cameras}
    assert {"cam1", "cam2", "cam3"} == cam_ids


def test_get_cameras_includes_fields():
    resp = client.get("/api/cameras")
    cam = resp.json()[0]
    assert "name" in cam
    assert "zone" in cam
    assert "demo" in cam
    assert "rules" in cam
    assert "enabled" in cam
    assert "status" in cam


# ── GET /api/alerts ──────────────────────────────────────────────────────────

def test_get_alerts_empty():
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_alerts_with_data():
    _create_test_alert()
    _create_test_alert(severity="P1")
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_alerts_filter_severity():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    _create_test_alert(severity="P2")
    resp = client.get("/api/alerts?severity=P2")
    assert len(resp.json()) == 2


def test_get_alerts_filter_status():
    a = _create_test_alert()
    _create_test_alert()
    alert_store.acknowledge_alert(a["id"])
    resp = client.get("/api/alerts?status=acknowledged")
    assert len(resp.json()) == 1


def test_get_alerts_filter_camera():
    _create_test_alert(camera_id="cam1")
    _create_test_alert(camera_id="cam2")
    resp = client.get("/api/alerts?cameraId=cam1")
    assert len(resp.json()) == 1


def test_get_alerts_limit():
    for _ in range(5):
        _create_test_alert()
    resp = client.get("/api/alerts?limit=3")
    assert len(resp.json()) == 3


# ── GET /api/alerts/stats ────────────────────────────────────────────────────

def test_alerts_stats_empty():
    resp = client.get("/api/alerts/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["active"] == 0


def test_alerts_stats_with_data():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = client.get("/api/alerts/stats")
    data = resp.json()
    assert data["total"] == 2
    assert data["active"] == 2
    assert data["bySeverity"]["P1"] == 1
    assert data["bySeverity"]["P2"] == 1


# ── GET /api/alerts/time-series ──────────────────────────────────────────────

def test_alert_time_series_endpoint():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = client.get("/api/alerts/time-series")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    bucket = data[0]
    assert "hour" in bucket
    assert "P1" in bucket
    assert "P2" in bucket


def test_alert_time_series_custom_hours():
    resp = client.get("/api/alerts/time-series?hours=1")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── PUT /api/alerts/{id}/acknowledge ─────────────────────────────────────────

def test_acknowledge_alert():
    a = _create_test_alert()
    resp = client.put(f"/api/alerts/{a['id']}/acknowledge")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "acknowledged"
    assert data["acknowledgedBy"] == "Admin"


def test_acknowledge_not_found():
    resp = client.put("/api/alerts/fake-id/acknowledge")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/resolve ─────────────────────────────────────────────

def test_resolve_alert():
    a = _create_test_alert()
    resp = client.put(f"/api/alerts/{a['id']}/resolve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"


def test_resolve_not_found():
    resp = client.put("/api/alerts/fake-id/resolve")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/snooze ──────────────────────────────────────────────

def test_snooze_alert_default():
    a = _create_test_alert()
    resp = client.put(f"/api/alerts/{a['id']}/snooze")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "snoozed"
    assert data["snoozedUntil"] is not None


def test_snooze_alert_custom_minutes():
    a = _create_test_alert()
    resp = client.put(f"/api/alerts/{a['id']}/snooze?minutes=60")
    assert resp.status_code == 200
    assert resp.json()["status"] == "snoozed"


def test_snooze_not_found():
    resp = client.put("/api/alerts/fake-id/snooze")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/false-positive ──────────────────────────────────────

def test_false_positive():
    a = _create_test_alert()
    resp = client.put(f"/api/alerts/{a['id']}/false-positive")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"
    assert data["falsePositive"] is True


def test_false_positive_not_found():
    resp = client.put("/api/alerts/fake-id/false-positive")
    assert resp.status_code == 404


# ── GET /api/snapshots/{filename} ────────────────────────────────────────────

def test_serve_snapshot():
    fake_jpeg = b"\xff\xd8\xff\xe0test"
    a = alert_store.create_alert(
        "cam1", "C", "Z", "R", "P2", 0.9, snapshot_jpeg=fake_jpeg,
    )
    filename = a["snapshotUrl"].split("/")[-1]
    resp = client.get(f"/api/snapshots/{filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == fake_jpeg


def test_serve_snapshot_not_found():
    resp = client.get("/api/snapshots/nonexistent.jpg")
    assert resp.status_code == 404


# ── GET /api/config ──────────────────────────────────────────────────────────

def test_get_config():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "global" in data
    assert "vlm" in data
    assert "cameras" in data


# ── PUT /api/config/global ───────────────────────────────────────────────────

@mock.patch("routers.config.restart_all_cameras")
def test_update_global_config(mock_restart):
    resp = client.put("/api/config/global", json={"target_fps": 10})
    assert resp.status_code == 200
    assert resp.json()["target_fps"] == 10
    mock_restart.assert_called_once()


@mock.patch("routers.config.restart_all_cameras")
def test_update_global_partial(mock_restart):
    resp = client.put("/api/config/global", json={"yolo_conf": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["yolo_conf"] == 0.5
    assert data["target_fps"] == 6


# ── PUT /api/config/vlm ─────────────────────────────────────────────────────

def test_update_vlm_config():
    resp = client.put("/api/config/vlm", json={"model": "qwen3.5:35b", "interval": 60})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3.5:35b"
    assert data["interval"] == 60


# ── Telegram config endpoints ────────────────────────────────────────────────

def test_telegram_config_endpoint():
    resp = client.put("/api/config/telegram", json={"enabled": True, "bot_token": "tok123", "chat_id": "456"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["bot_token"] == "tok123"
    assert data["chat_id"] == "456"


@mock.patch("telegram_notifier.test_connection")
def test_telegram_test_endpoint(mock_test):
    mock_test.return_value = {"ok": True}
    resp = client.post("/api/config/telegram/test")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── GET /api/alert-rules-available ───────────────────────────────────────────

def test_available_alert_rules_endpoint():
    resp = client.get("/api/alert-rules-available")
    assert resp.status_code == 200
    data = resp.json()
    assert "mobile_phone" in data
    assert "animal_intrusion" in data
    assert "person_detected" in data
    assert "vehicle_detected" in data
    assert data["mobile_phone"]["rule"] == "Mobile Phone Usage"
    assert data["mobile_phone"]["severity"] == "P3"


# ── GET /api/videos ──────────────────────────────────────────────────────────

def test_list_videos(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.avi").touch()
    (tmp_path / "c.mp4").touch()
    (tmp_path / "readme.txt").touch()
    from routers import misc
    misc.VIDEO_DIR = tmp_path
    resp = client.get("/api/videos")
    assert resp.status_code == 200
    videos = resp.json()
    assert "a.mp4" in videos
    assert "b.avi" in videos
    assert "c.mp4" in videos
    assert "readme.txt" not in videos


def test_videos_includes_avi(tmp_path):
    (tmp_path / "test.avi").touch()
    from routers import misc
    misc.VIDEO_DIR = tmp_path
    resp = client.get("/api/videos")
    assert resp.status_code == 200
    assert "test.avi" in resp.json()


# ── Camera CRUD ──────────────────────────────────────────────────────────────

@mock.patch("routers.cameras.start_camera")
def test_add_camera(mock_start):
    resp = client.post("/api/cameras", json={
        "name": "New Cam", "video": "test.mp4", "zone": "ZoneX",
        "demo": "yolo", "rules": ["Test Rule"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Cam"
    assert "id" in data
    mock_start.assert_called_once()


@mock.patch("routers.cameras.restart_camera")
def test_update_camera(mock_restart):
    resp = client.put("/api/cameras/cam1", json={"name": "Updated Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"
    mock_restart.assert_called_once_with("cam1")


def test_update_camera_not_found():
    resp = client.put("/api/cameras/cam999", json={"name": "X"})
    assert resp.status_code == 404


@mock.patch("routers.cameras.stop_camera")
def test_delete_camera(mock_stop):
    resp = client.delete("/api/cameras/cam1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "cam1"
    mock_stop.assert_called_once_with("cam1")

    cfg = config_manager.get_config()
    assert "cam1" not in cfg["cameras"]


def test_delete_camera_not_found():
    resp = client.delete("/api/cameras/cam999")
    assert resp.status_code == 404


# ── Detection Rules API ─────────────────────────────────────────────────────

def test_get_detection_rules_defaults():
    resp = client.get("/api/detection-rules")
    assert resp.status_code == 200
    rules = resp.json()
    assert len(rules) == 12
    assert rules[0]["name"] == "Hard Hat Detection"
    assert rules[0]["enabled"] is True


def test_get_detection_rules_from_config():
    cfg = config_manager.get_config()
    cfg["detection_rules"] = [
        {"id": "r1", "name": "Custom Rule", "enabled": True, "model": "YOLOE",
         "prompts": ["test"], "confidenceThreshold": 0.5, "severity": "P2",
         "promptType": "text", "camerasCount": 0, "category": "Custom"},
    ]
    config_manager.save_config(cfg)

    resp = client.get("/api/detection-rules")
    assert resp.status_code == 200
    rules = resp.json()
    assert len(rules) == 1
    assert rules[0]["name"] == "Custom Rule"


def test_toggle_detection_rule():
    resp = client.get("/api/detection-rules")
    rules = resp.json()
    first_rule = rules[0]
    assert first_rule["enabled"] is True

    resp = client.put(f"/api/detection-rules/{first_rule['id']}/toggle")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    resp = client.put(f"/api/detection-rules/{first_rule['id']}/toggle")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True


def test_toggle_detection_rule_persists():
    resp = client.get("/api/detection-rules")
    rule_id = resp.json()[0]["id"]

    client.put(f"/api/detection-rules/{rule_id}/toggle")

    config_manager._config = None
    cfg = config_manager.load_config()
    rule = next(r for r in cfg["detection_rules"] if r["id"] == rule_id)
    assert rule["enabled"] is False


def test_toggle_detection_rule_not_found():
    resp = client.put("/api/detection-rules/r999/toggle")
    assert resp.status_code == 404


def test_create_detection_rule():
    resp = client.post("/api/detection-rules", json={
        "name": "Welding Mask",
        "model": "YOLOE",
        "promptType": "text",
        "prompts": ["welding mask", "face shield"],
        "confidenceThreshold": 0.45,
        "severity": "P2",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Welding Mask"
    assert data["enabled"] is True
    assert data["category"] == "Custom"
    assert data["prompts"] == ["welding mask", "face shield"]
    assert "id" in data


def test_create_detection_rule_increments_id():
    client.post("/api/detection-rules", json={
        "name": "Rule A", "model": "YOLOE", "promptType": "text",
        "prompts": ["a"], "confidenceThreshold": 0.5, "severity": "P2",
    })
    resp = client.post("/api/detection-rules", json={
        "name": "Rule B", "model": "YOLOE", "promptType": "text",
        "prompts": ["b"], "confidenceThreshold": 0.5, "severity": "P2",
    })
    data = resp.json()
    rule_num = int(data["id"].replace("r", ""))
    assert rule_num > 12


def test_create_detection_rule_appears_in_list():
    client.post("/api/detection-rules", json={
        "name": "New Custom Rule", "model": "VLM", "promptType": "text",
        "prompts": ["test"], "confidenceThreshold": 0.6, "severity": "P3",
    })
    resp = client.get("/api/detection-rules")
    names = [r["name"] for r in resp.json()]
    assert "New Custom Rule" in names


# ── Enhanced stats ──────────────────────────────────────────────────────────

def test_stats_includes_breakdowns():
    alert_store.create_alert("cam1", "Cam A", "Zone 1", "Helmet", "P2", 0.9)
    alert_store.create_alert("cam2", "Cam B", "Zone 2", "Vest", "P1", 0.8)

    resp = client.get("/api/alerts/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["active"] == 2
    assert data["acknowledged"] == 0
    assert data["resolved"] == 0
    assert data["byRule"]["Helmet"] == 1
    assert data["byRule"]["Vest"] == 1
    assert data["byZone"]["Zone 1"] == 1
    assert data["byZone"]["Zone 2"] == 1
    assert data["byCamera"]["Cam A"] == 1
    assert data["byCamera"]["Cam B"] == 1
