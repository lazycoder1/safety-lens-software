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

import auth_store
import config_manager
config_manager.CONFIG_PATH = _test_config

# Mock YOLO so server doesn't try to load real models
with mock.patch("state.load_model"):
    import server
    import model_manager
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
    state.camera_runtime_status.clear()
    model_manager._INSTALL_JOBS.clear()
    model_manager._ACTIVE_JOB_ID = None
    with model_manager._MODEL_LOCK:
        for model_key in model_manager.MODEL_DEFINITIONS:
            model_manager._set_model_state(model_key, status="ready", error=None, active_path=config_manager.CONFIG_PATH, job_id=None)

    yield

    # Clean up snapshot files
    if _test_snapshots.exists():
        for f in _test_snapshots.iterdir():
            f.unlink(missing_ok=True)


client = TestClient(server.app, raise_server_exceptions=False)


def _admin_headers(extra: dict | None = None) -> dict[str, str]:
    token = auth_store.create_token("admin-test", "admin", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers


def api_get(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.get(path, headers=headers, **kwargs)


def api_post(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.post(path, headers=headers, **kwargs)


def api_put(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.put(path, headers=headers, **kwargs)


def api_delete(path: str, **kwargs):
    headers = _admin_headers(kwargs.pop("headers", None))
    return client.delete(path, headers=headers, **kwargs)


def _first_camera_id() -> str:
    return next(iter(config_manager.get_config()["cameras"]))


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
    assert data["status"] in {"ok", "degraded"}
    assert "cameras" in data


# ── GET /api/cameras ─────────────────────────────────────────────────────────

def test_get_cameras():
    resp = api_get("/api/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert isinstance(cameras, list)
    cfg = config_manager.get_config()
    assert len(cameras) == len(cfg["cameras"])
    cam_ids = {c["id"] for c in cameras}
    assert set(cfg["cameras"]) == cam_ids


def test_get_cameras_includes_fields():
    resp = api_get("/api/cameras")
    cam = resp.json()[0]
    assert "name" in cam
    assert "zone" in cam
    assert "zones" in cam
    assert "demo" in cam
    assert "rules" in cam
    assert "enabled" in cam
    assert "status" in cam


def test_get_cameras_includes_saved_zones():
    cfg = config_manager.get_config()
    cam_id = _first_camera_id()
    cfg["cameras"][cam_id]["zones"] = [
        {
            "id": "z1",
            "name": "Saved Zone",
            "type": "restricted",
            "color": "#dc2626",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        }
    ]
    config_manager.save_config(cfg)

    resp = api_get("/api/cameras")
    cam = next(item for item in resp.json() if item["id"] == cam_id)
    assert cam["zones"][0]["name"] == "Saved Zone"


# ── GET /api/alerts ──────────────────────────────────────────────────────────

def test_get_alerts_empty():
    resp = api_get("/api/alerts")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_alerts_with_data():
    _create_test_alert()
    _create_test_alert(severity="P1")
    resp = api_get("/api/alerts")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_alerts_filter_severity():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts?severity=P2")
    assert len(resp.json()) == 2


def test_get_alerts_filter_status():
    a = _create_test_alert()
    _create_test_alert()
    alert_store.acknowledge_alert(a["id"])
    resp = api_get("/api/alerts?status=acknowledged")
    assert len(resp.json()) == 1


def test_get_alerts_filter_camera():
    _create_test_alert(camera_id="cam1")
    _create_test_alert(camera_id="cam2")
    resp = api_get("/api/alerts?cameraId=cam1")
    assert len(resp.json()) == 1


def test_get_alerts_limit():
    for _ in range(5):
        _create_test_alert()
    resp = api_get("/api/alerts?limit=3")
    assert len(resp.json()) == 3


# ── GET /api/alerts/stats ────────────────────────────────────────────────────

def test_alerts_stats_empty():
    resp = api_get("/api/alerts/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["active"] == 0


def test_alerts_stats_with_data():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts/stats")
    data = resp.json()
    assert data["total"] == 2
    assert data["active"] == 2
    assert data["bySeverity"]["P1"] == 1
    assert data["bySeverity"]["P2"] == 1


# ── GET /api/alerts/time-series ──────────────────────────────────────────────

def test_alert_time_series_endpoint():
    _create_test_alert(severity="P1")
    _create_test_alert(severity="P2")
    resp = api_get("/api/alerts/time-series")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    bucket = data[0]
    assert "hour" in bucket
    assert "P1" in bucket
    assert "P2" in bucket


def test_alert_time_series_custom_hours():
    resp = api_get("/api/alerts/time-series?hours=1")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── PUT /api/alerts/{id}/acknowledge ─────────────────────────────────────────

def test_acknowledge_alert():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/acknowledge")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "acknowledged"
    assert data["acknowledgedBy"] == "admin"


def test_acknowledge_not_found():
    resp = api_put("/api/alerts/fake-id/acknowledge")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/resolve ─────────────────────────────────────────────

def test_resolve_alert():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/resolve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"


def test_resolve_not_found():
    resp = api_put("/api/alerts/fake-id/resolve")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/snooze ──────────────────────────────────────────────

def test_snooze_alert_default():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/snooze")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "snoozed"
    assert data["snoozedUntil"] is not None


def test_snooze_alert_custom_minutes():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/snooze?minutes=60")
    assert resp.status_code == 200
    assert resp.json()["status"] == "snoozed"


def test_snooze_not_found():
    resp = api_put("/api/alerts/fake-id/snooze")
    assert resp.status_code == 404


# ── PUT /api/alerts/{id}/false-positive ──────────────────────────────────────

def test_false_positive():
    a = _create_test_alert()
    resp = api_put(f"/api/alerts/{a['id']}/false-positive")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"
    assert data["falsePositive"] is True


def test_false_positive_not_found():
    resp = api_put("/api/alerts/fake-id/false-positive")
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
    resp = api_get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "global" in data
    assert "vlm" in data
    assert "cameras" in data


# ── PUT /api/config/global ───────────────────────────────────────────────────

@mock.patch("routers.config.restart_all_cameras")
def test_update_global_config(mock_restart):
    resp = api_put("/api/config/global", json={"target_fps": 10})
    assert resp.status_code == 200
    assert resp.json()["target_fps"] == 10
    mock_restart.assert_called_once()


@mock.patch("routers.config.restart_all_cameras")
def test_update_global_partial(mock_restart):
    resp = api_put("/api/config/global", json={"yolo_conf": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["yolo_conf"] == 0.5
    assert data["target_fps"] == 6


# ── PUT /api/config/vlm ─────────────────────────────────────────────────────

def test_update_vlm_config():
    resp = api_put("/api/config/vlm", json={"model": "qwen3.5:35b", "interval": 60})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3.5:35b"
    assert data["interval"] == 60


# ── Telegram config endpoints ────────────────────────────────────────────────

def test_telegram_config_endpoint():
    resp = api_put("/api/config/telegram", json={"enabled": True, "bot_token": "tok123", "chat_id": "456"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["bot_token"] == "tok123"
    assert data["chat_id"] == "456"


@mock.patch("telegram_notifier.test_connection")
def test_telegram_test_endpoint(mock_test):
    mock_test.return_value = {"ok": True}
    resp = api_post("/api/config/telegram/test")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── GET /api/alert-rules-available ───────────────────────────────────────────

def test_available_alert_rules_endpoint():
    resp = api_get("/api/alert-rules-available")
    assert resp.status_code == 200
    data = resp.json()
    assert "alert_mobile_phone" in data
    assert "alert_animal" in data
    assert "alert_person" in data
    assert "alert_vehicle" in data
    assert data["alert_mobile_phone"]["rule"] == "Mobile Phone Usage"
    assert data["alert_mobile_phone"]["severity"] == "P3"


# ── GET /api/videos ──────────────────────────────────────────────────────────

def test_list_videos(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.avi").touch()
    (tmp_path / "c.mp4").touch()
    (tmp_path / "readme.txt").touch()
    from routers import misc
    misc.VIDEO_DIR = tmp_path
    resp = api_get("/api/videos")
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
    resp = api_get("/api/videos")
    assert resp.status_code == 200
    assert "test.avi" in resp.json()


# ── Camera CRUD ──────────────────────────────────────────────────────────────

@mock.patch("routers.cameras.start_camera")
def test_add_camera(mock_start):
    resp = api_post("/api/cameras", json={
        "name": "New Cam", "video": "test.mp4", "zone": "ZoneX",
        "demo": "yolo", "rules": ["Test Rule"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Cam"
    assert "id" in data
    mock_start.assert_called_once()


@mock.patch("routers.cameras.model_manager.missing_model_keys", return_value=["coco_primary"])
def test_add_camera_returns_missing_models(_mock_missing):
    resp = api_post("/api/cameras", json={
        "name": "Needs Model", "video": "test.mp4", "zone": "ZoneX",
        "profile": "general_safety", "capabilities": ["person_presence"],
    })
    assert resp.status_code == 409
    data = resp.json()
    assert data["code"] == "missing_models"
    assert "coco_primary" in data["missing_model_keys"]


@mock.patch("routers.cameras.restart_camera")
def test_update_camera(mock_restart):
    cam_id = _first_camera_id()
    resp = api_put(f"/api/cameras/{cam_id}", json={"name": "Updated Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"
    mock_restart.assert_called_once_with(cam_id)


def test_get_cameras_derives_display_rules_from_safety_rule_ids():
    from routers.cameras import _camera_public_payload

    cfg = config_manager.get_config()
    cam_id = next(iter(cfg["cameras"]))
    cfg["cameras"][cam_id]["rules"] = ["Stale Rule Label"]
    cfg["cameras"][cam_id]["safety_rule_ids"] = ["ppe_helmet", "alert_mobile_phone"]
    config_manager.save_config(cfg)

    camera = _camera_public_payload(cam_id, cfg["cameras"][cam_id], cfg)
    assert camera["rules"] == ["Helmet", "Mobile Phone Usage"]


def test_update_camera_replaces_stale_legacy_detection_fields():
    cfg = config_manager.get_config()
    cam_id = _first_camera_id()
    cfg["cameras"][cam_id].update({
        "safety_rule_ids": ["alert_animal", "alert_mobile_phone", "ppe_helmet", "ppe_hairnet"],
        "alert_classes": ["animal_intrusion", "mobile_phone"],
        "yoloe_classes": ["person", "hard hat", "hairnet", "cell phone"],
        "capabilities": ["animal_presence", "mobile_phone", "helmet_required", "hairnet_required"],
    })
    config_manager.save_config(cfg)

    resp = api_put(f"/api/cameras/{cam_id}", json={
        "capabilities": ["hairnet_required"],
        "safety_rule_ids": ["ppe_hairnet"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_rule_ids"] == ["ppe_hairnet"]
    assert data["alert_classes"] == []
    assert data["capabilities"] == ["hairnet_required"]


def test_preview_camera_plan():
    resp = api_post("/api/camera-plans/preview", json={
        "profile": "work_zone_ppe",
        "capabilities": ["helmet_required", "zone_intrusion"],
        "stream_type": "file",
        "video": "test.mp4",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_plan"]["run_coco_primary"] is True
    assert data["execution_plan"]["run_ppe_specialist"] is True


def test_update_camera_not_found():
    resp = api_put("/api/cameras/cam999", json={"name": "X"})
    assert resp.status_code == 404


@mock.patch("routers.cameras.stop_camera")
def test_delete_camera(mock_stop):
    cam_id = _first_camera_id()
    resp = api_delete(f"/api/cameras/{cam_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == cam_id
    mock_stop.assert_called_once_with(cam_id)

    cfg = config_manager.get_config()
    assert cam_id not in cfg["cameras"]


def test_delete_camera_not_found():
    resp = api_delete("/api/cameras/cam999")
    assert resp.status_code == 404


# ── Enhanced stats ──────────────────────────────────────────────────────────

def test_stats_includes_breakdowns():
    alert_store.create_alert("cam1", "Cam A", "Zone 1", "Helmet", "P2", 0.9)
    alert_store.create_alert("cam2", "Cam B", "Zone 2", "Vest", "P1", 0.8)

    resp = api_get("/api/alerts/stats")
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
