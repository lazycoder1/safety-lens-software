"""Tests for camera discovery, redaction, and import flows."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/safetylens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = tempfile.mkdtemp()
_test_snapshots = Path(_tmpdir) / "snapshots"
_test_config = Path(_tmpdir) / "test_config.json"

import alert_store
alert_store.SNAPSHOTS_DIR = _test_snapshots

import audit_store
import auth_store
from camera_connection import build_rtsp_url, normalize_camera_connection
from camera_discovery import resolve_scan_networks
import config_manager
config_manager.CONFIG_PATH = _test_config

with mock.patch("state.load_model"):
    import server
    import model_manager
    import state

state.model = None
state.yoloe_model = None

client = TestClient(server.app, raise_server_exceptions=False)


def admin_headers() -> dict[str, str]:
    token = auth_store.create_token("admin-test", "admin", "admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def fresh_state():
    alert_store.SNAPSHOTS_DIR = _test_snapshots
    alert_store.init_db()
    audit_store.init_db()
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alerts")
        conn.commit()

    config_manager._config = None
    config_manager.CONFIG_PATH = _test_config
    if _test_config.exists():
        _test_config.unlink()
    config_manager.load_config()
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


def test_cameras_endpoint_redacts_structured_credentials():
    cfg = config_manager.get_config()
    cfg["cameras"]["cam1"].update({
        "stream_type": "rtsp",
        "host": "192.168.1.50",
        "rtsp_port": 554,
        "stream_path": "/stream1",
        "username": "admin",
        "password": "secret",
        "rtsp_url": "rtsp://192.168.1.50:554/stream1",
    })
    config_manager.save_config(cfg)

    resp = client.get("/api/cameras", headers=admin_headers())
    assert resp.status_code == 200
    camera = next(item for item in resp.json() if item["id"] == "cam1")
    assert "username" not in camera
    assert "password" not in camera
    assert camera["credentials_configured"] is True
    assert camera["rtsp_url"] == "rtsp://192.168.1.50:554/stream1"


def test_normalize_camera_connection_builds_structured_rtsp_fields():
    camera = {
        "stream_type": "rtsp",
        "rtsp_url": "rtsp://admin:secret@192.168.1.75:8554/live",
    }

    changed = normalize_camera_connection(camera)

    assert changed is True
    assert camera["host"] == "192.168.1.75"
    assert camera["rtsp_port"] == 8554
    assert camera["stream_path"] == "/live"
    assert camera["username"] == "admin"
    assert camera["password"] == "secret"
    assert camera["preferred_stream"] == "main"
    assert camera["rtsp_url"] == "rtsp://192.168.1.75:8554/live"
    assert build_rtsp_url(camera, include_credentials=True) == "rtsp://admin:secret@192.168.1.75:8554/live"


def test_resolve_scan_networks_ignores_invalid_cidrs():
    cidrs, warnings = resolve_scan_networks(["192.168.10.0/24", "bad-cidr"])

    assert cidrs == ["192.168.10.0/24"]
    assert warnings == ["Ignored invalid CIDR: bad-cidr"]


def test_config_endpoint_redacts_camera_credentials():
    cfg = config_manager.get_config()
    cfg["cameras"]["cam2"].update({
        "stream_type": "rtsp",
        "host": "192.168.1.60",
        "rtsp_port": 8554,
        "stream_path": "/sub",
        "username": "viewer",
        "password": "hidden",
        "rtsp_url": "rtsp://192.168.1.60:8554/sub",
    })
    config_manager.save_config(cfg)

    resp = client.get("/api/config", headers=admin_headers())
    assert resp.status_code == 200
    camera = resp.json()["cameras"]["cam2"]
    assert "username" not in camera
    assert "password" not in camera
    assert camera["credentials_configured"] is True
    assert camera["connection_summary"] == "rtsp://192.168.1.60:8554/sub"


@mock.patch("routers.cameras.discover_cameras")
def test_discover_endpoint_marks_duplicates(mock_discover):
    cfg = config_manager.get_config()
    cfg["cameras"]["cam3"].update({
        "stream_type": "rtsp",
        "host": "192.168.29.250",
        "rtsp_port": 554,
        "stream_path": "/unicaststream/1",
        "onvif_uuid": "uuid:existing-camera",
        "rtsp_url": "rtsp://192.168.29.250:554/unicaststream/1",
    })
    config_manager.save_config(cfg)
    mock_discover.return_value = {
        "cidrs": ["192.168.29.0/24"],
        "warnings": [],
        "devices": [
            {
                "fingerprint": "uuid:existing-camera",
                "host": "192.168.29.250",
                "ip": "192.168.29.250",
                "name": "Matrix SATATYA",
                "vendor": "Matrix",
                "model": "SATATYA",
                "onvif_uuid": "uuid:existing-camera",
                "rtsp_port": 554,
                "stream_path": "/unicaststream/1",
            }
        ],
    }

    resp = client.post("/api/cameras/discover", json={}, headers=admin_headers())
    assert resp.status_code == 200
    device = resp.json()["devices"][0]
    assert device["duplicate_state"] == "exact"
    assert device["existing_camera_id"] == "cam3"


@mock.patch("routers.cameras.test_camera_connection")
def test_test_endpoint_annotates_duplicate_state(mock_test):
    cfg = config_manager.get_config()
    cfg["cameras"]["cam1"].update({
        "stream_type": "rtsp",
        "host": "192.168.29.250",
        "rtsp_port": 554,
        "stream_path": "/unicaststream/1",
        "rtsp_url": "rtsp://192.168.29.250:554/unicaststream/1",
    })
    config_manager.save_config(cfg)
    mock_test.return_value = {
        "ok": True,
        "auth_state": "valid",
        "host": "192.168.29.250",
        "rtsp_port": 554,
        "stream_path": "/unicaststream/1",
        "preferred_stream": "main",
        "discovery_fingerprint": "rtsp:192.168.29.250:554:/unicaststream/1",
    }

    resp = client.post(
        "/api/cameras/discover/test",
        json={"host": "192.168.29.250", "rtsp_port": 554, "stream_path": "/unicaststream/1"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["duplicate_state"] == "exact"
    assert data["existing_camera_id"] == "cam1"


@mock.patch("routers.cameras.start_camera")
@mock.patch("routers.cameras.test_camera_connection")
@mock.patch("routers.cameras.licensing.can_add_camera")
def test_import_endpoint_partially_succeeds(mock_can_add, mock_test, mock_start_camera):
    mock_can_add.return_value = (True, "")
    mock_test.side_effect = [
        {
            "ok": True,
            "auth_state": "valid",
            "host": "192.168.29.250",
            "rtsp_port": 554,
            "stream_path": "/unicaststream/1",
            "preferred_stream": "main",
            "rtsp_url": "rtsp://192.168.29.250:554/unicaststream/1",
            "discovery_fingerprint": "rtsp:192.168.29.250:554:/unicaststream/1",
            "onvif_uuid": "uuid:new-camera",
        },
        {
            "ok": False,
            "auth_state": "failed",
            "error_code": "auth_failed",
            "error": "Authentication failed",
        },
    ]

    resp = client.post(
        "/api/cameras/discover/import",
        json={
            "devices": [
                {
                    "fingerprint": "row-1",
                    "host": "192.168.29.250",
                    "name": "Gate Camera",
                    "zone": "Main Gate",
                    "profile": "work_zone_ppe",
                    "capabilities": ["helmet_required", "zone_intrusion"],
                    "username": "admin",
                    "password": "admin",
                },
                {
                    "fingerprint": "row-2",
                    "host": "192.168.29.251",
                    "name": "Bad Camera",
                    "zone": "Yard",
                    "profile": "general_safety",
                    "capabilities": ["person_presence"],
                    "username": "admin",
                    "password": "wrong",
                },
            ]
        },
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["created"]) == 1
    assert len(data["failed"]) == 1
    assert data["needs_zone_setup"][0]["needs_zone_setup"] is True
    created_camera_id = data["created"][0]["camera_id"]
    cfg = config_manager.get_config()
    assert cfg["cameras"][created_camera_id]["username"] == "admin"
    assert cfg["cameras"][created_camera_id]["password"] == "admin"
    mock_start_camera.assert_called_once_with(created_camera_id)
