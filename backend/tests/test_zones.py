"""Tests for zone CRUD API and zone intrusion detection logic."""

import os
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Set up test DB
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/safetylens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = tempfile.mkdtemp()
_test_snapshots = Path(_tmpdir) / "snapshots"
_test_config = Path(_tmpdir) / "test_config.json"

import alert_store
alert_store.SNAPSHOTS_DIR = _test_snapshots

import config_manager
config_manager.CONFIG_PATH = _test_config

with mock.patch("server.load_model"):
    import server

server.model = None
server.yoloe_model = None

from fastapi.testclient import TestClient

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def fresh_state():
    alert_store.init_db()
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alerts")
        conn.commit()

    config_manager._config = None
    config_manager.CONFIG_PATH = _test_config
    if _test_config.exists():
        _test_config.unlink()
    config_manager.load_config()
    server.camera_threads.clear()
    server.camera_frames.clear()
    server.camera_detections.clear()
    yield


# ── Zone CRUD API Tests ──────────────────────────────────────────────────────

def test_get_zones_empty():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    resp = client.get(f"/api/cameras/{cam_id}/zones")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_zone():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    resp = client.post(f"/api/cameras/{cam_id}/zones", json={
        "name": "Test Restricted Zone",
        "type": "restricted",
        "color": "#dc2626",
        "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
    })
    assert resp.status_code == 200
    zone = resp.json()
    assert zone["name"] == "Test Restricted Zone"
    assert zone["type"] == "restricted"
    assert len(zone["points"]) == 4
    assert "id" in zone


def test_add_zone_auto_enables_zone_intrusion():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    client.post(f"/api/cameras/{cam_id}/zones", json={
        "name": "Auto Zone",
        "type": "restricted",
        "color": "#dc2626",
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    })
    # Reload config and check alert_classes
    cfg = config_manager.load_config()
    cam = cfg["cameras"][cam_id]
    assert "zone_intrusion" in cam.get("alert_classes", [])


def test_get_zones_after_add():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    client.post(f"/api/cameras/{cam_id}/zones", json={
        "name": "Zone A",
        "type": "restricted",
        "color": "#dc2626",
        "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
    })
    resp = client.get(f"/api/cameras/{cam_id}/zones")
    assert resp.status_code == 200
    zones = resp.json()
    assert len(zones) == 1
    assert zones[0]["name"] == "Zone A"


def test_delete_zone():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    resp = client.post(f"/api/cameras/{cam_id}/zones", json={
        "name": "To Delete",
        "type": "caution",
        "color": "#f59e0b",
        "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
    })
    zone_id = resp.json()["id"]
    resp = client.delete(f"/api/cameras/{cam_id}/zones/{zone_id}")
    assert resp.status_code == 200
    # Verify deleted
    resp = client.get(f"/api/cameras/{cam_id}/zones")
    assert len(resp.json()) == 0


def test_zone_not_found_camera():
    resp = client.get("/api/cameras/nonexistent/zones")
    assert resp.status_code == 404


def test_add_multiple_zones():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    for i in range(3):
        client.post(f"/api/cameras/{cam_id}/zones", json={
            "name": f"Zone {i}",
            "type": "restricted",
            "color": "#dc2626",
            "points": [[0.1 * i, 0.1], [0.1 * i + 0.1, 0.1], [0.1 * i + 0.1, 0.2]],
        })
    resp = client.get(f"/api/cameras/{cam_id}/zones")
    assert len(resp.json()) == 3


# ── Point-in-Polygon Tests ──────────────────────────────────────────────────

def test_point_in_polygon_inside():
    polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert server.point_in_polygon(0.5, 0.5, polygon) is True


def test_point_in_polygon_outside():
    polygon = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    assert server.point_in_polygon(0.8, 0.8, polygon) is False


def test_point_in_polygon_triangle():
    polygon = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    assert server.point_in_polygon(0.5, 0.3, polygon) is True
    assert server.point_in_polygon(0.9, 0.9, polygon) is False


# ── Zone Intrusion Detection Tests ───────────────────────────────────────────

def test_check_zone_intrusions_person_in_zone():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    # Add zone covering entire frame
    cfg["cameras"][cam_id]["zones"] = [{
        "id": "z1", "name": "Full Zone", "type": "restricted",
        "color": "#dc2626", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }]
    cfg["cameras"][cam_id]["alert_classes"] = ["zone_intrusion"]
    config_manager.save_config(cfg)

    dets = [{"class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
    candidates = server.check_zone_intrusions(dets, cam_id, 640, 480)
    assert len(candidates) == 1
    assert candidates[0]["rule"] == "Zone Intrusion"
    assert candidates[0]["severity"] == "P1"
    assert "Full Zone" in candidates[0]["description"]


def test_check_zone_intrusions_person_outside_zone():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    # Zone in top-left corner only
    cfg["cameras"][cam_id]["zones"] = [{
        "id": "z1", "name": "Corner", "type": "restricted",
        "color": "#dc2626", "points": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]
    }]
    cfg["cameras"][cam_id]["alert_classes"] = ["zone_intrusion"]
    config_manager.save_config(cfg)

    # Person in bottom-right
    dets = [{"class": "person", "confidence": 0.9, "bbox": [500, 400, 600, 450]}]
    candidates = server.check_zone_intrusions(dets, cam_id, 640, 480)
    assert len(candidates) == 0


def test_check_zone_intrusions_disabled():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    cfg["cameras"][cam_id]["zones"] = [{
        "id": "z1", "name": "Zone", "type": "restricted",
        "color": "#dc2626", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }]
    cfg["cameras"][cam_id]["alert_classes"] = ["mobile_phone"]  # zone_intrusion NOT enabled
    config_manager.save_config(cfg)

    dets = [{"class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
    candidates = server.check_zone_intrusions(dets, cam_id, 640, 480)
    assert len(candidates) == 0


def test_check_zone_intrusions_caution_zone():
    cfg = config_manager.get_config()
    cam_id = list(cfg["cameras"].keys())[0]
    cfg["cameras"][cam_id]["zones"] = [{
        "id": "z1", "name": "Caution Zone", "type": "caution",
        "color": "#f59e0b", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }]
    cfg["cameras"][cam_id]["alert_classes"] = ["zone_intrusion"]
    config_manager.save_config(cfg)

    dets = [{"class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
    candidates = server.check_zone_intrusions(dets, cam_id, 640, 480)
    assert len(candidates) == 1
    assert candidates[0]["severity"] == "P2"  # caution = P2, not P1
