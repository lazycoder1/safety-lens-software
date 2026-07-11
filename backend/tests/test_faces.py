"""Tests for face enrollment, logs, and API wiring."""

import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/rakshak_lens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = Path(tempfile.mkdtemp())

import alert_store
import audit_store
import auth_store
import config_manager
import face_analyzer
import face_store

config_manager.CONFIG_PATH = _tmpdir / "test_config.json"
face_store.FACE_PHOTOS_DIR = _tmpdir / "face_photos"
face_store.FACE_SNAPSHOTS_DIR = _tmpdir / "face_snapshots"

with mock.patch("state.load_model"):
    import server
    import state

client = TestClient(server.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def fresh_state():
    alert_store.init_db()
    audit_store.init_db()
    face_store.init_db()
    with face_store.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE face_logs, enrolled_faces RESTART IDENTITY CASCADE")
            cur.execute("TRUNCATE TABLE audit_log RESTART IDENTITY CASCADE")
        conn.commit()

    config_manager._config = None
    if config_manager.CONFIG_PATH.exists():
        config_manager.CONFIG_PATH.unlink()
    config_manager.load_config()
    state.camera_clean_frames.clear()
    state.camera_frames.clear()
    yield


def admin_headers() -> dict[str, str]:
    token = auth_store.create_token("admin-test", "admin", "admin")
    return {"Authorization": f"Bearer {token}"}


def test_face_init_creates_tables():
    with face_store.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE tablename IN ('enrolled_faces', 'face_logs')")
            names = {row[0] for row in cur.fetchall()}
    assert {"enrolled_faces", "face_logs"} <= names


def test_enroll_rejects_missing_consent():
    resp = client.post(
        "/api/faces/enroll",
        headers=admin_headers(),
        data={
            "name": "Rajesh Kumar",
            "group": "employees",
            "consentMethod": "Written form",
            "consentConfirmed": "false",
        },
        files={"photo": ("face.jpg", b"fake", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "Consent" in resp.json()["detail"]


def test_format_enrollment_photo_outputs_centered_jpeg():
    image = Image.new("RGB", (1200, 900), (20, 30, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    output = face_analyzer.format_enrollment_photo(
        buffer.getvalue(),
        {"x1": 450, "y1": 250, "x2": 650, "y2": 500},
    )

    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (512, 512)


@mock.patch("face_analyzer.extract_enrollment_embedding")
def test_enroll_upload_creates_face_and_audit(mock_extract):
    mock_extract.return_value.embedding = [0.01] * 512
    resp = client.post(
        "/api/faces/enroll",
        headers=admin_headers(),
        data={
            "name": "Rajesh Kumar",
            "group": "employees",
            "consentMethod": "Written form",
            "consentConfirmed": "true",
        },
        files={"photo": ("face.jpg", b"\xff\xd8fake", "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Rajesh Kumar"
    assert data["group"] == "employees"
    assert data["photoUrl"].startswith("/api/faces/")

    faces = client.get("/api/faces", headers=admin_headers()).json()
    assert len(faces) == 1
    assert "embedding" not in faces[0]


@mock.patch("face_analyzer.extract_enrollment_embedding")
def test_enroll_live_uses_latest_camera_frame(mock_extract):
    cfg = config_manager.get_config()
    camera_id = next(iter(cfg["cameras"]))
    state.camera_clean_frames[camera_id] = b"\xff\xd8live"
    mock_extract.return_value.embedding = [0.02] * 512

    resp = client.post(
        "/api/faces/enroll/live",
        headers=admin_headers(),
        json={
            "cameraId": camera_id,
            "name": "Priya Sharma",
            "group": "employees",
            "validUntil": None,
            "consentMethod": "Written form",
            "consentConfirmed": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Priya Sharma"


def test_enroll_live_returns_clear_error_without_frame():
    cfg = config_manager.get_config()
    camera_id = next(iter(cfg["cameras"]))
    resp = client.post(
        "/api/faces/enroll/live",
        headers=admin_headers(),
        json={
            "cameraId": camera_id,
            "name": "No Frame",
            "group": "employees",
            "consentMethod": "Written form",
            "consentConfirmed": True,
        },
    )
    assert resp.status_code == 409
    assert "No live frame" in resp.json()["detail"]


def test_face_logs_return_camel_case_payload():
    face_store.log_face_event(
        camera_id="cam1",
        camera_name="Gate 1",
        event_type="face_unknown",
        bbox={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        quality_reason=None,
    )
    resp = client.get("/api/faces/logs", headers=admin_headers())
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["cameraId"] == "cam1"
    assert row["eventType"] == "face_unknown"
    assert row["personName"] is None
    assert row["isUnknown"] is True


def test_delete_face_deactivates_record():
    face = face_store.create_enrollment(
        name="Delete Me",
        group="employees",
        valid_until=None,
        consent_method="Written form",
        embedding=[0.03] * 512,
        photo_bytes=b"\xff\xd8photo",
    )
    resp = client.delete(f"/api/faces/{face['id']}", headers=admin_headers())
    assert resp.status_code == 200
    assert resp.json()["active"] is False
    assert client.get("/api/faces", headers=admin_headers()).json() == []


def test_face_analyzer_emits_match_unknown_and_low_quality(monkeypatch):
    import numpy as np
    import face_analyzer

    class FakeFace:
        def __init__(self, bbox, score, emb):
            self.bbox = np.array(bbox)
            self.det_score = score
            self.normed_embedding = np.array(emb)

    faces = [
        FakeFace([10, 10, 90, 90], 0.9, [0.1] * 512),
        FakeFace([110, 10, 190, 90], 0.9, [0.2] * 512),
        FakeFace([210, 10, 230, 30], 0.9, [0.3] * 512),
    ]
    monkeypatch.setattr(face_analyzer, "_detect", lambda frame: faces)
    monkeypatch.setattr(
        face_store,
        "find_best_match",
        lambda embedding: {"id": "fc-1", "name": "Rajesh Kumar", "group": "employees", "confidence": 96.0}
        if embedding[0] == 0.1 else None,
    )

    events = face_analyzer.analyze_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert [event["eventType"] for event in events] == ["face_match", "face_unknown", "face_low_quality"]
    assert events[0]["personName"] == "Rajesh Kumar"
