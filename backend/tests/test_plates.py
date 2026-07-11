"""Tests for ANPR plate storage, API wiring, and video-processing integration."""

import os
import tempfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/rakshak_lens_test")
os.environ["DATABASE_URL"] = TEST_DB_URL

_tmpdir = Path(tempfile.mkdtemp())

import alert_store
import audit_store
import auth_store
import config_manager
import plate_store

config_manager.CONFIG_PATH = _tmpdir / "test_config.json"
plate_store.PLATE_SNAPSHOTS_DIR = _tmpdir / "plate_snapshots"
plate_store.PLATE_CROPS_DIR = _tmpdir / "plate_crops"

with mock.patch("state.load_model"):
    import plate_analyzer
    import server
    import state
    import video_processing

client = TestClient(server.app, raise_server_exceptions=False)


def admin_headers() -> dict[str, str]:
    token = auth_store.create_token("admin-test", "admin", "admin")
    return {"Authorization": f"Bearer {token}"}


def setup_function():
    alert_store.init_db()
    audit_store.init_db()
    plate_store.init_db()
    with plate_store.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE plate_reads, plate_lists RESTART IDENTITY CASCADE")
            cur.execute("TRUNCATE TABLE audit_log RESTART IDENTITY CASCADE")
        conn.commit()
    config_manager._config = None
    if config_manager.CONFIG_PATH.exists():
        config_manager.CONFIG_PATH.unlink()
    config_manager.load_config()
    state.camera_detections.clear()


def test_plate_init_creates_tables():
    with plate_store.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE tablename IN ('plate_reads', 'plate_lists')")
            names = {row[0] for row in cur.fetchall()}
    assert {"plate_reads", "plate_lists"} <= names


def test_plate_list_crud_returns_camel_case_payload():
    resp = client.post(
        "/api/plates/lists",
        headers=admin_headers(),
        json={
            "plateNumber": "ka 05 mn 4523",
            "list": "whitelist",
            "owner": "Rajesh Kumar",
            "vehicle": "White Innova",
            "validUntil": None,
        },
    )
    assert resp.status_code == 200
    entry = resp.json()
    assert entry["normalizedPlate"] == "KA05MN4523"
    assert entry["isIndianFormat"] is True

    resp = client.put(
        f"/api/plates/lists/{entry['id']}",
        headers=admin_headers(),
        json={**entry, "plateNumber": "KA05MN4524", "list": "blocked"},
    )
    assert resp.status_code == 200
    assert resp.json()["list"] == "blocked"

    resp = client.delete(f"/api/plates/lists/{entry['id']}", headers=admin_headers())
    assert resp.status_code == 200
    assert resp.json()["active"] is False
    assert client.get("/api/plates/lists", headers=admin_headers()).json() == []


def test_duplicate_active_plate_in_same_list_is_rejected():
    payload = {"plateNumber": "KA05MN4523", "list": "whitelist", "owner": "", "vehicle": ""}
    assert client.post("/api/plates/lists", headers=admin_headers(), json=payload).status_code == 200
    resp = client.post("/api/plates/lists", headers=admin_headers(), json=payload)
    assert resp.status_code == 409


def test_expired_visitor_does_not_match_read():
    plate_store.create_plate_entry(
        plate_text="KA05MN4523",
        list_type="visitors",
        owner_name="Expired Visitor",
        vehicle_desc="Sedan",
        valid_until="2020-01-01T00:00:00+00:00",
    )
    assert plate_store.find_matching_plate("KA05MN4523") is None


def test_similar_plate_match_uses_ocr_confusion_score():
    entry = plate_store.create_plate_entry(plate_text="KA19MD6157", list_type="whitelist")
    match = plate_store.find_similar_plate("KA19HD6157")

    assert match is not None
    assert match["id"] == entry["id"]
    assert match["similarityScore"] >= 0.88


def test_plate_analyzer_normalizes_common_indian_ocr_confusions():
    assert plate_analyzer.normalize_plate_text("MHOIA5755") == "MH01A5755"
    assert plate_analyzer.normalize_plate_text("KA05MN4523") == "KA05MN4523"
    assert plate_analyzer.normalize_plate_text("IND") == "IND"


def test_csv_import_reports_success_and_failures():
    csv_blob = b"plate,list,owner,vehicle\nKA05MN4523,whitelist,Rajesh,Car\n,blocked,Bad,Car\n"
    resp = client.post(
        "/api/plates/lists/import",
        headers=admin_headers(),
        files={"file": ("plates.csv", csv_blob, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["created"]) == 1
    assert len(data["failed"]) == 1


def test_reads_query_filters_by_plate_camera_event():
    plate_store.log_plate_read(
        plate_text="KA05MN4523",
        camera_id="gate-1",
        camera_name="Gate 1",
        event_type="plate_unknown",
        confidence=88.0,
        bbox={"x1": 1, "y1": 2, "x2": 3, "y2": 4},
    )
    plate_store.log_plate_read(
        plate_text="MH04JK2345",
        camera_id="gate-2",
        camera_name="Gate 2",
        event_type="plate_blocked",
        confidence=91.0,
    )
    resp = client.get("/api/plates/reads?plate=KA05&cameraId=gate-1&eventType=plate_unknown", headers=admin_headers())
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["normalizedPlate"] == "KA05MN4523"
    assert rows[0]["matchStatus"] == "unknown"


def test_plate_analyzer_rejects_implausible_detector_boxes(monkeypatch):
    frame = __import__("numpy").zeros((120, 240, 3), dtype="uint8")
    monkeypatch.setattr(
        plate_analyzer.model_manager,
        "predict_records",
        lambda *args, **kwargs: [
            {"bbox": [0, 0, 200, 20], "confidence": 0.95},
            {"bbox": [20, 40, 120, 64], "confidence": 0.90},
        ],
    )
    monkeypatch.setattr(plate_analyzer, "_read_plate_text", lambda crop: ("KA05MN4523", 0.92))

    candidates = plate_analyzer.analyze_frame(frame, conf=0.35, device="cpu", imgsz=640)

    assert len(candidates) == 1
    assert candidates[0]["normalizedPlate"] == "KA05MN4523"
    assert candidates[0]["bbox"] == {"x1": 20, "y1": 40, "x2": 120, "y2": 64}


def test_plate_analyzer_rejects_invalid_or_low_confidence_ocr(monkeypatch):
    frame = __import__("numpy").zeros((120, 240, 3), dtype="uint8")
    monkeypatch.setattr(
        plate_analyzer.model_manager,
        "predict_records",
        lambda *args, **kwargs: [
            {"bbox": [20, 40, 120, 64], "confidence": 0.95},
            {"bbox": [30, 70, 130, 94], "confidence": 0.95},
        ],
    )
    reads = iter([
        ("EXIT", 0.98),
        ("KA05MN4523", 0.20),
    ])
    monkeypatch.setattr(plate_analyzer, "_read_plate_text", lambda crop: next(reads))

    candidates = plate_analyzer.analyze_frame(frame, conf=0.35, device="cpu", imgsz=640)

    assert candidates == []


def test_plate_recognition_video_path_logs_and_dedupes(monkeypatch):
    plate_store.create_plate_entry(plate_text="KA05MN4523", list_type="blocked")
    frame = __import__("numpy").zeros((120, 240, 3), dtype="uint8")
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_plate_records",
        lambda *args, **kwargs: [{
            "plateText": "KA05MN4523",
            "normalizedPlate": "KA05MN4523",
            "confidence": 94.0,
            "detectionConfidence": 96.0,
            "ocrConfidence": 92.0,
            "bbox": {"x1": 10, "y1": 20, "x2": 90, "y2": 50},
            "qualityReason": None,
        }],
    )

    last_log = {}
    vote_window = []
    annotated, detections = video_processing._run_plate_recognition(
        "gate-1",
        frame,
        frame.copy(),
        {"name": "Gate 1"},
        last_log,
        vote_window,
        conf=0.35,
        device="cpu",
        imgsz=640,
    )
    assert annotated is not None
    assert detections[0]["class"] == "plate_blocked"
    assert len(plate_store.query_reads(camera_id="gate-1")) == 1

    video_processing._run_plate_recognition(
        "gate-1",
        frame,
        frame.copy(),
        {"name": "Gate 1"},
        last_log,
        vote_window,
        conf=0.35,
        device="cpu",
        imgsz=640,
    )
    assert len(plate_store.query_reads(camera_id="gate-1")) == 1


def test_plate_recognition_confirms_fuzzy_match_after_multiple_frames(monkeypatch):
    plate_store.create_plate_entry(plate_text="KA19MD6157", list_type="whitelist")
    frame = __import__("numpy").zeros((120, 240, 3), dtype="uint8")
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_plate_records",
        lambda *args, **kwargs: [{
            "plateText": "KA19HD6157",
            "normalizedPlate": "KA19HD6157",
            "confidence": 92.0,
            "detectionConfidence": 91.0,
            "ocrConfidence": 94.0,
            "bbox": {"x1": 10, "y1": 20, "x2": 120, "y2": 50},
            "qualityReason": None,
        }],
    )

    last_log = {}
    vote_window = []
    video_processing._run_plate_recognition(
        "gate-fuzzy",
        frame,
        frame.copy(),
        {"name": "Gate Fuzzy"},
        last_log,
        vote_window,
        conf=0.35,
        device="cpu",
        imgsz=640,
    )
    assert plate_store.query_reads(camera_id="gate-fuzzy") == []

    _annotated, detections = video_processing._run_plate_recognition(
        "gate-fuzzy",
        frame,
        frame.copy(),
        {"name": "Gate Fuzzy"},
        last_log,
        vote_window,
        conf=0.35,
        device="cpu",
        imgsz=640,
    )

    rows = plate_store.query_reads(camera_id="gate-fuzzy")
    assert len(rows) == 1
    assert rows[0]["normalizedPlate"] == "KA19HD6157"
    assert rows[0]["matchStatus"] == "whitelist"
    assert rows[0]["qualityReason"].startswith("Similar match to registered plate KA19MD6157")
    assert detections[0]["match_kind"] == "similar"
