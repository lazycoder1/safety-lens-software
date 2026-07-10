import base64
import threading

import cv2
import numpy as np
from fastapi.testclient import TestClient

import model_manager
import model_server


def _jpeg_bytes(frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return encoded.tobytes()


def test_remote_http_session_is_reused_per_thread():
    main_session = model_manager._remote_session()
    assert model_manager._remote_session() is main_session
    assert main_session.get_adapter("http://")._pool_maxsize == 16

    worker_sessions = []

    def capture_worker_session():
        worker_sessions.extend(
            [model_manager._remote_session(), model_manager._remote_session()]
        )

    worker = threading.Thread(target=capture_worker_session)
    worker.start()
    worker.join()

    assert worker_sessions[0] is worker_sessions[1]
    assert worker_sessions[0] is not main_session


def test_edge_uses_raw_jpeg_transport(monkeypatch):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    captured = {}
    expected = [{"class_id": 0, "confidence": 0.9, "bbox": [1, 2, 30, 40]}]

    def fake_remote_post_jpeg(path, frame_jpeg, *, params):
        captured.update(path=path, frame_jpeg=frame_jpeg, params=params)
        return {"detections": expected}

    def fail_legacy_transport(*_args, **_kwargs):
        raise AssertionError("legacy transport used")

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg", fake_remote_post_jpeg)
    monkeypatch.setattr(model_manager, "_remote_post", fail_legacy_transport)

    detections = model_manager.predict_records(
        "coco_primary",
        frame,
        conf=0.35,
        device="cuda",
        imgsz=640,
        classes=["person"],
    )

    decoded = cv2.imdecode(np.frombuffer(captured["frame_jpeg"], np.uint8), cv2.IMREAD_COLOR)
    assert captured["path"] == "/api/infer/jpeg"
    assert captured["params"] == {
        "model_key": "coco_primary",
        "conf": 0.35,
        "device": "cuda",
        "imgsz": 640,
        "classes": ["person"],
    }
    assert decoded.shape == frame.shape
    assert detections is expected


def test_edge_falls_back_to_legacy_json_transport(monkeypatch):
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    captured = {}

    def fake_remote_post(_path, payload):
        captured.update(payload)
        return {"detections": []}

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_manager, "_remote_post", fake_remote_post)

    model_manager.predict_records(
        "coco_primary",
        frame,
        conf=0.35,
        device="cuda",
        imgsz=640,
    )

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(captured["frame_jpeg_b64"]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == frame.shape


def test_model_server_accepts_raw_jpeg_and_metadata(monkeypatch):
    captured = {}

    def fake_predict(model_key, frame, *, conf, device, imgsz, classes):
        captured.update(
            model_key=model_key,
            shape=frame.shape,
            conf=conf,
            device=device,
            imgsz=imgsz,
            classes=classes,
        )
        return [{"class_id": 0, "confidence": 0.8, "bbox": [2, 3, 20, 30]}]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server.model_manager, "predict_records", fake_predict)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    response = client.post(
        "/api/infer/jpeg",
        params=[
            ("model_key", "coco_primary"),
            ("conf", "0.42"),
            ("device", "cuda"),
            ("imgsz", "640"),
            ("classes", "person"),
            ("classes", "car"),
        ],
        content=_jpeg_bytes(frame),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json()["detections"][0]["bbox"] == [2, 3, 20, 30]
    assert captured == {
        "model_key": "coco_primary",
        "shape": (120, 200, 3),
        "conf": 0.42,
        "device": "cuda",
        "imgsz": 640,
        "classes": ["person", "car"],
    }


def test_model_server_rejects_invalid_raw_jpeg(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/jpeg",
        params={"model_key": "coco_primary"},
        content=b"not-a-jpeg",
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not decode frame"
