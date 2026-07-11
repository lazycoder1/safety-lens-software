import base64
import json
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


def test_edge_resizes_oversized_remote_frame_and_restores_source_coordinates(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    captured = {}
    remote_records = [
        {"class_id": 0, "confidence": 0.9, "bbox": [100, 50, 900, 500]}
    ]

    def fake_remote_post_jpeg(path, frame_jpeg, *, params):
        captured.update(path=path, frame_jpeg=frame_jpeg, params=params)
        return {"detections": remote_records}

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg", fake_remote_post_jpeg)

    detections = model_manager.predict_records(
        "coco_primary",
        frame,
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    decoded = cv2.imdecode(
        np.frombuffer(captured["frame_jpeg"], np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == (540, 960, 3)
    assert detections == [
        {"class_id": 0, "confidence": 0.9, "bbox": [200, 100, 1800, 1000]}
    ]
    assert remote_records[0]["bbox"] == [100, 50, 900, 500]


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


def test_edge_runs_model_pairs_concurrently_with_one_jpeg_encode(monkeypatch):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    jpeg_objects = []
    encode_parameters = []
    encode_calls = 0
    barrier = threading.Barrier(2)
    real_imencode = cv2.imencode

    def counted_imencode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        encode_parameters.append(args[2])
        return real_imencode(*args, **kwargs)

    def fake_single(model_key, frame_jpeg, **_kwargs):
        jpeg_objects.append(frame_jpeg)
        barrier.wait(timeout=1.0)
        return [{"model_key": model_key}]

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(cv2, "imencode", counted_imencode)
    monkeypatch.setattr(model_manager, "_remote_predict_records_jpeg", fake_single)

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "conf": 0.3, "device": "cuda", "imgsz": 960},
        {"request_id": "ppe", "model_key": "ppe_specialist", "conf": 0.25, "device": "cuda", "imgsz": 960, "classes": ["helmet"]},
    ])

    assert encode_calls == 1
    assert encode_parameters == [[cv2.IMWRITE_JPEG_QUALITY, 85]]
    assert jpeg_objects[0] is jpeg_objects[1]
    assert results == {
        "coco": [{"model_key": "coco_primary"}],
        "ppe": [{"model_key": "ppe_specialist"}],
    }


def test_edge_bounds_grouped_transport_and_restores_source_coordinates(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    decoded_shapes = []
    encode_parameters = []
    real_imencode = cv2.imencode

    def capture_imencode(*args, **kwargs):
        encode_parameters.append(args[2])
        return real_imencode(*args, **kwargs)

    def fake_single(model_key, frame_jpeg, **_kwargs):
        decoded = cv2.imdecode(
            np.frombuffer(frame_jpeg, np.uint8),
            cv2.IMREAD_COLOR,
        )
        decoded_shapes.append(decoded.shape)
        return [{"model_key": model_key, "bbox": [100, 50, 900, 500]}]

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(cv2, "imencode", capture_imencode)
    monkeypatch.setattr(model_manager, "_remote_predict_records_jpeg", fake_single)

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "imgsz": 640},
        {"request_id": "ppe", "model_key": "ppe_specialist", "imgsz": 960},
    ])

    assert encode_parameters == [[cv2.IMWRITE_JPEG_QUALITY, 90]]
    assert decoded_shapes == [(540, 960, 3), (540, 960, 3)]
    assert results["coco"][0]["bbox"] == [200, 100, 1800, 1000]
    assert results["ppe"][0]["bbox"] == [200, 100, 1800, 1000]


def test_edge_uses_batch_route_for_more_than_two_models(monkeypatch):
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    captured = {}

    def fake_remote_batch(path, frame_jpeg, *, batch):
        captured.update(path=path, frame_jpeg=frame_jpeg, batch=batch)
        return {
            "results": {
                item["request_id"]: [{"model_key": item["model_key"]}]
                for item in batch
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg_batch", fake_remote_batch)

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary"},
        {"request_id": "ppe", "model_key": "ppe_specialist"},
        {"request_id": "fire", "model_key": "fire_smoke_specialist"},
    ])

    assert captured["path"] == "/api/infer/jpeg/batch"
    assert [item["request_id"] for item in captured["batch"]] == [
        "coco",
        "ppe",
        "fire",
    ]
    assert results["coco"] == [{"model_key": "coco_primary"}]


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


def test_model_server_batches_models_after_one_decode(monkeypatch):
    frame_ids = []

    def fake_predict(model_key, frame, *, conf, device, imgsz, classes):
        frame_ids.append(id(frame))
        return [{"class_id": 0, "model_key": model_key}]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server.model_manager, "predict_records", fake_predict)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    batch = [
        {"request_id": "coco", "model_key": "coco_primary", "conf": 0.3, "imgsz": 960},
        {"request_id": "ppe", "model_key": "ppe_specialist", "conf": 0.25, "imgsz": 960, "classes": ["helmet"]},
    ]
    response = client.post(
        "/api/infer/jpeg/batch",
        content=_jpeg_bytes(np.zeros((120, 200, 3), dtype=np.uint8)),
        headers={
            "Content-Type": "image/jpeg",
            "X-Rakshak-Inference-Batch": json.dumps(batch),
        },
    )

    assert response.status_code == 200
    assert response.json()["results"]["ppe"][0]["model_key"] == "ppe_specialist"
    assert len(frame_ids) == 2
    assert len(set(frame_ids)) == 1


def test_model_server_rejects_duplicate_batch_request_ids(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    batch = [
        {"request_id": "duplicate", "model_key": "coco_primary"},
        {"request_id": "duplicate", "model_key": "ppe_specialist"},
    ]
    response = client.post(
        "/api/infer/jpeg/batch",
        content=_jpeg_bytes(np.zeros((20, 20, 3), dtype=np.uint8)),
        headers={
            "Content-Type": "image/jpeg",
            "X-Rakshak-Inference-Batch": json.dumps(batch),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Inference batch request IDs must be unique"
