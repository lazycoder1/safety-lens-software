import json

import numpy as np
from fastapi.testclient import TestClient

import model_server
from rtdetr_phone_runtime import RTDETRPhoneRuntimeUnavailable


def _metadata(frames):
    return [
        {
            "request_id": f"frame-{index}",
            "person_conf": 0.3,
            "phone_conf": 0.15,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index, frame in enumerate(frames)
    ]


def test_model_server_accepts_rtdetr_phone_batch2(monkeypatch):
    frames = [
        np.full((4, 5, 3), 10, dtype=np.uint8),
        np.full((6, 7, 3), 20, dtype=np.uint8),
    ]
    captured = {}

    def fake_predict(raw_frames, *, person_conf, phone_conf):
        captured.update(
            shapes=[frame.shape for frame in raw_frames],
            values=[int(frame[0, 0, 0]) for frame in raw_frames],
            person_conf=person_conf,
            phone_conf=phone_conf,
        )
        return [[{"class_id": 0}], [{"class_id": 67}]]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server.RTDETR_PHONE_RUNTIME_POOL, "predict", fake_predict)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/raw/rtdetr-phone-batch2",
        content=b"".join(frame.tobytes() for frame in frames),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-RTDETR-Phone-Batch": json.dumps(_metadata(frames)),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": {
            "frame-0": [{"class_id": 0}],
            "frame-1": [{"class_id": 67}],
        }
    }
    assert captured == {
        "shapes": [(4, 5, 3), (6, 7, 3)],
        "values": [10, 20],
        "person_conf": 0.3,
        "phone_conf": 0.15,
    }


def test_model_server_rtdetr_phone_batch1_requires_auth(monkeypatch):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "secret")
    monkeypatch.setattr(
        model_server.RTDETR_PHONE_RUNTIME_POOL,
        "predict",
        lambda *_args, **_kwargs: [[]],
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Rakshak-RTDETR-Phone-Batch": json.dumps(_metadata([frame])),
    }

    assert (
        client.post(
            "/api/infer/raw/rtdetr-phone-batch1",
            content=frame.tobytes(),
            headers=headers,
        ).status_code
        == 401
    )
    response = client.post(
        "/api/infer/raw/rtdetr-phone-batch1",
        content=frame.tobytes(),
        headers={**headers, "Authorization": "Bearer secret"},
    )
    assert response.status_code == 200


def test_model_server_rtdetr_phone_rejects_bad_frame_length(monkeypatch):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/raw/rtdetr-phone-batch1",
        content=frame.tobytes()[:-1],
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-RTDETR-Phone-Batch": json.dumps(_metadata([frame])),
        },
    )

    assert response.status_code == 400


def test_model_server_rtdetr_phone_unavailable_maps_to_conflict(monkeypatch):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")

    def unavailable(*_args, **_kwargs):
        raise RTDETRPhoneRuntimeUnavailable("engine unavailable")

    monkeypatch.setattr(model_server.RTDETR_PHONE_RUNTIME_POOL, "predict", unavailable)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/raw/rtdetr-phone-batch1",
        content=frame.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-RTDETR-Phone-Batch": json.dumps(_metadata([frame])),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "engine unavailable"


def test_model_server_health_reports_optional_rtdetr_status(monkeypatch):
    expected = {"1": {"configured": False}, "2": {"configured": True}}
    monkeypatch.setattr(
        model_server.RTDETR_PHONE_RUNTIME_POOL,
        "status",
        lambda: expected,
    )

    assert model_server.health()["rtdetr_phone_runtimes"] == expected
