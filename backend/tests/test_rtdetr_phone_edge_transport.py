import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import model_manager


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_remote_rtdetr_phone_batch_posts_bounded_raw_contract(monkeypatch):
    captured = {}

    class Session:
        def post(self, url, *, data, headers, timeout):
            captured.update(
                url=url,
                data=bytes(data),
                headers=headers,
                timeout=timeout,
            )
            return _Response({"results": {"frame-0": [], "frame-1": []}})

    monkeypatch.setattr(model_manager, "_remote_raw_transport_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager,
        "_remote_settings",
        lambda: {
            "url": "http://model-server:8100",
            "token": "secret",
            "timeout_seconds": 5.0,
        },
    )
    monkeypatch.setattr(model_manager, "_remote_session", lambda: Session())
    model_manager._REMOTE_RTDETR_PHONE_SUPPORT.update(
        url=None,
        **{"1": None, "2": None},
    )
    frames = [
        np.full((4, 5, 3), 10, dtype=np.uint8),
        np.full((6, 7, 3), 20, dtype=np.uint8),
    ]
    items = [
        {"frame": frame, "person_conf": 0.3, "phone_conf": 0.15} for frame in frames
    ]

    result = model_manager._remote_post_raw_rtdetr_phone_batch(items)

    assert result == {"results": {"frame-0": [], "frame-1": []}}
    assert captured["url"].endswith("/api/infer/raw/rtdetr-phone-batch2")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    metadata = json.loads(captured["headers"]["X-Rakshak-RTDETR-Phone-Batch"])
    assert [item["byte_length"] for item in metadata] == [
        frames[0].nbytes,
        frames[1].nbytes,
    ]
    assert len(captured["data"]) == sum(frame.nbytes for frame in frames)


def test_rtdetr_phone_batcher_pairs_phase_aligned_frames(monkeypatch):
    calls = []

    def post(items):
        calls.append(len(items))
        return {
            "results": {
                f"frame-{index}": [
                    {
                        "class_id": 67,
                        "confidence": 0.8,
                        "bbox": [1, 2, 3, 4],
                    }
                ]
                for index in range(len(items))
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_raw_transport_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_raw_rtdetr_phone_batch", post)
    batcher = model_manager._RemoteRTDETRPhoneFrameBatcher(0.02)
    frames = [
        np.zeros((40, 60, 3), dtype=np.uint8),
        np.zeros((50, 70, 3), dtype=np.uint8),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                batcher.submit,
                frame,
                person_conf=0.3,
                phone_conf=0.15,
            )
            for frame in frames
        ]
        results = [future.result() for future in futures]

    assert calls == [2]
    assert [result[0]["class_id"] for result in results] == [67, 67]
    assert batcher.stats()["batch2_executed"] == 1
    assert batcher.stats()["batch2_frames"] == 2


def test_rtdetr_phone_batcher_uses_batch1_after_bounded_wait(monkeypatch):
    calls = []
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_raw_transport_enabled", lambda: True)

    def post(items):
        calls.append(len(items))
        return {"results": {"frame-0": []}}

    monkeypatch.setattr(model_manager, "_remote_post_raw_rtdetr_phone_batch", post)
    batcher = model_manager._RemoteRTDETRPhoneFrameBatcher(0.001)

    assert (
        batcher.submit(
            np.zeros((20, 30, 3), dtype=np.uint8),
            person_conf=0.3,
            phone_conf=0.15,
        )
        == []
    )
    assert calls == [1]
    assert batcher.stats()["batch1_executed"] == 1


def test_rtdetr_phone_batcher_fails_closed_without_remote_raw_transport(monkeypatch):
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_raw_transport_enabled", lambda: False)
    batcher = model_manager._RemoteRTDETRPhoneFrameBatcher(0.01)

    with pytest.raises(
        model_manager.RemoteRTDETRPhoneUnavailableError,
        match="remote raw transport",
    ):
        batcher.submit(
            np.zeros((20, 30, 3), dtype=np.uint8),
            person_conf=0.3,
            phone_conf=0.15,
        )
