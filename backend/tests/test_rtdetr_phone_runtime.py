import hashlib

import numpy as np
import pytest

import rtdetr_phone_runtime
from rtdetr_phone_runtime import (
    RTDETRPhoneRuntimePool,
    RTDETRPhoneRuntimeUnavailable,
    records_from_outputs,
)


def test_records_filter_normalize_and_class_aware_nms():
    records = records_from_outputs(
        np.asarray([0, 0, 67, 67, 2, 67, 0, 67], dtype=np.int32),
        np.asarray(
            [
                [-10.2, -5.8, 90.6, 80.2],
                [-9.0, -5.0, 91.0, 81.0],
                [10.0, 10.0, 30.0, 30.0],
                [10.5, 10.5, 29.5, 29.5],
                [1.0, 1.0, 20.0, 20.0],
                [5.0, 5.0, 4.0, 20.0],
                [0.0, 0.0, np.inf, 10.0],
                [2.0, 2.0, 8.0, 8.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.91, 0.80, 0.75, 0.70, 0.99, 0.99, 0.99, 0.14]),
        frame_width=80,
        frame_height=60,
        person_conf=0.3,
        phone_conf=0.15,
    )

    assert records == [
        {"class_id": 0, "confidence": pytest.approx(0.91), "bbox": [0, 0, 80, 60]},
        {"class_id": 67, "confidence": pytest.approx(0.75), "bbox": [10, 10, 30, 30]},
    ]


def test_pool_rejects_partial_configuration_and_reports_failure(tmp_path):
    engine = tmp_path / "phone.engine"
    engine.write_bytes(b"engine")
    pool = RTDETRPhoneRuntimePool(
        {"SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE": str(engine)}
    )

    with pytest.raises(RTDETRPhoneRuntimeUnavailable, match="exact SHA-256"):
        pool.predict(
            [np.zeros((4, 4, 3), dtype=np.uint8)],
            person_conf=0.3,
            phone_conf=0.15,
        )

    assert pool.status()["1"]["backend"] == "failed"
    assert "exact SHA-256" in pool.status()["1"]["error"]


def test_pool_rejects_engine_identity_mismatch_before_loading(tmp_path, monkeypatch):
    engine = tmp_path / "phone.engine"
    engine.write_bytes(b"engine")
    loaded = []
    monkeypatch.setattr(
        rtdetr_phone_runtime,
        "_TensorRTRuntime",
        lambda *_args: loaded.append(True),
    )
    pool = RTDETRPhoneRuntimePool(
        {
            "SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE": str(engine),
            "SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256": "0" * 64,
        }
    )

    with pytest.raises(RTDETRPhoneRuntimeUnavailable, match="SHA-256 mismatch"):
        pool.predict(
            [np.zeros((4, 4, 3), dtype=np.uint8)],
            person_conf=0.3,
            phone_conf=0.15,
        )

    assert loaded == []
    assert pool.status()["1"]["backend"] == "failed"


def test_pool_loads_verified_engine_and_tracks_warm_state(tmp_path, monkeypatch):
    engine = tmp_path / "phone.engine"
    engine.write_bytes(b"verified engine")
    expected_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    calls = []

    class FakeRuntime:
        def __init__(self, engine_path, batch_size):
            calls.append((engine_path, batch_size))

        def predict(self, frames, *, person_conf, phone_conf):
            calls.append((len(frames), person_conf, phone_conf))
            return [[{"class_id": 67}]]

    monkeypatch.setattr(rtdetr_phone_runtime, "_TensorRTRuntime", FakeRuntime)
    pool = RTDETRPhoneRuntimePool(
        {
            "SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE": str(engine),
            "SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256": expected_sha,
        }
    )

    assert pool.predict(
        [np.zeros((4, 5, 3), dtype=np.uint8)],
        person_conf=0.3,
        phone_conf=0.15,
    ) == [[{"class_id": 67}]]
    assert calls == [(engine, 1), (1, 0.3, 0.15)]
    assert pool.status()["1"] == {
        "configured": True,
        "backend": "tensorrt",
        "warmed": True,
        "engine": "phone.engine",
        "error": None,
    }


def test_pool_fences_failed_runtime_instead_of_reusing_it(tmp_path, monkeypatch):
    engine = tmp_path / "phone.engine"
    engine.write_bytes(b"verified engine")
    expected_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    predictions = []

    class BrokenRuntime:
        def __init__(self, *_args):
            pass

        def predict(self, *_args, **_kwargs):
            predictions.append(True)
            raise RuntimeError("cuda fault")

    monkeypatch.setattr(rtdetr_phone_runtime, "_TensorRTRuntime", BrokenRuntime)
    pool = RTDETRPhoneRuntimePool(
        {
            "SAFETYLENS_RTDETR_PHONE_BATCH1_TENSORRT_ENGINE": str(engine),
            "SAFETYLENS_RTDETR_PHONE_BATCH1_ENGINE_SHA256": expected_sha,
        }
    )
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="cuda fault"):
        pool.predict([frame], person_conf=0.3, phone_conf=0.15)
    with pytest.raises(RTDETRPhoneRuntimeUnavailable, match="cuda fault"):
        pool.predict([frame], person_conf=0.3, phone_conf=0.15)

    assert predictions == [True]


def test_pool_rejects_invalid_thresholds_and_frames():
    pool = RTDETRPhoneRuntimePool({})

    with pytest.raises(ValueError, match="thresholds"):
        pool.predict(
            [np.zeros((4, 4, 3), dtype=np.uint8)],
            person_conf=-0.1,
            phone_conf=0.15,
        )
    with pytest.raises(ValueError, match="BGR arrays"):
        pool.predict(
            [np.zeros((4, 4), dtype=np.uint8)],
            person_conf=0.3,
            phone_conf=0.15,
        )
