import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import model_manager
import model_server
import plate_analyzer


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


def test_edge_restores_pose_keypoints_to_source_coordinates(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    remote_records = [
        {
            "class_id": 0,
            "confidence": 0.9,
            "bbox": [100, 50, 900, 500],
            "keypoints": [[100.5, 50.25, 0.8], [900.0, 500.0, 0.6]],
        }
    ]
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager,
        "_remote_post_jpeg",
        lambda *_args, **_kwargs: {"detections": remote_records},
    )

    detections = model_manager.predict_records(
        "pose_specialist",
        frame,
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    assert detections == [
        {
            "class_id": 0,
            "confidence": 0.9,
            "bbox": [200, 100, 1800, 1000],
            "keypoints": [[201.0, 100.5, 0.8], [1800.0, 1000.0, 0.6]],
        }
    ]
    assert remote_records[0]["keypoints"] == [
        [100.5, 50.25, 0.8],
        [900.0, 500.0, 0.6],
    ]


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


def test_edge_sends_model_pairs_as_one_jpeg_batch(monkeypatch):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    encode_parameters = []
    encode_calls = 0
    captured = {}
    real_imencode = cv2.imencode

    def counted_imencode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        encode_parameters.append(args[2])
        return real_imencode(*args, **kwargs)

    def fake_batch(path, frame_jpeg, *, batch):
        captured.update(path=path, frame_jpeg=frame_jpeg, batch=batch)
        return {
            "results": {
                item["request_id"]: [{"model_key": item["model_key"]}]
                for item in batch
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(cv2, "imencode", counted_imencode)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg_batch", fake_batch)
    monkeypatch.setattr(
        model_manager,
        "_remote_predict_records_jpeg",
        lambda *_args, **_kwargs: pytest.fail("batch-capable pair used fallback transport"),
    )

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "conf": 0.3, "device": "cuda", "imgsz": 960},
        {"request_id": "ppe", "model_key": "ppe_specialist", "conf": 0.25, "device": "cuda", "imgsz": 960, "classes": ["helmet"]},
    ])

    assert encode_calls == 1
    assert encode_parameters == [[cv2.IMWRITE_JPEG_QUALITY, 85]]
    assert captured["path"] == "/api/infer/jpeg/batch"
    assert [item["request_id"] for item in captured["batch"]] == ["coco", "ppe"]
    assert results == {
        "coco": [{"model_key": "coco_primary"}],
        "ppe": [{"model_key": "ppe_specialist"}],
    }


def test_edge_rejects_stale_pair_before_model_request_queue(monkeypatch):
    class RejectAdmission:
        def acquire(self, **kwargs):
            assert kwargs == {
                "timeout": model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            }
            return False

        def release(self):
            pytest.fail("Rejected admission must not release an unclaimed slot")

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", RejectAdmission())
    monkeypatch.setattr(
        cv2,
        "imencode",
        lambda *_args, **_kwargs: pytest.fail("Rejected pair reached JPEG encoding"),
    )
    monkeypatch.setattr(
        model_manager,
        "_remote_predict_records_jpeg",
        lambda *_args, **_kwargs: pytest.fail("Rejected pair reached model transport"),
    )

    with pytest.raises(model_manager.RemoteInferenceOverloadedError):
        model_manager.predict_record_batches(
            np.zeros((180, 320, 3), dtype=np.uint8),
            [
                {"request_id": "coco", "model_key": "coco_primary"},
                {"request_id": "ppe", "model_key": "ppe_specialist"},
            ],
        )


def test_edge_rejects_stale_single_before_jpeg_encoding(monkeypatch):
    class RejectAdmission:
        def acquire(self, **kwargs):
            assert kwargs == {
                "timeout": model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            }
            return False

        def release(self):
            pytest.fail("Rejected admission must not release an unclaimed slot")

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", RejectAdmission())
    monkeypatch.setattr(
        cv2,
        "imencode",
        lambda *_args, **_kwargs: pytest.fail("Rejected single reached JPEG encoding"),
    )

    with pytest.raises(model_manager.RemoteInferenceOverloadedError):
        model_manager.predict_records(
            "coco_primary",
            np.zeros((180, 320, 3), dtype=np.uint8),
            conf=0.3,
            device="cuda",
            imgsz=960,
        )


def test_edge_pair_uses_parallel_fallback_for_older_model_server(monkeypatch):
    barrier = threading.Barrier(2)
    calls = []

    def fake_single(model_key, _frame_jpeg, **_kwargs):
        calls.append(model_key)
        barrier.wait(timeout=1.0)
        return [{"model_key": model_key}]

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg_batch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_manager, "_remote_predict_records_jpeg", fake_single)

    results = model_manager.predict_record_batches(
        np.zeros((180, 320, 3), dtype=np.uint8),
        [
            {"request_id": "coco", "model_key": "coco_primary"},
            {"request_id": "ppe", "model_key": "ppe_specialist"},
        ],
    )

    assert set(calls) == {"coco_primary", "ppe_specialist"}
    assert results == {
        "coco": [{"model_key": "coco_primary"}],
        "ppe": [{"model_key": "ppe_specialist"}],
    }


def test_long_tail_runtime_loads_on_first_prediction(monkeypatch, tmp_path):
    runtime = model_manager._MODEL_RUNTIMES["yoloe_long_tail"]
    source = tmp_path / "long-tail.pt"
    source.write_bytes(b"model")
    handle = object()
    load_calls = []

    monkeypatch.setitem(runtime, "handle", None)
    monkeypatch.setitem(runtime, "runtime_backend", "lazy")
    monkeypatch.setitem(
        model_manager._MODEL_STATES,
        "yoloe_long_tail",
        {
            **model_manager._MODEL_STATES["yoloe_long_tail"],
            "status": "ready",
            "active_path": source,
        },
    )

    def load_runtime(model_key, path):
        load_calls.append((model_key, path))
        runtime["handle"] = handle
        runtime["runtime_backend"] = "pytorch"

    monkeypatch.setattr(model_manager, "_load_runtime", load_runtime)
    monkeypatch.setattr(model_manager, "_sync_state_compat", lambda: None)
    monkeypatch.setattr(model_manager, "_runtime_device", lambda _device: "cpu")
    monkeypatch.setattr(
        model_manager,
        "_predict_with_runtime",
        lambda model_key, loaded_runtime, *_args, **_kwargs: (
            model_key,
            loaded_runtime["handle"],
        ),
    )

    result = model_manager.predict(
        "yoloe_long_tail",
        np.zeros((20, 20, 3), dtype=np.uint8),
        conf=0.3,
        device="cuda",
        imgsz=960,
        classes=["fire"],
    )

    assert load_calls == [("yoloe_long_tail", source)]
    assert result == ("yoloe_long_tail", handle)


def test_edge_bounds_grouped_transport_and_restores_source_coordinates(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    decoded_shapes = []
    encode_parameters = []
    real_imencode = cv2.imencode

    def capture_imencode(*args, **kwargs):
        encode_parameters.append(args[2])
        return real_imencode(*args, **kwargs)

    def fake_batch(_path, frame_jpeg, *, batch):
        decoded = cv2.imdecode(
            np.frombuffer(frame_jpeg, np.uint8),
            cv2.IMREAD_COLOR,
        )
        decoded_shapes.append(decoded.shape)
        return {
            "results": {
                item["request_id"]: [
                    {"model_key": item["model_key"], "bbox": [100, 50, 900, 500]}
                ]
                for item in batch
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.delenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", raising=False)
    monkeypatch.setattr(cv2, "imencode", capture_imencode)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg_batch", fake_batch)

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "imgsz": 640},
        {"request_id": "ppe", "model_key": "ppe_specialist", "imgsz": 960},
    ])

    assert encode_parameters == [[cv2.IMWRITE_JPEG_QUALITY, 90]]
    assert decoded_shapes == [(540, 960, 3)]
    assert results["coco"][0]["bbox"] == [200, 100, 1800, 1000]
    assert results["ppe"][0]["bbox"] == [200, 100, 1800, 1000]


def test_edge_uses_opt_in_raw_batch_and_restores_source_coordinates(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    captured = {}

    def fake_raw(path, inference_frame, *, batch):
        captured.update(
            path=path,
            shape=inference_frame.shape,
            contiguous=inference_frame.flags.c_contiguous,
            batch=batch,
        )
        return {
            "results": {
                item["request_id"]: [
                    {"model_key": item["model_key"], "bbox": [100, 50, 900, 500]}
                ]
                for item in batch
            }
        }

    monkeypatch.setenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", "true")
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_raw_batch", fake_raw)
    monkeypatch.setattr(
        model_manager,
        "_remote_post_jpeg_batch",
        lambda *_args, **_kwargs: pytest.fail("raw-capable edge encoded JPEG"),
    )

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "imgsz": 640},
        {"request_id": "ppe", "model_key": "ppe_specialist", "imgsz": 960},
    ])

    assert captured["path"] == "/api/infer/raw/batch"
    assert captured["shape"] == (540, 960, 3)
    assert captured["contiguous"] is True
    assert results["coco"][0]["bbox"] == [200, 100, 1800, 1000]
    assert results["ppe"][0]["bbox"] == [200, 100, 1800, 1000]


def test_edge_raw_batch_falls_back_to_jpeg_for_older_server(monkeypatch):
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    captured = {}

    def fake_jpeg(path, frame_jpeg, *, batch):
        captured.update(path=path, frame_jpeg=frame_jpeg, batch=batch)
        return {
            "results": {
                item["request_id"]: [{"model_key": item["model_key"]}]
                for item in batch
            }
        }

    monkeypatch.setenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", "true")
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_raw_batch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg_batch", fake_jpeg)

    results = model_manager.predict_record_batches(frame, [
        {"request_id": "coco", "model_key": "coco_primary", "imgsz": 640},
    ])

    decoded = cv2.imdecode(
        np.frombuffer(captured["frame_jpeg"], np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert captured["path"] == "/api/infer/jpeg/batch"
    assert decoded.shape == frame.shape
    assert results["coco"] == [{"model_key": "coco_primary"}]


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


def test_model_server_reuses_and_bounds_identical_jpeg_decodes(monkeypatch):
    model_server._clear_decode_cache()
    decode_calls = 0
    real_imdecode = model_server.cv2.imdecode

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(model_server.cv2, "imdecode", counted_decode)
    jpeg_frames = [
        _jpeg_bytes(np.full((120, 200, 3), value, dtype=np.uint8))
        for value in (0, 80, 160)
    ]

    first = model_server._decode_frame(jpeg_frames[0])
    repeated = model_server._decode_frame(jpeg_frames[0])
    model_server._decode_frame(jpeg_frames[1])
    model_server._decode_frame(jpeg_frames[2])
    model_server._decode_frame(jpeg_frames[0])

    assert first is repeated
    assert decode_calls == 4
    assert len(model_server._DECODE_CACHE) == model_server._DECODE_CACHE_MAX_ENTRIES
    model_server._clear_decode_cache()


def test_model_server_decodes_different_jpegs_concurrently(monkeypatch):
    model_server._clear_decode_cache()
    real_imdecode = model_server.cv2.imdecode
    both_decoders_entered = threading.Event()
    release_decoders = threading.Event()
    decode_calls = 0
    decode_calls_lock = threading.Lock()

    def blocked_decode(*args, **kwargs):
        nonlocal decode_calls
        with decode_calls_lock:
            decode_calls += 1
            if decode_calls == 2:
                both_decoders_entered.set()
        assert release_decoders.wait(1.0)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(model_server.cv2, "imdecode", blocked_decode)
    jpeg_frames = [
        _jpeg_bytes(np.full((120, 200, 3), value, dtype=np.uint8))
        for value in (20, 180)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model_server._decode_frame, jpeg) for jpeg in jpeg_frames]
        assert both_decoders_entered.wait(0.5)
        release_decoders.set()
        decoded = [future.result() for future in futures]

    assert decode_calls == 2
    assert [int(frame[0, 0, 0]) for frame in decoded] == [20, 180]
    model_server._clear_decode_cache()


def test_model_server_singleflights_concurrent_duplicate_jpeg(monkeypatch):
    model_server._clear_decode_cache()
    real_imdecode = model_server.cv2.imdecode
    decoder_entered = threading.Event()
    release_decoder = threading.Event()
    decode_calls = 0

    def blocked_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        decoder_entered.set()
        assert release_decoder.wait(1.0)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(model_server.cv2, "imdecode", blocked_decode)
    jpeg = _jpeg_bytes(np.full((120, 200, 3), 80, dtype=np.uint8))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(model_server._decode_frame, jpeg)
        assert decoder_entered.wait(0.5)
        second = pool.submit(model_server._decode_frame, jpeg)
        release_decoder.set()
        first_frame = first.result()
        second_frame = second.result()

    assert decode_calls == 1
    assert first_frame is second_frame
    model_server._clear_decode_cache()


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
    worker_threads = []
    both_models_entered = threading.Barrier(2)

    def fake_predict(model_key, frame, *, conf, device, imgsz, classes):
        frame_ids.append(id(frame))
        worker_threads.append(threading.get_ident())
        both_models_entered.wait(timeout=1.0)
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
    assert len(set(worker_threads)) == 2


def test_model_server_accepts_raw_bgr_batch(monkeypatch):
    frame_ids = []
    captured_shapes = []

    def fake_predict(model_key, frame, *, conf, device, imgsz, classes):
        frame_ids.append(id(frame))
        captured_shapes.append(frame.shape)
        return [{"class_id": 0, "model_key": model_key}]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server.model_manager, "predict_records", fake_predict)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    batch = [
        {"request_id": "coco", "model_key": "coco_primary", "imgsz": 640},
        {"request_id": "ppe", "model_key": "ppe_specialist", "imgsz": 640, "classes": ["helmet"]},
    ]
    response = client.post(
        "/api/infer/raw/batch",
        content=frame.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Inference-Batch": json.dumps(batch),
            "X-Rakshak-Frame-Width": "200",
            "X-Rakshak-Frame-Height": "120",
            "X-Rakshak-Frame-Channels": "3",
        },
    )

    assert response.status_code == 200
    assert response.json()["results"]["ppe"][0]["model_key"] == "ppe_specialist"
    assert captured_shapes == [(120, 200, 3), (120, 200, 3)]
    assert len(set(frame_ids)) == 1


def test_model_server_rejects_raw_frame_length_mismatch(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/raw/batch",
        content=b"too-short",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Inference-Batch": json.dumps([
                {"request_id": "coco", "model_key": "coco_primary"}
            ]),
            "X-Rakshak-Frame-Width": "20",
            "X-Rakshak-Frame-Height": "20",
            "X-Rakshak-Frame-Channels": "3",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Raw frame byte length does not match shape"


def test_model_server_rejects_invalid_raw_frame_shape(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/infer/raw/batch",
        content=np.zeros((20, 20, 4), dtype=np.uint8).tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Inference-Batch": json.dumps([
                {"request_id": "coco", "model_key": "coco_primary"}
            ]),
            "X-Rakshak-Frame-Width": "20",
            "X-Rakshak-Frame-Height": "20",
            "X-Rakshak-Frame-Channels": "4",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid raw frame shape"


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


def test_edge_uses_raw_jpeg_anpr_transport(monkeypatch):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    captured = {}
    expected = [{"plateText": "KA05MN4523"}]

    def fake_raw(path, frame_jpeg, *, params):
        captured.update(path=path, frame_jpeg=frame_jpeg, params=params)
        return {"plates": expected}

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg", fake_raw)
    monkeypatch.setattr(
        model_manager,
        "_remote_post",
        lambda *_args, **_kwargs: pytest.fail("raw-capable ANPR used JSON fallback"),
    )

    plates = model_manager.predict_plate_records(
        frame,
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    decoded = cv2.imdecode(
        np.frombuffer(captured["frame_jpeg"], np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == frame.shape
    assert captured["path"] == "/api/anpr/jpeg"
    assert captured["params"] == {"conf": 0.25, "device": "cuda", "imgsz": 960}
    assert plates is expected


def test_edge_anpr_falls_back_to_legacy_json(monkeypatch):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    captured = {}

    def fake_legacy(path, payload):
        captured.update(path=path, payload=payload)
        return {"plates": []}

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_remote_post_jpeg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_manager, "_remote_post", fake_legacy)

    model_manager.predict_plate_records(
        frame,
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(captured["payload"]["frame_jpeg_b64"]), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == frame.shape
    assert captured["path"] == "/api/anpr"


def test_edge_rejects_stale_anpr_before_jpeg_encoding(monkeypatch):
    class RejectAdmission:
        def acquire(self, **kwargs):
            assert kwargs == {
                "timeout": model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            }
            return False

        def release(self):
            pytest.fail("Rejected ANPR must not release an unclaimed slot")

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", RejectAdmission())
    monkeypatch.setattr(
        cv2,
        "imencode",
        lambda *_args, **_kwargs: pytest.fail("Rejected ANPR reached JPEG encoding"),
    )

    with pytest.raises(model_manager.RemoteInferenceOverloadedError):
        model_manager.predict_plate_records(
            np.zeros((120, 200, 3), dtype=np.uint8),
            conf=0.25,
            device="cuda",
            imgsz=960,
        )


def test_model_server_accepts_raw_jpeg_anpr(monkeypatch):
    captured = {}

    def fake_analyze(frame, *, conf, device, imgsz):
        captured.update(shape=frame.shape, conf=conf, device=device, imgsz=imgsz)
        return [{"plateText": "KA05MN4523"}]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(plate_analyzer, "analyze_frame", fake_analyze)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    response = client.post(
        "/api/anpr/jpeg",
        params={"conf": 0.27, "device": "cpu", "imgsz": 640},
        content=_jpeg_bytes(frame),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json()["plates"] == [{"plateText": "KA05MN4523"}]
    assert captured == {
        "shape": (120, 200, 3),
        "conf": 0.27,
        "device": "cpu",
        "imgsz": 640,
    }


def test_model_server_keeps_legacy_json_anpr(monkeypatch):
    captured = {}

    def fake_analyze(frame, *, conf, device, imgsz):
        captured.update(shape=frame.shape, conf=conf, device=device, imgsz=imgsz)
        return [{"plateText": "KA05MN4523"}]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(plate_analyzer, "analyze_frame", fake_analyze)
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    response = client.post(
        "/api/anpr",
        json={
            "frame_jpeg_b64": base64.b64encode(_jpeg_bytes(frame)).decode("ascii"),
            "conf": 0.27,
            "device": "cpu",
            "imgsz": 640,
        },
    )

    assert response.status_code == 200
    assert response.json()["plates"] == [{"plateText": "KA05MN4523"}]
    assert captured == {
        "shape": (120, 200, 3),
        "conf": 0.27,
        "device": "cpu",
        "imgsz": 640,
    }
