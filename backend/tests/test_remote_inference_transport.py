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


def test_remote_inference_session_prewarm_uses_thread_local_health_get(monkeypatch):
    captured = {}
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager,
        "_remote_get",
        lambda path, *, timeout_seconds: captured.update(
            path=path,
            timeout_seconds=timeout_seconds,
        ) or {"status": "ok"},
    )

    assert model_manager.warm_remote_inference_session() is True
    assert captured == {"path": "/api/health", "timeout_seconds": 2.0}


def test_remote_inference_session_prewarm_is_best_effort(monkeypatch):
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager,
        "_remote_get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    assert model_manager.warm_remote_inference_session() is False


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


def test_raw_batch_posts_a_full_byte_view_without_frame_copy(monkeypatch):
    source = np.arange(40 * 60 * 3, dtype=np.uint8).reshape((40, 60, 3))
    frame = source[:, ::2]
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"results": {"primary": []}}

    class Session:
        @staticmethod
        def post(url, *, data, headers, timeout):
            captured.update(
                url=url,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            return Response()

    monkeypatch.setattr(
        model_manager,
        "_remote_settings",
        lambda: {"url": "http://model", "token": "", "timeout_seconds": 2.0},
    )
    monkeypatch.setattr(model_manager, "_remote_session", Session)
    monkeypatch.setattr(
        model_manager,
        "_REMOTE_RAW_BATCH_SUPPORT",
        {"url": None, "supported": None},
    )

    result = model_manager._remote_post_raw_batch(
        "/api/infer/raw/batch",
        frame,
        batch=[{"request_id": "primary", "model_key": "coco_primary"}],
    )

    assert result == {"results": {"primary": []}}
    assert isinstance(captured["data"], memoryview)
    assert captured["data"].nbytes == frame.size
    assert bytes(captured["data"]) == np.ascontiguousarray(frame).tobytes()
    assert captured["headers"]["X-Rakshak-Frame-Width"] == "30"
    assert captured["headers"]["X-Rakshak-Frame-Height"] == "40"


def test_primary_frame_batch_uses_unique_transport_ids(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"results": {"frame-0": [], "frame-1": []}}

    class Session:
        @staticmethod
        def post(url, *, data, headers, timeout):
            captured.update(
                url=url,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            return Response()

    monkeypatch.setattr(
        model_manager,
        "_remote_settings",
        lambda: {"url": "http://model", "token": "", "timeout_seconds": 2.0},
    )
    monkeypatch.setattr(model_manager, "_remote_session", Session)
    monkeypatch.setattr(
        model_manager,
        "_REMOTE_PRIMARY_BATCH2_SUPPORT",
        {"url": "http://model", "supported": None},
    )
    frames = [
        np.full((4, 5, 3), value, dtype=np.uint8) for value in (10, 20)
    ]
    items = [
        {
            "frame": frame,
            "request": {
                "request_id": "coco_primary",
                "conf": 0.3,
                "device": "cuda",
                "imgsz": 640,
            },
        }
        for frame in frames
    ]

    result = model_manager._remote_post_raw_primary_batch2(items)

    metadata = json.loads(
        captured["headers"]["X-Rakshak-Primary-Frame-Batch"]
    )
    assert result == {"results": {"frame-0": [], "frame-1": []}}
    assert [item["request_id"] for item in metadata] == ["frame-0", "frame-1"]
    assert bytes(captured["data"]) == b"".join(
        frame.tobytes() for frame in frames
    )


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


def test_edge_pairs_two_primary_frames_before_remote_admission(monkeypatch):
    class CountingAdmission:
        def __init__(self):
            self.acquires = 0
            self.releases = 0

        def acquire(self, *, timeout):
            assert timeout == model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            self.acquires += 1
            return True

        def release(self):
            self.releases += 1

    admission = CountingAdmission()
    batcher = model_manager._RemotePrimaryFrameBatcher(0.1)
    captured = {}

    def fake_primary_batch(items):
        captured["shapes"] = [item["frame"].shape for item in items]
        captured["request_ids"] = [
            item["request"]["request_id"] for item in items
        ]
        return {
            "results": {
                "frame-0": [{"bbox": [10, 20, 100, 200]}],
                "frame-1": [{"bbox": [10, 20, 100, 200]}],
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager, "_remote_primary_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", admission)
    monkeypatch.setattr(model_manager, "_REMOTE_PRIMARY_FRAME_BATCHER", batcher)
    monkeypatch.setattr(
        model_manager, "_remote_post_raw_primary_batch2", fake_primary_batch
    )
    frames = [
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        np.zeros((720, 1280, 3), dtype=np.uint8),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                model_manager.predict_record_batches,
                frame,
                [{"model_key": "coco_primary", "imgsz": 640}],
            )
            for frame in frames
        ]
        results = [future.result() for future in futures]

    assert admission.acquires == 1
    assert admission.releases == 1
    assert captured["shapes"] == [(360, 640, 3), (360, 640, 3)]
    # The logical IDs intentionally collide, as they do across camera calls.
    assert captured["request_ids"] == ["coco_primary:0", "coco_primary:0"]
    assert results[0]["coco_primary:0"][0]["bbox"] == [30, 60, 300, 600]
    assert results[1]["coco_primary:0"][0]["bbox"] == [20, 40, 200, 400]
    assert batcher.stats()["paired_requests"] == 2
    assert batcher.stats()["pairs_executed"] == 1


def test_edge_unmatched_primary_frame_times_out_to_existing_transport(monkeypatch):
    batcher = model_manager._RemotePrimaryFrameBatcher(0.001)
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager, "_remote_primary_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_PRIMARY_FRAME_BATCHER", batcher)
    monkeypatch.setenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", "true")
    monkeypatch.setattr(
        model_manager,
        "_remote_post_raw_batch",
        lambda _path, _frame, *, batch: {
            "results": {batch[0]["request_id"]: [{"class_id": 0}]}
        },
    )

    result = model_manager.predict_record_batches(
        np.zeros((120, 200, 3), dtype=np.uint8),
        [{"request_id": "primary", "model_key": "coco_primary", "imgsz": 640}],
    )

    assert result == {"primary": [{"class_id": 0}]}
    assert batcher.stats()["timeout_fallbacks"] == 1
    assert batcher.stats()["pending"] == 0


def test_edge_primary_batch_requires_compatible_settings(monkeypatch):
    batcher = model_manager._RemotePrimaryFrameBatcher(0.005)
    monkeypatch.setattr(
        model_manager, "_remote_primary_batch2_route_may_run", lambda: True
    )
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    requests = [
        {
            "request_id": f"frame-{index}",
            "model_key": "coco_primary",
            "conf": conf,
            "device": "cuda",
            "imgsz": 640,
            "classes": [],
        }
        for index, conf in enumerate((0.3, 0.4))
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(batcher.submit, frames, requests))

    assert outcomes == [(False, []), (False, [])]
    assert batcher.stats()["pairs_executed"] == 0
    assert batcher.stats()["timeout_fallbacks"] == 2


def test_edge_primary_batch_route_fallback_reuses_single_frame_path(monkeypatch):
    batcher = model_manager._RemotePrimaryFrameBatcher(0.1)
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager, "_remote_primary_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_PRIMARY_FRAME_BATCHER", batcher)
    monkeypatch.setattr(
        model_manager, "_remote_post_raw_primary_batch2", lambda _items: None
    )
    monkeypatch.setenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", "true")
    monkeypatch.setattr(
        model_manager,
        "_remote_post_raw_batch",
        lambda _path, _frame, *, batch: {
            "results": {batch[0]["request_id"]: []}
        },
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                model_manager.predict_record_batches,
                frame,
                [{"request_id": f"camera-{index}", "model_key": "coco_primary", "imgsz": 640}],
            )
            for index in range(2)
        ]
        results = [future.result() for future in futures]

    assert results == [{"camera-0": []}, {"camera-1": []}]
    assert batcher.stats()["route_fallbacks"] == 2


def test_edge_primary_batch_propagates_admission_overload_to_both_callers(
    monkeypatch,
):
    class RejectAdmission:
        @staticmethod
        def acquire(*, timeout):
            assert timeout == model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            return False

        @staticmethod
        def release():
            pytest.fail("Rejected admission must not release an unclaimed slot")

    batcher = model_manager._RemotePrimaryFrameBatcher(0.1)
    monkeypatch.setattr(
        model_manager, "_remote_primary_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", RejectAdmission())
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    requests = [
        {
            "request_id": f"frame-{index}",
            "model_key": "coco_primary",
            "conf": 0.35,
            "device": "cuda",
            "imgsz": 640,
            "classes": [],
        }
        for index in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(batcher.submit, frame, request) for request in requests
        ]
        for future in futures:
            with pytest.raises(model_manager.RemoteInferenceOverloadedError):
                future.result()

    assert batcher.stats()["admission_overloads"] == 2
    assert batcher.stats()["pairs_executed"] == 0


def test_edge_pairs_matching_primary_ppe_frames_before_admission(monkeypatch):
    class CountingAdmission:
        def __init__(self):
            self.acquires = 0
            self.releases = 0

        def acquire(self, *, timeout):
            assert timeout == model_manager._REMOTE_JOB_ADMISSION_WAIT_SECONDS
            self.acquires += 1
            return True

        def release(self):
            self.releases += 1

    admission = CountingAdmission()
    batcher = model_manager._RemoteSpecialistFrameBatcher(0.1)
    captured = {}

    def fake_specialist_batch(items):
        captured["shapes"] = [item["frame"].shape for item in items]
        return {
            "results": {
                f"frame-{index}": {
                    "coco_primary": [{"bbox": [10, 20, 100, 200]}],
                    "ppe_specialist": [{"bbox": [20, 10, 80, 100]}],
                }
                for index in range(2)
            }
        }

    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager, "_remote_specialist_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_JOB_ADMISSION", admission)
    monkeypatch.setattr(model_manager, "_REMOTE_SPECIALIST_FRAME_BATCHER", batcher)
    monkeypatch.setattr(
        model_manager,
        "_remote_post_raw_specialist_batch2",
        fake_specialist_batch,
    )
    frames = [
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        np.zeros((720, 1280, 3), dtype=np.uint8),
    ]

    def requests():
        return [
            {
                "request_id": "coco",
                "model_key": "coco_primary",
                "conf": 0.15,
                "imgsz": 640,
            },
            {
                "request_id": "ppe",
                "model_key": "ppe_specialist",
                "conf": 0.2,
                "imgsz": 640,
                "classes": ["motorcycle helmet", "rider helmet", "helmet"],
            },
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(model_manager.predict_record_batches, frame, requests())
            for frame in frames
        ]
        results = [future.result() for future in futures]

    assert admission.acquires == 1
    assert admission.releases == 1
    assert captured["shapes"] == [(360, 640, 3), (360, 640, 3)]
    assert results[0]["coco"][0]["bbox"] == [30, 60, 300, 600]
    assert results[0]["ppe"][0]["bbox"] == [60, 30, 240, 300]
    assert results[1]["coco"][0]["bbox"] == [20, 40, 200, 400]
    assert results[1]["ppe"][0]["bbox"] == [40, 20, 160, 200]
    assert batcher.stats()["paired_requests"] == 2
    assert batcher.stats()["pairs_executed"] == 1


def test_edge_specialist_batch_requires_matching_prompt_sets(monkeypatch):
    batcher = model_manager._RemoteSpecialistFrameBatcher(0.001)
    monkeypatch.setattr(
        model_manager, "_remote_specialist_batch2_route_may_run", lambda: True
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    def requests(classes):
        return [
            {
                "request_id": "coco",
                "model_key": "coco_primary",
                "conf": 0.15,
                "device": "cuda",
                "imgsz": 640,
                "classes": [],
            },
            {
                "request_id": "ppe",
                "model_key": "ppe_specialist",
                "conf": 0.2,
                "device": "cuda",
                "imgsz": 640,
                "classes": classes,
            },
        ]

    request_sets = [requests(["helmet"]), requests(["hard hat"])]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(batcher.submit, [frame, frame], request_sets))

    assert outcomes == [(False, {}), (False, {})]
    assert batcher.stats()["pairs_executed"] == 0
    assert batcher.stats()["timeout_fallbacks"] == 2


def test_edge_specialist_batch_route_fallback_reuses_grouped_path(monkeypatch):
    batcher = model_manager._RemoteSpecialistFrameBatcher(0.1)
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    monkeypatch.setattr(
        model_manager, "_remote_specialist_batch2_route_may_run", lambda: True
    )
    monkeypatch.setattr(model_manager, "_REMOTE_SPECIALIST_FRAME_BATCHER", batcher)
    monkeypatch.setattr(
        model_manager, "_remote_post_raw_specialist_batch2", lambda _items: None
    )
    monkeypatch.setenv("SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT", "true")
    monkeypatch.setattr(
        model_manager,
        "_remote_post_raw_batch",
        lambda _path, _frame, *, batch: {
            "results": {item["request_id"]: [] for item in batch}
        },
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    def requests(index):
        return [
            {
                "request_id": f"coco-{index}",
                "model_key": "coco_primary",
                "conf": 0.15,
                "imgsz": 640,
            },
            {
                "request_id": f"ppe-{index}",
                "model_key": "ppe_specialist",
                "conf": 0.2,
                "imgsz": 640,
                "classes": ["motorcycle helmet", "rider helmet", "helmet"],
            },
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                model_manager.predict_record_batches,
                frame,
                requests(index),
            )
            for index in range(2)
        ]
        results = [future.result() for future in futures]

    assert results == [
        {"coco-0": [], "ppe-0": []},
        {"coco-1": [], "ppe-1": []},
    ]
    assert batcher.stats()["route_fallbacks"] == 2


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


def test_model_server_accepts_two_primary_raw_frames(monkeypatch):
    captured = {}

    def fake_batch(frames, *, conf, device, imgsz):
        captured.update(
            values=[int(frame[0, 0, 0]) for frame in frames],
            shapes=[frame.shape for frame in frames],
            conf=conf,
            device=device,
            imgsz=imgsz,
        )
        return [[{"side": "left"}], [{"side": "right"}]]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(
        model_server.model_manager, "predict_coco_record_batch", fake_batch
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    left = np.full((12, 20, 3), 10, dtype=np.uint8)
    right = np.full((10, 16, 3), 20, dtype=np.uint8)
    metadata = [
        {
            "request_id": "left",
            "conf": 0.3,
            "device": "cuda",
            "imgsz": 640,
            "frame_width": 20,
            "frame_height": 12,
            "frame_channels": 3,
            "byte_length": left.nbytes,
        },
        {
            "request_id": "right",
            "conf": 0.3,
            "device": "cuda",
            "imgsz": 640,
            "frame_width": 16,
            "frame_height": 10,
            "frame_channels": 3,
            "byte_length": right.nbytes,
        },
    ]

    response = client.post(
        "/api/infer/raw/primary-batch2",
        content=left.tobytes() + right.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Primary-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": {
            "left": [{"side": "left"}],
            "right": [{"side": "right"}],
        }
    }
    assert captured == {
        "values": [10, 20],
        "shapes": [(12, 20, 3), (10, 16, 3)],
        "conf": 0.3,
        "device": "cuda",
        "imgsz": 640,
    }


def test_model_server_accepts_two_primary_ppe_raw_frames(monkeypatch):
    captured = {}
    classes = ["motorcycle helmet", "rider helmet", "helmet"]

    def fake_primary(frames, *, conf, device, imgsz):
        captured["primary"] = (len(frames), conf, device, imgsz)
        return [[{"primary": "left"}], [{"primary": "right"}]]

    def fake_ppe(frames, *, conf, device, imgsz, classes):
        captured["ppe"] = (len(frames), conf, device, imgsz, classes)
        return [[{"ppe": "left"}], [{"ppe": "right"}]]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server, "_SPECIALIST_BATCH_CONCURRENT", False)
    monkeypatch.setattr(
        model_server.model_manager, "predict_coco_record_batch", fake_primary
    )
    monkeypatch.setattr(
        model_server.model_manager, "predict_ppe_record_batch", fake_ppe
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frames = [np.full((4, 5, 3), value, dtype=np.uint8) for value in (10, 20)]
    metadata = [
        {
            "request_id": f"frame-{index}",
            "primary_conf": 0.15,
            "primary_device": "cuda",
            "primary_imgsz": 640,
            "ppe_conf": 0.2,
            "ppe_device": "cuda",
            "ppe_imgsz": 640,
            "ppe_classes": classes,
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index, frame in enumerate(frames)
    ]

    response = client.post(
        "/api/infer/raw/specialist-batch2",
        content=b"".join(frame.tobytes() for frame in frames),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Specialist-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == {
        "frame-0": {
            "coco_primary": [{"primary": "left"}],
            "ppe_specialist": [{"ppe": "left"}],
        },
        "frame-1": {
            "coco_primary": [{"primary": "right"}],
            "ppe_specialist": [{"ppe": "right"}],
        },
    }
    assert captured == {
        "primary": (2, 0.15, "cuda", 640),
        "ppe": (2, 0.2, "cuda", 640, classes),
    }


def test_model_server_can_overlap_primary_and_ppe_batch_engines(monkeypatch):
    classes = ["motorcycle helmet", "rider helmet", "helmet"]
    started = threading.Barrier(2, timeout=2)

    def fake_primary(frames, *, conf, device, imgsz):
        started.wait()
        return [[{"primary": "left"}], [{"primary": "right"}]]

    def fake_ppe(frames, *, conf, device, imgsz, classes):
        started.wait()
        return [[{"ppe": "left"}], [{"ppe": "right"}]]

    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server, "_SPECIALIST_BATCH_CONCURRENT", True)
    monkeypatch.setattr(
        model_server.model_manager, "predict_coco_record_batch", fake_primary
    )
    monkeypatch.setattr(
        model_server.model_manager, "predict_ppe_record_batch", fake_ppe
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    metadata = [
        {
            "request_id": f"frame-{index}",
            "primary_conf": 0.15,
            "primary_device": "cuda",
            "primary_imgsz": 640,
            "ppe_conf": 0.2,
            "ppe_device": "cuda",
            "ppe_imgsz": 640,
            "ppe_classes": classes,
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index in range(2)
    ]

    response = client.post(
        "/api/infer/raw/specialist-batch2",
        content=frame.tobytes() * 2,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Specialist-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 200
    assert response.json()["results"]["frame-0"] == {
        "coco_primary": [{"primary": "left"}],
        "ppe_specialist": [{"ppe": "left"}],
    }


def test_model_server_specialist_batch_falls_back_when_runtime_unavailable(
    monkeypatch,
):
    classes = ["motorcycle helmet", "rider helmet", "helmet"]
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(model_server, "_SPECIALIST_BATCH_CONCURRENT", False)
    monkeypatch.setattr(
        model_server.model_manager,
        "predict_coco_record_batch",
        lambda *_args, **_kwargs: [[], []],
    )
    monkeypatch.setattr(
        model_server.model_manager,
        "predict_ppe_record_batch",
        lambda *_args, **_kwargs: None,
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    metadata = [
        {
            "request_id": f"frame-{index}",
            "primary_conf": 0.15,
            "primary_imgsz": 640,
            "ppe_conf": 0.2,
            "ppe_imgsz": 640,
            "ppe_classes": classes,
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index in range(2)
    ]

    response = client.post(
        "/api/infer/raw/specialist-batch2",
        content=frame.tobytes() * 2,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Specialist-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Batch-2 specialist runtimes are unavailable"


def test_model_server_primary_frame_batch_requires_auth(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "expected-token")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    metadata = [
        {
            "request_id": f"frame-{index}",
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index in range(2)
    ]

    response = client.post(
        "/api/infer/raw/primary-batch2",
        content=frame.tobytes() * 2,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Primary-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 401


def test_model_server_primary_batch_falls_back_when_engine_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    monkeypatch.setattr(
        model_server.model_manager,
        "predict_coco_record_batch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_server,
        "_run_inference_frame",
        lambda **kwargs: {
            "detections": [{"value": int(kwargs["frame"][0, 0, 0])}]
        },
    )
    client = TestClient(model_server.app, raise_server_exceptions=False)
    frames = [np.full((4, 5, 3), value, dtype=np.uint8) for value in (30, 40)]
    metadata = [
        {
            "request_id": f"frame-{index}",
            "conf": 0.3,
            "device": "cuda",
            "imgsz": 640,
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": frame.nbytes,
        }
        for index, frame in enumerate(frames)
    ]

    response = client.post(
        "/api/infer/raw/primary-batch2",
        content=b"".join(frame.tobytes() for frame in frames),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Primary-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == {
        "frame-0": [{"value": 30}],
        "frame-1": [{"value": 40}],
    }


def test_model_server_rejects_incompatible_primary_frame_batch(monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_SERVER_TOKEN", "")
    client = TestClient(model_server.app, raise_server_exceptions=False)
    metadata = [
        {
            "request_id": f"frame-{index}",
            "conf": conf,
            "imgsz": 640,
            "frame_width": 5,
            "frame_height": 4,
            "frame_channels": 3,
            "byte_length": 60,
        }
        for index, conf in enumerate((0.3, 0.4))
    ]

    response = client.post(
        "/api/infer/raw/primary-batch2",
        content=bytes(120),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Rakshak-Primary-Frame-Batch": json.dumps(metadata),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Primary frame batch settings must match"


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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 2),
        ("invalid", 2),
        ("0", 1),
        ("3", 3),
        ("99", 8),
    ],
)
def test_remote_inference_concurrency_is_safely_bounded(
    monkeypatch,
    configured,
    expected,
):
    name = "TEST_REMOTE_INFERENCE_MAX_INFLIGHT"
    if configured is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, configured)

    assert model_manager._bounded_env_int(
        name,
        2,
        minimum=1,
        maximum=8,
    ) == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 0.075),
        ("invalid", 0.075),
        ("nan", 0.075),
        ("-1", 0.0),
        ("0.08", 0.08),
        ("1", 0.2),
    ],
)
def test_remote_inference_admission_wait_is_safely_bounded(
    monkeypatch,
    configured,
    expected,
):
    name = "TEST_REMOTE_INFERENCE_ADMISSION_WAIT_SECONDS"
    if configured is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, configured)

    assert model_manager._bounded_env_float(
        name,
        0.075,
        minimum=0.0,
        maximum=0.2,
    ) == expected


def test_model_server_runs_singleton_batch_without_executor_hop(monkeypatch):
    detections = [{"class_id": 0, "confidence": 0.9, "bbox": [1, 2, 3, 4]}]
    captured = {}

    def fake_inference(**kwargs):
        captured.update(kwargs)
        return {"detections": detections}

    monkeypatch.setattr(model_server, "_run_inference_frame", fake_inference)
    monkeypatch.setattr(
        model_server._BATCH_INFERENCE_EXECUTOR,
        "submit",
        lambda *_args, **_kwargs: pytest.fail(
            "singleton batch reached the nested executor"
        ),
    )
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    item = model_server.InferenceBatchItem(
        request_id="primary",
        model_key="coco_primary",
        conf=0.3,
        device="cuda",
        imgsz=640,
    )

    result = model_server._run_inference_batch(frame, [item])

    assert result == {"results": {"primary": detections}}
    assert captured.pop("frame") is frame
    assert captured == {
        "model_key": "coco_primary",
        "conf": 0.3,
        "device": "cuda",
        "imgsz": 640,
        "classes": [],
    }
