import threading
import time

import cv2
import numpy as np
import pytest

from alert_pipeline import AlertPipeline
import video_processing


def _alert_from_payload(**payload):
    return {"id": str(payload["sequence"]), "snapshotUrl": None, **payload}


def test_persistence_and_broadcast_remain_in_submission_order():
    persisted = []
    broadcast = []

    def persist(**payload):
        persisted.append(payload["sequence"])
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: None,
        on_persisted=lambda alert: broadcast.append(alert["sequence"]),
        retry_delay=0,
    )

    futures = [pipeline.submit({"sequence": sequence}) for sequence in range(8)]

    assert [future.result(timeout=1)["sequence"] for future in futures] == list(range(8))
    assert pipeline.drain(timeout=1)
    assert persisted == list(range(8))
    assert broadcast == list(range(8))
    assert pipeline.shutdown(timeout=1)


def test_submit_does_not_wait_for_slow_persistence_or_delivery():
    release = threading.Event()
    persistence_started = threading.Event()

    def persist(**payload):
        persistence_started.set()
        assert release.wait(timeout=2)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: time.sleep(0.2),
    )

    started = time.perf_counter()
    future = pipeline.submit({"sequence": 1})
    submit_elapsed = time.perf_counter() - started

    assert submit_elapsed < 0.05
    assert persistence_started.wait(timeout=1)
    assert not future.done()
    release.set()
    assert future.result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=2)
    assert pipeline.shutdown(timeout=1)


def test_delivery_workers_run_concurrently():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def deliver(_alert, _outputs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with lock:
            active -= 1

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_workers=4,
    )
    futures = [pipeline.submit({"sequence": sequence}) for sequence in range(8)]

    for future in futures:
        future.result(timeout=1)
    assert pipeline.drain(timeout=2)
    assert max_active >= 2
    assert pipeline.stats()["delivered"] == 8
    assert pipeline.shutdown(timeout=1)


def test_transient_persistence_failures_are_retried():
    attempts = 0

    def persist(**payload):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("database temporarily unavailable")
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: None,
        persistence_attempts=3,
        retry_delay=0,
    )

    assert pipeline.submit({"sequence": 1}).result(timeout=1)["id"] == "1"
    assert attempts == 3
    assert pipeline.stats()["persistence_failures"] == 0
    assert pipeline.shutdown(timeout=1)


def test_permanent_persistence_failure_is_visible_on_future_and_stats():
    def persist(**_payload):
        raise RuntimeError("database unavailable")

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: None,
        persistence_attempts=2,
        retry_delay=0,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.stats()["persistence_failures"] == 1
    assert pipeline.shutdown(timeout=1)


def test_full_queue_applies_backpressure_without_dropping_alerts():
    first_started = threading.Event()
    release_first = threading.Event()
    persisted = []

    def persist(**payload):
        if payload["sequence"] == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        persisted.append(payload["sequence"])
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: None,
        persist_queue_size=1,
        submit_timeout=0.01,
    )
    first = pipeline.submit({"sequence": 1})
    assert first_started.wait(timeout=1)
    second = pipeline.submit({"sequence": 2})

    submitted_third = threading.Event()
    third_future = []

    def submit_third():
        third_future.append(pipeline.submit({"sequence": 3}))
        submitted_third.set()

    producer = threading.Thread(target=submit_third)
    producer.start()
    time.sleep(0.05)
    assert not submitted_third.is_set()
    assert pipeline.stats()["backpressure_events"] == 1

    release_first.set()
    producer.join(timeout=1)
    assert submitted_third.is_set()
    for future in (first, second, third_future[0]):
        future.result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert persisted == [1, 2, 3]
    assert pipeline.stats()["submitted"] == 3
    assert pipeline.shutdown(timeout=1)


def test_submit_copies_mutable_payload_before_worker_uses_it():
    release = threading.Event()
    received = []

    def persist(**payload):
        assert release.wait(timeout=1)
        received.append(payload)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(persist_alert=persist, deliver_alert=lambda _alert, _outputs: None)
    payload = {"sequence": 1, "bboxes": [{"x": 10}]}
    future = pipeline.submit(payload)
    payload["bboxes"][0]["x"] = 99
    release.set()

    future.result(timeout=1)
    assert received[0]["bboxes"] == [{"x": 10}]
    assert pipeline.shutdown(timeout=1)


def test_snapshot_pair_uses_same_dimensions_and_distinct_source_frames():
    annotated = np.zeros((100, 1000, 3), dtype=np.uint8)
    clean = np.zeros((100, 1000, 3), dtype=np.uint8)
    annotated[:, :] = (0, 0, 255)
    clean[:, :] = (0, 255, 0)

    annotated_jpeg, clean_jpeg = video_processing._encode_alert_snapshot_pair(
        annotated,
        clean,
        90,
    )
    decoded_annotated = cv2.imdecode(np.frombuffer(annotated_jpeg, np.uint8), cv2.IMREAD_COLOR)
    decoded_clean = cv2.imdecode(np.frombuffer(clean_jpeg, np.uint8), cv2.IMREAD_COLOR)

    assert decoded_annotated.shape == decoded_clean.shape == (85, 854, 3)
    assert decoded_annotated[:, :, 2].mean() > 240
    assert decoded_clean[:, :, 1].mean() > 240


def test_create_alert_prefers_explicit_inference_frame_snapshots(monkeypatch):
    captured = []

    class FakePipeline:
        def submit(self, payload, **_kwargs):
            captured.append(payload)
            return "queued"

    monkeypatch.setattr(video_processing, "_get_alert_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {"cameras": {"cam1": {"name": "Camera 1", "zone": "Loading"}}},
    )
    video_processing.state.camera_frames["cam1"] = b"newer-shared-frame"
    video_processing.state.camera_clean_frames["cam1"] = b"newer-shared-clean-frame"

    result = video_processing.create_alert(
        "cam1",
        "No helmet",
        "P2",
        0.9,
        snapshot_jpeg=b"inference-frame",
        clean_snapshot_jpeg=b"inference-clean-frame",
    )

    assert result == "queued"
    assert captured[0]["snapshot_jpeg"] == b"inference-frame"
    assert captured[0]["clean_snapshot_jpeg"] == b"inference-clean-frame"


def test_persisted_alert_broadcast_is_scheduled_on_server_event_loop(monkeypatch):
    scheduled = []

    class FakeLoop:
        @staticmethod
        def is_running():
            return True

    class FakeResult:
        @staticmethod
        def result(timeout):
            assert timeout == 5.0

    def schedule(coroutine, loop):
        scheduled.append(loop)
        coroutine.close()
        return FakeResult()

    loop = FakeLoop()
    monkeypatch.setattr(video_processing, "_alert_event_loop", loop)
    monkeypatch.setattr(video_processing.asyncio, "run_coroutine_threadsafe", schedule)

    video_processing._broadcast_persisted_alert({"id": "alert-1"})

    assert scheduled == [loop]
