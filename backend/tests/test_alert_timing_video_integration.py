from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

import alert_timing
import alert_delivery_worker
import video_processing
from pipeline_telemetry import PipelineTelemetry


@pytest.fixture(autouse=True)
def _clear_alert_timing_registry():
    alert_timing.registry.clear()
    yield
    alert_timing.registry.clear()


def _candidate(threshold: int = 2) -> dict:
    return {
        "camera_id": "cam1",
        "rule": "Person Detected",
        "severity": "P2",
        "confidence": 0.9,
        "description": "Person detected",
        "source": "test",
        "threshold": threshold,
    }


def _decision():
    return SimpleNamespace(
        fallback=False,
        output_ids=None,
        rule_id="",
        rule_name="Test",
        severity="P2",
        priority=2,
        message=None,
        cooldown_seconds=30,
    )


def test_detection_confirmation_carries_first_positive_timing(monkeypatch):
    times = iter(
        (
            datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 15, 10, 0, 1, tzinfo=timezone.utc),
        )
    )

    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return next(times)

    monotonic_ns = iter((100, 250))
    captured = []
    anchors = {}
    violation_window = {}
    active_violations = set()
    monkeypatch.setattr(video_processing, "datetime", FakeDateTime)
    monkeypatch.setattr(
        video_processing.time, "monotonic_ns", lambda: next(monotonic_ns)
    )
    monkeypatch.setattr(
        video_processing, "check_violations", lambda *_args: [_candidate()]
    )
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_args: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(
        video_processing.policy_engine,
        "evaluate_candidate",
        lambda *_args, **_kwargs: [_decision()],
    )
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: captured.append(kwargs) or {"id": "alert-1"},
    )

    observation = dict(
        camera_id="cam1",
        frame=np.zeros((40, 60, 3), dtype=np.uint8),
        annotated_frame=None,
        detections=[{"class": "person", "confidence": 0.9, "bbox": [1, 1, 20, 35]}],
        fresh_pose_results=None,
        scheduled_plan={
            "capabilities": ["person_presence"],
            "run_pose_specialist": False,
            "run_ppe_specialist": False,
        },
        current_cam={},
        current_cfg={"global": {"jpeg_quality": 70}},
        last_alert_by_rule={},
        active_violations=active_violations,
        violation_window=violation_window,
        alert_cooldown=30,
        window_size=3,
        first_positive_by_rule=anchors,
    )

    video_processing._process_detection_observation(**observation)
    assert captured == []
    assert anchors["Person Detected"] == (
        "2026-07-15T10:00:00+00:00",
        100,
    )

    video_processing._process_detection_observation(**observation)

    assert captured[0]["first_positive_at"] == "2026-07-15T10:00:00+00:00"
    assert captured[0]["confirmed_at"] == "2026-07-15T10:00:01+00:00"
    assert captured[0]["_first_positive_monotonic_ns"] == 100
    assert captured[0]["_confirmed_monotonic_ns"] == 250
    assert anchors == {}
    assert active_violations == {"Person Detected"}


def test_rearmed_incident_anchors_first_post_alert_positive(monkeypatch):
    wall_times = iter(
        datetime(2026, 7, 15, 10, 0, second, tzinfo=timezone.utc)
        for second in range(4)
    )

    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return next(wall_times)

    monotonic_ns = iter((100, 200, 300, 400))
    captured = []
    anchors = {}
    violation_window = {}
    active_violations = set()
    monkeypatch.setattr(video_processing, "datetime", FakeDateTime)
    monkeypatch.setattr(
        video_processing.time, "monotonic_ns", lambda: next(monotonic_ns)
    )
    monkeypatch.setattr(
        video_processing, "check_violations", lambda *_args: [_candidate()]
    )
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_a: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(
        video_processing.policy_engine,
        "evaluate_candidate",
        lambda *_args, **_kwargs: [_decision()],
    )
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: captured.append(kwargs) or {"id": f"alert-{len(captured)}"},
    )
    observation = dict(
        camera_id="cam1",
        frame=np.zeros((40, 60, 3), dtype=np.uint8),
        annotated_frame=None,
        detections=[{"class": "person", "confidence": 0.9, "bbox": [1, 1, 20, 35]}],
        fresh_pose_results=None,
        scheduled_plan={"capabilities": ["person_presence"]},
        current_cam={},
        current_cfg={"global": {"jpeg_quality": 70}},
        last_alert_by_rule={},
        active_violations=active_violations,
        violation_window=violation_window,
        alert_cooldown=30,
        window_size=3,
        first_positive_by_rule=anchors,
    )

    for _ in range(4):
        video_processing._process_detection_observation(**observation)

    assert len(captured) == 2
    assert captured[1]["first_positive_at"] == "2026-07-15T10:00:02+00:00"
    assert captured[1]["_first_positive_monotonic_ns"] == 300
    assert captured[1]["confirmed_at"] == "2026-07-15T10:00:03+00:00"


def test_fresh_negative_clears_first_positive_anchor(monkeypatch):
    anchors = {"Person Detected": ("2026-07-15T10:00:00+00:00", 100)}

    video_processing._refresh_first_positive_anchors(
        anchors,
        {},
        set(),
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
        fresh_ppe_evaluated=False,
        fresh_detection_rule_keys=None,
    )

    assert anchors == {}


def test_partial_observation_preserves_unobserved_rule_anchor():
    anchors = {"Person Detected": ("2026-07-15T10:00:00+00:00", 100)}

    video_processing._refresh_first_positive_anchors(
        anchors,
        {},
        set(),
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
        fresh_ppe_evaluated=False,
        fresh_detection_rule_keys={"Mobile Phone Detected"},
    )

    assert anchors == {"Person Detected": ("2026-07-15T10:00:00+00:00", 100)}


def test_first_positive_anchor_includes_decoded_frame_age(monkeypatch):
    class FakeDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(2026, 7, 15, 10, 0, 2, tzinfo=timezone.utc)

    monkeypatch.setattr(video_processing, "datetime", FakeDateTime)
    monkeypatch.setattr(video_processing.time, "monotonic_ns", lambda: 3_000_000_000)
    anchors = {}

    video_processing._refresh_first_positive_anchors(
        anchors,
        {"Person Detected": _candidate()},
        set(),
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
        fresh_ppe_evaluated=False,
        fresh_detection_rule_keys=None,
        fresh_observation_monotonic_ns=1_000_000_000,
    )

    assert anchors["Person Detected"] == (
        "2026-07-15T10:00:00+00:00",
        1_000_000_000,
    )


def test_persistence_callback_records_all_monotonic_alert_latencies(monkeypatch):
    captured = []
    observations = []

    class FakeFuture:
        def __init__(self):
            self.callback = None

        def add_done_callback(self, callback):
            self.callback = callback

        @staticmethod
        def result():
            return {"id": "persisted"}

    future = FakeFuture()

    class FakePipeline:
        def submit(self, payload, **_kwargs):
            captured.append(payload)
            return future

    monkeypatch.setattr(video_processing, "_get_alert_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        video_processing,
        "get_config_snapshot",
        lambda: {"cameras": {"cam1": {"name": "Camera 1", "zone": "A"}}},
    )
    monkeypatch.setattr(
        video_processing.notification_dispatcher,
        "resolve_delivery_targets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        video_processing.pipeline_telemetry,
        "observe_alert_elapsed_ns",
        lambda name, started, completed: observations.append(
            (name, started, completed)
        ),
    )

    submission = video_processing.create_alert(
        "cam1",
        "Person Detected",
        "P2",
        0.9,
        snapshot_jpeg=b"snapshot",
        first_positive_at="2026-07-15T10:00:00+00:00",
        confirmed_at="2026-07-15T10:00:01+00:00",
        _first_positive_monotonic_ns=100,
        _confirmed_monotonic_ns=250,
    )
    alert_id = captured[0]["alert_id"]
    assert submission is future
    assert captured[0]["first_positive_at"] == "2026-07-15T10:00:00+00:00"
    assert captured[0]["confirmed_at"] == "2026-07-15T10:00:01+00:00"
    assert alert_timing.registry.stats()["pending"] == 1

    monkeypatch.setattr(video_processing.time, "monotonic_ns", lambda: 400)
    video_processing._broadcast_persisted_alert({"id": alert_id})

    assert observations == [
        ("firstPositiveToConfirmedMs", 100, 250),
        ("confirmedToPersistedMs", 250, 400),
        ("firstPositiveToPersistedMs", 100, 400),
    ]
    assert alert_timing.registry.stats()["pending"] == 0


def test_alert_submission_failure_discards_process_local_timing(monkeypatch):
    class FailingPipeline:
        @staticmethod
        def submit(*_args, **_kwargs):
            raise RuntimeError("queue closed")

    monkeypatch.setattr(
        video_processing, "_get_alert_pipeline", lambda: FailingPipeline()
    )
    monkeypatch.setattr(
        video_processing,
        "get_config_snapshot",
        lambda: {"cameras": {"cam1": {}}},
    )
    monkeypatch.setattr(
        video_processing.notification_dispatcher,
        "resolve_delivery_targets",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(RuntimeError, match="queue closed"):
        video_processing.create_alert(
            "cam1",
            "Person Detected",
            "P2",
            0.9,
            snapshot_jpeg=b"snapshot",
            first_positive_at="2026-07-15T10:00:00+00:00",
            confirmed_at="2026-07-15T10:00:01+00:00",
            _first_positive_monotonic_ns=100,
            _confirmed_monotonic_ns=250,
        )

    assert alert_timing.registry.stats()["pending"] == 0


def test_provider_outcome_before_persistence_callback_is_deferred_without_loss(
    monkeypatch,
):
    telemetry = PipelineTelemetry(
        telemetry_epoch="00000000-0000-4000-8000-000000000099"
    )
    monkeypatch.setattr(video_processing, "pipeline_telemetry", telemetry)
    monkeypatch.setattr(alert_delivery_worker, "pipeline_telemetry", telemetry)
    alert_timing.registry.remember(
        "alert-race",
        first_positive_ns=100_000_000,
        confirmed_ns=200_000_000,
        initial_target_keys=("initial:webhook:target",),
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1)

    # The alert/outbox DB commit is visible, but AlertPipeline._on_persisted is
    # deliberately blocked. A fast provider finishes in that gap.
    worker._record_initial_outcome(
        {
            "kind": "initial",
            "alert_id": "alert-race",
            "target_key": "initial:webhook:target",
        },
        "delivered",
        completed_ns=400_000_000,
    )

    before = telemetry.public_snapshot()["alerts"]["deliveryCoverage"]
    assert before["counters"]["eligibleCount"] == 0
    assert before["counters"]["outcomeCensoredCount"] == 0
    assert alert_timing.registry.stats()["awaitingPersistence"] == 1

    monkeypatch.setattr(video_processing.time, "monotonic_ns", lambda: 300_000_000)
    video_processing._observe_persisted_alert_timing({"id": "alert-race"})

    alerts = telemetry.public_snapshot()["alerts"]
    coverage = alerts["deliveryCoverage"]
    assert coverage["counters"]["eligibleCount"] == 1
    assert coverage["counters"]["deliveredCount"] == 1
    assert coverage["counters"]["outcomeCensoredCount"] == 0
    assert coverage["pending"] == 0
    assert alerts["latency"]["firstPositiveToProviderSuccessMs"]["count"] == 1
    assert alert_timing.registry.stats()["pending"] == 0
