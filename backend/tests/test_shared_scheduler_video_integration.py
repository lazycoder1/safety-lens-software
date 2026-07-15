import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import state
import video_processing
from pipeline_telemetry import telemetry as pipeline_telemetry


def _scheduler_config(*camera_ids: str, mode: str = "active") -> dict:
    return {
        "global": {
            "shared_inference_scheduler_mode": mode,
            "shared_inference_max_batch_size": 4,
            "shared_inference_max_workers": 4,
            "coco_inference_width": 640,
            "ppe_inference_width": 640,
            "mobile_phone_inference_width": 640,
        },
        "cameras": {
            camera_id: {"enabled": True, "safety_rule_ids": []}
            for camera_id in camera_ids
        },
        "safety_rules": [],
    }


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    pytest.fail("condition was not met before timeout")


def _take_result(scheduler, camera_id: str, owner: str):
    result = None

    def result_available() -> bool:
        nonlocal result
        result = scheduler.take_result(camera_id, owner)
        return result is not None

    _wait_for(result_available)
    return result


@pytest.fixture(autouse=True)
def _reset_process_pipeline_runtime():
    assert video_processing.shutdown_pipeline_runtime(timeout=2.0)
    yield
    assert video_processing.shutdown_pipeline_runtime(timeout=2.0)


def test_mode_off_keeps_legacy_remote_batch_hint_semantics(monkeypatch):
    cfg = _scheduler_config("cam-1", "cam-2", mode="off")
    monkeypatch.delenv("SAFETYLENS_SHARED_INFERENCE_SCHEDULER_MODE", raising=False)
    monkeypatch.setattr(
        state,
        "camera_frames",
        {"cam-1": b"frame-1", "cam-2": b"frame-2"},
    )
    monkeypatch.setattr(
        state,
        "camera_frame_updated_at",
        {"cam-1": 100.0, "cam-2": 100.0},
    )

    assert video_processing._shared_scheduler_enabled(cfg) is False
    assert (
        video_processing._remote_frame_batch_size_hint(
            "cam-1",
            cfg,
            now_wall=101.0,
        )
        is None
    )

    state.camera_frames["cam-2"] = None
    assert (
        video_processing._remote_frame_batch_size_hint(
            "cam-1",
            cfg,
            now_wall=101.0,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("config_key", "environment_key"),
    [
        ("adaptive_inference_mode", "SAFETYLENS_ADAPTIVE_INFERENCE_MODE"),
        ("keyframe_tracking_mode", "SAFETYLENS_KEYFRAME_TRACKING_MODE"),
    ],
)
def test_runtime_mode_environment_override_wins_without_mutating_config(
    monkeypatch,
    config_key,
    environment_key,
):
    cfg = {"global": {config_key: "off"}}
    monkeypatch.setenv(environment_key, "active")

    assert (
        video_processing._configured_global_runtime_mode(
            cfg,
            config_key,
            environment_key,
            allowed={"off", "shadow", "active"},
        )
        == "active"
    )
    assert cfg["global"][config_key] == "off"


def test_invalid_runtime_mode_override_fails_closed(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_ADAPTIVE_INFERENCE_MODE", "surprise")

    assert (
        video_processing._configured_global_runtime_mode(
            {"global": {"adaptive_inference_mode": "shadow"}},
            "adaptive_inference_mode",
            "SAFETYLENS_ADAPTIVE_INFERENCE_MODE",
            allowed={"off", "shadow", "active"},
        )
        == "off"
    )


def test_adaptive_tracker_and_person_crop_runtime_counters_are_auditable():
    camera_id = "runtime-counter-test"
    pipeline_telemetry.remove_camera(camera_id)
    pipeline_telemetry.reset_camera(camera_id)
    try:
        decision = SimpleNamespace(
            mode="active",
            state="uncertain",
            target_fps=2.0,
            baseline_due=False,
            adaptive_due=True,
            urgent_reasons=("tracker_force_redetect",),
        )
        tracker_result = SimpleNamespace(
            projections=(object(), object()),
            aggregate_confidence=0.4,
            force_redetect=True,
            reasons=("low_tracker_confidence",),
        )

        video_processing._record_adaptive_runtime_state(
            camera_id,
            decision,
            tracker_result,
        )
        video_processing._record_adaptive_inference_admission(camera_id, decision)
        video_processing._record_adaptive_inference_completion(
            camera_id,
            decision.mode,
            decision.state,
        )
        video_processing._record_person_crop_runtime_counters(
            camera_id,
            {
                "ppe": {
                    "cropInferenceAttempts": 2,
                    "fallbackRequired": True,
                    "fullFrameInvocations": 1,
                },
                "phone": {
                    "cropInferenceAttempts": 3,
                    "fallbackRequired": False,
                    "fullFrameInvocations": 0,
                },
            },
        )

        counters = pipeline_telemetry.public_camera_snapshot(camera_id)["counters"]
        assert counters["adaptiveUncertainObservationCount"] == 1
        assert counters["adaptiveUncertainAdmissionCount"] == 1
        assert counters["adaptiveUncertainInferenceCount"] == 1
        assert counters["trackerProjectionFrameCount"] == 1
        assert counters["trackerProjectedPersonCount"] == 2
        assert counters["trackerForceRedetectSignalCount"] == 1
        assert counters["ppePersonCropAttemptCount"] == 2
        assert counters["phonePersonCropAttemptCount"] == 3
        assert counters["personCropFallbackCount"] == 1
        assert counters["personCropFullFrameInvocationCount"] == 1
    finally:
        pipeline_telemetry.remove_camera(camera_id)


def test_active_four_camera_cohort_reaches_model_boundary_with_exact_hint(
    monkeypatch,
):
    camera_ids = ("cam-1", "cam-2", "cam-3", "cam-4")
    cfg = _scheduler_config(*camera_ids)
    scheduler = video_processing.SharedInferenceScheduler(
        batch2_wait_seconds=0.2,
        singleton_wait_seconds=0.4,
    )
    scheduler.start()
    model_calls = []
    calls_lock = threading.Lock()

    def fake_predict_record_batches(
        _frame,
        requests,
        *,
        frame_batch_size_hint=None,
    ):
        with calls_lock:
            model_calls.append(
                {
                    "hint": frame_batch_size_hint,
                    "models": tuple(request["model_key"] for request in requests),
                }
            )
        return {request["request_id"]: [] for request in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "remote_frame_batch_route_may_run",
        lambda model_keys, batch_size: tuple(model_keys) == ("coco_primary",)
        and batch_size in {2, 4},
    )
    plan = {
        "run_coco_primary": True,
        "required_model_keys": ["coco_primary"],
        "capabilities": ["person_presence"],
    }
    profile = video_processing._inference_batch_profile(
        "cam-1",
        plan,
        cfg,
        yolo_conf=0.35,
        device="cuda",
        inference_width=640,
    )

    try:
        for camera_id in camera_ids:
            scheduler.register(camera_id, "owner")
            frame = np.zeros((24, 32, 3), dtype=np.uint8)

            def run(batch_size, *, selected_camera=camera_id, selected_frame=frame):
                return video_processing._run_detection_job(
                    selected_camera,
                    selected_frame,
                    plan,
                    {},
                    cfg["cameras"][selected_camera],
                    cfg,
                    yolo_conf=0.35,
                    device="cuda",
                    inference_width=640,
                    last_face_log_by_key={},
                    last_plate_log_by_key={},
                    plate_vote_window=[],
                    frame_batch_size_hint=batch_size,
                )

            now = time.monotonic()
            offer = scheduler.offer(
                camera_id,
                "owner",
                video_processing.InferenceWork(
                    sequence=1,
                    profile=profile,
                    run=run,
                    captured_at=now,
                    expires_at=now + 2.0,
                ),
            )
            assert offer.accepted

        outcomes = [
            _take_result(scheduler, camera_id, "owner") for camera_id in camera_ids
        ]

        assert all(outcome.succeeded for outcome in outcomes)
        assert {outcome.batch_size for outcome in outcomes} == {4}
        assert model_calls == [
            {"hint": 4, "models": ("coco_primary",)},
            {"hint": 4, "models": ("coco_primary",)},
            {"hint": 4, "models": ("coco_primary",)},
            {"hint": 4, "models": ("coco_primary",)},
        ]
        assert scheduler.stats()["counters"]["batches_4"] == 1
    finally:
        assert scheduler.stop(timeout=2.0)


def test_profile_stays_singleton_when_fixed_batch_route_is_unavailable(monkeypatch):
    cfg = _scheduler_config("cam-1", "cam-2")
    plan = {
        "run_coco_primary": True,
        "required_model_keys": ["coco_primary"],
        "capabilities": ["person_presence"],
    }
    monkeypatch.setattr(
        video_processing.model_manager,
        "remote_frame_batch_route_may_run",
        lambda _model_keys, _batch_size: False,
    )

    profile = video_processing._inference_batch_profile(
        "cam-1",
        plan,
        cfg,
        yolo_conf=0.35,
        device="cuda",
        inference_width=640,
    )

    assert profile.max_batch_size == 1


def test_tracker_keyframe_age_starts_at_decoded_ingress_not_completion():
    captured_at = video_processing._decoded_ingress_monotonic_seconds(
        1_000_000_000,
        completed_monotonic=3.0,
    )
    tracker = video_processing.PersonKLTTracker(max_frame_gap_seconds=0.5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    tracker.seed(
        frame,
        [{"class": "person", "bbox": [8, 8, 40, 44], "confidence": 0.9}],
        captured_at,
    )

    result = tracker.project(frame, 3.0)

    adaptive = video_processing.AdaptiveInferenceController(
        4,
        mode="active",
        max_keyframe_age_seconds=1.0,
    )
    adaptive.record_inference(
        3.0,
        keyframe=True,
        keyframe_at=captured_at,
    )
    adaptive_decision = adaptive.decide(3.0)

    primary_tracker = video_processing.PrimaryPersonTracker()
    primary_tracker.update(
        [
            {
                "class": "person",
                "model_family": "coco_primary",
                "bbox": [8, 8, 40, 44],
            }
        ],
        now=captured_at,
        ttl_seconds=1.0,
    )

    assert result.frame_gap is True
    assert result.force_redetect is True
    assert "max_keyframe_age" in adaptive_decision.urgent_reasons
    assert not primary_tracker.has_stable_person(
        now=3.0,
        min_hits=1,
        ttl_seconds=1.0,
    )


def test_latest_queued_work_replaces_older_frame_before_dispatch():
    scheduler = video_processing.SharedInferenceScheduler(
        batch2_wait_seconds=0.5,
        singleton_wait_seconds=1.0,
    )
    scheduler.start()
    calls = []
    profile = video_processing.BatchProfile("single", max_batch_size=1)

    try:
        scheduler.register("cam-1", "owner")
        now = time.monotonic()
        first = scheduler.offer(
            "cam-1",
            "owner",
            video_processing.InferenceWork(
                sequence=1,
                profile=profile,
                run=lambda batch_size: calls.append((1, batch_size)),
                captured_at=now,
                expires_at=now + 2.0,
            ),
        )
        replacement = scheduler.offer(
            "cam-1",
            "owner",
            video_processing.InferenceWork(
                sequence=2,
                profile=profile,
                run=lambda batch_size: calls.append((2, batch_size)),
                captured_at=now,
                expires_at=now + 2.0,
                urgent=True,
            ),
        )

        outcome = _take_result(scheduler, "cam-1", "owner")

        assert first.status is video_processing.OfferStatus.ACCEPTED
        assert replacement.status is video_processing.OfferStatus.REPLACED
        assert outcome.sequence == 2
        assert calls == [(2, 1)]
        assert scheduler.stats()["counters"]["replaced_queued"] == 1
    finally:
        assert scheduler.stop(timeout=2.0)


def test_unregister_generation_fences_late_camera_completion():
    scheduler = video_processing.SharedInferenceScheduler()
    scheduler.start()
    old_started = threading.Event()
    release_old = threading.Event()

    try:
        old_generation = scheduler.register("cam-1", "old-owner")

        def run_old(_batch_size):
            old_started.set()
            assert release_old.wait(timeout=2.0)
            return "old-result"

        now = time.monotonic()
        scheduler.offer(
            "cam-1",
            "old-owner",
            video_processing.InferenceWork(
                sequence=1,
                profile=video_processing.BatchProfile("detector"),
                run=run_old,
                captured_at=now,
                expires_at=now + 2.0,
                urgent=True,
            ),
        )
        assert old_started.wait(timeout=1.0)
        assert scheduler.unregister("cam-1", "old-owner")

        new_generation = scheduler.register("cam-1", "new-owner")
        scheduler.offer(
            "cam-1",
            "new-owner",
            video_processing.InferenceWork(
                sequence=1,
                profile=video_processing.BatchProfile("detector"),
                run=lambda _batch_size: "new-result",
                captured_at=now,
                expires_at=now + 2.0,
                urgent=True,
            ),
        )
        new_outcome = _take_result(scheduler, "cam-1", "new-owner")
        release_old.set()
        _wait_for(
            lambda: scheduler.stats()["counters"].get(
                "fenced_completions",
                0,
            )
            == 1
        )

        assert new_generation != old_generation
        assert new_outcome.value == "new-result"
        assert scheduler.take_result("cam-1", "new-owner") is None
    finally:
        release_old.set()
        assert scheduler.stop(timeout=2.0)


def test_pipeline_runtime_shutdown_resets_global_scheduler_for_clean_restart():
    cfg = _scheduler_config("cam-1")
    first = video_processing._get_shared_scheduler(cfg)

    assert video_processing.pipeline_runtime_stats()["sharedInferenceScheduler"][
        "running"
    ]
    assert video_processing.shutdown_pipeline_runtime(timeout=2.0)
    stopped_stats = video_processing.pipeline_runtime_stats()[
        "sharedInferenceScheduler"
    ]
    assert stopped_stats == {"running": False, "accepting": False}

    second = video_processing._get_shared_scheduler(cfg)

    assert second is not first
    assert second.stats()["running"] is True
