import threading
import time

import pytest

from shared_inference_scheduler import (
    COUNTER_MAX,
    BatchProfile,
    CameraOwnershipError,
    InferenceWork,
    OfferStatus,
    SchedulerNotRunningError,
    SharedInferenceScheduler,
)


_OWNER = "owner-1"
_PROFILE = BatchProfile("detector:640", max_batch_size=4)


def _work(
    sequence,
    run,
    *,
    profile=_PROFILE,
    urgent=False,
    urgent_reasons=(),
    expires_in=1.0,
):
    now = time.monotonic()
    return InferenceWork(
        sequence=sequence,
        profile=profile,
        run=run,
        captured_at=now,
        expires_at=None if expires_in is None else now + expires_in,
        urgent=urgent,
        urgent_reasons=urgent_reasons,
    )


def _wait_for(predicate, *, timeout=2.0, message="condition was not met"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    pytest.fail(message)


def _take_result(scheduler, camera_id, owner_token=_OWNER, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = scheduler.take_result(camera_id, owner_token)
        if result is not None:
            return result
        time.sleep(0.001)
    pytest.fail(f"no result became available for {camera_id}")


@pytest.fixture
def scheduler():
    value = SharedInferenceScheduler()
    value.start()
    try:
        yield value
    finally:
        value.stop()


def test_profile_work_and_scheduler_lifecycle_validation():
    with pytest.raises(ValueError, match="1, 2, or 4"):
        BatchProfile("detector", max_batch_size=3)
    with pytest.raises(TypeError, match="hashable"):
        BatchProfile([])
    with pytest.raises(ValueError, match="1, 2, or 4"):
        BatchProfile("detector", max_batch_size=True)
    with pytest.raises(ValueError, match="non-negative"):
        _work(-1, lambda _batch_size: None)

    value = SharedInferenceScheduler()
    with pytest.raises(SchedulerNotRunningError):
        value.register("cam-1", _OWNER)

    assert value.start() is True
    assert value.start() is False
    assert value.stats()["batch2WaitMs"] == 6.0
    assert value.stats()["singletonWaitMs"] == 14.0
    generation = value.register("cam-1", _OWNER)
    assert value.register("cam-1", _OWNER) == generation
    with pytest.raises(CameraOwnershipError):
        value.register("cam-1", "somebody-else")
    with pytest.raises(CameraOwnershipError):
        value.unregister("cam-1", "somebody-else")
    assert value.unregister("cam-1", _OWNER) is True
    assert value.unregister("cam-1", _OWNER) is False
    assert value.stop() is True
    with pytest.raises(SchedulerNotRunningError):
        value.start()


def test_offer_replaces_only_queued_item_and_rejects_invalid_offers(scheduler):
    calls = []
    profile = BatchProfile("single", max_batch_size=1)
    scheduler.register("cam-1", _OWNER)

    first = scheduler.offer(
        "cam-1",
        _OWNER,
        _work(1, lambda size: calls.append((1, size)), profile=profile),
    )
    replacement = scheduler.offer(
        "cam-1",
        _OWNER,
        _work(2, lambda size: calls.append((2, size)), profile=profile),
    )

    assert first.status is OfferStatus.ACCEPTED
    assert replacement.status is OfferStatus.REPLACED
    assert replacement.accepted is True
    assert (
        scheduler.offer(
            "cam-1",
            _OWNER,
            _work(2, lambda _size: None, profile=profile),
        ).status
        is OfferStatus.OUT_OF_ORDER
    )
    assert (
        scheduler.offer(
            "missing",
            _OWNER,
            _work(1, lambda _size: None),
        ).status
        is OfferStatus.NOT_REGISTERED
    )
    assert (
        scheduler.offer(
            "cam-1",
            "wrong-owner",
            _work(3, lambda _size: None),
        ).status
        is OfferStatus.OWNER_MISMATCH
    )
    assert (
        scheduler.offer(
            "cam-1",
            _OWNER,
            _work(3, lambda _size: None, expires_in=-0.001),
        ).status
        is OfferStatus.STALE
    )

    result = _take_result(scheduler, "cam-1")
    assert result.sequence == 2
    assert calls == [(2, 1)]
    assert scheduler.take_result("cam-1", _OWNER) is None
    stats = scheduler.stats()
    assert stats["counters"]["replaced_queued"] == 1
    assert stats["counters"]["out_of_order_offers"] == 1
    assert stats["counters"]["stale_offer_drops"] == 1


def test_one_inflight_one_queued_and_one_unconsumed_result_per_camera(scheduler):
    first_started = threading.Event()
    release_first = threading.Event()
    later_started = threading.Event()
    calls = []
    profile = BatchProfile("single", max_batch_size=1)
    scheduler.register("cam-1", _OWNER)

    def run_first(batch_size):
        calls.append((1, batch_size))
        first_started.set()
        assert release_first.wait(timeout=2)
        return "first"

    def run_later(sequence):
        def run(batch_size):
            calls.append((sequence, batch_size))
            later_started.set()
            return sequence

        return run

    scheduler.offer(
        "cam-1",
        _OWNER,
        _work(1, run_first, profile=profile, urgent=True),
    )
    assert first_started.wait(timeout=1)
    assert (
        scheduler.offer(
            "cam-1",
            _OWNER,
            _work(2, run_later(2), profile=profile),
        ).status
        is OfferStatus.ACCEPTED
    )
    assert (
        scheduler.offer(
            "cam-1",
            _OWNER,
            _work(3, run_later(3), profile=profile),
        ).status
        is OfferStatus.REPLACED
    )

    release_first.set()
    _wait_for(
        lambda: scheduler.stats()["completedUnconsumed"] == 1,
        message="first result was not deposited",
    )
    assert scheduler.stats()["queued"] == 1
    assert not later_started.wait(timeout=0.04)

    first_result = scheduler.take_result("cam-1", _OWNER)
    assert first_result is not None and first_result.value == "first"
    later_result = _take_result(scheduler, "cam-1")
    assert later_result.sequence == 3
    assert later_result.value == 3
    assert calls == [(1, 1), (3, 1)]


def test_four_compatible_items_dispatch_immediately_with_exact_batch_size():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.2,
        singleton_wait_seconds=0.4,
    )
    value.start()
    try:
        calls = []
        for camera_id in ("cam-1", "cam-2", "cam-3", "cam-4"):
            value.register(camera_id, _OWNER)
            value.offer(
                camera_id,
                _OWNER,
                _work(
                    1,
                    lambda size, camera=camera_id: calls.append((camera, size)),
                ),
            )

        results = [_take_result(value, f"cam-{index}") for index in range(1, 5)]
        assert {result.batch_size for result in results} == {4}
        assert sorted(calls) == [
            ("cam-1", 4),
            ("cam-2", 4),
            ("cam-3", 4),
            ("cam-4", 4),
        ]
        assert value.stats()["counters"]["batches_4"] == 1
    finally:
        value.stop()


def test_two_compatible_items_wait_six_millisecond_class_then_use_batch_two():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.03,
        singleton_wait_seconds=0.2,
    )
    value.start()
    try:
        started = threading.Event()
        start_times = []
        batch_sizes = []
        for camera_id in ("cam-1", "cam-2"):
            value.register(camera_id, _OWNER)
        offered_at = time.monotonic()

        def run(batch_size):
            batch_sizes.append(batch_size)
            start_times.append(time.monotonic())
            started.set()

        for camera_id in ("cam-1", "cam-2"):
            value.offer(camera_id, _OWNER, _work(1, run))

        assert not started.wait(timeout=0.012)
        results = [_take_result(value, camera_id) for camera_id in ("cam-1", "cam-2")]
        assert {result.batch_size for result in results} == {2}
        assert batch_sizes == [2, 2]
        assert min(start_times) - offered_at >= 0.02
        assert value.stats()["counters"]["batches_2"] == 1
    finally:
        value.stop()


def test_singleton_waits_fourteen_millisecond_class_and_uses_batch_one():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.03,
        singleton_wait_seconds=0.05,
    )
    value.start()
    try:
        started = threading.Event()
        start_times = []
        value.register("cam-1", _OWNER)
        offered_at = time.monotonic()

        def run(batch_size):
            start_times.append((time.monotonic(), batch_size))
            started.set()

        value.offer("cam-1", _OWNER, _work(1, run))
        assert not started.wait(timeout=0.02)
        result = _take_result(value, "cam-1")
        assert result.batch_size == 1
        assert start_times[0][1] == 1
        assert start_times[0][0] - offered_at >= 0.04
        assert value.stats()["counters"]["batches_1"] == 1
    finally:
        value.stop()


def test_urgent_work_bypasses_all_batch_waits():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.5,
        singleton_wait_seconds=0.8,
    )
    value.start()
    try:
        started = threading.Event()
        observed = []
        value.register("cam-1", _OWNER)

        def run(batch_size):
            observed.append(batch_size)
            started.set()

        offered_at = time.monotonic()
        value.offer(
            "cam-1",
            _OWNER,
            _work(1, run, urgent_reasons=("possible-violation",)),
        )
        assert started.wait(timeout=0.2)
        result = _take_result(value, "cam-1")
        assert result.started_at - offered_at < 0.2
        assert observed == [1]
        assert value.stats()["counters"]["urgent_dispatched"] == 1
    finally:
        value.stop()


def test_incompatible_profiles_never_share_a_cohort():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.02,
        singleton_wait_seconds=0.2,
    )
    value.start()
    try:
        observed = []
        profiles = {
            "cam-a1": BatchProfile("model-a", max_batch_size=4),
            "cam-b1": BatchProfile("model-b", max_batch_size=4),
            "cam-a2": BatchProfile("model-a", max_batch_size=4),
            "cam-b2": BatchProfile("model-b", max_batch_size=4),
        }
        for camera_id, profile in profiles.items():
            value.register(camera_id, _OWNER)
            value.offer(
                camera_id,
                _OWNER,
                _work(
                    1,
                    lambda size, camera=camera_id: observed.append((camera, size)),
                    profile=profile,
                ),
            )

        results = {
            camera_id: _take_result(value, camera_id)
            for camera_id in profiles
        }
        assert {result.batch_size for result in results.values()} == {2}
        assert {size for _camera, size in observed} == {2}
        counters = value.stats()["counters"]
        assert counters["batches_2"] == 2
        assert counters.get("batches_4", 0) == 0
    finally:
        value.stop()


def test_profile_batch_limit_two_splits_four_cameras_into_two_cohorts():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.02,
        singleton_wait_seconds=0.2,
    )
    value.start()
    try:
        observed = []
        profile = BatchProfile("max-two", max_batch_size=2)
        camera_ids = [f"cam-{index}" for index in range(4)]
        for camera_id in camera_ids:
            value.register(camera_id, _OWNER)
            value.offer(
                camera_id,
                _OWNER,
                _work(
                    1,
                    lambda size, camera=camera_id: observed.append((camera, size)),
                    profile=profile,
                ),
            )

        results = [_take_result(value, camera_id) for camera_id in camera_ids]
        assert {result.batch_size for result in results} == {2}
        assert {size for _camera, size in observed} == {2}
        counters = value.stats()["counters"]
        assert counters["batches_2"] == 2
        assert counters.get("batches_4", 0) == 0
    finally:
        value.stop()


def test_queued_work_is_rejected_when_it_becomes_stale_before_dispatch():
    observed_drops = []
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.1,
        singleton_wait_seconds=0.2,
        drop_observer=lambda camera_id, reason, amount: observed_drops.append(
            (camera_id, reason, amount)
        ),
    )
    value.start()
    try:
        called = threading.Event()
        value.register("cam-1", _OWNER)
        assert value.offer(
            "cam-1",
            _OWNER,
            _work(1, lambda _size: called.set(), expires_in=0.03),
        ).accepted

        _wait_for(
            lambda: value.stats()["queued"] == 0,
            message="expired work remained queued",
        )
        assert not called.is_set()
        assert value.take_result("cam-1", _OWNER) is None
        counters = value.stats()["counters"]
        assert counters["stale_drops"] == 1
        assert counters["stale_dispatch_drops"] == 1
        assert observed_drops == [("cam-1", "stale", 1)]
    finally:
        value.stop()


def test_result_that_expires_during_provider_execution_is_never_published():
    observed_drops = []
    value = SharedInferenceScheduler(
        drop_observer=lambda camera_id, reason, amount: observed_drops.append(
            (camera_id, reason, amount)
        )
    )
    value.start()
    try:
        started = threading.Event()

        def slow_result(_batch_size):
            started.set()
            time.sleep(0.04)
            return "stale evidence"

        value.register("cam-1", _OWNER)
        assert value.offer(
            "cam-1",
            _OWNER,
            _work(1, slow_result, urgent=True, expires_in=0.02),
        ).accepted
        assert started.wait(timeout=1)
        _wait_for(
            lambda: value.stats()["inflight"] == 0,
            message="slow provider result did not complete",
        )

        assert value.take_result("cam-1", _OWNER) is None
        counters = value.stats()["counters"]
        assert counters["stale_drops"] == 1
        assert counters["stale_completion_drops"] == 1
        assert counters.get("completed", 0) == 0
        assert observed_drops == [("cam-1", "stale", 1)]
    finally:
        value.stop()


def test_latest_replacement_inherits_queued_urgent_redetection_intent():
    value = SharedInferenceScheduler(max_workers=4)
    value.start()
    blocker_started = [threading.Event() for _ in range(4)]
    release_blocker = threading.Event()
    latest_called = threading.Event()
    single = BatchProfile("single", max_batch_size=1)
    try:
        value.register("cam-1", _OWNER)
        for index, started in enumerate(blocker_started):
            camera_id = f"blocker-{index}"
            value.register(camera_id, _OWNER)

            def block(_batch_size, *, marker=started):
                marker.set()
                assert release_blocker.wait(timeout=2)

            assert value.offer(
                camera_id, _OWNER, _work(1, block, profile=single)
            ).accepted
        assert all(marker.wait(timeout=1) for marker in blocker_started)
        assert value.offer(
            "cam-1",
            _OWNER,
            _work(
                1,
                lambda _size: pytest.fail("superseded frame executed"),
                profile=single,
                urgent_reasons=("new_entry",),
            ),
        ).accepted
        assert value.offer(
            "cam-1",
            _OWNER,
            _work(2, lambda _size: latest_called.set(), profile=single),
        ).status is OfferStatus.REPLACED

        release_blocker.set()
        for index in range(4):
            _take_result(value, f"blocker-{index}")
        _take_result(value, "cam-1")

        assert latest_called.is_set()
        assert value.stats()["counters"]["urgent_dispatched"] == 1
    finally:
        release_blocker.set()
        value.stop()


def test_old_owner_completion_is_fenced_even_when_token_is_reused():
    value = SharedInferenceScheduler()
    value.start()
    release_old = threading.Event()
    old_started = threading.Event()
    try:
        old_generation = value.register("cam-1", _OWNER)

        def run_old(_batch_size):
            old_started.set()
            assert release_old.wait(timeout=2)
            return "old"

        value.offer("cam-1", _OWNER, _work(1, run_old, urgent=True))
        assert old_started.wait(timeout=1)
        assert value.unregister("cam-1", _OWNER)
        new_generation = value.register("cam-1", _OWNER)
        assert new_generation != old_generation
        value.offer(
            "cam-1",
            _OWNER,
            _work(1, lambda _size: "new", urgent=True),
        )
        _wait_for(
            lambda: value.stats()["completedUnconsumed"] == 1,
            message="new lifecycle did not complete",
        )

        release_old.set()
        _wait_for(
            lambda: value.stats()["counters"].get("fenced_completions", 0) == 1,
            message="old lifecycle completion was not fenced",
        )
        result = value.take_result("cam-1", _OWNER)
        assert result is not None
        assert result.value == "new"
        assert result.sequence == 1
        assert value.take_result("cam-1", _OWNER) is None
    finally:
        release_old.set()
        value.stop()


def test_due_normal_work_gets_a_turn_during_sustained_urgent_load():
    value = SharedInferenceScheduler(
        batch2_wait_seconds=0.01,
        singleton_wait_seconds=0.03,
        urgent_batch_burst=1,
    )
    value.start()
    blocker_releases = [threading.Event() for _ in range(4)]
    blocker_started = [threading.Event() for _ in range(4)]
    normal_started = threading.Event()
    normal_release = threading.Event()
    urgent_started = threading.Event()
    try:
        profile = BatchProfile("single", max_batch_size=1)
        for index in range(4):
            camera_id = f"blocker-{index}"
            value.register(camera_id, _OWNER)

            def block(_size, item=index):
                blocker_started[item].set()
                assert blocker_releases[item].wait(timeout=2)

            value.offer(
                camera_id,
                _OWNER,
                _work(1, block, profile=profile, urgent=True),
            )
        assert all(event.wait(timeout=1) for event in blocker_started)

        value.register("normal", _OWNER)
        value.register("urgent", _OWNER)

        def run_normal(_batch_size):
            normal_started.set()
            assert normal_release.wait(timeout=2)

        value.offer(
            "normal",
            _OWNER,
            _work(1, run_normal, profile=profile),
        )
        value.offer(
            "urgent",
            _OWNER,
            _work(
                1,
                lambda _size: urgent_started.set(),
                profile=profile,
                urgent=True,
            ),
        )
        time.sleep(0.05)

        blocker_releases[0].set()
        assert normal_started.wait(timeout=1)
        assert not urgent_started.is_set()

        blocker_releases[1].set()
        assert urgent_started.wait(timeout=1)
    finally:
        normal_release.set()
        for release in blocker_releases:
            release.set()
        value.stop()


def test_latency_samples_are_fixed_size_and_counters_saturate():
    value = SharedInferenceScheduler(
        latency_window_size=3,
        per_camera_latency_window_size=2,
    )
    value.start()
    try:
        value.register("cam-1", _OWNER)
        for sequence in range(5):
            value.offer(
                "cam-1",
                _OWNER,
                _work(sequence, lambda size: size, urgent=True),
            )
            assert _take_result(value, "cam-1").succeeded

        stats = value.stats()
        assert stats["latency"]["queue"]["sampleCount"] == 3
        assert stats["latency"]["service"]["sampleCount"] == 3
        assert stats["latency"]["frameAge"]["sampleCount"] == 3
        camera_latency = stats["cameras"]["cam-1"]["latency"]
        assert camera_latency["queue"]["sampleCount"] == 2
        assert camera_latency["service"]["sampleCount"] == 2
        assert camera_latency["frameAge"]["sampleCount"] == 2

        counters = {"events": COUNTER_MAX - 1}
        value._increment(counters, "events", 100)
        assert counters["events"] == COUNTER_MAX
    finally:
        value.stop()


def test_execution_errors_are_returned_once_without_blocking_future_work(scheduler):
    scheduler.register("cam-1", _OWNER)

    def fail(_batch_size):
        raise ValueError("bad tensor")

    scheduler.offer("cam-1", _OWNER, _work(1, fail, urgent=True))
    failed = _take_result(scheduler, "cam-1")
    assert failed.succeeded is False
    assert failed.value is None
    assert isinstance(failed.error, ValueError)
    assert str(failed.error) == "bad tensor"
    assert scheduler.take_result("cam-1", _OWNER) is None

    scheduler.offer(
        "cam-1",
        _OWNER,
        _work(2, lambda size: f"ok-{size}", urgent=True),
    )
    recovered = _take_result(scheduler, "cam-1")
    assert recovered.succeeded is True
    assert recovered.value == "ok-1"
    assert scheduler.stats()["counters"]["failed"] == 1


def test_clean_shutdown_drops_queued_work_and_drains_inflight_call():
    value = SharedInferenceScheduler()
    value.start()
    value.register("cam-1", _OWNER)
    first_started = threading.Event()
    release_first = threading.Event()
    second_called = threading.Event()

    def run_first(_batch_size):
        first_started.set()
        assert release_first.wait(timeout=2)
        return "done"

    value.offer("cam-1", _OWNER, _work(1, run_first, urgent=True))
    assert first_started.wait(timeout=1)
    value.offer(
        "cam-1",
        _OWNER,
        _work(2, lambda _size: second_called.set(), urgent=True),
    )
    stopped = []
    stop_thread = threading.Thread(target=lambda: stopped.append(value.stop()))
    stop_thread.start()
    _wait_for(
        lambda: value.stats()["accepting"] is False,
        message="shutdown did not stop admission",
    )
    assert stop_thread.is_alive()
    release_first.set()
    stop_thread.join(timeout=2)

    assert not stop_thread.is_alive()
    assert stopped == [True]
    assert not second_called.is_set()
    stats = value.stats()
    assert stats["running"] is False
    assert stats["queued"] == 0
    assert stats["inflight"] == 0
    assert stats["counters"]["shutdown_queue_drops"] == 1
    assert (
        value.offer("cam-1", _OWNER, _work(3, lambda _size: None)).status
        is OfferStatus.NOT_RUNNING
    )


def test_shutdown_timeout_returns_without_waiting_for_hung_inference():
    value = SharedInferenceScheduler()
    value.start()
    value.register("cam-1", _OWNER)
    started = threading.Event()
    release = threading.Event()

    def run(_batch_size):
        started.set()
        release.wait(timeout=2)

    value.offer("cam-1", _OWNER, _work(1, run, urgent=True))
    assert started.wait(timeout=1)
    before = time.monotonic()
    try:
        assert value.stop(timeout=0.02) is False
        assert time.monotonic() - before < 0.2
        assert value.stats()["running"] is False
    finally:
        release.set()
