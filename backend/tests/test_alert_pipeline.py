import asyncio
import threading
import time

import cv2
import numpy as np
import pytest

import alert_pipeline
from alert_pipeline import AlertPipeline, DeliveryOutcome
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
        deliver_alert=lambda _alert, _outputs: True,
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
        deliver_alert=lambda _alert, _outputs: (time.sleep(0.2) or True),
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
        return True

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


def test_false_delivery_result_retries_then_succeeds():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        return len(calls) >= 3

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=3,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, None, None]
    assert pipeline.stats()["delivered"] == 1
    assert pipeline.stats()["delivery_failures"] == 0
    assert pipeline.stats()["delivery_attempts"] == 3
    assert pipeline.stats()["delivery_retries"] == 2
    assert pipeline.shutdown(timeout=1)


def test_delivery_callback_exception_is_retried():
    calls = 0

    def deliver(_alert, _output_ids):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary provider failure")
        return True

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == 2
    assert pipeline.stats()["delivered"] == 1
    assert pipeline.stats()["delivery_retries"] == 1
    assert pipeline.shutdown(timeout=1)


def test_retry_backoff_does_not_occupy_delivery_worker():
    attempts = {}
    first_attempt_finished = threading.Event()
    second_delivered = threading.Event()

    def deliver(alert, _output_ids):
        sequence = alert["sequence"]
        attempts[sequence] = attempts.get(sequence, 0) + 1
        if sequence == 1 and attempts[sequence] == 1:
            first_attempt_finished.set()
            return False
        if sequence == 2:
            second_delivered.set()
        return True

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_workers=1,
        delivery_attempts=2,
        delivery_retry_delay=0.3,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert first_attempt_finished.wait(timeout=1)
    pipeline.submit({"sequence": 2}).result(timeout=1)

    assert second_delivered.wait(timeout=0.15)
    assert pipeline.drain(timeout=1)
    assert attempts == {1: 2, 2: 1}
    assert pipeline.shutdown(timeout=1)


def test_partial_delivery_retries_only_the_failed_channel():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        if output_ids is None:
            return DeliveryOutcome(
                delivered_output_ids=("email",),
                retry_output_ids=("telegram",),
            )
        return DeliveryOutcome(delivered_output_ids=("telegram",))

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=3,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, ["telegram"]]
    assert pipeline.stats()["delivered"] == 1
    assert pipeline.stats()["partially_delivered"] == 0
    assert pipeline.stats()["delivery_retries"] == 1
    assert pipeline.shutdown(timeout=1)


def test_incomplete_explicit_delivery_outcome_is_retried_not_counted_success():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        return DeliveryOutcome(delivered_output_ids=("email",))

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit(
        {"sequence": 1},
        output_ids=["email", "telegram"],
    ).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [["email", "telegram"], ["email", "telegram"]]
    stats = pipeline.stats()
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 1
    assert stats["delivery_retry_exhausted"] == 1
    assert pipeline.shutdown(timeout=1)


def test_retry_all_cannot_claim_explicit_target_outcomes():
    calls = 0

    def deliver(_alert, _output_ids):
        nonlocal calls
        calls += 1
        return DeliveryOutcome(delivered_output_ids=("email",), retry_all=True)

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=1,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}, output_ids=["email"]).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == 1
    assert pipeline.stats()["delivered"] == 0
    assert pipeline.stats()["delivery_failures"] == 1
    assert pipeline.shutdown(timeout=1)


def test_terminal_partial_delivery_is_visible_without_retry():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        return DeliveryOutcome(
            delivered_output_ids=("email",),
            terminal_output_ids=("sms",),
        )

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None]
    stats = pipeline.stats()
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 1
    assert stats["partially_delivered"] == 1
    assert stats["delivery_terminal_failures"] == 1
    assert stats["delivery_retry_exhausted"] == 0
    assert pipeline.shutdown(timeout=1)


def test_terminal_failure_remains_visible_after_other_target_retry_succeeds():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        if output_ids is None:
            return DeliveryOutcome(
                delivered_output_ids=("email",),
                retry_output_ids=("telegram",),
                terminal_output_ids=("sms",),
            )
        return DeliveryOutcome(delivered_output_ids=("telegram",))

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, ["telegram"]]
    stats = pipeline.stats()
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 1
    assert stats["partially_delivered"] == 1
    assert stats["delivery_terminal_failures"] == 1
    assert stats["delivery_retry_exhausted"] == 0
    assert pipeline.shutdown(timeout=1)


def test_exhausted_delivery_is_not_counted_as_delivered():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        return False

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, None]
    stats = pipeline.stats()
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 1
    assert stats["delivery_attempts"] == 2
    assert stats["delivery_retries"] == 1
    assert stats["delivery_retry_exhausted"] == 1
    assert pipeline.shutdown(timeout=1)


def test_exhausted_partial_delivery_keeps_success_and_failure_distinct():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)
        if output_ids is None:
            return DeliveryOutcome(
                delivered_output_ids=("email",),
                retry_output_ids=("telegram",),
            )
        return DeliveryOutcome(retry_output_ids=("telegram",))

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, ["telegram"]]
    stats = pipeline.stats()
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 1
    assert stats["partially_delivered"] == 1
    assert stats["delivery_retry_exhausted"] == 1
    assert pipeline.shutdown(timeout=1)


def test_missing_delivery_return_is_a_failure_not_success():
    calls = []

    def deliver(_alert, output_ids):
        calls.append(output_ids)

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=2,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == [None, None]
    assert pipeline.stats()["delivered"] == 0
    assert pipeline.stats()["delivery_failures"] == 1
    assert pipeline.shutdown(timeout=1)


def test_malformed_typed_delivery_outcome_does_not_kill_worker():
    def deliver(alert, _output_ids):
        if alert["sequence"] == 1:
            return DeliveryOutcome(delivered_output_ids=([],))
        return True

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_workers=1,
        delivery_attempts=1,
        delivery_retry_delay=0,
    )

    first = pipeline.submit({"sequence": 1})
    second = pipeline.submit({"sequence": 2})
    first.result(timeout=1)
    second.result(timeout=1)
    assert pipeline.drain(timeout=1)
    stats = pipeline.stats()
    assert stats["delivery_failures"] == 1
    assert stats["delivered"] == 1
    assert stats["delivery_workers_alive"] == 1
    assert pipeline.shutdown(timeout=1)


def test_retry_backlog_is_bounded_and_overflow_is_visible():
    initial_attempts = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def deliver(_alert, _output_ids):
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls >= 4:
                initial_attempts.set()
        return False

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_queue_size=8,
        delivery_retry_queue_size=1,
        delivery_workers=1,
        delivery_attempts=2,
        delivery_retry_delay=0.2,
    )

    futures = [pipeline.submit({"sequence": sequence}) for sequence in range(4)]
    for future in futures:
        future.result(timeout=1)
    assert initial_attempts.wait(timeout=1)
    deadline = time.monotonic() + 1
    while pipeline.stats()["delivery_retry_queue_full"] < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    stats = pipeline.stats()
    assert stats["delivery_retry_queue_depth"] <= 1
    assert stats["delivery_retry_queue_full"] == 3
    assert pipeline.drain(timeout=1)
    stats = pipeline.stats()
    assert stats["delivery_failures"] == 4
    assert stats["delivery_retries"] == 1
    assert stats["delivery_retry_exhausted"] == 1
    assert pipeline.shutdown(timeout=1)


def test_retry_handoff_releases_capacity_before_fast_next_attempt():
    calls = 0

    def deliver(_alert, _output_ids):
        nonlocal calls
        calls += 1
        return False

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_queue_size=1,
        delivery_retry_queue_size=1,
        delivery_workers=1,
        delivery_attempts=3,
        delivery_retry_delay=0,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert calls == 3
    stats = pipeline.stats()
    assert stats["delivery_retries"] == 2
    assert stats["delivery_retry_queue_full"] == 0
    assert stats["delivery_failures"] == 1
    assert pipeline.shutdown(timeout=1)


def test_explicit_inapp_target_satisfies_contract_without_external_partial_metric():
    def deliver(_alert, output_ids):
        assert output_ids == ["inApp", "email"]
        return DeliveryOutcome(
            delivered_output_ids=("email",),
            handled_output_ids=("inapp",),
        )

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_attempts=1,
    )

    pipeline.submit(
        {"sequence": 1},
        output_ids=["inApp", "email"],
    ).result(timeout=1)
    assert pipeline.drain(timeout=1)
    stats = pipeline.stats()
    assert stats["delivered"] == 1
    assert stats["delivery_failures"] == 0
    assert stats["partially_delivered"] == 0
    assert pipeline.shutdown(timeout=1)


def test_unstarted_pipeline_stats_keep_the_operational_schema(monkeypatch):
    monkeypatch.setattr(video_processing, "_alert_pipeline", None)

    stats = video_processing.get_alert_pipeline_stats()

    assert stats["running"] is False
    assert stats["accepting"] is False
    assert stats["delivery_attempts"] == 0
    assert stats["delivery_retries"] == 0
    assert stats["delivery_failures"] == 0
    assert stats["partially_delivered"] == 0
    assert stats["delivery_retry_queue_full"] == 0
    assert stats["outbox_handoffs"] == 0
    assert stats["consecutive_persistence_failures"] == 0
    assert stats["last_persistence_failure_at"] is None
    assert stats["last_persistence_success_at"] is None
    assert stats["persistence_in_flight"] is False
    assert stats["oldest_persistence_age_seconds"] is None


def test_durable_outbox_handoff_is_not_counted_as_provider_delivery(monkeypatch):
    wakes = []
    monkeypatch.setattr(video_processing.alert_delivery_worker, "wake", lambda: wakes.append(True))
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=video_processing._outbox_handoff,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)

    stats = pipeline.stats()
    assert wakes == [True]
    assert stats["outbox_handoffs"] == 1
    assert stats["delivered"] == 0
    assert stats["delivery_failures"] == 0
    assert pipeline.shutdown(timeout=1)


def test_legacy_true_delivery_callback_keeps_delivered_semantics():
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )

    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)
    assert pipeline.stats()["delivered"] == 1
    assert pipeline.stats()["outbox_handoffs"] == 0
    assert pipeline.shutdown(timeout=1)


def test_invalid_delivery_routing_does_not_suppress_alert_persistence(
    monkeypatch,
    caplog,
):
    captured = []

    class FakePipeline:
        def submit(self, payload, **_kwargs):
            captured.append(payload)
            return "queued"

    monkeypatch.setattr(video_processing, "_get_alert_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        video_processing,
        "get_config_snapshot",
        lambda: {"cameras": {"cam1": {"name": "Camera 1", "zone": "Loading"}}},
    )

    def invalid_routing(*_args, **_kwargs):
        raise ValueError("smtp-password-must-not-leak")

    monkeypatch.setattr(
        video_processing.notification_dispatcher,
        "resolve_delivery_targets",
        invalid_routing,
    )

    result = video_processing.create_alert(
        "cam1",
        "No helmet",
        "P2",
        0.9,
        snapshot_jpeg=b"snapshot",
        clean_snapshot_jpeg=b"clean",
    )

    assert result == "queued"
    assert captured[0]["delivery_targets"] == []
    assert "smtp-password-must-not-leak" not in caplog.text


def test_start_failure_rolls_back_durable_outbox(monkeypatch):
    calls = []

    class FailingPipeline:
        def start(self):
            calls.append("pipeline-start")
            raise RuntimeError("pipeline start failed")

        def shutdown(self, **_kwargs):
            calls.append("pipeline-stop")
            return True

    monkeypatch.setattr(video_processing, "_get_alert_pipeline", lambda: FailingPipeline())
    monkeypatch.setattr(
        video_processing.alert_delivery_worker,
        "start",
        lambda: calls.append("outbox-start"),
    )
    monkeypatch.setattr(
        video_processing.alert_delivery_worker,
        "stop",
        lambda _timeout: calls.append("outbox-stop") or True,
    )

    with pytest.raises(RuntimeError, match="pipeline start failed"):
        video_processing.start_alert_pipeline()

    assert calls == ["outbox-start", "pipeline-start", "pipeline-stop", "outbox-stop"]
    assert video_processing._alert_event_loop is None


def test_stop_always_stops_durable_outbox_without_legacy_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(video_processing, "_alert_pipeline", None)
    monkeypatch.setattr(
        video_processing.alert_delivery_worker,
        "stop",
        lambda _timeout: calls.append("outbox-stop") or True,
    )

    assert video_processing.stop_alert_pipeline(1.0) is True
    assert calls == ["outbox-stop"]


def test_stop_still_stops_durable_outbox_when_legacy_shutdown_fails(monkeypatch):
    calls = []

    class FailingPipeline:
        def shutdown(self, **_kwargs):
            calls.append("pipeline-stop")
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(video_processing, "_alert_pipeline", FailingPipeline())
    monkeypatch.setattr(
        video_processing.alert_delivery_worker,
        "stop",
        lambda _timeout: calls.append("outbox-stop") or True,
    )

    assert video_processing.stop_alert_pipeline(1.0) is False
    assert calls == ["pipeline-stop", "outbox-stop"]


def test_stale_alerts_are_resolved_before_delivery_workers_start(monkeypatch):
    calls = []
    monkeypatch.setattr(
        video_processing.alert_store,
        "auto_resolve_stale_alerts_batch",
        lambda: (calls.append("auto-resolve") or (1, False)),
    )
    monkeypatch.setattr(
        video_processing.alert_delivery_store,
        "backfill_active_escalations_batch",
        lambda *_args, **_kwargs: (0, None),
    )
    monkeypatch.setattr(
        video_processing,
        "start_alert_pipeline",
        lambda: calls.append("workers-start"),
    )

    asyncio.run(video_processing.start_alert_delivery_workers())

    assert calls == ["auto-resolve", "workers-start"]


def test_upgrade_backfill_is_bounded_before_start_and_continues_online(monkeypatch):
    calls = []
    cutoffs = []

    def backfill(_cfg, _resolver, *, after_id=None, minimum_timestamp=None, **_kwargs):
        calls.append(f"backfill:{after_id or 'first'}")
        cutoffs.append(minimum_timestamp)
        return (1, "cursor-1") if after_id is None else (1, None)

    monkeypatch.setattr(
        video_processing.alert_store,
        "auto_resolve_stale_alerts_batch",
        lambda: (calls.append("auto-resolve") or (0, False)),
    )
    monkeypatch.setattr(
        video_processing.alert_delivery_store,
        "backfill_active_escalations_batch",
        backfill,
    )
    monkeypatch.setattr(
        video_processing,
        "start_alert_pipeline",
        lambda: calls.append("workers-start"),
    )

    async def run_startup():
        await video_processing.start_alert_delivery_workers()
        assert video_processing._alert_backfill_task is not None
        await video_processing._alert_backfill_task

    asyncio.run(run_startup())

    assert calls == [
        "auto-resolve",
        "backfill:first",
        "workers-start",
        "backfill:cursor-1",
    ]
    assert cutoffs[0]
    assert cutoffs == [cutoffs[0], cutoffs[0]]


def test_stale_reconciliation_is_bounded_before_start_and_continues_online(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        video_processing.alert_store,
        "auto_resolve_stale_alerts_batch",
        lambda: (calls.append("reconcile:first") or (250, True)),
    )

    async def finish_reconciliation():
        calls.append("reconcile:remaining")
        return 3

    monkeypatch.setattr(
        video_processing.alert_store,
        "reconcile_stale_alerts",
        finish_reconciliation,
    )
    monkeypatch.setattr(
        video_processing.alert_delivery_store,
        "backfill_active_escalations_batch",
        lambda *_args, **_kwargs: (0, None),
    )
    monkeypatch.setattr(
        video_processing,
        "start_alert_pipeline",
        lambda: calls.append("workers-start"),
    )

    async def run_startup():
        await video_processing.start_alert_delivery_workers()
        assert video_processing._alert_reconciliation_task is not None
        await video_processing._alert_reconciliation_task

    asyncio.run(run_startup())

    assert calls == [
        "reconcile:first",
        "workers-start",
        "reconcile:remaining",
    ]


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
        deliver_alert=lambda _alert, _outputs: True,
        persistence_attempts=3,
        retry_delay=0,
    )

    assert pipeline.submit({"sequence": 1}).result(timeout=1)["id"] == "1"
    assert attempts == 3
    assert pipeline.stats()["persistence_failures"] == 0
    assert pipeline.shutdown(timeout=1)


def test_started_idle_pipeline_has_no_false_persistence_failure_signal():
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )

    pipeline.start()
    stats = pipeline.stats()

    assert stats["consecutive_persistence_failures"] == 0
    assert stats["last_persistence_failure_at"] is None
    assert stats["last_persistence_success_at"] is None
    assert stats["persistence_in_flight"] is False
    assert stats["oldest_persistence_age_seconds"] is None
    assert pipeline.shutdown(timeout=1)


def test_permanent_persistence_failure_is_visible_on_future_and_stats():
    def persist(**_payload):
        raise RuntimeError("database unavailable")

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persistence_attempts=2,
        retry_delay=0,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        pipeline.submit({"sequence": 1}).result(timeout=1)
    stats = pipeline.stats()
    assert stats["persistence_failures"] == 1
    assert stats["consecutive_persistence_failures"] == 1
    assert stats["last_persistence_failure_at"] is not None
    assert stats["last_persistence_success_at"] is None
    assert pipeline.shutdown(timeout=1)


def test_success_after_permanent_persistence_failure_clears_unresolved_signal():
    attempts = 0

    def persist(**payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("database unavailable")
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persistence_attempts=1,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        pipeline.submit({"sequence": 1}).result(timeout=1)
    failure_stats = pipeline.stats()
    assert failure_stats["consecutive_persistence_failures"] == 1

    assert pipeline.submit({"sequence": 2}).result(timeout=1)["sequence"] == 2
    recovery_stats = pipeline.stats()
    assert recovery_stats["persistence_failures"] == 1
    assert recovery_stats["consecutive_persistence_failures"] == 0
    assert recovery_stats["last_persistence_success_at"] >= (
        failure_stats["last_persistence_failure_at"]
    )
    assert pipeline.shutdown(timeout=1)


def test_stats_expose_oldest_in_flight_persistence_age():
    persistence_started = threading.Event()
    release_persistence = threading.Event()

    def persist(**payload):
        persistence_started.set()
        assert release_persistence.wait(timeout=2)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
    )
    future = pipeline.submit({"sequence": 1})
    assert persistence_started.wait(timeout=1)

    stats = pipeline.stats()
    assert stats["persistence_in_flight"] is True
    assert stats["oldest_persistence_age_seconds"] is not None
    assert stats["consecutive_persistence_failures"] == 0
    assert stats["last_persistence_failure_at"] is None
    assert stats["last_persistence_success_at"] is None

    release_persistence.set()
    assert future.result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=1)
    stats = pipeline.stats()
    assert stats["persistence_in_flight"] is False
    assert stats["oldest_persistence_age_seconds"] is None
    assert pipeline.shutdown(timeout=1)


def test_non_mapping_persistence_result_is_rejected_without_killing_worker():
    def persist(**payload):
        if payload["sequence"] == 1:
            return []
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persistence_attempts=1,
    )

    with pytest.raises(TypeError, match="must return a mapping"):
        pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.submit({"sequence": 2}).result(timeout=1)["sequence"] == 2
    assert pipeline.drain(timeout=1)
    stats = pipeline.stats()
    assert stats["persisted"] == 1
    assert stats["persistence_failures"] == 1
    assert stats["delivered"] == 1
    assert stats["persist_worker_alive"] is True
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
        deliver_alert=lambda _alert, _outputs: True,
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


def test_optional_advisory_submission_rejects_full_queue_without_waiting():
    first_started = threading.Event()
    release_first = threading.Event()

    def persist(**payload):
        if payload["sequence"] == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persist_queue_size=1,
        submit_timeout=1.0,
    )
    first = pipeline.submit({"sequence": 1})
    assert first_started.wait(timeout=1)
    second = pipeline.submit({"sequence": 2})

    started = time.perf_counter()
    with pytest.raises(alert_pipeline.queue.Full):
        pipeline.submit(
            {"sequence": 3},
            allow_backpressure=False,
        )
    assert time.perf_counter() - started < 0.05
    assert pipeline.stats()["submitted"] == 2
    assert pipeline.stats()["backpressure_events"] == 1

    release_first.set()
    assert first.result(timeout=1)["sequence"] == 1
    assert second.result(timeout=1)["sequence"] == 2
    assert pipeline.drain(timeout=1)
    assert pipeline.shutdown(timeout=1)


def test_timed_out_shutdown_does_not_strand_inflight_persistence():
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    delivered = []

    def persist(**payload):
        persistence_started.set()
        assert release_persistence.wait(timeout=2)
        return _alert_from_payload(**payload)

    def deliver(alert, _output_ids):
        delivered.append(alert["sequence"])
        return True

    pipeline = AlertPipeline(persist_alert=persist, deliver_alert=deliver)
    future = pipeline.submit({"sequence": 1})
    assert persistence_started.wait(timeout=1)

    started = time.perf_counter()
    assert pipeline.shutdown(timeout=0.05) is False
    assert time.perf_counter() - started < 0.25
    assert pipeline.stats()["running"] is True
    assert pipeline.stats()["accepting"] is False
    with pytest.raises(RuntimeError, match="shutting down"):
        pipeline.submit({"sequence": 2})

    release_persistence.set()
    assert future.result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=1)
    assert delivered == [1]
    assert pipeline.shutdown(timeout=1)


def test_full_delivery_queue_shutdown_respects_timeout_and_later_drains():
    first_delivery_started = threading.Event()
    release_first_delivery = threading.Event()
    delivered = []

    def deliver(alert, _output_ids):
        if alert["sequence"] == 1:
            first_delivery_started.set()
            assert release_first_delivery.wait(timeout=2)
        delivered.append(alert["sequence"])
        return True

    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=deliver,
        delivery_queue_size=1,
        delivery_workers=1,
    )
    futures = [pipeline.submit({"sequence": sequence}) for sequence in (1, 2, 3)]
    assert first_delivery_started.wait(timeout=1)
    time.sleep(0.05)

    started = time.perf_counter()
    assert pipeline.shutdown(timeout=0.05) is False
    assert time.perf_counter() - started < 0.25
    assert pipeline.stats()["running"] is True
    assert pipeline.stats()["accepting"] is False

    release_first_delivery.set()
    for future in futures:
        assert future.result(timeout=1)["sequence"] in (1, 2, 3)
    assert pipeline.drain(timeout=1)
    assert delivered == [1, 2, 3]
    assert pipeline.shutdown(timeout=1)


def test_concurrent_shutdown_is_serialized_and_pipeline_can_restart():
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )
    pipeline.submit({"sequence": 1}).result(timeout=1)
    assert pipeline.drain(timeout=1)

    ready = threading.Barrier(3)
    results = []

    def stop():
        ready.wait(timeout=1)
        results.append(pipeline.shutdown(timeout=1))

    callers = [threading.Thread(target=stop) for _ in range(2)]
    for caller in callers:
        caller.start()
    ready.wait(timeout=1)
    for caller in callers:
        caller.join(timeout=2)

    assert sorted(results) == [True, True]
    assert pipeline._persist_queue.unfinished_tasks == 0
    assert pipeline._delivery_queue.unfinished_tasks == 0

    assert pipeline.submit({"sequence": 2}).result(timeout=1)["sequence"] == 2
    assert pipeline.drain(timeout=1)
    assert pipeline.stats()["delivered"] == 2
    assert pipeline.shutdown(timeout=1)


def test_worker_start_failure_rolls_back_and_next_start_recovers(monkeypatch):
    original_start = threading.Thread.start
    failed = False

    def fail_persistence_worker_once(thread):
        nonlocal failed
        if thread.name == "alert-persistence" and not failed:
            failed = True
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_persistence_worker_once)
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )

    with pytest.raises(RuntimeError, match="can't start new thread"):
        pipeline.start()
    stats = pipeline.stats()
    assert stats["running"] is False
    assert stats["accepting"] is False
    assert stats["persist_worker_alive"] is False
    assert stats["retry_worker_alive"] is False
    assert stats["delivery_workers_alive"] == 0

    assert pipeline.submit({"sequence": 1}).result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=1)
    assert pipeline.shutdown(timeout=1)


def test_worker_construction_failure_rolls_back_and_next_start_recovers(monkeypatch):
    original_thread = threading.Thread
    failed = False

    def fail_retry_worker_construction_once(*args, **kwargs):
        nonlocal failed
        if kwargs.get("name") == "alert-delivery-retry-scheduler" and not failed:
            failed = True
            raise RuntimeError("can't construct thread")
        return original_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", fail_retry_worker_construction_once)
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )

    with pytest.raises(RuntimeError, match="can't construct thread"):
        pipeline.start()
    stats = pipeline.stats()
    assert stats["running"] is False
    assert stats["accepting"] is False
    assert stats["persist_worker_alive"] is False
    assert stats["retry_worker_alive"] is False
    assert stats["delivery_workers_alive"] == 0

    assert pipeline.submit({"sequence": 1}).result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=1)
    assert pipeline.shutdown(timeout=1)


def test_callback_dispatcher_start_failure_prevents_admission_and_recovers(monkeypatch):
    original_start = threading.Thread.start
    alert_pipeline._CALLBACK_WORKERS_STARTED = False
    failed = False

    def fail_callback_worker_once(thread):
        nonlocal failed
        if thread.name == "alert-future-callback-1" and not failed:
            failed = True
            raise RuntimeError("can't start callback thread")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_callback_worker_once)
    pipeline = AlertPipeline(
        persist_alert=_alert_from_payload,
        deliver_alert=lambda _alert, _outputs: True,
    )

    with pytest.raises(RuntimeError, match="can't start callback thread"):
        pipeline.start()
    assert pipeline.stats()["running"] is False
    assert pipeline.stats()["accepting"] is False

    assert pipeline.submit({"sequence": 1}).result(timeout=1)["sequence"] == 1
    assert pipeline.drain(timeout=1)
    assert pipeline.stats()["delivered"] == 1
    assert pipeline.shutdown(timeout=1)


def test_future_callbacks_keep_registration_order_even_when_added_after_completion():
    alert_pipeline._ensure_future_callback_worker()
    release_blocker = threading.Event()
    callbacks_finished = threading.Event()
    order = []

    blocker = alert_pipeline._AlertFuture()
    blocker.add_done_callback(lambda _future: release_blocker.wait(timeout=2))
    blocker.set_result(True)

    observed = alert_pipeline._AlertFuture()

    def record(label):
        order.append(label)
        if len(order) == 2:
            callbacks_finished.set()

    observed.add_done_callback(lambda _future: record("first"))
    observed.set_result(True)
    observed.add_done_callback(lambda _future: record("second"))
    release_blocker.set()

    assert callbacks_finished.wait(timeout=1)
    assert order == ["first", "second"]


def test_future_callback_can_reenter_submit_without_deadlocking_persistence():
    first_started = threading.Event()
    release_first = threading.Event()
    callback_finished = threading.Event()
    callback_futures = []

    def persist(**payload):
        if payload["sequence"] == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persist_queue_size=1,
    )
    first = pipeline.submit({"sequence": 1})
    assert first_started.wait(timeout=1)
    second = pipeline.submit({"sequence": 2})

    def submit_from_callback(_future):
        callback_futures.append(pipeline.submit({"sequence": 3}))
        callback_finished.set()

    first.add_done_callback(submit_from_callback)
    release_first.set()

    assert second.result(timeout=1)["sequence"] == 2
    assert callback_finished.wait(timeout=1)
    assert callback_futures[0].result(timeout=1)["sequence"] == 3
    assert pipeline.drain(timeout=1)
    assert pipeline.stats()["persisted"] == 3
    assert pipeline.shutdown(timeout=1)


def test_cancelled_persistence_future_does_not_kill_worker_or_alert():
    first_started = threading.Event()
    release_first = threading.Event()
    delivered = []

    def persist(**payload):
        if payload["sequence"] == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda alert, _outputs: delivered.append(alert["sequence"]) or True,
    )
    cancelled = pipeline.submit({"sequence": 1})
    assert first_started.wait(timeout=1)
    assert cancelled.cancel()
    release_first.set()

    second = pipeline.submit({"sequence": 2})
    assert second.result(timeout=1)["sequence"] == 2
    assert pipeline.drain(timeout=1)
    assert delivered == [1, 2]
    assert pipeline.stats()["persist_worker_alive"] is True
    assert pipeline.shutdown(timeout=1)


def test_submit_copies_mutable_payload_before_worker_uses_it():
    release = threading.Event()
    received = []

    def persist(**payload):
        assert release.wait(timeout=1)
        received.append(payload)
        return _alert_from_payload(**payload)

    pipeline = AlertPipeline(persist_alert=persist, deliver_alert=lambda _alert, _outputs: True)
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
        "get_config_snapshot",
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
        def __init__(self):
            self.callback = None

        def add_done_callback(self, callback):
            self.callback = callback

    def schedule(coroutine, loop):
        scheduled.append(loop)
        coroutine.close()
        result = FakeResult()
        scheduled_result.append(result)
        return result

    loop = FakeLoop()
    scheduled_result = []
    monkeypatch.setattr(video_processing, "_alert_event_loop", loop)
    monkeypatch.setattr(video_processing.asyncio, "run_coroutine_threadsafe", schedule)

    video_processing._broadcast_persisted_alert({"id": "alert-1"})

    assert scheduled == [loop]
    assert callable(scheduled_result[0].callback)
