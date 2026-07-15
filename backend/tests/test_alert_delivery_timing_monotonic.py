from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import alert_delivery_store
import alert_delivery_worker
import alert_timing
from delivery_result import DeliveryDisposition, ProviderDeliveryResult
from pipeline_telemetry import PipelineTelemetry


@pytest.fixture(autouse=True)
def _clear_timing_registry():
    alert_timing.registry.clear()
    yield
    alert_timing.registry.clear()


@pytest.mark.parametrize("database_clock_offset", (-86_400, 86_400))
def test_provider_success_latency_ignores_positive_and_negative_database_skew(
    monkeypatch,
    database_clock_offset: int,
):
    telemetry = PipelineTelemetry(
        telemetry_epoch="00000000-0000-4000-8000-000000000101"
    )
    monkeypatch.setattr(alert_delivery_worker, "pipeline_telemetry", telemetry)
    alert_timing.registry.remember(
        "alert-1",
        first_positive_ns=1_000_000_000,
        confirmed_ns=2_000_000_000,
        initial_target_keys=("initial:webhook:target",),
    )
    assert alert_timing.registry.mark_persisted(
        "alert-1", persisted_ns=2_500_000_000
    ) is not None
    telemetry.register_alert_delivery_targets(1)
    assert alert_timing.registry.activate_delivery_tracking("alert-1") == ()

    database_now = datetime(2026, 7, 15, tzinfo=timezone.utc) + timedelta(
        seconds=database_clock_offset
    )
    durable_metadata = alert_delivery_store.InitialDeliveryTiming(
        alert_id="alert-1",
        camera_id="cam-1",
        first_positive_at=database_now - timedelta(seconds=3),
        confirmed_at=database_now - timedelta(seconds=2),
        persisted_at=database_now - timedelta(seconds=1),
        first_initial_delivery_at=database_now,
    )
    monkeypatch.setattr(
        alert_delivery_worker.alert_delivery_store,
        "mark_delivered_with_timing",
        lambda *_args, **_kwargs: alert_delivery_store.DeliveryCompletion(
            updated=True,
            initial_delivery_timing=durable_metadata,
        ),
    )
    monkeypatch.setattr(
        alert_delivery_worker.time,
        "monotonic_ns",
        lambda: 4_100_000_000,
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1)

    worker._record_result(
        {
            "id": "delivery-1",
            "lease_token": "lease-1",
            "kind": "initial",
            "alert_id": "alert-1",
            "target_key": "initial:webhook:target",
            "channel": "webhook",
            "attempt_count": 1,
        },
        ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered"),
    )

    alerts = telemetry.public_snapshot()["alerts"]
    canonical = alerts["latency"]["firstPositiveToProviderSuccessMs"]
    compatibility = alerts["latency"]["firstPositiveToDeliveryHandoffMs"]
    assert canonical["count"] == 1
    assert canonical["maximumMs"] == 3_100.0
    assert compatibility == canonical
    assert alerts["deliveryCoverage"]["counters"]["deliveredCount"] == 1
    assert alerts["deliveryCoverage"]["pending"] == 0


def test_terminal_provider_outcome_stays_in_coverage_without_latency_sample(
    monkeypatch,
):
    telemetry = PipelineTelemetry(
        telemetry_epoch="00000000-0000-4000-8000-000000000102"
    )
    monkeypatch.setattr(alert_delivery_worker, "pipeline_telemetry", telemetry)
    alert_timing.registry.remember(
        "alert-2",
        first_positive_ns=1,
        confirmed_ns=2,
        initial_target_keys=("initial:webhook:target",),
    )
    assert alert_timing.registry.mark_persisted("alert-2", persisted_ns=3)
    telemetry.register_alert_delivery_targets(1)
    assert alert_timing.registry.activate_delivery_tracking("alert-2") == ()
    monkeypatch.setattr(
        alert_delivery_worker.alert_delivery_store,
        "mark_terminal",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(alert_delivery_worker.time, "monotonic_ns", lambda: 4)
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1)

    worker._record_result(
        {
            "id": "delivery-2",
            "lease_token": "lease-2",
            "kind": "initial",
            "alert_id": "alert-2",
            "target_key": "initial:webhook:target",
            "channel": "webhook",
            "attempt_count": 1,
        },
        ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Rejected",
            error_code="provider_rejected",
        ),
    )

    alerts = telemetry.public_snapshot()["alerts"]
    coverage = alerts["deliveryCoverage"]
    assert coverage["counters"]["terminalCount"] == 1
    assert coverage["counters"]["failedAttemptCount"] == 1
    assert coverage["pending"] == 0
    assert alerts["latency"]["firstPositiveToProviderSuccessMs"]["count"] == 0


def test_begin_send_terminal_fence_resolves_tracked_pending_target(monkeypatch):
    telemetry = PipelineTelemetry(
        telemetry_epoch="00000000-0000-4000-8000-000000000103"
    )
    monkeypatch.setattr(alert_delivery_worker, "pipeline_telemetry", telemetry)
    alert_timing.registry.remember(
        "alert-3",
        first_positive_ns=1,
        confirmed_ns=2,
        initial_target_keys=("initial:webhook:target",),
    )
    assert alert_timing.registry.mark_persisted("alert-3", persisted_ns=3)
    telemetry.register_alert_delivery_targets(1)
    assert alert_timing.registry.activate_delivery_tracking("alert-3") == ()
    monkeypatch.setattr(
        alert_delivery_worker.alert_store,
        "get_alert",
        lambda _alert_id: {"id": "alert-3"},
    )
    monkeypatch.setattr(
        alert_delivery_worker.alert_delivery_store,
        "begin_send",
        lambda *_args, **_kwargs: {
            "started": False,
            "reason": "delivery_expired",
            "final_state": "terminal",
            "alert_id": "alert-3",
            "kind": "initial",
            "target_key": "initial:webhook:target",
        },
    )
    monkeypatch.setattr(alert_delivery_worker.time, "monotonic_ns", lambda: 4)
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1)

    worker._deliver_claimed(
        {
            "id": "delivery-3",
            "lease_token": "lease-3",
            "kind": "initial",
            "alert_id": "alert-3",
            "target_key": "initial:webhook:target",
            "channel": "webhook",
        }
    )

    coverage = telemetry.public_snapshot()["alerts"]["deliveryCoverage"]
    assert coverage["counters"]["terminalCount"] == 1
    assert coverage["pending"] == 0
    assert alert_timing.registry.stats()["pending"] == 0


def test_claim_cleanup_terminalization_resolves_tracked_pending_target(monkeypatch):
    telemetry = PipelineTelemetry(
        telemetry_epoch="00000000-0000-4000-8000-000000000104"
    )
    monkeypatch.setattr(alert_delivery_worker, "pipeline_telemetry", telemetry)
    alert_timing.registry.remember(
        "alert-4",
        first_positive_ns=1,
        confirmed_ns=2,
        initial_target_keys=("initial:webhook:target",),
    )
    assert alert_timing.registry.mark_persisted("alert-4", persisted_ns=3)
    telemetry.register_alert_delivery_targets(1)
    assert alert_timing.registry.activate_delivery_tracking("alert-4") == ()
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        poll_seconds=0.05,
    )

    def claim_once(**kwargs):
        assert kwargs["include_terminalized"] is True
        worker._stop_claiming.set()
        return {
            "delivery": None,
            "terminalized": [
                {
                    "alert_id": "alert-4",
                    "kind": "initial",
                    "target_key": "initial:webhook:target",
                }
            ],
        }

    monkeypatch.setattr(
        alert_delivery_worker.alert_delivery_store,
        "claim_due",
        claim_once,
    )
    monkeypatch.setattr(alert_delivery_worker.time, "monotonic_ns", lambda: 4)

    worker._claim_loop()

    coverage = telemetry.public_snapshot()["alerts"]["deliveryCoverage"]
    assert coverage["counters"]["terminalCount"] == 1
    assert coverage["pending"] == 0
    assert alert_timing.registry.stats()["pending"] == 0
