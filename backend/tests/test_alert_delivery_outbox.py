from __future__ import annotations

import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://localhost:5432/safetylens_test",
)
os.environ["DATABASE_URL"] = TEST_DB_URL

import alert_delivery_store
import alert_delivery_worker
import alert_store
import notification_dispatcher
from delivery_result import DeliveryDisposition, ProviderDeliveryResult


_snapshots = Path(tempfile.mkdtemp()) / "outbox-snapshots"


def _target(
    channel: str = "webhook",
    *,
    suffix: str = "target",
    kind: str = "initial",
    delay_seconds: float = 0,
    max_age_seconds: float | None = None,
    priority: int = 1,
) -> dict:
    target = {
        "kind": kind,
        "channel": channel,
        "target_key": f"{kind}:{channel}:{suffix}",
        "context": {},
        "priority": priority,
        "delay_seconds": delay_seconds,
    }
    if max_age_seconds is not None:
        target["max_age_seconds"] = max_age_seconds
    return target


def _create_alert(*, targets=None, alert_id=None, timestamp=None, snapshot=None):
    return alert_store.create_alert(
        camera_id="cam-1",
        camera_name="Camera 1",
        zone="Line 1",
        rule="No Helmet",
        severity="P1",
        confidence=0.9,
        alert_id=alert_id,
        timestamp=timestamp,
        delivery_targets=targets,
        snapshot_jpeg=snapshot,
    )


def _wait_for(predicate, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


@pytest.fixture(autouse=True)
def fresh_outbox():
    alert_store.SNAPSHOTS_DIR = _snapshots
    alert_store.init_db()
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alert_delivery_outbox, alerts")
        conn.commit()
    _snapshots.mkdir(parents=True, exist_ok=True)
    yield
    for path in _snapshots.glob("*"):
        path.unlink(missing_ok=True)


def test_alert_and_delivery_targets_commit_atomically():
    alert = _create_alert(
        targets=[
            _target("telegram", suffix="chat"),
            _target("webhook", suffix="hook"),
        ]
    )

    rows = alert_delivery_store.get_for_alert(alert["id"])

    assert len(rows) == 2
    assert {row["channel"] for row in rows} == {"telegram", "webhook"}
    assert {row["state"] for row in rows} == {"pending"}


def test_outbox_insert_failure_rolls_back_alert(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("forced outbox failure")

    monkeypatch.setattr(alert_delivery_store, "insert_targets", fail)

    with pytest.raises(RuntimeError, match="forced outbox failure"):
        _create_alert(targets=[_target()])

    assert alert_store.get_alerts() == []


def test_idempotent_persistence_replay_does_not_duplicate_alert_or_outbox():
    alert_id = "2fd36a7d-e535-4fdd-94af-651be56923af"
    timestamp = datetime.now(timezone.utc).isoformat()
    first = _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        targets=[_target()],
    )
    replay = _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        targets=[_target()],
    )

    assert first == replay
    assert len(alert_store.get_alerts()) == 1
    assert len(alert_delivery_store.get_for_alert(alert_id)) == 1


def test_idempotent_replay_returns_persisted_payload_and_restores_missing_evidence():
    alert_id = "46dcf08c-3fd4-4fb0-b057-8ffec1c09b39"
    timestamp = datetime.now(timezone.utc).isoformat()
    first = _create_alert(alert_id=alert_id, timestamp=timestamp)

    replay = _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        snapshot=b"late-replay-evidence",
    )

    assert replay == first
    # The original row has no evidence reference, so a late replay file is an
    # orphan and is removed. Referenced missing evidence is covered below.
    assert not (_snapshots / f"{alert_id}.jpg").exists()


def test_idempotent_replay_restores_evidence_referenced_by_persisted_row():
    alert_id = "c1427797-1b78-41d9-8872-365a046ca10f"
    timestamp = datetime.now(timezone.utc).isoformat()
    first = _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        snapshot=b"original-evidence",
    )
    evidence = _snapshots / f"{alert_id}.jpg"
    evidence.unlink()

    replay = _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        snapshot=b"restored-evidence",
    )

    assert replay == first
    assert evidence.read_bytes() == b"restored-evidence"


def test_identity_collision_cannot_overwrite_existing_snapshot():
    alert_id = "535457ce-22be-4ddb-8cb6-371df22e035e"
    timestamp = datetime.now(timezone.utc).isoformat()
    _create_alert(
        alert_id=alert_id,
        timestamp=timestamp,
        snapshot=b"first-evidence",
    )

    with pytest.raises(RuntimeError, match="identity collision"):
        alert_store.create_alert(
            camera_id="different-camera",
            camera_name="Different",
            zone="Other",
            rule="Different rule",
            severity="P2",
            confidence=0.1,
            alert_id=alert_id,
            timestamp=timestamp,
            snapshot_jpeg=b"replacement-evidence",
        )

    assert (_snapshots / f"{alert_id}.jpg").read_bytes() == b"first-evidence"


def test_identity_collision_cannot_restore_missing_evidence_with_wrong_payload():
    alert_id = "6dbc0cfa-cf87-4aa4-b87e-f56c6a804728"
    timestamp = datetime.now(timezone.utc).isoformat()
    _create_alert(alert_id=alert_id, timestamp=timestamp, snapshot=b"original")
    evidence = _snapshots / f"{alert_id}.jpg"
    evidence.unlink()

    with pytest.raises(RuntimeError, match="identity collision"):
        alert_store.create_alert(
            camera_id="different-camera",
            camera_name="Different",
            zone="Other",
            rule="Different rule",
            severity="P2",
            confidence=0.1,
            alert_id=alert_id,
            timestamp=timestamp,
            snapshot_jpeg=b"wrong-evidence",
        )

    assert not evidence.exists()


def test_concurrent_claimers_never_receive_same_live_lease():
    alert = _create_alert(targets=[_target()])
    barrier = threading.Barrier(3)
    claims = []

    def claim(worker):
        barrier.wait()
        claims.append(
            alert_delivery_store.claim_due(worker_id=worker, lease_seconds=30)
        )

    threads = [threading.Thread(target=claim, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 1
    assert claimed[0]["alert_id"] == alert["id"]


def test_expired_lease_is_reclaimed_and_stale_token_is_fenced():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = %s",
                (first["id"],),
            )
        conn.commit()

    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)

    assert second["id"] == first["id"]
    assert second["lease_token"] != first["lease_token"]
    assert not alert_delivery_store.mark_delivered(str(first["id"]), str(first["lease_token"]))
    assert alert_delivery_store.mark_delivered(str(second["id"]), str(second["lease_token"]))
    assert alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "delivered"


def test_claim_recovery_does_not_consume_provider_attempt_budget():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = %s",
                (first["id"],),
            )
        conn.commit()

    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["claim_count"] == 2
    assert row["attempt_count"] == 0

    began = alert_delivery_store.begin_send(
        str(second["id"]), str(second["lease_token"]), max_attempts=5
    )
    assert began["started"] is True
    assert began["attempt_count"] == 1


def test_expired_initial_delivery_is_terminalized_without_provider_send():
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    alert = _create_alert(
        timestamp=old_timestamp,
        targets=[_target(max_age_seconds=1)],
    )

    assert alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30) is None
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["state"] == "terminal"
    assert row["terminal_reason"] == "delivery_expired"
    assert row["attempt_count"] == 0


def test_delayed_escalation_gets_full_lifetime_after_becoming_eligible():
    alert = _create_alert(
        targets=[
            _target(
                "telegram",
                kind="escalation",
                delay_seconds=30 * 60,
                max_age_seconds=60,
            )
        ]
    )

    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert (row["eligible_at"] - row["created_at"]).total_seconds() > 29 * 60
    assert (row["expires_at"] - row["eligible_at"]).total_seconds() == pytest.approx(60)
    assert alert_delivery_store.claim_due(worker_id="early", lease_seconds=30) is None


def test_pending_row_created_before_worker_start_is_delivered(monkeypatch):
    alert = _create_alert(targets=[_target()])
    monkeypatch.setattr(
        notification_dispatcher,
        "deliver_outbox_target",
        lambda *_args: ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered"),
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        poll_seconds=0.02,
        rng=lambda _start, _end: 0,
    )

    worker.start()
    try:
        _wait_for(
            lambda: alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "delivered"
        )
    finally:
        assert worker.stop(timeout=2)


def test_worker_restart_uses_persisted_deadline(monkeypatch):
    timestamp = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    alert = _create_alert(
        targets=[_target(max_age_seconds=60)],
        timestamp=timestamp,
    )
    monkeypatch.setattr(
        notification_dispatcher,
        "deliver_outbox_target",
        lambda *_args: ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered"),
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        poll_seconds=0.02,
    )

    worker.start()
    try:
        _wait_for(
            lambda: alert_delivery_store.get_for_alert(alert["id"])[0]["state"]
            == "delivered"
        )
    finally:
        assert worker.stop(timeout=2)


def test_retry_due_time_survives_worker_recreation(monkeypatch):
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(first["id"]), str(first["lease_token"]), max_attempts=5
    )["started"]
    assert alert_delivery_store.schedule_retry(
        str(first["id"]),
        str(first["lease_token"]),
        delay_seconds=0.12,
        max_attempts=5,
        error_code="http_503",
        error_message="temporary",
        acceptance_unknown=False,
    ) == "pending"
    assert alert_delivery_store.claim_due(worker_id="too-early", lease_seconds=30) is None

    monkeypatch.setattr(
        notification_dispatcher,
        "deliver_outbox_target",
        lambda *_args: ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered"),
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1, poll_seconds=0.02)
    worker.start()
    try:
        _wait_for(
            lambda: alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "delivered"
        )
        assert alert_delivery_store.get_for_alert(alert["id"])[0]["attempt_count"] == 2
    finally:
        assert worker.stop(timeout=2)


def test_lease_renewal_prevents_second_worker_during_slow_provider(monkeypatch):
    alert = _create_alert(targets=[_target()])
    entered = threading.Event()
    release = threading.Event()

    def slow_delivery(*_args):
        entered.set()
        assert release.wait(timeout=2)
        return ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered")

    monkeypatch.setattr(notification_dispatcher, "deliver_outbox_target", slow_delivery)
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        lease_seconds=0.3,
        poll_seconds=0.02,
    )
    worker.start()
    try:
        assert entered.wait(timeout=2)
        time.sleep(0.45)
        assert alert_delivery_store.claim_due(worker_id="competitor", lease_seconds=30) is None
        release.set()
        _wait_for(
            lambda: alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "delivered"
        )
    finally:
        release.set()
        assert worker.stop(timeout=2)


def test_huge_retry_after_is_terminal_without_killing_worker(monkeypatch):
    alert = _create_alert(targets=[_target()])
    monkeypatch.setattr(
        notification_dispatcher,
        "deliver_outbox_target",
        lambda *_args: ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Provider asked us to wait",
            error_code="rate_limited",
            retry_after_seconds=1e308,
        ),
    )
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        poll_seconds=0.02,
    )
    worker.start()
    try:
        row = _wait_for(
            lambda: (
                current
                if (current := alert_delivery_store.get_for_alert(alert["id"])[0])["state"]
                == "terminal"
                else None
            )
        )
        assert row["last_error_code"] == "rate_limited"
        assert row["terminal_reason"] == "retry_deadline_exceeds_lifetime"
        assert worker.stats()["workers_alive"] == 1
    finally:
        assert worker.stop(timeout=2)


def test_cancelled_escalation_is_never_claimed():
    alert = _create_alert(
        targets=[_target("telegram", kind="escalation", delay_seconds=0)]
    )

    assert alert_delivery_store.cancel_escalations(alert["id"]) == 1
    assert alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30) is None
    assert alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "cancelled"


def test_acknowledgement_cancels_even_a_leased_escalation_before_send():
    alert = _create_alert(
        targets=[_target("telegram", kind="escalation", delay_seconds=0)]
    )
    claimed = alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30)

    alert_store.acknowledge_alert(alert["id"])
    began = alert_delivery_store.begin_send(
        str(claimed["id"]), str(claimed["lease_token"]), max_attempts=5
    )

    assert began == {"started": False, "reason": "lease_lost"}
    assert alert_delivery_store.get_for_alert(alert["id"])[0]["state"] == "cancelled"


def test_cancelling_an_in_flight_escalation_preserves_ambiguity_history():
    alert = _create_alert(
        targets=[_target("telegram", kind="escalation", delay_seconds=0)]
    )
    claimed = alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(claimed["id"]), str(claimed["lease_token"]), max_attempts=5
    )["started"]

    alert_store.acknowledge_alert(alert["id"])

    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["state"] == "cancelled"
    assert row["last_error_code"] == "cancelled_in_flight"
    assert row["ever_acceptance_unknown"] is True
    assert row["send_in_flight"] is False


def test_ambiguity_history_survives_later_terminal_outcome():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(first["id"]), str(first["lease_token"]), max_attempts=5
    )["started"]
    assert alert_delivery_store.schedule_retry(
        str(first["id"]),
        str(first["lease_token"]),
        delay_seconds=0,
        max_attempts=5,
        error_code="read_timeout",
        error_message="acceptance unknown",
        acceptance_unknown=True,
    ) == "pending"

    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(second["id"]), str(second["lease_token"]), max_attempts=5
    )["started"]
    assert alert_delivery_store.mark_terminal(
        str(second["id"]),
        str(second["lease_token"]),
        error_code="invalid_configuration",
        error_message="configuration changed",
    )

    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["last_acceptance_unknown"] is False
    assert row["ever_acceptance_unknown"] is True
    assert row["send_in_flight"] is False


def test_known_retry_clears_in_flight_without_inventing_ambiguity():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(first["id"]), str(first["lease_token"]), max_attempts=5
    )["started"]
    assert alert_delivery_store.schedule_retry(
        str(first["id"]),
        str(first["lease_token"]),
        delay_seconds=0,
        max_attempts=5,
        error_code="rate_limited",
        error_message="retry later",
        acceptance_unknown=False,
    ) == "pending"

    between = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert between["send_in_flight"] is False
    assert between["ever_acceptance_unknown"] is False

    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)
    reclaimed = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert reclaimed["last_error_code"] == "rate_limited"
    assert reclaimed["ever_acceptance_unknown"] is False
    assert alert_delivery_store.mark_delivered(
        str(second["id"]), str(second["lease_token"])
    )
    assert alert_delivery_store.get_for_alert(alert["id"])[0]["send_in_flight"] is False


def test_crash_after_provider_start_is_reclaimed_as_ambiguous():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    assert alert_delivery_store.begin_send(
        str(first["id"]), str(first["lease_token"]), max_attempts=5
    )["started"]
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox "
                "SET lease_expires_at = clock_timestamp() - interval '1 second' "
                "WHERE id = %s",
                (str(first["id"]),),
            )
        conn.commit()

    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)

    assert second is not None
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["last_error_code"] == "interrupted_in_flight_attempt"
    assert row["last_acceptance_unknown"] is True
    assert row["ever_acceptance_unknown"] is True
    assert row["send_in_flight"] is False


def test_begin_send_refuses_an_expired_lease_without_consuming_attempt():
    alert = _create_alert(targets=[_target()])
    claimed = alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30)
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox "
                "SET lease_expires_at = clock_timestamp() - interval '1 second' "
                "WHERE id = %s",
                (str(claimed["id"]),),
            )
        conn.commit()

    result = alert_delivery_store.begin_send(
        str(claimed["id"]), str(claimed["lease_token"]), max_attempts=5
    )

    assert result == {"started": False, "reason": "lease_expired"}
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["attempt_count"] == 0
    assert row["lease_token"] is None
    assert row["send_in_flight"] is False


def test_internal_failures_are_bounded_without_provider_attempts():
    alert = _create_alert(targets=[_target()])
    first = alert_delivery_store.claim_due(worker_id="first", lease_seconds=30)
    assert alert_delivery_store.schedule_internal_retry(
        str(first["id"]),
        str(first["lease_token"]),
        delay_seconds=0,
        max_claims=2,
    ) == "pending"
    second = alert_delivery_store.claim_due(worker_id="second", lease_seconds=30)
    assert alert_delivery_store.schedule_internal_retry(
        str(second["id"]),
        str(second["lease_token"]),
        delay_seconds=0,
        max_claims=2,
    ) == "terminal"

    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["attempt_count"] == 0
    assert row["claim_count"] == 2
    assert row["internal_failure_count"] == 2
    assert row["terminal_reason"] == "worker_failures_exhausted"


def test_internal_failure_budget_is_independent_of_normal_provider_retries():
    alert = _create_alert(targets=[_target()])
    for index in range(4):
        claimed = alert_delivery_store.claim_due(
            worker_id=f"provider-{index}", lease_seconds=30
        )
        assert alert_delivery_store.begin_send(
            str(claimed["id"]), str(claimed["lease_token"]), max_attempts=10
        )["started"]
        assert alert_delivery_store.schedule_retry(
            str(claimed["id"]),
            str(claimed["lease_token"]),
            delay_seconds=0,
            max_attempts=10,
            error_code="rate_limited",
            error_message="retry later",
            acceptance_unknown=False,
        ) == "pending"

    first_internal = alert_delivery_store.claim_due(
        worker_id="internal-1", lease_seconds=30
    )
    assert alert_delivery_store.schedule_internal_retry(
        str(first_internal["id"]),
        str(first_internal["lease_token"]),
        delay_seconds=0,
        max_claims=2,
    ) == "pending"
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["claim_count"] == 5
    assert row["attempt_count"] == 4
    assert row["internal_failure_count"] == 1

    second_internal = alert_delivery_store.claim_due(
        worker_id="internal-2", lease_seconds=30
    )
    assert alert_delivery_store.schedule_internal_retry(
        str(second_internal["id"]),
        str(second_internal["lease_token"]),
        delay_seconds=0,
        max_claims=2,
    ) == "terminal"


def test_repeated_graceful_shutdown_releases_do_not_consume_failure_budget():
    alert = _create_alert(targets=[_target()])
    for index in range(8):
        claimed = alert_delivery_store.claim_due(
            worker_id=f"shutdown-{index}", lease_seconds=30
        )
        assert alert_delivery_store.release_unstarted_claim(
            str(claimed["id"]),
            str(claimed["lease_token"]),
        )

    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["state"] == "pending"
    assert row["claim_count"] == 8
    assert row["internal_failure_count"] == 0
    assert row["attempt_count"] == 0


def test_terminal_replay_is_age_bounded_and_requires_ambiguity_confirmation():
    alert = _create_alert(targets=[_target()])
    claimed = alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30)
    assert alert_delivery_store.mark_terminal(
        str(claimed["id"]),
        str(claimed["lease_token"]),
        error_code="request_timeout",
        error_message="unknown acceptance",
        acceptance_unknown=True,
    )
    delivery_id = str(claimed["id"])

    assert not alert_delivery_store.requeue_terminal(delivery_id)
    assert alert_delivery_store.requeue_terminal(delivery_id, allow_ambiguous=True)
    row = alert_delivery_store.get_for_alert(alert["id"])[0]
    assert row["state"] == "pending"
    assert row["attempt_count"] == 0
    assert row["ever_acceptance_unknown"] is True


def test_retention_never_deletes_pending_work():
    pending_alert = _create_alert(targets=[_target(suffix="pending")])
    completed_alert = _create_alert(targets=[_target(suffix="done")])
    completed = alert_delivery_store.claim_due(worker_id="worker", lease_seconds=30)
    assert completed is not None
    # The earliest row may be either alert; finish the leased one and retain the other.
    assert alert_delivery_store.mark_delivered(
        str(completed["id"]), str(completed["lease_token"])
    )
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox SET updated_at = clock_timestamp() - interval '3 days' WHERE state = 'delivered'"
            )
        conn.commit()

    assert alert_delivery_store.cleanup_completed(delivered_days=1) == 1
    remaining = (
        alert_delivery_store.get_for_alert(pending_alert["id"])
        + alert_delivery_store.get_for_alert(completed_alert["id"])
    )
    assert len(remaining) == 1
    assert remaining[0]["state"] == "pending"


def test_stats_separate_due_work_from_future_escalations():
    _create_alert(
        targets=[
            _target(suffix="due"),
            _target("telegram", suffix="future", kind="escalation", delay_seconds=600),
        ]
    )

    stats = alert_delivery_store.get_stats()
    assert stats["pending"] == 2
    assert stats["due"] == 1
    assert stats["scheduled"] == 1
    assert stats["oldest_due_age_seconds"] is not None


def test_due_queue_prefers_new_critical_work_over_older_low_priority_work():
    low = _create_alert(targets=[_target(suffix="low", priority=4)])
    critical = _create_alert(targets=[_target(suffix="critical", priority=1)])
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox "
                "SET next_attempt_at = clock_timestamp() - interval '30 seconds' "
                "WHERE alert_id = %s",
                (low["id"],),
            )
        conn.commit()

    claimed = alert_delivery_store.claim_due(worker_id="priority", lease_seconds=30)

    assert claimed["alert_id"] == critical["id"]


def test_due_queue_ages_low_priority_work_to_prevent_starvation():
    low = _create_alert(targets=[_target(suffix="low", priority=4)])
    _create_alert(targets=[_target(suffix="newer-p2", priority=2)])
    with alert_store._get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alert_delivery_outbox "
                "SET next_attempt_at = clock_timestamp() - interval '4 minutes' "
                "WHERE alert_id = %s",
                (low["id"],),
            )
        conn.commit()

    claimed = alert_delivery_store.claim_due(worker_id="aging", lease_seconds=30)

    assert claimed["alert_id"] == low["id"]


def test_empty_delivery_routing_creates_no_outbox_rows():
    alert = _create_alert(targets=[])
    assert alert_delivery_store.get_for_alert(alert["id"]) == []


def test_upgrade_backfills_only_active_escalations_idempotently():
    alert = _create_alert(targets=[])
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {"P1": {"telegram": True}},
            "escalation_steps": [
                {"id": 1, "afterMinutes": 30, "role": "Manager", "channel": "telegram"}
            ],
        },
    }

    assert alert_delivery_store.backfill_active_escalations(
        cfg, notification_dispatcher.resolve_delivery_targets
    ) == 1
    assert alert_delivery_store.backfill_active_escalations(
        cfg, notification_dispatcher.resolve_delivery_targets
    ) == 0

    rows = alert_delivery_store.get_for_alert(alert["id"])
    assert len(rows) == 1
    assert rows[0]["kind"] == "escalation"


def test_upgrade_backfill_skips_already_due_legacy_escalations():
    old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    alert = _create_alert(targets=[], timestamp=old_timestamp)
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {"id": 1, "afterMinutes": 1, "role": "Manager", "channel": "telegram"}
            ],
        },
    }

    assert alert_delivery_store.backfill_active_escalations(
        cfg, notification_dispatcher.resolve_delivery_targets
    ) == 0
    assert alert_delivery_store.get_for_alert(alert["id"]) == []


def test_upgrade_backfill_uses_keyset_batches():
    alerts = [_create_alert(targets=[]) for _ in range(3)]
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {"id": 1, "afterMinutes": 30, "role": "Manager", "channel": "telegram"}
            ],
        },
    }

    first_count, cursor = alert_delivery_store.backfill_active_escalations_batch(
        cfg,
        notification_dispatcher.resolve_delivery_targets,
        batch_size=2,
    )
    second_count, final_cursor = alert_delivery_store.backfill_active_escalations_batch(
        cfg,
        notification_dispatcher.resolve_delivery_targets,
        after_id=cursor,
        batch_size=2,
    )

    assert first_count == 2
    assert cursor is not None
    assert second_count == 1
    assert final_cursor is None
    assert sum(len(alert_delivery_store.get_for_alert(alert["id"])) for alert in alerts) == 3


def test_upgrade_backfill_excludes_rows_older_than_reconciliation_cutoff():
    stale_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    stale = _create_alert(targets=[], timestamp=stale_timestamp)
    recent = _create_alert(targets=[])
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {
                    "id": 99,
                    "afterMinutes": 3 * 24 * 60,
                    "role": "Manager",
                    "channel": "telegram",
                }
            ],
        },
    }

    inserted, cursor = alert_delivery_store.backfill_active_escalations_batch(
        cfg,
        notification_dispatcher.resolve_delivery_targets,
        minimum_timestamp=cutoff,
    )

    assert inserted == 1
    assert cursor is None
    assert alert_delivery_store.get_for_alert(stale["id"]) == []
    assert len(alert_delivery_store.get_for_alert(recent["id"])) == 1


def test_resolver_creates_recipient_granular_initial_and_escalation_rows():
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1"],
        },
        "email": {
            "enabled": True,
            "smtp_host": "smtp.example",
            "from_address": "alerts@example.com",
            "to_addresses": ["one@example.com", "two@example.com"],
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {"P1": {"inApp": True, "telegram": True, "email": True}},
            "escalation_steps": [
                {"id": 7, "afterMinutes": 3, "role": "Manager", "channel": "email"},
            ],
        },
    }

    targets = notification_dispatcher.resolve_delivery_targets(
        cfg,
        {"id": "alert", "severity": "P1"},
    )

    initial = [target for target in targets if target["kind"] == "initial"]
    escalations = [target for target in targets if target["kind"] == "escalation"]
    assert [target["channel"] for target in initial].count("email") == 2
    assert [target["channel"] for target in initial].count("telegram") == 1
    assert len(escalations) == 2
    assert {target["delay_seconds"] for target in escalations} == {180}


def test_resolver_preserves_configured_low_priority_escalations():
    cfg = {
        "telegram": {
            "enabled": True,
            "chat_id": "chat-1",
            "bot_token": "token",
            "severities": ["P1", "P2", "P3", "P4"],
        },
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {"id": 7, "afterMinutes": 3, "role": "Manager", "channel": "telegram"},
            ],
        },
    }

    targets = notification_dispatcher.resolve_delivery_targets(
        cfg,
        {"id": "alert", "severity": "P4"},
    )

    assert len(targets) == 1
    assert targets[0]["kind"] == "escalation"


def test_resolver_skips_disabled_defaults_and_malformed_steps():
    cfg = {
        "telegram": {"enabled": False},
        "email": {"enabled": False},
        "alert_routing": {
            "channel_matrix": {"P1": "not-a-map"},
            "escalation_steps": [
                None,
                "bad",
                {
                    "id": 1,
                    "afterMinutes": 3,
                    "role": "Floor Manager",
                    "channel": "telegram",
                },
                {
                    "id": 2,
                    "afterMinutes": 10,
                    "role": "Plant Manager",
                    "channel": "email",
                },
            ],
        },
    }

    assert notification_dispatcher.resolve_delivery_targets(
        cfg, {"id": "alert", "severity": "P1"}
    ) == []


def test_resolver_keeps_unmarked_seed_examples_dormant_with_ready_providers():
    cfg = {
        "telegram": {
            "enabled": True,
            "bot_token": "123456:token",
            "chat_id": "chat-1",
            "severities": ["P1"],
        },
        "email": {
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "alerts@example.com",
            "to_addresses": ["safety@example.com"],
            "severities": ["P1"],
        },
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {
                    "id": 1,
                    "afterMinutes": 3,
                    "role": "Floor Manager",
                    "channel": "telegram",
                },
                {
                    "id": 2,
                    "afterMinutes": 10,
                    "role": "Plant Manager",
                    "channel": "email",
                },
            ],
        },
    }

    assert notification_dispatcher.resolve_delivery_targets(
        cfg,
        {"id": "alert", "severity": "P1"},
    ) == []


def test_explicit_invalid_escalation_becomes_visible_terminal_obligation(monkeypatch):
    cfg = {
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
        "alert_routing": {
            "channel_matrix": {},
            "escalation_steps": [
                {
                    "id": 9,
                    "enabled": True,
                    "afterMinutes": 1,
                    "role": "Safety Manager",
                    "channel": "telegram",
                    "severities": ["P1"],
                }
            ],
        },
    }

    targets = notification_dispatcher.resolve_delivery_targets(
        cfg, {"id": "alert", "severity": "P1"}
    )

    assert len(targets) == 1
    assert targets[0]["kind"] == "escalation"
    assert targets[0]["target_key"].endswith(":unconfigured")
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: cfg,
    )
    result = notification_dispatcher.deliver_outbox_target(
        {"id": "alert", "severity": "P1"},
        {"id": "delivery", **targets[0]},
        None,
    )
    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "channel_inactive"


def test_stale_attempt_cannot_remove_newer_active_lease():
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=1)
    worker._active_leases = {("delivery", "old-token"), ("delivery", "new-token")}
    worker._channel_inflight = {"webhook": 2}

    worker._complete_active(
        {"id": "delivery", "lease_token": "old-token", "channel": "webhook"}
    )

    assert worker._active_leases == {("delivery", "new-token")}
    assert worker._channel_inflight == {"webhook": 1}


def test_database_outage_uses_backoff_instead_of_hot_spinning(monkeypatch):
    attempts = 0

    def fail_claim(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(alert_delivery_store, "claim_due", fail_claim)
    worker = alert_delivery_worker.AlertDeliveryWorker(
        worker_count=1,
        poll_seconds=0.01,
        rng=lambda start, _end: start,
    )
    worker.start()
    try:
        time.sleep(0.16)
    finally:
        assert worker.stop(timeout=2)

    assert 1 <= attempts <= 3
    assert worker.stats()["claim_errors"] == attempts


def test_blocked_webhook_does_not_starve_telegram(monkeypatch):
    alert = _create_alert(
        targets=[_target("webhook", suffix="blocked"), _target("telegram", suffix="fast")]
    )
    webhook_entered = threading.Event()
    release_webhook = threading.Event()

    def deliver(_alert, target, _snapshot):
        if target["channel"] == "webhook":
            webhook_entered.set()
            assert release_webhook.wait(timeout=3)
        return ProviderDeliveryResult(DeliveryDisposition.DELIVERED, "Delivered")

    monkeypatch.setattr(notification_dispatcher, "deliver_outbox_target", deliver)
    worker = alert_delivery_worker.AlertDeliveryWorker(worker_count=3, poll_seconds=0.02)
    worker.start()
    try:
        assert webhook_entered.wait(timeout=2)
        _wait_for(
            lambda: any(
                row["channel"] == "telegram" and row["state"] == "delivered"
                for row in alert_delivery_store.get_for_alert(alert["id"])
            )
        )
    finally:
        release_webhook.set()
        assert worker.stop(timeout=3)
