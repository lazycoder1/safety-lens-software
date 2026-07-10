"""Focused health checks for the durable alert-delivery outbox."""

import pytest

from alert_pipeline import AlertPipeline
from routers import misc


def _snapshot(status="ok"):
    return {"status": status, "reasons": []}


def _healthy_outbox(**overrides):
    outbox = {
        "database_error": False,
        "running": True,
        "claimer_alive": True,
        "workers_alive": 3,
        "renewer_alive": True,
        "due": 0,
        "oldest_due_age_seconds": None,
    }
    outbox.update(overrides)
    return {"outbox": outbox}


def _healthy_persistence(**overrides):
    stats = {
        "running": True,
        "accepting": True,
        "persist_worker_alive": True,
        "consecutive_persistence_failures": 0,
        "last_persistence_failure_at": None,
        "last_persistence_success_at": None,
        "oldest_persistence_age_seconds": None,
    }
    stats.update(overrides)
    return stats


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"running": False},
            "alert persistence pipeline is not running",
        ),
        (
            {"accepting": False},
            "alert persistence pipeline is not accepting alerts",
        ),
        (
            {"persist_worker_alive": False},
            "alert persistence worker is not running",
        ),
    ],
)
def test_persistence_failures_degrade_health_even_with_healthy_outbox(
    overrides,
    reason,
):
    snapshot = _snapshot()
    pipeline_stats = _healthy_persistence(**overrides)
    pipeline_stats.update(_healthy_outbox())

    misc._apply_persistence_health(snapshot, pipeline_stats)
    misc._apply_outbox_health(snapshot, pipeline_stats)

    assert snapshot["status"] == "degraded"
    assert reason in snapshot["reasons"]


def test_healthy_persistence_pipeline_keeps_health_ok():
    snapshot = _snapshot()

    misc._apply_persistence_health(snapshot, _healthy_persistence())

    assert snapshot == {"status": "ok", "reasons": []}


def test_unresolved_persistence_failure_degrades_live_pipeline():
    snapshot = _snapshot()

    misc._apply_persistence_health(
        snapshot,
        _healthy_persistence(
            consecutive_persistence_failures=2,
            last_persistence_failure_at=123.0,
        ),
    )

    assert snapshot == {
        "status": "degraded",
        "reasons": ["alert persistence has unresolved failures"],
    }


def test_later_persistence_success_clears_failure_health():
    snapshot = _snapshot()

    misc._apply_persistence_health(
        snapshot,
        _healthy_persistence(
            consecutive_persistence_failures=0,
            last_persistence_failure_at=123.0,
            last_persistence_success_at=124.0,
        ),
    )

    assert snapshot == {"status": "ok", "reasons": []}


def test_stale_persistence_work_degrades_live_worker(monkeypatch):
    monkeypatch.setenv("ALERT_PERSISTENCE_STALE_SECONDS", "10")
    snapshot = _snapshot()

    misc._apply_persistence_health(
        snapshot,
        _healthy_persistence(oldest_persistence_age_seconds=10.1),
    )

    assert snapshot == {
        "status": "degraded",
        "reasons": ["alert persistence work is stale"],
    }


def test_no_persistence_attempt_is_not_mistaken_for_failure(monkeypatch):
    monkeypatch.setenv("ALERT_PERSISTENCE_STALE_SECONDS", "0.05")
    snapshot = _snapshot()

    misc._apply_persistence_health(snapshot, _healthy_persistence())

    assert snapshot == {"status": "ok", "reasons": []}


def test_live_pipeline_health_degrades_on_failure_and_recovers_on_success():
    database_available = False

    def persist(**payload):
        if not database_available:
            raise RuntimeError("database unavailable")
        return {"id": str(payload["sequence"]), **payload}

    pipeline = AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persistence_attempts=1,
    )
    try:
        with pytest.raises(RuntimeError, match="database unavailable"):
            pipeline.submit({"sequence": 1}).result(timeout=1)

        failed_snapshot = _snapshot()
        misc._apply_persistence_health(failed_snapshot, pipeline.stats())
        assert failed_snapshot == {
            "status": "degraded",
            "reasons": ["alert persistence has unresolved failures"],
        }

        database_available = True
        assert pipeline.submit({"sequence": 2}).result(timeout=1)["sequence"] == 2

        recovered_snapshot = _snapshot()
        misc._apply_persistence_health(recovered_snapshot, pipeline.stats())
        assert recovered_snapshot == {"status": "ok", "reasons": []}
    finally:
        assert pipeline.shutdown(timeout=1)


def test_missing_persistence_stats_degrades_health():
    snapshot = _snapshot()

    misc._apply_persistence_health(snapshot, None)

    assert snapshot == {
        "status": "degraded",
        "reasons": ["alert persistence pipeline statistics unavailable"],
    }


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"database_error": True},
            "alert delivery outbox statistics unavailable",
        ),
        (
            {"running": False},
            "alert delivery outbox is not running",
        ),
        (
            {"claimer_alive": False},
            "alert delivery outbox claimer is not running",
        ),
        (
            {"workers_alive": 0},
            "alert delivery provider workers are not running",
        ),
        (
            {"renewer_alive": False},
            "alert delivery lease renewer is not running",
        ),
    ],
)
def test_outbox_failures_degrade_health(overrides, reason):
    snapshot = _snapshot()

    misc._apply_outbox_health(snapshot, _healthy_outbox(**overrides))

    assert snapshot["status"] == "degraded"
    assert reason in snapshot["reasons"]


def test_stale_due_backlog_degrades_health(monkeypatch):
    monkeypatch.setenv("ALERT_OUTBOX_POLL_SECONDS", "5")
    snapshot = _snapshot()

    misc._apply_outbox_health(
        snapshot,
        _healthy_outbox(due=2, oldest_due_age_seconds=31),
    )

    assert snapshot == {
        "status": "degraded",
        "reasons": ["alert delivery due backlog is stale"],
    }


def test_recent_due_work_and_healthy_workers_keep_health_ok(monkeypatch):
    monkeypatch.setenv("ALERT_OUTBOX_POLL_SECONDS", "20")
    snapshot = _snapshot()

    misc._apply_outbox_health(
        snapshot,
        _healthy_outbox(due=1, oldest_due_age_seconds=39.9),
    )

    assert snapshot == {"status": "ok", "reasons": []}


def test_outbox_degradation_does_not_mask_existing_error():
    snapshot = _snapshot(status="error")

    misc._apply_outbox_health(snapshot, _healthy_outbox(running=False))

    assert snapshot["status"] == "error"
    assert snapshot["reasons"] == ["alert delivery outbox is not running"]


def test_missing_outbox_stats_degrades_health():
    snapshot = _snapshot()

    misc._apply_outbox_health(snapshot, {})

    assert snapshot == {
        "status": "degraded",
        "reasons": ["alert delivery outbox statistics unavailable"],
    }


def test_outbox_reason_does_not_mutate_cached_base_reasons():
    cached_reasons = ["camera offline"]
    snapshot = {"status": "degraded", "reasons": cached_reasons}

    misc._apply_outbox_health(snapshot, _healthy_outbox(running=False))

    assert cached_reasons == ["camera offline"]
    assert snapshot["reasons"] == [
        "camera offline",
        "alert delivery outbox is not running",
    ]
