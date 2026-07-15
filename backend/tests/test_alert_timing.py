from __future__ import annotations

import pytest

from alert_timing import AlertTimingRegistry


def test_registry_returns_context_once() -> None:
    registry = AlertTimingRegistry(maximum_pending=2)
    registry.remember(
        "alert-1",
        first_positive_ns=10,
        confirmed_ns=25,
    )

    context = registry.pop("alert-1")

    assert context is not None
    assert context.first_positive_ns == 10
    assert context.confirmed_ns == 25
    assert registry.pop("alert-1") is None


def test_registry_evicts_oldest_pending_context() -> None:
    registry = AlertTimingRegistry(maximum_pending=2)
    registry.remember("alert-1", first_positive_ns=1, confirmed_ns=2)
    registry.remember("alert-2", first_positive_ns=3, confirmed_ns=4)
    registry.remember("alert-3", first_positive_ns=5, confirmed_ns=6)

    assert registry.pop("alert-1") is None
    assert registry.stats() == {
        "pending": 2,
        "awaitingPersistence": 2,
        "awaitingDelivery": 0,
        "pendingDeliveryTargets": 0,
        "capacity": 2,
        "evicted": 1,
    }
    assert registry.pop("alert-2") is not None
    assert registry.pop("alert-3") is not None


def test_replacing_same_alert_id_does_not_consume_capacity() -> None:
    registry = AlertTimingRegistry(maximum_pending=1)
    registry.remember("alert-1", first_positive_ns=1, confirmed_ns=2)
    registry.remember("alert-1", first_positive_ns=3, confirmed_ns=4)

    assert registry.stats() == {
        "pending": 1,
        "awaitingPersistence": 1,
        "awaitingDelivery": 0,
        "pendingDeliveryTargets": 0,
        "capacity": 1,
        "evicted": 0,
    }
    assert registry.pop("alert-1").confirmed_ns == 4


@pytest.mark.parametrize(
    ("first_positive_ns", "confirmed_ns"),
    [(-1, 2), (2, -1), (3, 2)],
)
def test_registry_rejects_invalid_timing_order(
    first_positive_ns: int,
    confirmed_ns: int,
) -> None:
    registry = AlertTimingRegistry()

    with pytest.raises(ValueError):
        registry.remember(
            "alert-1",
            first_positive_ns=first_positive_ns,
            confirmed_ns=confirmed_ns,
        )


def test_registry_rejects_empty_id_and_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        AlertTimingRegistry(maximum_pending=0)

    registry = AlertTimingRegistry()
    with pytest.raises(ValueError):
        registry.remember("", first_positive_ns=1, confirmed_ns=2)


def test_registry_retains_monotonic_context_until_each_initial_target_finishes() -> None:
    registry = AlertTimingRegistry(maximum_pending=2)
    registry.remember(
        "alert-1",
        first_positive_ns=10,
        confirmed_ns=20,
        initial_target_keys=("webhook-a", "telegram-b"),
    )

    transition = registry.mark_persisted("alert-1", persisted_ns=30)

    assert transition is not None
    assert transition.newly_persisted is True
    assert transition.context.persisted_ns == 30
    assert registry.stats()["pendingDeliveryTargets"] == 2
    assert registry.activate_delivery_tracking("alert-1") == ()
    first_outcome = registry.complete_initial_target("alert-1", "webhook-a")
    assert first_outcome is not None
    assert first_outcome.first_positive_ns == transition.context.first_positive_ns
    assert first_outcome.delivery_tracking_active is True
    assert registry.tracks_initial_target("alert-1", "telegram-b") is True
    assert registry.complete_initial_target("alert-1", "telegram-b") is not None
    assert registry.stats()["pending"] == 0


def test_registry_rejects_reversed_monotonic_persistence_anchor() -> None:
    registry = AlertTimingRegistry()
    registry.remember("alert-1", first_positive_ns=10, confirmed_ns=20)

    with pytest.raises(ValueError, match="persisted anchor"):
        registry.mark_persisted("alert-1", persisted_ns=19)


def test_outcome_between_persist_anchor_and_denominator_activation_is_deferred() -> None:
    registry = AlertTimingRegistry()
    registry.remember(
        "alert-1",
        first_positive_ns=10,
        confirmed_ns=20,
        initial_target_keys=("target",),
    )
    assert registry.mark_persisted("alert-1", persisted_ns=30)

    outcome = registry.record_initial_outcome(
        "alert-1",
        "target",
        outcome="delivered",
        completed_ns=40,
    )

    assert outcome is not None and outcome.deferred is True
    deferred = registry.activate_delivery_tracking("alert-1")
    assert len(deferred) == 1
    assert deferred[0].completed_ns == 40
    assert registry.stats()["pending"] == 0
