"""Bounded process-local timing anchors for durable alert telemetry.

UTC timestamps travel with the alert into PostgreSQL.  Monotonic timestamps
stay in this registry so wall-clock adjustments cannot corrupt the real-time
latency histograms while persistence happens on another thread.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class AlertTimingContext:
    first_positive_ns: int
    confirmed_ns: int
    initial_target_keys: frozenset[str] = frozenset()
    remaining_initial_target_keys: frozenset[str] = frozenset()
    persisted_ns: int | None = None
    delivery_tracking_active: bool = False
    deferred_initial_outcomes: tuple["InitialOutcomeTransition", ...] = ()

    def __post_init__(self) -> None:
        if self.first_positive_ns < 0 or self.confirmed_ns < 0:
            raise ValueError("alert timing anchors must be non-negative")
        if self.confirmed_ns < self.first_positive_ns:
            raise ValueError("confirmed anchor cannot precede first positive")
        if self.persisted_ns is not None:
            if self.persisted_ns < self.confirmed_ns:
                raise ValueError("persisted anchor cannot precede confirmation")
        elif self.delivery_tracking_active:
            raise ValueError("delivery tracking requires a persistence anchor")
        if not self.remaining_initial_target_keys.issubset(self.initial_target_keys):
            raise ValueError("remaining targets must belong to the initial target set")
        deferred_keys = {outcome.target_key for outcome in self.deferred_initial_outcomes}
        if len(deferred_keys) != len(self.deferred_initial_outcomes):
            raise ValueError("a target cannot have multiple deferred outcomes")
        if not deferred_keys.issubset(self.initial_target_keys):
            raise ValueError("deferred targets must belong to the initial target set")
        if deferred_keys & self.remaining_initial_target_keys:
            raise ValueError("deferred targets cannot remain pending")


@dataclass(frozen=True)
class InitialOutcomeTransition:
    target_key: str
    outcome: str
    completed_ns: int
    context: AlertTimingContext | None = None
    deferred: bool = False


@dataclass(frozen=True)
class PersistenceTransition:
    context: AlertTimingContext
    newly_persisted: bool


def _target_keys(values: Iterable[object]) -> frozenset[str]:
    keys: set[str] = set()
    for value in values:
        key = str(value).strip()
        if key:
            keys.add(key)
    return frozenset(keys)


class AlertTimingRegistry:
    """Keep only timing contexts for alerts awaiting persistence callbacks."""

    def __init__(self, maximum_pending: int = 4_096) -> None:
        if isinstance(maximum_pending, bool) or maximum_pending < 1:
            raise ValueError("maximum_pending must be a positive integer")
        self._maximum_pending = int(maximum_pending)
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, AlertTimingContext] = OrderedDict()
        self._evicted = 0

    def remember(
        self,
        alert_id: str,
        *,
        first_positive_ns: int,
        confirmed_ns: int,
        initial_target_keys: Iterable[object] = (),
    ) -> tuple[AlertTimingContext, ...]:
        key = str(alert_id).strip()
        if not key:
            raise ValueError("alert_id cannot be empty")
        targets = _target_keys(initial_target_keys)
        context = AlertTimingContext(
            first_positive_ns=int(first_positive_ns),
            confirmed_ns=int(confirmed_ns),
            initial_target_keys=targets,
            remaining_initial_target_keys=targets,
        )
        evicted: list[AlertTimingContext] = []
        with self._lock:
            replaced = self._pending.pop(key, None)
            if replaced is not None and replaced.persisted_ns is not None:
                evicted.append(replaced)
            self._pending[key] = context
            while len(self._pending) > self._maximum_pending:
                _evicted_id, evicted_context = self._pending.popitem(last=False)
                evicted.append(evicted_context)
                self._evicted += 1
        return tuple(evicted)

    def mark_persisted(
        self,
        alert_id: str,
        *,
        persisted_ns: int,
    ) -> PersistenceTransition | None:
        """Attach the app-worker persistence clock without consulting DB time.

        Contexts without provider targets are removed after this transition;
        contexts with targets remain until every initial target reaches a final
        provider outcome so delivery timing stays in the same monotonic domain.
        """

        if isinstance(persisted_ns, bool):
            raise ValueError("persisted_ns must be a non-negative integer")
        try:
            parsed = int(persisted_ns)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("persisted_ns must be a non-negative integer") from exc
        if parsed != persisted_ns or parsed < 0:
            raise ValueError("persisted_ns must be a non-negative integer")
        key = str(alert_id)
        with self._lock:
            context = self._pending.get(key)
            if context is None:
                return None
            if context.persisted_ns is not None:
                return PersistenceTransition(context=context, newly_persisted=False)
            persisted = replace(context, persisted_ns=parsed)
            if (
                persisted.remaining_initial_target_keys
                or persisted.deferred_initial_outcomes
            ):
                self._pending[key] = persisted
            else:
                self._pending.pop(key, None)
            return PersistenceTransition(context=persisted, newly_persisted=True)

    def activate_delivery_tracking(
        self,
        alert_id: str,
    ) -> tuple[InitialOutcomeTransition, ...]:
        """Open the outcome fence after telemetry registers its denominator."""

        alert_key = str(alert_id)
        with self._lock:
            context = self._pending.get(alert_key)
            if (
                context is None
                or context.persisted_ns is None
                or context.delivery_tracking_active
            ):
                return ()
            deferred = context.deferred_initial_outcomes
            activated = replace(
                context,
                delivery_tracking_active=True,
                deferred_initial_outcomes=(),
            )
            if activated.remaining_initial_target_keys:
                self._pending[alert_key] = activated
            else:
                self._pending.pop(alert_key, None)
            return deferred

    def tracks_initial_target(self, alert_id: str, target_key: object) -> bool:
        key = str(target_key).strip()
        if not key:
            return False
        with self._lock:
            context = self._pending.get(str(alert_id))
            return bool(
                context is not None
                and key in context.remaining_initial_target_keys
            )

    def record_initial_outcome(
        self,
        alert_id: str,
        target_key: object,
        *,
        outcome: str,
        completed_ns: int,
    ) -> InitialOutcomeTransition | None:
        """Consume or defer one target outcome across the persistence race."""

        target = str(target_key).strip()
        if not target or outcome not in {"delivered", "terminal", "cancelled"}:
            return None
        if isinstance(completed_ns, bool):
            return None
        try:
            completed = int(completed_ns)
        except (TypeError, ValueError, OverflowError):
            return None
        if completed != completed_ns or completed < 0:
            return None
        alert_key = str(alert_id)
        with self._lock:
            context = self._pending.get(alert_key)
            if (
                context is None
                or target not in context.remaining_initial_target_keys
            ):
                return None
            remaining = context.remaining_initial_target_keys - {target}
            transition = InitialOutcomeTransition(
                target_key=target,
                outcome=outcome,
                completed_ns=completed,
                context=context,
                deferred=not context.delivery_tracking_active,
            )
            deferred = context.deferred_initial_outcomes
            if transition.deferred:
                deferred = (*deferred, replace(transition, context=None))
            if remaining:
                self._pending[alert_key] = replace(
                    context,
                    remaining_initial_target_keys=frozenset(remaining),
                    deferred_initial_outcomes=deferred,
                )
            elif transition.deferred:
                self._pending[alert_key] = replace(
                    context,
                    remaining_initial_target_keys=frozenset(),
                    deferred_initial_outcomes=deferred,
                )
            else:
                self._pending.pop(alert_key, None)
            return transition

    def complete_initial_target(
        self,
        alert_id: str,
        target_key: object,
    ) -> AlertTimingContext | None:
        """Compatibility helper for a completed target without latency data."""

        transition = self.record_initial_outcome(
            alert_id,
            target_key,
            outcome="delivered",
            completed_ns=0,
        )
        return None if transition is None else transition.context

    def pop(self, alert_id: str) -> AlertTimingContext | None:
        with self._lock:
            return self._pending.pop(str(alert_id), None)

    def discard(self, alert_id: str) -> bool:
        return self.pop(alert_id) is not None

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            awaiting_persistence = sum(
                context.persisted_ns is None for context in self._pending.values()
            )
            awaiting_delivery = len(self._pending) - awaiting_persistence
            return {
                "pending": len(self._pending),
                "awaitingPersistence": awaiting_persistence,
                "awaitingDelivery": awaiting_delivery,
                "pendingDeliveryTargets": sum(
                    len(context.remaining_initial_target_keys)
                    for context in self._pending.values()
                    if context.persisted_ns is not None
                ),
                "capacity": self._maximum_pending,
                "evicted": self._evicted,
            }


registry = AlertTimingRegistry()
