"""Bounded, fair cross-camera inference scheduling.

The scheduler deliberately owns only inference work and result mailboxes.  A
camera worker remains responsible for applying a completed result to mutable
camera, tracking, and alert state.  This boundary keeps capture non-blocking
and makes camera lifecycle fencing explicit.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Hashable


COUNTER_MAX = 2_147_483_647
_ALLOWED_BATCH_SIZES = (1, 2, 4)


class SchedulerNotRunningError(RuntimeError):
    """Raised when lifecycle operations require a running scheduler."""


class CameraOwnershipError(RuntimeError):
    """Raised when a camera operation uses the wrong ownership token."""


@dataclass(frozen=True)
class BatchProfile:
    """Compatibility identity and largest supported cohort for one work item."""

    key: Hashable
    max_batch_size: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_batch_size, bool)
            or self.max_batch_size not in _ALLOWED_BATCH_SIZES
        ):
            raise ValueError("max_batch_size must be one of 1, 2, or 4")
        try:
            hash(self.key)
        except TypeError as exc:
            raise TypeError("BatchProfile.key must be hashable") from exc


@dataclass(frozen=True)
class InferenceWork:
    """One immutable camera frame and the callable that processes it."""

    sequence: int
    profile: BatchProfile
    run: Callable[[int], Any]
    captured_at: float
    expires_at: float | None = None
    urgent: bool = False
    urgent_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not callable(self.run):
            raise TypeError("run must be callable")
        if not math.isfinite(float(self.captured_at)):
            raise ValueError("captured_at must be finite")
        if self.expires_at is not None and not math.isfinite(float(self.expires_at)):
            raise ValueError("expires_at must be finite when provided")
        object.__setattr__(
            self,
            "urgent_reasons",
            tuple(str(reason) for reason in self.urgent_reasons if str(reason)),
        )

    @property
    def is_urgent(self) -> bool:
        return self.urgent or bool(self.urgent_reasons)


@dataclass(frozen=True)
class InferenceOutcome:
    """Exactly one completed result available for camera-thread consumption."""

    camera_id: str
    sequence: int
    batch_size: int
    value: Any
    error: BaseException | None
    captured_at: float
    offered_at: float
    dispatched_at: float
    started_at: float
    completed_at: float

    @property
    def succeeded(self) -> bool:
        return self.error is None


class OfferStatus(str, Enum):
    ACCEPTED = "accepted"
    REPLACED = "replaced"
    STALE = "stale"
    OUT_OF_ORDER = "out_of_order"
    NOT_REGISTERED = "not_registered"
    OWNER_MISMATCH = "owner_mismatch"
    NOT_RUNNING = "not_running"


@dataclass(frozen=True)
class OfferResult:
    status: OfferStatus

    @property
    def accepted(self) -> bool:
        return self.status in {OfferStatus.ACCEPTED, OfferStatus.REPLACED}


@dataclass
class _QueuedWork:
    work: InferenceWork
    offered_at: float


@dataclass
class _InflightWork:
    work: InferenceWork
    offered_at: float
    dispatched_at: float
    batch_size: int


@dataclass
class _CameraState:
    owner_token: str
    registration_id: int
    latency_window_size: int
    queued: _QueuedWork | None = None
    inflight: _InflightWork | None = None
    completed: InferenceOutcome | None = None
    last_offered_sequence: int | None = None
    counters: dict[str, int] = field(default_factory=dict)
    queue_latency_ms: deque[float] = field(init=False)
    service_latency_ms: deque[float] = field(init=False)
    frame_age_ms: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.queue_latency_ms = deque(maxlen=self.latency_window_size)
        self.service_latency_ms = deque(maxlen=self.latency_window_size)
        self.frame_age_ms = deque(maxlen=self.latency_window_size)


@dataclass(frozen=True)
class _Dispatch:
    camera_id: str
    owner_token: str
    registration_id: int
    inflight: _InflightWork


@dataclass(frozen=True)
class _ExecutionRecord:
    value: Any
    error: BaseException | None
    started_at: float
    completed_at: float
    stale_before_run: bool = False


class SharedInferenceScheduler:
    """One latest-frame slot per camera with fair, bounded cohort dispatch."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        batch2_wait_seconds: float = 0.006,
        singleton_wait_seconds: float = 0.014,
        urgent_batch_burst: int = 4,
        latency_window_size: int = 256,
        per_camera_latency_window_size: int = 32,
        clock: Callable[[], float] = time.monotonic,
        drop_observer: Callable[[str, str, int], None] | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or max_workers < 4:
            raise ValueError("max_workers must be at least four")
        if not 0 <= batch2_wait_seconds <= singleton_wait_seconds:
            raise ValueError("batch waits must satisfy 0 <= batch2 <= singleton")
        if urgent_batch_burst < 1:
            raise ValueError("urgent_batch_burst must be positive")
        if latency_window_size < 1 or per_camera_latency_window_size < 1:
            raise ValueError("latency windows must be positive")

        self.max_workers = int(max_workers)
        self.batch2_wait_seconds = float(batch2_wait_seconds)
        self.singleton_wait_seconds = float(singleton_wait_seconds)
        self.urgent_batch_burst = int(urgent_batch_burst)
        self._latency_window_size = int(latency_window_size)
        self._per_camera_latency_window_size = int(
            per_camera_latency_window_size
        )
        self._clock = clock
        self._drop_observer = drop_observer

        self._condition = threading.Condition(threading.RLock())
        self._states: dict[str, _CameraState] = {}
        self._next_registration_id = 1
        self._fair_cursor = 0
        self._urgent_batch_streak = 0
        self._total_inflight = 0
        self._running = False
        self._accepting = False
        self._closing = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._counters: dict[str, int] = {}
        self._queue_latency_ms: deque[float] = deque(
            maxlen=self._latency_window_size
        )
        self._service_latency_ms: deque[float] = deque(
            maxlen=self._latency_window_size
        )
        self._frame_age_ms: deque[float] = deque(
            maxlen=self._latency_window_size
        )

    def start(self) -> bool:
        """Start the scheduler once; repeated calls while running are harmless."""
        with self._condition:
            if self._running:
                return False
            if self._closed:
                raise SchedulerNotRunningError("Scheduler cannot restart after stop")
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="shared-inference",
            )
            self._running = True
            self._accepting = True
            self._closing = False
            self._thread = threading.Thread(
                target=self._scheduler_loop,
                name="shared-inference-scheduler",
                daemon=True,
            )
            self._thread.start()
            self._increment(self._counters, "starts")
            return True

    def register(self, camera_id: str, owner_token: str) -> int:
        """Register one camera lifecycle and return its internal generation."""
        camera_id = self._validate_identity(camera_id, "camera_id")
        owner_token = self._validate_identity(owner_token, "owner_token")
        with self._condition:
            if not self._running or not self._accepting:
                raise SchedulerNotRunningError("Scheduler is not accepting cameras")
            existing = self._states.get(camera_id)
            if existing is not None:
                if existing.owner_token != owner_token:
                    raise CameraOwnershipError(
                        f"Camera {camera_id} is owned by another lifecycle"
                    )
                return existing.registration_id

            registration_id = self._next_registration_id
            self._next_registration_id += 1
            self._states[camera_id] = _CameraState(
                owner_token=owner_token,
                registration_id=registration_id,
                latency_window_size=self._per_camera_latency_window_size,
            )
            self._increment(self._counters, "registrations")
            self._condition.notify_all()
            return registration_id

    def unregister(self, camera_id: str, owner_token: str) -> bool:
        """Fence a camera lifecycle and discard its queued/mailbox state."""
        with self._condition:
            state = self._states.get(camera_id)
            if state is None:
                return False
            self._require_owner(camera_id, owner_token, state)
            if state.queued is not None:
                self._increment(self._counters, "unregister_queue_drops")
                self._notify_drop(camera_id, "lifecycle", 1)
            if state.completed is not None:
                self._increment(self._counters, "unregister_result_drops")
                self._notify_drop(camera_id, "lifecycle", 1)
            del self._states[camera_id]
            self._fair_cursor %= max(1, len(self._states))
            self._increment(self._counters, "unregistrations")
            self._condition.notify_all()
            return True

    def offer(
        self,
        camera_id: str,
        owner_token: str,
        work: InferenceWork,
    ) -> OfferResult:
        """Atomically retain the newest queued work without ever waiting."""
        now = self._clock()
        with self._condition:
            if not self._running or not self._accepting:
                self._increment(self._counters, "offers_not_running")
                return OfferResult(OfferStatus.NOT_RUNNING)
            state = self._states.get(camera_id)
            if state is None:
                self._increment(self._counters, "offers_not_registered")
                return OfferResult(OfferStatus.NOT_REGISTERED)
            if state.owner_token != owner_token:
                self._increment(self._counters, "offers_owner_mismatch")
                return OfferResult(OfferStatus.OWNER_MISMATCH)

            self._increment(self._counters, "offers")
            self._increment(state.counters, "offers")
            if self._is_expired(work, now):
                self._record_stale_locked(state, "stale_offer_drops")
                return OfferResult(OfferStatus.STALE)
            if (
                state.last_offered_sequence is not None
                and work.sequence <= state.last_offered_sequence
            ):
                self._increment(self._counters, "out_of_order_offers")
                self._increment(state.counters, "out_of_order_offers")
                return OfferResult(OfferStatus.OUT_OF_ORDER)

            replaced = state.queued is not None
            if replaced and state.queued is not None and state.queued.work.is_urgent:
                previous_work = state.queued.work
                merged_reasons = tuple(
                    dict.fromkeys(
                        (*previous_work.urgent_reasons, *work.urgent_reasons)
                    )
                )
                work = replace(
                    work,
                    urgent=previous_work.urgent or work.urgent,
                    urgent_reasons=merged_reasons,
                )
            state.queued = _QueuedWork(work=work, offered_at=now)
            state.last_offered_sequence = work.sequence
            if replaced:
                self._increment(self._counters, "replaced_queued")
                self._increment(state.counters, "replaced_queued")
            else:
                self._increment(self._counters, "accepted_offers")
                self._increment(state.counters, "accepted_offers")
            self._condition.notify_all()
            return OfferResult(
                OfferStatus.REPLACED if replaced else OfferStatus.ACCEPTED
            )

    def take_result(
        self,
        camera_id: str,
        owner_token: str,
    ) -> InferenceOutcome | None:
        """Consume a camera's completed result exactly once."""
        with self._condition:
            state = self._states.get(camera_id)
            if state is None:
                return None
            self._require_owner(camera_id, owner_token, state)
            outcome = state.completed
            if outcome is None:
                return None
            state.completed = None
            self._increment(self._counters, "results_taken")
            self._increment(state.counters, "results_taken")
            self._condition.notify_all()
            return outcome

    def stop(self, *, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop dispatch, drop queued work, and optionally drain running calls."""
        deadline = (
            None
            if timeout is None
            else time.monotonic() + max(0.0, float(timeout))
        )
        with self._condition:
            if self._closed:
                return True
            if not self._running:
                self._closed = True
                return True
            self._accepting = False
            self._closing = True
            for camera_id, state in self._states.items():
                if state.queued is not None:
                    state.queued = None
                    self._increment(self._counters, "shutdown_queue_drops")
                    self._increment(state.counters, "shutdown_queue_drops")
                    self._notify_drop(camera_id, "lifecycle", 1)
            thread = self._thread
            executor = self._executor
            self._condition.notify_all()

        if thread is not None:
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        drained = True
        if wait:
            with self._condition:
                while self._total_inflight:
                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    if remaining is not None and remaining <= 0:
                        drained = False
                        break
                    self._condition.wait(timeout=remaining)
        if executor is not None:
            executor.shutdown(
                wait=bool(wait and drained),
                cancel_futures=bool(not wait or not drained),
            )

        with self._condition:
            self._running = False
            self._closed = True
            self._thread = None
            self._executor = None
            self._increment(self._counters, "stops")
            self._condition.notify_all()
            return not wait or drained

    close = stop

    def stats(self) -> dict[str, Any]:
        """Return credential-free bounded counters and latency distributions."""
        with self._condition:
            cameras = {
                camera_id: {
                    "queued": state.queued is not None,
                    "inflight": state.inflight is not None,
                    "completedUnconsumed": state.completed is not None,
                    "lastOfferedSequence": state.last_offered_sequence,
                    "counters": dict(state.counters),
                    "latency": {
                        "queue": self._latency_summary(state.queue_latency_ms),
                        "service": self._latency_summary(
                            state.service_latency_ms
                        ),
                        "frameAge": self._latency_summary(state.frame_age_ms),
                    },
                }
                for camera_id, state in self._states.items()
            }
            return {
                "running": self._running,
                "accepting": self._accepting,
                "registeredCameras": len(self._states),
                "queued": sum(
                    state.queued is not None for state in self._states.values()
                ),
                "inflight": self._total_inflight,
                "completedUnconsumed": sum(
                    state.completed is not None for state in self._states.values()
                ),
                "maxWorkers": self.max_workers,
                "batch2WaitMs": round(self.batch2_wait_seconds * 1000, 3),
                "singletonWaitMs": round(
                    self.singleton_wait_seconds * 1000, 3
                ),
                "latencyWindowCapacity": self._latency_window_size,
                "counters": dict(self._counters),
                "latency": {
                    "queue": self._latency_summary(self._queue_latency_ms),
                    "service": self._latency_summary(self._service_latency_ms),
                    "frameAge": self._latency_summary(self._frame_age_ms),
                },
                "cameras": cameras,
            }

    def _scheduler_loop(self) -> None:
        while True:
            dispatches: list[_Dispatch] | None = None
            with self._condition:
                if self._closing:
                    return
                now = self._clock()
                self._drop_expired_locked(now)
                dispatches, wait_seconds = self._select_dispatch_locked(now)
                if dispatches is None:
                    self._condition.wait(timeout=wait_seconds)
                    continue
            self._submit_dispatches(dispatches)

    def _select_dispatch_locked(
        self,
        now: float,
    ) -> tuple[list[_Dispatch] | None, float | None]:
        capacity = self.max_workers - self._total_inflight
        ready = {
            camera_id: state
            for camera_id, state in self._states.items()
            if state.queued is not None
            and state.inflight is None
            and state.completed is None
        }
        if capacity <= 0 or not ready:
            return None, self._next_wake_delay_locked(now, ready, capacity)

        normal_candidates = self._normal_candidate_ids(ready, now, capacity)
        urgent_ids = {
            camera_id
            for camera_id, state in ready.items()
            if state.queued is not None and state.queued.work.is_urgent
        }
        force_normal = bool(
            normal_candidates
            and urgent_ids
            and self._urgent_batch_streak >= self.urgent_batch_burst
        )

        if urgent_ids and not force_normal:
            seed_id = self._fair_pick_locked(urgent_ids)
            selected_ids = self._cohort_for_seed_locked(
                seed_id,
                ready,
                capacity,
                immediate=True,
                now=now,
            )
            self._urgent_batch_streak = min(
                COUNTER_MAX,
                self._urgent_batch_streak + 1,
            )
        elif normal_candidates:
            seed_id = self._fair_pick_locked(normal_candidates)
            selected_ids = self._cohort_for_seed_locked(
                seed_id,
                ready,
                capacity,
                immediate=False,
                now=now,
            )
            self._urgent_batch_streak = 0
        else:
            return None, self._next_wake_delay_locked(now, ready, capacity)

        batch_size = len(selected_ids)
        if batch_size not in _ALLOWED_BATCH_SIZES:
            raise RuntimeError("Scheduler selected an invalid cohort size")
        dispatches: list[_Dispatch] = []
        for camera_id in selected_ids:
            state = self._states[camera_id]
            queued = state.queued
            if queued is None or self._is_expired(queued.work, now):
                if queued is not None:
                    state.queued = None
                    self._record_stale_locked(
                        state,
                        "stale_dispatch_drops",
                        camera_id=camera_id,
                        notify=True,
                    )
                continue
            inflight = _InflightWork(
                work=queued.work,
                offered_at=queued.offered_at,
                dispatched_at=now,
                batch_size=batch_size,
            )
            state.queued = None
            state.inflight = inflight
            self._total_inflight += 1
            self._increment(self._counters, "dispatched")
            self._increment(state.counters, "dispatched")
            if queued.work.is_urgent:
                self._increment(self._counters, "urgent_dispatched")
                self._increment(state.counters, "urgent_dispatched")
            dispatches.append(
                _Dispatch(
                    camera_id=camera_id,
                    owner_token=state.owner_token,
                    registration_id=state.registration_id,
                    inflight=inflight,
                )
            )

        if not dispatches:
            return None, 0.0
        if len(dispatches) != batch_size:
            raise RuntimeError("Scheduler dispatch cohort changed under its lock")
        self._increment(self._counters, f"batches_{batch_size}")
        self._advance_fair_cursor_locked(selected_ids[-1])
        return dispatches, 0.0

    def _normal_candidate_ids(
        self,
        ready: dict[str, _CameraState],
        now: float,
        capacity: int,
    ) -> set[str]:
        groups = self._profile_groups(ready)
        candidates: set[str] = set()
        for camera_ids in groups.values():
            first_queued = ready[camera_ids[0]].queued
            if first_queued is None:
                continue
            profile = first_queued.work.profile
            normal_ids = {
                camera_id
                for camera_id in camera_ids
                if ready[camera_id].queued is not None
                and not ready[camera_id].queued.work.is_urgent
            }
            if (
                profile.max_batch_size >= 4
                and capacity >= 4
                and len(camera_ids) >= 4
            ):
                candidates.update(normal_ids)
                continue
            oldest = min(
                queued.offered_at
                for camera_id in camera_ids
                if (queued := ready[camera_id].queued) is not None
            )
            if (
                profile.max_batch_size >= 2
                and capacity >= 2
                and len(camera_ids) >= 2
                and now - oldest >= self.batch2_wait_seconds
            ):
                candidates.update(normal_ids)
        candidates.update(
            camera_id
            for camera_id, state in ready.items()
            if state.queued is not None
            and not state.queued.work.is_urgent
            and now - state.queued.offered_at >= self.singleton_wait_seconds
        )
        return candidates

    def _cohort_for_seed_locked(
        self,
        seed_id: str,
        ready: dict[str, _CameraState],
        capacity: int,
        *,
        immediate: bool,
        now: float,
    ) -> list[str]:
        seed = ready[seed_id].queued
        if seed is None:
            return []
        compatible = {
            camera_id
            for camera_id, state in ready.items()
            if state.queued is not None
            and state.queued.work.profile == seed.work.profile
        }
        ordered = self._fair_order_locked(compatible, seed_id)
        maximum = min(seed.work.profile.max_batch_size, capacity, len(ordered))
        if maximum >= 4:
            batch_size = 4
        elif maximum >= 2 and (
            immediate
            or min(
                queued.offered_at
                for camera_id in compatible
                if (queued := ready[camera_id].queued) is not None
            )
            <= now - self.batch2_wait_seconds
        ):
            batch_size = 2
        else:
            batch_size = 1
        return ordered[:batch_size]

    def _profile_groups(
        self,
        ready: dict[str, _CameraState],
    ) -> dict[BatchProfile, list[str]]:
        groups: dict[BatchProfile, list[str]] = {}
        for camera_id, state in ready.items():
            if state.queued is not None:
                groups.setdefault(state.queued.work.profile, []).append(camera_id)
        return groups

    def _fair_pick_locked(self, eligible: set[str]) -> str:
        ordered = list(self._states)
        if not ordered:
            raise RuntimeError("No registered cameras are available")
        start = self._fair_cursor % len(ordered)
        for offset in range(len(ordered)):
            camera_id = ordered[(start + offset) % len(ordered)]
            if camera_id in eligible:
                return camera_id
        raise RuntimeError("Fair selection received no registered candidate")

    def _fair_order_locked(self, eligible: set[str], seed_id: str) -> list[str]:
        ordered = list(self._states)
        seed_index = ordered.index(seed_id)
        return [
            camera_id
            for offset in range(len(ordered))
            if (camera_id := ordered[(seed_index + offset) % len(ordered)])
            in eligible
        ]

    def _advance_fair_cursor_locked(self, last_camera_id: str) -> None:
        ordered = list(self._states)
        if not ordered:
            self._fair_cursor = 0
            return
        try:
            index = ordered.index(last_camera_id)
        except ValueError:
            self._fair_cursor %= len(ordered)
        else:
            self._fair_cursor = (index + 1) % len(ordered)

    def _next_wake_delay_locked(
        self,
        now: float,
        ready: dict[str, _CameraState],
        capacity: int,
    ) -> float | None:
        wake_times: list[float] = []
        for camera_id, state in self._states.items():
            if state.queued is not None and state.queued.work.expires_at is not None:
                wake_times.append(state.queued.work.expires_at)
        if capacity > 0:
            for state in ready.values():
                if state.queued is not None and not state.queued.work.is_urgent:
                    wake_times.append(
                        state.queued.offered_at + self.singleton_wait_seconds
                    )
            for camera_ids in self._profile_groups(ready).values():
                state = ready[camera_ids[0]]
                if (
                    state.queued is not None
                    and state.queued.work.profile.max_batch_size >= 2
                    and len(camera_ids) >= 2
                    and capacity >= 2
                ):
                    wake_times.append(
                        min(
                            queued.offered_at
                            for camera_id in camera_ids
                            if (queued := ready[camera_id].queued) is not None
                        )
                        + self.batch2_wait_seconds
                    )
        if not wake_times:
            return None
        return max(0.0, min(wake_times) - now)

    def _submit_dispatches(self, dispatches: list[_Dispatch]) -> None:
        executor = self._executor
        if executor is None:
            for dispatch in dispatches:
                self._complete_execution(
                    dispatch,
                    _ExecutionRecord(
                        value=None,
                        error=SchedulerNotRunningError("Executor is unavailable"),
                        started_at=self._clock(),
                        completed_at=self._clock(),
                    ),
                )
            return
        for dispatch in dispatches:
            try:
                future = executor.submit(self._execute, dispatch)
            except BaseException as exc:
                now = self._clock()
                self._complete_execution(
                    dispatch,
                    _ExecutionRecord(
                        value=None,
                        error=self._strip_traceback(exc),
                        started_at=now,
                        completed_at=now,
                    ),
                )
                continue
            future.add_done_callback(
                lambda completed, item=dispatch: self._future_done(item, completed)
            )

    def _execute(self, dispatch: _Dispatch) -> _ExecutionRecord:
        started_at = self._clock()
        if self._is_expired(dispatch.inflight.work, started_at):
            return _ExecutionRecord(
                value=None,
                error=None,
                started_at=started_at,
                completed_at=started_at,
                stale_before_run=True,
            )
        try:
            value = dispatch.inflight.work.run(dispatch.inflight.batch_size)
        except BaseException as exc:
            error = self._strip_traceback(exc)
            value = None
        else:
            error = None
        return _ExecutionRecord(
            value=value,
            error=error,
            started_at=started_at,
            completed_at=self._clock(),
        )

    def _future_done(self, dispatch: _Dispatch, future: Future) -> None:
        try:
            record = future.result()
        except BaseException as exc:  # pragma: no cover - defensive executor guard.
            now = self._clock()
            record = _ExecutionRecord(
                value=None,
                error=self._strip_traceback(exc),
                started_at=now,
                completed_at=now,
            )
        self._complete_execution(dispatch, record)

    def _complete_execution(
        self,
        dispatch: _Dispatch,
        record: _ExecutionRecord,
    ) -> None:
        with self._condition:
            self._total_inflight = max(0, self._total_inflight - 1)
            state = self._states.get(dispatch.camera_id)
            if (
                state is None
                or state.registration_id != dispatch.registration_id
                or state.owner_token != dispatch.owner_token
                or state.inflight is None
                or state.inflight.work.sequence != dispatch.inflight.work.sequence
            ):
                self._increment(self._counters, "fenced_completions")
                self._notify_drop(dispatch.camera_id, "lifecycle", 1)
                self._condition.notify_all()
                return

            state.inflight = None
            if record.stale_before_run:
                self._record_stale_locked(
                    state,
                    "stale_before_run_drops",
                    camera_id=dispatch.camera_id,
                    notify=True,
                )
                self._condition.notify_all()
                return
            if self._is_expired(dispatch.inflight.work, record.completed_at):
                # A provider call can begin while fresh and finish after its
                # evidence window. Never publish that result to alert/tracker
                # state; the camera's next captured frame remains eligible.
                self._record_stale_locked(
                    state,
                    "stale_completion_drops",
                    camera_id=dispatch.camera_id,
                    notify=True,
                )
                self._condition.notify_all()
                return

            queue_ms = max(
                0.0,
                (record.started_at - dispatch.inflight.offered_at) * 1000,
            )
            service_ms = max(
                0.0,
                (record.completed_at - record.started_at) * 1000,
            )
            frame_age_ms = max(
                0.0,
                (record.completed_at - dispatch.inflight.work.captured_at) * 1000,
            )
            self._record_latency(self._queue_latency_ms, queue_ms)
            self._record_latency(self._service_latency_ms, service_ms)
            self._record_latency(self._frame_age_ms, frame_age_ms)
            self._record_latency(state.queue_latency_ms, queue_ms)
            self._record_latency(state.service_latency_ms, service_ms)
            self._record_latency(state.frame_age_ms, frame_age_ms)

            state.completed = InferenceOutcome(
                camera_id=dispatch.camera_id,
                sequence=dispatch.inflight.work.sequence,
                batch_size=dispatch.inflight.batch_size,
                value=record.value,
                error=record.error,
                captured_at=dispatch.inflight.work.captured_at,
                offered_at=dispatch.inflight.offered_at,
                dispatched_at=dispatch.inflight.dispatched_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
            )
            self._increment(self._counters, "completed")
            self._increment(state.counters, "completed")
            if record.error is not None:
                self._increment(self._counters, "failed")
                self._increment(state.counters, "failed")
            self._condition.notify_all()

    def _drop_expired_locked(self, now: float) -> None:
        for camera_id, state in self._states.items():
            if state.queued is not None and self._is_expired(
                state.queued.work,
                now,
            ):
                state.queued = None
                self._record_stale_locked(
                    state,
                    "stale_dispatch_drops",
                    camera_id=camera_id,
                    notify=True,
                )

    def _record_stale_locked(
        self,
        state: _CameraState,
        counter: str,
        *,
        camera_id: str | None = None,
        notify: bool = False,
    ) -> None:
        self._increment(self._counters, "stale_drops")
        self._increment(self._counters, counter)
        self._increment(state.counters, "stale_drops")
        self._increment(state.counters, counter)
        if notify and camera_id is not None:
            self._notify_drop(camera_id, "stale", 1)

    def _notify_drop(self, camera_id: str, reason: str, amount: int) -> None:
        observer = self._drop_observer
        if observer is None:
            return
        try:
            observer(camera_id, reason, amount)
        except Exception:
            self._increment(self._counters, "drop_observer_failures")

    @staticmethod
    def _is_expired(work: InferenceWork, now: float) -> bool:
        return work.expires_at is not None and now >= work.expires_at

    @staticmethod
    def _strip_traceback(exc: BaseException) -> BaseException:
        try:
            return exc.with_traceback(None)
        except (AttributeError, TypeError):
            return RuntimeError(str(exc))

    @staticmethod
    def _validate_identity(value: str, name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{name} must not be empty")
        return normalized

    @staticmethod
    def _require_owner(
        camera_id: str,
        owner_token: str,
        state: _CameraState,
    ) -> None:
        if state.owner_token != owner_token:
            raise CameraOwnershipError(
                f"Camera {camera_id} is owned by another lifecycle"
            )

    @staticmethod
    def _increment(
        counters: dict[str, int],
        key: str,
        amount: int = 1,
    ) -> None:
        current = counters.get(key, 0)
        counters[key] = min(COUNTER_MAX, max(0, current + amount))

    @staticmethod
    def _record_latency(samples: deque[float], value: float) -> None:
        samples.append(round(float(value), 6))

    @staticmethod
    def _latency_summary(samples: deque[float]) -> dict[str, float | int | None]:
        if not samples:
            return {
                "sampleCount": 0,
                "medianMs": None,
                "p95Ms": None,
                "p99Ms": None,
                "maxMs": None,
            }
        ordered = sorted(samples)

        def percentile(fraction: float) -> float:
            index = min(
                len(ordered) - 1,
                max(0, math.ceil(len(ordered) * fraction) - 1),
            )
            return round(ordered[index], 3)

        return {
            "sampleCount": len(ordered),
            "medianMs": percentile(0.5),
            "p95Ms": percentile(0.95),
            "p99Ms": percentile(0.99),
            "maxMs": round(ordered[-1], 3),
        }


__all__ = [
    "BatchProfile",
    "CameraOwnershipError",
    "InferenceOutcome",
    "InferenceWork",
    "OfferResult",
    "OfferStatus",
    "SchedulerNotRunningError",
    "SharedInferenceScheduler",
]
