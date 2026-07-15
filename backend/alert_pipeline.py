"""Bounded asynchronous alert persistence and delivery pipeline."""

from __future__ import annotations

import copy
import heapq
import logging
import queue
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field, replace
from typing import Callable

logger = logging.getLogger("safetylens.alert_pipeline")

_CALLBACK_QUEUE: queue.SimpleQueue = queue.SimpleQueue()
_CALLBACK_WORKERS_LOCK = threading.Lock()
_CALLBACK_WORKERS_STARTED = False
_CALLBACK_WORKER_COUNT = 1


def _future_callback_loop() -> None:
    while True:
        callback, future = _CALLBACK_QUEUE.get()
        try:
            callback(future)
        except BaseException:
            logger.exception("Persistence Future callback failed")


def _ensure_future_callback_worker() -> None:
    global _CALLBACK_WORKERS_STARTED
    with _CALLBACK_WORKERS_LOCK:
        if _CALLBACK_WORKERS_STARTED:
            return
        workers = [
            threading.Thread(
                target=_future_callback_loop,
                name=f"alert-future-callback-{index + 1}",
                daemon=True,
            )
            for index in range(_CALLBACK_WORKER_COUNT)
        ]
        for worker in workers:
            worker.start()
        _CALLBACK_WORKERS_STARTED = True


def _dispatch_future_callbacks(callbacks: list[Callable], future: Future) -> None:
    if not callbacks:
        return
    for callback in callbacks:
        _CALLBACK_QUEUE.put((callback, future))
    try:
        _ensure_future_callback_worker()
    except Exception:
        # Future callbacks are observers. Failure to start their dispatcher
        # must never suppress persistence or external safety notification.
        logger.exception("Could not start persistence Future callback dispatcher")


class _AlertFuture(Future):
    """Future whose user callbacks never execute on the persistence worker."""

    def __init__(self) -> None:
        super().__init__()
        self._callbacks_dispatched = False

    def add_done_callback(self, fn: Callable[[Future], object]) -> None:
        with self._condition:
            if not self.done() or not self._callbacks_dispatched:
                self._done_callbacks.append(fn)
                return
            _dispatch_future_callbacks([fn], self)

    def _invoke_callbacks(self) -> None:
        with self._condition:
            callbacks = list(self._done_callbacks)
            self._done_callbacks.clear()
            self._callbacks_dispatched = True
            _dispatch_future_callbacks(callbacks, self)


@dataclass(frozen=True)
class DeliveryOutcome:
    """Describe only the targets that still need work after one dispatch call.

    ``retry_all`` is reserved for legacy boolean callbacks and unexpected
    callback exceptions where the pipeline cannot identify a narrower target.
    Explicit target IDs let a successful channel stay delivered while only a
    failed channel is retried. ``handled_output_ids`` completes local paths
    such as in-app persistence without counting them as external delivery.
    """

    delivered_output_ids: tuple[str, ...] = ()
    retry_output_ids: tuple[str, ...] = ()
    terminal_output_ids: tuple[str, ...] = ()
    handled_output_ids: tuple[str, ...] = ()
    retry_all: bool = False


@dataclass(frozen=True)
class _PersistJob:
    payload: dict
    output_ids: list[str] | None
    future: Future
    enqueued_at: float


@dataclass(frozen=True)
class _DeliveryJob:
    alert: dict
    output_ids: list[str] | None
    attempt: int = 1
    delivered_output_ids: tuple[str, ...] = ()
    terminal_output_ids: tuple[str, ...] = ()
    handoff_ready: threading.Event | None = field(default=None, compare=False, repr=False)


@dataclass(order=True, frozen=True)
class _ScheduledRetry:
    due_at: float
    sequence: int
    job: _DeliveryJob = field(compare=False)


class AlertPipeline:
    """Persist alerts in order, then deliver notifications concurrently.

    The persistence queue is deliberately bounded. When it fills, producers
    wait instead of silently dropping a safety alert. Under normal load,
    ``submit`` only copies the request and enqueues it.
    """

    def __init__(
        self,
        *,
        persist_alert: Callable[..., dict],
        deliver_alert: Callable[[dict, list[str] | None], DeliveryOutcome | bool],
        on_persisted: Callable[[dict], None] | None = None,
        persist_queue_size: int = 256,
        delivery_queue_size: int = 256,
        delivery_workers: int = 4,
        submit_timeout: float = 0.01,
        persistence_attempts: int = 3,
        retry_delay: float = 0.1,
        delivery_attempts: int = 3,
        delivery_retry_delay: float = 1.0,
        delivery_retry_queue_size: int | None = None,
    ):
        if persist_queue_size < 1 or delivery_queue_size < 1:
            raise ValueError("Alert queue sizes must be positive")
        if delivery_workers < 1:
            raise ValueError("At least one alert delivery worker is required")
        if persistence_attempts < 1:
            raise ValueError("At least one persistence attempt is required")
        if delivery_attempts < 1:
            raise ValueError("At least one delivery attempt is required")
        if delivery_retry_queue_size is not None and delivery_retry_queue_size < 1:
            raise ValueError("Alert delivery retry queue size must be positive")

        self._persist_alert = persist_alert
        self._deliver_alert = deliver_alert
        self._on_persisted = on_persisted
        self._delivery_worker_count = delivery_workers
        self._submit_timeout = max(0.0, submit_timeout)
        self._persistence_attempts = persistence_attempts
        self._retry_delay = max(0.0, retry_delay)
        self._delivery_attempts = delivery_attempts
        self._delivery_retry_delay = max(0.0, delivery_retry_delay)
        self._delivery_retry_queue_size = delivery_retry_queue_size or delivery_queue_size

        self._persist_queue: queue.Queue = queue.Queue(maxsize=persist_queue_size)
        self._delivery_queue: queue.Queue = queue.Queue(maxsize=delivery_queue_size)
        self._lifecycle_lock = threading.Lock()
        self._submission_condition = threading.Condition(self._lifecycle_lock)
        self._shutdown_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._retry_condition = threading.Condition()
        self._retry_heap: list[_ScheduledRetry] = []
        self._retry_sequence = 0
        self._pending_retries = 0
        self._stats_lock = threading.Lock()
        self._active_persist_enqueued_at: float | None = None
        self._consecutive_persistence_failures = 0
        self._last_persistence_failure_at: float | None = None
        self._last_persistence_success_at: float | None = None
        self._running = False
        self._accepting = False
        self._active_submitters = 0
        self._persist_thread: threading.Thread | None = None
        self._retry_thread: threading.Thread | None = None
        self._delivery_threads: list[threading.Thread] = []
        self._counters = {
            "submitted": 0,
            "persisted": 0,
            "persistence_failures": 0,
            "callback_failures": 0,
            "delivered": 0,
            "outbox_handoffs": 0,
            "delivery_failures": 0,
            "partially_delivered": 0,
            "delivery_attempts": 0,
            "delivery_retries": 0,
            "delivery_terminal_failures": 0,
            "delivery_retry_exhausted": 0,
            "delivery_retry_queue_full": 0,
            "backpressure_events": 0,
        }

    def start(self) -> None:
        """Start workers. Safe to call repeatedly."""
        with self._lifecycle_lock:
            if self._running:
                return
            # Start the observer callback dispatcher before accepting safety
            # work so a later thread-creation failure cannot interrupt delivery.
            _ensure_future_callback_worker()
            self._worker_stop.clear()
            self._running = True
            self._accepting = False
            workers: list[threading.Thread] = []
            try:
                self._retry_thread = threading.Thread(
                    target=self._retry_loop,
                    name="alert-delivery-retry-scheduler",
                    daemon=True,
                )
                self._delivery_threads = [
                    threading.Thread(
                        target=self._delivery_loop,
                        name=f"alert-delivery-{index + 1}",
                        daemon=True,
                    )
                    for index in range(self._delivery_worker_count)
                ]
                self._persist_thread = threading.Thread(
                    target=self._persistence_loop,
                    name="alert-persistence",
                    daemon=True,
                )
                workers = [self._retry_thread, *self._delivery_threads, self._persist_thread]
                for worker in workers:
                    worker.start()
            except Exception:
                self._worker_stop.set()
                with self._retry_condition:
                    self._retry_condition.notify_all()
                for worker in workers:
                    if worker.is_alive():
                        worker.join(timeout=0.25)
                self._running = False
                self._accepting = False
                self._persist_thread = None
                self._retry_thread = None
                self._delivery_threads = []
                raise
            self._accepting = True

    def submit(
        self,
        payload: dict,
        *,
        output_ids: list[str] | None = None,
        allow_backpressure: bool = True,
    ) -> Future:
        """Queue an immutable alert request and return its persistence future.

        Core safety alerts keep the default lossless producer backpressure.
        Optional advisory producers may set ``allow_backpressure=False`` so a
        full persistence queue rejects immediately instead of blocking the
        real-time or shutdown path.
        """
        self.start()
        future = _AlertFuture()
        job = _PersistJob(
            payload=copy.deepcopy(payload),
            output_ids=list(output_ids) if output_ids is not None else None,
            future=future,
            enqueued_at=time.monotonic(),
        )
        # Register accepted producers before touching the bounded queue.
        # Shutdown closes admission, then waits for these producers to enqueue;
        # no lifecycle lock is held while backpressure blocks a producer.
        with self._submission_condition:
            if not self._running or not self._accepting:
                raise RuntimeError("Alert pipeline is shutting down")
            self._active_submitters += 1
        try:
            if allow_backpressure:
                try:
                    self._persist_queue.put(job, timeout=self._submit_timeout)
                except queue.Full:
                    self._increment("backpressure_events")
                    logger.warning(
                        "Alert persistence queue full; applying producer backpressure",
                        extra={"queue_size": self._persist_queue.qsize()},
                    )
                    self._persist_queue.put(job)
            else:
                try:
                    self._persist_queue.put_nowait(job)
                except queue.Full:
                    self._increment("backpressure_events")
                    raise
            self._increment("submitted")
        finally:
            with self._submission_condition:
                self._active_submitters -= 1
                self._submission_condition.notify_all()
        return future

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until queued persistence and delivery work is complete."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._retry_condition:
                pending_retries = self._pending_retries
            if (
                self._persist_queue.unfinished_tasks == 0
                and self._delivery_queue.unfinished_tasks == 0
                and pending_retries == 0
            ):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    def shutdown(self, *, wait: bool = True, timeout: float = 10.0) -> bool:
        """Stop accepting work, then stop workers only after accepted work drains.

        A timed-out shutdown deliberately leaves workers alive so accepted
        work can finish. Callers may retry shutdown after the blocker clears.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        remaining = max(0.0, deadline - time.monotonic())
        if not self._shutdown_lock.acquire(timeout=remaining):
            return False
        try:
            with self._submission_condition:
                if not self._running:
                    return True
                self._accepting = False
                while self._active_submitters:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "Alert pipeline shutdown timed out waiting for accepted producers"
                        )
                        return False
                    self._submission_condition.wait(timeout=remaining)

            if wait:
                drained = self.drain(max(0.0, deadline - time.monotonic()))
            else:
                with self._retry_condition:
                    pending_retries = self._pending_retries
                drained = (
                    self._persist_queue.unfinished_tasks == 0
                    and self._delivery_queue.unfinished_tasks == 0
                    and pending_retries == 0
                )
            if not drained:
                logger.warning("Alert pipeline shutdown timed out; workers remain active")
                return False

            # Polling workers stop only after every accepted job has drained.
            # No queue sentinel is left behind for a later worker generation.
            self._worker_stop.set()
            with self._retry_condition:
                self._retry_condition.notify_all()

            workers = [
                self._persist_thread,
                self._retry_thread,
                *self._delivery_threads,
            ]
            for worker in workers:
                if worker is not None and worker.is_alive():
                    worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if any(worker is not None and worker.is_alive() for worker in workers):
                return False

            with self._lifecycle_lock:
                self._running = False
                self._accepting = False
                self._persist_thread = None
                self._retry_thread = None
                self._delivery_threads = []
            return True
        finally:
            self._shutdown_lock.release()

    def stats(self) -> dict:
        """Return counters and queue depths for diagnostics."""
        with self._stats_lock:
            result = dict(self._counters)
            active_persist_enqueued_at = self._active_persist_enqueued_at
            result.update({
                "consecutive_persistence_failures": (
                    self._consecutive_persistence_failures
                ),
                "last_persistence_failure_at": self._last_persistence_failure_at,
                "last_persistence_success_at": self._last_persistence_success_at,
                "persistence_in_flight": active_persist_enqueued_at is not None,
            })
        with self._lifecycle_lock:
            result["running"] = self._running
            result["accepting"] = self._accepting
            result["active_submitters"] = self._active_submitters
            result["persist_worker_alive"] = bool(
                self._persist_thread and self._persist_thread.is_alive()
            )
            result["retry_worker_alive"] = bool(
                self._retry_thread and self._retry_thread.is_alive()
            )
            result["delivery_workers_alive"] = sum(
                worker.is_alive() for worker in self._delivery_threads
            )
        with self._retry_condition:
            pending_retries = self._pending_retries
        # Queue.Queue has no public peek operation. Reading its bounded deque
        # under the queue's own mutex gives diagnostics a race-free snapshot
        # without maintaining a second, potentially divergent job registry.
        with self._persist_queue.mutex:
            queued_persist_enqueued_at = (
                self._persist_queue.queue[0].enqueued_at
                if self._persist_queue.queue
                else None
            )
        oldest_persist_enqueued_at = min(
            (
                enqueued_at
                for enqueued_at in (
                    active_persist_enqueued_at,
                    queued_persist_enqueued_at,
                )
                if enqueued_at is not None
            ),
            default=None,
        )
        oldest_persistence_age = (
            max(0.0, time.monotonic() - oldest_persist_enqueued_at)
            if oldest_persist_enqueued_at is not None
            else None
        )
        result.update({
            "persist_queue_depth": self._persist_queue.qsize(),
            "oldest_persistence_age_seconds": (
                round(oldest_persistence_age, 3)
                if oldest_persistence_age is not None
                else None
            ),
            "delivery_queue_depth": self._delivery_queue.qsize(),
            "delivery_retry_queue_depth": pending_retries,
            "delivery_retry_queue_capacity": self._delivery_retry_queue_size,
        })
        return result

    def _persistence_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                job = self._persist_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._stats_lock:
                self._active_persist_enqueued_at = job.enqueued_at
            try:
                try:
                    self._persist(job)
                except Exception as exc:
                    self._record_persistence_failure()
                    self._set_future_exception(job.future, exc)
                    logger.exception("Unexpected alert persistence worker failure")
            finally:
                with self._stats_lock:
                    self._active_persist_enqueued_at = None
                self._persist_queue.task_done()

    def _persist(self, job: _PersistJob) -> None:
        alert = None
        last_error: Exception | None = None
        for attempt in range(1, self._persistence_attempts + 1):
            try:
                persisted_alert = self._persist_alert(**job.payload)
                if not isinstance(persisted_alert, Mapping):
                    raise TypeError(
                        "Alert persistence callback must return a mapping, "
                        f"got {type(persisted_alert).__name__}"
                    )
                alert = dict(persisted_alert)
                break
            except Exception as exc:
                last_error = exc
                if attempt < self._persistence_attempts:
                    logger.warning(
                        "Alert persistence failed; retrying",
                        extra={"attempt": attempt, "max_attempts": self._persistence_attempts},
                        exc_info=True,
                    )
                    if self._retry_delay:
                        time.sleep(self._retry_delay)

        if alert is None:
            self._record_persistence_failure()
            if last_error is None:
                last_error = RuntimeError("Alert persistence returned no alert")
            self._set_future_exception(job.future, last_error)
            logger.error(
                "Alert persistence failed permanently: %s",
                last_error,
                extra={"attempts": self._persistence_attempts},
            )
            return

        self._record_persistence_success()
        if self._on_persisted is not None:
            try:
                self._on_persisted(alert)
            except Exception:
                self._increment("callback_failures")
                logger.exception("Post-persistence alert callback failed")

        # The Future reports persistence only. Cancelling that observation must
        # never kill the persistence worker or cancel a safety alert.
        self._set_future_result(job.future, alert)
        self._delivery_queue.put(_DeliveryJob(alert=alert, output_ids=job.output_ids))

    def _delivery_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                job = self._delivery_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if job.handoff_ready is not None:
                    # The scheduler releases retry-backlog capacity only after
                    # Queue.put registers this attempt. Do not let a fast
                    # provider race that bookkeeping and reject its next retry.
                    job.handoff_ready.wait()
                try:
                    self._deliver(job)
                except Exception:
                    # A malformed callback result or accounting bug must not
                    # silently kill a worker and strand every later alert.
                    self._record_delivery_failure(
                        job.alert,
                        delivered_output_ids=set(job.delivered_output_ids),
                        terminal_output_ids=set(job.terminal_output_ids),
                        failed_output_ids=self._pending_output_id_set(job.output_ids),
                        exhausted=True,
                    )
                    logger.exception(
                        "Unexpected alert delivery worker failure",
                        extra={"alert_id": self._alert_id(job.alert), "attempt": job.attempt},
                    )
            finally:
                self._delivery_queue.task_done()

    def _deliver(self, job: _DeliveryJob) -> None:
        pending_output_ids = list(job.output_ids) if job.output_ids is not None else None
        delivered_output_ids = set(job.delivered_output_ids)
        terminal_output_ids = set(job.terminal_output_ids)
        self._increment("delivery_attempts")
        try:
            raw_outcome = self._deliver_alert(job.alert, pending_output_ids)
            outcome = self._coerce_delivery_outcome(raw_outcome, pending_output_ids)
        except Exception:
            outcome = DeliveryOutcome(retry_all=True)
            logger.exception(
                "Alert delivery callback failed",
                extra={
                    "alert_id": self._alert_id(job.alert),
                    "attempt": job.attempt,
                    "max_attempts": self._delivery_attempts,
                },
            )

        delivered_output_ids.update(outcome.delivered_output_ids)
        terminal_output_ids.update(outcome.terminal_output_ids)
        has_retry = outcome.retry_all or bool(outcome.retry_output_ids)
        if not has_retry:
            if terminal_output_ids:
                self._record_delivery_failure(
                    job.alert,
                    delivered_output_ids=delivered_output_ids,
                    terminal_output_ids=terminal_output_ids,
                    failed_output_ids=set(terminal_output_ids),
                    exhausted=False,
                )
            elif outcome.handled_output_ids and not delivered_output_ids:
                if "durable_outbox" in outcome.handled_output_ids:
                    self._increment("outbox_handoffs")
            else:
                self._increment("delivered")
            return

        if job.attempt >= self._delivery_attempts:
            failed_output_ids = self._retry_output_id_set(
                outcome,
                pending_output_ids,
            ) | terminal_output_ids
            self._record_delivery_failure(
                job.alert,
                delivered_output_ids=delivered_output_ids,
                terminal_output_ids=terminal_output_ids,
                failed_output_ids=failed_output_ids,
                exhausted=True,
            )
            return

        retry_output_ids = pending_output_ids
        if not outcome.retry_all:
            retry_output_ids = list(outcome.retry_output_ids)
        delay = min(
            self._delivery_retry_delay * (2 ** min(job.attempt - 1, 10)),
            60.0,
        )
        scheduled = self._schedule_retry(
            _DeliveryJob(
                alert=job.alert,
                output_ids=retry_output_ids,
                attempt=job.attempt + 1,
                delivered_output_ids=tuple(sorted(delivered_output_ids)),
                terminal_output_ids=tuple(sorted(terminal_output_ids)),
            ),
            delay=delay,
        )
        if scheduled:
            self._increment("delivery_retries")
            logger.warning(
                "Alert delivery incomplete; scheduled failed targets for retry",
                extra={
                    "alert_id": self._alert_id(job.alert),
                    "attempt": job.attempt,
                    "max_attempts": self._delivery_attempts,
                    "retry_output_ids": retry_output_ids,
                },
            )
            return

        self._increment("delivery_retry_queue_full")
        failed_output_ids = self._pending_output_id_set(retry_output_ids) | terminal_output_ids
        self._record_delivery_failure(
            job.alert,
            delivered_output_ids=delivered_output_ids,
            terminal_output_ids=terminal_output_ids,
            failed_output_ids=failed_output_ids,
            exhausted=False,
        )
        logger.error(
            "Alert delivery retry queue full; retry not scheduled",
            extra={
                "alert_id": self._alert_id(job.alert),
                "retry_queue_size": self._delivery_retry_queue_size,
            },
        )

    @staticmethod
    def _coerce_delivery_outcome(
        raw_outcome: DeliveryOutcome | bool,
        requested_output_ids: list[str] | None,
    ) -> DeliveryOutcome:
        if raw_outcome is True:
            return DeliveryOutcome()
        if raw_outcome is False:
            return DeliveryOutcome(retry_all=True)
        if not isinstance(raw_outcome, DeliveryOutcome):
            raise TypeError(f"Unsupported delivery callback result: {type(raw_outcome).__name__}")
        if type(raw_outcome.retry_all) is not bool:
            raise TypeError("DeliveryOutcome.retry_all must be a bool")

        def normalize_ids(value: object, field_name: str) -> tuple[str, ...]:
            if isinstance(value, (str, bytes)):
                raise TypeError(f"DeliveryOutcome.{field_name} must be an iterable of strings")
            try:
                values = list(value)
            except TypeError as exc:
                raise TypeError(
                    f"DeliveryOutcome.{field_name} must be an iterable of strings"
                ) from exc
            normalized: list[str] = []
            seen: set[str] = set()
            for value_item in values:
                if not isinstance(value_item, str) or not value_item.strip():
                    raise TypeError(
                        f"DeliveryOutcome.{field_name} must contain non-empty strings"
                    )
                output_id = value_item.strip().lower()
                if output_id not in seen:
                    seen.add(output_id)
                    normalized.append(output_id)
            return tuple(normalized)

        delivered = normalize_ids(raw_outcome.delivered_output_ids, "delivered_output_ids")
        retry = normalize_ids(raw_outcome.retry_output_ids, "retry_output_ids")
        terminal = normalize_ids(raw_outcome.terminal_output_ids, "terminal_output_ids")
        handled = normalize_ids(raw_outcome.handled_output_ids, "handled_output_ids")
        partitions = [set(delivered), set(retry), set(terminal), set(handled)]
        if raw_outcome.retry_all and any(partitions):
            raise ValueError("DeliveryOutcome cannot combine retry_all with explicit targets")
        if any(
            partitions[left] & partitions[right]
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        ):
            raise ValueError("DeliveryOutcome target sets must be disjoint")
        if requested_output_ids is not None:
            requested = set(AlertPipeline._normalize_requested_output_ids(requested_output_ids))
            partitioned = set().union(*partitions)
            if not raw_outcome.retry_all and partitioned != requested:
                missing = requested - partitioned
                unexpected = partitioned - requested
                raise ValueError(
                    "DeliveryOutcome must partition every requested target "
                    f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
                )
        return DeliveryOutcome(
            delivered_output_ids=delivered,
            retry_output_ids=retry,
            terminal_output_ids=terminal,
            handled_output_ids=handled,
            retry_all=raw_outcome.retry_all,
        )

    @staticmethod
    def _normalize_requested_output_ids(output_ids: list[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for output_id in output_ids:
            if not isinstance(output_id, str) or not output_id.strip():
                raise TypeError("Requested delivery target IDs must be non-empty strings")
            target_id = output_id.strip().lower()
            if target_id not in seen:
                seen.add(target_id)
                normalized.append(target_id)
        return tuple(normalized)

    @staticmethod
    def _pending_output_id_set(output_ids: list[str] | None) -> set[str]:
        if output_ids is None:
            return {"<all-requested>"}
        try:
            return set(AlertPipeline._normalize_requested_output_ids(output_ids))
        except (TypeError, ValueError):
            return {"<invalid-requested-targets>"}

    @staticmethod
    def _retry_output_id_set(
        outcome: DeliveryOutcome,
        pending_output_ids: list[str] | None,
    ) -> set[str]:
        if outcome.retry_all:
            return AlertPipeline._pending_output_id_set(pending_output_ids)
        return set(outcome.retry_output_ids)

    @staticmethod
    def _alert_id(alert: object) -> object:
        return alert.get("id") if isinstance(alert, Mapping) else None

    def _schedule_retry(self, job: _DeliveryJob, *, delay: float) -> bool:
        with self._retry_condition:
            if self._pending_retries >= self._delivery_retry_queue_size:
                return False
            self._retry_sequence += 1
            self._pending_retries += 1
            job = replace(job, handoff_ready=threading.Event())
            heapq.heappush(
                self._retry_heap,
                _ScheduledRetry(
                    due_at=time.monotonic() + max(0.0, delay),
                    sequence=self._retry_sequence,
                    job=job,
                ),
            )
            self._retry_condition.notify()
            return True

    def _retry_loop(self) -> None:
        while not self._worker_stop.is_set():
            with self._retry_condition:
                while not self._retry_heap and not self._worker_stop.is_set():
                    self._retry_condition.wait(timeout=0.1)
                if self._worker_stop.is_set():
                    return
                scheduled = self._retry_heap[0]
                remaining = scheduled.due_at - time.monotonic()
                if remaining > 0:
                    self._retry_condition.wait(timeout=remaining)
                    continue
                heapq.heappop(self._retry_heap)

            # Keep the retry counted as pending until Queue.put has registered
            # the next attempt, leaving no false-zero window for drain().
            self._delivery_queue.put(scheduled.job)
            with self._retry_condition:
                self._pending_retries -= 1
                if scheduled.job.handoff_ready is not None:
                    scheduled.job.handoff_ready.set()
                self._retry_condition.notify_all()

    @staticmethod
    def _set_future_result(future: Future, value: object) -> None:
        try:
            future.set_result(value)
        except InvalidStateError:
            logger.debug("Persistence Future was already cancelled or completed")

    @staticmethod
    def _set_future_exception(future: Future, error: BaseException) -> None:
        try:
            future.set_exception(error)
        except InvalidStateError:
            logger.debug("Persistence Future was already cancelled or completed")

    def _record_delivery_failure(
        self,
        alert: dict,
        *,
        delivered_output_ids: set[str],
        terminal_output_ids: set[str],
        failed_output_ids: set[str],
        exhausted: bool,
    ) -> None:
        self._increment("delivery_failures")
        if delivered_output_ids:
            self._increment("partially_delivered")
        if terminal_output_ids:
            self._increment("delivery_terminal_failures")
        if exhausted:
            self._increment("delivery_retry_exhausted")
        logger.error(
            "Alert delivery failed permanently",
            extra={
                "alert_id": self._alert_id(alert),
                "delivered_output_ids": sorted(delivered_output_ids),
                "terminal_output_ids": sorted(terminal_output_ids),
                "failed_output_ids": sorted(failed_output_ids),
                "retry_exhausted": exhausted,
            },
        )

    def _increment(self, key: str) -> None:
        with self._stats_lock:
            self._counters[key] += 1

    def _record_persistence_failure(self) -> None:
        with self._stats_lock:
            self._counters["persistence_failures"] += 1
            self._consecutive_persistence_failures += 1
            self._last_persistence_failure_at = time.time()

    def _record_persistence_success(self) -> None:
        with self._stats_lock:
            self._counters["persisted"] += 1
            self._consecutive_persistence_failures = 0
            self._last_persistence_success_at = time.time()
