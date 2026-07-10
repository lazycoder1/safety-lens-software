"""Bounded asynchronous alert persistence and delivery pipeline."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("safetylens.alert_pipeline")


@dataclass(frozen=True)
class _PersistJob:
    payload: dict
    output_ids: list[str] | None
    future: Future


@dataclass(frozen=True)
class _DeliveryJob:
    alert: dict
    output_ids: list[str] | None


_STOP = object()


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
        deliver_alert: Callable[[dict, list[str] | None], None],
        on_persisted: Callable[[dict], None] | None = None,
        persist_queue_size: int = 256,
        delivery_queue_size: int = 256,
        delivery_workers: int = 4,
        submit_timeout: float = 0.01,
        persistence_attempts: int = 3,
        retry_delay: float = 0.1,
    ):
        if persist_queue_size < 1 or delivery_queue_size < 1:
            raise ValueError("Alert queue sizes must be positive")
        if delivery_workers < 1:
            raise ValueError("At least one alert delivery worker is required")
        if persistence_attempts < 1:
            raise ValueError("At least one persistence attempt is required")

        self._persist_alert = persist_alert
        self._deliver_alert = deliver_alert
        self._on_persisted = on_persisted
        self._delivery_worker_count = delivery_workers
        self._submit_timeout = max(0.0, submit_timeout)
        self._persistence_attempts = persistence_attempts
        self._retry_delay = max(0.0, retry_delay)

        self._persist_queue: queue.Queue = queue.Queue(maxsize=persist_queue_size)
        self._delivery_queue: queue.Queue = queue.Queue(maxsize=delivery_queue_size)
        self._lifecycle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._running = False
        self._persist_thread: threading.Thread | None = None
        self._delivery_threads: list[threading.Thread] = []
        self._counters = {
            "submitted": 0,
            "persisted": 0,
            "persistence_failures": 0,
            "callback_failures": 0,
            "delivered": 0,
            "delivery_failures": 0,
            "backpressure_events": 0,
        }

    def start(self) -> None:
        """Start workers. Safe to call repeatedly."""
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            self._delivery_threads = [
                threading.Thread(
                    target=self._delivery_loop,
                    name=f"alert-delivery-{index + 1}",
                    daemon=True,
                )
                for index in range(self._delivery_worker_count)
            ]
            for worker in self._delivery_threads:
                worker.start()
            self._persist_thread = threading.Thread(
                target=self._persistence_loop,
                name="alert-persistence",
                daemon=True,
            )
            self._persist_thread.start()

    def submit(self, payload: dict, *, output_ids: list[str] | None = None) -> Future:
        """Queue an immutable alert request and return its persistence future."""
        self.start()
        future = Future()
        job = _PersistJob(
            payload=copy.deepcopy(payload),
            output_ids=list(output_ids) if output_ids is not None else None,
            future=future,
        )
        try:
            self._persist_queue.put(job, timeout=self._submit_timeout)
        except queue.Full:
            self._increment("backpressure_events")
            logger.warning(
                "Alert persistence queue full; applying producer backpressure",
                extra={"queue_size": self._persist_queue.qsize()},
            )
            self._persist_queue.put(job)
        self._increment("submitted")
        return future

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until queued persistence and delivery work is complete."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() <= deadline:
            if self._persist_queue.unfinished_tasks == 0 and self._delivery_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def shutdown(self, *, wait: bool = True, timeout: float = 10.0) -> bool:
        """Drain and stop workers. Returns whether all work drained in time."""
        with self._lifecycle_lock:
            if not self._running:
                return True

        drained = self.drain(timeout) if wait else False
        self._persist_queue.put(_STOP)
        for _ in self._delivery_threads:
            self._delivery_queue.put(_STOP)

        deadline = time.monotonic() + max(0.0, timeout)
        persist_thread = self._persist_thread
        if persist_thread is not None:
            persist_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for worker in self._delivery_threads:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._lifecycle_lock:
            self._running = False
            self._persist_thread = None
            self._delivery_threads = []
        return drained

    def stats(self) -> dict:
        """Return counters and queue depths for diagnostics."""
        with self._stats_lock:
            result = dict(self._counters)
        with self._lifecycle_lock:
            result["running"] = self._running
        result.update({
            "persist_queue_depth": self._persist_queue.qsize(),
            "delivery_queue_depth": self._delivery_queue.qsize(),
        })
        return result

    def _persistence_loop(self) -> None:
        while True:
            job = self._persist_queue.get()
            try:
                if job is _STOP:
                    return
                self._persist(job)
            finally:
                self._persist_queue.task_done()

    def _persist(self, job: _PersistJob) -> None:
        alert = None
        last_error: Exception | None = None
        for attempt in range(1, self._persistence_attempts + 1):
            try:
                alert = self._persist_alert(**job.payload)
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
            self._increment("persistence_failures")
            if last_error is None:
                last_error = RuntimeError("Alert persistence returned no alert")
            job.future.set_exception(last_error)
            logger.error(
                "Alert persistence failed permanently: %s",
                last_error,
                extra={"attempts": self._persistence_attempts},
            )
            return

        self._increment("persisted")
        if self._on_persisted is not None:
            try:
                self._on_persisted(alert)
            except Exception:
                self._increment("callback_failures")
                logger.exception("Post-persistence alert callback failed")

        self._delivery_queue.put(_DeliveryJob(alert=alert, output_ids=job.output_ids))
        job.future.set_result(alert)

    def _delivery_loop(self) -> None:
        while True:
            job = self._delivery_queue.get()
            try:
                if job is _STOP:
                    return
                try:
                    self._deliver_alert(job.alert, job.output_ids)
                    self._increment("delivered")
                except Exception:
                    self._increment("delivery_failures")
                    logger.exception("Alert delivery failed")
            finally:
                self._delivery_queue.task_done()

    def _increment(self, key: str) -> None:
        with self._stats_lock:
            self._counters[key] += 1
