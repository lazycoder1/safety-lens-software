"""Restart-safe workers for the PostgreSQL alert delivery outbox."""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
from pathlib import Path
from uuid import uuid4

import alert_delivery_store
import alert_store
import notification_dispatcher
from delivery_result import DeliveryDisposition, ProviderDeliveryResult


logger = logging.getLogger("safetylens.alert_delivery")


class AlertDeliveryWorker:
    """One low-frequency claimer feeding isolated, lease-renewed send workers."""

    def __init__(
        self,
        *,
        worker_count: int = 3,
        lease_seconds: float = 60.0,
        poll_seconds: float = 5.0,
        max_attempts: int = 5,
        retry_base_seconds: float = 1.0,
        retry_cap_seconds: float = 60.0,
        rng=None,
    ) -> None:
        if worker_count < 1 or max_attempts < 1:
            raise ValueError("delivery worker and attempt counts must be positive")
        self._worker_count = worker_count
        self._lease_seconds = max(0.3, float(lease_seconds))
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._max_attempts = max_attempts
        self._retry_base_seconds = max(0.0, float(retry_base_seconds))
        self._retry_cap_seconds = max(self._retry_base_seconds, float(retry_cap_seconds))
        self._rng = rng or random.uniform
        self._instance_id = f"{os.getpid()}-{uuid4()}"

        self._stop_claiming = threading.Event()
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._dispatch_queue: queue.Queue = queue.Queue(maxsize=worker_count)
        self._active_leases: set[tuple[str, str]] = set()
        self._channel_inflight: dict[str, int] = {}
        # A stuck webhook must not occupy every provider worker. The default
        # three-worker deployment reserves capacity across the three channels.
        self._channel_limits = {"webhook": 1, "telegram": 1, "email": 1}

        self._threads: list[threading.Thread] = []
        self._claimer: threading.Thread | None = None
        self._renewer: threading.Thread | None = None
        self._running = False
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "claim_errors": 0,
            "renewal_failures": 0,
            "fencing_failures": 0,
            "last_claim_at": None,
            "last_delivery_at": None,
        }
        self._stats_cache: dict | None = None
        self._stats_cache_at = 0.0

    def start(self) -> None:
        with self._lifecycle_lock:
            self._reset_if_dead_locked()
            if self._running:
                return
            alert_delivery_store.init_db()
            self._stop_claiming.clear()
            self._dispatch_queue = queue.Queue(maxsize=self._worker_count)
            self._threads = [
                threading.Thread(
                    target=self._delivery_loop,
                    name=f"alert-outbox-send-{index + 1}",
                    daemon=True,
                )
                for index in range(self._worker_count)
            ]
            self._claimer = threading.Thread(
                target=self._claim_loop,
                name="alert-outbox-claimer",
                daemon=True,
            )
            self._renewer = threading.Thread(
                target=self._renew_loop,
                name="alert-outbox-lease-renewer",
                daemon=True,
            )
            all_threads = [self._renewer, *self._threads, self._claimer]
            started: list[threading.Thread] = []
            self._running = True
            try:
                for thread in all_threads:
                    thread.start()
                    started.append(thread)
            except Exception:
                self._stop_claiming.set()
                self._notify_waiters()
                for thread in started:
                    thread.join(timeout=0.5)
                self._running = False
                self._threads = []
                self._claimer = None
                self._renewer = None
                raise

    def wake(self) -> None:
        self._invalidate_stats()
        self._notify_waiters()

    def _notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def stop(self, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lifecycle_lock:
            self._reset_if_dead_locked()
            if not self._running:
                return True
            self._stop_claiming.set()
            self._notify_waiters()
            claimer = self._claimer
            workers = list(self._threads)
            renewer = self._renewer

        if claimer is not None and claimer.is_alive():
            claimer.join(timeout=max(0.0, deadline - time.monotonic()))
        for thread in workers:
            if thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        # The renewer intentionally outlives a timed-out shutdown while a
        # provider call is still active. It self-terminates after the last send.
        with self._active_lock:
            has_active = bool(self._active_leases)
        if not has_active and renewer is not None and renewer.is_alive():
            self._notify_waiters()
            renewer.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._lifecycle_lock:
            self._reset_if_dead_locked()
            return not self._running

    def stats(self) -> dict:
        now = time.monotonic()
        if self._stats_cache is not None and now - self._stats_cache_at < 1.0:
            durable = dict(self._stats_cache)
        else:
            try:
                durable = alert_delivery_store.get_stats()
                self._stats_cache = dict(durable)
                self._stats_cache_at = now
            except Exception:
                logger.error("Could not read durable alert-delivery statistics")
                durable = {
                    "pending": None,
                    "due": None,
                    "scheduled": None,
                    "leased": None,
                    "expired_leases": None,
                    "delivered": None,
                    "terminal": None,
                    "cancelled": None,
                    "ambiguous_history": None,
                    "oldest_due_age_seconds": None,
                    "oldest_pending_age_seconds": None,
                    "database_error": True,
                }
        with self._lifecycle_lock:
            self._reset_if_dead_locked()
            running = self._running
            workers_alive = sum(thread.is_alive() for thread in self._threads)
            claimer_alive = bool(self._claimer and self._claimer.is_alive())
            renewer_alive = bool(self._renewer and self._renewer.is_alive())
        with self._active_lock:
            active = len(self._active_leases)
            channel_inflight = dict(self._channel_inflight)
        with self._metrics_lock:
            metrics = dict(self._metrics)
        return {
            **durable,
            **metrics,
            "running": running,
            "workers_alive": workers_alive,
            "claimer_alive": claimer_alive,
            "renewer_alive": renewer_alive,
            "active_sends": active,
            "channel_inflight": channel_inflight,
        }

    def _claim_loop(self) -> None:
        failure_streak = 0
        next_error_log_at = 0.0
        try:
            while not self._stop_claiming.is_set():
                if self._dispatch_queue.full():
                    self._wait(0.1)
                    continue
                claimed = None
                try:
                    with self._active_lock:
                        excluded = [
                            channel
                            for channel, limit in self._channel_limits.items()
                            if self._channel_inflight.get(channel, 0) >= limit
                        ]
                    claimed = alert_delivery_store.claim_due(
                        worker_id=self._instance_id,
                        lease_seconds=self._lease_seconds,
                        excluded_channels=excluded,
                    )
                    failure_streak = 0
                    if claimed is None:
                        self._wait(self._poll_seconds)
                        continue

                    delivery_id = str(claimed["id"])
                    lease_token = str(claimed["lease_token"])
                    channel = str(claimed.get("channel") or "unknown")
                    with self._active_lock:
                        self._active_leases.add((delivery_id, lease_token))
                        self._channel_inflight[channel] = self._channel_inflight.get(channel, 0) + 1
                    self._set_metric("last_claim_at", time.time())
                    self._invalidate_stats()
                    while not self._stop_claiming.is_set():
                        try:
                            self._dispatch_queue.put(claimed, timeout=0.1)
                            claimed = None
                            break
                        except queue.Full:
                            continue
                    if claimed is not None:
                        # Shutdown won after the lease was claimed. Preserve the
                        # work for restart without counting a provider attempt.
                        self._release_claim(claimed, error_code="shutdown_before_send")
                except Exception as exc:
                    if claimed is not None:
                        self._release_claim(claimed, error_code="claim_handoff_error")
                    failure_streak += 1
                    self._increment_metric("claim_errors")
                    now = time.monotonic()
                    if now >= next_error_log_at:
                        logger.error(
                            "Alert outbox claim failed; backing off",
                            extra={
                                "error_phase": "claim",
                                "error_type": type(exc).__name__,
                                "failure_count": failure_streak,
                            },
                        )
                        next_error_log_at = now + 30.0
                    cap = min(10.0, 0.25 * (2 ** min(failure_streak - 1, 6)))
                    delay = max(0.05, float(self._rng(cap / 2, cap)))
                    self._wait(delay)
        finally:
            self._notify_waiters()

    def _delivery_loop(self) -> None:
        try:
            while True:
                if self._stop_claiming.is_set() and self._dispatch_queue.empty():
                    return
                try:
                    delivery = self._dispatch_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._deliver_claimed(delivery)
                except Exception as exc:
                    logger.error(
                        "Outbox worker failed unexpectedly",
                        extra={
                            "delivery_id": delivery.get("id"),
                            "alert_id": delivery.get("alert_id"),
                            "error_phase": "delivery",
                            "error_type": type(exc).__name__,
                        },
                    )
                    self._recover_unexpected_failure(
                        delivery,
                        provider_started=bool(delivery.get("_provider_started")),
                    )
                finally:
                    self._complete_active(delivery)
                    self._dispatch_queue.task_done()
                    self._notify_waiters()
        finally:
            self._notify_waiters()

    def _deliver_claimed(self, delivery: dict) -> None:
        delivery_id = str(delivery["id"])
        lease_token = str(delivery["lease_token"])
        alert = alert_store.get_alert(str(delivery["alert_id"]))
        if alert is None:
            self._terminal(delivery, "alert_missing", "Persisted alert no longer exists")
            return

        snapshot_path = None
        snapshot_url = alert.get("snapshotUrl")
        if snapshot_url:
            snapshot_path = str(alert_store.SNAPSHOTS_DIR / Path(snapshot_url).name)

        began = alert_delivery_store.begin_send(
            delivery_id,
            lease_token,
            max_attempts=self._max_attempts,
            lease_seconds=self._lease_seconds,
        )
        if not began.get("started"):
            logger.info(
                "Alert target was fenced before provider dispatch",
                extra={
                    "delivery_id": delivery_id,
                    "alert_id": delivery.get("alert_id"),
                    "channel": delivery.get("channel"),
                    "error_code": began.get("reason"),
                },
            )
            return
        delivery["_provider_started"] = True
        delivery = dict(delivery)
        delivery.update(began)

        result = notification_dispatcher.deliver_outbox_target(
            alert,
            delivery,
            snapshot_path,
        )
        if not isinstance(result, ProviderDeliveryResult):
            raise TypeError("Outbox dispatcher must return ProviderDeliveryResult")
        self._record_result(delivery, result)

    def _record_result(self, delivery: dict, result: ProviderDeliveryResult) -> None:
        delivery_id = str(delivery["id"])
        lease_token = str(delivery["lease_token"])
        attempt = int(delivery.get("attempt_count") or 1)
        common_log = {
            "delivery_id": delivery_id,
            "alert_id": delivery.get("alert_id"),
            "channel": delivery.get("channel"),
            "attempt": attempt,
            "error_code": result.error_code,
            "provider_status": result.provider_status,
            "acceptance_unknown": result.acceptance_unknown,
            "retry_after_seconds": result.retry_after_seconds,
        }
        if result.success:
            updated = alert_delivery_store.mark_delivered(delivery_id, lease_token)
            self._log_fenced(updated, "Alert target delivered", common_log)
            if updated:
                self._set_metric("last_delivery_at", time.time())
            self._invalidate_stats()
            return
        if result.disposition in {DeliveryDisposition.TERMINAL, DeliveryDisposition.SKIPPED}:
            updated = alert_delivery_store.mark_terminal(
                delivery_id,
                lease_token,
                error_code=result.error_code or "terminal",
                error_message=result.message,
                acceptance_unknown=result.acceptance_unknown,
            )
            self._log_fenced(updated, "Alert target failed terminally", common_log, error=True)
            self._invalidate_stats()
            return

        local_cap = min(
            self._retry_cap_seconds,
            self._retry_base_seconds * (2 ** min(attempt - 1, 10)),
        )
        local_delay = float(self._rng(0.0, local_cap)) if local_cap > 0 else 0.0
        delay = max(local_delay, float(result.retry_after_seconds or 0.0))
        state = alert_delivery_store.schedule_retry(
            delivery_id,
            lease_token,
            delay_seconds=delay,
            max_attempts=self._max_attempts,
            error_code=result.error_code or "retryable",
            error_message=result.message,
            acceptance_unknown=result.acceptance_unknown,
        )
        common_log["retry_after_seconds"] = delay
        if state == "pending":
            self._log_fenced(True, "Alert target scheduled for retry", common_log)
            self.wake()
        elif state in {"terminal", "cancelled"}:
            common_log["error_code"] = (
                "retry_budget_exhausted" if state == "terminal" else "alert_inactive"
            )
            self._log_fenced(
                True,
                "Alert target retry ended without another attempt",
                common_log,
                error=state == "terminal",
            )
        else:
            self._log_fenced(False, "Alert target retry lost lease fencing", common_log)
        self._invalidate_stats()

    def _terminal(
        self,
        delivery: dict,
        code: str,
        message: str,
        *,
        terminal_reason: str | None = None,
    ) -> None:
        updated = alert_delivery_store.mark_terminal(
            str(delivery["id"]),
            str(delivery["lease_token"]),
            error_code=code,
            error_message=message,
            terminal_reason=terminal_reason,
        )
        self._log_fenced(
            updated,
            "Alert target ended before provider dispatch",
            {
                "delivery_id": str(delivery["id"]),
                "alert_id": delivery.get("alert_id"),
                "channel": delivery.get("channel"),
                "error_code": code,
            },
            error=code != "alert_inactive",
        )
        self._invalidate_stats()

    def _recover_unexpected_failure(self, delivery: dict, *, provider_started: bool) -> None:
        try:
            state = alert_delivery_store.schedule_retry(
                str(delivery["id"]),
                str(delivery["lease_token"]),
                delay_seconds=self._retry_base_seconds,
                max_attempts=self._max_attempts,
                error_code="worker_error",
                error_message="Delivery worker failed unexpectedly",
                acceptance_unknown=provider_started,
            ) if provider_started else alert_delivery_store.schedule_internal_retry(
                str(delivery["id"]),
                str(delivery["lease_token"]),
                delay_seconds=self._retry_base_seconds,
                max_claims=self._max_attempts,
            )
            if state == "pending":
                self.wake()
        except Exception as exc:
            logger.error(
                "Could not release failed outbox lease",
                extra={
                    "delivery_id": delivery.get("id"),
                    "alert_id": delivery.get("alert_id"),
                    "error_phase": "recovery",
                    "error_type": type(exc).__name__,
                },
            )

    def _release_claim(self, delivery: dict, *, error_code: str) -> None:
        try:
            if error_code == "shutdown_before_send":
                alert_delivery_store.release_unstarted_claim(
                    str(delivery["id"]),
                    str(delivery["lease_token"]),
                    error_code=error_code,
                )
            else:
                alert_delivery_store.schedule_internal_retry(
                    str(delivery["id"]),
                    str(delivery["lease_token"]),
                    delay_seconds=self._retry_base_seconds,
                    max_claims=self._max_attempts,
                    error_code=error_code,
                )
        except Exception:
            logger.error(
                "Could not release an unstarted outbox claim",
                extra={"delivery_id": delivery.get("id"), "error_phase": "claim_handoff"},
            )
        finally:
            self._complete_active(delivery)

    def _complete_active(self, delivery: dict) -> None:
        delivery_id = str(delivery.get("id"))
        lease_token = str(delivery.get("lease_token"))
        channel = str(delivery.get("channel") or "unknown")
        with self._active_lock:
            # Exact tuple removal prevents a stale attempt from deleting the
            # renewal record for a newer lease of the same delivery row.
            self._active_leases.discard((delivery_id, lease_token))
            remaining = self._channel_inflight.get(channel, 0) - 1
            if remaining > 0:
                self._channel_inflight[channel] = remaining
            else:
                self._channel_inflight.pop(channel, None)

    def _renew_loop(self) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while True:
            with self._active_lock:
                leases = list(self._active_leases)
            if self._stop_claiming.is_set() and not leases:
                return
            self._wait(interval)
            with self._active_lock:
                leases = list(self._active_leases)
            for delivery_id, lease_token in leases:
                try:
                    renewed = alert_delivery_store.renew_lease(
                        delivery_id,
                        lease_token,
                        lease_seconds=self._lease_seconds,
                    )
                    if not renewed:
                        self._increment_metric("renewal_failures")
                        logger.warning(
                            "Outbox lease was cancelled or lost",
                            extra={"delivery_id": delivery_id},
                        )
                except Exception as exc:
                    self._increment_metric("renewal_failures")
                    logger.error(
                        "Outbox lease renewal failed",
                        extra={
                            "delivery_id": delivery_id,
                            "error_phase": "lease_renewal",
                            "error_type": type(exc).__name__,
                        },
                    )

    def _wait(self, seconds: float) -> None:
        with self._condition:
            self._condition.wait(timeout=max(0.0, seconds))

    def _invalidate_stats(self) -> None:
        self._stats_cache_at = 0.0

    def _increment_metric(self, key: str) -> None:
        with self._metrics_lock:
            self._metrics[key] = int(self._metrics.get(key) or 0) + 1

    def _set_metric(self, key: str, value) -> None:
        with self._metrics_lock:
            self._metrics[key] = value

    def _reset_if_dead_locked(self) -> None:
        all_threads = [*self._threads]
        if self._claimer is not None:
            all_threads.append(self._claimer)
        if self._renewer is not None:
            all_threads.append(self._renewer)
        if self._running and all_threads and any(thread.is_alive() for thread in all_threads):
            return
        self._running = False
        self._threads = []
        self._claimer = None
        self._renewer = None

    def _log_fenced(
        self,
        updated: bool,
        message: str,
        context: dict,
        *,
        error: bool = False,
    ) -> None:
        if not updated:
            self._increment_metric("fencing_failures")
            logger.warning("Outbox completion lost lease fencing", extra=context)
            return
        (logger.error if error else logger.info)(message, extra=context)


_worker = AlertDeliveryWorker(
    worker_count=int(os.getenv("ALERT_OUTBOX_WORKERS", "3")),
    lease_seconds=float(os.getenv("ALERT_OUTBOX_LEASE_SECONDS", "60")),
    poll_seconds=float(os.getenv("ALERT_OUTBOX_POLL_SECONDS", "5")),
    max_attempts=int(os.getenv("ALERT_DELIVERY_ATTEMPTS", "5")),
    retry_base_seconds=float(os.getenv("ALERT_DELIVERY_RETRY_DELAY_SECONDS", "1.0")),
    retry_cap_seconds=float(os.getenv("ALERT_DELIVERY_RETRY_CAP_SECONDS", "60")),
)


def start() -> None:
    _worker.start()


def wake() -> None:
    _worker.wake()


def stop(timeout: float = 20.0) -> bool:
    return _worker.stop(timeout)


def stats() -> dict:
    return _worker.stats()
