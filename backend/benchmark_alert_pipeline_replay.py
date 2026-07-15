#!/usr/bin/env python3
"""Benchmark the real alert persistence/outbox/provider path on an isolated DB.

This executable is copied into the edge image with the rest of ``backend/``.
Run it as the only process connected to a disposable clone, normally by
overriding the command in a one-shot container rather than using ``docker
exec`` in a serving edge container. It deliberately emits aggregate telemetry
only: no alert IDs, payloads, destinations, database names, URLs, credentials,
or snapshots enter the report.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import UUID, uuid4


SCHEMA_VERSION = 1
ISOLATION_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_ISOLATED_BENCHMARK_DATABASE"
DEFAULT_DATABASE_NAME_TOKEN = "benchmark"
MIN_ALERT_COUNT = 100
MAX_ALERT_COUNT = 2_000
MAX_SINK_BODY_BYTES = 64 * 1024
CONFIG_ID = "default"
CONFIG_SECTION_KEYS = (
    "alert_outputs",
    "alert_routing",
    "webhook",
    "telegram",
    "email",
)
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
DATABASE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")


class BenchmarkSafetyError(RuntimeError):
    """Raised before mutation when an isolation invariant is not satisfied."""


class BenchmarkRuntimeError(RuntimeError):
    """Raised when the real pipeline cannot produce a complete measurement."""


def _connect_database(database_url: str):
    import psycopg2

    return psycopg2.connect(database_url)


def _require_guarded_database_name(
    connection,
    *,
    database_name_token: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        row = cursor.fetchone()
    database_name = str(row[0] if row else "")
    if database_name_token.casefold() not in database_name.casefold():
        raise BenchmarkSafetyError("database name does not contain the guard token")


def _acquire_benchmark_lock(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtext(%s))",
            ("safetylens-alert-pipeline-benchmark",),
        )
        lock_row = cursor.fetchone()
    if not lock_row or lock_row[0] is not True:
        raise BenchmarkSafetyError("another alert benchmark owns the database lock")


def _require_exclusive_clone_client(connection) -> None:
    """Reject a clone that has any other connected application process."""

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM pg_catalog.pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
              AND backend_type = 'client backend'
            """
        )
        row = cursor.fetchone()
    if row is None or int(row[0]) != 0:
        raise BenchmarkSafetyError(
            "isolated benchmark database has another connected client"
        )


def _validate_isolation_inputs(
    *,
    acknowledgement: str,
    database_name_token: str,
    config_store: str,
) -> None:
    if acknowledgement != ISOLATION_ACKNOWLEDGEMENT:
        raise BenchmarkSafetyError("explicit isolation acknowledgement is required")
    if DATABASE_TOKEN_RE.fullmatch(database_name_token) is None:
        raise BenchmarkSafetyError("database name guard token is invalid")
    if database_name_token.casefold() != DEFAULT_DATABASE_NAME_TOKEN:
        raise BenchmarkSafetyError("database name guard token is immutable")
    if config_store.strip().lower() != "postgres":
        raise BenchmarkSafetyError("PostgreSQL must be the authoritative config store")


def _verify_isolated_database(connection, *, database_name_token: str) -> dict:
    """Read and lock the target only after its authoritative name passes."""

    _require_guarded_database_name(
        connection,
        database_name_token=database_name_token,
    )
    _acquire_benchmark_lock(connection)
    _require_exclusive_clone_client(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT config FROM public.app_config WHERE id = %s",
            (CONFIG_ID,),
        )
        config_row = cursor.fetchone()
        cursor.execute("SELECT to_regclass('public.alert_delivery_outbox')")
        outbox_table = cursor.fetchone()
        if outbox_table and outbox_table[0] is not None:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM public.alert_delivery_outbox
                WHERE state = 'pending'
                """
            )
            pending_row = cursor.fetchone()
            if pending_row and int(pending_row[0]) > 0:
                raise BenchmarkSafetyError(
                    "isolated database contains pre-existing pending alert work"
                )
    if not config_row:
        raise BenchmarkSafetyError("isolated database has no authoritative app config")
    raw_config = config_row[0]
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except json.JSONDecodeError as exc:
            raise BenchmarkSafetyError("isolated app config is invalid") from exc
    if not isinstance(raw_config, dict):
        raise BenchmarkSafetyError("isolated app config is invalid")
    return copy.deepcopy(raw_config)


def _original_config_sections(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(config[key]) for key in CONFIG_SECTION_KEYS if key in config
    }


def _benchmark_config_sections(
    config: Mapping[str, Any],
    *,
    loopback_url: str,
) -> dict[str, Any]:
    """Return only provider/routing sections that the benchmark may replace."""

    outputs: list[dict[str, Any]] = []
    webhook_seen = False
    raw_outputs = config.get("alert_outputs", [])
    if isinstance(raw_outputs, list):
        for raw_output in raw_outputs:
            if not isinstance(raw_output, dict):
                continue
            output = copy.deepcopy(raw_output)
            output_type = str(output.get("type") or "").strip().lower()
            output_id = str(output.get("id") or "").strip().lower()
            is_webhook = output_type == "webhook" or output_id == "webhook"
            output["enabled"] = is_webhook
            if is_webhook:
                webhook_seen = True
                output["id"] = "webhook"
                output["type"] = "webhook"
                output["severities"] = ["P1", "P2", "P3", "P4"]
                output["zones"] = []
                output["settings"] = {
                    "url": loopback_url,
                    "headers": {},
                    "include_snapshot": False,
                }
            outputs.append(output)
    if not webhook_seen:
        outputs.append(
            {
                "id": "webhook",
                "name": "Isolated Benchmark Loopback",
                "type": "webhook",
                "enabled": True,
                "severities": ["P1", "P2", "P3", "P4"],
                "zones": [],
                "mode": "http",
                "status": "ready",
                "lastTestAt": None,
                "lastFiredAt": None,
                "lastError": "",
                "settings": {
                    "url": loopback_url,
                    "headers": {},
                    "include_snapshot": False,
                },
            }
        )

    routing = copy.deepcopy(config.get("alert_routing", {}))
    if not isinstance(routing, dict):
        routing = {}
    routing["channel_matrix"] = {
        severity: {
            "inApp": False,
            "telegram": False,
            "email": False,
            "webhook": True,
            "whatsapp": False,
            "sms": False,
            "plc": False,
        }
        for severity in ("P1", "P2", "P3", "P4")
    }
    routing["escalation_steps"] = []

    return {
        "alert_outputs": outputs,
        "alert_routing": routing,
        "webhook": {
            "enabled": True,
            "url": loopback_url,
            "account_id": "isolated-alert-benchmark",
            "headers": {},
            "severities": ["P1", "P2", "P3", "P4"],
            "include_snapshot": False,
        },
        "telegram": {"enabled": False},
        "email": {"enabled": False},
    }


def _write_config_sections(connection, sections: Mapping[str, Any]) -> None:
    from psycopg2.extras import Json

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE public.app_config
            SET config = (config - %s::text[]) || %s::jsonb,
                updated_at = NOW()
            WHERE id = %s
            """,
            (list(CONFIG_SECTION_KEYS), Json(dict(sections)), CONFIG_ID),
        )
        if cursor.rowcount != 1:
            raise BenchmarkSafetyError("authoritative app config disappeared")
    connection.commit()


def _restore_config_sections(
    connection,
    sections: Mapping[str, Any],
    *,
    reconnect: Callable[[], Any],
) -> None:
    """Restore only owned sections, retrying on a fresh connection once."""

    try:
        _write_config_sections(connection, sections)
        return
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
    replacement = reconnect()
    try:
        _write_config_sections(replacement, sections)
    except Exception as exc:
        try:
            replacement.rollback()
        except Exception:
            pass
        raise BenchmarkSafetyError(
            "isolated alert configuration restore failed"
        ) from exc
    finally:
        replacement.close()


@contextlib.contextmanager
def _temporary_alert_configuration(
    connection,
    *,
    original_sections: Mapping[str, Any],
    benchmark_sections: Mapping[str, Any],
    reconnect: Callable[[], Any],
):
    try:
        _write_config_sections(connection, benchmark_sections)
        yield
    finally:
        _restore_config_sections(
            connection,
            original_sections,
            reconnect=reconnect,
        )


@contextlib.contextmanager
def _temporary_environment(updates: Mapping[str, str]):
    original = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _LoopbackSink:
    """Bounded local HTTP provider that retains receipt aggregates only."""

    def __init__(self) -> None:
        self._path = f"/alert-benchmark/{uuid4().hex}"
        self._lock = threading.Lock()
        self._received_ns: dict[str, int] = {}
        self._request_count = 0
        self._duplicate_count = 0
        self._malformed_count = 0
        self._oversize_count = 0
        self._snapshot_payload_count = 0
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path != sink._path:
                    self.send_error(404)
                    return
                raw_length = self.headers.get("Content-Length", "")
                try:
                    length = int(raw_length)
                except (TypeError, ValueError):
                    length = -1
                if length < 0 or length > MAX_SINK_BODY_BYTES:
                    with sink._lock:
                        sink._oversize_count += 1
                    self.send_error(413)
                    return
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                    alert = payload.get("alert") if isinstance(payload, dict) else None
                    alert_id = alert.get("id") if isinstance(alert, dict) else None
                    alert_id = str(UUID(str(alert_id)))
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                    with sink._lock:
                        sink._malformed_count += 1
                    self.send_error(400)
                    return
                received_ns = time.monotonic_ns()
                with sink._lock:
                    sink._request_count += 1
                    if "snapshot_base64" in payload:
                        sink._snapshot_payload_count += 1
                    if alert_id in sink._received_ns:
                        sink._duplicate_count += 1
                    else:
                        sink._received_ns[alert_id] = received_ns
                self.send_response(204)
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="alert-benchmark-loopback",
            daemon=True,
        )

    @property
    def url(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}{self._path}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "_LoopbackSink":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()

    def receipts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._received_ns)

    def aggregate(self) -> dict[str, int]:
        with self._lock:
            return {
                "acceptedRequests": self._request_count,
                "uniqueAlerts": len(self._received_ns),
                "duplicateRequests": self._duplicate_count,
                "malformedRequests": self._malformed_count,
                "oversizeRequests": self._oversize_count,
                "snapshotPayloadRequests": self._snapshot_payload_count,
            }


def _prime_process_config(config_manager, config: Mapping[str, Any]) -> None:
    """Install the already-guarded DB generation without normalizing/saving it."""

    with config_manager._lock:
        config_manager._config = copy.deepcopy(dict(config))
        config_manager._config_version = None
        config_manager._config_checked_at = time.monotonic()


def _query_run_counts(db, run_key: str) -> dict[str, Any]:
    with db.get_conn() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (
                           WHERE snapshot_path IS NOT NULL
                              OR clean_snapshot_path IS NOT NULL
                       )
                FROM public.alerts
                WHERE metadata ->> 'benchmarkRunKey' = %s
                """,
                (run_key,),
            )
            alert_row = cursor.fetchone() or (0, 0)
            cursor.execute(
                """
                SELECT delivery.state, COUNT(*)
                FROM public.alert_delivery_outbox AS delivery
                JOIN public.alerts AS alert ON alert.id = delivery.alert_id
                WHERE alert.metadata ->> 'benchmarkRunKey' = %s
                GROUP BY delivery.state
                """,
                (run_key,),
            )
            state_rows = cursor.fetchall()
    states: dict[str, int] = {}
    for raw_state, raw_count in state_rows:
        state = str(raw_state)
        if state not in {"pending", "delivered", "terminal", "cancelled"}:
            state = "unknown"
        states[state] = states.get(state, 0) + int(raw_count)
    return {
        "alertsPersisted": int(alert_row[0]),
        "alertsWithSnapshots": int(alert_row[1]),
        "outboxTotal": sum(states.values()),
        "outboxByState": dict(sorted(states.items())),
    }


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return round(ordered[rank - 1], 6)


def _latency_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"sampleCount": 0, "p95Ms": None, "p99Ms": None, "maxMs": None}
    return {
        "sampleCount": len(values),
        "p95Ms": _nearest_rank(values, 0.95),
        "p99Ms": _nearest_rank(values, 0.99),
        "maxMs": round(max(values), 6),
    }


def _has_exact_sample_count(values: Sequence[object], expected_count: int) -> bool:
    return len(values) == expected_count


def _stop_runtime_workers(
    pipeline,
    worker,
    *,
    pipeline_started: bool,
    worker_started: bool,
    timeout_seconds: float = 30.0,
) -> None:
    pipeline_stopped = not pipeline_started
    worker_stopped = not worker_started
    if pipeline_started:
        try:
            pipeline_stopped = pipeline.shutdown(
                wait=True,
                timeout=timeout_seconds,
            )
        except Exception:
            pipeline_stopped = False
    if worker_started:
        try:
            worker_stopped = worker.stop(timeout_seconds)
        except Exception:
            worker_stopped = False
    if not pipeline_stopped or not worker_stopped:
        raise BenchmarkRuntimeError("alert benchmark workers failed to stop")


def _cleanup_runtime_resources(
    pipeline,
    worker,
    *,
    pipeline_started: bool,
    worker_started: bool,
    alert_timing,
    db,
    db_cleanup_owned: bool,
) -> None:
    """Attempt every cleanup action before surfacing a shutdown failure."""

    shutdown_error: BaseException | None = None
    try:
        _stop_runtime_workers(
            pipeline,
            worker,
            pipeline_started=pipeline_started,
            worker_started=worker_started,
        )
    except BaseException as exc:
        shutdown_error = exc
    finally:
        try:
            alert_timing.registry.clear()
        finally:
            if db_cleanup_owned:
                db.close_pool()
    if shutdown_error is not None:
        raise shutdown_error


def _register_timing(alert_timing, alert_id: str, targets: Sequence[Mapping[str, Any]]):
    first_positive_ns = time.monotonic_ns()
    confirmed_ns = time.monotonic_ns()
    initial_keys = [
        str(target["target_key"])
        for target in targets
        if target.get("kind", "initial") == "initial"
        and target.get("channel")
        and target.get("target_key")
    ]
    evicted = alert_timing.registry.remember(
        alert_id,
        first_positive_ns=first_positive_ns,
        confirmed_ns=confirmed_ns,
        initial_target_keys=initial_keys,
    )
    if evicted:
        raise BenchmarkRuntimeError("alert timing registry capacity was exceeded")
    return first_positive_ns, confirmed_ns


def _run_real_pipeline(
    *,
    alert_count: int,
    timeout_seconds: float,
    worker_count: int,
    run_key: str,
    sink: _LoopbackSink,
    configured_app_config: Mapping[str, Any],
) -> dict[str, Any]:
    import alert_delivery_worker
    import alert_store
    import alert_timing
    import config_manager
    import db
    import notification_dispatcher
    import video_processing
    from alert_pipeline import AlertPipeline, DeliveryOutcome
    from pipeline_telemetry import telemetry as pipeline_telemetry

    _prime_process_config(config_manager, configured_app_config)
    telemetry_before = pipeline_telemetry.public_snapshot()["alerts"]
    if any(
        int(histogram.get("count") or 0)
        for histogram in telemetry_before["latency"].values()
    ) or int(telemetry_before["deliveryCoverage"].get("pending") or 0):
        raise BenchmarkRuntimeError("alert replay requires a fresh process")

    worker = None
    pipeline = None
    started = 0.0
    deadline = 0.0
    futures = []
    first_positive_by_alert: dict[str, int] = {}
    db_counts: dict[str, Any] = {
        "alertsPersisted": 0,
        "alertsWithSnapshots": 0,
        "outboxTotal": 0,
        "outboxByState": {},
    }
    pipeline_started = False
    worker_started = False
    db_cleanup_owned = False
    try:
        # Take cleanup ownership before initialization so a partial pool setup
        # cannot escape if initialization itself raises.
        db_cleanup_owned = True
        db.init_pool()
        alert_store.init_db()
        worker = alert_delivery_worker.AlertDeliveryWorker(
            worker_count=worker_count,
            poll_seconds=0.02,
            lease_seconds=30.0,
            max_attempts=3,
            retry_base_seconds=0.05,
            retry_cap_seconds=0.5,
        )

        def durable_handoff(_alert: dict, _output_ids: list[str] | None):
            worker.wake()
            return DeliveryOutcome(handled_output_ids=("durable_outbox",))

        pipeline = AlertPipeline(
            persist_alert=alert_store.create_alert,
            deliver_alert=durable_handoff,
            on_persisted=video_processing._observe_persisted_alert_timing,
            persist_queue_size=max(256, alert_count),
            delivery_queue_size=max(256, alert_count),
            delivery_workers=1,
            submit_timeout=0.1,
            persistence_attempts=3,
            retry_delay=0.05,
            delivery_attempts=1,
        )
        started = time.monotonic()
        deadline = started + timeout_seconds
        config = config_manager.get_config_snapshot()
        template_targets = notification_dispatcher.resolve_delivery_targets(
            config,
            {
                "id": str(uuid4()),
                "severity": "P1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cameraId": "benchmark-camera",
            },
        )
        if (
            len(template_targets) != 1
            or template_targets[0].get("channel") != "webhook"
            or template_targets[0].get("kind", "initial") != "initial"
        ):
            raise BenchmarkRuntimeError(
                "isolated routing did not produce one webhook target"
            )

        worker_started = True
        worker.start()
        pipeline_started = True
        pipeline.start()
        for sequence in range(alert_count):
            alert_id = str(uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            targets = copy.deepcopy(template_targets)
            first_positive_ns, confirmed_ns = _register_timing(
                alert_timing,
                alert_id,
                targets,
            )
            first_positive_by_alert[alert_id] = first_positive_ns
            confirmed_at = datetime.now(timezone.utc)
            futures.append(
                pipeline.submit(
                    {
                        "alert_id": alert_id,
                        "timestamp": timestamp,
                        "delivery_targets": targets,
                        "camera_id": "benchmark-camera",
                        "camera_name": "Benchmark Camera",
                        "zone": "Isolated Benchmark",
                        "rule": "Synthetic Pipeline Replay",
                        "severity": "P1",
                        "confidence": 0.99,
                        "description": "Synthetic isolated alert pipeline replay",
                        "source": "benchmark",
                        "snapshot_jpeg": None,
                        "bboxes": [],
                        "clean_snapshot_jpeg": None,
                        "policy_id": None,
                        "priority": 1,
                        "message": None,
                        "metadata": {
                            "benchmarkRunKey": run_key,
                            "sequence": sequence,
                        },
                        "first_positive_at": confirmed_at,
                        "confirmed_at": confirmed_at,
                    }
                )
            )

        for future in futures:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BenchmarkRuntimeError("persistence wait timed out")
            future.result(timeout=remaining)
        if not pipeline.drain(max(0.0, deadline - time.monotonic())):
            raise BenchmarkRuntimeError("persistence handoff drain timed out")

        worker.wake()
        while time.monotonic() < deadline:
            db_counts = _query_run_counts(db, run_key)
            delivered = int(db_counts["outboxByState"].get("delivered", 0))
            live_alert_telemetry = pipeline_telemetry.public_snapshot()["alerts"]
            live_coverage = live_alert_telemetry["deliveryCoverage"]
            live_provider_latency = live_alert_telemetry["latency"][
                "firstPositiveToProviderSuccessMs"
            ]
            if (
                db_counts["alertsPersisted"] == alert_count
                and db_counts["outboxTotal"] == alert_count
                and delivered == alert_count
                and sink.aggregate()["uniqueAlerts"] == alert_count
                and int(live_coverage.get("pending") or 0) == 0
                and int(
                    (live_coverage.get("counters") or {}).get("deliveredCount") or 0
                )
                == alert_count
                and int(live_provider_latency.get("count") or 0) == alert_count
            ):
                break
            time.sleep(0.05)
        db_counts = _query_run_counts(db, run_key)

        telemetry = pipeline_telemetry.public_snapshot()["alerts"]
        receipts = sink.receipts()
        receipt_latencies_ms = [
            (received_ns - first_positive_by_alert[alert_id]) / 1_000_000.0
            for alert_id, received_ns in receipts.items()
            if alert_id in first_positive_by_alert
            and received_ns >= first_positive_by_alert[alert_id]
        ]
        sink_aggregate = sink.aggregate()
        delivery_coverage = telemetry["deliveryCoverage"]
        latency = {
            name: {
                "sampleCount": int(histogram.get("count") or 0),
                "invalidCount": int(histogram.get("invalidCount") or 0),
                "p95UpperBoundMs": histogram.get("p95UpperBoundMs"),
                "p95Overflow": bool(histogram.get("p95Overflow")),
                "p99UpperBoundMs": histogram.get("p99UpperBoundMs"),
                "p99Overflow": bool(histogram.get("p99Overflow")),
                "maximumMs": histogram.get("maximumMs"),
            }
            for name, histogram in telemetry["latency"].items()
            if name
            in {
                "firstPositiveToConfirmedMs",
                "confirmedToPersistedMs",
                "firstPositiveToPersistedMs",
                "firstPositiveToProviderSuccessMs",
            }
        }
        states = db_counts["outboxByState"]
        coverage_counters = delivery_coverage.get("counters", {})
        valid = (
            db_counts["alertsPersisted"] == alert_count
            and db_counts["alertsWithSnapshots"] == 0
            and db_counts["outboxTotal"] == alert_count
            and int(states.get("delivered", 0)) == alert_count
            and sum(
                int(states.get(state, 0))
                for state in ("pending", "terminal", "cancelled")
            )
            == 0
            and sink_aggregate["acceptedRequests"] == alert_count
            and sink_aggregate["uniqueAlerts"] == alert_count
            and sink_aggregate["duplicateRequests"] == 0
            and sink_aggregate["malformedRequests"] == 0
            and sink_aggregate["oversizeRequests"] == 0
            and sink_aggregate["snapshotPayloadRequests"] == 0
            and _has_exact_sample_count(receipt_latencies_ms, alert_count)
            and int(delivery_coverage.get("pending") or 0) == 0
            and int(coverage_counters.get("eligibleCount") or 0) == alert_count
            and int(coverage_counters.get("deliveredCount") or 0) == alert_count
            and latency["firstPositiveToPersistedMs"]["sampleCount"] == alert_count
            and latency["firstPositiveToProviderSuccessMs"]["sampleCount"]
            == alert_count
            and not any(item["invalidCount"] for item in latency.values())
        )
        pipeline_stats = pipeline.stats()
        worker_stats = worker.stats()
        return {
            "outcome": "valid" if valid else "incomplete",
            "configuredAlertCount": alert_count,
            "elapsedSeconds": round(time.monotonic() - started, 6),
            "measurementValidity": {
                "valid": valid,
                "allAlertsPersisted": db_counts["alertsPersisted"] == alert_count,
                "allOutboxTargetsDelivered": int(states.get("delivered", 0))
                == alert_count,
                "allProviderReceiptsObserved": sink_aggregate["uniqueAlerts"]
                == alert_count,
                "allMonotonicProviderReceiptsMatched": _has_exact_sample_count(
                    receipt_latencies_ms,
                    alert_count,
                ),
                "noSnapshotsPersistedOrTransmitted": (
                    db_counts["alertsWithSnapshots"] == 0
                    and sink_aggregate["snapshotPayloadRequests"] == 0
                ),
                "deliveryCoverageFullyAccounted": (
                    int(delivery_coverage.get("pending") or 0) == 0
                    and int(coverage_counters.get("eligibleCount") or 0) == alert_count
                    and int(coverage_counters.get("deliveredCount") or 0) == alert_count
                ),
            },
            "latency": latency,
            "firstPositiveToLoopbackReceiptMs": _latency_distribution(
                receipt_latencies_ms
            ),
            "deliveryCoverage": {
                "unit": delivery_coverage.get("unit"),
                "pending": int(delivery_coverage.get("pending") or 0),
                "counters": {
                    str(key): int(value)
                    for key, value in sorted(coverage_counters.items())
                },
            },
            "databaseCounts": db_counts,
            "loopbackProvider": sink_aggregate,
            "pipelineCounts": {
                key: int(pipeline_stats.get(key) or 0)
                for key in (
                    "submitted",
                    "persisted",
                    "persistence_failures",
                    "outbox_handoffs",
                    "backpressure_events",
                )
            },
            "outboxWorkerCounts": {
                "claimErrors": int(worker_stats.get("claim_errors") or 0),
                "renewalFailures": int(worker_stats.get("renewal_failures") or 0),
                "fencingFailures": int(worker_stats.get("fencing_failures") or 0),
            },
            "provenance": {
                "path": "AlertPipeline -> PostgreSQL alerts/outbox -> AlertDeliveryWorker -> loopback webhook",
                "latencyClock": "process-monotonic",
                "latencyQuantiles": "fixed-histogram nearest-rank upper bounds",
                "loopbackReceiptQuantiles": "exact nearest-rank over in-memory monotonic receipts",
                "aggregateOnly": True,
            },
        }
    finally:
        _cleanup_runtime_resources(
            pipeline,
            worker,
            pipeline_started=pipeline_started,
            worker_started=worker_started,
            alert_timing=alert_timing,
            db=db,
            db_cleanup_owned=db_cleanup_owned,
        )


def _execute(
    args: argparse.Namespace,
    *,
    connect_factory: Callable[[str], Any] = _connect_database,
    runtime: Callable[..., dict[str, Any]] = _run_real_pipeline,
    sink_factory: Callable[[], _LoopbackSink] = _LoopbackSink,
) -> dict[str, Any]:
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise BenchmarkSafetyError("DATABASE_URL is required")
    _validate_isolation_inputs(
        acknowledgement=args.isolation_acknowledgement,
        database_name_token=DEFAULT_DATABASE_NAME_TOKEN,
        config_store=os.environ.get("SAFETYLENS_CONFIG_STORE", ""),
    )

    connection = connect_factory(database_url)
    report: dict[str, Any] | None = None
    try:
        original_config = _verify_isolated_database(
            connection,
            database_name_token=DEFAULT_DATABASE_NAME_TOKEN,
        )
        original_sections = _original_config_sections(original_config)
        with sink_factory() as sink:
            benchmark_sections = _benchmark_config_sections(
                original_config,
                loopback_url=sink.url,
            )
            configured_config = copy.deepcopy(original_config)
            for key in CONFIG_SECTION_KEYS:
                configured_config.pop(key, None)
            configured_config.update(copy.deepcopy(benchmark_sections))

            def reconnect():
                replacement = connect_factory(database_url)
                try:
                    _require_guarded_database_name(
                        replacement,
                        database_name_token=DEFAULT_DATABASE_NAME_TOKEN,
                    )
                    _acquire_benchmark_lock(replacement)
                    _require_exclusive_clone_client(replacement)
                except Exception:
                    replacement.close()
                    raise
                return replacement

            with _temporary_alert_configuration(
                connection,
                original_sections=original_sections,
                benchmark_sections=benchmark_sections,
                reconnect=reconnect,
            ):
                with _temporary_environment(
                    {
                        "WEBHOOK_ALLOWED_HTTP_HOSTS": "127.0.0.1",
                        "WEBHOOK_ALLOWED_PRIVATE_HOSTS": "127.0.0.1",
                        "SAFETYLENS_CONFIG_REFRESH_SECONDS": "86400",
                    }
                ):
                    report = runtime(
                        alert_count=args.alert_count,
                        timeout_seconds=args.timeout_seconds,
                        worker_count=args.worker_count,
                        run_key=uuid4().hex,
                        sink=sink,
                        configured_app_config=configured_config,
                    )
    finally:
        connection.close()
    if report is None:
        raise BenchmarkRuntimeError("benchmark produced no aggregate report")
    report["configurationRestored"] = True
    report["isolationGuardVerified"] = True
    report["exclusiveCloneClientVerified"] = True
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="isolated-alert-pipeline-replay")
    parser.add_argument("--alert-count", type=int, default=MIN_ALERT_COUNT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument(
        "--isolation-acknowledgement",
        required=True,
        help=f"must equal {ISOLATION_ACKNOWLEDGEMENT}",
    )
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if LABEL_RE.fullmatch(args.label) is None:
        parser.error("label must be a bounded identifier")
    if not MIN_ALERT_COUNT <= args.alert_count <= MAX_ALERT_COUNT:
        parser.error(
            f"alert count must be between {MIN_ALERT_COUNT} and {MAX_ALERT_COUNT}"
        )
    if not math.isfinite(args.timeout_seconds) or not 10 <= args.timeout_seconds <= 900:
        parser.error("timeout must be between 10 and 900 seconds")
    if not 1 <= args.worker_count <= 16:
        parser.error("worker count must be between 1 and 16")


def _safe_failure_report(label: str, exc: BaseException) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "label": label,
        "outcome": "failed",
        "errorType": type(exc).__name__,
        "aggregateOnly": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    exit_code = 0
    try:
        summary = _execute(args)
        exit_code = 0 if summary.get("outcome") == "valid" else 3
        report = {
            "schemaVersion": SCHEMA_VERSION,
            "label": args.label,
            "reportId": hashlib.sha256(
                f"{args.label}:{time.monotonic_ns()}".encode("utf-8")
            ).hexdigest()[:16],
            "summary": summary,
        }
    except Exception as exc:
        report = _safe_failure_report(args.label, exc)
        exit_code = 2
    _atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
