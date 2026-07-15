"""Safety and aggregate-output tests for the isolated alert replay benchmark."""

from __future__ import annotations

import importlib.util
import json
import stat
import urllib.request
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark_alert_pipeline_replay.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_alert_pipeline_replay",
        SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark():
    return _load_benchmark()


class _Cursor:
    def __init__(
        self,
        database_name: str,
        config: dict,
        *,
        lock_acquired: bool = True,
        pending_outbox: int | None = None,
        other_clients: int = 0,
    ):
        self.database_name = database_name
        self.config = config
        self.lock_acquired = lock_acquired
        self.pending_outbox = pending_outbox
        self.other_clients = other_clients
        self.executed: list[tuple[str, object]] = []
        self._result = None
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return None

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split())
        self.executed.append((normalized, params))
        if "pg_catalog.pg_stat_activity" in normalized:
            self._result = (self.other_clients,)
        elif "current_database()" in normalized:
            self._result = (self.database_name,)
        elif "pg_try_advisory_lock" in normalized:
            self._result = (self.lock_acquired,)
        elif normalized.startswith("SELECT config FROM public.app_config"):
            self._result = (self.config,)
        elif "to_regclass('public.alert_delivery_outbox')" in normalized:
            self._result = (
                None if self.pending_outbox is None else "public.alert_delivery_outbox",
            )
        elif "FROM public.alert_delivery_outbox" in normalized:
            self._result = (self.pending_outbox or 0,)
        else:
            self._result = None

    def fetchone(self):
        return self._result


class _Connection:
    def __init__(
        self,
        database_name: str,
        config: dict,
        *,
        lock_acquired: bool = True,
        pending_outbox: int | None = None,
        other_clients: int = 0,
    ):
        self.cursor_instance = _Cursor(
            database_name,
            config,
            lock_acquired=lock_acquired,
            pending_outbox=pending_outbox,
            other_clients=other_clients,
        )
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _args(benchmark, **overrides):
    values = {
        "isolation_acknowledgement": benchmark.ISOLATION_ACKNOWLEDGEMENT,
        "alert_count": 100,
        "timeout_seconds": 30.0,
        "worker_count": 3,
    }
    values.update(overrides)
    return Namespace(**values)


def test_isolation_acknowledgement_token_and_postgres_store_are_mandatory(benchmark):
    with pytest.raises(benchmark.BenchmarkSafetyError):
        benchmark._validate_isolation_inputs(
            acknowledgement="yes",
            database_name_token="benchmark",
            config_store="postgres",
        )
    with pytest.raises(benchmark.BenchmarkSafetyError):
        benchmark._validate_isolation_inputs(
            acknowledgement=benchmark.ISOLATION_ACKNOWLEDGEMENT,
            database_name_token="bench",
            config_store="postgres",
        )
    with pytest.raises(benchmark.BenchmarkSafetyError):
        benchmark._validate_isolation_inputs(
            acknowledgement=benchmark.ISOLATION_ACKNOWLEDGEMENT,
            database_name_token="benchmark",
            config_store="file",
        )


def test_database_name_guard_fails_before_lock_or_config_read(benchmark):
    connection = _Connection("rakshak_lens", {"alert_outputs": []})

    with pytest.raises(benchmark.BenchmarkSafetyError):
        benchmark._verify_isolated_database(
            connection,
            database_name_token="benchmark",
        )

    queries = [query for query, _params in connection.cursor_instance.executed]
    assert len(queries) == 1
    assert "current_database()" in queries[0]


def test_database_guard_locks_and_copies_authoritative_config(benchmark):
    original = {"alert_outputs": [{"id": "webhook", "enabled": False}]}
    connection = _Connection("rakshak_pipeline_benchmark", original)

    loaded = benchmark._verify_isolated_database(
        connection,
        database_name_token="benchmark",
    )

    assert loaded == original
    assert loaded is not original
    loaded["alert_outputs"][0]["enabled"] = True
    assert original["alert_outputs"][0]["enabled"] is False
    queries = [query for query, _params in connection.cursor_instance.executed]
    assert any("pg_try_advisory_lock" in query for query in queries)
    assert any("SELECT config FROM public.app_config" in query for query in queries)


def test_database_guard_rejects_preexisting_pending_outbox_work(benchmark):
    connection = _Connection(
        "rakshak_pipeline_benchmark",
        {"alert_outputs": []},
        pending_outbox=1,
    )

    with pytest.raises(benchmark.BenchmarkSafetyError, match="pending alert work"):
        benchmark._verify_isolated_database(
            connection,
            database_name_token="benchmark",
        )


def test_database_guard_rejects_another_connected_clone_client(benchmark):
    connection = _Connection(
        "rakshak_pipeline_benchmark",
        {"alert_outputs": []},
        other_clients=1,
    )

    with pytest.raises(benchmark.BenchmarkSafetyError, match="connected client"):
        benchmark._verify_isolated_database(
            connection,
            database_name_token="benchmark",
        )


def test_benchmark_sections_disable_every_non_loopback_output(benchmark):
    original = {
        "alert_outputs": [
            {
                "id": "telegram",
                "type": "telegram",
                "enabled": True,
                "settings": {"bot_token": "secret"},
            },
            {
                "id": "webhook",
                "type": "webhook",
                "enabled": True,
                "settings": {"url": "https://external.invalid"},
            },
        ],
        "alert_routing": {
            "templates": {"email_subject": "keep"},
            "escalation_steps": [{"channel": "email", "enabled": True}],
        },
    }
    loopback = "http://127.0.0.1:32123/alert-benchmark/token"

    sections = benchmark._benchmark_config_sections(
        original,
        loopback_url=loopback,
    )

    enabled = [output for output in sections["alert_outputs"] if output["enabled"]]
    assert len(enabled) == 1
    assert enabled[0]["id"] == "webhook"
    assert enabled[0]["settings"] == {
        "url": loopback,
        "headers": {},
        "include_snapshot": False,
    }
    assert sections["webhook"]["url"] == loopback
    assert sections["telegram"] == {"enabled": False}
    assert sections["email"] == {"enabled": False}
    assert sections["alert_routing"]["escalation_steps"] == []
    assert sections["alert_routing"]["templates"] == {"email_subject": "keep"}
    assert all(
        matrix
        == {
            "inApp": False,
            "telegram": False,
            "email": False,
            "webhook": True,
            "whatsapp": False,
            "sms": False,
            "plc": False,
        }
        for matrix in sections["alert_routing"]["channel_matrix"].values()
    )


def test_temporary_configuration_restores_owned_sections_on_failure(
    benchmark,
    monkeypatch: pytest.MonkeyPatch,
):
    writes = []
    connection = object()
    original = {"webhook": {"enabled": False}}
    candidate = {"webhook": {"enabled": True}}
    monkeypatch.setattr(
        benchmark,
        "_write_config_sections",
        lambda received_connection, sections: writes.append(
            (received_connection, sections)
        ),
    )

    with pytest.raises(RuntimeError, match="injected"):
        with benchmark._temporary_alert_configuration(
            connection,
            original_sections=original,
            benchmark_sections=candidate,
            reconnect=lambda: pytest.fail("reconnect should not be needed"),
        ):
            raise RuntimeError("injected")

    assert writes == [(connection, candidate), (connection, original)]


def test_ambiguous_initial_config_apply_still_attempts_restore(
    benchmark,
    monkeypatch: pytest.MonkeyPatch,
):
    writes = []
    connection = object()
    original = {"webhook": {"enabled": False}}
    candidate = {"webhook": {"enabled": True}}

    def ambiguous_write(received_connection, sections):
        writes.append((received_connection, sections))
        if len(writes) == 1:
            raise RuntimeError("commit outcome unknown")

    monkeypatch.setattr(benchmark, "_write_config_sections", ambiguous_write)

    with pytest.raises(RuntimeError, match="commit outcome unknown"):
        with benchmark._temporary_alert_configuration(
            connection,
            original_sections=original,
            benchmark_sections=candidate,
            reconnect=lambda: pytest.fail("reconnect should not be needed"),
        ):
            pytest.fail("an ambiguous apply must not enter the benchmark")

    assert writes == [(connection, candidate), (connection, original)]


def test_loopback_sink_retains_only_aggregate_receipts(benchmark):
    alert_id = str(uuid4())
    payload = json.dumps(
        {
            "source": "Rakshak Lens",
            "event": "alert",
            "alert": {"id": alert_id, "description": "must not be retained"},
        }
    ).encode()

    with benchmark._LoopbackSink() as sink:
        request = urllib.request.Request(
            sink.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2.0) as response:
            assert response.status == 204
        aggregate = sink.aggregate()

    assert aggregate == {
        "acceptedRequests": 1,
        "uniqueAlerts": 1,
        "duplicateRequests": 0,
        "malformedRequests": 0,
        "oversizeRequests": 0,
        "snapshotPayloadRequests": 0,
    }
    rendered = json.dumps(aggregate)
    assert alert_id not in rendered
    assert "must not be retained" not in rendered
    assert "127.0.0.1" not in rendered


def test_execute_restores_config_after_runtime_failure(
    benchmark,
    monkeypatch: pytest.MonkeyPatch,
):
    original = {
        "alert_outputs": [],
        "alert_routing": {},
        "webhook": {"enabled": False},
        "telegram": {"enabled": False},
        "email": {"enabled": False},
    }
    connection = _Connection("rakshak_benchmark_clone", original)
    writes = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret@db/rakshak_benchmark_clone")
    monkeypatch.setenv("SAFETYLENS_CONFIG_STORE", "postgres")
    monkeypatch.setattr(
        benchmark,
        "_write_config_sections",
        lambda received_connection, sections: writes.append(
            (received_connection, copy_sections(sections))
        ),
    )

    def fail_runtime(**_kwargs):
        raise RuntimeError("provider URL and secret must not escape")

    with pytest.raises(RuntimeError, match="provider URL"):
        benchmark._execute(
            _args(benchmark),
            connect_factory=lambda _url: connection,
            runtime=fail_runtime,
        )

    assert len(writes) == 2
    assert writes[0][1]["webhook"]["enabled"] is True
    assert writes[1][1] == original
    assert connection.closed is True


def test_restore_reconnect_rechecks_database_name_and_lock(
    benchmark,
    monkeypatch: pytest.MonkeyPatch,
):
    original = {
        "alert_outputs": [],
        "alert_routing": {},
        "webhook": {"enabled": False},
        "telegram": {"enabled": False},
        "email": {"enabled": False},
    }
    primary = _Connection("rakshak_benchmark_clone", original)
    replacement = _Connection("rakshak_benchmark_clone", original)
    connections = iter((primary, replacement))
    writes = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/rakshak_benchmark_clone")
    monkeypatch.setenv("SAFETYLENS_CONFIG_STORE", "postgres")

    def write_with_ambiguous_restore(connection, sections):
        writes.append((connection, copy_sections(sections)))
        primary_writes = sum(item[0] is primary for item in writes)
        if connection is primary and primary_writes == 2:
            raise RuntimeError("primary connection failed during restore")

    monkeypatch.setattr(
        benchmark,
        "_write_config_sections",
        write_with_ambiguous_restore,
    )

    result = benchmark._execute(
        _args(benchmark),
        connect_factory=lambda _url: next(connections),
        runtime=lambda **_kwargs: {"outcome": "valid"},
    )

    assert result["configurationRestored"] is True
    assert result["exclusiveCloneClientVerified"] is True
    assert writes[-1] == (replacement, original)
    replacement_queries = [
        query for query, _params in replacement.cursor_instance.executed
    ]
    assert any("current_database()" in query for query in replacement_queries)
    assert any("pg_try_advisory_lock" in query for query in replacement_queries)
    assert primary.closed is True
    assert replacement.closed is True


def copy_sections(value):
    return json.loads(json.dumps(value))


def test_latency_distribution_uses_nearest_rank(benchmark):
    values = [float(value) for value in range(1, 101)]

    result = benchmark._latency_distribution(values)

    assert result == {
        "sampleCount": 100,
        "p95Ms": 95.0,
        "p99Ms": 99.0,
        "maxMs": 100.0,
    }


def test_exact_receipt_sample_count_is_required(benchmark):
    assert benchmark._has_exact_sample_count([object()] * 100, 100) is True
    assert benchmark._has_exact_sample_count([object()] * 99, 100) is False
    assert benchmark._has_exact_sample_count([object()] * 101, 100) is False


def test_cleanup_has_independent_budget_and_fails_if_workers_remain(benchmark):
    calls = []

    class Pipeline:
        def shutdown(self, *, wait, timeout):
            calls.append(("pipeline", wait, timeout))
            return False

    class Worker:
        def stop(self, timeout):
            calls.append(("worker", timeout))
            return True

    with pytest.raises(benchmark.BenchmarkRuntimeError, match="failed to stop"):
        benchmark._stop_runtime_workers(
            Pipeline(),
            Worker(),
            pipeline_started=True,
            worker_started=True,
        )

    assert calls == [("pipeline", True, 30.0), ("worker", 30.0)]


def test_cleanup_closes_owned_pool_even_when_worker_shutdown_fails(
    benchmark,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class Registry:
        def clear(self):
            calls.append("registry-clear")

    class AlertTiming:
        registry = Registry()

    class Database:
        def close_pool(self):
            calls.append("pool-close")

    monkeypatch.setattr(
        benchmark,
        "_stop_runtime_workers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            benchmark.BenchmarkRuntimeError("workers failed to stop")
        ),
    )

    with pytest.raises(benchmark.BenchmarkRuntimeError, match="failed to stop"):
        benchmark._cleanup_runtime_resources(
            object(),
            object(),
            pipeline_started=True,
            worker_started=True,
            alert_timing=AlertTiming(),
            db=Database(),
            db_cleanup_owned=True,
        )

    assert calls == ["registry-clear", "pool-close"]


def test_cleanup_closes_partially_initialized_pool_without_workers(benchmark):
    calls = []

    class Registry:
        def clear(self):
            calls.append("registry-clear")

    class AlertTiming:
        registry = Registry()

    class Database:
        def close_pool(self):
            calls.append("pool-close")

    benchmark._cleanup_runtime_resources(
        None,
        None,
        pipeline_started=False,
        worker_started=False,
        alert_timing=AlertTiming(),
        db=Database(),
        db_cleanup_owned=True,
    )

    assert calls == ["registry-clear", "pool-close"]


def test_failure_report_and_private_output_never_echo_exception_text(
    benchmark,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "alert-replay.json"
    monkeypatch.setattr(
        benchmark,
        "_execute",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("postgresql://user:secret@private.example/production")
        ),
    )

    exit_code = benchmark.main(
        [
            "--out",
            str(destination),
            "--isolation-acknowledgement",
            benchmark.ISOLATION_ACKNOWLEDGEMENT,
        ]
    )

    assert exit_code == 2
    rendered = destination.read_text(encoding="utf-8")
    assert "secret" not in rendered
    assert "private.example" not in rendered
    assert "production" not in rendered
    assert json.loads(rendered)["errorType"] == "RuntimeError"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
