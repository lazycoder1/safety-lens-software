import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import constants
import model_manager


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = float(value)

    def advance(self, seconds):
        with self.lock:
            self.value += seconds


class FakeResponse:
    def __init__(self, payload=None, *, body=None, status_code=200, headers=None, chunks=None):
        if body is None:
            body = json.dumps(payload).encode("utf-8")
        self.body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        if self.chunks is not None:
            yield from self.chunks
            return
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _catalog(*, ready=True, model_key="coco_primary", **fields):
    item = {"model_key": model_key, "is_ready": ready, **fields}
    return model_manager._validate_remote_model_catalog({"models": [item]})


@pytest.fixture(autouse=True)
def isolated_remote_catalog(monkeypatch):
    monkeypatch.setattr(model_manager, "is_remote_inference_enabled", lambda: True)
    model_manager.invalidate_remote_model_catalog()
    yield
    model_manager.invalidate_remote_model_catalog()


def test_ttl_reuses_one_fetch_and_refreshes_with_monotonic_clock(monkeypatch):
    clock = FakeClock()
    calls = 0

    def fetch():
        nonlocal calls
        calls += 1
        return _catalog(ready=True)

    monkeypatch.setattr(model_manager, "_monotonic", clock)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TTL_SECONDS", 5.0)
    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)

    assert model_manager.model_readiness_snapshot(["coco_primary"])["coco_primary"]
    assert model_manager.model_readiness_snapshot(["coco_primary"])["coco_primary"]
    clock.advance(4.99)
    assert model_manager.list_models()[0]["is_ready"]
    assert calls == 1

    clock.advance(0.02)
    assert model_manager.list_models()[0]["is_ready"]
    assert calls == 2


def test_sixteen_threads_share_one_catalog_refresh(monkeypatch):
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(17)

    def fetch():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return _catalog(ready=True)

    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)

    def worker():
        start.wait(timeout=2)
        return model_manager.model_readiness_snapshot(["coco_primary"])

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(worker) for _ in range(16)]
        start.wait(timeout=2)
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert all(result == {"coco_primary": True} for result in results)


def test_follower_wait_is_bounded_when_leader_stalls(monkeypatch):
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    def fetch():
        fetch_started.set()
        assert release_fetch.wait(timeout=2)
        return _catalog(ready=True)

    monkeypatch.setattr(model_manager, "MODEL_METADATA_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)
    leader_result = []
    leader = threading.Thread(target=lambda: leader_result.append(model_manager.list_models()))
    leader.start()
    assert fetch_started.wait(timeout=1)

    started = time.monotonic()
    follower = model_manager.list_models()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert all(item["is_ready"] is False for item in follower)
    release_fetch.set()
    leader.join(timeout=2)
    assert not leader.is_alive()
    assert leader_result[0][0]["is_ready"] is True


def test_invalidation_fences_a_stale_inflight_refresh(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def fetch():
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
            return _catalog(ready=False)
        return _catalog(ready=True)

    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)
    stale_result = []
    stale = threading.Thread(target=lambda: stale_result.append(model_manager.list_models()))
    stale.start()
    assert first_started.wait(timeout=1)

    model_manager.invalidate_remote_model_catalog()
    assert model_manager.list_models()[0]["is_ready"] is True
    release_first.set()
    stale.join(timeout=2)

    assert not stale.is_alive()
    assert stale_result[0][0]["is_ready"] is False
    assert model_manager.list_models()[0]["is_ready"] is True
    assert calls == 2


def test_expired_ready_catalog_is_dropped_when_refresh_fails(monkeypatch):
    clock = FakeClock()
    responses = iter([_catalog(ready=True), RuntimeError("secret https://model/token")])

    def fetch():
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(model_manager, "_monotonic", clock)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TTL_SECONDS", 5.0)
    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)

    assert model_manager.list_models()[0]["is_ready"] is True
    clock.advance(6)
    models = model_manager.list_models()

    assert all(item["is_ready"] is False for item in models)
    assert {item["error"] for item in models} == {"Remote model metadata unavailable"}
    assert "secret" not in json.dumps(models)
    assert "https://model/token" not in json.dumps(model_manager.remote_model_metadata_health())


def test_breaker_uses_one_two_four_eight_ten_second_cadence_and_resets(monkeypatch):
    clock = FakeClock()
    attempts = []
    should_succeed = False

    def fetch():
        attempts.append(clock())
        if not should_succeed:
            raise RuntimeError("offline")
        return _catalog(ready=True)

    monkeypatch.setattr(model_manager, "_monotonic", clock)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_BREAKER_INITIAL_SECONDS", 1.0)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_BREAKER_MAX_SECONDS", 10.0)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TTL_SECONDS", 5.0)
    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)

    for attempt_at, retry_after in [(0, 1), (1, 2), (3, 4), (7, 8), (15, 10)]:
        clock.set(attempt_at)
        model_manager.list_models()
        assert model_manager.remote_model_metadata_health()["retry_after_seconds"] == retry_after
        clock.advance(retry_after - 0.01)
        model_manager.list_models()
        assert attempts[-1] == attempt_at

    should_succeed = True
    clock.set(25)
    assert model_manager.list_models()[0]["is_ready"] is True
    assert model_manager.remote_model_metadata_health()["failure_count"] == 0

    should_succeed = False
    clock.set(31)
    model_manager.list_models()
    assert model_manager.remote_model_metadata_health()["retry_after_seconds"] == 1.0
    assert attempts == [0.0, 1.0, 3.0, 7.0, 15.0, 25.0, 31.0]


def test_outage_logs_only_transition_reminder_and_recovery(monkeypatch, caplog):
    clock = FakeClock()
    results = iter([RuntimeError("first"), RuntimeError("second"), _catalog(ready=True)])

    def fetch():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(model_manager, "_monotonic", clock)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_LOG_REMINDER_SECONDS", 60.0)
    monkeypatch.setattr(model_manager, "_fetch_remote_model_catalog", fetch)
    caplog.set_level(logging.INFO, logger="safetylens.models")

    model_manager.list_models()  # transition
    clock.set(59)
    model_manager.list_models()  # second failed refresh, no repeated warning
    clock.set(60)
    model_manager.list_models()  # breaker-open reminder
    clock.set(61)
    model_manager.list_models()  # recovery

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Remote model metadata became unavailable",
        "Remote model metadata remains unavailable",
        "Remote model metadata recovered",
    ]
    assert "first" not in caplog.text
    assert "second" not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"models": {}, "extra": True},
        {"models": ["bad"]},
        {"models": [{"model_key": "coco_primary", "is_ready": "false"}]},
        {
            "models": [
                {"model_key": "coco_primary", "is_ready": True},
                {"model_key": "coco_primary", "is_ready": False},
            ]
        },
    ],
)
def test_catalog_validation_rejects_malformed_payloads(payload):
    with pytest.raises(model_manager._RemoteModelCatalogError):
        model_manager._validate_remote_model_catalog(payload)


def test_missing_known_models_are_synthesized_unavailable_and_unknown_requests_fail_closed(monkeypatch):
    monkeypatch.setattr(
        model_manager,
        "_fetch_remote_model_catalog",
        lambda: _catalog(ready=True, model_key="coco_primary"),
    )

    models = model_manager.list_models()
    assert len(models) == len(model_manager.MODEL_DEFINITIONS)
    assert models[0]["is_ready"] is True
    assert all(item["is_ready"] is False for item in models[1:])
    assert model_manager.model_readiness_snapshot(["unknown_model"]) == {"unknown_model": False}
    assert model_manager.missing_model_keys(["unknown_model"]) == ["unknown_model"]
    assert model_manager.missing_model_keys(
        ["coco_primary", "unknown_model"],
        {"coco_primary": True},
    ) == ["unknown_model"]


def test_bounded_future_catalog_fields_and_model_keys_are_ignored():
    models = model_manager._validate_remote_model_catalog(
        {
            "models": [
                {"model_key": "future_detector", "is_ready": True},
                {"model_key": "coco_primary", "is_ready": True},
            ],
            "generation": 2,
        }
    )

    assert len(models) == len(model_manager.MODEL_DEFINITIONS)
    assert models[0]["model_key"] == "coco_primary"
    assert models[0]["is_ready"] is True
    assert "future_detector" not in {item["model_key"] for item in models}


def test_empty_or_unknown_only_readiness_does_not_fetch(monkeypatch):
    monkeypatch.setattr(
        model_manager,
        "_fetch_remote_model_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected metadata fetch")),
    )

    assert model_manager.model_readiness_snapshot([]) == {}
    assert model_manager.model_readiness_snapshot(["future_detector"]) == {
        "future_detector": False,
    }


def test_returned_catalogs_are_deep_copy_isolated(monkeypatch):
    monkeypatch.setattr(
        model_manager,
        "_fetch_remote_model_catalog",
        lambda: _catalog(
            ready=True,
            runtime_fixed_classes=["person"],
            runtime_fixed_class_groups=["human"],
        ),
    )
    first = model_manager.list_models()
    first[0]["is_ready"] = False
    first[0]["runtime_fixed_classes"].append("mutated")

    second = model_manager.list_models()
    assert second[0]["is_ready"] is True
    assert second[0]["runtime_fixed_classes"] == ["person"]


def test_remote_fields_are_whitelisted_bounded_and_errors_are_generic():
    item = {
        "model_key": "coco_primary",
        "is_ready": False,
        "status": "failed",
        "error": "password at https://internal.example",
        "download_url": "https://attacker.example/payload",
        "active_path": "https://attacker.example/path",
        "runtime_path": "x" * 2_000,
        "runtime_fallback_error": "secret traceback",
        "unexpected": "not public",
    }

    model = model_manager._validate_remote_model_catalog({"models": [item]})[0]

    assert model["error"] == "Remote model reported an error"
    assert model["active_path"] is None
    assert model["runtime_path"] is None
    assert model["runtime_fallback_error"] == "Remote model runtime fallback active"
    assert model["download_url"] == model_manager.MODEL_DEFINITIONS["coco_primary"]["download_url"]
    assert "unexpected" not in model
    assert "attacker" not in json.dumps(model)
    assert "secret traceback" not in json.dumps(model)


def test_metadata_fetch_is_bounded_uses_dedicated_timeout_and_disables_redirects(monkeypatch):
    response = FakeResponse({"models": [{"model_key": "coco_primary", "is_ready": True}]})
    session = FakeSession(response)
    monkeypatch.setattr(model_manager, "_remote_session", lambda: session)
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(model_manager, "MODEL_SERVER_URL", "https://model.example")

    result = model_manager._fetch_remote_model_catalog()

    assert result[0]["is_ready"] is True
    assert session.calls == [
        (
            "https://model.example/api/models",
            {
                "headers": model_manager._remote_headers(),
                "timeout": (2.0, 0.5),
                "allow_redirects": False,
                "stream": True,
            },
        )
    ]
    assert response.closed


def test_metadata_fetch_rejects_redirect_and_closes_response(monkeypatch):
    response = FakeResponse({"models": []}, status_code=302)
    monkeypatch.setattr(model_manager, "_remote_session", lambda: FakeSession(response))

    with pytest.raises(model_manager._RemoteModelCatalogError):
        model_manager._fetch_remote_model_catalog()
    assert response.closed


def test_remote_install_mutations_invalidate_but_terminal_polling_does_not(monkeypatch):
    invalidations = []
    posts = []
    job_state = {"status": "running", "model_keys": ["coco_primary"]}

    monkeypatch.setattr(
        model_manager,
        "invalidate_remote_model_catalog",
        lambda: invalidations.append("invalidate"),
    )
    monkeypatch.setattr(
        model_manager,
        "_remote_post",
        lambda path, payload: posts.append((path, payload)) or dict(job_state),
    )
    monkeypatch.setattr(model_manager, "_remote_get", lambda _path: dict(job_state))

    model_manager.install_models(["coco_primary"])
    model_manager.retry_install_job("job-1")
    assert len(invalidations) == 2

    job_state["status"] = "ready"
    for _ in range(3):
        assert model_manager.get_install_job("job-1")["status"] == "ready"
    # Read-only terminal polling relies on the five-second catalogue TTL and
    # must not repeatedly reset the breaker or fence in-flight refreshes.
    assert len(invalidations) == 2
    assert posts == [
        ("/api/models/install", {"model_keys": ["coco_primary"]}),
        ("/api/models/install/job-1/retry", {}),
    ]


def test_metadata_slow_drip_hits_absolute_deadline(monkeypatch):
    class SlowDripResponse(FakeResponse):
        def iter_content(self, chunk_size):
            del chunk_size
            while True:
                time.sleep(0.02)
                yield b" "

    response = SlowDripResponse(body=b"")
    monkeypatch.setattr(model_manager, "_remote_session", lambda: FakeSession(response))
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    with pytest.raises(
        model_manager._RemoteModelCatalogError,
        match="deadline exceeded",
    ):
        model_manager._fetch_remote_model_catalog()

    assert time.monotonic() - started < 0.2
    deadline = time.monotonic() + 0.2
    while not response.closed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert response.closed


def test_metadata_deadline_includes_blocked_dns_or_response_headers(monkeypatch):
    release = threading.Event()

    class BlockingSession:
        def get(self, *_args, **_kwargs):
            assert release.wait(timeout=2)
            return FakeResponse({"models": []})

    monkeypatch.setattr(model_manager, "_remote_session", lambda: BlockingSession())
    monkeypatch.setattr(model_manager, "MODEL_METADATA_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    with pytest.raises(
        model_manager._RemoteModelCatalogError,
        match="deadline exceeded",
    ):
        model_manager._fetch_remote_model_catalog()
    assert time.monotonic() - started < 0.2

    release.set()
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        with model_manager._REMOTE_MODEL_TRANSPORT_LOCK:
            transport = model_manager._REMOTE_MODEL_TRANSPORT_THREAD
        if transport is None:
            break
        time.sleep(0.005)
    assert model_manager._REMOTE_MODEL_TRANSPORT_THREAD is None


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(
            {"models": []},
            headers={"Content-Length": str(model_manager._REMOTE_MODEL_CATALOG_MAX_BYTES + 1)},
        ),
        FakeResponse(
            body=b"",
            chunks=[b"x" * (model_manager._REMOTE_MODEL_CATALOG_MAX_BYTES + 1)],
        ),
    ],
)
def test_metadata_fetch_rejects_declared_or_streamed_oversize(monkeypatch, response):
    monkeypatch.setattr(model_manager, "_remote_session", lambda: FakeSession(response))

    with pytest.raises(model_manager._RemoteModelCatalogError):
        model_manager._fetch_remote_model_catalog()
    assert response.closed


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "garbage"])
def test_metadata_config_rejects_nonfinite_or_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("TEST_METADATA_FLOAT", raw)
    assert constants._finite_env_float(
        "TEST_METADATA_FLOAT",
        2.0,
        minimum=0.1,
        maximum=30.0,
    ) == 2.0
