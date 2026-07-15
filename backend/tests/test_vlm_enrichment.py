import threading
import time

import pytest

import vlm_enrichment
from vlm_enrichment import VLMEnrichmentDispatcher, remote_vlm_endpoint_allowed


def _install_dns(monkeypatch, mapping):
    monkeypatch.setattr(vlm_enrichment.socket, "gethostname", lambda: "jetson-edge")
    monkeypatch.setattr(vlm_enrichment.socket, "getfqdn", lambda: "jetson-edge.local")

    def fake_getaddrinfo(host, port, *args, **kwargs):
        result = mapping.get(host)
        if isinstance(result, BaseException):
            raise result
        if result is None:
            raise vlm_enrichment.socket.gaierror(f"no address for {host}")
        entries = []
        for address in result:
            if ":" in address:
                family = vlm_enrichment.socket.AF_INET6
                sockaddr = (address, port or 0, 0, 0)
            else:
                family = vlm_enrichment.socket.AF_INET
                sockaddr = (address, port or 0)
            entries.append(
                (
                    family,
                    vlm_enrichment.socket.SOCK_STREAM,
                    vlm_enrichment.socket.IPPROTO_TCP,
                    "",
                    sockaddr,
                )
            )
        return entries

    monkeypatch.setattr(vlm_enrichment.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(vlm_enrichment, "_interface_ip_addresses", lambda: set())


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/api/generate",
        "http://127.0.0.1:11434/api/generate",
        "http://[::1]:11434/api/generate",
        "file:///tmp/model",
        "not-a-url",
    ],
)
def test_remote_endpoint_rejects_on_device_and_invalid_urls(url):
    assert not remote_vlm_endpoint_allowed(url)


def test_remote_endpoint_allows_remote_ip_but_rejects_dns_names(monkeypatch):
    _install_dns(
        monkeypatch,
        {
            "jetson-edge": ["192.168.1.10"],
            "jetson-edge.local": ["192.168.1.10"],
            "vlm.example.test": ["192.168.1.50"],
        },
    )
    assert remote_vlm_endpoint_allowed("http://192.168.1.50:11434/api/generate")
    assert not remote_vlm_endpoint_allowed("https://vlm.example.test/generate")


def test_remote_endpoint_never_resolves_rebindable_target_hostname(monkeypatch):
    monkeypatch.setattr(vlm_enrichment.socket, "gethostname", lambda: "jetson-edge")
    monkeypatch.setattr(
        vlm_enrichment.socket,
        "getfqdn",
        lambda: "jetson-edge.local",
    )
    resolutions = []

    def rebinding_dns(host, port, *args, **kwargs):
        resolutions.append((host, port))
        address = "192.168.1.50" if len(resolutions) == 1 else "192.168.1.10"
        return [
            (
                vlm_enrichment.socket.AF_INET,
                vlm_enrichment.socket.SOCK_STREAM,
                vlm_enrichment.socket.IPPROTO_TCP,
                "",
                (address, port or 0),
            )
        ]

    monkeypatch.setattr(vlm_enrichment.socket, "getaddrinfo", rebinding_dns)

    assert not remote_vlm_endpoint_allowed(
        "http://attacker-controlled.example:11434/api/generate"
    )
    assert resolutions == []


@pytest.mark.parametrize(
    "unsafe_address",
    [
        "127.0.0.1",
        "0.0.0.0",
        "169.254.10.20",
        "224.0.0.10",
        "fe80::1234",
        "::ffff:127.0.0.1",
    ],
)
def test_remote_endpoint_rejects_hostname_when_any_dns_answer_is_unsafe(
    monkeypatch,
    unsafe_address,
):
    _install_dns(
        monkeypatch,
        {
            "jetson-edge": ["192.168.1.10"],
            "jetson-edge.local": ["192.168.1.10"],
            "vlm.example.test": ["192.168.1.50", unsafe_address],
        },
    )

    assert not remote_vlm_endpoint_allowed("https://vlm.example.test/generate")


def test_remote_endpoint_rejects_dns_alias_for_local_interface(monkeypatch):
    _install_dns(
        monkeypatch,
        {
            "jetson-edge": ["10.20.30.40"],
            "jetson-edge.local": ["10.20.30.40"],
            "vlm.example.test": ["10.20.30.40"],
        },
    )

    assert not remote_vlm_endpoint_allowed("https://vlm.example.test/generate")


def test_remote_endpoint_rejects_local_eth0_when_hostname_only_resolves_loopback(
    monkeypatch,
):
    _install_dns(
        monkeypatch,
        {
            "jetson-edge": ["127.0.1.1"],
            "jetson-edge.local": ["127.0.1.1"],
        },
    )
    monkeypatch.setattr(
        vlm_enrichment,
        "_interface_ip_addresses",
        lambda: {vlm_enrichment.ipaddress.ip_address("192.168.10.20")},
    )

    assert not remote_vlm_endpoint_allowed(
        "http://192.168.10.20:11434/api/generate"
    )


def test_remote_endpoint_hostname_resolution_failures_fail_closed(monkeypatch):
    _install_dns(
        monkeypatch,
        {
            "jetson-edge": ["10.20.30.40"],
            "jetson-edge.local": ["10.20.30.40"],
            "missing.example.test": vlm_enrichment.socket.gaierror("dns unavailable"),
        },
    )
    assert not remote_vlm_endpoint_allowed("https://missing.example.test/generate")

    _install_dns(
        monkeypatch,
        {
            "jetson-edge": vlm_enrichment.socket.gaierror("local dns unavailable"),
            "jetson-edge.local": vlm_enrichment.socket.gaierror(
                "local dns unavailable"
            ),
            "vlm.example.test": ["10.20.30.50"],
        },
    )
    assert not remote_vlm_endpoint_allowed("https://vlm.example.test/generate")
    assert not remote_vlm_endpoint_allowed("http://10.20.30.50:11434/generate")


@pytest.mark.parametrize(
    "alias",
    [
        "host.docker.internal",
        "gateway.docker.internal",
        "host.containers.internal",
        "kubernetes.docker.internal",
    ],
)
def test_remote_endpoint_rejects_container_host_aliases_without_dns(
    monkeypatch,
    alias,
):
    monkeypatch.setattr(vlm_enrichment.socket, "gethostname", lambda: "jetson-edge")
    monkeypatch.setattr(vlm_enrichment.socket, "getfqdn", lambda: "jetson-edge.local")

    def unexpected_dns(*args, **kwargs):
        raise AssertionError("container host aliases must be rejected before DNS")

    monkeypatch.setattr(vlm_enrichment.socket, "getaddrinfo", unexpected_dns)
    assert not remote_vlm_endpoint_allowed(f"http://{alias}:11434/api/generate")


def test_latest_offer_replaces_pending_work_without_blocking():
    gate = threading.Event()
    started = threading.Event()
    results = []

    def process(work):
        started.set()
        gate.wait(1)
        return work.payload

    dispatcher = VLMEnrichmentDispatcher(process=process, on_result=lambda *args: results.append(args))
    dispatcher.register_camera("cam1", "g1")
    assert dispatcher.offer("cam1", "g1", "first")
    assert started.wait(1)
    assert dispatcher.offer("cam1", "g1", "second")
    assert dispatcher.offer("cam1", "g1", "third")
    gate.set()

    deadline = time.monotonic() + 1
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert [entry[1] for entry in results] == ["first", "third"]
    assert dispatcher.stats()["replaced"] == 1
    dispatcher.shutdown(wait=True, timeout=1)


def test_generation_fence_discards_late_result_and_camera_stop_never_waits():
    gate = threading.Event()
    started = threading.Event()
    results = []

    def process(work):
        started.set()
        gate.wait(1)
        return work.payload

    dispatcher = VLMEnrichmentDispatcher(process=process, on_result=lambda *args: results.append(args))
    dispatcher.register_camera("cam1", "g1")
    assert dispatcher.offer("cam1", "g1", "old")
    assert started.wait(1)

    before = time.monotonic()
    dispatcher.discard_camera("cam1", "g1")
    assert time.monotonic() - before < 0.05
    dispatcher.register_camera("cam1", "g2")
    gate.set()

    deadline = time.monotonic() + 1
    while dispatcher.stats()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)

    assert results == []
    assert dispatcher.stats()["lateResultDiscarded"] == 1
    dispatcher.shutdown(wait=True, timeout=1)


def test_generation_barrier_blocks_result_action_after_camera_restart():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    mutations = []
    dispatcher = None

    def on_result(work, result, _elapsed):
        callback_entered.set()
        release_callback.wait(1)
        assert dispatcher is not None
        dispatcher.run_if_current(
            work.camera_id,
            work.generation,
            lambda: mutations.append(result),
        )

    dispatcher = VLMEnrichmentDispatcher(
        process=lambda work: work.payload,
        on_result=on_result,
    )
    dispatcher.register_camera("cam1", "g1")
    assert dispatcher.offer("cam1", "g1", "old-result")
    assert callback_entered.wait(1)

    dispatcher.discard_camera("cam1", "g1")
    dispatcher.register_camera("cam1", "g2")
    release_callback.set()

    deadline = time.monotonic() + 1
    while dispatcher.stats()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)

    assert mutations == []
    dispatcher.shutdown(wait=True, timeout=1)


def test_circuit_breaker_drops_work_after_repeated_provider_failures():
    calls = []

    def process(work):
        calls.append(work.payload)
        raise RuntimeError("provider unavailable")

    dispatcher = VLMEnrichmentDispatcher(
        process=process,
        on_result=lambda *_: None,
        failure_threshold=2,
        circuit_cooldown_seconds=10,
    )
    dispatcher.register_camera("cam1", "g1")
    for expected_failures, payload in enumerate(("one", "two"), start=1):
        assert dispatcher.offer("cam1", "g1", payload)
        deadline = time.monotonic() + 1
        while (
            dispatcher.stats()["failed"] < expected_failures
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        # Wait for the active callback to finish before the next offer.
        while dispatcher.stats()["active"] and time.monotonic() < deadline:
            time.sleep(0.005)
    assert dispatcher.offer("cam1", "g1", "three")

    deadline = time.monotonic() + 1
    while dispatcher.stats()["circuitDropped"] < 1 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert calls == ["one", "two"]
    assert dispatcher.stats()["circuitOpen"]
    assert dispatcher.stats()["circuitDropped"] == 1
    dispatcher.shutdown(wait=True, timeout=1)
