"""Fail-open, process-wide VLM enrichment scheduling.

The real-time detector must never wait for VLM work.  This dispatcher accepts
one replaceable item per camera, bounds total memory, expires old work, and
fences late results when a camera generation changes.  Provider calls run on a
single daemon worker so camera start/stop/restart never joins or waits for them.
"""

from __future__ import annotations

import ipaddress
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse


_LOCAL_ENDPOINT_ALIASES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
        "gateway.docker.internal",
        "host.containers.internal",
        "kubernetes.docker.internal",
        "docker.for.mac.host.internal",
        "docker.for.mac.localhost",
        "docker.for.win.localhost",
    }
)


def _ip_address(value: object) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        # Scoped IPv6 addresses identify a local interface and must never be
        # accepted as remote VLM targets.  Strip either URL-encoded or raw
        # scope syntax before applying the address safety checks.
        normalized = str(value).replace("%25", "%").split("%", 1)[0]
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _resolve_ip_addresses(
    host: str,
    port: int | None,
) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        if len(item) < 5 or not item[4]:
            continue
        address = _ip_address(item[4][0])
        if address is not None:
            addresses.add(address)
    return addresses


def _local_interface_addresses(
    local_names: set[str],
) -> tuple[set[ipaddress.IPv4Address | ipaddress.IPv6Address], bool]:
    addresses = _interface_ip_addresses()
    interface_enumeration_succeeded = bool(addresses)
    resolved_local_name = False
    for name in local_names:
        if not name:
            continue
        try:
            resolved = _resolve_ip_addresses(name, None)
        except (OSError, OverflowError, ValueError):
            continue
        if resolved:
            resolved_local_name = True
            addresses.update(resolved)
    return addresses, resolved_local_name or interface_enumeration_succeeded


def _interface_ip_addresses() -> set[
    ipaddress.IPv4Address | ipaddress.IPv6Address
]:
    """Enumerate local unicast interfaces without shelling out or logging them."""
    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    configured = os.environ.get("SAFETYLENS_LOCAL_INTERFACE_ADDRESSES", "")
    for raw in configured.split(","):
        address = _ip_address(raw.strip())
        if address is not None:
            addresses.add(address)

    # Jetson production is Linux. SIOCGIFADDR covers every configured IPv4
    # interface even when hostname/FQDN resolves only to 127.0.1.1.
    try:
        import fcntl

        for _index, name in socket.if_nameindex():
            request = struct.pack("256s", name.encode("utf-8")[:15])
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
                response = fcntl.ioctl(handle.fileno(), 0x8915, request)
            address = _ip_address(socket.inet_ntoa(response[20:24]))
            if address is not None:
                addresses.add(address)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        with open("/proc/net/if_inet6", encoding="ascii") as handle:
            for line in handle:
                raw = line.split(maxsplit=1)[0]
                if len(raw) != 32:
                    continue
                address = _ip_address(str(ipaddress.IPv6Address(int(raw, 16))))
                if address is not None:
                    addresses.add(address)
    except (OSError, ValueError):
        pass
    return addresses


@dataclass(frozen=True)
class VLMEnrichmentWork:
    camera_id: str
    generation: str
    payload: Any
    enqueued_monotonic: float
    sequence: int


def remote_vlm_endpoint_allowed(url: str) -> bool:
    """Reject on-device endpoints and DNS-rebindable remote-only targets.

    The HTTP client would otherwise resolve a hostname again after this
    validation, allowing DNS to change from a remote address to a Jetson-local
    address between the two lookups.  Remote-only VLM endpoints therefore use
    an explicit IP literal.  This is a fail-closed operational constraint, not
    a general URL validator.
    """
    try:
        parsed = urlparse(str(url))
        host = parsed.hostname
        _port = parsed.port
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    normalized = host.rstrip(".").lower()
    try:
        hostname = socket.gethostname().rstrip(".").lower()
        fqdn = socket.getfqdn().rstrip(".").lower()
    except OSError:
        return False
    local_names = {hostname, fqdn}
    if normalized in _LOCAL_ENDPOINT_ALIASES or normalized in local_names:
        return False

    literal_address = _ip_address(normalized)
    if literal_address is None:
        return False
    target_addresses = {literal_address}

    if any(
        address.is_loopback
        or address.is_unspecified
        or address.is_link_local
        or address.is_multicast
        for address in target_addresses
    ):
        return False

    local_addresses, local_resolution_succeeded = _local_interface_addresses(local_names)
    if not local_resolution_succeeded:
        # All targets fail closed when local interfaces cannot be enumerated;
        # otherwise a literal address for this device could slip through too.
        return False
    return target_addresses.isdisjoint(local_addresses)


class VLMEnrichmentDispatcher:
    """One-worker advisory enrichment queue with latest-per-camera semantics."""

    def __init__(
        self,
        *,
        process: Callable[[VLMEnrichmentWork], Any],
        on_result: Callable[[VLMEnrichmentWork, Any, float], None],
        maximum_pending_cameras: int = 32,
        maximum_queue_age_seconds: float = 15.0,
        failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if maximum_pending_cameras < 1:
            raise ValueError("maximum_pending_cameras must be positive")
        if not math.isfinite(maximum_queue_age_seconds) or maximum_queue_age_seconds <= 0:
            raise ValueError("maximum_queue_age_seconds must be positive and finite")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if not math.isfinite(circuit_cooldown_seconds) or circuit_cooldown_seconds < 0:
            raise ValueError("circuit_cooldown_seconds must be finite and non-negative")
        self._process = process
        self._on_result = on_result
        self._maximum_pending_cameras = int(maximum_pending_cameras)
        self._maximum_queue_age_seconds = float(maximum_queue_age_seconds)
        self._failure_threshold = int(failure_threshold)
        self._circuit_cooldown_seconds = float(circuit_cooldown_seconds)
        self._monotonic = monotonic
        self._condition = threading.Condition()
        self._pending: dict[str, VLMEnrichmentWork] = {}
        self._generations: dict[str, str] = {}
        self._sequence = 0
        self._accepting = False
        self._stop = False
        self._worker: threading.Thread | None = None
        self._active_camera: str | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._counters = {
            "offered": 0,
            "replaced": 0,
            "capacityDropped": 0,
            "staleDropped": 0,
            "circuitDropped": 0,
            "processed": 0,
            "failed": 0,
            "lateResultDiscarded": 0,
            "resultCallbackFailed": 0,
        }

    def start(self) -> None:
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                self._accepting = True
                return
            self._stop = False
            self._accepting = True
            self._worker = threading.Thread(
                target=self._run,
                name="vlm-enrichment",
                daemon=True,
            )
            self._worker.start()

    def register_camera(self, camera_id: str, generation: str) -> None:
        camera_id = str(camera_id)
        generation = str(generation)
        if not camera_id or not generation:
            raise ValueError("camera_id and generation are required")
        with self._condition:
            self._generations[camera_id] = generation
            existing = self._pending.get(camera_id)
            if existing is not None and existing.generation != generation:
                self._pending.pop(camera_id, None)
            self._condition.notify_all()

    def discard_camera(self, camera_id: str, generation: str | None = None) -> None:
        camera_id = str(camera_id)
        with self._condition:
            current = self._generations.get(camera_id)
            if generation is not None and current != str(generation):
                return
            self._generations.pop(camera_id, None)
            self._pending.pop(camera_id, None)
            self._condition.notify_all()

    def is_current_generation(self, camera_id: str, generation: str) -> bool:
        """Revalidate a completed work item immediately before result mutation."""
        with self._condition:
            return self._generations.get(camera_id) == generation

    def run_if_current(
        self,
        camera_id: str,
        generation: str,
        action: Callable[[], Any],
    ) -> bool:
        """Run one short result action before a matching discard can complete.

        A check followed by an unlocked mutation has a stop/restart race: the
        camera generation can change between those operations.  Holding the
        generation condition for the action makes ``discard_camera`` a barrier.
        Result actions must therefore remain short and non-blocking.
        """
        with self._condition:
            if self._generations.get(str(camera_id)) != str(generation):
                return False
            action()
            return True

    def offer(self, camera_id: str, generation: str, payload: Any) -> bool:
        """Offer enrichment without waiting for provider work or queue space."""
        self.start()
        camera_id = str(camera_id)
        generation = str(generation)
        now = self._monotonic()
        with self._condition:
            if not self._accepting or self._generations.get(camera_id) != generation:
                return False
            self._counters["offered"] += 1
            self._sequence += 1
            work = VLMEnrichmentWork(
                camera_id=camera_id,
                generation=generation,
                payload=payload,
                enqueued_monotonic=now,
                sequence=self._sequence,
            )
            if camera_id in self._pending:
                self._counters["replaced"] += 1
            elif len(self._pending) >= self._maximum_pending_cameras:
                self._counters["capacityDropped"] += 1
                return False
            self._pending[camera_id] = work
            self._condition.notify_all()
            return True

    def _next_work_locked(self) -> VLMEnrichmentWork | None:
        if not self._pending:
            return None
        return min(self._pending.values(), key=lambda item: item.sequence)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                work = self._next_work_locked()
                if work is None:
                    continue
                self._pending.pop(work.camera_id, None)
                now = self._monotonic()
                if now - work.enqueued_monotonic > self._maximum_queue_age_seconds:
                    self._counters["staleDropped"] += 1
                    continue
                if now < self._circuit_open_until:
                    self._counters["circuitDropped"] += 1
                    continue
                self._active_camera = work.camera_id

            started = self._monotonic()
            try:
                result = self._process(work)
            except Exception:
                with self._condition:
                    self._counters["failed"] += 1
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._failure_threshold:
                        self._circuit_open_until = (
                            self._monotonic() + self._circuit_cooldown_seconds
                        )
            else:
                elapsed = max(0.0, self._monotonic() - started)
                with self._condition:
                    self._consecutive_failures = 0
                    self._circuit_open_until = 0.0
                    current_generation = self._generations.get(work.camera_id)
                    if current_generation != work.generation:
                        self._counters["lateResultDiscarded"] += 1
                        deliver = False
                    else:
                        self._counters["processed"] += 1
                        deliver = True
                if deliver:
                    try:
                        self._on_result(work, result, elapsed)
                    except Exception:
                        with self._condition:
                            self._counters["resultCallbackFailed"] += 1
            finally:
                with self._condition:
                    self._active_camera = None
                    self._condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._condition:
            now = self._monotonic()
            return {
                "running": bool(self._worker and self._worker.is_alive()),
                "accepting": self._accepting,
                "pendingCameras": len(self._pending),
                "pendingCapacity": self._maximum_pending_cameras,
                "active": self._active_camera is not None,
                "circuitOpen": now < self._circuit_open_until,
                "circuitRetryAfterSeconds": round(
                    max(0.0, self._circuit_open_until - now), 3
                ),
                **self._counters,
            }

    def shutdown(self, *, wait: bool = False, timeout: float = 0.0) -> bool:
        """Stop accepting immediately; provider work is never camera-owned."""
        with self._condition:
            self._accepting = False
            self._pending.clear()
            self._generations.clear()
            self._stop = True
            worker = self._worker
            self._condition.notify_all()
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, timeout))
        return not bool(worker and worker.is_alive())
