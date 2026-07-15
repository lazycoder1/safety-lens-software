"""Bounded, credential-free telemetry for the real-time video pipeline.

This module deliberately owns no camera frames, alert payloads, stream URLs, or
provider identifiers.  Callers register configured camera IDs, then record only
fixed counters and elapsed durations.  All state is process-local, bounded, and
safe to serialize through the public health endpoint.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from uuid import UUID, uuid4


JS_SAFE_COUNTER_MAX = (1 << 53) - 1
DEFAULT_MAX_CAMERAS = 256

FRAME_HISTOGRAM_BOUNDS_MS = (
    5,
    10,
    20,
    40,
    80,
    160,
    320,
    640,
    1_280,
    2_560,
    5_120,
    10_240,
    20_480,
    40_960,
)

ALERT_HISTOGRAM_BOUNDS_MS = (
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    40_000,
    80_000,
    160_000,
    320_000,
    640_000,
    900_000,
)

CAMERA_COUNTER_NAMES = (
    "decodedFrameCount",
    "captureDropCount",
    "captureDuplicateCount",
    "latestSlotDropCount",
    "admissionBusyDropCount",
    "admissionStaleDropCount",
    "admissionOverloadDropCount",
    "schedulerLifecycleDropCount",
    "motionSkipCount",
    "cadenceSkipCount",
    "inferenceCompletedCount",
    "inferenceFailureCount",
    "adaptiveQuietObservationCount",
    "adaptiveUncertainObservationCount",
    "adaptiveActiveObservationCount",
    "adaptiveQuietAdmissionCount",
    "adaptiveUncertainAdmissionCount",
    "adaptiveActiveAdmissionCount",
    "adaptiveQuietInferenceCount",
    "adaptiveUncertainInferenceCount",
    "adaptiveActiveInferenceCount",
    "trackerProjectionFrameCount",
    "trackerProjectedPersonCount",
    "trackerForceRedetectSignalCount",
    "ppePersonCropAttemptCount",
    "phonePersonCropAttemptCount",
    "personCropFallbackCount",
    "personCropFullFrameInvocationCount",
)

CAMERA_LATENCY_NAMES = (
    "decodedIngressToResultMs",
    "decodedIngressToObservationMs",
    "submitToResultMs",
)

ALERT_LATENCY_NAMES = (
    "firstPositiveToConfirmedMs",
    "confirmedToPersistedMs",
    "firstPositiveToPersistedMs",
    "firstPositiveToProviderSuccessMs",
)

# Older collectors called provider acceptance a delivery "handoff". Keep the
# public alias during rollout, but record only the accurately named histogram.
ALERT_LATENCY_ALIASES = {
    "firstPositiveToDeliveryHandoffMs": "firstPositiveToProviderSuccessMs",
}

ALERT_DELIVERY_COUNTER_NAMES = (
    "eligibleCount",
    "deliveredCount",
    "terminalCount",
    "cancelledCount",
    "failedAttemptCount",
    "untrackedFailureAttemptCount",
    "persistenceCensoredCount",
    "outcomeCensoredCount",
    "evictedPendingCount",
)

# Existing deployments may use human-readable IDs such as ``Gate 1``. Keep
# those workers alive while still excluding URL/credential delimiters and
# control characters from the public telemetry key space.
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.: -]{0,127}$")
_OPAQUE_CAMERA_PREFIX = "telemetry-opaque-"


def _saturating_add(current: int, increment: int) -> int:
    return min(JS_SAFE_COUNTER_MAX, current + increment)


def _validated_bounds(bounds_ms: Iterable[float]) -> tuple[float, ...]:
    try:
        parsed = tuple(float(value) for value in bounds_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("histogram bounds must be finite positive numbers") from exc
    if not parsed:
        raise ValueError("histogram bounds cannot be empty")
    if any(not math.isfinite(value) or value <= 0 for value in parsed):
        raise ValueError("histogram bounds must be finite positive numbers")
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("histogram bounds must be strictly increasing")
    return parsed


def _validated_bucket_counts(
    bounds_ms: Sequence[float],
    bucket_counts: Sequence[int],
) -> tuple[int, ...]:
    if len(bucket_counts) != len(bounds_ms) + 1:
        raise ValueError("bucket counts must include one overflow bucket")
    parsed: list[int] = []
    for count in bucket_counts:
        if isinstance(count, bool):
            raise ValueError("bucket counts must be non-negative integers")
        try:
            value = int(count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bucket counts must be non-negative integers") from exc
        if value < 0 or value != count:
            raise ValueError("bucket counts must be non-negative integers")
        parsed.append(value)
    return tuple(parsed)


@dataclass(frozen=True)
class QuantileEstimate:
    """Nearest-rank result expressed as a histogram upper bound."""

    upper_bound_ms: float | None
    overflow: bool = False


def nearest_rank_upper_bound_ms(
    bounds_ms: Sequence[float],
    bucket_counts: Sequence[int],
    quantile: float,
) -> QuantileEstimate:
    """Return the fixed-bucket nearest-rank estimate for ``quantile``.

    ``bucket_counts`` are non-cumulative and must contain one final overflow
    bucket.  An overflow result is explicit because returning the largest finite
    boundary would understate the observed latency.
    """

    bounds = _validated_bounds(bounds_ms)
    counts = _validated_bucket_counts(bounds, bucket_counts)
    try:
        requested = float(quantile)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quantile must be finite and in the interval (0, 1]") from exc
    if not math.isfinite(requested) or not 0 < requested <= 1:
        raise ValueError("quantile must be finite and in the interval (0, 1]")

    sample_count = sum(counts)
    if sample_count == 0:
        return QuantileEstimate(None, False)
    rank = max(1, math.ceil(requested * sample_count))
    cumulative = 0
    for index, count in enumerate(counts):
        cumulative += count
        if cumulative < rank:
            continue
        if index == len(bounds):
            return QuantileEstimate(None, True)
        return QuantileEstimate(bounds[index], False)
    raise AssertionError("validated histogram counts did not contain the requested rank")


class FixedBucketHistogram:
    """Thread-safe histogram with fixed memory and explicit invalid samples."""

    def __init__(self, bounds_ms: Iterable[float]) -> None:
        self._bounds_ms = _validated_bounds(bounds_ms)
        self._bucket_counts = [0] * (len(self._bounds_ms) + 1)
        self._count = 0
        self._invalid_count = 0
        self._maximum_ms: float | None = None
        self._lock = threading.Lock()

    @property
    def bounds_ms(self) -> tuple[float, ...]:
        return self._bounds_ms

    def observe_ms(self, duration_ms: object) -> bool:
        """Observe a non-negative finite duration; return whether it was valid."""

        if isinstance(duration_ms, bool):
            parsed = math.nan
        else:
            try:
                parsed = float(duration_ms)
            except (TypeError, ValueError, OverflowError):
                parsed = math.nan
        if not math.isfinite(parsed) or parsed < 0:
            with self._lock:
                self._invalid_count = _saturating_add(self._invalid_count, 1)
            return False

        bucket = bisect.bisect_left(self._bounds_ms, parsed)
        with self._lock:
            self._count = _saturating_add(self._count, 1)
            self._bucket_counts[bucket] = _saturating_add(
                self._bucket_counts[bucket],
                1,
            )
            if self._maximum_ms is None or parsed > self._maximum_ms:
                self._maximum_ms = parsed
        return True

    def observe_elapsed_ns(self, started_ns: object, completed_ns: object) -> bool:
        """Observe two monotonic nanosecond anchors without mixing clock domains."""

        if isinstance(started_ns, bool) or isinstance(completed_ns, bool):
            return self.observe_ms(math.nan)
        try:
            started = int(started_ns)
            completed = int(completed_ns)
        except (TypeError, ValueError, OverflowError):
            return self.observe_ms(math.nan)
        if started != started_ns or completed != completed_ns:
            return self.observe_ms(math.nan)
        return self.observe_ms((completed - started) / 1_000_000.0)

    def snapshot(self) -> dict:
        with self._lock:
            counts = list(self._bucket_counts)
            sample_count = self._count
            invalid_count = self._invalid_count
            maximum_ms = self._maximum_ms
        p95 = nearest_rank_upper_bound_ms(self._bounds_ms, counts, 0.95)
        p99 = nearest_rank_upper_bound_ms(self._bounds_ms, counts, 0.99)
        return {
            "count": sample_count,
            "invalidCount": invalid_count,
            "overflowCount": counts[-1],
            "bucketCounts": counts,
            "maximumMs": maximum_ms,
            "p95UpperBoundMs": p95.upper_bound_ms,
            "p95Overflow": p95.overflow,
            "p99UpperBoundMs": p99.upper_bound_ms,
            "p99Overflow": p99.overflow,
        }


@dataclass
class _CameraTelemetry:
    generation: int
    active: bool = True
    counters: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in CAMERA_COUNTER_NAMES}
    )
    latency: dict[str, FixedBucketHistogram] = field(
        default_factory=lambda: {
            name: FixedBucketHistogram(FRAME_HISTOGRAM_BOUNDS_MS)
            for name in CAMERA_LATENCY_NAMES
        }
    )
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, counter_name: str, amount: int) -> None:
        with self.lock:
            self.counters[counter_name] = _saturating_add(
                self.counters[counter_name],
                amount,
            )

    def set_active(self, active: bool) -> None:
        with self.lock:
            self.active = bool(active)

    def public_snapshot(self) -> dict:
        with self.lock:
            generation = self.generation
            active = self.active
            counters = dict(self.counters)
        return {
            "generation": generation,
            "active": active,
            "counters": counters,
            "latency": {
                name: histogram.snapshot()
                for name, histogram in self.latency.items()
            },
        }


class TelemetryCapacityError(RuntimeError):
    """Raised when a caller attempts to retain more camera states than allowed."""


class PipelineTelemetry:
    """Process-local registry for bounded camera and real-time alert telemetry."""

    def __init__(
        self,
        *,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
        telemetry_epoch: str | None = None,
    ) -> None:
        if isinstance(max_cameras, bool) or not isinstance(max_cameras, int):
            raise ValueError("max_cameras must be a positive integer")
        if max_cameras < 1:
            raise ValueError("max_cameras must be a positive integer")
        if telemetry_epoch is None:
            telemetry_epoch = str(uuid4())
        else:
            try:
                telemetry_epoch = str(UUID(str(telemetry_epoch)))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("telemetry_epoch must be a UUID") from exc

        self._max_cameras = max_cameras
        self._telemetry_epoch = telemetry_epoch
        self._cameras: dict[str, _CameraTelemetry] = {}
        self._camera_lock = threading.RLock()
        self._alert_latency = {
            name: FixedBucketHistogram(ALERT_HISTOGRAM_BOUNDS_MS)
            for name in ALERT_LATENCY_NAMES
        }
        self._alert_delivery_lock = threading.Lock()
        self._alert_delivery_counters = {
            name: 0 for name in ALERT_DELIVERY_COUNTER_NAMES
        }
        self._alert_delivery_pending = 0

    @property
    def telemetry_epoch(self) -> str:
        return self._telemetry_epoch

    @staticmethod
    def _validate_camera_id(camera_id: object) -> str:
        if not isinstance(camera_id, str):
            raise ValueError("camera_id must be a string")
        if (
            _CAMERA_ID_PATTERN.fullmatch(camera_id)
            and not camera_id.startswith(_OPAQUE_CAMERA_PREFIX)
        ):
            return camera_id
        # Camera map keys predate telemetry validation and may contain slashes,
        # punctuation, Unicode, or even URL-like text.  Optional telemetry must
        # never kill those workers or publish the raw value.  Recompute this
        # deterministic key at every call so no unsafe alias is retained.
        digest = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()
        return f"{_OPAQUE_CAMERA_PREFIX}{digest[:32]}"

    @staticmethod
    def _validate_increment(amount: object) -> int:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("counter increment must be a non-negative integer")
        return min(JS_SAFE_COUNTER_MAX, amount)

    def reset_camera(self, camera_id: str) -> int:
        """Start a clean camera generation and return its generation number."""

        camera_key = self._validate_camera_id(camera_id)
        with self._camera_lock:
            previous = self._cameras.get(camera_key)
            if previous is None and len(self._cameras) >= self._max_cameras:
                raise TelemetryCapacityError("camera telemetry capacity reached")
            generation = 1 if previous is None else previous.generation + 1
            self._cameras[camera_key] = _CameraTelemetry(generation=generation)
            return generation

    def mark_camera_stopped(self, camera_id: str) -> bool:
        """Mark a camera inactive while retaining its final generation snapshot."""

        camera_key = self._validate_camera_id(camera_id)
        with self._camera_lock:
            camera = self._cameras.get(camera_key)
        if camera is None:
            return False
        camera.set_active(False)
        return True

    def remove_camera(self, camera_id: str) -> bool:
        """Forget telemetry only when a camera is terminally deleted."""

        camera_key = self._validate_camera_id(camera_id)
        with self._camera_lock:
            return self._cameras.pop(camera_key, None) is not None

    def increment_camera_counter(
        self,
        camera_id: str,
        counter_name: str,
        amount: int = 1,
    ) -> bool:
        camera_key = self._validate_camera_id(camera_id)
        if counter_name not in CAMERA_COUNTER_NAMES:
            raise ValueError(f"unknown camera counter: {counter_name}")
        increment = self._validate_increment(amount)
        with self._camera_lock:
            camera = self._cameras.get(camera_key)
        if camera is None:
            return False
        camera.increment(counter_name, increment)
        return True

    def observe_camera_latency_ms(
        self,
        camera_id: str,
        latency_name: str,
        duration_ms: object,
    ) -> bool:
        camera_key = self._validate_camera_id(camera_id)
        if latency_name not in CAMERA_LATENCY_NAMES:
            raise ValueError(f"unknown camera latency: {latency_name}")
        with self._camera_lock:
            camera = self._cameras.get(camera_key)
        if camera is None:
            return False
        return camera.latency[latency_name].observe_ms(duration_ms)

    def observe_camera_elapsed_ns(
        self,
        camera_id: str,
        latency_name: str,
        started_ns: object,
        completed_ns: object,
    ) -> bool:
        camera_key = self._validate_camera_id(camera_id)
        if latency_name not in CAMERA_LATENCY_NAMES:
            raise ValueError(f"unknown camera latency: {latency_name}")
        with self._camera_lock:
            camera = self._cameras.get(camera_key)
        if camera is None:
            return False
        return camera.latency[latency_name].observe_elapsed_ns(
            started_ns,
            completed_ns,
        )

    def observe_alert_latency_ms(
        self,
        latency_name: str,
        duration_ms: object,
    ) -> bool:
        canonical_name = ALERT_LATENCY_ALIASES.get(latency_name, latency_name)
        if canonical_name not in ALERT_LATENCY_NAMES:
            raise ValueError(f"unknown alert latency: {latency_name}")
        return self._alert_latency[canonical_name].observe_ms(duration_ms)

    def observe_alert_elapsed_ns(
        self,
        latency_name: str,
        started_ns: object,
        completed_ns: object,
    ) -> bool:
        canonical_name = ALERT_LATENCY_ALIASES.get(latency_name, latency_name)
        if canonical_name not in ALERT_LATENCY_NAMES:
            raise ValueError(f"unknown alert latency: {latency_name}")
        return self._alert_latency[canonical_name].observe_elapsed_ns(
            started_ns,
            completed_ns,
        )

    def register_alert_delivery_targets(self, amount: int) -> None:
        count = self._validate_increment(amount)
        if count == 0:
            return
        with self._alert_delivery_lock:
            self._alert_delivery_counters["eligibleCount"] = _saturating_add(
                self._alert_delivery_counters["eligibleCount"], count
            )
            self._alert_delivery_pending = _saturating_add(
                self._alert_delivery_pending, count
            )

    def record_alert_delivery_outcome(
        self,
        outcome: str,
        *,
        tracked: bool,
    ) -> bool:
        counter_name = {
            "delivered": "deliveredCount",
            "terminal": "terminalCount",
            "cancelled": "cancelledCount",
        }.get(str(outcome))
        if counter_name is None:
            raise ValueError("unknown alert delivery outcome")
        with self._alert_delivery_lock:
            if not tracked or self._alert_delivery_pending < 1:
                self._alert_delivery_counters["outcomeCensoredCount"] = (
                    _saturating_add(
                        self._alert_delivery_counters["outcomeCensoredCount"],
                        1,
                    )
                )
                return False
            self._alert_delivery_counters[counter_name] = _saturating_add(
                self._alert_delivery_counters[counter_name], 1
            )
            self._alert_delivery_pending -= 1
            return True

    def record_alert_delivery_failure_attempt(self, *, tracked: bool) -> None:
        with self._alert_delivery_lock:
            if tracked:
                self._alert_delivery_counters["failedAttemptCount"] = (
                    _saturating_add(
                        self._alert_delivery_counters["failedAttemptCount"],
                        1,
                    )
                )
            else:
                self._alert_delivery_counters["untrackedFailureAttemptCount"] = (
                    _saturating_add(
                        self._alert_delivery_counters[
                            "untrackedFailureAttemptCount"
                        ],
                        1,
                    )
                )

    def record_alert_persistence_censored(self) -> None:
        with self._alert_delivery_lock:
            self._alert_delivery_counters["persistenceCensoredCount"] = (
                _saturating_add(
                    self._alert_delivery_counters["persistenceCensoredCount"],
                    1,
                )
            )

    def censor_pending_alert_deliveries(self, amount: int) -> None:
        count = self._validate_increment(amount)
        if count == 0:
            return
        with self._alert_delivery_lock:
            removed = min(count, self._alert_delivery_pending)
            self._alert_delivery_pending -= removed
            self._alert_delivery_counters["evictedPendingCount"] = (
                _saturating_add(
                    self._alert_delivery_counters["evictedPendingCount"],
                    count,
                )
            )

    def _alert_delivery_snapshot(self) -> dict:
        with self._alert_delivery_lock:
            return {
                "unit": "initial-external-delivery-target",
                "counters": dict(self._alert_delivery_counters),
                "pending": self._alert_delivery_pending,
            }

    def public_camera_snapshot(self, camera_id: str) -> dict | None:
        """Return one aggregate snapshot without echoing the camera ID."""

        camera_key = self._validate_camera_id(camera_id)
        with self._camera_lock:
            camera = self._cameras.get(camera_key)
        return None if camera is None else camera.public_snapshot()

    def public_snapshot(self) -> dict:
        """Return the complete fixed-schema, credential-free public snapshot."""

        with self._camera_lock:
            camera_items = list(self._cameras.items())
        alert_latency = {
            name: histogram.snapshot()
            for name, histogram in self._alert_latency.items()
        }
        alert_latency.update(
            {
                alias: alert_latency[canonical]
                for alias, canonical in ALERT_LATENCY_ALIASES.items()
            }
        )
        return {
            "schemaVersion": 1,
            "telemetryEpoch": self._telemetry_epoch,
            "counterMaximum": JS_SAFE_COUNTER_MAX,
            "quantileMethod": "nearest-rank-upper-bucket-bound",
            "frameHistogramBoundsMs": list(FRAME_HISTOGRAM_BOUNDS_MS),
            "alertHistogramBoundsMs": list(ALERT_HISTOGRAM_BOUNDS_MS),
            "cameraCounterNames": list(CAMERA_COUNTER_NAMES),
            "cameras": {
                camera_id: camera.public_snapshot()
                for camera_id, camera in sorted(camera_items)
            },
            "alerts": {
                "realTimeOnly": True,
                "latency": alert_latency,
                "latencyAliases": dict(ALERT_LATENCY_ALIASES),
                "deliveryCoverage": self._alert_delivery_snapshot(),
            },
        }


telemetry = PipelineTelemetry()
