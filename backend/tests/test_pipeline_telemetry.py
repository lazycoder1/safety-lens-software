import json
import math
import threading
from uuid import UUID

import pytest

import pipeline_telemetry
from pipeline_telemetry import (
    ALERT_HISTOGRAM_BOUNDS_MS,
    ALERT_LATENCY_ALIASES,
    ALERT_LATENCY_NAMES,
    CAMERA_COUNTER_NAMES,
    CAMERA_LATENCY_NAMES,
    FRAME_HISTOGRAM_BOUNDS_MS,
    JS_SAFE_COUNTER_MAX,
    FixedBucketHistogram,
    PipelineTelemetry,
    QuantileEstimate,
    TelemetryCapacityError,
    nearest_rank_upper_bound_ms,
)


_EPOCH = "0f09fb70-523e-4459-8308-91922d0a94bd"


def test_fixed_histogram_places_boundaries_and_overflow_without_raw_samples():
    histogram = FixedBucketHistogram((5, 10, 20))

    for value in (0, 5, 5.1, 10, 19.9, 20, 20.1):
        assert histogram.observe_ms(value)

    snapshot = histogram.snapshot()
    assert snapshot == {
        "count": 7,
        "invalidCount": 0,
        "overflowCount": 1,
        "bucketCounts": [2, 2, 2, 1],
        "maximumMs": 20.1,
        "p95UpperBoundMs": None,
        "p95Overflow": True,
        "p99UpperBoundMs": None,
        "p99Overflow": True,
    }
    assert not hasattr(histogram, "samples")


@pytest.mark.parametrize(
    "value",
    (-1, math.nan, math.inf, -math.inf, None, "bad", True),
)
def test_fixed_histogram_rejects_invalid_samples_explicitly(value):
    histogram = FixedBucketHistogram((10, 20))

    assert histogram.observe_ms(value) is False
    snapshot = histogram.snapshot()
    assert snapshot["count"] == 0
    assert snapshot["invalidCount"] == 1
    assert snapshot["bucketCounts"] == [0, 0, 0]


def test_elapsed_nanosecond_observation_rejects_clock_reversal_and_fractional_anchors():
    histogram = FixedBucketHistogram((5, 10))

    assert histogram.observe_elapsed_ns(1_000_000, 6_000_000)
    assert histogram.observe_elapsed_ns(8_000_000, 7_000_000) is False
    assert histogram.observe_elapsed_ns(1.5, 2) is False

    assert histogram.snapshot()["bucketCounts"] == [1, 0, 0]
    assert histogram.snapshot()["invalidCount"] == 2


def test_nearest_rank_quantile_returns_upper_bound_and_explicit_overflow():
    bounds = (10, 20, 40)

    assert nearest_rank_upper_bound_ms(bounds, (0, 0, 0, 0), 0.99) == QuantileEstimate(
        None,
        False,
    )
    assert nearest_rank_upper_bound_ms(bounds, (94, 1, 4, 1), 0.95) == QuantileEstimate(
        20.0,
        False,
    )
    assert nearest_rank_upper_bound_ms(bounds, (94, 1, 4, 1), 0.99) == QuantileEstimate(
        40.0,
        False,
    )
    assert nearest_rank_upper_bound_ms(bounds, (94, 1, 4, 1), 1.0) == QuantileEstimate(
        None,
        True,
    )


@pytest.mark.parametrize(
    ("bounds", "counts", "quantile"),
    [
        ((), (0,), 0.95),
        ((10, 5), (0, 0, 0), 0.95),
        ((10,), (0,), 0.95),
        ((10,), (0, -1), 0.95),
        ((10,), (0, 1.5), 0.95),
        ((10,), (0, 0), 0),
        ((10,), (0, 0), 1.1),
    ],
)
def test_nearest_rank_quantile_rejects_malformed_histograms(bounds, counts, quantile):
    with pytest.raises(ValueError):
        nearest_rank_upper_bound_ms(bounds, counts, quantile)


def test_camera_generation_resets_but_stop_retains_final_snapshot():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    assert registry.reset_camera("cam-1") == 1
    assert registry.increment_camera_counter("cam-1", "decodedFrameCount", 4)
    assert registry.observe_camera_latency_ms(
        "cam-1",
        "decodedIngressToResultMs",
        17,
    )

    assert registry.mark_camera_stopped("cam-1") is True
    stopped = registry.public_camera_snapshot("cam-1")
    assert stopped["generation"] == 1
    assert stopped["active"] is False
    assert stopped["counters"]["decodedFrameCount"] == 4
    assert stopped["latency"]["decodedIngressToResultMs"]["count"] == 1

    assert registry.reset_camera("cam-1") == 2
    restarted = registry.public_camera_snapshot("cam-1")
    assert restarted["generation"] == 2
    assert restarted["active"] is True
    assert set(restarted["counters"]) == set(CAMERA_COUNTER_NAMES)
    assert all(value == 0 for value in restarted["counters"].values())
    assert all(
        histogram["count"] == 0
        for histogram in restarted["latency"].values()
    )


def test_unknown_camera_observations_are_safe_noops_until_registered():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    assert registry.increment_camera_counter("cam-1", "decodedFrameCount") is False
    assert registry.observe_camera_latency_ms(
        "cam-1",
        "submitToResultMs",
        5,
    ) is False
    assert registry.observe_camera_elapsed_ns(
        "cam-1",
        "submitToResultMs",
        0,
        1_000_000,
    ) is False
    assert registry.mark_camera_stopped("cam-1") is False
    assert registry.public_camera_snapshot("cam-1") is None


def test_camera_counters_saturate_at_javascript_safe_integer_maximum():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")

    registry.increment_camera_counter(
        "cam-1",
        "latestSlotDropCount",
        JS_SAFE_COUNTER_MAX - 1,
    )
    registry.increment_camera_counter("cam-1", "latestSlotDropCount", 100)

    snapshot = registry.public_camera_snapshot("cam-1")
    assert snapshot["counters"]["latestSlotDropCount"] == JS_SAFE_COUNTER_MAX


def test_histogram_counters_saturate_without_changing_memory_shape():
    histogram = FixedBucketHistogram((10,))
    histogram._count = JS_SAFE_COUNTER_MAX
    histogram._bucket_counts[0] = JS_SAFE_COUNTER_MAX

    assert histogram.observe_ms(1)

    snapshot = histogram.snapshot()
    assert snapshot["count"] == JS_SAFE_COUNTER_MAX
    assert snapshot["bucketCounts"] == [JS_SAFE_COUNTER_MAX, 0]


def test_all_distinct_drop_skip_and_latency_metrics_are_present():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")

    for counter_name in CAMERA_COUNTER_NAMES:
        assert registry.increment_camera_counter("cam-1", counter_name)
    for latency_name in CAMERA_LATENCY_NAMES:
        assert registry.observe_camera_elapsed_ns(
            "cam-1",
            latency_name,
            1_000_000,
            4_000_000,
        )
    for latency_name in ALERT_LATENCY_NAMES:
        assert registry.observe_alert_elapsed_ns(
            latency_name,
            1_000_000,
            6_000_000,
        )

    snapshot = registry.public_snapshot()
    camera = snapshot["cameras"]["cam-1"]
    assert camera["counters"] == {name: 1 for name in CAMERA_COUNTER_NAMES}
    assert set(camera["latency"]) == set(CAMERA_LATENCY_NAMES)
    assert all(value["count"] == 1 for value in camera["latency"].values())
    assert set(snapshot["alerts"]["latency"]) == (
        set(ALERT_LATENCY_NAMES) | set(ALERT_LATENCY_ALIASES)
    )
    assert all(
        value["count"] == 1
        for value in snapshot["alerts"]["latency"].values()
    )


def test_public_snapshot_has_fixed_bounds_and_no_event_or_credential_fields():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")
    registry.increment_camera_counter("cam-1", "captureDropCount", 3)
    registry.observe_alert_latency_ms("firstPositiveToPersistedMs", 125)

    snapshot = registry.public_snapshot()
    serialized = json.dumps(snapshot, sort_keys=True)
    assert snapshot["schemaVersion"] == 1
    assert snapshot["telemetryEpoch"] == _EPOCH
    assert snapshot["counterMaximum"] == JS_SAFE_COUNTER_MAX
    assert snapshot["frameHistogramBoundsMs"] == list(FRAME_HISTOGRAM_BOUNDS_MS)
    assert snapshot["alertHistogramBoundsMs"] == list(ALERT_HISTOGRAM_BOUNDS_MS)
    assert snapshot["alerts"]["realTimeOnly"] is True
    for forbidden in (
        "rtsp",
        "password",
        "token",
        "snapshot",
        "alertId",
        "ruleName",
        "outputId",
        "provider",
        "timestamp",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "camera_id",
    (
        "rtsp://user:secret@camera/live",
        "cam\nsecret",
        "@camera",
        "x" * 129,
        "",
    ),
)
def test_camera_identifiers_cannot_smuggle_credentials_into_public_telemetry(camera_id):
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    assert registry.reset_camera(camera_id) == 1
    snapshot = registry.public_snapshot()
    assert len(snapshot["cameras"]) == 1
    public_id = next(iter(snapshot["cameras"]))
    assert public_id.startswith("telemetry-opaque-")
    if camera_id:
        assert camera_id not in json.dumps(snapshot, sort_keys=True)
    assert registry.public_camera_snapshot(camera_id) is not None


def test_non_string_camera_identifier_is_rejected():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    with pytest.raises(ValueError, match="must be a string"):
        registry.reset_camera(None)


def test_legacy_human_readable_camera_identifier_keeps_worker_telemetry_alive():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    assert registry.reset_camera("Gate 1") == 1
    assert "Gate 1" in registry.public_snapshot()["cameras"]


def test_camera_capacity_and_terminal_removal_bound_registry_memory():
    registry = PipelineTelemetry(max_cameras=2, telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")
    registry.reset_camera("cam-2")

    with pytest.raises(TelemetryCapacityError):
        registry.reset_camera("cam-3")

    assert registry.remove_camera("cam-1") is True
    assert registry.remove_camera("cam-1") is False
    assert registry.reset_camera("cam-3") == 1
    assert set(registry.public_snapshot()["cameras"]) == {"cam-2", "cam-3"}


def test_metric_names_and_counter_increments_are_strictly_bounded():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")

    with pytest.raises(ValueError, match="unknown camera counter"):
        registry.increment_camera_counter("cam-1", "rtsp://secret")
    with pytest.raises(ValueError, match="unknown camera latency"):
        registry.observe_camera_latency_ms("cam-1", "secretLatency", 1)
    with pytest.raises(ValueError, match="unknown alert latency"):
        registry.observe_alert_latency_ms("secretLatency", 1)
    for amount in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            registry.increment_camera_counter("cam-1", "captureDropCount", amount)


def test_delivery_latency_alias_shares_one_histogram_without_double_observation():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    assert registry.observe_alert_latency_ms(
        "firstPositiveToDeliveryHandoffMs", 125
    )

    latency = registry.public_snapshot()["alerts"]["latency"]
    canonical = latency["firstPositiveToProviderSuccessMs"]
    compatibility = latency["firstPositiveToDeliveryHandoffMs"]
    assert canonical == compatibility
    assert canonical["count"] == 1


def test_delivery_coverage_keeps_latency_denominator_and_pending_visible():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.register_alert_delivery_targets(3)
    registry.record_alert_delivery_failure_attempt(tracked=True)
    assert registry.record_alert_delivery_outcome("delivered", tracked=True)
    assert registry.record_alert_delivery_outcome("terminal", tracked=True)

    coverage = registry.public_snapshot()["alerts"]["deliveryCoverage"]
    assert coverage == {
        "unit": "initial-external-delivery-target",
        "counters": {
            "eligibleCount": 3,
            "deliveredCount": 1,
            "terminalCount": 1,
            "cancelledCount": 0,
            "failedAttemptCount": 1,
            "untrackedFailureAttemptCount": 0,
            "persistenceCensoredCount": 0,
            "outcomeCensoredCount": 0,
            "evictedPendingCount": 0,
        },
        "pending": 1,
    }


def test_untracked_delivery_outcome_is_censored_not_added_to_success_numerator():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)

    assert not registry.record_alert_delivery_outcome("delivered", tracked=False)

    coverage = registry.public_snapshot()["alerts"]["deliveryCoverage"]
    assert coverage["counters"]["deliveredCount"] == 0
    assert coverage["counters"]["outcomeCensoredCount"] == 1


def test_threaded_observation_is_lossless_and_snapshot_is_consistent():
    registry = PipelineTelemetry(telemetry_epoch=_EPOCH)
    registry.reset_camera("cam-1")
    thread_count = 8
    observations_per_thread = 2_000

    def observe() -> None:
        for _ in range(observations_per_thread):
            assert registry.increment_camera_counter(
                "cam-1",
                "inferenceCompletedCount",
            )
            assert registry.observe_camera_latency_ms(
                "cam-1",
                "decodedIngressToResultMs",
                12,
            )
            assert registry.observe_alert_latency_ms(
                "firstPositiveToConfirmedMs",
                250,
            )

    threads = [threading.Thread(target=observe) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = thread_count * observations_per_thread
    snapshot = registry.public_snapshot()
    camera = snapshot["cameras"]["cam-1"]
    assert camera["counters"]["inferenceCompletedCount"] == expected
    assert camera["latency"]["decodedIngressToResultMs"]["count"] == expected
    assert sum(
        camera["latency"]["decodedIngressToResultMs"]["bucketCounts"]
    ) == expected
    alert = snapshot["alerts"]["latency"]["firstPositiveToConfirmedMs"]
    assert alert["count"] == expected
    assert sum(alert["bucketCounts"]) == expected


def test_module_singleton_has_a_valid_process_epoch_and_public_schema():
    assert str(UUID(pipeline_telemetry.telemetry.telemetry_epoch)) == (
        pipeline_telemetry.telemetry.telemetry_epoch
    )
    snapshot = pipeline_telemetry.telemetry.public_snapshot()
    assert snapshot["schemaVersion"] == 1
    assert isinstance(snapshot["cameras"], dict)
