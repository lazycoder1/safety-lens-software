"""Focused tests for the live pipeline runtime benchmark collector."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_live_pipeline_runtime.py"
EPOCH_A = "00000000-0000-4000-8000-000000000001"
EPOCH_B = "00000000-0000-4000-8000-000000000002"
FRAME_BOUNDS = [10, 20, 40]
ALERT_BOUNDS = [100, 200, 500]


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_live_pipeline_runtime",
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


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _raw_health(
    benchmark,
    *,
    epoch: str = EPOCH_A,
    generation: int = 1,
    decoded: int = 0,
    completed: int = 0,
    capture_drop: int = 0,
    latest_drop: int = 0,
    completed_histogram: list[int] | None = None,
    alert_histogram: list[int] | None = None,
    scheduler_counters: dict[str, int] | None = None,
    vlm_counters: dict[str, int] | None = None,
    frame_age_seconds: float = 0.2,
    camera_id: str = "cam1",
    stream_type: str = "rtsp",
    capture_backend: str = "gstreamer_nvdec",
    hardware_active: bool = True,
    hardware_fallback: bool = False,
    frame_width: int | None = 960,
    frame_height: int | None = 540,
    transport_counters: dict[str, dict[str, int]] | None = None,
    alert_invalid_count: int = 0,
    alert_delivery_counters: dict[str, int] | None = None,
    alert_delivery_pending: int = 0,
    server_timestamp: float = 999.0,
    camera_counters: dict[str, int] | None = None,
) -> dict:
    counters = {name: 0 for name in benchmark.CAMERA_COUNTER_NAMES}
    counters.update(
        {
            "decodedFrameCount": decoded,
            "inferenceCompletedCount": completed,
            "captureDropCount": capture_drop,
            "latestSlotDropCount": latest_drop,
        }
    )
    counters.update(camera_counters or {})
    frame_histogram = completed_histogram or [0, 0, 0, 0]
    alert_counts = alert_histogram or [0, 0, 0, 0]
    camera_latency = {
        name: {"bucketCounts": list(frame_histogram)}
        for name in benchmark.CAMERA_LATENCY_NAMES
    }
    alert_latency = {
        name: {
            "bucketCounts": list(alert_counts),
            "invalidCount": alert_invalid_count,
        }
        for name in benchmark.ALERT_LATENCY_NAMES
    }
    delivery_counters = {
        name: 0 for name in benchmark.ALERT_DELIVERY_COUNTER_NAMES
    }
    delivery_counters.update(alert_delivery_counters or {})
    return {
        "status": "ok",
        "timestamp": _iso_timestamp(server_timestamp),
        "cameras": [
            {
                "id": camera_id,
                "name": "must never be copied",
                "video": "rtsp://user:password@private.example/live",
                "frameFresh": frame_age_seconds < 2,
                "workerRunning": True,
                "lastFrameAgeSeconds": frame_age_seconds,
                "streamType": stream_type,
                "frameWidth": frame_width,
                "frameHeight": frame_height,
                "connection": {
                    "captureBackend": capture_backend,
                    "hardwareAccelerationActive": hardware_active,
                    "hardwareFallback": hardware_fallback,
                    "appsinkLatestBufferDropsObservable": False,
                    "appsinkLatestBufferDropMethod": "unavailable",
                    "captureDropAccounting": "videorate-only",
                    "captureDropCountIsLowerBound": True,
                    "decoderPolicyDropAccounting": "not-configured",
                },
            }
        ],
        "pipelineTelemetry": {
            "telemetryEpoch": epoch,
            "frameHistogramBoundsMs": FRAME_BOUNDS,
            "alertHistogramBoundsMs": ALERT_BOUNDS,
            "cameras": {
                camera_id: {
                    "generation": generation,
                    "active": True,
                    "counters": counters,
                    "latency": camera_latency,
                }
            },
            "alerts": {
                "realTimeOnly": True,
                "latency": alert_latency,
                "deliveryCoverage": {
                    "unit": "initial-external-delivery-target",
                    "counters": delivery_counters,
                    "pending": alert_delivery_pending,
                },
            },
        },
        "sharedInferenceScheduler": {
            "running": True,
            "accepting": True,
            "registeredCameras": 1,
            "counters": scheduler_counters or {},
            "latency": {
                "queue": {
                    "sampleCount": 5,
                    "medianMs": 2,
                    "p95Ms": 6,
                    "p99Ms": 8,
                    "maxMs": 9,
                }
            },
        },
        "vlmEnrichment": {
            "running": True,
            "accepting": True,
            "pendingCameras": 0,
            **(vlm_counters or {}),
        },
        "inferenceTransport": transport_counters or {},
    }


def _sample(benchmark, monotonic: float, **health_kwargs) -> dict:
    health_kwargs.setdefault("server_timestamp", 999.0 + monotonic)
    raw = _raw_health(benchmark, **health_kwargs)
    return {
        "monotonic": monotonic,
        "httpRoundTripMs": 5.0,
        "health": benchmark._sanitize_health_payload(
            raw,
            received_wall_seconds=1000.0 + monotonic,
        ),
    }


def _multi_camera_sample(benchmark, monotonic: float, cameras: list[dict]) -> dict:
    first_camera = dict(cameras[0])
    first_camera.setdefault("server_timestamp", 999.0 + monotonic)
    raw = _raw_health(benchmark, **first_camera)
    for camera in cameras[1:]:
        extra_camera = dict(camera)
        extra_camera.setdefault("server_timestamp", 999.0 + monotonic)
        extra = _raw_health(benchmark, **extra_camera)
        raw["cameras"].extend(extra["cameras"])
        raw["pipelineTelemetry"]["cameras"].update(
            extra["pipelineTelemetry"]["cameras"]
        )
    return {
        "monotonic": monotonic,
        "httpRoundTripMs": 5.0,
        "health": benchmark._sanitize_health_payload(
            raw,
            received_wall_seconds=1000.0 + monotonic,
        ),
    }


def _summarize(benchmark, samples: list[dict], **overrides) -> dict:
    observed_span = max(
        0.0,
        float(samples[-1]["monotonic"]) - float(samples[0]["monotonic"]),
    )
    configured_duration = max(10.0, observed_span)
    arguments = {
        "configured_duration_seconds": configured_duration,
        "configured_poll_interval_seconds": min(10.0, configured_duration),
        "demand_fps": {},
        "require_hardware_decode": False,
        "require_max_frame_dimension": None,
        "camera_basis": "physical",
        "equivalent_camera_count": None,
        "tegrastats_source": "none",
        "tegrastats_interval_seconds": 1.0,
        "tegrastats_errors": {},
        "thermal_warning_c": 85.0,
    }
    arguments.update(overrides)
    return benchmark._summarize(samples, [], **arguments)


def test_counter_deltas_segment_by_epoch_and_generation_and_ignore_resets(benchmark):
    samples = [
        _sample(benchmark, 0, decoded=100, completed=50),
        _sample(benchmark, 10, decoded=140, completed=70),
        # A generation change starts a new baseline and contributes no delta.
        _sample(benchmark, 20, generation=2, decoded=5, completed=2),
        _sample(benchmark, 30, generation=2, decoded=25, completed=12),
        # A decrease inside the same identity is a reset for only that counter.
        _sample(benchmark, 40, generation=2, decoded=10, completed=15),
        # A process epoch change also starts a new baseline.
        _sample(benchmark, 50, epoch=EPOCH_B, decoded=8, completed=3),
        _sample(benchmark, 60, epoch=EPOCH_B, decoded=28, completed=13),
    ]

    summary = _summarize(benchmark, samples)
    camera = summary["cameras"]["cam1"]

    assert camera["decodedFrames"] == 80
    assert camera["inferenceCompleted"] == 43
    assert camera["generationTransitions"] == 1
    assert camera["counterResetCounts"] == {"decodedFrameCount": 1}
    assert camera["telemetryIdentitySegments"] == 3
    assert camera["decodedFps"] == pytest.approx(80 / 30, abs=1e-6)
    assert camera["effectiveInferenceFps"] == pytest.approx(43 / 40, abs=1e-6)
    assert summary["health"]["telemetryEpochTransitions"] == 1


def test_demand_normalized_jain_uses_relative_satisfaction(benchmark):
    equal_satisfaction = benchmark._demand_normalized_jain(
        {"cam1": 5.0, "cam2": 10.0},
        {"cam1": 10.0, "cam2": 20.0},
    )
    unequal_satisfaction = benchmark._demand_normalized_jain(
        {"cam1": 10.0, "cam2": 2.0},
        {"cam1": 10.0, "cam2": 10.0},
    )

    assert equal_satisfaction == 1.0
    assert unequal_satisfaction == pytest.approx(0.692308)
    assert benchmark._demand_normalized_jain({}, {}) is None


def test_summary_reports_adaptive_tracker_and_crop_rates_by_camera(benchmark):
    first = {
        "adaptiveQuietObservationCount": 10,
        "adaptiveQuietAdmissionCount": 2,
        "adaptiveQuietInferenceCount": 1,
        "trackerProjectionFrameCount": 3,
        "trackerProjectedPersonCount": 4,
        "trackerForceRedetectSignalCount": 1,
        "ppePersonCropAttemptCount": 2,
        "phonePersonCropAttemptCount": 1,
        "personCropFallbackCount": 1,
        "personCropFullFrameInvocationCount": 1,
    }
    second = {
        **first,
        "adaptiveQuietObservationCount": 30,
        "adaptiveQuietAdmissionCount": 6,
        "adaptiveQuietInferenceCount": 5,
        "adaptiveActiveObservationCount": 10,
        "adaptiveActiveAdmissionCount": 4,
        "adaptiveActiveInferenceCount": 4,
        "trackerProjectionFrameCount": 13,
        "trackerProjectedPersonCount": 19,
        "trackerForceRedetectSignalCount": 3,
        "ppePersonCropAttemptCount": 8,
        "phonePersonCropAttemptCount": 5,
        "personCropFallbackCount": 3,
        "personCropFullFrameInvocationCount": 4,
    }
    summary = _summarize(
        benchmark,
        [
            _sample(
                benchmark,
                0,
                decoded=10,
                completed=5,
                camera_counters=first,
            ),
            _sample(
                benchmark,
                10,
                decoded=110,
                completed=15,
                camera_counters=second,
            ),
        ],
    )
    camera = summary["cameras"]["cam1"]

    assert camera["adaptiveInference"]["observationCountsByState"] == {
        "quiet": 20,
        "uncertain": 0,
        "active": 10,
    }
    assert camera["adaptiveInference"]["observationSharesByState"]["quiet"] == pytest.approx(2 / 3, abs=1e-6)
    assert camera["adaptiveInference"]["successfulInferenceRatesPerSecondByState"] == {
        "quiet": 0.4,
        "uncertain": 0.0,
        "active": 0.4,
    }
    assert camera["keyframeTracker"] == {
        "projectionFrames": 10,
        "projectedPeople": 15,
        "forceRedetectSignalObservations": 2,
        "projectionFrameRatePerSecond": 1.0,
        "forceRedetectSignalRatePerSecond": 0.2,
    }
    assert camera["personCropSpecialists"]["ppeCropAttempts"] == 6
    assert camera["personCropSpecialists"]["phoneCropAttempts"] == 4
    assert camera["personCropSpecialists"]["fallbacks"] == 2
    assert camera["personCropSpecialists"]["fullFrameInvocations"] == 3


def test_histogram_quantiles_are_exposed_upper_bounds_with_overflow(benchmark):
    result = benchmark._histogram_quantiles([10, 20], [0, 95, 5])

    assert result == {
        "samples": 100,
        "overflowSamples": 5,
        "p95UpperBoundMs": 20.0,
        "p95Overflow": False,
        "p99UpperBoundMs": None,
        "p99Overflow": True,
        "quantileMethod": "nearest-rank-upper-bucket-bound",
    }


def test_summary_uses_histogram_counter_deltas_and_scheduler_vlm_deltas(benchmark):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=5,
            completed_histogram=[0, 10, 0, 0],
            alert_histogram=[0, 4, 0, 0],
            scheduler_counters={
                "batches_1": 2,
                "batches_2": 3,
                "completed": 8,
                "stale_completion_drops": 1,
            },
            vlm_counters={"offered": 2, "processed": 1},
        ),
        _sample(
            benchmark,
            10,
            decoded=30,
            completed=15,
            completed_histogram=[0, 105, 0, 5],
            alert_histogram=[0, 99, 0, 1],
            scheduler_counters={
                "batches_1": 4,
                "batches_2": 7,
                "completed": 18,
                "stale_completion_drops": 4,
            },
            vlm_counters={"offered": 7, "processed": 4},
        ),
    ]

    summary = _summarize(benchmark, samples, demand_fps={"cam1": 1.0})

    frame_latency = summary["cameras"]["cam1"]["latency"]["decodedIngressToResultMs"]
    assert frame_latency["samples"] == 100
    assert frame_latency["p95UpperBoundMs"] == 20.0
    assert frame_latency["p99Overflow"] is True
    assert summary["cameras"]["cam1"]["latency"][
        "decodedIngressToObservationMs"
    ]["samples"] == 100
    assert (
        summary["alerts"]["latency"]["firstPositiveToPersistedMs"]["p95UpperBoundMs"]
        == 200.0
    )
    assert summary["sharedInferenceScheduler"]["schedulerCohortDeltas"] == {
        "1": 2,
        "2": 4,
        "4": 0,
    }
    assert summary["sharedInferenceScheduler"][
        "dispatchedItemsFromCohortCounters"
    ] == 10
    assert summary["sharedInferenceScheduler"][
        "meanSchedulerCohortSize"
    ] == pytest.approx(10 / 6)
    assert summary["sharedInferenceScheduler"]["counterDeltas"][
        "stale_completion_drops"
    ] == 3
    assert summary["vlmEnrichment"]["counterDeltas"] == {
        "offered": 5,
        "processed": 3,
    }


def test_provider_latency_is_paired_with_delivery_outcome_denominator(benchmark):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=5,
            alert_histogram=[0, 5, 0, 0],
            alert_delivery_counters={
                "eligibleCount": 10,
                "deliveredCount": 5,
                "terminalCount": 1,
                "failedAttemptCount": 2,
            },
            alert_delivery_pending=4,
        ),
        _sample(
            benchmark,
            60,
            decoded=70,
            completed=35,
            alert_histogram=[0, 7, 0, 0],
            alert_delivery_counters={
                "eligibleCount": 14,
                "deliveredCount": 7,
                "terminalCount": 2,
                "failedAttemptCount": 5,
            },
            alert_delivery_pending=5,
        ),
    ]

    summary = _summarize(benchmark, samples)
    coverage = summary["alerts"]["deliveryCoverage"]

    assert coverage["pendingAtStart"] == 4
    assert coverage["eligibleDuringWindow"] == 4
    assert coverage["denominator"] == 8
    assert coverage["deliveredDuringWindow"] == 2
    assert coverage["terminalDuringWindow"] == 1
    assert coverage["pendingAtEnd"] == 5
    assert coverage["failedAttemptsDuringWindow"] == 3
    assert coverage["deliveredCoverageRatio"] == 0.25
    assert coverage["providerSuccessLatencyValidSamples"] == 2
    assert coverage["providerSuccessLatencySampleCoverageRatio"] == 1.0
    assert coverage["flowAccounted"] is True
    assert coverage["validForProviderSuccessLatency"] is True
    assert summary["measurementValidity"][
        "validForProviderSuccessLatency"
    ] is True


def test_terminal_and_pending_collapse_cannot_make_provider_latency_look_valid(
    benchmark,
):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=5,
            alert_delivery_counters={"eligibleCount": 2},
            alert_delivery_pending=2,
        ),
        _sample(
            benchmark,
            60,
            decoded=70,
            completed=35,
            alert_delivery_counters={
                "eligibleCount": 12,
                "terminalCount": 2,
                "failedAttemptCount": 10,
            },
            alert_delivery_pending=10,
        ),
    ]

    coverage = _summarize(benchmark, samples)["alerts"]["deliveryCoverage"]

    assert coverage["denominator"] == 12
    assert coverage["deliveredDuringWindow"] == 0
    assert coverage["terminalDuringWindow"] == 2
    assert coverage["pendingAtEnd"] == 10
    assert coverage["deliveredCoverageRatio"] == 0.0
    assert coverage["validForProviderSuccessLatency"] is False


def test_invalid_provider_latency_samples_are_exposed_and_invalidate_result(benchmark):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=5,
            alert_delivery_pending=0,
        ),
        _sample(
            benchmark,
            60,
            decoded=70,
            completed=35,
            alert_invalid_count=1,
            alert_delivery_counters={
                "eligibleCount": 1,
                "deliveredCount": 1,
            },
            alert_delivery_pending=0,
        ),
    ]

    summary = _summarize(benchmark, samples)
    latency = summary["alerts"]["latency"][
        "firstPositiveToProviderSuccessMs"
    ]
    coverage = summary["alerts"]["deliveryCoverage"]

    assert latency["samples"] == 0
    assert latency["invalidSamples"] == 1
    assert coverage["providerSuccessLatencyInvalidSamples"] == 1
    assert coverage["providerSuccessLatencySampleCoverageRatio"] == 0.0
    assert coverage["validForProviderSuccessLatency"] is False


def test_actual_transport_batches_are_reported_separately_from_scheduler_cohorts(
    benchmark,
):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=5,
            scheduler_counters={"batches_4": 2},
            transport_counters={
                "primaryFrameBatch": {
                    "batch2_succeeded": 1,
                    "batch4_succeeded": 0,
                },
                "rtdetrPhoneFrameBatch": {
                    "batch1_executed": 3,
                    "batch2_executed": 2,
                },
            },
        ),
        _sample(
            benchmark,
            10,
            decoded=30,
            completed=15,
            scheduler_counters={"batches_4": 5},
            transport_counters={
                "primaryFrameBatch": {
                    "batch2_succeeded": 4,
                    "batch4_succeeded": 2,
                },
                "rtdetrPhoneFrameBatch": {
                    "batch1_executed": 5,
                    "batch2_executed": 5,
                },
            },
        ),
    ]

    summary = _summarize(benchmark, samples)

    assert summary["sharedInferenceScheduler"]["schedulerCohortDeltas"] == {
        "1": 0,
        "2": 0,
        "4": 3,
    }
    assert summary["sharedInferenceScheduler"]["meanSchedulerCohortSize"] == 4
    primary = summary["inferenceTransport"]["routes"]["primaryFrameBatch"]
    assert primary["successfulBatchExecutionDeltas"] == {"2": 3, "4": 2}
    assert primary["successfulBatchExecutions"] == 5
    assert primary["successfulFramesTransported"] == 14
    assert primary["meanSuccessfulTransportBatchSize"] == 2.8
    rtdetr = summary["inferenceTransport"]["routes"]["rtdetrPhoneFrameBatch"]
    assert rtdetr["successfulBatchExecutionDeltas"] == {"1": 2, "2": 3}
    assert rtdetr["successfulFramesTransported"] == 8
    assert summary["inferenceTransport"][
        "aggregateSuccessfulBatchExecutions"
    ] == 10
    assert summary["inferenceTransport"][
        "aggregateSuccessfulFramesTransported"
    ] == 22


def test_parse_and_summarize_tegrastats_resources_and_thermal_markers(benchmark):
    parsed = benchmark._parse_tegrastats_line(
        "RAM 5854/7622MB SWAP 1430/4096MB "
        "CPU [10%@729,20%@729,off,30%@729] GR3D_FREQ 47% "
        "VIC_FREQ 12% CPU@86.3C GPU@65.9C "
        "VDD_IN 8432mW/8000mW VDD_CPU_GPU_CV 3200mW/3000mW THROTTLED"
    )

    assert parsed["ramUsedMiB"] == 5854
    assert parsed["swapUsedMiB"] == 1430
    assert parsed["cpuPercent"] == 15.0
    assert parsed["cpuOnlinePercent"] == 20.0
    assert parsed["cpuOfflineCoreCount"] == 1
    assert parsed["gpuPercent"] == 47.0
    assert parsed["vicPercent"] == 12.0
    assert parsed["inputPowerW"] == 8.432
    assert parsed["temperaturesC"] == {"cpu": 86.3, "gpu": 65.9}
    assert parsed["thermalMarker"] is True

    resources = benchmark._summarize_tegrastats(
        [parsed],
        source="supplied-tegrastats-log",
        collection_errors={},
        thermal_warning_c=85.0,
    )
    assert resources["ram"]["usedMiB"]["maximum"] == 5854
    assert resources["inputPowerW"]["maximum"] == 8.432
    assert resources["thermalIndicators"] == {
        "warningThresholdC": 85.0,
        "samplesAtOrAboveThreshold": 1,
        "explicitThrottleOrThermalMarkerSamples": 1,
        "concernObserved": True,
    }


def test_output_is_atomic_private_and_excludes_secrets_frames_and_urls(
    benchmark,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    token = "super-secret-bearer-value"
    monkeypatch.setenv("PIPELINE_BENCHMARK_TOKEN", token)
    headers = benchmark._auth_headers_from_env(
        bearer_token_env="PIPELINE_BENCHMARK_TOKEN"
    )
    assert headers == {"Authorization": f"Bearer {token}"}

    first_raw = _raw_health(benchmark, decoded=10, completed=2)
    second_raw = _raw_health(
        benchmark,
        decoded=20,
        completed=6,
        server_timestamp=1009.0,
    )
    for payload in (first_raw, second_raw):
        payload["secret"] = token
        payload["frame"] = "base64-private-frame-material"
        payload["privateUrl"] = "https://user:password@private.example/health"
        payload["sharedInferenceScheduler"]["counters"]["token"] = 123456
        payload["vlmEnrichment"]["providerToken"] = token

    samples = [
        {
            "monotonic": 0.0,
            "httpRoundTripMs": 2.0,
            "health": benchmark._sanitize_health_payload(
                first_raw, received_wall_seconds=1000.0
            ),
        },
        {
            "monotonic": 10.0,
            "httpRoundTripMs": 3.0,
            "health": benchmark._sanitize_health_payload(
                second_raw, received_wall_seconds=1010.0
            ),
        },
    ]
    summary = _summarize(benchmark, samples)
    report = {"schemaVersion": 1, "label": "safe-run", "summary": summary}
    destination = tmp_path / "report.json"

    benchmark._atomic_write_json(destination, report)

    rendered = destination.read_text(encoding="utf-8")
    assert token not in rendered
    assert "base64-private-frame-material" not in rendered
    assert "password" not in rendered
    assert "private.example" not in rendered
    assert '"token"' not in rendered
    assert "must never be copied" not in rendered
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(rendered)["summary"]["cameras"]["cam1"]["inferenceCompleted"] == 4
    assert "PIPELINE_BENCHMARK_TOKEN" not in rendered
    assert os.environ["PIPELINE_BENCHMARK_TOKEN"] == token


def test_capacity_and_latency_provenance_are_explicit(benchmark):
    samples = [
        _sample(benchmark, 0, decoded=1, completed=0),
        _sample(benchmark, 10, decoded=11, completed=5),
    ]

    summary = _summarize(
        benchmark,
        samples,
        camera_basis="equivalent",
        equivalent_camera_count=24,
    )

    assert summary["capacityProvenance"]["basis"] == "equivalent"
    assert summary["capacityProvenance"]["declaredCameraCount"] == 24
    assert (
        summary["capacityProvenance"][
            "cameraEquivalentResultsAreNotPhysicalCameraCertification"
        ]
        is True
    )
    assert (
        "not sensor capture age"
        in summary["latencyProvenance"]["decodedIngressToResultMs"]
    )
    assert (
        "not sensor capture age"
        in summary["latencyProvenance"]["decodedIngressToObservationMs"]
    )
    assert summary["health"]["endpointSnapshotAgeSeconds"]["maximum"] == 1.0


def test_explicit_demands_define_completeness_and_exclude_extra_offline_camera(
    benchmark,
):
    samples = [
        _multi_camera_sample(
            benchmark,
            0,
            [
                {"camera_id": "cam1", "decoded": 10, "completed": 2},
                {"camera_id": "offline-extra", "decoded": 0, "completed": 0},
            ],
        ),
        _multi_camera_sample(
            benchmark,
            10,
            [
                {"camera_id": "cam1", "decoded": 20, "completed": 7},
                {"camera_id": "offline-extra", "decoded": 0, "completed": 0},
            ],
        ),
    ]

    summary = _summarize(benchmark, samples, demand_fps={"cam1": 1.0})

    assert summary["measurementValidity"]["validForPipelineDeltas"] is True
    assert summary["expectedCameraCompleteness"] == {
        "source": "explicit-camera-demand-fps",
        "expectedCameraIds": ["cam1"],
        "expectedCameraCount": 1,
        "completeExpectedCameraIds": ["cam1"],
        "incompleteExpectedCameraIds": [],
        "allExpectedCamerasComplete": True,
        "activeTelemetryCameraIds": ["cam1"],
        "excludedObservedCameraIds": ["offline-extra"],
        "requirements": {"hardwareDecode": False, "maxFrameDimension": None},
    }
    assert summary["fairness"]["cameraIds"] == ["cam1"]
    assert summary["fairness"]["cameraCount"] == 1
    assert summary["capacityProvenance"]["observedActiveCameraCount"] == 1


def test_missing_or_zero_decoded_expected_camera_invalidates_and_enters_fairness(
    benchmark,
):
    samples = [
        _sample(benchmark, 0, decoded=10, completed=2),
        _sample(benchmark, 10, decoded=20, completed=7),
    ]

    summary = _summarize(
        benchmark,
        samples,
        demand_fps={"cam1": 1.0, "cam-missing": 1.0},
    )

    assert summary["measurementValidity"]["validForPipelineDeltas"] is False
    assert summary["expectedCameraCompleteness"]["incompleteExpectedCameraIds"] == [
        "cam-missing"
    ]
    missing = summary["cameras"]["cam-missing"]["measurementCompleteness"]
    assert missing["expected"] is True
    assert missing["telemetryObserved"] is False
    assert missing["decodedCounterValidSeconds"] == 0.0
    assert missing["validDecodedCounterInterval"] is False
    assert missing["decodedFramesObserved"] is False
    assert missing["diagnosticSamplesPresent"] == 0
    assert missing["completeDiagnosticSampleCoverage"] is False
    assert missing["complete"] is False
    assert summary["fairness"]["cameraIds"] == ["cam-missing", "cam1"]
    assert summary["fairness"]["jainIndex"] == 0.5


@pytest.mark.parametrize(
    ("capture_backend", "hardware_active", "frame_width", "expected_valid"),
    [
        ("gstreamer_nvdec", True, 480, True),
        ("gstreamer_software", False, 480, False),
        ("gstreamer_nvdec", True, 960, False),
    ],
)
def test_hardware_decode_and_low_resolution_requirements_are_validity_gates(
    benchmark,
    capture_backend,
    hardware_active,
    frame_width,
    expected_valid,
):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=2,
            capture_backend=capture_backend,
            hardware_active=hardware_active,
            hardware_fallback=not hardware_active,
            frame_width=frame_width,
            frame_height=270,
        ),
        _sample(
            benchmark,
            10,
            decoded=20,
            completed=7,
            capture_backend=capture_backend,
            hardware_active=hardware_active,
            hardware_fallback=not hardware_active,
            frame_width=frame_width,
            frame_height=270,
        ),
    ]

    summary = _summarize(
        benchmark,
        samples,
        demand_fps={"cam1": 1.0},
        require_hardware_decode=True,
        require_max_frame_dimension=480,
    )

    assert (
        summary["measurementValidity"]["validForPipelineDeltas"]
        is expected_valid
    )
    completeness = summary["cameras"]["cam1"]["measurementCompleteness"]
    assert completeness["hardwareDecodeSatisfied"] is (
        capture_backend == "gstreamer_nvdec" and hardware_active
    )
    assert completeness["maxFrameDimensionSatisfied"] is (frame_width <= 480)


def test_hardware_and_dimension_gates_cover_every_diagnostic_sample(benchmark):
    samples = [
        _sample(
            benchmark,
            0,
            decoded=10,
            completed=2,
            capture_backend="gstreamer_software",
            hardware_active=False,
            hardware_fallback=True,
            frame_width=960,
            frame_height=540,
        ),
        _sample(
            benchmark,
            10,
            decoded=20,
            completed=7,
            capture_backend="gstreamer_nvdec",
            hardware_active=True,
            hardware_fallback=False,
            frame_width=480,
            frame_height=270,
        ),
    ]

    summary = _summarize(
        benchmark,
        samples,
        demand_fps={"cam1": 1.0},
        require_hardware_decode=True,
        require_max_frame_dimension=480,
    )

    assert summary["measurementValidity"]["validForPipelineDeltas"] is False
    completeness = summary["cameras"]["cam1"]["measurementCompleteness"]
    assert completeness["hardwareDecodeViolationSamples"] == 1
    assert completeness["maxFrameDimensionViolationSamples"] == 1
    assert completeness["hardwareDecodeSatisfied"] is False
    assert completeness["maxFrameDimensionSatisfied"] is False


def test_health_duration_and_poll_coverage_are_required(benchmark):
    early_only = [
        _sample(benchmark, 0, decoded=10, completed=2),
        _sample(benchmark, 1, decoded=20, completed=7),
    ]
    duration_gap = _summarize(
        benchmark,
        early_only,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )
    assert duration_gap["measurementValidity"]["sufficientDurationCoverage"] is False
    assert duration_gap["measurementValidity"]["validForPipelineDeltas"] is False

    endpoints_only = [
        _sample(benchmark, 0, decoded=10, completed=2),
        _sample(benchmark, 60, decoded=70, completed=32),
    ]
    sample_gap = _summarize(
        benchmark,
        endpoints_only,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )
    assert sample_gap["measurementValidity"]["sufficientDurationCoverage"] is True
    assert (
        sample_gap["measurementValidity"]["sufficientHealthSampleCoverage"]
        is False
    )
    assert sample_gap["measurementValidity"]["validForPipelineDeltas"] is False


def test_frozen_cached_health_snapshot_cannot_pass_a_full_client_poll_window(
    benchmark,
):
    samples = [
        _sample(
            benchmark,
            monotonic,
            server_timestamp=999.0,
            decoded=10,
            completed=2,
        )
        for monotonic in range(61)
    ]

    summary = _summarize(
        benchmark,
        samples,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )
    validity = summary["measurementValidity"]

    assert validity["sufficientDurationCoverage"] is True
    assert validity["sufficientHealthSampleCoverage"] is True
    assert validity["distinctServerSnapshotCount"] == 1
    assert validity["atLeastTwoDistinctServerSnapshots"] is False
    assert validity["serverSnapshotTimestampsMonotonic"] is True
    assert validity["serverSnapshotsFresh"] is False
    assert validity["serverSnapshotDurationCoverageRatio"] == 0.0
    assert validity["sufficientServerSnapshotDurationCoverage"] is False
    assert validity["validForPipelineDeltas"] is False
    assert summary["health"]["advancingServerSnapshotTransitions"] == 0
    assert (
        summary["health"]["skippedNonadvancingServerSnapshotTransitions"]
        == 60
    )
    assert summary["cameras"]["cam1"]["decodedFrames"] == 0


def test_five_second_cached_health_snapshots_use_server_time_for_coverage(
    benchmark,
):
    samples = []
    for monotonic in range(61):
        cache_generation = monotonic // 5
        samples.append(
            _sample(
                benchmark,
                monotonic,
                server_timestamp=999.0 + cache_generation * 5,
                decoded=10 + cache_generation * 5,
                completed=2 + cache_generation,
            )
        )

    summary = _summarize(
        benchmark,
        samples,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )
    validity = summary["measurementValidity"]
    camera = summary["cameras"]["cam1"]

    assert validity["allHealthSamplesHaveServerSnapshotTimestamp"] is True
    assert validity["serverSnapshotTimestampsMonotonic"] is True
    assert validity["distinctServerSnapshotCount"] == 13
    assert validity["atLeastTwoDistinctServerSnapshots"] is True
    assert validity["maximumServerSnapshotAgeSeconds"] == 5.0
    assert validity["serverSnapshotsFresh"] is True
    assert validity["serverSnapshotSpanSeconds"] == 60.0
    assert validity["serverSnapshotDurationCoverageRatio"] == 1.0
    assert validity["sufficientServerSnapshotDurationCoverage"] is True
    assert validity["validForPipelineDeltas"] is True
    assert summary["health"]["advancingServerSnapshotTransitions"] == 12
    assert (
        summary["health"]["skippedNonadvancingServerSnapshotTransitions"]
        == 48
    )
    assert camera["decodedFrames"] == 60
    assert camera["decodedFps"] == 1.0
    assert camera["measurementCompleteness"]["decodedCounterValidSeconds"] == 60.0


def test_stale_and_nonmonotonic_server_snapshots_are_invalid(benchmark):
    stale_samples = [
        _sample(
            benchmark,
            monotonic,
            server_timestamp=979.0 + monotonic,
            decoded=10 + monotonic,
            completed=2 + monotonic,
        )
        for monotonic in range(61)
    ]
    stale_summary = _summarize(
        benchmark,
        stale_samples,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )

    assert stale_summary["measurementValidity"]["serverSnapshotsFresh"] is False
    assert (
        stale_summary["measurementValidity"][
            "maximumServerSnapshotAgeSeconds"
        ]
        == 21.0
    )
    assert stale_summary["measurementValidity"]["validForPipelineDeltas"] is False

    nonmonotonic_samples = [
        _sample(
            benchmark,
            monotonic,
            server_timestamp=(
                1027.0 if monotonic == 30 else 999.0 + monotonic
            ),
            decoded=10 + monotonic,
            completed=2 + monotonic,
        )
        for monotonic in range(61)
    ]
    nonmonotonic_summary = _summarize(
        benchmark,
        nonmonotonic_samples,
        configured_duration_seconds=60.0,
        configured_poll_interval_seconds=1.0,
        demand_fps={"cam1": 1.0},
    )

    assert (
        nonmonotonic_summary["measurementValidity"][
            "serverSnapshotTimestampsMonotonic"
        ]
        is False
    )
    assert (
        nonmonotonic_summary["measurementValidity"]["serverSnapshotsFresh"]
        is True
    )
    assert (
        nonmonotonic_summary["measurementValidity"]["validForPipelineDeltas"]
        is False
    )
    assert nonmonotonic_summary["cameras"]["cam1"]["decodedFrames"] == 0


def test_sanitizer_retains_safe_legacy_camera_id_backend_and_dimensions(benchmark):
    raw = _raw_health(
        benchmark,
        camera_id="Gate 1",
        decoded=10,
        capture_backend="gstreamer_software",
        hardware_active=False,
        hardware_fallback=True,
        frame_width=640,
        frame_height=360,
    )

    sanitized = benchmark._sanitize_health_payload(
        raw,
        received_wall_seconds=1000.0,
    )

    assert sanitized["snapshotTimestampSeconds"] == 999.0
    assert set(sanitized["pipelineTelemetry"]["cameras"]) == {"Gate 1"}
    diagnostic = sanitized["diagnosticsCameras"]["Gate 1"]
    assert diagnostic["frameWidth"] == 640
    assert diagnostic["frameHeight"] == 360
    assert diagnostic["connection"]["captureBackend"] == "gstreamer_software"
    assert diagnostic["connection"]["hardwareAccelerationActive"] is False
    assert diagnostic["connection"]["hardwareFallback"] is True
    assert diagnostic["connection"]["appsinkLatestBufferDropMethod"] == "unavailable"
    assert benchmark._parse_demand_fps(["Gate 1=2.5"]) == {"Gate 1": 2.5}


def test_tegrastats_supplied_log_starts_at_eof_and_ignores_history(
    benchmark,
    tmp_path: Path,
):
    log = tmp_path / "tegrastats.log"
    log.write_text("RAM 7000/8000MB CPU@99C VDD_IN 9999mW/9999mW\n")
    collector = benchmark._TegrastatsCollector(log_path=log)
    started = time.monotonic()

    collector.start()
    with log.open("a", encoding="utf-8") as handle:
        handle.write("RAM 1000/8000MB CPU@45C VDD_IN 5000mW/5000mW\n")
        handle.flush()
    time.sleep(0.2)
    collector.stop()
    ended = time.monotonic()

    samples, errors = collector.result(started=started, ended=ended)
    assert errors == {}
    assert len(samples) == 1
    assert samples[0]["ramUsedMiB"] == 1000
    assert samples[0]["temperaturesC"] == {"cpu": 45.0}


def test_tegrastats_supplied_log_follows_rotated_inode(
    benchmark,
    tmp_path: Path,
):
    log = tmp_path / "tegrastats.log"
    rotated = tmp_path / "tegrastats.log.1"
    log.write_text("RAM 7000/8000MB CPU@99C\n")
    collector = benchmark._TegrastatsCollector(log_path=log)
    started = time.monotonic()

    collector.start()
    log.rename(rotated)
    log.write_text("RAM 1200/8000MB CPU@46C\n")
    time.sleep(0.2)
    collector.stop()
    ended = time.monotonic()

    samples, errors = collector.result(started=started, ended=ended)
    assert errors == {}
    assert [sample["ramUsedMiB"] for sample in samples] == [1200]


def test_resource_summary_rejects_sparse_samples_for_a_long_run(benchmark):
    sparse = benchmark._summarize_tegrastats(
        [{"monotonic": 0.0, "ramUsedMiB": 1000}],
        source="supplied-tegrastats-log",
        collection_errors={},
        thermal_warning_c=85.0,
        configured_duration_seconds=60.0,
        configured_interval_seconds=1.0,
    )

    assert sparse["measurementValidity"]["valid"] is False
    assert sparse["measurementValidity"]["sufficientSampleCoverage"] is False
    assert sparse["measurementValidity"]["sufficientDurationCoverage"] is False

    dense_samples = [
        {"monotonic": index * 1.25, "ramUsedMiB": 1000 + index}
        for index in range(49)
    ]
    dense = benchmark._summarize_tegrastats(
        dense_samples,
        source="supplied-tegrastats-log",
        collection_errors={},
        thermal_warning_c=85.0,
        configured_duration_seconds=60.0,
        configured_interval_seconds=1.0,
    )
    assert dense["measurementValidity"]["valid"] is True
