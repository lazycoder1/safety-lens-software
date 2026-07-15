import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark_pipeline_soak_jetson.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_pipeline_soak", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _health(successes: int, *, age: float = 0.2, backend: str = "gstreamer_nvdec"):
    return {
        "status": "ok",
        "alertsCount": 10,
        "cameras": [
            {
                "id": "cam1",
                "frameFresh": True,
                "lastFrameAgeSeconds": age,
                "connection": {
                    "captureBackend": backend,
                    "totalFailureCount": 2,
                },
                "inference": {
                    "successCount": successes,
                    "failureCount": 1,
                    "overloadDropCount": 0,
                },
            }
        ],
    }


def test_parse_tegrastats_jetpack5_metrics():
    benchmark = _load_benchmark()

    parsed = benchmark._parse_tegrastats_line(
        "RAM 5854/7622MB SWAP 1430/4096MB GR3D_FREQ 47% "
        "VIC_FREQ 12% CPU@67.3C GPU@65.9C VDD_IN 8432mW/8432mW"
    )

    assert parsed == {
        "ram_used_mb": 5854,
        "ram_total_mb": 7622,
        "swap_used_mb": 1430,
        "swap_total_mb": 4096,
        "gpu_percent": 47,
        "vic_percent": 12,
        "input_power_w": 8.432,
        "temperatures_c": {"cpu": 67.3, "gpu": 65.9},
    }


def test_safe_health_excludes_names_urls_and_unapproved_fields():
    benchmark = _load_benchmark()
    payload = {
        "status": "ok",
        "alerts_count": 1,
        "secret": "must-not-survive",
        "inferenceTransport": {
            "primary": {
                "eligible": 7,
                "enabled": True,
                "token": "nested-must-not-survive",
            }
        },
        "cameras": [
            {
                "id": "cam1",
                "name": "Private site",
                "video": "rtsp://user:password@example.test/live",
                "frameFresh": True,
                "workerRunning": True,
                "lastFrameAgeSeconds": 0.1,
                "runtimeStatus": "running",
                "connection": {"captureBackend": "gstreamer_nvdec"},
                "inference": {"successCount": 3},
            }
        ],
    }

    safe = benchmark._safe_health(payload)
    rendered = str(safe)

    assert "Private site" not in rendered
    assert "password" not in rendered
    assert "must-not-survive" not in rendered
    assert "nested-must-not-survive" not in rendered
    assert safe["inferenceTransport"] == {
        "primary": {"eligible": 7, "enabled": True}
    }
    assert safe["cameras"][0]["id"] == "cam1"


def test_summary_reports_deltas_frame_age_fairness_and_resources():
    benchmark = _load_benchmark()
    samples = [
        {
            "monotonic": 100.0,
            "health": _health(100, age=0.2),
            "containers": {
                "rakshak-edge": {"cpu_percent": 20.0, "memory_used_mib": 500.0}
            },
        },
        {
            "monotonic": 110.0,
            "health": _health(140, age=0.8),
            "containers": {
                "rakshak-edge": {"cpu_percent": 30.0, "memory_used_mib": 520.0}
            },
        },
    ]
    tegra = [
        {
            "ram_used_mb": 6000,
            "swap_used_mb": 1200,
            "gpu_percent": 10,
            "vic_percent": 20,
            "input_power_w": 8.0,
            "temperatures_c": {"cpu": 66.0},
        },
        {
            "ram_used_mb": 6100,
            "swap_used_mb": 1201,
            "gpu_percent": 50,
            "vic_percent": 30,
            "input_power_w": 9.0,
            "temperatures_c": {"cpu": 68.0},
        },
    ]

    summary = benchmark._summarize(
        samples,
        tegra,
        duration_seconds=10.0,
        expected_cameras=["cam1"],
    )

    assert summary["cameras"]["cam1"]["inference_success_delta"] == 40
    assert summary["cameras"]["cam1"]["inference_fps"] == 4.0
    assert summary["observed_health_interval_seconds"] == 10.0
    assert summary["cameras"]["cam1"]["frame_age_seconds"]["p99"] > 0.79
    assert summary["inference_fps_jain"] == 1.0
    assert summary["resources"]["input_power_w"]["maximum"] == 9.0
    assert summary["containers"]["rakshak-edge"]["memory_used_mib"]["maximum"] == 520.0


def test_stale_gate_rejects_absent_expected_camera():
    benchmark = _load_benchmark()
    reports = {
        "cam1": {
            "present": True,
            "missing_health_samples": 0,
            "stale_health_samples": 0,
        },
        "cam2": {
            "present": False,
            "missing_health_samples": 2,
            "stale_health_samples": 0,
        },
    }

    assert benchmark._stale_gate_failed(reports, ["cam1", "cam2"])
    assert not benchmark._stale_gate_failed({"cam1": reports["cam1"]}, ["cam1"])


def test_counter_reset_invalidates_soak():
    benchmark = _load_benchmark()

    with pytest.raises(benchmark.CounterResetError):
        benchmark._counter_delta({"count": 10}, {"count": 2}, "count")


def test_mid_window_counter_reset_invalidates_soak_even_if_endpoint_recovers():
    benchmark = _load_benchmark()
    samples = [
        {"monotonic": 100.0, "health": _health(100)},
        {"monotonic": 105.0, "health": _health(2)},
        {"monotonic": 110.0, "health": _health(140)},
    ]

    with pytest.raises(benchmark.CounterResetError, match=r"successCount \(100 -> 2\)"):
        benchmark._summarize(
            samples,
            [],
            duration_seconds=10.0,
            expected_cameras=["cam1"],
        )
