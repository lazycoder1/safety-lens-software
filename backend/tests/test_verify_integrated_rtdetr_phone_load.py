import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = ROOT / "scripts" / "verify_integrated_rtdetr_phone_load.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_integrated_rtdetr_phone_load",
        VERIFIER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report():
    return {
        "cameras": 20,
        "target_fps": 4.0,
        "duration_seconds": 60.0,
        "requests": 4740,
        "effective_requests": 4800,
        "substitution_source": "rtdetr",
        "substitution_attempts": 60,
        "substituted_requests": 60,
        "specialist_requests": 520,
        "overloads": 0,
        "failures": 0,
        "minimum_effective_camera_fps": 4.0,
        "latency_ms": {"maximum": 220.0},
        "rtdetr_latency_ms": {"maximum": 180.0},
        "edge_rtdetr_phone_batch": {
            "submitted_frames": 60,
            "batch1_frames": 0,
            "batch2_frames": 60,
            "batch2_executed": 30,
            "route_failures": 0,
            "admission_overloads": 0,
        },
        "per_camera": [{} for _ in range(20)],
    }


def test_verifier_accepts_integrated_fresh_batch2_substitution():
    result = _load_verifier().verify(
        _report(),
        maximum_primary_latency_ms=250.0,
        maximum_rtdetr_latency_ms=250.0,
    )

    assert result["ok"] is True
    assert result["errors"] == []


def test_verifier_rejects_batch1_or_missing_substitution():
    report = _report()
    report["substituted_requests"] = 59
    report["edge_rtdetr_phone_batch"]["batch1_frames"] = 1
    report["edge_rtdetr_phone_batch"]["batch2_frames"] = 59

    result = _load_verifier().verify(
        report,
        maximum_primary_latency_ms=250.0,
        maximum_rtdetr_latency_ms=250.0,
    )

    assert result["ok"] is False
    assert any("attempt" in error for error in result["errors"])
    assert any("batch-1" in error for error in result["errors"])


def test_verifier_rejects_primary_or_rtdetr_freshness_miss():
    report = _report()
    report["latency_ms"]["maximum"] = 260.0
    report["rtdetr_latency_ms"]["maximum"] = 270.0

    result = _load_verifier().verify(
        report,
        maximum_primary_latency_ms=250.0,
        maximum_rtdetr_latency_ms=250.0,
    )

    assert result["ok"] is False
    assert any("primary maximum" in error for error in result["errors"])
    assert any("RT-DETR maximum" in error for error in result["errors"])
