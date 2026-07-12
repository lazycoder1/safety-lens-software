import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = ROOT / "scripts" / "verify_rtdetrv4_substitution_load.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_rtdetrv4_substitution_load",
        VERIFIER_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payloads():
    per_camera = [
        {
            "successes": 240,
            "substituted_requests": 0,
            "specialist_requests": 26,
            "effective_fps": 4.0,
        }
        for _ in range(18)
    ]
    for item in per_camera[-2:]:
        item["successes"] = 210
        item["substituted_requests"] = 30
    edge = {
        "cameras": 18,
        "target_fps": 4.0,
        "duration_seconds": 60.0,
        "requests": 4260,
        "effective_requests": 4320,
        "substituted_requests": 60,
        "overloads": 0,
        "failures": 0,
        "minimum_effective_camera_fps": 4.0,
        "specialist_requests": 468,
        "latency_ms": {"median": 40.0, "p95": 130.0, "maximum": 220.0},
        "per_camera": per_camera,
    }
    specialist = {
        "target_fps": 1.0,
        "achieved_fps": 1.0,
        "duration_seconds": 60.0,
        "frames_completed": 60,
        "stale_groups": 0,
    }
    return edge, specialist


def test_verifier_accepts_exact_fresh_substitution():
    verifier = _load_verifier()
    edge, specialist = _payloads()

    result = verifier.verify(edge, specialist, maximum_primary_latency_ms=250.0)

    assert result["ok"] is True
    assert result["errors"] == []


def test_verifier_accepts_model_server_route_report_schema():
    verifier = _load_verifier()
    edge, specialist = _payloads()
    specialist = {
        "target_fps": 1.0,
        "frame_fps": 1.0,
        "duration_target_seconds": 60.0,
        "repeats": 30,
        "batch_size": 2,
        "stale_groups": 0,
    }

    result = verifier.verify(edge, specialist, maximum_primary_latency_ms=250.0)

    assert result["ok"] is True
    assert result["rtdetr_achieved_fps"] == 1.0


def test_verifier_rejects_missing_rtdetr_frames():
    verifier = _load_verifier()
    edge, specialist = _payloads()
    specialist["frames_completed"] = 58

    result = verifier.verify(edge, specialist, maximum_primary_latency_ms=250.0)

    assert result["ok"] is False
    assert "RT-DETR frames 58 != substituted frames 60" in result["errors"]


def test_verifier_rejects_primary_freshness_miss():
    verifier = _load_verifier()
    edge, specialist = _payloads()
    edge["latency_ms"]["maximum"] = 265.0

    result = verifier.verify(edge, specialist, maximum_primary_latency_ms=250.0)

    assert result["ok"] is False
    assert any("primary maximum latency" in error for error in result["errors"])
