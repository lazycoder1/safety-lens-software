"""Tests for Jetson raw YOLO benchmark metadata helpers."""

from argparse import Namespace
from pathlib import Path
import hashlib
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_yolo_jetson.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_yolo_jetson", BENCHMARK_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_report_payload(export_sha256: str = "d" * 64) -> dict:
    return {
        "ok": True,
        "promotion_manifest": {
            "candidate_status": "ready_for_side_by_side_runtime_test",
            "runtime_handoff": {
                "selected_export": {
                    "path": "/path/to/cleared/apron-harness-ppe.onnx",
                    "suffix": ".onnx",
                    "sha256": export_sha256,
                },
            },
        },
    }


def test_load_candidate_identity_derives_report_and_export_sha(tmp_path: Path):
    benchmark = _load_benchmark()
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report_payload())
    candidate_sha = _sha256_file(candidate_path)

    identity = benchmark.load_candidate_identity(candidate_path)

    assert identity["model_artifact_sha256"] == "d" * 64
    assert identity["candidate_report_sha256"] == candidate_sha
    assert identity["candidate_report"] == {
        "present": True,
        "path": str(candidate_path),
        "sha256": candidate_sha,
        "ok": True,
        "candidate_status": "ready_for_side_by_side_runtime_test",
        "selected_export_sha256": "d" * 64,
        "selected_export_path": "/path/to/cleared/apron-harness-ppe.onnx",
    }


def test_load_candidate_identity_rejects_explicit_mismatch(tmp_path: Path):
    benchmark = _load_benchmark()
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report_payload(export_sha256="e" * 64))

    try:
        benchmark.load_candidate_identity(candidate_path, model_artifact_sha256="d" * 64)
    except RuntimeError as exc:
        assert "--model-artifact-sha256 does not match candidate report" in str(exc)
    else:
        raise AssertionError("expected artifact mismatch to raise")


def test_build_identity_fields_requires_single_candidate_model(tmp_path: Path):
    benchmark = _load_benchmark()
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate_report_payload())

    args = Namespace(
        candidate_report=str(candidate_path),
        model_artifact_sha256="",
        candidate_report_sha256="",
        models=["apron-harness-ppe.onnx", "other.onnx"],
    )

    try:
        benchmark.build_identity_fields(args)
    except RuntimeError as exc:
        assert "exactly one model" in str(exc)
    else:
        raise AssertionError("expected multi-model candidate identity stamping to raise")


def test_build_identity_fields_allows_explicit_hashes_without_candidate_report():
    benchmark = _load_benchmark()
    args = Namespace(
        candidate_report="",
        model_artifact_sha256="d" * 64,
        candidate_report_sha256="c" * 64,
        models=["apron-harness-ppe.onnx"],
    )

    identity = benchmark.build_identity_fields(args)

    assert identity["candidate_report"] == {"present": False}
    assert identity["model_artifact_sha256"] == "d" * 64
    assert identity["candidate_report_sha256"] == "c" * 64
