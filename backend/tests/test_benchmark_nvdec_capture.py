"""Tests for the concurrent NVDEC capture benchmark helpers."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_nvdec_capture.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_nvdec_capture", BENCHMARK_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_readings_uses_each_capture_duration():
    benchmark = _load_benchmark()

    summary = benchmark._summarize_readings(
        [
            (120, 960, 540, 15.0),
            (123, 960, 540, 15.5),
        ]
    )

    assert summary["frame_counts"] == [120, 123]
    assert summary["read_seconds"] == [15.0, 15.5]
    assert summary["delivered_rates"] == pytest.approx([8.0, 123 / 15.5])
    assert summary["dimensions"] == [(960, 540)]


def test_summarize_readings_keeps_distinct_dimensions():
    benchmark = _load_benchmark()

    summary = benchmark._summarize_readings(
        [
            (80, 352, 288, 10.0),
            (80, 960, 540, 10.0),
        ]
    )

    assert summary["dimensions"] == [(352, 288), (960, 540)]


def test_channel_replacement_preserves_other_query_fields_and_credentials():
    benchmark = _load_benchmark()

    result = benchmark._replace_numeric_query_parameter(
        "rtsp://user:password@camera.test/cam/realmonitor?channel=1&subtype=0",
        "channel",
        4,
    )

    assert result == (
        "rtsp://user:password@camera.test/cam/realmonitor?channel=4&subtype=0"
    )


def test_channel_replacement_rejects_incompatible_rtsp_shape():
    benchmark = _load_benchmark()

    with pytest.raises(ValueError, match="channel"):
        benchmark._replace_numeric_query_parameter(
            "rtsp://camera.test/Streaming/Channels/101",
            "channel",
            2,
        )
