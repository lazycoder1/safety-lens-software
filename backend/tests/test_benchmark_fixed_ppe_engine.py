"""Tests for the fixed-prompt PPE engine comparison helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_fixed_ppe_engine.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_fixed_ppe_engine", BENCHMARK_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Coordinates:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, _index):
        return self

    def tolist(self):
        return self.values


def test_image_paths_are_sorted_and_filter_non_images(tmp_path: Path):
    benchmark = _load_benchmark()
    for name in ("b.PNG", "a.jpg", "notes.json", "directory.jpeg"):
        (tmp_path / name).write_bytes(b"test")

    assert [path.name for path in benchmark.image_paths(tmp_path)] == [
        "a.jpg",
        "b.PNG",
        "directory.jpeg",
    ]


def test_detections_from_result_preserves_prompt_and_rounds_values():
    benchmark = _load_benchmark()
    result = SimpleNamespace(
        names={0: "helmet"},
        boxes=[
            SimpleNamespace(
                cls=_Value(0),
                conf=_Value(0.85246),
                xyxy=_Coordinates([1.234, 5.678, 10.049, 20.051]),
            )
        ],
    )

    assert benchmark.detections_from_result(result) == [
        {
            "class": "helmet",
            "confidence": 0.8525,
            "bbox": [1.2, 5.7, 10.0, 20.1],
        }
    ]


def test_percentile_is_bounded_for_small_samples():
    benchmark = _load_benchmark()

    assert benchmark.percentile([30.0, 10.0, 20.0], 0.95) == 30.0
    assert benchmark.percentile([30.0, 10.0, 20.0], 1.0) == 30.0
