import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "scripts" / "benchmark_conditional_model_server_load.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_conditional_model_server_load",
        BENCHMARK_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_grouped_phase_places_eighteen_camera_remainder_at_200ms():
    benchmark = _load_benchmark()

    assert benchmark._phase_offset(16, 18, 0.25, "grouped", 4) == 0.2
    assert benchmark._phase_group_cardinality(16, 18, "grouped", 4) == 2
    assert benchmark._phase_group_cardinality(17, 18, "grouped", 4) == 2


def test_grouped_phase_can_reserve_less_time_for_a_cheaper_remainder():
    benchmark = _load_benchmark()

    offsets = [
        benchmark._phase_offset(camera, 22, 0.25, "grouped", 4, 0.8)
        for camera in range(22)
    ]

    assert offsets[:4] == [0.0] * 4
    assert offsets[4:8] == [0.25 / 5.8] * 4
    assert offsets[20:] == [1.25 / 5.8] * 2
    assert 0.25 - offsets[-1] == pytest.approx(0.25 * 0.8 / 5.8)


def test_full_group_layout_ignores_remainder_weight():
    benchmark = _load_benchmark()

    assert benchmark._phase_offset(16, 20, 0.25, "grouped", 4, 0.5) == 0.2


def test_one_eighth_duty_replaces_exactly_one_batch2_frame_per_second():
    benchmark = _load_benchmark()

    sequences = [
        sequence
        for sequence in range(240)
        if benchmark._specialist_due(sequence, 0.125)
    ]

    assert len(sequences) == 30
    assert sequences[0] == 7
    assert sequences[-1] == 239
    assert all(right - left == 8 for left, right in zip(sequences, sequences[1:]))


def test_substitution_and_ppe_collisions_are_bounded_and_deferable():
    benchmark = _load_benchmark()
    substitutions = {
        sequence
        for sequence in range(240)
        if benchmark._specialist_due(sequence, 0.125)
    }
    ppe = {
        sequence
        for sequence in range(240)
        if benchmark._specialist_due(sequence, 0.111)
    }

    assert substitutions & ppe == {63, 135, 207}


def test_sequence_offset_places_rt_work_beside_ppe_without_collision():
    benchmark = _load_benchmark()
    ppe = {
        sequence
        for sequence in range(240)
        if benchmark._specialist_due(sequence, 0.125)
    }
    shifted_rt = {
        sequence
        for sequence in range(240)
        if benchmark._specialist_due(sequence + 1, 0.125)
    }

    assert len(ppe) == 30
    assert len(shifted_rt) == 30
    assert ppe.isdisjoint(shifted_rt)
    assert all(rt + 1 in ppe for rt in shifted_rt)


def test_distribution_includes_p99_and_handles_empty_samples():
    benchmark = _load_benchmark()

    assert benchmark._distribution([]) == {
        "median": None,
        "p95": None,
        "p99": None,
        "maximum": None,
    }
    summary = benchmark._distribution([1.0, 2.0, 3.0, 100.0])

    assert summary["median"] == 2.5
    assert summary["p95"] == pytest.approx(85.45)
    assert summary["p99"] == pytest.approx(97.09)
    assert summary["maximum"] == 100.0


def test_jain_index_reports_equal_service_and_starvation():
    benchmark = _load_benchmark()

    assert benchmark._jain_index([1.0, 1.0, 1.0, 1.0]) == 1.0
    assert benchmark._jain_index([1.0, 0.0, 0.0, 0.0]) == 0.25
    assert benchmark._jain_index([0.0, 0.0]) is None


def test_camera_fps_profile_repeats_and_defaults():
    benchmark = _load_benchmark()

    assert benchmark._camera_fps_profile(
        None,
        cameras=3,
        default_fps=4.0,
    ) == [4.0, 4.0, 4.0]
    assert benchmark._camera_fps_profile(
        "1,2,4",
        cameras=5,
        default_fps=4.0,
    ) == [1.0, 2.0, 4.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="positive"):
        benchmark._camera_fps_profile("1,0", cameras=2, default_fps=4.0)


def test_maximum_service_gap_includes_window_edges_and_starvation():
    benchmark = _load_benchmark()

    assert benchmark._maximum_gap_ms(
        [], window_start=10.0, window_end=12.0
    ) == 2000.0
    assert benchmark._maximum_gap_ms(
        [10.25, 10.5, 11.0], window_start=10.0, window_end=12.0
    ) == 1000.0
    assert benchmark._maximum_gap_ms(
        [], window_start=12.0, window_end=11.0
    ) == 0.0
