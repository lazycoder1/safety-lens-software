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
