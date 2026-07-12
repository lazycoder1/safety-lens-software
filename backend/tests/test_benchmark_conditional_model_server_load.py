import importlib.util
from pathlib import Path


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
