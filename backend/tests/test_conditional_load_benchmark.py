from scripts.benchmark_conditional_model_server_load import _phase_group_cardinality


def test_grouped_phase_cardinality_exposes_singleton_remainder():
    assert _phase_group_cardinality(0, 21, "grouped", 4) == 4
    assert _phase_group_cardinality(19, 21, "grouped", 4) == 4
    assert _phase_group_cardinality(20, 21, "grouped", 4) == 1


def test_phase_cardinality_matches_each_supported_schedule_shape():
    assert _phase_group_cardinality(4, 5, "paired", 4) == 1
    assert _phase_group_cardinality(2, 5, "staggered", 4) == 1
    assert _phase_group_cardinality(2, 5, "aligned", 4) == 5
