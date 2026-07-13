import pytest

from inference_scheduler import camera_phase_offset, next_inference_slot


def _config(camera_count=5):
    return {
        "cameras": {
            f"cam{index + 1}": {"enabled": True}
            for index in range(camera_count)
        }
    }


def test_camera_phases_are_evenly_spaced_across_interval(monkeypatch):
    monkeypatch.delenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", raising=False)
    monkeypatch.delenv("SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT", raising=False)
    cfg = _config(5)

    offsets = [camera_phase_offset(f"cam{index + 1}", cfg, 1.0 / 3.0) for index in range(5)]

    assert offsets == pytest.approx([0.0, 1 / 15, 2 / 15, 3 / 15, 4 / 15])


def test_camera_phases_can_be_grouped_for_pre_admission_batching(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "2")
    monkeypatch.delenv("SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT", raising=False)
    cfg = _config(5)

    offsets = [camera_phase_offset(f"cam{index + 1}", cfg, 0.3) for index in range(5)]

    assert offsets == pytest.approx([0.0, 0.0, 0.1, 0.1, 0.2])


def test_partial_final_phase_can_reserve_less_of_the_inference_period(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "4")
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT", "0.7")
    cfg = _config(27)

    offsets = [camera_phase_offset(f"cam{index + 1}", cfg, 0.25) for index in range(27)]

    groups = sorted(set(offsets))
    assert groups == pytest.approx([index * 0.25 / 6.7 for index in range(7)])
    assert 0.25 - groups[-1] == pytest.approx(0.25 * 0.7 / 6.7)


def test_remainder_weight_does_not_change_full_group_layout(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "4")
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT", "0.7")
    cfg = _config(20)

    offsets = [camera_phase_offset(f"cam{index + 1}", cfg, 0.25) for index in range(20)]

    assert sorted(set(offsets)) == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2])


@pytest.mark.parametrize("value", ["invalid", "nan", "inf"])
def test_invalid_remainder_weight_falls_back_to_uniform_spacing(monkeypatch, value):
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "4")
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_REMAINDER_WEIGHT", value)

    assert camera_phase_offset("cam5", _config(5), 0.3) == pytest.approx(0.15)


def test_invalid_phase_group_size_falls_back_to_one(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_INFERENCE_PHASE_GROUP_SIZE", "invalid")

    assert camera_phase_offset("cam2", _config(2), 0.4) == pytest.approx(0.2)


def test_next_slot_preserves_shared_grid_after_missed_cycle():
    cfg = _config(2)

    assert next_inference_slot("cam1", cfg, 0.4, now=10.81, epoch=10.0) == pytest.approx(11.2)
    assert next_inference_slot("cam2", cfg, 0.4, now=10.81, epoch=10.0) == pytest.approx(11.0)


def test_disabled_cameras_do_not_consume_phase_slots():
    cfg = _config(3)
    cfg["cameras"]["cam2"]["enabled"] = False

    assert camera_phase_offset("cam1", cfg, 0.4) == pytest.approx(0.0)
    assert camera_phase_offset("cam3", cfg, 0.4) == pytest.approx(0.2)


def test_unknown_runtime_camera_gets_a_stable_slot():
    cfg = _config(2)

    assert camera_phase_offset("cam3", cfg, 0.3) == pytest.approx(0.2)


def test_non_positive_interval_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        camera_phase_offset("cam1", _config(), 0.0)
