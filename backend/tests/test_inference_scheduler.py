import pytest

from inference_scheduler import camera_phase_offset, next_inference_slot


def _config(camera_count=5):
    return {
        "cameras": {
            f"cam{index + 1}": {"enabled": True}
            for index in range(camera_count)
        }
    }


def test_camera_phases_are_evenly_spaced_across_interval():
    cfg = _config(5)

    offsets = [camera_phase_offset(f"cam{index + 1}", cfg, 1.0 / 3.0) for index in range(5)]

    assert offsets == pytest.approx([0.0, 1 / 15, 2 / 15, 3 / 15, 4 / 15])


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
