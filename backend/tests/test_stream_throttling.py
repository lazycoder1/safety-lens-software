import pytest

import config_manager
import state
import video_processing


def test_default_stream_fps_is_lower_than_capture_fps():
    global_config = config_manager.DEFAULT_CONFIG["global"]

    assert global_config["target_fps"] == 6
    assert global_config["stream_fps"] == 4


def test_configured_stream_fps_supports_override_and_capture_clamp():
    assert video_processing._configured_stream_fps({}, {"stream_fps": 4}, 8) == 4
    assert video_processing._configured_stream_fps({"stream_fps": 3}, {"stream_fps": 4}, 8) == 3
    assert video_processing._configured_stream_fps({"stream_fps": 12}, {"stream_fps": 4}, 8) == 8
    assert video_processing._configured_stream_fps({"stream_fps": "invalid"}, {"stream_fps": 4}, 8) == 4


def test_stream_publish_due_uses_monotonic_interval():
    assert video_processing._stream_publish_due(0.0, 10.0, 4)
    assert not video_processing._stream_publish_due(10.0, 10.24, 4)
    assert video_processing._stream_publish_due(10.0, 10.25, 4)


def test_mjpeg_generator_paces_cached_frames_at_stream_fps(monkeypatch):
    camera_id = "stream_test"
    state.camera_frames[camera_id] = b"jpeg"
    sleeps = []
    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {
            "global": {"target_fps": 8, "stream_fps": 4},
            "cameras": {camera_id: {"fps": 8}},
        },
    )
    monkeypatch.setattr(video_processing.time, "sleep", sleeps.append)

    generator = video_processing.mjpeg_generator(camera_id)
    try:
        assert b"jpeg" in next(generator)
        assert b"jpeg" in next(generator)
        assert sleeps == [pytest.approx(0.25)]
    finally:
        generator.close()
        state.camera_frames.pop(camera_id, None)
