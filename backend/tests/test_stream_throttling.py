import asyncio

import numpy as np

import config_manager
import video_processing
from mjpeg_fanout import MjpegFanout, build_mjpeg_chunk


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


def test_idle_stream_publication_uses_one_fps_heartbeat(monkeypatch):
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "has_subscribers",
        lambda _camera_id: False,
    )

    assert video_processing._stream_publication_due("cam1", 0.0, 10.0, 4.0, False) == (
        True,
        False,
    )
    assert video_processing._stream_publication_due("cam1", 10.0, 10.99, 4.0, False) == (
        False,
        False,
    )
    assert video_processing._stream_publication_due("cam1", 10.0, 11.0, 4.0, False) == (
        True,
        False,
    )


def test_stream_publication_restores_active_rate_and_forces_join_frame(monkeypatch):
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "has_subscribers",
        lambda _camera_id: True,
    )

    # Joining forces a fresh frame even when the configured interval has not elapsed.
    assert video_processing._stream_publication_due("cam1", 10.0, 10.01, 4.0, False) == (
        True,
        True,
    )
    assert video_processing._stream_publication_due("cam1", 10.0, 10.24, 4.0, True) == (
        False,
        True,
    )
    assert video_processing._stream_publication_due("cam1", 10.0, 10.25, 4.0, True) == (
        True,
        True,
    )


def test_idle_stream_does_not_raise_a_slower_configured_rate(monkeypatch):
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "has_subscribers",
        lambda _camera_id: False,
    )

    assert video_processing._stream_publication_due("cam1", 10.0, 11.5, 0.5, False) == (
        False,
        False,
    )
    assert video_processing._stream_publication_due("cam1", 10.0, 12.0, 0.5, False) == (
        True,
        False,
    )


def test_idle_stream_schedule_cuts_publication_work_by_three_quarters(monkeypatch):
    demand = {"active": False}
    monkeypatch.setattr(
        video_processing.stream_fanout,
        "has_subscribers",
        lambda _camera_id: demand["active"],
    )

    def publication_count(active: bool) -> int:
        demand["active"] = active
        last_published_at = 0.0
        had_subscribers = False
        published = 0
        for frame_index in range(64):  # Eight seconds of an 8 FPS capture.
            now = 10.0 + frame_index / 8.0
            due, had_subscribers = video_processing._stream_publication_due(
                "cam1",
                last_published_at,
                now,
                4.0,
                had_subscribers,
            )
            if due:
                last_published_at = now
                published += 1
        return published

    assert publication_count(active=True) == 32
    assert publication_count(active=False) == 8


def test_empty_scene_inference_skips_unchanged_frame_before_refresh():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    first_due, signature, _score = video_processing._empty_scene_inference_decision(
        frame,
        None,
        last_submitted_at=None,
        now=10.0,
        detections=[],
    )
    next_due, _next_signature, score = video_processing._empty_scene_inference_decision(
        frame.copy(),
        signature,
        last_submitted_at=10.0,
        now=10.25,
        detections=[],
    )

    assert first_due is True
    assert next_due is False
    assert score == 0.0


def test_empty_scene_inference_runs_on_small_visual_change():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    _due, signature, _score = video_processing._empty_scene_inference_decision(
        frame,
        None,
        last_submitted_at=None,
        now=10.0,
        detections=[],
    )
    changed = frame.copy()
    changed[70:110, 140:180] = 255

    due, _signature, score = video_processing._empty_scene_inference_decision(
        changed,
        signature,
        last_submitted_at=10.0,
        now=10.25,
        detections=[],
    )

    assert due is True
    assert score > video_processing.EMPTY_SCENE_CHANGED_FRACTION


def test_empty_scene_inference_forces_refresh_and_keeps_active_scenes_full_rate():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    _due, signature, _score = video_processing._empty_scene_inference_decision(
        frame,
        None,
        last_submitted_at=None,
        now=10.0,
        detections=[],
    )

    refresh_due, _signature, _score = video_processing._empty_scene_inference_decision(
        frame,
        signature,
        last_submitted_at=10.0,
        now=11.0,
        detections=[],
    )
    active_due, _signature, _score = video_processing._empty_scene_inference_decision(
        frame,
        signature,
        last_submitted_at=10.0,
        now=10.25,
        detections=[{"class": "person"}],
    )

    assert refresh_due is True
    assert active_due is True


def test_stream_consumer_waits_for_next_published_sequence():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("stream_test", b"jpeg-a")
        stream = hub.stream("stream_test")
        try:
            assert await anext(stream) == build_mjpeg_chunk(b"jpeg-a")
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert pending.done() is False

            hub.publish("stream_test", b"jpeg-b")
            assert await asyncio.wait_for(pending, 1) == build_mjpeg_chunk(b"jpeg-b")
        finally:
            await stream.aclose()

    asyncio.run(scenario())
