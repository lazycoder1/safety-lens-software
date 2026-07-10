import asyncio

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
