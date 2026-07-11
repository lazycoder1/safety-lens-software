import asyncio
import gc
import threading
import weakref
from contextlib import suppress

import mjpeg_fanout
import pytest
from mjpeg_fanout import (
    MjpegFanout,
    MjpegSubscriberLimitError,
    build_mjpeg_chunk,
)


def test_two_subscribers_share_one_prebuilt_chunk(monkeypatch):
    async def scenario():
        hub = MjpegFanout()
        build_calls = []
        original_build = mjpeg_fanout.build_mjpeg_chunk

        def tracked_build(jpeg):
            build_calls.append(jpeg)
            return original_build(jpeg)

        monkeypatch.setattr(mjpeg_fanout, "build_mjpeg_chunk", tracked_build)
        hub.publish("cam1", b"jpeg-a")
        assert build_calls == []
        assert hub.has_subscribers("cam1") is False
        first = hub.stream("cam1")
        second = hub.stream("cam1")
        assert hub.has_subscribers("cam1") is True
        try:
            first_chunk = await anext(first)
            second_chunk = await anext(second)

            assert first_chunk is second_chunk
            assert first_chunk == build_mjpeg_chunk(b"jpeg-a")
            assert build_calls == [b"jpeg-a"]
            assert hub.stats("cam1")["subscribers"] == 2
        finally:
            await first.aclose()
            await second.aclose()

        assert hub.stats("cam1")["subscribers"] == 0
        assert hub.has_subscribers("cam1") is False

    asyncio.run(scenario())


def test_subscriber_yields_cached_frame_once_then_waits_for_new_sequence():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"jpeg-a")
        stream = hub.stream("cam1")
        try:
            assert await anext(stream) == build_mjpeg_chunk(b"jpeg-a")

            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert pending.done() is False

            hub.publish("cam1", b"jpeg-b")
            assert await asyncio.wait_for(pending, 1) == build_mjpeg_chunk(b"jpeg-b")
        finally:
            await stream.aclose()

    asyncio.run(scenario())


def test_slow_subscriber_skips_intermediate_frames():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"jpeg-a")
        stream = hub.stream("cam1")
        try:
            assert await anext(stream) == build_mjpeg_chunk(b"jpeg-a")
            hub.publish("cam1", b"jpeg-b")
            hub.publish("cam1", b"jpeg-c")

            assert await anext(stream) == build_mjpeg_chunk(b"jpeg-c")
        finally:
            await stream.aclose()

    asyncio.run(scenario())


def test_clear_wakes_subscriber_without_replaying_stale_frame():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"jpeg-a")
        stream = hub.stream("cam1")
        try:
            assert await anext(stream) == build_mjpeg_chunk(b"jpeg-a")
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)

            hub.clear("cam1")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert pending.done() is False
            assert hub.stats("cam1")["has_frame"] is False

            hub.publish("cam1", b"jpeg-b")
            assert await asyncio.wait_for(pending, 1) == build_mjpeg_chunk(b"jpeg-b")
        finally:
            await stream.aclose()

    asyncio.run(scenario())


def test_cross_thread_publish_wakes_only_matching_camera():
    async def scenario():
        hub = MjpegFanout()
        first = hub.stream("cam1")
        second = hub.stream("cam2")
        first_pending = asyncio.create_task(anext(first))
        second_pending = asyncio.create_task(anext(second))
        await asyncio.sleep(0)

        publisher = threading.Thread(target=hub.publish, args=("cam1", b"jpeg-a"))
        publisher.start()
        publisher.join()

        try:
            assert await asyncio.wait_for(first_pending, 1) == build_mjpeg_chunk(b"jpeg-a")
            assert second_pending.done() is False
        finally:
            second_pending.cancel()
            with suppress(asyncio.CancelledError):
                await second_pending
            await first.aclose()

        assert hub.stats("cam1")["subscribers"] == 0
        assert hub.stats("cam2")["subscribers"] == 0

    asyncio.run(scenario())


def test_cancelled_waiter_unregisters_subscriber():
    async def scenario():
        hub = MjpegFanout()
        stream = hub.stream("cam1")
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        assert hub.stats("cam1")["subscribers"] == 1

        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await asyncio.sleep(0)

        assert hub.stats("cam1")["subscribers"] == 0

    asyncio.run(scenario())


def test_unstarted_or_immediately_cancelled_subscription_releases_cap():
    async def scenario():
        hub = MjpegFanout(max_subscribers_total=1)

        unused = hub.stream("cam1")
        unused_ref = weakref.ref(unused)
        del unused
        gc.collect()
        assert unused_ref() is None
        assert hub.operational_stats()["subscribers"] == 0

        cancelled = hub.stream("cam1")
        cancelled_ref = weakref.ref(cancelled)
        pending = asyncio.create_task(anext(cancelled))
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        del pending
        del cancelled
        await asyncio.sleep(0)
        gc.collect()

        assert cancelled_ref() is None
        assert hub.operational_stats()["subscribers"] == 0
        replacement = hub.stream("cam2")
        await replacement.aclose()

    asyncio.run(scenario())


def test_cycle_finalizer_cannot_deadlock_publish_while_hub_lock_is_held():
    async def scenario():
        hub = MjpegFanout(clock=lambda: float(gc.collect()))
        subscription = hub.stream("cam1")
        cycle = [subscription]
        cycle.append(cycle)
        del subscription
        del cycle

        publisher = threading.Thread(target=hub.publish, args=("cam1", b"jpeg"), daemon=True)
        publisher.start()
        publisher.join(timeout=1)

        assert publisher.is_alive() is False
        assert hub.operational_stats()["subscribers"] == 0

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(scenario())
    finally:
        if was_enabled:
            gc.enable()


def test_publish_wakes_are_coalesced_while_subscriber_is_slow(monkeypatch):
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"initial")
        stream = hub.stream("cam1")
        assert await anext(stream) == build_mjpeg_chunk(b"initial")

        loop = asyncio.get_running_loop()
        original_wake = loop.call_soon_threadsafe
        wake_calls = []

        def tracked_wake(callback, *args):
            wake_calls.append((callback, args))
            return original_wake(callback, *args)

        monkeypatch.setattr(loop, "call_soon_threadsafe", tracked_wake)
        for index in range(10_000):
            hub.publish("cam1", f"jpeg-{index}".encode())

        assert len(wake_calls) == 1
        assert hub.stats("cam1")["sequence"] == 10_001
        await stream.aclose()

    asyncio.run(scenario())


def test_subscriber_caps_release_immediately_on_close():
    async def scenario():
        hub = MjpegFanout(max_subscribers_per_camera=1, max_subscribers_total=1)
        first = hub.stream("cam1")

        with pytest.raises(MjpegSubscriberLimitError, match="Camera cam1"):
            hub.stream("cam1")
        with pytest.raises(MjpegSubscriberLimitError, match="Global"):
            hub.stream("cam2")

        assert hub.operational_stats() == {
            "subscribers": 1,
            "subscriberLimitPerCamera": 1,
            "subscriberLimitTotal": 1,
            "rejectedSubscribers": 2,
            "rejectedByCamera": {"cam1": 1, "cam2": 1},
        }

        await first.aclose()
        replacement = hub.stream("cam2")
        assert hub.stats("cam2")["subscribers"] == 1
        await replacement.aclose()

    asyncio.run(scenario())


def test_retire_ends_old_stream_and_reused_id_gets_new_channel():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"old")
        old_stream = hub.stream("cam1")
        assert await anext(old_stream) == build_mjpeg_chunk(b"old")

        pending = asyncio.create_task(anext(old_stream))
        await asyncio.sleep(0)
        assert hub.retire("cam1") is True
        assert hub.has_subscribers("cam1") is False
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(pending, 1)
        assert hub.stats("cam1")["subscribers"] == 0

        hub.publish("cam1", b"new")
        new_stream = hub.stream("cam1")
        try:
            assert await anext(new_stream) == build_mjpeg_chunk(b"new")
        finally:
            await new_stream.aclose()

        with pytest.raises(StopAsyncIteration):
            await anext(old_stream)

    asyncio.run(scenario())


def test_stats_use_monotonic_age_and_clear_resets_current_frame():
    now = [100.0]
    hub = MjpegFanout(clock=lambda: now[0])

    assert hub.stats("cam1") == {
        "sequence": 0,
        "subscribers": 0,
        "has_frame": False,
        "published_at": None,
        "frame_age_seconds": None,
    }

    assert hub.publish("cam1", b"jpeg") == 1
    now[0] = 102.5
    assert hub.stats("cam1") == {
        "sequence": 1,
        "subscribers": 0,
        "has_frame": True,
        "published_at": 100.0,
        "frame_age_seconds": 2.5,
    }

    assert hub.clear("cam1") == 2
    assert hub.clear("cam1") == 2
    assert hub.stats("cam1")["published_at"] is None
    assert hub.stats("cam1")["frame_age_seconds"] is None
    assert hub.stats("cam1")["has_frame"] is False


def test_environment_limits_fail_fast_on_invalid_values(monkeypatch):
    monkeypatch.delenv("TEST_MJPEG_LIMIT", raising=False)
    assert mjpeg_fanout._env_limit("TEST_MJPEG_LIMIT", 16) == 16

    monkeypatch.setenv("TEST_MJPEG_LIMIT", "0")
    assert mjpeg_fanout._env_limit("TEST_MJPEG_LIMIT", 16) == 0

    for invalid in ("-1", "many"):
        monkeypatch.setenv("TEST_MJPEG_LIMIT", invalid)
        with pytest.raises(ValueError, match="non-negative integer"):
            mjpeg_fanout._env_limit("TEST_MJPEG_LIMIT", 16)
