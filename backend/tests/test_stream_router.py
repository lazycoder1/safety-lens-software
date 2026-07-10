import asyncio
import gc
import weakref

import pytest
from fastapi import HTTPException
from starlette.requests import ClientDisconnect

from mjpeg_fanout import MjpegFanout, MjpegSubscriberLimitError
from routers import stream as stream_router


def test_stream_route_keeps_mjpeg_boundary_and_uses_async_fanout(monkeypatch):
    calls = []

    class FakeFanout:
        def stream(self, camera_id):
            calls.append(camera_id)

            async def frames():
                yield b"frame"

            return frames()

    monkeypatch.setattr(stream_router, "get_config", lambda: {"cameras": {"cam1": {}}})
    monkeypatch.setattr(stream_router, "stream_fanout", FakeFanout())

    response = asyncio.run(stream_router.stream("cam1"))

    assert calls == ["cam1"]
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
    assert hasattr(response.body_iterator, "__anext__")
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["x-accel-buffering"] == "no"

    asyncio.run(response.body_iterator.aclose())


def test_stream_route_rejects_unknown_camera(monkeypatch):
    monkeypatch.setattr(stream_router, "get_config", lambda: {"cameras": {}})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stream_router.stream("missing"))

    assert exc_info.value.status_code == 404


def test_stream_route_rejects_subscriber_overflow_before_response(monkeypatch):
    class FullFanout:
        def stream(self, _camera_id):
            raise MjpegSubscriberLimitError("Camera cam1 reached its stream subscriber limit")

        def operational_stats(self):
            return {
                "subscribers": 16,
                "subscriberLimitPerCamera": 16,
                "subscriberLimitTotal": 64,
                "rejectedSubscribers": 1,
            }

    monkeypatch.setattr(stream_router, "get_config", lambda: {"cameras": {"cam1": {}}})
    monkeypatch.setattr(stream_router, "stream_fanout", FullFanout())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stream_router.stream("cam1"))

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers == {"Retry-After": "5"}


def test_stream_response_closes_subscription_when_socket_send_fails():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"jpeg")
        response = stream_router.MjpegStreamingResponse(
            hub.stream("cam1"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
        messages = []

        async def send(message):
            messages.append(message["type"])
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        async def receive():
            return {"type": "http.disconnect"}

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)

        assert messages == ["http.response.start", "http.response.body"]
        assert hub.stats("cam1")["subscribers"] == 0

    asyncio.run(scenario())


def test_stream_response_releases_subscription_after_terminal_retire():
    async def scenario():
        hub = MjpegFanout()
        hub.publish("cam1", b"jpeg-a")
        response = stream_router.MjpegStreamingResponse(
            hub.stream("cam1"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
        bodies = []

        async def send(message):
            if message["type"] != "http.response.body" or not message.get("more_body"):
                return
            bodies.append(message["body"])
            if len(bodies) == 1:
                hub.publish("cam1", b"jpeg-b")
            else:
                hub.retire("cam1")

        async def receive():
            return {"type": "http.disconnect"}

        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        assert b"jpeg-a" in bodies[0]
        assert b"jpeg-b" in bodies[1]
        assert len(bodies) == 2
        assert hub.stats("cam1")["subscribers"] == 0

    asyncio.run(scenario())


def test_discarded_unstarted_response_releases_reserved_subscription():
    async def scenario():
        hub = MjpegFanout(max_subscribers_total=1)
        response = stream_router.MjpegStreamingResponse(hub.stream("cam1"))
        response_ref = weakref.ref(response)
        assert hub.operational_stats()["subscribers"] == 1

        del response
        gc.collect()

        assert response_ref() is None
        assert hub.operational_stats()["subscribers"] == 0
        replacement = hub.stream("cam2")
        await replacement.aclose()

    asyncio.run(scenario())
