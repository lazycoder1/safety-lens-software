"""Latest-frame MJPEG fanout shared by every viewer of a camera."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field


_MJPEG_PREFIX = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_MJPEG_SUFFIX = b"\r\n"


def build_mjpeg_chunk(jpeg: bytes) -> bytes:
    """Build one immutable multipart chunk for all subscribers."""
    return _MJPEG_PREFIX + jpeg + _MJPEG_SUFFIX


class MjpegSubscriberLimitError(RuntimeError):
    """Raised before streaming headers when a fanout subscriber cap is full."""


@dataclass(eq=False)
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    event: asyncio.Event
    notified: bool = False


@dataclass
class _CameraChannel:
    sequence: int = 0
    jpeg: bytes | None = None
    chunk: bytes | None = None
    published_at: float | None = None
    subscribers: set[_Subscriber] = field(default_factory=set)
    retired: bool = False


class _StreamSubscription(AsyncIterator[bytes]):
    def __init__(
        self,
        hub: "MjpegFanout",
        channel: _CameraChannel,
        subscriber: _Subscriber,
    ) -> None:
        self._hub = hub
        self._channel = channel
        self._subscriber = subscriber
        self._last_sequence = -1
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            hub._unsubscribe,
            channel,
            subscriber,
        )

    def __aiter__(self) -> "_StreamSubscription":
        return self

    async def __anext__(self) -> bytes:
        while not self._closed:
            self._subscriber.event.clear()
            with self._hub._lock:
                self._subscriber.notified = False
                retired = self._channel.retired
                sequence = self._channel.sequence
                if self._channel.jpeg is not None and self._channel.chunk is None:
                    self._channel.chunk = build_mjpeg_chunk(self._channel.jpeg)
                chunk = self._channel.chunk

            if retired:
                await self.aclose()
                raise StopAsyncIteration

            if sequence != self._last_sequence:
                self._last_sequence = sequence
                if chunk is not None:
                    return chunk
                continue

            try:
                await self._subscriber.event.wait()
            except asyncio.CancelledError:
                await self.aclose()
                raise

        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close()

    def close(self) -> None:
        """Release a reservation synchronously, including before iteration starts."""
        if self._closed:
            return
        self._closed = True
        self._finalizer()


class MjpegFanout:
    """Thread-safe, latest-only delivery from camera workers to async clients."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_subscribers_per_camera: int | None = None,
        max_subscribers_total: int | None = None,
    ) -> None:
        self._clock = clock
        self._max_subscribers_per_camera = max_subscribers_per_camera
        self._max_subscribers_total = max_subscribers_total
        # Finalizers may run during a GC-triggering clock/build call while this
        # same thread owns the hub lock. Reentrancy keeps lease cleanup safe.
        self._lock = threading.RLock()
        self._channels: dict[str, _CameraChannel] = {}
        self._subscriber_count = 0
        self._rejected_subscribers = 0
        self._rejected_by_camera: dict[str, int] = {}

    def publish(self, camera_id: str, jpeg: bytes) -> int:
        """Publish a JPEG; viewers share one chunk and slow viewers skip ahead."""
        with self._lock:
            channel = self._channels.setdefault(camera_id, _CameraChannel())
            self._prune_closed_subscribers_locked(channel)
            channel.sequence += 1
            channel.jpeg = jpeg
            channel.chunk = build_mjpeg_chunk(jpeg) if channel.subscribers else None
            channel.published_at = self._clock()
            if not channel.subscribers:
                channel.chunk = None
            sequence = channel.sequence
            subscribers = self._mark_wakes_locked(channel)
        self._wake(channel, subscribers)
        return sequence

    def clear(self, camera_id: str) -> int:
        """Invalidate a transiently offline camera while keeping viewers attached."""
        with self._lock:
            channel = self._channels.get(camera_id)
            if channel is None or channel.jpeg is None:
                return channel.sequence if channel is not None else 0
            channel.sequence += 1
            channel.jpeg = None
            channel.chunk = None
            channel.published_at = None
            sequence = channel.sequence
            subscribers = self._mark_wakes_locked(channel)
        self._wake(channel, subscribers)
        return sequence

    def retire(self, camera_id: str) -> bool:
        """Close existing viewers and detach a terminally deleted camera channel."""
        with self._lock:
            channel = self._channels.pop(camera_id, None)
            if channel is None:
                return False
            channel.retired = True
            channel.sequence += 1
            channel.jpeg = None
            channel.chunk = None
            channel.published_at = None
            subscribers = self._mark_wakes_locked(channel)
        self._wake(channel, subscribers)
        return True

    def stream(self, camera_id: str) -> AsyncIterator[bytes]:
        """Reserve and return a bounded subscription before response headers are sent."""
        subscriber = _Subscriber(asyncio.get_running_loop(), asyncio.Event())
        with self._lock:
            channel = self._channels.setdefault(camera_id, _CameraChannel())
            self._prune_closed_subscribers_locked(channel)
            if self._limit_reached(len(channel.subscribers), self._max_subscribers_per_camera):
                self._record_rejection_locked(camera_id)
                raise MjpegSubscriberLimitError(
                    f"Camera {camera_id} reached its stream subscriber limit"
                )
            if self._limit_reached(self._subscriber_count, self._max_subscribers_total):
                self._record_rejection_locked(camera_id)
                raise MjpegSubscriberLimitError("Global stream subscriber limit reached")
            channel.subscribers.add(subscriber)
            self._subscriber_count += 1
            if channel.jpeg is not None and channel.chunk is None:
                channel.chunk = build_mjpeg_chunk(channel.jpeg)
        return _StreamSubscription(self, channel, subscriber)

    def stats(self, camera_id: str) -> dict:
        """Return a consistent operational snapshot without exposing frame bytes."""
        with self._lock:
            channel = self._channels.get(camera_id)
            if channel is None:
                return {
                    "sequence": 0,
                    "subscribers": 0,
                    "has_frame": False,
                    "published_at": None,
                    "frame_age_seconds": None,
                }
            published_at = channel.published_at
            return {
                "sequence": channel.sequence,
                "subscribers": len(channel.subscribers),
                "has_frame": channel.jpeg is not None,
                "published_at": published_at,
                "frame_age_seconds": (
                    max(0.0, self._clock() - published_at)
                    if published_at is not None
                    else None
                ),
            }

    def has_subscribers(self, camera_id: str) -> bool:
        """Return whether a camera currently has live stream demand."""
        with self._lock:
            channel = self._channels.get(camera_id)
            if channel is None:
                return False
            self._prune_closed_subscribers_locked(channel)
            return bool(channel.subscribers)

    def operational_stats(self) -> dict:
        """Expose effective limits and aggregate load for health diagnostics."""
        with self._lock:
            return {
                "subscribers": self._subscriber_count,
                "subscriberLimitPerCamera": self._max_subscribers_per_camera,
                "subscriberLimitTotal": self._max_subscribers_total,
                "rejectedSubscribers": self._rejected_subscribers,
                "rejectedByCamera": dict(self._rejected_by_camera),
            }

    @staticmethod
    def _limit_reached(current: int, limit: int | None) -> bool:
        return limit is not None and current >= limit

    def _mark_wakes_locked(self, channel: _CameraChannel) -> tuple[_Subscriber, ...]:
        pending = []
        for subscriber in channel.subscribers:
            if not subscriber.notified:
                subscriber.notified = True
                pending.append(subscriber)
        return tuple(pending)

    def _record_rejection_locked(self, camera_id: str) -> None:
        self._rejected_subscribers += 1
        self._rejected_by_camera[camera_id] = self._rejected_by_camera.get(camera_id, 0) + 1

    def _wake(self, channel: _CameraChannel, subscribers: tuple[_Subscriber, ...]) -> None:
        stale: list[_Subscriber] = []
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.event.set)
            except RuntimeError:
                stale.append(subscriber)
        if not stale:
            return
        with self._lock:
            for subscriber in stale:
                self._unsubscribe_locked(channel, subscriber)

    def _unsubscribe(self, channel: _CameraChannel, subscriber: _Subscriber) -> None:
        with self._lock:
            removed = self._unsubscribe_locked(channel, subscriber)
        if removed:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.event.set)
            except RuntimeError:
                pass

    def _unsubscribe_locked(self, channel: _CameraChannel, subscriber: _Subscriber) -> bool:
        if subscriber not in channel.subscribers:
            return False
        channel.subscribers.remove(subscriber)
        self._subscriber_count = max(0, self._subscriber_count - 1)
        if not channel.subscribers:
            channel.chunk = None
        return True

    def _prune_closed_subscribers_locked(self, channel: _CameraChannel) -> None:
        for subscriber in tuple(channel.subscribers):
            if subscriber.loop.is_closed():
                self._unsubscribe_locked(channel, subscriber)


def _env_limit(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative integer") from None
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


stream_fanout = MjpegFanout(
    max_subscribers_per_camera=_env_limit("MJPEG_MAX_SUBSCRIBERS_PER_CAMERA", 16),
    max_subscribers_total=_env_limit("MJPEG_MAX_SUBSCRIBERS_TOTAL", 64),
)
