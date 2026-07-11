import asyncio

import state
import video_processing


def test_broadcast_alert_sends_to_subscribers_concurrently():
    active_sends = 0
    peak_active_sends = 0

    class Subscriber:
        async def send_json(self, _message):
            nonlocal active_sends, peak_active_sends
            active_sends += 1
            peak_active_sends = max(peak_active_sends, active_sends)
            await asyncio.sleep(0)
            active_sends -= 1

    subscribers = [Subscriber() for _ in range(4)]
    state.alert_subscribers[:] = subscribers
    try:
        asyncio.run(video_processing.broadcast_alert({"type": "alert"}))
    finally:
        state.alert_subscribers.clear()

    assert peak_active_sends == len(subscribers)


def test_broadcast_alert_removes_only_failed_subscribers():
    messages = []

    class HealthySubscriber:
        async def send_json(self, message):
            messages.append(message)

    class FailedSubscriber:
        async def send_json(self, _message):
            state.alert_subscribers.remove(self)
            raise RuntimeError("connection closed")

    healthy = HealthySubscriber()
    failed = FailedSubscriber()
    message = {"type": "alert", "data": {"id": "alert-1"}}
    state.alert_subscribers[:] = [healthy, failed]
    try:
        asyncio.run(video_processing.broadcast_alert(message))
        assert state.alert_subscribers == [healthy]
    finally:
        state.alert_subscribers.clear()

    assert messages == [message]
