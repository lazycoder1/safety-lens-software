import json
import logging
import threading

import camera_capture
from logging_config import JSONFormatter
import state
import video_processing


def test_redact_video_source_removes_credentials_and_query_values():
    source = "rtsp://admin:p%40ss@192.168.1.20:554/cam/live?token=secret&channel=1"

    redacted = camera_capture.redact_video_source(source)

    assert redacted == "rtsp://192.168.1.20:554/cam/live"
    assert "admin" not in redacted
    assert "secret" not in redacted


def test_open_rtsp_capture_forces_ffmpeg_and_bounded_timeouts(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.open_call = None
            self.set_calls = []

        def open(self, *args):
            self.open_call = args
            return True

        def isOpened(self):
            return True

        def set(self, *args):
            self.set_calls.append(args)

    capture = FakeCapture()
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda: capture)

    result = camera_capture.open_video_capture("rtsp://camera/live", stream_type="rtsp")

    assert result is capture
    source, backend, parameters = capture.open_call
    assert source == "rtsp://camera/live"
    assert backend == camera_capture.cv2.CAP_FFMPEG
    assert parameters == [
        camera_capture.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        camera_capture.RTSP_OPEN_TIMEOUT_MS,
        camera_capture.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        camera_capture.RTSP_READ_TIMEOUT_MS,
    ]
    assert capture.set_calls == [(camera_capture.cv2.CAP_PROP_BUFFERSIZE, 1)]


def test_open_rtsp_capture_does_not_fall_back_after_ffmpeg_error(monkeypatch):
    class FailedCapture:
        def __init__(self):
            self.open_calls = 0

        def open(self, *_args):
            self.open_calls += 1
            raise camera_capture.cv2.error("FFmpeg rejected stream")

        def isOpened(self):
            return False

    capture = FailedCapture()
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda: capture)

    result = camera_capture.open_video_capture("rtsp://user:secret@camera/live", stream_type="rtsp")

    assert result is capture
    assert capture.open_calls == 1


def test_reconnect_backoff_grows_and_caps():
    delays = [camera_capture.reconnect_delay_seconds(index, "cam1") for index in range(10)]

    assert delays == sorted(delays)
    assert delays[1] > delays[0]
    assert delays[-1] == camera_capture.RTSP_RECONNECT_MAX_SECONDS
    assert camera_capture.reconnect_delay_seconds(100, "cam2") == camera_capture.RTSP_RECONNECT_MAX_SECONDS


def test_restart_does_not_spawn_duplicate_when_worker_is_still_alive(monkeypatch):
    class StuckThread:
        def __init__(self):
            self.join_timeout = None

        def join(self, timeout):
            self.join_timeout = timeout

        def is_alive(self):
            return True

    thread = StuckThread()
    stop_event = threading.Event()
    monkeypatch.setattr(state, "camera_threads", {"cam1": (thread, stop_event)})
    monkeypatch.setattr(state, "vlm_threads", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(state, "camera_frames", {"cam1": b"frame"})
    monkeypatch.setattr(state, "camera_detections", {"cam1": []})
    starts = []
    monkeypatch.setattr(video_processing, "start_camera", lambda camera_id: starts.append(camera_id))

    restarted = video_processing.restart_camera("cam1")

    assert restarted is False
    assert stop_event.is_set()
    assert thread.join_timeout == camera_capture.CAMERA_STOP_TIMEOUT_SECONDS
    assert state.camera_threads["cam1"][0] is thread
    assert state.camera_runtime_status["cam1"] == "stopping"
    assert starts == []


def test_structured_camera_retry_log_keeps_backoff_telemetry():
    record = logging.LogRecord(
        name="safetylens",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="RTSP stream interrupted; retry scheduled",
        args=(),
        exc_info=None,
    )
    record.camera_id = "cam1"
    record.source = "rtsp://192.168.1.20:554/live"
    record.retry_seconds = 8.5
    record.failure_count = 4
    record.received_frame = False

    payload = json.loads(JSONFormatter().format(record))

    assert payload["retry_seconds"] == 8.5
    assert payload["failure_count"] == 4
    assert payload["received_frame"] is False
    assert payload["source"] == "rtsp://192.168.1.20:554/live"
