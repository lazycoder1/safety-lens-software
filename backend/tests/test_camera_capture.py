import json
import logging
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import camera_capture
import camera_runtime
import diagnostics
from logging_config import JSONFormatter
import state
import video_processing
from mjpeg_fanout import MjpegFanout


def _rtsp_worker_config() -> dict:
    return {
        "cameras": {
            "cam1": {
                "stream_type": "rtsp",
                "fps": 1,
                "inference_fps": 1,
                "execution_plan": {"required_model_keys": []},
            },
        },
        "global": {
            "target_fps": 1,
            "inference_fps": 1,
            "alert_cooldown": 30,
            "yolo_conf": 0.3,
            "jpeg_quality": 70,
            "inference_width": 960,
            "device": "0",
        },
    }


def _patch_rtsp_worker_dependencies(monkeypatch, cfg: dict) -> None:
    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)
    monkeypatch.setattr(
        video_processing,
        "build_rtsp_url",
        lambda *_args, **_kwargs: "rtsp://camera/live",
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "missing_model_keys",
        lambda _keys: [],
    )
    monkeypatch.setattr(
        video_processing.licensing,
        "is_inference_allowed",
        lambda: True,
    )
    monkeypatch.setattr(
        video_processing.inference_scheduler,
        "next_inference_slot",
        lambda *_args, **_kwargs: 1_000_000.0,
    )
    monkeypatch.setattr(
        video_processing,
        "_publish_stream_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(state, "camera_connection_health", {})


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
        camera_capture.cv2.CAP_PROP_N_THREADS,
        camera_capture.RTSP_DECODE_THREADS,
    ]
    assert capture.set_calls == [(camera_capture.cv2.CAP_PROP_BUFFERSIZE, 1)]


def test_open_rtsp_capture_uses_nvdec_when_requested(monkeypatch):
    capture = object()
    opened = {}
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")
    monkeypatch.setattr(
        camera_capture,
        "_open_gstreamer_capture",
        lambda source, **kwargs: (
            opened.update(source=source, **kwargs) or capture
        ),
    )
    monkeypatch.setattr(
        camera_capture.cv2,
        "VideoCapture",
        lambda: pytest.fail("FFmpeg fallback should not be opened"),
    )

    assert (
        camera_capture.open_video_capture(
            "rtsp://camera/live",
            stream_type="rtsp",
            max_fps=6.5,
        )
        is capture
    )
    assert opened == {"source": "rtsp://camera/live", "max_fps": 6.5}


def test_open_rtsp_capture_supports_short_software_probe_timeouts(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.open_call = None

        def open(self, *args):
            self.open_call = args
            return True

        def isOpened(self):
            return False

    capture = FakeCapture()
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")
    monkeypatch.setattr(
        camera_capture,
        "_open_gstreamer_capture",
        lambda _source: pytest.fail("A one-frame probe should not initialize NVDEC"),
    )
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda: capture)

    result = camera_capture.open_video_capture(
        "rtsp://camera/live",
        stream_type="rtsp",
        prefer_hardware=False,
        open_timeout_ms=1_000,
        read_timeout_ms=2_500,
    )

    assert result is capture
    assert capture.open_call[2] == [
        camera_capture.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        1_000,
        camera_capture.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        2_500,
        camera_capture.cv2.CAP_PROP_N_THREADS,
        camera_capture.RTSP_DECODE_THREADS,
    ]


def test_open_rtsp_capture_falls_back_when_nvdec_runtime_is_missing(monkeypatch):
    class FakeCapture:
        def open(self, *_args):
            return True

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

    capture = FakeCapture()
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")
    monkeypatch.setattr(camera_capture, "_open_gstreamer_capture", lambda _source: None)
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda: capture)

    assert (
        camera_capture.open_video_capture(
            "rtsp://camera/live",
            stream_type="rtsp",
        )
        is capture
    )


def test_nvdec_failure_cache_skips_repeated_hardware_probe(monkeypatch):
    class FakeCapture:
        def open(self, *_args):
            return True

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

    now = [100.0]
    hardware_calls = []
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")
    monkeypatch.setattr(camera_capture.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        camera_capture,
        "_open_gstreamer_capture",
        lambda source, **_kwargs: hardware_calls.append(source),
    )
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", FakeCapture)
    camera_capture._reset_nvdec_retry_cache()

    first = camera_capture.open_video_capture(
        "rtsp://user:secret@camera/live",
        stream_type="rtsp",
    )
    second = camera_capture.open_video_capture(
        "rtsp://user:secret@camera/live",
        stream_type="rtsp",
    )
    now[0] += camera_capture.NVDEC_RETRY_SECONDS + 0.1
    third = camera_capture.open_video_capture(
        "rtsp://user:secret@camera/live",
        stream_type="rtsp",
    )

    assert first.isOpened() and second.isOpened() and third.isOpened()
    assert hardware_calls == [
        "rtsp://user:secret@camera/live",
        "rtsp://user:secret@camera/live",
    ]
    assert all(
        b"user" not in key and b"secret" not in key
        for key in camera_capture._NVDEC_RETRY_AFTER
    )


def test_successful_nvdec_probe_clears_cached_failure(monkeypatch):
    fallback = SimpleNamespace(
        open=lambda *_args: True,
        isOpened=lambda: True,
        set=lambda *_args: True,
    )
    hardware_capture = object()
    hardware_results = [None, hardware_capture, hardware_capture]
    now = [100.0]
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "auto")
    monkeypatch.setattr(camera_capture.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        camera_capture,
        "_open_gstreamer_capture",
        lambda _source, **_kwargs: hardware_results.pop(0),
    )
    monkeypatch.setattr(camera_capture.cv2, "VideoCapture", lambda: fallback)
    camera_capture._reset_nvdec_retry_cache()

    assert camera_capture.open_video_capture(
        "rtsp://camera/live",
        stream_type="rtsp",
    ) is fallback
    now[0] += camera_capture.NVDEC_RETRY_SECONDS + 0.1
    assert camera_capture.open_video_capture(
        "rtsp://camera/live",
        stream_type="rtsp",
    ) is hardware_capture
    assert camera_capture.open_video_capture(
        "rtsp://camera/live",
        stream_type="rtsp",
    ) is hardware_capture
    assert camera_capture._NVDEC_RETRY_AFTER == {}


def test_gstreamer_capture_that_does_not_open_is_released_and_falls_back(
    monkeypatch,
    caplog,
):
    class ClosedCapture:
        def __init__(self, *_args, **_kwargs):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    closed = ClosedCapture()
    monkeypatch.setitem(
        sys.modules,
        "gstreamer_capture",
        SimpleNamespace(
            GStreamerCapture=lambda *_args, **_kwargs: closed,
            nvdec_runtime_available=lambda: True,
        ),
    )

    with caplog.at_level(logging.WARNING, logger="rakshak_lens"):
        result = camera_capture._open_gstreamer_capture(
            "rtsp://user:secret@camera/live"
        )

    assert result is None
    assert closed.released is True
    assert "did not open" in caplog.text
    assert "user" not in caplog.text
    assert "secret" not in caplog.text


def test_invalid_capture_backend_defaults_to_ffmpeg(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "shell-command")

    assert camera_capture._rtsp_capture_backend() == "ffmpeg"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", 0),
        ("invalid", 0),
        ("-1", 0),
        ("100", 320),
        ("960", 960),
        ("99999", 4096),
    ],
)
def test_rtsp_max_dimension_is_disabled_or_safely_bounded(
    monkeypatch,
    configured,
    expected,
):
    monkeypatch.setenv("SAFETYLENS_RTSP_MAX_DIMENSION", configured)

    assert camera_capture._rtsp_max_dimension() == expected


def test_nvdec_drop_interval_is_safely_bounded():
    assert 0 <= camera_capture.NVDEC_DROP_FRAME_INTERVAL <= 30


def test_nvdec_drop_interval_is_forwarded_to_gstreamer(monkeypatch):
    captured = {}

    class OpenCapture:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

        def isOpened(self):
            return True

    monkeypatch.setattr(camera_capture, "NVDEC_DROP_FRAME_INTERVAL", 3)
    monkeypatch.setitem(
        sys.modules,
        "gstreamer_capture",
        SimpleNamespace(
            GStreamerCapture=OpenCapture,
            nvdec_runtime_available=lambda: True,
        ),
    )

    result = camera_capture._open_gstreamer_capture("rtsp://camera/live")

    assert isinstance(result, OpenCapture)
    assert captured["decoder_drop_interval"] == 3


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


def test_video_worker_uses_bounded_rtsp_drain_budget():
    assert 0.004 <= camera_capture.RTSP_BUFFER_DRAIN_MAX_SECONDS <= 0.040
    assert (
        video_processing.RTSP_BUFFER_DRAIN_MAX_SECONDS
        == camera_capture.RTSP_BUFFER_DRAIN_MAX_SECONDS
    )


def test_reconnect_backoff_grows_and_caps():
    delays = [camera_capture.reconnect_delay_seconds(index, "cam1") for index in range(10)]

    assert delays == sorted(delays)
    assert delays[1] > delays[0]
    assert 0 < delays[-1] <= camera_capture.RTSP_RECONNECT_MAX_SECONDS
    assert delays[-1] == delays[-2]


def test_inference_health_tracks_bounded_outcomes_and_ages():
    state.clear_camera_inference_health("cam1")

    state.record_camera_inference_outcome("cam1", "success", now_monotonic=10.0)
    state.record_camera_inference_outcome("cam1", "success", now_monotonic=11.0)
    state.record_camera_inference_outcome("cam1", "overloaded", now_monotonic=12.0)
    state.record_camera_inference_outcome("cam1", "failed", now_monotonic=13.0)

    assert state.get_camera_inference_health("cam1", now_monotonic=15.0) == {
        "successCount": 2,
        "overloadDropCount": 1,
        "failureCount": 1,
        "lastSuccessAgeSeconds": 4.0,
        "lastOverloadDropAgeSeconds": 3.0,
        "lastFailureAgeSeconds": 2.0,
    }

    state.clear_camera_inference_health("cam1")
    assert state.get_camera_inference_health("cam1", now_monotonic=15.0) == {
        "successCount": 0,
        "overloadDropCount": 0,
        "failureCount": 0,
        "lastSuccessAgeSeconds": None,
        "lastOverloadDropAgeSeconds": None,
        "lastFailureAgeSeconds": None,
    }


def test_saturated_reconnect_backoff_keeps_stable_bounded_camera_jitter():
    cam1 = camera_capture.reconnect_delay_seconds(100, "cam1")
    cam2 = camera_capture.reconnect_delay_seconds(100, "cam2")

    assert cam1 == camera_capture.reconnect_delay_seconds(1_000, "cam1")
    assert cam2 == camera_capture.reconnect_delay_seconds(1_000, "cam2")
    assert cam1 != cam2
    assert 0 < cam1 <= camera_capture.RTSP_RECONNECT_MAX_SECONDS
    assert 0 < cam2 <= camera_capture.RTSP_RECONNECT_MAX_SECONDS


def test_saturated_outage_has_bounded_hourly_open_attempt_budget(monkeypatch):
    monkeypatch.setattr(camera_capture, "RTSP_RECONNECT_BASE_SECONDS", 1.0)
    monkeypatch.setattr(camera_capture, "RTSP_RECONNECT_MAX_SECONDS", 60.0)
    open_timeout_seconds = 5.0

    def attempts_in_one_hour(camera_id: str) -> int:
        elapsed = 0.0
        failures = 0
        attempts = 0
        while elapsed < 3_600.0:
            elapsed += open_timeout_seconds
            attempts += 1
            elapsed += camera_capture.reconnect_delay_seconds(failures, camera_id)
            failures += 1
        return attempts

    attempts = [attempts_in_one_hour(camera_id) for camera_id in ("cam1", "cam2")]

    # Stable jitter changes the exact per-camera count, while the 60-second
    # long-tail ceiling keeps both well below the former ~174 attempts/hour.
    assert all(50 <= count <= 70 for count in attempts)


@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf"])
def test_env_float_rejects_non_finite_values(monkeypatch, invalid):
    monkeypatch.setenv("SAFETYLENS_TEST_FLOAT", invalid)

    assert camera_capture._env_float(
        "SAFETYLENS_TEST_FLOAT",
        30.0,
        minimum=1.0,
        maximum=300.0,
    ) == 30.0


def test_outage_tracker_requires_stable_window_before_recovery():
    tracker = camera_capture.CameraConnectionTracker(
        stable_window_seconds=30,
        summary_interval_seconds=300,
        now=0,
    )

    first = tracker.record_failure(now=1)
    assert first is not None and first.kind == "outage"
    # The first decoded frame starts, but does not complete, recovery.
    assert tracker.record_frame(now=2) is None
    assert tracker.record_failure(now=3) is None
    assert tracker.record_frame(now=4) is None
    assert tracker.record_frame(now=33.999) is None

    recovered = tracker.record_frame(now=34)

    assert recovered is not None and recovered.kind == "recovered"
    assert recovered.failure_count == 2
    assert recovered.suppressed_failure_count == 1
    assert tracker.outage_active is False
    assert tracker.outage_failure_count == 0
    assert tracker.total_failure_count == 2
    assert tracker.record_frame(now=40) is None


def test_outage_tracker_periodic_summary_has_exact_counts_and_duration():
    tracker = camera_capture.CameraConnectionTracker(
        stable_window_seconds=30,
        summary_interval_seconds=10,
        now=0,
    )

    first = tracker.record_failure(now=5)
    assert tracker.record_failure(now=9) is None
    summary = tracker.record_failure(now=15)

    assert first == camera_capture.CameraConnectionEvent(
        kind="outage",
        failure_count=1,
        total_failure_count=1,
        suppressed_failure_count=0,
        outage_duration_seconds=0.0,
    )
    assert summary == camera_capture.CameraConnectionEvent(
        kind="summary",
        failure_count=3,
        total_failure_count=3,
        suppressed_failure_count=1,
        outage_duration_seconds=10.0,
    )


def test_outage_trackers_are_isolated_per_camera():
    cam1 = camera_capture.CameraConnectionTracker(now=0)
    cam2 = camera_capture.CameraConnectionTracker(now=0)

    cam1.record_failure(now=1)
    cam1.record_failure(now=2)
    cam2.record_frame(now=2)

    assert cam1.outage_failure_count == 2
    assert cam1.suppressed_failure_count == 1
    assert cam2.outage_active is False
    assert cam2.total_failure_count == 0
    assert cam2.last_transition == "connected"


def test_repeated_one_frame_flaps_keep_reconnect_debt(monkeypatch):
    class FakeClock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += max(0.0, seconds)

    class StopAfterRetries:
        def __init__(self, clock, retries):
            self.clock = clock
            self.retries = retries
            self.delays = []
            self.stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            self.delays.append(delay)
            self.clock.now += delay
            self.stopped = len(self.delays) >= self.retries
            return self.stopped

    class OneFrameCapture:
        def __init__(self, clock):
            self.clock = clock
            self.reads = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            self.clock.now += 0.01
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            return False, None

        def release(self):
            self.released = True

    clock = FakeClock()
    stop_event = StopAfterRetries(clock, retries=3)
    captures = [OneFrameCapture(clock) for _ in range(3)]
    cfg = _rtsp_worker_config()
    _patch_rtsp_worker_dependencies(monkeypatch, cfg)
    monkeypatch.setattr(video_processing.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(video_processing.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        video_processing,
        "open_video_capture",
        lambda *_args, **_kwargs: captures.pop(0),
    )

    video_processing._video_processor_loop("cam1", stop_event)

    assert stop_event.delays == [
        camera_capture.reconnect_delay_seconds(index, "cam1")
        for index in range(3)
    ]
    health = state.get_camera_connection_health("cam1", now_monotonic=clock.now)
    assert health["outageActive"] is True
    assert health["failureCount"] == 3
    assert health["suppressedFailureCount"] == 2


def test_stable_fake_capture_resets_reconnect_debt(monkeypatch):
    class FakeClock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += max(0.0, seconds)

    class StopAfterRetries:
        def __init__(self, clock):
            self.clock = clock
            self.delays = []

        def is_set(self):
            return len(self.delays) >= 3

        def wait(self, delay):
            self.delays.append(delay)
            self.clock.now += delay
            return len(self.delays) >= 3

    class FiniteCapture:
        def __init__(self, clock, successful_frames):
            self.clock = clock
            self.successful_frames = successful_frames
            self.reads = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            self.clock.now += 0.01
            self.reads += 1
            if self.reads <= self.successful_frames:
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            return False, None

        def release(self):
            self.released = True

    clock = FakeClock()
    stop_event = StopAfterRetries(clock)
    captures = [
        FiniteCapture(clock, 1),
        FiniteCapture(clock, 1),
        FiniteCapture(clock, 32),
    ]
    cfg = _rtsp_worker_config()
    _patch_rtsp_worker_dependencies(monkeypatch, cfg)
    monkeypatch.setattr(video_processing.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(video_processing.time, "sleep", clock.sleep)
    monkeypatch.setattr(
        video_processing,
        "open_video_capture",
        lambda *_args, **_kwargs: captures.pop(0),
    )

    video_processing._video_processor_loop("cam1", stop_event)

    base = camera_capture.reconnect_delay_seconds(0, "cam1")
    assert stop_event.delays == [
        base,
        camera_capture.reconnect_delay_seconds(1, "cam1"),
        base,
    ]
    health = state.get_camera_connection_health("cam1", now_monotonic=clock.now)
    assert health["outageActive"] is True
    assert health["failureCount"] == 1
    assert health["totalFailureCount"] == 3


def test_outage_logs_only_transition_summary_and_stable_recovery(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(state, "camera_connection_health", {})
    tracker = camera_capture.CameraConnectionTracker(
        stable_window_seconds=30,
        summary_interval_seconds=10,
        now=0,
    )
    caplog.set_level(logging.INFO, logger="rakshak_lens")

    for now in (0, 1, 10):
        video_processing._record_rtsp_connection_failure(
            "cam1",
            tracker,
            safe_source="rtsp://camera/live",
            retry_seconds=8.0,
            received_frame=False,
            now=now,
        )
    assert video_processing._record_rtsp_connection_frame(
        "cam1", tracker, now=11
    ) is False
    assert video_processing._record_rtsp_connection_frame(
        "cam1", tracker, now=41
    ) is True
    assert video_processing._record_rtsp_connection_frame(
        "cam1", tracker, now=42
    ) is False

    records = [record for record in caplog.records if record.camera_id == "cam1"]
    warnings = [record for record in records if record.levelno == logging.WARNING]
    recoveries = [record for record in records if record.levelno == logging.INFO]
    assert [record.getMessage() for record in warnings] == [
        "Camera connection outage detected; retry scheduled",
        "Camera connection outage persists",
    ]
    assert warnings[1].failure_count == 3
    assert warnings[1].suppressed_failure_count == 1
    assert warnings[1].outage_duration_seconds == 10.0
    assert len(recoveries) == 1
    assert recoveries[0].failure_count == 3
    assert recoveries[0].suppressed_failure_count == 1
    assert recoveries[0].outage_duration_seconds == 41.0


def test_failed_open_observes_shutdown_before_retry_state_or_log(
    monkeypatch,
    caplog,
):
    stop_event = threading.Event()

    class FailedCapture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = FailedCapture()
    cfg = _rtsp_worker_config()
    _patch_rtsp_worker_dependencies(monkeypatch, cfg)

    def finish_shutdown_during_open(*_args, **_kwargs):
        state.camera_runtime_status["cam1"] = "stopping"
        stop_event.set()
        return capture

    monkeypatch.setattr(
        video_processing,
        "open_video_capture",
        finish_shutdown_during_open,
    )
    caplog.set_level(logging.WARNING, logger="safetylens")

    video_processing._video_processor_loop("cam1", stop_event)

    assert capture.released is True
    assert state.camera_runtime_status["cam1"] == "stopping"
    assert not [
        record
        for record in caplog.records
        if "retry" in record.getMessage().lower()
        or "outage" in record.getMessage().lower()
    ]


def test_connection_health_bounds_fields_and_rejects_arbitrary_transition_text(
    monkeypatch,
):
    monkeypatch.setattr(state, "camera_connection_health", {})
    state.update_camera_connection_health(
        "cam1",
        outage_active=True,
        outage_started_monotonic=-1_000_000_000.0,
        outage_failure_count=10**20,
        total_failure_count=10**20,
        suppressed_failure_count=10**20,
        last_transition="rtsp://user:password@camera/private",
        last_transition_monotonic=-1_000_000_000.0,
    )

    snapshot = state.get_camera_connection_health(
        "cam1",
        now_monotonic=1_000_000_000.0,
    )

    assert snapshot == {
        "outageActive": True,
        "failureCount": state.CAMERA_CONNECTION_COUNTER_MAX,
        "totalFailureCount": state.CAMERA_CONNECTION_COUNTER_MAX,
        "suppressedFailureCount": state.CAMERA_CONNECTION_COUNTER_MAX,
        "outageAgeSeconds": state.CAMERA_CONNECTION_AGE_MAX_SECONDS,
        "lastTransition": "unknown",
        "lastTransitionAgeSeconds": state.CAMERA_CONNECTION_AGE_MAX_SECONDS,
        "captureBackend": "unknown",
    }
    serialized = json.dumps(snapshot)
    assert "password" not in serialized
    assert "rtsp" not in serialized


def test_non_rtsp_worker_clears_stale_connection_health(monkeypatch):
    cfg = _rtsp_worker_config()
    cfg["cameras"]["cam1"].update(
        {
            "stream_type": "file",
            "video": "demo.mp4",
        }
    )
    stop_event = threading.Event()
    stop_event.set()
    _patch_rtsp_worker_dependencies(monkeypatch, cfg)
    state.update_camera_connection_health(
        "cam1",
        outage_active=True,
        outage_started_monotonic=0,
        outage_failure_count=4,
        total_failure_count=4,
        suppressed_failure_count=3,
        last_transition="outage",
        last_transition_monotonic=0,
    )

    video_processing._video_processor_loop("cam1", stop_event)

    assert state.get_camera_connection_health("cam1") == {
        "outageActive": False,
        "failureCount": 0,
        "totalFailureCount": 0,
        "suppressedFailureCount": 0,
        "outageAgeSeconds": None,
        "lastTransition": "unknown",
        "lastTransitionAgeSeconds": None,
        "captureBackend": "unknown",
    }


def test_health_exposes_safe_connection_outage_and_degrades(monkeypatch):
    class AliveThread:
        def is_alive(self):
            return True

    cfg = {
        "retention": {},
        "cameras": {
            "cam1": {
                "enabled": True,
                "demo": "yolo",
                "name": "Camera 1",
                "stream_type": "rtsp",
            },
        },
    }
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")
    monkeypatch.setattr(diagnostics, "_health_cache", None)
    monkeypatch.setattr(diagnostics, "get_config", lambda: cfg)
    monkeypatch.setattr(diagnostics.db, "check_connection", lambda: True)
    monkeypatch.setattr(
        diagnostics.licensing,
        "get_status",
        lambda: SimpleNamespace(
            state=diagnostics.licensing.LicenseState.VALID,
            to_public_dict=lambda: {"state": "valid"},
        ),
    )
    monkeypatch.setattr(diagnostics, "_path_usage", lambda _path: {})
    monkeypatch.setattr(
        diagnostics.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024),
    )
    monkeypatch.setattr(diagnostics.model_manager, "list_models_for_status", lambda: [])
    monkeypatch.setattr(
        diagnostics.model_manager,
        "remote_model_metadata_health",
        lambda: {
            "enabled": True,
            "status": "unavailable",
            "cache_fresh": False,
        },
    )
    monkeypatch.setattr(
        diagnostics.stream_fanout,
        "stats",
        lambda _cam_id: {
            "sequence": 0,
            "subscribers": 0,
            "has_frame": False,
            "frame_age_seconds": None,
        },
    )
    monkeypatch.setattr(diagnostics.stream_fanout, "operational_stats", lambda: {})
    monkeypatch.setattr(
        state,
        "camera_threads",
        {"cam1": (AliveThread(), threading.Event())},
    )
    monkeypatch.setattr(state, "vlm_threads", {})
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {"cam1": "reconnecting"})
    monkeypatch.setattr(state, "camera_connection_health", {})
    now = state.time.monotonic()
    state.update_camera_connection_health(
        "cam1",
        outage_active=True,
        outage_started_monotonic=now - 12,
        outage_failure_count=7,
        total_failure_count=9,
        suppressed_failure_count=6,
        last_transition="outage",
        last_transition_monotonic=now - 12,
        capture_backend="ffmpeg",
    )

    snapshot = diagnostics.build_health_snapshot()
    connection = snapshot["cameras"][0]["connection"]

    assert snapshot["status"] == "degraded"
    assert "one or more cameras have an active connection outage" in snapshot["reasons"]
    assert (
        "one or more cameras fell back from NVDEC to CPU decoding"
        in snapshot["reasons"]
    )
    assert "remote model metadata unavailable" in snapshot["reasons"]
    assert snapshot["modelMetadata"]["status"] == "unavailable"
    assert connection["outageActive"] is True
    assert connection["failureCount"] == 7
    assert connection["totalFailureCount"] == 9
    assert connection["suppressedFailureCount"] == 6
    assert 12 <= connection["outageAgeSeconds"] < 13
    assert connection["lastTransition"] == "outage"
    assert set(connection) == {
        "outageActive",
        "failureCount",
        "totalFailureCount",
        "suppressedFailureCount",
        "outageAgeSeconds",
        "lastTransition",
        "lastTransitionAgeSeconds",
        "captureBackend",
        "hardwareAccelerationExpected",
        "hardwareAccelerationActive",
        "hardwareFallback",
    }
    assert connection["captureBackend"] == "ffmpeg"
    assert connection["hardwareAccelerationExpected"] is True
    assert connection["hardwareAccelerationActive"] is False
    assert connection["hardwareFallback"] is True


def test_health_identifies_requested_nvdec_cpu_fallback(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_RTSP_CAPTURE_BACKEND", "nvdec")

    assert diagnostics._capture_acceleration_health(
        {"stream_type": "rtsp"},
        {"captureBackend": "ffmpeg"},
    ) == {
        "hardwareAccelerationExpected": True,
        "hardwareAccelerationActive": False,
        "hardwareFallback": True,
    }
    assert diagnostics._capture_acceleration_health(
        {"stream_type": "rtsp"},
        {"captureBackend": "gstreamer_nvdec"},
    ) == {
        "hardwareAccelerationExpected": True,
        "hardwareAccelerationActive": True,
        "hardwareFallback": False,
    }


def test_clear_camera_observation_discards_stale_frame_and_detection_state(monkeypatch):
    fanout = MjpegFanout()
    fanout.publish("cam1", b"annotated")
    monkeypatch.setattr(state, "camera_frames", {"cam1": b"annotated"})
    monkeypatch.setattr(state, "camera_clean_frames", {"cam1": b"clean"})
    monkeypatch.setattr(state, "camera_detections", {"cam1": [{"class": "person"}]})
    monkeypatch.setattr(video_processing, "stream_fanout", fanout)

    video_processing._clear_camera_observation("cam1")

    assert state.camera_frames["cam1"] is None
    assert state.camera_clean_frames["cam1"] is None
    assert state.camera_detections["cam1"] == []
    assert fanout.stats("cam1")["has_frame"] is False


def test_read_failure_discards_observations_at_worker_boundary(monkeypatch):
    stop_event = threading.Event()

    class ReadFailureCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            stop_event.set()
            return False, None

        def release(self):
            self.released = True

    capture = ReadFailureCapture()
    cfg = {
        "cameras": {
            "cam1": {
                "stream_type": "rtsp",
                "fps": 3,
                "execution_plan": {"required_model_keys": []},
            },
        },
        "global": {
            "target_fps": 3,
            "alert_cooldown": 30,
            "yolo_conf": 0.3,
            "jpeg_quality": 70,
            "inference_width": 960,
            "device": "0",
        },
    }
    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)
    monkeypatch.setattr(video_processing, "build_rtsp_url", lambda *_args, **_kwargs: "rtsp://camera/live")
    monkeypatch.setattr(video_processing, "open_video_capture", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(video_processing.model_manager, "missing_model_keys", lambda _keys: [])
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(state, "camera_frames", {"cam1": b"annotated"})
    monkeypatch.setattr(state, "camera_clean_frames", {"cam1": b"clean"})
    monkeypatch.setattr(state, "camera_detections", {"cam1": [{"class": "person"}]})
    monkeypatch.setattr(state, "camera_runtime_status", {})

    video_processing.video_processor("cam1", stop_event)

    assert capture.released is True
    assert state.camera_frames["cam1"] is None
    assert state.camera_clean_frames["cam1"] is None
    assert state.camera_detections["cam1"] == []


def test_unexpected_worker_exit_cannot_leave_observations_published(monkeypatch):
    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("capture failed")

    cfg = {
        "cameras": {
            "cam1": {
                "stream_type": "rtsp",
                "execution_plan": {"required_model_keys": []},
            },
        },
        "global": {
            "target_fps": 3,
            "alert_cooldown": 30,
            "yolo_conf": 0.3,
            "jpeg_quality": 70,
            "inference_width": 960,
            "device": "0",
        },
    }
    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)
    monkeypatch.setattr(video_processing, "build_rtsp_url", lambda *_args, **_kwargs: "rtsp://camera/live")
    monkeypatch.setattr(video_processing, "open_video_capture", fail_capture)
    monkeypatch.setattr(video_processing.model_manager, "missing_model_keys", lambda _keys: [])
    monkeypatch.setattr(state, "camera_frames", {"cam1": b"annotated"})
    monkeypatch.setattr(state, "camera_clean_frames", {"cam1": b"clean"})
    monkeypatch.setattr(state, "camera_detections", {"cam1": [{"class": "person"}]})

    with pytest.raises(RuntimeError, match="capture failed"):
        video_processing.video_processor("cam1", threading.Event())

    assert state.camera_frames["cam1"] is None
    assert state.camera_clean_frames["cam1"] is None
    assert state.camera_detections["cam1"] == []


def test_runtime_status_preserves_reconnecting_until_a_frame_arrives(monkeypatch):
    class AliveThread:
        def is_alive(self):
            return True

    monkeypatch.setattr(state, "camera_threads", {"cam1": (AliveThread(), threading.Event())})
    monkeypatch.setattr(state, "camera_frames", {"cam1": None})
    monkeypatch.setattr(state, "camera_runtime_status", {"cam1": "reconnecting"})

    status = camera_runtime.derive_camera_runtime_status("cam1", {"enabled": True})

    assert status == "reconnecting"


def test_runtime_status_does_not_report_a_dead_registered_worker_online(monkeypatch):
    class DeadThread:
        def is_alive(self):
            return False

    monkeypatch.setattr(state, "camera_threads", {"cam1": (DeadThread(), threading.Event())})
    monkeypatch.setattr(state, "camera_frames", {"cam1": b"stale"})
    monkeypatch.setattr(state, "camera_runtime_status", {"cam1": "running"})

    status = camera_runtime.derive_camera_runtime_status("cam1", {"enabled": True})

    assert status == "offline"


def test_runtime_status_preserves_lifecycle_model_revalidation(monkeypatch):
    class ExitingThread:
        def is_alive(self):
            return True

    monkeypatch.setattr(
        state,
        "camera_threads",
        {"cam1": (ExitingThread(), threading.Event())},
    )
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(
        state,
        "camera_runtime_status",
        {"cam1": "awaiting_model_install"},
    )

    status = camera_runtime.derive_camera_runtime_status(
        "cam1",
        {"enabled": True},
        # Simulate an older API admission snapshot that said the model was
        # ready before start_camera performed its defensive revalidation.
        missing_models=False,
    )

    assert status == "awaiting_model_install"


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
    record.total_failure_count = 9
    record.suppressed_failure_count = 3
    record.outage_duration_seconds = 42.25
    record.stable_window_seconds = 30.0
    record.received_frame = False

    payload = json.loads(JSONFormatter().format(record))

    assert payload["retry_seconds"] == 8.5
    assert payload["failure_count"] == 4
    assert payload["total_failure_count"] == 9
    assert payload["suppressed_failure_count"] == 3
    assert payload["outage_duration_seconds"] == 42.25
    assert payload["stable_window_seconds"] == 30.0
    assert payload["received_frame"] is False
    assert payload["source"] == "rtsp://192.168.1.20:554/live"
