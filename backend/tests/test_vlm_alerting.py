import threading
from concurrent.futures import Future

import numpy as np
import pytest

import state
import video_processing


@pytest.fixture(autouse=True)
def no_vlm_interval_wait(monkeypatch):
    monkeypatch.setattr(video_processing, "_vlm_interval_seconds", lambda _value: 0)


def test_vlm_structured_verdict_and_keyword_fallback_avoid_safe_false_positives():
    keywords = [
        "not wearing",
        "missing",
        "blocked",
        "obstructed",
        "hazard",
        "unsafe",
        "forklift",
        "proximity",
        "clearance",
    ]

    assert not video_processing._vlm_result_is_violation(
        "Forklift proximity and clearance are safe. No hazards; aisle is unobstructed. STATUS: SAFE",
        keywords,
    )
    assert video_processing._vlm_result_is_violation(
        "A worker is not wearing a helmet. STATUS: VIOLATION",
        keywords,
    )
    assert not video_processing._vlm_result_is_violation(
        "The aisle is not blocked and no safety hazard is visible.",
        keywords,
    )
    assert video_processing._vlm_result_is_violation(
        "The emergency aisle is blocked by stored material.",
        keywords,
    )
    assert not video_processing._vlm_result_is_violation(
        "VLM unavailable: credential-bearing internal error",
        keywords,
    )


def test_safe_vlm_analysis_updates_result_without_creating_alert(monkeypatch):
    stop_event = threading.Event()
    frame_bytes = b"encoded-frame"
    config = {
        "vlm": {
            "enabled": True,
            "interval": 0,
            "model": "test-vlm",
            "violation_keywords": ["hazard", "blocked"],
        }
    }
    created = []
    analysis_calls = 0

    def analyze(_frame):
        nonlocal analysis_calls
        analysis_calls += 1
        return "No hazards are visible and the aisle is not blocked. STATUS: SAFE"

    def get_config():
        if analysis_calls:
            stop_event.set()
        return config

    monkeypatch.setattr(video_processing, "get_config", get_config)
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(video_processing, "call_vlm", analyze)
    monkeypatch.setattr(video_processing, "create_alert", lambda **kwargs: created.append(kwargs))
    monkeypatch.setattr(
        video_processing.cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(state, "camera_frames", {"cam-1": frame_bytes})
    monkeypatch.setattr(state, "vlm_last_results", {})

    video_processing.vlm_worker("cam-1", stop_event)

    assert created == []
    assert state.vlm_last_results["cam-1"]["text"].endswith("STATUS: SAFE")


def test_violation_vlm_analysis_creates_only_p2_alert(monkeypatch):
    stop_event = threading.Event()
    config = {
        "vlm": {
            "enabled": True,
            "interval": 0,
            "model": "test-vlm",
            "violation_keywords": ["blocked"],
        }
    }
    created = []

    def create(**kwargs):
        created.append(kwargs)
        stop_event.set()
        return {"id": "alert"}

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(
        video_processing,
        "call_vlm",
        lambda _frame: "The aisle is blocked. STATUS: VIOLATION",
    )
    monkeypatch.setattr(video_processing, "create_alert", create)
    monkeypatch.setattr(
        video_processing.cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(state, "camera_frames", {"cam-1": b"encoded-frame"})
    monkeypatch.setattr(state, "vlm_last_results", {})

    video_processing.vlm_worker("cam-1", stop_event)

    assert len(created) == 1
    assert created[0]["severity"] == "P2"


def test_persistent_vlm_violation_is_submitted_once_until_safe_reset(monkeypatch):
    stop_event = threading.Event()
    config = {
        "vlm": {
            "enabled": True,
            "interval": 0,
            "model": "test-vlm",
            "violation_keywords": ["blocked"],
        }
    }
    results = iter([
        "The aisle is blocked. STATUS: VIOLATION",
        "The aisle remains blocked. STATUS: VIOLATION",
        "The aisle is clear. STATUS: SAFE",
        "The aisle is blocked again. STATUS: VIOLATION",
    ])
    created = []

    def create(**kwargs):
        created.append(kwargs)
        if len(created) == 2:
            stop_event.set()
        return {"id": f"alert-{len(created)}"}

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(video_processing, "call_vlm", lambda _frame: next(results))
    monkeypatch.setattr(video_processing, "create_alert", create)
    monkeypatch.setattr(
        video_processing.cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(state, "camera_frames", {"cam-1": b"encoded-frame"})
    monkeypatch.setattr(state, "vlm_last_results", {})

    video_processing.vlm_worker("cam-1", stop_event)

    assert len(created) == 2
    assert created[0]["description"].startswith("The aisle is blocked")
    assert created[1]["description"].startswith("The aisle is blocked again")


def test_pending_vlm_persistence_suppresses_duplicate_unsafe_submission(monkeypatch):
    stop_event = threading.Event()
    config = {
        "vlm": {
            "enabled": True,
            "interval": 0,
            "model": "test-vlm",
            "violation_keywords": ["blocked"],
        }
    }
    persistence = Future()
    analysis_calls = 0
    created = []

    def get_config():
        if analysis_calls >= 2:
            stop_event.set()
        return config

    def analyze(_frame):
        nonlocal analysis_calls
        analysis_calls += 1
        if analysis_calls == 2:
            persistence.set_result({"id": "persisted-alert"})
        return "The aisle remains blocked. STATUS: VIOLATION"

    monkeypatch.setattr(video_processing, "get_config", get_config)
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(video_processing, "call_vlm", analyze)
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: created.append(kwargs) or persistence,
    )
    monkeypatch.setattr(
        video_processing.cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(state, "camera_frames", {"cam-1": b"encoded-frame"})
    monkeypatch.setattr(state, "vlm_last_results", {})

    video_processing.vlm_worker("cam-1", stop_event)

    assert analysis_calls == 2
    assert len(created) == 1


def test_vlm_result_returning_after_stop_is_discarded(monkeypatch):
    stop_event = threading.Event()
    config = {
        "vlm": {
            "enabled": True,
            "interval": 0,
            "model": "test-vlm",
            "violation_keywords": ["blocked"],
        }
    }
    created = []

    def analyze(_frame):
        stop_event.set()
        return "The aisle is blocked. STATUS: VIOLATION"

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(video_processing, "call_vlm", analyze)
    monkeypatch.setattr(video_processing, "create_alert", lambda **kwargs: created.append(kwargs))
    monkeypatch.setattr(
        video_processing.cv2,
        "imdecode",
        lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(state, "camera_frames", {"cam-1": b"encoded-frame"})
    monkeypatch.setattr(state, "vlm_last_results", {})

    video_processing.vlm_worker("cam-1", stop_event)

    assert state.vlm_last_results == {}
    assert created == []


def test_restart_waits_for_stuck_vlm_worker_without_spawning_duplicate(monkeypatch):
    class StuckThread:
        def __init__(self):
            self.join_timeout = None

        def join(self, timeout):
            self.join_timeout = timeout

        def is_alive(self):
            return True

    thread = StuckThread()
    stop_event = threading.Event()
    starts = []
    monkeypatch.setattr(state, "camera_threads", {})
    monkeypatch.setattr(state, "vlm_threads", {"cam-1": (thread, stop_event)})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(video_processing, "start_camera", lambda camera_id: starts.append(camera_id))

    restarted = video_processing.restart_camera("cam-1")

    assert restarted is False
    assert stop_event.is_set()
    assert thread.join_timeout == 5
    assert state.vlm_threads["cam-1"][0] is thread
    assert state.camera_runtime_status["cam-1"] == "stopping"
    assert starts == []
    assert video_processing.start_vlm_for_camera("cam-1") is False
    assert state.vlm_threads["cam-1"][0] is thread


def test_restart_all_fences_orphaned_live_vlm_worker(monkeypatch):
    class StuckThread:
        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    thread = StuckThread()
    stop_event = threading.Event()
    starts = []
    monkeypatch.setattr(state, "camera_threads", {})
    monkeypatch.setattr(state, "vlm_threads", {"cam-1": (thread, stop_event)})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(video_processing, "get_config", lambda: {"cameras": {"cam-1": {}}})
    monkeypatch.setattr(video_processing, "start_camera", lambda camera_id: starts.append(camera_id))

    video_processing.restart_all_cameras()

    assert stop_event.is_set()
    assert starts == []
    assert state.vlm_threads["cam-1"][0] is thread
