import threading
import time
import json
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


def test_call_vlm_never_follows_redirect_to_on_device_endpoint(monkeypatch):
    calls = []

    class RedirectResponse:
        status_code = 302
        headers = {"Location": "http://127.0.0.1:11434/api/generate"}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return RedirectResponse()

    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {
            "vlm": {
                "url": "https://remote.example.test/generate",
                "remote_only": True,
                "model": "test-vlm",
                "prompt": "inspect",
                "temperature": 0,
                "max_tokens": 8,
            }
        },
    )
    monkeypatch.setattr(
        video_processing, "remote_vlm_endpoint_allowed", lambda _url: True
    )
    monkeypatch.setattr(video_processing.requests, "post", post)
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (True, np.frombuffer(b"jpeg", dtype=np.uint8)),
    )

    result = video_processing.call_vlm(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result == "VLM error: 302"
    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["stream"] is True


def test_call_vlm_rejects_oversized_chunked_response_without_state_mutation(
    monkeypatch,
):
    closed = []

    class OversizedResponse:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size):
            assert chunk_size <= video_processing.VLM_MAX_RESPONSE_BYTES
            yield b"{" + b"x" * (video_processing.VLM_MAX_RESPONSE_BYTES // 2)
            yield b"x" * (video_processing.VLM_MAX_RESPONSE_BYTES // 2 + 1)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {
            "vlm": {
                "url": "http://192.0.2.20:11434/api/generate",
                "remote_only": True,
                "model": "test-vlm",
                "prompt": "inspect",
                "temperature": 0,
                "max_tokens": 8,
            }
        },
    )
    monkeypatch.setattr(
        video_processing, "remote_vlm_endpoint_allowed", lambda _url: True
    )
    monkeypatch.setattr(
        video_processing.requests,
        "post",
        lambda *_args, **_kwargs: OversizedResponse(),
    )
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (True, np.frombuffer(b"jpeg", dtype=np.uint8)),
    )
    original_results = {"cam-1": {"text": "existing"}}
    monkeypatch.setattr(state, "vlm_last_results", original_results.copy())

    result = video_processing.call_vlm(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result == "VLM unavailable"
    assert closed == [True]
    assert state.vlm_last_results == original_results


def test_call_vlm_caps_successful_response_text_and_closes_stream(monkeypatch):
    closed = []
    response_text = "safe " * (video_processing.VLM_MAX_RESULT_CHARS + 1)
    encoded = json.dumps({"response": response_text}).encode()

    class SuccessfulResponse:
        status_code = 200
        headers = {"Content-Length": str(len(encoded))}

        def iter_content(self, chunk_size):
            for offset in range(0, len(encoded), chunk_size):
                yield encoded[offset : offset + chunk_size]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: {
            "vlm": {
                "url": "http://192.0.2.20:11434/api/generate",
                "remote_only": True,
                "model": "test-vlm",
                "prompt": "inspect",
                "temperature": 0,
                "max_tokens": 8,
            }
        },
    )
    monkeypatch.setattr(
        video_processing, "remote_vlm_endpoint_allowed", lambda _url: True
    )
    monkeypatch.setattr(
        video_processing.requests,
        "post",
        lambda *_args, **_kwargs: SuccessfulResponse(),
    )
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (True, np.frombuffer(b"jpeg", dtype=np.uint8)),
    )

    result = video_processing.call_vlm(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result == response_text[: video_processing.VLM_MAX_RESULT_CHARS]
    assert closed == [True]


def test_stale_dispatcher_generation_cannot_mutate_result_or_create_alert(
    monkeypatch,
):
    created = []
    dispatcher = video_processing.VLMEnrichmentDispatcher(
        process=lambda work: work.payload,
        on_result=lambda *_args: None,
    )
    dispatcher.register_camera("cam-1", "old-generation")
    dispatcher.discard_camera("cam-1", "old-generation")
    dispatcher.register_camera("cam-1", "new-generation")
    monkeypatch.setattr(video_processing, "_vlm_dispatcher", dispatcher)
    monkeypatch.setattr(state, "vlm_last_results", {})
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: created.append(kwargs),
    )

    video_processing._handle_vlm_enrichment_result(
        video_processing.VLMEnrichmentWork(
            camera_id="cam-1",
            generation="old-generation",
            payload={"frame_bytes": b"jpeg"},
            enqueued_monotonic=0.0,
            sequence=1,
        ),
        "The aisle is blocked. STATUS: VIOLATION",
        1.0,
    )

    assert state.vlm_last_results == {}
    assert created == []
    dispatcher.shutdown(wait=True, timeout=1)


def test_full_alert_queue_cannot_block_vlm_result_or_camera_discard(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    persisted = []

    def persist(**payload):
        if payload.get("sequence") == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        persisted.append(payload)
        return {"id": str(payload.get("sequence", "alert")), **payload}

    pipeline = video_processing.AlertPipeline(
        persist_alert=persist,
        deliver_alert=lambda _alert, _outputs: True,
        persist_queue_size=1,
        submit_timeout=1.0,
    )
    first = pipeline.submit({"sequence": 1})
    assert first_started.wait(timeout=1)
    second = pipeline.submit({"sequence": 2})

    config = {
        "cameras": {
            "cam-1": {
                "enabled": True,
                "name": "Camera 1",
                "zone": "Test",
            }
        },
        "vlm": {
            "enabled": True,
            "alerting_enabled": True,
            "model": "remote-vlm",
            "violation_keywords": ["blocked"],
        },
    }
    dispatcher = video_processing.VLMEnrichmentDispatcher(
        process=lambda work: work.payload,
        on_result=lambda *_args: None,
    )
    dispatcher.register_camera("cam-1", "generation-1")
    monkeypatch.setattr(video_processing, "_vlm_dispatcher", dispatcher)
    monkeypatch.setattr(video_processing, "_alert_pipeline", pipeline)
    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing, "get_config_snapshot", lambda: config)
    monkeypatch.setattr(
        video_processing.notification_dispatcher,
        "resolve_delivery_targets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(state, "vlm_last_results", {})
    monkeypatch.setattr(video_processing, "_vlm_incident_active", {})

    started = time.perf_counter()
    video_processing._handle_vlm_enrichment_result(
        video_processing.VLMEnrichmentWork(
            camera_id="cam-1",
            generation="generation-1",
            payload={"frame_bytes": b"jpeg"},
            enqueued_monotonic=0.0,
            sequence=1,
        ),
        "The aisle is blocked. STATUS: VIOLATION",
        1.0,
    )
    assert time.perf_counter() - started < 0.1

    discard_started = time.perf_counter()
    dispatcher.discard_camera("cam-1", "generation-1")
    assert time.perf_counter() - discard_started < 0.05
    assert video_processing._vlm_incident_active == {}

    release_first.set()
    assert first.result(timeout=1)["sequence"] == 1
    assert second.result(timeout=1)["sequence"] == 2
    assert pipeline.drain(timeout=1)
    assert all(item.get("rule") != "VLM Scene Analysis" for item in persisted)
    assert pipeline.shutdown(timeout=1)
    dispatcher.shutdown(wait=True, timeout=1)


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
