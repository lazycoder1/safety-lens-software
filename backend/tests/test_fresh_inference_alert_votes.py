import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mjpeg_fanout import MjpegFanout
import state
import video_processing


class _ImmediateFuture:
    def __init__(self, function, *args, **kwargs):
        try:
            self._result = function(*args, **kwargs)
            self._error = None
        except Exception as exc:
            self._result = None
            self._error = exc

    def done(self):
        return True

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        return False


class _ImmediateExecutor:
    def __init__(self, **_kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        return _ImmediateFuture(function, *args, **kwargs)

    def shutdown(self, **_kwargs):
        pass


@pytest.mark.parametrize(
    (
        "second_result",
        "expected_violation_checks",
        "expected_alert_indices",
        "expected_preserved_detections",
        "expected_history_length",
    ),
    [
        ("positive", [2, 21], [21], 1, 2),
        ("empty", [2], [], 0, 2),
        ("error", [2], [], 1, 1),
        ("overloaded", [2], [], 1, 1),
    ],
)
def test_cached_detection_does_not_advance_alert_window_between_inferences(
    monkeypatch,
    second_result,
    expected_violation_checks,
    expected_alert_indices,
    expected_preserved_detections,
    expected_history_length,
):
    stop_event = threading.Event()

    class FixedCapture:
        def __init__(self):
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            self.read_count += 1
            if self.read_count <= 21:
                return True, np.zeros((90, 160, 3), dtype=np.uint8)
            stop_event.set()
            return False, None

        def release(self):
            self.released = True

    capture = FixedCapture()
    execution_plan = {
        "required_model_keys": [],
        "run_face_recognition": False,
        "run_pose_specialist": False,
        "run_ppe_specialist": False,
    }
    config = {
        "cameras": {
            "cam1": {
                "name": "Test Camera",
                "video": "test.avi",
                "stream_type": "file",
                "fps": 20,
                "inference_fps": 1,
                "execution_plan": execution_plan,
            }
        },
        "global": {
            "target_fps": 20,
            "inference_fps": 1,
            "stream_fps": 10,
            "alert_cooldown": 30,
            "yolo_conf": 0.3,
            "jpeg_quality": 70,
            "inference_width": 160,
            "device": "cpu",
        },
    }
    detection = {"class": "person", "confidence": 0.9, "bbox": [1, 2, 30, 40]}
    candidate = {
        "camera_id": "cam1",
        "rule": "Test Violation",
        "severity": "P2",
        "confidence": 0.9,
        "description": "Synthetic violation",
        "source": "test",
        "threshold": 2,
    }
    inference_calls = []
    violation_checks = []
    alert_capture_indices = []
    detections_before_clear = []
    signature_calls = []
    next_slots = iter((0.0, 0.0, 1.0, float("inf")))
    original_clear_observation = video_processing._clear_camera_observation
    original_frame_change_signature = video_processing._frame_change_signature

    def run_inference(_camera_id, frame, *_args, **_kwargs):
        inference_calls.append(capture.read_count)
        if len(inference_calls) == 2:
            if second_result == "overloaded":
                raise video_processing.model_manager.RemoteInferenceOverloadedError(
                    "synthetic overload"
                )
            if second_result == "error":
                raise RuntimeError("synthetic inference failure")
            if second_result == "empty":
                return frame.copy(), [], None, {}
        return frame.copy(), [detection], None, {}

    def check_violations(_detections, _camera_id):
        violation_checks.append(capture.read_count)
        return [candidate]

    def create_alert(**_kwargs):
        alert_capture_indices.append(capture.read_count)
        return {"id": "alert-1"}

    def record_then_clear(camera_id):
        detections_before_clear.append(list(state.camera_detections.get(camera_id, [])))
        original_clear_observation(camera_id)

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(
        video_processing, "open_video_capture", lambda *_args, **_kwargs: capture
    )
    monkeypatch.setattr(
        video_processing.model_manager, "missing_model_keys", lambda _keys: []
    )
    monkeypatch.setattr(
        video_processing.licensing, "is_inference_allowed", lambda: True
    )
    monkeypatch.setattr(
        video_processing.inference_scheduler,
        "next_inference_slot",
        lambda *_args, **_kwargs: next(next_slots),
    )
    monkeypatch.setattr(
        video_processing.time,
        "monotonic",
        lambda: 0.0 if capture.read_count < 20 else 1.0,
    )
    monkeypatch.setattr(video_processing.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(video_processing, "ThreadPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(
        video_processing,
        "_frame_change_signature",
        lambda frame: signature_calls.append(capture.read_count)
        or original_frame_change_signature(frame),
    )
    monkeypatch.setattr(video_processing, "_run_grouped_inference", run_inference)
    monkeypatch.setattr(video_processing, "check_violations", check_violations)
    monkeypatch.setattr(
        video_processing, "check_zone_intrusions", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        video_processing, "extract_violation_bboxes", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(video_processing, "create_alert", create_alert)
    monkeypatch.setattr(
        video_processing, "_publish_stream_frame", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        video_processing, "_clear_camera_observation", record_then_clear
    )
    monkeypatch.setattr(video_processing, "stream_fanout", MjpegFanout())
    monkeypatch.setattr(
        video_processing.policy_engine,
        "evaluate_candidate",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                fallback=True,
                output_ids=None,
                rule_id="",
                rule_name="Fallback",
                severity="P2",
                priority=2,
                message=None,
                cooldown_seconds=30,
            )
        ],
    )
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_detection_history", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})

    video_processing.video_processor("cam1", stop_event)

    assert capture.released is True
    assert inference_calls == [1, 20]
    assert violation_checks == expected_violation_checks
    assert alert_capture_indices == expected_alert_indices
    assert len(signature_calls) <= len(inference_calls)
    assert len(detections_before_clear[0]) == expected_preserved_detections
    assert len(state.camera_detection_history["cam1"]) == expected_history_length


def test_cached_pose_result_is_evaluated_once_per_inference(monkeypatch):
    stop_event = threading.Event()

    class FixedCapture:
        def __init__(self):
            self.read_count = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            self.read_count += 1
            if self.read_count <= 6:
                return True, np.zeros((90, 160, 3), dtype=np.uint8)
            stop_event.set()
            return False, None

        def release(self):
            self.released = True

    capture = FixedCapture()
    execution_plan = {
        "required_model_keys": [],
        "run_face_recognition": False,
        "run_pose_specialist": True,
        "run_ppe_specialist": False,
    }
    config = {
        "cameras": {
            "cam1": {
                "video": "test.avi",
                "stream_type": "file",
                "fps": 1000,
                "inference_fps": 10,
                "execution_plan": execution_plan,
            }
        },
        "global": {
            "target_fps": 1000,
            "inference_fps": 10,
            "stream_fps": 10,
            "alert_cooldown": 30,
            "yolo_conf": 0.3,
            "jpeg_quality": 70,
            "inference_width": 160,
            "device": "cpu",
        },
    }
    pose_checks = []
    alert_capture_indices = []
    next_slots = iter((0.0, 1.0, 5.0, float("inf")))
    fall_candidate = {
        "camera_id": "cam1",
        "rule": "Person Fall",
        "severity": "P1",
        "confidence": 0.95,
        "description": "Synthetic fall",
        "source": "pose",
        "threshold": 2,
    }

    def run_inference(_camera_id, frame, *_args, **_kwargs):
        return frame.copy(), [], object(), {}

    def check_fall(_result, _camera_id, _frame):
        pose_checks.append(capture.read_count)
        return [fall_candidate]

    def create_alert(**_kwargs):
        alert_capture_indices.append(capture.read_count)
        return {"id": "fall-alert"}

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(
        video_processing, "open_video_capture", lambda *_args, **_kwargs: capture
    )
    monkeypatch.setattr(
        video_processing.model_manager, "missing_model_keys", lambda _keys: []
    )
    monkeypatch.setattr(
        video_processing.licensing, "is_inference_allowed", lambda: True
    )
    monkeypatch.setattr(
        video_processing.inference_scheduler,
        "next_inference_slot",
        lambda *_args, **_kwargs: next(next_slots),
    )
    monkeypatch.setattr(
        video_processing.time, "monotonic", lambda: float(capture.read_count)
    )
    monkeypatch.setattr(video_processing.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(video_processing, "ThreadPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(video_processing, "_run_grouped_inference", run_inference)
    monkeypatch.setattr(video_processing, "check_fall_detections", check_fall)
    monkeypatch.setattr(
        video_processing, "extract_violation_bboxes", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(video_processing, "create_alert", create_alert)
    monkeypatch.setattr(
        video_processing, "_publish_stream_frame", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(video_processing, "stream_fanout", MjpegFanout())
    monkeypatch.setattr(
        video_processing.policy_engine,
        "evaluate_candidate",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                fallback=True,
                output_ids=None,
                rule_id="",
                rule_name="Fallback",
                severity="P1",
                priority=1,
                message=None,
                cooldown_seconds=30,
            )
        ],
    )
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})

    video_processing.video_processor("cam1", stop_event)

    assert capture.released is True
    assert pose_checks == [2, 6]
    assert alert_capture_indices == [6]
