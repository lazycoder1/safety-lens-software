import threading

import numpy as np
import pytest

from mjpeg_fanout import MjpegFanout
import state
import video_processing


@pytest.mark.parametrize(
    ("second_result", "expected_violation_checks", "expected_alert_indices"),
    [
        ("positive", [1, 20], [20]),
        ("empty", [1], []),
        ("error", [1], []),
    ],
)
def test_cached_detection_does_not_advance_alert_window_between_inferences(
    monkeypatch,
    second_result,
    expected_violation_checks,
    expected_alert_indices,
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
    next_slots = iter((0.0, 1.0, float("inf")))

    def run_inference(_camera_id, frame, *_args, **_kwargs):
        inference_calls.append(capture.read_count)
        if len(inference_calls) == 2:
            if second_result == "error":
                raise RuntimeError("synthetic inference failure")
            if second_result == "empty":
                return frame.copy(), []
        return frame.copy(), [detection]

    def check_violations(_detections, _camera_id):
        violation_checks.append(capture.read_count)
        return [candidate]

    def create_alert(**_kwargs):
        alert_capture_indices.append(capture.read_count)
        return {"id": "alert-1"}

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing, "open_video_capture", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(video_processing.model_manager, "missing_model_keys", lambda _keys: [])
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
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
    monkeypatch.setattr(video_processing, "_run_grouped_inference", run_inference)
    monkeypatch.setattr(video_processing, "check_violations", check_violations)
    monkeypatch.setattr(video_processing, "check_zone_intrusions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(video_processing, "create_alert", create_alert)
    monkeypatch.setattr(video_processing, "_publish_stream_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_processing, "stream_fanout", MjpegFanout())
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(video_processing, "_last_pose_results", {})

    video_processing.video_processor("cam1", stop_event)

    assert capture.released is True
    assert inference_calls == [1, 20]
    assert violation_checks == expected_violation_checks
    assert alert_capture_indices == expected_alert_indices


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
    next_slots = iter((0.0, 5.0, float("inf")))
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
        video_processing._last_pose_results["cam1"] = object()
        return frame.copy(), []

    def check_fall(_result, _camera_id, _frame):
        pose_checks.append(capture.read_count)
        return [fall_candidate]

    def create_alert(**_kwargs):
        alert_capture_indices.append(capture.read_count)
        return {"id": "fall-alert"}

    monkeypatch.setattr(video_processing, "get_config", lambda: config)
    monkeypatch.setattr(video_processing, "open_video_capture", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(video_processing.model_manager, "missing_model_keys", lambda _keys: [])
    monkeypatch.setattr(video_processing.licensing, "is_inference_allowed", lambda: True)
    monkeypatch.setattr(
        video_processing.inference_scheduler,
        "next_inference_slot",
        lambda *_args, **_kwargs: next(next_slots),
    )
    monkeypatch.setattr(video_processing.time, "monotonic", lambda: float(capture.read_count))
    monkeypatch.setattr(video_processing.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(video_processing, "_run_grouped_inference", run_inference)
    monkeypatch.setattr(video_processing, "check_fall_detections", check_fall)
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
    monkeypatch.setattr(video_processing, "create_alert", create_alert)
    monkeypatch.setattr(video_processing, "_publish_stream_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_processing, "stream_fanout", MjpegFanout())
    monkeypatch.setattr(state, "camera_frames", {})
    monkeypatch.setattr(state, "camera_clean_frames", {})
    monkeypatch.setattr(state, "camera_detections", {})
    monkeypatch.setattr(state, "camera_runtime_status", {})
    monkeypatch.setattr(video_processing, "_last_pose_results", {})

    video_processing.video_processor("cam1", stop_event)

    assert capture.released is True
    assert pose_checks == [1, 5]
    assert alert_capture_indices == [5]
