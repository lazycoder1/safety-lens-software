"""Tests for video processing runtime helpers."""

from datetime import datetime
from concurrent.futures import Future
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import camera_planner
import state
import video_processing


def test_policy_timestamp_waits_for_successful_alert_persistence(monkeypatch):
    calls = []
    submission = Future()
    monkeypatch.setattr(
        video_processing.policy_engine,
        "mark_rule_triggered",
        lambda rule_id: calls.append(rule_id),
    )

    video_processing._persist_policy_trigger_after_alert(submission, "rule-1")
    assert calls == []

    submission.set_result({"id": "alert-1"})
    assert calls == ["rule-1"]


def test_policy_timestamp_is_not_written_after_failed_persistence(monkeypatch):
    calls = []
    submission = Future()
    monkeypatch.setattr(
        video_processing.policy_engine,
        "mark_rule_triggered",
        lambda rule_id: calls.append(rule_id),
    )

    video_processing._persist_policy_trigger_after_alert(submission, "rule-1")
    submission.set_exception(RuntimeError("persistence failed"))

    assert calls == []


def _face_event(event_type="face_unknown", *, matched_face_id=None):
    return {
        "eventType": event_type,
        "confidence": 91.0 if matched_face_id else None,
        "matchedFaceId": matched_face_id,
        "personName": "Worker" if matched_face_id else None,
        "personGroup": "employees" if matched_face_id else None,
        "bbox": {"x1": 10, "y1": 20, "x2": 70, "y2": 90},
        "qualityReason": None,
    }


def test_face_recognition_skips_snapshot_encoding_without_events(monkeypatch):
    monkeypatch.setattr(video_processing.face_analyzer, "analyze_frame", lambda _frame: [])
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty face analysis encoded a snapshot")
        ),
    )

    annotated, detections = video_processing._run_face_recognition(
        "cam1",
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.zeros((120, 200, 3), dtype=np.uint8),
        {"name": "Gate"},
        {},
    )

    assert annotated.shape == (120, 200, 3)
    assert detections == []


def test_face_recognition_encodes_one_snapshot_for_multiple_loggable_events(monkeypatch):
    events = [
        _face_event("face_match", matched_face_id="face-1"),
        _face_event("face_unknown"),
    ]
    encode_calls = []
    logged = []
    monkeypatch.setattr(video_processing.face_analyzer, "analyze_frame", lambda _frame: events)

    def fake_encode(extension, frame, options):
        encode_calls.append((extension, frame.shape, options))
        return True, np.frombuffer(b"snapshot", dtype=np.uint8)

    monkeypatch.setattr(video_processing.cv2, "imencode", fake_encode)
    monkeypatch.setattr(
        video_processing.face_store,
        "log_face_event",
        lambda **kwargs: logged.append(kwargs),
    )

    _annotated, detections = video_processing._run_face_recognition(
        "cam1",
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.zeros((120, 200, 3), dtype=np.uint8),
        {"name": "Gate"},
        {},
    )

    assert len(detections) == 2
    assert len(encode_calls) == 1
    assert len(logged) == 2
    assert all(item["snapshot_jpeg"] == b"snapshot" for item in logged)


def test_face_recognition_skips_snapshot_encoding_during_log_cooldown(monkeypatch):
    now = 1000.0
    event = _face_event("face_unknown")
    monkeypatch.setattr(video_processing.face_analyzer, "analyze_frame", lambda _frame: [event])
    monkeypatch.setattr(video_processing.time, "time", lambda: now)
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cooldown face event encoded a snapshot")
        ),
    )
    monkeypatch.setattr(
        video_processing.face_store,
        "log_face_event",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cooldown face event was persisted")
        ),
    )

    _annotated, detections = video_processing._run_face_recognition(
        "cam1",
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.zeros((120, 200, 3), dtype=np.uint8),
        {"name": "Gate"},
        {"cam1:face_unknown:face_unknown": now - 1},
    )

    assert len(detections) == 1


def test_plate_recognition_skips_snapshot_encoding_without_candidates(monkeypatch):
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_plate_records",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        video_processing.cv2,
        "imencode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty plate analysis encoded a snapshot")
        ),
    )

    annotated, detections = video_processing._run_plate_recognition(
        "cam1",
        np.zeros((120, 200, 3), dtype=np.uint8),
        np.zeros((120, 200, 3), dtype=np.uint8),
        {"name": "Gate"},
        {},
        [],
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    assert annotated.shape == (120, 200, 3)
    assert detections == []


def test_plate_recognition_reuses_one_snapshot_for_multiple_reads(monkeypatch):
    candidates = [
        {
            "plateText": plate,
            "normalizedPlate": plate,
            "bbox": bbox,
            "confidence": 0.9,
            "qualityReason": "OCR confidence is low",
        }
        for plate, bbox in (
            ("KA01AA1111", {"x1": 10, "y1": 20, "x2": 70, "y2": 60}),
            ("KA01AA2222", {"x1": 80, "y1": 30, "x2": 150, "y2": 75}),
        )
    ]
    encoded_shapes = []
    logged = []
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_plate_records",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(
        video_processing.plate_store,
        "normalize_plate_text",
        lambda value: value,
    )
    monkeypatch.setattr(
        video_processing.plate_store,
        "log_plate_read",
        lambda **kwargs: logged.append(kwargs),
    )

    def fake_encode(_extension, image, _options):
        encoded_shapes.append(image.shape)
        return True, np.frombuffer(b"jpeg", dtype=np.uint8)

    monkeypatch.setattr(video_processing.cv2, "imencode", fake_encode)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)

    _annotated, detections = video_processing._run_plate_recognition(
        "cam1",
        frame,
        frame.copy(),
        {"name": "Gate"},
        {},
        [],
        conf=0.25,
        device="cuda",
        imgsz=960,
    )

    assert len(detections) == 2
    assert encoded_shapes.count(frame.shape) == 1
    assert len(encoded_shapes) == 3
    assert len(logged) == 2
    assert all(item["snapshot_jpeg"] == b"jpeg" for item in logged)


def test_executor_shutdown_error_classifier_is_narrow():
    assert video_processing._is_executor_shutdown_error(
        RuntimeError("cannot schedule new futures after shutdown")
    )
    assert video_processing._is_executor_shutdown_error(
        RuntimeError("cannot schedule new futures after interpreter shutdown")
    )
    assert not video_processing._is_executor_shutdown_error(RuntimeError("camera inference submit failed"))


def test_drain_inference_executor_waits_for_uncancellable_work():
    calls = []

    class PendingInference:
        def cancel(self):
            calls.append(("cancel",))
            return False

    class InferenceExecutor:
        def shutdown(self, **kwargs):
            calls.append(("shutdown", kwargs))

    video_processing._drain_inference_executor(InferenceExecutor(), PendingInference())

    assert calls == [
        ("cancel",),
        ("shutdown", {"wait": True, "cancel_futures": True}),
    ]


class _FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.grabbed = None
        self.grab_calls = 0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def grab(self):
        self.grab_calls += 1
        if not self.frames:
            return False
        self.grabbed = self.frames.pop(0)
        return True

    def retrieve(self):
        if self.grabbed is None:
            return False, None
        return True, self.grabbed


def test_read_live_frame_drains_rtsp_buffer_to_latest_frame(monkeypatch):
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(4)]
    cap = _FakeCapture(frames)
    monkeypatch.setattr(video_processing, "RTSP_BUFFER_DRAIN_MAX_FRAMES", 10)
    monkeypatch.setattr(video_processing, "RTSP_BUFFER_DRAIN_MAX_SECONDS", 1.0)
    monkeypatch.setattr(video_processing, "RTSP_BUFFER_DRAIN_BLOCK_SECONDS", 1.0)

    ok, frame = video_processing._read_live_frame(cap, "rtsp")

    assert ok
    assert frame is not None
    assert int(frame[0, 0, 0]) == 3
    assert cap.grab_calls >= 3


def test_read_live_frame_does_not_drain_file_sources():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(3)]
    cap = _FakeCapture(frames)

    ok, frame = video_processing._read_live_frame(cap, "file")

    assert ok
    assert frame is not None
    assert int(frame[0, 0, 0]) == 0
    assert cap.grab_calls == 0


def test_read_live_frame_does_not_redrain_latest_frame_capture():
    frame = np.full((2, 2, 3), 7, dtype=np.uint8)
    cap = _FakeCapture([frame])
    cap.delivers_latest_frame = True

    ok, result = video_processing._read_live_frame(cap, "rtsp")

    assert ok
    assert result is frame
    assert cap.grab_calls == 0


def test_ppe_model_confidence_uses_camera_rule_override(monkeypatch):
    cfg = {
        "cameras": {
            "cam_apron": {
                "safety_rule_ids": ["ppe_apron"],
                "safety_rule_overrides": {
                    "ppe_apron": {"confidence": 0.10},
                },
            }
        },
        "safety_rules": [
            {
                "id": "ppe_apron",
                "type": "ppe",
                "enabled": True,
                "confidence": 0.35,
            }
        ],
    }
    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)

    assert video_processing._rule_confidence_for_model_family("cam_apron", "ppe_specialist", 0.35) == 0.10


def test_ppe_model_confidence_falls_back_to_rule_confidence(monkeypatch):
    cfg = {
        "cameras": {
            "cam_harness": {
                "safety_rule_ids": ["ppe_harness"],
            }
        },
        "safety_rules": [
            {
                "id": "ppe_harness",
                "type": "ppe",
                "enabled": True,
                "confidence": 0.15,
            }
        ],
    }
    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)

    assert video_processing._rule_confidence_for_model_family("cam_harness", "ppe_specialist", 0.35) == 0.15


def _apron_plan():
    return {
        "capabilities": ["apron_required"],
        "required_model_keys": ["coco_primary", "ppe_specialist"],
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_face_recognition": False,
        "run_pose_specialist": False,
        "run_plate_recognition": False,
        "ppe_prompt_terms": ["apron", "denim apron", "work apron"],
        "yoloe_prompt_terms": [],
        "capability_windows": [
            {
                "id": "apron_monday_shift",
                "capabilities": ["apron_required"],
                "mode": "detection",
                "windows": [{"days": ["mon"], "from": "09:00", "to": "17:00"}],
            }
        ],
    }


def _apron_candidate_plan():
    return {
        "capabilities": ["apron_required"],
        "required_model_keys": ["ppe_closed_set_candidate"],
        "capability_model_overrides": {"apron_required": "ppe_closed_set_candidate"},
        "run_coco_primary": False,
        "run_ppe_specialist": False,
        "run_ppe_closed_set_candidate": True,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_face_recognition": False,
        "run_pose_specialist": False,
        "run_plate_recognition": False,
        "ppe_prompt_terms": [],
        "yoloe_prompt_terms": [],
        "capability_windows": [
            {
                "id": "apron_candidate_monday_shift",
                "capabilities": ["apron_required"],
                "mode": "detection",
                "windows": [{"days": ["mon"], "from": "09:00", "to": "17:00"}],
            }
        ],
    }


def test_detection_window_keeps_capability_active_inside_window():
    cfg = {"site": {"timezone": "Asia/Kolkata"}}
    schedule = video_processing._capability_schedule_state(
        {},
        cfg,
        _apron_plan(),
        now=datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    scheduled_plan = video_processing._scheduled_execution_plan(_apron_plan(), schedule)

    assert schedule["suppressedCapabilities"] == []
    assert scheduled_plan["run_ppe_specialist"] is True
    assert scheduled_plan["run_coco_primary"] is True
    assert "denim apron" in scheduled_plan["ppe_prompt_terms"]


def test_detection_window_suppresses_model_paths_outside_window():
    cfg = {"site": {"timezone": "Asia/Kolkata"}}
    schedule = video_processing._capability_schedule_state(
        {},
        cfg,
        _apron_plan(),
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    scheduled_plan = video_processing._scheduled_execution_plan(_apron_plan(), schedule)

    assert schedule["suppressedCapabilities"] == ["apron_required"]
    assert scheduled_plan["capabilities"] == []
    assert scheduled_plan["required_model_keys"] == []
    assert scheduled_plan["run_ppe_specialist"] is False
    assert scheduled_plan["run_coco_primary"] is False
    assert scheduled_plan["ppe_prompt_terms"] == []


def test_detection_window_preserves_closed_set_candidate_override_inside_window():
    cfg = {"site": {"timezone": "Asia/Kolkata"}}
    schedule = video_processing._capability_schedule_state(
        {},
        cfg,
        _apron_candidate_plan(),
        now=datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    scheduled_plan = video_processing._scheduled_execution_plan(_apron_candidate_plan(), schedule)

    assert schedule["suppressedCapabilities"] == []
    assert scheduled_plan["required_model_keys"] == ["ppe_closed_set_candidate"]
    assert scheduled_plan["run_ppe_closed_set_candidate"] is True
    assert scheduled_plan["run_ppe_specialist"] is False
    assert scheduled_plan["run_coco_primary"] is False


def test_detection_window_suppresses_closed_set_candidate_outside_window():
    cfg = {"site": {"timezone": "Asia/Kolkata"}}
    schedule = video_processing._capability_schedule_state(
        {},
        cfg,
        _apron_candidate_plan(),
        now=datetime(2026, 6, 17, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    scheduled_plan = video_processing._scheduled_execution_plan(_apron_candidate_plan(), schedule)

    assert schedule["suppressedCapabilities"] == ["apron_required"]
    assert scheduled_plan["required_model_keys"] == []
    assert scheduled_plan["run_ppe_closed_set_candidate"] is False
    assert scheduled_plan["run_ppe_specialist"] is False
    assert scheduled_plan["run_coco_primary"] is False


def test_crowd_count_detector_window_suppresses_coco_only_model_path():
    cfg = {"site": {"timezone": "Asia/Kolkata"}, "safety_rules": []}
    camera = {
        "id": "eval_office_crowd_count",
        "capabilities": ["crowd_count_threshold"],
        "safety_rule_ids": [],
        "capability_windows": {
            "crowd_count_threshold": {
                "id": "office_crowd_count_inactive_window",
                "mode": "detection",
                "active": False,
                "windows": [{"days": ["mon"], "from": "09:00", "to": "17:00"}],
            }
        },
    }

    plan = camera_planner.build_execution_plan(camera, cfg)
    schedule = video_processing._capability_schedule_state(
        camera,
        cfg,
        plan,
        now=datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    scheduled_plan = video_processing._scheduled_execution_plan(plan, schedule)

    assert plan["capabilities"] == ["crowd_count_threshold"]
    assert plan["required_model_keys"] == ["coco_primary"]
    assert plan["run_coco_primary"] is True
    assert plan["capability_windows"][0]["active"] is False
    assert schedule["capabilities"]["crowd_count_threshold"]["active"] is False
    assert schedule["suppressedCapabilities"] == ["crowd_count_threshold"]
    assert scheduled_plan["capabilities"] == []
    assert scheduled_plan["required_model_keys"] == []
    assert scheduled_plan["run_coco_primary"] is False


def test_crowd_count_threshold_candidate_is_person_count_policy_only():
    candidates = video_processing._crowd_count_threshold_candidates(
        "eval_office_crowd_count",
        [
            {"class": "person", "confidence": 0.62},
            {"class": "person", "confidence": 0.54},
            {"class": "chair", "confidence": 0.81},
        ],
    )

    assert candidates == [
        {
            "camera_id": "eval_office_crowd_count",
            "rule": "Person Detected",
            "severity": "P3",
            "confidence": 0.62,
            "count": 2,
            "classes": ["person"],
            "description": "2 person detected detection(s)",
            "source": "COCO Primary",
            "threshold": 1,
            "metadata": {"capability": "crowd_count_threshold"},
        }
    ]


def test_detection_history_records_schedule_suppression_telemetry():
    state.camera_detection_history.clear()
    state.camera_schedule_telemetry.clear()
    schedule = {
        "suppressedCapabilities": ["apron_required"],
        "suppressedCount": 1,
        "capabilities": {"apron_required": {"active": False, "suppressed": True}},
    }
    invocations = {"coco_primary": 0, "ppe_specialist": 0}

    video_processing._record_detection_history(
        "cam_apron",
        [],
        schedule_state=schedule,
        model_invocations=invocations,
    )

    sample = state.camera_detection_history["cam_apron"][-1]
    assert sample["detectionsCount"] == 0
    assert sample["scheduleState"]["suppressedCapabilities"] == ["apron_required"]
    assert sample["modelInvocationCounts"] == invocations
    assert state.camera_schedule_telemetry["cam_apron"]["scheduleState"] == schedule


def test_detection_history_records_high_resolution_phone_probe_telemetry():
    state.camera_detection_history.clear()
    state.camera_schedule_telemetry.clear()
    phone = {"class": "cell phone", "confidence": 0.58}

    video_processing._record_detection_history(
        "cam_phone",
        [phone],
        schedule_state={},
        model_invocations={"coco_primary": 1},
        runtime_probe_reason=video_processing._MOBILE_PHONE_PROBE_REASON,
    )
    video_processing._record_detection_history(
        "cam_phone",
        [],
        schedule_state={},
        model_invocations={"coco_primary": 1},
        runtime_probe_reason=video_processing._MOBILE_PHONE_PROBE_REASON,
    )

    telemetry = state.camera_schedule_telemetry["cam_phone"]["phoneProbe"]
    assert telemetry["probeCount"] == 2
    assert telemetry["hitProbeCount"] == 1
    assert telemetry["lastProbePhoneDetections"] == 0
    assert telemetry["lastHitAt"]
    assert state.camera_detection_history["cam_phone"][-1]["runtimeProbeReason"] == (
        "mobile_phone_small_object_recall"
    )


def test_detection_history_records_context_suppressed_phone_probe_telemetry():
    state.camera_detection_history.clear()
    state.camera_schedule_telemetry.clear()

    video_processing._record_detection_history(
        "cam_phone",
        [],
        schedule_state={},
        model_invocations={"coco_primary": 1},
        runtime_probe_suppression_reason=(
            video_processing._MOBILE_PHONE_PROBE_CONTEXT_SUPPRESSION_REASON
        ),
    )

    sample = state.camera_detection_history["cam_phone"][-1]
    telemetry = state.camera_schedule_telemetry["cam_phone"]["phoneProbe"]
    assert sample["runtimeProbeSuppressionReason"] == (
        "awaiting_primary_person_context"
    )
    assert telemetry["contextSuppressedCount"] == 1
    assert telemetry["lastContextSuppressionReason"] == (
        "awaiting_primary_person_context"
    )
    assert telemetry["lastContextSuppressedAt"]


def test_violation_window_ignores_stale_display_frames_for_detection_rules():
    windows = {"Missing gloves": [True, True]}
    video_processing._advance_violation_window(
        windows,
        {"Missing gloves": {"rule": "Missing gloves"}},
        window_size=15,
        fresh_detection_evaluated=False,
        fresh_fall_evaluated=False,
    )

    assert windows["Missing gloves"] == [True, True]


def test_violation_window_advances_on_fresh_detection_observation():
    windows = {"Missing gloves": [True, True]}
    video_processing._advance_violation_window(
        windows,
        {"Missing gloves": {"rule": "Missing gloves"}},
        window_size=15,
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
    )

    assert windows["Missing gloves"] == [True, True, True]


def test_empty_violation_observation_does_not_clear_on_stale_display_frame():
    windows = {"Missing gloves": [True]}
    active = {"Missing gloves"}

    video_processing._record_empty_violation_observation(
        windows,
        active,
        window_size=15,
        fresh_detection_evaluated=False,
        fresh_fall_evaluated=False,
    )

    assert windows["Missing gloves"] == [True]
    assert active == {"Missing gloves"}


def test_grouped_inference_runs_closed_set_candidate_with_rule_labels(monkeypatch):
    cfg = {
        "cameras": {
            "cam_candidate": {
                "safety_rule_ids": ["ppe_harness"],
                "safety_rule_overrides": {"ppe_harness": {"confidence": 0.22}},
            }
        },
        "safety_rules": [
            {"id": "ppe_harness", "type": "ppe", "enabled": True, "confidence": 0.31},
        ],
    }
    calls = []

    def fake_predict_records(model_key, frame, *, conf, device, imgsz, classes=None):
        calls.append({
            "model_key": model_key,
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
            "classes": classes,
        })
        return [
            {"class_id": 0, "confidence": 0.91, "bbox": [5, 5, 45, 70]},
            {"class_id": 2, "confidence": 0.88, "bbox": [10, 12, 42, 65]},
        ]

    monkeypatch.setattr(video_processing, "get_config", lambda: cfg)
    monkeypatch.setattr(video_processing.model_manager, "predict_records", fake_predict_records)

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    plan = {
        "capabilities": ["harness_required"],
        "required_model_keys": ["ppe_closed_set_candidate"],
        "capability_model_overrides": {"harness_required": "ppe_closed_set_candidate"},
        "run_coco_primary": False,
        "run_ppe_specialist": False,
        "run_ppe_closed_set_candidate": True,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_pose_specialist": False,
        "ppe_prompt_terms": [],
        "yoloe_prompt_terms": [],
    }

    _annotated, detections, _pose, invocations = video_processing._run_grouped_inference(
        "cam_candidate",
        frame,
        plan,
        conf=0.35,
        device="cpu",
        imgsz=640,
    )

    assert calls == [
        {
            "model_key": "ppe_closed_set_candidate",
            "conf": 0.22,
            "device": "cpu",
            "imgsz": 640,
            "classes": [],
        }
    ]
    assert invocations["ppe_closed_set_candidate"] == 1
    assert invocations["ppe_specialist"] == 0
    assert invocations["coco_primary"] == 0
    assert [d["class"] for d in detections] == ["person", "safety harness"]
    assert {d["model_family"] for d in detections} == {"ppe_closed_set_candidate"}
    assert detections[1]["capability_keys"] == ["harness_required"]


def test_closed_set_candidate_observation_runs_ppe_violation_check(monkeypatch):
    calls = []

    def fake_check_yoloe_violations(detections, camera_id, frame_w=None, frame_h=None):
        calls.append({
            "camera_id": camera_id,
            "classes": [d["class"] for d in detections],
            "frame_w": frame_w,
            "frame_h": frame_h,
        })
        return []

    monkeypatch.setattr(video_processing, "check_yoloe_violations", fake_check_yoloe_violations)
    monkeypatch.setattr(video_processing, "check_violations", lambda detections, camera_id: [])

    confirmation_required = video_processing._process_detection_observation(
        "cam_candidate",
        np.zeros((40, 60, 3), dtype=np.uint8),
        None,
        [{"class": "person", "confidence": 0.9, "bbox": [1, 1, 20, 35], "model_family": "ppe_closed_set_candidate"}],
        None,
        {
            "capabilities": ["harness_required"],
            "run_pose_specialist": False,
            "run_ppe_specialist": False,
            "run_ppe_closed_set_candidate": True,
        },
        {},
        {"site": {"timezone": "Asia/Kolkata"}},
        last_alert_by_rule={},
        active_violations=set(),
        violation_window={},
        alert_cooldown=30,
        window_size=3,
    )

    assert calls == [
        {
            "camera_id": "cam_candidate",
            "classes": ["person"],
            "frame_w": 60,
            "frame_h": 40,
        }
    ]
    assert confirmation_required is False


def test_alert_burst_encodes_one_snapshot_pair_per_observation(monkeypatch):
    candidates = [
        {
            "camera_id": "cam1",
            "rule": rule,
            "severity": "P2",
            "confidence": 0.9,
            "description": f"{rule} detected",
            "source": "test",
            "threshold": 1,
        }
        for rule in ("Missing Helmet", "Missing Vest")
    ]
    encode_calls = []
    alerts = []
    active_violations = set()

    monkeypatch.setattr(video_processing, "check_violations", lambda *_args: candidates)
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_args: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: encode_calls.append(True) or (b"annotated", b"clean"),
    )
    monkeypatch.setattr(
        video_processing.policy_engine,
        "evaluate_candidate",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                fallback=False,
                output_ids=None,
                rule_id="",
                rule_name="Test",
                severity="P2",
                priority=2,
                message=None,
                cooldown_seconds=30,
            )
        ],
    )
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: alerts.append(kwargs) or {"id": f"alert-{len(alerts)}"},
    )

    confirmation_required = video_processing._process_detection_observation(
        "cam1",
        np.zeros((540, 960, 3), dtype=np.uint8),
        None,
        [{"class": "person", "confidence": 0.9, "bbox": [1, 1, 20, 40]}],
        None,
        {"capabilities": [], "run_pose_specialist": False},
        {},
        {"global": {"jpeg_quality": 70}},
        last_alert_by_rule={},
        active_violations=active_violations,
        violation_window={},
        alert_cooldown=30,
        window_size=3,
    )

    assert encode_calls == [True]
    assert [alert["rule"] for alert in alerts] == ["Missing Helmet", "Missing Vest"]
    assert {alert["snapshot_jpeg"] for alert in alerts} == {b"annotated"}
    assert {alert["clean_snapshot_jpeg"] for alert in alerts} == {b"clean"}
    assert active_violations == {"Missing Helmet", "Missing Vest"}
    assert confirmation_required is True


def test_failed_alert_submission_does_not_activate_or_cool_down_incident(monkeypatch):
    candidate = {
        "camera_id": "cam1",
        "rule": "Missing Helmet",
        "severity": "P2",
        "confidence": 0.9,
        "description": "Missing helmet detected",
        "source": "test",
        "threshold": 1,
    }
    submission_results = iter((None, {"id": "alert-1"}))
    active_violations = set()
    violation_window = {}
    last_alert_by_rule = {}

    monkeypatch.setattr(video_processing.time, "time", lambda: 100.0)
    monkeypatch.setattr(video_processing, "check_violations", lambda *_args: [candidate])
    monkeypatch.setattr(video_processing, "extract_violation_bboxes", lambda *_args: [])
    monkeypatch.setattr(
        video_processing,
        "_encode_inference_snapshot_pair",
        lambda *_args, **_kwargs: (b"annotated", b"clean"),
    )
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
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **_kwargs: next(submission_results),
    )

    def observe():
        video_processing._process_detection_observation(
            "cam1",
            np.zeros((540, 960, 3), dtype=np.uint8),
            None,
            [{"class": "person", "confidence": 0.9, "bbox": [1, 1, 20, 40]}],
            None,
            {"capabilities": [], "run_pose_specialist": False},
            {},
            {"global": {"jpeg_quality": 70}},
            last_alert_by_rule=last_alert_by_rule,
            active_violations=active_violations,
            violation_window=violation_window,
            alert_cooldown=30,
            window_size=3,
        )

    observe()
    assert active_violations == set()
    assert last_alert_by_rule == {}
    assert violation_window["Missing Helmet"] == [True]

    observe()
    assert active_violations == {"Missing Helmet"}
    assert last_alert_by_rule == {"Missing Helmet": 100.0}
    assert violation_window["Missing Helmet"] == []


def test_empty_violation_observation_clears_after_fresh_absence():
    windows = {"Missing gloves": [False]}
    active = {"Missing gloves"}

    video_processing._record_empty_violation_observation(
        windows,
        active,
        window_size=15,
        fresh_detection_evaluated=True,
        fresh_fall_evaluated=False,
    )

    assert "Missing gloves" not in windows
    assert active == set()


def test_inference_frame_is_frozen_without_copying_pixels():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)

    frozen = video_processing._freeze_inference_frame(frame)

    assert frozen is frame
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0, 0, 0] = 1
