import numpy as np
import pytest

import video_processing


FRAME_SHAPE = (200, 300, 3)


def _record(class_id, bbox, confidence=0.9, **extra):
    return {
        "class_id": class_id,
        "confidence": confidence,
        "bbox": bbox,
        **extra,
    }


def _plan(*, ppe=True, phone=False):
    return {
        "capabilities": ["person_presence", "helmet_required", "mobile_phone"],
        "run_coco_primary": True,
        "run_ppe_specialist": ppe,
        "run_rtdetr_phone": phone,
        "ppe_prompt_terms": ["hard hat"],
    }


def _cfg(**global_values):
    return {
        "global": {
            "person_crop_padding_fraction": 0.0,
            "person_crop_boundary_margin": 0,
            **global_values,
        },
        "cameras": {"cam": {"safety_rule_ids": ["alert_mobile_phone"]}},
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "type": "alert",
                "enabled": True,
                "confidence": 0.15,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _clear_crop_environment(monkeypatch):
    monkeypatch.delenv("SAFETYLENS_PPE_PERSON_CROP_MODE", raising=False)
    monkeypatch.delenv("SAFETYLENS_PHONE_PERSON_CROP_MODE", raising=False)


def _run(frame, plan, cfg):
    telemetry = {}
    result = video_processing._run_grouped_inference(
        "cam",
        frame,
        plan,
        conf=0.35,
        device="cuda",
        imgsz=640,
        cfg=cfg,
        frame_batch_size_hint=1,
        person_crop_telemetry=telemetry,
    )
    return (*result, telemetry)


def test_active_ppe_crop_remaps_coordinates_and_suppresses_full_frame(monkeypatch):
    calls = []

    def predict(frame, requests, **_options):
        calls.extend((request["model_key"], frame.shape) for request in requests)
        output = {}
        for request in requests:
            if request["model_key"] == "coco_primary":
                output[request["request_id"]] = [_record(0, [50, 20, 150, 180])]
            else:
                assert frame.shape[:2] == (160, 100)
                output[request["request_id"]] = [_record(0, [10, 5, 30, 25])]
        return output

    monkeypatch.setattr(
        video_processing.model_manager, "predict_record_batches", predict
    )
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    _annotated, detections, _pose, invocations, telemetry = _run(
        frame,
        _plan(),
        _cfg(ppe_person_crop_mode="active"),
    )

    ppe = [detection for detection in detections if detection["class"] == "hard hat"]
    assert [detection["bbox"] for detection in ppe] == [[60.0, 25.0, 80.0, 45.0]]
    assert ppe[0]["coordinate_space"] == "full_frame"
    assert ppe[0]["inference_scope"] == "person_crop"
    assert calls == [
        ("coco_primary", FRAME_SHAPE),
        ("ppe_specialist", (160, 100, 3)),
    ]
    assert invocations["ppe_specialist"] == 1
    assert invocations["ppe_person_crop"] == 1
    assert telemetry["ppe"]["authoritativePath"] == "person_crop"
    assert telemetry["ppe"]["fullFrameInvocations"] == 0


def test_shadow_crop_never_replaces_full_frame_authority(monkeypatch):
    def predict(frame, requests, **_options):
        output = {}
        for request in requests:
            if request["model_key"] == "coco_primary":
                records = [_record(0, [50, 20, 150, 180])]
            elif frame.shape == FRAME_SHAPE:
                records = [_record(0, [200, 10, 230, 40])]
            else:
                records = [_record(0, [10, 5, 30, 25])]
            output[request["request_id"]] = records
        return output

    monkeypatch.setattr(
        video_processing.model_manager, "predict_record_batches", predict
    )
    _annotated, detections, _pose, invocations, telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        _plan(),
        _cfg(ppe_person_crop_mode="shadow"),
    )

    ppe_boxes = [
        detection["bbox"]
        for detection in detections
        if detection["class"] == "hard hat"
    ]
    assert ppe_boxes == [[200, 10, 230, 40]]
    assert invocations["ppe_specialist"] == 2
    assert telemetry["ppe"]["authoritativePath"] == "full_frame_shadow"
    assert telemetry["ppe"]["cropCandidateCount"] == 1


@pytest.mark.parametrize(
    ("people", "config", "reason"),
    [
        ([_record(0, [0, 20, 100, 180])], {}, "boundary_person"),
        (
            [
                _record(0, [20, 20, 100, 180]),
                _record(0, [160, 20, 240, 180]),
            ],
            {"person_crop_max_crops": 1},
            "crowd_limit",
        ),
    ],
)
def test_active_ppe_crop_falls_back_for_unsafe_geometry(
    monkeypatch,
    people,
    config,
    reason,
):
    specialist_shapes = []

    def predict(frame, requests, **_options):
        output = {}
        for request in requests:
            if request["model_key"] == "coco_primary":
                records = people
            else:
                specialist_shapes.append(frame.shape)
                records = [_record(0, [200, 10, 230, 40])]
            output[request["request_id"]] = records
        return output

    monkeypatch.setattr(
        video_processing.model_manager, "predict_record_batches", predict
    )
    _annotated, detections, _pose, _invocations, telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        _plan(),
        _cfg(ppe_person_crop_mode="active", **config),
    )

    assert specialist_shapes == [FRAME_SHAPE]
    assert any(detection["bbox"] == [200, 10, 230, 40] for detection in detections)
    assert reason in telemetry["ppe"]["fallbackReasons"]
    assert telemetry["ppe"]["authoritativePath"] == "full_frame"


def test_active_ppe_crop_failure_fails_open_to_full_frame(monkeypatch):
    specialist_shapes = []

    def predict(frame, requests, **_options):
        output = {}
        for request in requests:
            if request["model_key"] == "coco_primary":
                output[request["request_id"]] = [_record(0, [50, 20, 150, 180])]
            elif frame.shape != FRAME_SHAPE:
                specialist_shapes.append(frame.shape)
                raise RuntimeError("synthetic crop failure")
            else:
                specialist_shapes.append(frame.shape)
                output[request["request_id"]] = [_record(0, [200, 10, 230, 40])]
        return output

    monkeypatch.setattr(
        video_processing.model_manager, "predict_record_batches", predict
    )
    _annotated, detections, _pose, invocations, telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        _plan(),
        _cfg(ppe_person_crop_mode="active"),
    )

    assert specialist_shapes == [(160, 100, 3), FRAME_SHAPE]
    assert any(detection["bbox"] == [200, 10, 230, 40] for detection in detections)
    assert invocations["ppe_specialist"] == 2
    assert telemetry["ppe"]["fallbackReasons"] == ["crop_inference_failed"]
    assert telemetry["ppe"]["fullFrameInvocations"] == 1


def test_phone_active_uses_remapped_crop_but_confirm_uses_full_frame(monkeypatch):
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        lambda _frame, requests, **_options: {
            requests[0]["request_id"]: [_record(0, [50, 20, 150, 180])]
        },
    )
    phone_shapes = []

    def predict_phone(frame, **_options):
        phone_shapes.append(frame.shape)
        if frame.shape == FRAME_SHAPE:
            return [_record(67, [210, 30, 230, 60])]
        return [_record(67, [10, 5, 30, 25])]

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_rtdetr_phone_records",
        predict_phone,
    )
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    _a, active_detections, _p, _i, active_telemetry = _run(
        frame,
        _plan(ppe=False, phone=True),
        _cfg(phone_person_crop_mode="active"),
    )
    assert [
        detection["bbox"]
        for detection in active_detections
        if detection["class"] == "cell phone"
    ] == [[60.0, 25.0, 80.0, 45.0]]
    assert phone_shapes == [(160, 100, 3)]
    assert active_telemetry["phone"]["authoritativePath"] == "person_crop"

    phone_shapes.clear()
    _a, confirm_detections, _p, _i, confirm_telemetry = _run(
        frame,
        _plan(ppe=False, phone=True),
        _cfg(phone_person_crop_mode="confirm"),
    )
    assert [
        detection["bbox"]
        for detection in confirm_detections
        if detection["class"] == "cell phone"
    ] == [[210, 30, 230, 60]]
    assert phone_shapes == [(160, 100, 3), FRAME_SHAPE]
    assert confirm_telemetry["phone"]["authoritativePath"] == (
        "full_frame_confirmation"
    )


def test_production_phone_substitution_plan_reaches_person_crop_path(monkeypatch):
    monkeypatch.setattr(
        video_processing.RTDETR_PHONE_SUBSTITUTION_SCHEDULER,
        "consider",
        lambda *_args, **_kwargs: True,
    )
    cfg = _cfg(phone_person_crop_mode="active")
    plan = _plan(ppe=False, phone=False)
    plan["required_model_keys"] = ["coco_primary"]

    runtime_plan, selected = video_processing._rtdetr_phone_substitution_execution_plan(
        "cam",
        plan,
        cfg,
        now=10.0,
        stable_person_track=True,
    )

    assert selected is True
    assert runtime_plan["run_coco_primary"] is True
    assert runtime_plan["run_rtdetr_phone"] is True
    assert "partial_detection_capabilities" not in runtime_plan
    assert runtime_plan["runtime_probe_reason"] == "rtdetr_phone_person_crop"

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        lambda _frame, requests, **_options: {
            requests[0]["request_id"]: [_record(0, [50, 20, 150, 180])]
        },
    )
    phone_shapes = []

    def predict_phone(frame, **_options):
        phone_shapes.append(frame.shape)
        return [_record(67, [10, 5, 30, 25])]

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_rtdetr_phone_records",
        predict_phone,
    )

    _a, detections, _p, invocations, telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        runtime_plan,
        cfg,
    )

    assert phone_shapes == [(160, 100, 3)]
    assert invocations["phone_person_crop"] == 1
    assert telemetry["phone"]["cropInferenceAttempts"] == 1
    assert any(
        detection["class"] == "cell phone"
        and detection["inference_scope"] == "person_crop"
        for detection in detections
    )


def test_phone_crop_route_failure_reuses_primary_without_duplicate_people(
    monkeypatch,
):
    primary_calls = []

    def predict_primary(_frame, requests, **_options):
        primary_calls.append(tuple(request["model_key"] for request in requests))
        return {
            requests[0]["request_id"]: [_record(0, [50, 20, 150, 180])]
        }

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        predict_primary,
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_rtdetr_phone_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            video_processing.model_manager.RemoteRTDETRPhoneUnavailableError(
                "route unavailable"
            )
        ),
    )

    _a, detections, _p, invocations, telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        _plan(ppe=False, phone=True),
        _cfg(phone_person_crop_mode="active"),
    )

    assert primary_calls == [("coco_primary",)]
    assert invocations["coco_primary"] == 1
    assert invocations["rtdetr_phone_fallback"] == 1
    assert [item["class"] for item in detections].count("person") == 1
    assert telemetry["phone"]["fallbackReasons"] == ["crop_inference_failed"]


def test_tracker_projection_is_rejected_as_crop_seed():
    detections = [
        {
            "class": "person",
            "bbox": [20, 20, 100, 180],
            "model_family": "coco_primary",
            "observation_kind": "tracker_projection",
        },
        {
            "class": "person",
            "bbox": [120, 20, 200, 180],
            "model_family": "tracker_projection",
        },
        {
            "class": "person",
            "bbox": [210, 20, 290, 180],
            "model_family": "coco_primary",
            "observation_kind": "fresh_detector",
        },
    ]

    assert video_processing._fresh_primary_person_detections(detections) == [
        detections[2]
    ]


def test_environment_crop_mode_override_precedes_config(monkeypatch):
    monkeypatch.setenv("SAFETYLENS_PPE_PERSON_CROP_MODE", "shadow")
    assert (
        video_processing._configured_person_crop_mode(
            _cfg(ppe_person_crop_mode="off"),
            "ppe_person_crop_mode",
            "SAFETYLENS_PPE_PERSON_CROP_MODE",
        )
        == "shadow"
    )


def test_crop_telemetry_is_exposed_without_payload_or_config_secrets(monkeypatch):
    monkeypatch.setattr(video_processing.state, "camera_detection_history", {})
    monkeypatch.setattr(video_processing.state, "camera_schedule_telemetry", {})
    crop_telemetry = {
        "ppe": {
            "mode": "shadow",
            "cropCount": 1,
            "authoritativePath": "full_frame_shadow",
        }
    }

    video_processing._record_detection_history(
        "cam",
        [],
        model_invocations={"ppe_specialist": 2, "ppe_person_crop": 1},
        person_crop_telemetry=crop_telemetry,
    )

    stored = video_processing.state.camera_schedule_telemetry["cam"]
    assert stored["personCropTelemetry"] == crop_telemetry
    assert set(stored["personCropTelemetry"]["ppe"]) == {
        "mode",
        "cropCount",
        "authoritativePath",
    }


def test_crop_and_pose_rebuild_one_final_annotation_with_primary_and_crop_boxes(
    monkeypatch,
):
    def predict(frame, requests, **_options):
        output = {}
        for request in requests:
            if request["model_key"] == "coco_primary":
                records = [_record(0, [50, 20, 150, 180])]
            elif request["model_key"] == "ppe_specialist":
                assert frame.shape != FRAME_SHAPE
                records = [_record(0, [10, 5, 30, 25])]
            else:
                records = []
            output[request["request_id"]] = records
        return output

    rendered_detection_sets = []
    pose_layers = []
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        predict,
    )
    monkeypatch.setattr(
        video_processing,
        "_draw_stream_detection_records",
        lambda frame, detections, *_args, **_kwargs: (
            rendered_detection_sets.append(list(detections)) or frame.copy()
        ),
    )
    monkeypatch.setattr(
        video_processing,
        "draw_pose_detections",
        lambda annotated, results, **_kwargs: (
            pose_layers.append(results) or annotated,
            [],
        ),
    )
    monkeypatch.setattr(
        video_processing,
        "apply_camera_overlay",
        lambda annotated, **_kwargs: annotated,
    )
    plan = _plan()
    plan["run_pose_specialist"] = True

    annotated, detections, pose_results, _invocations, _telemetry = _run(
        np.zeros(FRAME_SHAPE, dtype=np.uint8),
        plan,
        _cfg(ppe_person_crop_mode="active"),
    )

    assert annotated.shape == FRAME_SHAPE
    assert pose_results == []
    assert len(pose_layers) == 2
    assert [item["class"] for item in rendered_detection_sets[-1]] == [
        "person",
        "hard hat",
    ]
    assert [item["class"] for item in detections] == ["person", "hard hat"]
