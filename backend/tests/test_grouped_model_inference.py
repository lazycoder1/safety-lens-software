from types import SimpleNamespace

import numpy as np

import video_processing


def _rider_plan():
    return {
        "capabilities": ["animal_presence", "rider_helmet_required"],
        "required_model_keys": ["coco_primary", "ppe_specialist"],
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["motorcycle helmet", "rider helmet", "helmet"],
    }


def _ppe_substitution_config(*, enabled=True):
    return {
        "global": {
            "ppe_specialist_target_fps": 0.5,
            "ppe_specialist_substitution_enabled": enabled,
        }
    }


def test_ppe_cadence_suppresses_additive_work_between_due_slots():
    video_processing.PPE_SUBSTITUTION_SCHEDULER.reset()
    plan = _rider_plan()
    cfg = _ppe_substitution_config(enabled=False)

    first, due, substituted = (
        video_processing._ppe_specialist_cadence_execution_plan(
            "cam1",
            plan,
            cfg,
            now=10.0,
            stable_person_track=True,
        )
    )
    suppressed, due_again, substituted_again = (
        video_processing._ppe_specialist_cadence_execution_plan(
            "cam1",
            plan,
            cfg,
            now=10.25,
            stable_person_track=True,
        )
    )

    assert first is plan
    assert (due, substituted) == (True, False)
    assert (due_again, substituted_again) == (False, False)
    assert suppressed["run_coco_primary"] is True
    assert suppressed["run_ppe_specialist"] is False
    assert suppressed["required_model_keys"] == ["coco_primary"]
    assert suppressed["runtime_suppression_reason"] == "ppe_specialist_cadence"


def test_ppe_due_slot_replaces_primary_with_stable_cached_context():
    video_processing.PPE_SUBSTITUTION_SCHEDULER.reset()
    plan = _rider_plan()
    context = [
        {
            "class": "person",
            "confidence": 0.9,
            "bbox": [40, 10, 100, 155],
            "model_family": "coco_primary",
        },
        {
            "class": "motorcycle",
            "confidence": 0.9,
            "bbox": [30, 100, 130, 190],
            "model_family": "coco_primary",
        },
        {
            "class": "dog",
            "confidence": 0.8,
            "bbox": [1, 2, 10, 12],
            "model_family": "coco_primary",
        },
    ]

    result, due, substituted = (
        video_processing._ppe_specialist_cadence_execution_plan(
            "cam1",
            plan,
            _ppe_substitution_config(),
            now=20.0,
            stable_person_track=True,
            previous_detections=context,
        )
    )

    assert (due, substituted) == (True, True)
    assert result["run_coco_primary"] is False
    assert result["run_ppe_specialist"] is True
    assert result["required_model_keys"] == ["ppe_specialist"]
    assert result["partial_detection_capabilities"] == ["rider_helmet_required"]
    assert [item["class"] for item in result["ppe_context_detections"]] == [
        "person",
        "motorcycle",
    ]


def test_ppe_substitution_stays_additive_with_unvalidated_companion():
    video_processing.PPE_SUBSTITUTION_SCHEDULER.reset()
    plan = _rider_plan()
    plan["run_pose_specialist"] = True

    result, due, substituted = (
        video_processing._ppe_specialist_cadence_execution_plan(
            "cam1",
            plan,
            _ppe_substitution_config(),
            now=30.0,
            stable_person_track=True,
        )
    )

    assert result is plan
    assert (due, substituted) == (True, False)


def test_grouped_ppe_substitution_requests_only_ppe_and_merges_cached_context(
    monkeypatch,
):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    context = {
        "class": "person",
        "confidence": 0.9,
        "bbox": [40, 10, 100, 155],
        "model_family": "coco_primary",
        "capability_keys": ["person_presence"],
    }
    captured = {}

    def fake_predict(_frame, requests, **_options):
        captured["requests"] = requests
        return {
            "ppe_specialist": [
                {
                    "class_id": 0,
                    "confidence": 0.8,
                    "bbox": [50, 15, 75, 40],
                }
            ]
        }

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict,
    )
    plan = {
        "capabilities": ["helmet_required"],
        "required_model_keys": ["ppe_specialist"],
        "run_coco_primary": False,
        "run_ppe_specialist": True,
        "run_ppe_substitution": True,
        "ppe_prompt_terms": ["hard hat"],
        "partial_detection_capabilities": ["helmet_required"],
        "ppe_context_detections": [context],
    }

    _annotated, detections, _pose, invocations = (
        video_processing._run_grouped_inference(
            "cam1",
            frame,
            plan,
            conf=0.35,
            device="cuda",
            imgsz=640,
            cfg={"global": {"ppe_inference_width": 640}},
            frame_batch_size_hint=4,
        )
    )

    assert [request["model_key"] for request in captured["requests"]] == [
        "ppe_specialist"
    ]
    assert invocations["coco_primary"] == 0
    assert invocations["ppe_specialist"] == 1
    assert [item["model_family"] for item in detections] == [
        "ppe_specialist",
        "coco_primary",
    ]
    assert detections[1] == context


def test_rider_only_ppe_waits_for_coco_vehicle_context():
    plan = _rider_plan()

    gated = video_processing._context_gated_execution_plan(plan, [])

    assert gated is not plan
    assert gated["run_coco_primary"] is True
    assert gated["run_ppe_specialist"] is False
    assert gated["required_model_keys"] == ["coco_primary"]
    assert gated["ppe_prompt_terms"] == []
    assert gated["runtime_suppression_reason"] == "awaiting_rider_vehicle_context"
    assert plan["run_ppe_specialist"] is True


def test_rider_only_ppe_waits_for_evaluable_person_vehicle_association():
    plan = _rider_plan()

    gated = video_processing._context_gated_execution_plan(
        plan,
        [
            {
                "class": "motorcycle",
                "model_family": "coco_primary",
                "confidence": 0.9,
                "bbox": [30, 100, 130, 190],
            }
        ],
        frame_w=200,
        frame_h=200,
    )

    assert gated["run_ppe_specialist"] is False


def test_rider_only_ppe_runs_after_evaluable_coco_rider_association():
    plan = _rider_plan()

    result = video_processing._context_gated_execution_plan(
        plan,
        [
            {
                "class": "person",
                "model_family": "coco_primary",
                "confidence": 0.9,
                "bbox": [40, 10, 100, 155],
            },
            {
                "class": "motorcycle",
                "model_family": "coco_primary",
                "confidence": 0.9,
                "bbox": [30, 100, 130, 190],
            },
        ],
        frame_w=200,
        frame_h=200,
    )

    assert result is plan


def test_rider_gate_ignores_vehicle_label_from_non_coco_model():
    plan = _rider_plan()

    gated = video_processing._context_gated_execution_plan(
        plan,
        [
            {
                "class": "motorcycle",
                "model_family": "ppe_specialist",
                "confidence": 0.9,
                "bbox": [30, 100, 130, 190],
            }
        ],
        frame_w=200,
        frame_h=200,
    )

    assert gated["run_ppe_specialist"] is False


def test_other_ppe_capabilities_wait_for_coco_person_context():
    plan = _rider_plan()
    plan["capabilities"].append("helmet_required")

    gated = video_processing._context_gated_execution_plan(plan, [])

    assert gated["run_ppe_specialist"] is False
    assert gated["runtime_suppression_reason"] == "awaiting_person_context"


def test_other_ppe_capabilities_run_after_coco_person_detection():
    plan = _rider_plan()
    plan["capabilities"].append("helmet_required")

    result = video_processing._context_gated_execution_plan(
        plan,
        [
            {
                "class": "person",
                "model_family": "coco_primary",
                "confidence": 0.9,
                "bbox": [40, 10, 100, 155],
            }
        ],
        frame_w=200,
        frame_h=200,
    )

    assert result is plan


def test_other_ppe_gate_ignores_person_from_non_coco_model():
    plan = _rider_plan()
    plan["capabilities"].append("helmet_required")

    gated = video_processing._context_gated_execution_plan(
        plan,
        [
            {
                "class": "person",
                "model_family": "ppe_specialist",
                "confidence": 0.9,
                "bbox": [40, 10, 100, 155],
            }
        ],
        frame_w=200,
        frame_h=200,
    )

    assert gated["run_ppe_specialist"] is False


def test_other_ppe_gate_requires_person_inside_ppe_evaluation_zone():
    plan = _rider_plan()
    plan["capabilities"].append("helmet_required")
    camera = {
        "zones": [
            {
                "id": "ppe_gate",
                "type": "ppe_evaluation",
                "points": [[0.0, 0.0], [0.45, 0.0], [0.45, 1.0], [0.0, 1.0]],
            }
        ]
    }
    detection = {
        "class": "person",
        "model_family": "coco_primary",
        "confidence": 0.9,
        "bbox": [120, 10, 190, 155],
    }

    gated = video_processing._context_gated_execution_plan(
        plan,
        [detection],
        camera=camera,
        frame_w=200,
        frame_h=200,
    )
    active = video_processing._context_gated_execution_plan(
        plan,
        [{**detection, "bbox": [10, 10, 80, 155]}],
        camera=camera,
        frame_w=200,
        frame_h=200,
    )

    assert gated["run_ppe_specialist"] is False
    assert active is plan


def test_mobile_phone_probe_defers_ppe_without_losing_the_due_probe():
    plan = {
        "capabilities": ["mobile_phone", "rider_helmet_required"],
        "required_model_keys": ["coco_primary", "ppe_specialist"],
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
    }
    cfg = {
        "global": {
            "inference_width": 960,
            "coco_inference_width": 640,
            "ppe_inference_width": 640,
            "mobile_phone_inference_width": 960,
            "mobile_phone_probe_interval_seconds": 1.0,
        }
    }

    probed, due, suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=8.9,
        previous_detections=[
            {"class": "person", "model_family": "coco_primary"},
        ],
    )
    waiting, waiting_due, waiting_suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=9.1,
        previous_detections=[
            {"class": "person", "model_family": "coco_primary"},
        ],
    )

    assert due is True
    assert suppressed is False
    assert probed is not plan
    assert probed["coco_inference_width_override"] == 960
    assert probed["runtime_probe_reason"] == "mobile_phone_small_object_recall"
    assert probed["run_ppe_specialist"] is False
    assert probed["ppe_prompt_terms"] == []
    assert probed["required_model_keys"] == ["coco_primary"]
    assert probed["runtime_deferred_model_keys"] == ["ppe_specialist"]
    assert probed["runtime_specialist_deferral_reason"] == (
        "deferred_for_mobile_phone_probe"
    )
    assert waiting is plan
    assert waiting_due is False
    assert waiting_suppressed is False
    assert "coco_inference_width_override" not in plan
    assert plan["run_ppe_specialist"] is True


def test_mobile_phone_probe_waits_for_primary_person_context():
    plan = {"capabilities": ["mobile_phone"], "run_coco_primary": True}
    cfg = {
        "global": {
            "coco_inference_width": 640,
            "mobile_phone_inference_width": 960,
            "mobile_phone_probe_interval_seconds": 1.0,
        }
    }

    empty, empty_due, empty_suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=None,
        previous_detections=[],
    )
    repeated, repeated_due, repeated_suppressed = (
        video_processing._mobile_phone_probe_execution_plan(
            plan,
            cfg,
            now=10.5,
            last_probe_at=None,
            last_context_suppressed_at=10.0,
            previous_detections=[],
        )
    )
    specialist_only, specialist_due, specialist_suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=None,
        previous_detections=[
            {"class": "person", "model_family": "ppe_specialist"},
        ],
    )

    assert empty is not plan
    assert empty_due is False
    assert empty_suppressed is True
    assert empty["runtime_probe_suppression_reason"] == (
        "awaiting_primary_person_context"
    )
    assert repeated is plan
    assert repeated_due is False
    assert repeated_suppressed is False
    assert specialist_only is not plan
    assert specialist_due is False
    assert specialist_suppressed is True


def test_mobile_phone_probe_is_disabled_for_non_phone_plan():
    plan = {"capabilities": ["animal_presence"], "run_coco_primary": True}
    cfg = {
        "global": {
            "coco_inference_width": 640,
            "mobile_phone_inference_width": 960,
            "mobile_phone_probe_interval_seconds": 1.0,
        }
    }

    result, due, suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=None,
    )

    assert result is plan
    assert due is False
    assert suppressed is False


def test_mobile_phone_probe_is_disabled_when_primary_already_uses_probe_width():
    plan = {
        "capabilities": ["mobile_phone"],
        "run_coco_primary": True,
    }
    cfg = {
        "global": {
            "coco_inference_width": 640,
            "mobile_phone_inference_width": 640,
            "mobile_phone_probe_interval_seconds": 1.0,
        }
    }

    result, due, suppressed = video_processing._mobile_phone_probe_execution_plan(
        plan,
        cfg,
        now=10.0,
        last_probe_at=None,
        previous_detections=[
            {"class": "person", "model_family": "coco_primary"},
        ],
    )

    assert result is plan
    assert due is False
    assert suppressed is False


def test_rtdetr_phone_substitution_replaces_primary_and_defers_ppe(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        video_processing.RTDETR_PHONE_SUBSTITUTION_SCHEDULER,
        "consider",
        lambda camera_id, cfg, *, now, stable_person: captured.update(
            camera_id=camera_id,
            cfg=cfg,
            now=now,
            stable_person=stable_person,
        )
        or True,
    )
    plan = {
        "capabilities": ["mobile_phone", "helmet_required"],
        "required_model_keys": ["coco_primary", "ppe_specialist"],
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
    }
    cfg = {"global": {"rtdetr_phone_substitution_enabled": True}}

    result, selected = video_processing._rtdetr_phone_substitution_execution_plan(
        "cam-phone",
        plan,
        cfg,
        now=12.5,
        stable_person_track=True,
    )

    assert selected is True
    assert result["run_coco_primary"] is False
    assert result["run_rtdetr_phone"] is True
    assert result["run_ppe_specialist"] is False
    assert result["partial_detection_capabilities"] == ["mobile_phone"]
    assert result["required_model_keys"] == []
    assert result["runtime_deferred_model_keys"] == ["ppe_specialist"]
    assert captured["stable_person"] is True


def test_rtdetr_phone_substitution_rejects_unvalidated_companion(monkeypatch):
    stable_values = []
    monkeypatch.setattr(
        video_processing.RTDETR_PHONE_SUBSTITUTION_SCHEDULER,
        "consider",
        lambda _camera_id, _cfg, *, now, stable_person: stable_values.append(
            stable_person
        )
        or False,
    )
    plan = {
        "capabilities": ["mobile_phone", "fall_detection"],
        "run_coco_primary": True,
        "run_pose_specialist": True,
    }

    result, selected = video_processing._rtdetr_phone_substitution_execution_plan(
        "cam-phone",
        plan,
        {"global": {"rtdetr_phone_substitution_enabled": True}},
        now=12.5,
        stable_person_track=True,
    )

    assert result is plan
    assert selected is False
    assert stable_values == [False]


def test_grouped_inference_runs_rtdetr_phone_records_without_primary(monkeypatch):
    captured = {}

    def fake_rtdetr(frame, *, person_conf, phone_conf, frame_batch_size_hint):
        captured.update(
            frame=frame,
            person_conf=person_conf,
            phone_conf=phone_conf,
            frame_batch_size_hint=frame_batch_size_hint,
        )
        return [
            {"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 100, 200]},
            {"class_id": 67, "confidence": 0.8, "bbox": [45, 50, 55, 70]},
        ]

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_rtdetr_phone_records",
        fake_rtdetr,
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("primary route must be substituted")
        ),
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cfg = {
        "cameras": {"cam-phone": {"safety_rule_ids": ["alert_mobile_phone"]}},
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "type": "alert",
                "enabled": True,
                "confidence": 0.15,
            }
        ],
    }

    _annotated, detections, _pose, invocations = video_processing._run_grouped_inference(
        "cam-phone",
        frame,
        {"capabilities": ["mobile_phone"], "run_rtdetr_phone": True},
        conf=0.3,
        device="cuda",
        imgsz=640,
        cfg=cfg,
        frame_batch_size_hint=2,
    )

    assert [detection["class"] for detection in detections] == [
        "person",
        "cell phone",
    ]
    assert {detection["model_family"] for detection in detections} == {
        "rtdetr_phone"
    }
    assert invocations["rtdetr_phone"] == 1
    assert invocations["coco_primary"] == 0
    assert captured["phone_conf"] == 0.15
    assert captured["frame_batch_size_hint"] == 2


def test_grouped_inference_falls_back_to_primary_when_rtdetr_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_rtdetr_phone_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            video_processing.model_manager.RemoteRTDETRPhoneUnavailableError(
                "unavailable"
            )
        ),
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        lambda _frame, requests, **_kwargs: {
            requests[0]["request_id"]: [
                {"class_id": 0, "confidence": 0.9, "bbox": [1, 2, 30, 40]}
            ]
        },
    )

    _annotated, detections, _pose, invocations = video_processing._run_grouped_inference(
        "cam-phone",
        np.zeros((90, 160, 3), dtype=np.uint8),
        {"capabilities": ["mobile_phone"], "run_rtdetr_phone": True},
        conf=0.3,
        device="cuda",
        imgsz=640,
    )

    assert detections[0]["model_family"] == "coco_primary"
    assert invocations["rtdetr_phone_fallback"] == 1
    assert invocations["coco_primary"] == 1


def test_partial_phone_observation_does_not_clear_unrelated_alert_state():
    active = {"Animal Intrusion"}
    windows = {
        "Animal Intrusion": [],
        "Mobile Phone Usage": [True],
    }
    cfg = {
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "name": "Mobile Phone Usage",
                "type": "alert",
                "enabled": True,
            },
            {
                "id": "alert_animal",
                "name": "Animal Intrusion",
                "type": "alert",
                "enabled": True,
            },
        ]
    }
    cam = {
        "safety_rule_ids": ["alert_mobile_phone", "alert_animal"],
    }

    video_processing._process_detection_observation(
        "cam-phone",
        np.zeros((90, 160, 3), dtype=np.uint8),
        None,
        [],
        None,
        {
            "capabilities": ["mobile_phone", "animal_presence"],
            "partial_detection_capabilities": ["mobile_phone"],
            "run_pose_specialist": False,
            "run_ppe_specialist": False,
            "run_ppe_closed_set_candidate": False,
        },
        cam,
        cfg,
        last_alert_by_rule={},
        active_violations=active,
        violation_window=windows,
        alert_cooldown=30,
        window_size=15,
    )

    assert active == {"Animal Intrusion"}
    assert windows["Animal Intrusion"] == []
    assert windows["Mobile Phone Usage"] == [True, False]


def test_partial_ppe_observation_advances_only_ppe_and_preserves_unrelated_state(
    monkeypatch,
):
    capability_filters = []

    def check_ppe(_detections, _camera_id, _frame_w, _frame_h, *, capability_filter=None):
        capability_filters.append(capability_filter)
        return [
            {
                "camera_id": "cam-ppe",
                "rule": "Missing helmet",
                "severity": "P2",
                "confidence": 0.9,
                "description": "Worker missing helmet",
                "source": "PPE Specialist",
                "threshold": 2,
            }
        ]

    monkeypatch.setattr(video_processing, "check_yoloe_violations", check_ppe)
    monkeypatch.setattr(
        video_processing,
        "check_violations",
        lambda *_args, **_kwargs: [],
    )
    active = {"Animal Intrusion"}
    windows = {"Animal Intrusion": [True, True]}

    video_processing._process_detection_observation(
        "cam-ppe",
        np.zeros((200, 200, 3), dtype=np.uint8),
        None,
        [
            {
                "class": "person",
                "confidence": 0.9,
                "bbox": [40, 10, 100, 155],
                "model_family": "coco_primary",
            }
        ],
        None,
        {
            "capabilities": ["helmet_required", "animal_presence"],
            "partial_detection_capabilities": ["helmet_required"],
            "run_pose_specialist": False,
            "run_ppe_specialist": True,
            "run_ppe_closed_set_candidate": False,
        },
        {"safety_rule_ids": ["ppe_helmet", "alert_animal"]},
        {
            "safety_rules": [
                {
                    "id": "ppe_helmet",
                    "name": "Helmet",
                    "type": "ppe",
                    "enabled": True,
                },
                {
                    "id": "alert_animal",
                    "name": "Animal Intrusion",
                    "type": "alert",
                    "enabled": True,
                },
            ]
        },
        last_alert_by_rule={},
        active_violations=active,
        violation_window=windows,
        alert_cooldown=30,
        window_size=15,
    )

    assert capability_filters == [{"helmet_required"}]
    assert active == {"Animal Intrusion"}
    assert windows["Animal Intrusion"] == [True, True]
    assert windows["Missing helmet"] == [True]


def test_partial_phone_observations_create_phone_alert_after_threshold(monkeypatch):
    alerts = []
    capability_filters = []
    candidate = {
        "camera_id": "cam-phone",
        "rule": "Mobile Phone Usage",
        "severity": "P3",
        "confidence": 0.8,
        "description": "Mobile phone detected near a worker",
        "source": "RT-DETR Phone Recall",
        "threshold": 2,
    }

    def check_violations(_detections, _camera_id, *, capability_filter=None):
        capability_filters.append(capability_filter)
        return [candidate]

    monkeypatch.setattr(video_processing, "check_violations", check_violations)
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
                severity="P3",
                priority=3,
                message=None,
                cooldown_seconds=60,
            )
        ],
    )
    monkeypatch.setattr(
        video_processing,
        "create_alert",
        lambda **kwargs: alerts.append(kwargs) or {"id": "phone-alert"},
    )
    plan = {
        "capabilities": ["mobile_phone", "animal_presence"],
        "partial_detection_capabilities": ["mobile_phone"],
        "run_pose_specialist": False,
        "run_ppe_specialist": False,
        "run_ppe_closed_set_candidate": False,
    }
    cam = {"safety_rule_ids": ["alert_mobile_phone", "alert_animal"]}
    cfg = {
        "global": {"jpeg_quality": 70},
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "name": "Mobile Phone Usage",
                "type": "alert",
                "enabled": True,
            },
            {
                "id": "alert_animal",
                "name": "Animal Intrusion",
                "type": "alert",
                "enabled": True,
            },
        ],
    }
    detections = [
        {
            "class": "person",
            "confidence": 0.9,
            "bbox": [0, 0, 100, 200],
            "model_family": "rtdetr_phone",
        },
        {
            "class": "cell phone",
            "confidence": 0.8,
            "bbox": [45, 50, 55, 70],
            "model_family": "rtdetr_phone",
        },
    ]
    active = {"Animal Intrusion"}
    windows = {"Animal Intrusion": [True, True, True]}

    for _ in range(2):
        video_processing._process_detection_observation(
            "cam-phone",
            np.zeros((90, 160, 3), dtype=np.uint8),
            None,
            detections,
            None,
            plan,
            cam,
            cfg,
            last_alert_by_rule={},
            active_violations=active,
            violation_window=windows,
            alert_cooldown=60,
            window_size=15,
        )

    assert capability_filters == [{"mobile_phone"}, {"mobile_phone"}]
    assert len(alerts) == 1
    assert alerts[0]["rule"] == "Mobile Phone Usage"
    assert "Animal Intrusion" in active
    assert windows["Animal Intrusion"] == [True, True, True]


def test_grouped_inference_submits_record_models_as_one_frame_batch(monkeypatch):
    captured = {}

    def fake_predict_record_batches(frame, requests):
        captured["frame"] = frame
        captured["requests"] = requests
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    monkeypatch.setattr(
        video_processing,
        "draw_detection_records",
        lambda annotated, *_args, **_kwargs: (annotated, []),
    )
    monkeypatch.setattr(
        video_processing,
        "apply_camera_overlay",
        lambda annotated, **_kwargs: annotated,
    )
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    _annotated, detections, pose_results, invocations = video_processing._run_grouped_inference(
        "cam-test",
        frame,
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
    )

    assert captured["frame"] is frame
    assert [item["request_id"] for item in captured["requests"]] == [
        "coco_primary",
        "ppe_specialist",
    ]
    assert captured["requests"][1]["classes"] == ["helmet"]
    assert detections == []
    assert pose_results is None
    assert invocations["coco_primary"] == 1
    assert invocations["ppe_specialist"] == 1


def test_grouped_inference_forwards_runtime_frame_batch_hint(monkeypatch):
    captured = {}

    def fake_predict_record_batches(_frame, requests, **kwargs):
        captured["options"] = kwargs
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": False,
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    video_processing._run_grouped_inference(
        "cam-test",
        np.zeros((90, 160, 3), dtype=np.uint8),
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=640,
        frame_batch_size_hint=1,
    )

    assert captured["options"] == {"frame_batch_size_hint": 1}


def test_grouped_inference_can_size_coco_without_downsizing_ppe(monkeypatch):
    captured = {}

    def fake_predict_record_batches(_frame, requests):
        captured["requests"] = requests
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    video_processing._run_grouped_inference(
        "cam-test",
        np.zeros((720, 960, 3), dtype=np.uint8),
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
        cfg={"global": {"coco_inference_width": 640}},
    )

    assert [request["imgsz"] for request in captured["requests"]] == [640, 960]


def test_grouped_inference_can_size_coco_and_ppe_independently(monkeypatch):
    captured = {}

    def fake_predict_record_batches(_frame, requests):
        captured["requests"] = requests
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    video_processing._run_grouped_inference(
        "cam-test",
        np.zeros((720, 960, 3), dtype=np.uint8),
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
        cfg={
            "global": {
                "coco_inference_width": 640,
                "ppe_inference_width": 640,
            }
        },
    )

    assert [request["imgsz"] for request in captured["requests"]] == [640, 640]


def test_grouped_inference_phone_probe_overrides_coco_but_not_ppe(monkeypatch):
    captured = {}

    def fake_predict_record_batches(_frame, requests):
        captured["requests"] = requests
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
        "coco_inference_width_override": 960,
    }

    video_processing._run_grouped_inference(
        "cam-test",
        np.zeros((720, 960, 3), dtype=np.uint8),
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
        cfg={
            "global": {
                "coco_inference_width": 640,
                "ppe_inference_width": 640,
            }
        },
    )

    assert [request["imgsz"] for request in captured["requests"]] == [960, 640]


def test_grouped_inference_ignores_invalid_coco_width(monkeypatch):
    captured = {}

    def fake_predict_record_batches(_frame, requests):
        captured["requests"] = requests
        return {item["request_id"]: [] for item in requests}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": False,
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    video_processing._run_grouped_inference(
        "cam-test",
        np.zeros((720, 960, 3), dtype=np.uint8),
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
        cfg={"global": {"coco_inference_width": "640"}},
    )

    assert captured["requests"][0]["imgsz"] == 960


def test_coco_inference_uses_mobile_rule_confidence_without_lowering_other_rules(monkeypatch):
    captured = {}
    cfg = {
        "cameras": {
            "cam-phone": {
                "safety_rule_ids": ["alert_mobile_phone", "alert_animal"],
            }
        },
        "safety_rules": [
            {
                "id": "alert_mobile_phone",
                "type": "alert",
                "enabled": True,
                "confidence": 0.15,
            },
            {
                "id": "alert_animal",
                "type": "alert",
                "enabled": True,
            },
        ],
    }

    def fake_predict_record_batches(_frame, requests):
        captured["requests"] = requests
        return {
            "coco_primary": [
                {"class_id": 67, "confidence": 0.18, "bbox": [1, 2, 10, 20]},
                {"class_id": 16, "confidence": 0.18, "bbox": [20, 2, 35, 20]},
            ]
        }

    monkeypatch.setattr(
        video_processing,
        "get_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("grouped inference should reuse the caller's config snapshot")
        ),
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    execution_plan = {
        "capabilities": ["mobile_phone", "animal_presence"],
        "run_coco_primary": True,
        "run_ppe_specialist": False,
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    _annotated, detections, _pose, _invocations = video_processing._run_grouped_inference(
        "cam-phone",
        np.zeros((90, 160, 3), dtype=np.uint8),
        execution_plan,
        conf=0.30,
        device="cuda",
        imgsz=960,
        cfg=cfg,
    )

    assert captured["requests"][0]["conf"] == 0.15
    assert [detection["class"] for detection in detections] == ["cell phone"]


def test_bbox_only_grouped_inference_skips_full_resolution_rendering(monkeypatch):
    def fake_predict_record_batches(_frame, requests):
        assert [request["request_id"] for request in requests] == ["coco_primary", "ppe_specialist"]
        return {
            "coco_primary": [{"class_id": 2, "confidence": 0.91, "bbox": [1, 2, 30, 40]}],
            "ppe_specialist": [{"class_id": 0, "confidence": 0.82, "bbox": [5, 6, 20, 25]}],
        }

    def fail_full_resolution_render(*_args, **_kwargs):
        raise AssertionError("full-resolution render should be deferred")

    def fail_full_resolution_overlay(*_args, **_kwargs):
        raise AssertionError("full-resolution overlay should be deferred")

    monkeypatch.setattr(video_processing.model_manager, "predict_record_batches", fake_predict_record_batches)
    monkeypatch.setattr(video_processing, "draw_detection_records", fail_full_resolution_render)
    monkeypatch.setattr(video_processing, "apply_camera_overlay", fail_full_resolution_overlay)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": True,
        "ppe_prompt_terms": ["helmet"],
        "run_yoloe_long_tail": False,
        "run_pose_specialist": False,
    }

    annotated, detections, pose_results, invocations = video_processing._run_grouped_inference(
        "cam-test",
        frame,
        execution_plan,
        conf=0.3,
        device="cuda",
        imgsz=960,
    )

    assert annotated is None
    assert pose_results is None
    assert invocations["coco_primary"] == 1
    assert invocations["ppe_specialist"] == 1
    assert detections == [
        {
            "class_id": 2,
            "class": "car",
            "confidence": 0.91,
            "bbox": [1, 2, 30, 40],
            "model_family": "coco_primary",
            "capability_keys": ["vehicle_presence"],
        },
        {
            "class_id": 0,
            "class": "helmet",
            "confidence": 0.82,
            "bbox": [5, 6, 20, 25],
            "model_family": "ppe_specialist",
            "capability_keys": ["rider_helmet_required"],
        },
    ]


def test_pose_uses_shared_record_batch_instead_of_local_predict(monkeypatch):
    pose_records = [
        {
            "class_id": 0,
            "confidence": 0.9,
            "bbox": [10, 20, 70, 80],
            "keypoints": [[20.0, 30.0, 0.8]] * 17,
        }
    ]
    captured = {}

    def fake_predict_record_batches(frame, requests):
        captured["frame"] = frame
        captured["requests"] = requests
        return {"coco_primary": [], "pose_specialist": pose_records}

    monkeypatch.setattr(
        video_processing.model_manager,
        "predict_record_batches",
        fake_predict_record_batches,
    )
    monkeypatch.setattr(
        video_processing.model_manager,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pose bypassed grouped remote inference")
        ),
    )
    monkeypatch.setattr(
        video_processing,
        "draw_pose_detections",
        lambda annotated, results, **_kwargs: (
            annotated,
            [] if results is pose_records else ["wrong pose records"],
        ),
    )
    monkeypatch.setattr(
        video_processing,
        "apply_camera_overlay",
        lambda annotated, **_kwargs: annotated,
    )
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    execution_plan = {
        "run_coco_primary": True,
        "run_ppe_specialist": False,
        "run_yoloe_long_tail": False,
        "run_fire_smoke_specialist": False,
        "run_pose_specialist": True,
    }

    _annotated, detections, pose_results, invocations = (
        video_processing._run_grouped_inference(
            "cam-test",
            frame,
            execution_plan,
            conf=0.3,
            device="cuda",
            imgsz=960,
        )
    )

    assert captured["frame"] is frame
    assert [item["request_id"] for item in captured["requests"]] == [
        "coco_primary",
        "pose_specialist",
    ]
    assert detections == []
    assert pose_results is pose_records
    assert invocations["pose_specialist"] == 1
