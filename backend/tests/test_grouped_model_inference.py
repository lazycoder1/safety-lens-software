import numpy as np

import video_processing


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
