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
