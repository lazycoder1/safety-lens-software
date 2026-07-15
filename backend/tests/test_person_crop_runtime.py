import numpy as np
import pytest

from person_crop_runtime import (
    FALLBACK_BOUNDARY_PERSON,
    FALLBACK_CROWD_LIMIT,
    FALLBACK_CROP_FAILED,
    FALLBACK_INVALID_PERSON,
    FALLBACK_MISSING_PLAN,
    FALLBACK_SMALL_PERSON,
    PersonCropPolicy,
    decide_crop_execution,
    plan_person_crops,
    remap_crop_detections,
)


def _frame(width=120, height=100):
    values = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    return np.repeat((values % 255).astype(np.uint8)[..., None], 3, axis=2)


def _person(bbox, confidence=0.9, track_id=None):
    detection = {"class": "person", "bbox": bbox, "confidence": confidence}
    if track_id is not None:
        detection["track_id"] = track_id
    return detection


def test_person_crop_is_padded_clipped_and_copied():
    frame = _frame()
    policy = PersonCropPolicy(
        padding_fraction=0.1,
        min_person_width=10,
        min_person_height=10,
    )

    plan = plan_person_crops(
        frame,
        [_person([20, 10, 60, 90], track_id="p-1")],
        policy,
    )
    original_pixel = int(plan.crops[0].image[0, 0, 0])
    frame[2, 16] = 255

    assert plan.fallback_required is False
    assert plan.crops[0].crop_bbox == (16, 2, 64, 98)
    assert plan.crops[0].image.shape == (96, 48, 3)
    assert plan.crops[0].track_id == "p-1"
    assert int(plan.crops[0].image[0, 0, 0]) == original_pixel


def test_invalid_small_and_boundary_people_retain_full_frame_fallback():
    policy = PersonCropPolicy(
        padding_fraction=0,
        min_person_width=20,
        min_person_height=40,
        boundary_margin=2,
    )

    plan = plan_person_crops(
        _frame(),
        [
            _person([20, 20, 10, 80]),
            _person([20, 20, 30, 40]),
            _person([0, 10, 30, 80]),
        ],
        policy,
    )

    assert plan.fallback_required is True
    assert plan.fallback_reasons == (
        FALLBACK_INVALID_PERSON,
        FALLBACK_SMALL_PERSON,
        FALLBACK_BOUNDARY_PERSON,
    )
    assert len(plan.crops) == 1
    assert plan.crops[0].boundary_person is True
    assert set(plan.skipped_detection_indices) == {0, 1}


def test_people_are_deduplicated_and_work_is_bounded_in_crowds():
    policy = PersonCropPolicy(
        padding_fraction=0,
        min_person_width=10,
        min_person_height=10,
        max_crops=2,
        person_dedup_iou=0.8,
    )
    detections = [
        _person([10, 10, 40, 80], 0.95),
        _person([11, 10, 41, 80], 0.70),
        _person([45, 10, 70, 80], 0.90),
        _person([80, 10, 110, 80], 0.85),
    ]

    plan = plan_person_crops(_frame(), detections, policy)

    assert len(plan.crops) == 2
    assert [crop.source_detection_index for crop in plan.crops] == [0, 2]
    assert plan.fallback_reasons == (FALLBACK_CROWD_LIMIT,)
    assert set(plan.skipped_detection_indices) == {1, 3}


def test_crop_results_remap_to_full_frame_and_deduplicate_across_people():
    policy = PersonCropPolicy(
        padding_fraction=0,
        min_person_width=10,
        min_person_height=10,
        person_dedup_iou=0.95,
    )
    plan = plan_person_crops(
        _frame(),
        [
            _person([10, 10, 70, 90], track_id="left"),
            _person([40, 10, 100, 90], track_id="right"),
        ],
        policy,
    )

    remapped = remap_crop_detections(
        plan,
        {
            "person-0": [
                {
                    "class": "phone",
                    "bbox": [30, 20, 50, 40],
                    "confidence": 0.7,
                    "keypoints": [[30, 20, 0.8]],
                }
            ],
            "person-1": [
                {
                    "class": "phone",
                    "bbox": [0, 20, 20, 40],
                    "confidence": 0.9,
                    "keypoints": [{"x": 5, "y": 25, "score": 0.9}],
                }
            ],
        },
    )

    assert len(remapped) == 1
    assert remapped[0]["confidence"] == 0.9
    assert remapped[0]["bbox"] == [40.0, 30.0, 60.0, 50.0]
    assert remapped[0]["coordinate_space"] == "full_frame"
    assert remapped[0]["source_person_track_id"] == "right"
    assert remapped[0]["keypoints"][0]["x"] == 45
    assert remapped[0]["keypoints"][0]["y"] == 35


def test_positional_crop_results_require_one_result_set_per_crop():
    policy = PersonCropPolicy(min_person_width=10, min_person_height=10)
    plan = plan_person_crops(_frame(), [_person([20, 10, 60, 90])], policy)

    with pytest.raises(ValueError):
        remap_crop_detections(plan, [])


def test_off_and_shadow_modes_preserve_full_frame_authority():
    policy = PersonCropPolicy(min_person_width=10, min_person_height=10)
    plan = plan_person_crops(_frame(), [_person([20, 10, 60, 90])], policy)

    off = decide_crop_execution("off", plan)
    shadow = decide_crop_execution("shadow", plan)

    assert off.run_person_crops is False
    assert off.run_full_frame_specialist is True
    assert off.full_frame_evidence_authoritative is True
    assert shadow.run_person_crops is True
    assert shadow.run_full_frame_specialist is True
    assert shadow.crop_evidence_authoritative is False
    assert shadow.full_frame_evidence_authoritative is True


def test_confirm_requires_fresh_full_frame_only_for_candidate_or_fallback():
    policy = PersonCropPolicy(min_person_width=10, min_person_height=10)
    plan = plan_person_crops(_frame(), [_person([20, 10, 60, 90])], policy)

    negative = decide_crop_execution("confirm", plan)
    candidate = decide_crop_execution("confirm", plan, crop_candidate=True)

    assert negative.run_person_crops is True
    assert negative.run_full_frame_specialist is False
    assert negative.crop_evidence_authoritative is False
    assert candidate.run_full_frame_specialist is True
    assert candidate.requires_full_frame_confirmation is True
    assert candidate.crop_evidence_authoritative is False


def test_active_uses_crops_only_when_every_person_is_safe():
    policy = PersonCropPolicy(min_person_width=10, min_person_height=10)
    good_plan = plan_person_crops(_frame(), [_person([20, 10, 60, 90])], policy)
    boundary_plan = plan_person_crops(_frame(), [_person([0, 10, 40, 90])], policy)

    active = decide_crop_execution("active", good_plan)
    fallback = decide_crop_execution("active", boundary_plan)
    failed = decide_crop_execution("active", good_plan, crop_failed=True)

    assert active.run_person_crops is True
    assert active.run_full_frame_specialist is False
    assert active.crop_evidence_authoritative is True
    assert fallback.run_person_crops is False
    assert fallback.run_full_frame_specialist is True
    assert fallback.full_frame_evidence_authoritative is True
    assert failed.reasons == (FALLBACK_CROP_FAILED,)
    assert failed.run_full_frame_specialist is True


def test_invalid_policy_and_mode_are_rejected():
    with pytest.raises(ValueError):
        PersonCropPolicy(max_crops=0)
    with pytest.raises(ValueError):
        decide_crop_execution("enabled")


def test_missing_plan_fails_safe_to_full_frame_in_active_mode():
    decision = decide_crop_execution("active")

    assert decision.run_person_crops is False
    assert decision.run_full_frame_specialist is True
    assert decision.crop_evidence_authoritative is False
    assert decision.reasons == (FALLBACK_MISSING_PLAN,)
