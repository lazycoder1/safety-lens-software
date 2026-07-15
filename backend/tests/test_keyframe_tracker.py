import cv2
import numpy as np
import pytest

from keyframe_tracker import (
    PersonKLTTracker,
    REDETECT_FRAME_GAP,
    REDETECT_LOW_CONFIDENCE,
    REDETECT_NEW_FOREGROUND,
    REDETECT_NO_KEYFRAME,
    REDETECT_NO_TRACKS,
    REDETECT_SCENE_CUT,
    REDETECT_ZONE_ENTRY,
)


def _textured_person_frame(width=160, height=120):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(24, 78, 8):
        for x in range(34, 68, 8):
            colour = (255, 255, 255) if (x + y) % 16 else (80, 180, 240)
            cv2.circle(frame, (x, y), 2, colour, -1)
    cv2.rectangle(frame, (30, 20), (70, 80), (90, 90, 90), 1)
    return frame


def _shift(frame, dx, dy=0):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def _person():
    return {
        "class": "person",
        "bbox": [30, 20, 70, 80],
        "confidence": 0.91,
        "track_id": "primary-7",
    }


def test_tracks_only_people_and_marks_projection_as_non_alert_evidence():
    frame = _textured_person_frame()
    tracker = PersonKLTTracker(confidence_threshold=0.35)
    count = tracker.seed(
        frame,
        [_person(), {"class": "phone", "bbox": [40, 40, 50, 55]}],
        1.0,
    )

    result = tracker.project(_shift(frame, 5), 1.1)

    assert count == 1
    assert result.force_redetect is False
    assert len(result.projections) == 1
    projection = result.projections[0]
    assert projection.track_id == "primary-7"
    assert projection.bbox[0] == pytest.approx(35, abs=1.5)
    assert projection.fresh_detection is False
    assert projection.fresh_alert_evidence is False
    assert projection.alert_eligible is False
    assert result.detections[0]["observation_kind"] == "tracker_projection"
    assert result.detections[0]["alert_eligible"] is False


def test_project_without_detector_keyframe_requests_immediate_detection():
    result = PersonKLTTracker().project(_textured_person_frame(), 1.0)

    assert result.force_redetect is True
    assert result.reasons == (REDETECT_NO_KEYFRAME,)
    assert result.projections == ()


def test_frame_gap_discards_tracks_and_requests_fresh_keyframe():
    frame = _textured_person_frame()
    tracker = PersonKLTTracker(max_frame_gap_seconds=0.2)
    tracker.seed(frame, [_person()], 1.0)

    result = tracker.project(frame, 1.21)
    after_clear = tracker.project(frame, 1.22)

    assert result.frame_gap is True
    assert result.reasons == (REDETECT_FRAME_GAP,)
    assert after_clear.reasons == (REDETECT_NO_KEYFRAME,)


def test_scene_cut_discards_projections():
    frame = _textured_person_frame()
    tracker = PersonKLTTracker(scene_cut_threshold=0.25)
    tracker.seed(frame, [_person()], 1.0)

    result = tracker.project(np.full_like(frame, 255), 1.1)

    assert result.scene_cut is True
    assert result.force_redetect is True
    assert result.reasons == (REDETECT_SCENE_CUT,)
    assert result.projections == ()


def test_lost_features_trigger_low_confidence_fallback():
    frame = _textured_person_frame()
    tracker = PersonKLTTracker(
        confidence_threshold=0.8,
        scene_cut_threshold=0.9,
    )
    tracker.seed(frame, [_person()], 1.0)

    result = tracker.project(np.zeros_like(frame), 1.1)

    assert result.force_redetect is True
    assert REDETECT_LOW_CONFIDENCE in result.reasons
    assert result.aggregate_confidence < 0.8


def test_unexplained_new_foreground_forces_detection():
    frame = _textured_person_frame()
    current = frame.copy()
    cv2.rectangle(current, (110, 70), (140, 100), (255, 255, 255), -1)
    tracker = PersonKLTTracker(
        confidence_threshold=0.3,
        new_foreground_min_area=100,
    )
    tracker.seed(frame, [_person()], 1.0)

    result = tracker.project(current, 1.1)

    assert result.new_foreground is True
    assert result.force_redetect is True
    assert REDETECT_NEW_FOREGROUND in result.reasons


def test_new_foreground_after_empty_keyframe_forces_immediate_detection():
    empty = np.zeros((120, 160, 3), dtype=np.uint8)
    entered = empty.copy()
    cv2.rectangle(entered, (70, 30), (110, 100), (255, 255, 255), -1)
    tracker = PersonKLTTracker(
        scene_cut_threshold=0.9,
        new_foreground_min_area=100,
    )
    assert tracker.seed(empty, [], 1.0) == 0

    result = tracker.project(entered, 1.1)

    assert result.new_foreground is True
    assert result.force_redetect is True
    assert result.reasons == (REDETECT_NO_TRACKS, REDETECT_NEW_FOREGROUND)


def test_projected_zone_entry_forces_fresh_detection():
    frame = _textured_person_frame()
    zone = [[55, 0], [100, 0], [100, 110], [55, 110]]
    tracker = PersonKLTTracker(confidence_threshold=0.3)
    tracker.seed(frame, [_person()], 1.0, zones=[zone])

    result = tracker.project(_shift(frame, 12), 1.1, zones=[zone])

    assert result.zone_entry is True
    assert result.force_redetect is True
    assert REDETECT_ZONE_ENTRY in result.reasons


def test_normalised_zones_and_invalid_timestamps_are_supported_safely():
    frame = _textured_person_frame()
    zone = {"points": [[0.35, 0], [0.8, 0], [0.8, 1], [0.35, 1]]}
    tracker = PersonKLTTracker(confidence_threshold=0.3)
    tracker.seed(frame, [_person()], 1.0, zones=[zone])
    tracker.project(_shift(frame, 12), 1.1, zones=[zone])

    with pytest.raises(ValueError):
        tracker.project(frame, 1.0)
