"""Focused tests for the gated helmet-colour post-process and compliance rule."""

import json
from unittest import mock

import numpy as np
import pytest

from camera_planner import build_execution_plan
from detection import (
    HELMET_COLOUR_MISMATCH_RULE,
    check_yoloe_violations,
    extract_violation_bboxes,
)
from helmet_colour import (
    annotate_helmet_colours,
    classify_helmet_colour,
    normalize_helmet_colour_policy,
)
from routers.safety_rules import DEFAULT_SAFETY_RULES


HELMET_ENGINE_PROMPTS = ["motorcycle helmet", "rider helmet", "helmet"]


@pytest.mark.parametrize(
    ("expected", "bgr"),
    [
        ("white", (240, 240, 240)),
        ("yellow", (0, 255, 255)),
        ("orange", (0, 128, 255)),
        ("red", (0, 0, 255)),
        ("blue", (255, 0, 0)),
        ("green", (0, 180, 0)),
        ("black", (20, 20, 20)),
    ],
)
def test_classify_helmet_colour_fixed_vocabulary(expected, bgr):
    frame = np.full((80, 80, 3), bgr, dtype=np.uint8)

    result = classify_helmet_colour(frame, [8, 8, 72, 72])

    assert result["colour"] == expected
    assert result["confidence"] >= 0.95
    assert result["sample_pixels"] > 0


def test_classify_helmet_colour_fails_open_for_tiny_crop():
    frame = np.full((20, 20, 3), (0, 0, 255), dtype=np.uint8)

    result = classify_helmet_colour(frame, [1, 1, 4, 4])

    assert result == {
        "colour": "unknown",
        "confidence": 0.0,
        "sample_pixels": 0,
        "colour_pixel_ratio": 0.0,
        "reason": "crop_too_small",
    }


def test_normalize_policy_accepts_color_alias_and_bounds_values():
    policy = normalize_helmet_colour_policy({
        "allowed_colors": ["amber", "WHITE", "not-a-colour", "white"],
        "min_confidence": 4,
        "threshold": 0,
        "severity": "P3",
    })

    assert policy == {
        "enabled": True,
        "allowed_colours": ["orange", "white"],
        "min_confidence": 1.0,
        "confirmation_threshold": 3,
        "severity": "P3",
    }


def test_worker_helmet_and_colour_share_the_jetson_fixed_prompt_profile():
    plan = build_execution_plan(
        {
            "profile": "work_zone_ppe",
            "capabilities": ["helmet_required", "helmet_color_compliance"],
            "safety_rule_ids": ["ppe_helmet"],
        },
        {"safety_rules": list(DEFAULT_SAFETY_RULES)},
    )

    assert plan["ppe_prompt_terms"] == HELMET_ENGINE_PROMPTS


def test_annotate_helmet_colours_marks_compliance_without_touching_people():
    frame = np.full((220, 160, 3), (0, 0, 255), dtype=np.uint8)
    detections = [
        {"class": "person", "confidence": 0.9, "bbox": [10, 10, 120, 210]},
        {"class": "hard hat", "confidence": 0.88, "bbox": [40, 20, 90, 70]},
    ]
    camera = {
        "capabilities": ["helmet_color_compliance"],
        "helmet_colour_policy": {
            "allowed_colours": ["white"],
            "min_confidence": 0.45,
        },
    }

    annotate_helmet_colours(frame, detections, camera)

    assert "helmet_colour" not in detections[0]
    assert detections[1]["helmet_colour"] == "red"
    assert detections[1]["helmet_colour_confidence"] >= 0.95
    assert detections[1]["helmet_colour_compliant"] is False


def _colour_config(allowed_colours):
    return {
        "cameras": {
            "cam1": {
                "capabilities": ["helmet_required", "helmet_color_compliance"],
                "safety_rule_ids": ["ppe_helmet"],
                "helmet_colour_policy": {
                    "enabled": True,
                    "allowed_colours": allowed_colours,
                    "min_confidence": 0.45,
                    "confirmation_threshold": 3,
                    "severity": "P2",
                },
            },
        },
        "safety_rules": list(DEFAULT_SAFETY_RULES),
    }


def _colour_detections(
    colour="red",
    compliant=False,
    colour_confidence=0.86,
    helmet_class="hard hat",
):
    return [
        {
            "class": "person",
            "confidence": 0.92,
            "bbox": [10, 10, 120, 220],
            "model_family": "coco_primary",
        },
        {
            "class": helmet_class,
            "confidence": 0.88,
            "bbox": [40, 20, 90, 70],
            "model_family": "ppe_specialist",
            "helmet_colour": colour,
            "helmet_colour_confidence": colour_confidence,
            "helmet_colour_compliant": compliant,
        },
    ]


@mock.patch("detection.get_config")
def test_check_yoloe_violations_emits_repeated_wrong_colour_candidate(mock_config):
    mock_config.return_value = _colour_config(["white", "yellow"])

    candidates = check_yoloe_violations(
        _colour_detections(),
        "cam1",
        frame_w=200,
        frame_h=240,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["rule"] == HELMET_COLOUR_MISMATCH_RULE
    assert candidate["threshold"] == 3
    assert candidate["confidence"] == 0.86
    assert candidate["metadata"]["allowedHelmetColours"] == ["white", "yellow"]
    assert candidate["metadata"]["observedHelmetColours"] == ["red"]
    json.dumps(candidate["metadata"])


@mock.patch("detection.get_config")
def test_check_yoloe_violations_accepts_allowed_colour(mock_config):
    mock_config.return_value = _colour_config(["red"])

    assert check_yoloe_violations(
        _colour_detections(colour="red", compliant=True),
        "cam1",
        frame_w=200,
        frame_h=240,
    ) == []


@mock.patch("detection.get_config")
def test_generic_helmet_counts_for_worker_presence_and_colour(mock_config):
    mock_config.return_value = _colour_config(["yellow"])

    assert check_yoloe_violations(
        _colour_detections(
            colour="yellow",
            compliant=True,
            helmet_class="helmet",
        ),
        "cam1",
        frame_w=200,
        frame_h=240,
    ) == []


@mock.patch("detection.get_config")
def test_check_yoloe_violations_ignores_uncertain_colour(mock_config):
    mock_config.return_value = _colour_config(["white"])

    assert check_yoloe_violations(
        _colour_detections(
            colour="unknown",
            compliant=None,
            colour_confidence=0.2,
        ),
        "cam1",
        frame_w=200,
        frame_h=240,
    ) == []


def test_extract_violation_bboxes_labels_observed_colour():
    bboxes = extract_violation_bboxes(
        HELMET_COLOUR_MISMATCH_RULE,
        _colour_detections(),
        frame_w=200,
        frame_h=240,
        camera_id="cam1",
    )

    assert bboxes == [{
        "label": "Wrong Helmet Colour — Red",
        "bbox": [0.2, 0.0833, 0.45, 0.2917],
        "confidence": 0.86,
    }]
