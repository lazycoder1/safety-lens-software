"""Lightweight helmet-colour classification for detected helmet crops.

The PPE detector remains responsible for finding a helmet.  This module only
examines pixels inside that detector box, so it adds no neural model and stays
cheap enough to run whenever the gated PPE specialist produces a fresh result.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

SUPPORTED_HELMET_COLOURS = (
    "white",
    "yellow",
    "orange",
    "red",
    "blue",
    "green",
    "black",
)
HELMET_COLOUR_CAPABILITY = "helmet_color_compliance"
DEFAULT_MIN_CONFIDENCE = 0.45
DEFAULT_CONFIRMATION_THRESHOLD = 3
DEFAULT_MIN_CROP_PIXELS = 64
HELMET_CLASS_TERMS = {
    "hard hat",
    "safety helmet",
    "helmet",
    "rider helmet",
    "motorcycle helmet",
}

_COLOUR_ALIASES = {
    "amber": "orange",
    "hi vis yellow": "yellow",
    "hi-vis yellow": "yellow",
    "navy": "blue",
}


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def normalize_helmet_colour(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", " ")
    normalized = _COLOUR_ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_HELMET_COLOURS else None


def normalize_helmet_colour_policy(
    value: Any,
    *,
    enabled: bool | None = None,
) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    allowed: list[str] = []
    raw_allowed = raw.get("allowed_colours")
    if raw_allowed is None:
        raw_allowed = raw.get("allowed_colors")
    if isinstance(raw_allowed, str):
        raw_allowed = raw_allowed.split(",")
    if isinstance(raw_allowed, list):
        for item in raw_allowed:
            colour = normalize_helmet_colour(item)
            if colour and colour not in allowed:
                allowed.append(colour)

    min_confidence = min(
        1.0,
        max(0.05, _number(raw.get("min_confidence"), DEFAULT_MIN_CONFIDENCE)),
    )
    threshold = _positive_int(
        raw.get("confirmation_threshold", raw.get("threshold")),
        DEFAULT_CONFIRMATION_THRESHOLD,
    )
    configured_enabled = bool(raw.get("enabled", True))

    return {
        "enabled": configured_enabled if enabled is None else bool(enabled),
        "allowed_colours": allowed,
        "min_confidence": round(min_confidence, 4),
        "confirmation_threshold": threshold,
        "severity": str(raw.get("severity") or "P2"),
    }


def _clipped_crop(frame: np.ndarray, bbox: list | tuple) -> np.ndarray | None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None
    height, width = frame.shape[:2]
    x1 = min(width, max(0, x1))
    x2 = min(width, max(0, x2))
    y1 = min(height, max(0, y1))
    y2 = min(height, max(0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def classify_helmet_colour(
    frame: np.ndarray,
    bbox: list | tuple,
    *,
    min_crop_pixels: int = DEFAULT_MIN_CROP_PIXELS,
) -> dict[str, Any]:
    """Classify one helmet box into a conservative fixed colour vocabulary.

    Confidence is the winning colour's share of the central elliptical crop.
    Callers should fail open below their configured threshold; low-confidence
    results are still returned as ``unknown`` for telemetry and tuning.
    """

    crop = _clipped_crop(frame, bbox)
    if crop is None or crop.shape[0] * crop.shape[1] < max(1, min_crop_pixels):
        return {
            "colour": "unknown",
            "confidence": 0.0,
            "sample_pixels": 0,
            "colour_pixel_ratio": 0.0,
            "reason": "crop_too_small",
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    height, width = hue.shape
    yy, xx = np.ogrid[:height, :width]
    ellipse = (
        ((xx - (width - 1) / 2.0) / max(1.0, width * 0.5)) ** 2
        + ((yy - (height - 1) / 2.0) / max(1.0, height * 0.5)) ** 2
        <= 1.0
    )

    masks = {
        "red": (((hue <= 10) | (hue >= 170)) & (saturation >= 70) & (value >= 55)),
        "orange": ((hue >= 11) & (hue <= 23) & (saturation >= 70) & (value >= 60)),
        "yellow": ((hue >= 24) & (hue <= 38) & (saturation >= 65) & (value >= 70)),
        "green": ((hue >= 39) & (hue <= 90) & (saturation >= 55) & (value >= 45)),
        "blue": ((hue >= 91) & (hue <= 135) & (saturation >= 55) & (value >= 45)),
        "white": ((saturation <= 45) & (value >= 145)),
        "black": (value <= 60),
    }
    counts = {
        colour: int(np.count_nonzero(mask & ellipse))
        for colour, mask in masks.items()
    }
    sample_pixels = int(np.count_nonzero(ellipse))
    if sample_pixels <= 0:
        return {
            "colour": "unknown",
            "confidence": 0.0,
            "sample_pixels": 0,
            "colour_pixel_ratio": 0.0,
            "reason": "empty_crop",
        }

    colour = max(counts, key=counts.get)
    winning_pixels = counts[colour]
    confidence = winning_pixels / sample_pixels
    if winning_pixels <= 0:
        colour = "unknown"

    return {
        "colour": colour,
        "confidence": round(float(confidence), 4),
        "sample_pixels": sample_pixels,
        "colour_pixel_ratio": round(float(confidence), 4),
        "reason": "classified" if colour != "unknown" else "no_colour_pixels",
    }


def annotate_helmet_colours(
    frame: np.ndarray,
    detections: list[dict],
    camera: dict,
) -> list[dict]:
    """Attach helmet-colour evidence to fresh helmet detections in place."""

    capabilities = set(camera.get("capabilities") or [])
    policy = normalize_helmet_colour_policy(
        camera.get("helmet_colour_policy") or camera.get("helmet_color_policy"),
        enabled=HELMET_COLOUR_CAPABILITY in capabilities,
    )
    if not policy["enabled"]:
        return detections

    allowed = set(policy["allowed_colours"])
    min_confidence = float(policy["min_confidence"])
    for detection in detections:
        class_name = str(detection.get("class") or "").strip().lower()
        if class_name not in HELMET_CLASS_TERMS or not detection.get("bbox"):
            continue
        result = classify_helmet_colour(frame, detection["bbox"])
        detected_colour = result["colour"]
        colour_confidence = float(result["confidence"])
        accepted = detected_colour != "unknown" and colour_confidence >= min_confidence
        detection["helmet_colour"] = detected_colour if accepted else "unknown"
        detection["helmet_colour_confidence"] = round(colour_confidence, 4)
        detection["helmet_colour_analysis"] = result
        detection["helmet_colour_compliant"] = (
            detection["helmet_colour"] in allowed
            if accepted and allowed
            else None
        )
    return detections
