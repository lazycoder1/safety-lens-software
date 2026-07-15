"""Safe person-crop planning for PPE and phone specialist models.

This module only prepares/remaps crops and decides which path may provide fresh
evidence.  It does not call a model, so callers can shadow the crop path beside
the existing full-frame specialist before making it authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np


BBox = tuple[float, float, float, float]
CropMode = Literal["off", "shadow", "confirm", "active"]
VALID_CROP_MODES = frozenset({"off", "shadow", "confirm", "active"})

FALLBACK_INVALID_PERSON = "invalid_person_bbox"
FALLBACK_SMALL_PERSON = "small_person"
FALLBACK_BOUNDARY_PERSON = "boundary_person"
FALLBACK_CROWD_LIMIT = "crowd_limit"
FALLBACK_CROP_FAILED = "crop_inference_failed"
FALLBACK_MISSING_PLAN = "missing_crop_plan"


@dataclass(frozen=True)
class PersonCropPolicy:
    """Geometry limits used before a specialist is allowed to replace full-frame."""

    padding_fraction: float = 0.12
    min_person_width: int = 24
    min_person_height: int = 48
    boundary_margin: int = 2
    max_crops: int = 8
    person_dedup_iou: float = 0.85
    result_dedup_iou: float = 0.55

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.padding_fraction)) or not 0 <= float(
            self.padding_fraction
        ) <= 1:
            raise ValueError("padding_fraction must be between 0 and 1")
        for name in ("min_person_width", "min_person_height", "max_crops"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.boundary_margin, bool) or int(self.boundary_margin) < 0:
            raise ValueError("boundary_margin must be a non-negative integer")
        for name in ("person_dedup_iou", "result_dedup_iou"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class PersonCrop:
    """One copied BGR/gray crop and its full-frame coordinate transform."""

    crop_id: str
    source_detection_index: int
    source_bbox: BBox
    crop_bbox: tuple[int, int, int, int]
    image: np.ndarray
    confidence: float
    track_id: str | None = None
    boundary_person: bool = False
    padding_clipped: bool = False

    @property
    def width(self) -> int:
        return self.crop_bbox[2] - self.crop_bbox[0]

    @property
    def height(self) -> int:
        return self.crop_bbox[3] - self.crop_bbox[1]


@dataclass(frozen=True)
class PersonCropPlan:
    """A bounded crop plan plus reasons to retain the full-frame fallback."""

    crops: tuple[PersonCrop, ...]
    person_detection_count: int
    valid_person_count: int
    fallback_reasons: tuple[str, ...] = ()
    skipped_detection_indices: tuple[int, ...] = ()

    @property
    def fallback_required(self) -> bool:
        return bool(self.fallback_reasons)


@dataclass(frozen=True)
class CropExecutionDecision:
    """Which specialist path runs and which path may become alert evidence."""

    mode: CropMode
    run_person_crops: bool
    run_full_frame_specialist: bool
    crop_evidence_authoritative: bool
    full_frame_evidence_authoritative: bool
    requires_full_frame_confirmation: bool
    fallback_required: bool
    reasons: tuple[str, ...] = ()


def _frame_shape(frame: np.ndarray) -> tuple[int, int]:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError("frame must be a non-empty numpy image")
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("frame must have positive dimensions")
    return height, width


def _bbox(value: Any, width: int, height: int) -> BBox | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    x1 = min(float(width), max(0.0, x1))
    x2 = min(float(width), max(0.0, x2))
    y1 = min(float(height), max(0.0, y1))
    y2 = min(float(height), max(0.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _confidence(detection: Mapping[str, Any]) -> float:
    try:
        value = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _iou(left: BBox, right: BBox) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def plan_person_crops(
    frame: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    policy: PersonCropPolicy | None = None,
) -> PersonCropPlan:
    """Build clipped, padded person crops with bounded work and safe fallbacks."""

    height, width = _frame_shape(frame)
    policy = policy or PersonCropPolicy()
    people: list[tuple[int, Mapping[str, Any], BBox, float]] = []
    reasons: list[str] = []
    skipped: list[int] = []
    person_detection_count = 0

    for index, detection in enumerate(detections):
        if str(detection.get("class") or "").strip().lower() != "person":
            continue
        person_detection_count += 1
        person_bbox = _bbox(detection.get("bbox"), width, height)
        if person_bbox is None:
            _append_reason(reasons, FALLBACK_INVALID_PERSON)
            skipped.append(index)
            continue
        people.append((index, detection, person_bbox, _confidence(detection)))

    # Prefer the strongest representation when two detectors report the same person.
    people.sort(key=lambda item: (-item[3], item[0]))
    deduplicated: list[tuple[int, Mapping[str, Any], BBox, float]] = []
    for person in people:
        if any(_iou(person[2], accepted[2]) >= policy.person_dedup_iou for accepted in deduplicated):
            skipped.append(person[0])
            continue
        deduplicated.append(person)

    valid_people: list[tuple[int, Mapping[str, Any], BBox, float, bool]] = []
    margin = int(policy.boundary_margin)
    for index, detection, person_bbox, confidence in deduplicated:
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        if person_width < int(policy.min_person_width) or person_height < int(
            policy.min_person_height
        ):
            _append_reason(reasons, FALLBACK_SMALL_PERSON)
            skipped.append(index)
            continue
        boundary = (
            person_bbox[0] <= margin
            or person_bbox[1] <= margin
            or person_bbox[2] >= width - margin
            or person_bbox[3] >= height - margin
        )
        if boundary:
            _append_reason(reasons, FALLBACK_BOUNDARY_PERSON)
        valid_people.append((index, detection, person_bbox, confidence, boundary))

    valid_person_count = len(valid_people)
    if valid_person_count > int(policy.max_crops):
        _append_reason(reasons, FALLBACK_CROWD_LIMIT)
        skipped.extend(person[0] for person in valid_people[int(policy.max_crops) :])
        valid_people = valid_people[: int(policy.max_crops)]

    crops: list[PersonCrop] = []
    for index, detection, person_bbox, confidence, boundary in valid_people:
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        pad_x = person_width * float(policy.padding_fraction)
        pad_y = person_height * float(policy.padding_fraction)
        raw_crop = (
            person_bbox[0] - pad_x,
            person_bbox[1] - pad_y,
            person_bbox[2] + pad_x,
            person_bbox[3] + pad_y,
        )
        x1 = max(0, int(math.floor(raw_crop[0])))
        y1 = max(0, int(math.floor(raw_crop[1])))
        x2 = min(width, int(math.ceil(raw_crop[2])))
        y2 = min(height, int(math.ceil(raw_crop[3])))
        if x2 <= x1 or y2 <= y1:
            _append_reason(reasons, FALLBACK_INVALID_PERSON)
            skipped.append(index)
            continue
        track_id = detection.get("track_id", detection.get("id"))
        crops.append(
            PersonCrop(
                crop_id=f"person-{index}",
                source_detection_index=index,
                source_bbox=person_bbox,
                crop_bbox=(x1, y1, x2, y2),
                image=np.ascontiguousarray(frame[y1:y2, x1:x2]).copy(),
                confidence=confidence,
                track_id=None if track_id is None else str(track_id),
                boundary_person=boundary,
                padding_clipped=(
                    x1 > raw_crop[0]
                    or y1 > raw_crop[1]
                    or x2 < raw_crop[2]
                    or y2 < raw_crop[3]
                ),
            )
        )

    return PersonCropPlan(
        crops=tuple(crops),
        person_detection_count=person_detection_count,
        valid_person_count=valid_person_count,
        fallback_reasons=tuple(reasons),
        skipped_detection_indices=tuple(sorted(set(skipped))),
    )


def _remap_keypoints(value: Any, offset_x: int, offset_y: int) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    remapped: list[Any] = []
    for point in value:
        if isinstance(point, Mapping) and "x" in point and "y" in point:
            updated = dict(point)
            try:
                updated["x"] = float(point["x"]) + offset_x
                updated["y"] = float(point["y"]) + offset_y
            except (TypeError, ValueError):
                pass
            remapped.append(updated)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                remapped.append(
                    [float(point[0]) + offset_x, float(point[1]) + offset_y, *point[2:]]
                )
            except (TypeError, ValueError):
                remapped.append(point)
        else:
            remapped.append(point)
    return remapped


def deduplicate_remapped_detections(
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Suppress same-class specialist duplicates after all boxes are full-frame."""

    if not math.isfinite(float(iou_threshold)) or not 0 <= float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-_confidence(item[1]), item[0]),
    )
    kept: list[dict[str, Any]] = []
    for _, detection in ranked:
        candidate = detection.get("bbox")
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            continue
        try:
            candidate_bbox = tuple(float(value) for value in candidate)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in candidate_bbox):
            continue
        if candidate_bbox[2] <= candidate_bbox[0] or candidate_bbox[3] <= candidate_bbox[1]:
            continue
        candidate_class = str(detection.get("class") or "").strip().lower()
        duplicate = False
        for existing in kept:
            if str(existing.get("class") or "").strip().lower() != candidate_class:
                continue
            existing_bbox = tuple(float(value) for value in existing["bbox"])
            if _iou(candidate_bbox, existing_bbox) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def remap_crop_detections(
    plan: PersonCropPlan,
    crop_results: Mapping[str, Sequence[Mapping[str, Any]]]
    | Sequence[Sequence[Mapping[str, Any]]],
    *,
    dedup_iou_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """Map crop-local specialist boxes/keypoints into full-frame coordinates."""

    if isinstance(crop_results, Mapping):
        result_sets = [crop_results.get(crop.crop_id, ()) for crop in plan.crops]
    else:
        result_sets = list(crop_results)
        if len(result_sets) != len(plan.crops):
            raise ValueError("crop_results must contain one result set per planned crop")

    remapped: list[dict[str, Any]] = []
    for crop, detections in zip(plan.crops, result_sets, strict=True):
        for detection in detections:
            local_bbox = _bbox(detection.get("bbox"), crop.width, crop.height)
            if local_bbox is None:
                continue
            offset_x, offset_y = crop.crop_bbox[:2]
            full_bbox = [
                local_bbox[0] + offset_x,
                local_bbox[1] + offset_y,
                local_bbox[2] + offset_x,
                local_bbox[3] + offset_y,
            ]
            record = dict(detection)
            record.update(
                bbox=full_bbox,
                coordinate_space="full_frame",
                source_crop_id=crop.crop_id,
                source_person_bbox=list(crop.source_bbox),
                source_person_track_id=crop.track_id,
            )
            if "keypoints" in record:
                record["keypoints"] = _remap_keypoints(
                    record["keypoints"],
                    offset_x,
                    offset_y,
                )
            remapped.append(record)
    return deduplicate_remapped_detections(
        remapped,
        iou_threshold=dedup_iou_threshold,
    )


def decide_crop_execution(
    mode: CropMode,
    plan: PersonCropPlan | None = None,
    *,
    crop_candidate: bool = False,
    crop_failed: bool = False,
) -> CropExecutionDecision:
    """Apply off/shadow/confirm/active evidence semantics without model calls."""

    if mode not in VALID_CROP_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_CROP_MODES)}")
    has_crops = bool(plan and plan.crops)
    missing_plan = plan is None
    plan_fallback = bool(plan and plan.fallback_required)
    fallback_required = missing_plan or plan_fallback or crop_failed
    reasons = list(plan.fallback_reasons if plan else ())
    if missing_plan:
        _append_reason(reasons, FALLBACK_MISSING_PLAN)
    if crop_failed:
        _append_reason(reasons, FALLBACK_CROP_FAILED)

    if mode == "off":
        return CropExecutionDecision(
            mode="off",
            run_person_crops=False,
            run_full_frame_specialist=True,
            crop_evidence_authoritative=False,
            full_frame_evidence_authoritative=True,
            requires_full_frame_confirmation=False,
            fallback_required=fallback_required,
            reasons=tuple(reasons),
        )
    if mode == "shadow":
        return CropExecutionDecision(
            mode="shadow",
            run_person_crops=has_crops and not crop_failed,
            run_full_frame_specialist=True,
            crop_evidence_authoritative=False,
            full_frame_evidence_authoritative=True,
            requires_full_frame_confirmation=False,
            fallback_required=fallback_required,
            reasons=tuple(reasons),
        )
    if mode == "confirm":
        return CropExecutionDecision(
            mode="confirm",
            run_person_crops=has_crops and not crop_failed,
            run_full_frame_specialist=(
                fallback_required or crop_candidate or not has_crops
            ),
            crop_evidence_authoritative=False,
            full_frame_evidence_authoritative=True,
            requires_full_frame_confirmation=crop_candidate and not crop_failed,
            fallback_required=fallback_required,
            reasons=tuple(reasons),
        )

    use_crops = has_crops and not fallback_required
    return CropExecutionDecision(
        mode="active",
        run_person_crops=use_crops,
        run_full_frame_specialist=fallback_required,
        crop_evidence_authoritative=use_crops,
        full_frame_evidence_authoritative=fallback_required,
        requires_full_frame_confirmation=False,
        fallback_required=fallback_required,
        reasons=tuple(reasons),
    )
