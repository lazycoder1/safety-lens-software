"""Conservative person-only optical-flow projections between detector keyframes.

The tracker is deliberately an optimisation primitive, not a detector.  Its
outputs are marked as stale projections and are never eligible to create or
confirm an alert.  Any ambiguity asks the caller for a fresh detector keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


BBox = tuple[float, float, float, float]

REDETECT_NO_KEYFRAME = "no_keyframe"
REDETECT_NO_TRACKS = "no_person_tracks"
REDETECT_FRAME_GAP = "frame_gap"
REDETECT_FRAME_GEOMETRY = "frame_geometry_change"
REDETECT_SCENE_CUT = "scene_cut"
REDETECT_LOW_CONFIDENCE = "low_tracker_confidence"
REDETECT_NEW_FOREGROUND = "new_foreground"
REDETECT_ZONE_ENTRY = "zone_entry"


@dataclass(frozen=True)
class TrackerProjection:
    """A non-authoritative person projection for display/association only."""

    track_id: str
    bbox: BBox
    confidence: float
    source_keyframe_timestamp: float
    timestamp: float
    class_name: str = "person"
    observation_kind: str = "tracker_projection"
    model_family: str = "tracker_projection"
    fresh_detection: bool = False
    fresh_alert_evidence: bool = False
    alert_eligible: bool = False

    def as_detection(self) -> dict[str, Any]:
        """Return a detection-shaped record that retains safety provenance."""

        return {
            "class": self.class_name,
            "bbox": [round(value, 3) for value in self.bbox],
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
            "model_family": self.model_family,
            "observation_kind": self.observation_kind,
            "fresh_detection": self.fresh_detection,
            "fresh_alert_evidence": self.fresh_alert_evidence,
            "alert_eligible": self.alert_eligible,
            "source_keyframe_timestamp": self.source_keyframe_timestamp,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class KeyframeTrackerResult:
    """One projection pass and the reasons a fresh detector frame is needed."""

    projections: tuple[TrackerProjection, ...] = ()
    aggregate_confidence: float = 0.0
    force_redetect: bool = False
    reasons: tuple[str, ...] = ()
    scene_cut: bool = False
    frame_gap: bool = False
    new_foreground: bool = False
    zone_entry: bool = False

    @property
    def detections(self) -> list[dict[str, Any]]:
        return [projection.as_detection() for projection in self.projections]


@dataclass
class _Track:
    track_id: str
    bbox: BBox
    points: np.ndarray
    source_keyframe_timestamp: float
    zone_memberships: frozenset[int] = field(default_factory=frozenset)


def _positive(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _fraction(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _timestamp(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError("timestamp must be finite")
    return parsed


def _gray(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty numpy array")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("frame must be grayscale, BGR, or BGRA")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(gray)


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
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _zone_points(zone: Any, width: int, height: int) -> np.ndarray | None:
    raw = zone
    if isinstance(zone, dict):
        raw = zone.get("points", zone.get("polygon", zone.get("vertices")))
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    points: list[tuple[float, float]] = []
    for point in raw:
        if isinstance(point, dict):
            values = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            values = point[0], point[1]
        else:
            return None
        try:
            x, y = float(values[0]), float(values[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((x, y))
    normalised = max(max(abs(x), abs(y)) for x, y in points) <= 1.5
    if normalised:
        points = [(x * width, y * height) for x, y in points]
    return np.asarray(points, dtype=np.float32)


def _memberships(
    bbox: BBox,
    zones: Iterable[Any],
    width: int,
    height: int,
) -> frozenset[int]:
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    memberships: set[int] = set()
    for index, zone in enumerate(zones):
        polygon = _zone_points(zone, width, height)
        if polygon is not None and cv2.pointPolygonTest(polygon, center, False) >= 0:
            memberships.add(index)
    return frozenset(memberships)


def _translated_box(bbox: BBox, dx: float, dy: float, width: int, height: int) -> BBox:
    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]
    x1 = min(max(0.0, bbox[0] + dx), max(0.0, width - box_width))
    y1 = min(max(0.0, bbox[1] + dy), max(0.0, height - box_height))
    return x1, y1, min(float(width), x1 + box_width), min(float(height), y1 + box_height)


class PersonKLTTracker:
    """Project fresh person boxes with KLT until a detector keyframe is due."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.55,
        max_frame_gap_seconds: float = 0.5,
        scene_cut_threshold: float = 0.35,
        max_forward_backward_error: float = 2.0,
        max_features_per_person: int = 40,
        min_features_per_person: int = 4,
        new_foreground_min_area: int = 144,
        new_foreground_threshold: int = 32,
    ) -> None:
        self.confidence_threshold = _fraction(
            confidence_threshold,
            "confidence_threshold",
        )
        self.max_frame_gap_seconds = _positive(
            max_frame_gap_seconds,
            "max_frame_gap_seconds",
        )
        self.scene_cut_threshold = _fraction(
            scene_cut_threshold,
            "scene_cut_threshold",
        )
        self.max_forward_backward_error = _positive(
            max_forward_backward_error,
            "max_forward_backward_error",
        )
        if isinstance(max_features_per_person, bool) or int(max_features_per_person) < 1:
            raise ValueError("max_features_per_person must be a positive integer")
        if isinstance(min_features_per_person, bool) or int(min_features_per_person) < 1:
            raise ValueError("min_features_per_person must be a positive integer")
        if int(min_features_per_person) > int(max_features_per_person):
            raise ValueError("min_features_per_person cannot exceed max_features_per_person")
        if isinstance(new_foreground_min_area, bool) or int(new_foreground_min_area) < 1:
            raise ValueError("new_foreground_min_area must be a positive integer")
        if not 1 <= int(new_foreground_threshold) <= 255:
            raise ValueError("new_foreground_threshold must be between 1 and 255")
        self.max_features_per_person = int(max_features_per_person)
        self.min_features_per_person = int(min_features_per_person)
        self.new_foreground_min_area = int(new_foreground_min_area)
        self.new_foreground_threshold = int(new_foreground_threshold)

        self._previous_gray: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def clear(self) -> None:
        self._previous_gray = None
        self._previous_timestamp = None
        self._tracks.clear()

    def _features(self, gray: np.ndarray, bbox: BBox) -> np.ndarray:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        margin_x = max(1, int((x2 - x1) * 0.05))
        margin_y = max(1, int((y2 - y1) * 0.05))
        cv2.rectangle(
            mask,
            (min(gray.shape[1] - 1, x1 + margin_x), min(gray.shape[0] - 1, y1 + margin_y)),
            (max(0, x2 - margin_x - 1), max(0, y2 - margin_y - 1)),
            255,
            -1,
        )
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features_per_person,
            qualityLevel=0.01,
            minDistance=4,
            mask=mask,
            blockSize=5,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32)

    def seed(
        self,
        frame: np.ndarray,
        detections: Sequence[dict[str, Any]],
        timestamp: float,
        zones: Iterable[Any] = (),
    ) -> int:
        """Replace state from a fresh detector keyframe and return person count."""

        gray = _gray(frame)
        observed_at = _timestamp(timestamp)
        height, width = gray.shape
        zone_list = tuple(zones)
        tracks: list[_Track] = []
        for detection in detections:
            if str(detection.get("class") or "").strip().lower() != "person":
                continue
            person_box = _bbox(detection.get("bbox"), width, height)
            if person_box is None:
                continue
            raw_id = detection.get("track_id", detection.get("id"))
            if raw_id is None:
                raw_id = self._next_track_id
                self._next_track_id += 1
            tracks.append(
                _Track(
                    track_id=str(raw_id),
                    bbox=person_box,
                    points=self._features(gray, person_box),
                    source_keyframe_timestamp=observed_at,
                    zone_memberships=_memberships(
                        person_box,
                        zone_list,
                        width,
                        height,
                    ),
                )
            )
        self._tracks = tracks
        self._previous_gray = gray.copy()
        self._previous_timestamp = observed_at
        return len(tracks)

    def _hard_fallback(
        self,
        reason: str,
        *,
        scene_cut: bool = False,
        frame_gap: bool = False,
    ) -> KeyframeTrackerResult:
        self.clear()
        return KeyframeTrackerResult(
            force_redetect=True,
            reasons=(reason,),
            scene_cut=scene_cut,
            frame_gap=frame_gap,
        )

    def _is_scene_cut(self, previous: np.ndarray, current: np.ndarray) -> bool:
        target = (64, 36)
        left = cv2.resize(previous, target, interpolation=cv2.INTER_AREA)
        right = cv2.resize(current, target, interpolation=cv2.INTER_AREA)
        score = float(np.mean(cv2.absdiff(left, right))) / 255.0
        return score >= self.scene_cut_threshold

    def _has_new_foreground(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        old_boxes: Sequence[BBox],
        new_boxes: Sequence[BBox],
    ) -> bool:
        changed = cv2.absdiff(previous, current)
        _, mask = cv2.threshold(
            changed,
            self.new_foreground_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        height, width = mask.shape
        for bbox in (*old_boxes, *new_boxes):
            margin_x = max(4, int((bbox[2] - bbox[0]) * 0.2))
            margin_y = max(4, int((bbox[3] - bbox[1]) * 0.2))
            x1 = max(0, int(math.floor(bbox[0])) - margin_x)
            y1 = max(0, int(math.floor(bbox[1])) - margin_y)
            x2 = min(width, int(math.ceil(bbox[2])) + margin_x)
            y2 = min(height, int(math.ceil(bbox[3])) + margin_y)
            mask[y1:y2, x1:x2] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return any(
            int(stats[index, cv2.CC_STAT_AREA]) >= self.new_foreground_min_area
            for index in range(1, count)
        )

    def project(
        self,
        frame: np.ndarray,
        timestamp: float,
        zones: Iterable[Any] = (),
    ) -> KeyframeTrackerResult:
        """Project all people once; ambiguity always requests fresh inference."""

        current = _gray(frame)
        observed_at = _timestamp(timestamp)
        if self._previous_gray is None or self._previous_timestamp is None:
            return KeyframeTrackerResult(
                force_redetect=True,
                reasons=(REDETECT_NO_KEYFRAME,),
            )
        if observed_at < self._previous_timestamp:
            raise ValueError("tracker timestamps must be monotonic")
        if current.shape != self._previous_gray.shape:
            return self._hard_fallback(REDETECT_FRAME_GEOMETRY)
        if observed_at - self._previous_timestamp > self.max_frame_gap_seconds:
            return self._hard_fallback(REDETECT_FRAME_GAP, frame_gap=True)
        if self._is_scene_cut(self._previous_gray, current):
            return self._hard_fallback(REDETECT_SCENE_CUT, scene_cut=True)
        if not self._tracks:
            new_foreground = self._has_new_foreground(
                self._previous_gray,
                current,
                (),
                (),
            )
            self._previous_gray = current.copy()
            self._previous_timestamp = observed_at
            reasons = [REDETECT_NO_TRACKS]
            if new_foreground:
                reasons.append(REDETECT_NEW_FOREGROUND)
            return KeyframeTrackerResult(
                force_redetect=True,
                reasons=tuple(reasons),
                new_foreground=new_foreground,
            )

        previous = self._previous_gray
        old_boxes = [track.bbox for track in self._tracks]
        zone_list = tuple(zones)
        height, width = current.shape
        projections: list[TrackerProjection] = []
        confidences: list[float] = []
        entered_zone = False

        for track in self._tracks:
            initial_count = len(track.points)
            confidence = 0.0
            translated = track.bbox
            good_next = np.empty((0, 1, 2), dtype=np.float32)
            if initial_count:
                next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                    previous,
                    current,
                    track.points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        20,
                        0.03,
                    ),
                )
                if next_points is not None and forward_status is not None:
                    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                        current,
                        previous,
                        next_points,
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            20,
                            0.03,
                        ),
                    )
                    if back_points is not None and backward_status is not None:
                        fb_error = np.linalg.norm(
                            track.points.reshape(-1, 2)
                            - back_points.reshape(-1, 2),
                            axis=1,
                        )
                        valid = (
                            forward_status.reshape(-1).astype(bool)
                            & backward_status.reshape(-1).astype(bool)
                            & (fb_error <= self.max_forward_backward_error)
                        )
                        if np.any(valid):
                            good_old = track.points.reshape(-1, 2)[valid]
                            good_next_flat = next_points.reshape(-1, 2)[valid]
                            displacement = np.median(good_next_flat - good_old, axis=0)
                            translated = _translated_box(
                                track.bbox,
                                float(displacement[0]),
                                float(displacement[1]),
                                width,
                                height,
                            )
                            good_next = good_next_flat.reshape(-1, 1, 2).astype(np.float32)
                            survival = len(good_next) / max(initial_count, 1)
                            median_error = float(np.median(fb_error[valid]))
                            geometric_quality = max(
                                0.0,
                                1.0 - median_error / self.max_forward_backward_error,
                            )
                            feature_quality = min(
                                1.0,
                                len(good_next) / self.min_features_per_person,
                            )
                            confidence = survival * geometric_quality * feature_quality

            memberships = _memberships(translated, zone_list, width, height)
            entered_zone = entered_zone or bool(memberships - track.zone_memberships)
            track.bbox = translated
            track.points = good_next
            track.zone_memberships = memberships
            projections.append(
                TrackerProjection(
                    track_id=track.track_id,
                    bbox=translated,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_keyframe_timestamp=track.source_keyframe_timestamp,
                    timestamp=observed_at,
                )
            )
            confidences.append(confidence)

        new_foreground = self._has_new_foreground(
            previous,
            current,
            old_boxes,
            [projection.bbox for projection in projections],
        )
        aggregate_confidence = min(confidences, default=0.0)
        reasons: list[str] = []
        if aggregate_confidence < self.confidence_threshold:
            reasons.append(REDETECT_LOW_CONFIDENCE)
        if new_foreground:
            reasons.append(REDETECT_NEW_FOREGROUND)
        if entered_zone:
            reasons.append(REDETECT_ZONE_ENTRY)

        self._previous_gray = current.copy()
        self._previous_timestamp = observed_at
        return KeyframeTrackerResult(
            projections=tuple(projections),
            aggregate_confidence=max(0.0, min(1.0, aggregate_confidence)),
            force_redetect=bool(reasons),
            reasons=tuple(reasons),
            new_foreground=new_foreground,
            zone_entry=entered_zone,
        )
