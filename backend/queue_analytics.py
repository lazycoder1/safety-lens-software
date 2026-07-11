"""Queue/crowd snapshot telemetry for camera QA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import state
from constants import VIDEO_DIR
from detection import bbox_intersects_polygon

DEFAULT_QUEUE_THRESHOLD = 3
DEFAULT_QUEUE_DURATION_SECONDS = 10
DEFAULT_SESSION_RESET_GAP_SECONDS = 30
DEFAULT_WAIT_TRACK_MAX_DISTANCE = 0.18
DEFAULT_WAIT_TRACK_STALE_SECONDS = 8

_sessions: dict[str, dict[str, Any]] = {}


def _camera_frame_size(camera: dict[str, Any]) -> tuple[int, int]:
    cached = state.get_camera_frame_dimensions(str(camera.get("id", "")))
    if cached is not None:
        return cached

    video = camera.get("video")
    if video:
        path = Path(video)
        if not path.is_absolute():
            path = VIDEO_DIR / video
        cap = cv2.VideoCapture(str(path))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width > 0 and height > 0:
                return width, height
        finally:
            cap.release()

    frame_bytes = state.camera_clean_frames.get(camera.get("id", "")) or b""
    if frame_bytes:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            height, width = frame.shape[:2]
            return width, height

    return 1280, 720


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _queue_zones(camera: dict[str, Any]) -> list[dict[str, Any]]:
    zones = []
    for zone in camera.get("zones", []):
        zone_type = str(zone.get("type") or "").lower()
        analytics = str(zone.get("analytics") or "").lower()
        if zone_type in {"queue", "queue_area"} or analytics == "queue" or zone.get("queue_enabled"):
            zones.append(zone)
    return zones


def _numeric_config(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_numeric_config(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _area_square_meters(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> float | None:
    calibration = zone.get("calibration")
    camera_calibration = (camera or {}).get("queue_calibration")
    zone_id = _zone_id(zone)
    camera_zone_calibration = None
    if isinstance(camera_calibration, dict):
        camera_zone_calibration = (camera_calibration.get("zones") or {}).get(zone_id)

    return _optional_numeric_config(
        zone.get("area_square_meters"),
        zone.get("area_m2"),
        zone.get("physical_area_m2"),
        calibration.get("area_square_meters") if isinstance(calibration, dict) else None,
        calibration.get("area_m2") if isinstance(calibration, dict) else None,
        camera_zone_calibration.get("area_square_meters") if isinstance(camera_zone_calibration, dict) else None,
        camera_zone_calibration.get("area_m2") if isinstance(camera_zone_calibration, dict) else None,
        (camera or {}).get("queue_area_square_meters"),
    )


def _density_threshold(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> float | None:
    return _optional_numeric_config(
        zone.get("density_threshold_people_per_square_meter"),
        zone.get("max_density_people_per_square_meter"),
        zone.get("density_threshold_per_m2"),
        (camera or {}).get("queue_density_threshold_people_per_square_meter"),
    )


def _wait_tracking_enabled(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> bool:
    for value in (
        zone.get("wait_tracking_enabled"),
        zone.get("queue_wait_tracking_enabled"),
        zone.get("wait_time_tracking_enabled"),
        (camera or {}).get("queue_wait_tracking_enabled"),
        (camera or {}).get("wait_time_tracking_enabled"),
    ):
        if value is None:
            continue
        return bool(value)
    return False


def _wait_threshold_seconds(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> float | None:
    return _optional_numeric_config(
        zone.get("wait_threshold_seconds"),
        zone.get("max_wait_seconds"),
        zone.get("queue_wait_threshold_seconds"),
        (camera or {}).get("queue_wait_threshold_seconds"),
    )


def _bbox_center(bbox: list[float], frame_w: int, frame_h: int) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return (
        ((x1 + x2) / 2.0) / max(float(frame_w), 1.0),
        ((y1 + y2) / 2.0) / max(float(frame_h), 1.0),
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _update_wait_tracklets(
    zone_state: dict[str, Any],
    *,
    persons: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
    now: float,
    max_distance: float,
    stale_seconds: float,
) -> dict[str, Any]:
    tracklets = zone_state.setdefault("queue_tracklets", {})
    if not isinstance(tracklets, dict):
        tracklets = {}
        zone_state["queue_tracklets"] = tracklets

    detections = [
        {
            "center": _bbox_center(person["bbox"], frame_w, frame_h),
            "confidence": float(person.get("confidence") or 0),
        }
        for person in persons
        if person.get("bbox")
    ]
    unmatched_track_ids = set(tracklets)
    next_id = int(zone_state.get("next_queue_track_id") or 1)

    for detection in detections:
        best_track_id = None
        best_distance = max_distance
        for track_id in list(unmatched_track_ids):
            tracklet = tracklets.get(track_id)
            if not isinstance(tracklet, dict):
                continue
            distance = _distance(detection["center"], tuple(tracklet.get("center") or (9.0, 9.0)))
            if distance <= best_distance:
                best_track_id = track_id
                best_distance = distance
        if best_track_id is None:
            best_track_id = f"q{next_id}"
            next_id += 1
            tracklets[best_track_id] = {
                "first_seen_at": now,
                "last_seen_at": now,
                "center": detection["center"],
                "max_confidence": detection["confidence"],
            }
        else:
            unmatched_track_ids.discard(best_track_id)
            tracklet = tracklets[best_track_id]
            tracklet["last_seen_at"] = now
            tracklet["center"] = detection["center"]
            tracklet["max_confidence"] = max(float(tracklet.get("max_confidence") or 0), detection["confidence"])

    for track_id, tracklet in list(tracklets.items()):
        if now - float(tracklet.get("last_seen_at") or now) > stale_seconds:
            tracklets.pop(track_id, None)

    zone_state["next_queue_track_id"] = next_id
    wait_seconds = [
        max(0.0, now - float(tracklet.get("first_seen_at") or now))
        for tracklet in tracklets.values()
        if isinstance(tracklet, dict) and now - float(tracklet.get("last_seen_at") or now) <= stale_seconds
    ]
    max_wait = max(wait_seconds, default=0.0)
    average_wait = sum(wait_seconds) / len(wait_seconds) if wait_seconds else 0.0
    return {
        "waitTimeTrackingEnabled": True,
        "trackedPersonCount": len(wait_seconds),
        "maxWaitSeconds": round(max_wait),
        "averageWaitSeconds": round(average_wait, 1),
        "oldestWaitSeconds": round(max_wait),
    }


def _zone_id(zone: dict[str, Any]) -> str:
    return str(zone.get("id") or zone.get("name") or "queue_zone")


def _empty_session(now: float) -> dict[str, Any]:
    return {
        "started_at": now,
        "last_sample_at": now,
        "active_seconds": 0.0,
        "inactive_seconds": 0.0,
        "active_events": 0,
        "active_since": None,
        "longest_active_seconds": 0.0,
        "last_active": None,
        "zones": {},
    }


def _session_for(camera_id: str, camera: dict[str, Any], now: float) -> tuple[dict[str, Any], float]:
    reset_gap = _numeric_config(camera.get("queue_session_reset_gap_seconds"), DEFAULT_SESSION_RESET_GAP_SECONDS)
    session = _sessions.setdefault(camera_id, _empty_session(now))
    elapsed = max(0.0, now - float(session.get("last_sample_at") or now))
    if elapsed > reset_gap:
        session = _empty_session(now)
        _sessions[camera_id] = session
        elapsed = 0.0
    session["last_sample_at"] = now
    return session, elapsed


def _update_duration_state(
    state: dict[str, Any],
    *,
    active: bool,
    now: float,
    elapsed: float,
    person_count: int,
) -> dict[str, Any]:
    if active:
        if state.get("last_active") is not True:
            state["active_events"] = int(state.get("active_events") or 0) + 1
            state["active_since"] = now
        state["active_seconds"] = float(state.get("active_seconds") or 0.0) + elapsed
        active_since = float(state.get("active_since") or now)
        current_active_seconds = max(0.0, now - active_since)
        state["longest_active_seconds"] = max(
            float(state.get("longest_active_seconds") or 0.0),
            current_active_seconds,
        )
        state["max_person_count"] = max(int(state.get("max_person_count") or 0), person_count)
    else:
        state["inactive_seconds"] = float(state.get("inactive_seconds") or 0.0) + elapsed
        current_active_seconds = 0.0
        state["active_since"] = None

    state["last_active"] = active
    return {
        "activeSeconds": round(float(state.get("active_seconds") or 0.0)),
        "currentActiveSeconds": round(current_active_seconds),
        "longestActiveSeconds": round(float(state.get("longest_active_seconds") or 0.0)),
        "activeEvents": int(state.get("active_events") or 0),
        "maxPersonCount": int(state.get("max_person_count") or person_count),
        "activeSince": state.get("active_since"),
    }


def reset_queue_state(camera_id: str | None = None) -> None:
    if camera_id is None:
        _sessions.clear()
    else:
        _sessions.pop(camera_id, None)


def get_queue_snapshot(camera_id: str, camera: dict[str, Any]) -> dict[str, Any]:
    """Return a live queue snapshot from current person detections.

    This is intentionally a snapshot, not a calibrated wait-time or people-per-
    square-meter model. It gives QA and the UI a concrete count, active flag, and
    zone-level evidence for queue/crowding claims that are configured in YAML.
    """
    frame_w, frame_h = _camera_frame_size({**camera, "id": camera_id})
    detections = state.camera_detections.get(camera_id, [])
    persons = [d for d in detections if d.get("class") == "person" and d.get("bbox")]
    zones = _queue_zones(camera)
    threshold = int(camera.get("queue_threshold") or DEFAULT_QUEUE_THRESHOLD)
    now = time.time()
    session, elapsed = _session_for(camera_id, camera, now)
    min_duration = int(_numeric_config(camera.get("queue_min_duration_seconds"), DEFAULT_QUEUE_DURATION_SECONDS))

    if not zones:
        person_count = len(persons)
        confidences = [float(p.get("confidence") or 0) for p in persons]
        area_square_meters = _optional_numeric_config(
            camera.get("queue_area_square_meters"),
            camera.get("area_square_meters"),
            camera.get("area_m2"),
        )
        density = person_count / area_square_meters if area_square_meters else None
        density_threshold = _density_threshold({}, camera)
        queue_active = person_count >= threshold or (
            density is not None
            and density_threshold is not None
            and density >= density_threshold
        )
        duration = _update_duration_state(
            session,
            active=queue_active,
            now=now,
            elapsed=elapsed,
            person_count=person_count,
        )
        return {
            "cameraId": camera_id,
            "title": "Queue duration snapshot",
            "generatedAt": now,
            "queueActive": queue_active,
            "personCount": person_count,
            "maxZoneCount": person_count,
            "threshold": threshold,
            "minDurationSeconds": min_duration,
            "durationReady": duration["activeSeconds"] >= min_duration,
            "sessionSeconds": round(max(0.0, now - float(session.get("started_at") or now))),
            **duration,
            "calibrated": area_square_meters is not None,
            "areaSquareMeters": round(area_square_meters, 2) if area_square_meters is not None else None,
            "densityPeoplePerSquareMeter": round(density, 3) if density is not None else None,
            "maxDensityPeoplePerSquareMeter": round(density, 3) if density is not None else None,
            "densityThresholdPeoplePerSquareMeter": (
                round(density_threshold, 3) if density_threshold is not None else None
            ),
            "queueZones": [],
            "detections": person_count,
            "maxConfidence": round(max(confidences), 2) if confidences else None,
            "waitTimeTrackingEnabled": False,
            "trackedPersonCount": 0,
            "maxWaitSeconds": 0,
            "averageWaitSeconds": 0.0,
            "oldestWaitSeconds": 0,
            "waitThresholdSeconds": None,
            "waitTimeReady": False,
        }

    counted_person_indexes: set[int] = set()
    rows = []
    for zone in zones:
        points = zone.get("points") or []
        if len(points) < 3:
            continue
        zone_person_indexes = []
        zone_confidences = []
        for idx, person in enumerate(persons):
            if bbox_intersects_polygon(person["bbox"], points, frame_w, frame_h):
                zone_person_indexes.append(idx)
                counted_person_indexes.add(idx)
                zone_confidences.append(float(person.get("confidence") or 0))
        zone_count = len(zone_person_indexes)
        zone_threshold = int(zone.get("threshold") or threshold)
        zone_id = _zone_id(zone)
        area_ratio = _polygon_area(points)
        area_square_meters = _area_square_meters(zone, camera)
        density = zone_count / area_square_meters if area_square_meters else None
        density_threshold = _density_threshold(zone, camera)
        zone_active = zone_count >= zone_threshold or (
            density is not None
            and density_threshold is not None
            and density >= density_threshold
        )
        zone_state = session["zones"].setdefault(zone_id, {})
        zone_duration = _update_duration_state(
            zone_state,
            active=zone_active,
            now=now,
            elapsed=elapsed,
            person_count=zone_count,
        )
        wait_threshold = _wait_threshold_seconds(zone, camera)
        wait_summary = {
            "waitTimeTrackingEnabled": False,
            "trackedPersonCount": 0,
            "maxWaitSeconds": 0,
            "averageWaitSeconds": 0.0,
            "oldestWaitSeconds": 0,
        }
        if _wait_tracking_enabled(zone, camera):
            wait_summary = _update_wait_tracklets(
                zone_state,
                persons=[persons[idx] for idx in zone_person_indexes],
                frame_w=frame_w,
                frame_h=frame_h,
                now=now,
                max_distance=_numeric_config(
                    zone.get("wait_track_max_distance") or camera.get("queue_wait_track_max_distance"),
                    DEFAULT_WAIT_TRACK_MAX_DISTANCE,
                ),
                stale_seconds=_numeric_config(
                    zone.get("wait_track_stale_seconds") or camera.get("queue_wait_track_stale_seconds"),
                    DEFAULT_WAIT_TRACK_STALE_SECONDS,
                ),
            )
        rows.append(
            {
                "id": zone_id,
                "name": zone.get("name", "Queue Zone"),
                "personCount": zone_count,
                "threshold": zone_threshold,
                "queueActive": zone_active,
                "minDurationSeconds": int(_numeric_config(zone.get("min_duration_seconds"), min_duration)),
                "durationReady": zone_duration["activeSeconds"] >= int(
                    _numeric_config(zone.get("min_duration_seconds"), min_duration)
                ),
                **zone_duration,
                "densityIndex": round(zone_count / max(area_ratio, 0.01), 2),
                "areaRatio": round(area_ratio, 4),
                "calibrated": area_square_meters is not None,
                "areaSquareMeters": round(area_square_meters, 2) if area_square_meters is not None else None,
                "densityPeoplePerSquareMeter": round(density, 3) if density is not None else None,
                "densityThresholdPeoplePerSquareMeter": (
                    round(density_threshold, 3) if density_threshold is not None else None
                ),
                **wait_summary,
                "waitThresholdSeconds": round(wait_threshold) if wait_threshold is not None else None,
                "waitTimeReady": (
                    wait_threshold is not None
                    and float(wait_summary.get("maxWaitSeconds") or 0) >= wait_threshold
                ),
                "maxConfidence": round(max(zone_confidences), 2) if zone_confidences else None,
            }
        )

    total_people = len(counted_person_indexes)
    max_zone_count = max((row["personCount"] for row in rows), default=0)
    max_density = max(
        (row["densityPeoplePerSquareMeter"] for row in rows if row.get("densityPeoplePerSquareMeter") is not None),
        default=None,
    )
    density_thresholds = [
        row["densityThresholdPeoplePerSquareMeter"]
        for row in rows
        if row.get("densityThresholdPeoplePerSquareMeter") is not None
    ]
    wait_rows = [row for row in rows if row.get("waitTimeTrackingEnabled")]
    max_wait = max((float(row.get("maxWaitSeconds") or 0) for row in wait_rows), default=0.0)
    tracked_person_count = sum(int(row.get("trackedPersonCount") or 0) for row in wait_rows)
    weighted_wait_sum = sum(
        float(row.get("averageWaitSeconds") or 0) * int(row.get("trackedPersonCount") or 0)
        for row in wait_rows
    )
    wait_thresholds = [row["waitThresholdSeconds"] for row in wait_rows if row.get("waitThresholdSeconds") is not None]
    queue_active = any(row["queueActive"] for row in rows)
    duration = _update_duration_state(
        session,
        active=queue_active,
        now=now,
        elapsed=elapsed,
        person_count=total_people,
    )
    return {
        "cameraId": camera_id,
        "title": "Queue duration snapshot",
        "generatedAt": now,
        "queueActive": queue_active,
        "personCount": total_people,
        "maxZoneCount": max_zone_count,
        "threshold": threshold,
        "minDurationSeconds": min_duration,
        "durationReady": duration["activeSeconds"] >= min_duration,
        "sessionSeconds": round(max(0.0, now - float(session.get("started_at") or now))),
        **duration,
        "calibrated": any(bool(row.get("calibrated")) for row in rows),
        "maxDensityPeoplePerSquareMeter": max_density,
        "densityThresholdPeoplePerSquareMeter": min(density_thresholds) if density_thresholds else None,
        "calibratedZoneCount": sum(1 for row in rows if row.get("calibrated")),
        "waitTimeTrackingEnabled": bool(wait_rows),
        "trackedPersonCount": tracked_person_count,
        "maxWaitSeconds": round(max_wait),
        "averageWaitSeconds": round(weighted_wait_sum / tracked_person_count, 1) if tracked_person_count else 0.0,
        "oldestWaitSeconds": round(max_wait),
        "waitThresholdSeconds": min(wait_thresholds) if wait_thresholds else None,
        "waitTimeReady": any(bool(row.get("waitTimeReady")) for row in wait_rows),
        "queueZones": rows,
        "detections": len(persons),
        "maxConfidence": max((row["maxConfidence"] for row in rows if row["maxConfidence"] is not None), default=None),
    }
