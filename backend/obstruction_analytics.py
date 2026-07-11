"""Route obstruction snapshot telemetry for camera QA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import state
from constants import VIDEO_DIR
from detection import bbox_intersects_polygon

DEFAULT_OBSTRUCTION_THRESHOLD = 1
DEFAULT_OBSTRUCTION_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
DEFAULT_OBSTRUCTION_DURATION_SECONDS = 10
DEFAULT_SESSION_RESET_GAP_SECONDS = 30
SEVERITY_RANKS = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
SEVERITY_LABELS = {value: key for key, value in SEVERITY_RANKS.items()}

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


def _split_classes(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _obstruction_classes(camera: dict[str, Any], zone: dict[str, Any] | None = None) -> set[str]:
    zone_classes = _split_classes((zone or {}).get("classes") or (zone or {}).get("obstruction_classes"))
    camera_classes = _split_classes(camera.get("obstruction_classes"))
    return zone_classes or camera_classes or set(DEFAULT_OBSTRUCTION_CLASSES)


def _obstruction_zones(camera: dict[str, Any]) -> list[dict[str, Any]]:
    zones = []
    for zone in camera.get("zones", []):
        zone_type = str(zone.get("type") or "").lower()
        analytics = str(zone.get("analytics") or "").lower()
        if zone_type in {"route", "obstruction", "keep_clear"} or analytics == "obstruction" or zone.get("obstruction_enabled"):
            zones.append(zone)
    return zones


def _matching_detections(detections: list[dict[str, Any]], classes: set[str]) -> list[dict[str, Any]]:
    return [d for d in detections if d.get("class") in classes and d.get("bbox")]


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


def _zone_id(zone: dict[str, Any]) -> str:
    return str(zone.get("id") or zone.get("name") or "route_zone")


def _area_square_meters(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> float | None:
    calibration = zone.get("calibration")
    camera_calibration = (camera or {}).get("obstruction_calibration")
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
        (camera or {}).get("obstruction_area_square_meters"),
    )


def _severity_thresholds(zone: dict[str, Any], camera: dict[str, Any] | None = None) -> dict[str, float]:
    zone_thresholds = zone.get("severity_thresholds")
    camera_thresholds = (camera or {}).get("obstruction_severity_thresholds")
    if not isinstance(zone_thresholds, dict):
        zone_thresholds = {}
    if not isinstance(camera_thresholds, dict):
        camera_thresholds = {}

    def pick(*names: str) -> float | None:
        values: list[Any] = []
        for name in names:
            values.extend(
                [
                    zone_thresholds.get(name),
                    zone.get(name),
                    camera_thresholds.get(name),
                    (camera or {}).get(f"obstruction_{name}"),
                ]
            )
        return _optional_numeric_config(*values)

    thresholds = {
        "mediumDensityObjectsPerSquareMeter": pick(
            "medium_density_objects_per_square_meter",
            "medium_density_per_m2",
            "medium_density",
        ),
        "highDensityObjectsPerSquareMeter": pick(
            "high_density_objects_per_square_meter",
            "high_density_per_m2",
            "high_density",
        ),
        "criticalDensityObjectsPerSquareMeter": pick(
            "critical_density_objects_per_square_meter",
            "critical_density_per_m2",
            "critical_density",
        ),
        "mediumObjectCount": pick("medium_object_count", "medium_count"),
        "highObjectCount": pick("high_object_count", "high_count"),
        "criticalObjectCount": pick("critical_object_count", "critical_count"),
    }
    return {key: float(value) for key, value in thresholds.items() if value is not None}


def _severity_for(
    object_count: int,
    density_objects_per_square_meter: float | None,
    thresholds: dict[str, float],
) -> tuple[str, int]:
    if object_count <= 0:
        return "none", SEVERITY_RANKS["none"]

    rank = SEVERITY_RANKS["low"]
    if density_objects_per_square_meter is not None:
        if density_objects_per_square_meter >= thresholds.get("criticalDensityObjectsPerSquareMeter", float("inf")):
            rank = max(rank, SEVERITY_RANKS["critical"])
        elif density_objects_per_square_meter >= thresholds.get("highDensityObjectsPerSquareMeter", float("inf")):
            rank = max(rank, SEVERITY_RANKS["high"])
        elif density_objects_per_square_meter >= thresholds.get("mediumDensityObjectsPerSquareMeter", float("inf")):
            rank = max(rank, SEVERITY_RANKS["medium"])

    if object_count >= thresholds.get("criticalObjectCount", 1_000_000):
        rank = max(rank, SEVERITY_RANKS["critical"])
    elif object_count >= thresholds.get("highObjectCount", 1_000_000):
        rank = max(rank, SEVERITY_RANKS["high"])
    elif object_count >= thresholds.get("mediumObjectCount", 1_000_000):
        rank = max(rank, SEVERITY_RANKS["medium"])

    return SEVERITY_LABELS[rank], rank


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
    reset_gap = _numeric_config(camera.get("obstruction_session_reset_gap_seconds"), DEFAULT_SESSION_RESET_GAP_SECONDS)
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
    object_count: int,
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
        state["max_object_count"] = max(int(state.get("max_object_count") or 0), object_count)
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
        "maxObjectCount": int(state.get("max_object_count") or object_count),
        "activeSince": state.get("active_since"),
    }


def reset_obstruction_state(camera_id: str | None = None) -> None:
    if camera_id is None:
        _sessions.clear()
    else:
        _sessions.pop(camera_id, None)


def get_obstruction_snapshot(camera_id: str, camera: dict[str, Any]) -> dict[str, Any]:
    """Return a live route-obstruction snapshot from current detections.

    This is a configured keep-clear/route-zone occupancy snapshot, not calibrated
    dwell time, parking flow, or object lifecycle analytics.
    """
    frame_w, frame_h = _camera_frame_size({**camera, "id": camera_id})
    detections = state.camera_detections.get(camera_id, [])
    zones = _obstruction_zones(camera)
    threshold = int(camera.get("obstruction_threshold") or DEFAULT_OBSTRUCTION_THRESHOLD)
    now = time.time()
    session, elapsed = _session_for(camera_id, camera, now)
    min_duration = int(
        _numeric_config(camera.get("obstruction_min_duration_seconds"), DEFAULT_OBSTRUCTION_DURATION_SECONDS)
    )

    if not zones:
        classes = _obstruction_classes(camera)
        objects = _matching_detections(detections, classes)
        confidences = [float(obj.get("confidence") or 0) for obj in objects]
        area_square_meters = _area_square_meters({}, camera)
        density = len(objects) / area_square_meters if area_square_meters else None
        severity_thresholds = _severity_thresholds({}, camera)
        severity, severity_rank = _severity_for(len(objects), density, severity_thresholds)
        obstruction_active = len(objects) >= threshold
        duration = _update_duration_state(
            session,
            active=obstruction_active,
            now=now,
            elapsed=elapsed,
            object_count=len(objects),
        )
        return {
            "cameraId": camera_id,
            "title": "Route obstruction duration snapshot",
            "generatedAt": now,
            "obstructionActive": obstruction_active,
            "objectCount": len(objects),
            "maxZoneCount": len(objects),
            "threshold": threshold,
            "minDurationSeconds": min_duration,
            "durationReady": duration["activeSeconds"] >= min_duration,
            "sessionSeconds": round(max(0.0, now - float(session.get("started_at") or now))),
            **duration,
            "classes": sorted(classes),
            "calibrated": area_square_meters is not None,
            "calibratedZoneCount": 1 if area_square_meters is not None else 0,
            "areaSquareMeters": round(area_square_meters, 2) if area_square_meters is not None else None,
            "obstructionDensityObjectsPerSquareMeter": round(density, 3) if density is not None else None,
            "maxObstructionDensityObjectsPerSquareMeter": round(density, 3) if density is not None else None,
            "maxSeverity": severity,
            "maxSeverityRank": severity_rank,
            "severityThresholds": severity_thresholds,
            "routeZones": [],
            "detections": len(objects),
            "maxConfidence": round(max(confidences), 2) if confidences else None,
        }

    counted_object_indexes: set[int] = set()
    rows = []
    for zone in zones:
        points = zone.get("points") or []
        if len(points) < 3:
            continue
        classes = _obstruction_classes(camera, zone)
        objects = _matching_detections(detections, classes)
        zone_object_indexes = []
        zone_confidences = []
        zone_class_counts: dict[str, int] = {}
        for idx, obj in enumerate(objects):
            if bbox_intersects_polygon(obj["bbox"], points, frame_w, frame_h):
                zone_object_indexes.append(idx)
                counted_object_indexes.add(idx)
                zone_confidences.append(float(obj.get("confidence") or 0))
                class_name = str(obj.get("class") or "unknown")
                zone_class_counts[class_name] = zone_class_counts.get(class_name, 0) + 1
        zone_count = len(zone_object_indexes)
        zone_threshold = int(zone.get("threshold") or threshold)
        zone_active = zone_count >= zone_threshold
        zone_id = _zone_id(zone)
        area_ratio = _polygon_area(points)
        area_square_meters = _area_square_meters(zone, camera)
        density = zone_count / area_square_meters if area_square_meters else None
        severity_thresholds = _severity_thresholds(zone, camera)
        severity, severity_rank = _severity_for(zone_count, density, severity_thresholds)
        zone_min_duration = int(_numeric_config(zone.get("min_duration_seconds"), min_duration))
        zone_state = session["zones"].setdefault(zone_id, {})
        zone_duration = _update_duration_state(
            zone_state,
            active=zone_active,
            now=now,
            elapsed=elapsed,
            object_count=zone_count,
        )
        rows.append(
            {
                "id": zone_id,
                "name": zone.get("name", "Route Zone"),
                "objectCount": zone_count,
                "threshold": zone_threshold,
                "obstructionActive": zone_active,
                "minDurationSeconds": zone_min_duration,
                "durationReady": zone_duration["activeSeconds"] >= zone_min_duration,
                **zone_duration,
                "classes": sorted(classes),
                "classCounts": zone_class_counts,
                "areaRatio": round(area_ratio, 4),
                "calibrated": area_square_meters is not None,
                "areaSquareMeters": round(area_square_meters, 2) if area_square_meters is not None else None,
                "obstructionDensityObjectsPerSquareMeter": round(density, 3) if density is not None else None,
                "severity": severity,
                "severityRank": severity_rank,
                "severityThresholds": severity_thresholds,
                "maxConfidence": round(max(zone_confidences), 2) if zone_confidences else None,
            }
        )

    total_objects = len(counted_object_indexes)
    max_zone_count = max((row["objectCount"] for row in rows), default=0)
    max_density = max(
        (
            row["obstructionDensityObjectsPerSquareMeter"]
            for row in rows
            if row.get("obstructionDensityObjectsPerSquareMeter") is not None
        ),
        default=None,
    )
    max_severity_rank = max((int(row.get("severityRank") or 0) for row in rows), default=0)
    all_classes = sorted({cls for row in rows for cls in row.get("classes", [])})
    obstruction_active = any(row["obstructionActive"] for row in rows)
    duration = _update_duration_state(
        session,
        active=obstruction_active,
        now=now,
        elapsed=elapsed,
        object_count=total_objects,
    )
    return {
        "cameraId": camera_id,
        "title": "Route obstruction duration snapshot",
        "generatedAt": now,
        "obstructionActive": obstruction_active,
        "objectCount": total_objects,
        "maxZoneCount": max_zone_count,
        "threshold": threshold,
        "minDurationSeconds": min_duration,
        "durationReady": duration["activeSeconds"] >= min_duration,
        "sessionSeconds": round(max(0.0, now - float(session.get("started_at") or now))),
        **duration,
        "classes": all_classes,
        "calibrated": any(bool(row.get("calibrated")) for row in rows),
        "calibratedZoneCount": sum(1 for row in rows if row.get("calibrated")),
        "maxObstructionDensityObjectsPerSquareMeter": max_density,
        "maxSeverity": SEVERITY_LABELS[max_severity_rank],
        "maxSeverityRank": max_severity_rank,
        "routeZones": rows,
        "detections": len(_matching_detections(detections, set(all_classes or DEFAULT_OBSTRUCTION_CLASSES))),
        "maxConfidence": max((row["maxConfidence"] for row in rows if row["maxConfidence"] is not None), default=None),
    }
