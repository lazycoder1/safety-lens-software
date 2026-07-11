"""Workstation occupancy analytics for demo cameras."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import state
from constants import VIDEO_DIR
from detection import bbox_intersects_polygon, point_in_polygon

WORK_START = "09:00"
WORK_END = "17:30"
LUNCH_START = "13:00"
LUNCH_END = "14:00"
GRACE_SECONDS = 5 * 60
DEFAULT_OCCUPANCY_DURATION_SECONDS = 60

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


def _zone_occupied(zone: dict[str, Any], persons: list[dict[str, Any]], frame_w: int, frame_h: int) -> bool:
    points = zone.get("points") or []
    if len(points) < 3:
        return False
    for person in persons:
        bbox = person.get("bbox")
        if not bbox:
            continue
        if zone.get("occupancy_mode") == "chair_anchor":
            x1, y1, x2, y2 = bbox
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            # For chair occupancy, use lower-body anchors instead of box overlap.
            # This prevents a standing/walking person near a chair from filling a
            # large zone just because their bounding box touches it.
            anchors = (
                ((x1 + x2) / 2, y1 + height * 0.72),
                (x1 + width * 0.35, y1 + height * 0.78),
                (x1 + width * 0.65, y1 + height * 0.78),
                ((x1 + x2) / 2, y1 + height * 0.9),
            )
            if any(point_in_polygon(px / frame_w, py / frame_h, points) for px, py in anchors):
                return True
        elif bbox_intersects_polygon(bbox, points, frame_w, frame_h):
            return True
    return False


def _numeric_config(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_occupancy_report(camera_id: str, camera: dict[str, Any]) -> dict[str, Any]:
    """Return current chair occupancy and update an in-memory report preview."""
    now = time.time()
    zones = [zone for zone in camera.get("zones", []) if zone.get("type") == "workstation"]
    detections = state.camera_detections.get(camera_id, [])
    persons = [d for d in detections if d.get("class") == "person"]
    frame_w, frame_h = _camera_frame_size({**camera, "id": camera_id})

    session = _sessions.setdefault(
        camera_id,
        {
            "started_at": now,
            "last_sample_at": now,
            "zones": {},
        },
    )
    elapsed = max(0.0, now - session["last_sample_at"])
    session["last_sample_at"] = now
    min_duration_seconds = int(
        _numeric_config(
            camera.get("occupancy_min_duration_seconds")
            or camera.get("occupancy_duration_threshold_seconds"),
            DEFAULT_OCCUPANCY_DURATION_SECONDS,
        )
    )

    rows = []
    for zone in zones:
        zone_id = zone.get("id") or zone.get("name")
        occupied = _zone_occupied(zone, persons, frame_w, frame_h)
        zone_min_duration_seconds = int(
            _numeric_config(
                zone.get("min_occupied_duration_seconds")
                or zone.get("occupancy_min_duration_seconds")
                or zone.get("occupancy_duration_threshold_seconds"),
                min_duration_seconds,
            )
        )
        zone_state = session["zones"].setdefault(
            zone_id,
            {
                "name": zone.get("name", "Chair"),
                "occupied_seconds": 0.0,
                "empty_seconds": 0.0,
                "empty_events": 0,
                "long_absence_events": 0,
                "occupied_since": None,
                "longest_occupied_seconds": 0.0,
                "last_status": None,
                "empty_since": None,
            },
        )

        if occupied:
            if zone_state["last_status"] is not True:
                zone_state["occupied_since"] = now
            zone_state["occupied_seconds"] += elapsed
            zone_state["empty_since"] = None
            occupied_since = float(zone_state.get("occupied_since") or now)
            current_occupied_seconds = max(0.0, now - occupied_since)
            zone_state["longest_occupied_seconds"] = max(
                float(zone_state.get("longest_occupied_seconds") or 0.0),
                current_occupied_seconds,
            )
        else:
            zone_state["empty_seconds"] += elapsed
            zone_state["occupied_since"] = None
            current_occupied_seconds = 0.0
            if zone_state["last_status"] is True:
                zone_state["empty_events"] += 1
                zone_state["empty_since"] = now
            elif zone_state["empty_since"] is None:
                zone_state["empty_since"] = now

        if not occupied and zone_state["empty_since"] is not None:
            empty_for = now - zone_state["empty_since"]
            if empty_for >= GRACE_SECONDS and not zone_state.get("current_long_absence"):
                zone_state["long_absence_events"] += 1
                zone_state["current_long_absence"] = True
        else:
            empty_for = 0.0
            zone_state["current_long_absence"] = False

        zone_state["last_status"] = occupied
        total = zone_state["occupied_seconds"] + zone_state["empty_seconds"]
        longest_occupied_seconds = round(float(zone_state.get("longest_occupied_seconds") or 0.0))

        rows.append(
            {
                "id": zone_id,
                "name": zone_state["name"],
                "status": "occupied" if occupied else "empty",
                "occupied": occupied,
                "sampleSeconds": round(total),
                "reportReady": False,
                "emptyEvents": zone_state["empty_events"],
                "longAbsenceEvents": zone_state["long_absence_events"],
                "emptyForSeconds": round(empty_for),
                "occupiedSeconds": round(zone_state["occupied_seconds"]),
                "currentOccupiedSeconds": round(current_occupied_seconds),
                "longestOccupiedSeconds": longest_occupied_seconds,
                "emptySeconds": round(zone_state["empty_seconds"]),
                "minDurationSeconds": zone_min_duration_seconds,
                "durationReady": longest_occupied_seconds >= zone_min_duration_seconds,
            }
        )

    sample_seconds = round(max(0.0, now - session["started_at"]))
    duration_ready_zone_count = sum(1 for row in rows if row.get("durationReady"))
    return {
        "cameraId": camera_id,
        "title": "Chair occupancy snapshot",
        "generatedAt": now,
        "sessionSeconds": sample_seconds,
        "sampleSeconds": sample_seconds,
        "reportReady": False,
        "durationReady": duration_ready_zone_count > 0,
        "durationReadyZoneCount": duration_ready_zone_count,
        "minDurationSeconds": min_duration_seconds,
        "maxOccupiedSeconds": max((int(row.get("occupiedSeconds") or 0) for row in rows), default=0),
        "maxCurrentOccupiedSeconds": max((int(row.get("currentOccupiedSeconds") or 0) for row in rows), default=0),
        "maxLongestOccupiedSeconds": max((int(row.get("longestOccupiedSeconds") or 0) for row in rows), default=0),
        "totalOccupiedSeconds": sum(int(row.get("occupiedSeconds") or 0) for row in rows),
        "workHours": {"start": WORK_START, "end": WORK_END},
        "excludedWindows": [{"label": "Lunch", "start": LUNCH_START, "end": LUNCH_END}],
        "gracePeriodSeconds": GRACE_SECONDS,
        "chairs": rows,
        "tables": rows,
    }
