"""Object lifecycle telemetry for watched zones."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import state
from constants import VIDEO_DIR
from detection import bbox_intersects_polygon

DEFAULT_OBJECT_CLASSES = {"backpack", "handbag", "suitcase"}
DEFAULT_REMOVAL_AFTER_SECONDS = 1.0
DEFAULT_DWELL_AFTER_SECONDS = 2.0
DEFAULT_EVENT_LINGER_SECONDS = 15.0

_LIFECYCLE_STATE: dict[str, dict[str, dict[str, Any]]] = {}


def reset_object_lifecycle_state(camera_id: str | None = None) -> None:
    if camera_id:
        _LIFECYCLE_STATE.pop(camera_id, None)
        return
    _LIFECYCLE_STATE.clear()


def is_object_lifecycle_enabled(camera: dict[str, Any]) -> bool:
    capabilities = set(camera.get("capabilities") or [])
    return "object_lifecycle" in capabilities or bool(_watch_zones(camera))


def _camera_frame_size(camera: dict[str, Any]) -> tuple[int, int]:
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


def _split_classes(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _object_classes(camera: dict[str, Any], zone: dict[str, Any] | None = None) -> set[str]:
    zone_classes = _split_classes((zone or {}).get("classes") or (zone or {}).get("object_classes"))
    camera_classes = _split_classes(camera.get("object_classes"))
    return zone_classes or camera_classes or set(DEFAULT_OBJECT_CLASSES)


def _watch_zones(camera: dict[str, Any]) -> list[dict[str, Any]]:
    zones = []
    for zone in camera.get("zones", []):
        zone_type = str(zone.get("type") or "").lower()
        analytics = str(zone.get("analytics") or "").lower()
        if (
            zone_type in {"object_watch", "object_lifecycle", "unattended_object"}
            or analytics in {"object_lifecycle", "object_removal", "unattended_object"}
            or zone.get("object_lifecycle_enabled")
        ):
            zones.append(zone)
    return zones


def _zone_key(zone: dict[str, Any], index: int) -> str:
    return str(zone.get("id") or zone.get("name") or f"zone_{index}")


def _matching_objects(
    detections: list[dict[str, Any]],
    classes: set[str],
    points: list[list[float]],
    frame_w: int,
    frame_h: int,
) -> list[dict[str, Any]]:
    if len(points) < 3:
        return []
    return [
        detection
        for detection in detections
        if detection.get("class") in classes
        and detection.get("bbox")
        and bbox_intersects_polygon(detection["bbox"], points, frame_w, frame_h)
    ]


def _class_counts(objects: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        class_name = str(obj.get("class") or "unknown")
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def update_object_lifecycle(
    camera_id: str,
    camera: dict[str, Any],
    detections: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
    *,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Advance watched-zone state and return one-shot lifecycle alert candidates."""
    now = now or time.time()
    zones = _watch_zones(camera)
    camera_state = _LIFECYCLE_STATE.setdefault(camera_id, {})
    events: list[dict[str, Any]] = []

    for idx, zone in enumerate(zones):
        points = zone.get("points") or []
        zone_id = _zone_key(zone, idx)
        classes = _object_classes(camera, zone)
        removal_after = float(zone.get("removal_after_seconds") or camera.get("object_removal_after_seconds") or DEFAULT_REMOVAL_AFTER_SECONDS)
        dwell_after = float(zone.get("dwell_after_seconds") or camera.get("object_dwell_after_seconds") or DEFAULT_DWELL_AFTER_SECONDS)
        event_linger = float(zone.get("event_linger_seconds") or camera.get("object_event_linger_seconds") or DEFAULT_EVENT_LINGER_SECONDS)
        objects = _matching_objects(detections, classes, points, frame_w, frame_h)
        confidences = [float(obj.get("confidence") or 0) for obj in objects]
        row = camera_state.setdefault(
            zone_id,
            {
                "seenEver": False,
                "objectPresent": False,
                "firstSeenAt": None,
                "lastSeenAt": None,
                "lastRemovedAt": None,
                "lastDwellAt": None,
                "dwellAlerted": False,
                "dwellCount": 0,
                "removalCount": 0,
                "lastClassCounts": {},
                "lastConfidence": None,
            },
        )

        if objects:
            if not row["objectPresent"]:
                row["firstSeenAt"] = now
                row["dwellAlerted"] = False
            row["seenEver"] = True
            row["objectPresent"] = True
            row["lastSeenAt"] = now
            row["lastClassCounts"] = _class_counts(objects)
            row["lastConfidence"] = round(max(confidences), 2) if confidences else None
            first_seen = float(row.get("firstSeenAt") or now)
            present_seconds = max(0.0, now - first_seen)
            if present_seconds >= dwell_after and not row.get("dwellAlerted"):
                row["dwellAlerted"] = True
                row["lastDwellAt"] = now
                row["dwellCount"] = int(row.get("dwellCount") or 0) + 1
                events.append(
                    {
                        "camera_id": camera_id,
                        "rule": "Unattended Object Dwell",
                        "severity": zone.get("dwell_severity") or camera.get("object_dwell_severity") or "P2",
                        "confidence": float(row.get("lastConfidence") or 0.5),
                        "count": 1,
                        "classes": [
                            "unattended object dwell",
                            "object dwell",
                            *sorted((row.get("lastClassCounts") or {}).keys()),
                        ],
                        "zone": zone.get("name") or camera.get("zone", "Object Watch Zone"),
                        "description": f"Watched object remained in {zone.get('name') or 'object watch zone'} for {round(present_seconds, 1)}s",
                        "source": "COCO Object Lifecycle",
                        "threshold": 1,
                        "metadata": {
                            "zoneId": zone_id,
                            "lastClassCounts": row.get("lastClassCounts") or {},
                            "dwellAfterSeconds": dwell_after,
                            "presentSeconds": round(present_seconds, 1),
                        },
                    }
                )
            continue

        last_seen = row.get("lastSeenAt")
        if row.get("objectPresent") and last_seen and now - float(last_seen) >= removal_after:
            row["objectPresent"] = False
            row["lastRemovedAt"] = now
            row["removalCount"] = int(row.get("removalCount") or 0) + 1
            events.append(
                {
                    "camera_id": camera_id,
                    "rule": "Object Removed",
                    "severity": zone.get("severity") or camera.get("object_removal_severity") or "P2",
                    "confidence": float(row.get("lastConfidence") or 0.5),
                    "count": 1,
                    "classes": ["object removed", *sorted((row.get("lastClassCounts") or {}).keys())],
                    "zone": zone.get("name") or camera.get("zone", "Object Watch Zone"),
                    "description": f"Watched object removed from {zone.get('name') or 'object watch zone'}",
                    "source": "COCO Object Lifecycle",
                    "threshold": 1,
                    "metadata": {
                        "zoneId": zone_id,
                        "lastClassCounts": row.get("lastClassCounts") or {},
                        "removalAfterSeconds": removal_after,
                    },
                }
            )

        if row.get("lastRemovedAt") and now - float(row["lastRemovedAt"]) > event_linger:
            row["lastRemovedAt"] = None
        if row.get("lastDwellAt") and now - float(row["lastDwellAt"]) > event_linger:
            row["lastDwellAt"] = None

    return events


def get_object_lifecycle_snapshot(camera_id: str, camera: dict[str, Any]) -> dict[str, Any]:
    frame_w, frame_h = _camera_frame_size({**camera, "id": camera_id})
    detections = state.camera_detections.get(camera_id, [])
    now = time.time()
    update_object_lifecycle(camera_id, camera, detections, frame_w, frame_h, now=now)

    camera_state = _LIFECYCLE_STATE.setdefault(camera_id, {})
    rows = []
    for idx, zone in enumerate(_watch_zones(camera)):
        zone_id = _zone_key(zone, idx)
        row = camera_state.get(zone_id, {})
        last_seen = row.get("lastSeenAt")
        last_removed = row.get("lastRemovedAt")
        last_dwell = row.get("lastDwellAt")
        event_linger = float(zone.get("event_linger_seconds") or camera.get("object_event_linger_seconds") or DEFAULT_EVENT_LINGER_SECONDS)
        dwell_after = float(zone.get("dwell_after_seconds") or camera.get("object_dwell_after_seconds") or DEFAULT_DWELL_AFTER_SECONDS)
        present_seconds = round(now - float(row["firstSeenAt"]), 1) if row.get("firstSeenAt") and row.get("objectPresent") else 0.0
        rows.append(
            {
                "id": zone_id,
                "name": zone.get("name", "Object Watch Zone"),
                "classes": sorted(_object_classes(camera, zone)),
                "objectPresent": bool(row.get("objectPresent")),
                "seenEver": bool(row.get("seenEver")),
                "dwellDetected": bool(last_dwell and now - float(last_dwell) <= event_linger),
                "dwellReady": bool(present_seconds >= dwell_after) if row.get("objectPresent") else False,
                "dwellCount": int(row.get("dwellCount") or 0),
                "dwellAfterSeconds": dwell_after,
                "removalDetected": bool(last_removed and now - float(last_removed) <= event_linger),
                "removalCount": int(row.get("removalCount") or 0),
                "lastClassCounts": row.get("lastClassCounts") or {},
                "lastConfidence": row.get("lastConfidence"),
                "presentSeconds": present_seconds,
                "absentSeconds": round(now - float(last_seen), 1) if last_seen and not row.get("objectPresent") else 0.0,
                "lastSeenAt": last_seen,
                "lastDwellAt": last_dwell,
                "lastRemovedAt": last_removed,
            }
        )

    return {
        "cameraId": camera_id,
        "title": "Object lifecycle snapshot",
        "generatedAt": now,
        "objectPresent": any(row["objectPresent"] for row in rows),
        "dwellDetected": any(row["dwellDetected"] for row in rows),
        "dwellReady": any(row["dwellReady"] for row in rows),
        "dwellCount": sum(row["dwellCount"] for row in rows),
        "maxPresentSeconds": max((row["presentSeconds"] for row in rows), default=0.0),
        "removalDetected": any(row["removalDetected"] for row in rows),
        "removalCount": sum(row["removalCount"] for row in rows),
        "watchZones": rows,
        "calibrated": False,
    }
