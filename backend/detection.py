"""
SafetyLens detection logic — drawing, PPE checks, zone intrusions, violations.
"""

import logging

import cv2
import numpy as np

from config_manager import get_config
from constants import COCO_NAMES, CLASS_COLORS, YOLOE_COLORS

logger = logging.getLogger("safetylens")

# ── Unified draw function ───────────────────────────────────────────────────


def draw_detections(
    frame: np.ndarray,
    results,
    camera_id: str,
    class_names=None,
    colors=None,
    demo_label: str | None = None,
) -> tuple[np.ndarray, list]:
    """Draw bounding boxes on frame.

    Parameters
    ----------
    class_names : dict | list | None
        - None  -> use COCO_NAMES dict lookup
        - list  -> use index lookup (YOLOe open-vocabulary)
    colors : dict | list | None
        - None  -> use CLASS_COLORS dict
        - list  -> use rotating index (YOLOe palette)
    demo_label : str | None
        - None  -> derive overlay label from camera config demo field
        - str   -> use this string directly (e.g. "YOLOE")
    """
    annotated = frame.copy()
    detections = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Resolve class name
                if class_names is None:
                    cls_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")
                elif isinstance(class_names, list):
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                else:
                    cls_name = class_names.get(cls_id, f"class_{cls_id}")

                # Resolve color
                if colors is None:
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                elif isinstance(colors, list):
                    color = colors[cls_id % len(colors)]
                else:
                    color = colors.get(cls_id, (200, 200, 200))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

    # Overlay camera info
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    cam_name = cam.get("name", camera_id)

    if demo_label is not None:
        overlay_text = f"{cam_name} | {demo_label}"
    else:
        cam_demo = cam.get("demo", "yolo")
        overlay_text = f"{cam_name} | {cam_demo.upper()}"

    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Count color: purple for YOLOe, green for COCO
    if colors is not None and isinstance(colors, list):
        count_color = (168, 85, 247)  # purple for YOLOe
    else:
        count_color = (34, 197, 94)   # green for COCO

    count_text = f"{len(detections)} detections"
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color, 1)

    return annotated, detections


# ── PPE detection ───────────────────────────────────────────────────────────

def get_ppe_groups() -> dict[str, list[str]]:
    """Get PPE groups from config (safety_rules where type=='ppe')."""
    cfg = get_config()
    rules = cfg.get("safety_rules", [])
    return {r["name"].lower(): r["classes"] for r in rules if r.get("type") == "ppe" and r.get("enabled", True)}


def get_ppe_severity_map() -> dict[str, str]:
    """Get severity per PPE group from config."""
    cfg = get_config()
    rules = cfg.get("safety_rules", [])
    return {r["name"].lower(): r.get("severity", "P2") for r in rules if r.get("type") == "ppe" and r.get("enabled", True)}


def _ppe_center_inside_person(person_bbox: list, ppe_dets: list) -> bool:
    """Check if the center of any PPE detection falls inside the person bbox."""
    px1, py1, px2, py2 = person_bbox
    for p in ppe_dets:
        cx = (p["bbox"][0] + p["bbox"][2]) / 2.0
        cy = (p["bbox"][1] + p["bbox"][3]) / 2.0
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
    return False


def check_yoloe_violations(detections: list, camera_id: str) -> list:
    """Return candidate violation dicts (NOT yet persisted to DB).
    Per-person check: only flags persons whose bbox does not contain any matching PPE item."""
    candidates = []
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    yoloe_classes = cam.get("yoloe_classes", [])

    persons = [d for d in detections if d["class"] == "person"]
    if not persons:
        return candidates

    # Get PPE groups and severity map
    safety_rule_ids = cam.get("safety_rule_ids", cam.get("ppe_rule_ids", []))
    if safety_rule_ids:
        # Camera has assigned safety rules — only check PPE ones
        all_rules = cfg.get("safety_rules", [])
        rule_map = {r["id"]: r for r in all_rules}
        ppe_groups = {}
        severity_map = {}
        for rid in safety_rule_ids:
            rule = rule_map.get(rid)
            if rule and rule.get("type") == "ppe" and rule.get("enabled", True):
                key = rule["name"].lower()
                ppe_groups[key] = rule["classes"]
                severity_map[key] = rule.get("severity", "P2")
    else:
        # Fallback: match yoloe_classes against all known PPE groups
        ppe_groups = get_ppe_groups()
        severity_map = get_ppe_severity_map()

    # Build set of PPE groups that this camera monitors
    checked_groups: set[str] = set()
    for cls in yoloe_classes:
        if cls == "person":
            continue
        for group_name, group_classes in ppe_groups.items():
            if cls in group_classes:
                checked_groups.add(group_name)

    # Per-person check
    for group_name in checked_groups:
        group_classes = ppe_groups[group_name]
        ppe_dets = [d for d in detections if d["class"] in group_classes]
        violating_persons = [p for p in persons if not _ppe_center_inside_person(p["bbox"], ppe_dets)]

        if violating_persons:
            print(f"[PPE DEBUG] cam={camera_id} group={group_name} persons={len(persons)} ppe_dets={len(ppe_dets)} ppe_classes={[d['class'] for d in ppe_dets]} violating={len(violating_persons)} all_classes={list({d['class'] for d in detections})}", flush=True)
            candidates.append({
                "camera_id": camera_id,
                "rule": f"Missing {group_name}",
                "severity": severity_map.get(group_name, "P2"),
                "confidence": max(p["confidence"] for p in violating_persons),
                "description": f"{len(violating_persons)} worker(s) detected without {group_name}",
                "source": "YOLOe",
            })

    return candidates


# ── Legacy alert mapping (backward compat) ─────────────────────────────────

_LEGACY_ALERT_MAP = {
    "mobile_phone": "alert_mobile_phone",
    "animal_intrusion": "alert_animal",
    "person_detected": "alert_person",
    "vehicle_detected": "alert_vehicle",
    "zone_intrusion": "alert_zone_intrusion",
}


# ── Zone helpers ────────────────────────────────────────────────────────────

def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test. Coords are normalized 0-1."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center_normalized(bbox: list[int], frame_w: int, frame_h: int) -> tuple[float, float]:
    """Get normalized center (0-1) of a bounding box [x1,y1,x2,y2]."""
    cx = (bbox[0] + bbox[2]) / 2.0 / frame_w
    cy = (bbox[1] + bbox[3]) / 2.0 / frame_h
    return cx, cy


def bbox_probe_points_normalized(bbox: list[int], frame_w: int, frame_h: int) -> list[tuple[float, float]]:
    """Return a set of normalized (0-1) test points for bbox-vs-polygon overlap.

    We can't easily do a true polygon-polygon intersection cheaply, so we
    approximate: test the four corners, the center, and the foot-center
    (bottom-center where a person actually stands). If ANY of these points
    lies inside the polygon we treat it as an intrusion. This matches user
    intent — "if a person is visibly in the drawn area, alert" — much better
    than a single torso point.
    """
    x1, y1, x2, y2 = bbox
    fw, fh = float(frame_w), float(frame_h)
    points = [
        (x1 / fw, y1 / fh),                 # top-left
        (x2 / fw, y1 / fh),                 # top-right
        (x1 / fw, y2 / fh),                 # bottom-left
        (x2 / fw, y2 / fh),                 # bottom-right
        ((x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh),  # center
        ((x1 + x2) / 2 / fw, y2 / fh),      # foot-center
    ]
    return points


def bbox_intersects_polygon(bbox: list[int], polygon: list[list[float]], frame_w: int, frame_h: int) -> bool:
    """True if the person's bounding box visibly overlaps the polygon.

    Fast approximation — checks if any probe point of the bbox lies inside
    the polygon, or if any polygon vertex lies inside the (normalized) bbox.
    The second check catches the edge case where the polygon is entirely
    contained inside the bbox (common when the user draws a tight zone on a
    close-up shot).
    """
    if len(polygon) < 3:
        return False
    for (px, py) in bbox_probe_points_normalized(bbox, frame_w, frame_h):
        if point_in_polygon(px, py, polygon):
            return True
    # Reverse check: any polygon vertex inside the person's bbox?
    x1n = bbox[0] / frame_w
    y1n = bbox[1] / frame_h
    x2n = bbox[2] / frame_w
    y2n = bbox[3] / frame_h
    for vx, vy in polygon:
        if x1n <= vx <= x2n and y1n <= vy <= y2n:
            return True
    return False


def check_zone_intrusions(detections: list, camera_id: str, frame_w: int, frame_h: int) -> list:
    """Check if detected persons are inside any restricted zones for this camera."""
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    zones = cam.get("zones", [])
    enabled_rules = cam.get("alert_classes", [])
    safety_rule_ids = cam.get("safety_rule_ids", [])

    zone_enabled = "zone_intrusion" in enabled_rules or "alert_zone_intrusion" in safety_rule_ids
    if not zone_enabled or not zones:
        return []

    persons = [d for d in detections if d["class"] == "person"]
    if not persons:
        return []

    candidates = []
    for zone in zones:
        zone_name = zone.get("name", "Unknown Zone")
        zone_type = zone.get("type", "restricted")
        points = zone.get("points", [])
        if len(points) < 3:
            continue

        intruders = 0
        max_conf = 0.0
        for p in persons:
            if bbox_intersects_polygon(p["bbox"], points, frame_w, frame_h):
                intruders += 1
                max_conf = max(max_conf, p["confidence"])

        if intruders > 0:
            severity = "P1" if zone_type == "restricted" else "P2"
            candidates.append({
                "camera_id": camera_id,
                "rule": "Zone Intrusion",
                "severity": severity,
                "confidence": max_conf,
                "description": f"{intruders} person(s) in {zone_type} zone '{zone_name}'",
                "source": "YOLO",
            })

    return candidates


def extract_violation_bboxes(rule: str, detections: list, frame_w: int, frame_h: int) -> list[dict]:
    """Extract and normalize bboxes relevant to a specific violation rule.
    Returns list of {label, bbox: [x1_norm, y1_norm, x2_norm, y2_norm], confidence}."""
    results = []

    def normalize(bbox):
        return [round(bbox[0] / frame_w, 4), round(bbox[1] / frame_h, 4),
                round(bbox[2] / frame_w, 4), round(bbox[3] / frame_h, 4)]

    if rule.startswith("Missing "):
        ppe_item = rule.replace("Missing ", "")
        ppe_classes = get_ppe_groups().get(ppe_item, [ppe_item])
        # Collect center points of all detected PPE items of this type
        ppe_centers = []
        for d in detections:
            if d["class"] in ppe_classes:
                cx = (d["bbox"][0] + d["bbox"][2]) / 2.0
                cy = (d["bbox"][1] + d["bbox"][3]) / 2.0
                ppe_centers.append((cx, cy))
        for d in detections:
            if d["class"] == "person":
                px1, py1, px2, py2 = d["bbox"]
                has_ppe = any(px1 <= cx <= px2 and py1 <= cy <= py2 for cx, cy in ppe_centers)
                if not has_ppe:
                    results.append({"label": f"Missing {ppe_item.title()}", "bbox": normalize(d["bbox"]), "confidence": round(d["confidence"], 2)})
    elif rule == "Zone Intrusion":
        for d in detections:
            if d["class"] == "person":
                results.append({"label": "Zone Intruder", "bbox": normalize(d["bbox"]), "confidence": round(d["confidence"], 2)})
    else:
        # Config-driven: look up the safety rule by name to find its classes
        cfg = get_config()
        all_rules = cfg.get("safety_rules", [])
        matched_rule = next((r for r in all_rules if r["name"] == rule and r.get("type") == "alert"), None)
        if matched_rule:
            rule_classes = matched_rule["classes"]
            for d in detections:
                if d["class"] in rule_classes:
                    results.append({"label": d["class"].title(), "bbox": normalize(d["bbox"]), "confidence": round(d["confidence"], 2)})

    return results


# Per-severity cooldown multipliers (base_cooldown * multiplier)
SEVERITY_COOLDOWN_MULT = {"P1": 1, "P2": 1, "P3": 2, "P4": 5}


def check_violations(detections: list, camera_id: str) -> list:
    """Return candidate violation dicts for COCO-based detections, config-driven from safety_rules."""
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    candidates = []

    # Resolve rule IDs
    rule_ids = cam.get("safety_rule_ids", [])
    if not rule_ids:
        # Backward compat: convert old alert_classes
        old_classes = cam.get("alert_classes", ["mobile_phone", "animal_intrusion"])
        rule_ids = [_LEGACY_ALERT_MAP.get(k, k) for k in old_classes]

    all_rules = {r["id"]: r for r in cfg.get("safety_rules", [])}
    persons = [d for d in detections if d["class"] == "person"]

    for rid in rule_ids:
        rule = all_rules.get(rid)
        if not rule or not rule.get("enabled", True):
            continue
        if rule["type"] != "alert":
            continue
        # Skip zone_intrusion — handled separately
        if rid == "alert_zone_intrusion":
            continue

        matching = [d for d in detections if d["class"] in rule["classes"]]
        if not matching:
            continue

        # Build description
        desc = f"{len(matching)} {rule['name'].lower()} detection(s)"
        if rule["classes"] == ["cell phone"] and persons:
            desc = f"Mobile phone detected near {len(persons)} worker(s)"
        elif "dog" in rule["classes"] or "cat" in rule["classes"]:
            dogs = [d for d in matching if d["class"] == "dog"]
            cats = [d for d in matching if d["class"] == "cat"]
            parts = []
            if dogs:
                parts.append(f"{len(dogs)} dog(s)")
            if cats:
                parts.append(f"{len(cats)} cat(s)")
            desc = f"Animal detected: {', '.join(parts)}" if parts else desc

        candidates.append({
            "camera_id": camera_id,
            "rule": rule["name"],
            "severity": rule["severity"],
            "confidence": max(d["confidence"] for d in matching),
            "description": desc,
            "source": "YOLO",
        })

    return candidates
