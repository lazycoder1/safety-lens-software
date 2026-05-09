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
    show_overlay: bool = True,
    count_override: int | None = None,
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
    records = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                records.append({
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bbox": list(map(int, box.xyxy[0])),
                })

    return draw_detection_records(
        frame,
        records,
        camera_id,
        class_names=class_names,
        colors=colors,
        demo_label=demo_label,
        show_overlay=show_overlay,
        count_override=count_override,
    )


def draw_detection_records(
    frame: np.ndarray,
    records: list[dict],
    camera_id: str,
    class_names=None,
    colors=None,
    demo_label: str | None = None,
    show_overlay: bool = True,
    count_override: int | None = None,
) -> tuple[np.ndarray, list]:
    """Draw normalized detection records from local or remote inference."""
    annotated = frame.copy()
    detections = []

    # Scale font and line thickness based on frame width for crisp text at any resolution
    _h_img, w_img = annotated.shape[:2]
    font_scale = max(0.4, w_img / 1600)
    font_thickness = max(1, int(w_img / 800))
    box_thickness = max(1, int(w_img / 600))

    for record in records:
        cls_id = int(record.get("class_id", record.get("cls", 0)))
        conf = float(record.get("confidence", record.get("conf", 0.0)))
        x1, y1, x2, y2 = map(int, record["bbox"])

        # Resolve class name
        if record.get("class"):
            cls_name = record["class"]
        elif class_names is None:
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

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    if show_overlay:
        count_color = (168, 85, 247) if colors is not None and isinstance(colors, list) else (34, 197, 94)
        annotated = apply_camera_overlay(
            annotated,
            camera_id=camera_id,
            detection_count=count_override if count_override is not None else len(detections),
            demo_label=demo_label,
            count_color=count_color,
        )

    return annotated, detections


def apply_camera_overlay(
    frame: np.ndarray,
    *,
    camera_id: str,
    detection_count: int,
    demo_label: str | None = None,
    count_color: tuple[int, int, int] = (34, 197, 94),
) -> np.ndarray:
    """Draw shared zone and camera overlay on an already-annotated frame."""
    annotated = frame
    h_img, w_img = annotated.shape[:2]
    font_scale = max(0.4, w_img / 1600)
    font_thickness = max(1, int(w_img / 800))
    box_thickness = max(1, int(w_img / 600))

    # Overlay camera info
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    cam_name = cam.get("name", camera_id)

    # Overlay zone polygons so operators can visually verify the configured zone
    zones = cam.get("zones", [])
    if zones:
        fh, fw = annotated.shape[:2]
        overlay = annotated.copy()
        for zone in zones:
            points = zone.get("points", [])
            if len(points) < 3:
                continue
            pts = np.array([[int(x * fw), int(y * fh)] for x, y in points], np.int32)
            # Hex "#rrggbb" -> BGR for OpenCV; default to red for restricted
            hex_color = (zone.get("color") or "#dc2626").lstrip("#")
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                bgr = (b, g, r)
            except ValueError:
                bgr = (38, 38, 220)
            cv2.fillPoly(overlay, [pts], bgr)
            cv2.polylines(annotated, [pts], isClosed=True, color=bgr, thickness=box_thickness, lineType=cv2.LINE_AA)
            # Zone name near centroid
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            zone_name = zone.get("name", "zone")
            (tw, th), _ = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 2), bgr, -1)
            cv2.putText(annotated, zone_name, (cx - tw // 2, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

    overlay_text = cam_name

    overlay_font_scale = max(0.5, w_img / 1400)
    overlay_thickness = max(1, int(w_img / 600))
    overlay_y = int(30 * (w_img / 1280))
    count_y = int(55 * (w_img / 1280))

    cv2.putText(annotated, overlay_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, overlay_font_scale, (255, 255, 255), overlay_thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, overlay_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, overlay_font_scale, (0, 0, 0), overlay_thickness, cv2.LINE_AA)

    count_text = f"{detection_count} detections"
    cv2.putText(annotated, count_text, (10, count_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
    cv2.putText(annotated, count_text, (10, count_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, count_color, font_thickness, cv2.LINE_AA)

    return annotated


# ── Fall / Man-Down detection (YOLO-pose) ──────────────────────────────────

# COCO keypoint indices (17-point format)
_KP_NOSE = 0
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12
_KP_LEFT_KNEE = 13
_KP_RIGHT_KNEE = 14

# Skeleton connections for visualization
_SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (0, 1), (0, 2), (1, 3), (2, 4),              # head
]
_SKELETON_COLOR = (0, 255, 255)  # cyan
_FALL_COLOR = (0, 0, 255)  # red


def _midpoint(kp_a, kp_b):
    """Average two keypoint (x, y) tuples."""
    return ((kp_a[0] + kp_b[0]) / 2.0, (kp_a[1] + kp_b[1]) / 2.0)


def _is_fall(keypoints, bbox, conf_threshold: float = 0.3) -> bool:
    """Determine if a person is in a fallen state using keypoint heuristics.

    Checks:
    1. Bounding box aspect ratio (wider than tall → lying down)
    2. Torso angle (shoulder-hip line deviates from vertical)
    3. Hip near or below shoulder level (inverted / horizontal posture)
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Heuristic 1: Aspect ratio — lying persons have wide, short boxes
    if bbox_h > 0 and bbox_w / bbox_h > 1.4:
        return True

    # Extract key body points (x, y, confidence)
    left_shoulder = keypoints[_KP_LEFT_SHOULDER]
    right_shoulder = keypoints[_KP_RIGHT_SHOULDER]
    left_hip = keypoints[_KP_LEFT_HIP]
    right_hip = keypoints[_KP_RIGHT_HIP]

    # Need sufficient confidence on at least shoulders and hips
    shoulder_conf = min(left_shoulder[2], right_shoulder[2])
    hip_conf = min(left_hip[2], right_hip[2])
    if shoulder_conf < conf_threshold or hip_conf < conf_threshold:
        return False

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)

    # Heuristic 2: Torso angle — angle of shoulder→hip line vs vertical
    dx = abs(hip_mid[0] - shoulder_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1])
    if dy < 1:
        # Nearly horizontal torso
        return True
    import math
    torso_angle_deg = math.degrees(math.atan2(dx, dy))
    if torso_angle_deg > 50:
        return True

    # Heuristic 3: Hip at same level or above shoulders (person lying flat)
    if hip_mid[1] <= shoulder_mid[1] + 10:
        # In image coords, lower Y = higher position. If hip Y ≤ shoulder Y,
        # person may be lying with feet up or fully horizontal.
        # Only flag if bbox is also somewhat wide
        if bbox_h > 0 and bbox_w / bbox_h > 0.9:
            return True

    return False


def check_fall_detections(results, camera_id: str, frame: np.ndarray) -> list:
    """Analyze YOLO-pose results to detect fallen persons.
    Returns candidate violation dicts."""
    candidates = []
    if not results or len(results) == 0:
        return candidates

    result = results[0]
    if result.keypoints is None or result.boxes is None:
        return candidates

    keypoints_data = result.keypoints.data  # (N, 17, 3) — x, y, conf
    boxes = result.boxes

    fallen_count = 0
    max_conf = 0.0

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        bbox = list(map(int, boxes.xyxy[i]))
        kps = keypoints_data[i].cpu().numpy()  # (17, 3)

        if _is_fall(kps, bbox):
            fallen_count += 1
            max_conf = max(max_conf, conf)

    if fallen_count > 0:
        candidates.append({
            "camera_id": camera_id,
            "rule": "Fall Detected",
            "severity": "P1",
            "confidence": max_conf,
            "description": f"{fallen_count} person(s) detected in fallen/man-down position",
            "source": "Pose Specialist",
            "threshold": 8,
        })

    return candidates


def draw_pose_detections(frame: np.ndarray, results, fall_only: bool = False) -> tuple[np.ndarray, list]:
    """Draw skeleton keypoints on frame. Returns annotated frame and fall detections list."""
    annotated = frame.copy()
    fall_detections = []

    if not results or len(results) == 0:
        return annotated, fall_detections

    result = results[0]
    if result.keypoints is None or result.boxes is None:
        return annotated, fall_detections

    h_img, w_img = annotated.shape[:2]
    font_scale = max(0.4, w_img / 1600)
    font_thickness = max(1, int(w_img / 800))
    kp_radius = max(3, int(w_img / 400))

    keypoints_data = result.keypoints.data
    boxes = result.boxes

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        bbox = list(map(int, boxes.xyxy[i]))
        kps = keypoints_data[i].cpu().numpy()

        is_fallen = _is_fall(kps, bbox)

        if fall_only and not is_fallen:
            continue

        color = _FALL_COLOR if is_fallen else _SKELETON_COLOR
        box_thickness = max(2, int(w_img / 400))

        # Draw bounding box
        if is_fallen:
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), _FALL_COLOR, box_thickness, cv2.LINE_AA)
            label = f"FALL {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated, (bbox[0], bbox[1] - th - 8), (bbox[0] + tw + 4, bbox[1]), _FALL_COLOR, -1)
            cv2.putText(annotated, label, (bbox[0] + 2, bbox[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            fall_detections.append({
                "class": "person_fall",
                "confidence": conf,
                "bbox": bbox,
            })

        # Draw skeleton
        for (a, b) in _SKELETON_PAIRS:
            if kps[a][2] > 0.3 and kps[b][2] > 0.3:
                pt1 = (int(kps[a][0]), int(kps[a][1]))
                pt2 = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(annotated, pt1, pt2, color, max(1, kp_radius // 2), cv2.LINE_AA)

        # Draw keypoints
        for kp in kps:
            if kp[2] > 0.3:
                cv2.circle(annotated, (int(kp[0]), int(kp[1])), kp_radius, color, -1, cv2.LINE_AA)

    return annotated, fall_detections


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


def get_ppe_threshold_map() -> dict[str, int | None]:
    """Get per-rule threshold per PPE group from config."""
    cfg = get_config()
    rules = cfg.get("safety_rules", [])
    return {r["name"].lower(): r.get("threshold") for r in rules if r.get("type") == "ppe" and r.get("enabled", True)}


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
        threshold_map = {}
        for rid in safety_rule_ids:
            rule = rule_map.get(rid)
            if rule and rule.get("type") == "ppe" and rule.get("enabled", True):
                key = rule["name"].lower()
                ppe_groups[key] = rule["classes"]
                severity_map[key] = rule.get("severity", "P2")
                threshold_map[key] = rule.get("threshold")
        checked_groups: set[str] = set(ppe_groups)
    else:
        # Fallback: match yoloe_classes against all known PPE groups
        ppe_groups = get_ppe_groups()
        severity_map = get_ppe_severity_map()
        threshold_map = get_ppe_threshold_map()
        yoloe_classes = cam.get("yoloe_classes", [])
        checked_groups = set()
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
            candidates.append({
                "camera_id": camera_id,
                "rule": f"Missing {group_name}",
                "severity": severity_map.get(group_name, "P2"),
                "confidence": max(p["confidence"] for p in violating_persons),
                "description": f"{len(violating_persons)} worker(s) detected without {group_name}",
                "source": "PPE Specialist",
                "threshold": threshold_map.get(group_name),
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

    # Per-rule threshold override (e.g. require sustained presence before firing)
    zone_rule = next(
        (r for r in cfg.get("safety_rules", []) if r.get("id") == "alert_zone_intrusion"),
        None,
    )
    rule_threshold = zone_rule.get("threshold") if zone_rule else None

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
                "threshold": rule_threshold,
            })

    return candidates


def extract_violation_bboxes(rule: str, detections: list, frame_w: int, frame_h: int, camera_id: str | None = None) -> list[dict]:
    """Extract and normalize bboxes relevant to a specific violation rule.
    Returns list of {label, bbox: [x1_norm, y1_norm, x2_norm, y2_norm], confidence}."""
    results = []

    def normalize(bbox):
        return [round(bbox[0] / frame_w, 4), round(bbox[1] / frame_h, 4),
                round(bbox[2] / frame_w, 4), round(bbox[3] / frame_h, 4)]

    if rule == "Fall Detected":
        for d in detections:
            if d["class"] == "person_fall":
                results.append({"label": "Fall / Man Down", "bbox": normalize(d["bbox"]), "confidence": round(d["confidence"], 2)})
        return results

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
        # Only include persons whose bbox actually overlaps one of this camera's zones.
        cfg = get_config()
        zones = cfg["cameras"].get(camera_id, {}).get("zones", []) if camera_id else []
        zones = [z for z in zones if len(z.get("points", [])) >= 3]
        if not zones:
            return results
        for d in detections:
            if d["class"] != "person":
                continue
            for zone in zones:
                if bbox_intersects_polygon(d["bbox"], zone["points"], frame_w, frame_h):
                    zone_name = zone.get("name", "zone")
                    results.append({
                        "label": f"Intruder — {zone_name}",
                        "bbox": normalize(d["bbox"]),
                        "confidence": round(d["confidence"], 2),
                    })
                    break  # don't double-count persons who overlap multiple zones
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
        elif "dog" in rule["classes"] or "cat" in rule["classes"] or "animal" in rule["classes"]:
            animal_counts: dict[str, int] = {}
            for d in matching:
                cls = d["class"]
                animal_counts[cls] = animal_counts.get(cls, 0) + 1
            parts = [f"{count} {name}(s)" for name, count in animal_counts.items()]
            desc = f"Animal detected: {', '.join(parts)}" if parts else desc

        matching_sources = {d.get("model_family") for d in matching if d.get("model_family")}
        source_label = "YOLO"
        if matching_sources == {"yoloe_long_tail"}:
            source_label = "YOLOE Long-Tail"
        elif matching_sources == {"ppe_specialist"}:
            source_label = "PPE Specialist"
        elif matching_sources == {"coco_primary"}:
            source_label = "COCO Primary"

        candidates.append({
            "camera_id": camera_id,
            "rule": rule["name"],
            "severity": rule["severity"],
            "confidence": max(d["confidence"] for d in matching),
            "description": desc,
            "source": source_label,
            "threshold": rule.get("threshold"),
        })

    return candidates
