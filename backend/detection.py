"""
Rakshak Lens detection logic — drawing, PPE checks, zone intrusions, violations.
"""

import logging
import math

import cv2
import numpy as np

from capability_registry import RULE_ID_TO_CAPABILITY
from config_manager import get_config
from constants import COCO_NAMES, CLASS_COLORS

logger = logging.getLogger("rakshak_lens")

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


def _pose_hit_counts(keypoints, conf_threshold: float = 0.3) -> tuple[int, int]:
    torso_points = (
        keypoints[_KP_LEFT_SHOULDER],
        keypoints[_KP_RIGHT_SHOULDER],
        keypoints[_KP_LEFT_HIP],
        keypoints[_KP_RIGHT_HIP],
    )
    leg_points = (
        keypoints[_KP_LEFT_KNEE],
        keypoints[_KP_RIGHT_KNEE],
    )
    torso_hits = sum(1 for point in torso_points if point[2] >= conf_threshold)
    leg_hits = sum(1 for point in leg_points if point[2] >= conf_threshold)
    return torso_hits, leg_hits


def _has_usable_body_pose(keypoints, conf_threshold: float = 0.3) -> bool:
    torso_hits, leg_hits = _pose_hit_counts(keypoints, conf_threshold)
    return torso_hits >= 3 and (torso_hits == 4 or leg_hits >= 1)


def _round_keypoint(point) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2), round(float(point[2]), 4)]


def _fall_detection_settings(camera_id: str | None) -> dict:
    if not camera_id:
        return {}
    cfg = get_config()
    camera = cfg.get("cameras", {}).get(camera_id, {})
    settings = camera.get("fall_detection") or camera.get("fallDetection") or {}
    return settings if isinstance(settings, dict) else {}


def _setting_enabled(settings: dict, *keys: str) -> bool:
    return any(settings.get(key) is True for key in keys)


def _fall_suppression_reason(analysis: dict, settings: dict) -> str | None:
    if (
        _setting_enabled(settings, "suppress_floor_exercise", "suppressFloorExercise")
        and bool(analysis.get("floorExercisePose"))
    ):
        return "floor_exercise_pose"
    return None


def _fall_confirmation_threshold(settings: dict) -> int:
    for key in ("confirmation_threshold", "confirmationThreshold", "threshold"):
        value = settings.get(key)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 8


def _fall_analysis(keypoints, bbox, conf_threshold: float = 0.3) -> dict:
    """Analyze a pose for fall/man-down heuristics and explain the decision."""
    # Extract key body points (x, y, confidence)
    left_shoulder = keypoints[_KP_LEFT_SHOULDER]
    right_shoulder = keypoints[_KP_RIGHT_SHOULDER]
    left_hip = keypoints[_KP_LEFT_HIP]
    right_hip = keypoints[_KP_RIGHT_HIP]
    left_knee = keypoints[_KP_LEFT_KNEE]
    right_knee = keypoints[_KP_RIGHT_KNEE]
    torso_hits, leg_hits = _pose_hit_counts(keypoints, conf_threshold)
    usable_pose = bool(torso_hits >= 3 and (torso_hits == 4 or leg_hits >= 1))
    x1, y1, x2, y2 = bbox
    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)
    aspect_ratio = (bbox_w / bbox_h) if bbox_h > 0 else 0.0

    analysis = {
        "isFall": False,
        "reason": "upright_or_not_fall",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "bboxWidth": int(bbox_w),
        "bboxHeight": int(bbox_h),
        "aspectRatio": round(float(aspect_ratio), 4),
        "confThreshold": conf_threshold,
        "torsoHits": torso_hits,
        "legHits": leg_hits,
        "usablePose": usable_pose,
        "keypoints": {
            "leftShoulder": _round_keypoint(left_shoulder),
            "rightShoulder": _round_keypoint(right_shoulder),
            "leftHip": _round_keypoint(left_hip),
            "rightHip": _round_keypoint(right_hip),
            "leftKnee": _round_keypoint(left_knee),
            "rightKnee": _round_keypoint(right_knee),
        },
    }

    # A P1 fall alert should never come from a box shape alone. Require a
    # usable body pose before applying posture heuristics.
    if not usable_pose:
        analysis["reason"] = "insufficient_pose"
        return analysis

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    visible_knees = [
        knee for knee in (left_knee, right_knee)
        if knee[2] >= conf_threshold
    ]
    knee_mid = None
    if visible_knees:
        knee_mid = (
            sum(float(knee[0]) for knee in visible_knees) / len(visible_knees),
            sum(float(knee[1]) for knee in visible_knees) / len(visible_knees),
        )

    knee_raise_margin = max(20.0, bbox_h * 0.12)
    raised_knee_count = sum(1 for knee in visible_knees if float(knee[1]) < hip_mid[1] - knee_raise_margin)
    hips_below_shoulders = bool(hip_mid[1] > shoulder_mid[1] + max(10.0, bbox_h * 0.08))
    analysis["shoulderMidpoint"] = [round(float(shoulder_mid[0]), 2), round(float(shoulder_mid[1]), 2)]
    analysis["hipMidpoint"] = [round(float(hip_mid[0]), 2), round(float(hip_mid[1]), 2)]
    analysis["kneeMidpoint"] = (
        [round(float(knee_mid[0]), 2), round(float(knee_mid[1]), 2)]
        if knee_mid is not None
        else None
    )
    analysis["raisedKneeCount"] = raised_knee_count
    analysis["hipsBelowShoulders"] = hips_below_shoulders

    # Torso angle — angle of shoulder→hip line vs vertical.
    dx = abs(hip_mid[0] - shoulder_mid[0])
    dy = abs(hip_mid[1] - shoulder_mid[1])
    if dy < 1:
        torso_angle_deg = 90.0
    else:
        torso_angle_deg = math.degrees(math.atan2(dx, dy))
    analysis["torsoAngleDeg"] = round(float(torso_angle_deg), 2)
    floor_exercise_pose = bool(
        raised_knee_count > 0
        and (
            hips_below_shoulders
            or (aspect_ratio > 2.0 and torso_angle_deg > 50)
        )
    )
    analysis["floorExercisePose"] = floor_exercise_pose

    # Heuristic 1: Aspect ratio — lying persons have wide, short boxes
    if aspect_ratio > 1.4:
        analysis["isFall"] = True
        analysis["reason"] = "wide_body_box"
        return analysis

    # Heuristic 2: nearly horizontal torso.
    if torso_angle_deg > 50:
        analysis["isFall"] = True
        analysis["reason"] = "horizontal_torso"
        return analysis

    # Heuristic 3: Hip at same level or above shoulders (person lying flat)
    hips_at_or_above_shoulders = bool(hip_mid[1] <= shoulder_mid[1] + 10)
    analysis["hipsAtOrAboveShoulders"] = hips_at_or_above_shoulders
    if hips_at_or_above_shoulders:
        # In image coords, lower Y = higher position. If hip Y ≤ shoulder Y,
        # person may be lying with feet up or fully horizontal.
        # Only flag if bbox is also somewhat wide
        if aspect_ratio > 0.9:
            analysis["isFall"] = True
            analysis["reason"] = "hips_at_or_above_shoulders"
            return analysis

    return analysis


def _is_fall(keypoints, bbox, conf_threshold: float = 0.3) -> bool:
    """Determine if a person is in a fallen state using keypoint heuristics."""
    return bool(_fall_analysis(keypoints, bbox, conf_threshold)["isFall"])


def _pose_observations(results) -> list[tuple[float, list[int], np.ndarray]]:
    """Normalize local Ultralytics results or remote pose records once."""
    if not results or len(results) == 0:
        return []
    if isinstance(results[0], dict):
        observations = []
        for record in results:
            keypoints = record.get("keypoints")
            bbox = record.get("bbox")
            if not isinstance(keypoints, list) or not isinstance(bbox, list):
                continue
            observations.append(
                (
                    float(record.get("confidence", 0.0)),
                    list(map(int, bbox)),
                    np.asarray(keypoints, dtype=np.float32),
                )
            )
        return observations

    result = results[0]
    if result.keypoints is None or result.boxes is None:
        return []
    keypoints_data = result.keypoints.data
    boxes = result.boxes
    return [
        (
            float(boxes.conf[index]),
            list(map(int, boxes.xyxy[index])),
            keypoints_data[index].cpu().numpy(),
        )
        for index in range(len(boxes))
    ]


def check_fall_detections(results, camera_id: str, frame: np.ndarray) -> list:
    """Analyze YOLO-pose results to detect fallen persons.
    Returns candidate violation dicts."""
    candidates = []
    observations = _pose_observations(results)
    if not observations:
        return candidates

    fallen_count = 0
    max_conf = 0.0
    pose_diagnostics = []
    fall_settings = _fall_detection_settings(camera_id)
    confirmation_threshold = _fall_confirmation_threshold(fall_settings)

    for i, (conf, bbox, kps) in enumerate(observations):
        analysis = _fall_analysis(kps, bbox)
        analysis["confidence"] = round(conf, 4)
        analysis["index"] = i
        suppression_reason = _fall_suppression_reason(analysis, fall_settings)
        if analysis["isFall"] and suppression_reason:
            analysis["rawIsFall"] = True
            analysis["isFall"] = False
            analysis["suppressed"] = True
            analysis["suppressionReason"] = suppression_reason
        pose_diagnostics.append(analysis)

        if analysis["isFall"]:
            fallen_count += 1
            max_conf = max(max_conf, conf)

    if fallen_count > 0:
        frame_h, frame_w = frame.shape[:2]
        candidates.append({
            "camera_id": camera_id,
            "rule": "Fall Detected",
            "severity": "P1",
            "confidence": max_conf,
            "description": f"{fallen_count} person(s) detected in fallen/man-down position",
            "source": "Pose Specialist",
            "threshold": confirmation_threshold,
            "metadata": {
                "frameWidth": frame_w,
                "frameHeight": frame_h,
                "fallenCount": fallen_count,
                "fallConfirmationThreshold": confirmation_threshold,
                "fallDetectionSettings": fall_settings,
                "poseDiagnostics": pose_diagnostics,
            },
        })

    return candidates


def draw_pose_detections(frame: np.ndarray, results, fall_only: bool = False, camera_id: str | None = None) -> tuple[np.ndarray, list]:
    """Draw skeleton keypoints on frame. Returns annotated frame and fall detections list."""
    annotated = frame.copy()
    fall_detections = []

    observations = _pose_observations(results)
    if not observations:
        return annotated, fall_detections

    h_img, w_img = annotated.shape[:2]
    font_scale = max(0.4, w_img / 1600)
    font_thickness = max(1, int(w_img / 800))
    kp_radius = max(3, int(w_img / 400))

    fall_settings = _fall_detection_settings(camera_id)

    for conf, bbox, kps in observations:
        analysis = _fall_analysis(kps, bbox)
        is_fallen = bool(analysis["isFall"]) and not _fall_suppression_reason(analysis, fall_settings)

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

RIDER_HELMET_RULE_ID = "ppe_rider_helmet"
RIDER_HELMET_GROUP = "rider helmet"
RIDER_VEHICLE_CLASSES = {"motorcycle", "motorbike", "scooter"}
RIDER_MIN_PERSON_CONFIDENCE = 0.65
RIDER_MIN_VEHICLE_CONFIDENCE = 0.70
RIDER_MIN_PERSON_HEIGHT_RATIO = 0.12
RIDER_MIN_VEHICLE_HEIGHT_RATIO = 0.06


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


def _positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _camera_ppe_threshold_override(cam: dict, rule_id: str | None, group_name: str) -> int | None:
    overrides = (
        cam.get("safety_rule_overrides")
        or cam.get("safetyRuleOverrides")
        or cam.get("rule_overrides")
        or {}
    )
    if not isinstance(overrides, dict):
        return None

    for key in (rule_id, group_name):
        if not key or key not in overrides:
            continue
        override = overrides.get(key)
        value = override.get("threshold") if isinstance(override, dict) else override
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    return None


def _source_label_for_model_families(model_families: set[str]) -> str:
    if model_families == {"yoloe_long_tail"}:
        return "YOLOE Long-Tail"
    if model_families == {"fire_smoke_specialist"}:
        return "Fire / Smoke Specialist"
    if model_families == {"ppe_closed_set_candidate"}:
        return "Closed-Set PPE Candidate"
    if model_families == {"ppe_specialist"}:
        return "PPE Specialist"
    if model_families == {"coco_primary"}:
        return "COCO Primary"
    if model_families == {"rtdetr_phone"}:
        return "RT-DETR Phone Recall"
    return "YOLO"


def _expanded_bbox(bbox: list, *, x_ratio: float = 0.12, y_ratio: float = 0.08) -> list[float]:
    x1, y1, x2, y2 = bbox
    width = max(0.0, float(x2 - x1))
    height = max(0.0, float(y2 - y1))
    return [
        float(x1) - width * x_ratio,
        float(y1) - height * y_ratio,
        float(x2) + width * x_ratio,
        float(y2) + height * y_ratio,
    ]


def _bbox_intersection_area(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _bbox_area(bbox: list) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _ppe_match_detail(person_bbox: list, ppe_bbox: list) -> dict:
    """Describe whether a PPE detection plausibly belongs to a person box."""
    px1, py1, px2, py2 = person_bbox
    cx = (ppe_bbox[0] + ppe_bbox[2]) / 2.0
    cy = (ppe_bbox[1] + ppe_bbox[3]) / 2.0
    center_inside_person = px1 <= cx <= px2 and py1 <= cy <= py2

    ex1, ey1, ex2, ey2 = _expanded_bbox(person_bbox)
    center_inside_expanded_person = ex1 <= cx <= ex2 and ey1 <= cy <= ey2

    overlap = _bbox_intersection_area(person_bbox, ppe_bbox)
    ppe_area = _bbox_area(ppe_bbox)
    person_area = _bbox_area(person_bbox)
    reference_area = min(ppe_area, person_area)
    overlap_reference_ratio = overlap / reference_area if reference_area > 0 else 0.0
    overlap_matched = overlap_reference_ratio >= 0.25
    matched = center_inside_person or center_inside_expanded_person or overlap_matched
    reason = None
    if center_inside_person:
        reason = "center_inside_person"
    elif center_inside_expanded_person:
        reason = "center_inside_expanded_person"
    elif overlap_matched:
        reason = "bbox_overlap"

    return {
        "matched": matched,
        "reason": reason,
        "ppeCenter": [round(float(cx), 2), round(float(cy), 2)],
        "centerInsidePerson": center_inside_person,
        "centerInsideExpandedPerson": center_inside_expanded_person,
        "overlapArea": round(float(overlap), 2),
        "overlapReferenceRatio": round(float(overlap_reference_ratio), 4),
    }


def _ppe_matches_person(person_bbox: list, ppe_bbox: list) -> bool:
    """True when a PPE detection plausibly belongs to a person box."""
    return bool(_ppe_match_detail(person_bbox, ppe_bbox)["matched"])


def _ppe_present_for_person(person_bbox: list, ppe_dets: list) -> bool:
    """Check if any PPE detection is geometrically associated with a person."""
    for p in ppe_dets:
        if _ppe_matches_person(person_bbox, p["bbox"]):
            return True
    return False


def _person_evaluable_for_ppe(person_bbox: list, frame_w: int | None = None, frame_h: int | None = None) -> bool:
    x1, y1, x2, y2 = map(float, person_bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < 24 or height < 48:
        return False
    if not frame_w or not frame_h:
        return True
    width_ratio = width / float(frame_w)
    height_ratio = height / float(frame_h)
    area_ratio = (width * height) / float(frame_w * frame_h)
    touches_horizontal_edge = x1 <= 2 or x2 >= frame_w - 2
    if width_ratio < 0.03 or height_ratio < 0.10 or area_ratio < 0.003:
        return False
    if touches_horizontal_edge and width_ratio < 0.08:
        return False
    return True


def _round_bbox(bbox: list) -> list[float]:
    return [round(float(v), 2) for v in bbox]


def _ppe_scope_zones(cam: dict) -> list[dict]:
    zones = cam.get("zones") or []
    if not isinstance(zones, list):
        return []
    scope_zones = []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        zone_type = str(zone.get("type") or "").lower()
        analytics = str(zone.get("analytics") or "").lower()
        if zone_type in {"ppe", "ppe_evaluation", "ppe_compliance"} or analytics in {"ppe", "ppe_evaluation", "ppe_compliance"}:
            if len(zone.get("points") or []) >= 3:
                scope_zones.append(zone)
    return scope_zones


def _person_in_ppe_scope(person_bbox: list, scope_zones: list[dict], frame_w: int | None, frame_h: int | None) -> bool:
    if not scope_zones:
        return True
    if not frame_w or not frame_h:
        return True
    return any(bbox_intersects_polygon(person_bbox, zone.get("points", []), frame_w, frame_h) for zone in scope_zones)


def _detection_class(value) -> str:
    return str(value or "").lower().replace("_", " ").replace("-", " ").strip()


def _rider_vehicle_match_detail(person_bbox: list, vehicle_bbox: list) -> dict:
    px1, py1, px2, py2 = map(float, person_bbox)
    vx1, vy1, vx2, vy2 = map(float, vehicle_bbox)
    person_w = max(0.0, px2 - px1)
    person_h = max(0.0, py2 - py1)
    vehicle_w = max(0.0, vx2 - vx1)
    vehicle_h = max(0.0, vy2 - vy1)
    if person_w <= 0 or person_h <= 0 or vehicle_w <= 0 or vehicle_h <= 0:
        return {"matched": False, "reason": "invalid_bbox", "score": 0.0}

    person_cx = (px1 + px2) / 2.0
    foot_x = person_cx
    foot_y = py2
    expanded_vehicle = _expanded_bbox(vehicle_bbox, x_ratio=0.35, y_ratio=0.45)
    ex1, ey1, ex2, ey2 = expanded_vehicle
    foot_inside_expanded_vehicle = ex1 <= foot_x <= ex2 and ey1 <= foot_y <= ey2
    center_x_near_vehicle = ex1 <= person_cx <= ex2

    horizontal_overlap = max(0.0, min(px2, vx2) - max(px1, vx1))
    horizontal_overlap_ratio = horizontal_overlap / min(person_w, vehicle_w)

    lower_person_bbox = [px1, py1 + person_h * 0.45, px2, py2]
    expanded_vehicle_core = _expanded_bbox(vehicle_bbox, x_ratio=0.20, y_ratio=0.25)
    lower_overlap = _bbox_intersection_area(lower_person_bbox, expanded_vehicle_core)
    lower_reference_area = min(_bbox_area(lower_person_bbox), _bbox_area(expanded_vehicle_core))
    lower_overlap_ratio = lower_overlap / lower_reference_area if lower_reference_area > 0 else 0.0

    posture_near_vehicle = center_x_near_vehicle and horizontal_overlap_ratio >= 0.18
    matched = (
        posture_near_vehicle
        and (foot_inside_expanded_vehicle or lower_overlap_ratio >= 0.08)
    ) or lower_overlap_ratio >= 0.18
    reason = None
    if matched and foot_inside_expanded_vehicle:
        reason = "foot_on_vehicle"
    elif matched:
        reason = "lower_body_vehicle_overlap"

    score = max(horizontal_overlap_ratio, lower_overlap_ratio)
    return {
        "matched": bool(matched),
        "reason": reason,
        "score": round(float(score), 4),
        "footInsideExpandedVehicle": bool(foot_inside_expanded_vehicle),
        "centerXNearVehicle": bool(center_x_near_vehicle),
        "horizontalOverlapRatio": round(float(horizontal_overlap_ratio), 4),
        "lowerBodyVehicleOverlapRatio": round(float(lower_overlap_ratio), 4),
    }


def _rider_vehicle_associations(persons: list, vehicles: list) -> list[dict]:
    associations = []
    for person in persons:
        best_match = None
        for vehicle in vehicles:
            detail = _rider_vehicle_match_detail(person["bbox"], vehicle["bbox"])
            if not detail["matched"]:
                continue
            candidate = {"person": person, "vehicle": vehicle, "detail": detail}
            if best_match is None or detail["score"] > best_match["detail"]["score"]:
                best_match = candidate
        if best_match:
            associations.append(best_match)
    return associations


def has_ppe_specialist_context(
    detections: list[dict],
    ppe_capabilities: set[str],
    camera: dict,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> bool:
    """Return whether a PPE specialist can produce an actionable result.

    Keep this gate aligned with the geometry, confidence, and zone filters used
    by ``check_yoloe_violations``. Running the specialist for context that the
    policy layer will immediately discard wastes GPU time and inflates model
    duty without improving alert recall.
    """
    coco_detections = [
        detection
        for detection in detections
        if detection.get("model_family") == "coco_primary"
        and detection.get("bbox")
    ]
    scope_zones = _ppe_scope_zones(camera)
    persons = [
        detection
        for detection in coco_detections
        if _detection_class(detection.get("class")) == "person"
        and _person_evaluable_for_ppe(detection["bbox"], frame_w, frame_h)
        and _person_in_ppe_scope(
            detection["bbox"],
            scope_zones,
            frame_w,
            frame_h,
        )
    ]
    if not persons:
        return False

    if ppe_capabilities != {"rider_helmet_required"}:
        return True

    rider_persons = [
        person
        for person in persons
        if _rider_person_evaluable(person, frame_h)
    ]
    vehicles = [
        detection
        for detection in coco_detections
        if _detection_class(detection.get("class")) in RIDER_VEHICLE_CLASSES
        and _rider_vehicle_evaluable(detection, frame_h)
    ]
    return bool(_rider_vehicle_associations(rider_persons, vehicles))


def _bbox_height_ratio(bbox: list, frame_h: int | None) -> float | None:
    if not frame_h:
        return None
    _x1, y1, _x2, y2 = map(float, bbox)
    return max(0.0, y2 - y1) / float(frame_h)


def _rider_person_evaluable(person: dict, frame_h: int | None) -> bool:
    if float(person.get("confidence", 0.0)) < RIDER_MIN_PERSON_CONFIDENCE:
        return False
    height_ratio = _bbox_height_ratio(person["bbox"], frame_h)
    if height_ratio is not None and height_ratio < RIDER_MIN_PERSON_HEIGHT_RATIO:
        return False
    return True


def _rider_vehicle_evaluable(vehicle: dict, frame_h: int | None) -> bool:
    if float(vehicle.get("confidence", 0.0)) < RIDER_MIN_VEHICLE_CONFIDENCE:
        return False
    height_ratio = _bbox_height_ratio(vehicle["bbox"], frame_h)
    if height_ratio is not None and height_ratio < RIDER_MIN_VEHICLE_HEIGHT_RATIO:
        return False
    return True


def _rider_helmet_rule_for_camera(cam: dict, cfg: dict) -> dict | None:
    assigned_rule_ids = cam.get("safety_rule_ids") or cam.get("ppe_rule_ids") or []
    if RIDER_HELMET_RULE_ID not in assigned_rule_ids:
        return None
    rule = {r["id"]: r for r in cfg.get("safety_rules", [])}.get(RIDER_HELMET_RULE_ID)
    if rule and rule.get("enabled", True) and rule.get("type") == "ppe":
        return rule
    return None


def _check_rider_helmet_violations(
    detections: list,
    camera_id: str,
    cam: dict,
    rule: dict,
    persons: list,
    all_persons: list,
    scope_zones: list[dict],
    frame_w: int | None,
    frame_h: int | None,
) -> list[dict]:
    raw_vehicles = [
        d for d in detections
        if d.get("bbox") and _detection_class(d.get("class")) in RIDER_VEHICLE_CLASSES
    ]
    vehicles = [d for d in raw_vehicles if _rider_vehicle_evaluable(d, frame_h)]
    if not vehicles:
        return []

    rider_persons = [p for p in persons if _rider_person_evaluable(p, frame_h)]
    if not rider_persons:
        return []

    ppe_classes = rule.get("classes") or ["motorcycle helmet", "rider helmet", "helmet"]
    ppe_dets = [d for d in detections if d.get("bbox") and d.get("class") in ppe_classes]
    rider_associations = _rider_vehicle_associations(rider_persons, vehicles)
    if not rider_associations:
        return []

    violating_associations = [
        association for association in rider_associations
        if not _ppe_present_for_person(association["person"]["bbox"], ppe_dets)
    ]
    if not violating_associations:
        return []

    rule_threshold = _positive_int(rule.get("threshold"))
    override_threshold = _camera_ppe_threshold_override(cam, rule.get("id"), RIDER_HELMET_GROUP)
    threshold = override_threshold if override_threshold is not None else rule_threshold
    threshold_source = "camera_override" if override_threshold is not None else (
        "safety_rule" if rule_threshold is not None else "default"
    )
    person_diagnostics = _ppe_person_diagnostics(all_persons, ppe_dets, frame_w, frame_h, scope_zones)
    confidence = max(
        max(
            float(association["person"].get("confidence", 0.0)),
            float(association["vehicle"].get("confidence", 0.0)),
        )
        for association in violating_associations
    )

    return [{
        "camera_id": camera_id,
        "rule": "Missing rider helmet",
        "severity": rule.get("severity", "P2"),
        "confidence": confidence,
        "count": len(violating_associations),
        "classes": [
            "rider helmet",
            "missing rider helmet",
            "no rider helmet",
            "person",
            *sorted(RIDER_VEHICLE_CLASSES),
            *ppe_classes,
        ],
        "description": f"{len(violating_associations)} rider(s) detected without helmet",
        "source": "COCO Primary + PPE Specialist",
        "threshold": threshold,
        "metadata": {
            "ppeGroup": RIDER_HELMET_GROUP,
            "safetyRuleId": rule.get("id"),
            "ruleThreshold": threshold,
            "thresholdSource": threshold_source,
            "personCount": len(all_persons),
            "evaluablePersonCount": len(persons),
            "riderCandidatePersonCount": len(rider_persons),
            "rawVehicleDetectionCount": len(raw_vehicles),
            "vehicleDetectionCount": len(vehicles),
            "riderCount": len(rider_associations),
            "coveredRiderCount": len(rider_associations) - len(violating_associations),
            "violatingRiderCount": len(violating_associations),
            "ppeDetectionCount": len(ppe_dets),
            "ppeEvaluationZoneIds": [zone.get("id") or zone.get("name") for zone in scope_zones],
            "frameWidth": frame_w,
            "frameHeight": frame_h,
            "riderVehicleClasses": sorted(RIDER_VEHICLE_CLASSES),
            "riderCandidateFilters": {
                "minPersonConfidence": RIDER_MIN_PERSON_CONFIDENCE,
                "minVehicleConfidence": RIDER_MIN_VEHICLE_CONFIDENCE,
                "minPersonHeightRatio": RIDER_MIN_PERSON_HEIGHT_RATIO,
                "minVehicleHeightRatio": RIDER_MIN_VEHICLE_HEIGHT_RATIO,
            },
            "ppeDetections": [
                {
                    "class": p["class"],
                    "confidence": round(float(p.get("confidence", 0.0)), 4),
                    "bbox": _round_bbox(p["bbox"]),
                }
                for p in ppe_dets
            ],
            "riderAssociations": [
                {
                    "person": {
                        "confidence": round(float(association["person"].get("confidence", 0.0)), 4),
                        "bbox": _round_bbox(association["person"]["bbox"]),
                    },
                    "vehicle": {
                        "class": association["vehicle"].get("class"),
                        "confidence": round(float(association["vehicle"].get("confidence", 0.0)), 4),
                        "bbox": _round_bbox(association["vehicle"]["bbox"]),
                    },
                    **association["detail"],
                }
                for association in rider_associations
            ],
            "personDiagnostics": person_diagnostics,
        },
    }]


def _ppe_person_diagnostics(
    all_persons: list,
    ppe_dets: list,
    frame_w: int | None = None,
    frame_h: int | None = None,
    scope_zones: list[dict] | None = None,
) -> list[dict]:
    diagnostics = []
    scope_zones = scope_zones or []
    for index, person in enumerate(all_persons):
        person_bbox = person["bbox"]
        geometry_evaluable = _person_evaluable_for_ppe(person_bbox, frame_w, frame_h)
        in_scope = _person_in_ppe_scope(person_bbox, scope_zones, frame_w, frame_h)
        ppe_matches = []
        for ppe in ppe_dets:
            detail = _ppe_match_detail(person_bbox, ppe["bbox"])
            ppe_matches.append({
                "class": ppe["class"],
                "confidence": round(float(ppe.get("confidence", 0.0)), 4),
                "bbox": _round_bbox(ppe["bbox"]),
                **detail,
            })
        best_match = max(
            ppe_matches,
            key=lambda match: (
                1 if match["matched"] else 0,
                match["overlapReferenceRatio"],
                1 if match["centerInsideExpandedPerson"] else 0,
            ),
            default=None,
        )
        diagnostics.append({
            "index": index,
            "class": person["class"],
            "confidence": round(float(person.get("confidence", 0.0)), 4),
            "bbox": _round_bbox(person_bbox),
            "evaluable": geometry_evaluable and in_scope,
            "geometryEvaluable": geometry_evaluable,
            "inPpeEvaluationScope": in_scope,
            "covered": bool(geometry_evaluable and in_scope and best_match and best_match["matched"]),
            "bestPpeMatch": best_match,
        })
    return diagnostics


def check_yoloe_violations(detections: list, camera_id: str, frame_w: int | None = None, frame_h: int | None = None) -> list:
    """Return candidate violation dicts (NOT yet persisted to DB).
    Per-person check: only flags persons whose bbox does not contain any matching PPE item."""
    candidates = []
    cfg = get_config()
    cam = cfg["cameras"].get(camera_id, {})
    all_persons = [d for d in detections if d["class"] == "person"]
    scope_zones = _ppe_scope_zones(cam)
    persons = [
        p for p in all_persons
        if _person_evaluable_for_ppe(p["bbox"], frame_w, frame_h)
        and _person_in_ppe_scope(p["bbox"], scope_zones, frame_w, frame_h)
    ]
    if not persons:
        return candidates

    # Get PPE groups and severity map
    safety_rule_ids = cam.get("safety_rule_ids", cam.get("ppe_rule_ids", []))
    rule_id_map = {}
    threshold_source_map = {}
    if safety_rule_ids:
        # Camera has assigned safety rules — only check PPE ones
        all_rules = cfg.get("safety_rules", [])
        rule_map = {r["id"]: r for r in all_rules}
        ppe_groups = {}
        severity_map = {}
        threshold_map = {}
        for rid in safety_rule_ids:
            rule = rule_map.get(rid)
            if rule and rule.get("id") == RIDER_HELMET_RULE_ID:
                continue
            if rule and rule.get("type") == "ppe" and rule.get("enabled", True):
                key = rule["name"].lower()
                ppe_groups[key] = rule["classes"]
                severity_map[key] = rule.get("severity", "P2")
                rule_id_map[key] = rule.get("id")
                rule_threshold = _positive_int(rule.get("threshold"))
                override_threshold = _camera_ppe_threshold_override(cam, rule.get("id"), key)
                threshold_map[key] = override_threshold if override_threshold is not None else rule_threshold
                threshold_source_map[key] = "camera_override" if override_threshold is not None else (
                    "safety_rule" if rule_threshold is not None else "default"
                )
        checked_groups: set[str] = set(ppe_groups)
    else:
        # Fallback: match yoloe_classes against all known PPE groups
        ppe_rules = [
            r for r in cfg.get("safety_rules", [])
            if r.get("type") == "ppe" and r.get("enabled", True) and r.get("id") != RIDER_HELMET_RULE_ID
        ]
        ppe_groups = {r["name"].lower(): r["classes"] for r in ppe_rules}
        severity_map = {r["name"].lower(): r.get("severity", "P2") for r in ppe_rules}
        threshold_map = {}
        for rule in ppe_rules:
            key = rule["name"].lower()
            rule_id_map[key] = rule.get("id")
            rule_threshold = _positive_int(rule.get("threshold"))
            override_threshold = _camera_ppe_threshold_override(cam, rule.get("id"), key)
            threshold_map[key] = override_threshold if override_threshold is not None else rule_threshold
            threshold_source_map[key] = "camera_override" if override_threshold is not None else (
                "safety_rule" if rule_threshold is not None else "default"
            )
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
        violating_persons = [p for p in persons if not _ppe_present_for_person(p["bbox"], ppe_dets)]

        if violating_persons:
            person_diagnostics = _ppe_person_diagnostics(all_persons, ppe_dets, frame_w, frame_h, scope_zones)
            matching_sources = {d.get("model_family") for d in ppe_dets if d.get("model_family")}
            if not matching_sources:
                person_sources = {p.get("model_family") for p in violating_persons if p.get("model_family")}
                matching_sources = (
                    person_sources
                    if person_sources == {"ppe_closed_set_candidate"}
                    else {"ppe_specialist"}
                )
            candidates.append({
                "camera_id": camera_id,
                "rule": f"Missing {group_name}",
                "severity": severity_map.get(group_name, "P2"),
                "confidence": max(p["confidence"] for p in violating_persons),
                "description": f"{len(violating_persons)} worker(s) detected without {group_name}",
                "source": _source_label_for_model_families(matching_sources),
                "threshold": threshold_map.get(group_name),
                "metadata": {
                    "ppeGroup": group_name,
                    "safetyRuleId": rule_id_map.get(group_name),
                    "ruleThreshold": threshold_map.get(group_name),
                    "thresholdSource": threshold_source_map.get(group_name, "default"),
                    "personCount": len(all_persons),
                    "evaluablePersonCount": len(persons),
                    "ignoredPersonCount": len(all_persons) - len(persons),
                    "ppeDetectionCount": len(ppe_dets),
                    "coveredPersonCount": len(persons) - len(violating_persons),
                    "violatingPersonCount": len(violating_persons),
                    "ppeEvaluationZoneIds": [zone.get("id") or zone.get("name") for zone in scope_zones],
                    "frameWidth": frame_w,
                    "frameHeight": frame_h,
                    "ppeDetections": [
                        {
                            "class": p["class"],
                            "confidence": round(float(p.get("confidence", 0.0)), 4),
                            "bbox": _round_bbox(p["bbox"]),
                        }
                        for p in ppe_dets
                    ],
                    "personDiagnostics": person_diagnostics,
                },
            })

    rider_helmet_rule = _rider_helmet_rule_for_camera(cam, cfg)
    if rider_helmet_rule:
        candidates.extend(_check_rider_helmet_violations(
            detections,
            camera_id,
            cam,
            rider_helmet_rule,
            persons,
            all_persons,
            scope_zones,
            frame_w,
            frame_h,
        ))

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

    if rule == "Missing rider helmet":
        cfg = get_config()
        cam = cfg["cameras"].get(camera_id, {}) if camera_id else {}
        scope_zones = _ppe_scope_zones(cam)
        persons = [
            d for d in detections
            if d.get("class") == "person"
            and d.get("bbox")
            and _person_evaluable_for_ppe(d["bbox"], frame_w, frame_h)
            and _rider_person_evaluable(d, frame_h)
            and _person_in_ppe_scope(d["bbox"], scope_zones, frame_w, frame_h)
        ]
        vehicles = [
            d for d in detections
            if d.get("bbox")
            and _detection_class(d.get("class")) in RIDER_VEHICLE_CLASSES
            and _rider_vehicle_evaluable(d, frame_h)
        ]
        rule_map = {r["id"]: r for r in cfg.get("safety_rules", [])}
        rider_rule = rule_map.get(RIDER_HELMET_RULE_ID) or {}
        ppe_classes = rider_rule.get("classes") or ["motorcycle helmet", "rider helmet", "helmet"]
        ppe_dets = [d for d in detections if d.get("bbox") and d.get("class") in ppe_classes]
        for association in _rider_vehicle_associations(persons, vehicles):
            person = association["person"]
            if _ppe_present_for_person(person["bbox"], ppe_dets):
                continue
            results.append({
                "label": "Missing Rider Helmet",
                "bbox": normalize(person["bbox"]),
                "confidence": round(person["confidence"], 2),
            })
        return results

    if rule.startswith("Missing "):
        ppe_item = rule.replace("Missing ", "")
        ppe_classes = get_ppe_groups().get(ppe_item, [ppe_item])
        ppe_dets = [d for d in detections if d["class"] in ppe_classes]
        cfg = get_config()
        cam = cfg["cameras"].get(camera_id, {}) if camera_id else {}
        scope_zones = _ppe_scope_zones(cam)
        for d in detections:
            if d["class"] == "person":
                if not _person_evaluable_for_ppe(d["bbox"], frame_w, frame_h):
                    continue
                if not _person_in_ppe_scope(d["bbox"], scope_zones, frame_w, frame_h):
                    continue
                has_ppe = _ppe_present_for_person(d["bbox"], ppe_dets)
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

# A hand-held phone often sits just outside the detector's torso-aligned person
# box. Jetson evaluation found 0.25 to be the smallest padding that recovered
# that pose without admitting any of the labelled desk/phone negatives.
_MOBILE_PHONE_PERSON_X_PADDING = 0.25
_MOBILE_PHONE_PERSON_Y_PADDING = 0.10
_MOBILE_PHONE_MAX_PERSON_AREA_RATIO = 0.08
_MOBILE_PHONE_MAX_PERSON_RELATIVE_Y = 0.85


def _bbox_coordinates(detection: dict) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _mobile_phone_matches_person(phone: dict, person: dict) -> bool:
    """Return whether a phone-sized box is plausibly being used by a person."""
    phone_bbox = _bbox_coordinates(phone)
    person_bbox = _bbox_coordinates(person)
    if phone_bbox is None or person_bbox is None:
        return False

    phone_x1, phone_y1, phone_x2, phone_y2 = phone_bbox
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1
    phone_center_x = (phone_x1 + phone_x2) / 2.0
    phone_center_y = (phone_y1 + phone_y2) / 2.0
    if not (
        person_x1 - person_width * _MOBILE_PHONE_PERSON_X_PADDING
        <= phone_center_x
        <= person_x2 + person_width * _MOBILE_PHONE_PERSON_X_PADDING
        and person_y1 - person_height * _MOBILE_PHONE_PERSON_Y_PADDING
        <= phone_center_y
        <= person_y2 + person_height * _MOBILE_PHONE_PERSON_Y_PADDING
    ):
        return False

    relative_y = (phone_center_y - person_y1) / person_height
    if not (
        -_MOBILE_PHONE_PERSON_Y_PADDING
        <= relative_y
        <= _MOBILE_PHONE_MAX_PERSON_RELATIVE_Y
    ):
        return False

    phone_area = (phone_x2 - phone_x1) * (phone_y2 - phone_y1)
    person_area = person_width * person_height
    return phone_area / person_area <= _MOBILE_PHONE_MAX_PERSON_AREA_RATIO


def _alert_rule_confidence(cfg: dict, cam: dict, rule: dict) -> float:
    default_confidence = cfg.get("global", {}).get("yolo_conf", 0.35)
    overrides = cam.get("safety_rule_overrides") or {}
    override = overrides.get(rule.get("id")) if isinstance(overrides, dict) else None
    values = []
    if isinstance(override, dict):
        values.extend(override.get(key) for key in ("confidence", "conf"))
    values.append(rule.get("confidence"))
    values.append(default_confidence)
    for value in values:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if 0 < confidence <= 1:
            return confidence
    return 0.35


def check_violations(
    detections: list,
    camera_id: str,
    *,
    capability_filter: set[str] | None = None,
) -> list:
    """Return COCO candidates, optionally limited to freshly evaluated capabilities."""
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
        if capability_filter is not None and RULE_ID_TO_CAPABILITY.get(rid) not in capability_filter:
            continue
        rule = all_rules.get(rid)
        if not rule or not rule.get("enabled", True):
            continue
        if rule["type"] != "alert":
            continue
        # Skip zone_intrusion — handled separately
        if rid == "alert_zone_intrusion":
            continue

        confidence_threshold = _alert_rule_confidence(cfg, cam, rule)
        matching = [
            detection
            for detection in detections
            if detection["class"] in rule["classes"]
            and float(detection.get("confidence") or 0) >= confidence_threshold
        ]
        if rule["classes"] == ["cell phone"]:
            matching = [
                phone
                for phone in matching
                if any(
                    _mobile_phone_matches_person(phone, person)
                    for person in persons
                )
            ]
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
        source_label = _source_label_for_model_families(matching_sources)

        candidates.append({
            "camera_id": camera_id,
            "rule": rule["name"],
            "severity": rule["severity"],
            "confidence": max(d["confidence"] for d in matching),
            "count": len(matching),
            "description": desc,
            "source": source_label,
            "threshold": rule.get("threshold"),
        })

    return candidates
