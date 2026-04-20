"""Helpers for normalizing camera config across legacy and current fields."""

from __future__ import annotations

from copy import deepcopy

LEGACY_ALERT_MAP = {
    "mobile_phone": "alert_mobile_phone",
    "animal_intrusion": "alert_animal",
    "person_detected": "alert_person",
    "vehicle_detected": "alert_vehicle",
    "zone_intrusion": "alert_zone_intrusion",
}

RULE_ID_TO_LEGACY_ALERT = {value: key for key, value in LEGACY_ALERT_MAP.items()}

RULE_ID_TO_CLASSES = {
    "ppe_helmet": ["hard hat", "safety helmet"],
    "ppe_vest": ["safety vest", "high visibility vest", "fluorescent vest"],
    "ppe_gloves": ["gloves"],
    "ppe_hairnet": ["hairnet"],
    "ppe_facemask": ["face mask"],
    "ppe_apron": ["apron"],
    "ppe_boots": ["safety boots", "steel-toe boots"],
    "ppe_goggles": ["safety goggles", "protective eyewear"],
    "alert_mobile_phone": ["cell phone"],
    "alert_animal": ["dog", "cat", "deer", "animal"],
    "alert_person": ["person"],
    "alert_vehicle": ["truck", "car", "motorcycle"],
    "alert_zone_intrusion": ["person"],
}

RULE_MATCH_TERMS = {
    "ppe_helmet": ["helmet", "hard hat", "safety helmet"],
    "ppe_vest": ["vest", "safety vest", "high visibility vest", "fluorescent vest"],
    "ppe_gloves": ["gloves"],
    "ppe_hairnet": ["hairnet"],
    "ppe_facemask": ["face mask", "mask"],
    "ppe_apron": ["apron"],
    "ppe_boots": ["safety boots", "steel-toe boots", "boots"],
    "ppe_goggles": ["safety goggles", "protective eyewear", "goggles"],
    "alert_mobile_phone": ["mobile phone", "cell phone", "phone"],
    "alert_animal": ["animal intrusion", "animal", "dog", "cat", "deer"],
    "alert_person": ["person detected", "person"],
    "alert_vehicle": ["vehicle detected", "vehicle", "car", "truck", "motorcycle", "forklift"],
    "alert_zone_intrusion": ["zone intrusion", "restricted zone", "intrusion"],
}

PPE_RULE_IDS = {rule_id for rule_id in RULE_ID_TO_CLASSES if rule_id.startswith("ppe_")}
CLASS_TO_RULE_ID = {
    normalized: rule_id
    for rule_id, classes in RULE_ID_TO_CLASSES.items()
    for normalized in [cls.lower().replace("_", " ").replace("-", " ").strip() for cls in classes]
    if normalized != "person"
}


def _ensure_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _normalize_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def _append_unique(items: list[str], value: str):
    if value and value not in items:
        items.append(value)


def infer_rule_ids_from_camera(camera: dict) -> list[str]:
    rule_ids: list[str] = []

    for rule_id in _ensure_list(camera.get("safety_rule_ids")):
        _append_unique(rule_ids, rule_id)

    for rule_id in _ensure_list(camera.get("ppe_rule_ids")):
        _append_unique(rule_ids, rule_id)

    for alert_key in _ensure_list(camera.get("alert_classes")):
        mapped = LEGACY_ALERT_MAP.get(alert_key, alert_key)
        _append_unique(rule_ids, mapped)

    for value in _ensure_list(camera.get("yoloe_classes")):
        mapped = CLASS_TO_RULE_ID.get(_normalize_text(value))
        if mapped:
            _append_unique(rule_ids, mapped)

    if not rule_ids:
        for value in _ensure_list(camera.get("rules")):
            normalized = _normalize_text(value)
            for rule_id, terms in RULE_MATCH_TERMS.items():
                if any(term in normalized for term in terms):
                    _append_unique(rule_ids, rule_id)

    return rule_ids


def sync_camera_rule_fields(camera: dict) -> bool:
    changed = False
    rule_ids = infer_rule_ids_from_camera(camera)

    if camera.get("safety_rule_ids") != rule_ids:
        camera["safety_rule_ids"] = rule_ids
        changed = True

    derived_alert_classes = [RULE_ID_TO_LEGACY_ALERT[rule_id] for rule_id in rule_ids if rule_id in RULE_ID_TO_LEGACY_ALERT]
    alert_classes = []
    for value in [*_ensure_list(camera.get("alert_classes")), *derived_alert_classes]:
        if value in LEGACY_ALERT_MAP or value in RULE_ID_TO_LEGACY_ALERT.values():
            _append_unique(alert_classes, value)
    if camera.get("alert_classes") != alert_classes:
        camera["alert_classes"] = alert_classes
        changed = True

    ppe_rule_ids = [rule_id for rule_id in rule_ids if rule_id in PPE_RULE_IDS]
    if camera.get("ppe_rule_ids") != ppe_rule_ids:
        camera["ppe_rule_ids"] = ppe_rule_ids
        changed = True

    existing_yoloe_classes = _ensure_list(camera.get("yoloe_classes"))
    derived_yoloe_classes = ["person"]
    for rule_id in rule_ids:
        for cls in RULE_ID_TO_CLASSES.get(rule_id, []):
            _append_unique(derived_yoloe_classes, cls)

    unknown_existing = [
        cls
        for cls in existing_yoloe_classes
        if _normalize_text(cls) not in CLASS_TO_RULE_ID and _normalize_text(cls) != "person"
    ]
    for cls in unknown_existing:
        _append_unique(derived_yoloe_classes, cls)

    next_yoloe_classes = derived_yoloe_classes if existing_yoloe_classes or ppe_rule_ids else existing_yoloe_classes
    if camera.get("yoloe_classes") != next_yoloe_classes:
        camera["yoloe_classes"] = next_yoloe_classes
        changed = True

    return changed


def normalize_config(config: dict) -> tuple[dict, bool]:
    normalized = deepcopy(config)
    changed = False

    cameras = normalized.get("cameras", {})
    for camera in cameras.values():
        if sync_camera_rule_fields(camera):
            changed = True
        from camera_planner import normalize_camera_record

        updated_camera, camera_changed = normalize_camera_record(camera, normalized)
        if camera_changed:
            camera.clear()
            camera.update(updated_camera)
            changed = True

    return normalized, changed
