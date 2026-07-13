"""Capability registry for Rakshak Lens camera planning."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

ModelKey = Literal[
    "coco_primary",
    "ppe_specialist",
    "ppe_closed_set_candidate",
    "yoloe_long_tail",
    "fire_smoke_specialist",
    "face_recognition",
    "pose_specialist",
    "plate_recognition",
]
CameraProfile = Literal["general_safety", "work_zone_ppe", "office_occupancy", "demo_advanced"]
CapabilityKey = Literal[
    "person_presence",
    "office_occupancy",
    "crowd_count_threshold",
    "queue_monitoring",
    "route_obstruction",
    "object_lifecycle",
    "vehicle_presence",
    "animal_presence",
    "mobile_phone",
    "zone_intrusion",
    "helmet_required",
    "helmet_color_compliance",
    "rider_helmet_required",
    "vest_required",
    "gloves_required",
    "hairnet_required",
    "face_mask_required",
    "face_shield_required",
    "apron_required",
    "boots_required",
    "harness_required",
    "goggles_required",
    "fire_smoke",
    "face_recognition",
    "plate_recognition",
    "fall_detection",
    "custom_long_tail",
]
InputScope = Literal["full_frame", "object_crop"]


class CapabilityDefinition(TypedDict, total=False):
    key: CapabilityKey
    label: str
    group: str
    model_family: ModelKey
    input_scope: InputScope
    requires_tracking: bool
    requires_zones: bool
    requires_association: bool
    prompt_terms: list[str]
    safety_rule_ids: list[str]


# Keep worker and rider helmet capabilities on the same fixed-prompt TensorRT
# profile.  The generic ``helmet`` prompt detects industrial hard hats in the
# validated TMEIC corpus, while the synonyms retain rider-camera coverage.
HELMET_DETECTOR_PROMPT_TERMS = [
    "motorcycle helmet",
    "rider helmet",
    "helmet",
]


CAPABILITY_REGISTRY: dict[CapabilityKey, CapabilityDefinition] = {
    "person_presence": {
        "key": "person_presence",
        "label": "People",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["person"],
        "safety_rule_ids": ["alert_person"],
    },
    "office_occupancy": {
        "key": "office_occupancy",
        "label": "Office Occupancy",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": True,
        "requires_association": False,
        "prompt_terms": ["person", "chair", "seat", "desk", "workstation"],
        "safety_rule_ids": [],
    },
    "crowd_count_threshold": {
        "key": "crowd_count_threshold",
        "label": "Crowd Count",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["person", "people count", "crowd"],
        "safety_rule_ids": [],
    },
    "queue_monitoring": {
        "key": "queue_monitoring",
        "label": "Queue Monitoring",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": True,
        "requires_association": False,
        "prompt_terms": ["person", "queue", "line"],
        "safety_rule_ids": [],
    },
    "route_obstruction": {
        "key": "route_obstruction",
        "label": "Route Obstruction",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": True,
        "requires_association": False,
        "prompt_terms": ["person", "car", "truck", "bus", "motorcycle", "bicycle", "obstruction"],
        "safety_rule_ids": [],
    },
    "object_lifecycle": {
        "key": "object_lifecycle",
        "label": "Object Lifecycle",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": True,
        "requires_association": False,
        "prompt_terms": ["backpack", "handbag", "suitcase", "watched object", "object removed"],
        "safety_rule_ids": [],
    },
    "vehicle_presence": {
        "key": "vehicle_presence",
        "label": "Vehicles",
        "group": "Core Monitoring",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["car", "truck", "motorcycle", "vehicle"],
        "safety_rule_ids": ["alert_vehicle"],
    },
    "animal_presence": {
        "key": "animal_presence",
        "label": "Animals",
        "group": "Safety Events",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["dog", "cat", "deer", "animal"],
        "safety_rule_ids": ["alert_animal"],
    },
    "mobile_phone": {
        "key": "mobile_phone",
        "label": "Mobile Phone",
        "group": "Safety Events",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["cell phone", "mobile phone", "phone"],
        "safety_rule_ids": ["alert_mobile_phone"],
    },
    "zone_intrusion": {
        "key": "zone_intrusion",
        "label": "Zone Intrusion",
        "group": "Safety Events",
        "model_family": "coco_primary",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": True,
        "requires_association": False,
        "prompt_terms": ["person", "zone intrusion"],
        "safety_rule_ids": ["alert_zone_intrusion"],
    },
    "helmet_required": {
        "key": "helmet_required",
        "label": "Helmet",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": list(HELMET_DETECTOR_PROMPT_TERMS),
        "safety_rule_ids": ["ppe_helmet"],
    },
    "helmet_color_compliance": {
        "key": "helmet_color_compliance",
        "label": "Helmet Colour",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": list(HELMET_DETECTOR_PROMPT_TERMS),
        "safety_rule_ids": [],
    },
    "rider_helmet_required": {
        "key": "rider_helmet_required",
        "label": "Rider Helmet",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": list(HELMET_DETECTOR_PROMPT_TERMS),
        "safety_rule_ids": ["ppe_rider_helmet"],
    },
    "vest_required": {
        "key": "vest_required",
        "label": "Safety Vest",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["safety vest", "high visibility vest", "fluorescent vest"],
        "safety_rule_ids": ["ppe_vest"],
    },
    "gloves_required": {
        "key": "gloves_required",
        "label": "Gloves",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["gloves"],
        "safety_rule_ids": ["ppe_gloves"],
    },
    "hairnet_required": {
        "key": "hairnet_required",
        "label": "Hairnet",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["hairnet"],
        "safety_rule_ids": ["ppe_hairnet"],
    },
    "face_mask_required": {
        "key": "face_mask_required",
        "label": "Face Mask",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["face mask", "surgical mask", "medical mask", "mask", "respirator"],
        "safety_rule_ids": ["ppe_face_mask"],
    },
    "face_shield_required": {
        "key": "face_shield_required",
        "label": "Face Shield",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["face shield", "protective face shield", "clear face shield", "visor", "protective visor"],
        "safety_rule_ids": ["ppe_face_shield"],
    },
    "apron_required": {
        "key": "apron_required",
        "label": "Apron",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["apron", "protective apron", "kitchen apron", "denim apron", "work apron"],
        "safety_rule_ids": ["ppe_apron"],
    },
    "boots_required": {
        "key": "boots_required",
        "label": "Safety Boots",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["safety boots", "steel-toe boots", "work boots", "protective boots", "rubber boots"],
        "safety_rule_ids": ["ppe_boots"],
    },
    "harness_required": {
        "key": "harness_required",
        "label": "Safety Harness",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["safety harness", "fall arrest harness", "body harness", "harness", "safety lanyard", "fall protection lanyard"],
        "safety_rule_ids": ["ppe_harness"],
    },
    "goggles_required": {
        "key": "goggles_required",
        "label": "Safety Goggles",
        "group": "PPE",
        "model_family": "ppe_specialist",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": True,
        "prompt_terms": ["safety goggles", "protective eyewear"],
        "safety_rule_ids": ["ppe_goggles"],
    },
    "fire_smoke": {
        "key": "fire_smoke",
        "label": "Fire / Smoke",
        "group": "Safety Events",
        "model_family": "fire_smoke_specialist",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["fire", "smoke"],
        "safety_rule_ids": ["alert_fire_smoke"],
    },
    "face_recognition": {
        "key": "face_recognition",
        "label": "Face Recognition",
        "group": "Advanced",
        "model_family": "face_recognition",
        "input_scope": "full_frame",
        "requires_tracking": True,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["face recognition", "face"],
        "safety_rule_ids": [],
    },
    "plate_recognition": {
        "key": "plate_recognition",
        "label": "Plate Recognition",
        "group": "Advanced",
        "model_family": "plate_recognition",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["plate recognition", "license plate", "number plate", "anpr"],
        "safety_rule_ids": [],
    },
    "fall_detection": {
        "key": "fall_detection",
        "label": "Fall / Man Down",
        "group": "Safety Events",
        "model_family": "pose_specialist",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["person_fall"],
        "safety_rule_ids": ["alert_fall_detection"],
    },
    "custom_long_tail": {
        "key": "custom_long_tail",
        "label": "Custom Long-Tail",
        "group": "Advanced",
        "model_family": "yoloe_long_tail",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": [],
        "safety_rule_ids": [],
    },
}

PROFILE_DEFAULT_CAPABILITIES: dict[CameraProfile, list[CapabilityKey]] = {
    "general_safety": ["person_presence", "mobile_phone"],
    "work_zone_ppe": ["helmet_required", "vest_required", "zone_intrusion"],
    "office_occupancy": ["office_occupancy", "mobile_phone"],
    "demo_advanced": ["fire_smoke", "fall_detection"],
}

RULE_ID_TO_CAPABILITY: dict[str, CapabilityKey] = {}
CLASS_TERM_TO_CAPABILITY: dict[str, CapabilityKey] = {}
CLASS_TERM_TO_CAPABILITIES: dict[str, list[CapabilityKey]] = {}
ALL_PPE_PROMPT_TERMS: list[str] = []

for key, definition in CAPABILITY_REGISTRY.items():
    for rule_id in definition.get("safety_rule_ids", []):
        RULE_ID_TO_CAPABILITY[rule_id] = key
    for prompt in definition.get("prompt_terms", []):
        normalized_prompt = prompt.lower().replace("_", " ").replace("-", " ").strip()
        CLASS_TERM_TO_CAPABILITY.setdefault(normalized_prompt, key)
        prompt_capabilities = CLASS_TERM_TO_CAPABILITIES.setdefault(normalized_prompt, [])
        if key not in prompt_capabilities:
            prompt_capabilities.append(key)
    if definition["model_family"] == "ppe_specialist":
        for prompt in definition.get("prompt_terms", []):
            if prompt not in ALL_PPE_PROMPT_TERMS:
                ALL_PPE_PROMPT_TERMS.append(prompt)

RULE_ID_TO_CAPABILITY["ppe_facemask"] = "face_mask_required"


def normalize_capability_key(value: str) -> Optional[CapabilityKey]:
    normalized = value.strip()
    return normalized if normalized in CAPABILITY_REGISTRY else None


def capability_label(key: CapabilityKey) -> str:
    return CAPABILITY_REGISTRY[key]["label"]


def default_rule_ids_for_capabilities(capabilities: list[CapabilityKey]) -> list[str]:
    rule_ids: list[str] = []
    for key in capabilities:
        for rule_id in CAPABILITY_REGISTRY[key].get("safety_rule_ids", []):
            if rule_id not in rule_ids:
                rule_ids.append(rule_id)
    return rule_ids


def infer_profile_from_capabilities(capabilities: list[CapabilityKey]) -> CameraProfile:
    if not capabilities:
        return "general_safety"
    capability_set = set(capabilities)
    has_long_tail = bool({"fire_smoke", "custom_long_tail"} & capability_set)
    has_ppe = any(CAPABILITY_REGISTRY[key]["model_family"] == "ppe_specialist" for key in capability_set)
    if "office_occupancy" in capability_set:
        return "office_occupancy"
    if has_long_tail:
        return "demo_advanced"
    if has_ppe:
        return "work_zone_ppe"
    return "general_safety"
