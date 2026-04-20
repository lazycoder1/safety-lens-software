"""Capability registry for SafetyLens camera planning."""

from __future__ import annotations

from typing import Literal, TypedDict

ModelKey = Literal["coco_primary", "ppe_specialist", "yoloe_long_tail"]
CameraProfile = Literal["general_safety", "work_zone_ppe", "demo_advanced"]
CapabilityKey = Literal[
    "person_presence",
    "vehicle_presence",
    "animal_presence",
    "mobile_phone",
    "zone_intrusion",
    "helmet_required",
    "vest_required",
    "gloves_required",
    "hairnet_required",
    "face_mask_required",
    "apron_required",
    "boots_required",
    "goggles_required",
    "fire_smoke",
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
        "prompt_terms": ["hard hat", "safety helmet"],
        "safety_rule_ids": ["ppe_helmet"],
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
        "prompt_terms": ["face mask"],
        "safety_rule_ids": ["ppe_facemask"],
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
        "prompt_terms": ["apron"],
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
        "prompt_terms": ["safety boots", "steel-toe boots"],
        "safety_rule_ids": ["ppe_boots"],
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
        "model_family": "yoloe_long_tail",
        "input_scope": "full_frame",
        "requires_tracking": False,
        "requires_zones": False,
        "requires_association": False,
        "prompt_terms": ["fire", "smoke", "flames"],
        "safety_rule_ids": ["alert_fire_smoke"],
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
    "demo_advanced": ["fire_smoke"],
}

RULE_ID_TO_CAPABILITY: dict[str, CapabilityKey] = {}
CLASS_TERM_TO_CAPABILITY: dict[str, CapabilityKey] = {}
ALL_PPE_PROMPT_TERMS: list[str] = []

for key, definition in CAPABILITY_REGISTRY.items():
    for rule_id in definition.get("safety_rule_ids", []):
        RULE_ID_TO_CAPABILITY[rule_id] = key
    for prompt in definition.get("prompt_terms", []):
        CLASS_TERM_TO_CAPABILITY[prompt.lower().replace("_", " ").replace("-", " ").strip()] = key
    if definition["model_family"] == "ppe_specialist":
        for prompt in definition.get("prompt_terms", []):
            if prompt not in ALL_PPE_PROMPT_TERMS:
                ALL_PPE_PROMPT_TERMS.append(prompt)


def normalize_capability_key(value: str) -> CapabilityKey | None:
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
    if has_long_tail:
        return "demo_advanced"
    if has_ppe:
        return "work_zone_ppe"
    return "general_safety"
