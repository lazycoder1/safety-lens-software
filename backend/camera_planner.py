"""Camera planning and execution-plan compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from camera_connection import normalize_camera_connection
from capability_registry import (
    ALL_PPE_PROMPT_TERMS,
    CAPABILITY_REGISTRY,
    CLASS_TERM_TO_CAPABILITY,
    PROFILE_DEFAULT_CAPABILITIES,
    RULE_ID_TO_CAPABILITY,
    CameraProfile,
    CapabilityKey,
    ModelKey,
    default_rule_ids_for_capabilities,
    infer_profile_from_capabilities,
    normalize_capability_key,
)


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


def _ordered_capabilities(capabilities: list[str]) -> list[CapabilityKey]:
    ordered: list[CapabilityKey] = []
    for key in CAPABILITY_REGISTRY:
        if key in capabilities:
            ordered.append(key)
    return ordered


def infer_capabilities_from_camera(camera: dict, cfg: dict) -> list[CapabilityKey]:
    rule_ids = _ensure_list(camera.get("safety_rule_ids"))
    if not rule_ids:
        explicit = [normalize_capability_key(value) for value in _ensure_list(camera.get("capabilities"))]
        explicit_caps = [value for value in explicit if value]
        if explicit_caps:
            return _ordered_capabilities(explicit_caps)

    capabilities: list[CapabilityKey] = []
    rule_map = {rule["id"]: rule for rule in cfg.get("safety_rules", [])}

    for rule_id in rule_ids:
        mapped = RULE_ID_TO_CAPABILITY.get(rule_id)
        if mapped:
            _append_unique(capabilities, mapped)
            continue
        rule = rule_map.get(rule_id)
        if rule and rule.get("model") == "yoloe":
            _append_unique(capabilities, "custom_long_tail")

    for class_name in _ensure_list(camera.get("yoloe_classes")):
        normalized = _normalize_text(class_name)
        if normalized == "person":
            continue
        mapped = CLASS_TERM_TO_CAPABILITY.get(normalized)
        if mapped:
            _append_unique(capabilities, mapped)
        else:
            _append_unique(capabilities, "custom_long_tail")

    if not rule_ids:
        for rule_name in _ensure_list(camera.get("rules")):
            normalized = _normalize_text(rule_name)
            if "zone intrusion" in normalized:
                _append_unique(capabilities, "zone_intrusion")
            if "person" in normalized:
                _append_unique(capabilities, "person_presence")
            if "vehicle" in normalized or "forklift" in normalized:
                _append_unique(capabilities, "vehicle_presence")
            if "animal" in normalized:
                _append_unique(capabilities, "animal_presence")
            if "phone" in normalized:
                _append_unique(capabilities, "mobile_phone")
            if "fire" in normalized or "smoke" in normalized:
                _append_unique(capabilities, "fire_smoke")

    return _ordered_capabilities(capabilities)


def derive_custom_long_tail_terms(camera: dict, cfg: dict) -> list[str]:
    terms: list[str] = []
    explicit_terms = _ensure_list(camera.get("custom_long_tail_terms"))
    for term in explicit_terms:
        _append_unique(terms, term)

    rule_ids = _ensure_list(camera.get("safety_rule_ids"))
    rule_map = {rule["id"]: rule for rule in cfg.get("safety_rules", [])}
    for rule_id in rule_ids:
        rule = rule_map.get(rule_id)
        if not rule or rule.get("id") in RULE_ID_TO_CAPABILITY:
            continue
        if rule.get("model") == "yoloe":
            for class_name in _ensure_list(rule.get("classes")):
                _append_unique(terms, class_name)

    for class_name in _ensure_list(camera.get("yoloe_classes")):
        normalized = _normalize_text(class_name)
        if normalized == "person":
            continue
        if normalized not in CLASS_TERM_TO_CAPABILITY:
            _append_unique(terms, class_name)

    return terms


def _resolve_profile(camera: dict, capabilities: list[CapabilityKey]) -> CameraProfile:
    raw_profile = camera.get("profile")
    if raw_profile in PROFILE_DEFAULT_CAPABILITIES:
        return raw_profile
    return infer_profile_from_capabilities(capabilities)


def _required_model_keys(capabilities: list[CapabilityKey]) -> list[ModelKey]:
    required: list[ModelKey] = []
    capability_set = set(capabilities)
    has_ppe = any(CAPABILITY_REGISTRY[key]["model_family"] == "ppe_specialist" for key in capability_set)
    has_coco = any(CAPABILITY_REGISTRY[key]["model_family"] == "coco_primary" for key in capability_set)
    has_long_tail = any(CAPABILITY_REGISTRY[key]["model_family"] == "yoloe_long_tail" for key in capability_set)

    if has_coco or has_ppe:
        required.append("coco_primary")
    if has_ppe:
        required.append("ppe_specialist")
    if has_long_tail:
        required.append("yoloe_long_tail")
    return required


def build_execution_plan(camera: dict, cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        from config_manager import get_config

        cfg = get_config()

    capabilities = infer_capabilities_from_camera(camera, cfg)
    profile = _resolve_profile(camera, capabilities)
    required_model_keys = _required_model_keys(capabilities)
    capability_set = set(capabilities)
    zones_required = any(CAPABILITY_REGISTRY[key]["requires_zones"] for key in capability_set)
    association_enabled = any(CAPABILITY_REGISTRY[key]["requires_association"] for key in capability_set)
    tracking_enabled = association_enabled or any(
        CAPABILITY_REGISTRY[key]["requires_tracking"] for key in capability_set
    )

    ppe_prompt_terms: list[str] = []
    yoloe_prompt_terms: list[str] = []
    for key in capabilities:
        definition = CAPABILITY_REGISTRY[key]
        if definition["model_family"] == "ppe_specialist":
            for prompt in definition.get("prompt_terms", []):
                _append_unique(ppe_prompt_terms, prompt)
        elif definition["model_family"] == "yoloe_long_tail":
            for prompt in definition.get("prompt_terms", []):
                _append_unique(yoloe_prompt_terms, prompt)

    for term in derive_custom_long_tail_terms(camera, cfg):
        _append_unique(yoloe_prompt_terms, term)

    if "ppe_specialist" in required_model_keys and not ppe_prompt_terms:
        ppe_prompt_terms = list(ALL_PPE_PROMPT_TERMS)

    runtime_load = "low"
    if len(required_model_keys) == 2:
        runtime_load = "medium"
    elif len(required_model_keys) >= 3:
        runtime_load = "high"

    derived_demo = "yolo"
    if "yoloe_long_tail" in required_model_keys and "coco_primary" not in required_model_keys:
        derived_demo = "yoloe"
    elif "ppe_specialist" in required_model_keys and "coco_primary" not in required_model_keys:
        derived_demo = "yoloe"
    elif any(model_key in required_model_keys for model_key in ("ppe_specialist", "yoloe_long_tail")):
        derived_demo = "yolo+yoloe"

    execution_plan = {
        "profile": profile,
        "capabilities": capabilities,
        "required_model_keys": required_model_keys,
        "run_coco_primary": "coco_primary" in required_model_keys,
        "run_ppe_specialist": "ppe_specialist" in required_model_keys,
        "run_yoloe_long_tail": "yoloe_long_tail" in required_model_keys,
        "tracking_enabled": tracking_enabled,
        "zones_required": zones_required,
        "association_enabled": association_enabled,
        "runtime_load": runtime_load,
        "ppe_prompt_terms": ppe_prompt_terms,
        "yoloe_prompt_terms": yoloe_prompt_terms,
        "derived_demo": derived_demo,
        "model_stack": [
            "COCO Primary" if model_key == "coco_primary" else
            "PPE Specialist" if model_key == "ppe_specialist" else
            "YOLOE Long-Tail"
            for model_key in required_model_keys
        ],
    }
    return execution_plan


def normalize_camera_record(camera: dict, cfg: dict | None = None) -> tuple[dict, bool]:
    if cfg is None:
        from config_manager import get_config

        cfg = get_config()

    updated = deepcopy(camera)
    changed = False

    if normalize_camera_connection(updated, existing_camera=camera):
        changed = True

    # Keep rule-derived fields coherent even when only safety_rule_ids changed.
    from camera_config_utils import sync_camera_rule_fields

    if sync_camera_rule_fields(updated):
        changed = True

    execution_plan = build_execution_plan(updated, cfg)
    capabilities = execution_plan["capabilities"]
    profile = execution_plan["profile"]
    custom_long_tail_terms = derive_custom_long_tail_terms(updated, cfg)
    if not _ensure_list(updated.get("safety_rule_ids")) and capabilities:
        updated["safety_rule_ids"] = default_rule_ids_for_capabilities(capabilities)
        changed = True

    derived_ppe_rule_ids = [value for value in _ensure_list(updated.get("safety_rule_ids")) if value.startswith("ppe_")]
    if updated.get("ppe_rule_ids") != derived_ppe_rule_ids:
        updated["ppe_rule_ids"] = derived_ppe_rule_ids
        changed = True

    next_fields = {
        "profile": profile,
        "capabilities": capabilities,
        "custom_long_tail_terms": custom_long_tail_terms,
        "execution_plan": execution_plan,
        "demo": execution_plan["derived_demo"],
    }

    compatibility_classes: list[str] = []
    for value in execution_plan["ppe_prompt_terms"]:
        _append_unique(compatibility_classes, value)
    for value in execution_plan["yoloe_prompt_terms"]:
        _append_unique(compatibility_classes, value)
    if compatibility_classes:
        compatibility_classes.insert(0, "person")
    next_fields["yoloe_classes"] = compatibility_classes

    for key, value in next_fields.items():
        if updated.get(key) != value:
            updated[key] = value
            changed = True

    return updated, changed
