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
from helmet_colour import HELMET_COLOUR_CAPABILITY, normalize_helmet_colour_policy

ALLOWED_CAPABILITY_MODEL_OVERRIDES: dict[CapabilityKey, set[ModelKey]] = {
    "apron_required": {"ppe_closed_set_candidate"},
    "harness_required": {"ppe_closed_set_candidate"},
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


def normalize_capability_model_overrides(camera: dict) -> dict[CapabilityKey, ModelKey]:
    """Return allowed camera-scoped capability -> model-key overrides."""
    raw = camera.get("capability_model_overrides")
    if raw is None:
        raw = camera.get("model_overrides")
    if not isinstance(raw, dict):
        return {}

    normalized: dict[CapabilityKey, ModelKey] = {}
    for raw_capability, raw_model in raw.items():
        if not isinstance(raw_capability, str):
            continue
        capability = normalize_capability_key(raw_capability)
        if not capability:
            continue
        model_key = raw_model.get("model_key") if isinstance(raw_model, dict) else raw_model
        if not isinstance(model_key, str):
            continue
        if model_key in ALLOWED_CAPABILITY_MODEL_OVERRIDES.get(capability, set()):
            normalized[capability] = model_key  # type: ignore[assignment]
    return normalized


def _model_key_for_capability(
    capability: CapabilityKey,
    capability_model_overrides: dict[CapabilityKey, ModelKey] | None = None,
) -> ModelKey:
    overrides = capability_model_overrides or {}
    return overrides.get(capability) or CAPABILITY_REGISTRY[capability]["model_family"]


def _ordered_capabilities(capabilities: list[str]) -> list[CapabilityKey]:
    ordered: list[CapabilityKey] = []
    for key in CAPABILITY_REGISTRY:
        if key in capabilities:
            ordered.append(key)
    return ordered


def infer_capabilities_from_camera(camera: dict, cfg: dict) -> list[CapabilityKey]:
    explicit = [normalize_capability_key(value) for value in _ensure_list(camera.get("capabilities"))]
    explicit_caps = [value for value in explicit if value]
    rule_ids = _ensure_list(camera.get("safety_rule_ids"))
    if explicit_caps and not rule_ids:
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

    if not rule_ids:
        for capability in explicit_caps:
            _append_unique(capabilities, capability)
        for class_name in _ensure_list(camera.get("yoloe_classes")):
            normalized = _normalize_text(class_name)
            if normalized == "person":
                continue
            mapped = CLASS_TERM_TO_CAPABILITY.get(normalized)
            if mapped:
                _append_unique(capabilities, mapped)
            else:
                _append_unique(capabilities, "custom_long_tail")
        for rule_name in _ensure_list(camera.get("rules")):
            normalized = _normalize_text(rule_name)
            if "zone intrusion" in normalized:
                _append_unique(capabilities, "zone_intrusion")
            if "person" in normalized:
                _append_unique(capabilities, "person_presence")
            if "vehicle" in normalized or "forklift" in normalized:
                _append_unique(capabilities, "vehicle_presence")
            if "plate" in normalized or "anpr" in normalized:
                _append_unique(capabilities, "plate_recognition")
            if "animal" in normalized:
                _append_unique(capabilities, "animal_presence")
            if "phone" in normalized:
                _append_unique(capabilities, "mobile_phone")
            if "fire" in normalized or "smoke" in normalized:
                _append_unique(capabilities, "fire_smoke")
    else:
        for capability in explicit_caps:
            if not CAPABILITY_REGISTRY[capability].get("safety_rule_ids"):
                _append_unique(capabilities, capability)

    return _ordered_capabilities(capabilities)


def derive_custom_long_tail_terms(camera: dict, cfg: dict) -> list[str]:
    terms: list[str] = []
    explicit_terms = _ensure_list(camera.get("custom_long_tail_terms"))
    for term in explicit_terms:
        _append_unique(terms, term)

    rule_ids = _ensure_list(camera.get("safety_rule_ids"))
    rule_map = {rule["id"]: rule for rule in cfg.get("safety_rules", [])}
    assigned_rule_terms: set[str] = set()
    for rule_id in rule_ids:
        rule = rule_map.get(rule_id)
        if not rule or rule.get("id") in RULE_ID_TO_CAPABILITY:
            if rule and rule.get("type") == "ppe":
                assigned_rule_terms.update(_normalize_text(class_name) for class_name in _ensure_list(rule.get("classes")))
            continue
        if rule.get("model") == "yoloe":
            for class_name in _ensure_list(rule.get("classes")):
                _append_unique(terms, class_name)
                assigned_rule_terms.add(_normalize_text(class_name))

    for class_name in _ensure_list(camera.get("yoloe_classes")):
        normalized = _normalize_text(class_name)
        if normalized == "person":
            continue
        if normalized not in CLASS_TERM_TO_CAPABILITY and normalized not in assigned_rule_terms:
            _append_unique(terms, class_name)

    return terms


def _append_ppe_rule_prompt_terms(
    terms: list[str],
    camera: dict,
    cfg: dict,
    capability_model_overrides: dict[CapabilityKey, ModelKey],
):
    """Let YAML safety-rule classes tune known PPE specialist prompts."""
    rule_map = {rule["id"]: rule for rule in cfg.get("safety_rules", [])}
    for rule_id in _ensure_list(camera.get("safety_rule_ids")):
        capability = RULE_ID_TO_CAPABILITY.get(rule_id)
        if capability and _model_key_for_capability(capability, capability_model_overrides) != "ppe_specialist":
            continue
        # Worker-helmet rules historically name ``hard hat`` and
        # ``safety helmet``.  The Jetson fixed engine uses the shared generic
        # helmet profile from the capability registry, so adding the legacy
        # rule terms here would force a slow PyTorch fallback.
        if capability == "helmet_required":
            continue
        rule = rule_map.get(rule_id)
        if not rule or rule.get("type") != "ppe" or not rule.get("enabled", True):
            continue
        for class_name in _ensure_list(rule.get("classes")):
            _append_unique(terms, class_name)


def _resolve_profile(camera: dict, capabilities: list[CapabilityKey]) -> CameraProfile:
    raw_profile = camera.get("profile")
    if raw_profile in PROFILE_DEFAULT_CAPABILITIES:
        return raw_profile
    return infer_profile_from_capabilities(capabilities)


def required_model_keys_for_capabilities(
    capabilities: list[CapabilityKey],
    capability_model_overrides: dict[CapabilityKey, ModelKey] | None = None,
) -> list[ModelKey]:
    required: list[ModelKey] = []
    capability_set = set(capabilities)
    model_keys = {
        _model_key_for_capability(key, capability_model_overrides)
        for key in capability_set
    }
    has_ppe_specialist = "ppe_specialist" in model_keys
    has_coco = "coco_primary" in model_keys

    if has_coco or has_ppe_specialist:
        required.append("coco_primary")
    if has_ppe_specialist:
        required.append("ppe_specialist")
    if "ppe_closed_set_candidate" in model_keys:
        required.append("ppe_closed_set_candidate")
    if "yoloe_long_tail" in model_keys:
        required.append("yoloe_long_tail")
    if "fire_smoke_specialist" in model_keys:
        required.append("fire_smoke_specialist")
    if "face_recognition" in model_keys:
        required.append("face_recognition")
    if "pose_specialist" in model_keys:
        required.append("pose_specialist")
    if "plate_recognition" in model_keys:
        required.append("plate_recognition")
    return required


def normalize_capability_windows(camera: dict) -> list[dict[str, Any]]:
    """Normalize camera-scoped capability active windows for execution-plan metadata."""
    raw = camera.get("capability_windows")
    if raw is None:
        raw = camera.get("capability_active_windows")
    entries: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        iterable = []
        for capability, schedule in raw.items():
            if isinstance(schedule, list):
                iterable.append({"capability": capability, "windows": schedule})
            elif isinstance(schedule, dict):
                iterable.append({"capability": capability, **schedule})
    elif isinstance(raw, list):
        iterable = [item for item in raw if isinstance(item, dict)]
    else:
        iterable = []

    for index, item in enumerate(iterable, start=1):
        raw_capabilities = item.get("capabilities")
        if raw_capabilities is None:
            raw_capabilities = [item.get("capability")]
        elif isinstance(raw_capabilities, str):
            raw_capabilities = [raw_capabilities]
        if not isinstance(raw_capabilities, list):
            raw_capabilities = []

        capabilities: list[CapabilityKey] = []
        for raw_capability in raw_capabilities:
            if not isinstance(raw_capability, str):
                continue
            capability = normalize_capability_key(raw_capability)
            if capability and capability not in capabilities:
                capabilities.append(capability)
        if not capabilities:
            continue

        windows = item.get("windows")
        normalized = {
            "id": str(item.get("id") or f"capability_window_{index}"),
            "capabilities": capabilities,
            "mode": str(item.get("mode") or "detection"),
            "windows": windows if isinstance(windows, list) else [],
        }
        if isinstance(item.get("active"), bool):
            normalized["active"] = item["active"]
        entries.append(normalized)

    return entries


def build_execution_plan(camera: dict, cfg: dict | None = None) -> dict[str, Any]:
    if cfg is None:
        from config_manager import get_config

        cfg = get_config()

    capabilities = infer_capabilities_from_camera(camera, cfg)
    profile = _resolve_profile(camera, capabilities)
    capability_model_overrides = normalize_capability_model_overrides(camera)
    required_model_keys = required_model_keys_for_capabilities(capabilities, capability_model_overrides)
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
        model_key = _model_key_for_capability(key, capability_model_overrides)
        if model_key == "ppe_specialist":
            for prompt in definition.get("prompt_terms", []):
                _append_unique(ppe_prompt_terms, prompt)
        elif model_key == "yoloe_long_tail":
            for prompt in definition.get("prompt_terms", []):
                _append_unique(yoloe_prompt_terms, prompt)

    _append_ppe_rule_prompt_terms(ppe_prompt_terms, camera, cfg, capability_model_overrides)

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
    if "fire_smoke_specialist" in required_model_keys and len(required_model_keys) == 1:
        derived_demo = "fire_smoke"
    elif "yoloe_long_tail" in required_model_keys and "coco_primary" not in required_model_keys:
        derived_demo = "yoloe"
    elif "ppe_specialist" in required_model_keys and "coco_primary" not in required_model_keys:
        derived_demo = "yoloe"
    elif any(model_key in required_model_keys for model_key in ("ppe_specialist", "yoloe_long_tail")):
        derived_demo = "yolo+yoloe"

    _MODEL_KEY_LABELS = {
        "coco_primary": "COCO Primary",
        "ppe_specialist": "PPE Specialist",
        "ppe_closed_set_candidate": "Closed-Set PPE Candidate",
        "yoloe_long_tail": "YOLOE Long-Tail",
        "fire_smoke_specialist": "Fire / Smoke Specialist",
        "face_recognition": "Face Recognition",
        "pose_specialist": "Pose Specialist",
        "plate_recognition": "Plate Recognition",
    }

    execution_plan = {
        "profile": profile,
        "capabilities": capabilities,
        "required_model_keys": required_model_keys,
        "capability_model_overrides": capability_model_overrides,
        "run_coco_primary": "coco_primary" in required_model_keys,
        "run_ppe_specialist": "ppe_specialist" in required_model_keys,
        "run_ppe_closed_set_candidate": "ppe_closed_set_candidate" in required_model_keys,
        "run_yoloe_long_tail": "yoloe_long_tail" in required_model_keys,
        "run_fire_smoke_specialist": "fire_smoke_specialist" in required_model_keys,
        "run_face_recognition": "face_recognition" in required_model_keys,
        "run_pose_specialist": "pose_specialist" in required_model_keys,
        "run_plate_recognition": "plate_recognition" in required_model_keys,
        "tracking_enabled": tracking_enabled,
        "zones_required": zones_required,
        "association_enabled": association_enabled,
        "runtime_load": runtime_load,
        "capability_windows": normalize_capability_windows(camera),
        "ppe_prompt_terms": ppe_prompt_terms,
        "yoloe_prompt_terms": yoloe_prompt_terms,
        "derived_demo": derived_demo,
        "model_stack": [
            _MODEL_KEY_LABELS.get(model_key, model_key)
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
    helmet_colour_policy = normalize_helmet_colour_policy(
        updated.get("helmet_colour_policy") or updated.get("helmet_color_policy"),
        enabled=HELMET_COLOUR_CAPABILITY in capabilities,
    )
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
        "helmet_colour_policy": helmet_colour_policy,
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
