"""Unified safety rules CRUD endpoints (PPE + alert rules)."""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from dependencies import require_admin

from config_manager import get_config, save_config
from video_processing import restart_camera

logger = logging.getLogger("safetylens.safety_rules")

router = APIRouter(prefix="/api", tags=["safety-rules"])

DEFAULT_SAFETY_RULES = [
    # PPE rules (type="ppe", model="yoloe")
    {"id": "ppe_helmet", "name": "Helmet", "type": "ppe", "classes": ["hard hat", "safety helmet"], "model": "yoloe", "severity": "P2", "enabled": True},
    {"id": "ppe_vest", "name": "Safety Vest", "type": "ppe", "classes": ["safety vest", "high visibility vest", "fluorescent vest"], "model": "yoloe", "severity": "P2", "enabled": True},
    {"id": "ppe_gloves", "name": "Gloves", "type": "ppe", "classes": ["gloves"], "model": "yoloe", "severity": "P2", "enabled": True},
    {"id": "ppe_hairnet", "name": "Hairnet", "type": "ppe", "classes": ["hairnet"], "model": "yoloe", "severity": "P3", "enabled": True},
    {"id": "ppe_facemask", "name": "Face Mask", "type": "ppe", "classes": ["face mask"], "model": "yoloe", "severity": "P3", "enabled": True},
    {"id": "ppe_apron", "name": "Apron", "type": "ppe", "classes": ["apron"], "model": "yoloe", "severity": "P3", "enabled": True},
    {"id": "ppe_boots", "name": "Safety Boots", "type": "ppe", "classes": ["safety boots", "steel-toe boots"], "model": "yoloe", "severity": "P2", "enabled": True},
    {"id": "ppe_goggles", "name": "Safety Goggles", "type": "ppe", "classes": ["safety goggles", "protective eyewear"], "model": "yoloe", "severity": "P2", "enabled": True},
    # Alert rules (type="alert", model="yolo")
    {"id": "alert_mobile_phone", "name": "Mobile Phone Usage", "type": "alert", "classes": ["cell phone"], "model": "yolo", "severity": "P3", "enabled": True},
    {"id": "alert_animal", "name": "Animal Intrusion", "type": "alert", "classes": ["dog", "cat"], "model": "yolo", "severity": "P3", "enabled": True},
    {"id": "alert_person", "name": "Person Detected", "type": "alert", "classes": ["person"], "model": "yolo", "severity": "P4", "enabled": True},
    {"id": "alert_vehicle", "name": "Vehicle Detected", "type": "alert", "classes": ["truck", "car", "motorcycle"], "model": "yolo", "severity": "P4", "enabled": True},
    {"id": "alert_zone_intrusion", "name": "Zone Intrusion", "type": "alert", "classes": ["person"], "model": "yolo", "severity": "P1", "enabled": True},
]

# Default alert rule IDs (for migration)
_DEFAULT_ALERT_IDS = [r["id"] for r in DEFAULT_SAFETY_RULES if r["type"] == "alert"]

# Legacy alert_classes key -> safety rule ID mapping
_LEGACY_ALERT_MAP = {
    "mobile_phone": "alert_mobile_phone",
    "animal_intrusion": "alert_animal",
    "person_detected": "alert_person",
    "vehicle_detected": "alert_vehicle",
    "zone_intrusion": "alert_zone_intrusion",
}


class SafetyRuleCreate(BaseModel):
    name: str
    type: str  # "ppe" | "alert"
    classes: list[str]
    model: str = "yoloe"  # "yolo" | "yoloe"
    severity: str = "P2"


class SafetyRuleUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    classes: Optional[list[str]] = None
    model: Optional[str] = None
    severity: Optional[str] = None
    enabled: Optional[bool] = None


def _migrate_config(cfg: dict):
    """Migrate old ppe_rules + alert_classes config to unified safety_rules."""
    if "safety_rules" in cfg:
        return

    safety_rules = []

    if "ppe_rules" in cfg:
        # Convert existing PPE rules
        for old in cfg["ppe_rules"]:
            safety_rules.append({
                "id": old["id"],
                "name": old["name"],
                "type": "ppe",
                "classes": old.get("yoloe_classes", []),
                "model": "yoloe",
                "severity": old.get("severity", "P2"),
                "enabled": old.get("enabled", True),
            })
    else:
        # Seed default PPE rules
        for r in DEFAULT_SAFETY_RULES:
            if r["type"] == "ppe":
                safety_rules.append(json.loads(json.dumps(r)))

    # Add default alert rules
    for r in DEFAULT_SAFETY_RULES:
        if r["type"] == "alert":
            safety_rules.append(json.loads(json.dumps(r)))

    cfg["safety_rules"] = safety_rules

    # Migrate camera references
    for cam_id, cam in cfg.get("cameras", {}).items():
        rule_ids = []
        # Keep existing PPE rule IDs as-is
        old_ppe_ids = cam.get("ppe_rule_ids", [])
        rule_ids.extend(old_ppe_ids)
        # Map old alert_classes to alert rule IDs
        old_alert_classes = cam.get("alert_classes", [])
        for key in old_alert_classes:
            mapped = _LEGACY_ALERT_MAP.get(key, key)
            if mapped not in rule_ids:
                rule_ids.append(mapped)
        cam["safety_rule_ids"] = rule_ids

    # Remove old key
    cfg.pop("ppe_rules", None)
    save_config(cfg)
    logger.info("Migrated config to unified safety_rules")


def _ensure_safety_rules(cfg: dict) -> list[dict]:
    """Seed default safety rules if missing, or migrate from old format."""
    if "safety_rules" not in cfg:
        if "ppe_rules" in cfg:
            _migrate_config(cfg)
        else:
            cfg["safety_rules"] = json.loads(json.dumps(DEFAULT_SAFETY_RULES))
            save_config(cfg)
    return cfg["safety_rules"]


def derive_yoloe_classes(safety_rule_ids: list[str], cfg: dict) -> list[str]:
    """Compute yoloe_classes from assigned safety rule IDs (PPE type only)."""
    rules = cfg.get("safety_rules", [])
    rule_map = {r["id"]: r for r in rules}
    classes = ["person"]
    for rid in safety_rule_ids:
        rule = rule_map.get(rid)
        if rule and rule.get("enabled", True):
            classes.extend(rule["classes"])
    return list(dict.fromkeys(classes))  # deduplicate, preserve order


def _refresh_cameras_for_rule(rule_id: str, cfg: dict):
    """Re-derive yoloe_classes for all cameras using this safety rule."""
    changed = False
    for cam_id, cam in cfg.get("cameras", {}).items():
        rule_ids = cam.get("safety_rule_ids", [])
        if rule_id in rule_ids and cam.get("demo") == "yoloe":
            cam["yoloe_classes"] = derive_yoloe_classes(rule_ids, cfg)
            changed = True
    if changed:
        save_config(cfg)


@router.get("/safety-rules")
async def api_get_safety_rules():
    cfg = get_config()
    return _ensure_safety_rules(cfg)


@router.post("/safety-rules")
async def api_create_safety_rule(body: SafetyRuleCreate):
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    # Validate unique name
    if any(r["name"].lower() == body.name.lower() for r in rules):
        raise HTTPException(status_code=409, detail="A safety rule with this name already exists")
    # Auto-generate ID
    prefix = "ppe_" if body.type == "ppe" else "alert_"
    rule_id = prefix + body.name.lower().replace(" ", "_")
    # Ensure unique ID
    existing_ids = {r["id"] for r in rules}
    base_id = rule_id
    counter = 1
    while rule_id in existing_ids:
        rule_id = f"{base_id}_{counter}"
        counter += 1
    rule = {"id": rule_id, "enabled": True, **body.model_dump()}
    rules.append(rule)
    save_config(cfg)
    logger.info("Safety rule created", extra={"rule_id": rule_id, "name": body.name})
    return rule


@router.put("/safety-rules/{rule_id}")
async def api_update_safety_rule(rule_id: str, body: SafetyRuleUpdate):
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    for rule in rules:
        if rule["id"] == rule_id:
            updates = body.model_dump(exclude_none=True)
            # Validate name uniqueness if changing name
            if "name" in updates:
                if any(r["name"].lower() == updates["name"].lower() and r["id"] != rule_id for r in rules):
                    raise HTTPException(status_code=409, detail="A safety rule with this name already exists")
            rule.update(updates)
            save_config(cfg)
            _refresh_cameras_for_rule(rule_id, cfg)
            logger.info("Safety rule updated", extra={"rule_id": rule_id})
            return rule
    raise HTTPException(status_code=404, detail="Safety rule not found")


@router.put("/safety-rules/{rule_id}/toggle")
async def api_toggle_safety_rule(rule_id: str):
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    for rule in rules:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_config(cfg)
            _refresh_cameras_for_rule(rule_id, cfg)
            return rule
    raise HTTPException(status_code=404, detail="Safety rule not found")


@router.delete("/safety-rules/{rule_id}")
async def api_delete_safety_rule(rule_id: str):
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    original_len = len(rules)
    cfg["safety_rules"] = [r for r in rules if r["id"] != rule_id]
    if len(cfg["safety_rules"]) == original_len:
        raise HTTPException(status_code=404, detail="Safety rule not found")
    # Remove from cameras
    for cam_id, cam in cfg.get("cameras", {}).items():
        rule_ids = cam.get("safety_rule_ids", [])
        if rule_id in rule_ids:
            cam["safety_rule_ids"] = [rid for rid in rule_ids if rid != rule_id]
            if cam.get("demo") == "yoloe":
                cam["yoloe_classes"] = derive_yoloe_classes(cam["safety_rule_ids"], cfg)
    save_config(cfg)
    logger.info("Safety rule deleted", extra={"rule_id": rule_id})
    return {"deleted": rule_id}


class RuleCameraAssign(BaseModel):
    camera_ids: list[str]


@router.put("/safety-rules/{rule_id}/cameras", dependencies=[Depends(require_admin)])
async def api_assign_rule_cameras(rule_id: str, body: RuleCameraAssign):
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    if not any(r["id"] == rule_id for r in rules):
        raise HTTPException(status_code=404, detail="Safety rule not found")

    desired = set(body.camera_ids)
    changed_cams: list[str] = []

    for cam_id, cam in cfg.get("cameras", {}).items():
        rule_ids = cam.get("safety_rule_ids", [])
        had_rule = rule_id in rule_ids

        if cam_id in desired and not had_rule:
            rule_ids.append(rule_id)
            cam["safety_rule_ids"] = rule_ids
            changed_cams.append(cam_id)
        elif cam_id not in desired and had_rule:
            cam["safety_rule_ids"] = [rid for rid in rule_ids if rid != rule_id]
            changed_cams.append(cam_id)

        if cam_id in changed_cams and cam.get("demo") == "yoloe":
            cam["yoloe_classes"] = derive_yoloe_classes(cam["safety_rule_ids"], cfg)

    save_config(cfg)
    for cam_id in changed_cams:
        restart_camera(cam_id)

    logger.info("Rule camera assignment updated", extra={"rule_id": rule_id, "cameras": list(desired), "changed": changed_cams})
    return {"rule_id": rule_id, "camera_ids": list(desired), "changed": changed_cams}
