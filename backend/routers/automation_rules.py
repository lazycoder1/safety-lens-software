"""Automation rules CRUD endpoints."""
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config_manager import get_config, save_config

logger = logging.getLogger("rakshak_lens.automation_rules")

router = APIRouter(prefix="/api", tags=["automation-rules"])

# Seed data — written to config on first access so rules persist from day one.
DEFAULT_AUTOMATION_RULES = [
    {
        "id": "rule-001",
        "name": "Gate Entry — Auto Open",
        "description": "Open gate for authorised vehicles and personnel",
        "enabled": True,
        "trigger": "plate_read",
        "cameras": ["cam-01"],
        "conditions": [
            {"type": "plate_in_list", "params": {"list": "Whitelist"}},
            {"type": "face_in_group", "params": {"group": "Employees"}},
        ],
        "thenActions": [
            {"type": "open_gate", "params": {"device": "main-gate"}},
            {"type": "log_entry", "params": {}},
        ],
        "elseActions": [
            {"type": "create_alert", "params": {"severity": "P1"}},
            {"type": "send_telegram", "params": {}},
            {"type": "close_gate", "params": {"device": "main-gate"}},
        ],
        "cooldownSeconds": 10,
        "priority": 8,
        "lastTriggered": "2026-03-28T09:14:00Z",
        "preset": "gate_entry",
    },
    {
        "id": "rule-002",
        "name": "PPE — Helmet Required",
        "description": "Detect missing helmets on the assembly line",
        "enabled": True,
        "trigger": "detection",
        "cameras": ["cam-02", "cam-03"],
        "conditions": [
            {"type": "class_is", "params": {"classes": "no_helmet"}},
            {"type": "zone_is", "params": {"zone": "Assembly Line"}},
            {"type": "confidence_above", "params": {"value": "0.7"}},
        ],
        "thenActions": [
            {"type": "create_alert", "params": {"severity": "P2"}},
            {"type": "send_telegram", "params": {}},
        ],
        "elseActions": [],
        "cooldownSeconds": 60,
        "priority": 5,
        "lastTriggered": "2026-03-28T08:52:00Z",
        "preset": "ppe",
    },
    {
        "id": "rule-003",
        "name": "Fire Emergency",
        "description": "Detect fire or smoke and trigger emergency response",
        "enabled": True,
        "trigger": "detection",
        "cameras": [],
        "conditions": [
            {"type": "class_is", "params": {"classes": "fire,smoke"}},
            {"type": "confidence_above", "params": {"value": "0.6"}},
        ],
        "thenActions": [
            {"type": "create_alert", "params": {"severity": "P1"}},
            {"type": "send_telegram", "params": {}},
            {"type": "trigger_plc", "params": {"device": "siren"}},
            {"type": "webhook", "params": {"url": "https://hooks.example.com/fire"}},
        ],
        "elseActions": [],
        "cooldownSeconds": 30,
        "priority": 10,
        "lastTriggered": None,
        "preset": "fire",
    },
    {
        "id": "rule-004",
        "name": "After-Hours Intrusion",
        "description": "Alert when motion is detected in restricted areas after hours",
        "enabled": True,
        "trigger": "zone_enter",
        "cameras": ["cam-04"],
        "conditions": [
            {"type": "zone_is", "params": {"zone": "Warehouse"}},
            {"type": "time_between", "params": {"from": "22:00", "to": "06:00"}},
        ],
        "thenActions": [
            {"type": "create_alert", "params": {"severity": "P1"}},
            {"type": "send_telegram", "params": {}},
            {"type": "play_sound", "params": {}},
        ],
        "elseActions": [],
        "cooldownSeconds": 120,
        "priority": 9,
        "lastTriggered": "2026-03-27T23:45:00Z",
        "preset": "after_hours",
    },
    {
        "id": "rule-005",
        "name": "Overcrowding Alert",
        "description": "Trigger when too many people are in Assembly Line zone",
        "enabled": True,
        "trigger": "count_threshold",
        "cameras": ["cam-02", "cam-03"],
        "conditions": [
            {"type": "zone_is", "params": {"zone": "Assembly Line"}},
            {"type": "count_exceeds", "params": {"count": "15"}},
        ],
        "thenActions": [
            {"type": "create_alert", "params": {"severity": "P2"}},
            {"type": "send_telegram", "params": {}},
        ],
        "elseActions": [],
        "cooldownSeconds": 300,
        "priority": 6,
        "lastTriggered": "2026-03-28T07:30:00Z",
        "preset": "overcrowding",
    },
]


def _ensure_automation_rules(cfg: dict) -> list[dict]:
    """Seed default automation rules if missing."""
    if "automation_rules" not in cfg:
        cfg["automation_rules"] = json.loads(json.dumps(DEFAULT_AUTOMATION_RULES))
        save_config(cfg)
    return cfg["automation_rules"]


# ── Pydantic models ──────────────────────────────────────────────

class ConditionModel(BaseModel):
    type: str
    params: dict[str, str] = {}


class ActionModel(BaseModel):
    type: str
    params: dict[str, str] = {}


class AutomationRuleCreate(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    trigger: str = "detection"
    cameras: list[str] = []
    conditions: list[ConditionModel] = []
    thenActions: list[ActionModel] = []
    elseActions: list[ActionModel] = []
    cooldownSeconds: int = 60
    priority: int = 5
    severity: Optional[str] = None
    outputIds: list[str] = []
    messageTemplate: str = ""
    schedule: Optional[dict] = None
    preset: Optional[str] = None


class AutomationRuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    trigger: Optional[str] = None
    cameras: Optional[list[str]] = None
    conditions: Optional[list[ConditionModel]] = None
    thenActions: Optional[list[ActionModel]] = None
    elseActions: Optional[list[ActionModel]] = None
    cooldownSeconds: Optional[int] = None
    priority: Optional[int] = None
    severity: Optional[str] = None
    outputIds: Optional[list[str]] = None
    messageTemplate: Optional[str] = None
    schedule: Optional[dict] = None
    preset: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────

@router.get("/automation-rules")
async def api_get_automation_rules():
    cfg = get_config()
    return _ensure_automation_rules(cfg)


@router.post("/automation-rules")
async def api_create_automation_rule(body: AutomationRuleCreate):
    cfg = get_config()
    rules = _ensure_automation_rules(cfg)

    if any(r["name"].lower() == body.name.lower() for r in rules):
        raise HTTPException(status_code=409, detail="An automation rule with this name already exists")

    # Generate unique ID
    rule_id = "rule_" + body.name.lower().replace(" ", "_").replace("—", "").replace("-", "_")
    existing_ids = {r["id"] for r in rules}
    base_id = rule_id
    counter = 1
    while rule_id in existing_ids:
        rule_id = f"{base_id}_{counter}"
        counter += 1

    rule = {"id": rule_id, "lastTriggered": None, **body.model_dump()}
    rules.append(rule)
    save_config(cfg)
    logger.info("Automation rule created", extra={"rule_id": rule_id, "name": body.name})
    return rule


@router.put("/automation-rules/{rule_id}")
async def api_update_automation_rule(rule_id: str, body: AutomationRuleUpdate):
    cfg = get_config()
    rules = _ensure_automation_rules(cfg)
    for rule in rules:
        if rule["id"] == rule_id:
            updates = body.model_dump(exclude_none=True)
            if "name" in updates:
                if any(r["name"].lower() == updates["name"].lower() and r["id"] != rule_id for r in rules):
                    raise HTTPException(status_code=409, detail="An automation rule with this name already exists")
            rule.update(updates)
            save_config(cfg)
            logger.info("Automation rule updated", extra={"rule_id": rule_id})
            return rule
    raise HTTPException(status_code=404, detail="Automation rule not found")


@router.put("/automation-rules/{rule_id}/toggle")
async def api_toggle_automation_rule(rule_id: str):
    cfg = get_config()
    rules = _ensure_automation_rules(cfg)
    for rule in rules:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_config(cfg)
            logger.info("Automation rule toggled", extra={"rule_id": rule_id, "enabled": rule["enabled"]})
            return rule
    raise HTTPException(status_code=404, detail="Automation rule not found")


@router.delete("/automation-rules/{rule_id}")
async def api_delete_automation_rule(rule_id: str):
    cfg = get_config()
    rules = _ensure_automation_rules(cfg)
    original_len = len(rules)
    cfg["automation_rules"] = [r for r in rules if r["id"] != rule_id]
    if len(cfg["automation_rules"]) == original_len:
        raise HTTPException(status_code=404, detail="Automation rule not found")
    save_config(cfg)
    logger.info("Automation rule deleted", extra={"rule_id": rule_id})
    return {"deleted": rule_id}
