"""
SafetyLens detection rules CRUD endpoints.
"""

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config_manager import get_config, save_config

router = APIRouter(prefix="/api", tags=["detection-rules"])

DEFAULT_DETECTION_RULES = [
    {"id": "r1", "name": "Hard Hat Detection", "model": "YOLOE", "promptType": "text", "prompts": ["hard hat", "safety helmet"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 3, "category": "PPE"},
    {"id": "r2", "name": "Safety Vest Detection", "model": "YOLOE", "promptType": "text", "prompts": ["safety vest", "high visibility vest"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 3, "category": "PPE"},
    {"id": "r3", "name": "Safety Goggles", "model": "YOLOE", "promptType": "text", "prompts": ["safety goggles", "protective eyewear"], "confidenceThreshold": 0.45, "severity": "P2", "enabled": True, "camerasCount": 2, "category": "PPE"},
    {"id": "r4", "name": "Safety Harness", "model": "YOLOE", "promptType": "visual", "prompts": ["harness-ref.jpg"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 1, "category": "PPE"},
    {"id": "r5", "name": "Zone Intrusion", "model": "YOLO26", "promptType": "internal", "prompts": ["person"], "confidenceThreshold": 0.6, "severity": "P1", "enabled": True, "camerasCount": 2, "category": "Zone Safety"},
    {"id": "r6", "name": "Person Fall Detection", "model": "YOLO-pose", "promptType": "internal", "prompts": ["pose-keypoints"], "confidenceThreshold": 0.55, "severity": "P1", "enabled": True, "camerasCount": 3, "category": "Emergency"},
    {"id": "r7", "name": "Mobile Phone Usage", "model": "YOLO26", "promptType": "internal", "prompts": ["cell phone"], "confidenceThreshold": 0.6, "severity": "P2", "enabled": True, "camerasCount": 2, "category": "Behavior"},
    {"id": "r8", "name": "Animal Detection", "model": "YOLOE", "promptType": "text", "prompts": ["snake", "dog", "cat"], "confidenceThreshold": 0.4, "severity": "P3", "enabled": True, "camerasCount": 1, "category": "Environment"},
    {"id": "r9", "name": "Gangway Blockage", "model": "VLM", "promptType": "text", "prompts": ["Is the gangway clear and unobstructed?"], "confidenceThreshold": 0.5, "severity": "P3", "enabled": True, "camerasCount": 1, "category": "Environment"},
    {"id": "r10", "name": "Fire / Smoke Detection", "model": "YOLOE", "promptType": "text", "prompts": ["fire", "smoke", "flames"], "confidenceThreshold": 0.35, "severity": "P1", "enabled": True, "camerasCount": 3, "category": "Emergency"},
    {"id": "r11", "name": "Head Cap vs Helmet", "model": "YOLOE", "promptType": "text", "prompts": ["hair net", "head cap", "hard hat"], "confidenceThreshold": 0.45, "severity": "P2", "enabled": False, "camerasCount": 0, "category": "PPE"},
    {"id": "r12", "name": "Forklift Operator Helmet", "model": "YOLOE", "promptType": "text", "prompts": ["forklift", "hard hat"], "confidenceThreshold": 0.5, "severity": "P2", "enabled": True, "camerasCount": 1, "category": "PPE"},
]


class DetectionRuleCreate(BaseModel):
    name: str
    model: str
    promptType: str
    prompts: list[str]
    confidenceThreshold: float
    severity: str
    category: str = "Custom"


@router.get("/detection-rules")
async def api_get_detection_rules():
    cfg = get_config()
    rules = cfg.get("detection_rules", DEFAULT_DETECTION_RULES)
    return rules


@router.put("/detection-rules/{rule_id}/toggle")
async def api_toggle_detection_rule(rule_id: str):
    cfg = get_config()
    if "detection_rules" not in cfg:
        cfg["detection_rules"] = json.loads(json.dumps(DEFAULT_DETECTION_RULES))
    for rule in cfg["detection_rules"]:
        if rule["id"] == rule_id:
            rule["enabled"] = not rule["enabled"]
            save_config(cfg)
            return rule
    raise HTTPException(status_code=404, detail="Rule not found")


@router.post("/detection-rules")
async def api_create_detection_rule(body: DetectionRuleCreate):
    cfg = get_config()
    if "detection_rules" not in cfg:
        cfg["detection_rules"] = json.loads(json.dumps(DEFAULT_DETECTION_RULES))
    existing_ids = [int(r["id"].replace("r", "")) for r in cfg["detection_rules"] if r["id"].startswith("r") and r["id"][1:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    rule = {
        "id": f"r{next_id}",
        "enabled": True,
        "camerasCount": 0,
        **body.model_dump(),
    }
    cfg["detection_rules"].append(rule)
    save_config(cfg)
    return rule
