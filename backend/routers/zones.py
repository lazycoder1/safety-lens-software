"""
Rakshak Lens zone CRUD endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from camera_config_utils import sync_camera_rule_fields
from config_manager import get_config, save_config

logger = logging.getLogger("rakshak_lens")

router = APIRouter(prefix="/api", tags=["zones"])


class ZoneCreate(BaseModel):
    name: str
    type: str = "restricted"
    color: str = "#dc2626"
    points: list[list[float]]  # [[x,y], ...] normalized 0-1
    analytics: Optional[str] = None
    classes: list[str] = Field(default_factory=list)
    threshold: Optional[float] = None
    min_duration_seconds: Optional[float] = None
    area_square_meters: Optional[float] = None
    severity_thresholds: dict[str, float] = Field(default_factory=dict)


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    color: Optional[str] = None
    points: Optional[list[list[float]]] = None
    analytics: Optional[str] = None
    classes: Optional[list[str]] = None
    threshold: Optional[float] = None
    min_duration_seconds: Optional[float] = None
    area_square_meters: Optional[float] = None
    severity_thresholds: Optional[dict[str, float]] = None


@router.get("/cameras/{cam_id}/zones")
async def api_get_zones(cam_id: str):
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam.get("zones", [])


@router.post("/cameras/{cam_id}/zones")
async def api_add_zone(cam_id: str, body: ZoneCreate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    if "zones" not in cfg["cameras"][cam_id]:
        cfg["cameras"][cam_id]["zones"] = []

    zone_id = f"z{len(cfg['cameras'][cam_id]['zones']) + 1}"
    zone = {"id": zone_id, **body.model_dump()}
    cfg["cameras"][cam_id]["zones"].append(zone)

    # Intrusion-style zones should enable intrusion alerts. Workstation zones
    # are used by office occupancy cameras and should not alert on normal work.
    if body.type in {"restricted", "caution"}:
        alert_classes = cfg["cameras"][cam_id].get("alert_classes", [])
        if "zone_intrusion" not in alert_classes:
            alert_classes.append("zone_intrusion")
            cfg["cameras"][cam_id]["alert_classes"] = alert_classes
        safety_rule_ids = cfg["cameras"][cam_id].get("safety_rule_ids", [])
        if "alert_zone_intrusion" not in safety_rule_ids:
            safety_rule_ids.append("alert_zone_intrusion")
            cfg["cameras"][cam_id]["safety_rule_ids"] = safety_rule_ids
        sync_camera_rule_fields(cfg["cameras"][cam_id])

    save_config(cfg)
    logger.info("Zone created", extra={"camera_id": cam_id, "zone": zone_id})
    return zone


@router.put("/cameras/{cam_id}/zones/{zone_id}")
async def api_update_zone(cam_id: str, zone_id: str, body: ZoneUpdate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    zones = cfg["cameras"][cam_id].get("zones", [])
    for z in zones:
        if z["id"] == zone_id:
            updates = body.model_dump(exclude_none=True)
            z.update(updates)
            save_config(cfg)
            return z

    raise HTTPException(status_code=404, detail="Zone not found")


@router.delete("/cameras/{cam_id}/zones/{zone_id}")
async def api_delete_zone(cam_id: str, zone_id: str):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    zones = cfg["cameras"][cam_id].get("zones", [])
    cfg["cameras"][cam_id]["zones"] = [z for z in zones if z["id"] != zone_id]
    save_config(cfg)
    logger.info("Zone deleted", extra={"camera_id": cam_id, "zone": zone_id})
    return {"deleted": zone_id}
