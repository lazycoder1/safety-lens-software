"""
SafetyLens camera CRUD endpoints.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config_manager import get_config, save_config
from dependencies import require_admin
from routers.safety_rules import derive_yoloe_classes
from video_processing import start_camera, stop_camera, restart_camera
import state

router = APIRouter(prefix="/api", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    video: str = ""
    zone: str
    demo: str = "yolo"
    rules: list[str] = []
    enabled: bool = True
    fps: int = 6
    yoloe_classes: list[str] = ["person"]
    stream_type: str = "file"  # "file" | "rtsp"
    rtsp_url: str = ""
    alert_classes: list[str] = ["mobile_phone", "animal_intrusion"]
    ppe_rule_ids: list[str] = []
    safety_rule_ids: list[str] = []


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    video: Optional[str] = None
    zone: Optional[str] = None
    demo: Optional[str] = None
    rules: Optional[list[str]] = None
    enabled: Optional[bool] = None
    fps: Optional[int] = None
    yoloe_classes: Optional[list[str]] = None
    stream_type: Optional[str] = None
    rtsp_url: Optional[str] = None
    alert_classes: Optional[list[str]] = None
    ppe_rule_ids: Optional[list[str]] = None
    safety_rule_ids: Optional[list[str]] = None


@router.get("/cameras")
async def get_cameras():
    cfg = get_config()
    result = []
    for cam_id, cam in cfg["cameras"].items():
        result.append({
            "id": cam_id,
            "name": cam["name"],
            "zone": cam["zone"],
            "demo": cam["demo"],
            "rules": cam["rules"],
            "enabled": cam.get("enabled", True),
            "fps": cam.get("fps", cfg["global"]["target_fps"]),
            "video": cam.get("video", ""),
            "yoloe_classes": cam.get("yoloe_classes", ["person"]),
            "stream_type": cam.get("stream_type", "file"),
            "rtsp_url": cam.get("rtsp_url", ""),
            "alert_classes": cam.get("alert_classes", ["mobile_phone", "animal_intrusion"]),
            "ppe_rule_ids": cam.get("ppe_rule_ids", []),
            "safety_rule_ids": cam.get("safety_rule_ids", []),
            "status": "online" if cam_id in state.camera_threads else "offline",
            "detectionsCount": len(state.camera_detections.get(cam_id, [])),
        })
    return result


@router.post("/cameras", dependencies=[Depends(require_admin)])
async def api_add_camera(body: CameraCreate):
    cfg = get_config()
    # Auto-generate next cam ID
    existing_ids = [int(k.replace("cam", "")) for k in cfg["cameras"] if k.startswith("cam") and k[3:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    cam_id = f"cam{next_id}"

    cam_data = body.model_dump()
    rule_ids = cam_data.get("safety_rule_ids") or cam_data.get("ppe_rule_ids", [])
    if rule_ids and cam_data.get("demo") == "yoloe":
        cam_data["yoloe_classes"] = derive_yoloe_classes(rule_ids, cfg)
    cfg["cameras"][cam_id] = cam_data
    save_config(cfg)
    start_camera(cam_id)
    return {"id": cam_id, **cam_data}


@router.put("/cameras/{cam_id}", dependencies=[Depends(require_admin)])
async def api_update_camera(cam_id: str, body: CameraUpdate):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    updates = body.model_dump(exclude_none=True)
    cfg["cameras"][cam_id].update(updates)
    cam = cfg["cameras"][cam_id]
    rule_ids = cam.get("safety_rule_ids") or cam.get("ppe_rule_ids", [])
    if rule_ids and cam.get("demo") == "yoloe":
        cam["yoloe_classes"] = derive_yoloe_classes(rule_ids, cfg)
    save_config(cfg)
    restart_camera(cam_id)
    return {"id": cam_id, **cfg["cameras"][cam_id]}


@router.delete("/cameras/{cam_id}", dependencies=[Depends(require_admin)])
async def api_delete_camera(cam_id: str):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    stop_camera(cam_id)
    del cfg["cameras"][cam_id]
    save_config(cfg)
    return {"deleted": cam_id}
