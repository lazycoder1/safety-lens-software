"""SafetyLens camera CRUD endpoints."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import audit_store
import licensing
import model_manager
import state
from camera_config_utils import sync_camera_rule_fields
from camera_planner import build_execution_plan, normalize_camera_record
from config_manager import get_config, save_config
from dependencies import require_admin
from video_processing import restart_camera, start_camera, stop_camera

router = APIRouter(prefix="/api", tags=["cameras"])


def _camera_runtime_status(cam_id: str, camera: dict) -> str:
    required_model_keys = camera.get("execution_plan", {}).get("required_model_keys", [])
    if model_manager.missing_model_keys(required_model_keys):
        return "awaiting_model_install"
    if not camera.get("enabled", True):
        return "offline"
    if cam_id in state.camera_threads:
        return "running" if state.camera_frames.get(cam_id) is not None else "starting"
    return "offline"


def _camera_public_payload(cam_id: str, cam: dict, cfg: dict) -> dict[str, Any]:
    runtime_status = _camera_runtime_status(cam_id, cam)
    state.camera_runtime_status[cam_id] = runtime_status
    return {
        "id": cam_id,
        "name": cam["name"],
        "zone": cam["zone"],
        "profile": cam.get("profile", "general_safety"),
        "capabilities": cam.get("capabilities", []),
        "execution_plan": cam.get("execution_plan", build_execution_plan(cam, cfg)),
        "runtime_status": runtime_status,
        "demo": cam.get("demo", "yolo"),
        "rules": cam.get("rules", []),
        "enabled": cam.get("enabled", True),
        "fps": cam.get("fps", cfg["global"]["target_fps"]),
        "video": cam.get("video", ""),
        "yoloe_classes": cam.get("yoloe_classes", []),
        "stream_type": cam.get("stream_type", "file"),
        "rtsp_url": cam.get("rtsp_url", ""),
        "alert_classes": cam.get("alert_classes", []),
        "ppe_rule_ids": cam.get("ppe_rule_ids", []),
        "safety_rule_ids": cam.get("safety_rule_ids", []),
        "custom_long_tail_terms": cam.get("custom_long_tail_terms", []),
        "status": "online" if cam_id in state.camera_threads else "offline",
        "detectionsCount": len(state.camera_detections.get(cam_id, [])),
    }


def _missing_models_response(execution_plan: dict, missing_model_keys: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "message": "Required models must be downloaded and set up before this camera can run.",
            "code": "missing_models",
            "missing_model_keys": missing_model_keys,
            "execution_plan": execution_plan,
        },
    )


class CameraPlanPreviewRequest(BaseModel):
    name: str = ""
    zone: str = ""
    profile: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    stream_type: str = "file"
    video: str = ""
    rtsp_url: str = ""
    safety_rule_ids: list[str] = Field(default_factory=list)
    yoloe_classes: list[str] = Field(default_factory=list)
    custom_long_tail_terms: list[str] = Field(default_factory=list)


class CameraCreate(BaseModel):
    name: str
    video: str = ""
    zone: str
    profile: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    demo: str = "yolo"
    rules: list[str] = Field(default_factory=list)
    enabled: bool = True
    fps: int = 6
    yoloe_classes: list[str] = Field(default_factory=list)
    stream_type: str = "file"
    rtsp_url: str = ""
    alert_classes: list[str] = Field(default_factory=list)
    ppe_rule_ids: list[str] = Field(default_factory=list)
    safety_rule_ids: list[str] = Field(default_factory=list)
    custom_long_tail_terms: list[str] = Field(default_factory=list)


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    video: Optional[str] = None
    zone: Optional[str] = None
    profile: Optional[str] = None
    capabilities: Optional[list[str]] = None
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
    custom_long_tail_terms: Optional[list[str]] = None


@router.get("/cameras")
async def get_cameras():
    cfg = get_config()
    result = []
    changed_any = False
    for cam_id, cam in cfg["cameras"].items():
        normalized_camera, changed = normalize_camera_record(cam, cfg)
        if changed:
            cfg["cameras"][cam_id] = normalized_camera
            cam = normalized_camera
            changed_any = True
        result.append(_camera_public_payload(cam_id, cam, cfg))
    if changed_any:
        save_config(cfg)
    return result


@router.post("/camera-plans/preview", dependencies=[Depends(require_admin)])
async def api_preview_camera_plan(body: CameraPlanPreviewRequest):
    cfg = get_config()
    draft = body.model_dump()
    sync_camera_rule_fields(draft)
    draft, _changed = normalize_camera_record(draft, cfg)
    execution_plan = draft["execution_plan"]
    missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
    return {
        "profile": draft.get("profile"),
        "capabilities": draft.get("capabilities", []),
        "execution_plan": execution_plan,
        "missing_model_keys": missing_model_keys,
    }


@router.post("/cameras", dependencies=[Depends(require_admin)])
async def api_add_camera(body: CameraCreate, request: Request):
    cfg = get_config()

    allowed, reason = licensing.can_add_camera(len(cfg["cameras"]))
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    existing_ids = [int(k.replace("cam", "")) for k in cfg["cameras"] if k.startswith("cam") and k[3:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    cam_id = f"cam{next_id}"

    cam_data = body.model_dump()
    sync_camera_rule_fields(cam_data)
    cam_data, _changed = normalize_camera_record(cam_data, cfg)
    execution_plan = cam_data["execution_plan"]
    missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
    if missing_model_keys:
        return _missing_models_response(execution_plan, missing_model_keys)

    cfg["cameras"][cam_id] = cam_data
    save_config(cfg)
    start_camera(cam_id)
    audit_store.log_event(
        "camera.create",
        target_type="camera",
        target_id=cam_id,
        details={
            "name": cam_data.get("name"),
            "zone": cam_data.get("zone"),
            "stream_type": cam_data.get("stream_type", "file"),
            "profile": cam_data.get("profile"),
            "capabilities": cam_data.get("capabilities", []),
        },
        **audit_store.build_actor_context(request),
    )
    return _camera_public_payload(cam_id, cam_data, cfg)


@router.put("/cameras/{cam_id}", dependencies=[Depends(require_admin)])
async def api_update_camera(cam_id: str, body: CameraUpdate, request: Request):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    current_camera = dict(cfg["cameras"][cam_id])
    updates = body.model_dump(exclude_none=True)
    current_camera.update(updates)
    sync_camera_rule_fields(current_camera)
    current_camera, _changed = normalize_camera_record(current_camera, cfg)
    execution_plan = current_camera["execution_plan"]
    missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
    if missing_model_keys:
        return _missing_models_response(execution_plan, missing_model_keys)

    cfg["cameras"][cam_id] = current_camera
    save_config(cfg)
    restart_camera(cam_id)
    audit_store.log_event(
        "camera.update",
        target_type="camera",
        target_id=cam_id,
        details={"updates": updates, "profile": current_camera.get("profile"), "capabilities": current_camera.get("capabilities", [])},
        **audit_store.build_actor_context(request),
    )
    return _camera_public_payload(cam_id, current_camera, cfg)


@router.delete("/cameras/{cam_id}", dependencies=[Depends(require_admin)])
async def api_delete_camera(cam_id: str, request: Request):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    stop_camera(cam_id)
    del cfg["cameras"][cam_id]
    save_config(cfg)
    audit_store.log_event(
        "camera.delete",
        target_type="camera",
        target_id=cam_id,
        **audit_store.build_actor_context(request),
    )
    return {"deleted": cam_id}
