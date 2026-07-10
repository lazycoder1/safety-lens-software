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
from camera_connection import (
    DEFAULT_RTSP_PORT,
    build_rtsp_url,
    normalize_camera_connection,
    normalize_stream_path,
    public_camera_connection_fields,
)
from camera_config_utils import sync_camera_rule_fields
from camera_discovery import discover_cameras, resolve_scan_networks, test_camera_connection
from camera_planner import build_execution_plan, normalize_camera_record
from camera_runtime import derive_camera_runtime_status
from config_manager import get_config, save_config
from dependencies import require_admin
from mjpeg_fanout import stream_fanout
from video_processing import restart_camera, start_camera, stop_camera

router = APIRouter(prefix="/api", tags=["cameras"])


def _camera_runtime_status(cam_id: str, camera: dict) -> str:
    required_model_keys = camera.get("execution_plan", {}).get("required_model_keys", [])
    return derive_camera_runtime_status(
        cam_id,
        camera,
        missing_models=bool(model_manager.missing_model_keys(required_model_keys)),
    )


def _camera_public_payload(cam_id: str, cam: dict, cfg: dict) -> dict[str, Any]:
    runtime_status = _camera_runtime_status(cam_id, cam)
    state.camera_runtime_status[cam_id] = runtime_status
    from routers.safety_rules import _ensure_safety_rules

    rule_map = {rule["id"]: rule for rule in _ensure_safety_rules(cfg)}
    display_rules = [
        rule_map[rule_id]["name"]
        for rule_id in cam.get("safety_rule_ids", [])
        if rule_id in rule_map
    ] or cam.get("rules", [])
    payload = {
        "id": cam_id,
        "name": cam["name"],
        "zone": cam["zone"],
        "profile": cam.get("profile", "general_safety"),
        "capabilities": cam.get("capabilities", []),
        "execution_plan": cam.get("execution_plan", build_execution_plan(cam, cfg)),
        "runtime_status": runtime_status,
        "demo": cam.get("demo", "yolo"),
        "rules": display_rules,
        "enabled": cam.get("enabled", True),
        "fps": cam.get("fps", cfg["global"]["target_fps"]),
        "video": cam.get("video", ""),
        "yoloe_classes": cam.get("yoloe_classes", []),
        "stream_type": cam.get("stream_type", "file"),
        "zones": cam.get("zones", []),
        "alert_classes": cam.get("alert_classes", []),
        "ppe_rule_ids": cam.get("ppe_rule_ids", []),
        "safety_rule_ids": cam.get("safety_rule_ids", []),
        "custom_long_tail_terms": cam.get("custom_long_tail_terms", []),
        "status": "online" if runtime_status == "running" else "offline",
        "detectionsCount": len(state.camera_detections.get(cam_id, [])),
    }
    payload.update(public_camera_connection_fields(cam))
    return payload


def _next_camera_id(cfg: dict) -> str:
    existing_ids = [int(k.replace("cam", "")) for k in cfg["cameras"] if k.startswith("cam") and k[3:].isdigit()]
    next_id = max(existing_ids, default=0) + 1
    return f"cam{next_id}"


def _prepare_camera_submission(payload: dict[str, Any], cfg: dict, *, existing_camera: dict[str, Any] | None = None) -> dict[str, Any]:
    camera_data = dict(payload)
    if camera_data.get("stream_type", existing_camera.get("stream_type") if existing_camera else "file") == "rtsp":
        normalize_camera_connection(camera_data, existing_camera=existing_camera)
        if not camera_data.get("rtsp_url"):
            camera_data["rtsp_url"] = build_rtsp_url(camera_data, include_credentials=False)
    else:
        for field in (
            "host",
            "rtsp_port",
            "onvif_port",
            "stream_path",
            "preferred_stream",
            "username",
            "password",
            "onvif_uuid",
            "discovery_fingerprint",
        ):
            camera_data.pop(field, None)

    sync_camera_rule_fields(camera_data)
    camera_data, _changed = normalize_camera_record(camera_data, cfg)
    return camera_data


def _find_duplicate_camera(
    cfg: dict,
    *,
    host: str = "",
    rtsp_port: int | None = None,
    stream_path: str = "",
    onvif_uuid: str = "",
) -> tuple[str | None, str | None]:
    normalized_path = normalize_stream_path(stream_path)
    for camera_id, camera in cfg["cameras"].items():
        public_fields = public_camera_connection_fields(camera)
        existing_host = public_fields.get("host") or ""
        existing_port = public_fields.get("rtsp_port") or DEFAULT_RTSP_PORT
        existing_path = normalize_stream_path(public_fields.get("stream_path") or "")
        existing_onvif_uuid = public_fields.get("onvif_uuid") or ""
        if onvif_uuid and existing_onvif_uuid and onvif_uuid == existing_onvif_uuid:
            return "exact", camera_id
        if host and existing_host and host == existing_host:
            if normalized_path and normalized_path == existing_path and int(rtsp_port or DEFAULT_RTSP_PORT) == int(existing_port):
                return "exact", camera_id
            return "potential", camera_id
    return None, None


def _annotate_duplicate_state(device: dict[str, Any], cfg: dict) -> dict[str, Any]:
    duplicate_state, existing_camera_id = _find_duplicate_camera(
        cfg,
        host=device.get("host") or device.get("ip") or "",
        rtsp_port=device.get("rtsp_port"),
        stream_path=device.get("stream_path", ""),
        onvif_uuid=device.get("onvif_uuid") or "",
    )
    return {
        **device,
        "duplicate_state": duplicate_state or "none",
        "existing_camera_id": existing_camera_id,
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
    host: str = ""
    rtsp_port: Optional[int] = None
    onvif_port: Optional[int] = None
    stream_path: str = ""
    preferred_stream: str = ""
    username: str = ""
    password: str = ""
    onvif_uuid: str = ""
    discovery_fingerprint: str = ""
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
    host: str = ""
    rtsp_port: Optional[int] = None
    onvif_port: Optional[int] = None
    stream_path: str = ""
    preferred_stream: str = ""
    username: str = ""
    password: str = ""
    onvif_uuid: str = ""
    discovery_fingerprint: str = ""
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
    host: Optional[str] = None
    rtsp_port: Optional[int] = None
    onvif_port: Optional[int] = None
    stream_path: Optional[str] = None
    preferred_stream: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    onvif_uuid: Optional[str] = None
    discovery_fingerprint: Optional[str] = None
    alert_classes: Optional[list[str]] = None
    ppe_rule_ids: Optional[list[str]] = None
    safety_rule_ids: Optional[list[str]] = None
    custom_long_tail_terms: Optional[list[str]] = None


class DiscoveryScanRequest(BaseModel):
    cidrs: list[str] = Field(default_factory=list)
    timeout_seconds: float = 4.0


class DiscoveryTestRequest(BaseModel):
    fingerprint: str = ""
    host: str = ""
    ip: str = ""
    name: str = ""
    vendor: str = ""
    model: str = ""
    onvif_uuid: str = ""
    onvif_xaddr: str = ""
    onvif_port: Optional[int] = None
    rtsp_port: Optional[int] = None
    preferred_stream: str = "main"
    stream_path: str = ""
    username: str = ""
    password: str = ""


class DiscoveryImportCamera(BaseModel):
    fingerprint: str = ""
    host: str = ""
    ip: str = ""
    name: str = ""
    zone: str = ""
    profile: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    preferred_stream: str = "main"
    stream_path: str = ""
    username: str = ""
    password: str = ""
    onvif_uuid: str = ""
    onvif_xaddr: str = ""
    onvif_port: Optional[int] = None
    rtsp_port: Optional[int] = None


class DiscoveryImportRequest(BaseModel):
    devices: list[DiscoveryImportCamera] = Field(default_factory=list)


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


@router.post("/cameras/discover", dependencies=[Depends(require_admin)])
async def api_discover_cameras(body: DiscoveryScanRequest):
    cfg = get_config()
    discovery = discover_cameras(body.cidrs or None, timeout_seconds=body.timeout_seconds)
    devices = [_annotate_duplicate_state(device, cfg) for device in discovery["devices"]]
    resolved_cidrs, cidr_warnings = resolve_scan_networks(body.cidrs or None)
    return {
        "cidrs": discovery.get("cidrs") or resolved_cidrs,
        "warnings": [*discovery.get("warnings", []), *cidr_warnings],
        "devices": devices,
    }


@router.post("/cameras/discover/test", dependencies=[Depends(require_admin)])
async def api_test_discovered_camera(body: DiscoveryTestRequest):
    cfg = get_config()
    result = test_camera_connection(body.model_dump())
    return _annotate_duplicate_state(result, cfg)


@router.post("/cameras/discover/import", dependencies=[Depends(require_admin)])
async def api_import_discovered_cameras(body: DiscoveryImportRequest, request: Request):
    cfg = get_config()
    created: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for row in body.devices:
        row_payload = row.model_dump()
        if not row_payload.get("name", "").strip():
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": "missing_name",
                "error": "Display name is required.",
            })
            continue
        if not row_payload.get("zone", "").strip():
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": "missing_zone",
                "error": "Zone / location is required.",
            })
            continue

        test_result = test_camera_connection(row_payload)
        if not test_result.get("ok"):
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": test_result.get("error_code", "test_failed"),
                "error": test_result.get("error", "Camera validation failed."),
            })
            continue

        duplicate_state, existing_camera_id = _find_duplicate_camera(
            cfg,
            host=test_result.get("host", ""),
            rtsp_port=test_result.get("rtsp_port"),
            stream_path=test_result.get("stream_path", ""),
            onvif_uuid=test_result.get("onvif_uuid", "") or "",
        )
        if duplicate_state == "exact":
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": "duplicate_camera",
                "error": "This camera is already configured.",
                "existing_camera_id": existing_camera_id,
            })
            continue

        allowed, reason = licensing.can_add_camera(len(cfg["cameras"]) + len(created))
        if not allowed:
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": "license_limit",
                "error": reason,
            })
            continue

        cam_id = _next_camera_id(cfg)
        cam_data = _prepare_camera_submission(
            {
                "name": row_payload["name"].strip(),
                "zone": row_payload["zone"].strip(),
                "profile": row_payload.get("profile"),
                "capabilities": row_payload.get("capabilities", []),
                "demo": "yolo",
                "rules": [],
                "enabled": True,
                "fps": cfg["global"]["target_fps"],
                "stream_type": "rtsp",
                "rtsp_url": test_result.get("rtsp_url", ""),
                "host": test_result.get("host", ""),
                "rtsp_port": test_result.get("rtsp_port"),
                "onvif_port": test_result.get("onvif_port"),
                "stream_path": test_result.get("stream_path", ""),
                "preferred_stream": test_result.get("preferred_stream") or row_payload.get("preferred_stream", "main"),
                "username": row_payload.get("username", ""),
                "password": row_payload.get("password", ""),
                "onvif_uuid": test_result.get("onvif_uuid", "") or row_payload.get("onvif_uuid", ""),
                "discovery_fingerprint": test_result.get("discovery_fingerprint", "") or row_payload.get("fingerprint", ""),
            },
            cfg,
        )
        execution_plan = cam_data["execution_plan"]
        missing_model_keys = model_manager.missing_model_keys(execution_plan["required_model_keys"])
        if missing_model_keys:
            failed.append({
                "fingerprint": row_payload.get("fingerprint"),
                "error_code": "missing_models",
                "error": "Required models must be installed before this camera can run.",
                "missing_model_keys": missing_model_keys,
            })
            continue

        cfg["cameras"][cam_id] = cam_data
        created.append({
            "camera_id": cam_id,
            "name": cam_data["name"],
            "zone": cam_data["zone"],
            "needs_zone_setup": bool(cam_data.get("execution_plan", {}).get("zones_required")),
        })

    if created:
        save_config(cfg)
        for item in created:
            camera = cfg["cameras"][item["camera_id"]]
            start_camera(item["camera_id"])
            audit_store.log_event(
                "camera.discover_import",
                target_type="camera",
                target_id=item["camera_id"],
                details={
                    "name": item["name"],
                    "zone": item["zone"],
                    "connection": public_camera_connection_fields(camera),
                },
                **audit_store.build_actor_context(request),
            )

    return {
        "created": created,
        "failed": failed,
        "needs_zone_setup": [item for item in created if item["needs_zone_setup"]],
    }


@router.post("/camera-plans/preview", dependencies=[Depends(require_admin)])
async def api_preview_camera_plan(body: CameraPlanPreviewRequest):
    cfg = get_config()
    draft = body.model_dump()
    draft = _prepare_camera_submission(draft, cfg)
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

    cam_id = _next_camera_id(cfg)
    cam_data = _prepare_camera_submission(body.model_dump(), cfg)
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
            "connection": public_camera_connection_fields(cam_data),
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
    current_camera = _prepare_camera_submission(current_camera, cfg, existing_camera=cfg["cameras"][cam_id])
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
        details={
            "updates": updates,
            "profile": current_camera.get("profile"),
            "capabilities": current_camera.get("capabilities", []),
            "connection": public_camera_connection_fields(current_camera),
        },
        **audit_store.build_actor_context(request),
    )
    return _camera_public_payload(cam_id, current_camera, cfg)


@router.delete("/cameras/{cam_id}", dependencies=[Depends(require_admin)])
async def api_delete_camera(cam_id: str, request: Request):
    cfg = get_config()
    if cam_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not stop_camera(cam_id):
        raise HTTPException(
            status_code=409,
            detail="Camera worker is still stopping; retry deletion",
        )
    del cfg["cameras"][cam_id]
    save_config(cfg)
    stream_fanout.retire(cam_id)
    audit_store.log_event(
        "camera.delete",
        target_type="camera",
        target_id=cam_id,
        **audit_store.build_actor_context(request),
    )
    return {"deleted": cam_id}
