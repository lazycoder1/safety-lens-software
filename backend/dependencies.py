"""
SafetyLens FastAPI dependency functions.
"""

from fastapi import Request, HTTPException

from config_manager import get_config


def get_current_user(request: Request) -> dict:
    return request.state.user


def require_admin(request: Request):
    if request.state.user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def require_operator_or_admin(request: Request):
    if request.state.user["role"] == "viewer":
        raise HTTPException(status_code=403, detail="Insufficient permissions")


def get_camera_or_404(cam_id: str) -> dict:
    cfg = get_config()
    cam = cfg["cameras"].get(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam
