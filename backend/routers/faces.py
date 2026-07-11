"""Face enrollment and recognition log endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import audit_store
import face_analyzer
import face_store
import state
from config_manager import get_config
from dependencies import require_admin

router = APIRouter(prefix="/api", tags=["faces"])

FACE_GROUPS = [
    {"id": "employees", "label": "Employees"},
    {"id": "visitors", "label": "Visitors"},
    {"id": "contractors", "label": "Contractors"},
    {"id": "watchlist", "label": "Watchlist"},
]


class LiveEnrollRequest(BaseModel):
    cameraId: str
    name: str
    group: str = "employees"
    validUntil: str | None = None
    consentMethod: str
    consentConfirmed: bool


@router.get("/faces")
async def api_get_faces():
    return face_store.list_faces()


@router.get("/faces/groups")
async def api_get_face_groups():
    return FACE_GROUPS


@router.post("/faces/enroll")
async def api_enroll_face(
    request: Request,
    name: str = Form(...),
    group: str = Form("employees"),
    validUntil: str | None = Form(None),
    consentMethod: str = Form(...),
    consentConfirmed: bool = Form(...),
    photo: UploadFile = File(...),
    _admin=Depends(require_admin),
):
    if not consentConfirmed:
        raise HTTPException(status_code=400, detail="Consent confirmation is required")
    if group not in {item["id"] for item in FACE_GROUPS}:
        raise HTTPException(status_code=400, detail="Unsupported face group")
    image_bytes = await photo.read()
    return _create_enrollment_from_bytes(
        request,
        name=name,
        group=group,
        valid_until=validUntil,
        consent_method=consentMethod,
        image_bytes=image_bytes,
    )


@router.post("/faces/enroll/live")
async def api_enroll_face_live(request: Request, body: LiveEnrollRequest, _admin=Depends(require_admin)):
    if not body.consentConfirmed:
        raise HTTPException(status_code=400, detail="Consent confirmation is required")
    if body.group not in {item["id"] for item in FACE_GROUPS}:
        raise HTTPException(status_code=400, detail="Unsupported face group")
    cfg = get_config()
    if body.cameraId not in cfg.get("cameras", {}):
        raise HTTPException(status_code=404, detail="Camera not found")
    image_bytes = state.camera_clean_frames.get(body.cameraId) or state.camera_frames.get(body.cameraId)
    if not image_bytes:
        raise HTTPException(status_code=409, detail="No live frame is available for this camera")
    return _create_enrollment_from_bytes(
        request,
        name=body.name,
        group=body.group,
        valid_until=body.validUntil,
        consent_method=body.consentMethod,
        image_bytes=image_bytes,
    )


@router.delete("/faces/{face_id}")
async def api_delete_face(face_id: str, request: Request, _admin=Depends(require_admin)):
    face = face_store.deactivate_face(face_id)
    if not face:
        raise HTTPException(status_code=404, detail="Enrolled face not found")
    audit_store.log_event(
        "face.deactivate",
        target_type="face",
        target_id=face_id,
        details={"name": face["name"], "group": face["group"]},
        **audit_store.build_actor_context(request),
    )
    return face


@router.get("/faces/logs")
async def api_get_face_logs(cameraId: str | None = None, eventType: str | None = None, limit: int = 100, offset: int = 0):
    return face_store.query_logs(camera_id=cameraId, event_type=eventType, limit=min(limit, 500), offset=offset)


@router.get("/faces/{face_id}/photo")
async def api_get_face_photo(face_id: str):
    face = next((item for item in face_store.list_faces(include_inactive=True) if item["id"] == face_id), None)
    if not face:
        raise HTTPException(status_code=404, detail="Enrolled face not found")
    path = face_store.FACE_PHOTOS_DIR / f"{face_id}.jpg"
    return _file_or_404(path)


@router.get("/faces/logs/{log_id}/snapshot")
async def api_get_face_log_snapshot(log_id: str):
    path = face_store.FACE_SNAPSHOTS_DIR / f"{log_id}.jpg"
    return _file_or_404(path)


def _create_enrollment_from_bytes(
    request: Request,
    *,
    name: str,
    group: str,
    valid_until: str | None,
    consent_method: str,
    image_bytes: bytes,
) -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Enrollment photo is required")

    try:
        payload = face_analyzer.extract_enrollment_embedding(image_bytes)
    except face_analyzer.DuplicateFace as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except face_analyzer.FaceAnalyzerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        photo_bytes = face_analyzer.format_enrollment_photo(image_bytes, payload.bbox)
    except Exception:
        photo_bytes = image_bytes

    face = face_store.create_enrollment(
        name=clean_name,
        group=group,
        valid_until=valid_until or None,
        consent_method=consent_method,
        embedding=payload.embedding,
        photo_bytes=photo_bytes,
    )
    audit_store.log_event(
        "face.enroll",
        target_type="face",
        target_id=face["id"],
        details={"name": face["name"], "group": face["group"]},
        **audit_store.build_actor_context(request),
    )
    return face


def _file_or_404(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
