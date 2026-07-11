"""ANPR plate list and read-log endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import audit_store
import plate_store
from dependencies import require_admin

router = APIRouter(prefix="/api", tags=["plates"])


class PlateEntryRequest(BaseModel):
    plateNumber: str
    list: str
    owner: str = ""
    vehicle: str = ""
    validFrom: str | None = None
    validUntil: str | None = None


@router.get("/plates/reads")
async def api_get_plate_reads(
    plate: str | None = None,
    cameraId: str | None = None,
    eventType: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    return plate_store.query_reads(
        plate=plate,
        camera_id=cameraId,
        event_type=eventType,
        limit=min(limit, 500),
        offset=offset,
    )


@router.get("/plates/lists")
async def api_get_plate_lists(list: str | None = None):
    if list and list not in plate_store.LIST_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported plate list type")
    return plate_store.list_plate_entries(list_type=list)


@router.post("/plates/lists")
async def api_create_plate_entry(body: PlateEntryRequest, request: Request, _admin=Depends(require_admin)):
    try:
        entry = plate_store.create_plate_entry(
            plate_text=body.plateNumber,
            list_type=body.list,
            owner_name=body.owner,
            vehicle_desc=body.vehicle,
            valid_from=body.validFrom,
            valid_until=body.validUntil,
        )
    except ValueError as exc:
        status_code = 409 if "already exists" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    _audit(request, "plate.list.create", entry)
    return entry


@router.put("/plates/lists/{entry_id}")
async def api_update_plate_entry(entry_id: str, body: PlateEntryRequest, request: Request, _admin=Depends(require_admin)):
    try:
        entry = plate_store.update_plate_entry(
            entry_id,
            plate_text=body.plateNumber,
            list_type=body.list,
            owner_name=body.owner,
            vehicle_desc=body.vehicle,
            valid_from=body.validFrom,
            valid_until=body.validUntil,
        )
    except ValueError as exc:
        status_code = 409 if "already exists" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    if not entry:
        raise HTTPException(status_code=404, detail="Plate list entry not found")
    _audit(request, "plate.list.update", entry)
    return entry


@router.delete("/plates/lists/{entry_id}")
async def api_delete_plate_entry(entry_id: str, request: Request, _admin=Depends(require_admin)):
    entry = plate_store.deactivate_plate_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Plate list entry not found")
    _audit(request, "plate.list.delete", entry)
    return entry


@router.post("/plates/lists/import")
async def api_import_plate_entries(request: Request, file: UploadFile = File(...), _admin=Depends(require_admin)):
    blob = await file.read()
    result = plate_store.import_plate_csv(blob)
    audit_store.log_event(
        "plate.list.import",
        target_type="plate_list",
        target_id="bulk",
        details={"created": len(result["created"]), "failed": len(result["failed"])},
        **audit_store.build_actor_context(request),
    )
    return result


@router.get("/plates/search")
async def api_search_plates(q: str, limit: int = 25):
    return plate_store.search_plates(q, limit=min(limit, 100))


@router.get("/plates/reads/{read_id}/snapshot")
async def api_get_plate_read_snapshot(read_id: str):
    return _file_or_404(plate_store.PLATE_SNAPSHOTS_DIR / f"{read_id}.jpg")


@router.get("/plates/reads/{read_id}/crop")
async def api_get_plate_read_crop(read_id: str):
    return _file_or_404(plate_store.PLATE_CROPS_DIR / f"{read_id}.jpg")


def _audit(request: Request, action: str, entry: dict):
    audit_store.log_event(
        action,
        target_type="plate_list",
        target_id=entry["id"],
        details={"plateNumber": entry["plateNumber"], "list": entry["list"]},
        **audit_store.build_actor_context(request),
    )


def _file_or_404(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
