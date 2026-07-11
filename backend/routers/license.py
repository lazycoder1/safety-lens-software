"""
Rakshak Lens license API endpoints — read current status and accept uploads.

All endpoints require admin authentication. The license page in the admin UI
remains reachable even when the license is suspended, so a customer can
always recover by uploading a fresh .lic file.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

import audit_store
import licensing
from dependencies import require_admin

logger = logging.getLogger("rakshak_lens.routers.license")

router = APIRouter(prefix="/api/license", tags=["license"])

# Reject anything obviously larger than a real license/heartbeat — they're
# only a few hundred bytes. This guards against accidental uploads of huge
# files which would otherwise be buffered into memory.
MAX_LICENSE_BYTES = 64 * 1024


@router.get("")
async def get_license_status():
    """Return the current license status: state, license, heartbeat, reason."""
    return licensing.get_status().to_public_dict()


@router.post("/upload", dependencies=[Depends(require_admin)])
async def upload_license(request: Request, file: UploadFile = File(...)):
    """Accept a signed .lic file, verify it, and install it on the edge.

    On success the new license replaces any previously installed one and the
    license state is recomputed. On signature failure the existing license
    (if any) is left untouched.
    """
    blob = await file.read()
    if len(blob) > MAX_LICENSE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"License file is too large (max {MAX_LICENSE_BYTES} bytes)",
        )
    if not blob:
        raise HTTPException(status_code=400, detail="License file is empty")

    try:
        license = licensing.save_license(blob)
    except licensing.InvalidLicense as e:
        logger.warning("Rejected license upload: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info(
        "License uploaded by admin: %s for %s, expires %s",
        license.license_id,
        license.customer_name,
        license.expires_at.date(),
    )
    audit_store.log_event(
        "license.upload",
        target_type="license",
        target_id=license.license_id,
        details={"customerName": license.customer_name, "expiresAt": license.expires_at.isoformat()},
        **audit_store.build_actor_context(request),
    )
    return licensing.get_status().to_public_dict()


@router.post("/heartbeat", dependencies=[Depends(require_admin)])
async def upload_heartbeat(request: Request, file: UploadFile = File(...)):
    """Manually install a heartbeat token. Used by air-gapped customers who
    can't reach License Hub from the edge — the partner downloads a fresh
    heartbeat from Hub on a connected device, then uploads it here.
    """
    blob = await file.read()
    if len(blob) > MAX_LICENSE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Heartbeat file is too large (max {MAX_LICENSE_BYTES} bytes)",
        )
    if not blob:
        raise HTTPException(status_code=400, detail="Heartbeat file is empty")

    try:
        heartbeat = licensing.save_heartbeat(blob)
    except licensing.InvalidHeartbeat as e:
        logger.warning("Rejected heartbeat upload: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info(
        "Heartbeat uploaded by admin for %s, valid until %s",
        heartbeat.license_id,
        heartbeat.valid_until.date(),
    )
    audit_store.log_event(
        "license.heartbeat_upload",
        target_type="license",
        target_id=heartbeat.license_id,
        details={"validUntil": heartbeat.valid_until.isoformat()},
        **audit_store.build_actor_context(request),
    )
    return licensing.get_status().to_public_dict()
