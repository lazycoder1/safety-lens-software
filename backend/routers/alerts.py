"""
SafetyLens alert endpoints — list, stats, time-series, acknowledge, resolve, snooze, false-positive, snapshots.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Request, Query, HTTPException
from fastapi.responses import FileResponse

import alert_store
import audit_store
from dependencies import require_operator_or_admin
from video_processing import broadcast_alert

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    limit: int = Query(200, le=1000),
    offset: int = Query(0),
):
    return alert_store.get_alerts(severity=severity, status=status, camera_id=camera_id, limit=limit, offset=offset)


@router.get("/alerts/stats")
async def get_alert_stats(camera_id: Optional[str] = Query(None, alias="cameraId")):
    return alert_store.get_stats(camera_id=camera_id)


@router.get("/alerts/time-series")
async def get_alert_time_series(hours: int = Query(24), camera_id: Optional[str] = Query(None, alias="cameraId")):
    return alert_store.get_time_series(hours, camera_id=camera_id)


@router.get("/alerts/compliance")
async def get_alert_compliance(hours: int = Query(24, ge=1, le=720), camera_id: Optional[str] = Query(None, alias="cameraId")):
    return alert_store.get_compliance_metrics(window_hours=hours, camera_id=camera_id)


@router.put("/alerts/{alert_id}/acknowledge", dependencies=[Depends(require_operator_or_admin)])
async def api_acknowledge_alert(alert_id: str, request: Request):
    by = request.state.user.get("username", "Admin")
    result = alert_store.acknowledge_alert(alert_id, by=by)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    audit_store.log_event(
        "alert.acknowledge",
        target_type="alert",
        target_id=alert_id,
        details={"status": "acknowledged"},
        **audit_store.build_actor_context(request),
    )
    await broadcast_alert({"type": "updated", "data": result})
    return result


@router.put("/alerts/{alert_id}/resolve", dependencies=[Depends(require_operator_or_admin)])
async def api_resolve_alert(alert_id: str, request: Request):
    result = alert_store.resolve_alert(alert_id)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    audit_store.log_event(
        "alert.resolve",
        target_type="alert",
        target_id=alert_id,
        details={"status": "resolved"},
        **audit_store.build_actor_context(request),
    )
    await broadcast_alert({"type": "updated", "data": result})
    return result


@router.put("/alerts/{alert_id}/snooze", dependencies=[Depends(require_operator_or_admin)])
async def api_snooze_alert(alert_id: str, request: Request, minutes: int = Query(15)):
    result = alert_store.snooze_alert(alert_id, minutes)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    audit_store.log_event(
        "alert.snooze",
        target_type="alert",
        target_id=alert_id,
        details={"minutes": minutes, "status": "snoozed"},
        **audit_store.build_actor_context(request),
    )
    await broadcast_alert({"type": "updated", "data": result})
    return result


@router.put("/alerts/{alert_id}/false-positive", dependencies=[Depends(require_operator_or_admin)])
async def api_false_positive_alert(alert_id: str, request: Request):
    result = alert_store.mark_false_positive(alert_id)
    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")
    audit_store.log_event(
        "alert.false_positive",
        target_type="alert",
        target_id=alert_id,
        details={"falsePositive": True, "status": "resolved"},
        **audit_store.build_actor_context(request),
    )
    await broadcast_alert({"type": "updated", "data": result})
    return result


@router.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    filepath = alert_store.SNAPSHOTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(filepath, media_type="image/jpeg")
