"""
Rakshak Lens alert endpoints — list, stats, time-series, acknowledge, resolve, snooze, false-positive, snapshots.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import alert_delivery_store
import alert_delivery_worker
import alert_store
import audit_store
from dependencies import require_admin, require_operator_or_admin
from video_processing import broadcast_alert

router = APIRouter(prefix="/api", tags=["alerts"])


class DeliveryReplayRequest(BaseModel):
    allow_ambiguous: bool = Field(False, alias="allowAmbiguous")


def _public_delivery(row: dict) -> dict:
    """Project operational state without destination or provider secrets."""
    def timestamp(name: str):
        value = row.get(name)
        return value.isoformat() if isinstance(value, datetime) else value

    return {
        "deliveryId": str(row.get("id")),
        "alertId": str(row.get("alert_id")),
        "kind": row.get("kind"),
        "channel": row.get("channel"),
        "priority": row.get("priority"),
        "state": row.get("state"),
        "claimCount": row.get("claim_count"),
        "internalFailureCount": row.get("internal_failure_count"),
        "attemptCount": row.get("attempt_count"),
        "eligibleAt": timestamp("eligible_at"),
        "expiresAt": timestamp("expires_at"),
        "nextAttemptAt": timestamp("next_attempt_at"),
        "lastAttemptAt": timestamp("last_attempt_at"),
        "lastErrorCode": row.get("last_error_code"),
        "lastAcceptanceUnknown": bool(row.get("last_acceptance_unknown")),
        "everAcceptanceUnknown": bool(row.get("ever_acceptance_unknown")),
        "terminalReason": row.get("terminal_reason"),
        "deliveredAt": timestamp("delivered_at"),
        "terminalAt": timestamp("terminal_at"),
        "createdAt": timestamp("created_at"),
        "updatedAt": timestamp("updated_at"),
    }


@router.get("/alerts/detection-classes")
async def get_detection_classes():
    return alert_store.get_detection_classes()


@router.get("/alerts/search")
async def search_alerts(
    q: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    severity: Optional[str] = Query(None),
    detection_class: Optional[str] = Query(None, alias="detectionClass"),
    time_range: Optional[str] = Query(None, alias="timeRange"),
    sort: str = Query("relevance"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    return alert_store.search_alerts(
        query=q, camera_id=camera_id, severity=severity,
        detection_class=detection_class, time_range=time_range,
        sort=sort, limit=limit, offset=offset,
    )


@router.get("/alerts/heatmap/zone-time")
async def get_zone_time_heatmap(
    hours: int = Query(24),
    camera_id: Optional[str] = Query(None, alias="cameraId"),
    severity: Optional[str] = Query(None),
    bucket: str = Query("hour"),
):
    return alert_store.get_zone_time_heatmap(hours=hours, camera_id=camera_id, severity=severity, bucket=bucket)


@router.get("/alerts/heatmap/spatial")
async def get_spatial_heatmap(
    camera_id: str = Query(..., alias="cameraId"),
    hours: int = Query(24),
    severity: Optional[str] = Query(None),
    grid_size: int = Query(10, alias="gridSize"),
):
    return alert_store.get_spatial_heatmap(camera_id=camera_id, hours=hours, severity=severity, grid_size=grid_size)


@router.get("/alerts/{alert_id}/similar")
async def get_similar_alerts(alert_id: str, limit: int = Query(20, le=50)):
    return alert_store.find_similar_alerts(alert_id, limit=limit)


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


@router.get("/alert-deliveries", dependencies=[Depends(require_admin)])
async def api_list_alert_deliveries(
    state: Optional[str] = Query(None),
    alert_id: Optional[str] = Query(None, alias="alertId"),
    limit: int = Query(100, ge=1, le=500),
):
    """Expose replayable IDs and safe delivery state to administrators."""
    try:
        rows = alert_delivery_store.list_deliveries(
            state=state,
            alert_id=alert_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_public_delivery(row) for row in rows]


@router.post(
    "/alert-deliveries/{delivery_id}/replay",
    dependencies=[Depends(require_admin)],
)
async def api_replay_terminal_delivery(
    delivery_id: UUID,
    body: DeliveryReplayRequest,
    request: Request,
):
    """Explicitly replay recent terminal work after an operator fixes config."""
    delivery_id = str(delivery_id)
    existing = alert_delivery_store.get_delivery(delivery_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Alert delivery not found")
    if existing.get("state") != "terminal":
        raise HTTPException(status_code=409, detail="Only terminal delivery work can be replayed")
    # Record the authorized request before making work eligible. If audit
    # persistence fails, the delivery remains terminal and cannot be sent.
    audit_store.log_event(
        "alert.delivery_replay",
        target_type="alert_delivery",
        target_id=delivery_id,
        details={
            "channel": existing.get("channel"),
            "alertId": existing.get("alert_id"),
            "allowAmbiguous": body.allow_ambiguous,
            "everAcceptanceUnknown": bool(existing.get("ever_acceptance_unknown")),
            "outcome": "requested",
        },
        **audit_store.build_actor_context(request),
    )
    replayed = alert_delivery_store.requeue_terminal(
        delivery_id,
        allow_ambiguous=body.allow_ambiguous,
    )
    if not replayed:
        raise HTTPException(
            status_code=409,
            detail="Delivery is too old or may already have been accepted; explicit confirmation is required",
        )
    alert_delivery_worker.wake()
    return {"ok": True, "deliveryId": delivery_id, "state": "pending"}


@router.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    filepath = alert_store.SNAPSHOTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(filepath, media_type="image/jpeg")
