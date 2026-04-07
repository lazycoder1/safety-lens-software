"""
SafetyLens streaming endpoints — MJPEG stream, WebSocket alerts, VLM latest.
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse

from config_manager import get_config
from video_processing import mjpeg_generator, broadcast_alert
import state
import alert_store
import auth_store

logger = logging.getLogger("safetylens")

router = APIRouter(tags=["stream"])


@router.get("/api/stream/{camera_id}")
async def stream(camera_id: str):
    cfg = get_config()
    if camera_id not in cfg["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/vlm/latest")
async def get_vlm_latest():
    with state.vlm_lock:
        return state.vlm_last_results


@router.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    token = ws.query_params.get("token")
    if not token:
        await ws.close(code=4001, reason="Missing token")
        return
    try:
        auth_store.decode_token(token)
    except Exception:
        await ws.close(code=4001, reason="Invalid or expired token")
        return
    await ws.accept()
    state.alert_subscribers.append(ws)
    logger.info("WebSocket connected", extra={"subscribers": len(state.alert_subscribers)})
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "acknowledge":
                alert_id = msg.get("alertId")
                result = alert_store.acknowledge_alert(alert_id)
                if result:
                    await broadcast_alert({"type": "updated", "data": result})
    except WebSocketDisconnect:
        state.alert_subscribers.remove(ws)
        logger.info("WebSocket disconnected", extra={"subscribers": len(state.alert_subscribers)})
