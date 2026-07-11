"""
Rakshak Lens streaming endpoints — MJPEG stream, WebSocket alerts, VLM latest,
RTSP connection testing.
"""

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config_manager import get_config
from dependencies import require_admin
from video_processing import mjpeg_generator, broadcast_alert
import state
import alert_store
import auth_store

logger = logging.getLogger("rakshak_lens")

router = APIRouter(tags=["stream"])

_rtsp_pool = ThreadPoolExecutor(max_workers=2)


class RtspTestRequest(BaseModel):
    rtsp_url: str


def _test_rtsp_sync(url: str) -> dict:
    """Open an RTSP URL, grab one frame, return result dict.  Runs in a thread."""
    cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return {"success": False, "resolution": None, "snapshot_b64": None,
                    "error": "Cannot open RTSP URL — check address/credentials"}
        ret, frame = cap.read()
        if not ret or frame is None:
            return {"success": False, "resolution": None, "snapshot_b64": None,
                    "error": "Connected but failed to read a frame"}
        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode()
        return {"success": True, "resolution": [w, h], "snapshot_b64": b64, "error": None}
    finally:
        cap.release()


@router.post("/api/rtsp/test", dependencies=[Depends(require_admin)])
async def test_rtsp_connection(body: RtspTestRequest):
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_rtsp_pool, _test_rtsp_sync, body.rtsp_url),
            timeout=15,
        )
    except asyncio.TimeoutError:
        result = {"success": False, "resolution": None, "snapshot_b64": None,
                  "error": "Connection timed out after 15 seconds"}
    return result


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
