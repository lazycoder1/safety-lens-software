"""SafetyLens model server.

This process owns heavyweight model runtimes. Edge backends send decoded camera
frames here and keep local streaming, alerting, config, and frontend APIs intact.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import model_manager
from logging_config import setup_logging

logger = logging.getLogger("safetylens.model_server")

app = FastAPI(title="SafetyLens Model Server")
MODEL_SERVER_TOKEN = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
# The model-server process must always use local model runtimes, even if it
# inherits edge-backend environment variables from a shared shell or container.
model_manager.MODEL_SERVER_URL = ""


class InferenceRequest(BaseModel):
    model_key: str
    frame_jpeg_b64: str
    conf: float = 0.35
    device: str = "cuda"
    imgsz: int = 960
    classes: list[str] = Field(default_factory=list)


class ModelInstallRequest(BaseModel):
    model_keys: list[str] = Field(default_factory=list)


def _require_model_server_token(authorization: str | None):
    if not MODEL_SERVER_TOKEN:
        return
    expected = f"Bearer {MODEL_SERVER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid model server token")


@app.on_event("startup")
async def startup():
    setup_logging()
    logger.info("SafetyLens model server starting")
    model_manager.initialize()


@app.get("/api/health")
async def health():
    models = model_manager.list_models()
    ready = [model for model in models if model.get("is_ready")]
    return {
        "status": "ok" if ready else "degraded",
        "role": "model_server",
        "models_ready": len(ready),
        "models_total": len(models),
    }


@app.get("/api/models")
async def list_models(authorization: str | None = Header(default=None)):
    _require_model_server_token(authorization)
    return {"models": model_manager.list_models()}


@app.post("/api/models/install")
async def install_models(body: ModelInstallRequest, authorization: str | None = Header(default=None)):
    _require_model_server_token(authorization)
    try:
        return model_manager.install_models(body.model_keys)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/models/install/{job_id}")
async def get_install_job(job_id: str, authorization: str | None = Header(default=None)):
    _require_model_server_token(authorization)
    job = model_manager.get_install_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Install job not found")
    return job


@app.post("/api/models/install/{job_id}/retry")
async def retry_install_job(job_id: str, authorization: str | None = Header(default=None)):
    _require_model_server_token(authorization)
    try:
        return model_manager.retry_install_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/infer")
async def infer(body: InferenceRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_model_server_token(authorization)
    if body.model_key not in model_manager.MODEL_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {body.model_key}")

    try:
        frame_bytes = base64.b64decode(body.frame_jpeg_b64)
        arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame_jpeg_b64") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    try:
        detections = model_manager.predict_records(
            body.model_key,  # type: ignore[arg-type]
            frame,
            conf=body.conf,
            device=body.device,
            imgsz=body.imgsz,
            classes=body.classes or None,
        )
    except Exception as exc:
        logger.exception("Inference failed", extra={"model_key": body.model_key})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"detections": detections}
