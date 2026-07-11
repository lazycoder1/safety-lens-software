"""Rakshak Lens model server.

This process owns heavyweight model runtimes. Edge backends send decoded camera
frames here and keep local streaming, alerting, config, and frontend APIs intact.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import model_manager
from logging_config import setup_logging

logger = logging.getLogger("rakshak_lens.model_server")

app = FastAPI(title="Rakshak Lens Model Server")
MODEL_SERVER_TOKEN = os.environ.get("SAFETYLENS_MODEL_SERVER_TOKEN", "")
# The model-server process must always use local model runtimes, even if it
# inherits edge-backend environment variables from a shared shell or container.
model_manager.force_local_inference()


def _runtime_device(requested: str) -> str:
    requested = (requested or "").strip().lower()
    if requested in {"cpu"}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


class InferenceRequest(BaseModel):
    model_key: str
    frame_jpeg_b64: str
    conf: float = 0.35
    device: str = "cuda"
    imgsz: int = 960
    classes: List[str] = Field(default_factory=list)


class ModelInstallRequest(BaseModel):
    model_keys: List[str] = Field(default_factory=list)


class AnprRequest(BaseModel):
    frame_jpeg_b64: str
    conf: float = 0.35
    device: str = "cuda"
    imgsz: int = 960


def _require_model_server_token(authorization: Optional[str]):
    if not MODEL_SERVER_TOKEN:
        return
    expected = f"Bearer {MODEL_SERVER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid model server token")


@app.on_event("startup")
async def startup():
    setup_logging()
    logger.info("Rakshak Lens model server starting")
    model_manager.initialize()


@app.get("/api/health")
def health():
    models = model_manager.list_models()
    ready = [model for model in models if model.get("is_ready")]
    anpr_ocr = None
    try:
        import plate_analyzer

        anpr_ocr = plate_analyzer.ocr_runtime_config()
    except Exception:
        logger.debug("ANPR OCR config unavailable", exc_info=True)
    return {
        "status": "ok" if ready else "degraded",
        "role": "model_server",
        "models_ready": len(ready),
        "models_total": len(models),
        "anpr_ocr": anpr_ocr,
    }


@app.get("/api/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    _require_model_server_token(authorization)
    return {"models": model_manager.list_models()}


@app.post("/api/models/install")
async def install_models(body: ModelInstallRequest, authorization: Optional[str] = Header(default=None)):
    _require_model_server_token(authorization)
    try:
        return model_manager.install_models(body.model_keys)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/models/install/{job_id}")
async def get_install_job(job_id: str, authorization: Optional[str] = Header(default=None)):
    _require_model_server_token(authorization)
    job = model_manager.get_install_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Install job not found")
    return job


@app.post("/api/models/install/{job_id}/retry")
async def retry_install_job(job_id: str, authorization: Optional[str] = Header(default=None)):
    _require_model_server_token(authorization)
    try:
        return model_manager.retry_install_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/infer")
def infer(body: InferenceRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
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
            device=_runtime_device(body.device),
            imgsz=body.imgsz,
            classes=body.classes or None,
        )
    except Exception as exc:
        logger.exception("Inference failed", extra={"model_key": body.model_key})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"detections": detections}


@app.post("/api/anpr")
def anpr(body: AnprRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_model_server_token(authorization)
    try:
        frame_bytes = base64.b64decode(body.frame_jpeg_b64)
        arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame_jpeg_b64") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    try:
        import plate_analyzer

        plates = plate_analyzer.analyze_frame(
            frame,
            conf=body.conf,
            device=_runtime_device(body.device),
            imgsz=body.imgsz,
        )
    except Exception as exc:
        logger.exception("ANPR inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"plates": plates}
