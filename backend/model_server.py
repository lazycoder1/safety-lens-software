"""Rakshak Lens model server.

This process owns heavyweight model runtimes. Edge backends send decoded camera
frames here and keep local streaming, alerting, config, and frontend APIs intact.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

import cv2
import numpy as np
from fastapi import Body, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

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


class InferenceBatchItem(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    model_key: str
    conf: float = Field(default=0.35, ge=0.0, le=1.0)
    device: str = "cuda"
    imgsz: int = Field(default=960, gt=0)
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


def _decode_frame(frame_bytes: bytes):
    try:
        arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JPEG frame") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")
    return frame


def _run_inference_frame(
    *,
    model_key: str,
    frame,
    conf: float,
    device: str,
    imgsz: int,
    classes: List[str],
) -> Dict[str, Any]:
    if model_key not in model_manager.MODEL_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model_key}")

    try:
        detections = model_manager.predict_records(
            model_key,  # type: ignore[arg-type]
            frame,
            conf=conf,
            device=device,
            imgsz=imgsz,
            classes=classes or None,
        )
    except Exception as exc:
        logger.exception("Inference failed", extra={"model_key": model_key})
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"detections": detections}


def _run_inference(
    *,
    model_key: str,
    frame_bytes: bytes,
    conf: float,
    device: str,
    imgsz: int,
    classes: List[str],
) -> Dict[str, Any]:
    return _run_inference_frame(
        model_key=model_key,
        frame=_decode_frame(frame_bytes),
        conf=conf,
        device=device,
        imgsz=imgsz,
        classes=classes,
    )


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
    try:
        frame_bytes = base64.b64decode(body.frame_jpeg_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame_jpeg_b64") from exc
    return _run_inference(
        model_key=body.model_key,
        frame_bytes=frame_bytes,
        conf=body.conf,
        device=body.device,
        imgsz=body.imgsz,
        classes=body.classes,
    )

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


@app.post("/api/infer/jpeg")
def infer_jpeg(
    frame_jpeg: bytes = Body(..., media_type="image/jpeg"),
    model_key: str = Query(...),
    conf: float = Query(0.35),
    device: str = Query("cuda"),
    imgsz: int = Query(960, gt=0),
    classes: List[str] = Query(default=[]),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_model_server_token(authorization)
    return _run_inference(
        model_key=model_key,
        frame_bytes=frame_jpeg,
        conf=conf,
        device=device,
        imgsz=imgsz,
        classes=classes,
    )


@app.post("/api/infer/jpeg/batch")
def infer_jpeg_batch(
    frame_jpeg: bytes = Body(..., media_type="image/jpeg"),
    batch_json: str = Header(..., alias="X-Rakshak-Inference-Batch"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_model_server_token(authorization)
    if len(batch_json) > 16_384:
        raise HTTPException(status_code=413, detail="Inference batch metadata is too large")
    try:
        payload = json.loads(batch_json)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid inference batch metadata") from exc
    if not isinstance(payload, list) or not 1 <= len(payload) <= 8:
        raise HTTPException(status_code=400, detail="Inference batch must contain 1 to 8 requests")
    try:
        batch = [InferenceBatchItem(**item) for item in payload]
    except (TypeError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail="Invalid inference batch request") from exc
    request_ids = [item.request_id for item in batch]
    if len(set(request_ids)) != len(request_ids):
        raise HTTPException(status_code=400, detail="Inference batch request IDs must be unique")

    frame = _decode_frame(frame_jpeg)
    results = {}
    for item in batch:
        results[item.request_id] = _run_inference_frame(
            model_key=item.model_key,
            frame=frame,
            conf=item.conf,
            device=item.device,
            imgsz=item.imgsz,
            classes=item.classes,
        )["detections"]
    return {"results": results}
