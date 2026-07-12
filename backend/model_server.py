"""Rakshak Lens model server.

This process owns heavyweight model runtimes. Edge backends send decoded camera
frames here and keep local streaming, alerting, config, and frontend APIs intact.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
_DECODE_CACHE_MAX_ENTRIES = 2
_DECODE_CACHE_TTL_SECONDS = 0.75
_DECODE_CACHE_LOCK = threading.Lock()
_DECODE_CACHE = OrderedDict()
_DECODE_INFLIGHT: Dict[bytes, threading.Event] = {}
_DECODE_SLOTS = threading.BoundedSemaphore(2)
_MAX_RAW_FRAME_DIMENSION = 1920
_BATCH_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="model-batch",
)
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


def _parse_inference_batch(batch_json: str) -> List[InferenceBatchItem]:
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
    return batch


def _run_inference_batch(frame, batch: List[InferenceBatchItem]) -> Dict[str, Any]:
    futures = {
        item.request_id: _BATCH_INFERENCE_EXECUTOR.submit(
            _run_inference_frame,
            model_key=item.model_key,
            frame=frame,
            conf=item.conf,
            device=item.device,
            imgsz=item.imgsz,
            classes=item.classes,
        )
        for item in batch
    }
    results = {
        request_id: future.result()["detections"]
        for request_id, future in futures.items()
    }
    return {"results": results}


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
    cache_key = hashlib.blake2b(frame_bytes, digest_size=16).digest()
    while True:
        now = time.monotonic()
        with _DECODE_CACHE_LOCK:
            cached = _DECODE_CACHE.pop(cache_key, None)
            if cached is not None and now - cached[0] <= _DECODE_CACHE_TTL_SECONDS:
                _DECODE_CACHE[cache_key] = cached
                return cached[1]
            decode_complete = _DECODE_INFLIGHT.get(cache_key)
            if decode_complete is None:
                decode_complete = threading.Event()
                _DECODE_INFLIGHT[cache_key] = decode_complete
                break
        decode_complete.wait()

    try:
        with _DECODE_SLOTS:
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        _finish_decode(cache_key)
        raise HTTPException(status_code=400, detail="Invalid JPEG frame") from exc
    if frame is None:
        _finish_decode(cache_key)
        raise HTTPException(status_code=400, detail="Could not decode frame")

    with _DECODE_CACHE_LOCK:
        _DECODE_CACHE[cache_key] = (time.monotonic(), frame)
        while len(_DECODE_CACHE) > _DECODE_CACHE_MAX_ENTRIES:
            _DECODE_CACHE.popitem(last=False)
        decode_complete = _DECODE_INFLIGHT.pop(cache_key, None)
        if decode_complete is not None:
            decode_complete.set()
    return frame


def _finish_decode(cache_key: bytes) -> None:
    """Release same-frame waiters after a failed decode so they can retry."""
    with _DECODE_CACHE_LOCK:
        decode_complete = _DECODE_INFLIGHT.pop(cache_key, None)
        if decode_complete is not None:
            decode_complete.set()


def _clear_decode_cache() -> None:
    with _DECODE_CACHE_LOCK:
        _DECODE_CACHE.clear()


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
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame_jpeg_b64") from exc
    return _run_anpr(
        frame_bytes,
        conf=body.conf,
        device=body.device,
        imgsz=body.imgsz,
    )


@app.post("/api/anpr/jpeg")
def anpr_jpeg(
    frame_jpeg: bytes = Body(..., media_type="image/jpeg"),
    conf: float = Query(0.35),
    device: str = Query("cuda"),
    imgsz: int = Query(960, gt=0),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_model_server_token(authorization)
    return _run_anpr(frame_jpeg, conf=conf, device=device, imgsz=imgsz)


def _run_anpr(
    frame_bytes: bytes,
    *,
    conf: float,
    device: str,
    imgsz: int,
) -> Dict[str, Any]:
    try:
        arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JPEG frame") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    try:
        import plate_analyzer

        plates = plate_analyzer.analyze_frame(
            frame,
            conf=conf,
            device=_runtime_device(device),
            imgsz=imgsz,
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
    batch = _parse_inference_batch(batch_json)
    frame = _decode_frame(frame_jpeg)
    return _run_inference_batch(frame, batch)


@app.post("/api/infer/raw/batch")
def infer_raw_batch(
    frame_bytes: bytes = Body(..., media_type="application/octet-stream"),
    batch_json: str = Header(..., alias="X-Rakshak-Inference-Batch"),
    frame_width: int = Header(..., alias="X-Rakshak-Frame-Width"),
    frame_height: int = Header(..., alias="X-Rakshak-Frame-Height"),
    frame_channels: int = Header(..., alias="X-Rakshak-Frame-Channels"),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_model_server_token(authorization)
    batch = _parse_inference_batch(batch_json)
    if not (
        1 <= frame_width <= _MAX_RAW_FRAME_DIMENSION
        and 1 <= frame_height <= _MAX_RAW_FRAME_DIMENSION
        and frame_channels == 3
    ):
        raise HTTPException(status_code=400, detail="Invalid raw frame shape")
    expected_bytes = frame_width * frame_height * frame_channels
    if len(frame_bytes) != expected_bytes:
        raise HTTPException(status_code=400, detail="Raw frame byte length does not match shape")
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
        (frame_height, frame_width, frame_channels)
    )
    return _run_inference_batch(frame, batch)
