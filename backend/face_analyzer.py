"""InsightFace-based detection, enrollment validation, and frame matching."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

import face_store
from config_manager import get_config
from constants import PROJECT_ROOT

logger = logging.getLogger("rakshak_lens.faces")

MIN_FACE_SIZE = 48
LOW_QUALITY_MIN_SCORE = 0.55

_app = None
_app_lock = threading.RLock()


class FaceAnalyzerError(RuntimeError):
    code = "face_analysis_failed"


class FaceModelUnavailable(FaceAnalyzerError):
    code = "face_model_unavailable"


class NoFaceFound(FaceAnalyzerError):
    code = "no_face_found"


class MultipleFacesFound(FaceAnalyzerError):
    code = "multiple_faces_found"


class LowQualityFace(FaceAnalyzerError):
    code = "low_quality_face"


class DuplicateFace(FaceAnalyzerError):
    code = "duplicate_face"


@dataclass
class FacePayload:
    bbox: dict
    embedding: list[float]
    confidence: float


def get_app():
    global _app
    with _app_lock:
        if _app is not None:
            return _app
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise FaceModelUnavailable("InsightFace is not installed") from exc

        providers = ["CPUExecutionProvider"]
        device = get_config().get("global", {}).get("device", "cpu")
        if isinstance(device, str) and device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        root = PROJECT_ROOT / "models" / "face_recognition"
        app = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers)
        app.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1, det_size=(640, 640))
        _app = app
        return app


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        from io import BytesIO

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except Exception:
            pass
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise FaceAnalyzerError("Could not read image")
        return frame


def extract_enrollment_embedding(image_bytes: bytes) -> FacePayload:
    frame = decode_image(image_bytes)
    faces = _detect(frame)
    if not faces:
        raise NoFaceFound("No face found in the image")
    if len(faces) > 1:
        raise MultipleFacesFound("Use an image with exactly one face")
    payload = _payload_from_face(faces[0])
    reason = quality_reason(payload)
    if reason:
        raise LowQualityFace(reason)
    duplicate = face_store.find_best_match(payload.embedding)
    if duplicate:
        err = DuplicateFace(f"Face already looks enrolled as {duplicate['name']}")
        err.match = duplicate  # type: ignore[attr-defined]
        raise err
    return payload


def format_enrollment_photo(image_bytes: bytes, bbox: dict, *, output_size: int = 512) -> bytes:
    """Return a face-centered JPEG suitable for enrollment cards."""
    frame = decode_image(image_bytes)
    height, width = frame.shape[:2]
    x1 = max(0, int(bbox.get("x1", 0)))
    y1 = max(0, int(bbox.get("y1", 0)))
    x2 = min(width, int(bbox.get("x2", width)))
    y2 = min(height, int(bbox.get("y2", height)))
    if x2 <= x1 or y2 <= y1:
        raise FaceAnalyzerError("Face crop could not be generated")

    face_w = x2 - x1
    face_h = y2 - y1
    crop_size = int(max(face_w, face_h) * 2.2)
    crop_size = max(crop_size, face_w, face_h)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    crop_x1 = max(0, cx - crop_size // 2)
    crop_y1 = max(0, cy - crop_size // 2)
    crop_x2 = min(width, crop_x1 + crop_size)
    crop_y2 = min(height, crop_y1 + crop_size)
    crop_x1 = max(0, crop_x2 - crop_size)
    crop_y1 = max(0, crop_y2 - crop_size)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise FaceAnalyzerError("Face crop could not be generated")

    crop_h, crop_w = crop.shape[:2]
    side = max(crop_h, crop_w)
    top = (side - crop_h) // 2
    bottom = side - crop_h - top
    left = (side - crop_w) // 2
    right = side - crop_w - left
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    if side > output_size:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise FaceAnalyzerError("Face crop could not be encoded")
    return encoded.tobytes()


def analyze_frame(frame: np.ndarray) -> list[dict]:
    events: list[dict] = []
    for face in _detect(frame):
        payload = _payload_from_face(face)
        reason = quality_reason(payload)
        if reason:
            events.append({
                "eventType": "face_low_quality",
                "confidence": None,
                "matchedFaceId": None,
                "personName": None,
                "bbox": payload.bbox,
                "qualityReason": reason,
            })
            continue

        match = face_store.find_best_match(payload.embedding)
        if match:
            events.append({
                "eventType": "face_match",
                "confidence": round(float(match["confidence"]), 1),
                "matchedFaceId": match["id"],
                "personName": match["name"],
                "personGroup": match["group"],
                "bbox": payload.bbox,
                "qualityReason": None,
            })
        else:
            events.append({
                "eventType": "face_unknown",
                "confidence": None,
                "matchedFaceId": None,
                "personName": None,
                "bbox": payload.bbox,
                "qualityReason": None,
            })
    return events


def quality_reason(payload: FacePayload) -> str | None:
    width = payload.bbox["x2"] - payload.bbox["x1"]
    height = payload.bbox["y2"] - payload.bbox["y1"]
    if min(width, height) < MIN_FACE_SIZE:
        return "Face is too small for reliable recognition"
    if payload.confidence < LOW_QUALITY_MIN_SCORE:
        return "Face detection confidence is too low"
    if not payload.embedding:
        return "Face embedding could not be generated"
    return None


def _detect(frame: np.ndarray):
    app = get_app()
    with _app_lock:
        return app.get(frame)


def _payload_from_face(face) -> FacePayload:
    bbox_values = [int(round(float(value))) for value in face.bbox.tolist()]
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    embedding_list = [float(value) for value in (embedding.tolist() if embedding is not None else [])]
    return FacePayload(
        bbox={"x1": bbox_values[0], "y1": bbox_values[1], "x2": bbox_values[2], "y2": bbox_values[3]},
        embedding=embedding_list,
        confidence=float(getattr(face, "det_score", 0.0)),
    )
