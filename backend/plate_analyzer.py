"""Model-server ANPR analyzer: plate detection plus PaddleOCR text reading."""

from __future__ import annotations

import logging
import os
import re
import threading
from importlib import metadata
from typing import Any

import cv2
import numpy as np

import model_manager

logger = logging.getLogger("rakshak_lens.anpr")

MIN_PLATE_WIDTH = 24
MIN_PLATE_HEIGHT = 10
MIN_PLATE_ASPECT_RATIO = 1.6
MAX_PLATE_ASPECT_RATIO = 7.5
LOW_CONFIDENCE_THRESHOLD = 0.70
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
OCR_DIGIT_TRANSLATION = str.maketrans({
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
})
OCR_LETTER_TRANSLATION = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
})
_OCR_LOCK = threading.RLock()
_OCR = None

PPOCRV6_TIERS = {
    "tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}


def _env_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize_plate_text(value: str | None) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    return _correct_indian_plate_ocr(normalized)


def _correct_indian_plate_ocr(value: str) -> str:
    if INDIAN_PLATE_RE.match(value):
        return value
    if len(value) < 7 or len(value) > 11:
        return value

    for district_len in (2, 1):
        for series_len in (1, 2, 3):
            if 2 + district_len + series_len + 4 != len(value):
                continue
            state = value[:2].translate(OCR_LETTER_TRANSLATION)
            district = value[2:2 + district_len].translate(OCR_DIGIT_TRANSLATION)
            series_start = 2 + district_len
            series_end = series_start + series_len
            series = value[series_start:series_end].translate(OCR_LETTER_TRANSLATION)
            serial = value[series_end:].translate(OCR_DIGIT_TRANSLATION)
            candidate = f"{state}{district}{series}{serial}"
            if INDIAN_PLATE_RE.match(candidate):
                return candidate
    return value


def analyze_frame(
    frame: np.ndarray,
    *,
    conf: float,
    device: str,
    imgsz: int,
) -> list[dict[str, Any]]:
    """Return normalized ANPR candidates from a BGR frame."""
    records = model_manager.predict_records(
        "plate_recognition",
        frame,
        conf=conf,
        device=device,
        imgsz=imgsz,
    )
    candidates: list[dict[str, Any]] = []
    height, width = frame.shape[:2]
    for record in records:
        x1, y1, x2, y2 = _clamp_bbox(record.get("bbox", []), width, height)
        if not _is_plausible_plate_box(x1, y1, x2, y2):
            continue
        crop = _padded_crop(frame, x1, y1, x2, y2)
        text, ocr_confidence = _read_plate_text(crop)
        normalized = normalize_plate_text(text)
        detection_confidence = float(record.get("confidence") or 0.0)
        confidence = _combined_confidence(detection_confidence, ocr_confidence)
        if not _is_confident_indian_plate(normalized, confidence):
            continue
        candidates.append({
            "plateText": normalized or text,
            "normalizedPlate": normalized,
            "confidence": round(confidence * 100.0, 1),
            "detectionConfidence": round(detection_confidence * 100.0, 1),
            "ocrConfidence": round(ocr_confidence * 100.0, 1) if ocr_confidence is not None else None,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "qualityReason": None,
        })
    return candidates


def _is_confident_indian_plate(normalized: str, confidence: float) -> bool:
    if not INDIAN_PLATE_RE.match(normalized or ""):
        return False
    return confidence >= LOW_CONFIDENCE_THRESHOLD


def _get_ocr():
    global _OCR
    with _OCR_LOCK:
        if _OCR is not None:
            return _OCR
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError("PaddleOCR is not installed on the model server") from exc
        for kwargs in _ocr_init_candidates():
            try:
                _OCR = PaddleOCR(**kwargs)
                logger.info(
                    "PaddleOCR initialized",
                    extra={
                        "ocr_version": kwargs.get("ocr_version"),
                        "text_detection_model_name": kwargs.get("text_detection_model_name"),
                        "text_recognition_model_name": kwargs.get("text_recognition_model_name"),
                    },
                )
                break
            except (TypeError, ValueError):
                logger.warning("PaddleOCR init candidate rejected", extra={"kwargs": kwargs}, exc_info=True)
                continue
        if _OCR is None:
            raise RuntimeError("PaddleOCR could not be initialized")
        return _OCR


def _ocr_init_candidates() -> tuple[dict[str, Any], ...]:
    base = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
    }
    tier = _env_text("SAFETYLENS_ANPR_OCR_TIER", "tiny").lower()
    det_default, rec_default = PPOCRV6_TIERS.get(tier, PPOCRV6_TIERS["tiny"])
    ocr_version = _env_text("SAFETYLENS_ANPR_OCR_VERSION", "PP-OCRv6")
    detection_model = _env_text("SAFETYLENS_ANPR_TEXT_DETECTION_MODEL", det_default)
    recognition_model = _env_text("SAFETYLENS_ANPR_TEXT_RECOGNITION_MODEL", rec_default)
    candidates: list[dict[str, Any]] = []
    if detection_model and recognition_model:
        candidates.append({
            **base,
            "text_detection_model_name": detection_model,
            "text_recognition_model_name": recognition_model,
        })
    if ocr_version.lower() not in {"", "auto", "default"}:
        candidates.append({
            **base,
            "ocr_version": ocr_version,
        })
    candidates.extend((
        {**base, "lang": "en"},
        {"use_textline_orientation": False, "enable_mkldnn": False, "lang": "en"},
        {"use_angle_cls": False, "enable_mkldnn": False, "lang": "en"},
        {"enable_mkldnn": False, "lang": "en"},
    ))
    return tuple(candidates)


def ocr_runtime_config() -> dict[str, Any]:
    candidate = _ocr_init_candidates()[0]
    try:
        paddleocr_version = metadata.version("paddleocr")
    except metadata.PackageNotFoundError:
        paddleocr_version = None
    try:
        paddlepaddle_version = metadata.version("paddlepaddle")
    except metadata.PackageNotFoundError:
        paddlepaddle_version = None
    return {
        "paddleocr_version": paddleocr_version,
        "paddlepaddle_version": paddlepaddle_version,
        "tier": _env_text("SAFETYLENS_ANPR_OCR_TIER", "tiny").lower(),
        "ocr_version": _env_text("SAFETYLENS_ANPR_OCR_VERSION", "PP-OCRv6"),
        "text_detection_model_name": candidate.get("text_detection_model_name"),
        "text_recognition_model_name": candidate.get("text_recognition_model_name"),
    }


def _read_plate_text(crop: np.ndarray) -> tuple[str, float | None]:
    ocr = _get_ocr()
    with _OCR_LOCK:
        if hasattr(ocr, "ocr"):
            try:
                result = ocr.ocr(crop, cls=True)
            except TypeError:
                result = ocr.ocr(crop)
        else:
            result = ocr.predict(crop)
    best_text = ""
    best_confidence: float | None = None
    for text, confidence in _iter_ocr_entries(result):
        normalized = normalize_plate_text(text)
        if not normalized:
            continue
        if best_confidence is None or confidence > best_confidence:
            best_text = normalized
            best_confidence = confidence
    return best_text, best_confidence


def _iter_ocr_entries(result) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    if not result:
        return entries
    for page in result:
        if isinstance(page, dict):
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for text, confidence in zip(texts, scores):
                entries.append((str(text), float(confidence)))
            continue
        if not page:
            continue
        for item in page:
            try:
                text, confidence = item[1]
                entries.append((str(text), float(confidence)))
            except Exception:
                continue
    return entries


def _clamp_bbox(values, width: int, height: int) -> tuple[int, int, int, int]:
    if len(values) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _padded_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_x = max(4, int((x2 - x1) * 0.08))
    pad_y = max(3, int((y2 - y1) * 0.15))
    return frame[
        max(0, y1 - pad_y): min(height, y2 + pad_y),
        max(0, x1 - pad_x): min(width, x2 + pad_x),
    ]


def _is_plausible_plate_box(x1: int, y1: int, x2: int, y2: int) -> bool:
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
        return False
    aspect_ratio = box_width / max(box_height, 1)
    return MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO


def _combined_confidence(detection_confidence: float, ocr_confidence: float | None) -> float:
    if ocr_confidence is None:
        return detection_confidence * 0.35
    return max(0.0, min(1.0, detection_confidence * 0.45 + ocr_confidence * 0.55))
