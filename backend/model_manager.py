"""Model registry, lifecycle management, and install jobs."""

from __future__ import annotations

import logging
import os
import base64
import gc
import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from capability_registry import ALL_PPE_PROMPT_TERMS, CLASS_TERM_TO_CAPABILITY, ModelKey
from constants import (
    MODEL_METADATA_BREAKER_INITIAL_SECONDS,
    MODEL_METADATA_BREAKER_MAX_SECONDS,
    MODEL_METADATA_LOG_REMINDER_SECONDS,
    MODEL_METADATA_TIMEOUT_SECONDS,
    MODEL_METADATA_TTL_SECONDS,
    MODEL_SERVER_TOKEN,
    MODEL_SERVER_URL,
    PROJECT_ROOT,
    resolve_coco_model_variant,
)
from tensorrt_engine import validate_engine

logger = logging.getLogger("rakshak_lens.models")


class RemoteInferenceOverloadedError(RuntimeError):
    """The bounded remote inference pool is busy with fresher admitted work."""


def _bounded_env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


MODELS_ROOT = PROJECT_ROOT / "models"
TMP_MODELS_ROOT = Path(tempfile.gettempdir()) / "rakshak-lens-models"
_JOB_POLL_FINAL_STATES = {"ready", "failed"}
_REMOTE_SESSION_LOCAL = threading.local()
_REMOTE_PAIR_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="remote-model-pair",
)
_REMOTE_JOB_MAX_INFLIGHT = _bounded_env_int(
    "SAFETYLENS_REMOTE_INFERENCE_MAX_INFLIGHT",
    2,
    minimum=1,
    maximum=8,
)
_REMOTE_JOB_ADMISSION = threading.BoundedSemaphore(_REMOTE_JOB_MAX_INFLIGHT)
# The phase scheduler keeps normal camera arrivals staggered. A 65 ms bounded
# wait absorbs short service-time jitter without allowing stale camera work to
# build an unbounded model-server queue.
_REMOTE_JOB_ADMISSION_WAIT_SECONDS = 0.065
_REMOTE_JPEG_QUALITY = 85
_RESIZED_GROUPED_REMOTE_JPEG_QUALITY = 90
_REMOTE_RAW_TRANSPORT_ENV = "SAFETYLENS_MODEL_SERVER_RAW_TRANSPORT"
_REMOTE_RAW_BATCH_SUPPORT_LOCK = threading.Lock()
_REMOTE_RAW_BATCH_SUPPORT: dict[str, Any] = {"url": None, "supported": None}
_COCO_LOW_RES_ENGINE_ENV = "SAFETYLENS_COCO_LOW_RES_TENSORRT_ENGINE"
_OPEN_VOCAB_MODEL_KEYS = {"ppe_specialist", "yoloe_long_tail"}
_REMOTE_MODEL_CACHE_LOCK = threading.RLock()
_REMOTE_MODEL_CACHE: dict[str, Any] = {
    "models": None,
    "updated_at": 0.0,
    "failure_error": None,
    "failure_at": 0.0,
}

_COCO_MODEL_OPTIONS: dict[str, dict[str, str]] = {
    "yolo26n": {
        "display_name": "COCO Primary Nano",
        "filename": "yolo26n.pt",
        "warmup_behavior": "Full-frame nano COCO detect warmup",
    },
    "yolo26s": {
        "display_name": "COCO Primary Small",
        "filename": "yolo26s.pt",
        "warmup_behavior": "Full-frame small COCO detect warmup",
    },
    "yolo26m": {
        "display_name": "COCO Primary Medium",
        "filename": "yolo26m.pt",
        "warmup_behavior": "Full-frame medium COCO detect warmup",
    },
}
def _resolve_coco_model_variant(configured_model: str | None) -> str:
    configured_model = str(configured_model or "").strip()
    selected_model = resolve_coco_model_variant(configured_model)
    if configured_model and configured_model != selected_model:
        logger.warning(
            "Unknown COCO model variant configured; falling back to small",
            extra={"configured_model": configured_model},
        )
    return selected_model


_COCO_MODEL_VARIANT = _resolve_coco_model_variant(
    os.environ.get("SAFETYLENS_COCO_MODEL")
)
_COCO_MODEL_CONFIG = _COCO_MODEL_OPTIONS[_COCO_MODEL_VARIANT]


MODEL_DEFINITIONS: dict[ModelKey, dict[str, Any]] = {
    "coco_primary": {
        "model_key": "coco_primary",
        "display_name": _COCO_MODEL_CONFIG["display_name"],
        "filename": _COCO_MODEL_CONFIG["filename"],
        "local_path": MODELS_ROOT / "coco_primary" / _COCO_MODEL_CONFIG["filename"],
        "legacy_paths": [],
        "download_url": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{_COCO_MODEL_CONFIG['filename']}",
        "warmup_behavior": _COCO_MODEL_CONFIG["warmup_behavior"],
        "shared_asset_key": _COCO_MODEL_VARIANT,
    },
    "ppe_specialist": {
        "model_key": "ppe_specialist",
        "display_name": "PPE Specialist",
        "filename": "yoloe-26s-seg.pt",
        "local_path": MODELS_ROOT / "yoloe_open_vocab" / "yoloe-26s-seg.pt",
        "legacy_paths": [],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt",
        "warmup_behavior": "YOLOE-26S open-vocab warmup with stable PPE prompts",
        "shared_asset_key": "yoloe-26s-seg",
    },
    "ppe_closed_set_candidate": {
        "model_key": "ppe_closed_set_candidate",
        "display_name": "Closed-Set Apron/Harness PPE",
        "filename": "apron-harness-ppe.onnx",
        "local_path": MODELS_ROOT / "ppe_closed_set_candidate" / "apron-harness-ppe.onnx",
        "legacy_paths": [],
        "download_url": "",
        "warmup_behavior": "Closed-set apron/harness PPE detect warmup",
        "shared_asset_key": "apron-harness-ppe",
    },
    "yoloe_long_tail": {
        "model_key": "yoloe_long_tail",
        "display_name": "YOLOE Long-Tail",
        "filename": "yoloe-26s-seg.pt",
        "local_path": MODELS_ROOT / "yoloe_open_vocab" / "yoloe-26s-seg.pt",
        "legacy_paths": [],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt",
        "warmup_behavior": "YOLOE-26S open-vocab warmup with long-tail prompts",
        "shared_asset_key": "yoloe-26s-seg",
    },
    "fire_smoke_specialist": {
        "model_key": "fire_smoke_specialist",
        "display_name": "Fire / Smoke Specialist",
        "filename": "wildfire-smoke-fire.pt",
        "local_path": MODELS_ROOT / "fire_smoke_specialist" / "wildfire-smoke-fire.pt",
        "legacy_paths": [
            PROJECT_ROOT / "models" / "fire_smoke_specialist_benchmark" / "odiug77-wildfire-smoke-fire-yolo26.pt",
        ],
        "download_url": "https://huggingface.co/odiug77/wildfire-smoke-fire/resolve/main/wildfire-smoke-fire.pt",
        "warmup_behavior": "YOLO26 fire/smoke specialist warmup",
        "shared_asset_key": "wildfire-smoke-fire-yolo26",
    },
    "face_recognition": {
        "model_key": "face_recognition",
        "display_name": "Face Recognition",
        "filename": "buffalo_l",
        "local_path": MODELS_ROOT / "face_recognition" / "buffalo_l",
        "legacy_paths": [PROJECT_ROOT / "models" / "face_recognition" / "buffalo_l"],
        "download_url": "",
        "warmup_behavior": "InsightFace SCRFD + ArcFace lazy load",
        "shared_asset_key": "insightface-buffalo-l",
    },
    "pose_specialist": {
        "model_key": "pose_specialist",
        "display_name": "Pose Specialist",
        "filename": "yolo26n-pose.pt",
        "local_path": MODELS_ROOT / "pose_specialist" / "yolo26n-pose.pt",
        "legacy_paths": [],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
        "warmup_behavior": "YOLO26 nano pose estimation warmup",
        "shared_asset_key": "yolo26n-pose",
    },
    "plate_recognition": {
        "model_key": "plate_recognition",
        "display_name": "Plate Recognition",
        "filename": "plate-detector.pt",
        "local_path": MODELS_ROOT / "plate_recognition" / "plate-detector.pt",
        "legacy_paths": [PROJECT_ROOT / "plate-detector.pt"],
        "download_url": "",
        "warmup_behavior": "Plate detector + PaddleOCR on model server",
        "shared_asset_key": "plate-detector",
    },
}

_DEFAULT_LONG_TAIL_PROMPTS = ["fire", "smoke", "flames"]
_OPEN_VOCAB_MODEL_KEYS = {"ppe_specialist", "yoloe_long_tail"}
_LAZY_START_MODEL_KEYS = {"yoloe_long_tail"}
_MAX_CLASS_EMBEDDING_CACHE_ENTRIES = 16
_MODEL_LOCK = threading.RLock()


def _new_model_runtime() -> dict[str, Any]:
    return {
        "handle": None,
        "lock": threading.Lock(),
        "loaded_path": None,
        "runtime_path": None,
        "runtime_backend": None,
        "runtime_fallback_error": None,
        "fallback_path": None,
        "fixed_imgsz": None,
        "fixed_classes": [],
        "fixed_class_groups": [],
        "current_classes": [],
        "class_embeddings": {},
        "warmed": False,
    }


def _build_model_runtimes() -> dict[ModelKey, dict[str, Any]]:
    """Use one runtime for model keys that point at the same model asset."""
    runtimes: dict[ModelKey, dict[str, Any]] = {}
    runtimes_by_asset: dict[str, dict[str, Any]] = {}
    for model_key, definition in MODEL_DEFINITIONS.items():
        asset_key = str(definition["shared_asset_key"])
        if model_key == "ppe_specialist" and os.environ.get("SAFETYLENS_PPE_TENSORRT_ENGINE", "").strip():
            asset_key = f"{asset_key}:ppe-tensorrt"
        runtime = runtimes_by_asset.get(asset_key)
        if runtime is None:
            runtime = _new_model_runtime()
            runtimes_by_asset[asset_key] = runtime
        runtimes[model_key] = runtime
    return runtimes


_MODEL_RUNTIMES = _build_model_runtimes()
_COCO_LOW_RES_RUNTIME = _new_model_runtime()
_MODEL_STATES: dict[ModelKey, dict[str, Any]] = {
    model_key: {
        "status": "not_downloaded",
        "error": None,
        "active_path": None,
        "job_id": None,
        "shared_asset_key": definition["shared_asset_key"],
    }
    for model_key, definition in MODEL_DEFINITIONS.items()
}
_INSTALL_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_JOB_ID: Optional[str] = None
_FORCE_LOCAL_INFERENCE = False
_DEVICE_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


REMOTE_MODEL_STATUS_TIMEOUT_SECONDS = max(
    0.5,
    _env_float("SAFETYLENS_MODEL_STATUS_TIMEOUT_SECONDS", 2.0),
)
REMOTE_MODEL_STATUS_CACHE_TTL_SECONDS = max(
    0.0,
    _env_float("SAFETYLENS_MODEL_STATUS_CACHE_TTL_SECONDS", 5.0),
)
REMOTE_MODEL_FAILURE_CACHE_TTL_SECONDS = max(
    0.0,
    _env_float("SAFETYLENS_MODEL_FAILURE_CACHE_TTL_SECONDS", 5.0),
)


def force_local_inference() -> None:
    global _FORCE_LOCAL_INFERENCE
    _FORCE_LOCAL_INFERENCE = True


def _remote_settings() -> dict[str, Any]:
    if _FORCE_LOCAL_INFERENCE:
        return {"enabled": False, "url": "", "token": "", "timeout_seconds": 30.0}
    try:
        from config_manager import get_config

        settings = get_config().get("model_server", {})
    except Exception:
        logger.debug("Model server config unavailable", exc_info=True)
        settings = {}
    url = str(settings.get("url") or "").strip().rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    enabled = bool(settings.get("enabled", bool(url)))
    if not enabled:
        url = ""
    timeout_seconds = settings.get("timeout_seconds", 30.0)
    try:
        timeout_seconds = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout_seconds = 30.0
    return {
        "enabled": enabled,
        "url": url,
        "token": str(settings.get("token") or ""),
        "timeout_seconds": max(1.0, timeout_seconds),
    }


def _runtime_device(requested: str | int) -> str:
    """Return a usable local inference device, normalizing CUDA indexes."""
    device = str(requested or "cpu").strip().lower()
    if device.isdigit():
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{device}"
        except Exception:
            pass
        fallback = "cpu"
        key = (device, fallback)
        if key not in _DEVICE_FALLBACK_WARNED:
            _DEVICE_FALLBACK_WARNED.add(key)
            logger.warning("Configured inference device unavailable; falling back", extra={"requested_device": device, "runtime_device": fallback})
        return fallback
    if device in {"cuda", "gpu"}:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0" if device == "gpu" else "cuda"
        except Exception:
            pass
        fallback = "cpu"
        key = (device, fallback)
        if key not in _DEVICE_FALLBACK_WARNED:
            _DEVICE_FALLBACK_WARNED.add(key)
            logger.warning("Configured inference device unavailable; falling back", extra={"requested_device": device, "runtime_device": fallback})
        return fallback
    if device != "mps":
        return device or "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    fallback = "cpu"
    key = (device, fallback)
    if key not in _DEVICE_FALLBACK_WARNED:
        _DEVICE_FALLBACK_WARNED.add(key)
        logger.warning("Configured inference device unavailable; falling back", extra={"requested_device": device, "runtime_device": fallback})
    return fallback


def _configured_local_device() -> str:
    """Resolve the device for model-manager-owned warmup/load paths."""
    env_device = os.environ.get("SAFETYLENS_INFERENCE_DEVICE")
    if env_device:
        return env_device

    if not _FORCE_LOCAL_INFERENCE:
        try:
            from config_manager import get_config

            return str(get_config().get("global", {}).get("device") or "cpu")
        except Exception:
            logger.debug("Configured inference device unavailable", exc_info=True)

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_REMOTE_MODEL_CATALOG_CONDITION = threading.Condition()
_REMOTE_MODEL_CATALOG: list[dict[str, Any]] | None = None
_REMOTE_MODEL_CATALOG_EXPIRES_AT = 0.0
_REMOTE_MODEL_CATALOG_REFRESHING = False
_REMOTE_MODEL_CATALOG_EPOCH = 0
_REMOTE_MODEL_CATALOG_FAILURE_COUNT = 0
_REMOTE_MODEL_CATALOG_NEXT_RETRY_AT = 0.0
_REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT: float | None = None
_REMOTE_MODEL_CATALOG_LAST_LOGGED_AT: float | None = None
_REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT: float | None = None
_REMOTE_MODEL_CATALOG_LAST_FAILURE_AT: float | None = None
_REMOTE_MODEL_CATALOG_MAX_BYTES = 256 * 1024
_REMOTE_MODEL_CATALOG_CHUNK_BYTES = 16 * 1024
_REMOTE_MODEL_TRANSPORT_LOCK = threading.Lock()
_REMOTE_MODEL_TRANSPORT_THREAD: threading.Thread | None = None


class _RemoteModelCatalogError(RuntimeError):
    """Internal marker whose details must never cross the public boundary."""


def is_remote_inference_enabled() -> bool:
    return bool(_remote_settings()["url"])


def _remote_headers(token: str | None = None) -> dict[str, str]:
    if token is None:
        token = str(_remote_settings().get("token") or MODEL_SERVER_TOKEN)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _remote_session():
    """Return one keep-alive HTTP session per caller thread.

    Camera workers call the model server concurrently, while requests.Session
    does not promise cross-thread safety. A thread-local session keeps sockets
    warm without sharing mutable connection-pool state between cameras.
    """
    session = getattr(_REMOTE_SESSION_LOCAL, "session", None)
    if session is None:
        import requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _REMOTE_SESSION_LOCAL.session = session
    return session


def _remote_get(path: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
    settings = _remote_settings()
    timeout = settings["timeout_seconds"] if timeout_seconds is None else max(0.25, float(timeout_seconds))
    response = _remote_session().get(
        f"{settings['url']}{path}",
        headers=_remote_headers(settings["token"]),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _monotonic() -> float:
    return time.monotonic()


def _bounded_text(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str) or len(value) > maximum:
        return None
    return value


def _bounded_text_list(value: Any, *, maximum_items: int = 128) -> list[str] | None:
    if not isinstance(value, list) or len(value) > maximum_items:
        return None
    result: list[str] = []
    for item in value:
        text = _bounded_text(item, maximum=256)
        if text is None:
            return None
        result.append(text)
    return result


def _remote_unavailable_model(model_key: ModelKey) -> dict[str, Any]:
    """Build a compatibility-shaped, credential-safe fail-closed record."""
    definition = MODEL_DEFINITIONS[model_key]
    return {
        "model_key": model_key,
        "display_name": definition["display_name"],
        "filename": definition["filename"],
        "local_path": str(definition["local_path"]),
        "active_path": None,
        "download_url": definition["download_url"],
        "warmup_behavior": definition["warmup_behavior"],
        "status": "remote_unavailable",
        "error": "Remote model metadata unavailable",
        "job_id": None,
        "shared_asset_key": definition["shared_asset_key"],
        "is_ready": False,
        "is_downloaded": False,
        "runtime_backend": None,
        "runtime_path": None,
        "runtime_fallback_error": None,
        "runtime_fixed_imgsz": None,
        "runtime_fixed_classes": [],
        "runtime_fixed_class_groups": [],
        "runtime_low_res_backend": None,
        "runtime_low_res_path": None,
        "runtime_low_res_fallback_error": None,
        "runtime_low_res_fixed_imgsz": None,
        "runtime_low_res_warmed": None,
    }


def _unavailable_remote_catalog() -> list[dict[str, Any]]:
    return [_remote_unavailable_model(model_key) for model_key in MODEL_DEFINITIONS]


def _sanitize_remote_model(item: dict[str, Any], model_key: ModelKey) -> dict[str, Any]:
    """Whitelist bounded catalogue fields while retaining the public shape."""
    result = _remote_unavailable_model(model_key)
    is_ready = item["is_ready"]
    result["is_ready"] = is_ready

    status = _bounded_text(item.get("status"), maximum=64)
    allowed_statuses = {
        "ready",
        "not_downloaded",
        "installing",
        "failed",
    }
    if is_ready:
        result["status"] = "ready"
        result["error"] = None
    else:
        result["status"] = status if status in allowed_statuses - {"ready"} else "not_downloaded"
        # Do not proxy arbitrary exception text from another process.
        result["error"] = "Remote model reported an error" if item.get("error") is not None else None

    active_path = _bounded_text(item.get("active_path"), maximum=1_024)
    if active_path is not None and "://" not in active_path:
        result["active_path"] = active_path
    job_id = _bounded_text(item.get("job_id"), maximum=128)
    if job_id is not None:
        result["job_id"] = job_id
    if type(item.get("is_downloaded")) is bool:
        result["is_downloaded"] = item["is_downloaded"]

    runtime_backend = _bounded_text(item.get("runtime_backend"), maximum=64)
    if runtime_backend is not None:
        result["runtime_backend"] = runtime_backend
    runtime_path = _bounded_text(item.get("runtime_path"), maximum=1_024)
    if runtime_path is not None and "://" not in runtime_path:
        result["runtime_path"] = runtime_path
    if item.get("runtime_fallback_error") is not None:
        result["runtime_fallback_error"] = "Remote model runtime fallback active"

    fixed_imgsz = item.get("runtime_fixed_imgsz")
    if type(fixed_imgsz) is int and 1 <= fixed_imgsz <= 16_384:
        result["runtime_fixed_imgsz"] = fixed_imgsz
    fixed_classes = _bounded_text_list(item.get("runtime_fixed_classes"))
    if fixed_classes is not None:
        result["runtime_fixed_classes"] = fixed_classes
    fixed_groups = _bounded_text_list(item.get("runtime_fixed_class_groups"))
    if fixed_groups is not None:
        result["runtime_fixed_class_groups"] = fixed_groups
    low_res_backend = _bounded_text(item.get("runtime_low_res_backend"), maximum=64)
    if low_res_backend is not None:
        result["runtime_low_res_backend"] = low_res_backend
    low_res_path = _bounded_text(item.get("runtime_low_res_path"), maximum=1_024)
    if low_res_path is not None and "://" not in low_res_path:
        result["runtime_low_res_path"] = low_res_path
    if item.get("runtime_low_res_fallback_error") is not None:
        result["runtime_low_res_fallback_error"] = "Remote low-resolution runtime fallback active"
    low_res_imgsz = item.get("runtime_low_res_fixed_imgsz")
    if type(low_res_imgsz) is int and 1 <= low_res_imgsz <= 16_384:
        result["runtime_low_res_fixed_imgsz"] = low_res_imgsz
    if type(item.get("runtime_low_res_warmed")) is bool:
        result["runtime_low_res_warmed"] = item["runtime_low_res_warmed"]
    return result


def _validate_remote_model_catalog(payload: Any) -> list[dict[str, Any]]:
    if (
        type(payload) is not dict
        or "models" not in payload
        or len(payload) > 16
        or any(not isinstance(key, str) or len(key) > 128 for key in payload)
    ):
        raise _RemoteModelCatalogError("invalid catalogue envelope")
    raw_models = payload["models"]
    if not isinstance(raw_models, list) or len(raw_models) > 32:
        raise _RemoteModelCatalogError("invalid model collection")

    models_by_key: dict[ModelKey, dict[str, Any]] = {}
    for item in raw_models:
        if type(item) is not dict:
            raise _RemoteModelCatalogError("invalid model item")
        model_key = item.get("model_key")
        if not isinstance(model_key, str) or len(model_key) > 128:
            raise _RemoteModelCatalogError("invalid model key")
        # Bounded future model keys are ignored until this edge version knows
        # how to route them. Known keys remain strict and duplicate-free.
        if model_key not in MODEL_DEFINITIONS:
            continue
        if model_key in models_by_key:
            raise _RemoteModelCatalogError("duplicate model key")
        if type(item.get("is_ready")) is not bool:
            raise _RemoteModelCatalogError("invalid readiness value")
        models_by_key[model_key] = _sanitize_remote_model(item, model_key)

    return [
        models_by_key.get(model_key, _remote_unavailable_model(model_key))
        for model_key in MODEL_DEFINITIONS
    ]


def _decode_catalog_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")

        def reject_constant(_value: str):
            raise ValueError("non-finite JSON constant")

        return json.loads(text, parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise _RemoteModelCatalogError("invalid catalogue JSON") from exc


def _fetch_remote_model_catalog_blocking() -> list[dict[str, Any]]:
    """Perform the bounded HTTP exchange inside an isolated daemon thread."""
    deadline = time.monotonic() + MODEL_METADATA_TIMEOUT_SECONDS
    read_timeout = min(0.5, MODEL_METADATA_TIMEOUT_SECONDS)
    settings = _remote_settings()
    model_server_url = str(settings.get("url") or MODEL_SERVER_URL).rstrip("/")
    response = _remote_session().get(
        f"{model_server_url}/api/models",
        headers=_remote_headers(str(settings.get("token") or MODEL_SERVER_TOKEN)),
        timeout=(MODEL_METADATA_TIMEOUT_SECONDS, read_timeout),
        allow_redirects=False,
        stream=True,
    )
    try:
        if time.monotonic() >= deadline:
            raise _RemoteModelCatalogError("catalogue deadline exceeded")
        if 300 <= response.status_code < 400:
            raise _RemoteModelCatalogError("redirect rejected")
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise _RemoteModelCatalogError("invalid content length") from exc
            if declared_length < 0 or declared_length > _REMOTE_MODEL_CATALOG_MAX_BYTES:
                raise _RemoteModelCatalogError("catalogue response too large")

        body = bytearray()
        for chunk in response.iter_content(chunk_size=_REMOTE_MODEL_CATALOG_CHUNK_BYTES):
            if time.monotonic() >= deadline:
                raise _RemoteModelCatalogError("catalogue deadline exceeded")
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > _REMOTE_MODEL_CATALOG_MAX_BYTES:
                raise _RemoteModelCatalogError("catalogue response too large")
        return _validate_remote_model_catalog(_decode_catalog_json(bytes(body)))
    finally:
        response.close()


def _fetch_remote_model_catalog() -> list[dict[str, Any]]:
    """Fetch metadata with a hard caller deadline, including DNS and headers.

    Requests cannot bound DNS resolution. A single daemon transport thread
    contains that gap: callers return at the configured deadline, and no new
    transport thread is launched while a timed-out one is still alive. Once it
    exits, the breaker may probe again normally.
    """
    global _REMOTE_MODEL_TRANSPORT_THREAD

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    with _REMOTE_MODEL_TRANSPORT_LOCK:
        current = _REMOTE_MODEL_TRANSPORT_THREAD
        if current is not None and current.is_alive():
            raise _RemoteModelCatalogError("catalogue transport still in flight")

        def run_transport() -> None:
            global _REMOTE_MODEL_TRANSPORT_THREAD
            outcome: tuple[str, Any] = (
                "error",
                _RemoteModelCatalogError("catalogue transport aborted"),
            )
            try:
                outcome = ("ok", _fetch_remote_model_catalog_blocking())
            except Exception as exc:
                outcome = ("error", exc)
            finally:
                with _REMOTE_MODEL_TRANSPORT_LOCK:
                    if _REMOTE_MODEL_TRANSPORT_THREAD is threading.current_thread():
                        _REMOTE_MODEL_TRANSPORT_THREAD = None
            # Publish only after clearing the in-flight marker so an immediate
            # caller can never observe a completed thread as still active.
            result_queue.put(outcome)

        transport = threading.Thread(
            target=run_transport,
            name="model-metadata-http",
            daemon=True,
        )
        _REMOTE_MODEL_TRANSPORT_THREAD = transport
        transport.start()

    try:
        outcome, value = result_queue.get(timeout=MODEL_METADATA_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise _RemoteModelCatalogError("catalogue deadline exceeded") from exc
    if outcome == "error":
        raise value
    return value


def _catalog_backoff_seconds(failure_count: int) -> float:
    exponent = min(30, max(0, failure_count - 1))
    return min(
        MODEL_METADATA_BREAKER_MAX_SECONDS,
        MODEL_METADATA_BREAKER_INITIAL_SECONDS * (2**exponent),
    )


def _catalog_log_fields(now: float) -> dict[str, Any]:
    outage_started = _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT
    return {
        "failure_count": _REMOTE_MODEL_CATALOG_FAILURE_COUNT,
        "retry_after_seconds": round(max(0.0, _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT - now), 3),
        "outage_duration_seconds": round(max(0.0, now - outage_started), 3)
        if outage_started is not None
        else 0.0,
    }


def _log_catalog_unavailable(*, transition: bool, reminder: bool, now: float) -> None:
    if transition:
        logger.warning("Remote model metadata became unavailable", extra=_catalog_log_fields(now))
    elif reminder:
        logger.warning("Remote model metadata remains unavailable", extra=_catalog_log_fields(now))


def _get_remote_model_catalog() -> list[dict[str, Any]]:
    """Return one fail-closed catalogue using a process-wide singleflight."""
    global _REMOTE_MODEL_CATALOG
    global _REMOTE_MODEL_CATALOG_EXPIRES_AT
    global _REMOTE_MODEL_CATALOG_REFRESHING
    global _REMOTE_MODEL_CATALOG_FAILURE_COUNT
    global _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT
    global _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT
    global _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT
    global _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT
    global _REMOTE_MODEL_CATALOG_LAST_FAILURE_AT

    now = _monotonic()
    reminder = False
    with _REMOTE_MODEL_CATALOG_CONDITION:
        if _REMOTE_MODEL_CATALOG is not None and now < _REMOTE_MODEL_CATALOG_EXPIRES_AT:
            return deepcopy(_REMOTE_MODEL_CATALOG)

        if _REMOTE_MODEL_CATALOG_REFRESHING:
            # A stuck leader must not strand camera/API callers indefinitely.
            wait_deadline = time.monotonic() + MODEL_METADATA_TIMEOUT_SECONDS + 0.25
            while _REMOTE_MODEL_CATALOG_REFRESHING:
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    return _unavailable_remote_catalog()
                _REMOTE_MODEL_CATALOG_CONDITION.wait(timeout=remaining)
            now = _monotonic()
            if _REMOTE_MODEL_CATALOG is not None and now < _REMOTE_MODEL_CATALOG_EXPIRES_AT:
                return deepcopy(_REMOTE_MODEL_CATALOG)
            if now < _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT:
                return _unavailable_remote_catalog()

        if now < _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT:
            if (
                _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT is not None
                and (
                    _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT is None
                    or now - _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT >= MODEL_METADATA_LOG_REMINDER_SECONDS
                )
            ):
                _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT = now
                reminder = True
            catalog = _unavailable_remote_catalog()
            epoch = None
        else:
            _REMOTE_MODEL_CATALOG_REFRESHING = True
            epoch = _REMOTE_MODEL_CATALOG_EPOCH
            catalog = None

    if reminder:
        _log_catalog_unavailable(transition=False, reminder=True, now=now)
    if catalog is not None:
        return catalog

    try:
        fetched = _fetch_remote_model_catalog()
    except Exception:
        now = _monotonic()
        with _REMOTE_MODEL_CATALOG_CONDITION:
            if epoch != _REMOTE_MODEL_CATALOG_EPOCH:
                _REMOTE_MODEL_CATALOG_CONDITION.notify_all()
                return _unavailable_remote_catalog()
            _REMOTE_MODEL_CATALOG_REFRESHING = False
            _REMOTE_MODEL_CATALOG = None
            _REMOTE_MODEL_CATALOG_EXPIRES_AT = 0.0
            _REMOTE_MODEL_CATALOG_FAILURE_COUNT += 1
            _REMOTE_MODEL_CATALOG_LAST_FAILURE_AT = now
            _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT = now + _catalog_backoff_seconds(
                _REMOTE_MODEL_CATALOG_FAILURE_COUNT
            )
            transition = _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT is None
            if transition:
                _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT = now
            reminder = not transition and (
                _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT is None
                or now - _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT >= MODEL_METADATA_LOG_REMINDER_SECONDS
            )
            if transition or reminder:
                _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT = now
            _REMOTE_MODEL_CATALOG_CONDITION.notify_all()
        _log_catalog_unavailable(transition=transition, reminder=reminder, now=now)
        return _unavailable_remote_catalog()

    now = _monotonic()
    with _REMOTE_MODEL_CATALOG_CONDITION:
        if epoch != _REMOTE_MODEL_CATALOG_EPOCH:
            _REMOTE_MODEL_CATALOG_CONDITION.notify_all()
            return _unavailable_remote_catalog()
        _REMOTE_MODEL_CATALOG_REFRESHING = False
        recovery_started = _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT
        _REMOTE_MODEL_CATALOG = deepcopy(fetched)
        _REMOTE_MODEL_CATALOG_EXPIRES_AT = now + MODEL_METADATA_TTL_SECONDS
        _REMOTE_MODEL_CATALOG_FAILURE_COUNT = 0
        _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT = 0.0
        _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT = None
        _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT = None
        _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT = now
        _REMOTE_MODEL_CATALOG_LAST_FAILURE_AT = None
        _REMOTE_MODEL_CATALOG_CONDITION.notify_all()
    if recovery_started is not None:
        logger.info(
            "Remote model metadata recovered",
            extra={"outage_duration_seconds": round(max(0.0, now - recovery_started), 3)},
        )
    return deepcopy(fetched)


def _remote_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = _remote_settings()
    response = _remote_session().post(
        f"{settings['url']}{path}",
        json=payload,
        headers=_remote_headers(settings["token"]),
        timeout=settings["timeout_seconds"],
    )
    response.raise_for_status()
    return response.json()


def _remote_post_jpeg(
    path: str,
    frame_jpeg: bytes,
    *,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Post JPEG bytes directly, returning None when the server lacks the route."""
    settings = _remote_settings()
    headers = _remote_headers(settings["token"])
    headers["Content-Type"] = "image/jpeg"
    response = _remote_session().post(
        f"{settings['url']}{path}",
        params=params,
        data=frame_jpeg,
        headers=headers,
        timeout=settings["timeout_seconds"],
    )
    if response.status_code == 404:
        try:
            if response.json().get("detail") == "Not Found":
                return None
        except (AttributeError, TypeError, ValueError):
            pass
    response.raise_for_status()
    return response.json()


def _remote_post_jpeg_batch(
    path: str,
    frame_jpeg: bytes,
    *,
    batch: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Post one JPEG with bounded model requests, or None on an older server."""
    settings = _remote_settings()
    headers = _remote_headers(settings["token"])
    headers["Content-Type"] = "image/jpeg"
    headers["X-Rakshak-Inference-Batch"] = json.dumps(batch, separators=(",", ":"))
    response = _remote_session().post(
        f"{settings['url']}{path}",
        data=frame_jpeg,
        headers=headers,
        timeout=settings["timeout_seconds"],
    )
    if response.status_code == 404:
        try:
            if response.json().get("detail") == "Not Found":
                return None
        except (AttributeError, TypeError, ValueError):
            pass
    response.raise_for_status()
    return response.json()


def _remote_raw_transport_enabled() -> bool:
    return os.environ.get(_REMOTE_RAW_TRANSPORT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _remote_post_raw_batch(
    path: str,
    frame,
    *,
    batch: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Post one contiguous BGR frame, or return None for an older server."""
    settings = _remote_settings()
    with _REMOTE_RAW_BATCH_SUPPORT_LOCK:
        if _REMOTE_RAW_BATCH_SUPPORT["url"] != settings["url"]:
            _REMOTE_RAW_BATCH_SUPPORT.update(url=settings["url"], supported=None)
        if _REMOTE_RAW_BATCH_SUPPORT["supported"] is False:
            return None

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    contiguous = np.ascontiguousarray(frame)
    height, width, channels = contiguous.shape
    headers = _remote_headers(settings["token"])
    headers.update({
        "Content-Type": "application/octet-stream",
        "X-Rakshak-Inference-Batch": json.dumps(batch, separators=(",", ":")),
        "X-Rakshak-Frame-Width": str(width),
        "X-Rakshak-Frame-Height": str(height),
        "X-Rakshak-Frame-Channels": str(channels),
    })
    response = _remote_session().post(
        f"{settings['url']}{path}",
        data=contiguous.tobytes(),
        headers=headers,
        timeout=settings["timeout_seconds"],
    )
    if response.status_code == 404:
        try:
            route_missing = response.json().get("detail") == "Not Found"
        except (AttributeError, TypeError, ValueError):
            route_missing = False
        if route_missing:
            with _REMOTE_RAW_BATCH_SUPPORT_LOCK:
                if _REMOTE_RAW_BATCH_SUPPORT["url"] == settings["url"]:
                    _REMOTE_RAW_BATCH_SUPPORT["supported"] = False
            return None
    response.raise_for_status()
    with _REMOTE_RAW_BATCH_SUPPORT_LOCK:
        if _REMOTE_RAW_BATCH_SUPPORT["url"] == settings["url"]:
            _REMOTE_RAW_BATCH_SUPPORT["supported"] = True
    return response.json()


def _now() -> float:
    return time.time()


def _copy_open_vocab_text_encoder_assets(target_dir: Path) -> list[str]:
    copied: list[str] = []
    for filename in ("mobileclip_blt.ts", "mobileclip2_b.ts"):
        candidates = [
            PROJECT_ROOT / "backend" / filename,
            Path(__file__).parent / filename,
            PROJECT_ROOT / filename,
            PROJECT_ROOT / "models" / filename,
            PROJECT_ROOT.parent / filename,
        ]
        for source in candidates:
            if not source.exists():
                continue
            target = target_dir / filename
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(str(source), str(target))
            copied.append(filename)
            break
    return copied


def _sync_state_compat():
    try:
        import state

        state.model = _MODEL_RUNTIMES["coco_primary"]["handle"]
        state.yoloe_model = _MODEL_RUNTIMES["yoloe_long_tail"]["handle"]
    except Exception:
        logger.debug("State compatibility sync skipped", exc_info=True)


def _set_model_state(model_key: ModelKey, **updates):
    _MODEL_STATES[model_key].update(updates)


def _serialize_model_state(model_key: ModelKey) -> dict[str, Any]:
    definition = MODEL_DEFINITIONS[model_key]
    state = deepcopy(_MODEL_STATES[model_key])
    active_path = state.get("active_path")
    runtime = _MODEL_RUNTIMES[model_key]
    low_res_runtime = {
        "runtime_low_res_backend": None,
        "runtime_low_res_path": None,
        "runtime_low_res_fallback_error": None,
        "runtime_low_res_fixed_imgsz": None,
        "runtime_low_res_warmed": None,
    }
    if model_key == "coco_primary":
        with _COCO_LOW_RES_RUNTIME["lock"]:
            low_res_runtime = {
                "runtime_low_res_backend": _COCO_LOW_RES_RUNTIME.get("runtime_backend"),
                "runtime_low_res_path": _COCO_LOW_RES_RUNTIME.get("runtime_path"),
                "runtime_low_res_fallback_error": _COCO_LOW_RES_RUNTIME.get("runtime_fallback_error"),
                "runtime_low_res_fixed_imgsz": _COCO_LOW_RES_RUNTIME.get("fixed_imgsz"),
                "runtime_low_res_warmed": bool(_COCO_LOW_RES_RUNTIME.get("warmed")),
            }
    return {
        "model_key": model_key,
        "display_name": definition["display_name"],
        "filename": definition["filename"],
        "local_path": str(definition["local_path"]),
        "active_path": str(active_path) if active_path else None,
        "download_url": definition["download_url"],
        "warmup_behavior": definition["warmup_behavior"],
        "status": state["status"],
        "error": state["error"],
        "job_id": state["job_id"],
        "shared_asset_key": definition["shared_asset_key"],
        "is_ready": state["status"] == "ready",
        "is_downloaded": bool(active_path),
        "runtime_backend": runtime.get("runtime_backend"),
        "runtime_path": runtime.get("runtime_path"),
        "runtime_fallback_error": runtime.get("runtime_fallback_error"),
        "runtime_fixed_imgsz": runtime.get("fixed_imgsz"),
        "runtime_fixed_classes": list(runtime.get("fixed_classes") or []),
        "runtime_fixed_class_groups": list(runtime.get("fixed_class_groups") or []),
        **low_res_runtime,
    }


def _remote_unavailable_models(error: Any) -> list[dict[str, Any]]:
    error_text = str(error)
    return [
        {
            **_serialize_model_state(model_key),
            "status": "remote_unavailable",
            "error": error_text,
            "is_ready": False,
        }
        for model_key in MODEL_DEFINITIONS
    ]


def _fresh_remote_model_cache(now: float, cache_ttl_seconds: float) -> list[dict[str, Any]] | None:
    if cache_ttl_seconds <= 0:
        return None
    with _REMOTE_MODEL_CACHE_LOCK:
        cached_models = _REMOTE_MODEL_CACHE.get("models")
        updated_at = float(_REMOTE_MODEL_CACHE.get("updated_at") or 0.0)
        if cached_models is not None and (now - updated_at) < cache_ttl_seconds:
            return deepcopy(cached_models)
    return None


def _fresh_remote_failure_cache(now: float) -> list[dict[str, Any]] | None:
    if REMOTE_MODEL_FAILURE_CACHE_TTL_SECONDS <= 0:
        return None
    with _REMOTE_MODEL_CACHE_LOCK:
        failure_at = float(_REMOTE_MODEL_CACHE.get("failure_at") or 0.0)
        failure_error = _REMOTE_MODEL_CACHE.get("failure_error")
        if failure_error is not None and (now - failure_at) < REMOTE_MODEL_FAILURE_CACHE_TTL_SECONDS:
            return _remote_unavailable_models(failure_error)
    return None


def clear_remote_model_status_cache() -> None:
    with _REMOTE_MODEL_CACHE_LOCK:
        _REMOTE_MODEL_CACHE.update({
            "models": None,
            "updated_at": 0.0,
            "failure_error": None,
            "failure_at": 0.0,
        })
    invalidate_remote_model_catalog()


def list_models(
    *,
    timeout_seconds: float | None = None,
    allow_cached: bool = False,
    cache_ttl_seconds: float | None = None,
) -> list[dict[str, Any]]:
    if is_remote_inference_enabled():
        return _get_remote_model_catalog()
    with _MODEL_LOCK:
        return [_serialize_model_state(model_key) for model_key in MODEL_DEFINITIONS]


def list_models_for_status() -> list[dict[str, Any]]:
    return list_models(
        timeout_seconds=REMOTE_MODEL_STATUS_TIMEOUT_SECONDS,
        allow_cached=True,
    )


def get_model(model_key: str) -> dict[str, Any]:
    if is_remote_inference_enabled():
        for model in list_models():
            if model.get("model_key") == model_key:
                return model
        raise KeyError(f"Unknown model key: {model_key}")
    with _MODEL_LOCK:
        if model_key not in MODEL_DEFINITIONS:
            raise KeyError(f"Unknown model key: {model_key}")
        return _serialize_model_state(model_key)  # type: ignore[arg-type]


def model_readiness_snapshot(model_keys: list[str] | None = None) -> dict[str, bool]:
    """Return one isolated readiness mapping suitable for request-wide reuse."""
    requested = list(MODEL_DEFINITIONS) if model_keys is None else list(model_keys)
    if not requested:
        return {}
    known_requested = [key for key in requested if key in MODEL_DEFINITIONS]
    readiness = {key: False for key in requested if key not in MODEL_DEFINITIONS}
    if not known_requested:
        return readiness
    if is_remote_inference_enabled():
        ready_by_key = {
            item["model_key"]: item["is_ready"] is True
            for item in _get_remote_model_catalog()
        }
        readiness.update({
            model_key: model_key in MODEL_DEFINITIONS and ready_by_key.get(model_key) is True
            for model_key in known_requested
        })
        return readiness
    with _MODEL_LOCK:
        readiness.update({
            model_key: (
                model_key in MODEL_DEFINITIONS
                and _MODEL_STATES[model_key]["status"] == "ready"
            )
            for model_key in known_requested
        })
        return readiness


def missing_model_keys(
    model_keys: list[str],
    readiness: dict[str, bool] | None = None,
    *,
    timeout_seconds: float | None = None,
    allow_cached: bool = False,
) -> list[ModelKey]:
    snapshot = readiness if readiness is not None else model_readiness_snapshot(model_keys)
    return [
        model_key  # type: ignore[misc]
        for model_key in model_keys
        if snapshot.get(model_key) is not True
    ]


def remote_model_metadata_health() -> dict[str, Any]:
    """Expose safe cache/breaker state without URLs or exception details."""
    if not is_remote_inference_enabled():
        return {
            "enabled": False,
            "status": "disabled",
            "cache_fresh": False,
            "refresh_in_flight": False,
            "failure_count": 0,
            "retry_after_seconds": 0.0,
            "outage_age_seconds": None,
            "last_success_age_seconds": None,
        }
    now = _monotonic()
    with _REMOTE_MODEL_CATALOG_CONDITION:
        cache_fresh = (
            _REMOTE_MODEL_CATALOG is not None
            and now < _REMOTE_MODEL_CATALOG_EXPIRES_AT
        )
        if cache_fresh:
            status = "healthy"
        elif _REMOTE_MODEL_CATALOG_REFRESHING:
            status = "refreshing"
        else:
            status = "unavailable"
        return {
            "enabled": True,
            "status": status,
            "cache_fresh": cache_fresh,
            "refresh_in_flight": _REMOTE_MODEL_CATALOG_REFRESHING,
            "failure_count": _REMOTE_MODEL_CATALOG_FAILURE_COUNT,
            "retry_after_seconds": round(
                max(0.0, _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT - now),
                3,
            ),
            "outage_age_seconds": round(
                max(0.0, now - _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT),
                3,
            )
            if _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT is not None
            else None,
            "last_success_age_seconds": round(
                max(0.0, now - _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT),
                3,
            )
            if _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT is not None
            else None,
        }


def invalidate_remote_model_catalog() -> None:
    """Invalidate cached metadata and reset its breaker; fence in-flight work."""
    global _REMOTE_MODEL_CATALOG
    global _REMOTE_MODEL_CATALOG_EXPIRES_AT
    global _REMOTE_MODEL_CATALOG_REFRESHING
    global _REMOTE_MODEL_CATALOG_EPOCH
    global _REMOTE_MODEL_CATALOG_FAILURE_COUNT
    global _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT
    global _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT
    global _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT
    global _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT
    global _REMOTE_MODEL_CATALOG_LAST_FAILURE_AT

    with _REMOTE_MODEL_CATALOG_CONDITION:
        _REMOTE_MODEL_CATALOG = None
        _REMOTE_MODEL_CATALOG_EXPIRES_AT = 0.0
        _REMOTE_MODEL_CATALOG_EPOCH += 1
        _REMOTE_MODEL_CATALOG_REFRESHING = False
        _REMOTE_MODEL_CATALOG_FAILURE_COUNT = 0
        _REMOTE_MODEL_CATALOG_NEXT_RETRY_AT = 0.0
        _REMOTE_MODEL_CATALOG_OUTAGE_STARTED_AT = None
        _REMOTE_MODEL_CATALOG_LAST_LOGGED_AT = None
        _REMOTE_MODEL_CATALOG_LAST_SUCCESS_AT = None
        _REMOTE_MODEL_CATALOG_LAST_FAILURE_AT = None
        _REMOTE_MODEL_CATALOG_CONDITION.notify_all()


def get_install_job(job_id: str) -> Optional[dict[str, Any]]:
    if is_remote_inference_enabled():
        import requests
        try:
            return _remote_get(f"/api/models/install/{job_id}")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
    with _MODEL_LOCK:
        job = _INSTALL_JOBS.get(job_id)
        return deepcopy(job) if job else None


def _update_job(job_id: str, **updates):
    with _MODEL_LOCK:
        job = _INSTALL_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _now()


def _resolve_existing_path(model_key: ModelKey) -> Optional[Path]:
    definition = MODEL_DEFINITIONS[model_key]
    local_path = definition["local_path"]
    if local_path.exists():
        return local_path
    for legacy_path in definition.get("legacy_paths", []):
        if legacy_path.exists():
            return _promote_legacy_asset(model_key, legacy_path)
    return None


def _promote_legacy_asset(model_key: ModelKey, legacy_path: Path) -> Path:
    definition = MODEL_DEFINITIONS[model_key]
    local_path = definition["local_path"]
    if legacy_path == local_path:
        return local_path
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if legacy_path.is_dir():
            shutil.copytree(str(legacy_path), str(local_path), dirs_exist_ok=True)
        else:
            shutil.copy2(str(legacy_path), str(local_path))
        logger.info(
            "Promoted legacy model asset into registry path",
            extra={
                "model_key": model_key,
                "legacy_path": str(legacy_path),
                "local_path": str(local_path),
            },
        )
        return local_path
    except Exception as exc:
        logger.warning(
            "Failed to promote legacy model asset; using legacy path",
            extra={
                "model_key": model_key,
                "legacy_path": str(legacy_path),
                "local_path": str(local_path),
                "error": str(exc),
            },
        )
        return legacy_path


def _copy_to_local_fs(model_key: ModelKey, source_path: Path) -> Path:
    TMP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    target = TMP_MODELS_ROOT / f"{model_key}-{source_path.name}"
    if not target.exists() or target.stat().st_size != source_path.stat().st_size:
        shutil.copy2(str(source_path), str(target))
    return target


def _set_open_vocab_classes(handle: Any, classes: list[str]):
    tmp_models_dir = TMP_MODELS_ROOT
    tmp_models_dir.mkdir(parents=True, exist_ok=True)
    copied_text_encoders = _copy_open_vocab_text_encoder_assets(tmp_models_dir)
    if not copied_text_encoders:
        raise RuntimeError("MobileCLIP text encoder artifact not found for open-vocabulary model setup")

    original_cwd = os.getcwd()
    try:
        handle.to("cpu")
        os.chdir(str(tmp_models_dir))
        embeddings = handle.get_text_pe(classes)
        handle.set_classes(classes, embeddings)
        return embeddings.detach().cpu()
    finally:
        os.chdir(original_cwd)


def _default_open_vocab_classes(model_key: ModelKey) -> list[str]:
    if model_key == "ppe_specialist":
        return list(ALL_PPE_PROMPT_TERMS)
    return list(_DEFAULT_LONG_TAIL_PROMPTS)


def _apply_open_vocab_classes(
    runtime: dict[str, Any],
    handle,
    classes: list[str],
    *,
    device: str,
) -> None:
    cache_key = tuple(classes)
    embedding_cache = runtime["class_embeddings"]
    embeddings = embedding_cache.pop(cache_key, None)
    if embeddings is None:
        embeddings = _set_open_vocab_classes(handle, classes)
        handle.to(device)
    else:
        handle.set_classes(classes, embeddings)
    embedding_cache[cache_key] = embeddings
    while len(embedding_cache) > _MAX_CLASS_EMBEDDING_CACHE_ENTRIES:
        oldest_key = next(iter(embedding_cache))
        embedding_cache.pop(oldest_key)
    runtime["current_classes"] = classes


def _configured_runtime_path(
    model_key: ModelKey,
    source_path: Path,
) -> tuple[Path, str, int | None, list[str], list[str], str | None]:
    settings = {
        "coco_primary": ("SAFETYLENS_COCO_TENSORRT_ENGINE", "detect"),
        "ppe_specialist": ("SAFETYLENS_PPE_TENSORRT_ENGINE", "segment"),
    }
    setting = settings.get(model_key)
    if setting is None:
        return source_path, "pytorch", None, [], [], None
    environment_name, expected_task = setting
    configured = os.environ.get(environment_name, "").strip()
    if not configured:
        return source_path, "pytorch", None, [], [], None
    engine_path = Path(configured).expanduser()
    if not engine_path.is_absolute():
        engine_path = source_path.parent / engine_path
    manifest, error = validate_engine(
        source_path=source_path,
        engine_path=engine_path,
        expected_task=expected_task,
    )
    if error:
        logger.warning(
            "TensorRT engine rejected; using PyTorch",
            extra={"model_key": model_key, "engine_path": str(engine_path), "reason": error},
        )
        return source_path, "pytorch", None, [], [], error
    fixed_classes = list(manifest.get("classes") or [])
    fixed_class_groups = list(manifest.get("classGroups") or [])
    if model_key == "ppe_specialist" and (
        not fixed_classes or len(fixed_class_groups) != len(fixed_classes)
    ):
        error = "PPE TensorRT manifest must declare fixed prompt classes and semantic groups"
        logger.warning(
            "TensorRT engine rejected; using PyTorch",
            extra={"model_key": model_key, "engine_path": str(engine_path), "reason": error},
        )
        return source_path, "pytorch", None, [], [], error
    return engine_path, "tensorrt", int(manifest["imgsz"]), fixed_classes, fixed_class_groups, None


def _set_runtime_metadata(
    runtime: dict[str, Any],
    *,
    source_path: Path,
    runtime_path: Path,
    backend: str,
    fixed_imgsz: int | None,
    fixed_classes: list[str],
    fixed_class_groups: list[str],
    fallback_error: str | None,
) -> None:
    runtime["loaded_path"] = str(source_path)
    runtime["runtime_path"] = str(runtime_path)
    runtime["runtime_backend"] = backend
    runtime["runtime_fallback_error"] = fallback_error
    runtime["fallback_path"] = str(source_path) if backend == "tensorrt" else None
    runtime["fixed_imgsz"] = fixed_imgsz
    runtime["fixed_classes"] = list(fixed_classes)
    runtime["fixed_class_groups"] = list(fixed_class_groups)


def _load_runtime(model_key: ModelKey, source_path: Path) -> None:
    runtime = _MODEL_RUNTIMES[model_key]
    (
        runtime_path,
        backend,
        fixed_imgsz,
        fixed_classes,
        fixed_class_groups,
        fallback_error,
    ) = _configured_runtime_path(model_key, source_path)
    if (
        runtime["handle"] is not None
        and runtime["loaded_path"] == str(source_path)
        and runtime.get("runtime_path") == str(runtime_path)
    ):
        if model_key in _OPEN_VOCAB_MODEL_KEYS and backend != "tensorrt":
            from config_manager import get_config

            with runtime["lock"]:
                initial_classes = _default_open_vocab_classes(model_key)
                if runtime["current_classes"] != initial_classes:
                    _apply_open_vocab_classes(
                        runtime,
                        runtime["handle"],
                        initial_classes,
                        device=get_config()["global"]["device"],
                    )
        return

    if model_key == "face_recognition":
        # InsightFace owns its own runtime and lazy-loads in face_recognition.py.
        # The model manager only tracks whether the expected model directory exists.
        runtime["handle"] = "insightface"
        _set_runtime_metadata(
            runtime,
            source_path=source_path,
            runtime_path=source_path,
            backend="insightface",
            fixed_imgsz=None,
            fixed_classes=[],
            fixed_class_groups=[],
            fallback_error=None,
        )
        runtime["current_classes"] = []
        return

    from ultralytics import YOLO

    device = _runtime_device(_configured_local_device())

    if model_key not in _OPEN_VOCAB_MODEL_KEYS:
        try:
            handle = (
                YOLO(str(runtime_path), task="detect")
                if backend == "tensorrt"
                else YOLO(str(runtime_path))
            )
        except Exception as exc:
            if backend != "tensorrt":
                raise
            logger.exception(
                "TensorRT model load failed; using PyTorch",
                extra={"model_key": model_key, "runtime_path": str(runtime_path)},
            )
            runtime_path = source_path
            backend = "pytorch_fallback"
            fixed_imgsz = None
            fallback_error = f"TensorRT load failed: {exc}"
            handle = YOLO(str(source_path))
        runtime["handle"] = handle
        _set_runtime_metadata(
            runtime,
            source_path=source_path,
            runtime_path=runtime_path,
            backend=backend,
            fixed_imgsz=fixed_imgsz,
            fixed_classes=fixed_classes,
            fixed_class_groups=fixed_class_groups,
            fallback_error=fallback_error,
        )
        runtime["current_classes"] = []
        runtime["warmed"] = False
        return

    if model_key == "ppe_specialist" and backend == "tensorrt":
        try:
            handle = YOLO(str(runtime_path), task="segment")
        except Exception as exc:
            logger.exception(
                "TensorRT model load failed; using PyTorch",
                extra={"model_key": model_key, "runtime_path": str(runtime_path)},
            )
            runtime_path = source_path
            backend = "pytorch_fallback"
            fixed_imgsz = None
            fixed_classes = []
            fixed_class_groups = []
            fallback_error = f"TensorRT load failed: {exc}"
        else:
            runtime["handle"] = handle
            _set_runtime_metadata(
                runtime,
                source_path=source_path,
                runtime_path=runtime_path,
                backend=backend,
                fixed_imgsz=fixed_imgsz,
                fixed_classes=fixed_classes,
                fixed_class_groups=fixed_class_groups,
                fallback_error=fallback_error,
            )
            runtime["current_classes"] = list(fixed_classes)
            runtime["class_embeddings"] = {}
            runtime["warmed"] = False
            return

    local_model = _copy_to_local_fs(model_key, source_path)
    handle = YOLO(str(local_model))
    initial_classes = _default_open_vocab_classes(model_key)

    device = _runtime_device(_configured_local_device())
    runtime["handle"] = handle
    _set_runtime_metadata(
        runtime,
        source_path=source_path,
        runtime_path=source_path,
        backend=backend,
        fixed_imgsz=None,
        fixed_classes=[],
        fixed_class_groups=[],
        fallback_error=fallback_error,
    )
    runtime["current_classes"] = []
    runtime["class_embeddings"] = {}
    runtime["warmed"] = False
    _apply_open_vocab_classes(runtime, handle, initial_classes, device=device)


def _mark_runtime_lazy(model_key: ModelKey, source_path: Path) -> None:
    runtime = _MODEL_RUNTIMES[model_key]
    if runtime["handle"] is not None:
        return
    _set_runtime_metadata(
        runtime,
        source_path=source_path,
        runtime_path=source_path,
        backend="lazy",
        fixed_imgsz=None,
        fixed_classes=[],
        fixed_class_groups=[],
        fallback_error=None,
    )
    runtime["current_classes"] = []
    runtime["class_embeddings"] = {}
    runtime["warmed"] = False


def initialize() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    with _MODEL_LOCK:
        for model_key in MODEL_DEFINITIONS:
            _set_model_state(model_key, status="not_downloaded", error=None, active_path=None, job_id=None)

    for model_key in MODEL_DEFINITIONS:
        existing_path = _resolve_existing_path(model_key)
        if existing_path is None:
            continue
        with _MODEL_LOCK:
            _set_model_state(model_key, status="loading", error=None, active_path=existing_path, job_id=None)
        try:
            if model_key in _LAZY_START_MODEL_KEYS:
                _mark_runtime_lazy(model_key, existing_path)
            else:
                _load_runtime(model_key, existing_path)
            with _MODEL_LOCK:
                _set_model_state(model_key, status="ready", error=None, active_path=existing_path, job_id=None)
        except Exception as exc:
            logger.exception("Model load failed during initialization", extra={"model_key": model_key})
            with _MODEL_LOCK:
                _set_model_state(model_key, status="failed", error=str(exc), active_path=existing_path, job_id=None)
    for model_key in MODEL_DEFINITIONS:
        _warm_configured_fixed_runtime(model_key)
    _warm_configured_low_res_coco_runtime()
    _sync_state_compat()


def _asset_keys_for_models(model_keys: list[ModelKey]) -> list[str]:
    asset_keys: list[str] = []
    for model_key in model_keys:
        shared_asset_key = MODEL_DEFINITIONS[model_key]["shared_asset_key"]
        if shared_asset_key not in asset_keys:
            asset_keys.append(shared_asset_key)
    return asset_keys


def install_models(model_keys: list[str]) -> dict[str, Any]:
    if is_remote_inference_enabled():
        job = _remote_post("/api/models/install", {"model_keys": model_keys})
        invalidate_remote_model_catalog()
        return job

    normalized_keys: list[ModelKey] = []
    for raw_key in model_keys:
        if raw_key in MODEL_DEFINITIONS and raw_key not in normalized_keys:
            normalized_keys.append(raw_key)  # type: ignore[arg-type]
    if not normalized_keys:
        raise ValueError("No valid model keys requested")

    with _MODEL_LOCK:
        global _ACTIVE_JOB_ID
        if _ACTIVE_JOB_ID:
            active = _INSTALL_JOBS.get(_ACTIVE_JOB_ID)
            if active and active.get("status") not in _JOB_POLL_FINAL_STATES:
                active_assets = set(active.get("asset_keys", []))
                requested_assets = set(_asset_keys_for_models(normalized_keys))
                if requested_assets.issubset(active_assets):
                    return deepcopy(active)
                raise RuntimeError("Another model install is already in progress")

        job_id = uuid.uuid4().hex[:12]
        asset_keys = _asset_keys_for_models(normalized_keys)
        job = {
            "id": job_id,
            "model_keys": normalized_keys,
            "asset_keys": asset_keys,
            "status": "queued",
            "stage": "queued",
            "current_model_key": normalized_keys[0],
            "progress_percent": 0.0,
            "bytes_downloaded": 0,
            "total_bytes": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        _INSTALL_JOBS[job_id] = job
        _ACTIVE_JOB_ID = job_id
        for model_key in normalized_keys:
            _set_model_state(model_key, status="installing", error=None, job_id=job_id)

    thread = threading.Thread(target=_run_install_job, args=(job_id,), daemon=True)
    thread.start()
    return get_install_job(job_id) or job


def retry_install_job(job_id: str) -> dict[str, Any]:
    job = get_install_job(job_id)
    if not job:
        raise KeyError("Install job not found")
    if is_remote_inference_enabled():
        retried = _remote_post(f"/api/models/install/{job_id}/retry", {})
        invalidate_remote_model_catalog()
        return retried
    return install_models(job.get("model_keys", []))


def _download_asset(job_id: str, model_key: ModelKey, destination: Path, url: str, asset_index: int, asset_count: int) -> Path:
    import requests
    if not url:
        raise RuntimeError(f"Model {model_key} must be installed manually at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    _update_job(
        job_id,
        status="running",
        stage="downloading",
        current_model_key=model_key,
        progress_percent=asset_index / max(asset_count, 1) * 100.0,
        bytes_downloaded=0,
        total_bytes=None,
        error=None,
    )
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", "0")) or None
        bytes_downloaded = 0
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_downloaded += len(chunk)
                asset_progress = (bytes_downloaded / total_bytes) if total_bytes else 0.0
                progress_percent = ((asset_index + asset_progress) / max(asset_count, 1)) * 100.0
                _update_job(
                    job_id,
                    bytes_downloaded=bytes_downloaded,
                    total_bytes=total_bytes,
                    progress_percent=min(progress_percent, 99.0),
                )
    os.replace(tmp_path, destination)
    return destination


def _verify_asset(path: Path, total_bytes: Optional[int]):
    if not path.exists():
        raise RuntimeError("Downloaded model file missing after transfer")
    if path.stat().st_size <= 0:
        raise RuntimeError("Downloaded model file is empty")
    if total_bytes and path.stat().st_size != total_bytes:
        raise RuntimeError("Downloaded model file size does not match expected size")


def _run_install_job(job_id: str):
    job = get_install_job(job_id)
    if not job:
        return

    try:
        asset_keys = job["asset_keys"]
        model_keys = job["model_keys"]
        for asset_index, asset_key in enumerate(asset_keys):
            asset_model_keys = [
                model_key
                for model_key in model_keys
                if MODEL_DEFINITIONS[model_key]["shared_asset_key"] == asset_key
            ]
            primary_model_key = asset_model_keys[0]
            definition = MODEL_DEFINITIONS[primary_model_key]

            _update_job(job_id, status="running", stage="checking_disk", current_model_key=primary_model_key)
            existing_path = _resolve_existing_path(primary_model_key)
            active_path = existing_path
            if active_path is None:
                active_path = _download_asset(
                    job_id,
                    primary_model_key,
                    definition["local_path"],
                    definition["download_url"],
                    asset_index,
                    len(asset_keys),
                )

            _update_job(job_id, stage="verifying", current_model_key=primary_model_key)
            _verify_asset(active_path, get_install_job(job_id).get("total_bytes") if get_install_job(job_id) else None)

            _update_job(job_id, stage="preparing_assets", current_model_key=primary_model_key)
            MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            TMP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

            for model_key in asset_model_keys:
                _update_job(job_id, stage="loading", current_model_key=model_key)
                _load_runtime(model_key, active_path)
                _update_job(job_id, stage="warming_up", current_model_key=model_key)
                _warm_configured_fixed_runtime(model_key)
                if model_key == "coco_primary":
                    _warm_configured_low_res_coco_runtime(active_path)
                with _MODEL_LOCK:
                    _set_model_state(model_key, status="ready", error=None, active_path=active_path, job_id=None)

        _update_job(job_id, status="ready", stage="ready", progress_percent=100.0, error=None)
        _sync_state_compat()
        try:
            from video_processing import restart_all_cameras

            restart_all_cameras()
        except Exception:
            logger.exception("Failed to restart cameras after model install")
    except Exception as exc:
        logger.exception("Model install failed", extra={"job_id": job_id})
        _update_job(job_id, status="failed", stage="failed", error=str(exc))
        with _MODEL_LOCK:
            for model_key in job["model_keys"]:
                _set_model_state(model_key, status="failed", error=str(exc), job_id=None)
    finally:
        with _MODEL_LOCK:
            global _ACTIVE_JOB_ID
            if _ACTIVE_JOB_ID == job_id:
                _ACTIVE_JOB_ID = None


def _predict_with_runtime(
    model_key: ModelKey,
    runtime: dict[str, Any],
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: list[str] | None = None,
):
    handle = runtime["handle"]
    if handle is None:
        raise RuntimeError(f"Model {model_key} is not loaded")
    effective_imgsz = runtime.get("fixed_imgsz") or imgsz

    requested_classes = None
    if model_key in _OPEN_VOCAB_MODEL_KEYS:
        requested = classes if classes is not None else _default_open_vocab_classes(model_key)
        requested_classes = [value for value in requested if isinstance(value, str) and value]
        fixed_classes = list(runtime.get("fixed_classes") or [])
        if fixed_classes and requested_classes != fixed_classes:
            raise RuntimeError("Configured prompts do not match the fixed TensorRT engine classes")

    if not runtime.get("warmed"):
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        handle.predict(dummy, verbose=False, device=device, imgsz=effective_imgsz)
        runtime["warmed"] = True
    if model_key in _OPEN_VOCAB_MODEL_KEYS:
        if runtime["current_classes"] != requested_classes:
            _apply_open_vocab_classes(runtime, handle, requested_classes, device=device)
    return handle.predict(frame, conf=conf, verbose=False, device=device, imgsz=effective_imgsz)


def _configured_optional_engine_path(source_path: Path, configured: str) -> Path:
    engine_path = Path(configured).expanduser()
    if not engine_path.is_absolute():
        engine_path = source_path.parent / engine_path
    return engine_path


def _predict_with_low_res_coco_runtime(
    frame,
    *,
    source_path: Path,
    conf: float,
    device: str,
    imgsz: int,
) -> tuple[bool, Any]:
    """Use an optional compact engine for fitting sources or an exact size request."""
    configured = os.environ.get(_COCO_LOW_RES_ENGINE_ENV, "").strip()
    if not configured:
        return False, None

    runtime = _COCO_LOW_RES_RUNTIME
    engine_path = _configured_optional_engine_path(source_path, configured)
    with runtime["lock"]:
        same_artifacts = (
            runtime.get("loaded_path") == str(source_path)
            and runtime.get("runtime_path") == str(engine_path)
        )
        if not same_artifacts:
            old_handle = runtime.get("handle")
            runtime_lock = runtime["lock"]
            runtime.clear()
            runtime.update(_new_model_runtime())
            runtime["lock"] = runtime_lock
            del old_handle
            manifest, error = validate_engine(
                source_path=source_path,
                engine_path=engine_path,
                expected_task="detect",
            )
            if error:
                _set_runtime_metadata(
                    runtime,
                    source_path=source_path,
                    runtime_path=engine_path,
                    backend="tensorrt_rejected",
                    fixed_imgsz=None,
                    fixed_classes=[],
                    fixed_class_groups=[],
                    fallback_error=error,
                )
                logger.warning(
                    "Low-resolution TensorRT engine rejected",
                    extra={"engine_path": str(engine_path), "reason": error},
                )
                return False, None
            _set_runtime_metadata(
                runtime,
                source_path=source_path,
                runtime_path=engine_path,
                backend="tensorrt_lazy",
                fixed_imgsz=int(manifest["imgsz"]),
                fixed_classes=[],
                fixed_class_groups=[],
                fallback_error=None,
            )

        fixed_imgsz = runtime.get("fixed_imgsz")
        source_fits_runtime = (
            type(fixed_imgsz) is int
            and max(frame.shape[:2]) <= fixed_imgsz
        )
        request_targets_runtime = fixed_imgsz == imgsz
        if (
            runtime.get("runtime_backend") in {"tensorrt_rejected", "tensorrt_failed"}
            or type(fixed_imgsz) is not int
            or fixed_imgsz > imgsz
            or not (source_fits_runtime or request_targets_runtime)
        ):
            return False, None
        if runtime["handle"] is None:
            try:
                from ultralytics import YOLO

                runtime["handle"] = YOLO(str(engine_path), task="detect")
                runtime["runtime_backend"] = "tensorrt"
                runtime["warmed"] = False
                logger.info(
                    "Loaded low-resolution COCO TensorRT runtime",
                    extra={"engine_path": str(engine_path), "fixed_imgsz": fixed_imgsz},
                )
            except Exception as exc:
                runtime["handle"] = None
                runtime["runtime_backend"] = "tensorrt_failed"
                runtime["runtime_fallback_error"] = f"TensorRT load failed: {exc}"
                logger.exception(
                    "Low-resolution TensorRT model load failed; using primary runtime",
                    extra={"engine_path": str(engine_path)},
                )
                return False, None

        try:
            return True, _predict_with_runtime(
                "coco_primary",
                runtime,
                frame,
                conf=conf,
                device=device,
                imgsz=imgsz,
            )
        except Exception as exc:
            old_handle = runtime.get("handle")
            runtime["handle"] = None
            runtime["runtime_backend"] = "tensorrt_failed"
            runtime["runtime_fallback_error"] = f"TensorRT inference failed: {exc}"
            runtime["warmed"] = False
            del old_handle
            logger.exception(
                "Low-resolution TensorRT inference failed; using primary runtime",
                extra={"engine_path": str(engine_path)},
            )
            return False, None


def _warm_configured_low_res_coco_runtime(source_path: Optional[Path] = None) -> bool:
    """Make configured low-resolution inference ready before cameras start."""
    if source_path is None:
        with _MODEL_LOCK:
            state = _MODEL_STATES["coco_primary"]
            active_path = state.get("active_path") if state.get("status") == "ready" else None
        if not active_path:
            return False
        source_path = Path(active_path)
    device = _runtime_device(_configured_local_device())
    used_low_res, _result = _predict_with_low_res_coco_runtime(
        np.zeros((32, 32, 3), dtype=np.uint8),
        source_path=source_path,
        conf=0.25,
        device=device,
        imgsz=16_384,
    )
    return used_low_res


def _warm_configured_fixed_runtime(model_key: ModelKey) -> bool:
    """Deserialize and execute fixed TensorRT runtimes before reporting ready."""
    runtime = _MODEL_RUNTIMES[model_key]
    with runtime["lock"]:
        if runtime.get("runtime_backend") != "tensorrt" or runtime.get("warmed"):
            return False
        try:
            _predict_with_runtime(
                model_key,
                runtime,
                np.zeros((32, 32, 3), dtype=np.uint8),
                conf=0.25,
                device=_runtime_device(_configured_local_device()),
                imgsz=16_384,
                classes=list(runtime.get("fixed_classes") or []) or None,
            )
            if model_key == "ppe_specialist":
                engine_classes = [str(value) for value in runtime["handle"].names.values()]
                if engine_classes != list(runtime.get("fixed_classes") or []):
                    raise RuntimeError("TensorRT engine classes do not match its manifest")
        except Exception as exc:
            logger.exception(
                "TensorRT warm-up failed; activating PyTorch fallback",
                extra={"model_key": model_key, "runtime_path": runtime.get("runtime_path")},
            )
            try:
                _activate_pytorch_fallback(runtime, exc)
            except Exception:
                logger.exception(
                    "TensorRT warm-up fallback failed",
                    extra={"model_key": model_key},
                )
            return False
        return True


def _activate_pytorch_fallback(runtime: dict[str, Any], error: Exception) -> None:
    fallback_path = runtime.get("fallback_path")
    if not fallback_path:
        raise error
    old_handle = runtime.get("handle")
    runtime["handle"] = None
    del old_handle
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("CUDA cache cleanup skipped during TensorRT fallback", exc_info=True)
    from ultralytics import YOLO

    runtime["handle"] = YOLO(str(fallback_path))
    runtime["runtime_path"] = str(fallback_path)
    runtime["runtime_backend"] = "pytorch_fallback"
    runtime["runtime_fallback_error"] = str(error)
    runtime["fallback_path"] = None
    runtime["fixed_imgsz"] = None
    runtime["fixed_classes"] = []
    runtime["fixed_class_groups"] = []
    runtime["current_classes"] = []
    runtime["class_embeddings"] = {}
    runtime["warmed"] = False


def predict(
    model_key: ModelKey,
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: Optional[list[str]] = None,
):
    runtime_device = _runtime_device(device)
    runtime = _MODEL_RUNTIMES[model_key]

    if model_key == "coco_primary":
        with _MODEL_LOCK:
            active_path = _MODEL_STATES[model_key].get("active_path")
        if active_path:
            used_low_res, result = _predict_with_low_res_coco_runtime(
                frame,
                source_path=Path(active_path),
                conf=conf,
                device=runtime_device,
                imgsz=imgsz,
            )
            if used_low_res:
                return result

    with runtime["lock"]:
        if runtime["handle"] is None and runtime.get("runtime_backend") == "lazy":
            with _MODEL_LOCK:
                active_path = _MODEL_STATES[model_key].get("active_path")
            if not active_path:
                raise RuntimeError(f"Model {model_key} has no active lazy-load path")
            _load_runtime(model_key, Path(active_path))
            _sync_state_compat()
        try:
            return _predict_with_runtime(
                model_key,
                runtime,
                frame,
                conf=conf,
                device=runtime_device,
                imgsz=imgsz,
                classes=classes,
            )
        except Exception as exc:
            if runtime.get("runtime_backend") != "tensorrt":
                raise
            logger.exception(
                "TensorRT inference failed; activating PyTorch fallback",
                extra={"model_key": model_key, "runtime_path": runtime.get("runtime_path")},
            )
            _activate_pytorch_fallback(runtime, exc)
            return _predict_with_runtime(
                model_key,
                runtime,
                frame,
                conf=conf,
                device=runtime_device,
                imgsz=imgsz,
                classes=classes,
            )


def _records_from_results(results) -> list[dict[str, Any]]:
    if not results or len(results) == 0:
        return []
    boxes = results[0].boxes
    if boxes is None:
        return []

    # Ultralytics box tensors live on the inference device. Converting cls,
    # confidence, and each coordinate separately forces several CUDA scalar
    # synchronizations per detection. Transfer the compact Nx6/Nx7 tensor to
    # host memory once, then normalize ordinary Python rows.
    rows = boxes.data.tolist()
    result_keypoints = getattr(results[0], "keypoints", None)
    keypoint_rows = (
        result_keypoints.data.tolist()
        if result_keypoints is not None
        else []
    )
    records = []
    for index, row in enumerate(rows):
        record = {
            "class_id": int(row[-1]),
            "confidence": float(row[-2]),
            "bbox": [int(value) for value in row[:4]],
        }
        if index < len(keypoint_rows):
            record["keypoints"] = [
                [float(value) for value in keypoint[:3]]
                for keypoint in keypoint_rows[index]
            ]
        records.append(record)
    return records


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_prompt_synonyms(
    records: list[dict[str, Any]],
    classes: list[str],
    *,
    class_groups: list[str] | None = None,
    iou_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    class_groups = list(class_groups or [])

    def semantic_group(class_id: int) -> str | None:
        if len(class_groups) == len(classes):
            return class_groups[class_id]
        class_name = classes[class_id].lower().replace("_", " ").replace("-", " ").strip()
        return CLASS_TERM_TO_CAPABILITY.get(class_name)

    kept: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda value: float(value["confidence"]), reverse=True):
        class_id = int(record["class_id"])
        if class_id < 0 or class_id >= len(classes):
            kept.append(record)
            continue
        capability = semantic_group(class_id)
        duplicate = False
        if capability:
            for existing in kept:
                existing_class_id = int(existing["class_id"])
                if existing_class_id < 0 or existing_class_id >= len(classes) or existing_class_id == class_id:
                    continue
                if (
                    semantic_group(existing_class_id) == capability
                    and _bbox_iou(record["bbox"], existing["bbox"]) >= iou_threshold
                ):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(record)
    return kept


def _remote_predict_records_jpeg(
    model_key: ModelKey,
    frame_jpeg: bytes,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    metadata = {
        "model_key": model_key,
        "conf": conf,
        "device": device,
        "imgsz": imgsz,
        "classes": classes or [],
    }
    response = _remote_post_jpeg("/api/infer/jpeg", frame_jpeg, params=metadata)
    if response is not None:
        return response.get("detections", [])

    # Keep rolling upgrades safe when the edge reaches an older model server.
    payload = {
        "model_key": model_key,
        "frame_jpeg_b64": base64.b64encode(frame_jpeg).decode("ascii"),
        "conf": conf,
        "device": device,
        "imgsz": imgsz,
        "classes": classes or [],
    }
    return _remote_post("/api/infer", payload).get("detections", [])


def _prepare_remote_inference_frame(
    frame,
    imgsz: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return a contiguous frame with no more pixels than the model consumes."""
    import cv2

    source_shape = frame.shape
    source_height, source_width = source_shape[:2]
    maximum_dimension = max(source_height, source_width)
    target_dimension = max(1, int(imgsz))
    inference_frame = frame
    if maximum_dimension > target_dimension:
        scale = target_dimension / maximum_dimension
        inference_frame = cv2.resize(
            frame,
            (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            ),
        )
    inference_frame = np.ascontiguousarray(inference_frame)
    return inference_frame, inference_frame.shape


def _prepare_remote_inference_jpeg(
    frame,
    imgsz: int,
    *,
    jpeg_quality: int = _REMOTE_JPEG_QUALITY,
) -> tuple[bytes, tuple[int, ...]]:
    """Encode no more pixels than the remote model can consume."""
    import cv2

    inference_frame, inference_shape = _prepare_remote_inference_frame(frame, imgsz)
    ok, buffer = cv2.imencode(
        ".jpg",
        inference_frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("Could not encode frame for remote inference")
    return buffer.tobytes(), inference_shape


def _scale_remote_records_to_source(
    records: list[dict[str, Any]],
    source_shape: tuple[int, ...],
    inference_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    if source_shape[:2] == inference_shape[:2]:
        return records
    source_height, source_width = source_shape[:2]
    inference_height, inference_width = inference_shape[:2]
    scale_x = source_width / inference_width
    scale_y = source_height / inference_height
    scaled: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            item["bbox"] = [
                min(source_width, max(0, round(float(x1) * scale_x))),
                min(source_height, max(0, round(float(y1) * scale_y))),
                min(source_width, max(0, round(float(x2) * scale_x))),
                min(source_height, max(0, round(float(y2) * scale_y))),
            ]
        keypoints = item.get("keypoints")
        if isinstance(keypoints, list):
            item["keypoints"] = [
                [
                    min(source_width, max(0.0, float(keypoint[0]) * scale_x)),
                    min(source_height, max(0.0, float(keypoint[1]) * scale_y)),
                    float(keypoint[2]),
                ]
                for keypoint in keypoints
                if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 3
            ]
        scaled.append(item)
    return scaled


def predict_records(
    model_key: ModelKey,
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    if is_remote_inference_enabled():
        if not _REMOTE_JOB_ADMISSION.acquire(
            timeout=_REMOTE_JOB_ADMISSION_WAIT_SECONDS
        ):
            raise RemoteInferenceOverloadedError(
                "Remote inference is at its bounded concurrency limit"
            )
        try:
            frame_jpeg, inference_shape = _prepare_remote_inference_jpeg(frame, imgsz)
            records = _remote_predict_records_jpeg(
                model_key,
                frame_jpeg,
                conf=conf,
                device=device,
                imgsz=imgsz,
                classes=classes,
            )
            return _scale_remote_records_to_source(
                records,
                frame.shape,
                inference_shape,
            )
        finally:
            _REMOTE_JOB_ADMISSION.release()

    records = _records_from_results(
        predict(model_key, frame, conf=conf, device=device, imgsz=imgsz, classes=classes)
    )
    if model_key in _OPEN_VOCAB_MODEL_KEYS:
        active_classes = classes if classes is not None else _default_open_vocab_classes(model_key)
        class_groups = list(_MODEL_RUNTIMES[model_key].get("fixed_class_groups") or [])
        return _deduplicate_prompt_synonyms(records, active_classes, class_groups=class_groups)
    return records


def predict_record_batches(frame, requests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Run multiple record-producing models against one immutable frame."""
    if not requests:
        return {}
    if len(requests) > 8:
        raise ValueError("At most eight model requests may share one frame")

    normalized = []
    request_ids = set()
    for index, request in enumerate(requests):
        model_key = request.get("model_key")
        if model_key not in MODEL_DEFINITIONS:
            raise KeyError(f"Unknown model key: {model_key}")
        request_id = str(request.get("request_id") or f"{model_key}:{index}")
        if request_id in request_ids:
            raise ValueError(f"Duplicate inference request ID: {request_id}")
        request_ids.add(request_id)
        normalized.append({
            "request_id": request_id,
            "model_key": model_key,
            "conf": float(request.get("conf", 0.35)),
            "device": str(request.get("device", "cuda")),
            "imgsz": int(request.get("imgsz", 960)),
            "classes": list(request.get("classes") or []),
        })

    remote_inference = is_remote_inference_enabled()
    if len(normalized) == 1 and not remote_inference:
        item = normalized[0]
        return {
            item["request_id"]: predict_records(
                item["model_key"],
                frame,
                conf=item["conf"],
                device=item["device"],
                imgsz=item["imgsz"],
                classes=item["classes"],
            )
        }

    if not remote_inference:
        return {
            item["request_id"]: predict_records(
                item["model_key"],
                frame,
                conf=item["conf"],
                device=item["device"],
                imgsz=item["imgsz"],
                classes=item["classes"],
            )
            for item in normalized
        }

    maximum_imgsz = max(item["imgsz"] for item in normalized)
    resize_required = max(frame.shape[:2]) > maximum_imgsz
    paired_request = len(normalized) == 2
    if not _REMOTE_JOB_ADMISSION.acquire(
        timeout=_REMOTE_JOB_ADMISSION_WAIT_SECONDS
    ):
        raise RemoteInferenceOverloadedError(
            "Remote inference is at its bounded concurrency limit"
        )
    try:
        response = None
        inference_shape = frame.shape
        if _remote_raw_transport_enabled():
            inference_frame, inference_shape = _prepare_remote_inference_frame(
                frame,
                maximum_imgsz,
            )
            response = _remote_post_raw_batch(
                "/api/infer/raw/batch",
                inference_frame,
                batch=normalized,
            )
        if response is not None:
            results = response.get("results")
            if not isinstance(results, dict) or set(results) != request_ids:
                raise RuntimeError("Model server returned an incomplete inference batch")
            return {
                request_id: _scale_remote_records_to_source(
                    records,
                    frame.shape,
                    inference_shape,
                )
                for request_id, records in results.items()
            }

        # Preserve the existing byte path for camera frames that already fit.
        # Oversized grouped frames use the smallest higher quality that retained
        # operational class presence across the Jetson validation corpus.
        frame_jpeg, inference_shape = _prepare_remote_inference_jpeg(
            frame,
            maximum_imgsz,
            jpeg_quality=(
                _RESIZED_GROUPED_REMOTE_JPEG_QUALITY
                if resize_required
                else _REMOTE_JPEG_QUALITY
            ),
        )
        response = _remote_post_jpeg_batch(
            "/api/infer/jpeg/batch",
            frame_jpeg,
            batch=normalized,
        )
        if response is not None:
            results = response.get("results")
            if not isinstance(results, dict) or set(results) != request_ids:
                raise RuntimeError("Model server returned an incomplete inference batch")
            return {
                request_id: _scale_remote_records_to_source(
                    records,
                    frame.shape,
                    inference_shape,
                )
                for request_id, records in results.items()
            }

        if paired_request:
            # Rolling-upgrade fallback for a model server without the batch route.
            futures = [
                _REMOTE_PAIR_EXECUTOR.submit(
                    _remote_predict_records_jpeg,
                    item["model_key"],
                    frame_jpeg,
                    conf=item["conf"],
                    device=item["device"],
                    imgsz=item["imgsz"],
                    classes=item["classes"],
                )
                for item in normalized
            ]
            return {
                item["request_id"]: _scale_remote_records_to_source(
                    future.result(),
                    frame.shape,
                    inference_shape,
                )
                for item, future in zip(normalized, futures)
            }
        return {
            item["request_id"]: _scale_remote_records_to_source(
                _remote_predict_records_jpeg(
                    item["model_key"],
                    frame_jpeg,
                    conf=item["conf"],
                    device=item["device"],
                    imgsz=item["imgsz"],
                    classes=item["classes"],
                ),
                frame.shape,
                inference_shape,
            )
            for item in normalized
        }
    finally:
        _REMOTE_JOB_ADMISSION.release()


def predict_plate_records(
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
) -> list[dict[str, Any]]:
    if not is_remote_inference_enabled():
        raise RuntimeError("ANPR inference must run through the model server")

    if not _REMOTE_JOB_ADMISSION.acquire(
        timeout=_REMOTE_JOB_ADMISSION_WAIT_SECONDS
    ):
        raise RemoteInferenceOverloadedError(
            "Remote inference is at its bounded concurrency limit"
        )
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode frame for remote ANPR")
        frame_jpeg = buffer.tobytes()
        metadata = {
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
        }
        response = _remote_post_jpeg(
            "/api/anpr/jpeg",
            frame_jpeg,
            params=metadata,
        )
        if response is not None:
            return response.get("plates", [])

        payload = {
            "frame_jpeg_b64": base64.b64encode(frame_jpeg).decode("ascii"),
            **metadata,
        }
        return _remote_post("/api/anpr", payload).get("plates", [])
    finally:
        _REMOTE_JOB_ADMISSION.release()
