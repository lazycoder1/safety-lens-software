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
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from capability_registry import ALL_PPE_PROMPT_TERMS, CLASS_TERM_TO_CAPABILITY, ModelKey
from constants import (
    MODEL_METADATA_BREAKER_INITIAL_SECONDS,
    MODEL_METADATA_BREAKER_MAX_SECONDS,
    MODEL_METADATA_LOG_REMINDER_SECONDS,
    MODEL_METADATA_TIMEOUT_SECONDS,
    MODEL_METADATA_TTL_SECONDS,
    MODEL_SERVER_TIMEOUT_SECONDS,
    MODEL_SERVER_TOKEN,
    MODEL_SERVER_URL,
    PROJECT_ROOT,
)
from tensorrt_engine import validate_engine

logger = logging.getLogger("safetylens.models")

MODELS_ROOT = PROJECT_ROOT / "models"
TMP_MODELS_ROOT = Path(tempfile.gettempdir()) / "safetylens-models"
_JOB_POLL_FINAL_STATES = {"ready", "failed"}


MODEL_DEFINITIONS: dict[ModelKey, dict[str, Any]] = {
    "coco_primary": {
        "model_key": "coco_primary",
        "display_name": "COCO Primary",
        "filename": "yolo26m.pt",
        "local_path": MODELS_ROOT / "coco_primary" / "yolo26m.pt",
        "legacy_paths": [PROJECT_ROOT / "yolo26m.pt", PROJECT_ROOT.parent / "yolo26m.pt"],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
        "warmup_behavior": "Full-frame COCO detect warmup",
        "shared_asset_key": "yolo26m",
    },
    "ppe_specialist": {
        "model_key": "ppe_specialist",
        "display_name": "PPE Specialist",
        "filename": "yoloe-11s-seg.pt",
        "local_path": MODELS_ROOT / "yoloe_open_vocab" / "yoloe-11s-seg.pt",
        "legacy_paths": [PROJECT_ROOT / "yoloe-11s-seg.pt"],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt",
        "warmup_behavior": "Open-vocab warmup with stable PPE prompts",
        "shared_asset_key": "yoloe-11s-seg",
    },
    "yoloe_long_tail": {
        "model_key": "yoloe_long_tail",
        "display_name": "YOLOE Long-Tail",
        "filename": "yoloe-11s-seg.pt",
        "local_path": MODELS_ROOT / "yoloe_open_vocab" / "yoloe-11s-seg.pt",
        "legacy_paths": [PROJECT_ROOT / "yoloe-11s-seg.pt"],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt",
        "warmup_behavior": "Open-vocab warmup with long-tail prompts",
        "shared_asset_key": "yoloe-11s-seg",
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
        "filename": "yolo11n-pose.pt",
        "local_path": MODELS_ROOT / "pose_specialist" / "yolo11n-pose.pt",
        "legacy_paths": [PROJECT_ROOT / "yolo11n-pose.pt"],
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt",
        "warmup_behavior": "Pose estimation warmup",
        "shared_asset_key": "yolo11n-pose",
    },
}

_DEFAULT_LONG_TAIL_PROMPTS = ["fire", "smoke", "flames"]
_OPEN_VOCAB_MODEL_KEYS = {"ppe_specialist", "yoloe_long_tail"}
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
_ACTIVE_JOB_ID: str | None = None
_REMOTE_SESSION_LOCAL = threading.local()
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
    return bool(MODEL_SERVER_URL)


def _remote_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MODEL_SERVER_TOKEN:
        headers["Authorization"] = f"Bearer {MODEL_SERVER_TOKEN}"
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


def _remote_get(path: str) -> dict[str, Any]:
    response = _remote_session().get(
        f"{MODEL_SERVER_URL}{path}",
        headers=_remote_headers(),
        timeout=MODEL_SERVER_TIMEOUT_SECONDS,
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
    response = _remote_session().get(
        f"{MODEL_SERVER_URL}/api/models",
        headers=_remote_headers(),
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
    response = _remote_session().post(
        f"{MODEL_SERVER_URL}{path}",
        json=payload,
        headers=_remote_headers(),
        timeout=MODEL_SERVER_TIMEOUT_SECONDS,
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
    headers = _remote_headers()
    headers["Content-Type"] = "image/jpeg"
    response = _remote_session().post(
        f"{MODEL_SERVER_URL}{path}",
        params=params,
        data=frame_jpeg,
        headers=headers,
        timeout=MODEL_SERVER_TIMEOUT_SECONDS,
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
    headers = _remote_headers()
    headers["Content-Type"] = "image/jpeg"
    headers["X-Rakshak-Inference-Batch"] = json.dumps(batch, separators=(",", ":"))
    response = _remote_session().post(
        f"{MODEL_SERVER_URL}{path}",
        data=frame_jpeg,
        headers=headers,
        timeout=MODEL_SERVER_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        try:
            if response.json().get("detail") == "Not Found":
                return None
        except (AttributeError, TypeError, ValueError):
            pass
    response.raise_for_status()
    return response.json()


def _now() -> float:
    return time.time()


def _resolve_mobileclip_source() -> Path | None:
    candidates = [
        PROJECT_ROOT / "backend" / "mobileclip_blt.ts",
        Path(__file__).parent / "mobileclip_blt.ts",
        PROJECT_ROOT / "mobileclip_blt.ts",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    }


def list_models() -> list[dict[str, Any]]:
    if is_remote_inference_enabled():
        return _get_remote_model_catalog()
    with _MODEL_LOCK:
        return [_serialize_model_state(model_key) for model_key in MODEL_DEFINITIONS]


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


def get_install_job(job_id: str) -> dict[str, Any] | None:
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


def _resolve_existing_path(model_key: ModelKey) -> Path | None:
    definition = MODEL_DEFINITIONS[model_key]
    if definition["local_path"].exists():
        return definition["local_path"]
    for legacy_path in definition.get("legacy_paths", []):
        if legacy_path.exists():
            return legacy_path
    return None


def _copy_to_local_fs(model_key: ModelKey, source_path: Path) -> Path:
    TMP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    target = TMP_MODELS_ROOT / f"{model_key}-{source_path.name}"
    if not target.exists() or target.stat().st_size != source_path.stat().st_size:
        shutil.copy2(str(source_path), str(target))
    return target


def _set_open_vocab_classes(handle: Any, classes: list[str]):
    tmp_models_dir = TMP_MODELS_ROOT
    tmp_models_dir.mkdir(parents=True, exist_ok=True)
    mobileclip_source = _resolve_mobileclip_source()
    if mobileclip_source is None:
        raise RuntimeError("mobileclip_blt.ts not found for open-vocabulary model setup")
    local_mobileclip = tmp_models_dir / "mobileclip_blt.ts"
    if not local_mobileclip.exists() or local_mobileclip.stat().st_size != mobileclip_source.stat().st_size:
        shutil.copy2(str(mobileclip_source), str(local_mobileclip))

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

    from config_manager import get_config
    from ultralytics import YOLO

    device = get_config()["global"]["device"]

    if model_key in ("coco_primary", "pose_specialist"):
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
            engine_classes = [str(value) for value in handle.names.values()]
            if engine_classes != fixed_classes:
                raise RuntimeError("TensorRT engine classes do not match its manifest")
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
            _load_runtime(model_key, existing_path)
            with _MODEL_LOCK:
                _set_model_state(model_key, status="ready", error=None, active_path=existing_path, job_id=None)
        except Exception as exc:
            logger.exception("Model load failed during initialization", extra={"model_key": model_key})
            with _MODEL_LOCK:
                _set_model_state(model_key, status="failed", error=str(exc), active_path=existing_path, job_id=None)
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


def _verify_asset(path: Path, total_bytes: int | None):
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
                with _MODEL_LOCK:
                    _set_model_state(model_key, status="ready", error=None, active_path=active_path, job_id=None)
                _update_job(job_id, stage="warming_up", current_model_key=model_key)

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
    runtime["warmed"] = False


def predict(
    model_key: ModelKey,
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: list[str] | None = None,
):
    runtime = _MODEL_RUNTIMES[model_key]

    with runtime["lock"]:
        try:
            return _predict_with_runtime(
                model_key,
                runtime,
                frame,
                conf=conf,
                device=device,
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
                device=device,
                imgsz=imgsz,
                classes=classes,
            )


def _records_from_results(results) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not results or len(results) == 0:
        return records
    boxes = results[0].boxes
    if boxes is None:
        return records
    for box in boxes:
        records.append({
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": list(map(int, box.xyxy[0])),
        })
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


def predict_records(
    model_key: ModelKey,
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
    classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    if is_remote_inference_enabled():
        import cv2
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode frame for remote inference")
        return _remote_predict_records_jpeg(
            model_key,
            buffer.tobytes(),
            conf=conf,
            device=device,
            imgsz=imgsz,
            classes=classes,
        )

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

    if len(normalized) == 1:
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

    if not is_remote_inference_enabled():
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

    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode frame for remote inference")
    frame_jpeg = buffer.tobytes()
    response = _remote_post_jpeg_batch(
        "/api/infer/jpeg/batch",
        frame_jpeg,
        batch=normalized,
    )
    if response is not None:
        results = response.get("results")
        if not isinstance(results, dict) or set(results) != request_ids:
            raise RuntimeError("Model server returned an incomplete inference batch")
        return results

    return {
        item["request_id"]: _remote_predict_records_jpeg(
            item["model_key"],
            frame_jpeg,
            conf=item["conf"],
            device=item["device"],
            imgsz=item["imgsz"],
            classes=item["classes"],
        )
        for item in normalized
    }
