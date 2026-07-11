"""Model registry, lifecycle management, and install jobs."""

from __future__ import annotations

import logging
import os
import base64
import shutil
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np

from capability_registry import ALL_PPE_PROMPT_TERMS, ModelKey
from constants import PROJECT_ROOT

logger = logging.getLogger("rakshak_lens.models")

MODELS_ROOT = PROJECT_ROOT / "models"
TMP_MODELS_ROOT = Path(tempfile.gettempdir()) / "rakshak-lens-models"
_JOB_POLL_FINAL_STATES = {"ready", "failed"}
_REMOTE_SESSION_LOCAL = threading.local()
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
_COCO_MODEL_VARIANT = os.environ.get("SAFETYLENS_COCO_MODEL", "yolo26n").strip()
if _COCO_MODEL_VARIANT not in _COCO_MODEL_OPTIONS:
    logger.warning(
        "Unknown COCO model variant configured; falling back to nano",
        extra={"configured_model": _COCO_MODEL_VARIANT},
    )
    _COCO_MODEL_VARIANT = "yolo26n"
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
_MODEL_LOCK = threading.RLock()
_MODEL_RUNTIMES: dict[ModelKey, dict[str, Any]] = {
    model_key: {
        "handle": None,
        "lock": threading.Lock(),
        "loaded_path": None,
        "current_classes": [],
    }
    for model_key in MODEL_DEFINITIONS
}
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


def is_remote_inference_enabled() -> bool:
    return bool(_remote_settings()["url"])


def _remote_headers(token: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _remote_session():
    import requests

    session = getattr(_REMOTE_SESSION_LOCAL, "session", None)
    if session is None:
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


def list_models(
    *,
    timeout_seconds: float | None = None,
    allow_cached: bool = False,
    cache_ttl_seconds: float | None = None,
) -> list[dict[str, Any]]:
    if is_remote_inference_enabled():
        now = time.time()
        if allow_cached:
            cached = _fresh_remote_model_cache(
                now,
                REMOTE_MODEL_STATUS_CACHE_TTL_SECONDS if cache_ttl_seconds is None else cache_ttl_seconds,
            )
            if cached is not None:
                return cached
            cached_failure = _fresh_remote_failure_cache(now)
            if cached_failure is not None:
                return cached_failure
        try:
            models = _remote_get("/api/models", timeout_seconds=timeout_seconds).get("models", [])
            with _REMOTE_MODEL_CACHE_LOCK:
                _REMOTE_MODEL_CACHE.update({
                    "models": deepcopy(models),
                    "updated_at": time.time(),
                    "failure_error": None,
                    "failure_at": 0.0,
                })
            return models
        except Exception as exc:
            logger.warning("Remote model list unavailable", extra={"model_server_url": _remote_settings()["url"], "error": str(exc)})
            with _REMOTE_MODEL_CACHE_LOCK:
                _REMOTE_MODEL_CACHE.update({
                    "failure_error": str(exc),
                    "failure_at": time.time(),
                })
            return _remote_unavailable_models(exc)
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


def missing_model_keys(
    model_keys: list[str],
    *,
    timeout_seconds: float | None = None,
    allow_cached: bool = False,
) -> list[ModelKey]:
    if not model_keys:
        return []
    if is_remote_inference_enabled():
        try:
            models = {
                item.get("model_key"): item
                for item in list_models(timeout_seconds=timeout_seconds, allow_cached=allow_cached)
            }
            return [
                model_key  # type: ignore[misc]
                for model_key in model_keys
                if not models.get(model_key, {}).get("is_ready")
            ]
        except Exception as exc:
            logger.warning("Remote model readiness check failed", extra={"model_server_url": _remote_settings()["url"], "error": str(exc)})
            return list(model_keys)  # type: ignore[return-value]
    missing: list[ModelKey] = []
    with _MODEL_LOCK:
        for model_key in model_keys:
            if model_key not in MODEL_DEFINITIONS:
                continue
            if _MODEL_STATES[model_key]["status"] != "ready":
                missing.append(model_key)  # type: ignore[arg-type]
    return missing


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


def _set_open_vocab_classes(handle: YOLO, classes: list[str]):
    tmp_models_dir = TMP_MODELS_ROOT
    tmp_models_dir.mkdir(parents=True, exist_ok=True)
    copied_text_encoders = _copy_open_vocab_text_encoder_assets(tmp_models_dir)
    if not copied_text_encoders:
        raise RuntimeError("MobileCLIP text encoder artifact not found for open-vocabulary model setup")

    original_cwd = os.getcwd()
    try:
        handle.to("cpu")
        os.chdir(str(tmp_models_dir))
        handle.set_classes(classes)
    finally:
        os.chdir(original_cwd)


def _load_runtime(model_key: ModelKey, source_path: Path) -> None:
    runtime = _MODEL_RUNTIMES[model_key]
    if runtime["handle"] is not None and runtime["loaded_path"] == str(source_path):
        return

    if model_key == "face_recognition":
        # InsightFace owns its own runtime and lazy-loads in face_recognition.py.
        # The model manager only tracks whether the expected model directory exists.
        runtime["handle"] = "insightface"
        runtime["loaded_path"] = str(source_path)
        runtime["current_classes"] = []
        return

    from ultralytics import YOLO

    if model_key not in _OPEN_VOCAB_MODEL_KEYS:
        handle = YOLO(str(source_path))
        runtime["handle"] = handle
        runtime["loaded_path"] = str(source_path)
        runtime["current_classes"] = []
        runtime["warmed"] = False
        return

    local_model = _copy_to_local_fs(model_key, source_path)
    handle = YOLO(str(local_model))
    if model_key == "ppe_specialist":
        initial_classes = list(ALL_PPE_PROMPT_TERMS)
    else:
        initial_classes = list(_DEFAULT_LONG_TAIL_PROMPTS)

    device = _runtime_device(_configured_local_device())
    _set_open_vocab_classes(handle, initial_classes)
    handle.to(device)
    runtime["handle"] = handle
    runtime["loaded_path"] = str(source_path)
    runtime["current_classes"] = initial_classes
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
        return _remote_post("/api/models/install", {"model_keys": model_keys})

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
        return _remote_post(f"/api/models/install/{job_id}/retry", {})
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
    handle = runtime["handle"]
    if handle is None:
        raise RuntimeError(f"Model {model_key} is not loaded")

    with runtime["lock"]:
        if not runtime.get("warmed"):
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            handle.predict(dummy, verbose=False, device=runtime_device, imgsz=imgsz)
            runtime["warmed"] = True
        if model_key in _OPEN_VOCAB_MODEL_KEYS and classes is not None:
            requested_classes = [value for value in classes if isinstance(value, str) and value]
            if runtime["current_classes"] != requested_classes:
                _set_open_vocab_classes(handle, requested_classes)
                handle.to(runtime_device)
                runtime["current_classes"] = requested_classes
        return handle.predict(frame, conf=conf, verbose=False, device=runtime_device, imgsz=imgsz)


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
        import cv2
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode frame for remote inference")
        payload = {
            "model_key": model_key,
            "frame_jpeg_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "conf": conf,
            "device": device,
            "imgsz": imgsz,
            "classes": classes or [],
        }
        return _remote_post("/api/infer", payload).get("detections", [])

    return _records_from_results(
        predict(model_key, frame, conf=conf, device=device, imgsz=imgsz, classes=classes)
    )


def predict_plate_records(
    frame,
    *,
    conf: float,
    device: str,
    imgsz: int,
) -> list[dict[str, Any]]:
    if not is_remote_inference_enabled():
        raise RuntimeError("ANPR inference must run through the model server")

    import cv2
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode frame for remote ANPR")
    payload = {
        "frame_jpeg_b64": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "conf": conf,
        "device": device,
        "imgsz": imgsz,
    }
    return _remote_post("/api/anpr", payload).get("plates", [])
