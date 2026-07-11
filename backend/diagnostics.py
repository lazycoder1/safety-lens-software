"""
Health, diagnostics bundle creation, and retention cleanup helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import alert_store
import audit_store
import db
import licensing
import model_manager
import state
from config_manager import get_config, get_redacted_config
from logging_config import LOGS_DIR
from mjpeg_fanout import stream_fanout
from runtime_storage import STATE_DIR
from secret_redaction import redact_text_secrets

logger = logging.getLogger("rakshak_lens.diagnostics")

DIAGNOSTICS_DIR = STATE_DIR / "diagnostics"
APP_START_TIME = time.time()

_health_cache: dict | None = None
_health_cache_time: float = 0.0
_HEALTH_CACHE_TTL = 5.0
_health_cache_lock = __import__("threading").Lock()


def _registered_worker_is_alive(
    workers: dict,
    camera_id: str,
) -> bool:
    """Return actual thread liveness, not mere registry membership."""
    ownership = workers.get(camera_id)
    if ownership is None:
        return False
    try:
        return bool(ownership[0].is_alive())
    except (AttributeError, IndexError, TypeError):
        return False


def _diagnostic_secret_values(config: dict) -> list[str]:
    values = [
        config.get("database", {}).get("url"),
        config.get("telegram", {}).get("bot_token"),
        config.get("email", {}).get("smtp_pass"),
        config.get("webhook", {}).get("url"),
        *config.get("webhook", {}).get("headers", {}).values(),
    ]
    for camera in config.get("cameras", {}).values():
        values.extend(
            [camera.get("username"), camera.get("password"), camera.get("video")]
        )
    return [str(value) for value in values if value not in (None, "")]


def build_health_snapshot() -> dict:
    global _health_cache, _health_cache_time
    now = time.time()
    if _health_cache is not None and (now - _health_cache_time) < _HEALTH_CACHE_TTL:
        return _health_cache.copy()

    # Serialize rebuilds so concurrent health polls don't all do expensive work.
    with _health_cache_lock:
        # Re-check after acquiring lock — another thread may have refreshed.
        now = time.time()
        if _health_cache is not None and (now - _health_cache_time) < _HEALTH_CACHE_TTL:
            return _health_cache.copy()

        cfg = get_config()
        retention = cfg.get("retention", {})
        license_status = licensing.get_status()
        db_ok = db.check_connection()
        storage = {
            "logs": _path_usage(LOGS_DIR),
            "snapshots": alert_store.get_snapshot_usage(),
            "diagnostics": _path_usage(DIAGNOSTICS_DIR),
        }
        free_bytes = shutil.disk_usage(Path(__file__).parent).free
        models = model_manager.list_models()
        model_metadata = model_manager.remote_model_metadata_health()

        cameras = []
        enabled_cameras = 0
        running_cameras = 0
        now = time.time()
        missing_vlm_companions = 0
        active_camera_outages = 0
        for cam_id, cam in cfg.get("cameras", {}).items():
            enabled = bool(cam.get("enabled", True))
            worker_running = _registered_worker_is_alive(state.camera_threads, cam_id)
            vlm_expected = cam.get("demo") == "yolo+vlm"
            vlm_worker_running = _registered_worker_is_alive(state.vlm_threads, cam_id)
            last_frame_at = state.camera_frame_updated_at.get(cam_id)
            last_frame_age = None if last_frame_at is None else now - last_frame_at
            frame_available = (
                state.camera_frames.get(cam_id) is not None
                and last_frame_age is not None
                and last_frame_age <= state.CAMERA_FRAME_STALE_SECONDS
            )
            runtime_status = state.camera_runtime_status.get(cam_id, "offline")
            if worker_running and not frame_available and runtime_status == "running":
                runtime_status = "stale"
            stream_stats = stream_fanout.stats(cam_id)
            connection_health = state.get_camera_connection_health(cam_id)
            if enabled:
                enabled_cameras += 1
            if enabled and worker_running:
                running_cameras += 1
            if enabled and vlm_expected and not vlm_worker_running:
                missing_vlm_companions += 1
            if enabled and connection_health["outageActive"]:
                active_camera_outages += 1
            cameras.append(
                {
                    "id": cam_id,
                    "name": cam.get("name", cam_id),
                    "enabled": enabled,
                    "workerRunning": worker_running,
                    "vlmExpected": vlm_expected,
                    "vlmWorkerRunning": vlm_worker_running,
                    "frameAvailable": frame_available,
                    "frameFresh": frame_available,
                    "lastFrameAgeSeconds": None if last_frame_age is None else round(last_frame_age, 1),
                    "runtimeStatus": runtime_status,
                    "connection": connection_health,
                    "inference": state.get_camera_inference_health(cam_id),
                    "detectionsCount": len(state.camera_detections.get(cam_id, [])),
                    "stream": {
                        "sequence": stream_stats["sequence"],
                        "subscribers": stream_stats["subscribers"],
                        "frameAvailable": stream_stats["has_frame"],
                        "frameAgeSeconds": (
                            round(stream_stats["frame_age_seconds"], 3)
                            if stream_stats["frame_age_seconds"] is not None
                            else None
                        ),
                    },
                }
            )

        models = model_manager.list_models_for_status()

        status = "ok"
        reasons: list[str] = []
        if not db_ok:
            status = "error"
            reasons.append("database unavailable")
        if license_status.state == licensing.LicenseState.SUSPENDED and status != "error":
            status = "degraded"
            reasons.append("license suspended")
        if any(model.get("status") == "remote_unavailable" for model in models) and status == "ok":
            status = "degraded"
            reasons.append("model server unavailable")
        if enabled_cameras and running_cameras < enabled_cameras:
            if status == "ok":
                status = "degraded"
            reasons.append("one or more enabled cameras are not running")
        if missing_vlm_companions:
            if status == "ok":
                status = "degraded"
            reasons.append("one or more enabled VLM camera companions are not running")
        if active_camera_outages:
            if status == "ok":
                status = "degraded"
            reasons.append("one or more cameras have an active connection outage")
        if (
            model_metadata.get("enabled") is True
            and model_metadata.get("status") != "healthy"
        ):
            if status == "ok":
                status = "degraded"
            reasons.append("remote model metadata unavailable")

        result = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptimeSeconds": int(max(0, time.time() - APP_START_TIME)),
            "reasons": reasons,
            "database": {"ok": db_ok},
            "license": license_status.to_public_dict(),
            "models": models,
            "modelMetadata": model_metadata,
            "cameras": cameras,
            "streamFanout": stream_fanout.operational_stats(),
            "storage": {
                **storage,
                "freeBytes": free_bytes,
            },
            "retention": retention,
        }
        _health_cache = result
        _health_cache_time = time.time()
        return result


def create_diagnostics_bundle() -> Path:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(DIAGNOSTICS_DIR, 0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = DIAGNOSTICS_DIR / f"rakshak-lens-diagnostics-{timestamp}.zip"

    recent_alerts = alert_store.get_alerts(limit=200)
    recent_audit = audit_store.get_recent(limit=200)
    health = build_health_snapshot()
    license_status = licensing.get_status().to_public_dict()
    stats = alert_store.get_stats()
    config = get_redacted_config()
    secret_values = _diagnostic_secret_values(get_config())

    fd, tmp_name = tempfile.mkstemp(prefix=".diagnostics-", suffix=".zip", dir=DIAGNOSTICS_DIR)
    os.fchmod(fd, 0o600)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zf:
            zf.writestr("health.json", json.dumps(health, indent=2))
            zf.writestr("license.json", json.dumps(license_status, indent=2))
            zf.writestr("alerts-stats.json", json.dumps(stats, indent=2))
            zf.writestr("recent-alerts.json", json.dumps(recent_alerts, indent=2))
            zf.writestr("audit-log.json", json.dumps(recent_audit, indent=2))
            zf.writestr("config.redacted.json", json.dumps(config, indent=2))

            if LOGS_DIR.exists():
                for path in sorted(LOGS_DIR.glob("*.log*")):
                    if not path.is_file():
                        continue
                    with path.open("r", encoding="utf-8", errors="replace") as source:
                        with zf.open(f"logs/{path.name}", "w") as destination:
                            for line in source:
                                safe_line = redact_text_secrets(line, secret_values)
                                destination.write(safe_line.encode("utf-8"))
        with tmp_path.open("rb") as bundle_file:
            os.fsync(bundle_file.fileno())
        os.replace(tmp_path, bundle_path)
        os.chmod(bundle_path, 0o600)
        directory_fd = os.open(DIAGNOSTICS_DIR, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("Diagnostics bundle created", extra={"path": str(bundle_path)})
    return bundle_path


def cleanup_diagnostics_files(*, retention_days: int, max_bytes: int) -> dict:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(DIAGNOSTICS_DIR, 0o700)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(retention_days, 0))
    deleted_files = 0
    reclaimed_bytes = 0

    files = [path for path in DIAGNOSTICS_DIR.iterdir() if path.is_file()]
    for path in files:
        os.chmod(path, 0o600)
    files.sort(key=lambda path: path.stat().st_mtime)

    for path in files:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified >= cutoff:
            continue
        reclaimed_bytes += path.stat().st_size
        deleted_files += 1
        path.unlink(missing_ok=True)

    usage = _path_usage(DIAGNOSTICS_DIR)
    files = [path for path in DIAGNOSTICS_DIR.iterdir() if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime)
    for path in files:
        if usage["bytes"] <= max_bytes:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        deleted_files += 1
        reclaimed_bytes += size
        usage["bytes"] = max(0, usage["bytes"] - size)

    final_usage = _path_usage(DIAGNOSTICS_DIR)
    return {
        "deletedFiles": deleted_files,
        "reclaimedBytes": reclaimed_bytes,
        "remainingBytes": final_usage["bytes"],
        "remainingFiles": final_usage["files"],
    }


def run_retention_cleanup() -> dict:
    cfg = get_config()
    retention = cfg.get("retention", {})

    snapshot_result = alert_store.cleanup_snapshots(
        retention_days=int(retention.get("snapshot_retention_days", 30)),
        max_bytes=int(retention.get("snapshot_max_bytes", 10 * 1024 * 1024 * 1024)),
        orphan_grace_hours=int(retention.get("orphan_grace_hours", 24)),
    )
    diagnostics_result = cleanup_diagnostics_files(
        retention_days=int(retention.get("diagnostics_retention_days", 14)),
        max_bytes=int(retention.get("diagnostics_max_bytes", 1 * 1024 * 1024 * 1024)),
    )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshots": snapshot_result,
        "diagnostics": diagnostics_result,
    }
    logger.info("Retention cleanup completed", extra={"action": "retention_cleanup"})
    return result


async def retention_cleanup_loop() -> None:
    cfg = get_config()
    retention = cfg.get("retention", {})
    interval_hours = max(int(retention.get("cleanup_interval_hours", 24)), 1)

    await asyncio.to_thread(run_retention_cleanup)
    while True:
        await asyncio.sleep(interval_hours * 60 * 60)
        try:
            await asyncio.to_thread(run_retention_cleanup)
        except Exception:
            logger.exception("Retention cleanup loop failed")


def _path_usage(path: Path) -> dict:
    total_bytes = 0
    total_files = 0
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                total_files += 1
                total_bytes += child.stat().st_size
    return {
        "dir": str(path),
        "files": total_files,
        "bytes": total_bytes,
    }
