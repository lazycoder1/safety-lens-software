"""
Health, diagnostics bundle creation, and retention cleanup helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
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

logger = logging.getLogger("safetylens.diagnostics")

DIAGNOSTICS_DIR = Path(__file__).parent / "diagnostics"
APP_START_TIME = time.time()

_health_cache: dict | None = None
_health_cache_time: float = 0.0
_HEALTH_CACHE_TTL = 5.0
_health_cache_lock = __import__("threading").Lock()


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

        cameras = []
        enabled_cameras = 0
        running_cameras = 0
        for cam_id, cam in cfg.get("cameras", {}).items():
            enabled = bool(cam.get("enabled", True))
            worker_running = cam_id in state.camera_threads
            frame_available = state.camera_frames.get(cam_id) is not None
            stream_stats = stream_fanout.stats(cam_id)
            if enabled:
                enabled_cameras += 1
            if enabled and worker_running:
                running_cameras += 1
            cameras.append(
                {
                    "id": cam_id,
                    "name": cam.get("name", cam_id),
                    "enabled": enabled,
                    "workerRunning": worker_running,
                    "frameAvailable": frame_available,
                    "runtimeStatus": state.camera_runtime_status.get(cam_id, "offline"),
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

        status = "ok"
        reasons: list[str] = []
        if not db_ok:
            status = "error"
            reasons.append("database unavailable")
        if license_status.state == licensing.LicenseState.SUSPENDED and status != "error":
            status = "degraded"
            reasons.append("license suspended")
        if enabled_cameras and running_cameras < enabled_cameras and status == "ok":
            status = "degraded"
            reasons.append("one or more enabled cameras are not running")

        result = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptimeSeconds": int(max(0, time.time() - APP_START_TIME)),
            "reasons": reasons,
            "database": {"ok": db_ok},
            "license": license_status.to_public_dict(),
            "models": model_manager.list_models(),
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
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = DIAGNOSTICS_DIR / f"safetylens-diagnostics-{timestamp}.zip"

    recent_alerts = alert_store.get_alerts(limit=200)
    recent_audit = audit_store.get_recent(limit=200)
    health = build_health_snapshot()
    license_status = licensing.get_status().to_public_dict()
    stats = alert_store.get_stats()
    config = get_redacted_config()

    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("health.json", json.dumps(health, indent=2))
        zf.writestr("license.json", json.dumps(license_status, indent=2))
        zf.writestr("alerts-stats.json", json.dumps(stats, indent=2))
        zf.writestr("recent-alerts.json", json.dumps(recent_alerts, indent=2))
        zf.writestr("audit-log.json", json.dumps(recent_audit, indent=2))
        zf.writestr("config.redacted.json", json.dumps(config, indent=2))

        if LOGS_DIR.exists():
            for path in sorted(LOGS_DIR.glob("safetylens.log*")):
                if path.is_file():
                    zf.write(path, arcname=f"logs/{path.name}")

    logger.info("Diagnostics bundle created", extra={"path": str(bundle_path)})
    return bundle_path


def cleanup_diagnostics_files(*, retention_days: int, max_bytes: int) -> dict:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(retention_days, 0))
    deleted_files = 0
    reclaimed_bytes = 0

    files = [path for path in DIAGNOSTICS_DIR.iterdir() if path.is_file()]
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
