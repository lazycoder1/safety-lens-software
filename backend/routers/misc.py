"""
Rakshak Lens miscellaneous endpoints — health, videos, alert rules available.
"""

import asyncio
import math
import os

from fastapi import APIRouter

import alert_store
import diagnostics
import model_manager
from config_manager import get_config
from constants import VIDEO_DIR
from routers.safety_rules import _ensure_safety_rules

router = APIRouter(prefix="/api", tags=["misc"])


def _mark_degraded(snapshot: dict, reason: str) -> None:
    # diagnostics snapshots are shallow copies of a short-lived cache. Own the
    # reasons list before extending it so an outbox failure does not poison the
    # cached base health after the worker recovers.
    reasons = list(snapshot.get("reasons") or [])
    snapshot["reasons"] = reasons
    if reason not in reasons:
        reasons.append(reason)
    if snapshot.get("status") not in {"degraded", "error"}:
        snapshot["status"] = "degraded"


def _stale_due_threshold_seconds() -> float:
    try:
        poll_seconds = float(os.getenv("ALERT_OUTBOX_POLL_SECONDS", "5"))
    except (TypeError, ValueError):
        poll_seconds = 5.0
    if not math.isfinite(poll_seconds):
        poll_seconds = 5.0
    return max(30.0, max(0.05, poll_seconds) * 2.0)


def _stale_persistence_threshold_seconds() -> float:
    try:
        threshold = float(os.getenv("ALERT_PERSISTENCE_STALE_SECONDS", "30"))
    except (TypeError, ValueError):
        threshold = 30.0
    if not math.isfinite(threshold):
        threshold = 30.0
    return max(0.05, threshold)


def _apply_outbox_health(snapshot: dict, pipeline_stats: dict) -> None:
    outbox = pipeline_stats.get("outbox")
    if not isinstance(outbox, dict):
        _mark_degraded(snapshot, "alert delivery outbox statistics unavailable")
        return

    if outbox.get("database_error"):
        _mark_degraded(snapshot, "alert delivery outbox statistics unavailable")

    running = outbox.get("running") is True
    if not running:
        _mark_degraded(snapshot, "alert delivery outbox is not running")
    else:
        if outbox.get("claimer_alive") is not True:
            _mark_degraded(snapshot, "alert delivery outbox claimer is not running")
        if int(outbox.get("workers_alive") or 0) < 1:
            _mark_degraded(snapshot, "alert delivery provider workers are not running")
        if outbox.get("renewer_alive") is not True:
            _mark_degraded(snapshot, "alert delivery lease renewer is not running")

    try:
        due = int(outbox.get("due") or 0)
        oldest_due_age = float(outbox.get("oldest_due_age_seconds") or 0.0)
    except (TypeError, ValueError):
        due = 0
        oldest_due_age = 0.0
    if due > 0 and oldest_due_age > _stale_due_threshold_seconds():
        _mark_degraded(snapshot, "alert delivery due backlog is stale")


def _apply_persistence_health(snapshot: dict, pipeline_stats: dict) -> None:
    """Expose an alert-ingestion outage even when delivery workers are healthy."""
    if not isinstance(pipeline_stats, dict):
        _mark_degraded(snapshot, "alert persistence pipeline statistics unavailable")
        return

    if pipeline_stats.get("running") is not True:
        _mark_degraded(snapshot, "alert persistence pipeline is not running")
        return
    if pipeline_stats.get("accepting") is not True:
        _mark_degraded(snapshot, "alert persistence pipeline is not accepting alerts")
    if pipeline_stats.get("persist_worker_alive") is not True:
        _mark_degraded(snapshot, "alert persistence worker is not running")

    try:
        consecutive_failures = int(
            pipeline_stats.get("consecutive_persistence_failures") or 0
        )
    except (TypeError, ValueError):
        consecutive_failures = 0
    if consecutive_failures > 0:
        _mark_degraded(snapshot, "alert persistence has unresolved failures")

    try:
        oldest_age = float(
            pipeline_stats.get("oldest_persistence_age_seconds") or 0.0
        )
    except (TypeError, ValueError):
        oldest_age = 0.0
    if oldest_age > _stale_persistence_threshold_seconds():
        _mark_degraded(snapshot, "alert persistence work is stale")


def _build_health():
    from video_processing import get_alert_pipeline_stats

    snapshot = diagnostics.build_health_snapshot()
    cfg = get_config()
    snapshot["cameraIds"] = list(cfg["cameras"].keys())
    snapshot["alerts_count"] = alert_store.get_stats()["total"]
    pipeline_stats = get_alert_pipeline_stats()
    snapshot["alertPipeline"] = pipeline_stats
    snapshot["inferenceTransport"] = {
        "primaryFrameBatch": model_manager.remote_primary_batch_stats(),
        "specialistFrameBatch": model_manager.remote_specialist_batch_stats(),
    }
    _apply_persistence_health(snapshot, pipeline_stats)
    _apply_outbox_health(snapshot, pipeline_stats)
    return snapshot


@router.get("/ping")
async def ping():
    return {"ok": True}


@router.get("/health")
async def health():
    return await asyncio.to_thread(_build_health)


@router.get("/alert-rules-available")
async def api_available_alert_rules():
    """Return the available alert rules from unified safety_rules config."""
    cfg = get_config()
    rules = _ensure_safety_rules(cfg)
    return {
        r["id"]: {"rule": r["name"], "severity": r["severity"], "classes": r["classes"]}
        for r in rules if r.get("type") == "alert"
    }


@router.get("/videos")
async def api_list_videos():
    videos = sorted(
        str(f.relative_to(VIDEO_DIR))
        for f in VIDEO_DIR.rglob("*")
        if f.suffix.lower() in (".mp4", ".avi") and f.is_file()
    )
    return videos
