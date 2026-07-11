"""
Rakshak Lens miscellaneous endpoints — health, videos, alert rules available.
"""

import asyncio

from fastapi import APIRouter

import alert_store
import diagnostics
from config_manager import get_config
from constants import VIDEO_DIR
from routers.safety_rules import _ensure_safety_rules

router = APIRouter(prefix="/api", tags=["misc"])


def _build_health():
    snapshot = diagnostics.build_health_snapshot()
    cfg = get_config()
    snapshot["cameraIds"] = list(cfg["cameras"].keys())
    snapshot["alerts_count"] = alert_store.get_stats()["total"]
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
