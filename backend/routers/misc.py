"""
SafetyLens miscellaneous endpoints — health, videos, alert rules available.
"""

from fastapi import APIRouter

from config_manager import get_config
from constants import YOLO_MODEL_PATH, VIDEO_DIR
from routers.safety_rules import _ensure_safety_rules
import alert_store

router = APIRouter(prefix="/api", tags=["misc"])


@router.get("/health")
async def health():
    cfg = get_config()
    return {
        "status": "ok",
        "model": str(YOLO_MODEL_PATH) if YOLO_MODEL_PATH.exists() else "yolov8n.pt (pretrained)",
        "cameras": list(cfg["cameras"].keys()),
        "vlm": f"{cfg['vlm']['model']} via Ollama",
        "alerts_count": alert_store.get_stats()["total"],
    }


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
