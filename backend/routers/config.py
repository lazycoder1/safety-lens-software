"""
SafetyLens config endpoints — global, VLM, telegram settings.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

import audit_store
from config_manager import get_config, save_config
from dependencies import require_admin
from video_processing import restart_all_cameras
import telegram_notifier

router = APIRouter(prefix="/api", tags=["config"])


class GlobalConfigUpdate(BaseModel):
    target_fps: Optional[int] = None
    yolo_conf: Optional[float] = None
    jpeg_quality: Optional[int] = None
    inference_width: Optional[int] = None
    device: Optional[str] = None
    alert_cooldown: Optional[int] = None


class VlmConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    interval: Optional[int] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    violation_keywords: Optional[list[str]] = None


class TelegramConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    severities: Optional[list[str]] = None


@router.get("/config")
async def api_get_config():
    return get_config()


@router.put("/config/global", dependencies=[Depends(require_admin)])
async def api_update_global(body: GlobalConfigUpdate, request: Request):
    cfg = get_config()
    updates = body.model_dump(exclude_none=True)
    cfg["global"].update(updates)
    save_config(cfg)
    restart_all_cameras()
    audit_store.log_event(
        "config.global_update",
        target_type="config",
        target_id="global",
        details={"updates": updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["global"]


@router.put("/config/vlm", dependencies=[Depends(require_admin)])
async def api_update_vlm(body: VlmConfigUpdate, request: Request):
    cfg = get_config()
    updates = body.model_dump(exclude_none=True)
    cfg["vlm"].update(updates)
    save_config(cfg)
    audit_store.log_event(
        "config.vlm_update",
        target_type="config",
        target_id="vlm",
        details={"updates": updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["vlm"]


@router.put("/config/telegram", dependencies=[Depends(require_admin)])
async def api_update_telegram(body: TelegramConfigUpdate, request: Request):
    cfg = get_config()
    if "telegram" not in cfg:
        cfg["telegram"] = {"enabled": False, "bot_token": "", "chat_id": "", "severities": ["P1", "P2"]}
    updates = body.model_dump(exclude_none=True)
    cfg["telegram"].update(updates)
    save_config(cfg)
    safe_updates = dict(updates)
    if "bot_token" in safe_updates:
        safe_updates["bot_token"] = "***redacted***"
    audit_store.log_event(
        "config.telegram_update",
        target_type="config",
        target_id="telegram",
        details={"updates": safe_updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["telegram"]


@router.post("/config/telegram/test", dependencies=[Depends(require_admin)])
async def api_test_telegram():
    cfg = get_config()
    tg = cfg.get("telegram", {})
    result = telegram_notifier.test_connection(tg.get("bot_token", ""), tg.get("chat_id", ""))
    return result


@router.post("/config/telegram/groups", dependencies=[Depends(require_admin)])
async def api_telegram_groups(body: TelegramConfigUpdate):
    """Fetch groups/chats the bot has been added to via getUpdates."""
    bot_token = body.bot_token
    if not bot_token:
        return {"ok": False, "error": "Bot token is required", "groups": []}
    groups = telegram_notifier.fetch_groups(bot_token)
    return groups
