"""
SafetyLens config endpoints — global, VLM, telegram, email, webhook, alert routing settings.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

import audit_store
from config_manager import get_config, get_public_config, save_config
from dependencies import require_admin
from video_processing import restart_all_cameras
import telegram_notifier
import email_notifier
import webhook_notifier

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


class EmailConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: Optional[list[str]] = None
    severities: Optional[list[str]] = None


class WebhookConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = None
    headers: Optional[dict] = None
    severities: Optional[list[str]] = None
    include_snapshot: Optional[bool] = None


class AlertRoutingUpdate(BaseModel):
    channel_matrix: Optional[dict] = None
    timeouts: Optional[dict] = None
    escalation_steps: Optional[list[dict]] = None
    templates: Optional[dict] = None


class ScheduledReportsUpdate(BaseModel):
    enabled: Optional[bool] = None
    schedule: Optional[str] = None
    day_of_week: Optional[int] = None
    hour: Optional[int] = None
    recipients: Optional[list[str]] = None


class TestAlertRequest(BaseModel):
    severity: str = "P1"
    rule: str = "Test Alert"
    cameraName: str = "Test Camera"
    zone: str = "Test Zone"


@router.get("/config")
async def api_get_config():
    return get_public_config()


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


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

@router.put("/config/email", dependencies=[Depends(require_admin)])
async def api_update_email(body: EmailConfigUpdate, request: Request):
    cfg = get_config()
    if "email" not in cfg:
        cfg["email"] = {"enabled": False, "smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_pass": "", "from_address": "", "to_addresses": [], "severities": ["P1", "P2"]}
    updates = body.model_dump(exclude_none=True)
    cfg["email"].update(updates)
    save_config(cfg)
    safe_updates = dict(updates)
    if "smtp_pass" in safe_updates:
        safe_updates["smtp_pass"] = "***redacted***"
    audit_store.log_event(
        "config.email_update",
        target_type="config",
        target_id="email",
        details={"updates": safe_updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["email"]


@router.post("/config/email/test", dependencies=[Depends(require_admin)])
async def api_test_email():
    cfg = get_config()
    em = cfg.get("email", {})
    to_addrs = em.get("to_addresses", [])
    if not to_addrs:
        return {"ok": False, "error": "No recipient addresses configured"}
    result = email_notifier.test_connection(
        em.get("smtp_host", ""),
        em.get("smtp_port", 587),
        em.get("smtp_user", ""),
        em.get("smtp_pass", ""),
        em.get("from_address", ""),
        to_addrs[0],
    )
    return result


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

@router.put("/config/webhook", dependencies=[Depends(require_admin)])
async def api_update_webhook(body: WebhookConfigUpdate, request: Request):
    cfg = get_config()
    if "webhook" not in cfg:
        cfg["webhook"] = {"enabled": False, "url": "", "headers": {}, "severities": ["P1", "P2"], "include_snapshot": False}
    updates = body.model_dump(exclude_none=True)
    cfg["webhook"].update(updates)
    save_config(cfg)
    audit_store.log_event(
        "config.webhook_update",
        target_type="config",
        target_id="webhook",
        details={"updates": updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["webhook"]


@router.post("/config/webhook/test", dependencies=[Depends(require_admin)])
async def api_test_webhook():
    cfg = get_config()
    wh = cfg.get("webhook", {})
    url = wh.get("url", "")
    if not url:
        return {"ok": False, "error": "No webhook URL configured"}
    result = webhook_notifier.test_connection(url, wh.get("headers"))
    return result


# ---------------------------------------------------------------------------
# Alert Routing
# ---------------------------------------------------------------------------

@router.put("/config/alert-routing", dependencies=[Depends(require_admin)])
async def api_update_alert_routing(body: AlertRoutingUpdate, request: Request):
    cfg = get_config()
    if "alert_routing" not in cfg:
        cfg["alert_routing"] = {}
    updates = body.model_dump(exclude_none=True)
    cfg["alert_routing"].update(updates)
    save_config(cfg)
    audit_store.log_event(
        "config.alert_routing_update",
        target_type="config",
        target_id="alert_routing",
        details={"updated_keys": list(updates.keys())},
        **audit_store.build_actor_context(request),
    )
    return cfg["alert_routing"]


@router.post("/config/alert-routing/test", dependencies=[Depends(require_admin)])
async def api_test_alert_routing(body: TestAlertRequest):
    """Fire a test alert through the notification dispatcher."""
    import notification_dispatcher
    from uuid import uuid4
    from datetime import datetime, timezone

    test_alert = {
        "id": f"test-{uuid4().hex[:8]}",
        "severity": body.severity,
        "status": "active",
        "rule": body.rule,
        "cameraId": "test",
        "cameraName": body.cameraName,
        "zone": body.zone,
        "confidence": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "test",
        "description": "[TEST] This is a test alert from SafetyLens.",
    }
    notification_dispatcher.notify(test_alert, None)
    return {"ok": True, "message": "Test alert dispatched to configured channels"}


# ---------------------------------------------------------------------------
# Scheduled Reports
# ---------------------------------------------------------------------------

@router.get("/config/scheduled-reports")
async def api_get_scheduled_reports():
    cfg = get_config()
    return cfg.get("scheduled_reports", {})


@router.put("/config/scheduled-reports", dependencies=[Depends(require_admin)])
async def api_update_scheduled_reports(body: ScheduledReportsUpdate, request: Request):
    cfg = get_config()
    if "scheduled_reports" not in cfg:
        cfg["scheduled_reports"] = {"enabled": False, "schedule": "weekly", "day_of_week": 1, "hour": 6, "recipients": []}
    updates = body.model_dump(exclude_none=True)
    cfg["scheduled_reports"].update(updates)
    save_config(cfg)
    audit_store.log_event(
        "config.scheduled_reports_update",
        target_type="config",
        target_id="scheduled_reports",
        details={"updates": updates},
        **audit_store.build_actor_context(request),
    )
    return cfg["scheduled_reports"]
