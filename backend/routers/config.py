"""
SafetyLens config endpoints — global, VLM, telegram, email, webhook, alert routing settings.
"""

import asyncio
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import audit_store
from config_manager import get_config, get_config_snapshot, get_public_config, save_config
from dependencies import require_admin
from secret_redaction import REDACTED_VALUE, is_redacted, redact_sensitive_data
from video_processing import restart_all_cameras
import telegram_notifier
import email_notifier
import webhook_notifier

router = APIRouter(prefix="/api", tags=["config"])

_ROUTING_CHANNELS = {"inapp", "telegram", "email", "webhook"}
_PROVIDER_CHANNELS = {"telegram", "email", "webhook"}
_ALERT_SEVERITIES = ("P1", "P2", "P3", "P4")


class GlobalConfigUpdate(BaseModel):
    target_fps: Optional[int] = Field(None, ge=1, le=60)
    yolo_conf: Optional[float] = Field(None, ge=0, le=1)
    jpeg_quality: Optional[int] = Field(None, ge=20, le=100)
    inference_width: Optional[int] = Field(None, ge=160, le=1920)
    device: Optional[str] = None
    alert_cooldown: Optional[int] = Field(None, ge=0, le=86400)


class VlmConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    interval: Optional[int] = Field(None, ge=5, le=86400)
    model: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
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
    account_id: Optional[str] = None
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
    channels: Optional[list[str]] = None


def _drop_redacted(updates: dict, *keys: str) -> dict:
    for key in keys:
        if is_redacted(updates.get(key)):
            updates.pop(key)
    return updates


def _preserve_redacted_headers(updates: dict, current: dict) -> dict:
    incoming = updates.get("headers")
    if not isinstance(incoming, dict):
        return updates
    existing = current.get("headers", {})
    updates["headers"] = {
        key: existing[key] if is_redacted(value) and key in existing else value
        for key, value in incoming.items()
        if not is_redacted(value) or key in existing
    }
    return updates


def _is_configured_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _provider_configuration_errors(channel: str, section: object) -> list[str]:
    """Return safe field names that keep an enabled provider from delivering."""
    if not isinstance(section, dict):
        return ["configuration"]

    if channel == "telegram":
        return [
            field
            for field in ("bot_token", "chat_id")
            if not _is_configured_text(section.get(field))
        ]

    if channel == "webhook":
        return [] if _is_configured_text(section.get("url")) else ["url"]

    if channel == "email":
        errors = []
        if not _is_configured_text(section.get("smtp_host")):
            errors.append("smtp_host")
        smtp_port = section.get("smtp_port")
        if (
            not isinstance(smtp_port, int)
            or isinstance(smtp_port, bool)
            or not 1 <= smtp_port <= 65535
        ):
            errors.append("smtp_port")
        if not _is_configured_text(section.get("from_address")):
            errors.append("from_address")
        recipients = section.get("to_addresses")
        if not isinstance(recipients, list) or not any(
            _is_configured_text(address) for address in recipients
        ):
            errors.append("to_addresses")

        # Anonymous SMTP is valid, but a half-configured authentication pair is
        # never usable: the notifier only logs in when both values are present.
        has_user = _is_configured_text(section.get("smtp_user"))
        has_password = _is_configured_text(section.get("smtp_pass"))
        if has_user != has_password:
            errors.append("smtp_user/smtp_pass")
        return errors

    return ["configuration"]


def _validate_enabled_provider(channel: str, section: object) -> None:
    if not isinstance(section, dict) or not section.get("enabled", False):
        return
    errors = _provider_configuration_errors(channel, section)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Cannot enable {channel}: incomplete configuration "
                f"({', '.join(errors)})"
            ),
        )


def _validate_channel_matrix(config: dict) -> None:
    routing = config.get("alert_routing", {})
    matrix = routing.get("channel_matrix", {}) if isinstance(routing, dict) else {}
    if not isinstance(matrix, dict):
        raise HTTPException(status_code=422, detail="channel_matrix must be an object")

    for severity, channels in matrix.items():
        if not isinstance(channels, dict):
            raise HTTPException(
                status_code=422,
                detail=f"channel_matrix.{severity} must be an object",
            )
        for raw_channel, enabled in channels.items():
            if not isinstance(enabled, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"channel_matrix.{severity}.{raw_channel} must be a boolean",
                )
            if not enabled:
                continue
            channel = str(raw_channel or "").strip().lower()
            if channel not in _ROUTING_CHANNELS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Unsupported enabled alert-routing channel: "
                        f"{channel or 'empty'} ({severity})"
                    ),
                )
            if channel not in _PROVIDER_CHANNELS:
                continue

            provider = config.get(channel, {})
            if not isinstance(provider, dict) or not provider.get("enabled", False):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Cannot enable {channel} routing for {severity}: "
                        "the provider is disabled"
                    ),
                )
            errors = _provider_configuration_errors(channel, provider)
            if errors:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Cannot enable {channel} routing for {severity}: "
                        f"incomplete provider configuration ({', '.join(errors)})"
                    ),
                )
            severities = provider.get("severities", [])
            if not isinstance(severities, list) or severity not in severities:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Cannot enable {channel} routing for {severity}: "
                        "the provider filters that severity"
                    ),
                )


def _normalized_severity_scope(raw: object, *, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise HTTPException(status_code=422, detail=f"{field} must be an array")
    normalized = []
    for item in raw:
        severity = str(item or "").strip().upper()
        if severity not in _ALERT_SEVERITIES:
            raise HTTPException(
                status_code=422,
                detail=f"{field} contains unsupported severity: {severity or 'empty'}",
            )
        if severity not in normalized:
            normalized.append(severity)
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{field} cannot be empty")
    return normalized


def _escalation_step_identity(step: dict, *, index: int) -> tuple[str, str]:
    """Return the exact identity fields used by durable escalation keys."""
    channel = str(step.get("channel") or "").strip().lower()
    raw_id = step.get("id")
    if raw_id in (None, ""):
        raw_id = (
            f"{step.get('afterMinutes', 0)}:"
            f"{step.get('role', '')}:"
            f"{channel}"
        )
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int, float)):
        raise HTTPException(
            status_code=422,
            detail=f"escalation_steps[{index}].id must be a string or number",
        )
    if isinstance(raw_id, float) and not math.isfinite(raw_id):
        raise HTTPException(
            status_code=422,
            detail=f"escalation_steps[{index}].id must be finite",
        )
    canonical_id = str(raw_id)
    if not canonical_id.strip() or len(canonical_id) > 160:
        raise HTTPException(
            status_code=422,
            detail=(
                f"escalation_steps[{index}].id must contain 1 to 160 characters"
            ),
        )
    return channel, canonical_id


def _validate_explicit_escalations(config: dict) -> None:
    """Protect active API-declared escalation steps from provider invalidation."""
    routing = config.get("alert_routing", {})
    steps = routing.get("escalation_steps", []) if isinstance(routing, dict) else []
    if not isinstance(steps, list):
        raise HTTPException(status_code=422, detail="escalation_steps must be an array")

    active_identities: set[tuple[str, str]] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("enabled") is not True:
            # Unmarked steps are legacy/default data. They are handled by the
            # resolver's migration policy, not silently promoted by an update
            # to an unrelated provider field.
            continue
        identity = _escalation_step_identity(step, index=index)
        if identity in active_identities:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Enabled escalation steps must have a unique id per channel: "
                    f"{identity[1]} ({identity[0]})"
                ),
            )
        active_identities.add(identity)
        channel = str(step.get("channel") or "").strip().lower()
        if channel not in _PROVIDER_CHANNELS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported escalation channel: {channel or 'empty'}",
            )
        provider = config.get(channel, {})
        if not isinstance(provider, dict) or not provider.get("enabled", False):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot activate escalation step {index + 1} ({channel}): "
                    "the provider is disabled"
                ),
            )
        errors = _provider_configuration_errors(channel, provider)
        if errors:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot activate escalation step {index + 1} ({channel}): "
                    f"incomplete provider configuration ({', '.join(errors)})"
                ),
            )

        scope = _normalized_severity_scope(
            step.get("severities", list(_ALERT_SEVERITIES)),
            field=f"escalation_steps[{index}].severities",
        )
        provider_scope = provider.get("severities", [])
        if not isinstance(provider_scope, list) or any(
            severity not in provider_scope for severity in scope
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cannot activate escalation step {index + 1} ({channel}): "
                    "the provider filters a severity required by the step"
                ),
            )


def _normalize_escalation_steps(config: dict, raw_steps: object) -> list[dict]:
    if not isinstance(raw_steps, list):
        raise HTTPException(status_code=422, detail="escalation_steps must be an array")
    normalized_steps = []
    active_identities: set[tuple[str, str]] = set()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise HTTPException(
                status_code=422,
                detail=f"escalation_steps[{index}] must be an object",
            )
        normalized = dict(raw_step)
        channel = str(normalized.get("channel", "telegram")).strip().lower()
        if channel not in _PROVIDER_CHANNELS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported escalation channel: {channel or 'empty'}",
            )
        enabled = normalized.get("enabled", True)
        if not isinstance(enabled, bool):
            raise HTTPException(
                status_code=422,
                detail=f"escalation_steps[{index}].enabled must be a boolean",
            )
        try:
            delay = float(normalized.get("afterMinutes", 0))
        except (TypeError, ValueError):
            delay = math.nan
        if not math.isfinite(delay) or delay < 0:
            raise HTTPException(
                status_code=422,
                detail=f"escalation_steps[{index}].afterMinutes must be non-negative",
            )

        normalized["channel"] = channel
        normalized["enabled"] = enabled
        if enabled:
            identity = _escalation_step_identity(normalized, index=index)
            if identity in active_identities:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Enabled escalation steps must have a unique id per channel: "
                        f"{identity[1]} ({identity[0]})"
                    ),
                )
            active_identities.add(identity)
            provider = config.get(channel, {})
            default_scope = (
                provider.get("severities", [])
                if isinstance(provider, dict)
                else []
            )
            normalized["severities"] = _normalized_severity_scope(
                normalized.get("severities", default_scope),
                field=f"escalation_steps[{index}].severities",
            )
        elif "severities" in normalized:
            normalized["severities"] = _normalized_severity_scope(
                normalized["severities"],
                field=f"escalation_steps[{index}].severities",
            )
        normalized_steps.append(normalized)
    return normalized_steps


@router.get("/config")
async def api_get_config():
    return get_public_config()


@router.put("/config/global", dependencies=[Depends(require_admin)])
async def api_update_global(body: GlobalConfigUpdate, request: Request):
    cfg = get_config_snapshot()
    updates = body.model_dump(exclude_none=True)
    cfg["global"].update(updates)
    save_config(cfg)
    await asyncio.to_thread(restart_all_cameras)
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
    cfg = get_config_snapshot()
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
    cfg = get_config_snapshot()
    if not isinstance(cfg.get("telegram"), dict):
        cfg["telegram"] = {"enabled": False, "bot_token": "", "chat_id": "", "severities": ["P1", "P2"]}
    updates = _drop_redacted(body.model_dump(exclude_none=True), "bot_token")
    cfg["telegram"].update(updates)
    _validate_enabled_provider("telegram", cfg["telegram"])
    _validate_channel_matrix(cfg)
    _validate_explicit_escalations(cfg)
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
    return get_public_config()["telegram"]


@router.post("/config/telegram/test", dependencies=[Depends(require_admin)])
async def api_test_telegram():
    cfg = get_config_snapshot()
    tg = cfg.get("telegram", {})
    result = await asyncio.to_thread(
        telegram_notifier.test_connection,
        tg.get("bot_token", ""),
        tg.get("chat_id", ""),
    )
    return result


@router.post("/config/telegram/groups", dependencies=[Depends(require_admin)])
async def api_telegram_groups(body: TelegramConfigUpdate):
    """Fetch groups/chats the bot has been added to via getUpdates."""
    bot_token = body.bot_token
    if is_redacted(bot_token):
        bot_token = get_config_snapshot("telegram").get("telegram", {}).get("bot_token")
    if not bot_token:
        return {"ok": False, "error": "Bot token is required", "groups": []}
    groups = await asyncio.to_thread(telegram_notifier.fetch_groups, bot_token)
    return groups


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

@router.put("/config/email", dependencies=[Depends(require_admin)])
async def api_update_email(body: EmailConfigUpdate, request: Request):
    cfg = get_config_snapshot()
    if not isinstance(cfg.get("email"), dict):
        cfg["email"] = {"enabled": False, "smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_pass": "", "from_address": "", "to_addresses": [], "severities": ["P1", "P2"]}
    updates = _drop_redacted(body.model_dump(exclude_none=True), "smtp_pass")
    cfg["email"].update(updates)
    _validate_enabled_provider("email", cfg["email"])
    _validate_channel_matrix(cfg)
    _validate_explicit_escalations(cfg)
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
    return get_public_config()["email"]


@router.post("/config/email/test", dependencies=[Depends(require_admin)])
async def api_test_email():
    cfg = get_config_snapshot()
    em = cfg.get("email", {})
    to_addrs = em.get("to_addresses", [])
    if not to_addrs:
        return {"ok": False, "error": "No recipient addresses configured"}
    result = await asyncio.to_thread(
        email_notifier.test_connection,
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
    cfg = get_config_snapshot()
    if not isinstance(cfg.get("webhook"), dict):
        cfg["webhook"] = {"enabled": False, "url": "", "headers": {}, "severities": ["P1", "P2"], "include_snapshot": False}
    updates = _drop_redacted(body.model_dump(exclude_none=True), "url")
    updates = _preserve_redacted_headers(updates, cfg["webhook"])
    cfg["webhook"].update(updates)
    _validate_enabled_provider("webhook", cfg["webhook"])
    _validate_channel_matrix(cfg)
    _validate_explicit_escalations(cfg)
    save_config(cfg)
    safe_updates = redact_sensitive_data(updates)
    if safe_updates.get("url"):
        safe_updates["url"] = REDACTED_VALUE
    audit_store.log_event(
        "config.webhook_update",
        target_type="config",
        target_id="webhook",
        details={"updates": safe_updates},
        **audit_store.build_actor_context(request),
    )
    return get_public_config()["webhook"]


@router.post("/config/webhook/test", dependencies=[Depends(require_admin)])
async def api_test_webhook():
    cfg = get_config_snapshot()
    wh = cfg.get("webhook", {})
    url = wh.get("url", "")
    if not url:
        return {"ok": False, "error": "No webhook URL configured"}
    result = await asyncio.to_thread(
        webhook_notifier.test_connection,
        url,
        wh.get("headers"),
    )
    return result


# ---------------------------------------------------------------------------
# Alert Routing
# ---------------------------------------------------------------------------

@router.put("/config/alert-routing", dependencies=[Depends(require_admin)])
async def api_update_alert_routing(body: AlertRoutingUpdate, request: Request):
    cfg = get_config_snapshot()
    if not isinstance(cfg.get("alert_routing"), dict):
        cfg["alert_routing"] = {}
    updates = body.model_dump(exclude_none=True)
    if "escalation_steps" in updates:
        updates["escalation_steps"] = _normalize_escalation_steps(
            cfg,
            updates["escalation_steps"],
        )
    cfg["alert_routing"].update(updates)
    _validate_channel_matrix(cfg)
    _validate_explicit_escalations(cfg)
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
    channels = body.channels
    if channels is None:
        matrix = get_config().get("alert_routing", {}).get("channel_matrix", {})
        channels = [
            channel
            for channel, enabled in matrix.get(body.severity, {}).items()
            if enabled
        ]

    results = await asyncio.to_thread(
        notification_dispatcher.notify_with_results,
        test_alert,
        None,
        channels=channels,
        test_request=True,
    )
    ok = bool(results) and all(result.get("success", False) for result in results)
    return {
        "ok": ok,
        "message": (
            "Test alert delivered to every requested channel"
            if ok
            else "One or more requested channels did not accept the test alert"
        ),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Scheduled Reports
# ---------------------------------------------------------------------------

@router.get("/config/scheduled-reports")
async def api_get_scheduled_reports():
    cfg = get_config()
    return cfg.get("scheduled_reports", {})


@router.put("/config/scheduled-reports", dependencies=[Depends(require_admin)])
async def api_update_scheduled_reports(body: ScheduledReportsUpdate, request: Request):
    cfg = get_config_snapshot()
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
