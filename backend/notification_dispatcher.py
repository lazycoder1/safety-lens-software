"""
Central notification dispatcher for Rakshak Lens.
Routes alerts through configurable alert outputs and records per-output results.

Also runs the escalation background loop.
"""

import asyncio
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from config_manager import get_config, save_config
import email_notifier
import telegram_notifier
import webhook_notifier

logger = logging.getLogger("rakshak_lens.dispatcher")

DELIVERED = "delivered"
SIMULATED = "simulated"
SKIPPED = "skipped"
FAILED = "failed"

_escalation_sent: dict[str, int] = {}
ESCALATION_CHECK_INTERVAL = 60  # seconds
ESCALATION_MAX_ALERT_AGE_HOURS = 24


def notify(alert: dict, snapshot_path: str | None = None, output_ids: list[str] | None = None) -> list[dict]:
    """Route alert to all enabled alert outputs. Never raises."""
    try:
        results = dispatch_alert(alert, snapshot_path, output_ids=output_ids)
        alert["deliveryResults"] = results
        alert_id = alert.get("id", "")
        if alert_id and not str(alert_id).startswith("test-"):
            try:
                import alert_store
                updated = alert_store.update_delivery_results(alert_id, results)
                if updated:
                    alert.update(updated)
            except Exception:
                logger.exception("Failed to persist delivery results")
        return results
    except Exception:
        logger.exception("Notification dispatch failed")
        return []


def dispatch_alert(alert: dict, snapshot_path: str | None = None, output_ids: list[str] | None = None) -> list[dict]:
    """Dispatch an alert and return one result per applicable output."""
    cfg = get_config()
    outputs = cfg.get("alert_outputs", [])
    if output_ids:
        requested = set(output_ids)
        outputs = [output for output in outputs if output.get("id") in requested]

    results = []
    for output in outputs:
        result = _dispatch_one(output, alert, snapshot_path)
        if result:
            results.append(result)

    if results:
        _record_output_results(results)
    return results


def test_output(output: dict, alert: dict | None = None) -> dict:
    """Send a single test alert through one output."""
    test_alert = alert or _build_test_alert()
    result = _dispatch_one(output, test_alert, None, force=True)
    if result:
        _record_output_results([result], test=True)
    return result or _result(output, SKIPPED, "Output is not applicable for this alert")


def _dispatch_one(output: dict, alert: dict, snapshot_path: str | None, force: bool = False) -> dict | None:
    if not output.get("enabled", False) and not force:
        return None

    severity = alert.get("severity", "P4")
    if not force and severity not in output.get("severities", []):
        return None

    zones = output.get("zones") or []
    zone = alert.get("zone")
    if not force and zones and zone not in zones:
        return None

    output_type = output.get("type")
    status = output.get("status")
    if status == "not_implemented":
        return _result(output, SKIPPED, "Adapter is not implemented yet")

    try:
        if output_type == "in_app":
            return _result(output, DELIVERED, "Delivered through Rakshak Lens WebSocket")
        if output_type == "browser_sound":
            return _result(output, SIMULATED, "Browser sound will play in armed Rakshak Lens clients")
        if output_type == "telegram":
            return _send_telegram(output, alert, snapshot_path)
        if output_type == "email":
            return _send_email(output, alert, snapshot_path)
        if output_type == "webhook":
            return _send_webhook(output, alert, snapshot_path)
        if output_type == "pushover":
            return _send_pushover(output, alert)
        if output_type == "ip_speaker":
            return _send_speaker(output, alert)
        if output_type == "relay":
            return _send_relay(output, alert)
        if output_type == "plc":
            return _result(output, SKIPPED, "PLC/Modbus adapter is not implemented yet")
        return _result(output, SKIPPED, f"Unknown output type: {output_type}")
    except Exception as exc:
        logger.exception("Alert output failed", extra={"output_id": output.get("id"), "type": output_type})
        return _result(output, FAILED, str(exc))


def _send_telegram(output: dict, alert: dict, snapshot_path: str | None) -> dict:
    settings = output.get("settings", {})
    token = settings.get("bot_token") or get_config().get("telegram", {}).get("bot_token", "")
    chat_id = settings.get("chat_id") or get_config().get("telegram", {}).get("chat_id", "")
    if not token or not chat_id:
        return _result(output, SKIPPED, "Telegram bot token and chat ID are required")
    caption = telegram_notifier._format_caption(alert)
    if snapshot_path:
        telegram_notifier._send_photo(token, chat_id, snapshot_path, caption)
    else:
        telegram_notifier._send_message(token, chat_id, caption)
    return _result(output, DELIVERED, "Telegram message sent")


def _send_email(output: dict, alert: dict, snapshot_path: str | None) -> dict:
    settings = output.get("settings", {})
    provider = settings.get("provider", "smtp")
    to_addrs = settings.get("to_addresses") or get_config().get("email", {}).get("to_addresses", [])
    from_addr = settings.get("from_address") or get_config().get("email", {}).get("from_address", "")
    if not from_addr or not to_addrs:
        return _result(output, SKIPPED, "From address and recipients are required")

    subject, html_body = email_notifier._build_email(alert, snapshot_path)
    if provider == "sendgrid":
        return _send_sendgrid(output, subject, html_body, alert, snapshot_path)

    host = settings.get("smtp_host") or get_config().get("email", {}).get("smtp_host", "")
    if not host:
        return _result(output, SKIPPED, "SMTP host is required")
    msg = email_notifier.build_message(subject, html_body, from_addr, to_addrs, snapshot_path)
    email_notifier._send(
        host,
        int(settings.get("smtp_port") or 587),
        settings.get("smtp_user", ""),
        settings.get("smtp_pass", ""),
        from_addr,
        to_addrs,
        msg,
        use_tls=bool(settings.get("use_tls", True)),
    )
    return _result(output, DELIVERED, "SMTP email sent")


def _send_sendgrid(output: dict, subject: str, html_body: str, alert: dict, snapshot_path: str | None) -> dict:
    settings = output.get("settings", {})
    api_key = settings.get("sendgrid_api_key", "")
    from_addr = settings.get("from_address", "")
    from_name = settings.get("from_name", "Rakshak Lens")
    to_addrs = settings.get("to_addresses", [])
    if not api_key:
        return _result(output, SKIPPED, "SendGrid API key is required")
    payload = {
        "personalizations": [{"to": [{"email": addr} for addr in to_addrs]}],
        "from": {"email": from_addr, "name": from_name},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }
    template_id = settings.get("sendgrid_template_id")
    if template_id:
        payload["template_id"] = template_id
        payload["personalizations"][0]["dynamic_template_data"] = _template_data(alert)
        payload.pop("content", None)
    if snapshot_path and Path(snapshot_path).exists():
        payload["attachments"] = [
            {
                "content": base64.b64encode(Path(snapshot_path).read_bytes()).decode("ascii"),
                "filename": "snapshot.jpg",
                "type": "image/jpeg",
                "disposition": "attachment",
            }
        ]

    resp = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=15,
    )
    if 200 <= resp.status_code < 300:
        return _result(output, DELIVERED, "SendGrid email accepted", {"httpStatus": resp.status_code})
    return _result(output, FAILED, f"SendGrid HTTP {resp.status_code}: {resp.text[:200]}", {"httpStatus": resp.status_code})


def _send_webhook(output: dict, alert: dict, snapshot_path: str | None) -> dict:
    settings = output.get("settings", {})
    url = settings.get("url", "")
    if not url:
        return _result(output, SKIPPED, "Webhook URL is required")
    payload = webhook_notifier._build_payload(alert, snapshot_path, include_snapshot=settings.get("include_snapshot", False))
    headers = {"Content-Type": "application/json"}
    headers.update(settings.get("headers") or {})
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    if 200 <= resp.status_code < 300:
        return _result(output, DELIVERED, f"Webhook HTTP {resp.status_code}", {"httpStatus": resp.status_code})
    return _result(output, FAILED, f"Webhook HTTP {resp.status_code}: {resp.text[:200]}", {"httpStatus": resp.status_code})


def _send_pushover(output: dict, alert: dict) -> dict:
    settings = output.get("settings", {})
    app_token = settings.get("app_token", "")
    user_key = settings.get("user_key", "")
    if not app_token or not user_key:
        return _result(output, SKIPPED, "Pushover app token and user key are required")

    priority = int(settings.get("priority") or 1)
    if alert.get("severity") == "P1":
        priority = max(priority, 2)
    payload = {
        "token": app_token,
        "user": user_key,
        "title": f"Rakshak Lens {alert.get('severity', '')} Alert",
        "message": alert.get("message") or f"{alert.get('rule', 'Alert')} at {alert.get('zone', 'Unknown')} ({alert.get('cameraName', 'Camera')})",
        "priority": priority,
        "sound": settings.get("sound") or "siren",
        "tags": f"rakshak_lens,output-{output.get('id', 'pushover')},alert-{alert.get('id', 'unknown')}",
    }
    if settings.get("device"):
        payload["device"] = settings["device"]
    if priority == 2:
        payload["retry"] = int(settings.get("emergency_retry") or 60)
        payload["expire"] = int(settings.get("emergency_expire") or 3600)

    resp = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=15)
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code == 200 and data.get("status") == 1:
        details = {"request": data.get("request")}
        if data.get("receipt"):
            details["receipt"] = data.get("receipt")
        return _result(output, DELIVERED, "Pushover notification sent", details)
    error = data.get("errors") or data.get("error") or resp.text[:200]
    return _result(output, FAILED, f"Pushover failed: {error}", {"httpStatus": resp.status_code})


def cancel_pushover_emergency_retries(output: dict, limit: int = 500) -> dict:
    """Cancel outstanding emergency-priority Pushover retries for this output."""
    settings = output.get("settings", {})
    app_token = settings.get("app_token", "")
    if not app_token:
        return {"attempted": 0, "cancelled": 0, "failed": 0, "errors": ["Pushover app token is required"]}

    receipts: set[str] = set()
    try:
        import alert_store
        for alert in alert_store.get_alerts(limit=limit):
            for result in alert.get("deliveryResults") or []:
                if result.get("outputId") != output.get("id"):
                    continue
                receipt = (result.get("details") or {}).get("receipt")
                if receipt:
                    receipts.add(str(receipt))
    except Exception:
        logger.exception("Failed to inspect Pushover receipts")

    attempted = 0
    cancelled = 0
    failed = 0
    errors: list[str] = []
    for receipt in receipts:
        attempted += 1
        try:
            resp = requests.post(
                f"https://api.pushover.net/1/receipts/{receipt}/cancel.json",
                data={"token": app_token},
                timeout=10,
            )
            try:
                data = resp.json()
            except Exception:
                data = {}
            if resp.status_code == 200 and data.get("status") == 1:
                cancelled += 1
            else:
                failed += 1
                errors.append(f"{receipt}: HTTP {resp.status_code}")
        except Exception as exc:
            failed += 1
            errors.append(f"{receipt}: {exc}")

    # Future sends include this tag, so this cancels anything newer even if its
    # receipt has not been persisted onto an alert yet.
    tag = f"output-{output.get('id', 'pushover')}"
    try:
        requests.post(
            f"https://api.pushover.net/1/receipts/cancel_by_tag/{tag}.json",
            data={"token": app_token},
            timeout=10,
        )
    except Exception:
        logger.debug("Pushover cancel_by_tag failed", exc_info=True)

    return {"attempted": attempted, "cancelled": cancelled, "failed": failed, "errors": errors[:5]}


def _send_speaker(output: dict, alert: dict) -> dict:
    settings = output.get("settings", {})
    mode = output.get("mode", "http")
    message = _render_template(settings.get("message") or alert.get("message") or "Safety alert in {zone}.", alert)
    if mode in {"audio_relay", "dry_run"}:
        return _result(output, SIMULATED, f"Speaker simulated: {message}")
    if mode != "http":
        return _result(output, SKIPPED, f"{mode} speaker adapter is not implemented yet")
    url = settings.get("url", "")
    if not url:
        return _result(output, SKIPPED, "Speaker HTTP URL is required")
    method = (settings.get("method") or "POST").upper()
    headers = {"Content-Type": "application/json"}
    headers.update(settings.get("headers") or {})
    resp = requests.request(method, url, json={"message": message, "alert": _template_data(alert)}, headers=headers, timeout=10)
    if 200 <= resp.status_code < 300:
        return _result(output, DELIVERED, f"Speaker HTTP {resp.status_code}", {"httpStatus": resp.status_code, "message": message})
    return _result(output, FAILED, f"Speaker HTTP {resp.status_code}: {resp.text[:200]}", {"httpStatus": resp.status_code, "message": message})


def _send_relay(output: dict, alert: dict) -> dict:
    settings = output.get("settings", {})
    mode = output.get("mode", "dry_run")
    pulse = int(settings.get("pulseSeconds") or 5)
    if mode == "dry_run":
        return _result(output, SIMULATED, f"Relay dry-run pulse {pulse}s")
    if mode == "http":
        url = settings.get("url", "")
        if not url:
            return _result(output, SKIPPED, "Relay HTTP URL is required")
        resp = requests.post(url, json={"state": "pulse", "seconds": pulse, "alert": _template_data(alert)}, timeout=10)
        if 200 <= resp.status_code < 300:
            return _result(output, DELIVERED, f"Relay HTTP {resp.status_code}", {"httpStatus": resp.status_code})
        return _result(output, FAILED, f"Relay HTTP {resp.status_code}: {resp.text[:200]}", {"httpStatus": resp.status_code})
    return _result(output, SKIPPED, f"{mode} relay adapter is not implemented yet")


def _record_output_results(results: list[dict], test: bool = False) -> None:
    cfg = get_config()
    outputs = cfg.get("alert_outputs", [])
    now = datetime.now(timezone.utc).isoformat()
    by_id = {item["outputId"]: item for item in results}
    changed = False
    for output in outputs:
        result = by_id.get(output.get("id"))
        if not result:
            continue
        if test:
            output["lastTestAt"] = now
        else:
            output["lastFiredAt"] = now
        output["lastError"] = result.get("message", "") if result.get("status") == FAILED else ""
        if result.get("status") == FAILED:
            output["status"] = "failed"
        elif output.get("status") in {"failed", "needs_setup"} and result.get("status") == DELIVERED:
            output["status"] = "ready"
        changed = True
    if changed:
        save_config(cfg)


def _result(output: dict, status: str, message: str, details: dict | None = None) -> dict:
    return {
        "id": str(uuid4())[:8],
        "outputId": output.get("id"),
        "outputName": output.get("name"),
        "type": output.get("type"),
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }


def _build_test_alert() -> dict:
    return {
        "id": f"test-{uuid4().hex[:8]}",
        "severity": "P1",
        "status": "active",
        "rule": "Test Alert",
        "cameraId": "test",
        "cameraName": "Test Camera",
        "zone": "Test Zone",
        "confidence": 0.95,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "test",
        "description": "[TEST] This is a test alert from Rakshak Lens.",
    }


def _template_data(alert: dict) -> dict:
    return {
        "alert_id": alert.get("id"),
        "severity": alert.get("severity"),
        "violation_type": alert.get("rule"),
        "camera": alert.get("cameraName"),
        "zone": alert.get("zone"),
        "timestamp": alert.get("timestamp"),
        "confidence": alert.get("confidence"),
        "description": alert.get("description"),
        "message": alert.get("message"),
        "policy_id": alert.get("policyId"),
        "priority": alert.get("priority"),
    }


def _render_template(template: str, alert: dict) -> str:
    data = _template_data(alert)
    value = template
    for key, replacement in data.items():
        value = value.replace("{" + key + "}", "" if replacement is None else str(replacement))
    return value


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

def clear_escalation(alert_id: str) -> None:
    """Call when an alert is acknowledged/resolved to stop further escalation."""
    _escalation_sent.pop(alert_id, None)


def _check_escalation() -> None:
    """Check all active alerts against escalation rules and send notifications."""
    import alert_store  # deferred to avoid circular import

    cfg = get_config()
    routing = cfg.get("alert_routing", {})
    steps = routing.get("escalation_steps", [])

    if not steps:
        return

    steps = sorted(steps, key=lambda s: s.get("afterMinutes", 0))

    try:
        active_alerts = alert_store.get_alerts(status="active", limit=200)
    except Exception:
        logger.exception("Escalation: failed to query active alerts")
        return

    now = datetime.now(timezone.utc)
    active_ids = set()

    for alert in active_alerts:
        alert_id = alert.get("id")
        if not alert_id:
            continue
        active_ids.add(alert_id)

        ts_str = alert.get("timestamp", "")
        try:
            alert_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if alert_time.tzinfo is None:
                alert_time = alert_time.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            continue

        age_minutes = (now - alert_time).total_seconds() / 60
        if age_minutes > ESCALATION_MAX_ALERT_AGE_HOURS * 60:
            continue
        last_sent_step = _escalation_sent.get(alert_id, 0)

        for step in steps:
            step_id = step.get("id", 0)
            after_min = step.get("afterMinutes", 0)
            if age_minutes >= after_min and step_id > last_sent_step:
                _send_escalation(alert, step)
                _escalation_sent[alert_id] = step_id

    for aid in [aid for aid in _escalation_sent if aid not in active_ids]:
        del _escalation_sent[aid]


def _send_escalation(alert: dict, step: dict) -> None:
    """Send an escalation notification for a specific step."""
    role = step.get("role", "Manager")
    channel = str(step.get("channel", "telegram")).lower()
    after_min = step.get("afterMinutes", 0)
    escalated = dict(alert)
    escalated["description"] = f"[ESCALATED to {role} after {after_min}min] {escalated.get('description', '')}"
    outputs = [
        output for output in get_config().get("alert_outputs", [])
        if output.get("type") == channel or output.get("id") == channel
    ]
    if not outputs:
        logger.warning("Escalation channel %s not available", channel)
        return
    try:
        snap_url = alert.get("snapshotUrl")
        snap_path = None
        if snap_url:
            import alert_store
            snap_path = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1])
        results = dispatch_alert(escalated, snap_path, output_ids=[outputs[0]["id"]])
        sent_results = [result for result in results if result.get("status") in {DELIVERED, SIMULATED}]
        if sent_results:
            logger.info("Escalation sent", extra={"alert_id": alert.get("id"), "role": role, "channel": channel})
        else:
            logger.info(
                "Escalation skipped",
                extra={"alert_id": alert.get("id"), "role": role, "channel": channel, "reason": "no enabled output"},
            )
    except Exception:
        logger.exception("Escalation send failed for step %s", step.get("id"))


async def escalation_check_loop() -> None:
    """Background loop that checks for alerts needing escalation."""
    while True:
        await asyncio.sleep(ESCALATION_CHECK_INTERVAL)
        try:
            await asyncio.to_thread(_check_escalation)
        except Exception:
            logger.exception("Escalation check loop failed")
