"""
Central notification dispatcher for SafetyLens.
Routes alerts to the appropriate channels based on the channel matrix config.

Also runs the escalation background loop.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from config_manager import get_config
import telegram_notifier
import email_notifier
import webhook_notifier

logger = logging.getLogger("safetylens.dispatcher")

# Maps channel names to their notifier modules
_CHANNEL_HANDLERS = {
    "telegram": telegram_notifier,
    "email": email_notifier,
    "webhook": webhook_notifier,
}

_NOT_IMPLEMENTED = {"whatsapp", "sms", "plc"}

# Track which escalation step has been sent per alert: {alert_id: step_id}
_escalation_sent: dict[str, int] = {}

ESCALATION_CHECK_INTERVAL = 60  # seconds


def notify(alert: dict, snapshot_path: str | None = None) -> None:
    """Route alert to all enabled channels for its severity. Never raises."""
    try:
        cfg = get_config()
        routing = cfg.get("alert_routing", {})
        matrix = routing.get("channel_matrix", {})

        severity = alert.get("severity", "P4")
        channels = matrix.get(severity, {})

        for channel, enabled in channels.items():
            if not enabled:
                continue

            if channel == "inApp":
                # In-app is handled by WebSocket broadcast in video_processing.py
                continue

            if channel in _NOT_IMPLEMENTED:
                logger.debug("Channel %s not implemented, skipping", channel)
                continue

            handler = _CHANNEL_HANDLERS.get(channel)
            if not handler:
                logger.debug("No handler for channel %s", channel)
                continue

            try:
                handler.send_alert(alert, snapshot_path)
            except Exception:
                logger.exception("Failed to send via %s", channel)
    except Exception:
        logger.exception("Notification dispatch failed")


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

    # Sort by afterMinutes ascending
    steps = sorted(steps, key=lambda s: s.get("afterMinutes", 0))

    try:
        result = alert_store.get_alerts(status="active", limit=200)
        active_alerts = result.get("alerts", [])
    except Exception:
        logger.exception("Escalation: failed to query active alerts")
        return

    now = datetime.now(timezone.utc)

    # Track which alert_ids are still active for cleanup
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
        last_sent_step = _escalation_sent.get(alert_id, 0)

        for step in steps:
            step_id = step.get("id", 0)
            after_min = step.get("afterMinutes", 0)

            if age_minutes >= after_min and step_id > last_sent_step:
                _send_escalation(alert, step)
                _escalation_sent[alert_id] = step_id

    # Clean up tracking for alerts that are no longer active
    stale_ids = [aid for aid in _escalation_sent if aid not in active_ids]
    for aid in stale_ids:
        del _escalation_sent[aid]


def _send_escalation(alert: dict, step: dict) -> None:
    """Send an escalation notification for a specific step."""
    role = step.get("role", "Manager")
    channel = step.get("channel", "telegram")
    after_min = step.get("afterMinutes", 0)

    # Build an escalated copy of the alert with modified description
    escalated = dict(alert)
    original_desc = escalated.get("description", "")
    escalated["description"] = f"[ESCALATED to {role} after {after_min}min] {original_desc}"

    handler = _CHANNEL_HANDLERS.get(channel)
    if not handler:
        logger.warning("Escalation channel %s not available", channel)
        return

    try:
        snap_url = alert.get("snapshotUrl")
        snap_path = None
        if snap_url:
            import alert_store
            snap_path = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1])
        handler.send_alert(escalated, snap_path)
        logger.info(
            "Escalation sent",
            extra={"alert_id": alert.get("id"), "role": role, "channel": channel, "after_min": after_min},
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
