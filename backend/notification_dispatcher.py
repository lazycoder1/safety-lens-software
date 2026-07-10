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

# Escalation state is process-local; sent and exhausted steps remain distinct.
_escalation_sent: dict[str, set[int | str]] = {}
_escalation_exhausted: dict[str, set[int | str]] = {}
_escalation_attempts: dict[tuple[str, int | str], int] = {}

ESCALATION_CHECK_INTERVAL = 60  # seconds
ESCALATION_MAX_ATTEMPTS = 3
ESCALATION_DELIVERED = "delivered"
ESCALATION_RETRY = "retry"
ESCALATION_TERMINAL = "terminal"


def _normalize_channel(channel: str) -> str:
    return str(channel or "").strip().lower()


def _terminal_channel_reason(
    cfg: dict,
    channel: str,
    alert: dict,
) -> tuple[str, str] | None:
    """Return why a configured channel cannot succeed without an operator change."""
    channel_cfg = cfg.get(channel, {})
    if not isinstance(channel_cfg, dict) or not channel_cfg.get("enabled", False):
        return ("inactive", "Channel is disabled")

    if channel == "telegram":
        configured = bool(channel_cfg.get("bot_token") and channel_cfg.get("chat_id"))
    elif channel == "email":
        configured = bool(
            channel_cfg.get("smtp_host")
            and channel_cfg.get("from_address")
            and channel_cfg.get("to_addresses")
        )
    elif channel == "webhook":
        configured = bool(channel_cfg.get("url"))
    else:
        configured = True
    if not configured:
        return ("invalid", "Channel configuration is incomplete")

    severities = channel_cfg.get("severities", ["P1", "P2"])
    if not isinstance(severities, (list, tuple, set)) or alert.get("severity") not in severities:
        return ("inactive", f"Severity {alert.get('severity', 'unknown')} is filtered")
    return None


def notify_with_results(
    alert: dict,
    snapshot_path: str | None = None,
    *,
    channels: list[str] | None = None,
    test_request: bool = False,
) -> list[dict]:
    """Route an alert and return one honest result per requested channel."""
    cfg = get_config()
    implicit_routing = channels is None
    if implicit_routing:
        routing = cfg.get("alert_routing", {})
        matrix = routing.get("channel_matrix", {})
        severity_channels = matrix.get(alert.get("severity", "P4"), {})
        raw_channels = [channel for channel, enabled in severity_channels.items() if enabled]
    else:
        raw_channels = channels

    requested_channels: list[str] = []
    seen_channels: set[str] = set()
    for raw_channel in raw_channels:
        channel = _normalize_channel(raw_channel)
        if channel in seen_channels:
            continue
        seen_channels.add(channel)
        requested_channels.append(channel)

    results: list[dict] = []
    for channel in requested_channels:
        result_channel = "inApp" if channel == "inapp" else channel
        if channel == "inapp":
            success = not test_request
            results.append({
                "channel": result_channel,
                "success": success,
                "status": "handled" if success else "skipped",
                "message": (
                    "Handled by the persisted alert and WebSocket path"
                    if success
                    else "In-app delivery cannot be verified by this external-channel test"
                ),
            })
            continue
        if channel in _NOT_IMPLEMENTED:
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "skipped",
                "message": "Channel is not implemented",
            })
            continue

        handler = _CHANNEL_HANDLERS.get(channel)
        if not handler:
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "skipped",
                "message": "No channel handler is configured",
            })
            continue
        terminal = _terminal_channel_reason(cfg, channel, alert)
        if terminal is not None:
            reason_code, terminal_reason = terminal
            # The channel matrix and global provider switch are both routing
            # gates. On normal implicit routing, a disabled or severity-filtered
            # provider was not requested for this alert. Explicit tests/retries
            # still surface the terminal state honestly.
            if implicit_routing and reason_code == "inactive":
                continue
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "skipped",
                "message": terminal_reason,
            })
            continue
        try:
            success = bool(handler.send_alert(alert, snapshot_path))
            results.append({
                "channel": result_channel,
                "success": success,
                "status": "delivered" if success else "failed",
                "message": "Delivered" if success else "Channel did not accept the alert",
            })
        except Exception as exc:
            logger.exception("Failed to send via %s", channel)
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "failed",
                "message": str(exc),
            })
    return results


def notify(alert: dict, snapshot_path: str | None = None) -> bool:
    """Route an alert and report whether every requested external path succeeded.

    In-app persistence is sufficient only when no external channel was
    requested; it must not mask a failed Telegram, email, or webhook target.
    """
    try:
        results = notify_with_results(alert, snapshot_path)
        external_results = [
            result
            for result in results
            if _normalize_channel(result.get("channel")) != "inapp"
        ]
        if external_results:
            return all(result.get("success", False) for result in external_results)
        return all(result.get("success", False) for result in results)
    except Exception:
        logger.exception("Notification dispatch failed")
        return False


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

def clear_escalation(alert_id: str) -> None:
    """Call when an alert is acknowledged/resolved to stop further escalation."""
    _escalation_sent.pop(alert_id, None)
    _escalation_exhausted.pop(alert_id, None)
    for key in [key for key in _escalation_attempts if key[0] == alert_id]:
        _escalation_attempts.pop(key, None)


def _check_escalation() -> None:
    """Check all active alerts against escalation rules and send notifications."""
    import alert_store  # deferred to avoid circular import

    cfg = get_config()
    routing = cfg.get("alert_routing", {})
    steps = routing.get("escalation_steps", [])

    if not steps:
        _escalation_sent.clear()
        _escalation_exhausted.clear()
        _escalation_attempts.clear()
        return

    # Sort by afterMinutes ascending
    steps = sorted(steps, key=lambda s: s.get("afterMinutes", 0))

    try:
        active_alerts = []
        page_size = 200
        offset = 0
        while True:
            page = alert_store.get_alerts(status="active", limit=page_size, offset=offset)
            active_alerts.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
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
        sent_steps = _escalation_sent.get(alert_id, set())
        exhausted_steps = _escalation_exhausted.get(alert_id, set())

        for step in steps:
            step_id = step.get("id", f"{step.get('afterMinutes', 0)}:{step.get('role', '')}:{step.get('channel', '')}")
            after_min = step.get("afterMinutes", 0)

            if age_minutes < after_min or step_id in sent_steps or step_id in exhausted_steps:
                continue

            outcome = _send_escalation(alert, step)
            attempt_key = (alert_id, step_id)
            if outcome == ESCALATION_DELIVERED:
                _escalation_sent.setdefault(alert_id, set()).add(step_id)
                sent_steps = _escalation_sent[alert_id]
                _escalation_attempts.pop(attempt_key, None)
                continue
            if outcome == ESCALATION_TERMINAL:
                _escalation_exhausted.setdefault(alert_id, set()).add(step_id)
                exhausted_steps = _escalation_exhausted[alert_id]
                _escalation_attempts.pop(attempt_key, None)
                continue

            attempts = _escalation_attempts.get(attempt_key, 0) + 1
            if attempts >= ESCALATION_MAX_ATTEMPTS:
                _escalation_attempts.pop(attempt_key, None)
                _escalation_exhausted.setdefault(alert_id, set()).add(step_id)
                exhausted_steps = _escalation_exhausted[alert_id]
                logger.error(
                    "Escalation retry budget exhausted",
                    extra={"alert_id": alert_id, "step_id": step_id, "attempts": attempts},
                )
                continue
            _escalation_attempts[attempt_key] = attempts
            break

    # Clean up tracking for alerts that are no longer active
    stale_ids = [aid for aid in _escalation_sent if aid not in active_ids]
    for aid in stale_ids:
        del _escalation_sent[aid]
    for aid in [aid for aid in _escalation_exhausted if aid not in active_ids]:
        del _escalation_exhausted[aid]
    for key in [key for key in _escalation_attempts if key[0] not in active_ids]:
        del _escalation_attempts[key]


def _send_escalation(alert: dict, step: dict) -> str:
    """Send one escalation step and classify success, retry, or terminal failure."""
    role = step.get("role", "Manager")
    channel = _normalize_channel(step.get("channel", "telegram"))
    after_min = step.get("afterMinutes", 0)

    # Build an escalated copy of the alert with modified description
    escalated = dict(alert)
    original_desc = escalated.get("description", "")
    escalated["description"] = f"[ESCALATED to {role} after {after_min}min] {original_desc}"

    handler = _CHANNEL_HANDLERS.get(channel)
    if not handler:
        logger.warning("Escalation channel %s not available", channel)
        return ESCALATION_TERMINAL

    terminal = _terminal_channel_reason(get_config(), channel, escalated)
    if terminal is not None:
        _reason_code, reason = terminal
        logger.warning(
            "Escalation channel is not deliverable",
            extra={
                "alert_id": alert.get("id"),
                "role": role,
                "channel": channel,
                "reason": reason,
            },
        )
        return ESCALATION_TERMINAL

    try:
        snap_url = alert.get("snapshotUrl")
        snap_path = None
        if snap_url:
            import alert_store
            snap_path = str(alert_store.SNAPSHOTS_DIR / snap_url.split("/")[-1])
        if not handler.send_alert(escalated, snap_path):
            logger.warning(
                "Escalation not delivered",
                extra={"alert_id": alert.get("id"), "role": role, "channel": channel, "after_min": after_min},
            )
            return ESCALATION_RETRY
        logger.info(
            "Escalation sent",
            extra={"alert_id": alert.get("id"), "role": role, "channel": channel, "after_min": after_min},
        )
        return ESCALATION_DELIVERED
    except Exception:
        logger.exception("Escalation send failed for step %s", step.get("id"))
        return ESCALATION_RETRY


async def escalation_check_loop() -> None:
    """Background loop that checks for alerts needing escalation."""
    while True:
        await asyncio.sleep(ESCALATION_CHECK_INTERVAL)
        try:
            await asyncio.to_thread(_check_escalation)
        except Exception:
            logger.exception("Escalation check loop failed")
