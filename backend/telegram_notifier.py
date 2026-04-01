"""
Telegram alert notifications for SafetyLens.
Uses raw requests to Telegram Bot API — no extra dependency needed.

Sync calls, fire-and-forget. Called from video processing threads.
"""

import logging

import requests

from config_manager import get_config

logger = logging.getLogger("safetylens.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def send_alert(alert: dict, snapshot_path: str | None = None) -> None:
    """Send alert notification to Telegram. Never raises — logs errors instead."""
    try:
        cfg = get_config()
        tg = cfg.get("telegram", {})

        if not tg.get("enabled", False):
            return

        bot_token = tg.get("bot_token", "")
        chat_id = tg.get("chat_id", "")
        severity_filter = tg.get("severities", ["P1", "P2"])

        if not bot_token or not chat_id:
            return

        if alert.get("severity") not in severity_filter:
            return

        caption = _format_caption(alert)

        if snapshot_path:
            _send_photo(bot_token, chat_id, snapshot_path, caption)
        else:
            _send_message(bot_token, chat_id, caption)

        logger.info("Telegram alert sent", extra={"alert_id": alert.get("id"), "camera_id": alert.get("cameraId")})
    except Exception:
        logger.exception("Telegram notification failed")


def test_connection(bot_token: str, chat_id: str) -> dict:
    """Test Telegram config by sending a test message."""
    try:
        url = f"{TELEGRAM_API.format(token=bot_token)}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": "SafetyLens — connection test successful.",
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return {"ok": True}
        data = resp.json()
        return {"ok": False, "error": data.get("description", resp.text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _format_caption(alert: dict) -> str:
    severity = alert.get("severity", "?")
    rule = alert.get("rule", "Unknown")
    camera = alert.get("cameraName", "Unknown")
    zone = alert.get("zone", "Unknown")
    desc = alert.get("description", "")
    ts = alert.get("timestamp", "")

    lines = [
        f"*{severity}* — {rule}",
        f"Camera: {camera} ({zone})",
    ]
    if desc:
        lines.append(desc)
    if ts:
        lines.append(f"Time: {ts[:19]}")
    return "\n".join(lines)


def _send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> None:
    url = f"{TELEGRAM_API.format(token=token)}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
                files={"photo": f},
                timeout=15,
            )
    except FileNotFoundError:
        # Snapshot file missing — send text-only
        _send_message(token, chat_id, caption)


def _send_message(token: str, chat_id: str, text: str) -> None:
    url = f"{TELEGRAM_API.format(token=token)}/sendMessage"
    requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
