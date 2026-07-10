"""
Telegram alert notifications for Rakshak Lens.
Uses raw requests to Telegram Bot API — no extra dependency needed.

Sync calls, fire-and-forget. Called from video processing threads.
"""

import logging

import requests

from config_manager import get_config
from secret_redaction import redact_text_secrets

logger = logging.getLogger("rakshak_lens.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """Send an alert to Telegram and return whether Telegram accepted it."""
    bot_token = ""
    try:
        cfg = get_config()
        tg = cfg.get("telegram", {})

        if not tg.get("enabled", False):
            return False

        bot_token = tg.get("bot_token", "")
        chat_id = tg.get("chat_id", "")
        severity_filter = tg.get("severities", ["P1", "P2"])

        if not bot_token or not chat_id:
            return False

        if alert.get("severity") not in severity_filter:
            return False

        caption = _format_caption(alert)

        if snapshot_path:
            _send_photo(bot_token, chat_id, snapshot_path, caption)
        else:
            _send_message(bot_token, chat_id, caption)

        logger.info("Telegram alert sent", extra={"alert_id": alert.get("id"), "camera_id": alert.get("cameraId")})
        return True
    except Exception as exc:
        safe_error = redact_text_secrets(str(exc), [bot_token])
        logger.error("Telegram notification failed: %s", safe_error)
        return False


def test_connection(bot_token: str, chat_id: str) -> dict:
    """Test Telegram config by sending a test message."""
    try:
        url = f"{TELEGRAM_API.format(token=bot_token)}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": "Rakshak Lens — connection test successful.",
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        _validate_response(resp)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": redact_text_secrets(str(e), [bot_token])}


def _format_caption(alert: dict) -> str:
    severity = alert.get("severity", "?")
    rule = alert.get("rule", "Unknown")
    camera = alert.get("cameraName", "Unknown")
    zone = alert.get("zone", "Unknown")
    desc = alert.get("message") or alert.get("description", "")
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
            response = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
                files={"photo": f},
                timeout=15,
            )
            _validate_response(response)
    except FileNotFoundError:
        # Snapshot file missing — send text-only
        _send_message(token, chat_id, caption)


def _send_message(token: str, chat_id: str, text: str) -> None:
    url = f"{TELEGRAM_API.format(token=token)}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
    _validate_response(response)


def _validate_response(response: requests.Response) -> dict:
    """Raise when Telegram rejects a request, including HTTP-200 API errors."""
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Telegram returned invalid JSON (HTTP {response.status_code})") from exc

    description = str(payload.get("description") or response.text or "request rejected")
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"Telegram HTTP {response.status_code}: {description}")
    if payload.get("ok") is not True:
        raise RuntimeError(f"Telegram API rejected request: {description}")
    return payload


def fetch_groups(bot_token: str) -> dict:
    """Call getUpdates and return unique group/supergroup chats the bot belongs to."""
    try:
        url = f"{TELEGRAM_API.format(token=bot_token)}/getUpdates"
        resp = requests.get(url, timeout=10)
        results = _validate_response(resp).get("result", [])
        seen: dict[str, dict] = {}
        for update in results:
            msg = update.get("message") or update.get("my_chat_member", {}).get("chat")
            if not msg:
                continue
            chat = msg.get("chat") or msg
            chat_type = chat.get("type", "")
            if chat_type in ("group", "supergroup"):
                chat_id = str(chat.get("id"))
                if chat_id not in seen:
                    seen[chat_id] = {
                        "chat_id": chat_id,
                        "title": chat.get("title", f"Group {chat_id}"),
                        "type": chat_type,
                    }
        return {"ok": True, "groups": list(seen.values())}
    except Exception as e:
        return {"ok": False, "error": redact_text_secrets(str(e), [bot_token]), "groups": []}
