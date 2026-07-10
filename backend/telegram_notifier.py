"""
Telegram alert notifications for SafetyLens.
Uses raw requests to Telegram Bot API — no extra dependency needed.

Synchronous provider call, fenced and retried by the durable outbox worker.
"""

import logging
from collections.abc import Callable
from copy import deepcopy

import requests

from config_manager import get_config
from delivery_result import (
    DeliveryDisposition,
    ProviderDeliveryResult,
    parse_retry_after,
)
from secret_redaction import redact_text_secrets

logger = logging.getLogger("safetylens.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}"


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """Send an alert to Telegram and return whether Telegram accepted it."""
    return send_alert_result(alert, snapshot_path).success


def send_alert_result(
    alert: dict,
    snapshot_path: str | None = None,
    *,
    telegram_config_override: dict | None = None,
    chat_id_override: str | int | None = None,
) -> ProviderDeliveryResult:
    """Send an alert and preserve whether a rejection is safe to retry."""
    bot_token = ""
    try:
        tg = (
            deepcopy(telegram_config_override)
            if telegram_config_override is not None
            else deepcopy(get_config().get("telegram", {}))
        )

        if not tg.get("enabled", False):
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Telegram is disabled",
                error_code="channel_disabled",
            )

        bot_token = tg.get("bot_token", "")
        chat_id = (
            chat_id_override
            if chat_id_override is not None
            else tg.get("chat_id", "")
        )
        severity_filter = tg.get("severities", ["P1", "P2"])

        if not bot_token or not chat_id:
            return ProviderDeliveryResult(
                DeliveryDisposition.TERMINAL,
                "Telegram configuration is incomplete",
                error_code="invalid_configuration",
            )

        if alert.get("severity") not in severity_filter:
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Alert severity is filtered",
                error_code="severity_filtered",
            )

        text = _format_caption(alert)

        if snapshot_path:
            result, effective_chat_id, migration_used = _send_with_migration(
                lambda destination: _send_photo(
                    bot_token,
                    destination,
                    snapshot_path,
                    _truncate(text, 1024),
                ),
                chat_id,
                allow_migration=True,
            )
        else:
            result, effective_chat_id, migration_used = _send_with_migration(
                lambda destination: _send_message(
                    bot_token,
                    destination,
                    _truncate(text, 4096),
                ),
                chat_id,
                allow_migration=True,
            )

        if (
            snapshot_path
            and result.disposition is DeliveryDisposition.TERMINAL
            and result.provider_status == 400
            and result.error_code not in {
                "chat_migrated",
                "chat_migration_loop",
                "invalid_chat_migration",
            }
        ):
            # Telegram may reject image bytes/caption constraints while still
            # accepting the safety message. A single plain-text fallback avoids
            # dropping the entire channel without retrying a rate limit.
            result, _effective_chat_id, _fallback_migration_used = _send_with_migration(
                lambda destination: _send_message(
                    bot_token,
                    destination,
                    _truncate(text, 4096),
                ),
                effective_chat_id,
                allow_migration=not migration_used,
            )
        if not result.success:
            logger.warning(
                "Telegram rejected alert",
                extra={
                    "alert_id": alert.get("id"),
                    "error_code": result.error_code,
                    "provider_status": result.provider_status,
                    "retry_after_seconds": result.retry_after_seconds,
                },
            )
            return result

        logger.info("Telegram alert sent", extra={"alert_id": alert.get("id"), "camera_id": alert.get("cameraId")})
        return result
    except (
        requests.exceptions.InvalidSchema,
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
        requests.exceptions.TooManyRedirects,
    ):
        logger.error("Telegram endpoint configuration is invalid")
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Telegram endpoint configuration is invalid",
            error_code="invalid_endpoint",
        )
    except requests.exceptions.SSLError:
        logger.warning("Telegram TLS transport failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram TLS transport failed temporarily",
            error_code="tls_transport_error",
            acceptance_unknown=True,
        )
    except requests.exceptions.ConnectTimeout:
        logger.warning("Telegram connection timed out")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram is temporarily unreachable",
            error_code="connect_timeout",
        )
    except requests.exceptions.ReadTimeout:
        logger.warning("Telegram response timed out")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram acceptance could not be confirmed",
            error_code="read_timeout",
            acceptance_unknown=True,
        )
    except requests.exceptions.ConnectionError:
        logger.warning("Telegram connection failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram acceptance could not be confirmed",
            error_code="connection_error",
            acceptance_unknown=True,
        )
    except requests.exceptions.RequestException:
        logger.warning("Telegram request failed")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram request failed temporarily",
            error_code="request_error",
            acceptance_unknown=True,
        )
    except Exception:
        logger.error("Telegram notification failed unexpectedly")
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "Telegram delivery failed unexpectedly",
            error_code="unexpected_error",
            acceptance_unknown=True,
        )


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
            allow_redirects=False,
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
    desc = alert.get("description", "")
    ts = alert.get("timestamp", "")

    lines = [
        f"{severity} — {rule}",
        f"Camera: {camera} ({zone})",
    ]
    if desc:
        lines.append(desc)
    if ts:
        lines.append(f"Time: {ts[:19]}")
    return "\n".join(lines)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _send_photo(token: str, chat_id: str, photo_path: str, caption: str) -> requests.Response:
    url = f"{TELEGRAM_API.format(token=token)}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            return requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": f},
                timeout=15,
                allow_redirects=False,
            )
    except FileNotFoundError:
        # Snapshot file missing — send text-only
        return _send_message(token, chat_id, caption)


def _send_message(token: str, chat_id: str, text: str) -> requests.Response:
    url = f"{TELEGRAM_API.format(token=token)}/sendMessage"
    return requests.post(
        url,
        json={"chat_id": chat_id, "text": text},
        timeout=10,
        allow_redirects=False,
    )


def _send_with_migration(
    sender: Callable[[str | int], requests.Response],
    chat_id: str | int,
    *,
    allow_migration: bool,
) -> tuple[ProviderDeliveryResult, str | int, bool]:
    """Follow at most one Telegram-provided chat migration for this delivery."""
    response = sender(chat_id)
    has_migration, migrated_chat_id = _migration_chat_id(response)
    if not has_migration:
        return _classify_response(response), chat_id, False
    if not allow_migration:
        return _migration_failure(response, "chat_migration_loop"), chat_id, False
    if migrated_chat_id is None:
        return _migration_failure(response, "invalid_chat_migration"), chat_id, False

    migrated_response = sender(migrated_chat_id)
    repeated_migration, _ignored_chat_id = _migration_chat_id(migrated_response)
    if repeated_migration:
        return (
            _migration_failure(migrated_response, "chat_migration_loop"),
            migrated_chat_id,
            True,
        )
    return _classify_response(migrated_response), migrated_chat_id, True


def _migration_chat_id(response: requests.Response) -> tuple[bool, int | None]:
    try:
        payload = response.json()
    except Exception:
        return False, None
    if not isinstance(payload, dict):
        return False, None
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or "migrate_to_chat_id" not in parameters:
        return False, None
    raw_chat_id = parameters.get("migrate_to_chat_id")
    if type(raw_chat_id) is not int:
        return True, None
    chat_id = raw_chat_id
    if chat_id == 0 or not -(2**63) <= chat_id <= (2**63) - 1:
        return True, None
    return True, chat_id


def _migration_failure(
    response: requests.Response,
    error_code: str,
) -> ProviderDeliveryResult:
    try:
        provider_status = int(response.status_code)
    except (TypeError, ValueError):
        provider_status = None
    return ProviderDeliveryResult(
        DeliveryDisposition.TERMINAL,
        (
            "Telegram returned an invalid chat migration"
            if error_code == "invalid_chat_migration"
            else "Telegram repeated a chat migration"
        ),
        error_code=error_code,
        provider_status=provider_status,
    )


def _classify_response(response: requests.Response) -> ProviderDeliveryResult:
    http_status = int(response.status_code)
    try:
        payload = response.json()
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        retryable = (
            200 <= http_status < 300
            or http_status in {408, 425, 429}
            or 500 <= http_status < 600
        )
        disposition = DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL
        return ProviderDeliveryResult(
            disposition,
            "Telegram returned an invalid response",
            error_code="invalid_response",
            provider_status=http_status,
            retry_after_seconds=(
                _response_retry_after(response)
                if disposition is DeliveryDisposition.RETRYABLE
                else None
            ),
            acceptance_unknown=(
                200 <= http_status < 300
                or http_status == 408
                or 500 <= http_status < 600
            ),
        )

    if 200 <= http_status < 300 and payload.get("ok") is True:
        return ProviderDeliveryResult(
            DeliveryDisposition.DELIVERED,
            "Delivered",
            provider_status=http_status,
        )

    try:
        api_status = int(payload.get("error_code") or http_status)
    except (TypeError, ValueError):
        api_status = http_status
    parameters = payload.get("parameters")
    if isinstance(parameters, dict) and parameters.get("migrate_to_chat_id") is not None:
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Telegram returned a chat migration that was not followed",
            error_code="chat_migrated",
            provider_status=api_status,
        )
    retryable = (
        http_status in {408, 425, 429}
        or 500 <= http_status < 600
        or api_status in {408, 425, 429}
        or 500 <= api_status < 600
    )
    retry_after = None
    if retryable:
        if isinstance(parameters, dict):
            retry_after = parse_retry_after(parameters.get("retry_after"))
        if retry_after is None:
            retry_after = _response_retry_after(response)
    return ProviderDeliveryResult(
        DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL,
        "Telegram temporarily rejected the request" if retryable else "Telegram rejected the request",
        error_code=f"telegram_{api_status}",
        provider_status=api_status,
        retry_after_seconds=retry_after,
        acceptance_unknown=(
            http_status == 408 or 500 <= http_status < 600
        ),
    )


def _response_retry_after(response: requests.Response) -> float | None:
    headers = getattr(response, "headers", {})
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    return parse_retry_after(value)


def _validate_response(response: requests.Response) -> dict:
    """Raise when Telegram rejects a request, including HTTP-200 API errors."""
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Telegram returned invalid JSON (HTTP {response.status_code})") from exc

    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"Telegram HTTP {response.status_code}: request rejected")
    if payload.get("ok") is not True:
        raise RuntimeError("Telegram API rejected request")
    return payload


def fetch_groups(bot_token: str) -> dict:
    """Call getUpdates and return unique group/supergroup chats the bot belongs to."""
    try:
        url = f"{TELEGRAM_API.format(token=bot_token)}/getUpdates"
        resp = requests.get(url, timeout=10, allow_redirects=False)
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
