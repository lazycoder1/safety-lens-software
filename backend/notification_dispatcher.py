"""
Central notification dispatcher for SafetyLens.
Routes alerts to the appropriate channels based on the channel matrix config.
"""

import hashlib
import json
import logging
import math
import re

from config_manager import get_config_snapshot
from delivery_result import DeliveryDisposition, ProviderDeliveryResult
import telegram_notifier
import email_notifier
import webhook_notifier

logger = logging.getLogger("safetylens.dispatcher")

# Retain the module-level name used by existing integrations/tests while making
# every non-durable test dispatch read one detached configuration generation.
get_config = get_config_snapshot

# Maps channel names to their notifier modules
_CHANNEL_HANDLERS = {
    "telegram": telegram_notifier,
    "email": email_notifier,
    "webhook": webhook_notifier,
}

_NOT_IMPLEMENTED = {"whatsapp", "sms", "plc"}
_SEVERITY_PRIORITY = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

# These are seeded examples, not active routing declarations.  Older installs
# have them in config even when no provider was ever configured, so treating
# them as obligations would create two terminal rows for every alert.
_BUILTIN_ESCALATION_EXAMPLES = (
    {"id": 1, "afterMinutes": 3, "role": "Floor Manager", "channel": "telegram"},
    {"id": 2, "afterMinutes": 10, "role": "Plant Manager", "channel": "email"},
)

_WEBHOOK_CREDENTIAL_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "x-auth-token",
    "x-access-token",
}
_WEBHOOK_STABLE_ACCOUNT_HEADERS = {
    "x-account-id",
    "x-tenant-id",
    "x-organization-id",
    "x-org-id",
    "x-workspace-id",
    "x-project-id",
}

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


def _target_fingerprint(value: object) -> str:
    """Legacy destination-only fingerprint retained for migration tests.

    New durable targets use ``_target_identity_fingerprint``.  Keeping this
    helper makes it explicit that old rows cannot accidentally match the v2
    account-bound identity after an upgrade.
    """
    if value in (None, ""):
        return "unconfigured"
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def _telegram_bot_boundary(token: object) -> dict:
    """Identify a Telegram bot while allowing a token rotation for that bot."""
    value = str(token or "").strip()
    match = re.fullmatch(r"([1-9][0-9]*):(.+)", value)
    if match:
        return {"kind": "bot_id", "value": match.group(1)}
    # A non-standard token has no independently verifiable account identifier.
    # Bind the opaque credential itself so a replacement fails closed.
    return {
        "kind": "opaque_token",
        "value": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _webhook_boundary(section: dict) -> dict:
    configured = section.get("headers", {})
    headers = configured if isinstance(configured, dict) else {}
    normalized = [
        (str(name).strip().lower(), str(name), str(value))
        for name, value in headers.items()
        if str(name).strip()
    ]
    stable_account = any(
        name in _WEBHOOK_STABLE_ACCOUNT_HEADERS and bool(value.strip())
        for name, _original_name, value in normalized
    )
    explicit_account = str(
        section.get("account_id") or section.get("tenant_id") or ""
    ).strip()

    if explicit_account or stable_account:
        # Once an independently stable account/tenant boundary is present,
        # bearer/API-key values may rotate without abandoning accepted work.
        # Header names remain bound so switching authentication mechanisms is
        # still visible.  Every non-credential routing header stays exact.
        bound_headers = sorted(
            (
                name,
                original_name,
                "<credential>" if name in _WEBHOOK_CREDENTIAL_HEADERS else value,
            )
            for name, original_name, value in normalized
        )
    else:
        # Without an independent account identifier, an auth credential may be
        # the only tenant boundary available.  Include it and fail closed on
        # rotation rather than risk delivering old work to a different tenant.
        bound_headers = sorted(normalized)

    return {
        "account_id": explicit_account,
        "headers": bound_headers,
    }


def _provider_boundary(cfg: dict, channel: str) -> dict:
    raw_section = cfg.get(channel, {})
    section = raw_section if isinstance(raw_section, dict) else {}
    if channel == "telegram":
        return _telegram_bot_boundary(section.get("bot_token"))
    if channel == "email":
        return {
            "smtp_host": str(section.get("smtp_host") or "").strip().lower(),
            "smtp_port": section.get("smtp_port", 587),
            "smtp_user": str(section.get("smtp_user") or "").strip(),
            "from_address": str(section.get("from_address") or "").strip(),
        }
    if channel == "webhook":
        return _webhook_boundary(section)
    return {"handler": channel}


def _target_identity_fingerprint(cfg: dict, channel: str, destination: object) -> str:
    """Hash destination plus provider account/routing boundary.

    Only the digest is stored in ``target_key``; credentials, tenant IDs,
    recipients and URLs never enter the durable row in reversible form.
    """
    if destination in (None, ""):
        return "unconfigured"
    canonical = json.dumps(
        {
            "version": 2,
            "channel": channel,
            "destination": str(destination).strip(),
            "provider": _provider_boundary(cfg, channel),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_builtin_escalation_example(step: dict) -> bool:
    comparable = {
        "id": step.get("id"),
        "afterMinutes": step.get("afterMinutes"),
        "role": step.get("role"),
        "channel": _normalize_channel(step.get("channel", "telegram")),
    }
    return comparable in _BUILTIN_ESCALATION_EXAMPLES


def _channel_destinations(cfg: dict, channel: str) -> list[object]:
    channel_cfg = cfg.get(channel, {})
    if not isinstance(channel_cfg, dict):
        return [None]
    if channel == "telegram":
        return [channel_cfg.get("chat_id")]
    if channel == "webhook":
        return [channel_cfg.get("url")]
    if channel == "email":
        recipients = channel_cfg.get("to_addresses", [])
        if isinstance(recipients, list) and recipients:
            return recipients
        return [None]
    return [None]


def resolve_delivery_targets(cfg: dict, alert: dict) -> list[dict]:
    """Snapshot one durable row per initial/escalation destination."""
    severity = str(alert.get("severity") or "P4")
    priority = _SEVERITY_PRIORITY.get(severity, 4)
    targets: list[dict] = []
    routing = cfg.get("alert_routing", {})
    if not isinstance(routing, dict):
        return targets
    matrix = routing.get("channel_matrix", {})
    if not isinstance(matrix, dict):
        matrix = {}
    severity_matrix = matrix.get(severity, {})
    if not isinstance(severity_matrix, dict):
        severity_matrix = {}

    for raw_channel, enabled in severity_matrix.items():
        channel = _normalize_channel(raw_channel)
        if not enabled or not channel or channel == "inapp":
            continue
        destinations = (
            _channel_destinations(cfg, channel)
            if channel in _CHANNEL_HANDLERS
            else [None]
        )
        for destination in destinations:
            fingerprint = _target_identity_fingerprint(cfg, channel, destination)
            targets.append({
                "kind": "initial",
                "channel": channel,
                "target_key": f"initial:{channel}:{fingerprint}",
                "context": _delivery_context(cfg, channel),
                "priority": priority,
                "delay_seconds": 0,
            })

    steps = routing.get("escalation_steps", [])
    if not isinstance(steps, list):
        steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("enabled") is False:
            continue
        explicitly_active = step.get("enabled") is True
        builtin_example = _is_builtin_escalation_example(step)
        if builtin_example and not explicitly_active:
            # Old installs persisted these UI examples without an activation
            # bit. Configuring a provider later must not silently turn sample
            # data into real escalation obligations.
            continue
        # Custom legacy steps and explicitly active steps are intentional.
        # Invalid routes become visible terminal obligations below.
        channel = _normalize_channel(step.get("channel", "telegram"))
        if not channel:
            continue

        step_severities = step.get("severities")
        if (
            explicitly_active
            and isinstance(step_severities, list)
            and severity not in step_severities
        ):
            continue

        step_id = step.get(
            "id",
            f"{step.get('afterMinutes', 0)}:{step.get('role', '')}:{channel}",
        )
        routing_error = None
        try:
            after_minutes = float(step.get("afterMinutes", 0))
        except (TypeError, ValueError):
            after_minutes = 0.0
            routing_error = "Escalation delay is invalid"
        if not math.isfinite(after_minutes) or after_minutes < 0:
            after_minutes = 0.0
            routing_error = "Escalation delay is invalid"
        destinations = (
            _channel_destinations(cfg, channel)
            if channel in _CHANNEL_HANDLERS
            else [None]
        )
        for destination in destinations:
            fingerprint = _target_identity_fingerprint(cfg, channel, destination)
            context = {
                **_delivery_context(cfg, channel),
                "step_id": str(step_id),
                "role": str(step.get("role") or "Manager")[:80],
                "after_minutes": after_minutes,
            }
            if routing_error:
                context["routing_error"] = routing_error
            targets.append({
                "kind": "escalation",
                "channel": channel,
                "target_key": f"escalation:{step_id}:{channel}:{fingerprint}",
                "context": context,
                "priority": priority,
                "delay_seconds": after_minutes * 60,
            })
    return targets


def _delivery_context(cfg: dict, channel: str) -> dict:
    """Freeze non-secret options that affect a retry's payload bytes."""
    channel_cfg = cfg.get(channel, {})
    if channel == "webhook" and isinstance(channel_cfg, dict):
        return {"include_snapshot": bool(channel_cfg.get("include_snapshot", False))}
    return {}


def deliver_outbox_target(
    alert: dict,
    target: dict,
    snapshot_path: str | None,
) -> ProviderDeliveryResult:
    """Deliver exactly one frozen outbox destination."""
    channel = _normalize_channel(target.get("channel"))
    context = target.get("context") or {}
    if isinstance(context, dict) and context.get("routing_error"):
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            str(context["routing_error"]),
            error_code="routing_configuration_invalid",
        )
    handler = _CHANNEL_HANDLERS.get(channel)
    if handler is None:
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "No channel handler is configured",
            error_code="handler_missing",
        )

    cfg = get_config_snapshot(channel)
    terminal = _terminal_channel_reason(cfg, channel, alert)
    if terminal is not None:
        reason_code, reason = terminal
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            reason,
            error_code=f"channel_{reason_code}",
        )

    target_key = str(target.get("target_key") or "")
    expected_fingerprint = target_key.rsplit(":", 1)[-1]
    matched_destination = None
    for destination in _channel_destinations(cfg, channel):
        if _target_identity_fingerprint(cfg, channel, destination) == expected_fingerprint:
            matched_destination = destination
            break
    if matched_destination in (None, ""):
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Delivery destination changed after the alert was accepted",
            error_code="target_configuration_changed",
        )

    outbound = dict(alert)
    outbound["deliveryId"] = str(target.get("id") or target_key)
    if target.get("kind") == "escalation":
        context = target.get("context") or {}
        role = str(context.get("role") or "Manager")
        after_minutes = context.get("after_minutes", 0)
        outbound["description"] = (
            f"[ESCALATED to {role} after {after_minutes:g}min] "
            f"{outbound.get('description', '')}"
        )

    if channel == "email":
        return email_notifier.send_alert_result(
            outbound,
            snapshot_path,
            to_addrs_override=[str(matched_destination)],
            email_config_override=cfg.get("email", {}),
        )
    if channel == "webhook":
        context = target.get("context") or {}
        return webhook_notifier.send_alert_result(
            outbound,
            snapshot_path,
            include_snapshot_override=bool(context.get("include_snapshot", False)),
            webhook_config_override=cfg.get("webhook", {}),
            url_override=str(matched_destination),
        )
    if channel == "telegram":
        return telegram_notifier.send_alert_result(
            outbound,
            snapshot_path,
            telegram_config_override=cfg.get("telegram", {}),
            chat_id_override=matched_destination,
        )
    return _invoke_handler(handler, outbound, snapshot_path)


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
                "status": "terminal",
                "message": "Channel is not implemented",
                "errorCode": "not_implemented",
            })
            continue

        handler = _CHANNEL_HANDLERS.get(channel)
        if not handler:
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "terminal",
                "message": "No channel handler is configured",
                "errorCode": "handler_missing",
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
                "status": "terminal" if reason_code == "invalid" else "skipped",
                "message": terminal_reason,
                "errorCode": f"channel_{reason_code}",
            })
            continue
        try:
            outcome = _invoke_handler(
                handler,
                alert,
                snapshot_path,
                channel=channel,
                config=cfg,
            )
            (logger.info if outcome.success else logger.warning)(
                "Notification channel outcome",
                extra={
                    "alert_id": alert.get("id"),
                    "channel": channel,
                    "outcome": outcome.disposition.value,
                    "error_code": outcome.error_code,
                    "provider_status": outcome.provider_status,
                    "retry_after_seconds": outcome.retry_after_seconds,
                    "acceptance_unknown": outcome.acceptance_unknown,
                },
            )
            results.append(outcome.to_dispatch_dict(result_channel))
        except Exception:
            logger.error(
                "Channel delivery raised unexpectedly",
                extra={"alert_id": alert.get("id"), "channel": channel},
            )
            results.append({
                "channel": result_channel,
                "success": False,
                "status": "retryable",
                "message": "Channel delivery failed unexpectedly",
                "errorCode": "unexpected_error",
                "acceptanceUnknown": True,
            })
    return results


def _invoke_handler(
    handler,
    alert: dict,
    snapshot_path: str | None,
    *,
    channel: str | None = None,
    config: dict | None = None,
) -> ProviderDeliveryResult:
    # Built-in handlers get the same immutable generation used by routing.
    # Third-party/test handlers keep the generic typed contract below.
    if config is not None and channel == "email" and handler is email_notifier:
        return handler.send_alert_result(
            alert,
            snapshot_path,
            email_config_override=config.get("email", {}),
        )
    if config is not None and channel == "telegram" and handler is telegram_notifier:
        return handler.send_alert_result(
            alert,
            snapshot_path,
            telegram_config_override=config.get("telegram", {}),
        )
    if config is not None and channel == "webhook" and handler is webhook_notifier:
        return handler.send_alert_result(
            alert,
            snapshot_path,
            webhook_config_override=config.get("webhook", {}),
        )
    typed_handler = getattr(handler, "send_alert_result", None)
    if callable(typed_handler):
        result = typed_handler(alert, snapshot_path)
        if not isinstance(result, ProviderDeliveryResult):
            raise TypeError("send_alert_result must return ProviderDeliveryResult")
        return result
    success = bool(handler.send_alert(alert, snapshot_path))
    return ProviderDeliveryResult(
        DeliveryDisposition.DELIVERED if success else DeliveryDisposition.RETRYABLE,
        "Delivered" if success else "Channel did not accept the alert",
        error_code=None if success else "legacy_boolean_failure",
        acceptance_unknown=not success,
    )


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
        logger.error("Notification dispatch failed unexpectedly")
        return False
