"""
Email alert notifications for SafetyLens.
Uses stdlib smtplib + email.mime — no extra dependency needed.

Synchronous provider call, fenced and retried by the durable outbox worker.
"""

import logging
import smtplib
import socket
import ssl
from copy import deepcopy
from html import escape
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config_manager import get_config
from delivery_result import (
    DeliveryDisposition,
    ProviderDeliveryResult,
    stable_delivery_identity,
)

logger = logging.getLogger("safetylens.email")


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """Send an email alert and return whether SMTP accepted it."""
    return send_alert_result(alert, snapshot_path).success


def send_alert_result(
    alert: dict,
    snapshot_path: str | None = None,
    *,
    to_addrs_override: list[str] | None = None,
    email_config_override: dict | None = None,
) -> ProviderDeliveryResult:
    """Send an email and preserve SMTP retry/partial-acceptance semantics."""
    to_addrs: list[str] = []
    try:
        email_cfg = (
            deepcopy(email_config_override)
            if email_config_override is not None
            else deepcopy(get_config().get("email", {}))
        )

        if not email_cfg.get("enabled", False):
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Email is disabled",
                error_code="channel_disabled",
            )

        smtp_host = email_cfg.get("smtp_host", "")
        smtp_port = email_cfg.get("smtp_port", 587)
        smtp_user = email_cfg.get("smtp_user", "")
        smtp_pass = email_cfg.get("smtp_pass", "")
        from_addr = email_cfg.get("from_address", "")
        configured_to_addrs = email_cfg.get("to_addresses", [])
        to_addrs = (
            list(to_addrs_override)
            if to_addrs_override is not None
            else configured_to_addrs
        )
        severity_filter = email_cfg.get("severities", ["P1", "P2"])

        if not smtp_host or not from_addr or not to_addrs:
            return ProviderDeliveryResult(
                DeliveryDisposition.TERMINAL,
                "Email configuration is incomplete",
                error_code="invalid_configuration",
            )

        if alert.get("severity") not in severity_filter:
            return ProviderDeliveryResult(
                DeliveryDisposition.SKIPPED,
                "Alert severity is filtered",
                error_code="severity_filtered",
            )

        subject, html_body = _build_email(alert, snapshot_path)
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg["Message-ID"] = _message_id(alert)

        msg.attach(MIMEText(html_body, "html"))

        if snapshot_path:
            _attach_snapshot(msg, snapshot_path)

        _send(smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, to_addrs, msg)

        logger.info("Email alert sent", extra={"alert_id": alert.get("id"), "camera_id": alert.get("cameraId")})
        return ProviderDeliveryResult(
            DeliveryDisposition.DELIVERED,
            "Delivered",
        )
    except Exception as exc:
        result = _classify_smtp_failure(exc, to_addrs)
        log = logger.warning if result.retryable else logger.error
        log(
            "Email delivery failed",
            extra={
                "alert_id": alert.get("id"),
                "error_code": result.error_code,
                "provider_status": result.provider_status,
                "acceptance_unknown": result.acceptance_unknown,
            },
        )
        return result


def test_connection(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
                    from_addr: str, to_addr: str) -> dict:
    """Send a test email to verify SMTP configuration."""
    try:
        msg = MIMEText(
            "<h3>SafetyLens — Email Test</h3>"
            "<p>If you see this message, your email configuration is working correctly.</p>",
            "html",
        )
        msg["Subject"] = "[SafetyLens] Connection Test"
        msg["From"] = from_addr
        msg["To"] = to_addr

        _send(smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, [to_addr], msg)
        return {"ok": True}
    except Exception as exc:
        result = _classify_smtp_failure(exc, [to_addr])
        return {
            "ok": False,
            "error": result.message,
            "errorCode": result.error_code,
            "retryable": result.retryable,
        }


def _message_id(alert: dict) -> str:
    return f"<safetylens-{stable_delivery_identity(alert)}@alerts.safetylens.local>"


def _classify_smtp_failure(
    exc: Exception,
    recipients: list[str],
) -> ProviderDeliveryResult:
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        refused = exc.recipients if isinstance(exc.recipients, dict) else {}
        codes = [
            int(value[0])
            for value in refused.values()
            if isinstance(value, (tuple, list)) and value and str(value[0]).isdigit()
        ]
        all_refused = bool(recipients) and len(refused) >= len(set(recipients))
        provider_status: int | str | None = None
        unique_codes = sorted(set(codes))
        if len(unique_codes) == 1:
            provider_status = unique_codes[0]
        elif unique_codes:
            provider_status = ",".join(str(code) for code in unique_codes)
        if not all_refused:
            return ProviderDeliveryResult(
                DeliveryDisposition.RETRYABLE,
                "SMTP accepted only some recipients; duplicate delivery is possible on retry",
                error_code="partial_recipient_refusal",
                provider_status=provider_status,
                acceptance_unknown=True,
            )
        retryable = bool(codes) and all(400 <= code < 500 for code in codes)
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL,
            "SMTP temporarily refused all recipients" if retryable else "SMTP rejected all recipients",
            error_code="recipients_refused",
            provider_status=provider_status,
        )

    if isinstance(exc, smtplib.SMTPResponseException):
        code = int(exc.smtp_code)
        retryable = 400 <= code < 500
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL,
            "SMTP returned a temporary error" if retryable else "SMTP rejected the message",
            error_code=f"smtp_{code}",
            provider_status=code,
        )

    if isinstance(exc, (ssl.SSLCertVerificationError, smtplib.SMTPNotSupportedError)):
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "SMTP TLS or protocol configuration is invalid",
            error_code="smtp_configuration_error",
        )

    if isinstance(exc, ssl.SSLError):
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "SMTP TLS transport failed temporarily",
            error_code="tls_transport_error",
            acceptance_unknown=True,
        )

    if isinstance(exc, socket.gaierror):
        retryable = exc.errno != socket.EAI_NONAME
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE if retryable else DeliveryDisposition.TERMINAL,
            "SMTP hostname lookup failed temporarily" if retryable else "SMTP hostname is invalid",
            error_code="dns_temporary" if retryable else "dns_invalid_hostname",
        )

    if isinstance(
        exc,
        (smtplib.SMTPServerDisconnected, TimeoutError, ConnectionError, OSError),
    ):
        return ProviderDeliveryResult(
            DeliveryDisposition.RETRYABLE,
            "SMTP acceptance could not be confirmed",
            error_code="smtp_transport_error",
            acceptance_unknown=True,
        )

    if isinstance(exc, (ValueError, TypeError)):
        return ProviderDeliveryResult(
            DeliveryDisposition.TERMINAL,
            "Email payload or address configuration is invalid",
            error_code="invalid_payload",
        )

    return ProviderDeliveryResult(
        DeliveryDisposition.RETRYABLE,
        "Email delivery failed unexpectedly",
        error_code="unexpected_error",
        acceptance_unknown=True,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {"P1": "#dc2626", "P2": "#ea580c", "P3": "#ca8a04", "P4": "#2563eb"}


def _build_email(alert: dict, snapshot_path: str | None) -> tuple[str, str]:
    """Return (subject, html_body) for an alert email."""
    raw_severity = str(alert.get("severity", "?"))
    severity = escape(raw_severity, quote=True)
    rule = escape(str(alert.get("rule", "Unknown")), quote=True)
    camera = escape(str(alert.get("cameraName", "Unknown")), quote=True)
    zone = escape(str(alert.get("zone", "Unknown")), quote=True)
    desc = escape(str(alert.get("description", "")), quote=True)
    ts = escape(str(alert.get("timestamp", ""))[:19], quote=True)
    try:
        confidence_text = f"{float(alert.get('confidence', 0)):.0%}"
    except (TypeError, ValueError, OverflowError):
        confidence_text = str(alert.get("confidence", "Unknown"))
    confidence = escape(confidence_text, quote=True)
    color = _SEVERITY_COLORS.get(raw_severity, "#6b7280")

    # MIME headers reject CR/LF, but remove them here as an explicit boundary
    # because the alert fields can originate outside this process.
    subject = (
        f"[SafetyLens {severity}] {rule} — {camera}"
        .replace("\r", " ")
        .replace("\n", " ")
    )

    snapshot_html = ""
    if snapshot_path and Path(snapshot_path).exists():
        snapshot_html = '<img src="cid:snapshot" style="max-width:600px;border-radius:6px;margin-top:12px;" />'

    html = f"""\
<div style="font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;">
  <div style="background:{color};color:#fff;padding:12px 20px;border-radius:8px 8px 0 0;">
    <strong>{severity}</strong> &mdash; {rule}
  </div>
  <div style="border:1px solid #e5e7eb;border-top:none;padding:20px;border-radius:0 0 8px 8px;">
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
      <tr><td style="padding:4px 8px;color:#6b7280;">Camera</td><td style="padding:4px 8px;">{camera}</td></tr>
      <tr><td style="padding:4px 8px;color:#6b7280;">Zone</td><td style="padding:4px 8px;">{zone}</td></tr>
      <tr><td style="padding:4px 8px;color:#6b7280;">Confidence</td><td style="padding:4px 8px;">{confidence}</td></tr>
      <tr><td style="padding:4px 8px;color:#6b7280;">Time</td><td style="padding:4px 8px;">{ts}</td></tr>
    </table>
    {"<p style='margin-top:12px;'>" + desc + "</p>" if desc else ""}
    {snapshot_html}
    <p style="margin-top:16px;font-size:12px;color:#9ca3af;">
      This is an automated alert from SafetyLens. Do not reply to this email.
    </p>
  </div>
</div>"""
    return subject, html


def _attach_snapshot(msg: MIMEMultipart, snapshot_path: str) -> None:
    """Attach snapshot image as inline CID for HTML embedding."""
    try:
        with open(snapshot_path, "rb") as f:
            img = MIMEImage(f.read(), _subtype="jpeg")
            img.add_header("Content-ID", "<snapshot>")
            img.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
            msg.attach(img)
    except FileNotFoundError:
        pass


def _send(host: str, port: int, user: str, password: str,
          from_addr: str, to_addrs: list[str], msg: MIMEMultipart | MIMEText) -> None:
    """Connect to SMTP server and send message."""
    # Be explicit rather than relying on smtplib defaults: this enables CA
    # verification and hostname checking for both implicit TLS and STARTTLS.
    tls_context = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=15, context=tls_context)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
        server.ehlo()
        server.starttls(context=tls_context)
        server.ehlo()

    try:
        if user and password:
            server.login(user, password)
        refused_recipients = server.sendmail(from_addr, to_addrs, msg.as_string())
        if refused_recipients:
            # SMTP may accept only a subset. Reporting the channel as delivered
            # would silently hide the rejected safety-alert recipients.
            raise smtplib.SMTPRecipientsRefused(refused_recipients)
    except Exception:
        try:
            server.quit()
        except Exception:
            logger.debug("SMTP cleanup failed after send failure", exc_info=True)
        raise

    # Once sendmail succeeds, a QUIT failure does not mean the server rejected
    # the message. Treating it as a delivery failure would trigger duplicates.
    try:
        server.quit()
    except Exception:
        logger.warning("SMTP cleanup failed after message was accepted", exc_info=True)
