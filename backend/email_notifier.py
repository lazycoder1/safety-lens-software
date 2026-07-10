"""
Email alert notifications for Rakshak Lens.
Uses stdlib smtplib + email.mime — no extra dependency needed.

Sync calls, fire-and-forget. Called from notification dispatcher.
"""

import logging
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config_manager import get_config

logger = logging.getLogger("rakshak_lens.email")


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """Send an email alert and return whether SMTP accepted it."""
    try:
        cfg = get_config()
        email_cfg = cfg.get("email", {})

        if not email_cfg.get("enabled", False):
            return False

        smtp_host = email_cfg.get("smtp_host", "")
        smtp_port = email_cfg.get("smtp_port", 587)
        smtp_user = email_cfg.get("smtp_user", "")
        smtp_pass = email_cfg.get("smtp_pass", "")
        from_addr = email_cfg.get("from_address", "")
        to_addrs = email_cfg.get("to_addresses", [])
        severity_filter = email_cfg.get("severities", ["P1", "P2"])

        if not smtp_host or not from_addr or not to_addrs:
            return False

        if alert.get("severity") not in severity_filter:
            return False

        subject, html_body = _build_email(alert, snapshot_path)
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        msg.attach(MIMEText(html_body, "html"))

        if snapshot_path:
            _attach_snapshot(msg, snapshot_path)

        _send(smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, to_addrs, msg)

        logger.info("Email alert sent", extra={"alert_id": alert.get("id"), "camera_id": alert.get("cameraId")})
        return True
    except Exception:
        logger.exception("Email notification failed")
        return False


def test_connection(smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str,
                    from_addr: str, to_addr: str) -> dict:
    """Send a test email to verify SMTP configuration."""
    try:
        msg = MIMEText(
            "<h3>Rakshak Lens — Email Test</h3>"
            "<p>If you see this message, your email configuration is working correctly.</p>",
            "html",
        )
        msg["Subject"] = "[Rakshak Lens] Connection Test"
        msg["From"] = from_addr
        msg["To"] = to_addr

        _send(smtp_host, smtp_port, smtp_user, smtp_pass, from_addr, [to_addr], msg)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def build_message(
    subject: str,
    html_body: str,
    from_addr: str,
    to_addrs: list[str],
    snapshot_path: str | None = None,
) -> MIMEMultipart:
    """Build an HTML alert email with an optional inline snapshot."""
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))
    if snapshot_path:
        _attach_snapshot(msg, snapshot_path)
    return msg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {"P1": "#dc2626", "P2": "#ea580c", "P3": "#ca8a04", "P4": "#2563eb"}


def _build_email(alert: dict, snapshot_path: str | None) -> tuple[str, str]:
    """Return (subject, html_body) for an alert email."""
    severity = alert.get("severity", "?")
    rule = alert.get("rule", "Unknown")
    camera = alert.get("cameraName", "Unknown")
    zone = alert.get("zone", "Unknown")
    desc = alert.get("message") or alert.get("description", "")
    ts = alert.get("timestamp", "")[:19]
    confidence = alert.get("confidence", 0)
    color = _SEVERITY_COLORS.get(severity, "#6b7280")

    subject = f"[Rakshak Lens {severity}] {rule} — {camera}"

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
      <tr><td style="padding:4px 8px;color:#6b7280;">Confidence</td><td style="padding:4px 8px;">{confidence:.0%}</td></tr>
      <tr><td style="padding:4px 8px;color:#6b7280;">Time</td><td style="padding:4px 8px;">{ts}</td></tr>
    </table>
    {"<p style='margin-top:12px;'>" + desc + "</p>" if desc else ""}
    {snapshot_html}
    <p style="margin-top:16px;font-size:12px;color:#9ca3af;">
      This is an automated alert from Rakshak Lens. Do not reply to this email.
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


def _send(
    host: str,
    port: int,
    user: str,
    password: str,
    from_addr: str,
    to_addrs: list[str],
    msg: MIMEMultipart | MIMEText,
    *,
    use_tls: bool = True,
) -> None:
    """Connect to SMTP server and send message."""
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()

    try:
        if user and password:
            server.login(user, password)
        server.sendmail(from_addr, to_addrs, msg.as_string())
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
