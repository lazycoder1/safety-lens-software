"""
Webhook alert notifications for Rakshak Lens.
HTTP POST JSON payloads to a configurable endpoint.

Sync calls, fire-and-forget. Called from notification dispatcher.
"""

import base64
import logging
from pathlib import Path

import requests

from config_manager import get_config
from secret_redaction import REDACTED_VALUE, redact_text_secrets

logger = logging.getLogger("rakshak_lens.webhook")


def send_alert(alert: dict, snapshot_path: str | None = None) -> bool:
    """POST an alert and return whether the endpoint accepted it."""
    url = ""
    try:
        cfg = get_config()
        wh = cfg.get("webhook", {})

        if not wh.get("enabled", False):
            return False

        url = wh.get("url", "")
        if not url:
            return False

        severity_filter = wh.get("severities", ["P1", "P2"])
        if alert.get("severity") not in severity_filter:
            return False

        payload = _build_payload(alert, snapshot_path, include_snapshot=wh.get("include_snapshot", False))
        headers = {"Content-Type": "application/json"}
        headers.update(wh.get("headers", {}))

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if not 200 <= response.status_code < 300:
            logger.error(
                "Webhook rejected alert (HTTP %s)",
                response.status_code,
                extra={"alert_id": alert.get("id"), "url": REDACTED_VALUE},
            )
            return False

        logger.info("Webhook alert sent", extra={"alert_id": alert.get("id"), "url": REDACTED_VALUE})
        return True
    except Exception as exc:
        safe_error = redact_text_secrets(str(exc), [url])
        logger.error("Webhook notification failed: %s", safe_error)
        return False


def test_connection(url: str, headers: dict | None = None) -> dict:
    """Send a test payload to the webhook URL."""
    try:
        test_payload = {
            "type": "test",
            "source": "Rakshak Lens",
            "message": "Webhook connection test successful.",
        }
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)

        resp = requests.post(url, json=test_payload, headers=req_headers, timeout=10)

        if 200 <= resp.status_code < 300:
            return {"ok": True}
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": redact_text_secrets(str(e), [url])}


def _build_payload(alert: dict, snapshot_path: str | None, include_snapshot: bool) -> dict:
    """Build the JSON payload to POST."""
    payload = {
        "source": "Rakshak Lens",
        "event": "alert",
        "alert": {
            "id": alert.get("id"),
            "severity": alert.get("severity"),
            "status": alert.get("status"),
            "rule": alert.get("rule"),
            "cameraId": alert.get("cameraId"),
            "cameraName": alert.get("cameraName"),
            "zone": alert.get("zone"),
            "confidence": alert.get("confidence"),
            "timestamp": alert.get("timestamp"),
            "description": alert.get("description"),
            "message": alert.get("message"),
            "policyId": alert.get("policyId"),
            "priority": alert.get("priority"),
        },
    }

    if include_snapshot and snapshot_path:
        try:
            data = Path(snapshot_path).read_bytes()
            payload["snapshot_base64"] = base64.b64encode(data).decode("ascii")
        except FileNotFoundError:
            pass

    return payload
