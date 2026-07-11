"""Tests for alert notification dispatch adapters."""

import notification_dispatcher


def test_email_output_can_disable_starttls_for_local_capture(monkeypatch):
    sent = {}

    def fake_send(host, port, user, password, from_addr, to_addrs, msg, *, use_tls=True):
        sent.update(
            {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "from_addr": from_addr,
                "to_addrs": to_addrs,
                "use_tls": use_tls,
                "subject": msg["Subject"],
            }
        )

    monkeypatch.setattr(notification_dispatcher.email_notifier, "_send", fake_send)
    monkeypatch.setattr(notification_dispatcher, "get_config", lambda: {"email": {}})

    result = notification_dispatcher._send_email(
        {
            "id": "local_email_capture",
            "name": "Local Email Capture",
            "type": "email",
            "enabled": True,
            "severities": ["P4"],
            "settings": {
                "smtp_host": "127.0.0.1",
                "smtp_port": 18081,
                "smtp_user": "",
                "smtp_pass": "",
                "use_tls": False,
                "from_address": "alerts@example.test",
                "to_addresses": ["qa@example.test"],
            },
        },
        {
            "id": "alert-1",
            "severity": "P4",
            "rule": "Person Detected",
            "cameraName": "Eval Camera",
            "zone": "Lane",
            "confidence": 0.9,
            "message": "email delivery proof",
        },
        None,
    )

    assert result["status"] == notification_dispatcher.DELIVERED
    assert sent["host"] == "127.0.0.1"
    assert sent["port"] == 18081
    assert sent["to_addrs"] == ["qa@example.test"]
    assert sent["use_tls"] is False
    assert "Person Detected" in sent["subject"]
