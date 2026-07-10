from types import SimpleNamespace

import email_notifier
import webhook_notifier


def _alert():
    return {
        "id": "alert-1",
        "severity": "P1",
        "rule": "No Helmet",
        "cameraName": "Camera 1",
        "zone": "Line 1",
        "confidence": 0.9,
        "timestamp": "2026-07-10T10:00:00+00:00",
    }


def test_email_reports_smtp_acceptance(monkeypatch):
    monkeypatch.setattr(
        email_notifier,
        "get_config",
        lambda: {
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
                "smtp_user": "",
                "smtp_pass": "",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
        },
    )
    monkeypatch.setattr(email_notifier, "_send", lambda *_args, **_kwargs: None)

    assert email_notifier.send_alert(_alert()) is True


def test_email_reports_smtp_failure(monkeypatch):
    monkeypatch.setattr(
        email_notifier,
        "get_config",
        lambda: {
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
        },
    )

    def fail_send(*_args, **_kwargs):
        raise OSError("SMTP unavailable")

    monkeypatch.setattr(email_notifier, "_send", fail_send)

    assert email_notifier.send_alert(_alert()) is False


def test_email_remains_delivered_when_smtp_quit_fails_after_acceptance(monkeypatch):
    class AcceptedThenQuitFails:
        def __init__(self):
            self.sendmail_calls = 0
            self.quit_calls = 0

        def login(self, *_args):
            raise AssertionError("login should not be attempted without credentials")

        def sendmail(self, *_args):
            self.sendmail_calls += 1
            return {}

        def quit(self):
            self.quit_calls += 1
            raise OSError("connection closed after acceptance")

    smtp = AcceptedThenQuitFails()
    monkeypatch.setattr(email_notifier.smtplib, "SMTP_SSL", lambda *_args, **_kwargs: smtp)
    monkeypatch.setattr(
        email_notifier,
        "get_config",
        lambda: {
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "smtp_port": 465,
                "smtp_user": "",
                "smtp_pass": "",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
        },
    )

    assert email_notifier.send_alert(_alert()) is True
    assert smtp.sendmail_calls == 1
    assert smtp.quit_calls == 1


def test_email_reports_partial_recipient_refusal_as_failure(monkeypatch):
    class PartiallyRefused:
        def sendmail(self, *_args):
            return {"missed@example.com": (550, b"mailbox unavailable")}

        def quit(self):
            return None

    monkeypatch.setattr(
        email_notifier.smtplib,
        "SMTP_SSL",
        lambda *_args, **_kwargs: PartiallyRefused(),
    )
    monkeypatch.setattr(
        email_notifier,
        "get_config",
        lambda: {
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "smtp_port": 465,
                "smtp_user": "",
                "smtp_pass": "",
                "from_address": "alerts@example.com",
                "to_addresses": ["accepted@example.com", "missed@example.com"],
                "severities": ["P1"],
            },
        },
    )

    assert email_notifier.send_alert(_alert()) is False


def test_webhook_reports_http_acceptance_and_rejection(monkeypatch):
    monkeypatch.setattr(
        webhook_notifier,
        "get_config",
        lambda: {
            "webhook": {
                "enabled": True,
                "url": "https://hooks.example.com/alerts",
                "severities": ["P1"],
            },
        },
    )
    responses = [SimpleNamespace(status_code=500), SimpleNamespace(status_code=204)]
    monkeypatch.setattr(webhook_notifier.requests, "post", lambda *_args, **_kwargs: responses.pop(0))

    assert webhook_notifier.send_alert(_alert()) is False
    assert webhook_notifier.send_alert(_alert()) is True
