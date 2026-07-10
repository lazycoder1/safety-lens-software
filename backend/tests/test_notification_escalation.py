from types import SimpleNamespace

import notification_dispatcher


def test_notify_with_results_normalizes_requested_channel_casing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "telegram": {
                "enabled": True,
                "bot_token": "token",
                "chat_id": "chat",
                "severities": ["P1"],
            },
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
        },
    )
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {
            "telegram": SimpleNamespace(
                send_alert=lambda *_args: calls.append("telegram") or True
            ),
            "email": SimpleNamespace(
                send_alert=lambda *_args: calls.append("email") or False
            ),
        },
    )

    results = notification_dispatcher.notify_with_results(
        {"severity": "P1"},
        channels=["Telegram", "telegram", "EMAIL", "SMS", "inApp"],
        test_request=True,
    )

    assert calls == ["telegram", "email"]
    assert [
        (result["channel"], result["success"], result["status"])
        for result in results
    ] == [
        ("telegram", True, "delivered"),
        ("email", False, "retryable"),
        ("sms", False, "terminal"),
        ("inApp", False, "skipped"),
    ]


def test_explicit_channels_are_real_delivery_targets_unless_marked_as_test(monkeypatch):
    monkeypatch.setattr(notification_dispatcher, "get_config", lambda: {})

    results = notification_dispatcher.notify_with_results(
        {"severity": "P1"},
        channels=["inApp"],
    )

    assert results == [
        {
            "channel": "inApp",
            "success": True,
            "status": "handled",
            "message": "Handled by the persisted alert and WebSocket path",
        }
    ]


def test_disabled_incomplete_and_filtered_channels_are_terminal_without_send(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {
            channel: SimpleNamespace(
                send_alert=lambda *_args, channel=channel: calls.append(channel) or True
            )
            for channel in ("telegram", "email", "webhook")
        },
    )
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "telegram": {"enabled": False},
            "email": {
                "enabled": True,
                "smtp_host": "",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
            "webhook": {
                "enabled": True,
                "url": "https://hooks.example.com/alerts",
                "severities": ["P2"],
            },
        },
    )

    results = notification_dispatcher.notify_with_results(
        {"severity": "P1"},
        channels=["telegram", "email", "webhook"],
    )

    assert calls == []
    assert [
        (result["channel"], result["status"], result["message"])
        for result in results
    ] == [
        ("telegram", "skipped", "Channel is disabled"),
        ("email", "terminal", "Channel configuration is incomplete"),
        ("webhook", "skipped", "Severity P1 is filtered"),
    ]


def test_implicit_routing_treats_globally_disabled_matrix_channels_as_inactive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notification_dispatcher,
        "_CHANNEL_HANDLERS",
        {
            channel: SimpleNamespace(
                send_alert=lambda *_args, channel=channel: calls.append(channel) or True
            )
            for channel in ("telegram", "email", "webhook")
        },
    )
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "telegram": {"enabled": False},
            "email": {"enabled": False},
            "webhook": {"enabled": False},
            "alert_routing": {
                "channel_matrix": {
                    "P1": {
                        "inApp": True,
                        "telegram": True,
                        "email": True,
                        "webhook": True,
                    },
                },
            },
        },
    )

    results = notification_dispatcher.notify_with_results({"severity": "P1"})

    assert calls == []
    assert results == [
        {
            "channel": "inApp",
            "success": True,
            "status": "handled",
            "message": "Handled by the persisted alert and WebSocket path",
        }
    ]


def test_notify_requires_every_requested_external_delivery(monkeypatch):
    outcomes = {
        "telegram": SimpleNamespace(send_alert=lambda *_args: False),
        "email": SimpleNamespace(send_alert=lambda *_args: True),
    }
    monkeypatch.setattr(notification_dispatcher, "_CHANNEL_HANDLERS", outcomes)
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config",
        lambda: {
            "telegram": {
                "enabled": True,
                "bot_token": "token",
                "chat_id": "chat",
                "severities": ["P1"],
            },
            "email": {
                "enabled": True,
                "smtp_host": "smtp.example.com",
                "from_address": "alerts@example.com",
                "to_addresses": ["manager@example.com"],
                "severities": ["P1"],
            },
            "alert_routing": {
                "channel_matrix": {
                    "P1": {"inApp": True, "telegram": True, "email": True},
                },
            },
        },
    )

    assert notification_dispatcher.notify({"severity": "P1"}) is False
