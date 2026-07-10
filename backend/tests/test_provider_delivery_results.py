import smtplib
import socket
import ssl
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import email_notifier
import notification_dispatcher
import telegram_notifier
import webhook_notifier
from delivery_result import (
    DeliveryDisposition,
    ProviderDeliveryResult,
    parse_retry_after,
)


def _alert() -> dict:
    return {
        "id": "alert-123",
        "severity": "P1",
        "status": "active",
        "rule": "No Helmet",
        "cameraId": "cam-1",
        "cameraName": "Camera 1",
        "zone": "Line 1",
        "confidence": 0.9,
        "timestamp": "2026-07-10T10:00:00+00:00",
    }


def _telegram_config() -> dict:
    return {
        "telegram": {
            "enabled": True,
            "bot_token": "telegram-secret-token",
            "chat_id": "chat-1",
            "severities": ["P1"],
        }
    }


def _telegram_response(status: int, payload: dict, headers: dict | None = None):
    response = mock.MagicMock(status_code=status, headers=headers or {})
    response.json.return_value = payload
    return response


@pytest.fixture(autouse=True)
def public_webhook_dns(monkeypatch):
    """Keep provider tests deterministic and make DNS trust explicit."""
    monkeypatch.setattr(
        webhook_notifier.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port)),
        ],
    )


@pytest.mark.parametrize(
    ("http_status", "payload", "expected"),
    [
        (401, {"ok": False, "error_code": 401}, DeliveryDisposition.TERMINAL),
        (403, {"ok": False, "error_code": 403}, DeliveryDisposition.TERMINAL),
        (408, {"ok": False, "error_code": 408}, DeliveryDisposition.RETRYABLE),
        (501, {"ok": False, "error_code": 501}, DeliveryDisposition.RETRYABLE),
        (503, {"ok": False, "error_code": 503}, DeliveryDisposition.RETRYABLE),
    ],
)
def test_telegram_classifies_permanent_and_temporary_rejections(
    monkeypatch,
    http_status,
    payload,
    expected,
):
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    monkeypatch.setattr(
        telegram_notifier.requests,
        "post",
        lambda *_args, **_kwargs: _telegram_response(http_status, payload),
    )

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is expected
    assert result.provider_status == http_status
    assert result.acceptance_unknown is (
        http_status == 408 or 500 <= http_status < 600
    )
    assert telegram_notifier.send_alert(_alert()) is (expected is DeliveryDisposition.DELIVERED)


def test_telegram_honors_json_retry_after_before_header(monkeypatch):
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    response = _telegram_response(
        429,
        {
            "ok": False,
            "error_code": 429,
            "parameters": {"retry_after": 73},
        },
        headers={"Retry-After": "12"},
    )
    monkeypatch.setattr(telegram_notifier.requests, "post", lambda *_args, **_kwargs: response)

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.RETRYABLE
    assert result.retry_after_seconds == 73


def test_telegram_invalid_success_response_is_ambiguous_and_retryable(monkeypatch):
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    response = mock.MagicMock(status_code=200, headers={})
    response.json.side_effect = ValueError("not json")
    monkeypatch.setattr(telegram_notifier.requests, "post", lambda *_args, **_kwargs: response)

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.RETRYABLE
    assert result.acceptance_unknown is True
    assert result.error_code == "invalid_response"


def test_telegram_read_timeout_is_ambiguous_without_exposing_token(monkeypatch, caplog):
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    monkeypatch.setattr(
        telegram_notifier.requests,
        "post",
        mock.Mock(side_effect=requests.exceptions.ReadTimeout("telegram-secret-token timed out")),
    )

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.RETRYABLE
    assert result.acceptance_unknown is True
    assert "telegram-secret-token" not in result.message
    assert "telegram-secret-token" not in caplog.text


def test_telegram_follows_one_valid_chat_migration_for_same_message(monkeypatch, caplog):
    config = _telegram_config()
    migrated_chat_id = -1001234567890
    monkeypatch.setattr(telegram_notifier, "get_config", lambda: config)
    responses = [
        _telegram_response(
            400,
            {
                "ok": False,
                "error_code": 400,
                "parameters": {"migrate_to_chat_id": migrated_chat_id},
            },
        ),
        _telegram_response(200, {"ok": True, "result": {}}),
    ]
    calls = []

    def post(*_args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(telegram_notifier.requests, "post", post)

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.DELIVERED
    assert len(calls) == 2
    assert calls[0]["json"]["chat_id"] == "chat-1"
    assert calls[1]["json"]["chat_id"] == migrated_chat_id
    assert calls[0]["json"]["text"] == calls[1]["json"]["text"]
    assert config["telegram"]["chat_id"] == "chat-1"
    assert str(migrated_chat_id) not in result.message
    assert str(migrated_chat_id) not in caplog.text


def test_telegram_follows_one_valid_chat_migration_for_same_photo(
    monkeypatch,
    tmp_path,
):
    config = _telegram_config()
    migrated_chat_id = -1009876543210
    snapshot = tmp_path / "snapshot.jpg"
    snapshot.write_bytes(b"jpeg")
    monkeypatch.setattr(telegram_notifier, "get_config", lambda: config)
    responses = [
        _telegram_response(
            400,
            {
                "ok": False,
                "error_code": 400,
                "parameters": {"migrate_to_chat_id": migrated_chat_id},
            },
        ),
        _telegram_response(200, {"ok": True, "result": {}}),
    ]
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(telegram_notifier.requests, "post", post)

    result = telegram_notifier.send_alert_result(_alert(), str(snapshot))

    assert result.disposition is DeliveryDisposition.DELIVERED
    assert len(calls) == 2
    assert all("sendPhoto" in args[0] for args, _kwargs in calls)
    assert calls[0][1]["data"]["chat_id"] == "chat-1"
    assert calls[1][1]["data"]["chat_id"] == migrated_chat_id
    assert calls[0][1]["data"]["caption"] == calls[1][1]["data"]["caption"]


@pytest.mark.parametrize(
    "migrated_chat_id",
    [True, "123", "not-an-integer", 0, 2**63],
)
def test_telegram_rejects_invalid_chat_migration_without_exposing_id(
    monkeypatch,
    caplog,
    migrated_chat_id,
):
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    post = mock.Mock(
        return_value=_telegram_response(
            400,
            {
                "ok": False,
                "error_code": 400,
                "parameters": {"migrate_to_chat_id": migrated_chat_id},
            },
        )
    )
    monkeypatch.setattr(telegram_notifier.requests, "post", post)

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "invalid_chat_migration"
    assert post.call_count == 1
    assert str(migrated_chat_id) not in result.message
    assert str(migrated_chat_id) not in caplog.text


def test_telegram_caps_repeated_chat_migration_at_one_hop(monkeypatch, caplog):
    first_chat_id = -1001111111111
    second_chat_id = -1002222222222
    monkeypatch.setattr(telegram_notifier, "get_config", _telegram_config)
    post = mock.Mock(
        side_effect=[
            _telegram_response(
                400,
                {
                    "ok": False,
                    "error_code": 400,
                    "parameters": {"migrate_to_chat_id": first_chat_id},
                },
            ),
            _telegram_response(
                400,
                {
                    "ok": False,
                    "error_code": 400,
                    "parameters": {"migrate_to_chat_id": second_chat_id},
                },
            ),
        ]
    )
    monkeypatch.setattr(telegram_notifier.requests, "post", post)

    result = telegram_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "chat_migration_loop"
    assert post.call_count == 2
    assert str(first_chat_id) not in result.message
    assert str(second_chat_id) not in result.message
    assert str(first_chat_id) not in caplog.text
    assert str(second_chat_id) not in caplog.text


def _webhook_config() -> dict:
    return {
        "webhook": {
            "enabled": True,
            "url": "https://hooks.example/alerts",
            "headers": {"Authorization": "Bearer header-secret"},
            "severities": ["P1"],
        }
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (204, DeliveryDisposition.DELIVERED),
        (400, DeliveryDisposition.TERMINAL),
        (409, DeliveryDisposition.TERMINAL),
        (408, DeliveryDisposition.RETRYABLE),
        (429, DeliveryDisposition.RETRYABLE),
        (500, DeliveryDisposition.RETRYABLE),
        (501, DeliveryDisposition.RETRYABLE),
        (505, DeliveryDisposition.RETRYABLE),
    ],
)
def test_webhook_http_classification(monkeypatch, status, expected):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    monkeypatch.setattr(
        webhook_notifier,
        "_post_pinned",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=status, headers={}),
    )

    result = webhook_notifier.send_alert_result(_alert())

    assert result.disposition is expected
    assert result.provider_status == status
    assert result.acceptance_unknown is (status == 408 or 500 <= status < 600)


def test_webhook_disables_redirects_and_uses_stable_idempotency_key(monkeypatch):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    calls = []

    def post(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=204, headers={})

    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    assert webhook_notifier.send_alert_result(_alert()).success
    assert webhook_notifier.send_alert_result(_alert()).success
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["headers"]["Idempotency-Key"] == calls[1]["headers"]["Idempotency-Key"]


def test_webhook_honors_retry_after_and_marks_read_timeout_ambiguous(monkeypatch, caplog):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    responses = [
        SimpleNamespace(status_code=429, headers={"Retry-After": "41"}),
        requests.exceptions.ReadTimeout("header-secret response timed out"),
    ]

    def post(*_args, **_kwargs):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    limited = webhook_notifier.send_alert_result(_alert())
    ambiguous = webhook_notifier.send_alert_result(_alert())
    assert limited.retry_after_seconds == 41
    assert ambiguous.disposition is DeliveryDisposition.RETRYABLE
    assert ambiguous.acceptance_unknown is True
    assert "header-secret" not in ambiguous.message
    assert "header-secret" not in caplog.text
    assert "webhook-password" not in caplog.text


def test_webhook_invalid_tls_certificate_is_terminal(monkeypatch):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    monkeypatch.setattr(
        webhook_notifier,
        "_post_pinned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            webhook_notifier._WebhookPolicyError("tls_certificate_invalid")
        ),
    )

    result = webhook_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "tls_certificate_invalid"


def test_webhook_public_https_target_is_validated_on_every_attempt(monkeypatch):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    resolver = mock.Mock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.4.4", 443)),
        ]
    )
    monkeypatch.setattr(webhook_notifier.socket, "getaddrinfo", resolver)
    post = mock.Mock(return_value=SimpleNamespace(status_code=204, headers={}))
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    assert webhook_notifier.send_alert_result(_alert()).success
    assert webhook_notifier.send_alert_result(_alert()).success
    assert resolver.call_count == 2
    assert post.call_count == 2
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["verify"] is True


def test_webhook_uses_validated_ip_without_second_dns_or_environment_proxy(
    monkeypatch,
):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    resolver = mock.Mock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.4.4", 443)),
        ]
    )
    monkeypatch.setattr(webhook_notifier.socket, "getaddrinfo", resolver)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    observed = {}

    class FakeSocket:
        def settimeout(self, value):
            observed.setdefault("timeouts", []).append(value)

        def connect(self, endpoint):
            observed["endpoint"] = endpoint

        def close(self):
            observed["socket_closed"] = True

    raw_socket = FakeSocket()

    def socket_factory(family, socktype):
        observed["socket_family"] = family
        observed["socket_type"] = socktype
        return raw_socket

    class FakeTLSContext:
        def wrap_socket(self, wrapped, *, server_hostname):
            observed["sni"] = server_hostname
            assert wrapped is raw_socket
            return wrapped

    class FakeResponse:
        status = 204
        headers = {}

        def close(self):
            observed["response_closed"] = True

    class FakeConnection:
        def __init__(self, host, port, timeout):
            observed["connection"] = (host, port, timeout)
            self.sock = None

        def request(self, method, request_target, *, body, headers):
            observed["request"] = (method, request_target, body, headers)

        def getresponse(self):
            return FakeResponse()

        def close(self):
            observed["connection_closed"] = True

    monkeypatch.setattr(webhook_notifier.socket, "socket", socket_factory)
    monkeypatch.setattr(
        webhook_notifier.ssl,
        "create_default_context",
        lambda: FakeTLSContext(),
    )
    monkeypatch.setattr(
        webhook_notifier.http.client,
        "HTTPConnection",
        FakeConnection,
    )

    result = webhook_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.DELIVERED
    assert resolver.call_count == 1
    assert observed["endpoint"] == ("8.8.4.4", 443)
    assert observed["sni"] == "hooks.example"
    assert observed["connection"][:2] == ("hooks.example", 443)
    assert observed["request"][0:2] == ("POST", "/alerts")
    assert observed["request"][3]["Host"] == "hooks.example"
    assert observed["response_closed"] is True
    assert observed["connection_closed"] is True


def test_webhook_ipv6_dns_answer_does_not_bracket_domain_host_header(monkeypatch):
    monkeypatch.setattr(
        webhook_notifier.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("2001:4860:4860::8888", port, 0, 0),
            ),
        ],
    )

    target = webhook_notifier._validate_webhook_url(
        "https://hooks.example:8443/alerts",
    )

    assert target.address.version == 6
    assert target.host_header == "hooks.example:8443"


def test_outbox_dispatch_uses_one_config_generation_for_validation_and_send(
    monkeypatch,
):
    config_a = _webhook_config()
    config_a["webhook"]["url"] = "https://a.example/alerts"
    config_a["webhook"]["headers"] = {"Authorization": "Bearer generation-a"}
    config_b = _webhook_config()
    config_b["webhook"]["url"] = "https://b.example/alerts"
    config_b["webhook"]["headers"] = {"Authorization": "Bearer generation-b"}
    target = {
        "id": "delivery-1",
        "kind": "initial",
        "channel": "webhook",
        "target_key": (
            "initial:webhook:"
            f"{notification_dispatcher._target_identity_fingerprint(config_a, 'webhook', config_a['webhook']['url'])}"
        ),
        "context": {"include_snapshot": False},
    }
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: config_a,
    )
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: config_b)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=204, headers={})

    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = notification_dispatcher.deliver_outbox_target(
        _alert(),
        target,
        None,
    )

    assert result.disposition is DeliveryDisposition.DELIVERED
    assert calls[0][0] == "https://a.example/alerts"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer generation-a"
    assert calls[0][1]["target"].hostname == "a.example"


def test_resolver_preserves_invalid_initial_and_custom_legacy_escalation_routes():
    cfg = {
        "telegram": {"enabled": False},
        "email": {"enabled": False},
        "alert_routing": {
            "channel_matrix": {
                "P1": {
                    "inApp": True,
                    "telegram": True,
                    "email": False,
                    "sms": True,
                }
            },
            "escalation_steps": [
                {
                    "id": 1,
                    "afterMinutes": 3,
                    "role": "Manager",
                    "channel": "telegram",
                }
            ],
        },
    }

    targets = notification_dispatcher.resolve_delivery_targets(
        cfg,
        _alert(),
    )

    assert {target["channel"] for target in targets} == {"telegram", "sms"}
    assert {target["kind"] for target in targets} == {"initial", "escalation"}
    assert all(target["target_key"].endswith(":unconfigured") for target in targets)


def test_durable_target_identity_binds_telegram_bot_but_allows_token_rotation():
    first = {
        "telegram": {"bot_token": "123456:old-secret"},
    }
    rotated = {
        "telegram": {"bot_token": "123456:new-secret"},
    }
    other_bot = {
        "telegram": {"bot_token": "999999:new-secret"},
    }

    fingerprint = notification_dispatcher._target_identity_fingerprint(
        first, "telegram", "chat-1"
    )

    assert fingerprint == notification_dispatcher._target_identity_fingerprint(
        rotated, "telegram", "chat-1"
    )
    assert fingerprint != notification_dispatcher._target_identity_fingerprint(
        other_bot, "telegram", "chat-1"
    )
    assert "123456" not in fingerprint


def test_nonstandard_telegram_token_rotation_fails_closed():
    first = {"telegram": {"bot_token": "opaque-old"}}
    rotated = {"telegram": {"bot_token": "opaque-new"}}

    assert notification_dispatcher._target_identity_fingerprint(
        first, "telegram", "chat-1"
    ) != notification_dispatcher._target_identity_fingerprint(
        rotated, "telegram", "chat-1"
    )


def test_durable_email_identity_allows_password_rotation_but_binds_account():
    base = {
        "email": {
            "smtp_host": "smtp.example",
            "smtp_port": 587,
            "smtp_user": "mailer",
            "smtp_pass": "old-password",
            "from_address": "safety@example.com",
        }
    }
    rotated = {"email": {**base["email"], "smtp_pass": "new-password"}}
    other_account = {"email": {**base["email"], "smtp_user": "other-mailer"}}

    fingerprint = notification_dispatcher._target_identity_fingerprint(
        base, "email", "recipient@example.com"
    )

    assert fingerprint == notification_dispatcher._target_identity_fingerprint(
        rotated, "email", "recipient@example.com"
    )
    assert fingerprint != notification_dispatcher._target_identity_fingerprint(
        other_account, "email", "recipient@example.com"
    )
    assert "mailer" not in fingerprint


def test_webhook_auth_rotation_requires_stable_tenant_boundary():
    without_tenant = {
        "webhook": {"headers": {"Authorization": "Bearer old"}},
    }
    without_tenant_rotated = {
        "webhook": {"headers": {"Authorization": "Bearer new"}},
    }
    with_tenant = {
        "webhook": {
            "headers": {
                "Authorization": "Bearer old",
                "X-Tenant-ID": "tenant-a",
                "X-Route": "safety",
            }
        },
    }
    with_tenant_rotated = {
        "webhook": {
            "headers": {
                "Authorization": "Bearer new",
                "X-Tenant-ID": "tenant-a",
                "X-Route": "safety",
            }
        },
    }
    other_tenant = {
        "webhook": {
            "headers": {
                "Authorization": "Bearer new",
                "X-Tenant-ID": "tenant-b",
                "X-Route": "safety",
            }
        },
    }
    other_route = {
        "webhook": {
            "headers": {
                "Authorization": "Bearer new",
                "X-Tenant-ID": "tenant-a",
                "X-Route": "operations",
            }
        },
    }
    url = "https://hooks.example/alerts"

    assert notification_dispatcher._target_identity_fingerprint(
        without_tenant, "webhook", url
    ) != notification_dispatcher._target_identity_fingerprint(
        without_tenant_rotated, "webhook", url
    )
    tenant_fingerprint = notification_dispatcher._target_identity_fingerprint(
        with_tenant, "webhook", url
    )
    assert tenant_fingerprint == notification_dispatcher._target_identity_fingerprint(
        with_tenant_rotated, "webhook", url
    )
    assert tenant_fingerprint != notification_dispatcher._target_identity_fingerprint(
        other_tenant, "webhook", url
    )
    assert tenant_fingerprint != notification_dispatcher._target_identity_fingerprint(
        other_route, "webhook", url
    )
    assert "tenant-a" not in tenant_fingerprint

    explicit_account_old = {
        "webhook": {
            "account_id": "tenant-a",
            "headers": {"Authorization": "Bearer old"},
        }
    }
    explicit_account_new = {
        "webhook": {
            "account_id": "tenant-a",
            "headers": {"Authorization": "Bearer new"},
        }
    }
    assert notification_dispatcher._target_identity_fingerprint(
        explicit_account_old, "webhook", url
    ) == notification_dispatcher._target_identity_fingerprint(
        explicit_account_new, "webhook", url
    )


def test_destination_only_legacy_outbox_key_fails_closed(monkeypatch):
    cfg = _webhook_config()
    target = {
        "id": "delivery-legacy",
        "kind": "initial",
        "channel": "webhook",
        "target_key": (
            "initial:webhook:"
            f"{notification_dispatcher._target_fingerprint(cfg['webhook']['url'])}"
        ),
        "context": {},
    }
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: cfg,
    )

    result = notification_dispatcher.deliver_outbox_target(_alert(), target, None)

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "target_configuration_changed"


def test_outbox_webhook_auth_rotation_uses_new_credential_only_with_same_tenant(
    monkeypatch,
):
    accepted = _webhook_config()
    accepted["webhook"]["headers"] = {
        "Authorization": "Bearer old",
        "X-Tenant-ID": "tenant-a",
    }
    current = _webhook_config()
    current["webhook"]["headers"] = {
        "Authorization": "Bearer new",
        "X-Tenant-ID": "tenant-a",
    }
    fingerprint = notification_dispatcher._target_identity_fingerprint(
        accepted, "webhook", accepted["webhook"]["url"]
    )
    target = {
        "id": "delivery-rotation",
        "kind": "initial",
        "channel": "webhook",
        "target_key": f"initial:webhook:{fingerprint}",
        "context": {},
    }
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: current,
    )
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=204, headers={})

    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = notification_dispatcher.deliver_outbox_target(_alert(), target, None)

    assert result.disposition is DeliveryDisposition.DELIVERED
    assert calls[0][1]["headers"]["Authorization"] == "Bearer new"
    assert calls[0][1]["headers"]["X-Tenant-ID"] == "tenant-a"


def test_outbox_webhook_auth_rotation_without_tenant_does_not_send(monkeypatch):
    accepted = _webhook_config()
    accepted["webhook"]["headers"] = {"Authorization": "Bearer old"}
    current = _webhook_config()
    current["webhook"]["headers"] = {"Authorization": "Bearer new"}
    target = {
        "id": "delivery-unsafe-rotation",
        "kind": "initial",
        "channel": "webhook",
        "target_key": (
            "initial:webhook:"
            + notification_dispatcher._target_identity_fingerprint(
                accepted, "webhook", accepted["webhook"]["url"]
            )
        ),
        "context": {},
    }
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: current,
    )
    post = mock.Mock()
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = notification_dispatcher.deliver_outbox_target(_alert(), target, None)

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "target_configuration_changed"
    post.assert_not_called()


def test_outbox_target_key_with_colons_in_escalation_id_still_matches(monkeypatch):
    cfg = _webhook_config()
    destination = cfg["webhook"]["url"]
    fingerprint = notification_dispatcher._target_identity_fingerprint(
        cfg, "webhook", destination
    )
    target = {
        "id": "delivery-colon-step",
        "kind": "escalation",
        "channel": "webhook",
        "target_key": f"escalation:night:ops:1:webhook:{fingerprint}",
        "context": {"role": "Night Manager", "after_minutes": 3},
    }
    monkeypatch.setattr(
        notification_dispatcher,
        "get_config_snapshot",
        lambda _section: cfg,
    )
    monkeypatch.setattr(
        webhook_notifier,
        "_post_pinned",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=204, headers={}),
    )

    result = notification_dispatcher.deliver_outbox_target(_alert(), target, None)

    assert result.disposition is DeliveryDisposition.DELIVERED


def test_webhook_response_cleanup_failure_does_not_retry_accepted_alert(monkeypatch):
    monkeypatch.setattr(webhook_notifier, "get_config", _webhook_config)
    response = mock.Mock(status_code=204, headers={})
    response.close.side_effect = OSError("cleanup failed")
    monkeypatch.setattr(webhook_notifier, "_post_pinned", mock.Mock(return_value=response))

    result = webhook_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.DELIVERED


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/alerts",
        "https://10.0.0.8/alerts",
        "https://[::1]/alerts",
        "https://[fe80::1]/alerts",
        "https://[ff02::1]/alerts",
    ],
)
def test_webhook_rejects_direct_non_public_addresses(monkeypatch, url):
    cfg = _webhook_config()
    cfg["webhook"]["url"] = url
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    post = mock.Mock()
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = webhook_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.TERMINAL
    assert result.error_code == "endpoint_address_forbidden"
    post.assert_not_called()


def test_webhook_rejects_hostname_resolving_to_private_address(monkeypatch):
    cfg = _webhook_config()
    cfg["webhook"]["url"] = "https://internal.example/alerts"
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    monkeypatch.setattr(
        webhook_notifier.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.168.1.5", port)),
        ],
    )
    post = mock.Mock()
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = webhook_notifier.send_alert_result(_alert())

    assert result.error_code == "endpoint_address_forbidden"
    post.assert_not_called()


def test_webhook_rejects_userinfo_without_leaking_it(monkeypatch, caplog):
    secret = "webhook-userinfo-secret"
    cfg = _webhook_config()
    cfg["webhook"]["url"] = f"https://user:{secret}@hooks.example/alerts"
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    post = mock.Mock()
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = webhook_notifier.send_alert_result(_alert())

    assert result.error_code == "endpoint_userinfo_forbidden"
    assert secret not in result.message
    assert secret not in caplog.text
    post.assert_not_called()


def test_webhook_http_requires_exact_host_allowlist(monkeypatch):
    cfg = _webhook_config()
    cfg["webhook"]["url"] = "http://hooks.example/alerts"
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    post = mock.Mock(return_value=SimpleNamespace(status_code=204, headers={}))
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    rejected = webhook_notifier.send_alert_result(_alert())
    assert rejected.error_code == "endpoint_https_required"
    post.assert_not_called()

    monkeypatch.setenv("WEBHOOK_ALLOWED_HTTP_HOSTS", "other.example, hooks.example")
    accepted = webhook_notifier.send_alert_result(_alert())
    assert accepted.disposition is DeliveryDisposition.DELIVERED
    post.assert_called_once()


def test_webhook_private_target_requires_exact_host_allowlist(monkeypatch):
    cfg = _webhook_config()
    cfg["webhook"]["url"] = "https://internal.example/alerts"
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    monkeypatch.setattr(
        webhook_notifier.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.10.0.4", port)),
        ],
    )
    post = mock.Mock(return_value=SimpleNamespace(status_code=204, headers={}))
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    assert webhook_notifier.send_alert_result(_alert()).error_code == "endpoint_address_forbidden"
    monkeypatch.setenv("WEBHOOK_ALLOWED_PRIVATE_HOSTS", "other.example,internal.example")
    assert webhook_notifier.send_alert_result(_alert()).success


@pytest.mark.parametrize("header", ["Host", "Content-Length", "Connection", "Transfer-Encoding"])
def test_webhook_rejects_forbidden_request_headers(monkeypatch, header):
    cfg = _webhook_config()
    cfg["webhook"]["headers"][header] = "unsafe"
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    post = mock.Mock()
    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)

    result = webhook_notifier.send_alert_result(_alert())

    assert result.error_code == "forbidden_header"
    post.assert_not_called()


def test_webhook_retry_payload_is_stable_and_snapshot_policy_can_be_frozen(
    monkeypatch,
    tmp_path,
):
    cfg = _webhook_config()
    cfg["webhook"]["include_snapshot"] = True
    monkeypatch.setattr(webhook_notifier, "get_config", lambda: cfg)
    snapshot = tmp_path / "snapshot.jpg"
    snapshot.write_bytes(b"jpeg-bytes")
    calls = []

    def post(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=500, headers={})

    monkeypatch.setattr(webhook_notifier, "_post_pinned", post)
    active = _alert()
    resolved = {**active, "status": "resolved"}

    webhook_notifier.send_alert_result(
        active,
        str(snapshot),
        include_snapshot_override=False,
    )
    webhook_notifier.send_alert_result(
        resolved,
        str(snapshot),
        include_snapshot_override=False,
    )

    assert calls[0]["json"] == calls[1]["json"]
    assert "status" not in calls[0]["json"]["alert"]
    assert "snapshot_base64" not in calls[0]["json"]

    cfg["webhook"]["include_snapshot"] = False
    webhook_notifier.send_alert_result(
        active,
        str(snapshot),
        include_snapshot_override=True,
    )
    assert calls[2]["json"]["snapshot_base64"] == "anBlZy1ieXRlcw=="


def _email_config() -> dict:
    return {
        "email": {
            "enabled": True,
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "user",
            "smtp_pass": "password",
            "from_address": "alerts@example.com",
            "to_addresses": ["one@example.com", "two@example.com"],
            "severities": ["P1"],
        }
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (smtplib.SMTPConnectError(421, b"temporarily unavailable"), DeliveryDisposition.RETRYABLE),
        (smtplib.SMTPAuthenticationError(535, b"bad credentials"), DeliveryDisposition.TERMINAL),
    ],
)
def test_email_classifies_smtp_response_codes(monkeypatch, error, expected):
    monkeypatch.setattr(email_notifier, "get_config", _email_config)
    monkeypatch.setattr(email_notifier, "_send", mock.Mock(side_effect=error))

    result = email_notifier.send_alert_result(_alert())

    assert result.disposition is expected
    assert result.provider_status == error.smtp_code


def test_email_marks_partial_acceptance_ambiguous_until_recipient_outbox(monkeypatch):
    monkeypatch.setattr(email_notifier, "get_config", _email_config)
    refusal = smtplib.SMTPRecipientsRefused(
        {"two@example.com": (450, b"try later")}
    )
    monkeypatch.setattr(email_notifier, "_send", mock.Mock(side_effect=refusal))

    result = email_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.RETRYABLE
    assert result.error_code == "partial_recipient_refusal"
    assert result.acceptance_unknown is True


def test_email_retries_when_every_recipient_is_temporarily_refused(monkeypatch):
    monkeypatch.setattr(email_notifier, "get_config", _email_config)
    refusal = smtplib.SMTPRecipientsRefused(
        {
            "one@example.com": (450, b"try later"),
            "two@example.com": (451, b"try later"),
        }
    )
    monkeypatch.setattr(email_notifier, "_send", mock.Mock(side_effect=refusal))

    result = email_notifier.send_alert_result(_alert())

    assert result.disposition is DeliveryDisposition.RETRYABLE
    assert result.provider_status == "450,451"


def test_email_message_id_is_stable_across_attempts(monkeypatch):
    monkeypatch.setattr(email_notifier, "get_config", _email_config)
    message_ids = []

    def capture(*args):
        message_ids.append(args[-1]["Message-ID"])

    monkeypatch.setattr(email_notifier, "_send", capture)

    assert email_notifier.send_alert_result(_alert()).success
    assert email_notifier.send_alert_result(_alert()).success
    assert message_ids == [message_ids[0], message_ids[0]]


def test_email_implicit_tls_uses_verified_default_context(monkeypatch):
    context = mock.Mock(check_hostname=True, verify_mode=ssl.CERT_REQUIRED)
    monkeypatch.setattr(email_notifier.ssl, "create_default_context", mock.Mock(return_value=context))
    server = mock.Mock()
    server.sendmail.return_value = {}
    smtp_ssl = mock.Mock(return_value=server)
    monkeypatch.setattr(email_notifier.smtplib, "SMTP_SSL", smtp_ssl)

    email_notifier._send(
        "smtp.example.com",
        465,
        "",
        "",
        "alerts@example.com",
        ["ops@example.com"],
        email_notifier.MIMEText("safe"),
    )

    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert smtp_ssl.call_args.kwargs["context"] is context


def test_email_starttls_uses_verified_default_context(monkeypatch):
    context = mock.Mock(check_hostname=True, verify_mode=ssl.CERT_REQUIRED)
    monkeypatch.setattr(email_notifier.ssl, "create_default_context", mock.Mock(return_value=context))
    server = mock.Mock()
    server.sendmail.return_value = {}
    monkeypatch.setattr(email_notifier.smtplib, "SMTP", mock.Mock(return_value=server))

    email_notifier._send(
        "smtp.example.com",
        587,
        "",
        "",
        "alerts@example.com",
        ["ops@example.com"],
        email_notifier.MIMEText("safe"),
    )

    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    server.starttls.assert_called_once_with(context=context)


def test_email_html_escapes_all_alert_controlled_fields():
    alert = {
        **_alert(),
        "severity": "<P1>",
        "rule": "<script>rule</script>",
        "cameraName": '<img src=x onerror="camera">',
        "zone": "<b>zone</b>",
        "confidence": "<confidence>",
        "timestamp": "<time>",
        "description": "<svg onload=description>",
    }

    subject, body = email_notifier._build_email(alert, None)

    assert "<script>" not in subject
    assert "<script>" not in body
    assert "<img src=x" not in body
    assert "<b>zone</b>" not in body
    assert "<svg onload=description>" not in body
    assert "&lt;script&gt;rule&lt;/script&gt;" in body
    assert "&lt;confidence&gt;" in body


def test_retry_after_parses_http_date_and_result_serializes_safe_fields():
    now = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)
    retry_at = now + timedelta(seconds=90)
    assert parse_retry_after(retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=now) == 90

    result = ProviderDeliveryResult(
        DeliveryDisposition.RETRYABLE,
        "Provider asked us to wait",
        error_code="rate_limited",
        provider_status=429,
        retry_after_seconds=90,
        acceptance_unknown=True,
    ).to_dispatch_dict("telegram")
    assert result == {
        "channel": "telegram",
        "success": False,
        "status": "retryable",
        "message": "Provider asked us to wait",
        "errorCode": "rate_limited",
        "providerStatus": 429,
        "retryAfterSeconds": 90.0,
        "acceptanceUnknown": True,
    }
