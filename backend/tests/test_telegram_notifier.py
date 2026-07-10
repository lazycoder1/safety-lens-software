"""Tests for telegram_notifier module — Telegram Bot API notifications."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ── send_alert ───────────────────────────────────────────────────────────────

def _make_alert(**overrides):
    """Build a test alert dict."""
    alert = {
        "id": "abc12345",
        "severity": "P1",
        "status": "active",
        "rule": "No Helmet",
        "cameraId": "cam1",
        "cameraName": "Welding Bay",
        "zone": "Zone A",
        "confidence": 0.92,
        "timestamp": "2026-03-23T10:30:00.000000",
        "source": "YOLO",
        "description": "Worker without helmet",
        "snapshotUrl": None,
    }
    alert.update(overrides)
    return alert


def _telegram_response(status_code=200, payload=None):
    response = mock.MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {"ok": True, "result": {}}
    response.text = str(response.json.return_value)
    return response


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_disabled_does_not_send(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": False, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}
    assert telegram_notifier.send_alert(_make_alert()) is False
    mock_post.assert_not_called()


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_no_token_does_not_send(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "", "chat_id": "123", "severities": ["P1"]}}
    telegram_notifier.send_alert(_make_alert())
    mock_post.assert_not_called()


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_severity_filtered_out(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}
    telegram_notifier.send_alert(_make_alert(severity="P3"))
    mock_post.assert_not_called()


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_sends_photo_when_snapshot(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok123", "chat_id": "456", "severities": ["P1"]}}
    mock_post.return_value = _telegram_response()

    # Create a temp snapshot file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"\xff\xd8fake_jpeg")
        snap_path = f.name

    assert telegram_notifier.send_alert(_make_alert(), snapshot_path=snap_path) is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "sendPhoto" in call_kwargs[0][0]
    assert call_kwargs[1]["data"]["chat_id"] == "456"

    Path(snap_path).unlink(missing_ok=True)


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_sends_message_when_no_snapshot(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok123", "chat_id": "456", "severities": ["P1"]}}
    mock_post.return_value = _telegram_response()

    assert telegram_notifier.send_alert(_make_alert()) is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "sendMessage" in call_kwargs[0][0]
    assert call_kwargs[1]["json"]["chat_id"] == "456"


@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_correct_caption_format(mock_post, mock_cfg):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}
    mock_post.return_value = _telegram_response()

    alert = _make_alert()
    telegram_notifier.send_alert(alert)

    call_kwargs = mock_post.call_args
    text = call_kwargs[1]["json"]["text"]
    assert "P1 —" in text
    assert "No Helmet" in text
    assert "Welding Bay" in text
    assert "Zone A" in text


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (401, {"ok": False, "description": "Unauthorized"}),
        (429, {"ok": False, "description": "Too Many Requests"}),
        (200, {"ok": False, "description": "Chat not found"}),
    ],
)
@mock.patch("telegram_notifier.get_config")
@mock.patch("telegram_notifier.requests.post")
def test_send_alert_reports_telegram_rejection(mock_post, mock_cfg, status_code, payload):
    import telegram_notifier
    mock_cfg.return_value = {"telegram": {"enabled": True, "bot_token": "tok", "chat_id": "123", "severities": ["P1"]}}
    mock_post.return_value = _telegram_response(status_code, payload)

    assert telegram_notifier.send_alert(_make_alert()) is False


# ── test_connection ──────────────────────────────────────────────────────────

@mock.patch("telegram_notifier.requests.post")
def test_test_connection_success(mock_post):
    import telegram_notifier
    mock_post.return_value = _telegram_response()

    result = telegram_notifier.test_connection("tok123", "chat456")
    assert result["ok"] is True
    mock_post.assert_called_once()
    assert "sendMessage" in mock_post.call_args[0][0]


@mock.patch("telegram_notifier.requests.post")
def test_test_connection_failure(mock_post):
    import telegram_notifier
    mock_post.return_value = _telegram_response(401, {"ok": False, "description": "Unauthorized"})

    result = telegram_notifier.test_connection("bad_tok", "chat456")
    assert result["ok"] is False
    assert result["error"] == "Telegram HTTP 401: request rejected"


@mock.patch("telegram_notifier.requests.post", side_effect=ConnectionError("no network"))
def test_test_connection_exception(mock_post):
    import telegram_notifier
    result = telegram_notifier.test_connection("tok", "chat")
    assert result["ok"] is False
    assert "no network" in result["error"]


# ── fetch_groups ─────────────────────────────────────────────────────────────

@mock.patch("telegram_notifier.requests.get")
def test_fetch_groups_reports_http_200_api_rejection(mock_get):
    import telegram_notifier
    mock_get.return_value = _telegram_response(
        200,
        {"ok": False, "description": "Unauthorized bot token"},
    )

    result = telegram_notifier.fetch_groups("bad-token")

    assert result["ok"] is False
    assert result["groups"] == []
    assert result["error"] == "Telegram API rejected request"


@mock.patch("telegram_notifier.requests.get")
def test_fetch_groups_reports_invalid_json(mock_get):
    import telegram_notifier
    response = mock.MagicMock(status_code=200, text="<html>gateway error</html>")
    response.json.side_effect = ValueError("not JSON")
    mock_get.return_value = response

    result = telegram_notifier.fetch_groups("token")

    assert result["ok"] is False
    assert result["groups"] == []
    assert "invalid JSON" in result["error"]


@mock.patch("telegram_notifier.requests.get")
def test_fetch_groups_returns_unique_groups_after_valid_response(mock_get):
    import telegram_notifier
    mock_get.return_value = _telegram_response(
        payload={
            "ok": True,
            "result": [
                {"message": {"chat": {"id": -1001, "title": "Safety", "type": "group"}}},
                {"message": {"chat": {"id": -1001, "title": "Safety", "type": "group"}}},
                {
                    "my_chat_member": {
                        "chat": {"id": -1002, "title": "Operations", "type": "supergroup"},
                    },
                },
                {"message": {"chat": {"id": 42, "first_name": "Private", "type": "private"}}},
            ],
        },
    )

    result = telegram_notifier.fetch_groups("token")

    assert result == {
        "ok": True,
        "groups": [
            {"chat_id": "-1001", "title": "Safety", "type": "group"},
            {"chat_id": "-1002", "title": "Operations", "type": "supergroup"},
        ],
    }
